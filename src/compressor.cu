#include "compressor.hpp"
#include "InterpolationPredictor.hpp"
#include "bitplane.hpp"
#include "zbpc.hpp"
#include "dataloader.hpp"

namespace prism{

template <int bytewidth>
struct matchby;
template <>
struct matchby<4> {
  using utype = unsigned int;
  using itype = int;
  using ftype = float;
};
template <>
struct matchby<8> {
  using utype = unsigned long long;
  using itype = long long;
  using ftype = double;
};

#define __ATOMIC_PLUGIN                                                     \
  constexpr auto bytewidth = sizeof(T);                                     \
  using itype = typename matchby<bytewidth>::itype;                         \
  using utype = typename matchby<bytewidth>::utype;                         \
  using ftype = typename matchby<bytewidth>::ftype;                         \
  static_assert(std::is_same<T, ftype>::value, "T and ftype don't match."); \
  auto fp_as_int = [](T fpval) -> itype {                                   \
    return *reinterpret_cast<itype *>(&fpval);                              \
  };                                                                        \
  auto fp_as_uint = [](T fpval) -> utype {                                  \
    return *reinterpret_cast<utype *>(&fpval);                              \
  };                                                                        \
  auto int_as_fp = [](itype ival) -> T {                                    \
    return *reinterpret_cast<T *>(&ival);                                   \
  };                                                                        \
  auto uint_as_fp = [](utype uval) -> T {                                   \
    return *reinterpret_cast<T *>(&uval);                                   \
  };

template <typename T>
__device__ __forceinline__ T atomicMinFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value)
                 ? int_as_fp(atomicMin((itype *)addr, fp_as_int(value)))
                 : uint_as_fp(atomicMax((utype *)addr, fp_as_uint(value)));
  return old;
}

template <typename T>
__device__ __forceinline__ T atomicMaxFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value)
                 ? int_as_fp(atomicMax((itype *)addr, fp_as_int(value)))
                 : uint_as_fp(atomicMin((utype *)addr, fp_as_uint(value)));
  return old;
}

template<typename T, typename E>
Compressor<T,E>::~Compressor() {
    delete qc;
    delete ap;
    delete ol;
    delete qc_tmp;
    // delete bitplane;
    delete bp;
}

template<typename T, typename E>
void Compressor<T,E>::init(prism_context* config) {
    auto x = config->x;
    auto y = config->y;
    auto z = config->z;
    radius = config->radius;
    len = x * y * z;
    bytes_ = len * sizeof(T);
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
    prism_dtype dtype_;
    if constexpr (std::is_same<E, uint8_t>::value) {
        dtype_ = U1;
    } else if constexpr (std::is_same<E, uint16_t>::value) {
        dtype_ = U2;
    } else if constexpr (std::is_same<E, uint32_t>::value) {
        dtype_ = U4;
    }
    else if constexpr (std::is_same<E, int8_t>::value) {
        dtype_ = I1;
    }
    else if constexpr (std::is_same<E, int16_t>::value) {
        dtype_ = I2;
    }
    else if constexpr (std::is_same<E, int32_t>::value) {
        dtype_ = I4;
    }
    else {
        fprintf(stderr, "error quantization code type!");
    }
    qc = new inBuffer(dtype_, 32, x, y, z);
    qc_tmp = new inBuffer(dtype_, 32, x, y, z);
    ap = new inBuffer(config->dtype, 0, div(x, BLOCK_SIZE), div(y, BLOCK_SIZE), div(z, BLOCK_SIZE));
    ol = new olBuffer<T>(config->dtype, 0, x, y, z);
    compressed_data = new Buffer(I4, 32, x, y, z);
    profiling_errors = new Buffer(config->dtype, 0, 18, 1, 1);
    bp = new Bitplane(x, y, z);
    this->begin = config->begin;
    this->end = config->end;
    cudaMalloc(&compressedSize_bp_d, sizeof(size_t) * 4 * 32);
}


template<typename T, typename E>
    void Compressor<T, E>::compress_pipeline(context* config, StatBuffer<T>* input, void* stream) {
    compress_predict(config, input, stream);
    if(config->bt == SM)
        convert_to_bitplane<E, SM>(qc, bp, ap->len, time_bitplane, stream);
    else convert_to_bitplane<E, NB>(qc, bp, ap->len, time_bitplane, stream);
    compressed_lossless_size = lossless_encode(bp, compressed_bp, compressedSize_bp_d, input->bytes, time_encode, stream);
    compress_merge(config, input->range, stream); 
}

template<typename T, typename E>
void Compressor<T, E>::compress_merge(context* config, double input_range, void* stream) {
    total_compressed_size =  8 + compressed_lossless_size + sizeof(double) + 4 * 32 * sizeof(size_t)  + ap->bytes;
    cudaMemcpyAsync((uint8_t*)compressed_data->d, config->intp_param.use_md, sizeof(config->intp_param.use_md), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync((uint8_t*)compressed_data->d + 4 , config->intp_param.reverse, sizeof(config->intp_param.reverse), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    cudaMemcpyAsync((uint8_t*)compressed_data->d + 8, &input_range, sizeof(double), cudaMemcpyHostToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync((uint8_t*)compressed_data->d + 8 + sizeof(double), compressedSize_bp_d, 4 * 32 * 8, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync((uint8_t*)compressed_data->d + 8 + 4 * 32 * 8 + sizeof(double), ap->d, ap->bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync((uint8_t*)compressed_data->d + 8 + ap->bytes + 4 * 32 * 8 + sizeof(double), compressed_bp, 
    compressed_lossless_size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}

template<typename T, typename E>
void Compressor<T, E>::compress_predict(context* config, StatBuffer<T>* input, void* stream) {
    double eb = config->eb;
    double rel_eb = config->rel_eb;
    spline_construct<T,E,T>(input, ap, qc, qc_tmp, ol, eb, rel_eb, radius, config->intp_param, profiling_errors, time_pred, stream);
}

template<typename T, typename E>
void Compressor<T, E>::decompress_pipeline(context* config, StatBuffer<T>* output, void* stream) {
    if(config->bt == NB)
        throw std::invalid_argument("Please use progressive mode instead.");
    decompress_scatter<1>(config, stream);
    if(config->error_mode == REL) {
        config->eb = config->rel_eb * range;
    }
    lossless_decode(compressed_bp, bp, output->bytes, itime_decode, stream);
    inverse_convert_to_bitplane<E, SM>(bp, qc, ap->len, itime_bitplane, stream);
    decompress_predict(config, output, stream);
}

template<typename T, typename E>
void Compressor<T, E>::decompress_progressive_pipeline(context* config, StatBuffer<T>* output_old, 
StatBuffer<T>* output_new, double targetError, double lastError, void* stream) {
    
    if(config->bt == SM)
        throw std::invalid_argument("Please use non-progressive mode instead.");
    if(lastError == 0)
        decompress_scatter<1>(config, stream);
    else  decompress_scatter<0>(config, stream);
    if(config->error_mode == REL) {
        config->eb = config->rel_eb * range;
        targetError *= range;
    }
    // printf("%lf %lf %lf\n", targetError, range, config->eb);
    findStrategy_h<E>(compressedSize_bp_d, begin, end, config->eb, targetError, itime_enum, stream);
    lossless_decode_progressive(compressed_bp, bp, output_new->bytes, begin, end, itime_decode, stream);
    inverse_convert_to_bitplane_progressive<E, NB>(bp, qc, ap->len, begin, end, itime_bitplane, stream);
    // inverse_convert_to_bitplane<E>(bp, qc, ap->len, itime_bitplane, stream);
    
    bool first = lastError == 0; //std::all_of(lastcache.begin(), lastcache.end(), [](int v){ return v == 0; });
    if(first) {
        decompress_predict(config, output_new, stream);
    }
    else decompress_progressive_predict(config, output_old, output_new, stream);
}

template<typename T, typename E>
template<int init_ap>  
void Compressor<T, E>::decompress_scatter(context* config, void* stream) {

    // GPUTimer dtimer;
    // dtimer.start(stream);
    total_compressed_size = compressed_data->bytes;
    compressed_lossless_size = compressed_data->bytes - sizeof(double) - 4 * 32 * sizeof(size_t) - 8;
    cudaMalloc(&compressed_bp, compressed_lossless_size);
    cudaMemcpyAsync(config->intp_param.use_md, (uint8_t*)compressed_data->d, 4, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    cudaMemcpyAsync(config->intp_param.reverse, (uint8_t*)compressed_data->d + 4, 4, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    cudaMemcpyAsync(&range, (uint8_t*)compressed_data->d +8 , sizeof(double), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    cudaMemcpyAsync(compressedSize_bp_d, (uint8_t*)compressed_data->d + sizeof(double) + 8, 4 * 32 * sizeof(size_t), cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    if(init_ap == 1)
    cudaMemcpyAsync(ap->d, (uint8_t*)compressed_data->d + 4 * 32 * sizeof(size_t) + sizeof(double) + 8, ap->bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    cudaMemcpyAsync(compressed_bp, (uint8_t*)compressed_data->d + ap->bytes + 4 * 32 * sizeof(size_t) + sizeof(double) + 8,
    compressed_lossless_size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);    
}

template<typename T, typename E>
void Compressor<T, E>::decompress_predict(context* config, StatBuffer<T>* output, void* stream) {

    double eb = config->eb;
    double rel_eb = config->rel_eb;
    spline_reconstruct<T,E,T>(ap, qc, qc_tmp, ol, output, eb, rel_eb, radius, config->intp_param, itime_pred, stream);
}

template<typename T, typename E>
void Compressor<T, E>::decompress_progressive_predict(context* config, StatBuffer<T>* output_old, StatBuffer<T>* output_new, void* stream) {

    double eb = config->eb;
    double rel_eb = config->rel_eb;
    spline_progressive_reconstruct<T,E,T>(ap, qc, qc_tmp, ol, output_old, output_new, eb, rel_eb, radius, config->intp_param, itime_pred, stream);
}

template<typename T>
//__global__ void extrema_scan_kernel(T* data, T* result) {
__global__ void extrema_kernel(T *in, size_t const len, T *minel, T *maxel, T const failsafe, int const R) {
    extern __shared__ float sdata[];  // 动态共享内存

    __shared__ T shared_minv, shared_maxv;
    T tp_minv, tp_maxv;

    auto entry = (blockDim.x * R) * blockIdx.x + threadIdx.x;
    auto _idx = [&](auto r) { return entry + (r * blockDim.x); };

    // failsafe; require external setup
    tp_minv = failsafe, tp_maxv = failsafe;
    if (threadIdx.x == 0) shared_minv = failsafe, shared_maxv = failsafe;

    __syncthreads();

    for (auto r = 0; r < R; r++) {
    auto idx = _idx(r);
    if (idx < len) {
        auto val = in[idx];

        tp_minv = min(tp_minv, val);
        tp_maxv = max(tp_maxv, val);
    }
    }
    __syncthreads();

    atomicMinFp<T>(&shared_minv, tp_minv);
    atomicMaxFp<T>(&shared_maxv, tp_maxv);
    __syncthreads();

    if (threadIdx.x == 0) {
        auto oldmin = atomicMinFp<T>(minel, shared_minv);
        auto oldmax = atomicMaxFp<T>(maxel, shared_maxv);
    }
}

template<typename T>
void extrema_scan(T* in, size_t len, T* result) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

    auto chunk = 32768;
    int nworker = 512;
    auto R = chunk / nworker;

    T h_min, h_max, failsafe;
    T *d_minel, *d_maxel;
    T *min_data, *max_data;
    CHECK_CUDA(cudaMalloc(&d_minel, sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_maxel, sizeof(T)));
    // failsafe init
    CHECK_CUDA(cudaMemcpy(&failsafe, in, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(d_minel, in, sizeof(T), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_maxel, in, sizeof(T), cudaMemcpyDeviceToDevice));
    extrema_kernel<T><<<div(len, chunk), nworker, sizeof(T) * 2, stream>>>(
     in, len, d_minel, d_maxel, failsafe, R);
    //extrema<T, 512><<< div(len, nworker * 32) , nworker, 0, stream>>>(in, len, d_minel, d_maxel);
    cudaStreamSynchronize((cudaStream_t)stream);

  // collect results
    CHECK_CUDA(cudaMemcpy(&h_min, d_minel, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_max, d_maxel, sizeof(T), cudaMemcpyDeviceToHost));

    result[0] = h_min;
    result[1] = h_max;

    CHECK_CUDA(cudaFree(d_minel));
    CHECK_CUDA(cudaFree(d_maxel));
    CHECK_CUDA(cudaFree(min_data));
    CHECK_CUDA(cudaFree(max_data));
    cudaStreamDestroy(stream);
}

__device__ uint8_t calcUncertaintyNegaBinary(unsigned int bit) {
    return (bit == 0) ? 0 : (0xaaaaaaaau >> (32 - bit)) << 1;
}

template void extrema_scan<float>(float* data, size_t len, float* result);
template void extrema_scan<double>(double* data, size_t len, double* result);

}
