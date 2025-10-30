#include "lossless.hpp"
#include "timer.hpp"
#include "io.hpp"
#include "assert.h"
#include "d_zero_elimination.hpp"
#include "zbpc.hpp"
#include "RZE.hpp"

#define DEFAULT_BLOCK_SIZE 384
using Bitplane = prism::Bitplane;
using byte = uint8_t;

static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}

static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


static __device__ unsigned long long g_chunk_counter;

static __global__ void d_reset()
{
  g_chunk_counter = 0LL;
}

static inline __device__ void propagate_block(const int value, const long long chunkID, volatile int* const __restrict__ fullcarry, long long* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? (long long)value : (long long)-value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const long long cidm1ml = chunkID - 1 - lane;
      long long val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(0xFFFFFFFF, val == 0)) || (__all_sync(0xFFFFFFFF, val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot_sync(0xFFFFFFFF, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot_sync(0xFFFFFFFF, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      long long partc = (lane < pos) ? -val : 0;
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 1);
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 2);
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 4);
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 8);
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 16);
#if defined(WS) && (WS == 64)
      partc += __shfl_xor_sync(0xFFFFFFFF, partc, 32);
#endif
      if (lane == pos) {
        const long long fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void zbpc_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, 
    size_t* const __restrict__ outsize, int aligned_strides[4], size_t* const __restrict__ compressedSize_bp_d, 
    int* const __restrict__ fullcarry) {

    // allocate shared memory buffer
    __shared__ long long chunk [3 * (CS / sizeof(long long))];
    // split into 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];
    
    // initialize
    const int tid = threadIdx.x;
    const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
    // const long long chunks = (insize + CS - 1) / CS;  // round up

    const long long chunks_level[4] = {(aligned_strides[0] + CS - 1) / CS << 5, (aligned_strides[1] + CS - 1) / CS << 5,
    (aligned_strides[2] + CS - 1) / CS << 5, (aligned_strides[3] + CS - 1) / CS << 5};
    const long long chunks = chunks_level[0] + chunks_level[1] + chunks_level[2] + chunks_level[3];
    long long* const head_out = (long long*)output;
    unsigned short* const size_out = (unsigned short*)&head_out[1];
    byte* const data_out = (byte*)&size_out[chunks];
        
    const int LEVELS = 3;
    // loop over chunks
    do {
        // assign work dynamically
        if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();  // chunk[last] produced, chunk consumed

        // terminate if done
        const long long chunkID = chunk[last];
        long long realID = chunkID;
        if(realID >= chunks) 
            break;
        int l = LEVELS;
        while(realID >= chunks_level[l]) {
            realID -= chunks_level[l--];
        }
        int chunks_bp = chunks_level[l] >> 5;
        long long base = 0;
        int bp_id = realID / chunks_bp;
        int bp_offset = realID % chunks_bp;
        for(int i = LEVELS; i > l; --i)
            base += aligned_strides[i] << 5; 
        base +=  bp_id * aligned_strides[l] + bp_offset * CS;
        const int osize = (int)min(CS, aligned_strides[l] - bp_offset * CS);
        //  const long long base = chunkID * CS;
        // if (base >= insize) break;
        // const int osize = (int)min((long long)CS, insize - base);
        
        long long* const input_l = (long long*)&input[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
            out_l[i] = input_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];

        // encode chunk
        __syncthreads();  // chunk produced, chunk[last] consumed
        int csize = osize;
        bool good = true;
        if (good) {
            byte* tmp = in; in = out; out = tmp;
            good = d_ZBPC_1(csize, in, out, temp);
            __syncthreads();
        }

        // handle carry
        if (!good || (csize >= osize)) csize = osize;
        propagate_block(csize, chunkID, fullcarry, (long long*)temp);

        // reload chunk if incompressible
        if (tid == 0) size_out[chunkID] = csize;
        if (csize == osize) {
        // store original data
            long long* const out_l = (long long*)out;
            for (long long i = tid; i < osize / 8; i += TPB) {
                out_l[i] = input_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
        }
        __syncthreads();  // "out" done, temp produced

        // store chunk
        const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
        s2g(&data_out[offs], out, csize);

        if(tid == 0 && bp_offset == chunks_bp -1) {
            compressedSize_bp_d[(3 - l) * 32 + bp_id] = fullcarry[chunkID];
        }
        // finalize if last chunk
        // if ((tid == 0) && (base + CS >= insize)) {
        if ((tid == 0) && chunkID + 1 == chunks) {
            // output header
            head_out[0] = insize;
            // compute compressed size
            *outsize = &data_out[fullcarry[chunkID]] - output;
            
        }
    } while (true);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void zbpc_decode(const byte* const __restrict__ input, byte* const __restrict__ output, 
    int* const __restrict__ g_outsize, int aligned_strides[4])
{
    // allocate shared memory buffer
    __shared__ long long chunk [3 * (CS / sizeof(long long))];
    const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

    // input header
    long long* const head_in = (long long*)input;
    const long long outsize = head_in[0];

    // initialize
    //   const long long chunks = (outsize + CS - 1) / CS;  // round up

    const long long chunks_level[4] = {(aligned_strides[0] + CS - 1) / CS << 5, (aligned_strides[1] + CS - 1) / CS << 5,
    (aligned_strides[2] + CS - 1) / CS << 5, (aligned_strides[3] + CS - 1) / CS << 5};
    const long long chunks = chunks_level[0] + chunks_level[1] + chunks_level[2] + chunks_level[3];

    unsigned short* const size_in = (unsigned short*)&head_in[1];
    byte* const data_in = (byte*)&size_in[chunks];

  
    // loop over chunks
    const int tid = threadIdx.x;
    long long prevChunkID = 0;
    long long prevOffset = 0;
    do {
        // assign work dynamically
        if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();  // chunk[last] produced, chunk consumed

        // terminate if done
        // const long long chunkID = chunk[last];
        // const long long base = chunkID * CS;
        // if (base >= outsize) break;
        const long long chunkID = chunk[last];
        if(chunkID >= chunks) 
            break;
        long long realID = chunkID;
        int l = 3;
        while(realID >= chunks_level[l]) {
            realID -= chunks_level[l--];
        }
        long long chunks_bp = chunks_level[l] >> 5;
        long long base = 0;
        long long bp_id = realID / chunks_bp;
        long long bp_offset = realID % chunks_bp;
        for(int i = 3; i > l; --i)
            base += aligned_strides[i] << 5; 
        base +=  bp_id * aligned_strides[l] + bp_offset * CS;

        // compute sum of all prior csizes (start where left off in previous iteration)
        long long sum = 0;
        for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
        sum += (long long)size_in[i];
        }
        int csize = (int)size_in[chunkID];
        const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
        prevChunkID = chunkID;
        prevOffset = offs;

        // create the 3 shared memory buffers
        byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
        byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
        byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

        // load chunk
        g2s(in, &data_in[offs], csize, out);
        byte* tmp = in; in = out; out = tmp;
        __syncthreads();  // chunk produced, chunk[last] consumed

        // decode
        const int osize = (int)min((long long)CS, aligned_strides[l] - bp_offset * CS);

        // const int osize = (int)min((long long)CS, outsize - base);
        if (csize < osize) {
            byte* tmp;
            tmp = in; in = out; out = tmp;
            d_iZBPC_1(csize, in, out,temp);
            __syncthreads();
        }

        // if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
        long long* const output_l = (long long*)&output[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
            output_l[i] = out_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void zbpc_decode_progressive(const byte* const __restrict__ input, byte* const __restrict__ output, 
    long long* const __restrict__ g_outsize, int aligned_strides[4], int* begin, int* end)
{
    // allocate shared memory buffer
    __shared__ long long chunk [3 * (CS / sizeof(long long))];
    const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

    // input header
    long long* const head_in = (long long*)input;
    const long long outsize = head_in[0];

    // initialize
    //   const long long chunks = (outsize + CS - 1) / CS;  // round up

    const long long chunks_level[4] = {(aligned_strides[0] + CS - 1) / CS << 5, (aligned_strides[1] + CS - 1) / CS << 5,
    (aligned_strides[2] + CS - 1) / CS << 5, (aligned_strides[3] + CS - 1) / CS << 5};
    const long long chunks = chunks_level[0] + chunks_level[1] + chunks_level[2] + chunks_level[3];

    unsigned short* const size_in = (unsigned short*)&head_in[1];
    byte* const data_in = (byte*)&size_in[chunks];

  
    // loop over chunks
    const int tid = threadIdx.x;
    long long prevChunkID = 0;
    long long prevOffset = 0;
    do {
        // assign work dynamically
        if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();  // chunk[last] produced, chunk consumed

        // terminate if done
        // const long long chunkID = chunk[last];
        // const long long base = chunkID * CS;
        // if (base >= outsize) break;
        const long long chunkID = chunk[last];
        if(chunkID >= chunks) 
            break;
        long long realID = chunkID;
        int l = 3;
        while(realID >= chunks_level[l]) {
            realID -= chunks_level[l--];
        }
        long long chunks_bp = chunks_level[l] >> 5;

        long long base = 0;
        long long bp_id = realID / chunks_bp;
        if((realID < begin[3-l] * chunks_bp  || realID >= end[3-l] * chunks_bp)) { //  && (realID >= chunks_bp)

            continue;
        }
        long long bp_offset = realID % chunks_bp;
        for(int i = 3; i > l; --i)
            base += aligned_strides[i] << 5; 
        base +=  bp_id * aligned_strides[l] + bp_offset * CS;

        // compute sum of all prior csizes (start where left off in previous iteration)
        long long sum = 0;
        for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
        sum += (long long)size_in[i];
        }
        int csize = (int)size_in[chunkID];
        const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
        prevChunkID = chunkID;
        prevOffset = offs;

        // create the 3 shared memory buffers
        byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
        byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
        byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

        // load chunk
        g2s(in, &data_in[offs], csize, out);
        byte* tmp = in; in = out; out = tmp;
        __syncthreads();  // chunk produced, chunk[last] consumed

        // decode
        const int osize = (int)min((long long)CS, aligned_strides[l] - bp_offset * CS);

        // const int osize = (int)min((long long)CS, outsize - base);
        if (csize < osize) {
            byte* tmp;
            tmp = in; in = out; out = tmp;
            d_iZBPC_1(csize, in, out,temp);
            __syncthreads();
        }

        // if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
        long long* const output_l = (long long*)&output[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
        output_l[i] = out_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


size_t lossless_encode(Bitplane* bp,  uint8_t*& compressed_bp, size_t*& compressedSize_bp_d, size_t ori_size, double& time, void* stream) {

    // const long long chunks = (bp->aligned_size + CS - 1) / CS;  // round up
    const long long chunks_level[4] = {(bp->strides[0] + CS - 1) / CS << 5, (bp->strides[1] + CS - 1) / CS << 5,
    (bp->strides[2] + CS - 1) / CS << 5, (bp->strides[3] + CS - 1) / CS << 5};

    const long long chunks = chunks_level[0] + chunks_level[1] + chunks_level[2] + chunks_level[3];
    
    const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS + 7;
    
    byte* d_encoded;
    cudaMalloc((void **)&d_encoded, maxsize);
    cudaMemset(d_encoded, 0, maxsize);
    size_t* d_encsize;
    int* d_fullcarry;
    // size_t* compressedSize_bp_d;

    cudaMalloc((void **)&d_encsize, sizeof(size_t));
    cudaMalloc((void **)&d_fullcarry, chunks * sizeof(int));

    d_reset<<<1, 1>>>();
    cudaMemset(d_fullcarry, 0, chunks * sizeof(int));

    GPUTimer dtimer;
    dtimer.start(stream);
    zbpc_encode<<<DEFAULT_BLOCK_SIZE, TPB, 0, (cudaStream_t)stream>>>
    ((uint8_t*)bp->data, bp->aligned_size, d_encoded, d_encsize, bp->aligned_strides_d, compressedSize_bp_d, d_fullcarry);
    time = dtimer.stop(stream);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA encode kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    size_t dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(size_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&compressedSize_bp[0][0], compressedSize_bp_d, 4 * 32 * sizeof(size_t), cudaMemcpyDeviceToHost);


    size_t padding = (8 - (dencsize % 8)) % 8;

    // Round up size to 8-byte alignment
    dencsize += padding;
    compressed_bp = d_encoded;

    return dencsize;
}


size_t lossless_decode(uint8_t*& compressed_bp, Bitplane* bp, size_t ori_size, double& time, void* stream) {

    int pre_size;
    uint8_t* input = compressed_bp;
    void** output = (void**)(&bp->data);
    cudaMemcpy(&pre_size, input, sizeof(long long), cudaMemcpyDeviceToHost);

    byte* d_decoded;
    cudaMalloc((void **)&d_decoded, pre_size);
    int* d_decsize;
    cudaMalloc((void **)&d_decsize, sizeof(int));
    
    // warm up
    // byte* d_decoded_dummy;
    // cudaMalloc((void **)&d_decoded_dummy, pre_size);
    // int* d_decsize_dummy;
    // cudaMalloc((void **)&d_decsize_dummy, sizeof(int));
    // zbpc_decode<<<DEFAULT_BLOCK_SIZE, TPB>>>(input, d_decoded_dummy, d_decsize_dummy, bp->aligned_strides_d);
    // cudaFree(d_decoded_dummy);
    // cudaFree(d_decsize_dummy);

    // time GPU decoding
    GPUTimer dtimer;
    dtimer.start(stream);
    d_reset<<<1, 1>>>();
    zbpc_decode<<<DEFAULT_BLOCK_SIZE, TPB, 0, (cudaStream_t)stream>>>(input, d_decoded, d_decsize, bp->aligned_strides_d);
    time = dtimer.stop(stream);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA decode kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    
    *output = d_decoded;
    return 0;
}

size_t lossless_decode_progressive(uint8_t*& compressed_bp, Bitplane* bp, size_t ori_size, int* begin, int* end,
double& time, void* stream) {

    int pre_size;
    uint8_t* input = compressed_bp;
    void** output = (void**)(&bp->data);
    cudaMemcpy(&pre_size, input, sizeof(long long), cudaMemcpyDeviceToHost);

    // CheckCuda(__LINE__);
    byte* d_decoded;
    cudaMalloc((void **)&d_decoded, pre_size);
    cudaMemset(d_decoded, 0, pre_size);
    long long* d_decsize;
    cudaMalloc((void **)&d_decsize, sizeof(long long));
    // CheckCuda(__LINE__);
    
    // warm up
    // byte* d_decoded_dummy;
    // cudaMalloc((void **)&d_decoded_dummy, pre_size);
    // long long* d_decsize_dummy;
    // cudaMalloc((void **)&d_decsize_dummy, sizeof(long long));
    // zbpc_decode_progressive<<<DEFAULT_BLOCK_SIZE, TPB>>>(input, d_decoded_dummy, d_decsize_dummy, bp->aligned_strides_d, begin, end);
    // cudaFree(d_decoded_dummy);
    // cudaFree(d_decsize_dummy);

    // time GPU decoding
    GPUTimer dtimer;
    dtimer.start(stream);
    d_reset<<<1, 1>>>();
    zbpc_decode_progressive<<<DEFAULT_BLOCK_SIZE, TPB, 0, (cudaStream_t)stream>>>(input, d_decoded, d_decsize, bp->aligned_strides_d, begin, end);
    time = dtimer.stop(stream);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA progressive decode kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    *output = d_decoded;
    return 0;
}