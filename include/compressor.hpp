
#pragma once

#include "parameters.hpp"
#include "io.hpp"
#include "timer.hpp"

namespace prism{


template<typename T>
void extrema_scan(T* data, size_t len,T* result);

template<typename T>
struct StatBuffer : Buffer{
    double minVal{0.0}, maxVal{0.0}, range{1.0};
    StatBuffer(): Buffer(){}
    template<typename... Dim>
    StatBuffer(prism_dtype dtype, Dim... args)
        : Buffer(dtype, args...)
    {}

    double findMinMax()  {
        T result[2];
        extrema_scan<T>(reinterpret_cast<T*>(d), len, result);
        minVal = result[0];
        maxVal = result[1];
        range = maxVal - minVal;
        // printf("%.20lf\n%.20lf\n%.20lf\n", maxVal, minVal, range);
        return range;
    }
};


template<typename T, typename E>
class Compressor {
    public:
        size_t len;
        size_t bytes_;
        // int splen;
        int radius;
        int align{0};
        double range{1.0};
        double time_pred{0.0}, time_bitplane{0.0}, time_encode{0.0};
        double itime_pred{0.0}, itime_bitplane{0.0}, itime_decode{0.0}, itime_enum{0.0};
        inBuffer* qc_tmp; // quantization code
        inBuffer* qc; // quantization code
        inBuffer* ap; // anchor point
        Buffer* compressed_data;
        olBuffer<T>* ol; // outlier
        // inBuffer* bitplane;
        Bitplane* bp;
        uint8_t* compressed_bp;
        int *begin, *end;
        size_t compressed_lossless_size{0};
        size_t total_compressed_size{0};
        // std::array<std::array<uint8_t*, 32>, 4> compressed_bp;
        std::array<std::array<size_t, 32>, 4> compressedSize_bp;
        size_t* compressedSize_bp_d;
        // std::array<std::array<size_t, 32>, 4> compressedSize;

        // prism_dtype dtype;
        Compressor(){};
        ~Compressor();
        void init(prism_context* config);
        void compute_align();
        //void setlocation(DataLocation dl_);
        void compress_pipeline(context* config, StatBuffer<T>* input, void* stream);
        void compress_predict(context* config, StatBuffer<T>* input, void* stream);
        void compress_merge(context* config, double input_range, void* stream);
        // void compress_output_time(StatBuffer<T>* input);

        void decompress_pipeline(context* config, StatBuffer<T>* output, void* stream);
        template<int init_ap>
        void decompress_scatter(context* config, void* stream);
        void decompress_predict(context* config, StatBuffer<T>* output, void* stream);

        void decompress_progressive_pipeline(context* config, StatBuffer<T>* output_old, StatBuffer<T>* output_new, 
        double targetError, double lastError, void* stream);
        void decompress_progressive_predict(context* config, StatBuffer<T>* output_old, StatBuffer<T>* output_new, void* stream);
};


// template class Compressor<float, u1>;
// template class Compressor<float, u2>;
// template class Compressor<float, u4>;
template class Compressor<float, i4>;
template class Compressor<double, i4>;

}