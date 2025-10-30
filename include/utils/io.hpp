
#pragma once

#include "parameters.hpp"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <array>

void* readData(char* srcFilePath);

namespace prism{

enum class typetofile {
    hostTofile,
    deviceTofile
};


typedef struct Buffer {
    prism_dtype dtype;
    int tsize, ndim{-1};
    size_t len{1}, bytes{1};
    uint32_t lx{1}, ly{1}, lz{1};
    size_t sty{1}, stz{1};  // stride
    int align{0};
    //DataLocation dl;
    void* d;
    void* h;
    Buffer() {};
    ~Buffer();
    template<itype file_type>
    void load_fromfile(const std::string filename);
    void unload_tofile(const std::string filename, typetofile tf = typetofile::hostTofile);
    void unload_tofile(const std::string  filename, long long numBytes,typetofile tf = typetofile::hostTofile);
    void H2D(long long numBytes = -1);
    void D2H(long long numBytes = -1);
    void H2D_cudaasync(void* stream, long long numBytes = -1);
    void D2H_cudaasync(void* stream, long long numBytes = -1);
    
    template<typename D>
    D len3() const {
        return D(lx, ly, lz);
    }
    template<typename D>
    D st3() const {
        return D(1, sty, stz);
    }
    template<typename... Dim>
    Buffer(prism_dtype type, int align_, Dim... args):dtype(type), align(align_) {
        uint32_t tmp[] = { static_cast<uint32_t>(args)... };
        switch(sizeof...(Dim)) {
            // case 4: lw = tmp[3];
            case 3: lz = tmp[2];
            case 2: ly = tmp[1];
            case 1: lx = tmp[0]; break;
            default:
                //printf("Dim error!\n");
                break;
        }
        ndim = 3;
        if (lz == 1) ndim = 2;
        if (ly == 1) ndim = 1;
        tsize = dtype % 10;
        len = lx * ly * lz;
        sty = lx;
        stz = lx * ly;
        bytes = tsize * len + align;
        CHECK_CUDA(cudaMallocHost(&h, bytes));
        CHECK_CUDA(cudaMalloc(&d, bytes));
        CHECK_CUDA(cudaMemset(d, 0, bytes));
    }

} inBuffer;

struct Bitplane {
    // uint8_t** d;
    // uint8_t** h;
    uint8_t* data;
    uint8_t* data_h;
    int* aligned_strides_d;
    uint32_t lx{1}, ly{1}, lz{1};
    size_t len{1}, aligned_len{1};
    size_t ori_size{1}, aligned_size{1};
    int align{0};
    int strides[4]{0, 0, 0, 0};
    int aligned_strides[4]{0, 0, 0, 0};
    int prefix_nums[4]{0, 0, 0, 0};
    int compressed_size[4][32];

    Bitplane(int x, int y, int z) : lx(x), ly(y), lz(z) {
        aligned_len = len = lx * ly * lz;
        ori_size = aligned_size = len * 4;
        h_comput<4>();
        bitplane_malloc<4>();
    }
    ~Bitplane() {
        // if (d) cudaFree(d);
        // if (h) cudaFreeHost(h);
        if (data) cudaFree(data);
        if (data_h) cudaFree(data_h);
    }
    template<int LEVEL>
    void bitplane_malloc() {
        aligned_size = calculate_aligned_buffer_size<LEVEL>();

        // cudaMallocHost(&h, sizeof(uint8_t*) * LEVEL * 32);
        // cudaMalloc(&d, sizeof(uint8_t*) * LEVEL * 32);
        cudaMalloc(&aligned_strides_d, sizeof(int) * 4);
        cudaError_t err = cudaMalloc(&data, aligned_size);
        cudaMemset(data, 0, aligned_size);
        // if (err != cudaSuccess) {
        //     printf("Failed to allocate aligned buffer: %s\n", cudaGetErrorString(err));
        // }

        // uint8_t* tmp = data;
        // for(int l = LEVEL - 1; l >= 0; --l) {
        //     tmp += aligned_strides[l] * 32;
        //     for(int b = 0; b < 32; ++b) {
        //         h[l * 32 + b] = tmp + strides[l] * b;
        //     }
        // }
        // for(int i = 0; i < LEVEL; ++i)
        //     printf("aligned_strides: %d, %d, %d\n", i, aligned_strides[i], strides[i]);
        // cudaMemcpy(d, h, sizeof(uint8_t*) * LEVEL * 32, cudaMemcpyHostToDevice);
        cudaMemcpy(aligned_strides_d, aligned_strides, sizeof(int) * 4, cudaMemcpyHostToDevice);
    }


    template<int LEVEL>
    inline size_t calculate_aligned_buffer_size(size_t alignment = 8) {
        const int segments_per_level = 32;
        size_t total_size = 0;
        
        for(int level = 0; level < LEVEL; ++level) {
            size_t segment_size = strides[level];
            aligned_strides[level] = ((segment_size + alignment - 1) / alignment) * alignment;
            total_size += aligned_strides[level] * segments_per_level;
        }
        
        return total_size;
    }

    template <int LEVEL> __forceinline__ void h_comput(){
        dim3 d_size = dim3(lx, ly, lz);
        int level = 0;
        while(level < LEVEL){
            d_size.x = (d_size.x + 1) >> 1;
            d_size.y = (d_size.y + 1) >> 1;
            d_size.z = (d_size.z + 1) >> 1;
            prefix_nums[level] = d_size.x * d_size.y * d_size.z;
            ++level;
        }
        prefix_nums[LEVEL] = 0;
        prefix_nums[0] -=  prefix_nums[3];
        prefix_nums[1] -=  prefix_nums[3];
        prefix_nums[2] -=  prefix_nums[3];
        prefix_nums[3] -=  prefix_nums[3];

        for(int i = LEVEL - 2; i >= 0; --i) {
            align += (8 - ((prefix_nums[i] - prefix_nums[i+1] + align) % 8)) % 8;
            prefix_nums[i] += align;
        }

        align += (8 - ((len - prefix_nums[0] + align) % 8)) % 8;

        strides[3] = (prefix_nums[2] - prefix_nums[3])/ 8;
        strides[2] = (prefix_nums[1] - prefix_nums[2])/ 8;
        strides[1] = (prefix_nums[0] - prefix_nums[1])/ 8;
        strides[0] = (len - prefix_nums[0] + align)/ 8;
        aligned_len += align;
        
    }
    void unload_tofile(const char* filename, typetofile tf) {
        if(tf == typetofile::deviceTofile) {
            cudaMallocHost((void**)&data_h, aligned_size);
            CHECK_CUDA(cudaMemcpy(data_h, data, aligned_size, cudaMemcpyDeviceToHost));
        }
        // std::cout << aligned_size << std::endl;
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            std::cerr << "can't open file" << std::endl;
            return ;
        }
        outfile.write(reinterpret_cast<char*>(data_h), aligned_size);
        outfile.close();
    }

};


template<typename T>
struct olBuffer : Buffer { ///outlier
    uint32_t *d_idx, *h_idx;
    uint32_t *d_num, h_num{0};

    olBuffer(){}

    template<typename... Dim>
    olBuffer(prism_dtype dtype, Dim... args)
        : Buffer(dtype, args...)
    {
        cudaMalloc(&d_idx, sizeof(uint32_t) * len);
        cudaMalloc(&d_num, sizeof(uint32_t) * 1);
        cudaMemset(d_num, 0x0, sizeof(uint32_t) * 1);
        cudaMallocHost(&h_idx, sizeof(uint32_t) * len);

    }
    
    ~olBuffer() {
        if (d_idx) cudaFree(d_idx);
        if (d_num) cudaFree(d_num);
        if (h_idx) cudaFreeHost(h_idx);
    }
};
}