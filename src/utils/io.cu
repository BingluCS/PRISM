
#include "io.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <iostream>


namespace prism {

Buffer::~Buffer() {
    if(d)
    cudaFree(d);
    if(h)
    cudaFreeHost(h);
}

template<itype file_type>
void Buffer::load_fromfile(const std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    file.seekg(0, std::ios::end);
    auto length = file.tellg();
    file.seekg(0, std::ios::beg);
    cudaError_t err = cudaMallocHost((void**)&h, length);

    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    if(file_type == ori_File) {
        if (length < bytes) {
            std::cerr << "Error: Read " << bytes << " bytes, but expected " << length << " bytes." << std::endl;
            std::cerr << "File read incomplete or corrupted." << std::endl;
            exit(1);
        }
    }
    else  bytes = length;
    // file.seekg(bytes, std::ios::beg);
    file.read(reinterpret_cast<char*>(h), bytes);
    file.close();
    // dl = DataLocation::OnHost;
}

void Buffer::unload_tofile(const std::string filename, typetofile tf) {
    if(tf == typetofile::deviceTofile) {
        cudaMallocHost((void**)&h, bytes);
        D2H();
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "can't open file" << std::endl;
        return ;
    }
    outfile.write(reinterpret_cast<char*>(h), bytes);
    std::cout <<  "decompressed file written: " << filename << '\n';
    outfile.close();
}

void Buffer::unload_tofile(const std::string filename, long long numBytes, typetofile tf) {
    if(tf == typetofile::deviceTofile) {
        cudaMallocHost((void**)&h, numBytes);
        D2H(numBytes);
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "can't open file" << std::endl;
        return ;
    }
    std::cout <<  "compressed file written: " << filename << '\n';
    outfile.write(reinterpret_cast<char*>(h), numBytes);
    outfile.close();
}

void Buffer::D2H(long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpy(h, d, numBytes, cudaMemcpyDeviceToHost));
}

void Buffer::D2H_cudaasync(void* stream, long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpyAsync(h, d, numBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

void Buffer::H2D(long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpy(d, h, numBytes, cudaMemcpyHostToDevice));
}

void Buffer::H2D_cudaasync(void* stream, long long numBytes)
{
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpyAsync(d, h, numBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

template void Buffer::load_fromfile<ori_File>(const std::string);
template void Buffer::load_fromfile<cmp_File>(const std::string);

void Bitplane::bitplane_malloc() {
    calculate_aligned_buffer_size(8);

    cudaMalloc(&aligned_strides_d, sizeof(int) * 4);
    cudaMalloc(&d, aligned_size);
    cudaMemset(d, 0, aligned_size);
    cudaMemcpy(aligned_strides_d, aligned_strides, sizeof(int) * 4, cudaMemcpyHostToDevice);
}

void Bitplane::calculate_aligned_buffer_size(size_t alignment = 8) {
    const int LEVEL = 4;
    const int segments_per_level = 32;
    aligned_size = 0;
    
    for(int level = 0; level < LEVEL; ++level) {
        size_t segment_size = strides[level];
        aligned_strides[level] = ((segment_size + alignment - 1) / alignment) * alignment;
        aligned_size += aligned_strides[level] * segments_per_level;
    }
}

}