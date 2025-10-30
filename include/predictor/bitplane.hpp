#pragma once

#include "io.hpp"
#include <cuda_runtime.h>

using Buffer = prism::Buffer;
using Bitplane = prism::Bitplane;
using byte = unsigned char;
// #define NB
// #define SM
// #define SA
template <int LEVEL> __forceinline__ __device__ void d_comput(int size, dim3 data_size, int& align,
 volatile int prefix_nums[LEVEL], volatile int aligned_prefix_nums[LEVEL], volatile int shmem_stride[LEVEL]);

template<typename E, btype bt>
bool convert_to_bitplane(Buffer* qc, Bitplane* bitplane, size_t anchor_size, double& time, void* stream);

template<typename E, btype bt>
bool inverse_convert_to_bitplane(Bitplane* bitplane, Buffer* qc, size_t anchor_size, double& time, void* stream);

template<typename E, btype bt>
bool inverse_convert_to_bitplane_progressive(Bitplane* bitplane, Buffer* qc, size_t anchor_size, 
int* begin, int* end, double& time, void* stream);