#pragma once

// template<typename E, int LEVEL>
// static __global__
// void findOptimalStrategy(size_t* compressedSize_bp_d, int* begin, int* end, size_t targetCost);
extern __device__ int maxlevel;

template<typename E>
void findStrategy_h(size_t* compressedSize_bp_d, int* begin, int* end,double eb, double targetError, double& time, void* stream);