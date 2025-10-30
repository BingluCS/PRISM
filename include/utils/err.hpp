#pragma once
#include <iostream>
#define CHECK_CUDA(call)                                                          \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " - "\
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)
