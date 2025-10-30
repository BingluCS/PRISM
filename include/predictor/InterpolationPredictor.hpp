#pragma once

#include "compressor.hpp"


using Buffer = prism::Buffer;
template<typename T>
using StatBuffer = prism::StatBuffer<T>;
template<typename T>
using olBuffer = prism::olBuffer<T>;

template<typename T, typename E, typename FP>
void spline_construct(StatBuffer<T>* input, Buffer* anchor, Buffer* ectrl, Buffer* ectrl_tmp, olBuffer<T>* outlier, 
    double eb, double rel_eb, uint32_t radius, interpolation_parameters& intp_param, double& time, void* stream);

template<typename T, typename E, typename FP>
void spline_reconstruct(Buffer* anchor, Buffer* ectrl, Buffer* ectrl_tmp, olBuffer<T>* outlier, StatBuffer<T>* output, double eb, double rel_eb, 
uint32_t radius, interpolation_parameters& intp_param, double& time, void* stream);

template<typename T, typename E, typename FP>
void spline_progressive_reconstruct(Buffer* anchor, Buffer* ectrl, Buffer* ectrl_tmp, olBuffer<T>* outlier, StatBuffer<T>* output_old, StatBuffer<T>* output,
double eb, double rel_eb,  uint32_t radius, interpolation_parameters& intp_param, double& time, void* stream);