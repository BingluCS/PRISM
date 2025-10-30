#pragma once

#include "io.hpp"
#include "lossless.hpp"
#include "timer.hpp"

using Bitplane = prism::Bitplane;



size_t lossless_encode(Bitplane* bp,  uint8_t*& compressed_bp, size_t*& compressedSize_bp_d, size_t ori_size, double& time, void* stream);

size_t lossless_decode(uint8_t*& compressed_bp, Bitplane* bp, size_t ori_size, double& time, void* stream);

size_t lossless_decode_progressive(uint8_t*& compressed_bp, Bitplane* bp, size_t ori_size, int* begin, int* end,
double& time, void* stream);