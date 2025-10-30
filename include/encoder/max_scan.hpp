#pragma once
#include "lossless.hpp"

template <typename T>
static __device__ inline T block_max_scan(T val, void* buffer)  // returns inclusive maximum scan
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const carry = (T*)buffer;
  assert(WS >= warps);

  T tmp = __shfl_up_sync(0xFFFFFFFF, val, 1);
  if (lane >= 1) val = max(val, tmp);
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 2);
  if (lane >= 2) val = max(val, tmp);
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 4);
  if (lane >= 4) val = max(val, tmp);
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 8);
  if (lane >= 8) val = max(val, tmp);
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 16);
  if (lane >= 16) val = max(val, tmp);
#if defined(WS) && (WS == 64)
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 32);
  if (lane >= 32) val = max(val, tmp);
#endif

  if (lane == WS - 1) carry[warp] = val;
  __syncthreads();  // carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      T res = carry[lane];
      T tmp = __shfl_up_sync(0xFFFFFFFF, res, 1);
      if (lane >= 1) res = max(res, tmp);
      if constexpr (warps > 2) {
        tmp = __shfl_up_sync(0xFFFFFFFF, res, 2);
        if (lane >= 2) res = max(res, tmp);
        if constexpr (warps > 4) {
          tmp = __shfl_up_sync(0xFFFFFFFF, res, 4);
          if (lane >= 4) res = max(res, tmp);
          if constexpr (warps > 8) {
            tmp = __shfl_up_sync(0xFFFFFFFF, res, 8);
            if (lane >= 8) res = max(res, tmp);
            if constexpr (warps > 16) {
              tmp = __shfl_up_sync(0xFFFFFFFF, res, 16);
              if (lane >= 16) res = max(res, tmp);
              #if defined(WS) && (WS == 64)
              if constexpr (warps > 32) {
                tmp = __shfl_up_sync(0xFFFFFFFF, res, 32);
                if (lane >= 32) res = max(res, tmp);
              }
              #endif
            }
          }
        }
      }
      carry[lane] = res;
    }
    __syncthreads();  // carry updated

    if (warp > 0) val = max(val, carry[warp - 1]);
  }

  return val;
}
