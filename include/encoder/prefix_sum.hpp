#pragma once

#include "lossless.hpp"
template <typename T>
static __device__ inline T block_prefix_sum(T val, void* buffer)  // returns inclusive prefix sum
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const carry = (T*)buffer;
  assert(WS >= warps);

  T tmp = __shfl_up_sync(0xFFFFFFFF, val, 1);
  if (lane >= 1) val += tmp;
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 2);
  if (lane >= 2) val += tmp;
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 4);
  if (lane >= 4) val += tmp;
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 8);
  if (lane >= 8) val += tmp;
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 16);
  if (lane >= 16) val += tmp;
#if defined(WS) && (WS == 64)
  tmp = __shfl_up_sync(0xFFFFFFFF, val, 32);
  if (lane >= 32) val += tmp;
#endif

  if (lane == WS - 1) carry[warp] = val;
  __syncthreads();  // carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      T sum = carry[lane];
      T tmp = __shfl_up_sync(0xFFFFFFFF, sum, 1);
      if (lane >= 1) sum += tmp;
      if constexpr (warps > 2) {
        tmp = __shfl_up_sync(0xFFFFFFFF, sum, 2);
        if (lane >= 2) sum += tmp;
        if constexpr (warps > 4) {
          tmp = __shfl_up_sync(0xFFFFFFFF, sum, 4);
          if (lane >= 4) sum += tmp;
          if constexpr (warps > 8) {
            tmp = __shfl_up_sync(0xFFFFFFFF, sum, 8);
            if (lane >= 8) sum += tmp;
            if constexpr (warps > 16) {
              tmp = __shfl_up_sync(0xFFFFFFFF, sum, 16);
              if (lane >= 16) sum += tmp;
              #if defined(WS) && (WS == 64)
              if constexpr (warps > 32) {
                tmp = __shfl_up_sync(0xFFFFFFFF, sum, 32);
                if (lane >= 32) sum += tmp;
              }
              #endif
            }
          }
        }
      }
      carry[lane] = sum;
    }
    __syncthreads();  // carry updated

    if (warp > 0) val += carry[warp - 1];
  }
  return val;
}

template <typename T>
static __device__ inline T block_sum_reduction(T val, void* buffer)  // returns sum to all threads
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  T* const s_carry = (T*)buffer;
  assert(WS >= warps);

  val += __shfl_xor_sync(0xFFFFFFFF, val, 1);  // MB: use reduction on 8.6 CC
  val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
#if defined(WS) && (WS == 64)
  val += __shfl_xor_sync(0xFFFFFFFF, val, 32);
#endif
  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      val += __shfl_xor_sync(0xFFFFFFFF, val, 1);  // MB: use reduction on 8.6 CC
      if constexpr (warps > 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
        if constexpr (warps > 4) {
          val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
          if constexpr (warps > 8) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
            if constexpr (warps > 16) {
              val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
              #if defined(WS) && (WS == 64)
              if constexpr (warps > 32) {
                val += __shfl_xor_sync(0xFFFFFFFF, val, 32);
              }
              #endif
            }
          }
        }
      }
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }

  return s_carry[0];
}
