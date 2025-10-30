#pragma once

#include <vector>
#include <chrono>
#define CREATE_GPUEVENT_PAIR \
  cudaEvent_t start_time, end_time;          \
  cudaEventCreate(&start_time);       \
  cudaEventCreate(&end_time);

#define DESTROY_GPUEVENT_PAIR \
  cudaEventDestroy(start_time);        \
  cudaEventDestroy(end_time);

#define START_GPUEVENT_RECORDING(STREAM) \
  cudaEventRecord(start_time, (cudaStream_t)STREAM);
  
#define STOP_GPUEVENT_RECORDING(STREAM)     \
  cudaEventRecord(end_time, (cudaStream_t)STREAM); \
  cudaEventSynchronize(end_time);

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  cudaEventElapsedTime(PTR_MILLISEC, start_time, end_time);

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start(void* stream) {cudaEventRecord(beg, (cudaStream_t)stream);}
  double stop(void* stream) {cudaEventRecord(end, (cudaStream_t)stream); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return ms;}
};