#include "dataloader.hpp"
#include "traits.hpp"
#include "timer.hpp"
#include "err.hpp"

std::array<int, 4> lastcache{};
__device__ unsigned long long value_blk[65] = {0}; 
__device__ int maxbid[65] = {0};  
__device__ int target_blk[65][4] = {0};
__device__ int blockCounter = 0; 
__device__ int maxlevel = 4; 

__global__ void resetCounter() {
    maxlevel = 4;
    blockCounter = 0;
    // for(int i = 0; i < 65; ++i) {
    //     value_blk[i] = maxbid[i] = target_blk[i][0] = target_blk[i][1] = target_blk[i][2] = target_blk[i][3] =0;
    // }
}

__global__ void loadall(int* begin, int* end) {
    begin[0] = end[0];
    begin[1] = end[1];
    begin[2] = end[2];
    begin[3] = end[3];
    end[0] = end[1] = end[2] = end[3] = 32;
    maxlevel = begin[0] < 32 ? 3 : begin[1] < 32 ? 2 : begin[2] < 32 ? 1 : begin[3] < 32 ? 0 : 4;
}

template<typename E, int LEVEL>
static __global__ void findOptimalStrategy(size_t* compressedSize_bp_d, int* begin, int* end, size_t targetCost) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int tablesize = sizeof(E) * 8 + 1;
    const int TPB = blockDim.x ;

    __shared__ uint64_t cost[LEVEL][tablesize];
    __shared__ uint64_t valueTable[LEVEL][tablesize];
    __shared__ unsigned long long maxValue[1024];
    __shared__ int maxtid[1024];

    int r_end[4] = {32, 32, 32, 32};

    if(tid == 0) {
        #pragma unroll
        for (int i = 3; i >= 0; --i) {
            valueTable[i][0] = compressedSize_bp_d[i * 32 + 31];
            valueTable[i][32] = 0;
             #pragma unroll
            for (int j = 31; j > 0; --j) {
                valueTable[i][j] = compressedSize_bp_d[i * 32 + 31] - compressedSize_bp_d[i * 32 + j - 1];
            }
        }
        maxbid[bid] = bid;
        value_blk[bid] = 0;
        target_blk[bid][0] = target_blk[bid][1] = target_blk[bid][2] = target_blk[bid][3] = 32;
    }

    if(tid < LEVEL * tablesize) {
        int l = tid / tablesize;
        int b = tid % tablesize;
        auto bitshift = 32 - b;
        double error = 0;
        error = bitshift == 0 ? 0 :  (1ULL << bitshift);
        double p = l == 0 ? 1.0 : 1.25;
        error *= (pow(p, LEVEL * 3 - l * 3 - 1) + pow(p, LEVEL * 3 - l * 3 - 2) 
            + pow(p, LEVEL * 3 - l * 3 - 3));
        cost[l][b] = (uint64_t)ceil(error);
    }
    
    __syncthreads();
    
    const int b0 = blockIdx.x;
    unsigned long long maxSize = 0;
    #pragma unroll
    for(int pos = tid; pos < 33 * 33 * 33; pos += TPB) {
        int b1 = pos / (tablesize * tablesize);
        int b2 = (pos / tablesize) % tablesize;
        int b3 = pos % tablesize;

        size_t totalCost = cost[0][b0] + cost[1][b1] + cost[2][b2] + cost[3][b3];
        if(totalCost <=  targetCost) {
            unsigned long long value = 0;
            value += valueTable[0][b0] + valueTable[1][b1] + valueTable[2][b2] + valueTable[3][b3];
            if(value > maxSize) {
                r_end[0] = b0;
                r_end[1] = b1;
                r_end[2] = b2;
                r_end[3] = b3;
                maxSize = value;

            }
        }
    }
    maxtid[tid] = tid;
    maxValue[tid] = maxSize;
    __syncthreads();

    #pragma unroll
    #pragma unroll
    for (int stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if(maxValue[tid] < maxValue[tid + stride]) {
                maxValue[tid] =  maxValue[tid + stride];
                maxtid[tid] =  maxtid[tid + stride];
            }
        }
        __syncthreads();
    }

    if(tid == maxtid[0]) {
        value_blk[bid] = maxValue[0];
        target_blk[bid][0] = r_end[0];
        target_blk[bid][1] = r_end[1];
        target_blk[bid][2] = r_end[2];
        target_blk[bid][3] = r_end[3];
        atomicAdd(&blockCounter, 1);
    }

    if(bid == 0) {
        while (atomicAdd(&blockCounter, 0) < gridDim.x) {
        }
        #pragma unroll
        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if(value_blk[tid] < value_blk[tid + stride]) {
                    value_blk[tid] =  value_blk[tid + stride];
                    maxbid[tid] =  maxbid[tid + stride];
                }
            }
            __syncthreads();
        }   
    }
    
    if(bid == 0 && tid == 0) {
        begin[0] = end[0];
        begin[1] = end[1];
        begin[2] = end[2];
        begin[3] = end[3];
        if(begin[3] < target_blk[maxbid[0]][3]) {
            end[3] =  target_blk[maxbid[0]][3];
            maxlevel = 0;
        }
        else end[3] = begin[3];

        if(begin[2] < target_blk[maxbid[0]][2]) {
            end[2] =  target_blk[maxbid[0]][2];
            maxlevel = 1;
        }
        else end[2] = begin[2];

        if(begin[1] < target_blk[maxbid[0]][1]) {
            end[1] =  target_blk[maxbid[0]][1];
            maxlevel = 2;
        }
        else end[1] = begin[1];


        if(begin[0] < target_blk[maxbid[0]][0]) {
            end[0] =  target_blk[maxbid[0]][0];
            maxlevel = 3;
        }
        else end[0] = begin[0];
        // printf("maxlevel: %d\n", maxlevel);
        // end[1] = begin[1] < target_blk[maxbid[0]][1] ? target_blk[maxbid[0]][1] : begin[1];
        // end[2] = begin[2] < target_blk[maxbid[0]][2] ? target_blk[maxbid[0]][2] : begin[2];
        // end[3] = begin[3] < target_blk[maxbid[0]][3] ? target_blk[maxbid[0]][3] : begin[3];
        // printf("begin: %d %d %d %d\n", begin[0],begin[1], begin[2], begin[3]);
        // printf("end: %d %d %d %d\n", end[0],end[1], end[2], end[3]);
        // printf("%d %d %d %d %llu\n", target_blk[maxbid[0]][0], target_blk[maxbid[0]][1],
        //     target_blk[maxbid[0]][2],target_blk[maxbid[0]][3], value_blk[0]);
        // printf("%llu\n", valueTable[0][32] + valueTable[1][32] +
        //     valueTable[2][32] + valueTable[3][31]);
        // printf("%llu %lld\n", cost[0][32] + cost[1][32] +
        //     cost[2][32] + cost[3][31], targetCost);
    }
}

template<typename E>
void findStrategy_h(size_t* compressedSize_bp_d, int* begin, int* end, double eb, double targetError, double& time, void* stream) {
    size_t targetCost = (size_t)floor(targetError * 0.954  / eb) - 1;
    int *begin_tmp, *end_tmp;
    cudaMalloc(&begin_tmp, 1 * 4 * sizeof(int));
    cudaMalloc(&end_tmp, 1 * 4 * sizeof(int));

    resetCounter<<<1,1>>>();
    findOptimalStrategy<E, 4><<<33, 1024>>>(compressedSize_bp_d, begin_tmp, end_tmp, targetCost);
    loadall<<<1,1>>>(begin_tmp, end_tmp); //warm up

    resetCounter<<<1,1>>>();
    GPUTimer dtimer;
    dtimer.start(stream);
    if(targetError * 0.954  >= eb) {
        findOptimalStrategy<E, 4><<<33, 1024, 0, (cudaStream_t)stream>>> (compressedSize_bp_d, 
            begin, end, targetCost); // 4 * (sizeof(E) * 8 + 1
    }
    else {
        loadall<<<1,1>>>(begin, end);
    }

    time = dtimer.stop(stream);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA strategy kernel launch error: %s\n", cudaGetErrorString(err));
    // }

    // CHECK_CUDA(cudaMemcpy(&lastcache[0],  end, 1 * 4 * sizeof(int), cudaMemcpyDeviceToHost));
}


#define Strategy(E) \
template void findStrategy_h<E>(size_t* compressedSize_bp_d, int* begin, int* end,double eb, double targetError, double& time, void* stream);
// template static __global__ void findOptimalStrategy(size_t* compressedSize_bp_d, int* begin, int* end, size_t targetCost);

Strategy(i4)
// Strategy(u1)