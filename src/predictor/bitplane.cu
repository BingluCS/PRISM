#include "bitplane.hpp"
#include "timer.hpp"


#define WS 32
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
static const int CL = CS ;  // chunk len

template <int LEVEL> __forceinline__ __device__ void d_comput(int size, dim3 data_size, int& align,
 volatile int prefix_nums[LEVEL], volatile int aligned_prefix_nums[LEVEL], volatile int shmem_stride[LEVEL]){
    if(threadIdx.x==0){
        auto d_size = data_size;
        int level = 0;
        while(level < LEVEL){
            d_size.x = (d_size.x + 1) >> 1;
            d_size.y = (d_size.y + 1) >> 1;
            d_size.z = (d_size.z + 1) >> 1;
            prefix_nums[level] = d_size.x * d_size.y * d_size.z;
            ++level;
        } 
        prefix_nums[LEVEL] = 0;
        prefix_nums[0] -=  prefix_nums[3];
        prefix_nums[1] -=  prefix_nums[3];
        prefix_nums[2] -=  prefix_nums[3];
        prefix_nums[3] -=  prefix_nums[3];

        // shmem_stride[4] = prefix_nums[3] / 8;
        // int align = 0;
        for(int i = LEVEL - 2; i >= 0; --i) {
            align += (8 - ((prefix_nums[i] - prefix_nums[i+1] + align) % 8)) % 8;
            prefix_nums[i] += align;
        
        }
        align += (8 - ((size - prefix_nums[0] + align) % 8)) % 8;

        // shmem_stride[3] = (prefix_nums[2] - prefix_nums[3])/ 8;
        // shmem_stride[2] = (prefix_nums[1] - prefix_nums[2])/ 8;
        // shmem_stride[1] = (prefix_nums[0] - prefix_nums[1])/ 8;
        // shmem_stride[0] = (size - prefix_nums[0] + align)/ 8;

        shmem_stride[3] = (((prefix_nums[2])/ 8 - 1) / 8 + 1) * 8;
        shmem_stride[2] = (((prefix_nums[1] - prefix_nums[2])/ 8 - 1) / 8 + 1) * 8;
        shmem_stride[1] = (((prefix_nums[0] - prefix_nums[1])/ 8 - 1) / 8 + 1) * 8;
        shmem_stride[0] = (((size - prefix_nums[0] + align)/ 8 - 1) / 8 + 1) * 8;
        aligned_prefix_nums[3] = 0;
        aligned_prefix_nums[2] = shmem_stride[3] * 32;
        aligned_prefix_nums[1] = shmem_stride[2] * 32 + aligned_prefix_nums[2];
        aligned_prefix_nums[0] = shmem_stride[1] * 32 + aligned_prefix_nums[1];

    }
    __syncthreads(); 
}

template<typename E, int LEVEL, btype bt>
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// static __global__ __launch_bounds__(TPB, 3)
// #else
// static __global__ __launch_bounds__(TPB, 2)
// #endif
static __global__
void bitShuffle(E* ectrl, uint8_t* bitplane, int size, dim3 data_size) {
    __shared__ int shmem_prefix_nums[LEVEL];
    __shared__ int shmem_stride[LEVEL];
    __shared__ int aligned_prefix_nums[LEVEL];
    __shared__ unsigned int shmem_bitplane[513];
    
    int align = 0;
    d_comput<LEVEL>(size, data_size, align, shmem_prefix_nums, aligned_prefix_nums, shmem_stride);

    int start = blockIdx.x * CS;
    int tid = threadIdx.x;
    const int sublane = tid % 32;
    int* const in_w = (int*)&ectrl[start];

    if(blockIdx.x != gridDim.x - 1) {
        uint8_t* const out_w = (uint8_t*)bitplane;
        for (int pos = tid; pos < CL; pos += TPB) {
            unsigned int a = in_w[pos];
            if constexpr(bt == NB)
                a = (a + (uint32_t) 0xaaaaaaaau) ^ (uint32_t) 0xaaaaaaaau; //TCNBB
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = ((a ^ (mask_)) - mask_) ^ (a & 0x80000000); // convert to sign-maginitude
            }
            unsigned int q = __shfl_xor_sync(0xffffffff, a, 4);
            a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 2);
            a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 1);
            unsigned int mask = 0x0F0F0F0F;
            if ((sublane & 1) == 0) {
            a = (a & ~mask) | ((q >> 4) & mask);
            } else {
            a = ((q << 4) & ~mask) | (a & mask);
            }
            
            q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
            a = a ^ q ^ (q << 3);
            q = (a ^ (a >> 6)) & 0x00CC00CCU;
            a = a ^ q ^ (q << 6);
            q = (a ^ (a >> 12)) & 0x0000F0F0U;
            a = a ^ q ^ (q << 12);

            // int idx = start + pos;
            // int level = 0;
            // while(idx < shmem_prefix_nums[level]) {
            //     ++level;
            // }
            // idx -= shmem_prefix_nums[level];
            // int s = idx / 8 + shmem_stride[level] * (idx % 8) * 4 + aligned_prefix_nums[level];
            shmem_bitplane[tid] = a;
            __syncthreads();
            int write_pos = ((tid & 63) << 3) + (tid >> 6);
            int idx = start + write_pos + pos / TPB * TPB;
            int level = 0;
            while(idx < shmem_prefix_nums[level]) {
                ++level;
            }
            idx -= shmem_prefix_nums[level];
            int s = (idx >> 3) + shmem_stride[level] * ((idx & 7) << 2) + aligned_prefix_nums[level];
            a = shmem_bitplane[write_pos];
            out_w[s] = (a >> 24) & 0x000000FFU;
            out_w[s + shmem_stride[level]] = (a >> 8) & 0x000000FFU;
            out_w[s + shmem_stride[level] * 2] = (a >> 16) & 0x000000FFU;
            out_w[s + shmem_stride[level] * 3] = (a) & 0x000000FFU;

            // unsigned int q = __shfl_xor_sync(0xffffffff, a, 16);
            // a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            // q = __shfl_xor_sync(0xffffffff, a, 8);
            // a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            // q = __shfl_xor_sync(0xffffffff, a, 4);
            // unsigned int mask = 0x0F0F0F0F;
            // if ((sublane & 4) == 0) {
            // a = (a & ~mask) | ((q >> 4) & mask);
            // } else {
            // a = ((q << 4) & ~mask) | (a & mask);
            // }

            // q = __shfl_xor_sync(0xffffffff, a, 2);
            // mask = 0x33333333;
            // if ((sublane & 2) == 0) {
            // a = (a & ~mask) | ((q >> 2) & mask);
            // } else {
            // a = ((q << 2) & ~mask) | (a & mask);
            // }

            // q = __shfl_xor_sync(0xffffffff, a, 1);
            // mask = 0x55555555;
            // if ((sublane & 1) == 0) {
            //     a = (a & ~mask) | ((q >> 1) & mask);
            // } else {
            //     a = ((q << 1) & ~mask) | (a & mask);
            // }
            // a = __byte_perm(a, 0,  (0 << 12) | (1 << 8) | (2 << 4) | 3);
            // shmem_bitplane[tid] = a;
            // __syncthreads();
            // int write_pos = (tid >> 4) + ((tid & 15) << 5);
            // int idx = start + write_pos + pos / TPB * TPB;
            // int level = 0;
            // while(idx < shmem_prefix_nums[level]) {
            //     ++level;
            // }
            // idx -= shmem_prefix_nums[level];
            // int s = (idx % 32) *  (shmem_stride[level]>>2) + (idx / 32 )+  (aligned_prefix_nums[level]>>2);
            // out_w[s] = shmem_bitplane[write_pos];

        }
    }
    else {
        uint8_t* const out_w = bitplane;
        int tmp = size % CL;
        int res = align + (tmp == 0 ? CL : tmp);
        int align_res = (res + 511) / 512 * 512;
        for (int pos = tid; pos < align_res; pos += TPB) {
            unsigned int a = 0;
            if(pos < res) {
                a = in_w[pos];
            if constexpr(bt == NB)
                a = (a + (uint32_t) 0xaaaaaaaau) ^ (uint32_t) 0xaaaaaaaau; //TCNBB
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = ((a ^ (mask_)) - mask_) ^ (a & 0x80000000); // convert to sign-maginitude
            }

                unsigned active = __activemask();
                unsigned int q = __shfl_xor_sync(active, a, 4);
                a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

                q = __shfl_xor_sync(active, a, 2);
                a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

                q = __shfl_xor_sync(active, a, 1);
                unsigned int mask = 0x0F0F0F0F;
                if ((sublane & 1) == 0) {
                a = (a & ~mask) | ((q >> 4) & mask);
                } else {
                a = ((q << 4) & ~mask) | (a & mask);
                }
                
                q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
                a = a ^ q ^ (q << 3);
                q = (a ^ (a >> 6)) & 0x00CC00CCU;
                a = a ^ q ^ (q << 6);
                q = (a ^ (a >> 12)) & 0x0000F0F0U;
                a = a ^ q ^ (q << 12);
            }

            shmem_bitplane[tid] = a;
            __syncthreads();
            int write_pos = ((tid & 63) << 3) + (tid >> 6);
            if(write_pos < res) {
                int idx = start + write_pos + pos / TPB * TPB;
                int level = 0;
                while(idx < shmem_prefix_nums[level]) {
                    ++level;
                }
                idx -= shmem_prefix_nums[level];
                int s = (idx >> 3) + shmem_stride[level] * ((idx & 7) << 2) + aligned_prefix_nums[level];
                //int ss = (idx % 128) / 16 * 4 * shmem_stride[level] + aligned_prefix_nums[level] + (idx & 15) + ((idx >> 7) << 4);
                a = shmem_bitplane[write_pos];
                out_w[s] = (a >> 24) & 0x000000FFU;
                out_w[s + shmem_stride[level]] = (a >> 8) & 0x000000FFU;
                out_w[s + shmem_stride[level] * 2] = (a >> 16) & 0x000000FFU;
                out_w[s + shmem_stride[level] * 3] = (a) & 0x000000FFU;
            }

            // int idx = start + pos;
            // int level = 0;
            // while(idx < shmem_prefix_nums[level]) {
            //     ++level;
            // }
            // idx -= shmem_prefix_nums[level];
            // int s = idx / 8 + shmem_stride[level] * (idx % 8) * 4 + aligned_prefix_nums[level];
            // out_w[s] = (a >> 24) & 0x000000FFU;
            // out_w[s + shmem_stride[level]] = (a >> 8) & 0x000000FFU;
            // out_w[s + shmem_stride[level] * 2] = (a >> 16) & 0x000000FFU;
            // out_w[s + shmem_stride[level] * 3] = a & 0x000000FFU;
        }
    }
}


template<typename E, int LEVEL, btype bt>
// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// static __global__ __launch_bounds__(TPB, 3)
// #else
// static __global__ __launch_bounds__(TPB, 2)
// #endif
static __global__
void inverse_bitShuffle(uint8_t* bitplane, E* ectrl, int size, dim3 data_size) {
    __shared__ int shmem_prefix_nums[LEVEL];
    __shared__ int shmem_stride[LEVEL];
    __shared__ int aligned_prefix_nums[LEVEL];
    __shared__ unsigned int shmem_bitplane[513];

    int align = 0;
    // __shared__ long long in[(CS / sizeof(long long))];
    d_comput<LEVEL>(size, data_size, align, shmem_prefix_nums, aligned_prefix_nums, shmem_stride);
    int start = blockIdx.x * CS;
    int tid = threadIdx.x;
    const int sublane = tid % 32;
    uint8_t* const in_w = bitplane;
    E* const out_w = (E*)&ectrl[start];
    
    if(blockIdx.x != gridDim.x - 1) {
        for (int pos = tid; pos < CL; pos += TPB) {
            unsigned int a = 0, q = 0;
            // int idx = start + pos;
            
            int write_pos = ((tid & 63) << 3) + (tid >> 6);
            int idx = start + write_pos + pos / TPB * TPB;
            int level = 0;
            while(idx < shmem_prefix_nums[level]) {
                ++level;
            }
            idx -= shmem_prefix_nums[level];
            int s = (idx >> 3) + shmem_stride[level] * ((idx & 7) << 2) + aligned_prefix_nums[level];

            a |= in_w[s] << 24;
            a |= in_w[s + shmem_stride[level]] << 8;
            a |= in_w[s + shmem_stride[level] * 2] << 16;
            a |= in_w[s + shmem_stride[level] * 3];
            shmem_bitplane[write_pos] = a;
            __syncthreads();
            a = shmem_bitplane[tid];
            q = (a ^ (a >> 12)) & 0x0000F0F0U;
            a = a ^ q ^ (q << 12);
            q = (a ^ (a >> 6)) & 0x00CC00CCU;
            a = a ^ q ^ (q << 6);
            q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
            a = a ^ q ^ (q << 3);

            q = __shfl_xor_sync(0xffffffff, a, 4);
            a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 2);
            a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 1);
            unsigned int mask = 0x0F0F0F0F;
            if ((sublane & 1) == 0) {
                a = (a & ~mask) | ((q >> 4) & mask);
            } else {
                a = ((q << 4) & ~mask) | (a & mask);
            }
            // a = in__[pos];
            // int write_pos = (tid >> 4) + ((tid & 15) << 5);
            // int idx = start + write_pos + pos / TPB * TPB;
            // // idx = start + pos;
            // int level = 0;
            // while(idx < shmem_prefix_nums[level]) {
            //     ++level;
            // }
            // idx -= shmem_prefix_nums[level];
            // int s = (idx % 32) *  (shmem_stride[level]>>2) + (idx / 32 )+  (aligned_prefix_nums[level]>>2);
            // // a = in_u4[s];
            // a = in__[s - aligned_prefix_nums[l] / 4];
            // a = __byte_perm(a, 0,  (0 << 12) | (1 << 8) | (2 << 4) | 3);
            // shmem_bitplane[write_pos] = a;
            // __syncthreads();
            // a = shmem_bitplane[tid];
            // q = __shfl_xor_sync(0xffffffff, a, 16);
            // a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            // q = __shfl_xor_sync(0xffffffff, a, 8);
            // a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            // q = __shfl_xor_sync(0xffffffff, a, 4);
            // unsigned int mask = 0x0F0F0F0F;
            // if ((sublane & 4) == 0) {
            // a = (a & ~mask) | ((q >> 4) & mask);
            // } else {
            // a = ((q << 4) & ~mask) | (a & mask);
            // }

            // q = __shfl_xor_sync(0xffffffff, a, 2);
            // mask = 0x33333333;
            // if ((sublane & 2) == 0) {
            // a = (a & ~mask) | ((q >> 2) & mask);
            // } else {
            // a = ((q << 2) & ~mask) | (a & mask);
            // }

            // q = __shfl_xor_sync(0xffffffff, a, 1);
            // mask = 0x55555555;
            // if ((sublane & 1) == 0) {
            //     a = (a & ~mask) | ((q >> 1) & mask);
            // } else {
            //     a = ((q << 1) & ~mask) | (a & mask);
            // }

            if constexpr(bt == NB)
                a = (a ^ (uint32_t) 0xaaaaaaaau) - (uint32_t) 0xaaaaaaaau;
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = (((a & 0x7FFFFFFF) ^ (mask_)) - mask_);
            }
            out_w[pos] = a;
        }
    }
    else {
        int tmp = size % CL;
        int res = align + (tmp == 0 ? CL : tmp);
        for (int pos = tid; pos < res; pos += TPB) {
            unsigned int a = 0, q = 0;
            int idx = start + pos;
            int level = 0;
            while(idx < shmem_prefix_nums[level]) {
                ++level;
            }
            idx -= shmem_prefix_nums[level];
            int s = idx / 8 + shmem_stride[level] * (idx % 8) * 4 + aligned_prefix_nums[level];
            a |= in_w[s] << 24;
            a |= in_w[s + shmem_stride[level]] << 8;
            a |= in_w[s + shmem_stride[level] * 2] << 16;
            a |= in_w[s + shmem_stride[level] * 3];

            q = (a ^ (a >> 12)) & 0x0000F0F0U;
            a = a ^ q ^ (q << 12);
            q = (a ^ (a >> 6)) & 0x00CC00CCU;
            a = a ^ q ^ (q << 6);
            q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
            a = a ^ q ^ (q << 3);

            q = __shfl_xor_sync(0xffffffff, a, 4);
            a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 2);
            a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 1);
            unsigned int mask = 0x0F0F0F0F;
            if ((sublane & 1) == 0) {
            a = (a & ~mask) | ((q >> 4) & mask);
            } else {
            a = ((q << 4) & ~mask) | (a & mask);
            }
            if constexpr(bt == NB)
                a = (a ^ (uint32_t) 0xaaaaaaaau) - (uint32_t) 0xaaaaaaaau;
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = (((a & 0x7FFFFFFF) ^ (mask_)) - mask_);
            }

            out_w[pos] = a;

        }
    }
}

template<typename E, int LEVEL, btype bt>
static __global__ void inverse_bitShuffle_progressive(uint8_t* bitplane, E* ectrl, int size, dim3 data_size, 
int* begin, int* end) {
    __shared__ int shmem_prefix_nums[LEVEL];
    __shared__ int shmem_stride[LEVEL];
    __shared__ int aligned_prefix_nums[LEVEL];
    int align = 0;

    d_comput<LEVEL>(size, data_size, align, shmem_prefix_nums, aligned_prefix_nums, shmem_stride);

    int start = blockIdx.x * CS;
    int tid = threadIdx.x;

    const int sublane = tid % 32;
    uint8_t* const in_w = bitplane;
    E* const out_w = (E*)ectrl;
    unsigned int valid = 0xFFFFFFFF;
    
    if(blockIdx.x != gridDim.x - 1) {
        for (int pos = tid; pos < CL; pos += TPB) {
            unsigned int a = 0, q = 0;
            int idx = start + pos;
            // int write_pos = ((tid & 63) << 3) + (tid >> 6);
            // int idx = start + write_pos + pos / TPB * TPB;
            int level = 0;
            while(idx < shmem_prefix_nums[level]) {
                ++level;
            }
            idx -= shmem_prefix_nums[level];
            int lane = (idx & 7) << 2;
            int s = (idx >> 3) + shmem_stride[level] * lane + aligned_prefix_nums[level];
            int level_reverse = 3 - level;
            if(begin[level_reverse] < end[level_reverse]) {
                valid =  ((1ull << (32 - begin[level_reverse])) - 1) ^ ((1u << (32 - end[level_reverse])) - 1);
                // int s = idx / 8 + shmem_stride[level] * lane + aligned_prefix_nums[level];
                constexpr bool add_lane0 = (bt == SA || bt == SM);

                if ((lane >= begin[level_reverse] && lane < end[level_reverse])
                    || (add_lane0 && lane == 0))
                    a |= in_w[s] << 24;
                ++lane;
                if((lane >= begin[level_reverse] && lane < end[level_reverse]))
                    a |= in_w[s + shmem_stride[level]] << 8;
                ++lane;
                if(lane >= begin[level_reverse] && lane < end[level_reverse])
                    a |= in_w[s + shmem_stride[level] * 2] << 16;
                ++lane;
                if(lane >= begin[level_reverse] && lane < end[level_reverse]) {
                    
                    a |= in_w[s + shmem_stride[level] * 3];
                }
                    
            //             a |= in_w[s] << 24;
            // a |= in_w[s + shmem_stride[level]] << 8;
            // a |= in_w[s + shmem_stride[level] * 2] << 16;
            // a |= in_w[s + shmem_stride[level] * 3];

            }
            // if(level == 4) {
            //     int s = idx / 8 + shmem_stride[level] * lane + shmem_prefix_nums[level] * 4;
            //     a |= in_w[s] << 24;
            //     a |= in_w[s + shmem_stride[level]] << 8;
            //     a |= in_w[s + shmem_stride[level] * 2] << 16;
            //     a |= in_w[s + shmem_stride[level] * 3];
            // }


            q = (a ^ (a >> 12)) & 0x0000F0F0U;
            a = a ^ q ^ (q << 12);
            q = (a ^ (a >> 6)) & 0x00CC00CCU;
            a = a ^ q ^ (q << 6);
            q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
            a = a ^ q ^ (q << 3);

            q = __shfl_xor_sync(0xffffffff, a, 4);
            a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 2);
            a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 1);
            unsigned int mask = 0x0F0F0F0F;
            if ((sublane & 1) == 0) {
                a = (a & ~mask) | ((q >> 4) & mask);
            } else {
                a = ((q << 4) & ~mask) | (a & mask);
            }
            if constexpr(bt == NB)
                a = (a ^ (uint32_t) 0xaaaaaaaau) - (uint32_t) 0xaaaaaaaau;
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = (((a & 0x7FFFFFFF) ^ (mask_)) - mask_);
            }
            out_w[start + pos] = a;
        }
    }
    else {
        int tmp = size % CL;
        int res = align + (tmp == 0 ? CL : tmp);
        for (int pos = tid; pos < res; pos += TPB) {
            unsigned int a = 0, q = 0;
            int idx = start + pos;
            int level = 0;
            while(idx < shmem_prefix_nums[level]) {
                ++level;
            }
            idx -= shmem_prefix_nums[level];
            int lane = (idx % 8) * 4;
            int s = idx / 8 + shmem_stride[level] * lane + aligned_prefix_nums[level];
            int level_reverse = 3 - level;
            if(begin[level_reverse] < end[level_reverse]) {
                // valid =  ((1ull << (32 - begin[level_reverse])) - 1) ^ ((1u << (32 - end[level_reverse])) - 1);
                int s = idx / 8 + shmem_stride[level] * lane + aligned_prefix_nums[level];
                constexpr bool add_lane0 = (bt == SA || bt == SM);

                if ((lane >= begin[level_reverse] && lane < end[level_reverse])
                    || (add_lane0 && lane == 0))
                    a |= in_w[s] << 24;
                ++lane;
                if(lane >= begin[level_reverse] && lane < end[level_reverse])
                    a |= in_w[s + shmem_stride[level]] << 8;
                ++lane;
                if(lane >= begin[level_reverse] && lane < end[level_reverse])
                    a |= in_w[s + shmem_stride[level] * 2] << 16;
                ++lane;
                if(lane >= begin[level_reverse] && lane < end[level_reverse])
                    a |= in_w[s + shmem_stride[level] * 3];

            }

            q = (a ^ (a >> 12)) & 0x0000F0F0U;
            a = a ^ q ^ (q << 12);
            q = (a ^ (a >> 6)) & 0x00CC00CCU;
            a = a ^ q ^ (q << 6);
            q = (a ^ (a >> 3)) & 0x0A0A0A0AU;
            a = a ^ q ^ (q << 3);

            q = __shfl_xor_sync(0xffffffff, a, 4);
            a = ((sublane & 4) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 2);
            a = ((sublane & 2) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

            q = __shfl_xor_sync(0xffffffff, a, 1);
            unsigned int mask = 0x0F0F0F0F;
            if ((sublane & 1) == 0) {
                a = (a & ~mask) | ((q >> 4) & mask);
            } else {
                a = ((q << 4) & ~mask) | (a & mask);
            }

            if constexpr(bt == NB)
                a = (a ^ (uint32_t) 0xaaaaaaaau) - (uint32_t) 0xaaaaaaaau;
            else if constexpr(bt == SM) {
                const unsigned int sign = ((int)a) >> 31; // sign bit
                a = (a & 0x7FFFFFFF) ^ sign; // convert to sign-maginitude
            }
            else if constexpr(bt == SA) {
                int mask_ = ((int)a) >> 31; // sign bit
                a = (((a & 0x7FFFFFFF) ^ (mask_)) - mask_);
            }

            out_w[start + pos] = a;
        }
    }
}


template<typename E, btype bt>
// bool convert_to_bitplane(Buffer* qc, Buffer* bitplane, size_t anchor_size, void* stream) {
bool convert_to_bitplane(Buffer* qc, Bitplane* bp, size_t anchor_size, double& time, void* stream) {
    auto data_size = qc->template len3<dim3>();
    int size = qc->len;
    int gridDim = (bp->aligned_len - anchor_size - 1) / CS + 1;

    GPUTimer dtimer;
    dtimer.start(stream);
    bitShuffle<E,4,bt><<<gridDim, TPB, 0, (cudaStream_t)stream>>>(reinterpret_cast<E*>(qc->d) + anchor_size, 
    reinterpret_cast<uint8_t*>(bp->d), size, data_size);
    time = dtimer.stop(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA bitplane kernel launch error: %s\n", cudaGetErrorString(err));
    }
    return true;
}

template<typename E, btype bt>
bool inverse_convert_to_bitplane(Bitplane* bp, Buffer* qc, size_t anchor_size, double& time, void* stream) {
    dim3 data_size = qc->template len3<dim3>();
    int size = qc->len;
    int gridDim = (bp->aligned_len - anchor_size - 1) / CS + 1;


    GPUTimer dtimer;
    dtimer.start(stream);
    inverse_bitShuffle<E,4,bt><<<gridDim, TPB, 0, (cudaStream_t)stream>>>(reinterpret_cast<uint8_t*>(bp->d), 
    reinterpret_cast<E*>(qc->d) + anchor_size, size, data_size);
    time = dtimer.stop(stream);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA xbitplane kernel launch error: %s\n", cudaGetErrorString(err));
    // }

    return true;
}

template<typename E, btype bt>
bool inverse_convert_to_bitplane_progressive(Bitplane* bp, Buffer* qc, size_t anchor_size, 
int* begin, int* end, double& time, void* stream) {
    dim3 data_size = qc->template len3<dim3>();
    int size = qc->len;
    int gridDim = (bp->aligned_len - anchor_size - 1) / CS + 1;
    E* d_tt, *h_t, *h_t_pro;
    cudaMalloc((void**)&d_tt, bp->aligned_size);
    cudaMallocHost((void**)&h_t, bp->aligned_size);
    cudaMallocHost((void**)&h_t_pro, bp->aligned_size);

    GPUTimer dtimer;
    dtimer.start(stream);
    inverse_bitShuffle_progressive<E,4, bt><<<gridDim, TPB, 0, (cudaStream_t)stream>>>(reinterpret_cast<uint8_t*>(bp->d), 
    reinterpret_cast<E*>(qc->d) + anchor_size, size, data_size, begin, end);
    time = dtimer.stop(stream);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA xbitplane kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    return true;
}


#define BITPLANE(E, bt) \
template bool convert_to_bitplane<E, bt>(Buffer* qc, Bitplane* bitplane, size_t anchor_size, double& time, void* stream);\
template bool inverse_convert_to_bitplane<E, bt>(Bitplane* bitplane, Buffer* qc, size_t anchor_size, double& time, void* stream);\
template bool inverse_convert_to_bitplane_progressive<E, bt>(Bitplane* bitplane, Buffer* qc, size_t anchor_size, \
int* begin, int* end, double& time, void* stream);

BITPLANE(i4, SM)
BITPLANE(i4, SA)
BITPLANE(i4, NB)
// BITPLANE(u1)