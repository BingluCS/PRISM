
#include "InterpolationPredictor.hpp"
#include "dataloader.hpp"

#define BLOCK16 16
#define DEFAULT_BLOCK_SIZE 384
#define LINEAR_BLOCK_SIZE DEFAULT_BLOCK_SIZE
#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z
#define CONSTEXPR constexpr
#define SPLINE3_COMPR true
#define SPLINE3_DECOMPR false
#define SPLINE_DIM_3 3
using DIM3 = dim3;

template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT);
template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT);
template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT);


template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * (N+1) * (N+1); 
        auto Q = (N+1) * (N+1); 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q) / (N+1);
        auto k = (m % Q) % (N+1);
        if(group == 0)
            return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 2 * UNIT * k);
        else if (group == 1)
            return std::make_tuple(2 * UNIT * k, 2 * UNIT * i + UNIT, 2 * UNIT * j);
        else
            return std::make_tuple(2 * UNIT * j, 2 * UNIT * k, 2 * UNIT * i + UNIT);
    }
    if constexpr (SPLINE_DIM == 2) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * (N+1); 
        auto Q = (N+1); 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q);
        if(group == 0)
            return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 0);
        else
            return std::make_tuple(2 * UNIT * j, 2 * UNIT * i + UNIT, 0);
    }
}

template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * N * (N+1);
        auto Q = N * N; 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q) / N;
        auto k = (m % Q) % N;
        if(group == 0)
            return std::make_tuple(2 * UNIT * i, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
        else if (group == 1)
            return std::make_tuple(2 * UNIT * k + UNIT, 2 * UNIT * i, 2 * UNIT * j + UNIT);
        else
            return std::make_tuple(2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT, 2 * UNIT * i);
    }
    if constexpr (SPLINE_DIM == 2) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * N;
        auto Q = N * N; 
        // auto group = _tix / L ;
        auto m = _tix % L ;
        
        auto i = (m % Q) / N;
        auto j = (m % Q) % N;
        return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 0);
    }
}


template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto Q = N * N; 
        auto i = _tix / Q;
        auto j = (_tix % Q) / N;
        auto k = (_tix % Q) % N;
        return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
    }
}



template <typename T, int SPLINE_DIM, int LEVEL>
__forceinline__ __device__ void pre_compute_att(DIM3 sam_starts, DIM3 sam_bgs, DIM3 sam_strides, DIM3 &global_starts, interpolation_parameters &intp_param, uint8_t &level, uint8_t &unit, volatile T err[9], bool workflow){

    if(TIX < 9) err[TIX] = 0.0;

    auto grid_idx_x = BIX % sam_bgs.x;
    auto grid_idx_y = (BIX / sam_bgs.x) % sam_bgs.y;
    auto grid_idx_z = (BIX / sam_bgs.x) / sam_bgs.y;
    global_starts.x = sam_starts.x + grid_idx_x * sam_strides.x;
    global_starts.y = sam_starts.y + grid_idx_y * sam_strides.y;
    global_starts.z = sam_starts.z + grid_idx_z * sam_strides.z;
    
    bool use_natural = false, use_md = false, reverse = false;
    if (BIY == 0){
        level = 2;
    }
    else if (BIY < 3){
        level = 1;
        use_natural = (BIY == 2);
    }
    else{
        level = 0;
        use_natural = BIY > 5;
        use_md = (BIY == 5 or BIY == 8);
        reverse = BIY % 3;
    }       
    intp_param.use_natural[level] = use_natural;
    intp_param.use_md[level] = use_md;
    intp_param.reverse[level] = reverse;
    
    __syncthreads();
}


template <typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, int numAnchorBlockX, 
int numAnchorBlockY,  int numAnchorBlockZ>
__device__ void global2shmem_data_att(T1* data, DIM3 data_size, DIM3 data_leap, volatile T2 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], DIM3 global_starts, uint8_t unit)
{
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * 
    (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + global_starts.x);
        auto gy  = (y + global_starts.y);
        auto gz  = (z + global_starts.z);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z  ) s_data[z][y][x] = data[gid];
    }
    __syncthreads();
}

template<int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate_att(unsigned int x, unsigned int y, unsigned int z,
    const DIM3 &data_size, const DIM3 & global_starts)
{
    if CONSTEXPR (INCLUSIVE) {
            return (x <= (AnchorBlockSizeX * numAnchorBlockX) and y <= (AnchorBlockSizeY * numAnchorBlockY) and z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and global_starts.x + x < data_size.x and global_starts.y + y < data_size.y and global_starts.z + z < data_size.z;
    }
    else {
        return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) and y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) and z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) and global_starts.x + x < data_size.x and global_starts.y + y < data_size.y and global_starts.z + z < data_size.z;
    }
}

template <typename T, typename FP, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, typename LAMBDA, bool LINE, 
bool FACE, bool CUBE, bool COARSEN, bool BORDER_INCLUSIVE, bool WORKFLOW, typename INTERP>
__forceinline__ __device__ void interpolate_stage_md_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)
    ][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3 data_size, DIM3 global_starts, LAMBDA xyzmap, int unit,
    FP eb_r, FP ebx2, INTERP cubic_interpolator, volatile T* error, int NUM_ELE)
{
    // static_assert(COARSEN or (NUM_ELE <= BLOCK_DIM_SIZE), "block oversized");
    // static_assert((LINE or FACE or CUBE) == true, "must be one hot");
    // static_assert((LINE and FACE) == false, "must be only one hot (1)");
    // static_assert((LINE and CUBE) == false, "must be only one hot (2)");
    // static_assert((FACE and CUBE) == false, "must be only one hot (3)");
    //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate_att<SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z, data_size, global_starts)) {
            T pred = 0;

            auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
        //    T tmp_z[4], tmp_y[4], tmp_x[4];
           int id_z[4], id_y[4], id_x[4];
           id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
           id_z[1] = (z - unit >= 0) ? z - unit : 0;
           id_z[2] = (z + unit <=  AnchorBlockSizeZ * numAnchorBlockZ) ? z + unit : 0;
           id_z[3] = (z + 3 * unit <=  AnchorBlockSizeZ * numAnchorBlockZ) ? z + 3 * unit : 0;
           
           id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
           id_y[1] = (y - unit >= 0) ? y - unit : 0;
           id_y[2] = (y + unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + unit : 0;
           id_y[3] = (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + 3 * unit : 0;
           
           id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
           id_x[1] = (x - unit >= 0) ? x - unit : 0;
           id_x[2] = (x + unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + unit : 0;
           id_x[3] = (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + 3 * unit : 0;
           
            if CONSTEXPR (LINE) {
                bool I_Y = (y % (2*unit) )> 0; 
                bool I_Z = (z % (2*unit) )> 0; 

                pred = 0;
                auto input_x = x;
                //auto input_BI = BIX;
                //auto input_GD = GDX;
                auto input_gx = global_x;
                auto input_gs = data_size.x;
                auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                int global_start_ = global_starts.x;
                if (I_Z){
                    input_x = z;
                    //input_BI = BIZ;
                    //input_GD = GDZ;
                    input_gx = global_z;
                    input_gs = data_size.z;
                    global_start_ = global_starts.z;
                    right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                }
                else if (I_Y){
                    input_x = y;
                    //input_BI = BIY;
                    //input_GD = GDY;
                    input_gx = global_y;
                    input_gs = data_size.y;
                    global_start_ = global_starts.y;
                    right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                }
                
                int id_[4], s_id[4];
                id_[0] =  input_x - 3 * unit;
                id_[0] =  id_[0] >= 0 ? id_[0] : 0;
            
                id_[1] = input_x - unit;
                id_[1] = id_[1] >= 0 ? id_[1] : 0;
            
                id_[2] = input_x + unit;
                id_[2] = id_[2] < right_bound ? id_[2] : 0;
                
                id_[3] = input_x + 3 * unit;
                id_[3] = id_[3] < right_bound ? id_[3] : 0;
                
                s_id[0] = x_size * y_size * z + x_size * y + id_[0];
                s_id[1] = x_size * y_size * z + x_size * y + id_[1];
                s_id[2] = x_size * y_size * z + x_size * y + id_[2];
                s_id[3] = x_size * y_size * z + x_size * y + id_[3];
                if (I_Z){
                    s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
                    s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
                    s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
                    s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
                }
                else if (I_Y){
                    s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                    s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                    s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                    s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
                }

            
                bool case1 = (global_start_ + AnchorBlockSizeX * numAnchorBlockX < input_gs);
                bool case2 = (input_x >= 3 * unit);
                bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
                bool case4 = (input_gx + 3 * unit < input_gs);
                bool case5 = (input_gx + unit < input_gs);
                
                
                // 预加载 shared memory 数据到寄存器
                T tmp0 = *((T*)s_data + s_id[0]); 
                T tmp1 = *((T*)s_data + s_id[1]); 
                T tmp2 = *((T*)s_data + s_id[2]); 
                T tmp3 = *((T*)s_data + s_id[3]); 
    
                // 初始预测值
                pred = tmp1;
    
                // 计算不同 case 对应的 pred
                if ( (case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                    pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
                    
                }
                else if ((case1 && case2 && !case3) || ( !case1 && case2 && !(case3 && case4) && case5)) {
                    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
                }
                else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4 )){
                    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;   
                }
                else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                    pred = (tmp1 + tmp2) / 2;
                }

            }
            auto get_interp_order = [&](auto x, auto gx, auto gs){
                int b = (x >= 3 * unit) ? 3 : 1;
                int f = ((x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) && ((gx + 3 * unit < gs))) ? 3 :
                (((gx + unit < gs)) ? 1 : 0);

                return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0)) 
                                : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
            };
            if CONSTEXPR (FACE) {  //

                bool I_YZ = (x % (2*unit) ) == 0;
                bool I_XZ = (y % (2*unit ) )== 0;
                int x_1,BI_1,GD_1,gx_1,gs_1;
                int x_2,BI_2,GD_2,gx_2,gs_2;
                int s_id_1[4], s_id_2[4];
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                if (I_YZ){
                   
                 x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                 x_2 = y,BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
                 s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                 s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                 s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                 s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                 s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
                 s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
                 s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
                 s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
                 pred = s_data[id_z[1]][id_y[1]][x];

                }
                else if (I_XZ){
                    x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                    s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                    s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                    s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                    
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[id_z[1]][y][id_x[1]];
                    
                }
                else{
                    x_1 = y,BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
                    s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
                    s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
                    s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[z][id_y[1]][id_x[1]];
                }

                    auto interp_1 = get_interp_order(x_1,gx_1,gs_1);
                    auto interp_2 = get_interp_order(x_2,gx_2,gs_2);

                    int case_num = interp_1 + interp_2 * 5;

                    if (interp_1 == 4 && interp_2 == 4) {
                        pred = (cubic_interpolator(*((T*)s_data + s_id_1[0]), 
                        *((T*)s_data + s_id_1[1]), 
                        *((T*)s_data + s_id_1[2]), 
                        *((T*)s_data + s_id_1[3])) +
                         cubic_interpolator(*((T*)s_data + s_id_2[0]), 
                        *((T*)s_data + s_id_2[1]), 
                        *((T*)s_data + s_id_2[2]), 
                        *((T*)s_data + s_id_2[3]))) / 2;
                    } else if (interp_1 != 4 && interp_2 == 4) {
                        pred = cubic_interpolator(*((T*)s_data + s_id_2[0]), 
                        *((T*)s_data + s_id_2[1]), 
                        *((T*)s_data + s_id_2[2]), 
                        *((T*)s_data + s_id_2[3]));
                    } else if (interp_1 == 4 && interp_2 != 4) {
                        pred = cubic_interpolator(*((T*)s_data + s_id_1[0]), 
                        *((T*)s_data + s_id_1[1]), 
                        *((T*)s_data + s_id_1[2]), 
                        *((T*)s_data + s_id_1[3]));
                    } else if (interp_1 == 3 && interp_2 == 3) {
                        pred = (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                        pred += (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                        pred += (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 < 2) {
                        pred = (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                    } else if (interp_1 == 2 && interp_2 == 3) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                        pred += (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                        pred += (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 < 2) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 3) {
                        pred = (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                    } else if (interp_1 == 1 && interp_2 == 1) {
                        pred = ((*((T*)s_data + s_id_2[1]))+(*((T*)s_data + s_id_2[2]))) / 2;
                        pred += ((*((T*)s_data + s_id_1[1]))+(*((T*)s_data + s_id_1[2]))) / 2;
                        pred /= 2;
                    } else if (interp_1 == 1 && interp_2 < 1) {
                        
                        pred = ((*((T*)s_data + s_id_1[1]))+(*((T*)s_data + s_id_1[2]))) / 2;
                    } else if (interp_1 == 0 && interp_2 == 1) {
                        pred = ((*((T*)s_data + s_id_2[1]))+(*((T*)s_data + s_id_2[2]))) / 2;
                    }
                    else{
                        pred = (*((T*)s_data + s_id_1[1])) + (*((T*)s_data + s_id_2[1])) - pred;
                    }
                    
            }

            if CONSTEXPR (CUBE) {  //
                T tmp_z[4], tmp_y[4], tmp_x[4];
                auto interp_z = get_interp_order(z,global_z,data_size.z);
                auto interp_y = get_interp_order(y,global_y,data_size.y);
                auto interp_x = get_interp_order(x,global_x,data_size.x);
                
                #pragma unroll
                for(int id_itr = 0; id_itr < 4; ++id_itr){
                 tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; 
                }
                if(interp_z == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                        tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
                       }
                }
                if(interp_y == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                     tmp_y[id_itr] = s_data[z][id_y[id_itr]][x]; 
                    }
                }


                T pred_z[5], pred_y[5], pred_x[5];
                pred_x[0] = tmp_x[1];
                pred_x[1] = cubic_interpolator(tmp_x[0],tmp_x[1],tmp_x[2],tmp_x[3]);
                pred_x[2] = (-tmp_x[0]+6*tmp_x[1] + 3*tmp_x[2]) / 8;
                pred_x[3] = (3*tmp_x[1] + 6*tmp_x[2]-tmp_x[3]) / 8;
                pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;
                
                pred_y[1] = cubic_interpolator(tmp_y[0],tmp_y[1],tmp_y[2],tmp_y[3]);

                
                pred_z[1] = cubic_interpolator(tmp_z[0],tmp_z[1],tmp_z[2],tmp_z[3]);
                
            
                
                pred = pred_x[0];
                pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] +  pred_y[1] + pred_z[1]) / 3 : pred;
                
                pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2 : pred;
                pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2 : pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2 : pred;
                
                pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]: pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]: pred;


                pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]: pred;
                // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
            }


  
            // if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                
            //     auto          err = s_data[z][y][x] - pred;
            //     decltype(err) code;
            //     // TODO unsafe, did not deal with the out-of-cap case
            //     {
            //         code = fabs(err) * eb_r + 1;
            //         code = err < 0 ? -code : code;
            //         code = int(code / 2) ;
            //     }
            //     s_data[z][y][x]  = pred + code * ebx2;
            //     atomicAdd(const_cast<T*>(error),code!=0);
            // }
            // else{
                atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
            // }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        auto TOTAL = NUM_ELE;
            for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                auto [x,y,z]    = xyzmap(_tix, unit);
                run(x, y, z);
            }   
    }
    else {
        if(TIX<NUM_ELE){
            auto [x,y,z]    = xyzmap(TIX, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}

template < typename T, typename FP, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ, 
bool BLUE, bool YELLOW, bool HOLLOW, bool COARSEN, bool BORDER_INCLUSIVE, bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3 data_size, DIM3 global_starts, LAMBDAX xmap, LAMBDAY ymap, LAMBDAZ zmap, int unit,
    FP eb_r, FP ebx2, bool interpolator, volatile T* error, int BLOCK_DIMX, int BLOCK_DIMY, int BLOCK_DIMZ)
{
    // static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= BLOCK_DIM_SIZE, "block oversized");
    // static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    // static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    // static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    // static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
    //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate_att<SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z,data_size, global_starts)) {
            T pred = 0;

            auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
            auto input_x = x;
            // auto input_BI = BIX;
            // auto input_GD = GDX;
            auto input_gx = global_x;
            auto input_gs = data_size.x;
            auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            int global_start_ = global_starts.x;
            int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
            if(interpolator == 1){
               p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40;
           }
           if CONSTEXPR (BLUE){
               input_x = z;
            //    input_BI = BIZ;
            //    input_GD = GDZ;
               input_gx = global_z;
               input_gs = data_size.z;
               global_start_ = global_starts.z;
               right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
           }
           if CONSTEXPR (YELLOW){
               input_x = y;
            //    input_BI = BIY;
            //    input_GD = GDY;
               input_gx = global_y;
               input_gs = data_size.y;
               global_start_ = global_starts.y;
               right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
           }
           
           int id_[4], s_id[4];
           id_[0] =  input_x - 3 * unit;
           id_[0] =  id_[0] >= 0 ? id_[0] : 0;
           
           id_[1] = input_x - unit;
           id_[1] = id_[1] >= 0 ? id_[1] : 0;
           
           id_[2] = input_x + unit;
           id_[2] = id_[2] < right_bound ? id_[2] : 0;
           
           id_[3] = input_x + 3 * unit;
           id_[3] = id_[3] < right_bound ? id_[3] : 0;
           
           s_id[0] = x_size * y_size * z + x_size * y + id_[0];
           s_id[1] = x_size * y_size * z + x_size * y + id_[1];
           s_id[2] = x_size * y_size * z + x_size * y + id_[2];
           s_id[3] = x_size * y_size * z + x_size * y + id_[3];
           if CONSTEXPR (BLUE){
            s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
            s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
            s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
            s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
           }
           if CONSTEXPR (YELLOW){
            s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
            s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
            s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
            s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
           }
           
           
           bool case1 = (global_start_ + AnchorBlockSizeX * numAnchorBlockX < input_gs);
           bool case2 = (input_x >= 3 * unit);
           bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
           bool case4 = (input_gx + 3 * unit < input_gs);
           bool case5 = (input_gx + unit < input_gs);
           
           
           // 预加载 shared memory 数据到寄存器
           T tmp0 = *((T*)s_data + s_id[0]); 
           T tmp1 = *((T*)s_data + s_id[1]); 
           T tmp2 = *((T*)s_data + s_id[2]); 
           T tmp3 = *((T*)s_data + s_id[3]); 
           
           // 初始预测值
           pred = tmp1;
           
           // 计算不同 case 对应的 pred
           if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
               pred = (tmp1 + tmp2) / 2;
           }
           else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
               pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
           }
           else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
               pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
           }
           else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
               pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
           }

            //  if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                
            //     auto          err = s_data[z][y][x] - pred;
            //     decltype(err) code;
            //     // TODO unsafe, did not deal with the out-of-cap case
            //     {
            //         code = fabs(err) * eb_r + 1;
            //         code = err < 0 ? -code : code;
            //         code = int(code / 2) ;
            //     }
                
            //     s_data[z][y][x]  = pred + code * ebx2;
            //     atomicAdd(const_cast<T*>(error),code!=0);
                

            // }
            // else{
                // if(TIX == 0 and BIX == 0) printf("BIY=%d s_data[%d][%d][%d]=%f, pred=%f\n", BIY, z, y, x, s_data[z][y][x], pred);
                atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
            // }
        }
    };
    // -------------------------------------------------------------------------------- //
    auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
    // if(TIX == 0 and BIX == 0) printf("interpolate_stage_att BIY=%d, BLOCK_DIMX=%d, BLOCK_DIMY=%d, BLOCK_DIMZ=%d, TOTAL=%d\n", BIY, BLOCK_DIMX, BLOCK_DIMY, BLOCK_DIMZ, TOTAL);
    if CONSTEXPR (COARSEN) {
        for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
            auto itix = (_tix % BLOCK_DIMX);
            auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);
            
            run(x, y, z);
        }
    }
    else {
        if (TIX < TOTAL){
            auto itix = (TIX % BLOCK_DIMX);
            auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}

template <typename T, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, bool WORKFLOW>
__device__ void spline_layout_interpolate_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    DIM3 global_starts, FP eb_r, FP ebx2, uint8_t level, interpolation_parameters intp_param, volatile T *error)
{
    auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
    auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
    auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };


    auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
    auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
    auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
    auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };

    auto nan_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
        return (-a+9*b+9*c-d) / 16;
    };

    auto nat_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
        return (-3*a+23*b+23*c-3*d) / 40;
    };
    constexpr auto COARSEN          = true;
    // constexpr auto NO_COARSEN       = false;
    constexpr auto BORDER_INCLUSIVE = true;
    // constexpr auto BORDER_EXCLUSIVE = false;


    int unit;

    FP cur_ebx2 = ebx2, cur_eb_r = eb_r;


    auto calc_eb = [&](auto unit) {
        cur_ebx2 = ebx2, cur_eb_r = eb_r;
        int temp = 1;
        while(temp < unit){
            temp *= 2;
            cur_eb_r *= intp_param.alpha;
            cur_ebx2 /= intp_param.alpha;
        }
        if(cur_ebx2 < ebx2 / intp_param.beta){
            cur_ebx2 = ebx2 / intp_param.beta;
            cur_eb_r = eb_r * intp_param.beta;
        }
    };

    unit = 1 << level;
    int unit_x = (SPLINE_DIM >= 1) ? unit * 2 : 1;
    int unit_y = (SPLINE_DIM >= 2) ? unit * 2 : 1;
    int unit_z = (SPLINE_DIM >= 3) ? unit * 2 : 1;
    if(intp_param.use_md[level]){
        int N_x = AnchorBlockSizeX / (unit * 2);
        int N_y = AnchorBlockSizeY / (unit * 2);
        int N_z = AnchorBlockSizeZ / (unit * 2);
        int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
        int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
        int N_cube = N_x * N_y * N_z;

        auto cubic_interp = (intp_param.use_natural[level] == 0) ? nan_cubic_interp : nat_cubic_interp;

        if(intp_param.use_natural[level] == 0){
            if CONSTEXPR (SPLINE_DIM >= 1)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), 
            true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data,data_size,global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_line);

            if CONSTEXPR (SPLINE_DIM >= 2)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), 
            false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data, data_size,global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_face);

            if CONSTEXPR (SPLINE_DIM >= 3)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), 
            false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data,data_size,global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp, error, N_cube);
        }
        else{
            if CONSTEXPR (SPLINE_DIM >= 1)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), 
            true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data, data_size, global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nat_cubic_interp, error, N_line);

            if CONSTEXPR (SPLINE_DIM >= 2)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), 
            false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data, data_size, global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_face);

            if CONSTEXPR (SPLINE_DIM >= 3)
            interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), 
            false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
            (s_data, data_size, global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_cube);
        }

    }
    else{
        if(intp_param.reverse[level]){
            if CONSTEXPR (SPLINE_DIM >= 1){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse), 
                false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xhollow_reverse, 
                yhollow_reverse, zhollow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x, 
                numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                unit_x /= 2;
            }
            if CONSTEXPR (SPLINE_DIM >= 2){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), 
                false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyellow_reverse, 
                yyellow_reverse, zyellow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), 
                numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                unit_y /= 2;
            }
            if CONSTEXPR (SPLINE_DIM >= 3){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), 
                true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xblue_reverse, 
                yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), 
                numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                unit_z /= 2;
            }
        }
        else{
            if CONSTEXPR (SPLINE_DIM >= 3){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xblue), decltype(yblue), decltype(zblue), 
                true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xblue, 
                    yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), 
                numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                unit_z /= 2;
            }
            if CONSTEXPR (SPLINE_DIM >= 2){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xyellow), decltype(yyellow), decltype(zyellow), 
                false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyellow, 
                yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), 
                numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                unit_y /= 2;
            }
            if CONSTEXPR (SPLINE_DIM >= 1){
                interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
                numAnchorBlockY, numAnchorBlockZ, decltype(xhollow), decltype(yhollow), decltype(zhollow), 
                false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xhollow, 
                yhollow, zhollow, unit, cur_eb_r, cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x, 
                numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                unit_x /= 2;
            }
        }
    }
}

template <typename T, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__global__ void auto_tuning_interpolation(T* data, DIM3 data_size, DIM3 data_leap, DIM3 sample_starts, 
DIM3 sample_block_grid_sizes, DIM3 sample_strides, FP eb_r, FP eb_x2, interpolation_parameters intp_param, T* errors, bool workflow)
{       
    __shared__    T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
    __shared__    T shmem_err[9];

    DIM3 global_starts;
    uint8_t level = 0;
    uint8_t unit = 1;
    pre_compute_att<T, SPLINE_DIM, LEVEL>(sample_starts, sample_block_grid_sizes, sample_strides, global_starts, intp_param, level, unit, shmem_err, workflow);
        
    global2shmem_data_att<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, 
    numAnchorBlockY, numAnchorBlockZ>(data, data_size, data_leap, shmem_data, global_starts, unit);
        
    if CONSTEXPR (SPLINE_DIM == 3){
        //if(workflow){
            if(level==2){
                uint8_t level3 = 3;
                intp_param.use_natural[3] = false;
                intp_param.use_natural[2] = false;
                intp_param.use_md[3] = false;
                intp_param.reverse[3] = false;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r,
                    eb_x2, level3, intp_param, shmem_err);
                intp_param.reverse[3] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r,
                    eb_x2, level3, intp_param, shmem_err + 1);
                intp_param.use_md[3] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r, 
                    eb_x2, level3, intp_param, shmem_err + 2);


                intp_param.use_md[2] = false;
                intp_param.reverse[2] = false;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r,
                    eb_x2, level, intp_param, shmem_err + 3);
                intp_param.reverse[2] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r, 
                    eb_x2, level, intp_param, shmem_err + 4);
                intp_param.use_md[2] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r, 
                    eb_x2, level, intp_param, shmem_err + 5);
                if(TIX<6){
                    atomicAdd(const_cast<T*>(errors+TIX),shmem_err[TIX]);
                }
            }
            else if (level == 1){
                intp_param.use_md[1] = false;
                intp_param.reverse[1] = false;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r,
                    eb_x2, level, intp_param, shmem_err);
                intp_param.reverse[1] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size,global_starts, eb_r,
                    eb_x2, level, intp_param, shmem_err + 1);
                intp_param.use_md[1] = true;
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r,
                    eb_x2, level, intp_param, shmem_err + 2);

                if(TIX<3){
                    atomicAdd(const_cast<T*>(errors + 3 + BIY * 3 + TIX),shmem_err[TIX]);
                }
            }
            else{
                spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
                numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, true>(shmem_data, data_size, global_starts, eb_r, 
                    eb_x2, level, intp_param, shmem_err);
                if(TIX==0){
                    atomicAdd(const_cast<T*>(errors + 9 + BIY), shmem_err[0]);
                }
            }
             
    }
        //}

}

template <int LEVEL> __forceinline__ __device__ void pre_compute(DIM3 data_size, volatile size_t grid_leaps[LEVEL + 1][2], volatile size_t prefix_nums[LEVEL + 1]){
    if(TIX==0){
        auto d_size = data_size;
        auto total_size = d_size.x * d_size.y * d_size.z;
        int level = 0;
        while(level <= LEVEL){
            //grid_leaps[level][0] = 1;
            grid_leaps[level][0] = d_size.x;
            grid_leaps[level][1] = d_size.x * d_size.y;
            if(level < LEVEL){
                d_size.x = (d_size.x + 1) >> 1;
                d_size.y = (d_size.y + 1) >> 1;
                d_size.z = (d_size.z + 1) >> 1;
                prefix_nums[level] = d_size.x * d_size.y * d_size.z;
            }
            ++level;
        }   
        prefix_nums[LEVEL] = 0;
        int align = 0;
        for(int i = LEVEL - 2; i >= 0; --i) {
            align += (8 - ((prefix_nums[i] - prefix_nums[i+1] + align) % 8)) % 8;
            prefix_nums[i] += align;
        }
        align += (8 - ((total_size - prefix_nums[0] + align) % 8)) % 8;
    }
    __syncthreads(); 
}

template<typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void c_reset_scratch_data(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    // volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //                     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //                     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    int radius)
{
    for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
            (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
            _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));

        s_data[z][y][x] = 0;
        // if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
        //     z % AnchorBlockSizeZ == 0)
        //     s_ectrl
        // s_ectrl[z][y][x] = radius;
        // else s_ectrl[z][y][x] = 0;
    }
    __syncthreads();
}

template <typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY , int AnchorBlockSizeZ,
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void x_reset_scratch_data(
    volatile T1 s_xdata[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    // volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //                 [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //                 [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    T1*         anchor,
    DIM3        anchor_size, 
    DIM3     anchor_leap)
{
    for (auto _tix = TIX; _tix <  (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) * (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)); _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));

        s_xdata[z][y][x] = 0;  // TODO explicitly handle zero-padding
        /*****************************************************************************
         okay to use
         ******************************************************************************/
        if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
            z % AnchorBlockSizeZ == 0) {
            s_xdata[z][y][x] = 0;

            auto ax = ((x / AnchorBlockSizeX) + BIX * numAnchorBlockX);
            auto ay = ((y / AnchorBlockSizeY) + BIY * numAnchorBlockY);
            auto az = ((z / AnchorBlockSizeZ) + BIZ * numAnchorBlockZ);

            if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
                s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];

        }

    }

    __syncthreads();
}

template <typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void global2shmem_data(T1* data, DIM3 data_size, DIM3 data_leap,
    volatile T2 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)])
{
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
                        (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * 
                        (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
        auto gy  = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
        auto gz  = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];

    }
    __syncthreads();
}

template <typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY,int numAnchorBlockZ>
__device__ void shmem2global_data(volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], T2* xdata, DIM3 xdata_size, DIM3 data_leap)
{
    auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
    auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
    auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
        auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
        auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < xdata_size.x and gy < xdata_size.y and gz < xdata_size.z) 
            xdata[gid] = s_data[z][y][x];
    }
    __syncthreads();
}

template <typename T1, typename T2, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY,int numAnchorBlockZ>
__device__ void shmem2global_data_progressive(volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], T2* xdata_old, T2* xdata, DIM3 xdata_size, DIM3 data_leap)
{
    auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
    auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
    auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
        auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
        auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < xdata_size.x and gy < xdata_size.y and gz < xdata_size.z) 
            xdata[gid] = s_data[z][y][x] + xdata_old[gid];
    }
    __syncthreads();
}

template <typename T1, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void c_gather_anchor(T1* data, DIM3 data_size, DIM3 data_leap, T1* anchor, DIM3 anchor_leap)
{
    auto ax = BIX ;//1 is block16 by anchor stride
    auto ay = BIY ;
    auto az = BIZ ;
    // 2d bug may be here! 
    auto x = (AnchorBlockSizeX * numAnchorBlockX) * ax;
    auto y = (AnchorBlockSizeY * numAnchorBlockY) * ay;
    auto z = (AnchorBlockSizeZ * numAnchorBlockZ) * az;

    bool pred1 = TIX < 1;//1 is num of anchor
    bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

    if (pred1 and pred2) {
        auto data_id      = x + y * data_leap.y + z * data_leap.z;
        auto anchor_id    = ax + ay * anchor_leap.y + az * anchor_leap.z;
        anchor[anchor_id] = data[data_id];
    }
    __syncthreads();
}


template<int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size)
{
    if CONSTEXPR (INCLUSIVE) {  //
            return (x <= (AnchorBlockSizeX * numAnchorBlockX) and y <= (AnchorBlockSizeY * numAnchorBlockY) and z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and 
            BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and BIY *  (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
    }
    else {
        return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) * (SPLINE_DIM <= 1) and y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) * (SPLINE_DIM <= 2) and z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) * (SPLINE_DIM <= 3) 
        and BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
    }
}
template <
    typename T1,
    typename T2,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
    int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ,
    typename LAMBDAX,
    typename LAMBDAY,
    typename LAMBDAZ,
    bool BLUE,
    bool YELLOW,
    bool HOLLOW,
    bool COARSEN,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    T2* s_ectrl,
    // volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //                 [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //                 [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    LAMBDAX     xmap,
    LAMBDAY     ymap,
    LAMBDAZ     zmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    bool interpolator,
    int  BLOCK_DIMX,
    int  BLOCK_DIMY,
    int  BLOCK_DIMZ)
{
    // static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= BLOCK_DIM_SIZE, "block oversized");
    static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");


    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate<SPLINE_DIM,
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
            BORDER_INCLUSIVE>(x, y, z,data_size)) {
            auto global_x = BIX * AnchorBlockSizeX * numAnchorBlockX + x;
            auto global_y = BIY * AnchorBlockSizeY * numAnchorBlockY + y;
            auto global_z = BIZ * AnchorBlockSizeZ * numAnchorBlockZ + z;  
            
            T1 pred = 0;
            auto input_x = x;
            auto input_BI = BIX;
            auto input_GD = GDX;
            auto input_gx = global_x;
            auto input_gs = data_size.x;
            auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
            if(interpolator == 1){
                p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40;
            }
            if CONSTEXPR (BLUE){
                input_x = z;
                input_BI = BIZ;
                input_GD = GDZ;
                input_gx = global_z;
                input_gs = data_size.z;
                right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            }
            if CONSTEXPR (YELLOW){
                input_x = y;
                input_BI = BIY;
                input_GD = GDY;
                input_gx = global_y;
                input_gs = data_size.y;
                right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            }
            
            int id_[4], s_id[4];
            id_[0] =  input_x - 3 * unit;
            id_[0] =  id_[0] >= 0 ? id_[0] : 0;
        
            id_[1] = input_x - unit;
            id_[1] = id_[1] >= 0 ? id_[1] : 0;
        
            id_[2] = input_x + unit;
            id_[2] = id_[2] < right_bound ? id_[2] : 0;
            
            id_[3] = input_x + 3 * unit;
            id_[3] = id_[3] < right_bound ? id_[3] : 0;
            
            s_id[0] = x_size * y_size * z + x_size * y + id_[0];
            s_id[1] = x_size * y_size * z + x_size * y + id_[1];
            s_id[2] = x_size * y_size * z + x_size * y + id_[2];
            s_id[3] = x_size * y_size * z + x_size * y + id_[3];
            if CONSTEXPR (BLUE){
            s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
            s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
            s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
            s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
            }
            if CONSTEXPR (YELLOW){
                s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
            }

        
            bool case1 = (input_BI != input_GD - 1);
            bool case2 = (input_x >= 3 * unit);
            bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
            bool case4 = (input_gx + 3 * unit < input_gs);
            bool case5 = (input_gx + unit < input_gs);
            
            // 预加载 shared memory 数据到寄存器
            T1 tmp0 = *((T1*)s_data + s_id[0]); 
            T1 tmp1 = *((T1*)s_data + s_id[1]); 
            T1 tmp2 = *((T1*)s_data + s_id[2]); 
            T1 tmp3 = *((T1*)s_data + s_id[3]); 

            // 初始预测值
            pred = tmp1;
            // 计算不同 case 对应的 pred
            if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                pred = (tmp1 + tmp2) / 2;
            }
            else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
                pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
            }
            else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
                pred = (- tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
            }
            else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
            }
            
            if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
                
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;

                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) + radius;
                }
                // s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                auto data_gid = global_x + global_y * data_size.x + global_z * data_size.x * data_size.y;
                s_ectrl[data_gid] = code;
                s_data[z][y][x]  = pred + (code - radius) * ebx2;

            }
            else {  // TODO == DECOMPRESSS and static_assert
                auto data_gid = global_x + global_y * data_size.x + global_z * data_size.x * data_size.y;
                // s_ectrl[data_gid] = code;
                auto code       = s_ectrl[data_gid];
                s_data[z][y][x] = pred + (code - radius) * ebx2;

            }
        }
    };
    // -------------------------------------------------------------------------------- //
    auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
    if CONSTEXPR (COARSEN) {
        
        //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
            for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                auto itix = (_tix % BLOCK_DIMX);
                auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
                auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
                auto x    = xmap(itix, unit);
                auto y    = ymap(itiy, unit);
                auto z    = zmap(itiz, unit);
                
                run(x, y, z);
            }

        
    }
    else {
        if(TIX < TOTAL){
            auto itix = (TIX % BLOCK_DIMX);
            auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);


            run(x, y, z);
        }
    }
    __syncthreads();
}

template <typename T1, typename T2, typename FP, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, typename LAMBDA, 
bool LINE, bool FACE, bool CUBE, bool COARSEN, bool BORDER_INCLUSIVE, bool WORKFLOW, typename INTERP>
__forceinline__ __device__ void interpolate_stage_md(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    T2* s_ectrl,
// volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
//  [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
//  [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    LAMBDA xyzmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    INTERP cubic_interpolator,
    int NUM_ELE)
{
    // static_assert(COARSEN or (NUM_ELE <= BLOCK_DIM_SIZE), "block oversized");
    static_assert((LINE or FACE or CUBE) == true, "must be one hot");
    static_assert((LINE and FACE) == false, "must be only one hot (1)");
    static_assert((LINE and CUBE) == false, "must be only one hot (2)");
    static_assert((FACE and CUBE) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {

        if (xyz_predicate<SPLINE_DIM,
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z,data_size)) {
            T1 pred = 0;
            auto global_x = BIX * AnchorBlockSizeX * numAnchorBlockX + x;
            auto global_y = BIY * AnchorBlockSizeY * numAnchorBlockY + y;
            auto global_z = BIZ * AnchorBlockSizeZ * numAnchorBlockZ + z;  

          
           int id_z[4], id_y[4], id_x[4];
           id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
           id_z[1] = (z - unit >= 0) ? z - unit : 0;
           id_z[2] = (z + unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + unit : 0;
           id_z[3] = (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + 3 * unit : 0;
           
           id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
           id_y[1] = (y - unit >= 0) ? y - unit : 0;
           id_y[2] = (y + unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + unit : 0;
           id_y[3] = (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + 3 * unit : 0;
           
           id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
           id_x[1] = (x - unit >= 0) ? x - unit : 0;
           id_x[2] = (x + unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + unit : 0;
           id_x[3] = (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + 3 * unit : 0;
           
            if CONSTEXPR (LINE) {
                
                bool I_Y = (y % (2*unit) )> 0; 
                bool I_Z = (z % (2*unit) )> 0; 

                pred = 0;
                auto input_x = x;
                auto input_BI = BIX;
                auto input_GD = GDX;
                auto input_gx = global_x;
                auto input_gs = data_size.x;

                auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                
                if (I_Z){
                    input_x = z;
                    input_BI = BIZ;
                    input_GD = GDZ;
                    input_gx = global_z;
                    input_gs = data_size.z;
                    right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                }
                else if (I_Y){
                    input_x = y;
                    input_BI = BIY;
                    input_GD = GDY;
                    input_gx = global_y;
                    input_gs = data_size.y;
                    right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                }
                
                int id_[4], s_id[4];
                id_[0] =  input_x - 3 * unit;
                id_[0] =  id_[0] >= 0 ? id_[0] : 0;
            
                id_[1] = input_x - unit;
                id_[1] = id_[1] >= 0 ? id_[1] : 0;
            
                id_[2] = input_x + unit;
                id_[2] = id_[2] < right_bound ? id_[2] : 0;
                
                id_[3] = input_x + 3 * unit;
                id_[3] = id_[3] < right_bound ? id_[3] : 0;
                
                s_id[0] = x_size * y_size * z + x_size * y + id_[0];
                s_id[1] = x_size * y_size * z + x_size * y + id_[1];
                s_id[2] = x_size * y_size * z + x_size * y + id_[2];
                s_id[3] = x_size * y_size * z + x_size * y + id_[3];
                if (I_Z){
                s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
                s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
                s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
                s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
                }
                else if (I_Y){
                    s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                    s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                    s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                    s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
                }

            
                bool case1 = (input_BI != input_GD - 1);
                bool case2 = (input_x >= 3 * unit);
                bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
                bool case4 = (input_gx + 3 * unit < input_gs);
                bool case5 = (input_gx + unit < input_gs);
                
                
                // 预加载 shared memory 数据到寄存器
                T1 tmp0 = *((T1*)s_data + s_id[0]); 
                T1 tmp1 = *((T1*)s_data + s_id[1]); 
                T1 tmp2 = *((T1*)s_data + s_id[2]); 
                T1 tmp3 = *((T1*)s_data + s_id[3]); 
    
                // 初始预测值
                pred = tmp1;
    
                // 计算不同 case 对应的 pred
                if ( (case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                    pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
                    
                }
                else if ((case1 && case2 && !case3) || ( !case1 && case2 && !(case3 && case4) && case5)) {
                    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
                }
                else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4 )){
                    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;   
                }
                else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                    pred = (tmp1 + tmp2) / 2;
                }

            }
            auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs){
                int b = (x >= 3 * unit) ? 3 : 1;
                int f = ((x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) && ((BI != GD - 1) || (gx + 3 * unit < gs))) ? 3 :
                (((BI != GD - 1) || (gx + unit < gs)) ? 1 : 0);

                return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0)) 
                                : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
            };
            if CONSTEXPR (FACE) {  //
               // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
               //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
              //  }

                bool I_YZ = (x % (2*unit) ) == 0;
                bool I_XZ = (y % (2*unit ) )== 0;

                //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
               //     printf("face %d %d\n", I_YZ,I_XZ);
                int x_1,BI_1,GD_1,gx_1,gs_1;
                int x_2,BI_2,GD_2,gx_2,gs_2;
                int s_id_1[4], s_id_2[4];
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                if (I_YZ){
                   
                 x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                 x_2 = y,BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
                 s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                 s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                 s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                 s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                 s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
                 s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
                 s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
                 s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
                 pred = s_data[id_z[1]][id_y[1]][x];

                }
                else if (I_XZ){
                    x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                    s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                    s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                    s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                    
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[id_z[1]][y][id_x[1]];
                    
                }
                else{
                    x_1 = y,BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
                    s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
                    s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
                    s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[z][id_y[1]][id_x[1]];
                }

                    auto interp_1 = get_interp_order(x_1,BI_1,GD_1,gx_1,gs_1);
                    auto interp_2 = get_interp_order(x_2,BI_2,GD_2,gx_2,gs_2);

                    int case_num = interp_1 + interp_2 * 5;


                    if (interp_1 == 4 && interp_2 == 4) {
                        pred = (cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3])) +
                         cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]))) / 2;
                    } else if (interp_1 != 4 && interp_2 == 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]));
                    } else if (interp_1 == 4 && interp_2 != 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3]));
                    } else if (interp_1 == 3 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 < 2) {
                        pred = (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                    } else if (interp_1 == 2 && interp_2 == 3) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 < 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                    } else if (interp_1 == 1 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                        pred += ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                        pred /= 2;
                    } else if (interp_1 == 1 && interp_2 < 1) {
                        
                        pred = ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                    } else if (interp_1 == 0 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                    }
                    else{
                        pred = (*((T1*)s_data + s_id_1[1])) + (*((T1*)s_data + s_id_2[1])) - pred;
                    }
                    
            }

            if CONSTEXPR (CUBE) {  //
                T1 tmp_z[4], tmp_y[4], tmp_x[4];
                auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
                auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
                
                #pragma unroll
                for(int id_itr = 0; id_itr < 4; ++id_itr){
                 tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; 
                }
                if(interp_z == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                        tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
                       }
                }
                if(interp_y == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                     tmp_y[id_itr] = s_data[z][id_y[id_itr]][x]; 
                    }
                }


                T1 pred_z[5], pred_y[5], pred_x[5];
                pred_x[0] = tmp_x[1];
                pred_x[1] = cubic_interpolator(tmp_x[0],tmp_x[1],tmp_x[2],tmp_x[3]);
                pred_x[2] = (-tmp_x[0]+6*tmp_x[1] + 3*tmp_x[2]) / 8;
                pred_x[3] = (3*tmp_x[1] + 6*tmp_x[2]-tmp_x[3]) / 8;
                pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;
                
                pred_y[1] = cubic_interpolator(tmp_y[0],tmp_y[1],tmp_y[2],tmp_y[3]);

                
                pred_z[1] = cubic_interpolator(tmp_z[0],tmp_z[1],tmp_z[2],tmp_z[3]);
                
                pred = pred_x[0];
                pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] +  pred_y[1] + pred_z[1]) / 3 : pred;
                
                pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2 : pred;
                pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2 : pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2 : pred;
                
                pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]: pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]: pred;


                pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]: pred;
                // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
            }

            if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
                
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;
                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) + radius;
                }
                // s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                auto data_gid = global_x + global_y * data_size.x + global_z * data_size.x * data_size.y;
                s_ectrl[data_gid] = code;
                s_data[z][y][x]  = pred + (code - radius) * ebx2;
                

            }
            else {  // TODO == DECOMPRESSS and static_assert

                auto data_gid = global_x + global_y * data_size.x + global_z * data_size.x * data_size.y;
                auto code       = s_ectrl[data_gid];
                s_data[z][y][x] = pred + (code - radius) * ebx2;
                // printf("code: %f\n", code);
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        auto TOTAL = NUM_ELE;
        for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
            auto [x,y,z]    = xyzmap(_tix, unit);
            run(x, y, z);
        }
        
    }
    else {
        if(TIX<NUM_ELE){
            auto [x,y,z]    = xyzmap(TIX, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}


template<typename T1, typename T2, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, bool WORKFLOW>
__device__ void spline_layout_interpolate(volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    T2* s_ectrl,
    // volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //                 [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //                 [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    interpolation_parameters intp_param)
    {
    auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
    auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };


    auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
    auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
    auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
    auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };

    auto nan_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
        return (-a+9*b+9*c-d) / 16;
    };

    auto nat_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
        return (-3*a+23*b+23*c-3*d) / 40;
    };

    constexpr auto COARSEN          = true;
    // constexpr auto NO_COARSEN       = false;
    constexpr auto BORDER_INCLUSIVE = true;
    constexpr auto BORDER_EXCLUSIVE = false;
    FP cur_ebx2=ebx2,cur_eb_r=eb_r;

    auto calc_eb = [&](auto unit) {
        cur_ebx2=ebx2,cur_eb_r=eb_r;
        int temp=1;
        while(temp<unit){
            temp*=2;
            cur_eb_r *= intp_param.alpha;
            cur_ebx2 /= intp_param.alpha;

        }
        if(cur_ebx2 < ebx2 / intp_param.beta){
            cur_ebx2 = ebx2 / intp_param.beta;
            cur_eb_r = eb_r * intp_param.beta;

        }
    };
    int max_unit = ((AnchorBlockSizeX >= AnchorBlockSizeY) ? AnchorBlockSizeX : AnchorBlockSizeY);
    max_unit = ((max_unit >= AnchorBlockSizeZ) ? max_unit : AnchorBlockSizeZ);
    max_unit /= 2;
    int unit_x = AnchorBlockSizeX, unit_y = AnchorBlockSizeY, unit_z = AnchorBlockSizeZ;
    int level_id = LEVEL;
    level_id -= 1;
    #pragma unroll
    for(int unit = max_unit; unit >= 1; unit /= 2, level_id--) {
        if(level_id > maxlevel)
            continue;
        calc_eb(unit);
        unit_x = (SPLINE_DIM >= 1) ? unit * 2 : 1;
        unit_y = (SPLINE_DIM >= 2) ? unit * 2 : 1;
        unit_z = (SPLINE_DIM >= 3) ? unit * 2 : 1;
        if(level_id != 0){
            if(intp_param.use_md[level_id]){
                int N_x = AnchorBlockSizeX / (unit * 2);
                int N_y = AnchorBlockSizeY / (unit * 2);
                int N_z = AnchorBlockSizeZ / (unit * 2);
                int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
                int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
                int N_cube = N_x * N_y * N_z;
                if(intp_param.use_natural[level_id]==0){
                    if constexpr (SPLINE_DIM >= 1)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_cube);
                }
                else{
                    if constexpr (SPLINE_DIM >= 1)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                        interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_cube);
                }
            }
            else {
                if(intp_param.reverse[level_id]){
                    if constexpr (SPLINE_DIM >= 1) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse), false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 3) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                }
                else{
                    if constexpr (SPLINE_DIM >= 3) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 1) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                        numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                }                   
            }
        }
        else {
            if(intp_param.use_md[level_id]){
                int N_x = AnchorBlockSizeX / (unit * 2);
                int N_y = AnchorBlockSizeY / (unit * 2);
                int N_z = AnchorBlockSizeZ / (unit * 2);
                int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
                int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
                int N_cube = N_x * N_y * N_z;
                if(intp_param.use_natural[level_id]==0){
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_cube);
                }
                else{
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                    decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>
                    (s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_cube);
                }
            }
            else {
                if(intp_param.reverse[level_id]){
                    if constexpr (SPLINE_DIM >= 1) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse), false, false, true, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y,numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 3) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                }
                else{
                    if constexpr (SPLINE_DIM >= 3) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, BORDER_INCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 1) {
                        interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, 
                        decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>
                        (s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], 
                        numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                }
            }
        }
    }
}
template<typename T, typename T2, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void shmem2global_data_with_compaction(T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], T2* ectrl, T2* ectrl_tmp, DIM3 ectrl_leap, DIM3 data_size, uint32_t radius, 
volatile size_t grid_leaps[LEVEL + 1][2], volatile size_t prefix_nums[LEVEL + 1], T* ol, uint32_t* ol_idx, uint32_t* ol_num ) {

    auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
    auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
    auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
    auto TOTAL = x_size * y_size * z_size;

    // for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
    //     auto x   = (_tix % x_size);
    //     auto y   = (_tix / x_size) % y_size;
    //     auto z   = (_tix / x_size) / y_size;
    //     auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
    //     auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
    //     auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
    //     auto gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
    //     if (gx < data_size.x and gy < data_size.y and gz < data_size.z) 
    //     ectrl[gid] = ectrl_tmp[gid];

    // }
    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
        auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
        auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
        // auto candidate = s_data[z][y][x];
        // bool quantizable = (candidate >= 0) and (candidate < 2*radius);
        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) {
            
            auto gid_tmp = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
            // if (not quantizable) {
            //     auto data_gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
            //     auto cur_idx = atomicAdd(ol_num, 1);
            //     ol_idx[cur_idx] = data_gid;
            //     ol[cur_idx] = candidate;
            // }
            int level = 0;
            //todo: pre-compute the leaps and their halves
            while(gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL){
                gx = gx >> 1;
                gy = gy >> 1;
                gz = gz >> 1;
                level++;
            }
            auto gid = gx + gy * grid_leaps[level][0] + gz * grid_leaps[level][1];

            if(level < LEVEL){//non-anchor
                gid += prefix_nums[level]-((gz + 1) >> 1) * grid_leaps[level + 1][1] - 
                (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] - (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
            } 
            
            ectrl[gid] = ectrl_tmp[gid_tmp]; //s_ectrl[z][y][x]; // static_cast<T2>(candidate);// * quantizable;
        }
    }
}

template <typename T, typename E, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, 
int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__device__ void global2shmem_fuse(E* ectrl, E* ectrl_tmp, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier, 
    volatile T s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile size_t grid_leaps[LEVEL + 1][2],volatile size_t prefix_nums[LEVEL + 1])
{
    
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) *
    (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y   = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) % (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z   = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) / (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
        auto gy  = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
        auto gz  = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
        if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z) {
            //todo: pre-compute the leaps and their halves
           auto gid_tmp = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
            int level = 0;
            // auto data_gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
            // if(gz == 0 && gy ==0 && gx == 1)
            //     printf("ectrl: %u\n", ectrl[data_gid]);
            while(gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL){
                gx = gx >> 1;
                gy = gy >> 1;
                gz = gz >> 1;
                level++;
            }
            auto gid = gx + gy*grid_leaps[level][0] + gz*grid_leaps[level][1];

            if(level < LEVEL){//non-anchor
                gid += prefix_nums[level] - ((gz + 1) >> 1) * grid_leaps[level + 1][1] - 
                (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] - (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
            }
            ectrl_tmp[gid_tmp] = ectrl[gid];
            // s_ectrl[z][y][x] = static_cast<T>(ectrl[gid]);// + scattered_outlier[data_gid];
        }
    }
    __syncthreads();
}


template<typename T, typename E, typename FP, int LEVEL, int SPLINE_DIM, 
int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__global__ void interpolation(T* data, dim3 data_size, dim3 data_leap, E* ectrl, E* ectrl_tmp, T* anchor, 
dim3 anchor_leap, T* ol, uint32_t* ol_idx, uint32_t* ol_num, FP eb_r, FP ebx2, 
uint32_t radius, interpolation_parameters intp_param) 
{
    __shared__ T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        // __shared__ T shmem_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
            // [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
            // [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        __shared__ size_t shmem_grid_leaps[LEVEL + 1][2];
        __shared__ size_t shmem_prefix_nums[LEVEL + 1];
    pre_compute<LEVEL>(data_size, shmem_grid_leaps, shmem_prefix_nums);

    c_reset_scratch_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    // numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, shmem_ectrl, radius);
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, radius);

    global2shmem_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(data, data_size, data_leap, shmem_data);

    c_gather_anchor<T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(data, data_size, data_leap, anchor, anchor_leap);

    spline_layout_interpolate<T, E, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, SPLINE3_COMPR>
    // (shmem_data, shmem_ectrl, data_size, eb_r, ebx2, radius, intp_param);
    (shmem_data, ectrl_tmp, data_size, eb_r, ebx2, radius, intp_param);

    shmem2global_data_with_compaction<T, E, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, ectrl, ectrl_tmp, data_leap, data_size, radius, 
    shmem_grid_leaps, shmem_prefix_nums, ol, ol_idx, ol_num);
    //convert_to_bitplane<E, LEVEL>(ectrl, bitplane, data_leap, data_size, shmem_grid_leaps, shmem_prefix_nums);
}



template<typename T, typename E, typename FP, int LEVEL, int SPLINE_DIM, 
int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__global__ void interpolation_reverse(E* ectrl, E* ectrl_tmp, dim3 data_size, dim3 data_leap, T* anchor, dim3 anchor_size,
dim3 anchor_leap, T* xdata, T* outlier, FP eb_r, FP ebx2, uint32_t radius, interpolation_parameters intp_param) 
{
    __shared__ T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        // __shared__ T shmem_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        __shared__ size_t shmem_grid_leaps[LEVEL + 1][2];
        __shared__ size_t shmem_prefix_nums[LEVEL + 1];
    pre_compute<LEVEL>(data_size, shmem_grid_leaps, shmem_prefix_nums);

    x_reset_scratch_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, anchor, anchor_size, anchor_leap);

    global2shmem_fuse<T, E, LEVEL, SPLINE_DIM,AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(ectrl, ectrl_tmp, data_size, data_leap, outlier, 
    shmem_data, shmem_grid_leaps, shmem_prefix_nums);

    spline_layout_interpolate<T, E, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, SPLINE3_DECOMPR>
    (shmem_data, ectrl_tmp, data_size, eb_r, ebx2, radius, intp_param);

    shmem2global_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, xdata, data_size, data_leap);

}

template<typename T, typename E, typename FP, int LEVEL, int SPLINE_DIM, 
int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, 
int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ>
__global__ void interpolation_progressive_reverse(E* ectrl, E* ectrl_tmp, dim3 data_size, dim3 data_leap, T* anchor, dim3 anchor_size,
dim3 anchor_leap, T* xdata_old, T* xdata, T* outlier, FP eb_r, FP ebx2, uint32_t radius, interpolation_parameters intp_param) 
{
    __shared__ T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        // __shared__ T shmem_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        __shared__ size_t shmem_grid_leaps[LEVEL + 1][2];
        __shared__ size_t shmem_prefix_nums[LEVEL + 1];
    pre_compute<LEVEL>(data_size, shmem_grid_leaps, shmem_prefix_nums);

    x_reset_scratch_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, anchor, anchor_size, anchor_leap);

    global2shmem_fuse<T, E, LEVEL, SPLINE_DIM,AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(ectrl, ectrl_tmp, data_size, data_leap, outlier, 
    shmem_data, shmem_grid_leaps, shmem_prefix_nums);

    spline_layout_interpolate<T, E, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, SPLINE3_DECOMPR>
    (shmem_data, ectrl_tmp, data_size, eb_r, ebx2, radius, intp_param);

    shmem2global_data_progressive<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, 
    numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(shmem_data, xdata_old, xdata, data_size, data_leap);

}

template<typename T, typename E, typename FP>
void spline_construct(StatBuffer<T>* input, Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, olBuffer<T>* outlier, double eb, double rel_eb, 
uint32_t radius, interpolation_parameters& intp_param, Buffer* profiling_errors, double& time, void* stream){
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    double a1=2.0;
    double a2=1.75;
    double a3=1.5;
    double a4=1.25;
    double a5=1;
    double e1=1e-1;
    double e2=1e-2;
    double e3=1e-3;
    double e4=1e-4;
    double e5=1e-5;

    intp_param.beta=4.0;
    if(rel_eb>=e1)
        intp_param.alpha=a1;
    else if(rel_eb>=e2)
        intp_param.alpha=a2+(a1-a2)*(rel_eb-e2)/(e1-e2);
    else if(rel_eb>=e3)
        intp_param.alpha=a3+(a2-a3)*(rel_eb-e3)/(e2-e3);
    else if(rel_eb>=e4)
        intp_param.alpha=a4+(a3-a4)*(rel_eb-e4)/(e3-e4);
    else if(rel_eb>=e5)
        intp_param.alpha=a5+(a4-a5)*(rel_eb-e5)/(e4-e5);
    else
        intp_param.alpha=a5;

    auto grid_dim = dim3(div(input->lx, BLOCK16),
          div(input->ly, BLOCK16),
          div(input->lz, BLOCK16));
    int S_STRIDE = 8 * BLOCK16;
    auto calc_start_size = [&](auto dim, auto & s_start, auto &s_size, auto BLOCKSIZE) {
          auto mid = dim / 2;
          auto k = (mid - BLOCKSIZE / 2) / S_STRIDE;  
          auto t = (dim - BLOCKSIZE / 2 - 1 - mid) / S_STRIDE;
          s_start = mid - k * S_STRIDE;
          s_size = k + t + 1;
    };
    int s_start_x, s_start_y, s_start_z, s_size_x, s_size_y, s_size_z;
    calc_start_size(input->lx, s_start_x, s_size_x, BLOCK16);
    calc_start_size(input->ly, s_start_y, s_size_y, BLOCK16);
    calc_start_size(input->lz, s_start_z, s_size_z, BLOCK16);

    auto block_num = s_size_x * s_size_y * s_size_z;
    T best_error;
    int best_idx;  

    GPUTimer dtimer;
    dtimer.start(stream);
    auto_tuning_interpolation<T, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1>
    <<<dim3(block_num, 6, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>
    (reinterpret_cast<T*>(input->d), input->template len3<dim3>(), input->template st3<dim3>(), 
    dim3(s_start_x, s_start_y, s_start_z), dim3(s_size_x, s_size_y, s_size_z), 
    dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r, ebx2, intp_param, reinterpret_cast<T*>(profiling_errors->d), true);
    time = dtimer.stop(stream);

    profiling_errors->D2H();
    auto errors = reinterpret_cast<T*>(profiling_errors->h);


    if(errors[0] > errors[1]){
        best_error = errors[1];
        intp_param.reverse[3] = true;
    }
    else{
        best_error = errors[0];
        intp_param.reverse[3] = false;
    }
    
    //printf("use_md[3] errors[2]=%f, best_error=%f\n", errors[2], best_error);
    intp_param.use_md[3] = errors[2] < best_error; 
    best_error = fmin(errors[2],best_error);


    if(errors[3] > errors[4]){
        best_error = errors[4];
        intp_param.reverse[2] = true;
    }
    else{
        best_error = errors[3];
        intp_param.reverse[2] = false;
    }
    //printf("use_md[2] errors[5]=%f, best_error=%f\n", errors[5], best_error);
    intp_param.use_md[2] = errors[5] < best_error; 
    best_error = fmin(errors[5],best_error);

    best_error = errors[6];
    best_idx = 6; 
    for(auto i = 6; i < 12; i++){
        if(errors[i] < best_error){
        best_error = errors[i];
        best_idx = i;
        }
    }
    // intp_param.use_natural[1] = best_idx >  8;
    intp_param.use_md[1] = (best_idx ==  8 or best_idx ==  11) ;
    intp_param.reverse[1] = best_idx%3;

    best_error = errors[12];
    best_idx = 12; 

    for(auto i = 12;i<15;i++){
        if(errors[i]<best_error){
        best_error=errors[i];
        best_idx = i;
        }
    }
    // intp_param.use_natural[0] = best_idx >  14;
    intp_param.use_md[0] = (best_idx ==  14);
    intp_param.reverse[0] = best_idx%3;

    dtimer.start(stream);
    interpolation<T, E, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1>
    <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>
    (reinterpret_cast<T*>(input->d), input->template len3<dim3>(), input->template st3<dim3>(), reinterpret_cast<E*>(ectrl->d), 
    reinterpret_cast<E*>(ectrl_tmp), reinterpret_cast<T*>(anchor->d), anchor->template st3<dim3>(), 
    reinterpret_cast<T*>(outlier->d), outlier->d_idx,
    outlier->d_num, eb_r, ebx2, radius, intp_param);
    time += dtimer.stop(stream);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA predict kernel launch error: %s\n", cudaGetErrorString(err2));
    }
}


template<typename T, typename E, typename FP>
void spline_reconstruct(Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, olBuffer<T>* outlier, StatBuffer<T>* output, double eb, double rel_eb, 
uint32_t radius, interpolation_parameters& intp_param, double& time, void* stream){
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    auto grid_dim = dim3(div(output->lx, BLOCK16),
          div(output->ly, BLOCK16),
          div(output->lz, BLOCK16));
    // CREATE_GPUEVENT_PAIR
    // START_GPUEVENT_RECORDING(stream);
    double a1=2.0;
    double a2=1.75;
    double a3=1.5;
    double a4=1.25;
    double a5=1;
    double e1=1e-1;
    double e2=1e-2;
    double e3=1e-3;
    double e4=1e-4;
    double e5=1e-5;
    intp_param.beta=4.0;
    if(rel_eb>=e1)
    intp_param.alpha=a1;
    else if(rel_eb>=e2)
    intp_param.alpha=a2+(a1-a2)*(rel_eb-e2)/(e1-e2);
    else if(rel_eb>=e3)
    intp_param.alpha=a3+(a2-a3)*(rel_eb-e3)/(e2-e3);
    else if(rel_eb>=e4)
    intp_param.alpha=a4+(a3-a4)*(rel_eb-e4)/(e3-e4);
    else if(rel_eb>=e5)
    intp_param.alpha=a5+(a4-a5)*(rel_eb-e5)/(e4-e5);
    else
    intp_param.alpha=a5;

    GPUTimer dtimer;
    dtimer.start(stream);
    interpolation_reverse<T, E, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1>
    <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>
    (reinterpret_cast<E*>(ectrl->d), reinterpret_cast<E*>(ectrl_tmp), output->template len3<dim3>(), output->template st3<dim3>(), 
    reinterpret_cast<T*>(anchor->d), anchor->template len3<dim3>(), anchor->template st3<dim3>(), 
    reinterpret_cast<T*>(output->d), reinterpret_cast<T*>(outlier->d), eb_r, ebx2, radius, intp_param);

    time = dtimer.stop(stream);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("CUDA ipredict kernel launch error: %s\n", cudaGetErrorString(err1));
    }
}

template<typename T, typename E, typename FP>
void spline_progressive_reconstruct(Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, olBuffer<T>* outlier, StatBuffer<T>* output_old, StatBuffer<T>* output,
double eb, double rel_eb,  uint32_t radius, interpolation_parameters& intp_param, double& time, void* stream){
    auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    // for(int i = 0; i < 6; ++i) {
    //     printf("use_natural: %d\n", intp_param.use_natural[i]);
    // }
    // for(int i = 0; i < 6; ++i) {
    //     printf("use_md: %d\n", intp_param.use_md[i]);
    // }
    // for(int i = 0; i < 6; ++i) {
    //     printf("reverse: %d\n", intp_param.reverse[i]);
    // }
    double a1=2.0;
    double a2=1.75;
    double a3=1.5;
    double a4=1.25;
    double a5=1;
    double e1=1e-1;
    double e2=1e-2;
    double e3=1e-3;
    double e4=1e-4;
    double e5=1e-5;

    intp_param.beta=4.0;
    if(rel_eb>=e1)
    intp_param.alpha=a1;
    else if(rel_eb>=e2)
    intp_param.alpha=a2+(a1-a2)*(rel_eb-e2)/(e1-e2);
    else if(rel_eb>=e3)
    intp_param.alpha=a3+(a2-a3)*(rel_eb-e3)/(e2-e3);
    else if(rel_eb>=e4)
    intp_param.alpha=a4+(a3-a4)*(rel_eb-e4)/(e3-e4);
    else if(rel_eb>=e5)
    intp_param.alpha=a5+(a4-a5)*(rel_eb-e5)/(e4-e5);
    else
    intp_param.alpha=a5;

    auto grid_dim = dim3(div(output->lx, BLOCK16),
          div(output->ly, BLOCK16),
          div(output->lz, BLOCK16));

    GPUTimer dtimer;
    dtimer.start(stream);
    interpolation_progressive_reverse<T, E, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1>
    <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>
    (reinterpret_cast<E*>(ectrl->d), reinterpret_cast<E*>(ectrl_tmp), output->template len3<dim3>(), 
    output->template st3<dim3>(), reinterpret_cast<T*>(anchor->d), 
    anchor->template len3<dim3>(), anchor->template st3<dim3>(), reinterpret_cast<T*>(output_old->d),
    reinterpret_cast<T*>(output->d), reinterpret_cast<T*>(outlier->d), eb_r, ebx2, radius, intp_param);

    time = dtimer.stop(stream);

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("CUDA ipredict_prog kernel launch error: %s\n", cudaGetErrorString(err1));
    }

}

#define SPLINE(T, E, FP) \
  template void spline_construct<T, E, FP> (StatBuffer<T> *input, \
  Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, olBuffer<T>* outlier, double eb, double rel_eb, uint32_t radius, \
  interpolation_parameters& intp_param, Buffer* profiling_errors, double& time, void* stream);\
  template void spline_reconstruct<T, E, FP> (Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, olBuffer<T>* outlier, \
  StatBuffer<T> *output, double eb, double rel_eb, uint32_t radius, \
  interpolation_parameters& intp_param, double& time, void* stream);\
  template void spline_progressive_reconstruct<T, E, FP> (Buffer* anchor, Buffer* ectrl, void* ectrl_tmp, \
  olBuffer<T>* outlier, StatBuffer<T> *output_old, StatBuffer<T> *output, double eb, double rel_eb, uint32_t radius, \
  interpolation_parameters& intp_param, double& time, void* stream);

// SPLINE(f4, u1, f4)
// SPLINE(f4, u2, f4)
// SPLINE(f4, u4, f4)
SPLINE(f4, i4, f4)
SPLINE(f8, i4, f8)
