#pragma once

#include <iostream>
#include <cmath>
template<typename T>
void statistic(T *ori_data, T *data, size_t num_elements) {
    size_t i = 0;
    double psnr, nrmse, max_err, range, l2_err = 0;
    psnr = nrmse = max_err = range = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    // float Max = ori_data[0];
    // float Min = ori_data[0];
    // float psnr, nrmse, max_err, range, l2_err = 0;
    // psnr = nrmse = max_err = range = 0;
    // max_err = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;
    size_t max_err_idx = 0;

    double sum3 = 0, sum4 = 0;
    double prodSum = 0, relerr = 0;
    double *diff = (double *) malloc(num_elements * sizeof(double));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr)
                maxpw_relerr = relerr;
        }

        if (max_err < err) {
            max_err = err;
            max_err_idx = i;
        }
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        l2_err += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = l2_err / num_elements;
    range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    // printf("[Verify]L2 error = %.10G\n", l2_err);
    // printf("[Verify]Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("  Max_E  = %.15lf, idx = %d\n", max_err, (int)max_err_idx);
    printf("  Max_RE = %G\n", max_err / (Max - Min));
//        printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("  PSNR = %.7lf, NRMSE = %.10G\n", psnr, nrmse);
//        printf("PSNR = %f, NRMSE= %.10G L2Error= %.10G\n", psnr, nrmse, l2_err);
//        printf("acEff=%f\n", acEff);
//        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

template<typename T, int decmp = 1, int progressive = 1>
void print_result(size_t size, size_t total_compressed_size, double targeteb, double targetreb,
    double time_enum, double time_pred, double time_bitplane, double time_lossless) {
    printf(
    "\n  \e[1m%-12s %-12s %-20s\e[0m\n",  //
    const_cast<char*>("type"),              //
    const_cast<char*>("time (ms)"),            //
    const_cast<char*>("throughtput (GB/s)")                //
    );
    size_t total_bytes = size * sizeof(T);
    auto th = [total_bytes](auto time) { return 1.0 * total_bytes / 1024 / 1024 / 1024 / time * 1000; }; 
    if constexpr (decmp == 1) {
        if constexpr (progressive ==1) {
            printf("  %-12s %'-12f %'-10.4f\n", "loadStrategy", time_enum, th(time_enum));
        }   
        printf("  %-12s %'-12f %'-10.4f\n", "ipredict", time_pred, th(time_pred));
        printf("  %-12s %'-12f %'-10.4f\n", "ibitplane", time_bitplane, th(time_bitplane));
        printf("  %-12s %'-12f %'-10.4f\n", "ilossless", time_lossless, th(time_lossless));
        double total_time = time_pred + time_bitplane + time_lossless;
        if constexpr (progressive == 1) {}
            total_time += time_enum;
        printf("  \e[1m%-12s\e[0m %'-12f %'-10.4f\n\n", "itotal", total_time, th(total_time));
        // printf("  target Error: %G\n", targeteb);
        // printf("  target RError: %G\n",  targetreb);
        // if constexpr (progressive == 0)
        //     printf("--------------------------------------\n");
    }
    if constexpr (decmp == 0) {
        printf("  %-12s %'-12f %'-10.4f\n", "predict", time_pred, th(time_pred));
        printf("  %-12s %'-12f %'-10.4f\n", "bitplane", time_bitplane, th(time_bitplane));
        printf("  %-12s %'-12f %'-10.4f\n", "lossless", time_lossless, th(time_lossless));
        double total_time =  time_lossless + time_bitplane + time_pred;
        printf("  \e[1m%-12s\e[0m %'-12f %'-10.4f\n\n", "total", total_time, th(total_time));
        double cr = 1.0 * total_bytes / total_compressed_size;
        printf("  compression ratio: %lf\n", 
            cr);
        printf("--------------------------------------\n");
    }
    // printf("  target Error: %G\n", config->target_ebs[i] * input->range);
    // printf("  target RError: %G\n",  config->target_ebs[i]);
    // 
}

template<typename T>
void calculate_accumulate_size(size_t& loaded_size, size_t total_size, int* begin_d, int* end_d, size_t* compressedSize_bp_d) {
    int bitplane_end[4], bitplane_start[4];
    size_t compressedSize_bp[4][32];
    size_t total_bytes = total_size * sizeof(T);
    CHECK_CUDA(cudaMemcpy(bitplane_start, begin_d, 1 * 4 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bitplane_end, end_d, 1 * 4 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&compressedSize_bp[0][0], compressedSize_bp_d, 4 * 32 * sizeof(size_t), cudaMemcpyDeviceToHost));

    for(int l = 3; l >= 0; --l) {
        for(int b = 31; b >= 0; --b) {
            if(b == 0) {
                if(l == 0)
                    continue;
                compressedSize_bp[l][b] -= compressedSize_bp[l-1][31]; // - chunks_level[3-l] / 16 ;
            }
            else  compressedSize_bp[l][b] -= compressedSize_bp[l][b-1]; //- chunks_level[3-l] / 16 ;
        }
    }

    // for(int l = 3; l >= 0; --l) {
    //     for(int b = 31; b >= 0; --b) {
    //         printf("%ld ", compressedSize_bp[l][b]);
    //     }
    //     printf("\n");
    // }

    size_t retrival_size = 0;
    for(int i = 0; i < 4; ++i) {
        // printf("begin: %d, end: %d\n", bitplane_start[i], bitplane_end[i]);
        int j = bitplane_start[i];// >= 1 ? bitplane_start[i] : 1;
        for(; j < bitplane_end[i]; ++j) {
            retrival_size += compressedSize_bp[i][j];
        }
    }
    loaded_size += retrival_size;
    // // printf("sign size: (%lf%%)\n", 1.0 * sign / bytes_ * 100);
    printf("  Data Chunk: %ld (%lf%%)\n", retrival_size, 1.0 * retrival_size / total_bytes * 100);
    printf("  Total Retrieved size: %ld (%lf%%)\n", loaded_size, 1.0 * loaded_size / total_bytes * 100);
    printf("  Bit rate: %lf\n", sizeof(T) * 8.0 * loaded_size / total_bytes);
    printf("--------------------------------------\n");
}