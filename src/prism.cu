#include "prism.hpp"
#include "io.hpp"
#include "compressor.hpp"
#include "err.hpp"
#include "timer.hpp"
#include "dataloader.hpp"
#include "analyze.hpp"

template<typename T>
void prism_compress(prism_context* config, void* stream) {
    auto* cmp = new prism::Compressor<T,i4>();
    auto input = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);      
    input->template load_fromfile<ori_File>(config->oriFilePath);
    input->H2D();

    if(config->error_mode == REL) {
        config->eb = config->rel_eb * input->findMinMax();
        // printf("error bound: %.60lf\n", config->eb );
    }

    if(config->isComp) {

    if(config->report_time == 1) {
        auto* cmp_tmp = new prism::Compressor<T,i4>();
        cmp_tmp->init(config);
        for(int i = 0; i < 1; ++i) {
            cmp_tmp->compress_pipeline(config, input, stream);
        }
	delete cmp_tmp;
    } // warm up

    cmp->init(config);
    cmp->compress_pipeline(config, input, stream);
    cmp->compressed_data->unload_tofile(config->cmpFilePath, cmp->total_compressed_size, prism::typetofile::deviceTofile);
    
    if(config->report_time == 1)
        print_result<T, 0, 0>(config->size, cmp->total_compressed_size, config->eb, 
            config->rel_eb, cmp->itime_enum, cmp->time_pred, cmp->time_bitplane, cmp->time_encode);
    }
    delete cmp;
    delete input;
}

template<typename T>
void prism_decompress(prism_context* config, void* stream) {
    if(config->report_time == 1) {
        for(int i = 0; i < 1; ++i) {
            auto output_tmp = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
            auto decmp_tmp = new prism::Compressor<T,i4>();
            decmp_tmp->init(config);
            decmp_tmp->compressed_data->template load_fromfile<cmp_File>(config->cmpFilePath);
            decmp_tmp->compressed_data->H2D();
            decmp_tmp->decompress_pipeline(config, output_tmp, stream);
            delete decmp_tmp;
            delete output_tmp;
        }
    } // warm up

    auto decmp = new prism::Compressor<T,i4>();
    decmp->init(config);
    decmp->compressed_data->template load_fromfile<cmp_File>(config->cmpFilePath);
    decmp->compressed_data->H2D();
    auto output_new = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
    decmp->decompress_pipeline(config, output_new, stream);
    output_new->D2H();
    output_new->unload_tofile(config->decFilePath, prism::typetofile::deviceTofile);
    if(config->report_time == 1) {
        print_result<T, 1, 0>(config->size, decmp->total_compressed_size, config->eb, config->rel_eb, 
            decmp->itime_enum, decmp->itime_pred, decmp->itime_bitplane, decmp->itime_decode);
    }
    if (config->report_cr == 1) {
        auto input = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);      
        input->template load_fromfile<ori_File>(config->oriFilePath);
        statistic<T>(static_cast<T*>(input->h), static_cast<T*>(output_new->h), config->size);
    }
}

template<typename T>
void prism_progressive_decompress(prism_context* config, void* stream) {
    //progressive decompression

    prism::StatBuffer<T>* input = nullptr; 
    if(config->report_cr == 1) {
        input = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);      
        input->template load_fromfile<ori_File>(config->oriFilePath);
    }

    if(config->report_time == 1) {
        auto* decmp_tmp = new prism::Compressor<T,i4>();
        decmp_tmp->init(config);
        decmp_tmp->compressed_data->template load_fromfile<cmp_File>(config->cmpFilePath);
        decmp_tmp->compressed_data->H2D();
        auto output_tmp_new = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
        for(int i = 0; i < 1; ++i) {
            decmp_tmp->decompress_progressive_pipeline(config, nullptr, output_tmp_new, config->target_ebs[0],
            0, stream);
        }
        delete output_tmp_new;
        delete decmp_tmp;
        cudaMemset(config->begin, 0, 1 * 4 * sizeof(int));
        cudaMemset(config->end, 0, 1 * 4 * sizeof(int));
    } // warm up

    auto decmp = new prism::Compressor<T,i4>();
    decmp->init(config);
    decmp->compressed_data->template load_fromfile<cmp_File>(config->cmpFilePath);
    decmp->compressed_data->H2D();
    auto output_new = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
    auto output_old = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
    size_t loaded_size = decmp->ap->bytes;
    decmp->decompress_progressive_pipeline(config, output_old, output_new, config->target_ebs[0],
    0, stream);
    output_new->D2H();

    if(config->report_time == 1) {
        print_result<T, 1, 1>(config->size, decmp->total_compressed_size, config->eb, config->rel_eb, 
            decmp->itime_enum, decmp->itime_pred, decmp->itime_bitplane, decmp->itime_decode);
    }

    if (config->report_cr == 1) {
        statistic<T>(static_cast<T*>(input->h), static_cast<T*>(output_new->h), config->size);
        calculate_accumulate_size<T>(loaded_size, config->size, config->begin, config->end, decmp->compressedSize_bp_d);
    }
    int progressive_nums = config->target_ebs.size() - 1;
    for (int i = 1; i <= progressive_nums; ++i) {
        delete output_old;
        delete decmp;
        output_old = output_new;
        output_new = new prism::StatBuffer<T>(config->dtype, 0, config->x, config->y, config->z);
        decmp = new prism::Compressor<T,i4>();

        decmp->init(config);
        decmp->compressed_data->template load_fromfile<cmp_File>(config->cmpFilePath);
        decmp->compressed_data->H2D();

        decmp->decompress_progressive_pipeline(config, output_old, output_new, config->target_ebs[i], 
            config->target_ebs[i - 1], stream);
        output_new->D2H();
        if(config->report_time == 1) {
            print_result<T, 1, 1>(config->size, decmp->total_compressed_size, config->eb, config->rel_eb, 
                decmp->itime_enum, decmp->itime_pred, decmp->itime_bitplane, decmp->itime_decode);
        }
        if (config->report_cr == 1) {
            statistic<T>(static_cast<T*>(input->h), static_cast<T*>(output_new->h), config->size);
            calculate_accumulate_size<T>(loaded_size, config->size, config->begin, config->end, decmp->compressedSize_bp_d);
        }
        if(i == progressive_nums)
            output_new->unload_tofile(config->decFilePath, prism::typetofile::deviceTofile);
    }
}

void initialize(prism_context* config) {
   cudaStream_t stream;
   CHECK_CUDA(cudaStreamCreate(&stream));

   if(config->dtype == PRISM_TYPE_FLOAT) {

        if(config->isComp) {
            prism_compress<f4>(config, stream);
        }
        if(config->isDecomp) {
            if(config->compMode && config->target_ebs.size() > 0) {
                prism_progressive_decompress<f4>(config, stream);
            }
            else if(!config->compMode || config->target_ebs.size() == 0) { 
                prism_decompress<f4>(config, stream);
            }
            else {
                printf("unknown decompression mode.\n");
            }
       }
   }
   else if (config->dtype == PRISM_TYPE_DOUBLE ) {
        if(config->isComp) {
            prism_compress<f8>(config, stream);
        }

       if(config->isDecomp) {
            if(config->compMode && config->target_ebs.size() > 0) {
                prism_progressive_decompress<f8>(config, stream);
            }
            else if(!config->compMode || config->target_ebs.size() == 0) {  
                prism_decompress<f8>(config, stream);
            }
            else {
                printf("unknown decompression mode.\n");
            }
       }
   }

}
