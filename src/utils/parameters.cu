#include "parameters.hpp"
#include "stdint.h"
#include <stdexcept>  
#include <string>  
#include <stdlib.h>
#include <cstring> 
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

void usage() {
        std::cout << R"(
Options:
  -i <path> : Path to binary input file containing the original data
  -x <path> : Path to decompressed binary output file (optional)
  -z <path> : Path to compressed binary output/input file (optional)
  -f : single precision (float type)
  -d : double precision (double type)
  -1 <nx> : Dimensions for 1D array a[nx]
  -2 <nx> <ny> : Dimensions for 2D array a[ny][nx]
  -3 <nx> <ny> <nz> : Dimensions for 3D array a[nz][ny][nx]
  -4 <nx> <ny> <nz> <nw> : Dimensions for 4D array a[nw][nz][ny][nx]
  -A [errorBound] : Absolute error bound
  -R [errorBound] : Relative error bound
  --report [time,cr] : output the compression ratio/time
  -prog : Enable progressive compression/decompression
  -errors -<nums> <error_1> <error_2> ... : Perform progressive decompression multiple times with gradually decreasing error bounds
Examples:
  ./prism -i [oriFilePath] -f -3 [dim_z] [dim_y] [dim_x] -R [errorBound] -z [cmpFilePath] -x [decFilePath] --report time,cr
  ./prism -i [oriFilePath] -d -3 [dim_z] [dim_y] [dim_x] -R [errorBound] -z [cmpFilePath] -x [decFilePath] -prog -errors -2 1e-1 1e-2
)" << std::endl;

}

void check(prism_context* config) {
    if(config->ndim == -1) {
        throw std::invalid_argument("dim must be specified.");
    }
    if (config->report_cr == 1) {
        if (config->oriFilePath.empty())
            throw std::invalid_argument("input file path must be specified.");
    }
    if (config->isComp == 1) {
        if (config->oriFilePath.empty())
            throw std::invalid_argument("input file path must be specified.");
        if (config->cmpFilePath.empty())
            config->cmpFilePath = config->oriFilePath + ".prisma";
    }
    if(config->isDecomp == 1) {
        if (config->oriFilePath.empty() && config->cmpFilePath.empty())
            throw std::invalid_argument("compressed file path must be specified.");
        if (config->decFilePath.empty()) {
            if (!config->cmpFilePath.empty())
                config->decFilePath = config->cmpFilePath + ".out";
            else 
                config->decFilePath = config->oriFilePath + ".prisma.out";
        }
    }
}

void parse_argv(prism_context* config, int argc, char** argv) {

    if(argc == 1)
        usage();

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            usage();
            exit(0);
        }
        else if (strcmp(argv[i], "-f") == 0) {
            config->dtype = F4;
        }
        else if (strcmp(argv[i], "-d") == 0) {
            config->dtype = F8;
        }  
        else if (strcmp(argv[i], "-A") == 0) {
            // Error bound mode and value
            config->error_mode = ABS;
            if (i + 1 < argc) {
                config->eb = std::stod(argv[++i]);
            }
        } else if (strcmp(argv[i], "-R") == 0) {
            // Error bound mode and value
            config->error_mode = REL;
            if (i + 1 < argc) {
                config->rel_eb = std::stod(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-PW_R") == 0) {
            // Error bound mode and value
            config->error_mode = PW_REL;
            if (i + 1 < argc) {
                config->pw_rel_eb = std::stod(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-z") == 0) {
            config->isComp = true;
            // Compressed file path (optional)
            if (i + 1 < argc && argv[i + 1][0] != '-'){
                config->cmpFilePath = argv[++i];
            }
        } else if (strcmp(argv[i], "-x") == 0) {
            config->isDecomp = true;
            // Decompressed file path (optional)
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                config->decFilePath = argv[++i];
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            // Original file path (optional)
            if (i + 1 < argc) {
                config->oriFilePath = argv[++i];
            }
        }
        else if (strcmp(argv[i], "-1") == 0) {
            config->ndim = 1;
            if(i + 1 < argc) {
                config->x = std::stoi(argv[++i]);   
            }
            else 
                usage();
        }
        else if (strcmp(argv[i], "-2") == 0) {
            config->ndim = 2;
            if(i + 2 < argc) {
                config->x = std::stoi(argv[i+1]);
                config->y = std::stoi(argv[i+2]);
                i += 2;
            }
            else 
                usage();

        }
        else if (strcmp(argv[i], "-3") == 0) {
            config->ndim = 3;
            if(i + 3 < argc) {
                config->x = std::stoi(argv[i+1]);
                config->y = std::stoi(argv[i+2]);
                config->z = std::stoi(argv[i+3]);
                i += 3;
            }
            else
                usage();
        }
        else if (strcmp(argv[i], "-errors") == 0) {
            // multiple target eb for progressive coding
            // parse the number of errors
            config->compMode = 1;
            if (i + 1 < argc) {
                int num_errors = 0;
                try
                {
                    num_errors = std::stoi(argv[++i] + 1);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
                config->target_ebs.resize(num_errors);
                for (int j = 0; j < num_errors; ++j) {
                    if (i + 1 < argc) {
                        config->target_ebs[j] = std::stod(argv[++i]);
                    } else {
                        throw std::invalid_argument("Not enough error bounds provided.");
                    }
                }
            } else {
                throw std::invalid_argument("Number of error bounds not specified.");
            }
        }
        else if (strcmp(argv[i], "--report") == 0) {
            if (i + 1 < argc) {
                std::string value = argv[++i];
                std::stringstream ss(value);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    if (token == "time") {
                        config->report_time = true;
                    } else if (token == "cr") {
                        config->report_cr = true;
                    } else {
                        std::cerr << "Warning: unknown report option: " << token << std::endl;
                    }
                }
            } else {
                std::cerr << "Error: missing argument after --report\n";
                return ;
            }
        }
        else if (strcmp(argv[i], "-prog") == 0) {
            // if(strcmp(argv[++i], "prog") == 0) 
            config->bt = NB;
        }
        // test
        else if (strcmp(argv[i], "-cubic") == 0) {
            if (i + 1 < argc) {
                if(strcmp(argv[++i], "natural") == 0) 
                    for(int k = 0; k < 6; ++k)
                        config->intp_param.use_natural[k]=1;
            }
        }
        else if (strcmp(argv[i], "-md") == 0) {
            if (i + 1 < argc) {
                if(strcmp(argv[i+1], "true") == 0) 
                    for(int k = 0; k < 6; ++k)
                        config->intp_param.use_md[k]=1;
                else if (strcmp(argv[i+1], "false") == 0) 
                    for(int k = 0; k < 6; ++k)
                        config->intp_param.use_md[k]=0;
                ++i;
            }
        }
        else if (strcmp(argv[i], "-reverse") == 0) {
            for(int k = 0; k < 6; ++k)
                config->intp_param.reverse[k]=1;
        }
        //test
        else {
            printf("Error: Unrecognized option '%s'.\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }
    config->size = config->x * config->y * config->z * config->w;
    check(config);
    cudaMalloc(&config->begin, 1 * 4 * sizeof(int));
    cudaMalloc(&config->end, 1 * 4 * sizeof(int));
    cudaMemset(config->begin, 0, 1 * 4 * sizeof(int));
    cudaMemset(config->end, 0, 1 * 4 * sizeof(int));
}
