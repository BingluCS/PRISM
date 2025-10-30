#include <iostream>
#include <cuda_runtime.h>

#include "prism.hpp"
#include "compressor.hpp"



int main(int argc, char* argv[])
{
    prism_context* config = new prism_context;
    parse_argv(config, argc, argv);
    initialize(config);
    return 0;
}