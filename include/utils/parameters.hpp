#pragma once

#include "traits.hpp"
#include "err.hpp"
#include <stdint.h>
#include <vector>

struct interpolation_parameters
{
    double alpha{1.75};
    double beta{4.0};
    //
    //bool interpolators[3];
    bool use_md[4];
    bool use_natural[4];
    //
    bool reverse[4];
    uint8_t auto_tuning{3};

    //
    interpolation_parameters() : use_md{true, true, true, true}, 
    use_natural{false, false, false, false},
    reverse{false, false, false, false} {}
};


typedef struct prism_context {
    double eb{0.0};
    // double abs_eb{0.0};
    double rel_eb{0.0};
    double pw_rel_eb{0.0};
    uint32_t radius{0};// radius{128};
    int ndim{-1};
    uint32_t x{1}, y{1}, z{1}, w{1};
    uint32_t size{1};
    bool isComp{false};
    bool isDecomp{false};
    prism_dtype dtype{F4};
    prism_mode error_mode{REL};
    std::string oriFilePath;
    std::string cmpFilePath;
    std::string decFilePath;
    int blocks_num{1};
    int levels{15};
    int bitgroups{8};
    int compMode{0};
    // bool show_result{0};
    bool report_time{false};
    bool report_cr{false};
    size_t loaded_size{0};
    btype bt{SM};
    std::vector<double> target_ebs;
    interpolation_parameters intp_param;
    int* begin;
    int* end;
} context;


void usage();
void parse_argv(prism_context* config, int argc, char** argv);
