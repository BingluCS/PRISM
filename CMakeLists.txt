# Specify the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(PRISM VERSION 0.0.0 DESCRIPTION "Progressive lossy compression library")
add_library(prismcompile_settings INTERFACE)
target_compile_definitions(
    prismcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
    prismcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
            --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(prismcompile_settings INTERFACE cxx_std_17 cuda_std_17)
# Enable languages
enable_language(CXX CUDA)

# Find required CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # Set CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 89 90)
endif()
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "prefer shared libaries" ON)
# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE Release)
endif()

# Disable specific warnings (adjust as needed)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unused-result")

# List of source files
set(PRISM_SOURCES
    src/utils/parameters.cu
    src/utils/io.cu
    src/compressor.cu
    src/predictor/InterpolationPredictor.cu
    src/predictor/bitplane.cu
    src/loader/dataloader.cu
    src/encoder/zbpc.cu
    src/prism.cu
)

# Create static and shared libraries
add_library(prism_shared SHARED ${PRISM_SOURCES})

target_compile_options(prism_shared
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -O3
      >
    $<$<COMPILE_LANGUAGE:CXX>:
      -O3
     >
    )

# Include directories for both libraries
target_include_directories(prism_shared
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utils>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/predictor>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/loader>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/encoder>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link CUDA runtime
# target_link_libraries(prism_static PRIVATE prismcompile_settings CUDA::cudart)
target_link_libraries(prism_shared PRIVATE prismcompile_settings CUDA::cudart)

# Set output name for libraries
# set_target_properties(prism_static PROPERTIES OUTPUT_NAME prism)
set_target_properties(prism_shared PROPERTIES OUTPUT_NAME prism)

# Headers for installation
set(public_headers
    include/prism.hpp
    include/utils/io.hpp
    include/utils/parameters.hpp
    include/utils/err.hpp
    include/utils/analyze.hpp
    include/compressor.hpp
    include/predictor/InterpolationPredictor.hpp
    include/predictor/bitplane.hpp
    include/loader/dataloader.hpp
    include/encoder/lossless.hpp
    include/encoder/zbpc.hpp
)

# Optionally build examples
option(PRISM_BUILD_EXAMPLES "Option to enable building example programs" ON)
if (PRISM_BUILD_EXAMPLES)
    add_subdirectory(examples)  
endif ()
