# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Define the install directory for binaries
set(install_dir ${PROJECT_BINARY_DIR}/)
set(prism "prism")

# Include the main include directory
include_directories(${PROJECT_SOURCE_DIR}/include 
                    ${CUDAToolkit_INCLUDE_DIRS})

# Compile executable binaries
add_executable(${prism} prism.cpp)

# Link the existing prism_static library
# target_link_libraries(${prism} PRIVATE prism_static)
target_link_libraries(${prism} PRIVATE prism_shared)

# Set output paths for the compiled binary
set_target_properties(${prism} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary
install(TARGETS ${prism} DESTINATION bin)

message("[prism::info] $\{CMAKE_VERSION\}: " ${CMAKE_VERSION})
message("[prism::info] $\{CMAKE_C_STANDARD\}$: " ${CMAKE_C_STANDARD})
message("[prism::info] $\{CMAKE_C_COMPILER\}$: " ${CMAKE_C_COMPILER})
message("[prism::info] $\{CMAKE_C_COMPILER_ID\}$: " ${CMAKE_C_COMPILER_ID})
message("[prism::info] $\{CMAKE_C_COMPILER_VERSION\}$: " ${CMAKE_C_COMPILER_VERSION})
message("[prism::info] $\{CMAKE_CXX_STANDARD\}$: " ${CMAKE_CXX_STANDARD})
message("[prism::info] $\{CMAKE_CXX_COMPILER\}$: " ${CMAKE_CXX_COMPILER})
message("[prism::info] $\{CMAKE_CXX_COMPILER_ID\}$: " ${CMAKE_CXX_COMPILER_ID})
message("[prism::info] $\{CMAKE_CXX_COMPILER_VERSION\}$: " ${CMAKE_CXX_COMPILER_VERSION})
message("C++ standard: ${CMAKE_CXX_STANDARD}")
message("[prism::info] $\{CMAKE_CUDA_STANDARD\}: " ${CMAKE_CUDA_STANDARD})
message("[prism::info] $\{CMAKE_CUDA_COMPILER\}: " ${CMAKE_CUDA_COMPILER})
message("[prism::info] $\{CMAKE_CUDA_COMPILER_ID\}: " ${CMAKE_CUDA_COMPILER_ID})
message("[prism::info] $\{CMAKE_CUDA_COMPILER_VERSION\}: " ${CMAKE_CUDA_COMPILER_VERSION})
message("[prism::info] $\{CMAKE_CUDA_ARCHITECTURES\}: " ${CMAKE_CUDA_ARCHITECTURES})