cmake_minimum_required(VERSION 3.18)
project(tiny_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# nlohmann/json: use bundled single-header (include/nlohmann/json.hpp)
# No network required.

# Core library
add_library(tiny_inference_lib
    src/tensor.cu
    src/safetensors.cpp
    src/kernels.cu
    src/transformer.cu
)
target_include_directories(tiny_inference_lib PUBLIC include)
target_link_libraries(tiny_inference_lib PUBLIC
    CUDA::cublas CUDA::cudart
)
set_target_properties(tiny_inference_lib PROPERTIES
    CUDA_ARCHITECTURES "80;89"
    POSITION_INDEPENDENT_CODE ON
)

# Main executable
add_executable(tiny_inference src/main.cpp)
target_link_libraries(tiny_inference PRIVATE tiny_inference_lib)
