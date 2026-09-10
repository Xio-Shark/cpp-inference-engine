cmake_minimum_required(VERSION 3.19)

if(APPLE)
    project(tiny_inference LANGUAGES CXX OBJCXX)
else()
    project(tiny_inference LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# nlohmann/json: use bundled single-header (include/nlohmann/json.hpp)

if(APPLE)
    message(STATUS "Configuring for macOS with Apple Metal & MetalPerformanceShaders (MPS)")

    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    # Compile Metal shader into metallib if metal toolchain is available
    find_program(METAL_COMPILER metal)
    find_program(METALLIB_TOOL metallib)

    set(LIB_SOURCES
        src/tensor_metal.mm
        src/kernels_metal.mm
        src/safetensors.cpp
        src/weight_loader.cpp
        src/transformer.cpp
    )

    add_library(tiny_inference_lib ${LIB_SOURCES})
    target_include_directories(tiny_inference_lib PUBLIC include)

    # Link Apple frameworks
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

    target_link_libraries(tiny_inference_lib PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${MPS_FRAMEWORK}
    )

    target_compile_definitions(tiny_inference_lib PUBLIC -D__APPLE__=1)
    target_compile_options(tiny_inference_lib PUBLIC "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")

else()
    message(STATUS "Configuring for Linux/Windows with NVIDIA CUDA & cuBLAS")

    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)

    set(LIB_SOURCES
        src/tensor.cu
        src/safetensors.cpp
        src/weight_loader.cpp
        src/kernels.cu
        src/transformer.cpp
    )

    add_library(tiny_inference_lib ${LIB_SOURCES})
    target_include_directories(tiny_inference_lib PUBLIC include)
    target_link_libraries(tiny_inference_lib PUBLIC
        CUDA::cublas CUDA::cudart
    )
    target_compile_definitions(tiny_inference_lib PUBLIC -DUSE_CUDA=1)
    set_target_properties(tiny_inference_lib PROPERTIES
        CUDA_ARCHITECTURES "80;89"
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Main executable
add_executable(tiny_inference src/main.cpp)
target_link_libraries(tiny_inference PRIVATE tiny_inference_lib)

option(TINY_INFERENCE_BUILD_TESTS "Build kernel correctness tests" ON)

if(TINY_INFERENCE_BUILD_TESTS)
    enable_testing()
    add_executable(test_kernels tests/test_kernels.cpp)
    target_link_libraries(test_kernels PRIVATE tiny_inference_lib)
    add_test(NAME kernel_correctness COMMAND test_kernels)

    add_executable(test_model_loader tests/test_model_loader.cpp)
    target_link_libraries(test_model_loader PRIVATE tiny_inference_lib)
    add_test(NAME model_loader COMMAND test_model_loader)
endif()
