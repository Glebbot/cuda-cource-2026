cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu)

target_include_directories(submission PRIVATE include)

set_target_properties(
  submission
  PROPERTIES CXX_STANDARD 17
             CUDA_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD_REQUIRED ON)

# Кластер: Quadro RTX 6000 (compute capability 7.5). PTX compute_75 JIT-компилируется и на более
# новых GPU (локально RTX 3070, 8.6). Переопределить: -DLAB1_CUDA_ARCH=86.
set(LAB1_CUDA_ARCH
    75
    CACHE STRING "CUDA architecture for the submission target")
set_property(TARGET submission PROPERTY CUDA_ARCHITECTURES ${LAB1_CUDA_ARCH})

target_compile_options(submission PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
