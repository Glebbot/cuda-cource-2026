#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Ошибка CUDA — ошибка выполнения: сообщение в stderr и код возврата 1.
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                         cudaGetErrorString(err));                                                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)
