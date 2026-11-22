#include "cuda_check.h"
#include "kernels.h"

namespace {

// В некоалесцированных ядрах каждый поток обрабатывает непрерывный кусок из kChunk элементов.
// Соседние потоки варпа обращаются к адресам на расстоянии kChunk * 4 = 256 байт, поэтому
// одно обращение варпа затрагивает 32 разных сегмента памяти вместо одного.
constexpr std::size_t kChunk = 64;

// ---------------------------------------------------------------------------
// Коалесцированные ядра: соседние потоки читают и пишут соседние float.
// ---------------------------------------------------------------------------

__global__ void add_coalesced(const float *x, const float *y, float *out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

__global__ void saxpy_coalesced(float alpha, const float *x, const float *y, float *out,
                                std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        out[i] = alpha * x[i] + y[i];
    }
}

__global__ void relu_coalesced(const float *x, float *out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Некоалесцированные ядра: каждый поток обрабатывает свой кусок [start, start + kChunk).
// ---------------------------------------------------------------------------

__global__ void add_noncoalesced(const float *x, const float *y, float *out, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = start + kChunk < n ? start + kChunk : n;
    for (std::size_t i = start; i < end; ++i) {
        out[i] = x[i] + y[i];
    }
}

__global__ void saxpy_noncoalesced(float alpha, const float *x, const float *y, float *out,
                                   std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = start + kChunk < n ? start + kChunk : n;
    for (std::size_t i = start; i < end; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}

__global__ void relu_noncoalesced(const float *x, float *out, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t start = tid * kChunk;
    const std::size_t end = start + kChunk < n ? start + kChunk : n;
    for (std::size_t i = start; i < end; ++i) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

// Число блоков: в коалесцированном режиме один поток на элемент,
// в некоалесцированном — один поток на kChunk элементов.
unsigned grid_size(std::size_t n, AccessMode mode, int block_size) {
    const std::size_t threads = mode == AccessMode::Coalesced ? n : (n + kChunk - 1) / kChunk;
    return static_cast<unsigned>((threads + block_size - 1) / block_size);
}

} // namespace

void launch_add(const float *x, const float *y, float *out, std::size_t n, AccessMode mode,
                int block_size) {
    if (n == 0) {
        return;
    }
    const unsigned grid = grid_size(n, mode, block_size);
    if (mode == AccessMode::Coalesced) {
        add_coalesced<<<grid, block_size>>>(x, y, out, n);
    } else {
        add_noncoalesced<<<grid, block_size>>>(x, y, out, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_saxpy(float alpha, const float *x, const float *y, float *out, std::size_t n,
                  AccessMode mode, int block_size) {
    if (n == 0) {
        return;
    }
    const unsigned grid = grid_size(n, mode, block_size);
    if (mode == AccessMode::Coalesced) {
        saxpy_coalesced<<<grid, block_size>>>(alpha, x, y, out, n);
    } else {
        saxpy_noncoalesced<<<grid, block_size>>>(alpha, x, y, out, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu(const float *x, float *out, std::size_t n, AccessMode mode, int block_size) {
    if (n == 0) {
        return;
    }
    const unsigned grid = grid_size(n, mode, block_size);
    if (mode == AccessMode::Coalesced) {
        relu_coalesced<<<grid, block_size>>>(x, out, n);
    } else {
        relu_noncoalesced<<<grid, block_size>>>(x, out, n);
    }
    CUDA_CHECK(cudaGetLastError());
}
