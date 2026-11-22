#pragma once

#include <cstddef>

enum class AccessMode { Coalesced, NonCoalesced };

// Все указатели — на память устройства. Результат пишется в отдельный буфер out.

// out[i] = x[i] + y[i]
void launch_add(const float *x, const float *y, float *out, std::size_t n, AccessMode mode,
                int block_size);

// out[i] = alpha * x[i] + y[i]
void launch_saxpy(float alpha, const float *x, const float *y, float *out, std::size_t n,
                  AccessMode mode, int block_size);

// out[i] = max(0, x[i])
void launch_relu(const float *x, float *out, std::size_t n, AccessMode mode, int block_size);
