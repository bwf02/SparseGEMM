#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/arch/copy_sm90_desc.hpp>

// Reduction-only iteration for the 128x128 compute kernels. A 64-column tile
// restores grid parallelism without changing the dense or sparse mainloop.
template <int = 0>
__global__ void hybrid_sparse_reduce64_wgmma_tma_block128x128(
        const long long*, const unsigned char*, const float* dense_partial,
        const float* sparse_partial, __nv_bfloat16* output,
        const cute::TmaDescriptor, const cute::TmaDescriptor,
        const cute::TmaDescriptor, const int m, const int n, const int,
        const int, const int) {
    constexpr int kTileM = 64;
    constexpr int kTileN = 64;
    const int tile_m = static_cast<int>(blockIdx.x) * kTileM;
    const int tile_n = static_cast<int>(blockIdx.y) * kTileN;
    for (int linear = static_cast<int>(threadIdx.x); linear < kTileM * kTileN;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kTileN;
        const int row_n = tile_n + linear % kTileN;
        if (row_m < m && row_n < n) {
            const long long index = static_cast<long long>(row_m) * n + row_n;
            output[index] = __float2bfloat16_rn(
                dense_partial[index] + sparse_partial[index]);
        }
    }
}
