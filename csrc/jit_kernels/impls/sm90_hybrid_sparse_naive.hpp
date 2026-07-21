#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

enum class HybridSparseNaiveKernel {
    Dense,
    Sparse,
    Reduce,
};

class SM90HybridSparseNaiveRuntime final:
        public LaunchRuntime<SM90HybridSparseNaiveRuntime> {
public:
    struct Args {
        HybridSparseNaiveKernel kernel;
        void* a;
        void* block_selector;
        void* dense_values;
        void* sparse_values;
        void* sparse_metadata;
        void* dense_partial;
        void* sparse_partial;
        void* d;
        int m;
        int n;
        int k;
        int block_n;
        int block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        static constexpr auto common = R"(
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int kBlock = 64;

__device__ __forceinline__ void decode_pair(
        const unsigned char code, int& first, int& second) {
    switch (code) {
        case 0: first = 0; second = 1; break;
        case 1: first = 0; second = 2; break;
        case 2: first = 0; second = 3; break;
        case 3: first = 1; second = 2; break;
        case 4: first = 1; second = 3; break;
        default: first = 2; second = 3; break;
    }
}
)";

        static constexpr auto dense = R"(
extern "C" __global__ void hybrid_sparse_dense_naive(
        const __nv_bfloat16* a,
        const long long* block_selector,
        const __nv_bfloat16* dense_values,
        const __nv_bfloat16*,
        const unsigned char*,
        float* dense_partial,
        float*,
        __nv_bfloat16*,
        const int m,
        const int n,
        const int k,
        const int block_n,
        const int block_m) {
    const int tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        if (row_m >= m || row_n >= n)
            continue;

        const int block_row = row_n / kBlock;
        const int row_in_block = row_n % kBlock;
        float accumulator = 0.0f;

        for (int group = 0; group < block_groups; ++group) {
            const unsigned long long selector = static_cast<unsigned long long>(
                block_selector[block_row * block_groups + group]);
            int dense_slot = 0;
            for (int local_block = 0; local_block < block_m; ++local_block) {
                if ((selector >> local_block) & 1ULL)
                    continue;

                const int global_k = (group * block_m + local_block) * kBlock;
                const long long value_base =
                    (((static_cast<long long>(block_row) * block_groups + group)
                       * dense_count + dense_slot) * kBlock + row_in_block)
                    * kBlock;
                for (int inner_k = 0; inner_k < kBlock; ++inner_k) {
                    accumulator += __bfloat162float(a[row_m * k + global_k + inner_k])
                                 * __bfloat162float(dense_values[value_base + inner_k]);
                }
                ++dense_slot;
            }
        }
        dense_partial[row_m * n + row_n] = accumulator;
    }
}
)";

        static constexpr auto sparse = R"(
extern "C" __global__ void hybrid_sparse_2_4_naive(
        const __nv_bfloat16* a,
        const long long* block_selector,
        const __nv_bfloat16*,
        const __nv_bfloat16* sparse_values,
        const unsigned char* sparse_metadata,
        float*,
        float* sparse_partial,
        __nv_bfloat16*,
        const int m,
        const int n,
        const int k,
        const int block_n,
        const int block_m) {
    const int tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_groups = k / (kBlock * block_m);

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        if (row_m >= m || row_n >= n)
            continue;

        const int block_row = row_n / kBlock;
        const int row_in_block = row_n % kBlock;
        float accumulator = 0.0f;

        for (int group = 0; group < block_groups; ++group) {
            const unsigned long long selector = static_cast<unsigned long long>(
                block_selector[block_row * block_groups + group]);
            int sparse_slot = 0;
            for (int local_block = 0; local_block < block_m; ++local_block) {
                if (((selector >> local_block) & 1ULL) == 0)
                    continue;

                const int global_k = (group * block_m + local_block) * kBlock;
                const long long stream_base =
                    ((static_cast<long long>(block_row) * block_groups + group)
                      * block_n + sparse_slot) * kBlock + row_in_block;
                const long long value_base = stream_base * (kBlock / 2);
                const long long metadata_base = stream_base * (kBlock / 4);

                for (int quartet = 0; quartet < kBlock / 4; ++quartet) {
                    int first;
                    int second;
                    decode_pair(sparse_metadata[metadata_base + quartet], first, second);
                    const int quartet_k = global_k + quartet * 4;
                    const long long pair_base = value_base + quartet * 2;
                    accumulator += __bfloat162float(a[row_m * k + quartet_k + first])
                                 * __bfloat162float(sparse_values[pair_base]);
                    accumulator += __bfloat162float(a[row_m * k + quartet_k + second])
                                 * __bfloat162float(sparse_values[pair_base + 1]);
                }
                ++sparse_slot;
            }
        }
        sparse_partial[row_m * n + row_n] = accumulator;
    }
}
)";

        static constexpr auto reduce = R"(
extern "C" __global__ void hybrid_sparse_reduce_naive(
        const __nv_bfloat16*,
        const long long*,
        const __nv_bfloat16*,
        const __nv_bfloat16*,
        const unsigned char*,
        const float* dense_partial,
        const float* sparse_partial,
        __nv_bfloat16* d,
        const int m,
        const int n,
        const int,
        const int,
        const int) {
    const int tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        if (row_m < m && row_n < n) {
            const long long index = static_cast<long long>(row_m) * n + row_n;
            d[index] = __float2bfloat16_rn(
                dense_partial[index] + sparse_partial[index]);
        }
    }
}
)";

        switch (args.kernel) {
            case HybridSparseNaiveKernel::Dense:
                return std::string(common) + dense;
            case HybridSparseNaiveKernel::Sparse:
                return std::string(common) + sparse;
            case HybridSparseNaiveKernel::Reduce:
                return std::string(common) + reduce;
        }
        DG_HOST_UNREACHABLE("Unknown hybrid sparse naive kernel");
    }

    static void launch_impl(
            const KernelHandle& kernel,
            const LaunchConfigHandle& config,
            Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel,
            config,
            args.a,
            args.block_selector,
            args.dense_values,
            args.sparse_values,
            args.sparse_metadata,
            args.dense_partial,
            args.sparse_partial,
            args.d,
            args.m,
            args.n,
            args.k,
            args.block_n,
            args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_gemm_naive(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& dense_partial,
        const torch::Tensor& sparse_partial,
        const torch::Tensor& d,
        const int m,
        const int n,
        const int k,
        const int block_n,
        const int block_m) {
    const auto grid = std::make_pair((m + 63) / 64, (n + 63) / 64);
    auto args = SM90HybridSparseNaiveRuntime::Args {
        .kernel = HybridSparseNaiveKernel::Dense,
        .a = a.data_ptr(),
        .block_selector = block_selector.data_ptr(),
        .dense_values = dense_values.data_ptr(),
        .sparse_values = sparse_values.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .dense_partial = dense_partial.data_ptr(),
        .sparse_partial = sparse_partial.data_ptr(),
        .d = d.data_ptr(),
        .m = m,
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = block_m,
        .launch_args = LaunchArgs(grid, 256),
    };

    const auto dense_runtime = compiler->build(
        "sm90_hybrid_sparse_dense_naive",
        SM90HybridSparseNaiveRuntime::generate(args));
    SM90HybridSparseNaiveRuntime::launch(dense_runtime, args);

    args.kernel = HybridSparseNaiveKernel::Sparse;
    const auto sparse_runtime = compiler->build(
        "sm90_hybrid_sparse_2_4_naive",
        SM90HybridSparseNaiveRuntime::generate(args));
    SM90HybridSparseNaiveRuntime::launch(sparse_runtime, args);

    args.kernel = HybridSparseNaiveKernel::Reduce;
    const auto reduce_runtime = compiler->build(
        "sm90_hybrid_sparse_reduce_naive",
        SM90HybridSparseNaiveRuntime::generate(args));
    SM90HybridSparseNaiveRuntime::launch(reduce_runtime, args);
}

} // namespace deep_gemm
