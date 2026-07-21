#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

enum class HybridSparseGroupedNaiveKernel {
    Dense,
    Sparse,
    Reduce,
};

enum class HybridSparseGroupedMode {
    Contiguous,
    Masked,
};

class SM90HybridSparseGroupedNaiveRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupedNaiveRuntime> {
public:
    struct Args {
        HybridSparseGroupedNaiveKernel kernel;
        void* a;
        void* block_selector;
        void* dense_values;
        void* sparse_values;
        void* sparse_metadata;
        void* grouped_index;
        void* dense_partial;
        void* sparse_partial;
        void* d;
        int total_rows;
        int n;
        int k;
        int num_experts;
        int max_m;
        int m_alignment;
        int block_n;
        int block_m;
        int grouped_mode;
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

__device__ __forceinline__ bool resolve_grouped_row(
        const int tile,
        const int row_in_tile,
        const int* grouped_index,
        const int total_rows,
        const int num_experts,
        const int max_m,
        const int m_alignment,
        const int grouped_mode,
        int& expert,
        int& row_m) {
    if (grouped_mode == 0) {
        row_m = tile * kBlock + row_in_tile;
        if (row_m >= total_rows)
            return false;

        int previous_end = 0;
        for (int current_expert = 0; current_expert < num_experts; ++current_expert) {
            const int start = current_expert == 0
                ? 0
                : ((previous_end + m_alignment - 1) / m_alignment) * m_alignment;
            const int end = grouped_index[current_expert];
            if (row_m >= start && row_m < end) {
                expert = current_expert;
                return true;
            }
            previous_end = end;
        }
        return false;
    }

    const int tiles_per_expert = (max_m + kBlock - 1) / kBlock;
    expert = tile / tiles_per_expert;
    const int local_tile = tile % tiles_per_expert;
    const int local_m = local_tile * kBlock + row_in_tile;
    if (expert >= num_experts || local_m >= max_m) {
        row_m = total_rows;
        return false;
    }
    row_m = expert * max_m + local_m;
    return local_m < grouped_index[expert];
}
)";

        static constexpr auto dense = R"(
extern "C" __global__ void hybrid_sparse_grouped_dense_naive(
        const __nv_bfloat16* a,
        const long long* block_selector,
        const __nv_bfloat16* dense_values,
        const __nv_bfloat16*,
        const unsigned char*,
        const int* grouped_index,
        float* dense_partial,
        float*,
        __nv_bfloat16*,
        const int total_rows,
        const int n,
        const int k,
        const int num_experts,
        const int max_m,
        const int m_alignment,
        const int block_n,
        const int block_m,
        const int grouped_mode) {
    const int tile = static_cast<int>(blockIdx.x);
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_rows = n / kBlock;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_in_tile = linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        int expert;
        int row_m;
        if (row_n >= n || !resolve_grouped_row(
                tile, row_in_tile, grouped_index, total_rows, num_experts,
                max_m, m_alignment, grouped_mode, expert, row_m))
            continue;

        const int block_row = row_n / kBlock;
        const int row_in_block = row_n % kBlock;
        float accumulator = 0.0f;

        for (int group = 0; group < block_groups; ++group) {
            const long long topology_index =
                (static_cast<long long>(expert) * block_rows + block_row)
                * block_groups + group;
            const unsigned long long selector = static_cast<unsigned long long>(
                block_selector[topology_index]);
            int dense_slot = 0;
            for (int local_block = 0; local_block < block_m; ++local_block) {
                if ((selector >> local_block) & 1ULL)
                    continue;

                const int global_k = (group * block_m + local_block) * kBlock;
                const long long value_base =
                    ((topology_index * dense_count + dense_slot) * kBlock
                      + row_in_block) * kBlock;
                for (int inner_k = 0; inner_k < kBlock; ++inner_k) {
                    accumulator += __bfloat162float(a[row_m * k + global_k + inner_k])
                                 * __bfloat162float(dense_values[value_base + inner_k]);
                }
                ++dense_slot;
            }
        }
        dense_partial[static_cast<long long>(row_m) * n + row_n] = accumulator;
    }
}
)";

        static constexpr auto sparse = R"(
extern "C" __global__ void hybrid_sparse_grouped_2_4_naive(
        const __nv_bfloat16* a,
        const long long* block_selector,
        const __nv_bfloat16*,
        const __nv_bfloat16* sparse_values,
        const unsigned char* sparse_metadata,
        const int* grouped_index,
        float*,
        float* sparse_partial,
        __nv_bfloat16*,
        const int total_rows,
        const int n,
        const int k,
        const int num_experts,
        const int max_m,
        const int m_alignment,
        const int block_n,
        const int block_m,
        const int grouped_mode) {
    const int tile = static_cast<int>(blockIdx.x);
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_rows = n / kBlock;
    const int block_groups = k / (kBlock * block_m);

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_in_tile = linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        int expert;
        int row_m;
        if (row_n >= n || !resolve_grouped_row(
                tile, row_in_tile, grouped_index, total_rows, num_experts,
                max_m, m_alignment, grouped_mode, expert, row_m))
            continue;

        const int block_row = row_n / kBlock;
        const int row_in_block = row_n % kBlock;
        float accumulator = 0.0f;

        for (int group = 0; group < block_groups; ++group) {
            const long long topology_index =
                (static_cast<long long>(expert) * block_rows + block_row)
                * block_groups + group;
            const unsigned long long selector = static_cast<unsigned long long>(
                block_selector[topology_index]);
            int sparse_slot = 0;
            for (int local_block = 0; local_block < block_m; ++local_block) {
                if (((selector >> local_block) & 1ULL) == 0)
                    continue;

                const int global_k = (group * block_m + local_block) * kBlock;
                const long long stream_base =
                    (topology_index * block_n + sparse_slot) * kBlock
                    + row_in_block;
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
        sparse_partial[static_cast<long long>(row_m) * n + row_n] = accumulator;
    }
}
)";

        static constexpr auto reduce = R"(
extern "C" __global__ void hybrid_sparse_grouped_reduce_naive(
        const __nv_bfloat16*,
        const long long*,
        const __nv_bfloat16*,
        const __nv_bfloat16*,
        const unsigned char*,
        const int* grouped_index,
        const float* dense_partial,
        const float* sparse_partial,
        __nv_bfloat16* d,
        const int total_rows,
        const int n,
        const int,
        const int num_experts,
        const int max_m,
        const int m_alignment,
        const int,
        const int,
        const int grouped_mode) {
    const int tile = static_cast<int>(blockIdx.x);
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;

    for (int linear = static_cast<int>(threadIdx.x);
         linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_in_tile = linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        int expert;
        int row_m;
        if (row_n >= n)
            continue;
        const bool valid = resolve_grouped_row(
            tile, row_in_tile, grouped_index, total_rows, num_experts,
            max_m, m_alignment, grouped_mode, expert, row_m);
        if (row_m >= total_rows)
            continue;

        const long long index = static_cast<long long>(row_m) * n + row_n;
        d[index] = valid
            ? __float2bfloat16_rn(dense_partial[index] + sparse_partial[index])
            : __float2bfloat16_rn(0.0f);
    }
}
)";

        switch (args.kernel) {
            case HybridSparseGroupedNaiveKernel::Dense:
                return std::string(common) + dense;
            case HybridSparseGroupedNaiveKernel::Sparse:
                return std::string(common) + sparse;
            case HybridSparseGroupedNaiveKernel::Reduce:
                return std::string(common) + reduce;
        }
        DG_HOST_UNREACHABLE("Unknown hybrid sparse grouped naive kernel");
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
            args.grouped_index,
            args.dense_partial,
            args.sparse_partial,
            args.d,
            args.total_rows,
            args.n,
            args.k,
            args.num_experts,
            args.max_m,
            args.m_alignment,
            args.block_n,
            args.block_m,
            args.grouped_mode));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_gemm_naive(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& grouped_index,
        const torch::Tensor& dense_partial,
        const torch::Tensor& sparse_partial,
        const torch::Tensor& d,
        const int total_rows,
        const int n,
        const int k,
        const int num_experts,
        const int max_m,
        const int m_alignment,
        const int block_n,
        const int block_m,
        const HybridSparseGroupedMode grouped_mode) {
    const int grid_m = grouped_mode == HybridSparseGroupedMode::Contiguous
        ? (total_rows + 63) / 64
        : num_experts * ((max_m + 63) / 64);
    const auto grid = std::make_pair(grid_m, (n + 63) / 64);
    auto args = SM90HybridSparseGroupedNaiveRuntime::Args {
        .kernel = HybridSparseGroupedNaiveKernel::Dense,
        .a = a.data_ptr(),
        .block_selector = block_selector.data_ptr(),
        .dense_values = dense_values.data_ptr(),
        .sparse_values = sparse_values.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .grouped_index = grouped_index.data_ptr(),
        .dense_partial = dense_partial.data_ptr(),
        .sparse_partial = sparse_partial.data_ptr(),
        .d = d.data_ptr(),
        .total_rows = total_rows,
        .n = n,
        .k = k,
        .num_experts = num_experts,
        .max_m = max_m,
        .m_alignment = m_alignment,
        .block_n = block_n,
        .block_m = block_m,
        .grouped_mode = static_cast<int>(grouped_mode),
        .launch_args = LaunchArgs(grid, 256),
    };

    const auto dense_runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_dense_naive",
        SM90HybridSparseGroupedNaiveRuntime::generate(args));
    SM90HybridSparseGroupedNaiveRuntime::launch(dense_runtime, args);

    args.kernel = HybridSparseGroupedNaiveKernel::Sparse;
    const auto sparse_runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_2_4_naive",
        SM90HybridSparseGroupedNaiveRuntime::generate(args));
    SM90HybridSparseGroupedNaiveRuntime::launch(sparse_runtime, args);

    args.kernel = HybridSparseGroupedNaiveKernel::Reduce;
    const auto reduce_runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_reduce_naive",
        SM90HybridSparseGroupedNaiveRuntime::generate(args));
    SM90HybridSparseGroupedNaiveRuntime::launch(reduce_runtime, args);
}

} // namespace deep_gemm
