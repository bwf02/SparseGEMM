#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

class SM90HybridSparseGroupedContiguousOutput128x128NM12Stage2DualWGRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupedContiguousOutput128x128NM12Stage2DualWGRuntime> {
public:
    struct Args {
        void* block_selector;
        void* hardware_metadata;
        void* grouped_index;
        CUtensorMap tensor_map_activation;
        CUtensorMap tensor_map_dense;
        CUtensorMap tensor_map_sparse;
        CUtensorMap tensor_map_output;
        int total_m;
        int num_experts;
        int m_alignment;
        int n;
        int k;
        int block_n;
        int block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_contiguous_output128x128_nm12_stage2_dual_wg.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_grouped_contiguous_output128x128_nm12_stage2_dual_wg<
            {}, {}, {}, {}, {}, {}, {}>);
    (void)ptr;
}}
)", args.block_n, args.block_m, 2, args.num_experts,
            args.total_m, args.n, args.k);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata,
            args.grouped_index, args.tensor_map_activation,
            args.tensor_map_dense, args.tensor_map_sparse,
            args.tensor_map_output, args.total_m, args.num_experts,
            args.m_alignment, args.n, args.k, args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_contiguous_output128x128_nm12_stage2_dual_wg(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& grouped_index, const torch::Tensor& d,
        const int total_m, const int num_experts, const int m_alignment,
        const int n, const int k, const int block_n, const int block_m) {
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(total_m % 128 == 0 and m_alignment == 128);
    DG_HOST_ASSERT(n % 128 == 0);
    constexpr int num_stages = 2;
    constexpr int output_bytes = 128 * 128 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    const int dense_count = block_m - block_n;
    const int stage_bytes =
        2 * dense_count * 64 * 64 * sizeof(__nv_bfloat16) +
        2 * block_n * 64 * 32 * sizeof(__nv_bfloat16) +
        128 * 128 * sizeof(__nv_bfloat16) +
        2 * block_n * 2 * 4 * 16 * sizeof(std::uint32_t);
    const int pipeline_bytes =
        num_stages * stage_bytes +
        2 * num_stages * sizeof(std::uint64_t);
    const int output_offset = ((pipeline_bytes + 1023) / 1024) * 1024;
    const int smem_bytes = output_offset + output_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, total_m, 128, 128, k, 128);
    const auto tensor_map_dense = make_tma_2d_desc(
        dense_values, 64,
        num_experts * block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128);
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32,
        num_experts * block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, total_m, n, 128, 128, n, 1, 128);
    const int total_tiles = (total_m / 128) * (n / 128);
    const auto args = SM90HybridSparseGroupedContiguousOutput128x128NM12Stage2DualWGRuntime::Args {
        .block_selector = block_selector.data_ptr(),
        .hardware_metadata = hardware_metadata.data_ptr(),
        .grouped_index = grouped_index.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .tensor_map_output = tensor_map_output,
        .total_m = total_m, .num_experts = num_experts,
        .m_alignment = m_alignment,
        .n = n, .k = k, .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(total_tiles, 384, smem_bytes),
    };
    const auto runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_contiguous_output128x128_nm12_stage2_dual_wg",
        SM90HybridSparseGroupedContiguousOutput128x128NM12Stage2DualWGRuntime::generate(args));
    SM90HybridSparseGroupedContiguousOutput128x128NM12Stage2DualWGRuntime::launch(runtime, args);
}

} // namespace deep_gemm
