#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

class SM90HybridSparseGroupedMaskedOutput64x64NM12FixedStage2MaskedEpilogueRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupedMaskedOutput64x64NM12FixedStage2MaskedEpilogueRuntime> {
public:
    struct Args {
        void* block_selector;
        void* hardware_metadata;
        void* grouped_index;
        CUtensorMap tensor_map_activation;
        CUtensorMap tensor_map_dense;
        CUtensorMap tensor_map_sparse;
        CUtensorMap tensor_map_output;
        int num_experts;
        int max_m;
        int n;
        int k;
        int block_n;
        int block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_masked_output64x64_nm12_fixed_stage2_masked_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_grouped_masked_output64x64_nm12_fixed_stage2_masked_epilogue<
            {}, {}, {}, {}, {}, {}, {}>);
    (void)ptr;
}}
)",
            args.block_n, args.block_m,
            2, args.num_experts, args.max_m, args.n, args.k);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata,
            args.grouped_index,
            args.tensor_map_activation, args.tensor_map_dense,
            args.tensor_map_sparse, args.tensor_map_output,
            args.num_experts, args.max_m, args.n, args.k,
            args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_masked_output64x64_nm12_fixed_stage2_masked_epilogue(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& grouped_index, const torch::Tensor& d,
        const int num_experts, const int max_m, const int n, const int k,
        const int block_n, const int block_m) {
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    constexpr int output_bytes = 64 * 64 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    constexpr int num_stages = 2;
    const int stage_bytes =
        dense_count * 64 * 64 * sizeof(__nv_bfloat16) +
        block_n * 64 * 32 * sizeof(__nv_bfloat16) +
        64 * 64 * block_m * sizeof(__nv_bfloat16) +
        block_n * 2 * 4 * 16 * sizeof(std::uint32_t);
    const int pipeline_bytes =
        num_stages * stage_bytes +
        2 * num_stages * sizeof(std::uint64_t);
    const int output_offset =
        ((pipeline_bytes + 1023) / 1024) * 1024;
    const int smem_bytes = output_offset + output_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, num_experts * max_m, 64 * block_m, 64, k, 128);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 64,
        num_experts * block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32,
        num_experts * block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, num_experts * max_m, n, 64, 64, n, 1, 128);
    const int total_tiles =
        num_experts * (max_m / 64) * ((n + 63) / 64);
    const auto args = SM90HybridSparseGroupedMaskedOutput64x64NM12FixedStage2MaskedEpilogueRuntime::Args {
        .block_selector = block_selector.data_ptr(),
        .hardware_metadata = hardware_metadata.data_ptr(),
        .grouped_index = grouped_index.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .tensor_map_output = tensor_map_output,
        .num_experts = num_experts, .max_m = max_m,
        .n = n, .k = k,
        .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(total_tiles, 256, smem_bytes),
    };
    const auto runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_masked_output64x64_nm12_fixed_stage2_masked_epilogue",
        SM90HybridSparseGroupedMaskedOutput64x64NM12FixedStage2MaskedEpilogueRuntime::generate(args));
    SM90HybridSparseGroupedMaskedOutput64x64NM12FixedStage2MaskedEpilogueRuntime::launch(runtime, args);
}

} // namespace deep_gemm
