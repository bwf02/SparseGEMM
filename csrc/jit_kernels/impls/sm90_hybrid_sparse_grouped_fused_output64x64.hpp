#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64.hpp"

namespace deep_gemm {

class SM90HybridSparseGroupedFusedOutput64x64Runtime final:
        public LaunchRuntime<SM90HybridSparseGroupedFusedOutput64x64Runtime> {
public:
    struct Args {
        void* block_selector;
        void* hardware_metadata;
        void* grouped_index;
        cute::TmaDescriptor tensor_map_activation;
        cute::TmaDescriptor tensor_map_dense;
        cute::TmaDescriptor tensor_map_sparse;
        cute::TmaDescriptor tensor_map_output;
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
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_fused_output64x64.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_grouped_fused_output64x64<{}, {}, {}, {}>);
    (void)ptr;
}}
)",
            args.block_n, args.block_m,
            get_group_stage_output64x64_num_stages(
                args.block_n, args.block_m),
            args.grouped_mode);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata,
            args.grouped_index, args.tensor_map_activation,
            args.tensor_map_dense, args.tensor_map_sparse,
            args.tensor_map_output, args.total_rows, args.n, args.k,
            args.num_experts, args.max_m, args.m_alignment,
            args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_fused_output64x64(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& grouped_index, const torch::Tensor& d,
        const int total_rows, const int n, const int k,
        const int num_experts, const int max_m, const int m_alignment,
        const int block_n, const int block_m, const int grouped_mode) {
    constexpr int output_bytes = 64 * 64 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const int num_stages =
        get_group_stage_output64x64_num_stages(block_n, block_m);
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
        a, k, total_rows, 64 * block_m, 64, k, 128);
    const auto tensor_map_dense = dense_count > 0
        ? make_tma_2d_desc(
              dense_values, 64,
              num_experts * block_rows * block_groups * dense_count * 64,
              64, 64, 64, 128)
        : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32,
        num_experts * block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, total_rows, n, 64, 64, n, 1, 128);

    const int tiles_m = grouped_mode == 0
        ? (total_rows + 63) / 64
        : num_experts * ((max_m + 63) / 64);
    const int total_tiles = tiles_m * ((n + 63) / 64);
    const int ctas_per_sm = std::max(1, 232448 / smem_bytes);
    const int persistent_ctas = std::min(
        total_tiles, device_runtime->get_num_sms() * ctas_per_sm);
    auto args = SM90HybridSparseGroupedFusedOutput64x64Runtime::Args {
        .block_selector = block_selector.data_ptr(),
        .hardware_metadata = hardware_metadata.data_ptr(),
        .grouped_index = grouped_index.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .tensor_map_output = tensor_map_output,
        .total_rows = total_rows,
        .n = n,
        .k = k,
        .num_experts = num_experts,
        .max_m = max_m,
        .m_alignment = m_alignment,
        .block_n = block_n,
        .block_m = block_m,
        .grouped_mode = grouped_mode,
        .launch_args = LaunchArgs(persistent_ctas, 256, smem_bytes),
    };
    const auto kernel_name = grouped_mode == 0
        ? "sm90_hybrid_sparse_grouped_contiguous_fused_output64x64"
        : "sm90_hybrid_sparse_grouped_masked_fused_output64x64";
    const auto runtime = compiler->build(
        kernel_name,
        SM90HybridSparseGroupedFusedOutput64x64Runtime::generate(args));
    SM90HybridSparseGroupedFusedOutput64x64Runtime::launch(runtime, args);
}

} // namespace deep_gemm
