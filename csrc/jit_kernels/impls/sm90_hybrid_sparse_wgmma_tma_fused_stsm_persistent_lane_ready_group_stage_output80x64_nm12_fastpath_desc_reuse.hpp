#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

static int get_group_stage_output80x64_nm12_fastpath_desc_reuse_num_stages(
        int block_n, int block_m);

class SM90HybridSparseGroupStageOutput80x64NM12FastpathDescReuseRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupStageOutput80x64NM12FastpathDescReuseRuntime> {
public:
    using Args =
        SM90HybridSparseProducerMetadataCopyOutput128x64Runtime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_group_stage_output80x64_nm12_fastpath_desc_reuse<{}, {}, {}>);
    (void)ptr;
}}
)",
            args.block_n, args.block_m,
            get_group_stage_output80x64_nm12_fastpath_desc_reuse_num_stages(
                args.block_n, args.block_m));
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata, args.d,
            args.tensor_map_activation, args.tensor_map_dense,
            args.tensor_map_sparse, args.tensor_map_output,
            args.m, args.n, args.k, args.block_n, args.block_m));
    }
};

static int get_group_stage_output80x64_nm12_fastpath_desc_reuse_num_stages(
        const int block_n, const int block_m) {
    constexpr int max_smem_bytes = 232448;
    constexpr int output_bytes = 80 * 64 * sizeof(__nv_bfloat16);
    const int dense_count = block_m - block_n;
    const int stage_bytes =
        dense_count * 64 * 64 * sizeof(__nv_bfloat16) +
        block_n * 64 * 32 * sizeof(__nv_bfloat16) +
        80 * 64 * block_m * sizeof(__nv_bfloat16) +
        block_n * 2 * 4 * 16 * sizeof(std::uint32_t);
    for (int num_stages = 5; num_stages >= 2; --num_stages) {
        const int pipeline_bytes =
            num_stages * stage_bytes +
            2 * num_stages * sizeof(std::uint64_t);
        const int output_offset =
            ((pipeline_bytes + 1023) / 1024) * 1024;
        if (output_offset + output_bytes <= max_smem_bytes)
            return num_stages;
    }
    DG_HOST_ASSERT(false and "N:M group does not fit two TMA stages");
    return 0;
}

static void sm90_hybrid_block_sparse_bf16_gemm_group_stage_output80x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    constexpr int output_bytes = 80 * 64 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const int num_stages =
        get_group_stage_output80x64_nm12_fastpath_desc_reuse_num_stages(block_n, block_m);
    const int stage_bytes =
        dense_count * 64 * 64 * sizeof(__nv_bfloat16) +
        block_n * 64 * 32 * sizeof(__nv_bfloat16) +
        80 * 64 * block_m * sizeof(__nv_bfloat16) +
        block_n * 2 * 4 * 16 * sizeof(std::uint32_t);
    const int pipeline_bytes =
        num_stages * stage_bytes +
        2 * num_stages * sizeof(std::uint64_t);
    const int output_offset =
        ((pipeline_bytes + 1023) / 1024) * 1024;
    const int smem_bytes = output_offset + output_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, m, 64 * block_m, 80, k, 128);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 64, block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32, block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, m, n, 80, 64, n, 1, 128);
    const int total_tiles = ((m + 79) / 80) * ((n + 63) / 64);
    const int ctas_per_sm = std::max(1, 232448 / smem_bytes);
    const int persistent_ctas = std::min(
        total_tiles, device_runtime->get_num_sms() * ctas_per_sm);
    const auto args = SM90HybridSparseGroupStageOutput80x64NM12FastpathDescReuseRuntime::Args {
        .block_selector = block_selector.data_ptr(),
        .hardware_metadata = hardware_metadata.data_ptr(),
        .d = d.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .tensor_map_output = tensor_map_output,
        .m = m, .n = n, .k = k,
        .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(persistent_ctas, 256, smem_bytes),
    };
    const auto runtime = compiler->build(
        "sm90_hybrid_sparse_group_stage_output80x64_nm12_fastpath_desc_reuse",
        SM90HybridSparseGroupStageOutput80x64NM12FastpathDescReuseRuntime::generate(args));
    SM90HybridSparseGroupStageOutput80x64NM12FastpathDescReuseRuntime::launch(runtime, args);
}

} // namespace deep_gemm
