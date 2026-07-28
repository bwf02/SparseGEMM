#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

class SM90HybridSparseOutput32x64StageKindRuntime final:
        public LaunchRuntime<SM90HybridSparseOutput32x64StageKindRuntime> {
public:
    using Args =
        SM90HybridSparseProducerMetadataCopyOutput128x64Runtime::Args;

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind.cuh>

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&hybrid_sparse_output32x64_stage_kind<>);
    (void)ptr;
}
)";
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

static void sm90_hybrid_block_sparse_bf16_gemm_output32x64_stage_kind(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int barrier_bytes = 6 * sizeof(std::uint64_t);
    constexpr int stage_bytes =
        (64 * 64 + 32 * 64) * sizeof(__nv_bfloat16) +
        2 * 4 * 16 * sizeof(std::uint32_t);
    constexpr int pipeline_bytes = 3 * stage_bytes + barrier_bytes;
    constexpr int output_offset =
        ((pipeline_bytes + 1023) / 1024) * 1024;
    constexpr int output_bytes = 32 * 64 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const int smem_bytes = output_offset + output_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, m, 64, 32, k, 128);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 64, block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32, block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, m, n, 32, 64, n, 1, 128);
    const int total_tiles = ((m + 31) / 32) * ((n + 63) / 64);
    constexpr int ctas_per_sm = 3;
    const int persistent_ctas = std::min(
        total_tiles, device_runtime->get_num_sms() * ctas_per_sm);
    const auto args = SM90HybridSparseOutput32x64StageKindRuntime::Args {
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
        "sm90_hybrid_sparse_output32x64_stage_kind",
        SM90HybridSparseOutput32x64StageKindRuntime::generate(args));
    SM90HybridSparseOutput32x64StageKindRuntime::launch(runtime, args);
}

} // namespace deep_gemm
