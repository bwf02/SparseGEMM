#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

class SM90HybridSparseGroupedMaskedOutput128x64NM12Stage2PersistentOutputReuseWarpHandshakeWarpHandshakeRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupedMaskedOutput128x64NM12Stage2PersistentOutputReuseWarpHandshakeWarpHandshakeRuntime> {
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
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_masked_output128x64_nm12_stage2_persistent_output_reuse_warp_handshake.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_grouped_masked_output128x64_nm12_stage2_persistent_output_reuse_warp_handshake<
            {}, {}, {}, {}, {}, {}, {}>);
    (void)ptr;
}}
)",
            args.block_n, args.block_m, 2, args.num_experts,
            args.max_m, args.n, args.k);
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata,
            args.grouped_index, args.tensor_map_activation,
            args.tensor_map_dense, args.tensor_map_sparse,
            args.tensor_map_output, args.num_experts, args.max_m,
            args.n, args.k, args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_masked_output128x64_nm12_stage2_persistent_output_reuse_warp_handshake(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& grouped_index, const torch::Tensor& d,
        const int num_experts, const int max_m, const int n, const int k,
        const int block_n, const int block_m) {
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(max_m % 128 == 0);
    constexpr int num_stages = 2;
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    const int stage_bytes =
        64 * 64 * sizeof(__nv_bfloat16) +
        64 * 32 * sizeof(__nv_bfloat16) +
        128 * 128 * sizeof(__nv_bfloat16) +
        2 * 4 * 16 * sizeof(std::uint32_t);
    const int pipeline_bytes =
        num_stages * stage_bytes +
        2 * num_stages * sizeof(std::uint64_t);
    const int stage_control_bytes =
        num_stages * sizeof(std::uint64_t);
    const int smem_bytes = pipeline_bytes + stage_control_bytes;
    const int total_rows = num_experts * max_m;

    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, total_rows, 128, 128, k, 128);
    const auto tensor_map_dense = make_tma_2d_desc(
        dense_values, 64,
        num_experts * block_rows * block_groups * 64,
        64, 64, 64, 128);
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32,
        num_experts * block_rows * block_groups * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, total_rows, n, 128, 64, n, 1, 128);
    const int total_tiles =
        num_experts * (max_m / 128) * ((n + 63) / 64);
    const auto args =
        SM90HybridSparseGroupedMaskedOutput128x64NM12Stage2PersistentOutputReuseWarpHandshakeWarpHandshakeRuntime::Args {
            .block_selector = block_selector.data_ptr(),
            .hardware_metadata = hardware_metadata.data_ptr(),
            .grouped_index = grouped_index.data_ptr(),
            .tensor_map_activation = tensor_map_activation,
            .tensor_map_dense = tensor_map_dense,
            .tensor_map_sparse = tensor_map_sparse,
            .tensor_map_output = tensor_map_output,
            .num_experts = num_experts,
            .max_m = max_m,
            .n = n,
            .k = k,
            .block_n = block_n,
            .block_m = block_m,
            .launch_args = LaunchArgs(
                std::min(total_tiles, 2 * device_runtime->get_num_sms()),
                256, smem_bytes),
        };
    const auto runtime = compiler->build(
        "sm90_hybrid_sparse_grouped_masked_output128x64_nm12_stage2_persistent_output_reuse_warp_handshake",
        SM90HybridSparseGroupedMaskedOutput128x64NM12Stage2PersistentOutputReuseWarpHandshakeWarpHandshakeRuntime::generate(args));
    SM90HybridSparseGroupedMaskedOutput128x64NM12Stage2PersistentOutputReuseWarpHandshakeWarpHandshakeRuntime::launch(runtime, args);
}

} // namespace deep_gemm
