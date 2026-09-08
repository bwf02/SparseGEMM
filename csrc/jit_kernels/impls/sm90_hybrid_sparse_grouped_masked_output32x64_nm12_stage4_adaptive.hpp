#pragma once

#include "sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"

namespace deep_gemm {

class SM90HybridSparseGroupedMaskedOutput32x64NM12Stage4AdaptiveRuntime final:
        public LaunchRuntime<SM90HybridSparseGroupedMaskedOutput32x64NM12Stage4AdaptiveRuntime> {
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
        int num_workers;
        bool use_active_expert_prebind;
        bool use_bitmask_selector_fast_path;
        void* scheduler_trace;
        int scheduler_trace_max_tasks;
        bool enable_scheduler_trace;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_masked_output32x64_nm12_stage4_adaptive.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_sparse_grouped_masked_output32x64_nm12_stage4_adaptive<
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
    (void)ptr;
}}
)",
            args.block_n, args.block_m,
            4, args.num_experts, args.max_m, args.n, args.k, args.num_workers,
            args.use_active_expert_prebind ? "true" : "false",
            args.use_bitmask_selector_fast_path ? "true" : "false",
            args.enable_scheduler_trace ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.hardware_metadata,
            args.grouped_index,
            args.tensor_map_activation, args.tensor_map_dense,
            args.tensor_map_sparse, args.tensor_map_output,
            args.num_experts, args.max_m, args.n, args.k,
            args.block_n, args.block_m, args.scheduler_trace,
            args.scheduler_trace_max_tasks));
    }
};

static void sm90_hybrid_block_sparse_bf16_grouped_masked_output32x64_nm12_stage4_adaptive(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& grouped_index, const torch::Tensor& d,
        const int num_experts, const int max_m, const int n, const int k,
        const int block_n, const int block_m,
        const bool use_active_expert_prebind,
        const bool use_bitmask_selector_fast_path,
        const std::optional<torch::Tensor>& scheduler_trace) {
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(max_m == 64);
    constexpr int output_bytes = 32 * 64 * sizeof(__nv_bfloat16);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    constexpr int num_stages = 4;
    const int stage_bytes =
        dense_count * 64 * 64 * sizeof(__nv_bfloat16) +
        block_n * 64 * 32 * sizeof(__nv_bfloat16) +
        32 * 64 * block_m * sizeof(__nv_bfloat16) +
        block_n * 2 * 4 * 16 * sizeof(std::uint32_t);
    const int pipeline_bytes =
        num_stages * stage_bytes +
        2 * num_stages * sizeof(std::uint64_t);
    const int output_offset =
        ((pipeline_bytes + 1023) / 1024) * 1024;
    const int smem_bytes = output_offset + output_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, num_experts * max_m, 64 * block_m, 32, k, 128);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 64,
        num_experts * block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32,
        num_experts * block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto tensor_map_output = make_tma_cd_desc(
        d, num_experts * max_m, n, 32, 64, n, 1, 128);
    const int total_tiles =
        num_experts * 2 * ((n + 63) / 64);
    const int num_sms = device_runtime->get_num_sms();
    const int num_workers = 2 * num_sms;
    const bool enable_prebind =
        use_active_expert_prebind && num_experts <= num_workers;
    const bool enable_scheduler_trace = scheduler_trace.has_value();
    int scheduler_trace_max_tasks = 0;
    void* scheduler_trace_ptr = nullptr;
    if (enable_scheduler_trace) {
        const auto& trace = scheduler_trace.value();
        DG_HOST_ASSERT(trace.is_cuda() && trace.is_contiguous());
        DG_HOST_ASSERT(trace.scalar_type() == torch::kInt64 && trace.dim() == 4);
        DG_HOST_ASSERT(trace.size(0) >= num_workers && trace.size(1) > 0);
        DG_HOST_ASSERT(trace.size(2) == 7 && trace.size(3) == 9);
        scheduler_trace_max_tasks = static_cast<int>(trace.size(1));
        scheduler_trace_ptr = trace.data_ptr();
    }
    const auto args = SM90HybridSparseGroupedMaskedOutput32x64NM12Stage4AdaptiveRuntime::Args {
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
        .num_workers = num_workers,
        .use_active_expert_prebind = enable_prebind,
        .use_bitmask_selector_fast_path = use_bitmask_selector_fast_path,
        .scheduler_trace = scheduler_trace_ptr,
        .scheduler_trace_max_tasks = scheduler_trace_max_tasks,
        .enable_scheduler_trace = enable_scheduler_trace,
        .launch_args = LaunchArgs(
            enable_prebind ? num_workers : std::min(total_tiles, num_workers),
            256, smem_bytes),
    };
    const auto runtime = compiler->build(
        fmt::format(
            "sm90_hybrid_sparse_grouped_masked_output32x64_nm12_stage4_adaptive_{}_{}{}",
            enable_prebind ? "prebind" : "global_persistent",
            use_bitmask_selector_fast_path ? "bitmask" : "per_block",
            enable_scheduler_trace ? "_trace" : ""),
        SM90HybridSparseGroupedMaskedOutput32x64NM12Stage4AdaptiveRuntime::generate(args));
    SM90HybridSparseGroupedMaskedOutput32x64NM12Stage4AdaptiveRuntime::launch(runtime, args);
}

} // namespace deep_gemm
