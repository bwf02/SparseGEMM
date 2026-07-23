#pragma once

#include <cstdint>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90HybridSparseFusedDirectRuntime final:
        public LaunchRuntime<SM90HybridSparseFusedDirectRuntime> {
public:
    struct Args {
        void *block_selector, *sparse_metadata, *d;
        CUtensorMap tensor_map_activation;
        CUtensorMap tensor_map_dense;
        CUtensorMap tensor_map_sparse;
        int m, n, k, block_n, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_direct.cuh>

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&hybrid_sparse_fused_wgmma_tma_direct<>);
    (void)ptr;
}
)";
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.sparse_metadata, args.d,
            args.tensor_map_activation, args.tensor_map_dense,
            args.tensor_map_sparse, args.m, args.n, args.k,
            args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int barrier_bytes = 4 * sizeof(std::uint64_t);
    constexpr int pipeline_bytes =
        2 * (64 * 64 + 64 * 64) * sizeof(__nv_bfloat16) + barrier_bytes;
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const int metadata_bytes = block_groups * block_n * 64 * 16;
    const int smem_bytes = pipeline_bytes + metadata_bytes;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, m, 64, 64, k, 128);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 64, block_rows * block_groups * dense_count * 64,
        64, 64, 64, 128) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 32, block_rows * block_groups * block_n * 64,
        32, 64, 32, 64);
    const auto grid = std::make_pair((m + 63) / 64, (n + 63) / 64);
    const auto args = SM90HybridSparseFusedDirectRuntime::Args {
        .block_selector = block_selector.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .d = d.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .m = m, .n = n, .k = k, .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(grid, 256, smem_bytes),
    };
    const auto runtime = compiler->build(
        "sm90_hybrid_sparse_fused_wgmma_tma_direct",
        SM90HybridSparseFusedDirectRuntime::generate(args));
    SM90HybridSparseFusedDirectRuntime::launch(runtime, args);
}

} // namespace deep_gemm
