#pragma once

#include <cstdint>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

enum class HybridSparseWgmmaTmaBlock128x32Kernel { Dense, Sparse, Reduce };

class SM90HybridSparseWgmmaTmaBlock128x32Runtime final:
        public LaunchRuntime<SM90HybridSparseWgmmaTmaBlock128x32Runtime> {
public:
    struct Args {
        HybridSparseWgmmaTmaBlock128x32Kernel kernel;
        void *block_selector, *sparse_metadata;
        void *dense_partial, *sparse_partial, *d;
        CUtensorMap tensor_map_activation;
        CUtensorMap tensor_map_dense;
        CUtensorMap tensor_map_sparse;
        int m, n, k, block_n, block_m;
        LaunchArgs launch_args;
    };

    static const char* kernel_symbol(
            const HybridSparseWgmmaTmaBlock128x32Kernel kernel) {
        switch (kernel) {
            case HybridSparseWgmmaTmaBlock128x32Kernel::Dense:
                return "hybrid_sparse_dense_wgmma_tma_block128x32";
            case HybridSparseWgmmaTmaBlock128x32Kernel::Sparse:
                return "hybrid_sparse_2_4_wgmma_tma_block128x32";
            case HybridSparseWgmmaTmaBlock128x32Kernel::Reduce:
                return "hybrid_sparse_reduce_wgmma_tma_block128x32";
        }
        DG_HOST_UNREACHABLE("Unknown hybrid sparse WGMMA TMA kernel");
    }

    static std::string generate_impl(const Args& args) {
        const char* symbol = kernel_symbol(args.kernel);
        return std::string(R"(
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_block128x32.cuh>

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&)") + symbol + R"();
    (void)ptr;
}
)";
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.block_selector, args.sparse_metadata,
            args.dense_partial, args.sparse_partial, args.d,
            args.tensor_map_activation, args.tensor_map_dense,
            args.tensor_map_sparse, args.m, args.n, args.k,
            args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& dense_partial,
        const torch::Tensor& sparse_partial, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int barrier_bytes = 4 * sizeof(std::uint64_t);
    constexpr int dense_smem_bytes =
        2 * (128 * 32 + 64 * 32) * sizeof(__nv_bfloat16) + barrier_bytes;
    constexpr int sparse_smem_bytes =
        2 * (128 * 16 + 64 * 32) * sizeof(__nv_bfloat16) + barrier_bytes;
    const int block_rows = n / 128;
    const int block_groups = k / (32 * block_m);
    const int dense_count = block_m - block_n;
    const auto tensor_map_activation = make_tma_2d_desc(
        a, k, m, 32, 64, k, 64);
    const auto tensor_map_dense = dense_count > 0 ? make_tma_2d_desc(
        dense_values, 32, block_rows * block_groups * dense_count * 128,
        32, 128, 32, 64) : tensor_map_activation;
    const auto tensor_map_sparse = make_tma_2d_desc(
        sparse_values, 16, block_rows * block_groups * block_n * 128,
        16, 128, 16, 32);
    const auto grid = std::make_pair((m + 63) / 64, (n + 127) / 128);
    auto args = SM90HybridSparseWgmmaTmaBlock128x32Runtime::Args {
        .kernel = HybridSparseWgmmaTmaBlock128x32Kernel::Dense,
        .block_selector = block_selector.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .dense_partial = dense_partial.data_ptr(),
        .sparse_partial = sparse_partial.data_ptr(), .d = d.data_ptr(),
        .tensor_map_activation = tensor_map_activation,
        .tensor_map_dense = tensor_map_dense,
        .tensor_map_sparse = tensor_map_sparse,
        .m = m, .n = n, .k = k, .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(grid, 256, dense_smem_bytes),
    };
    const auto dense_runtime = compiler->build(
        "sm90_hybrid_sparse_dense_wgmma_tma_block128x32",
        SM90HybridSparseWgmmaTmaBlock128x32Runtime::generate(args));
    SM90HybridSparseWgmmaTmaBlock128x32Runtime::launch(dense_runtime, args);
    args.kernel = HybridSparseWgmmaTmaBlock128x32Kernel::Sparse;
    args.launch_args = LaunchArgs(grid, 256, sparse_smem_bytes);
    const auto sparse_runtime = compiler->build(
        "sm90_hybrid_sparse_2_4_wgmma_tma_block128x32",
        SM90HybridSparseWgmmaTmaBlock128x32Runtime::generate(args));
    SM90HybridSparseWgmmaTmaBlock128x32Runtime::launch(sparse_runtime, args);
    args.kernel = HybridSparseWgmmaTmaBlock128x32Kernel::Reduce;
    args.launch_args = LaunchArgs(grid, 256);
    const auto reduce_runtime = compiler->build(
        "sm90_hybrid_sparse_reduce_wgmma_tma_block128x32",
        SM90HybridSparseWgmmaTmaBlock128x32Runtime::generate(args));
    SM90HybridSparseWgmmaTmaBlock128x32Runtime::launch(reduce_runtime, args);
}

} // namespace deep_gemm
