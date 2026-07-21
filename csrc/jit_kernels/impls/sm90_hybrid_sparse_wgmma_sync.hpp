#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

enum class HybridSparseWgmmaSyncKernel { Dense, Sparse, Reduce };

class SM90HybridSparseWgmmaSyncRuntime final:
        public LaunchRuntime<SM90HybridSparseWgmmaSyncRuntime> {
public:
    struct Args {
        HybridSparseWgmmaSyncKernel kernel;
        void *a, *block_selector, *dense_values, *sparse_values, *sparse_metadata;
        void *dense_partial, *sparse_partial, *d;
        int m, n, k, block_n, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        static constexpr auto common = R"(
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

constexpr int kBlock = 64;
constexpr int kThreads = 128;

using DenseMMA = typename deep_gemm::mma::sm90::BF16MMASelector<64>::type;
using DenseSmemLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_SW128_Atom<__nv_bfloat16>{},
    cute::make_shape(cute::_64{}, cute::_64{})));
using SparseSmemLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_SW64_Atom<__nv_bfloat16>{},
    cute::make_shape(cute::_64{}, cute::_32{})));

template <size_t... I>
__device__ __forceinline__ void sparse_wgmma_impl(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate,
        cute::index_sequence<I...>) {
    using SparseMMA = cute::SM90::GMMA::SPARSE::
        GMMA_64x64x32_F32BF16BF16_SS<cute::GMMA::Major::K,
                                     cute::GMMA::Major::K>;
    SparseMMA::fma(
        desc_a, desc_b, accumulator[I]..., metadata,
        accumulate ? cute::GMMA::ScaleOut::One : cute::GMMA::ScaleOut::Zero);
}

__device__ __forceinline__ void sparse_wgmma(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate) {
    sparse_wgmma_impl(desc_a, desc_b, accumulator, metadata, accumulate,
                      cute::make_index_sequence<32>{});
}

__device__ __forceinline__ unsigned metadata_nibble(const unsigned char code) {
    constexpr unsigned table = (0x4u << 0) | (0x8u << 4) | (0xcu << 8) |
                               (0x9u << 12) | (0xdu << 16) | (0xeu << 20);
    return (table >> (static_cast<unsigned>(code) * 4)) & 0xfu;
}

__device__ __forceinline__ void publish_smem_to_wgmma() {
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void finish_wgmma_batch() {
    deep_gemm::ptx::warpgroup_commit_batch();
    deep_gemm::ptx::warpgroup_wait<0>();
    __syncthreads();
}

__device__ __forceinline__ void store_wgmma_accumulator(
        const float* accumulator, float* partial, const int output_tile_m,
        const int output_tile_n, const int m, const int n) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row_n_0 = output_tile_n + warp * 16 + lane / 4;
    const int row_n_1 = row_n_0 + 8;
    const int column_base = output_tile_m + (lane & 3) * 2;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
        const int column_m = column_base + group * 8;
        if (row_n_0 < n) {
            if (column_m < m)
                partial[static_cast<long long>(column_m) * n + row_n_0] =
                    accumulator[group * 4];
            if (column_m + 1 < m)
                partial[static_cast<long long>(column_m + 1) * n + row_n_0] =
                    accumulator[group * 4 + 1];
        }
        if (row_n_1 < n) {
            if (column_m < m)
                partial[static_cast<long long>(column_m) * n + row_n_1] =
                    accumulator[group * 4 + 2];
            if (column_m + 1 < m)
                partial[static_cast<long long>(column_m + 1) * n + row_n_1] =
                    accumulator[group * 4 + 3];
        }
    }
}
)";

        static constexpr auto dense = R"(
extern "C" __global__ __launch_bounds__(kThreads)
void hybrid_sparse_dense_wgmma_sync(
        const __nv_bfloat16* activation, const long long* block_selector,
        const __nv_bfloat16* dense_values, const __nv_bfloat16*,
        const unsigned char*, float* dense_partial, float*, __nv_bfloat16*,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    __shared__ alignas(128) __nv_bfloat16 smem_weight[kBlock * kBlock];
    __shared__ alignas(128) __nv_bfloat16 smem_activation[kBlock * kBlock];
    const DenseSmemLayout smem_layout;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_row = output_tile_n / kBlock;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;
    float accumulator[32] = {};
    bool has_accumulator = false;

    for (int block_group = 0; block_group < block_groups; ++block_group) {
        const unsigned long long selector = static_cast<unsigned long long>(
            block_selector[block_row * block_groups + block_group]);
        int dense_slot = 0;
        for (int local_block = 0; local_block < block_m; ++local_block) {
            if ((selector >> local_block) & 1ULL)
                continue;
            const int block_k = (block_group * block_m + local_block) * kBlock;
            const long long weight_base =
                (((static_cast<long long>(block_row) * block_groups + block_group)
                   * dense_count + dense_slot) * kBlock) * kBlock;
            for (int linear = static_cast<int>(threadIdx.x);
                 linear < kBlock * kBlock; linear += kThreads) {
                const int row = linear / kBlock;
                const int column = linear % kBlock;
                const int smem_offset = smem_layout(row, column);
                smem_weight[smem_offset] = dense_values[weight_base + linear];
                const int row_m = output_tile_m + row;
                smem_activation[smem_offset] = row_m < m ? activation[
                    static_cast<long long>(row_m) * k + block_k + column] :
                    __float2bfloat16_rn(0.0f);
            }
            publish_smem_to_wgmma();
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (int k_tile = 0; k_tile < 4; ++k_tile) {
                const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                    smem_weight + k_tile * 16,
                    static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_activation + k_tile * 16,
                    static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                DenseMMA::wgmma(desc_a.desc_, desc_b.desc_, accumulator,
                                has_accumulator || k_tile != 0);
            }
            finish_wgmma_batch();
            has_accumulator = true;
            ++dense_slot;
        }
    }
    store_wgmma_accumulator(accumulator, dense_partial, output_tile_m,
                            output_tile_n, m, n);
}
)";

        static constexpr auto sparse = R"(
extern "C" __global__ __launch_bounds__(kThreads)
void hybrid_sparse_2_4_wgmma_sync(
        const __nv_bfloat16* activation, const long long* block_selector,
        const __nv_bfloat16*, const __nv_bfloat16* sparse_values,
        const unsigned char* sparse_metadata, float*, float* sparse_partial,
        __nv_bfloat16*, const int m, const int n, const int k,
        const int block_n, const int block_m) {
    __shared__ alignas(128) __nv_bfloat16 smem_weight[kBlock * (kBlock / 2)];
    __shared__ alignas(128) __nv_bfloat16 smem_activation[kBlock * kBlock];
    const SparseSmemLayout sparse_layout;
    const DenseSmemLayout activation_layout;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_row = output_tile_n / kBlock;
    const int block_groups = k / (kBlock * block_m);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    float accumulator[32] = {};
    bool has_accumulator = false;

    for (int block_group = 0; block_group < block_groups; ++block_group) {
        const unsigned long long selector = static_cast<unsigned long long>(
            block_selector[block_row * block_groups + block_group]);
        int sparse_slot = 0;
        for (int local_block = 0; local_block < block_m; ++local_block) {
            if (((selector >> local_block) & 1ULL) == 0)
                continue;
            const int block_k = (block_group * block_m + local_block) * kBlock;
            const long long stream_base =
                ((static_cast<long long>(block_row) * block_groups + block_group)
                  * block_n + sparse_slot) * kBlock;
            const long long values_base = stream_base * (kBlock / 2);
            const long long metadata_base = stream_base * (kBlock / 4);
            for (int linear = static_cast<int>(threadIdx.x);
                 linear < kBlock * (kBlock / 2); linear += kThreads) {
                const int row = linear / (kBlock / 2);
                const int column = linear % (kBlock / 2);
                smem_weight[sparse_layout(row, column)] =
                    sparse_values[values_base + linear];
            }
            for (int linear = static_cast<int>(threadIdx.x);
                 linear < kBlock * kBlock; linear += kThreads) {
                const int row = linear / kBlock;
                const int column = linear % kBlock;
                const int row_m = output_tile_m + row;
                smem_activation[activation_layout(row, column)] = row_m < m ?
                    activation[static_cast<long long>(row_m) * k + block_k + column] :
                    __float2bfloat16_rn(0.0f);
            }
            publish_smem_to_wgmma();
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                unsigned hardware_metadata = 0;
                if (thread_in_group < 2) {
                    const int metadata_row = warp * 16 + group;
                    const int quartet_base = k_tile * 8 + thread_in_group * 4;
#pragma unroll
                    for (int quartet = 0; quartet < 4; ++quartet) {
                        const unsigned char lower_code = sparse_metadata[
                            metadata_base + static_cast<long long>(metadata_row) *
                            (kBlock / 4) + quartet_base + quartet];
                        const unsigned char upper_code = sparse_metadata[
                            metadata_base + static_cast<long long>(metadata_row + 8) *
                            (kBlock / 4) + quartet_base + quartet];
                        hardware_metadata |= metadata_nibble(lower_code) << (quartet * 4);
                        hardware_metadata |= metadata_nibble(upper_code) << ((quartet + 4) * 4);
                    }
                }
                const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                    smem_weight + k_tile * 16,
                    static_cast<int>(cute::GMMA::LayoutType::B64), 0, 512);
                const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_activation + k_tile * 32,
                    static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                sparse_wgmma(desc_a.desc_, desc_b.desc_, accumulator,
                             hardware_metadata, has_accumulator || k_tile != 0);
            }
            finish_wgmma_batch();
            has_accumulator = true;
            ++sparse_slot;
        }
    }
    store_wgmma_accumulator(accumulator, sparse_partial, output_tile_m,
                            output_tile_n, m, n);
}
)";

        static constexpr auto reduce = R"(
extern "C" __global__ void hybrid_sparse_reduce_wgmma_sync(
        const __nv_bfloat16*, const long long*, const __nv_bfloat16*,
        const __nv_bfloat16*, const unsigned char*, const float* dense_partial,
        const float* sparse_partial, __nv_bfloat16* output,
        const int m, const int n, const int, const int, const int) {
    const int tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int tile_n = static_cast<int>(blockIdx.y) * kBlock;
    for (int linear = static_cast<int>(threadIdx.x); linear < kBlock * kBlock;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kBlock;
        const int row_n = tile_n + linear % kBlock;
        if (row_m < m && row_n < n) {
            const long long index = static_cast<long long>(row_m) * n + row_n;
            output[index] = __float2bfloat16_rn(
                dense_partial[index] + sparse_partial[index]);
        }
    }
}
)";

        switch (args.kernel) {
            case HybridSparseWgmmaSyncKernel::Dense:
                return std::string(common) + dense;
            case HybridSparseWgmmaSyncKernel::Sparse:
                return std::string(common) + sparse;
            case HybridSparseWgmmaSyncKernel::Reduce:
                return std::string(common) + reduce;
        }
        DG_HOST_UNREACHABLE("Unknown hybrid sparse synchronous WGMMA kernel");
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.a, args.block_selector, args.dense_values,
            args.sparse_values, args.sparse_metadata, args.dense_partial,
            args.sparse_partial, args.d, args.m, args.n, args.k,
            args.block_n, args.block_m));
    }
};

static void sm90_hybrid_block_sparse_bf16_gemm_wgmma_sync(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& dense_partial,
        const torch::Tensor& sparse_partial, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    const auto grid = std::make_pair((m + 63) / 64, (n + 63) / 64);
    auto args = SM90HybridSparseWgmmaSyncRuntime::Args {
        .kernel = HybridSparseWgmmaSyncKernel::Dense,
        .a = a.data_ptr(), .block_selector = block_selector.data_ptr(),
        .dense_values = dense_values.data_ptr(),
        .sparse_values = sparse_values.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .dense_partial = dense_partial.data_ptr(),
        .sparse_partial = sparse_partial.data_ptr(), .d = d.data_ptr(),
        .m = m, .n = n, .k = k, .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(grid, 128),
    };
    const auto dense_runtime = compiler->build(
        "sm90_hybrid_sparse_dense_wgmma_sync",
        SM90HybridSparseWgmmaSyncRuntime::generate(args));
    SM90HybridSparseWgmmaSyncRuntime::launch(dense_runtime, args);
    args.kernel = HybridSparseWgmmaSyncKernel::Sparse;
    const auto sparse_runtime = compiler->build(
        "sm90_hybrid_sparse_2_4_wgmma_sync",
        SM90HybridSparseWgmmaSyncRuntime::generate(args));
    SM90HybridSparseWgmmaSyncRuntime::launch(sparse_runtime, args);
    args.kernel = HybridSparseWgmmaSyncKernel::Reduce;
    args.launch_args = LaunchArgs(grid, 256);
    const auto reduce_runtime = compiler->build(
        "sm90_hybrid_sparse_reduce_wgmma_sync",
        SM90HybridSparseWgmmaSyncRuntime::generate(args));
    SM90HybridSparseWgmmaSyncRuntime::launch(reduce_runtime, args);
}

} // namespace deep_gemm
