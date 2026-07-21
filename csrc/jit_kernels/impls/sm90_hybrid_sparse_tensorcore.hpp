#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

enum class HybridSparseTensorCoreKernel { Dense, Sparse, Reduce };

class SM90HybridSparseTensorCoreRuntime final:
        public LaunchRuntime<SM90HybridSparseTensorCoreRuntime> {
public:
    struct Args {
        HybridSparseTensorCoreKernel kernel;
        void *a, *block_selector, *dense_values, *sparse_values, *sparse_metadata;
        void *dense_partial, *sparse_partial, *d;
        int m, n, k, block_n, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        static constexpr auto common = R"(
#include <cuda_bf16.h>
#include <cuda_runtime.h>
constexpr int kBlock = 64;

__device__ __forceinline__ unsigned pack_bf16(
        const __nv_bfloat16 low, const __nv_bfloat16 high) {
    return static_cast<unsigned>(__bfloat16_as_ushort(low)) |
           (static_cast<unsigned>(__bfloat16_as_ushort(high)) << 16);
}

__device__ __forceinline__ unsigned metadata_nibble(const unsigned char code) {
    constexpr unsigned table = (0x4u << 0) | (0x8u << 4) | (0xcu << 8) |
                               (0x9u << 12) | (0xdu << 16) | (0xeu << 20);
    return (table >> (static_cast<unsigned>(code) * 4)) & 0xfu;
}

__device__ __forceinline__ void mma_dense_bf16(
        float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_sparse_bf16(
        float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[4],
        const unsigned metadata) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
        "{%0,%1,%2,%3}, %12, 0x0;\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(metadata));
}

__device__ __forceinline__ void store_accumulator(
        const float (&accumulator)[4], float* partial, const int tile_m,
        const int tile_n, const int m, const int n, const int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int element = 0; element < 4; ++element) {
        const int row_n = tile_n + group + (element >= 2 ? 8 : 0);
        const int row_m = tile_m + thread_in_group * 2 + (element & 1);
        if (row_m < m && row_n < n)
            partial[static_cast<long long>(row_m) * n + row_n] = accumulator[element];
    }
}
)";

        static constexpr auto dense = R"(
extern "C" __global__ void hybrid_sparse_dense_tensorcore(
        const __nv_bfloat16* activation, const long long* block_selector,
        const __nv_bfloat16* dense_values, const __nv_bfloat16*,
        const unsigned char*, float* dense_partial, float*, __nv_bfloat16*,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlock;

    for (int output_subtile = warp; output_subtile < 32; output_subtile += 8) {
        const int tile_n = output_tile_n + (output_subtile >> 3) * 16;
        const int tile_m = output_tile_m + (output_subtile & 7) * 8;
        const int group = lane >> 2;
        const int thread_in_group = lane & 3;
        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int block_row = tile_n / kBlock;
        const int row_base = tile_n % kBlock;

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
#pragma unroll
                for (int k_tile = 0; k_tile < 4; ++k_tile) {
                    unsigned weight_fragment[4];
#pragma unroll
                    for (int reg = 0; reg < 4; ++reg) {
                        const int row = row_base + group + ((reg & 1) ? 8 : 0);
                        const int column = k_tile * 16 + thread_in_group * 2 +
                                           ((reg >= 2) ? 8 : 0);
                        const long long offset = weight_base +
                            static_cast<long long>(row) * kBlock + column;
                        weight_fragment[reg] = pack_bf16(
                            dense_values[offset], dense_values[offset + 1]);
                    }
                    unsigned activation_fragment[2];
#pragma unroll
                    for (int reg = 0; reg < 2; ++reg) {
                        const int row_k = block_k + k_tile * 16 +
                                          thread_in_group * 2 + reg * 8;
                        const int column_m = tile_m + group;
                        const __nv_bfloat16 low = column_m < m ? activation[
                            static_cast<long long>(column_m) * k + row_k] :
                            __float2bfloat16_rn(0.0f);
                        const __nv_bfloat16 high = column_m < m ? activation[
                            static_cast<long long>(column_m) * k + row_k + 1] :
                            __float2bfloat16_rn(0.0f);
                        activation_fragment[reg] = pack_bf16(low, high);
                    }
                    mma_dense_bf16(accumulator, weight_fragment, activation_fragment);
                }
                ++dense_slot;
            }
        }
        store_accumulator(accumulator, dense_partial, tile_m, tile_n, m, n, lane);
    }
}
)";

        static constexpr auto sparse = R"(
extern "C" __global__ void hybrid_sparse_2_4_tensorcore(
        const __nv_bfloat16* activation, const long long* block_selector,
        const __nv_bfloat16*, const __nv_bfloat16* sparse_values,
        const unsigned char* sparse_metadata, float*, float* sparse_partial,
        __nv_bfloat16*, const int m, const int n, const int k,
        const int block_n, const int block_m) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block_groups = k / (kBlock * block_m);
    const int output_tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlock;

    for (int output_subtile = warp; output_subtile < 32; output_subtile += 8) {
        const int tile_n = output_tile_n + (output_subtile >> 3) * 16;
        const int tile_m = output_tile_m + (output_subtile & 7) * 8;
        const int group = lane >> 2;
        const int thread_in_group = lane & 3;
        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int block_row = tile_n / kBlock;
        const int row_base = tile_n % kBlock;

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
#pragma unroll
                for (int k_tile = 0; k_tile < 2; ++k_tile) {
                    unsigned weight_fragment[4];
#pragma unroll
                    for (int reg = 0; reg < 4; ++reg) {
                        const int row = row_base + group + ((reg & 1) ? 8 : 0);
                        const int column = k_tile * 16 + thread_in_group * 2 +
                                           ((reg >= 2) ? 8 : 0);
                        const long long offset = values_base +
                            static_cast<long long>(row) * (kBlock / 2) + column;
                        weight_fragment[reg] = pack_bf16(
                            sparse_values[offset], sparse_values[offset + 1]);
                    }
                    unsigned activation_fragment[4];
#pragma unroll
                    for (int reg = 0; reg < 4; ++reg) {
                        const int row_k = block_k + k_tile * 32 +
                                          thread_in_group * 2 + reg * 8;
                        const int column_m = tile_m + group;
                        const __nv_bfloat16 low = column_m < m ? activation[
                            static_cast<long long>(column_m) * k + row_k] :
                            __float2bfloat16_rn(0.0f);
                        const __nv_bfloat16 high = column_m < m ? activation[
                            static_cast<long long>(column_m) * k + row_k + 1] :
                            __float2bfloat16_rn(0.0f);
                        activation_fragment[reg] = pack_bf16(low, high);
                    }
                    unsigned hardware_metadata = 0;
                    if (thread_in_group < 2) {
                        const int metadata_row = row_base + group +
                                                 (thread_in_group ? 8 : 0);
#pragma unroll
                        for (int quartet = 0; quartet < 8; ++quartet) {
                            const unsigned char code = sparse_metadata[
                                metadata_base + static_cast<long long>(metadata_row) *
                                (kBlock / 4) + k_tile * 8 + quartet];
                            hardware_metadata |= metadata_nibble(code) << (quartet * 4);
                        }
                    }
                    mma_sparse_bf16(accumulator, weight_fragment,
                                    activation_fragment, hardware_metadata);
                }
                ++sparse_slot;
            }
        }
        store_accumulator(accumulator, sparse_partial, tile_m, tile_n, m, n, lane);
    }
}
)";

        static constexpr auto reduce = R"(
extern "C" __global__ void hybrid_sparse_reduce_tensorcore(
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
            case HybridSparseTensorCoreKernel::Dense:
                return std::string(common) + dense;
            case HybridSparseTensorCoreKernel::Sparse:
                return std::string(common) + sparse;
            case HybridSparseTensorCoreKernel::Reduce:
                return std::string(common) + reduce;
        }
        DG_HOST_UNREACHABLE("Unknown hybrid sparse Tensor Core kernel");
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

static void sm90_hybrid_block_sparse_bf16_gemm_tensorcore(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& dense_partial,
        const torch::Tensor& sparse_partial, const torch::Tensor& d,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    const auto grid = std::make_pair((m + 63) / 64, (n + 63) / 64);
    auto args = SM90HybridSparseTensorCoreRuntime::Args {
        .kernel = HybridSparseTensorCoreKernel::Dense,
        .a = a.data_ptr(), .block_selector = block_selector.data_ptr(),
        .dense_values = dense_values.data_ptr(),
        .sparse_values = sparse_values.data_ptr(),
        .sparse_metadata = sparse_metadata.data_ptr(),
        .dense_partial = dense_partial.data_ptr(),
        .sparse_partial = sparse_partial.data_ptr(), .d = d.data_ptr(),
        .m = m, .n = n, .k = k, .block_n = block_n, .block_m = block_m,
        .launch_args = LaunchArgs(grid, 256),
    };
    const auto dense_runtime = compiler->build(
        "sm90_hybrid_sparse_dense_tensorcore",
        SM90HybridSparseTensorCoreRuntime::generate(args));
    SM90HybridSparseTensorCoreRuntime::launch(dense_runtime, args);
    args.kernel = HybridSparseTensorCoreKernel::Sparse;
    const auto sparse_runtime = compiler->build(
        "sm90_hybrid_sparse_2_4_tensorcore",
        SM90HybridSparseTensorCoreRuntime::generate(args));
    SM90HybridSparseTensorCoreRuntime::launch(sparse_runtime, args);
    args.kernel = HybridSparseTensorCoreKernel::Reduce;
    const auto reduce_runtime = compiler->build(
        "sm90_hybrid_sparse_reduce_tensorcore",
        SM90HybridSparseTensorCoreRuntime::generate(args));
    SM90HybridSparseTensorCoreRuntime::launch(reduce_runtime, args);
}

} // namespace deep_gemm
