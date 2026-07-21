#pragma once

// Hybrid sparse 128x64 weight blocks with a 64x128 output tile.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

constexpr int kBlockH = 128;
constexpr int kBlockW = 64;
constexpr int kTileM = 64;
constexpr int kStages = 2;
constexpr int kMathThreads = 128;
constexpr int kThreads = 256;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
using DenseMMA = typename deep_gemm::mma::sm90::BF16MMASelector<64>::type;

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

__device__ __forceinline__ void advance_pipeline(
        int& stage, unsigned& phase) {
    stage = stage == kStages - 1 ? 0 : stage + 1;
    phase ^= stage == 0;
}

__device__ __forceinline__ void release_stage(Barrier* empty_barrier) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0)
        empty_barrier->arrive();
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


template <int = 0>
__global__ __launch_bounds__(kThreads, 1)
void hybrid_sparse_dense_wgmma_tma_block128x64(
        const long long* block_selector, const unsigned char*,
        float* dense_partial, float*, __nv_bfloat16*,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int kWeightBytes = kBlockH * kBlockW * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes = kTileM * kBlockW * sizeof(__nv_bfloat16);
    constexpr int kStageBytes = kWeightBytes + kActivationBytes;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kTileM;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlockH;
    const int block_row = output_tile_n / kBlockH;
    const int block_groups = k / (kBlockW * block_m);
    const int dense_count = block_m - block_n;

    extern __shared__ __align__(1024) unsigned char smem[];
    auto stage_base = [&](const int stage) { return smem + stage * kStageBytes; };
    auto smem_weight = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage));
    };
    auto smem_activation = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage) + kWeightBytes);
    };
    auto full_barrier = reinterpret_cast<Barrier*>(smem + kStages * kStageBytes);
    auto empty_barrier = full_barrier + kStages;

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(4);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (warp >= 4) {
        if (warp == 6 && cute::elect_one_sync()) {
            cute::prefetch_tma_descriptor(&tensor_map_activation);
            cute::prefetch_tma_descriptor(&tensor_map_dense);
            int stage = 0;
            unsigned phase = 0;
            for (int block_group = 0; block_group < block_groups; ++block_group) {
                const unsigned long long selector = static_cast<unsigned long long>(
                    block_selector[block_row * block_groups + block_group]);
                int dense_slot = 0;
                for (int local_block = 0; local_block < block_m; ++local_block) {
                    if ((selector >> local_block) & 1ULL)
                        continue;
                    empty_barrier[stage].wait(phase ^ 1);
                    const int block_k =
                        (block_group * block_m + local_block) * kBlockW;
                    const int packed_block =
                        (block_row * block_groups + block_group) * dense_count +
                        dense_slot;
                    deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                        &tensor_map_dense, &full_barrier[stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(smem_weight(stage)),
                        0, packed_block * kBlockH);
                    deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                        &tensor_map_activation, &full_barrier[stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(smem_activation(stage)),
                        block_k, output_tile_m);
                    full_barrier[stage].arrive_and_expect_tx(
                        kWeightBytes + kActivationBytes);
                    ++dense_slot;
                    advance_pipeline(stage, phase);
                }
            }
        }
        return;
    }

    float accumulator[64] = {};
    bool has_accumulator = false;
    int stage = 0;
    unsigned phase = 0;
    for (int block_group = 0; block_group < block_groups; ++block_group) {
        const unsigned long long selector = static_cast<unsigned long long>(
            block_selector[block_row * block_groups + block_group]);
        for (int local_block = 0; local_block < block_m; ++local_block) {
            if ((selector >> local_block) & 1ULL)
                continue;
            full_barrier[stage].wait(phase);
#pragma unroll
            for (int i = 0; i < 64; ++i)
                deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
#pragma unroll
                for (int k_tile = 0; k_tile < 4; ++k_tile) {
                    const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                        smem_weight(stage) + n_tile * 64 * kBlockW + k_tile * 16,
                        static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                    const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                        smem_activation(stage) + k_tile * 16,
                        static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                    DenseMMA::wgmma(desc_a.desc_, desc_b.desc_,
                                    accumulator + n_tile * 32,
                                    has_accumulator || k_tile != 0);
                }
            }
            deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < 64; ++i)
                deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
            deep_gemm::ptx::warpgroup_wait<0>();
            release_stage(&empty_barrier[stage]);
            has_accumulator = true;
            advance_pipeline(stage, phase);
        }
    }
    store_wgmma_accumulator(accumulator, dense_partial, output_tile_m,
                            output_tile_n, m, n);
    store_wgmma_accumulator(accumulator + 32, dense_partial, output_tile_m,
                            output_tile_n + 64, m, n);
}


template <int = 0>
__global__ __launch_bounds__(kThreads, 1)
void hybrid_sparse_2_4_wgmma_tma_block128x64(
        const long long* block_selector, const unsigned char* sparse_metadata,
        float*, float* sparse_partial, __nv_bfloat16*,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int kWeightBytes = kBlockH * (kBlockW / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes = kTileM * kBlockW * sizeof(__nv_bfloat16);
    constexpr int kStageBytes = kWeightBytes + kActivationBytes;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kTileM;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlockH;
    const int block_row = output_tile_n / kBlockH;
    const int block_groups = k / (kBlockW * block_m);

    extern __shared__ __align__(1024) unsigned char smem[];
    auto stage_base = [&](const int stage) { return smem + stage * kStageBytes; };
    auto smem_weight = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage));
    };
    auto smem_activation = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage) + kWeightBytes);
    };
    auto full_barrier = reinterpret_cast<Barrier*>(smem + kStages * kStageBytes);
    auto empty_barrier = full_barrier + kStages;

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(4);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (warp >= 4) {
        if (warp == 6 && cute::elect_one_sync()) {
            cute::prefetch_tma_descriptor(&tensor_map_activation);
            cute::prefetch_tma_descriptor(&tensor_map_sparse);
            int stage = 0;
            unsigned phase = 0;
            for (int block_group = 0; block_group < block_groups; ++block_group) {
                const unsigned long long selector = static_cast<unsigned long long>(
                    block_selector[block_row * block_groups + block_group]);
                int sparse_slot = 0;
                for (int local_block = 0; local_block < block_m; ++local_block) {
                    if (((selector >> local_block) & 1ULL) == 0)
                        continue;
                    empty_barrier[stage].wait(phase ^ 1);
                    const int block_k =
                        (block_group * block_m + local_block) * kBlockW;
                    const int packed_block =
                        (block_row * block_groups + block_group) * block_n +
                        sparse_slot;
                    deep_gemm::tma::copy<32, 128, 64, cutlass::bfloat16_t>(
                        &tensor_map_sparse, &full_barrier[stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(smem_weight(stage)),
                        0, packed_block * kBlockH);
                    deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                        &tensor_map_activation, &full_barrier[stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(smem_activation(stage)),
                        block_k, output_tile_m);
                    full_barrier[stage].arrive_and_expect_tx(
                        kWeightBytes + kActivationBytes);
                    ++sparse_slot;
                    advance_pipeline(stage, phase);
                }
            }
        }
        return;
    }

    float accumulator[64] = {};
    bool has_accumulator = false;
    int stage = 0;
    unsigned phase = 0;
    for (int block_group = 0; block_group < block_groups; ++block_group) {
        const unsigned long long selector = static_cast<unsigned long long>(
            block_selector[block_row * block_groups + block_group]);
        int sparse_slot = 0;
        for (int local_block = 0; local_block < block_m; ++local_block) {
            if (((selector >> local_block) & 1ULL) == 0)
                continue;
            const long long metadata_base =
                (((static_cast<long long>(block_row) * block_groups + block_group)
                   * block_n + sparse_slot) * kBlockH) * (kBlockW / 4);
            full_barrier[stage].wait(phase);
#pragma unroll
            for (int i = 0; i < 64; ++i)
                deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (int n_tile = 0; n_tile < 2; ++n_tile) {
#pragma unroll
                for (int k_tile = 0; k_tile < 2; ++k_tile) {
                    unsigned hardware_metadata = 0;
                    if (thread_in_group < 2) {
                        const int metadata_row = n_tile * 64 + warp * 16 + group;
                        const int quartet_base =
                            k_tile * 8 + thread_in_group * 4;
#pragma unroll
                        for (int quartet = 0; quartet < 4; ++quartet) {
                            const unsigned char lower_code = sparse_metadata[
                                metadata_base + static_cast<long long>(metadata_row) *
                                (kBlockW / 4) + quartet_base + quartet];
                            const unsigned char upper_code = sparse_metadata[
                                metadata_base + static_cast<long long>(metadata_row + 8) *
                                (kBlockW / 4) + quartet_base + quartet];
                            hardware_metadata |=
                                metadata_nibble(lower_code) << (quartet * 4);
                            hardware_metadata |=
                                metadata_nibble(upper_code) << ((quartet + 4) * 4);
                        }
                    }
                    const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                        smem_weight(stage) +
                            n_tile * 64 * (kBlockW / 2) + k_tile * 16,
                        static_cast<int>(cute::GMMA::LayoutType::B64), 0, 512);
                    const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                        smem_activation(stage) + k_tile * 32,
                        static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                    sparse_wgmma(desc_a.desc_, desc_b.desc_,
                                 accumulator + n_tile * 32,
                                 hardware_metadata,
                                 has_accumulator || k_tile != 0);
                }
            }
            deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < 64; ++i)
                deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
            deep_gemm::ptx::warpgroup_wait<0>();
            release_stage(&empty_barrier[stage]);
            has_accumulator = true;
            ++sparse_slot;
            advance_pipeline(stage, phase);
        }
    }
    store_wgmma_accumulator(accumulator, sparse_partial, output_tile_m,
                            output_tile_n, m, n);
    store_wgmma_accumulator(accumulator + 32, sparse_partial, output_tile_m,
                            output_tile_n + 64, m, n);
}


template <int = 0>
__global__ void hybrid_sparse_reduce_wgmma_tma_block128x64(
        const long long*, const unsigned char*, const float* dense_partial,
        const float* sparse_partial, __nv_bfloat16* output,
        const cute::TmaDescriptor, const cute::TmaDescriptor,
        const cute::TmaDescriptor, const int m, const int n, const int,
        const int, const int) {
    const int tile_m = static_cast<int>(blockIdx.x) * kTileM;
    const int tile_n = static_cast<int>(blockIdx.y) * kBlockH;
    for (int linear = static_cast<int>(threadIdx.x); linear < kTileM * kBlockH;
         linear += static_cast<int>(blockDim.x)) {
        const int row_m = tile_m + linear / kBlockH;
        const int row_n = tile_n + linear % kBlockH;
        if (row_m < m && row_n < n) {
            const long long index = static_cast<long long>(row_m) * n + row_n;
            output[index] = __float2bfloat16_rn(
                dense_partial[index] + sparse_partial[index]);
        }
    }
}
