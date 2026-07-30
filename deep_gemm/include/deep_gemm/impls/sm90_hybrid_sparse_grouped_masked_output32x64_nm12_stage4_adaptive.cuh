#pragma once

// Adaptive grouped masked M32xN64 kernel. Empty M32 tiles bypass the GEMM
// mainloop and only materialize the required zero output.

#ifndef CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
#endif
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

constexpr int kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive = 32;
constexpr int kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive = 64;
constexpr int kMathThreadsGroupedMaskedOutput32x64Stage4Adaptive = 128;
constexpr int kThreadsGroupedMaskedOutput32x64Stage4Adaptive = 256;
constexpr int kAccumulatorCountGroupedMaskedOutput32x64Stage4Adaptive = 16;

constexpr unsigned long long make_grouped_masked_output32x64_stage4_desc(
        const unsigned byte_offset, const unsigned stride_byte_offset,
        const unsigned layout_type) {
    return static_cast<unsigned long long>(byte_offset >> 4) |
           (static_cast<unsigned long long>(stride_byte_offset >> 4) << 32) |
           (static_cast<unsigned long long>(layout_type) << 62);
}

using DenseMMAGroupedMaskedOutput32x64Stage4Adaptive =
    typename deep_gemm::mma::sm90::BF16MMASelector<32>::type;

template <size_t... I>
__device__ __forceinline__ void sparse_wgmma_grouped_masked_output32x64_stage4_adaptive_impl(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate,
        cute::index_sequence<I...>) {
    using SparseMMA = cute::SM90::GMMA::SPARSE::
        GMMA_64x32x32_F32BF16BF16_SS<cute::GMMA::Major::K,
                                     cute::GMMA::Major::K>;
    SparseMMA::fma(
        desc_a, desc_b, accumulator[I]..., metadata,
        accumulate ? cute::GMMA::ScaleOut::One : cute::GMMA::ScaleOut::Zero);
}

__device__ __forceinline__ void sparse_wgmma_grouped_masked_output32x64_stage4_adaptive(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate) {
    sparse_wgmma_grouped_masked_output32x64_stage4_adaptive_impl(
        desc_a, desc_b, accumulator, metadata, accumulate,
        cute::make_index_sequence<kAccumulatorCountGroupedMaskedOutput32x64Stage4Adaptive>{});
}

template <int kPipelineStages>
__device__ __forceinline__ void advance_pipeline_grouped_masked_output32x64_stage4_adaptive(
        int& stage, unsigned& phase) {
    stage = stage == kPipelineStages - 1
        ? 0
        : stage + 1;
    phase ^= stage == 0;
}

template <int kBlockN, int kBlockM, int kPipelineStages,
          int kNumExperts, int kMaxM, int kN, int kK>
__global__ __launch_bounds__(kThreadsGroupedMaskedOutput32x64Stage4Adaptive, 1)
void hybrid_sparse_grouped_masked_output32x64_nm12_stage4_adaptive(
        const long long* block_selector, const unsigned* hardware_metadata,
        const int* grouped_index,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int num_experts, const int max_m, const int n, const int k,
        const int block_n, const int block_m) {
    static_assert(kBlockN == 1 && kBlockM == 2);
    static_assert(kPipelineStages == 4);
    static_assert(kNumExperts > 0 && kMaxM == 64);
    static_assert(kN % 64 == 0 && kK % 128 == 0);
    if (block_n != kBlockN || block_m != kBlockM ||
        num_experts != kNumExperts || max_m != kMaxM ||
        n != kN || k != kK)
        return;
    constexpr int kDenseCount = kBlockM - kBlockN;
    constexpr int kDenseWeightBytes =
        kDenseCount * kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseWeightBytes =
        kBlockN * kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes =
        kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive * kBlock * kBlockM *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes =
        kBlockN * 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kStageBytes =
        kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes =
        2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
        kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes +
          kBarrierBytes + 1023) / 1024) * 1024;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    constexpr int block_groups = kK / (kBlock * kBlockM);
    constexpr int tiles_per_expert = 2;
    constexpr int tiles_m = kNumExperts * tiles_per_expert;
    constexpr int tiles_n =
        (kN + kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive - 1) /
        kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive;
    constexpr int total_tiles = tiles_m * tiles_n;

    extern __shared__ __align__(1024) unsigned char smem[];
    auto stage_base = [&](const int stage) { return smem + stage * kStageBytes; };
    auto smem_dense = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage));
    };
    auto smem_sparse = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(
            stage_base(stage) + kDenseWeightBytes);
    };
    auto smem_activation = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(
            stage_base(stage) + kDenseWeightBytes + kSparseWeightBytes);
    };
    auto smem_metadata = [&](const int stage) {
        return reinterpret_cast<unsigned*>(
            stage_base(stage) + kDenseWeightBytes + kSparseWeightBytes +
            kActivationBytes);
    };
    auto full_barrier = reinterpret_cast<Barrier*>(
        smem + kPipelineStages * kStageBytes);
    auto empty_barrier =
        full_barrier + kPipelineStages;
    auto stage_control = reinterpret_cast<volatile unsigned long long*>(
        empty_barrier + kPipelineStages);
    auto smem_output =
        reinterpret_cast<__nv_bfloat16*>(smem + kOutputOffset);

    auto dense_desc = deep_gemm::mma::sm90::make_smem_desc(
        smem_dense(0), static_cast<int>(cute::GMMA::LayoutType::B128),
        0, 1024);
    auto sparse_desc = deep_gemm::mma::sm90::make_smem_desc(
        smem_sparse(0), static_cast<int>(cute::GMMA::LayoutType::B64),
        0, 512);
    auto activation_desc = deep_gemm::mma::sm90::make_smem_desc(
        smem_activation(0),
        static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
    const unsigned dense_desc_base_lo =
        __shfl_sync(0xffffffff, dense_desc.reg32_[0], 0);
    const unsigned sparse_desc_base_lo =
        __shfl_sync(0xffffffff, sparse_desc.reg32_[0], 0);
    const unsigned activation_desc_base_lo =
        __shfl_sync(0xffffffff, activation_desc.reg32_[0], 0);

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0;
             stage < kPipelineStages; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(4);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp == 6 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_activation);
        cute::prefetch_tma_descriptor(&tensor_map_dense);
        cute::prefetch_tma_descriptor(&tensor_map_sparse);
        cute::prefetch_tma_descriptor(&tensor_map_output);
    }
    __syncthreads();

    int producer_stage = 0;
    unsigned producer_phase = 0;
    int consumer_stage = 0;
    unsigned consumer_phase = 0;
    for (int tile_idx = static_cast<int>(blockIdx.x);
         tile_idx < total_tiles; tile_idx += static_cast<int>(gridDim.x)) {
        const int tile_m = tile_idx % tiles_m;
        const int tile_n = tile_idx / tiles_m;
        const int expert = tile_m / tiles_per_expert;
        const int local_tile_m = tile_m % tiles_per_expert;
        const int local_m =
            local_tile_m *
            kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive;
        const int output_tile_m =
            expert * kMaxM + local_m;
        const int output_tile_n =
            tile_n * kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive;
        const int block_row = tile_n;
        const int remaining = grouped_index[expert] - local_m;
        const int valid_rows = remaining <= 0
            ? 0
            : (remaining < 32 ? remaining : 32);

        if (valid_rows == 0) {
            for (int index = static_cast<int>(threadIdx.x);
                 index <
                     kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
                         kOutputTileNGroupedMaskedOutput32x64Stage4Adaptive;
                 index += kThreadsGroupedMaskedOutput32x64Stage4Adaptive)
                smem_output[index] = __float2bfloat16(0.0f);
            __syncthreads();
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncthreads();
            continue;
        }

        if (warp == 6) {
            const bool is_leader = cute::elect_one_sync();
#pragma unroll
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                if (is_leader)
                    empty_barrier[producer_stage].wait(
                        producer_phase ^ 1);
                __syncwarp();
                const unsigned long long selector =
                    static_cast<unsigned long long>(
                        block_selector[
                            (static_cast<long long>(expert) *
                                 (kN / kBlock) +
                             block_row) *
                                block_groups +
                            block_group]);
                if (is_leader)
                    stage_control[producer_stage] = selector;
#pragma unroll
                for (int dense_slot = 0; dense_slot < kDenseCount;
                     ++dense_slot) {
                    const long long packed_block =
                        ((static_cast<long long>(expert) *
                              (kN / kBlock) +
                          block_row) *
                             block_groups +
                         block_group) *
                            kDenseCount +
                        dense_slot;
                    if (is_leader) {
                        deep_gemm::tma::copy<
                            64, 64, 128, cutlass::bfloat16_t>(
                            &tensor_map_dense,
                            &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_dense(producer_stage) +
                                dense_slot * kBlock * kBlock),
                            0, packed_block * kBlock);
                    }
                }
#pragma unroll
                for (int sparse_slot = 0; sparse_slot < kBlockN;
                     ++sparse_slot) {
                    const long long packed_block =
                        ((static_cast<long long>(expert) *
                              (kN / kBlock) +
                          block_row) *
                             block_groups +
                         block_group) *
                            kBlockN +
                        sparse_slot;
                    if (is_leader) {
                        deep_gemm::tma::copy<
                            32, 64, 64, cutlass::bfloat16_t>(
                            &tensor_map_sparse,
                            &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_sparse(producer_stage) +
                                sparse_slot * kBlock * (kBlock / 2)),
                            0, packed_block * kBlock);
                    }
                    reinterpret_cast<uint4*>(
                        smem_metadata(producer_stage) +
                        sparse_slot * 128)[lane] =
                        reinterpret_cast<const uint4*>(
                            hardware_metadata +
                            static_cast<long long>(packed_block) *
                                128)[lane];
                }
                __syncwarp();
                if (is_leader) {
                    deep_gemm::tma::copy<
                        kBlock * kBlockM,
                        kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive, 128,
                        cutlass::bfloat16_t>(
                        &tensor_map_activation,
                        &full_barrier[producer_stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(
                            smem_activation(producer_stage)),
                        block_group * kBlock * kBlockM,
                        output_tile_m);
                    full_barrier[producer_stage].arrive_and_expect_tx(
                        kDenseWeightBytes + kSparseWeightBytes +
                        kActivationBytes);
                }
                advance_pipeline_grouped_masked_output32x64_stage4_adaptive<kPipelineStages>(
                    producer_stage, producer_phase);
            }
        }

        if (warp < 4) {
            float accumulator[kAccumulatorCountGroupedMaskedOutput32x64Stage4Adaptive] = {};
            bool has_accumulator = false;
#pragma unroll
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                full_barrier[consumer_stage].wait(consumer_phase);
                const unsigned long long selector =
                    stage_control[consumer_stage];
                const unsigned stage_desc_offset =
                    consumer_stage * (kStageBytes / 16);
#pragma unroll
                for (int i = 0;
                     i < kAccumulatorCountGroupedMaskedOutput32x64Stage4Adaptive; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                deep_gemm::ptx::warpgroup_arrive();
                if constexpr (kBlockN == 1 && kBlockM == 2) {
                    const int sparse_local_block =
                        (selector & 1ULL) != 0 ? 0 : 1;
                    const int dense_local_block = 1 - sparse_local_block;
#pragma unroll
                    for (int k_tile = 0; k_tile < 2; ++k_tile) {
                        unsigned metadata = 0;
                        if (thread_in_metadata_group < 2) {
                            const int active_lane =
                                (lane >> 2) * 2 + thread_in_metadata_group;
                            metadata = smem_metadata(consumer_stage)[
                                (k_tile * 4 + warp_in_math_wg) * 16 +
                                active_lane];
                        }
                        sparse_desc.reg32_[0] =
                            sparse_desc_base_lo + stage_desc_offset +
                            k_tile * 2;
                        activation_desc.reg32_[0] =
                            activation_desc_base_lo + stage_desc_offset +
                            (sparse_local_block *
                                 kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
                                 kBlock +
                             k_tile * 32) /
                                8;
                        sparse_wgmma_grouped_masked_output32x64_stage4_adaptive(
                            sparse_desc.desc_, activation_desc.desc_, accumulator,
                            metadata, has_accumulator || k_tile != 0);
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < 4; ++k_tile) {
                        dense_desc.reg32_[0] =
                            dense_desc_base_lo + stage_desc_offset +
                            k_tile * 2;
                        activation_desc.reg32_[0] =
                            activation_desc_base_lo + stage_desc_offset +
                            (dense_local_block *
                                 kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
                                 kBlock +
                             k_tile * 16) /
                                8;
                        DenseMMAGroupedMaskedOutput32x64Stage4Adaptive::wgmma(
                            dense_desc.desc_, activation_desc.desc_,
                            accumulator, true);
                    }
                } else {
#pragma unroll
                for (int local_block = 0; local_block < kBlockM;
                     ++local_block) {
                    const bool is_sparse =
                        static_cast<bool>(
                            (selector >> local_block) & 1ULL);
                    const unsigned long long lower_mask =
                        (1ULL << local_block) - 1ULL;
                    const int sparse_slot =
                        __popcll(selector & lower_mask);
                    const int dense_slot = local_block - sparse_slot;
                    if (is_sparse) {
#pragma unroll
                        for (int k_tile = 0; k_tile < 2; ++k_tile) {
                            unsigned metadata = 0;
                            if (thread_in_metadata_group < 2) {
                                const int active_lane =
                                    (lane >> 2) * 2 +
                                    thread_in_metadata_group;
                                metadata = smem_metadata(consumer_stage)[
                                    sparse_slot * 128 +
                                    (k_tile * 4 + warp_in_math_wg) * 16 +
                                    active_lane];
                            }
                            const auto desc_a =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_sparse(consumer_stage) +
                                        sparse_slot * kBlock *
                                            (kBlock / 2) +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B64),
                                    0, 512);
                            const auto desc_b =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_activation(consumer_stage) +
                                        local_block *
                                            kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
                                            kBlock +
                                        k_tile * 32,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            sparse_wgmma_grouped_masked_output32x64_stage4_adaptive(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                metadata,
                                has_accumulator || local_block != 0 ||
                                    k_tile != 0);
                        }
                    } else {
#pragma unroll
                        for (int k_tile = 0; k_tile < 4; ++k_tile) {
                            const auto desc_a =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_dense(consumer_stage) +
                                        dense_slot * kBlock * kBlock +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            const auto desc_b =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_activation(consumer_stage) +
                                        local_block *
                                            kOutputTileMGroupedMaskedOutput32x64Stage4Adaptive *
                                            kBlock +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            DenseMMAGroupedMaskedOutput32x64Stage4Adaptive::wgmma(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                has_accumulator || local_block != 0 ||
                                    k_tile != 0);
                        }
                    }
                }
                }
                deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                for (int i = 0;
                     i < kAccumulatorCountGroupedMaskedOutput32x64Stage4Adaptive; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                deep_gemm::ptx::warpgroup_wait<0>();
                release_stage(&empty_barrier[consumer_stage]);
                has_accumulator = true;
                advance_pipeline_grouped_masked_output32x64_stage4_adaptive<kPipelineStages>(
                    consumer_stage, consumer_phase);
            }

#pragma unroll
            for (int atom = 0; atom < 4; ++atom) {
                const auto bf16_0 = __float22bfloat162_rn(
                    {accumulator[atom * 4],
                     accumulator[atom * 4 + 1]});
                const auto bf16_1 = __float22bfloat162_rn(
                    {accumulator[atom * 4 + 2],
                     accumulator[atom * 4 + 3]});
                const int row = lane & 7;
                const int col = warp_in_math_wg * 2 + lane / 8;
                auto* smem_ptr =
                    smem_output + (atom * 8 + row) * kBlock +
                    ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<
                    __nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedMaskedOutput32x64Stage4Adaptive, 0);
            for (int index = static_cast<int>(threadIdx.x);
                 index < (32 - valid_rows) * 64;
                 index += kMathThreadsGroupedMaskedOutput32x64Stage4Adaptive) {
                const int row = valid_rows + index / 64;
                const int col = index % 64;
                const int physical_col =
                    ((col / 8) ^ (row & 7)) * 8 + col % 8;
                smem_output[row * 64 + physical_col] =
                    __float2bfloat16(0.0f);
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedMaskedOutput32x64Stage4Adaptive, 0);
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedMaskedOutput32x64Stage4Adaptive, 1);
        }
    }
}
