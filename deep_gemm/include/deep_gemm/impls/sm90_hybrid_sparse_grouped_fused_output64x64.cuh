#pragma once

// Fused hybrid sparse grouped GEMM for contiguous psum and masked layouts.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64.cuh>

template <int kGroupedMode>
__device__ __forceinline__ void resolve_hybrid_grouped_tile_64x64(
        const int tile_m, const int* grouped_index, const int total_rows,
        const int num_experts, const int max_m, const int m_alignment,
        int& expert, int& output_tile_m, int& valid_rows) {
    if constexpr (kGroupedMode == 0) {
        output_tile_m = tile_m * 64;
        expert = 0;
        valid_rows = 0;
        int previous_end = 0;
        for (int current_expert = 0; current_expert < num_experts;
             ++current_expert) {
            const int start = current_expert == 0
                ? 0
                : ((previous_end + m_alignment - 1) / m_alignment) *
                    m_alignment;
            const int end = grouped_index[current_expert];
            const int tiled_end = ((end + 63) / 64) * 64;
            if (output_tile_m >= start && output_tile_m < tiled_end) {
                expert = current_expert;
                const int remaining = end - output_tile_m;
                valid_rows = remaining <= 0 ? 0 : (remaining < 64 ? remaining : 64);
                return;
            }
            previous_end = end;
        }
        return;
    }

    const int tiles_per_expert = (max_m + 63) / 64;
    expert = tile_m / tiles_per_expert;
    const int local_tile = tile_m % tiles_per_expert;
    const int local_m = local_tile * 64;
    output_tile_m = expert * max_m + local_m;
    const int remaining = expert < num_experts
        ? grouped_index[expert] - local_m
        : 0;
    valid_rows = remaining <= 0 ? 0 : (remaining < 64 ? remaining : 64);
    if (output_tile_m >= total_rows)
        valid_rows = 0;
}

template <int kBlockN, int kBlockM, int kPipelineStages, int kGroupedMode>
__global__ __launch_bounds__(kThreadsProducerMetadataGroupStage64x64, 1)
void hybrid_sparse_grouped_fused_output64x64(
        const long long* block_selector, const unsigned* hardware_metadata,
        const int* grouped_index,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int total_rows, const int n, const int k,
        const int num_experts, const int max_m, const int m_alignment,
        const int block_n, const int block_m) {
    static_assert(kGroupedMode == 0 || kGroupedMode == 1);
    static_assert(0 < kBlockN && kBlockN <= kBlockM);
    if (block_n != kBlockN || block_m != kBlockM)
        return;
    constexpr int kDenseCount = kBlockM - kBlockN;
    constexpr int kDenseWeightBytes =
        kDenseCount * kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseWeightBytes =
        kBlockN * kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes =
        kOutputTileMProducerMetadataGroupStage64x64 * kBlock * kBlockM *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes =
        kBlockN * 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kStageBytes = kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes =
        2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMProducerMetadataGroupStage64x64 *
        kOutputTileNProducerMetadataGroupStage64x64 *
        sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes + kBarrierBytes + 1023) / 1024) *
        1024;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    const int block_rows = n / kBlock;
    const int block_groups = k / (kBlock * kBlockM);
    const int tiles_m = kGroupedMode == 0
        ? (total_rows + 63) / 64
        : num_experts * ((max_m + 63) / 64);
    const int tiles_n =
        (n + kOutputTileNProducerMetadataGroupStage64x64 - 1) /
        kOutputTileNProducerMetadataGroupStage64x64;
    const int total_tiles = tiles_m * tiles_n;

    extern __shared__ __align__(1024) unsigned char smem[];
    auto stage_base = [&](const int stage) {
        return smem + stage * kStageBytes;
    };
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
    auto empty_barrier = full_barrier + kPipelineStages;
    auto stage_control = reinterpret_cast<volatile unsigned long long*>(
        empty_barrier + kPipelineStages);
    auto smem_output = reinterpret_cast<__nv_bfloat16*>(
        smem + kOutputOffset);

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kPipelineStages; ++stage) {
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
         tile_idx < total_tiles;
         tile_idx += static_cast<int>(gridDim.x)) {
        const int tile_m = tile_idx % tiles_m;
        const int tile_n = tile_idx / tiles_m;
        const int output_tile_n =
            tile_n * kOutputTileNProducerMetadataGroupStage64x64;
        const int block_row = tile_n;
        int expert;
        int output_tile_m;
        int valid_rows;
        resolve_hybrid_grouped_tile_64x64<kGroupedMode>(
            tile_m, grouped_index, total_rows, num_experts, max_m,
            m_alignment, expert, output_tile_m, valid_rows);

        if (warp == 6 && valid_rows > 0) {
            const bool is_leader = cute::elect_one_sync();
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                if (is_leader)
                    empty_barrier[producer_stage].wait(producer_phase ^ 1);
                __syncwarp();
                const long long selector_index =
                    (static_cast<long long>(expert) * block_rows + block_row) *
                        block_groups +
                    block_group;
                const unsigned long long selector =
                    static_cast<unsigned long long>(
                        block_selector[selector_index]);
                if (is_leader)
                    stage_control[producer_stage] = selector;
#pragma unroll
                for (int dense_slot = 0; dense_slot < kDenseCount;
                     ++dense_slot) {
                    const long long packed_block =
                        selector_index * kDenseCount + dense_slot;
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
                        selector_index * kBlockN + sparse_slot;
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
                            hardware_metadata + packed_block * 128)[lane];
                }
                __syncwarp();
                if (is_leader) {
                    deep_gemm::tma::copy<
                        kBlock * kBlockM, 64, 128,
                        cutlass::bfloat16_t>(
                        &tensor_map_activation,
                        &full_barrier[producer_stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(
                            smem_activation(producer_stage)),
                        block_group * kBlock * kBlockM, output_tile_m);
                    full_barrier[producer_stage].arrive_and_expect_tx(
                        kDenseWeightBytes + kSparseWeightBytes +
                        kActivationBytes);
                }
                advance_pipeline_group_stage_64x64<kPipelineStages>(
                    producer_stage, producer_phase);
            }
        }

        if (warp < 4) {
            float accumulator[
                kAccumulatorCountProducerMetadataGroupStage64x64] = {};
            bool has_accumulator = false;
            if (valid_rows > 0) {
                for (int block_group = 0; block_group < block_groups;
                     ++block_group) {
                    full_barrier[consumer_stage].wait(consumer_phase);
                    const unsigned long long selector =
                        stage_control[consumer_stage];
#pragma unroll
                    for (int i = 0;
                         i < kAccumulatorCountProducerMetadataGroupStage64x64;
                         ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
                    for (int local_block = 0; local_block < kBlockM;
                         ++local_block) {
                        const bool is_sparse = static_cast<bool>(
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
                                            local_block * kBlock * kBlock +
                                            k_tile * 32,
                                        static_cast<int>(
                                            cute::GMMA::LayoutType::B128),
                                        0, 1024);
                                sparse_wgmma_group_stage_64x64(
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
                                            local_block * kBlock * kBlock +
                                            k_tile * 16,
                                        static_cast<int>(
                                            cute::GMMA::LayoutType::B128),
                                        0, 1024);
                                DenseMMAProducerMetadataGroupStage64x64::wgmma(
                                    desc_a.desc_, desc_b.desc_, accumulator,
                                    has_accumulator || local_block != 0 ||
                                        k_tile != 0);
                            }
                        }
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0;
                         i < kAccumulatorCountProducerMetadataGroupStage64x64;
                         ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release_stage(&empty_barrier[consumer_stage]);
                    has_accumulator = true;
                    advance_pipeline_group_stage_64x64<kPipelineStages>(
                        consumer_stage, consumer_phase);
                }
            }

#pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
                const int row = atom * 8 + (lane & 7);
                const auto bf16_0 = __float22bfloat162_rn(
                    make_float2(
                        accumulator[atom * 4],
                        accumulator[atom * 4 + 1]));
                const auto bf16_1 = __float22bfloat162_rn(
                    make_float2(
                        accumulator[atom * 4 + 2],
                        accumulator[atom * 4 + 3]));
                const int swizzle_row = lane & 7;
                const int col = warp_in_math_wg * 2 + lane / 8;
                auto* smem_ptr =
                    smem_output + row * kBlock +
                    ((col ^ swizzle_row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<
                    __nv_bfloat162>::copy(bf16_0, bf16_1, smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataGroupStage64x64, 0);
            for (int index = static_cast<int>(threadIdx.x);
                 index < (64 - valid_rows) * 64;
                 index += kMathThreadsProducerMetadataGroupStage64x64) {
                const int row = valid_rows + index / 64;
                const int col = index % 64;
                const int physical_col =
                    ((col / 8) ^ (row & 7)) * 8 + col % 8;
                smem_output[row * 64 + physical_col] =
                    __float2bfloat16(0.0f);
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataGroupStage64x64, 0);
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataGroupStage64x64, 1);
        }
    }
}
