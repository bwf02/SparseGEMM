#pragma once

// Split K across two CTAs and let the second arrival reduce in-kernel.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

constexpr int kOutputTileMProducerMetadataGroupStage64x64SplitK2 = 64;
constexpr int kOutputTileNProducerMetadataGroupStage64x64SplitK2 = 64;
constexpr int kMathThreadsProducerMetadataGroupStage64x64SplitK2 = 128;
constexpr int kThreadsProducerMetadataGroupStage64x64SplitK2 = 256;
constexpr int kAccumulatorCountProducerMetadataGroupStage64x64SplitK2 = 32;

using DenseMMAProducerMetadataGroupStage64x64SplitK2 =
    typename deep_gemm::mma::sm90::BF16MMASelector<64>::type;

template <size_t... I>
__device__ __forceinline__ void sparse_wgmma_group_stage_64x64_splitk2_impl(
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

__device__ __forceinline__ void sparse_wgmma_group_stage_64x64_splitk2(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate) {
    sparse_wgmma_group_stage_64x64_splitk2_impl(
        desc_a, desc_b, accumulator, metadata, accumulate,
        cute::make_index_sequence<kAccumulatorCountProducerMetadataGroupStage64x64SplitK2>{});
}

__device__ __forceinline__ void store_group_stage_64x64_splitk2_partial(
        const float* accumulator, float* partial,
        const int output_tile_m, const int output_tile_n,
        const int m, const int n) {
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

template <int kPipelineStages>
__device__ __forceinline__ void advance_pipeline_group_stage_64x64_splitk2(
        int& stage, unsigned& phase) {
    stage = stage == kPipelineStages - 1
        ? 0
        : stage + 1;
    phase ^= stage == 0;
}

template <int kBlockN, int kBlockM, int kPipelineStages>
__global__ __launch_bounds__(kThreadsProducerMetadataGroupStage64x64SplitK2, 1)
void hybrid_sparse_group_stage_output64x64_splitk2_fused_reduce(
        const long long* block_selector, const unsigned* hardware_metadata,
        float* partial, int* tile_counters, __nv_bfloat16* output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    static_assert(kBlockN == 1 && kBlockM == 2);
    if (block_n != kBlockN || block_m != kBlockM)
        return;
    constexpr int kDenseCount = kBlockM - kBlockN;
    constexpr int kDenseWeightBytes =
        kDenseCount * kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseWeightBytes =
        kBlockN * kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes =
        kOutputTileMProducerMetadataGroupStage64x64SplitK2 * kBlock * kBlockM *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes =
        kBlockN * 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kStageBytes =
        kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes =
        2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMProducerMetadataGroupStage64x64SplitK2 *
        kOutputTileNProducerMetadataGroupStage64x64SplitK2 * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes +
          kBarrierBytes + 1023) / 1024) * 1024;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    const int block_groups = k / (kBlock * kBlockM);
    const int tiles_m =
        (m + kOutputTileMProducerMetadataGroupStage64x64SplitK2 - 1) /
        kOutputTileMProducerMetadataGroupStage64x64SplitK2;
    const int tiles_n =
        (n + kOutputTileNProducerMetadataGroupStage64x64SplitK2 - 1) /
        kOutputTileNProducerMetadataGroupStage64x64SplitK2;
    const int total_tiles = tiles_m * tiles_n;
    const int total_tasks = total_tiles * 2;

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
    auto reduction_flag = reinterpret_cast<int*>(smem + kOutputOffset);

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
    }
    __syncthreads();

    int producer_stage = 0;
    unsigned producer_phase = 0;
    int consumer_stage = 0;
    unsigned consumer_phase = 0;
    for (int task_idx = static_cast<int>(blockIdx.x);
         task_idx < total_tasks; task_idx += static_cast<int>(gridDim.x)) {
        const int tile_idx = task_idx >> 1;
        const int split_idx = task_idx & 1;
        const int tile_m = tile_idx % tiles_m;
        const int tile_n = tile_idx / tiles_m;
        const int output_tile_m =
            tile_m * kOutputTileMProducerMetadataGroupStage64x64SplitK2;
        const int output_tile_n =
            tile_n * kOutputTileNProducerMetadataGroupStage64x64SplitK2;
        const int block_row = tile_n;
        const int groups_per_split = (block_groups + 1) / 2;
        const int group_begin = split_idx * groups_per_split;
        const int group_end = min(block_groups, group_begin + groups_per_split);

        if (warp == 6) {
            const bool is_leader = cute::elect_one_sync();
            for (int block_group = group_begin; block_group < group_end;
                 ++block_group) {
                if (is_leader)
                    empty_barrier[producer_stage].wait(
                        producer_phase ^ 1);
                __syncwarp();
                const unsigned long long selector =
                    static_cast<unsigned long long>(
                        block_selector[
                            block_row * block_groups + block_group]);
                if (is_leader)
                    stage_control[producer_stage] = selector;
#pragma unroll
                for (int dense_slot = 0; dense_slot < kDenseCount;
                     ++dense_slot) {
                    const int packed_block =
                        (block_row * block_groups + block_group) *
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
                    const int packed_block =
                        (block_row * block_groups + block_group) *
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
                        kBlock * kBlockM, 64, 128,
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
                advance_pipeline_group_stage_64x64_splitk2<kPipelineStages>(
                    producer_stage, producer_phase);
            }
        }

        if (warp < 4) {
            float accumulator[kAccumulatorCountProducerMetadataGroupStage64x64SplitK2] = {};
            bool has_accumulator = false;
            for (int block_group = group_begin; block_group < group_end;
                 ++block_group) {
                full_barrier[consumer_stage].wait(consumer_phase);
                const unsigned long long selector =
                    stage_control[consumer_stage];
#pragma unroll
                for (int i = 0;
                     i < kAccumulatorCountProducerMetadataGroupStage64x64SplitK2; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                deep_gemm::ptx::warpgroup_arrive();
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
                                        local_block * kBlock * kBlock +
                                        k_tile * 32,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            sparse_wgmma_group_stage_64x64_splitk2(
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
                            DenseMMAProducerMetadataGroupStage64x64SplitK2::wgmma(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                has_accumulator || local_block != 0 ||
                                    k_tile != 0);
                        }
                    }
                }
                deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                for (int i = 0;
                     i < kAccumulatorCountProducerMetadataGroupStage64x64SplitK2; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                deep_gemm::ptx::warpgroup_wait<0>();
                release_stage(&empty_barrier[consumer_stage]);
                has_accumulator = true;
                advance_pipeline_group_stage_64x64_splitk2<kPipelineStages>(
                    consumer_stage, consumer_phase);
            }

            float* split_partial =
                partial + static_cast<long long>(split_idx) * m * n;
            store_group_stage_64x64_splitk2_partial(
                accumulator, split_partial, output_tile_m,
                output_tile_n, m, n);
        }

        __syncthreads();
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int previous = atomicAdd(tile_counters + tile_idx, 1);
            *reduction_flag = previous == 1;
        }
        __syncthreads();

        if (*reduction_flag) {
            for (int linear = static_cast<int>(threadIdx.x);
                 linear < kBlock * kBlock;
                 linear += static_cast<int>(blockDim.x)) {
                const int row_m = output_tile_m + linear / kBlock;
                const int row_n = output_tile_n + linear % kBlock;
                if (row_m < m && row_n < n) {
                    const long long index =
                        static_cast<long long>(row_m) * n + row_n;
                    output[index] = __float2bfloat16_rn(
                        partial[index] +
                        partial[static_cast<long long>(m) * n + index]);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && *reduction_flag)
            atomicExch(tile_counters + tile_idx, 0);
        __syncthreads();
    }
}
