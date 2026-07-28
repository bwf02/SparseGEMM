#pragma once

// Small-M kernel: pair one 64x64 weight tile with a 32x64 activation tile.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

constexpr int kOutputTileMProducerMetadataCopy32x64 = 32;
constexpr int kOutputTileNProducerMetadataCopy32x64 = 64;
constexpr int kMathThreadsProducerMetadataCopy32x64 = 128;
constexpr int kThreadsProducerMetadataCopy32x64 = 256;
constexpr int kPipelineStagesProducerMetadataCopy32x64 = 3;
constexpr int kAccumulatorCountProducerMetadataCopy32x64 = 16;

using DenseMMAProducerMetadataCopy32x64 =
    typename deep_gemm::mma::sm90::BF16MMASelector<32>::type;

template <size_t... I>
__device__ __forceinline__ void sparse_wgmma_producer_metadata_copy_32x64_impl(
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

__device__ __forceinline__ void sparse_wgmma_producer_metadata_copy_32x64(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate) {
    sparse_wgmma_producer_metadata_copy_32x64_impl(
        desc_a, desc_b, accumulator, metadata, accumulate,
        cute::make_index_sequence<kAccumulatorCountProducerMetadataCopy32x64>{});
}

__device__ __forceinline__ void advance_pipeline_producer_metadata_copy_32x64(
        int& stage, unsigned& phase) {
    stage = stage == kPipelineStagesProducerMetadataCopy32x64 - 1
        ? 0
        : stage + 1;
    phase ^= stage == 0;
}

template <int = 0>
__global__ __launch_bounds__(kThreadsProducerMetadataCopy32x64, 1)
void hybrid_sparse_output32x64_stage_kind(
        const long long* block_selector, const unsigned* hardware_metadata,
        __nv_bfloat16*,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int kDenseWeightBytes =
        kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseWeightBytes =
        kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes =
        kOutputTileMProducerMetadataCopy32x64 * kBlock *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytesPerBlock = 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kStageBytes =
        kDenseWeightBytes + kActivationBytes + kMetadataBytesPerBlock;
    constexpr int kBarrierBytes =
        2 * kPipelineStagesProducerMetadataCopy32x64 * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMProducerMetadataCopy32x64 *
        kOutputTileNProducerMetadataCopy32x64 * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStagesProducerMetadataCopy32x64 * kStageBytes +
          kBarrierBytes + 1023) / 1024) * 1024;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;
    const int tiles_m =
        (m + kOutputTileMProducerMetadataCopy32x64 - 1) /
        kOutputTileMProducerMetadataCopy32x64;
    const int tiles_n =
        (n + kOutputTileNProducerMetadataCopy32x64 - 1) /
        kOutputTileNProducerMetadataCopy32x64;
    const int total_tiles = tiles_m * tiles_n;

    extern __shared__ __align__(1024) unsigned char smem[];
    auto stage_base = [&](const int stage) { return smem + stage * kStageBytes; };
    auto smem_weight = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_base(stage));
    };
    auto smem_activation = [&](const int stage) {
        return reinterpret_cast<__nv_bfloat16*>(
            stage_base(stage) + kDenseWeightBytes);
    };
    auto smem_metadata = [&](const int stage) {
        return reinterpret_cast<unsigned*>(
            stage_base(stage) + kDenseWeightBytes + kActivationBytes);
    };
    auto full_barrier = reinterpret_cast<Barrier*>(
        smem + kPipelineStagesProducerMetadataCopy32x64 * kStageBytes);
    auto empty_barrier =
        full_barrier + kPipelineStagesProducerMetadataCopy32x64;
    auto stage_control = reinterpret_cast<volatile unsigned long long*>(
        empty_barrier + kPipelineStagesProducerMetadataCopy32x64);
    auto smem_output =
        reinterpret_cast<__nv_bfloat16*>(smem + kOutputOffset);

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0;
             stage < kPipelineStagesProducerMetadataCopy32x64; ++stage) {
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
        const int output_tile_m =
            tile_m * kOutputTileMProducerMetadataCopy32x64;
        const int output_tile_n =
            tile_n * kOutputTileNProducerMetadataCopy32x64;
        const int block_row = tile_n;

        if (warp == 6) {
            const bool is_leader = cute::elect_one_sync();
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                const unsigned long long selector =
                    static_cast<unsigned long long>(
                        block_selector[
                            block_row * block_groups + block_group]);
                int dense_slot = 0;
                int sparse_slot = 0;
                for (int local_block = 0; local_block < block_m;
                     ++local_block) {
                    if (is_leader)
                        empty_barrier[producer_stage].wait(
                            producer_phase ^ 1);
                    __syncwarp();
                    const bool is_sparse =
                        (selector >> local_block) & 1ULL;
                    if (is_leader)
                        stage_control[producer_stage] =
                            static_cast<unsigned long long>(is_sparse);
                    __syncwarp();
                    const int block_k =
                        (block_group * block_m + local_block) * kBlock;
                    int weight_bytes;
                    if (is_sparse) {
                        const int packed_block =
                            (block_row * block_groups + block_group) *
                                block_n +
                            sparse_slot;
                        if (is_leader) {
                            deep_gemm::tma::copy<
                                32, 64, 64, cutlass::bfloat16_t>(
                                &tensor_map_sparse,
                                &full_barrier[producer_stage],
                                reinterpret_cast<cutlass::bfloat16_t*>(
                                    smem_weight(producer_stage)),
                                0, packed_block * kBlock);
                        }
                        reinterpret_cast<uint4*>(
                            smem_metadata(producer_stage))[lane] =
                            reinterpret_cast<const uint4*>(
                                hardware_metadata +
                                static_cast<long long>(packed_block) *
                                    128)[lane];
                        __syncwarp();
                        weight_bytes = kSparseWeightBytes;
                        ++sparse_slot;
                    } else {
                        const int packed_block =
                            (block_row * block_groups + block_group) *
                                dense_count +
                            dense_slot;
                        if (is_leader) {
                            deep_gemm::tma::copy<
                                64, 64, 128, cutlass::bfloat16_t>(
                                &tensor_map_dense,
                                &full_barrier[producer_stage],
                                reinterpret_cast<cutlass::bfloat16_t*>(
                                    smem_weight(producer_stage)),
                                0, packed_block * kBlock);
                        }
                        weight_bytes = kDenseWeightBytes;
                        ++dense_slot;
                    }
                    if (is_leader) {
                        deep_gemm::tma::copy<
                            64, 32, 128, cutlass::bfloat16_t>(
                            &tensor_map_activation,
                            &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_activation(producer_stage)),
                            block_k, output_tile_m);
                        full_barrier[producer_stage].arrive_and_expect_tx(
                            weight_bytes + kActivationBytes);
                    }
                    advance_pipeline_producer_metadata_copy_32x64(
                        producer_stage, producer_phase);
                }
            }
        }

        if (warp < 4) {
            float accumulator[kAccumulatorCountProducerMetadataCopy32x64] = {};
            bool has_accumulator = false;
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                for (int local_block = 0; local_block < block_m;
                     ++local_block) {
                    full_barrier[consumer_stage].wait(consumer_phase);
                    const bool is_sparse =
                        static_cast<bool>(stage_control[consumer_stage]);
#pragma unroll
                    for (int i = 0;
                         i < kAccumulatorCountProducerMetadataCopy32x64; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_arrive();
                    if (is_sparse) {
#pragma unroll
                        for (int k_tile = 0; k_tile < 2; ++k_tile) {
                            unsigned metadata = 0;
                            if (thread_in_metadata_group < 2) {
                                const int active_lane =
                                    (lane >> 2) * 2 +
                                    thread_in_metadata_group;
                                metadata = smem_metadata(consumer_stage)[
                                    (k_tile * 4 + warp_in_math_wg) * 16 +
                                    active_lane];
                            }
                            const auto desc_a =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_weight(consumer_stage) +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B64),
                                    0, 512);
                            const auto desc_b =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_activation(consumer_stage) +
                                        k_tile * 32,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            sparse_wgmma_producer_metadata_copy_32x64(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                metadata,
                                has_accumulator || k_tile != 0);
                        }
                    } else {
#pragma unroll
                        for (int k_tile = 0; k_tile < 4; ++k_tile) {
                            const auto desc_a =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_weight(consumer_stage) +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            const auto desc_b =
                                deep_gemm::mma::sm90::make_smem_desc(
                                    smem_activation(consumer_stage) +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            DenseMMAProducerMetadataCopy32x64::wgmma(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                has_accumulator || k_tile != 0);
                        }
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0;
                         i < kAccumulatorCountProducerMetadataCopy32x64; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release_stage(&empty_barrier[consumer_stage]);
                    has_accumulator = true;
                    advance_pipeline_producer_metadata_copy_32x64(
                        consumer_stage, consumer_phase);
                }
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
                kMathThreadsProducerMetadataCopy32x64, 0);
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataCopy32x64, 1);
        }
    }
}
