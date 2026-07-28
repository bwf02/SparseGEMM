#pragma once

// Reuse each 64x64 weight block across two 64x64 activation tiles.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

#ifndef HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME
#define HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME \
    hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready_producer_metadata_copy_output128x64
#endif

#ifndef HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND
#define HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND 0
#endif

#ifndef HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
#define HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE 0
#endif

#ifndef HYBRID_SPARSE_OUTPUT128X64_THREADS
#define HYBRID_SPARSE_OUTPUT128X64_THREADS 384
#endif

#ifndef HYBRID_SPARSE_OUTPUT128X64_BARRIER_WARP
#define HYBRID_SPARSE_OUTPUT128X64_BARRIER_WARP 8
#endif

#ifndef HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP
#define HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP 10
#endif

constexpr int kOutputTileMProducerMetadataCopy128x64 = 128;
constexpr int kOutputTileNProducerMetadataCopy128x64 = 64;
constexpr int kMathThreadsProducerMetadataCopy128x64 = 256;
constexpr int kThreadsProducerMetadataCopy128x64 =
    HYBRID_SPARSE_OUTPUT128X64_THREADS;
constexpr int kPipelineStagesProducerMetadataCopy128x64 = 3;

__device__ __forceinline__ void advance_pipeline_producer_metadata_copy_128x64(
        int& stage, unsigned& phase) {
    stage = stage == kPipelineStagesProducerMetadataCopy128x64 - 1
        ? 0
        : stage + 1;
    phase ^= stage == 0;
}

template <int = 0>
__global__ __launch_bounds__(kThreadsProducerMetadataCopy128x64, 1)
void HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME(
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
        kOutputTileMProducerMetadataCopy128x64 * kBlock *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytesPerBlock = 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kStageBytes =
        kDenseWeightBytes + kActivationBytes + kMetadataBytesPerBlock;
    constexpr int kBarrierBytes =
        2 * kPipelineStagesProducerMetadataCopy128x64 * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMProducerMetadataCopy128x64 *
        kOutputTileNProducerMetadataCopy128x64 * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStagesProducerMetadataCopy128x64 * kStageBytes +
          kBarrierBytes + 1023) / 1024) * 1024;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int math_wg = warp >> 2;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;
    const int tiles_m =
        (m + kOutputTileMProducerMetadataCopy128x64 - 1) /
        kOutputTileMProducerMetadataCopy128x64;
    const int tiles_n =
        (n + kOutputTileNProducerMetadataCopy128x64 - 1) /
        kOutputTileNProducerMetadataCopy128x64;
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
        smem + kPipelineStagesProducerMetadataCopy128x64 * kStageBytes);
    auto empty_barrier =
        full_barrier + kPipelineStagesProducerMetadataCopy128x64;
    auto stage_control = reinterpret_cast<volatile unsigned long long*>(
        empty_barrier + kPipelineStagesProducerMetadataCopy128x64);
    auto smem_output =
        reinterpret_cast<__nv_bfloat16*>(smem + kOutputOffset);

#if HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
    auto dense_desc = deep_gemm::mma::sm90::make_smem_desc(
        smem_weight(0), static_cast<int>(cute::GMMA::LayoutType::B128),
        0, 1024);
    auto sparse_desc = deep_gemm::mma::sm90::make_smem_desc(
        smem_weight(0), static_cast<int>(cute::GMMA::LayoutType::B64),
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
#endif

    if (warp == HYBRID_SPARSE_OUTPUT128X64_BARRIER_WARP && lane == 0) {
#pragma unroll
        for (int stage = 0;
             stage < kPipelineStagesProducerMetadataCopy128x64; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(8);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp == HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP &&
        cute::elect_one_sync()) {
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
            tile_m * kOutputTileMProducerMetadataCopy128x64;
        const int output_tile_n =
            tile_n * kOutputTileNProducerMetadataCopy128x64;
        const int block_row = tile_n;

        if (warp == HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP) {
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
                    if constexpr (HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND) {
                        if (is_leader &&
                            (HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND == 1 ||
                             local_block == 0)) {
                            stage_control[producer_stage] =
                                HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND == 1
                                ? static_cast<unsigned long long>(is_sparse)
                                : selector;
                        }
                        __syncwarp();
                    }
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
                            64, 128, 128, cutlass::bfloat16_t>(
                            &tensor_map_activation,
                            &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_activation(producer_stage)),
                            block_k, output_tile_m);
                        full_barrier[producer_stage].arrive_and_expect_tx(
                            weight_bytes + kActivationBytes);
                    }
                    advance_pipeline_producer_metadata_copy_128x64(
                        producer_stage, producer_phase);
                }
            }
        }

        if (warp < 8) {
            float accumulator[32] = {};
            bool has_accumulator = false;
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                unsigned long long selector = 0;
                if constexpr (HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND == 0) {
                    selector = static_cast<unsigned long long>(
                        block_selector[
                            block_row * block_groups + block_group]);
                }
                for (int local_block = 0; local_block < block_m;
                     ++local_block) {
                    full_barrier[consumer_stage].wait(consumer_phase);
                    if constexpr (
                        HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND == 2) {
                        if (local_block == 0)
                            selector = stage_control[consumer_stage];
                    }
                    const bool is_sparse =
                        HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND == 1
                        ? static_cast<bool>(
                            stage_control[consumer_stage])
                        : static_cast<bool>(
                            (selector >> local_block) & 1ULL);
#pragma unroll
                    for (int i = 0; i < 32; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_arrive();
#if HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
                    const unsigned stage_desc_offset =
                        consumer_stage * (kStageBytes / 16);
                    const unsigned activation_math_offset =
                        math_wg * (kBlock * kBlock * sizeof(__nv_bfloat16) / 16);
#endif
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
#if HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
                            sparse_desc.reg32_[0] =
                                sparse_desc_base_lo + stage_desc_offset +
                                k_tile * (16 * sizeof(__nv_bfloat16) / 16);
                            activation_desc.reg32_[0] =
                                activation_desc_base_lo + stage_desc_offset +
                                activation_math_offset +
                                k_tile * (32 * sizeof(__nv_bfloat16) / 16);
                            sparse_wgmma(
                                sparse_desc.desc_, activation_desc.desc_,
                                accumulator, metadata,
                                has_accumulator || k_tile != 0);
#else
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
                                        math_wg * kBlock * kBlock +
                                        k_tile * 32,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            sparse_wgmma(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                metadata,
                                has_accumulator || k_tile != 0);
#endif
                        }
                    } else {
#pragma unroll
                        for (int k_tile = 0; k_tile < 4; ++k_tile) {
#if HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
                            dense_desc.reg32_[0] =
                                dense_desc_base_lo + stage_desc_offset +
                                k_tile * (16 * sizeof(__nv_bfloat16) / 16);
                            activation_desc.reg32_[0] =
                                activation_desc_base_lo + stage_desc_offset +
                                activation_math_offset +
                                k_tile * (16 * sizeof(__nv_bfloat16) / 16);
                            DenseMMA::wgmma(
                                dense_desc.desc_, activation_desc.desc_,
                                accumulator,
                                has_accumulator || k_tile != 0);
#else
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
                                        math_wg * kBlock * kBlock +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            DenseMMA::wgmma(
                                desc_a.desc_, desc_b.desc_, accumulator,
                                has_accumulator || k_tile != 0);
#endif
                        }
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0; i < 32; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[i]);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release_stage(&empty_barrier[consumer_stage]);
                    has_accumulator = true;
                    advance_pipeline_producer_metadata_copy_128x64(
                        consumer_stage, consumer_phase);
                }
            }

#pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
                const auto bf16_0 = __float22bfloat162_rn(
                    {accumulator[atom * 4],
                     accumulator[atom * 4 + 1]});
                const auto bf16_1 = __float22bfloat162_rn(
                    {accumulator[atom * 4 + 2],
                     accumulator[atom * 4 + 3]});
                const int row = lane & 7;
                const int col = warp_in_math_wg * 2 + lane / 8;
                auto* smem_ptr =
                    smem_output +
                    math_wg * kBlock * kBlock +
                    (atom * 8 + row) * kBlock +
                    ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<
                    __nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataCopy128x64, 0);
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsProducerMetadataCopy128x64, 1);
        }
    }
}

#undef HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME
#undef HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND
#undef HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE
#undef HYBRID_SPARSE_OUTPUT128X64_THREADS
#undef HYBRID_SPARSE_OUTPUT128X64_BARRIER_WARP
#undef HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP
