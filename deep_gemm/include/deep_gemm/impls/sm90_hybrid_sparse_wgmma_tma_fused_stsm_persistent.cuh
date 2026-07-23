#pragma once

// Hybrid sparse 64x64 fused mainloop with persistent tile scheduling.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

template <int = 0>
__global__ __launch_bounds__(kThreads, 1)
void hybrid_sparse_fused_wgmma_tma_stsm_persistent(
        const long long* block_selector, const unsigned char* sparse_metadata,
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
        kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kStageBytes = kDenseWeightBytes + kActivationBytes;
    constexpr int kMetadataBytesPerBlock = kBlock * (kBlock / 4);
    constexpr int kBarrierBytes = 2 * kStages * sizeof(Barrier);
    constexpr int kOutputBytes = kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kStages * kStageBytes + kBarrierBytes + 1023) / 1024) * 1024;
    constexpr int kVectorBytes = sizeof(uint4);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int metadata_group = lane >> 2;
    const int thread_in_metadata_group = lane & 3;
    const int block_groups = k / (kBlock * block_m);
    const int dense_count = block_m - block_n;
    const int metadata_bytes = block_groups * block_n * kMetadataBytesPerBlock;
    const int metadata_vectors = metadata_bytes / kVectorBytes;
    const int tiles_m = (m + kBlock - 1) / kBlock;
    const int tiles_n = (n + kBlock - 1) / kBlock;
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
    auto full_barrier = reinterpret_cast<Barrier*>(smem + kStages * kStageBytes);
    auto empty_barrier = full_barrier + kStages;
    auto smem_output = reinterpret_cast<__nv_bfloat16*>(smem + kOutputOffset);
    auto smem_metadata = smem + kOutputOffset + kOutputBytes;

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
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
        const int output_tile_m = tile_m * kBlock;
        const int output_tile_n = tile_n * kBlock;
        const int block_row = tile_n;

        const auto* global_metadata = sparse_metadata +
            static_cast<long long>(block_row) * metadata_bytes;
        for (int vector = static_cast<int>(threadIdx.x);
             vector < metadata_vectors; vector += kThreads) {
            reinterpret_cast<uint4*>(smem_metadata)[vector] =
                reinterpret_cast<const uint4*>(global_metadata)[vector];
        }
        __syncthreads();

        if (warp == 6 && cute::elect_one_sync()) {
            for (int block_group = 0; block_group < block_groups; ++block_group) {
                const unsigned long long selector = static_cast<unsigned long long>(
                    block_selector[block_row * block_groups + block_group]);
                int dense_slot = 0;
                int sparse_slot = 0;
                for (int local_block = 0; local_block < block_m; ++local_block) {
                    empty_barrier[producer_stage].wait(producer_phase ^ 1);
                    const bool is_sparse = (selector >> local_block) & 1ULL;
                    const int block_k =
                        (block_group * block_m + local_block) * kBlock;
                    int weight_bytes;
                    if (is_sparse) {
                        const int packed_block =
                            (block_row * block_groups + block_group) * block_n +
                            sparse_slot;
                        deep_gemm::tma::copy<32, 64, 64, cutlass::bfloat16_t>(
                            &tensor_map_sparse, &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_weight(producer_stage)),
                            0, packed_block * kBlock);
                        weight_bytes = kSparseWeightBytes;
                        ++sparse_slot;
                    } else {
                        const int packed_block =
                            (block_row * block_groups + block_group) * dense_count +
                            dense_slot;
                        deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                            &tensor_map_dense, &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_weight(producer_stage)),
                            0, packed_block * kBlock);
                        weight_bytes = kDenseWeightBytes;
                        ++dense_slot;
                    }
                    deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                        &tensor_map_activation, &full_barrier[producer_stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(
                            smem_activation(producer_stage)),
                        block_k, output_tile_m);
                    full_barrier[producer_stage].arrive_and_expect_tx(
                        weight_bytes + kActivationBytes);
                    advance_pipeline(producer_stage, producer_phase);
                }
            }
        }

        if (warp < 4) {
            float accumulator[32] = {};
            bool has_accumulator = false;
            for (int block_group = 0; block_group < block_groups; ++block_group) {
                const unsigned long long selector = static_cast<unsigned long long>(
                    block_selector[block_row * block_groups + block_group]);
                int sparse_slot = 0;
                for (int local_block = 0; local_block < block_m; ++local_block) {
                    const bool is_sparse = (selector >> local_block) & 1ULL;
                    full_barrier[consumer_stage].wait(consumer_phase);
#pragma unroll
                    for (int i = 0; i < 32; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
                    deep_gemm::ptx::warpgroup_arrive();
                    if (is_sparse) {
                        const int metadata_base =
                            (block_group * block_n + sparse_slot) *
                            kMetadataBytesPerBlock;
#pragma unroll
                        for (int k_tile = 0; k_tile < 2; ++k_tile) {
                            unsigned hardware_metadata = 0;
                            if (thread_in_metadata_group < 2) {
                                const int metadata_row = warp * 16 + metadata_group;
                                const int quartet_base =
                                    k_tile * 8 + thread_in_metadata_group * 4;
#pragma unroll
                                for (int quartet = 0; quartet < 4; ++quartet) {
                                    const unsigned char lower_code = smem_metadata[
                                        metadata_base +
                                        metadata_row * (kBlock / 4) +
                                        quartet_base + quartet];
                                    const unsigned char upper_code = smem_metadata[
                                        metadata_base +
                                        (metadata_row + 8) * (kBlock / 4) +
                                        quartet_base + quartet];
                                    hardware_metadata |=
                                        metadata_nibble(lower_code) << (quartet * 4);
                                    hardware_metadata |=
                                        metadata_nibble(upper_code) <<
                                        ((quartet + 4) * 4);
                                }
                            }
                            const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                                smem_weight(consumer_stage) + k_tile * 16,
                                static_cast<int>(cute::GMMA::LayoutType::B64),
                                0, 512);
                            const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                                smem_activation(consumer_stage) + k_tile * 32,
                                static_cast<int>(cute::GMMA::LayoutType::B128),
                                0, 1024);
                            sparse_wgmma(desc_a.desc_, desc_b.desc_, accumulator,
                                         hardware_metadata,
                                         has_accumulator || k_tile != 0);
                        }
                        ++sparse_slot;
                    } else {
#pragma unroll
                        for (int k_tile = 0; k_tile < 4; ++k_tile) {
                            const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                                smem_weight(consumer_stage) + k_tile * 16,
                                static_cast<int>(cute::GMMA::LayoutType::B128),
                                0, 1024);
                            const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                                smem_activation(consumer_stage) + k_tile * 16,
                                static_cast<int>(cute::GMMA::LayoutType::B128),
                                0, 1024);
                            DenseMMA::wgmma(desc_a.desc_, desc_b.desc_, accumulator,
                                            has_accumulator || k_tile != 0);
                        }
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0; i < 32; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release_stage(&empty_barrier[consumer_stage]);
                    has_accumulator = true;
                    advance_pipeline(consumer_stage, consumer_phase);
                }
            }

#pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
                const auto bf16_0 = __float22bfloat162_rn(
                    {accumulator[atom * 4], accumulator[atom * 4 + 1]});
                const auto bf16_1 = __float22bfloat162_rn(
                    {accumulator[atom * 4 + 2], accumulator[atom * 4 + 3]});
                const int row = lane & 7;
                const int col = warp * 2 + lane / 8;
                auto* smem_ptr = smem_output + (atom * 8 + row) * kBlock +
                    ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<__nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kMathThreads, 0);
            if (warp == 0 && cute::elect_one_sync()) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output, smem_output,
                    output_tile_n, output_tile_m);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }
        __syncthreads();
    }
}
