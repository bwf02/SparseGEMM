#pragma once

// Hybrid sparse 64x64 kernel variant that stages one block-row of metadata.

#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma.cuh>

template <int = 0>
__global__ __launch_bounds__(kThreads, 1)
void hybrid_sparse_2_4_wgmma_tma_metadata_prefetch(
        const long long* block_selector, const unsigned char* sparse_metadata,
        float*, float* sparse_partial, __nv_bfloat16*,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const int m, const int n, const int k,
        const int block_n, const int block_m) {
    constexpr int kWeightBytes = kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kActivationBytes = kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kStageBytes = kWeightBytes + kActivationBytes;
    constexpr int kMetadataBytesPerBlock = kBlock * (kBlock / 4);
    constexpr int kBarrierBytes = 2 * kStages * sizeof(Barrier);
    constexpr int kVectorBytes = sizeof(uint4);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int output_tile_m = static_cast<int>(blockIdx.x) * kBlock;
    const int output_tile_n = static_cast<int>(blockIdx.y) * kBlock;
    const int block_row = output_tile_n / kBlock;
    const int block_groups = k / (kBlock * block_m);
    const int metadata_bytes = block_groups * block_n * kMetadataBytesPerBlock;

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
    auto smem_metadata = smem + kStages * kStageBytes + kBarrierBytes;

    if (warp == 4 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(4);
        }
        cutlass::arch::fence_barrier_init();
    }

    const auto* global_metadata = sparse_metadata +
        static_cast<long long>(block_row) * metadata_bytes;
    const int metadata_vectors = metadata_bytes / kVectorBytes;
    for (int vector = static_cast<int>(threadIdx.x);
         vector < metadata_vectors; vector += kThreads) {
        reinterpret_cast<uint4*>(smem_metadata)[vector] =
            reinterpret_cast<const uint4*>(global_metadata)[vector];
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
                        (block_group * block_m + local_block) * kBlock;
                    const int packed_block =
                        (block_row * block_groups + block_group) * block_n +
                        sparse_slot;
                    deep_gemm::tma::copy<32, 64, 64, cutlass::bfloat16_t>(
                        &tensor_map_sparse, &full_barrier[stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(smem_weight(stage)),
                        0, packed_block * kBlock);
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

    float accumulator[32] = {};
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
            const int metadata_base =
                (block_group * block_n + sparse_slot) * kMetadataBytesPerBlock;
            full_barrier[stage].wait(phase);
#pragma unroll
            for (int i = 0; i < 32; ++i)
                deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (int k_tile = 0; k_tile < 2; ++k_tile) {
                unsigned hardware_metadata = 0;
                if (thread_in_group < 2) {
                    const int metadata_row = warp * 16 + group;
                    const int quartet_base = k_tile * 8 + thread_in_group * 4;
#pragma unroll
                    for (int quartet = 0; quartet < 4; ++quartet) {
                        const unsigned char lower_code = smem_metadata[
                            metadata_base + metadata_row * (kBlock / 4) +
                            quartet_base + quartet];
                        const unsigned char upper_code = smem_metadata[
                            metadata_base + (metadata_row + 8) * (kBlock / 4) +
                            quartet_base + quartet];
                        hardware_metadata |=
                            metadata_nibble(lower_code) << (quartet * 4);
                        hardware_metadata |=
                            metadata_nibble(upper_code) << ((quartet + 4) * 4);
                    }
                }
                const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                    smem_weight(stage) + k_tile * 16,
                    static_cast<int>(cute::GMMA::LayoutType::B64), 0, 512);
                const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_activation(stage) + k_tile * 32,
                    static_cast<int>(cute::GMMA::LayoutType::B128), 0, 1024);
                sparse_wgmma(desc_a.desc_, desc_b.desc_, accumulator,
                             hardware_metadata, has_accumulator || k_tile != 0);
            }
            deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < 32; ++i)
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
}
