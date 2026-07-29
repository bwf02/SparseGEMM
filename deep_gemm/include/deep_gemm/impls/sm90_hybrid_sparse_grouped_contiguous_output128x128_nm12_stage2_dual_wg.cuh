#pragma once

#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_contiguous_output128x64_nm12_persistent.cuh>

constexpr int kGroupedContiguousOutput128x128DualWGM = 128;
constexpr int kGroupedContiguousOutput128x128DualWGN = 128;
constexpr int kGroupedContiguousOutput128x128DualWGMathThreads = 256;
constexpr int kGroupedContiguousOutput128x128DualWGThreads = 384;
constexpr int kGroupedContiguousOutput128x128DualWGAccumulators = 64;

template <int kBlockN, int kBlockM, int kPipelineStages,
          int kNumExperts, int kTotalM, int kN, int kK>
__global__ __launch_bounds__(kGroupedContiguousOutput128x128DualWGThreads, 1)
void hybrid_sparse_grouped_contiguous_output128x128_nm12_stage2_dual_wg(
        const long long* block_selector, const unsigned* hardware_metadata,
        const int* grouped_index,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int total_m, const int num_experts, const int m_alignment,
        const int n, const int k, const int block_n, const int block_m) {
    static_assert(kBlockN == 1 && kBlockM == 2);
    static_assert(kPipelineStages == 2);
    static_assert(kNumExperts > 0 && kTotalM > 0 && kTotalM % 128 == 0);
    static_assert(kN % kGroupedContiguousOutput128x128DualWGN == 0 && kK % 128 == 0);
    if (block_n != kBlockN || block_m != kBlockM ||
        total_m != kTotalM || num_experts != kNumExperts ||
        m_alignment != 128 ||
        n != kN || k != kK)
        return;

    constexpr int kWeightRows = 2;
    constexpr int kDenseCount = kBlockM - kBlockN;
    constexpr int kDenseRowBytes =
        kDenseCount * kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseRowBytes =
        kBlockN * kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kMetadataRowBytes = kBlockN * 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kDenseWeightBytes = kWeightRows * kDenseRowBytes;
    constexpr int kSparseWeightBytes = kWeightRows * kSparseRowBytes;
    constexpr int kActivationBytes =
        kGroupedContiguousOutput128x128DualWGM * kBlock * kBlockM * sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes = kWeightRows * kMetadataRowBytes;
    constexpr int kStageBytes =
        kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes = 2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kGroupedContiguousOutput128x128DualWGM * kGroupedContiguousOutput128x128DualWGN * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes + kBarrierBytes + 1023) / 1024) * 1024;
    constexpr int kBlockGroups = kK / (kBlock * kBlockM);
    constexpr int kTilesM = kTotalM / kGroupedContiguousOutput128x128DualWGM;
    constexpr int kTilesN = kN / kGroupedContiguousOutput128x128DualWGN;
    constexpr int kTotalTiles = kTilesM * kTilesN;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int math_group = warp >> 2;
    const int warp_in_math_group = warp & 3;
    const int thread_in_metadata_group = lane & 3;

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
    auto empty_barrier = full_barrier + kPipelineStages;
    auto stage_control = reinterpret_cast<volatile unsigned long long*>(
        empty_barrier + kPipelineStages);
    auto smem_output = reinterpret_cast<__nv_bfloat16*>(smem + kOutputOffset);

    if (warp == 8 && lane == 0) {
#pragma unroll
        for (int stage = 0; stage < kPipelineStages; ++stage) {
            full_barrier[stage].init(1);
            empty_barrier[stage].init(8);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp == 10 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_activation);
        cute::prefetch_tma_descriptor(&tensor_map_dense);
        cute::prefetch_tma_descriptor(&tensor_map_sparse);
        cute::prefetch_tma_descriptor(&tensor_map_output);
    }
    __syncthreads();

    const int tile_m = static_cast<int>(blockIdx.x) % kTilesM;
    const int tile_n = static_cast<int>(blockIdx.x) / kTilesM;
    const int block_row_base = tile_n * kWeightRows;
    const int output_tile_m =
        tile_m * kGroupedContiguousOutput128x128DualWGM;
    const int output_tile_n = tile_n * kGroupedContiguousOutput128x128DualWGN;
    int expert = 0;
    int valid_rows = 0;
    int previous_end = 0;
#pragma unroll
    for (int current_expert = 0; current_expert < kNumExperts;
         ++current_expert) {
        const int start = current_expert == 0
            ? 0
            : ((previous_end + 127) / 128) * 128;
        const int end = grouped_index[current_expert];
        const int allocated_end = current_expert + 1 == kNumExperts
            ? kTotalM
            : ((end + 127) / 128) * 128;
        if (output_tile_m >= start && output_tile_m < allocated_end) {
            expert = current_expert;
            const int remaining = end - output_tile_m;
            valid_rows = remaining <= 0
                ? 0
                : (remaining < 128 ? remaining : 128);
            break;
        }
        previous_end = end;
    }

    if (valid_rows == 0) {
        for (int index = static_cast<int>(threadIdx.x);
             index < kGroupedContiguousOutput128x128DualWGM *
                         kGroupedContiguousOutput128x128DualWGN;
             index += kGroupedContiguousOutput128x128DualWGThreads)
            smem_output[index] = __float2bfloat16(0.0f);
        __syncthreads();
        if (threadIdx.x == 0) {
            cute::tma_store_fence();
#pragma unroll
            for (int output_atom = 0; output_atom < 2; ++output_atom) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output,
                    smem_output + output_atom * 128 * 64,
                    output_tile_n + output_atom * 64, output_tile_m);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        __syncthreads();
        return;
    }

    if (warp == 10) {
        const bool is_leader = cute::elect_one_sync();
        int producer_stage = 0;
        unsigned producer_phase = 0;
#pragma unroll
        for (int block_group = 0; block_group < kBlockGroups; ++block_group) {
            if (is_leader)
                empty_barrier[producer_stage].wait(producer_phase ^ 1);
            __syncwarp();
#pragma unroll
            for (int weight_row = 0; weight_row < kWeightRows; ++weight_row) {
                const long long selector_index =
                    (static_cast<long long>(expert) * (kN / kBlock) +
                     block_row_base + weight_row) * kBlockGroups + block_group;
                const unsigned long long selector =
                    static_cast<unsigned long long>(block_selector[selector_index]);
                if (is_leader)
                    stage_control[producer_stage * kWeightRows + weight_row] = selector;
                const long long dense_block = selector_index * kDenseCount;
                const long long sparse_block = selector_index * kBlockN;
                if (is_leader) {
                    deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                        &tensor_map_dense, &full_barrier[producer_stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(
                            smem_dense(producer_stage) +
                            weight_row * kDenseRowBytes / sizeof(__nv_bfloat16)),
                        0, dense_block * kBlock);
                    deep_gemm::tma::copy<32, 64, 64, cutlass::bfloat16_t>(
                        &tensor_map_sparse, &full_barrier[producer_stage],
                        reinterpret_cast<cutlass::bfloat16_t*>(
                            smem_sparse(producer_stage) +
                            weight_row * kSparseRowBytes / sizeof(__nv_bfloat16)),
                        0, sparse_block * kBlock);
                }
                reinterpret_cast<uint4*>(
                    smem_metadata(producer_stage) + weight_row * 128)[lane] =
                    reinterpret_cast<const uint4*>(
                        hardware_metadata + sparse_block * 128)[lane];
            }
            __syncwarp();
            if (is_leader) {
                deep_gemm::tma::copy<128, 128, 128, cutlass::bfloat16_t>(
                    &tensor_map_activation, &full_barrier[producer_stage],
                    reinterpret_cast<cutlass::bfloat16_t*>(
                        smem_activation(producer_stage)),
                    block_group * 128, output_tile_m);
                full_barrier[producer_stage].arrive_and_expect_tx(
                    kDenseWeightBytes + kSparseWeightBytes + kActivationBytes);
            }
            advance_pipeline_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid<kPipelineStages>(
                producer_stage, producer_phase);
        }
    }

    if (warp < 8) {
        auto run_math = [&]<int kWeightRow>() {
            float accumulator[kGroupedContiguousOutput128x128DualWGAccumulators] = {};
            [&]<size_t... Group>(cute::index_sequence<Group...>) {
                ([&] {
                    constexpr int kGroup = static_cast<int>(Group);
                    constexpr int kStage = kGroup % kPipelineStages;
                    constexpr unsigned kPhase =
                        (kGroup / kPipelineStages) & 1;
                    constexpr unsigned kStageByteOffset =
                        1024 + kStage * kStageBytes;
                    constexpr unsigned kDenseByteOffset =
                        kStageByteOffset + kWeightRow * kDenseRowBytes;
                    constexpr unsigned kSparseByteOffset =
                        kStageByteOffset + kDenseWeightBytes +
                        kWeightRow * kSparseRowBytes;
                    constexpr unsigned kActivationByteOffset =
                        kStageByteOffset + kDenseWeightBytes +
                        kSparseWeightBytes;
                    constexpr unsigned long long kSparseDesc0 =
                        make_constexpr_gmma_desc(kSparseByteOffset, 512,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B64));
                    constexpr unsigned long long kSparseDesc1 =
                        make_constexpr_gmma_desc(kSparseByteOffset + 32, 512,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B64));
                    constexpr unsigned long long kDenseDesc0 =
                        make_constexpr_gmma_desc(kDenseByteOffset, 1024,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B128));
                    constexpr unsigned long long kDenseDesc1 =
                        make_constexpr_gmma_desc(kDenseByteOffset + 32, 1024,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B128));
                    constexpr unsigned long long kDenseDesc2 =
                        make_constexpr_gmma_desc(kDenseByteOffset + 64, 1024,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B128));
                    constexpr unsigned long long kDenseDesc3 =
                        make_constexpr_gmma_desc(kDenseByteOffset + 96, 1024,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B128));
                    constexpr auto activation_desc = [=](
                            const unsigned local_block,
                            const unsigned k_byte_offset) constexpr {
                        return make_constexpr_gmma_desc(
                            kActivationByteOffset +
                                local_block * kGroupedContiguousOutput128x128DualWGM * kBlock *
                                    sizeof(__nv_bfloat16) +
                                k_byte_offset,
                            1024, static_cast<unsigned>(
                                cute::GMMA::LayoutType::B128));
                    };

                    full_barrier[kStage].wait(kPhase);
                    const unsigned long long selector =
                        stage_control[kStage * kWeightRows + kWeightRow];
                    const int active_lane =
                        (lane >> 2) * 2 + thread_in_metadata_group;
                    unsigned metadata_0 = 0;
                    unsigned metadata_1 = 0;
                    if (thread_in_metadata_group < 2) {
                        metadata_0 = smem_metadata(kStage)[
                            kWeightRow * 128 +
                            warp_in_math_group * 16 + active_lane];
                        metadata_1 = smem_metadata(kStage)[
                            kWeightRow * 128 +
                            (4 + warp_in_math_group) * 16 + active_lane];
                    }
#pragma unroll
                    for (int i = 0; i < kGroupedContiguousOutput128x128DualWGAccumulators; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
                    deep_gemm::ptx::warpgroup_arrive();
                    if ((selector & 1ULL) != 0) {
                        wgmma_group_stage_128x64_nm12_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage4_async_group2(
                            kSparseDesc0, activation_desc(0, 0), metadata_0,
                            kSparseDesc1, activation_desc(0, 64), metadata_1,
                            kDenseDesc0, activation_desc(1, 0),
                            kDenseDesc1, activation_desc(1, 32),
                            kDenseDesc2, activation_desc(1, 64),
                            kDenseDesc3, activation_desc(1, 96),
                            accumulator, kGroup != 0);
                    } else {
                        wgmma_group_stage_128x64_nm12_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage4_async_group2(
                            kSparseDesc0, activation_desc(1, 0), metadata_0,
                            kSparseDesc1, activation_desc(1, 64), metadata_1,
                            kDenseDesc0, activation_desc(0, 0),
                            kDenseDesc1, activation_desc(0, 32),
                            kDenseDesc2, activation_desc(0, 64),
                            kDenseDesc3, activation_desc(0, 96),
                            accumulator, kGroup != 0);
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
                    for (int i = 0; i < kGroupedContiguousOutput128x128DualWGAccumulators; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(accumulator[i]);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    release_stage(&empty_barrier[kStage]);
                }(), ...);
            }(cute::make_index_sequence<kBlockGroups>{});

#pragma unroll
            for (int atom = 0; atom < 16; ++atom) {
                const auto bf16_0 = __float22bfloat162_rn(
                    {accumulator[atom * 4], accumulator[atom * 4 + 1]});
                const auto bf16_1 = __float22bfloat162_rn(
                    {accumulator[atom * 4 + 2], accumulator[atom * 4 + 3]});
                const int row = lane & 7;
                const int col = warp_in_math_group * 2 + lane / 8;
                auto* smem_ptr =
                    smem_output + kWeightRow * 128 * 64 +
                    (atom * 8 + row) * 64 + ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<__nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            for (int index = warp_in_math_group * 32 + lane;
                 index < (128 - valid_rows) * 64; index += 128) {
                const int row = valid_rows + index / 64;
                const int col = index % 64;
                const int physical_col =
                    ((col / 8) ^ (row & 7)) * 8 + col % 8;
                smem_output[kWeightRow * 128 * 64 + row * 64 + physical_col] =
                    __float2bfloat16(0.0f);
            }
        };

        if (math_group == 0)
            run_math.template operator()<0>();
        else
            run_math.template operator()<1>();
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kGroupedContiguousOutput128x128DualWGMathThreads, 0);
        if (warp == 0 && cute::elect_one_sync()) {
#pragma unroll
            for (int output_atom = 0; output_atom < 2; ++output_atom) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output,
                    smem_output + output_atom * 128 * 64,
                    output_tile_n + output_atom * 64, output_tile_m);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        cutlass::arch::NamedBarrier::sync(kGroupedContiguousOutput128x128DualWGMathThreads, 1);
    }
}
