#pragma once

#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_masked_output64x64_nm12_fixed_full_grid.cuh>

constexpr int kGroupedContiguousOutput64x128SingleWGM = 64;
constexpr int kGroupedContiguousOutput64x128SingleWGN = 128;
constexpr int kGroupedContiguousOutput64x128SingleWGMathThreads = 128;
constexpr int kGroupedContiguousOutput64x128SingleWGThreads = 256;
constexpr int kGroupedContiguousOutput64x128SingleWGAccumulators = 32;

template <int kBlockN, int kBlockM, int kPipelineStages,
          int kNumExperts, int kMAlignment, int kN, int kK>
__global__ __launch_bounds__(kGroupedContiguousOutput64x128SingleWGThreads, 1)
void hybrid_sparse_grouped_contiguous_output64x128_nm12_stage2_single_wg(
        const long long* block_selector, const unsigned* hardware_metadata,
        const int* grouped_index,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int total_m, const int num_experts, const int m_alignment,
        const int n, const int k, const int block_n, const int block_m,
        const int active_tail_block) {
    static_assert(kBlockM == 2 && (kBlockN == 1 || kBlockN == 2));
    static_assert(kPipelineStages == 2);
    static_assert(kNumExperts > 0);
    static_assert(kMAlignment == 64 || kMAlignment == 128);
    static_assert(kN % kGroupedContiguousOutput64x128SingleWGN == 0 && kK % 128 == 0);
    if (block_n != kBlockN || block_m != kBlockM ||
        total_m <= 0 || total_m % 64 != 0 || num_experts != kNumExperts ||
        m_alignment != kMAlignment ||
        n != kN || k != kK)
        return;
    if (active_tail_block < -1 || active_tail_block >= kBlockM)
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
        kGroupedContiguousOutput64x128SingleWGM * kBlock * kBlockM * sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes = kWeightRows * kMetadataRowBytes;
    constexpr int kStageBytes =
        kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes = 2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kGroupedContiguousOutput64x128SingleWGM * kGroupedContiguousOutput64x128SingleWGN * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes + kBarrierBytes + 1023) / 1024) * 1024;
    constexpr int kBlockGroups = kK / (kBlock * kBlockM);
    const int tiles_m = total_m / kGroupedContiguousOutput64x128SingleWGM;
    constexpr int kTilesN = kN / kGroupedContiguousOutput64x128SingleWGN;
    const int total_tiles = tiles_m * kTilesN;
    constexpr bool kUsePersistentGrid =
        ((kN == 1408 || kN == 1536) && kK == 2048) ||
        (kN == 2048 && (kK == 640 || kK == 768));

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
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
    unsigned consumer_tile_phase = 0;
    int tile_end;
    int tile_stride;
    if constexpr (kUsePersistentGrid) {
        tile_end = total_tiles;
        tile_stride = static_cast<int>(gridDim.x);
    } else {
        tile_end = static_cast<int>(blockIdx.x) + 1;
        tile_stride = total_tiles;
    }
    for (int tile_idx = static_cast<int>(blockIdx.x);
         tile_idx < tile_end;
         tile_idx += tile_stride) {
        const int tile_m = tile_idx % tiles_m;
        const int tile_n = tile_idx / tiles_m;
        const int block_row_base = tile_n * kWeightRows;
        const int output_tile_m =
            tile_m * kGroupedContiguousOutput64x128SingleWGM;
        const int output_tile_n =
            tile_n * kGroupedContiguousOutput64x128SingleWGN;
        int expert_low = 0;
        int expert_high = kNumExperts - 1;
        while (expert_low < expert_high) {
            const int expert_mid = (expert_low + expert_high) / 2;
            const int aligned_end =
                ((grouped_index[expert_mid] + kMAlignment - 1) /
                 kMAlignment) * kMAlignment;
            if (output_tile_m < aligned_end)
                expert_high = expert_mid;
            else
                expert_low = expert_mid + 1;
        }
        const int expert = expert_low;
        const int remaining = grouped_index[expert] - output_tile_m;
        const int valid_rows = remaining <= 0
            ? 0
            : (remaining < 64 ? remaining : 64);

        if (valid_rows == 0) {
            if constexpr (!kUsePersistentGrid) {
                for (int index = static_cast<int>(threadIdx.x);
                     index < kGroupedContiguousOutput64x128SingleWGM *
                                 kGroupedContiguousOutput64x128SingleWGN;
                     index += 256)
                    smem_output[index] = __float2bfloat16(0.0f);
                __syncthreads();
                if (threadIdx.x == 0) {
                    cute::tma_store_fence();
#pragma unroll
                    for (int output_atom = 0; output_atom < 2; ++output_atom) {
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_output,
                            smem_output + output_atom * 64 * 64,
                            output_tile_n + output_atom * 64, output_tile_m);
                    }
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                __syncthreads();
                return;
            }
            if (warp < 4) {
                for (int index = static_cast<int>(threadIdx.x);
                     index < kGroupedContiguousOutput64x128SingleWGM *
                                 kGroupedContiguousOutput64x128SingleWGN;
                     index += kGroupedContiguousOutput64x128SingleWGMathThreads)
                    smem_output[index] = __float2bfloat16(0.0f);
                cutlass::arch::NamedBarrier::sync(
                    kGroupedContiguousOutput64x128SingleWGMathThreads, 0);
                if (warp == 0 && cute::elect_one_sync()) {
                    cute::tma_store_fence();
#pragma unroll
                    for (int output_atom = 0; output_atom < 2; ++output_atom) {
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_output,
                            smem_output + output_atom * 64 * 64,
                            output_tile_n + output_atom * 64, output_tile_m);
                    }
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                cutlass::arch::NamedBarrier::sync(
                    kGroupedContiguousOutput64x128SingleWGMathThreads, 1);
            }
            continue;
        }

        if (warp == 6) {
            const bool is_leader = cute::elect_one_sync();
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
                if constexpr (kBlockN == 1) {
                    const unsigned long long selector =
                        static_cast<unsigned long long>(block_selector[selector_index]);
                    if (is_leader)
                        stage_control[producer_stage * kWeightRows + weight_row] = selector;
                }
                const long long dense_block = selector_index * kDenseCount;
                const long long sparse_block = selector_index * kBlockN;
                if (is_leader) {
                    if constexpr (kDenseCount > 0) {
                        deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                            &tensor_map_dense, &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_dense(producer_stage) +
                                weight_row * kDenseRowBytes / sizeof(__nv_bfloat16)),
                            0, dense_block * kBlock);
                    }
#pragma unroll
                    for (int sparse_idx = 0; sparse_idx < kBlockN; ++sparse_idx) {
                        deep_gemm::tma::copy<32, 64, 64, cutlass::bfloat16_t>(
                            &tensor_map_sparse, &full_barrier[producer_stage],
                            reinterpret_cast<cutlass::bfloat16_t*>(
                                smem_sparse(producer_stage) +
                                weight_row * kSparseRowBytes /
                                    sizeof(__nv_bfloat16) +
                                sparse_idx * kBlock * (kBlock / 2)),
                            0, (sparse_block + sparse_idx) * kBlock);
                    }
                }
#pragma unroll
                for (int sparse_idx = 0; sparse_idx < kBlockN; ++sparse_idx) {
                    reinterpret_cast<uint4*>(
                        smem_metadata(producer_stage) +
                        (weight_row * kBlockN + sparse_idx) * 128)[lane] =
                        reinterpret_cast<const uint4*>(
                            hardware_metadata +
                            (sparse_block + sparse_idx) * 128)[lane];
                }
            }
            __syncwarp();
            if (is_leader) {
                deep_gemm::tma::copy<128, 64, 128, cutlass::bfloat16_t>(
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

        if (warp < 4) {
            float accumulator[kWeightRows][kGroupedContiguousOutput64x128SingleWGAccumulators] = {};
        [&]<size_t... Group>(cute::index_sequence<Group...>) {
            ([&] {
                constexpr int kGroup = static_cast<int>(Group);
                constexpr int kStage = kGroup % kPipelineStages;
                constexpr unsigned kPhase =
                    (kGroup / kPipelineStages) & 1;
                constexpr unsigned kStageByteOffset =
                    1024 + kStage * kStageBytes;
                full_barrier[kStage].wait(kPhase ^ consumer_tile_phase);

                auto issue_weight_row = [&]<int kWeightRow>() {
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
                    constexpr unsigned kSparseBlockBytes =
                        kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
                    constexpr unsigned long long kSparseDesc2 =
                        make_constexpr_gmma_desc(
                            kSparseByteOffset + kSparseBlockBytes, 512,
                            static_cast<unsigned>(cute::GMMA::LayoutType::B64));
                    constexpr unsigned long long kSparseDesc3 =
                        make_constexpr_gmma_desc(
                            kSparseByteOffset + kSparseBlockBytes + 32, 512,
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
                                local_block * kGroupedContiguousOutput64x128SingleWGM * kBlock *
                                    sizeof(__nv_bfloat16) +
                                k_byte_offset,
                            1024, static_cast<unsigned>(
                                cute::GMMA::LayoutType::B128));
                    };

                    const int active_lane =
                        (lane >> 2) * 2 + thread_in_metadata_group;
                    unsigned metadata[kBlockN][2] = {};
                    if (thread_in_metadata_group < 2) {
#pragma unroll
                        for (int sparse_idx = 0; sparse_idx < kBlockN;
                             ++sparse_idx) {
                            const int metadata_base =
                                (kWeightRow * kBlockN + sparse_idx) * 128;
                            metadata[sparse_idx][0] = smem_metadata(kStage)[
                                metadata_base + warp_in_math_group * 16 +
                                active_lane];
                            metadata[sparse_idx][1] = smem_metadata(kStage)[
                                metadata_base +
                                (4 + warp_in_math_group) * 16 + active_lane];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < kGroupedContiguousOutput64x128SingleWGAccumulators; ++i)
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[kWeightRow][i]);
                    deep_gemm::ptx::warpgroup_arrive();
                    if constexpr (kBlockN == 2) {
                        sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                            kSparseDesc0, activation_desc(0, 0),
                            accumulator[kWeightRow], metadata[0][0], kGroup != 0);
                        sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                            kSparseDesc1, activation_desc(0, 64),
                            accumulator[kWeightRow], metadata[0][1], true);
                        sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                            kSparseDesc2, activation_desc(1, 0),
                            accumulator[kWeightRow], metadata[1][0], true);
                        sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                            kSparseDesc3, activation_desc(1, 64),
                            accumulator[kWeightRow], metadata[1][1], true);
                    } else {
                        const unsigned long long selector =
                            stage_control[kStage * kWeightRows + kWeightRow];
                        if constexpr (kK == 768) {
                            const bool is_partial_k_group =
                                (active_tail_block == 0 &&
                                 kGroup == kBlockGroups - 1) ||
                                (active_tail_block == 1 && kGroup == 0);
                            if (is_partial_k_group) {
                                const bool active_block_is_sparse =
                                    ((selector >> active_tail_block) & 1ULL) != 0;
                                if (active_block_is_sparse) {
                                    sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                        kSparseDesc0, activation_desc(active_tail_block, 0),
                                        accumulator[kWeightRow], metadata[0][0], true);
                                    sparse_wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                        kSparseDesc1, activation_desc(active_tail_block, 64),
                                        accumulator[kWeightRow], metadata[0][1], true);
                                } else {
                                    DenseMMAProducerMetadataGroupStage64x64NM12DescReuseFixedShapeStage7FusedMMAGroupConstexprDescBranchGroupFullGrid::wgmma(
                                        kDenseDesc0, activation_desc(active_tail_block, 0),
                                        accumulator[kWeightRow], true);
                                    DenseMMAProducerMetadataGroupStage64x64NM12DescReuseFixedShapeStage7FusedMMAGroupConstexprDescBranchGroupFullGrid::wgmma(
                                        kDenseDesc1, activation_desc(active_tail_block, 32),
                                        accumulator[kWeightRow], true);
                                    DenseMMAProducerMetadataGroupStage64x64NM12DescReuseFixedShapeStage7FusedMMAGroupConstexprDescBranchGroupFullGrid::wgmma(
                                        kDenseDesc2, activation_desc(active_tail_block, 64),
                                        accumulator[kWeightRow], true);
                                    DenseMMAProducerMetadataGroupStage64x64NM12DescReuseFixedShapeStage7FusedMMAGroupConstexprDescBranchGroupFullGrid::wgmma(
                                        kDenseDesc3, activation_desc(active_tail_block, 96),
                                        accumulator[kWeightRow], true);
                                }
                            } else if ((selector & 1ULL) != 0) {
                                wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                    kSparseDesc0, activation_desc(0, 0), metadata[0][0],
                                    kSparseDesc1, activation_desc(0, 64), metadata[0][1],
                                    kDenseDesc0, activation_desc(1, 0),
                                    kDenseDesc1, activation_desc(1, 32),
                                    kDenseDesc2, activation_desc(1, 64),
                                    kDenseDesc3, activation_desc(1, 96),
                                    accumulator[kWeightRow], kGroup != 0);
                            } else {
                                wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                    kSparseDesc0, activation_desc(1, 0), metadata[0][0],
                                    kSparseDesc1, activation_desc(1, 64), metadata[0][1],
                                    kDenseDesc0, activation_desc(0, 0),
                                    kDenseDesc1, activation_desc(0, 32),
                                    kDenseDesc2, activation_desc(0, 64),
                                    kDenseDesc3, activation_desc(0, 96),
                                    accumulator[kWeightRow], kGroup != 0);
                            }
                        } else if ((selector & 1ULL) != 0) {
                            wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                kSparseDesc0, activation_desc(0, 0), metadata[0][0],
                                kSparseDesc1, activation_desc(0, 64), metadata[0][1],
                                kDenseDesc0, activation_desc(1, 0),
                                kDenseDesc1, activation_desc(1, 32),
                                kDenseDesc2, activation_desc(1, 64),
                                kDenseDesc3, activation_desc(1, 96),
                                accumulator[kWeightRow], kGroup != 0);
                        } else {
                            wgmma_group_stage_64x64_nm12_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
                                kSparseDesc0, activation_desc(1, 0), metadata[0][0],
                                kSparseDesc1, activation_desc(1, 64), metadata[0][1],
                                kDenseDesc0, activation_desc(0, 0),
                                kDenseDesc1, activation_desc(0, 32),
                                kDenseDesc2, activation_desc(0, 64),
                                kDenseDesc3, activation_desc(0, 96),
                                accumulator[kWeightRow], kGroup != 0);
                        }
                    }
                    deep_gemm::ptx::warpgroup_commit_batch();
                };

                issue_weight_row.template operator()<0>();
                issue_weight_row.template operator()<1>();
#pragma unroll
                for (int weight_row = 0; weight_row < kWeightRows; ++weight_row) {
#pragma unroll
                    for (int i = 0; i < kGroupedContiguousOutput64x128SingleWGAccumulators; ++i) {
                        deep_gemm::ptx::warpgroup_fence_operand(
                            accumulator[weight_row][i]);
                    }
                }
                deep_gemm::ptx::warpgroup_wait<0>();
                release_stage(&empty_barrier[kStage]);
            }(), ...);
        }(cute::make_index_sequence<kBlockGroups>{});

#pragma unroll
        for (int weight_row = 0; weight_row < kWeightRows; ++weight_row) {
#pragma unroll
            for (int atom = 0; atom < 8; ++atom) {
                const auto bf16_0 = __float22bfloat162_rn(
                    {accumulator[weight_row][atom * 4],
                     accumulator[weight_row][atom * 4 + 1]});
                const auto bf16_1 = __float22bfloat162_rn(
                    {accumulator[weight_row][atom * 4 + 2],
                     accumulator[weight_row][atom * 4 + 3]});
                const int row = lane & 7;
                const int col = warp_in_math_group * 2 + lane / 8;
                auto* smem_ptr =
                    smem_output + weight_row * 64 * 64 +
                    (atom * 8 + row) * 64 + ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<__nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            cutlass::arch::NamedBarrier::sync(
                kGroupedContiguousOutput64x128SingleWGMathThreads, 2);
            for (int index = warp_in_math_group * 32 + lane;
                 index < (64 - valid_rows) * 64; index += 128) {
                const int row = valid_rows + index / 64;
                const int col = index % 64;
                const int physical_col =
                    ((col / 8) ^ (row & 7)) * 8 + col % 8;
                smem_output[weight_row * 64 * 64 + row * 64 + physical_col] =
                    __float2bfloat16(0.0f);
            }
        }

        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kGroupedContiguousOutput64x128SingleWGMathThreads, 0);
        if (warp == 0 && cute::elect_one_sync()) {
#pragma unroll
            for (int output_atom = 0; output_atom < 2; ++output_atom) {
                cute::SM90_TMA_STORE_2D::copy(
                    &tensor_map_output,
                    smem_output + output_atom * 64 * 64,
                    output_tile_n + output_atom * 64, output_tile_m);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
            cutlass::arch::NamedBarrier::sync(kGroupedContiguousOutput64x128SingleWGMathThreads, 1);
            if constexpr ((kBlockGroups / kPipelineStages) % 2 != 0)
                consumer_tile_phase ^= 1;
        }
    }
}
