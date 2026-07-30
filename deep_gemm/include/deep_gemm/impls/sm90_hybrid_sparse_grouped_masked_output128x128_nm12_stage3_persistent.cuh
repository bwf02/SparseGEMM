#pragma once

// Three-stage persistent masked grouped 128x128 output tile. Two math
// warpgroups share each activation tile and own adjacent N64 weight rows.

#ifndef CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
#endif
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.cuh>

constexpr int kOutputTileMGroupedOutput128x128Stage3 = 128;
constexpr int kOutputTileNGroupedOutput128x128Stage3 = 128;
constexpr int kMathThreadsGroupedOutput128x128Stage3 = 256;
constexpr int kThreadsGroupedOutput128x128Stage3 = 384;
constexpr int kAccumulatorCountGroupedOutput128x128Stage3 = 64;

#define DG_WGMMA_ACCUMULATORS_64 \
    "{%0, %1, %2, %3, %4, %5, %6, %7, " \
    "%8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, " \
    "%24, %25, %26, %27, %28, %29, %30, %31, " \
    "%32, %33, %34, %35, %36, %37, %38, %39, " \
    "%40, %41, %42, %43, %44, %45, %46, %47, " \
    "%48, %49, %50, %51, %52, %53, %54, %55, " \
    "%56, %57, %58, %59, %60, %61, %62, %63}"

#define DG_WGMMA_OUTPUTS_64(accumulator) \
    "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3]), \
    "+f"(accumulator[4]), "+f"(accumulator[5]), "+f"(accumulator[6]), "+f"(accumulator[7]), \
    "+f"(accumulator[8]), "+f"(accumulator[9]), "+f"(accumulator[10]), "+f"(accumulator[11]), \
    "+f"(accumulator[12]), "+f"(accumulator[13]), "+f"(accumulator[14]), "+f"(accumulator[15]), \
    "+f"(accumulator[16]), "+f"(accumulator[17]), "+f"(accumulator[18]), "+f"(accumulator[19]), \
    "+f"(accumulator[20]), "+f"(accumulator[21]), "+f"(accumulator[22]), "+f"(accumulator[23]), \
    "+f"(accumulator[24]), "+f"(accumulator[25]), "+f"(accumulator[26]), "+f"(accumulator[27]), \
    "+f"(accumulator[28]), "+f"(accumulator[29]), "+f"(accumulator[30]), "+f"(accumulator[31]), \
    "+f"(accumulator[32]), "+f"(accumulator[33]), "+f"(accumulator[34]), "+f"(accumulator[35]), \
    "+f"(accumulator[36]), "+f"(accumulator[37]), "+f"(accumulator[38]), "+f"(accumulator[39]), \
    "+f"(accumulator[40]), "+f"(accumulator[41]), "+f"(accumulator[42]), "+f"(accumulator[43]), \
    "+f"(accumulator[44]), "+f"(accumulator[45]), "+f"(accumulator[46]), "+f"(accumulator[47]), \
    "+f"(accumulator[48]), "+f"(accumulator[49]), "+f"(accumulator[50]), "+f"(accumulator[51]), \
    "+f"(accumulator[52]), "+f"(accumulator[53]), "+f"(accumulator[54]), "+f"(accumulator[55]), \
    "+f"(accumulator[56]), "+f"(accumulator[57]), "+f"(accumulator[58]), "+f"(accumulator[59]), \
    "+f"(accumulator[60]), "+f"(accumulator[61]), "+f"(accumulator[62]), "+f"(accumulator[63])

__device__ __forceinline__ void wgmma_grouped_output128x128_stage3(
        const unsigned long long sparse_desc_0,
        const unsigned long long sparse_activation_desc_0,
        const unsigned metadata_0,
        const unsigned long long sparse_desc_1,
        const unsigned long long sparse_activation_desc_1,
        const unsigned metadata_1,
        const unsigned long long dense_desc_0,
        const unsigned long long dense_activation_desc_0,
        const unsigned long long dense_desc_1,
        const unsigned long long dense_activation_desc_1,
        const unsigned long long dense_desc_2,
        const unsigned long long dense_activation_desc_2,
        const unsigned long long dense_desc_3,
        const unsigned long long dense_activation_desc_3,
        float* accumulator, const bool accumulate) {
    asm volatile(
        "{\n"
        ".reg .pred p, one;\n"
        "setp.ne.b32 p, %78, 0;\n"
        "setp.ne.b32 one, 1, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %64, %65, %66, 0, p, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %67, %68, %69, 0, one, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %70, %71, one, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %72, %73, one, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %74, %75, one, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        DG_WGMMA_ACCUMULATORS_64 ", %76, %77, one, 1, 1, 0, 0;\n"
        "}\n"
        : DG_WGMMA_OUTPUTS_64(accumulator)
        : "l"(sparse_desc_0), "l"(sparse_activation_desc_0),
          "r"(metadata_0), "l"(sparse_desc_1),
          "l"(sparse_activation_desc_1), "r"(metadata_1),
          "l"(dense_desc_0), "l"(dense_activation_desc_0),
          "l"(dense_desc_1), "l"(dense_activation_desc_1),
          "l"(dense_desc_2), "l"(dense_activation_desc_2),
          "l"(dense_desc_3), "l"(dense_activation_desc_3),
          "r"(static_cast<int>(accumulate)));
}

#undef DG_WGMMA_OUTPUTS_64
#undef DG_WGMMA_ACCUMULATORS_64

using DenseMMAGroupedOutput128x128Stage3 =
    typename deep_gemm::mma::sm90::BF16MMASelector<128>::type;

template <size_t... I>
__device__ __forceinline__ void sparse_wgmma_grouped_output128x128_stage3_impl(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate,
        cute::index_sequence<I...>) {
    using SparseMMA = cute::SM90::GMMA::SPARSE::
        GMMA_64x128x32_F32BF16BF16_SS<cute::GMMA::Major::K,
                                     cute::GMMA::Major::K>;
    SparseMMA::fma(
        desc_a, desc_b, accumulator[I]..., metadata,
        accumulate ? cute::GMMA::ScaleOut::One : cute::GMMA::ScaleOut::Zero);
}

__device__ __forceinline__ void sparse_wgmma_grouped_output128x128_stage3(
        const unsigned long long desc_a, const unsigned long long desc_b,
        float* accumulator, const unsigned metadata, const bool accumulate) {
    sparse_wgmma_grouped_output128x128_stage3_impl(
        desc_a, desc_b, accumulator, metadata, accumulate,
        cute::make_index_sequence<kAccumulatorCountGroupedOutput128x128Stage3>{});
}

template <int kPipelineStages>
__device__ __forceinline__ void advance_pipeline_grouped_output128x128_stage3(
        int& stage, unsigned& phase) {
    stage = stage == kPipelineStages - 1
        ? 0
        : stage + 1;
    phase ^= stage == 0;
}

template <int kBlockN, int kBlockM, int kPipelineStages,
          int kNumExperts, int kMaxM, int kN, int kK>
__global__ __launch_bounds__(kThreadsGroupedOutput128x128Stage3, 1)
void hybrid_sparse_grouped_masked_output128x128_nm12_stage3_persistent(
        const long long* block_selector, const unsigned* hardware_metadata,
        const int* grouped_index,
        const __grid_constant__ cute::TmaDescriptor tensor_map_activation,
        const __grid_constant__ cute::TmaDescriptor tensor_map_dense,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sparse,
        const __grid_constant__ cute::TmaDescriptor tensor_map_output,
        const int num_experts, const int max_m, const int n, const int k,
        const int block_n, const int block_m) {
    static_assert(kBlockN == 1 && kBlockM == 2);
    static_assert(kPipelineStages == 3);
    static_assert(kNumExperts > 0 && kMaxM == 128);
    static_assert(kN % 128 == 0 && kK % 128 == 0);
    if (block_n != kBlockN || block_m != kBlockM ||
        num_experts != kNumExperts || max_m != kMaxM ||
        n != kN || k != kK)
        return;
    constexpr int kDenseCount = kBlockM - kBlockN;
    constexpr int kWeightRows = 2;
    constexpr int kDenseRowBytes =
        kDenseCount * kBlock * kBlock * sizeof(__nv_bfloat16);
    constexpr int kSparseRowBytes =
        kBlockN * kBlock * (kBlock / 2) * sizeof(__nv_bfloat16);
    constexpr int kMetadataRowBytes =
        kBlockN * 2 * 4 * 16 * sizeof(unsigned);
    constexpr int kDenseWeightBytes = kWeightRows * kDenseRowBytes;
    constexpr int kSparseWeightBytes = kWeightRows * kSparseRowBytes;
    constexpr int kActivationBytes =
        kOutputTileMGroupedOutput128x128Stage3 * kBlock * kBlockM *
        sizeof(__nv_bfloat16);
    constexpr int kMetadataBytes = kWeightRows * kMetadataRowBytes;
    constexpr int kStageBytes =
        kDenseWeightBytes + kSparseWeightBytes +
        kActivationBytes + kMetadataBytes;
    constexpr int kBarrierBytes =
        2 * kPipelineStages * sizeof(Barrier);
    constexpr int kOutputBytes =
        kOutputTileMGroupedOutput128x128Stage3 *
        kOutputTileNGroupedOutput128x128Stage3 * sizeof(__nv_bfloat16);
    constexpr int kOutputOffset =
        ((kPipelineStages * kStageBytes +
          kBarrierBytes + 1023) / 1024) * 1024;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int math_group = warp >> 2;
    const int warp_in_math_wg = warp & 3;
    const int thread_in_metadata_group = lane & 3;
    constexpr int block_groups = kK / (kBlock * kBlockM);
    constexpr int tiles_per_expert =
        kMaxM /
        kOutputTileMGroupedOutput128x128Stage3;
    constexpr int tiles_m = kNumExperts * tiles_per_expert;
    constexpr int tiles_n =
        (kN + kOutputTileNGroupedOutput128x128Stage3 - 1) /
        kOutputTileNGroupedOutput128x128Stage3;
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

    if (warp == 8 && lane == 0) {
#pragma unroll
        for (int stage = 0;
             stage < kPipelineStages; ++stage) {
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

    int producer_stage = 0;
    unsigned producer_phase = 0;
    int consumer_stage = 0;
    unsigned consumer_phase = 0;
    for (int tile_idx = static_cast<int>(blockIdx.x);
         tile_idx < total_tiles;
         tile_idx += static_cast<int>(gridDim.x)) {
        const int tile_m = tile_idx % tiles_m;
        const int tile_n = tile_idx / tiles_m;
        const int expert = tile_m / tiles_per_expert;
        const int local_tile_m = tile_m % tiles_per_expert;
        const int local_m =
            local_tile_m *
            kOutputTileMGroupedOutput128x128Stage3;
        const int output_tile_m =
            expert * kMaxM + local_m;
        const int output_tile_n =
            tile_n * kOutputTileNGroupedOutput128x128Stage3;
        const int block_row_base = tile_n * kWeightRows;
        const int remaining = grouped_index[expert] - local_m;
        const int valid_rows = remaining <= 0
            ? 0
            : (remaining < 128 ? remaining : 128);

        if (warp == 10) {
            const bool is_leader = cute::elect_one_sync();
#pragma unroll
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                if (is_leader)
                    empty_barrier[producer_stage].wait(
                        producer_phase ^ 1);
                __syncwarp();
#pragma unroll
                for (int weight_row = 0; weight_row < kWeightRows;
                     ++weight_row) {
                    const long long selector_index =
                        (static_cast<long long>(expert) * (kN / kBlock) +
                         block_row_base + weight_row) *
                            block_groups +
                        block_group;
                    const unsigned long long selector =
                        static_cast<unsigned long long>(
                            block_selector[selector_index]);
                    if (is_leader)
                        stage_control[
                            producer_stage * kWeightRows + weight_row] =
                            selector;
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
                                    weight_row * kDenseRowBytes /
                                        sizeof(__nv_bfloat16) +
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
                                    weight_row * kSparseRowBytes /
                                        sizeof(__nv_bfloat16) +
                                    sparse_slot * kBlock * (kBlock / 2)),
                                0, packed_block * kBlock);
                        }
                        reinterpret_cast<uint4*>(
                            smem_metadata(producer_stage) +
                            weight_row * kMetadataRowBytes /
                                sizeof(unsigned) +
                            sparse_slot * 128)[lane] =
                            reinterpret_cast<const uint4*>(
                                hardware_metadata +
                                static_cast<long long>(packed_block) *
                                    128)[lane];
                    }
                }
                __syncwarp();
                if (is_leader) {
                    deep_gemm::tma::copy<
                        kBlock * kBlockM,
                        kOutputTileMGroupedOutput128x128Stage3, 128,
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
                advance_pipeline_grouped_output128x128_stage3<kPipelineStages>(
                    producer_stage, producer_phase);
            }
        }

        if (warp < 8) {
            const int weight_row = math_group;
            float accumulator[kAccumulatorCountGroupedOutput128x128Stage3] = {};
            bool has_accumulator = false;
            int pending_stage = -1;
#pragma unroll
            for (int block_group = 0; block_group < block_groups;
                 ++block_group) {
                full_barrier[consumer_stage].wait(consumer_phase);
                const unsigned long long selector =
                    stage_control[
                        consumer_stage * kWeightRows + weight_row];
                const unsigned stage_desc_offset =
                    consumer_stage * (kStageBytes / 16);
                const unsigned dense_row_desc_offset =
                    weight_row * (kDenseRowBytes / 16);
                const unsigned sparse_row_desc_offset =
                    weight_row * (kSparseRowBytes / 16);
#pragma unroll
                for (int i = 0;
                     i < kAccumulatorCountGroupedOutput128x128Stage3; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                deep_gemm::ptx::warpgroup_arrive();
                if constexpr (kBlockN == 1 && kBlockM == 2) {
                    const int sparse_local_block =
                        (selector & 1ULL) != 0 ? 0 : 1;
                    const int dense_local_block = 1 - sparse_local_block;
                    const int active_lane =
                        (lane >> 2) * 2 + thread_in_metadata_group;
                    unsigned metadata_0 = 0;
                    unsigned metadata_1 = 0;
                    if (thread_in_metadata_group < 2) {
                        metadata_0 = smem_metadata(consumer_stage)[
                            weight_row * kMetadataRowBytes /
                                    sizeof(unsigned) +
                            warp_in_math_wg * 16 + active_lane];
                        metadata_1 = smem_metadata(consumer_stage)[
                            weight_row * kMetadataRowBytes /
                                    sizeof(unsigned) +
                            (4 + warp_in_math_wg) * 16 + active_lane];
                    }
                    auto sparse_desc_0 = sparse_desc;
                    auto sparse_desc_1 = sparse_desc;
                    auto sparse_activation_desc_0 = activation_desc;
                    auto sparse_activation_desc_1 = activation_desc;
                    sparse_desc_0.reg32_[0] =
                        sparse_desc_base_lo + stage_desc_offset +
                        sparse_row_desc_offset;
                    sparse_desc_1.reg32_[0] =
                        sparse_desc_base_lo + stage_desc_offset +
                        sparse_row_desc_offset + 2;
                    const unsigned sparse_activation_base =
                        activation_desc_base_lo + stage_desc_offset +
                        sparse_local_block *
                            kOutputTileMGroupedOutput128x128Stage3 *
                            kBlock / 8;
                    sparse_activation_desc_0.reg32_[0] =
                        sparse_activation_base;
                    sparse_activation_desc_1.reg32_[0] =
                        sparse_activation_base + 4;

                    auto dense_desc_0 = dense_desc;
                    auto dense_desc_1 = dense_desc;
                    auto dense_desc_2 = dense_desc;
                    auto dense_desc_3 = dense_desc;
                    auto dense_activation_desc_0 = activation_desc;
                    auto dense_activation_desc_1 = activation_desc;
                    auto dense_activation_desc_2 = activation_desc;
                    auto dense_activation_desc_3 = activation_desc;
                    const unsigned dense_activation_base =
                        activation_desc_base_lo + stage_desc_offset +
                        dense_local_block *
                            kOutputTileMGroupedOutput128x128Stage3 *
                            kBlock / 8;
                    dense_desc_0.reg32_[0] =
                        dense_desc_base_lo + stage_desc_offset +
                        dense_row_desc_offset;
                    dense_desc_1.reg32_[0] =
                        dense_desc_base_lo + stage_desc_offset +
                        dense_row_desc_offset + 2;
                    dense_desc_2.reg32_[0] =
                        dense_desc_base_lo + stage_desc_offset +
                        dense_row_desc_offset + 4;
                    dense_desc_3.reg32_[0] =
                        dense_desc_base_lo + stage_desc_offset +
                        dense_row_desc_offset + 6;
                    dense_activation_desc_0.reg32_[0] =
                        dense_activation_base;
                    dense_activation_desc_1.reg32_[0] =
                        dense_activation_base + 2;
                    dense_activation_desc_2.reg32_[0] =
                        dense_activation_base + 4;
                    dense_activation_desc_3.reg32_[0] =
                        dense_activation_base + 6;
                    wgmma_grouped_output128x128_stage3(
                        sparse_desc_0.desc_, sparse_activation_desc_0.desc_,
                        metadata_0, sparse_desc_1.desc_,
                        sparse_activation_desc_1.desc_, metadata_1,
                        dense_desc_0.desc_, dense_activation_desc_0.desc_,
                        dense_desc_1.desc_, dense_activation_desc_1.desc_,
                        dense_desc_2.desc_, dense_activation_desc_2.desc_,
                        dense_desc_3.desc_, dense_activation_desc_3.desc_,
                        accumulator, has_accumulator);
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
                                            kOutputTileMGroupedOutput128x128Stage3 *
                                            kBlock +
                                        k_tile * 32,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            sparse_wgmma_grouped_output128x128_stage3(
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
                                            kOutputTileMGroupedOutput128x128Stage3 *
                                            kBlock +
                                        k_tile * 16,
                                    static_cast<int>(
                                        cute::GMMA::LayoutType::B128),
                                    0, 1024);
                            DenseMMAGroupedOutput128x128Stage3::wgmma(
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
                     i < kAccumulatorCountGroupedOutput128x128Stage3; ++i)
                    deep_gemm::ptx::warpgroup_fence_operand(
                        accumulator[i]);
                if (pending_stage >= 0) {
                    deep_gemm::ptx::warpgroup_wait<1>();
                    release_stage(&empty_barrier[pending_stage]);
                }
                pending_stage = consumer_stage;
                has_accumulator = true;
                advance_pipeline_grouped_output128x128_stage3<kPipelineStages>(
                    consumer_stage, consumer_phase);
            }
            deep_gemm::ptx::warpgroup_wait<0>();
            release_stage(&empty_barrier[pending_stage]);

#pragma unroll
            for (int atom = 0; atom < 16; ++atom) {
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
                    weight_row *
                        kOutputTileMGroupedOutput128x128Stage3 * kBlock +
                    (atom * 8 + row) * kBlock +
                    ((col ^ row) * 8);
                deep_gemm::ptx::SM90_U32x2_STSM_T<
                    __nv_bfloat162>::copy(
                    bf16_0, bf16_1, smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedOutput128x128Stage3, 0);
            for (int index = warp_in_math_wg * 32 + lane;
                 index < (128 - valid_rows) * 64;
                 index += 128) {
                const int row = valid_rows + index / 64;
                const int col = index % 64;
                const int physical_col =
                    ((col / 8) ^ (row & 7)) * 8 + col % 8;
                smem_output[
                    weight_row *
                            kOutputTileMGroupedOutput128x128Stage3 * kBlock +
                    row * 64 + physical_col] =
                    __float2bfloat16(0.0f);
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedOutput128x128Stage3, 0);
            if (warp == 0 && cute::elect_one_sync()) {
#pragma unroll
                for (int output_panel = 0; output_panel < kWeightRows;
                     ++output_panel) {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_output,
                        smem_output +
                            output_panel *
                                kOutputTileMGroupedOutput128x128Stage3 *
                                kBlock,
                        output_tile_n + output_panel * kBlock,
                        output_tile_m);
                }
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            cutlass::arch::NamedBarrier::sync(
                kMathThreadsGroupedOutput128x128Stage3, 1);
        }
    }
}
