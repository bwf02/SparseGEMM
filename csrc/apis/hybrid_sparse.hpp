#pragma once

#include <torch/python.h>

#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_grouped_naive.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_naive.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_tensorcore.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_sync.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_direct.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_stage3.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_stage4.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_stage5.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_metadata_prefetch.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_128x64.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_block128x32.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_block128x32_stage3.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_block128x32_output128x128.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_block128x64.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_wgmma_tma_block128x128.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::hybrid_sparse {

static void hybrid_block_sparse_bf16_gemm_naive(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);

    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);

    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);

    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const auto [selector_rows, selector_groups] = get_shape<2>(block_selector);
    DG_HOST_ASSERT(selector_rows == block_rows and selector_groups == block_groups);

    const auto [dense_rows, dense_groups, dense_slots, dense_h, dense_w] =
        get_shape<5>(dense_values);
    DG_HOST_ASSERT(dense_rows == block_rows and dense_groups == block_groups);
    DG_HOST_ASSERT(dense_slots == dense_count and dense_h == 64 and dense_w == 64);

    const auto [sparse_rows, sparse_groups, sparse_slots, sparse_h, sparse_w] =
        get_shape<5>(sparse_values);
    DG_HOST_ASSERT(sparse_rows == block_rows and sparse_groups == block_groups);
    DG_HOST_ASSERT(sparse_slots == block_n and sparse_h == 64 and sparse_w == 32);

    const auto [metadata_rows, metadata_groups, metadata_slots, metadata_h, metadata_w] =
        get_shape<5>(sparse_metadata);
    DG_HOST_ASSERT(metadata_rows == block_rows and metadata_groups == block_groups);
    DG_HOST_ASSERT(metadata_slots == block_n and metadata_h == 64 and metadata_w == 16);

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_naive(
        a,
        block_selector,
        dense_values,
        sparse_values,
        sparse_metadata,
        dense_partial,
        sparse_partial,
        d,
        m,
        n,
        k,
        block_n,
        block_m);
}

static void hybrid_block_sparse_bf16_gemm_tensorcore(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_tensorcore(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_sync(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_sync(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_metadata_prefetch(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_metadata_prefetch(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_producer_metadata_copy_output128x64(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output128x64_stage_kind(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output128x64_stage_kind_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output128x64_stage_kind_desc_reuse_compact(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output32x64_stage_kind(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output32x64_stage_kind_merge_k3_stage6(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output64x64_stage_kind(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output64x64_stage_kind_async_group3(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& partial,
        const torch::Tensor& tile_counters,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and partial.is_cuda());
    DG_HOST_ASSERT(tile_counters.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and partial.is_contiguous());
    DG_HOST_ASSERT(tile_counters.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(partial.get_device() == device);
    DG_HOST_ASSERT(tile_counters.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(partial.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(tile_counters.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    const int total_tiles = ((m + 63) / 64) * ((n + 63) / 64);
    DG_HOST_ASSERT(block_groups >= 2);
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));
    DG_HOST_ASSERT(get_shape<3>(partial) == std::make_tuple(2, m, n));
    DG_HOST_ASSERT(get_shape<1>(tile_counters) ==
                   std::make_tuple(total_tiles));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_splitk2_fused_reduce(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        partial, tile_counters, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64_nm12_fastpath(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64_nm12_fastpath_compact(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64_nm12_fastpath_merge2(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output48x64_nm12_fastpath_tma_metadata(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output80x64_nm12_fastpath(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output96x64_nm12_fastpath(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output96x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output88x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output80x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output128x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n == 1 and block_m == 2);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0 and k % 128 == 0);
    const int block_rows = n / 64;
    const int block_groups = k / 128;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, 1, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, 1, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_output128x64_stage_selector(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& hardware_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(hardware_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(hardware_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(hardware_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(hardware_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<6>(hardware_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 2, 4, 16));

    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5(
        a, block_selector, dense_values, sparse_values, hardware_metadata,
        d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_128x64(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 64, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 64, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_128x64(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_impl(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m,
        const int variant) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 128 == 0);
    DG_HOST_ASSERT(k % (32 * block_m) == 0);
    const int block_rows = n / 128;
    const int block_groups = k / (32 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 128, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 16));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 8));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    if (variant == 1) {
        sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_output128x128(
            a, block_selector, dense_values, sparse_values, sparse_metadata,
            dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
    } else if (variant == 2) {
        sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_stage3(
            a, block_selector, dense_values, sparse_values, sparse_metadata,
            dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
    } else {
        sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32(
            a, block_selector, dense_values, sparse_values, sparse_metadata,
            dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
    }
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& d,
        const int& block_n, const int& block_m) {
    hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_impl(
        a, block_selector, dense_values, sparse_values, sparse_metadata, d,
        block_n, block_m, 0);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_stage3(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& d,
        const int& block_n, const int& block_m) {
    hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_impl(
        a, block_selector, dense_values, sparse_values, sparse_metadata, d,
        block_n, block_m, 2);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_output128x128(
        const torch::Tensor& a, const torch::Tensor& block_selector,
        const torch::Tensor& dense_values, const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata, const torch::Tensor& d,
        const int& block_n, const int& block_m) {
    hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_impl(
        a, block_selector, dense_values, sparse_values, sparse_metadata, d,
        block_n, block_m, 1);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x64(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 128 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);
    const int block_rows = n / 128;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 128, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 32));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 16));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x64(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x128(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and d.is_contiguous());
    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const auto [m, k] = get_shape<2>(a);
    const auto [m_, n] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 128 == 0);
    DG_HOST_ASSERT(k % (128 * block_m) == 0);
    const int block_rows = n / 128;
    const int block_groups = k / (128 * block_m);
    const int dense_count = block_m - block_n;
    DG_HOST_ASSERT(get_shape<2>(block_selector) ==
                   std::make_tuple(block_rows, block_groups));
    DG_HOST_ASSERT(get_shape<5>(dense_values) ==
                   std::make_tuple(block_rows, block_groups, dense_count, 128, 128));
    DG_HOST_ASSERT(get_shape<5>(sparse_values) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 64));
    DG_HOST_ASSERT(get_shape<5>(sparse_metadata) ==
                   std::make_tuple(block_rows, block_groups, block_n, 128, 32));

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({m, n}, partial_options);
    const auto sparse_partial = torch::empty({m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x128(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        dense_partial, sparse_partial, d, m, n, k, block_n, block_m);
}

static void check_hybrid_grouped_common(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& grouped_index,
        const torch::Tensor& d,
        const int block_n,
        const int block_m,
        const int num_experts,
        const int n,
        const int k) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(block_n > 0 and block_n <= block_m and block_m <= 63);
    DG_HOST_ASSERT(num_experts > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(n % 64 == 0);
    DG_HOST_ASSERT(k % (64 * block_m) == 0);

    DG_HOST_ASSERT(a.is_cuda() and block_selector.is_cuda());
    DG_HOST_ASSERT(dense_values.is_cuda() and sparse_values.is_cuda());
    DG_HOST_ASSERT(sparse_metadata.is_cuda() and grouped_index.is_cuda());
    DG_HOST_ASSERT(d.is_cuda());
    DG_HOST_ASSERT(a.is_contiguous() and block_selector.is_contiguous());
    DG_HOST_ASSERT(dense_values.is_contiguous() and sparse_values.is_contiguous());
    DG_HOST_ASSERT(sparse_metadata.is_contiguous() and grouped_index.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    const auto device = a.get_device();
    DG_HOST_ASSERT(block_selector.get_device() == device);
    DG_HOST_ASSERT(dense_values.get_device() == device);
    DG_HOST_ASSERT(sparse_values.get_device() == device);
    DG_HOST_ASSERT(sparse_metadata.get_device() == device);
    DG_HOST_ASSERT(grouped_index.get_device() == device);
    DG_HOST_ASSERT(d.get_device() == device);

    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(dense_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sparse_values.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(block_selector.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(sparse_metadata.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(grouped_index.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);

    const int block_rows = n / 64;
    const int block_groups = k / (64 * block_m);
    const int dense_count = block_m - block_n;
    const auto [selector_experts, selector_rows, selector_groups] =
        get_shape<3>(block_selector);
    DG_HOST_ASSERT(selector_experts == num_experts);
    DG_HOST_ASSERT(selector_rows == block_rows and selector_groups == block_groups);

    const auto [dense_experts, dense_rows, dense_groups, dense_slots, dense_h, dense_w] =
        get_shape<6>(dense_values);
    DG_HOST_ASSERT(dense_experts == num_experts and dense_rows == block_rows);
    DG_HOST_ASSERT(dense_groups == block_groups and dense_slots == dense_count);
    DG_HOST_ASSERT(dense_h == 64 and dense_w == 64);

    const auto [sparse_experts, sparse_rows, sparse_groups, sparse_slots, sparse_h, sparse_w] =
        get_shape<6>(sparse_values);
    DG_HOST_ASSERT(sparse_experts == num_experts and sparse_rows == block_rows);
    DG_HOST_ASSERT(sparse_groups == block_groups and sparse_slots == block_n);
    DG_HOST_ASSERT(sparse_h == 64 and sparse_w == 32);

    const auto [metadata_experts, metadata_rows, metadata_groups,
                metadata_slots, metadata_h, metadata_w] =
        get_shape<6>(sparse_metadata);
    DG_HOST_ASSERT(metadata_experts == num_experts and metadata_rows == block_rows);
    DG_HOST_ASSERT(metadata_groups == block_groups and metadata_slots == block_n);
    DG_HOST_ASSERT(metadata_h == 64 and metadata_w == 16);

    const auto [index_experts] = get_shape<1>(grouped_index);
    DG_HOST_ASSERT(index_experts == num_experts);
}

static void hybrid_block_sparse_bf16_grouped_contiguous_naive(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& grouped_layout,
        const torch::Tensor& d,
        const int& m_alignment,
        const int& block_n,
        const int& block_m) {
    const auto [total_m, k] = get_shape<2>(a);
    const auto [total_m_, n] = get_shape<2>(d);
    const auto [num_experts, selector_rows, selector_groups] =
        get_shape<3>(block_selector);
    static_cast<void>(selector_rows);
    static_cast<void>(selector_groups);
    DG_HOST_ASSERT(total_m == total_m_ and total_m > 0);
    DG_HOST_ASSERT(m_alignment > 0);
    check_hybrid_grouped_common(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        grouped_layout, d, block_n, block_m, num_experts, n, k);

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({total_m, n}, partial_options);
    const auto sparse_partial = torch::empty({total_m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_grouped_gemm_naive(
        a,
        block_selector,
        dense_values,
        sparse_values,
        sparse_metadata,
        grouped_layout,
        dense_partial,
        sparse_partial,
        d,
        total_m,
        n,
        k,
        num_experts,
        total_m,
        m_alignment,
        block_n,
        block_m,
        HybridSparseGroupedMode::Contiguous);
}

static void hybrid_block_sparse_bf16_grouped_masked_naive(
        const torch::Tensor& a,
        const torch::Tensor& block_selector,
        const torch::Tensor& dense_values,
        const torch::Tensor& sparse_values,
        const torch::Tensor& sparse_metadata,
        const torch::Tensor& masked_m,
        const torch::Tensor& d,
        const int& block_n,
        const int& block_m) {
    const auto [num_experts, max_m, k] = get_shape<3>(a);
    const auto [num_experts_, max_m_, n] = get_shape<3>(d);
    DG_HOST_ASSERT(num_experts == num_experts_ and max_m == max_m_);
    DG_HOST_ASSERT(max_m > 0);
    check_hybrid_grouped_common(
        a, block_selector, dense_values, sparse_values, sparse_metadata,
        masked_m, d, block_n, block_m, num_experts, n, k);

    const auto partial_options = a.options().dtype(torch::kFloat);
    const auto dense_partial = torch::empty({num_experts, max_m, n}, partial_options);
    const auto sparse_partial = torch::empty({num_experts, max_m, n}, partial_options);
    sm90_hybrid_block_sparse_bf16_grouped_gemm_naive(
        a,
        block_selector,
        dense_values,
        sparse_values,
        sparse_metadata,
        masked_m,
        dense_partial,
        sparse_partial,
        d,
        num_experts * max_m,
        n,
        k,
        num_experts,
        max_m,
        1,
        block_n,
        block_m,
        HybridSparseGroupedMode::Masked);
}

static void register_apis(pybind11::module_& m) {
    m.def(
        "hybrid_block_sparse_bf16_gemm_naive",
        &hybrid_block_sparse_bf16_gemm_naive,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_tensorcore",
        &hybrid_block_sparse_bf16_gemm_tensorcore,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_sync",
        &hybrid_block_sparse_bf16_gemm_wgmma_sync,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_metadata_prefetch",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_metadata_prefetch,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("partial"),
        pybind11::arg("tile_counters"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("hardware_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_128x64",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_128x64,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_stage3",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_stage3,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_output128x128",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_output128x128,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x64",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x64,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x128",
        &hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x128,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_grouped_contiguous_naive",
        &hybrid_block_sparse_bf16_grouped_contiguous_naive,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("grouped_layout"),
        pybind11::arg("d"),
        pybind11::arg("m_alignment"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
    m.def(
        "hybrid_block_sparse_bf16_grouped_masked_naive",
        &hybrid_block_sparse_bf16_grouped_masked_naive,
        pybind11::arg("a"),
        pybind11::arg("block_selector"),
        pybind11::arg("dense_values"),
        pybind11::arg("sparse_values"),
        pybind11::arg("sparse_metadata"),
        pybind11::arg("masked_m"),
        pybind11::arg("d"),
        pybind11::arg("block_n"),
        pybind11::arg("block_m"));
}

} // namespace deep_gemm::hybrid_sparse
