#pragma once

#include <torch/python.h>

#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_grouped_naive.hpp"
#include "../jit_kernels/impls/sm90_hybrid_sparse_naive.hpp"
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
