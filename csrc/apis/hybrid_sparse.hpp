#pragma once

#include <torch/python.h>

#include "../jit/device_runtime.hpp"
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
}

} // namespace deep_gemm::hybrid_sparse
