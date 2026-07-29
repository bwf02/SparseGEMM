#pragma once

// Three-stage fixed-shape grouped kernel. The implementation remains in the
// full-grid template so pipeline depth stays a compile-time parameter.
#include <deep_gemm/impls/sm90_hybrid_sparse_grouped_masked_output64x64_nm12_fixed_full_grid.cuh>
