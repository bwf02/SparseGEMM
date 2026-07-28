#pragma once

// Remove two inactive warps while retaining separate barrier and TMA warps.

#define HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME \
    hybrid_sparse_output128x64_stage_kind_desc_reuse_compact
#define HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND 1
#define HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE 1
#define HYBRID_SPARSE_OUTPUT128X64_THREADS 320
#define HYBRID_SPARSE_OUTPUT128X64_BARRIER_WARP 8
#define HYBRID_SPARSE_OUTPUT128X64_PRODUCER_WARP 9
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.cuh>
