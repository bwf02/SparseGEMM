#pragma once

// Broadcast block kind per stage and reuse prebuilt GMMA descriptors.

#define HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME \
    hybrid_sparse_output128x64_stage_kind_desc_reuse
#define HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND 1
#define HYBRID_SPARSE_OUTPUT128X64_DESC_REUSE 1
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.cuh>
