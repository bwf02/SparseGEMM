#pragma once

// Broadcast one selector per block group and reuse it from registers.

#define HYBRID_SPARSE_OUTPUT128X64_KERNEL_NAME \
    hybrid_sparse_output128x64_stage_selector
#define HYBRID_SPARSE_OUTPUT128X64_STAGE_KIND 2
#include <deep_gemm/impls/sm90_hybrid_sparse_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64.cuh>
