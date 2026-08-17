#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::sched {

template <uint32_t BLOCK_M, uint32_t kNumGroups, uint32_t kNumWorkers, bool kEnabled>
struct ActiveExpertPrebindScheduler {
    int current_iter = -1;
    const int* grouped_layout;
    uint32_t current_group_idx = 0;
    uint32_t num_n_blocks;
    uint32_t num_m_blocks = 0;
    uint32_t num_active_groups = 0;
    uint32_t lane_in_group = 0;
    uint32_t lanes_for_group = 0;

    CUTLASS_DEVICE explicit ActiveExpertPrebindScheduler(
            const uint32_t& num_n_blocks, const int* grouped_layout):
            grouped_layout(grouped_layout), num_n_blocks(num_n_blocks) {
        DG_STATIC_ASSERT(!kEnabled || kNumGroups <= kNumWorkers,
                         "Prebinding requires num_groups <= num_workers");
        if constexpr (!kEnabled)
            return;
        for (uint32_t group_idx = 0; group_idx < kNumGroups; ++group_idx)
            num_active_groups += grouped_layout[group_idx] > 0;

        if (num_active_groups == 0)
            return;

        const auto active_group_rank = blockIdx.x % num_active_groups;
        lane_in_group = blockIdx.x / num_active_groups;
        lanes_for_group = math::ceil_div(kNumWorkers - active_group_rank, num_active_groups);

        uint32_t rank = 0;
        for (uint32_t group_idx = 0; group_idx < kNumGroups; ++group_idx) {
            if (grouped_layout[group_idx] > 0 and rank++ == active_group_rank) {
                current_group_idx = group_idx;
                break;
            }
        }
        num_m_blocks = math::ceil_div(
            static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
    }

    CUTLASS_DEVICE bool get_next_tile(
            uint32_t& group_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        if constexpr (!kEnabled)
            return false;
        if (num_active_groups == 0)
            return false;

        const auto block_idx_in_group =
            lane_in_group + static_cast<uint32_t>(++current_iter) * lanes_for_group;
        if (block_idx_in_group >= num_m_blocks * num_n_blocks)
            return false;

        group_idx = current_group_idx;
        m_block_idx = block_idx_in_group / num_n_blocks;
        n_block_idx = block_idx_in_group % num_n_blocks;
        return true;
    }
};

} // namespace deep_gemm::sched
