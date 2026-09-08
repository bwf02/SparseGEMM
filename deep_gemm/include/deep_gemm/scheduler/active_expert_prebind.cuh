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

template <uint32_t BLOCK_M, bool kEnabled>
struct ContiguousExpertPrebindScheduler {
    int current_iter = -1;
    const int* grouped_layout;
    uint32_t current_group_idx = 0;
    uint32_t current_group_start = 0;
    uint32_t current_group_end = 0;
    uint32_t num_n_blocks;
    uint32_t num_m_blocks = 0;
    uint32_t num_active_groups = 0;
    uint32_t num_groups = 0;
    uint32_t num_workers = 0;
    uint32_t m_alignment = 0;
    uint32_t active_group_rank = 0;
    uint32_t current_group_tile_start = 0;
    uint32_t current_group_tile_end = 0;
    uint32_t next_group_idx = 0;
    int previous_group_end = 0;
    uint32_t lane_in_group = 0;
    uint32_t lanes_for_group = 0;
    bool use_flat_schedule = false;

    CUTLASS_DEVICE bool bind_active_group(const uint32_t target_rank) {
        uint32_t rank = 0;
        int previous_end = 0;
        for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
            const uint32_t start = group_idx == 0
                ? 0
                : math::ceil_div(
                      static_cast<uint32_t>(previous_end), m_alignment) *
                      m_alignment;
            const int end = grouped_layout[group_idx];
            if (end > static_cast<int>(start)) {
                if (rank == target_rank) {
                    current_group_idx = group_idx;
                    current_group_start = start;
                    current_group_end = static_cast<uint32_t>(end);
                    const uint32_t aligned_end =
                        math::ceil_div(current_group_end, m_alignment) *
                        m_alignment;
                    num_m_blocks = math::ceil_div(
                        aligned_end - current_group_start, BLOCK_M);
                    return true;
                }
                ++rank;
            }
            previous_end = end;
        }
        return false;
    }

    CUTLASS_DEVICE bool bind_next_active_group() {
        while (next_group_idx < num_groups) {
            const uint32_t group_idx = next_group_idx++;
            const uint32_t start = group_idx == 0
                ? 0
                : math::ceil_div(
                      static_cast<uint32_t>(previous_group_end), m_alignment) *
                      m_alignment;
            const int end = grouped_layout[group_idx];
            previous_group_end = end;
            if (end <= static_cast<int>(start))
                continue;

            current_group_idx = group_idx;
            current_group_start = start;
            current_group_end = static_cast<uint32_t>(end);
            const uint32_t aligned_end =
                math::ceil_div(current_group_end, m_alignment) * m_alignment;
            num_m_blocks = math::ceil_div(
                aligned_end - current_group_start, BLOCK_M);
            current_group_tile_start = current_group_tile_end;
            current_group_tile_end += num_m_blocks * num_n_blocks;
            return true;
        }
        return false;
    }

    CUTLASS_DEVICE explicit ContiguousExpertPrebindScheduler(
            const uint32_t& num_groups, const uint32_t& num_workers,
            const uint32_t& num_n_blocks, const uint32_t& m_alignment,
            const int* grouped_layout):
            grouped_layout(grouped_layout), num_n_blocks(num_n_blocks),
            num_groups(num_groups), num_workers(num_workers),
            m_alignment(m_alignment) {
        if constexpr (!kEnabled)
            return;

        int previous_end = 0;
        for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
            const uint32_t start = group_idx == 0
                ? 0
                : math::ceil_div(
                      static_cast<uint32_t>(previous_end), m_alignment) *
                      m_alignment;
            const int end = grouped_layout[group_idx];
            num_active_groups += end > static_cast<int>(start);
            previous_end = end;
        }
        if (num_active_groups == 0)
            return;

        if (num_active_groups <= num_workers) {
            active_group_rank = blockIdx.x % num_active_groups;
            lane_in_group = blockIdx.x / num_active_groups;
            lanes_for_group = math::ceil_div(
                num_workers - active_group_rank, num_active_groups);
        } else {
            use_flat_schedule = true;
            while (current_group_tile_end <= blockIdx.x &&
                   bind_next_active_group()) {}
        }
        if (!use_flat_schedule)
            bind_active_group(active_group_rank);
    }

    CUTLASS_DEVICE bool get_next_tile(
            uint32_t& group_idx, uint32_t& output_m,
            uint32_t& valid_rows, uint32_t& n_block_idx) {
        if constexpr (!kEnabled)
            return false;
        if (num_active_groups == 0)
            return false;

        uint32_t block_idx_in_group = 0;
        if (use_flat_schedule) {
            const uint32_t flat_tile_idx =
                blockIdx.x + static_cast<uint32_t>(++current_iter) *
                    num_workers;
            while (flat_tile_idx >= current_group_tile_end) {
                if (!bind_next_active_group())
                    return false;
            }
            block_idx_in_group = flat_tile_idx - current_group_tile_start;
        } else {
            block_idx_in_group =
                lane_in_group + static_cast<uint32_t>(++current_iter) *
                    lanes_for_group;
            if (block_idx_in_group >= num_m_blocks * num_n_blocks)
                return false;
        }

        group_idx = current_group_idx;
        const uint32_t m_block_idx = block_idx_in_group / num_n_blocks;
        n_block_idx = block_idx_in_group % num_n_blocks;
        output_m = current_group_start + m_block_idx * BLOCK_M;
        const uint32_t remaining = current_group_end > output_m
            ? current_group_end - output_m
            : 0;
        valid_rows = remaining < BLOCK_M ? remaining : BLOCK_M;
        return true;
    }
};

} // namespace deep_gemm::sched
