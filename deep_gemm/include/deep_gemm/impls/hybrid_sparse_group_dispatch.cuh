#pragma once

namespace deep_gemm::hybrid_sparse {

constexpr int group_mask_popcount(unsigned mask) {
    int count = 0;
    while (mask != 0) {
        count += mask & 1U;
        mask >>= 1;
    }
    return count;
}

template <int kMask, int kBlockM, int kLocalBlock = 0,
          typename SparseMMA, typename DenseMMA>
__device__ __forceinline__ void issue_group_pattern(
        SparseMMA& issue_sparse, DenseMMA& issue_dense,
        const bool has_accumulator) {
    if constexpr (kLocalBlock < kBlockM) {
        constexpr unsigned kLowerMask =
            (1U << kLocalBlock) - 1U;
        constexpr int kSparseSlot =
            group_mask_popcount(kMask & kLowerMask);
        constexpr bool kIsSparse =
            ((kMask >> kLocalBlock) & 1U) != 0;
        constexpr int kDenseSlot = kLocalBlock - kSparseSlot;
        const bool accumulate = has_accumulator || kLocalBlock != 0;
        if constexpr (kIsSparse)
            issue_sparse(kLocalBlock, kSparseSlot, accumulate);
        else
            issue_dense(kLocalBlock, kDenseSlot, accumulate);
        issue_group_pattern<
            kMask, kBlockM, kLocalBlock + 1>(
                issue_sparse, issue_dense, has_accumulator);
    }
}

template <int kBlockN, int kBlockM, int kCandidateMask = 0,
          typename SparseMMA, typename DenseMMA>
__device__ __forceinline__ bool dispatch_group_pattern(
        const unsigned selector, SparseMMA& issue_sparse,
        DenseMMA& issue_dense, const bool has_accumulator) {
    static_assert(0 < kBlockN && kBlockN <= kBlockM);
    static_assert(kBlockM <= 4);
    if constexpr (kCandidateMask == (1 << kBlockM)) {
        return false;
    } else {
        if constexpr (group_mask_popcount(kCandidateMask) == kBlockN) {
            if (selector == kCandidateMask) {
                issue_group_pattern<kCandidateMask, kBlockM>(
                    issue_sparse, issue_dense, has_accumulator);
                return true;
            }
        }
        return dispatch_group_pattern<
            kBlockN, kBlockM, kCandidateMask + 1>(
                selector, issue_sparse, issue_dense, has_accumulator);
    }
}

} // namespace deep_gemm::hybrid_sparse
