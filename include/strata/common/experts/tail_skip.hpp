// include/strata/common/experts/tail_skip.hpp - the route tail skip (--route-tail-skip, STRATA_ROUTE_TAIL_SKIP): which
// of a window's routed experts that are missing from the GPU are left out, neither copied nor computed (changes the
// answers). Shared by Qwen3.8's verify windows (src/core/expert_source.cpp) and DeepSeek V4.1's CPU lane.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

/// The most entries (tokens times routed experts per token) a window passed to mark_tail_skips may have.
constexpr int kMaxTailSkipEntries = 256;

inline int get_pick_rank(const float* weights, int row, int k, int i);

/// Marks the routed experts of a window that the tail skip leaves out: each expert that `is_missing(e)` says is not on
/// the GPU and that no token of the window ranks among its top `rank` picks. Token t's `k` expert ids are at
/// ids[t * stride], ranked by `weights` (laid out as `ids`) when given, else by their order (best first). Sets
/// skip[t * stride + i] for each entry of such an expert, clears it for every other entry, and returns how many
/// distinct experts were marked. Ids outside [0, n_expert) are never marked; `rank` 0 marks nothing.
template <class IsMissing>
int mark_tail_skips(const int32_t* ids, const float* weights, int n_tok, int k, int stride, int32_t n_expert, int rank,
                    IsMissing&& is_missing, bool* skip) {
    int32_t kept[kMaxTailSkipEntries], marked[kMaxTailSkipEntries];   // (kept: experts some token ranks in its top)
    int n_kept = 0, n_marked = 0;
    for (int t = 0; t < n_tok; ++t)
        for (int i = 0; i < k; ++i) {
            const int at = t * stride + i;
            skip[at] = rank > 0 && get_pick_rank(weights, t * stride, k, i) >= rank;   // (for now: ranked too low)
            if (!skip[at])
                kept[n_kept++] = ids[at];
        }
    for (int t = 0; t < n_tok; ++t)
        for (int i = 0; i < k; ++i) {
            const int at = t * stride + i;
            const int32_t e = ids[at];
            if (!skip[at])
                continue;

            skip[at] = e >= 0 && e < n_expert && std::find(kept, kept + n_kept, e) == kept + n_kept && is_missing(e);
            if (skip[at] && std::find(marked, marked + n_marked, e) == marked + n_marked)
                marked[n_marked++] = e;
        }
    return n_marked;
}

/// Gets the 0-based rank of entry i among the k entries of a token that start at `row`: its position when `weights` is
/// null, else how many of them weigh more (ties go to the earlier one). Used by mark_tail_skips.
inline int get_pick_rank(const float* weights, int row, int k, int i) {
    if (weights == nullptr)
        return i;
    int rank = 0;
    for (int j = 0; j < k; ++j)
        rank += weights[row + j] > weights[row + i] || (weights[row + j] == weights[row + i] && j < i);
    return rank;
}

}  // namespace strata::core
