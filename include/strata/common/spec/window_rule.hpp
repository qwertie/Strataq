// include/strata/common/spec/window_rule.hpp - which rule picks a round's verify window.
//
// The decisions below read what the MTP chain just produced (the cost-aware rule's `chain_rows`, the draft
// probabilities).  Header only, no device types: the paged engine's decode step of every backend calls it.
#pragma once

#include "strata/common/spec/confidence.hpp"

#include <algorithm>

namespace strata::spec {

struct WindowRuleIn {
    bool use_mtp = false;        ///< the MTP layer is loaded
    bool draft_on = false;       ///< some drafter proposes
    bool spec_cost = false;      ///< STRATA_SPEC_COST=1: the cost-aware MTP window
    bool first_window = false;   ///< the prompt's last token alone: the caller sets its own width afterwards
    int chain_rows = 1;          ///< the window the cost rule kept in the MTP chain just drafted (MTP only)
    int cap = 1;                 ///< S_mtp: the widest window of this source
    int start = 1;               ///< the width before any rule (CLI: cap; serve: cap with the MTP, 1 without)
    double spec_min_p = 0.0;     ///< --spec-min-p: stop at the first draft whose probability is below it (0: off)
    const float* dprob = nullptr;   ///< the drafts' probabilities, dprob[j] for draft j (the row j + 1 of the window)
};

/// Gets the width of a round's verify window: the cost-aware MTP rule's, else the --spec-min-p cut, else `start`.
inline int choose_window(const WindowRuleIn& in) {
    if (in.use_mtp && in.spec_cost && !in.first_window)
        return std::clamp(in.chain_rows, 1, in.cap);
    if (in.draft_on && in.spec_min_p > 0.0 && in.dprob != nullptr)
        return 1 + count_confident_drafts(in.dprob, in.cap - 1, in.spec_min_p);
    return in.start;
}

}  // namespace strata::spec
