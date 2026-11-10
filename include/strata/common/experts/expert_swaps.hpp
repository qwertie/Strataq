// include/strata/common/experts/expert_swaps.hpp - the adaptive tier's choice of which experts a GPU's cache takes in
// and which it gives up, from how often decoding routed to each (plan v0.3 P6).  Header-only, so that every engine
// that compiles src/program/generate.cpp (the Metal build too) needs no other source for it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// One exchange of the adaptive tier: in `layer`, expert `in` (routed to often, not on the GPU) takes the slot of
/// expert `out` (on the GPU, routed to least), or free pages when `out` is -1; `gain` is how many more routings `in`
/// had (all of its routings when `out` is -1).
struct ExpertSwap {
    float gain;
    int32_t layer, in, out;
};

/// The decode routings that a prompt chunk adds to an expert all of its tokens route to, in the paged engine's
/// adaptive tier: the old engine's STRATA_RAM_ADAPT_PROMPT_W default (0.25) times 20.
constexpr float kPromptRoutings = 5.0f;

/// Gets what each of a prompt chunk's `n` routings, `k` per token, adds to its expert's count, so that an expert all
/// of the chunk's tokens route to gains `whole` routings and a long prompt does not drown out the decoding's.
inline float get_prompt_routing_weight(float whole, int64_t k, int64_t n) { return whole * (float) k / (float) n; }

/// Gets the exchanges an adaptive round would make, the largest gain first.  `usage` and `res` hold one entry per
/// expert of every layer (layer * n_expert + expert): its decayed routing count, and its GPU slot or a negative value
/// when it has none.  In each layer, the most-routed experts without a slot (routed at least twice, and not
/// `held_elsewhere(layer, expert)`) are paired with the least-routed ones with a slot while the first beats the second
/// by at least `min_gain`; that margin keeps two experts of about equal use from trading places back and forth.  With
/// `unpaired`, the candidates left without such a partner come too, with `out` -1: the paged engine puts them into
/// free pages, and drops them when there are none.
template <class HeldElsewhere>
std::vector<ExpertSwap> choose_expert_swaps(const std::vector<float>& usage, const std::vector<int32_t>& res,
                                            int64_t n_layers, int64_t n_expert, float min_gain,
                                            HeldElsewhere&& held_elsewhere, bool unpaired = false) {
    std::vector<ExpertSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert;
        const int32_t* r = res.data() + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] < 0) {
                if (u[e] >= 2.0f && !held_elsewhere(l, e))
                    cand.emplace_back(u[e], e);
            } else {
                vict.emplace_back(u[e], e);
            }
        }
        if (cand.empty() || (vict.empty() && !unpaired))
            continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        size_t i = 0;
        for (; i < nc && cand[i].first >= vict[i].first + min_gain; ++i)
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        for (; unpaired && i < cand.size(); ++i)
            swaps.push_back({cand[i].first, (int32_t) l, cand[i].second, -1});
    }
    std::sort(swaps.begin(), swaps.end(), [](const ExpertSwap& a, const ExpertSwap& b) { return a.gain > b.gain; });
    return swaps;
}

}  // namespace strata::core
