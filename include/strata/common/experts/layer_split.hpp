// include/strata/common/experts/layer_split.hpp - the layer split's planner: where to cut a model's layers between
// cards, so that the cards' expert pools hold most of the expert profile and a decode window takes least time
// (docs/MEMORY_MANAGER_PLAN.md, "Several GPUs", step M8).  No GPU: the model gives its per-layer sizes and the cards
// their speeds and room, so the old engine (`--layer-split auto`), strata-q38 and the unit tests share it.
//
// A placement's predicted decode window is the layers' time on their cards plus the cost of the routed experts no
// card holds (each card fills its pool from its layers' share of the profile, hottest first).  Two to four cards are
// searched exhaustively; beyond, the layers are shared in proportion to speed.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// What the planner knows of one card of a split, in the order the cards run the layers.
struct SplitCard {
    double layer_ms = 0;       ///< the predicted time one layer of a decode window takes on it
    double miss_scale = 1.0;   ///< the factor on the cost of its layers' missed experts (its link, split_rules)
};

/// One card's part of a placement: its layers and what its pool holds of the profile.
struct SplitFill {
    int64_t lb = 0, le = 0;   ///< it runs layers [lb, le)
    int64_t used = 0;         ///< the bytes of the profiled experts it holds
    int64_t held = 0;         ///< how many profiled experts it holds
    int64_t room = 0;         ///< the bytes it has for experts beside its layers' other memory
};

/// The question the planner answers.
struct SplitProblem {
    int64_t n_layers = 0;
    std::vector<SplitCard> cards;
    /// The layer of each expert of the profile, hottest first
    std::vector<int32_t> ranked_layers;
    /// Gets the bytes one expert of a layer takes in a card's pool
    std::function<int64_t(int64_t layer)> expert_bytes;
    /// Gets the bytes card `card` has for experts when it runs layers [lb, le): its memory less what those layers
    /// need besides experts (their conversation state, and their dense weights where the card loads only its own)
    std::function<int64_t(int card, int64_t lb, int64_t le)> room;
    /// Tells whether the engine can start on a placement (the old engine: its prompt path's buffers); null: always
    std::function<bool(const std::vector<SplitFill>&)> can_start;
    /// When no placement can start: true chooses one by decode time alone (SplitPlan::start_unchecked), false gives
    /// no placement (the old engine first tries a smaller K/V window, split_walk.hpp)
    bool choose_unchecked = true;
    double miss_ms = 190.0;        ///< the predicted cost of a window whose routed experts no card holds
    /// Weights the profile by the coverage curve 1 - (1 - f)^cover_b of the held share f, not by (rank + 1)^-1.2
    bool coverage_curve = false;
    double cover_b = 3.0;
    /// A placement within this many percent of the chosen one's time and at least 3 layers from it is named as a
    /// near tie when the cards differ (2 and 3 cards)
    double near_tie_pct = 4.0;
};

/// The planner's answer.
struct SplitPlan {
    std::vector<int64_t> cuts;     ///< the first layer of each later card; empty when there is no placement
    double ms = 0;                 ///< its predicted decode window
    double mass = 0;               ///< the share of the profile's routed mass the pools hold
    int64_t held = 0;              ///< the profiled experts the pools hold
    int64_t tried = 0;             ///< the placements scored
    std::vector<SplitFill> fills;  ///< each card's part of `cuts`
    /// No placement could start (SplitProblem::can_start), so `cuts` was chosen by decode time alone
    bool start_unchecked = false;
    std::vector<int64_t> near_tie;   ///< a different placement almost as fast (empty: none)
    double near_tie_ms = 0;          ///< the near tie's predicted decode window
};

/// Gets the placement with the least predicted decode window that can start.  Ties go to the placement whose slowest
/// card has the least work; a pick that leaves a card fewer than split_rules::kMinStageLayers layers is searched again
/// (on two cards first with each card's share of the layers by its room).  Says the retries on stderr after
/// `log_prefix` (for example "strata generate: layer split auto: ").
SplitPlan plan_layer_split(const SplitProblem& p, const char* log_prefix);

/// Says `plan` on stderr after `log_prefix`: its cuts, time and held share, and its near tie.  `n_ranked`: the
/// profile's experts.
void print_layer_split(const SplitPlan& plan, size_t n_ranked, const char* log_prefix);

/// Gets the cuts as text: "24,36,42".
std::string format_cuts(const std::vector<int64_t>& cuts);

/// The sizes the planner prices for each layer of a model whose cards each hold their layers' dense weights,
/// conversation state and experts (strata-q38).  Given by the model.
struct LayerSizes {
    std::vector<uint64_t> dense;          ///< per layer: the dense weights' bytes on the card that runs it
    std::vector<uint64_t> expert;         ///< per layer: one expert's bytes in a card's pool
    std::vector<uint64_t> fixed_state;    ///< per layer: a conversation's fixed state (recurrent state) bytes
    std::vector<uint64_t> token_bytes;    ///< per layer: a conversation's bytes per token (K/V)

    /// Gets what layers [lb, le) take on a card besides experts: their dense weights, and the state of
    /// `conversations` conversations of `tokens` tokens each.
    uint64_t get_range_bytes(int64_t lb, int64_t le, int64_t conversations, int64_t tokens) const;
};

/// Gets the problem of splitting a model of `sizes` over cards whose pools take `pool_bytes` each (their memory
/// settings), keeping the state of `conversations` conversations of `tokens` tokens each on the card that runs their
/// layers.  The cards' speeds and the profile are left to the caller.
SplitProblem make_paged_split_problem(const LayerSizes& sizes, const std::vector<uint64_t>& pool_bytes,
                                      int64_t conversations, int64_t tokens);

}  // namespace strata::core
