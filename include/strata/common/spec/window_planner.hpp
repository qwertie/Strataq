// include/strata/common/spec/window_planner.hpp - what one verify window holds after its first row: the MTP's drafts,
// a lookup (suffix) window, or the MTP's drafts with a lookup chain after them; and what the draft policy learns from
// the window's verdict. Used by the original Qwen decode loops and the shared paged step.
#pragma once

#include "strata/common/spec/confidence.hpp"
#include "strata/common/spec/draft_policy.hpp"
#include "strata/common/spec/draft_source.hpp"
#include "strata/common/spec/gated_proposal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::spec {

/// The full verify width and the cap reserved for model proposals, in rows including the current token.
struct WindowRows {
    int total, model;
};

/// Reserves lookup capacity while retaining the original model cap; zero model_rows selects it from rows.
inline WindowRows get_window_rows(int rows, int model_rows, int suffix_draft, int64_t lookup_chain,
                                  int max_rows = 8);

/// The make-up of one verify window, chosen by WindowPlanner.
struct Window {
    int T = 1;              ///< rows before the chained lookup tokens: the fed-back token, then MTP or lookup drafts
    bool first = false;     ///< the request's first window (no drafts of its own; not timed)
    bool mtp = false;       ///< the MTP drafter is on
    bool lookup = false;    ///< rows 1 .. T - 1 are WindowPlanner::lookup_drafts, not the MTP's
    int match = 0;          ///< the context match behind the suffix drafter's last proposal (tokens)
    int chain = 0;          ///< chained lookup tokens after the T rows (WindowPlanner::chain_drafts)
    int chain_match = 0;
    int rows() const { return T + chain; }
};

/// Windows, drafts offered and drafts accepted of one kind of draft.
struct DraftTally {
    int64_t windows = 0, drafts = 0, ok = 0;
    void add(int64_t offered, int64_t accepted) { ++windows; drafts += offered; ok += accepted; }
};

/// Gets how many of `n` drafts a verify window accepted: draft i holds when it is `picks[i]`, the token the target
/// picked after the row before it, and every draft after the first that does not hold is dropped.
inline int accepted_prefix(const int32_t* drafts, const int32_t* picks, int n) {
    int a = 0;
    while (a < n && drafts[a] == picks[a]) ++a;
    return a;
}

/// Chooses the rows of each verify window of a decode loop, and keeps what the choice learns from: the draft sources
/// (the request's text) and the draft policy (window costs and acceptance rates, over every request it sees).
class WindowPlanner {
public:
    /// `rows`: the widest window (--spec); `mtp_rows`: the MTP's widest (--mtp-max-t); `suffix_draft`, `lookup_chain`
    /// and `lookup_chain_min`: the options of those names; `max_context` sizes the suffix drafter's history.
    WindowPlanner(int rows, int mtp_rows, int suffix_draft, int lookup_chain, int lookup_chain_min,
                  int64_t max_context)
        : policy(rows), source(rows), sfx(std::max(1, suffix_draft), 64, (size_t) max_context + 4096), rows_(rows),
          mtp_rows_(mtp_rows), suffix_draft_(suffix_draft), lookup_chain_(lookup_chain),
          lookup_chain_min_(lookup_chain_min), lookup_src_(sfx), sbuf_((size_t) rows), cbuf_((size_t) rows),
          probs_((size_t) rows) {
        policy.set_robust(cost_aware);
    }

    /// STRATA_SPEC_COST=1 (A/B arm): the MTP window by expected tokens per ms at this machine's measured row costs
    /// (DraftPolicy::keep_draft, draft by draft) instead of "draft probability >= --spec-min-p".
    const bool cost_aware = [] { const char* v = std::getenv("STRATA_SPEC_COST"); return v && v[0] == '1'; }();
    DraftPolicy policy;
    /// STRATA_SPEC_PROB's source planner: a copy of `policy` taken at the start of a request (the caller copies it),
    /// which then learns only committed outcomes (DraftPolicy::observe_committed), so that a window's choice of source
    /// never depends on a draw the next rounds draw again.
    DraftPolicy source;
    SuffixDrafter sfx;
    DraftTally lookup, chained;   ///< this request's lookup windows and lookup-chain tokens

    /// The MTP drafter's chain rule with cost_aware: says whether to compute draft `idx` + 1 after draft `idx`, whose
    /// probability was `p`.
    bool keep_draft(int idx, float p) {
        const bool keep = policy.keep_draft(chain_st_, idx, p, mtp_rows_);
        chain_rows_ = chain_st_.best_rows;
        return keep;
    }
    /// Starts a request whose context so far is `tokens`: the tallies restart, the draft sources see the context.
    void start(const std::vector<int64_t>& tokens) {
        lookup = chained = {};
        if (suffix_draft_ > 0 || lookup_chain_ > 0) sfx.reset();
        if (lookup_chain_ > 0)
            for (auto& src : extra_draft_sources()) src->reset();
        for (int64_t t : tokens) append((int32_t) t);
    }
    /// Shows the draft sources a token the context gained (a prompt token or a committed one).
    void append(int32_t token) {
        if (suffix_draft_ > 0 || lookup_chain_ > 0) sfx.append(token);
        if (lookup_chain_ > 0)
            for (auto& src : extra_draft_sources()) src->append(&token, 1);
    }

    /// Begins a round at context position `pos`: a window of `rows` rows, or with the MTP the rows its rule keeps of
    /// the drafts that came with probabilities `probs` (`min_p` = --spec-min-p).
    Window begin(int64_t pos, int rows, bool use_mtp, const float* probs, double min_p, bool first) {
        Window w{rows, first, use_mtp};
        policy.set_context(pos);
        if (use_mtp && cost_aware && !first) w.T = std::clamp(chain_rows_, 1, mtp_rows_);
        else if (use_mtp && min_p > 0.0)
            w.T = 1 + count_confident_drafts(probs, mtp_rows_ - 1, min_p);
        std::copy_n(probs, rows_, probs_.begin());
        return w;
    }
    /// Takes a lookup window instead where the policy expects it to pay and its first token agrees with `mtp_drafts`.
    /// `probabilistic` (the MTP's drafts are sampled, STRATA_SPEC_PROB): the choice must not depend on the draw, so
    /// `source` chooses for the MTP's widest window and the first tokens need not agree.  `room` caps a
    /// conversation's share of a joint step; 0 uses the configured window width.
    void pick_lookup(Window& w, const int32_t* mtp_drafts, bool probabilistic, int room = 0) {
        if (suffix_draft_ <= 0 || w.first) return;
        const int k = sfx.propose((room > 0 ? std::min(room, rows_) : rows_) - 1, sbuf_.data());
        w.match = sfx.last_match();
        if (k <= 0 || (w.mtp && !probabilistic && sbuf_[0] != mtp_drafts[0])) return;
        const DraftPolicy::Pick pick = probabilistic ? source.choose(mtp_rows_, k, w.match)
                                                     : policy.choose(w.T, k, w.match);
        w.lookup = pick.lookup;
        if (pick.lookup) w.T = pick.t;
    }
    /// Gets the lookup root of a gated proposal (gated_proposal.hpp) for a window of at most `room` rows: the suffix
    /// drafter's proposal, eligible where `source` would take it.  Only the context decides, never a draw.
    core::gated::RootChoice choose_gated_root(Window& w, int room) {
        core::gated::RootChoice root;
        w.match = 0;
        if (suffix_draft_ <= 0 || w.first || room <= 1) return root;
        const int k = sfx.propose(room - 1, sbuf_.data());
        w.match = sfx.last_match();
        if (k > 0) {
            const DraftPolicy::Pick pick = source.choose(std::min(mtp_rows_, room), k, w.match);
            root.eligible = pick.lookup;
            root.rescue_id = sbuf_[0];
            root.window = pick.t;
        }
        return root;
    }
    /// Appends lookup-chain tokens after the MTP's `w.T - 1` drafts that pay, keeping the window within `room` rows.
    void pick_chain(Window& w, const int32_t* mtp_drafts, int room) {
        static const bool fixed = std::getenv("STRATA_LOOKUP_CHAIN_FIXED") != nullptr;   // no policy gate
        if (lookup_chain_ <= 0 || w.first || w.lookup || w.T >= room) return;
        // the draft sources' context: its last tokens (at most 64), the MTP's pending drafts last
        const int pending = w.T - 1;
        const std::vector<int32_t>& h = sfx.history();
        ctail_.assign(h.end() - std::max(0, std::min((int) h.size(), 64 - pending)), h.end());
        ctail_.insert(ctail_.end(), mtp_drafts, mtp_drafts + pending);
        int source = -1;
        w.chain = propose_from_sources(lookup_src_, ctail_.data(), (int) ctail_.size(), pending,
                                       std::min(lookup_chain_, room - w.T), lookup_chain_min_, cbuf_.data(),
                                       &w.chain_match, &source);
        if (w.chain <= 0 || fixed) return;
        double p_mtp = 1.0;
        for (int i = 0; i < pending; ++i) p_mtp *= probs_[(size_t) i];
        w.chain = policy.chain(w.T, p_mtp, w.chain, w.chain_match);
    }
    /// After the verdict: gets how many drafts of `window` (its rows) the target's `picks` accepted, and counts the
    /// lookup drafts.
    int accept(const Window& w, const int32_t* window, const int32_t* picks) {
        const int a = accepted_prefix(window + 1, picks, w.rows() - 1);
        if (w.lookup) lookup.add(w.T - 1, a);
        if (w.chain > 0) chained.add(w.chain, std::max(0, a - (w.T - 1)));
        return a;
    }
    /// After the round (`round_ms` long, `eos`: it ended the answer): teaches the policy.
    void observe(const Window& w, int accepted, double round_ms, bool eos) {
        if (w.first) return;
        if (cost_aware && w.mtp && !w.lookup) policy.observe_probs(probs_.data(), w.T - 1, accepted);
        if (eos) return;
        if (w.chain == 0) policy.observe(w.lookup, w.T, accepted, w.match, round_ms);
        else policy.observe_chain(w.T, w.chain, accepted, w.chain_match, round_ms);
    }

    const int32_t* lookup_drafts() const { return sbuf_.data(); }
    const int32_t* chain_drafts() const { return cbuf_.data(); }
    /// The probabilities the current window was chosen from (STRATA_SPEC_COST), `rows` of them.
    const float* window_probs() const { return probs_.data(); }

private:
    int rows_, mtp_rows_, suffix_draft_, lookup_chain_, lookup_chain_min_;
    int chain_rows_ = 1;   ///< the window the cost-aware rule kept in the MTP chain just drafted
    DraftPolicy::ChainState chain_st_;
    PromptLookupSource lookup_src_;
    std::vector<int32_t> sbuf_, cbuf_, ctail_;
    std::vector<float> probs_;
};

inline WindowRows get_window_rows(int rows, int model_rows, int suffix_draft, int64_t lookup_chain, int max_rows) {
    if (suffix_draft > 0 && rows >= 2 && model_rows == 0) {
        model_rows = rows;
        rows = (int) std::min<int64_t>((int64_t) rows + 2, max_rows);
    }
    if (lookup_chain > 0 && rows >= 2) {
        if (model_rows == 0)
            model_rows = rows;
        rows = std::max(rows, (int) std::min<int64_t>((int64_t) std::max(rows, model_rows) +
                                                    std::min<int64_t>(lookup_chain, max_rows), max_rows));
    }
    return {rows, model_rows};
}

}  // namespace strata::spec
