// include/strata/common/spec/window_planner.hpp - what one verify window holds after its first row: the drafter's
// drafts (the MTP's or DFlash2's), a lookup (suffix) window, or the drafts with a lookup chain after them; and what
// the draft policy learns from the window's verdict. Used by both decode loops of src/program/generate.cpp (`--serve`
// and `--spec`).
#pragma once

#include "strata/common/spec/draft_policy.hpp"
#include "strata/common/spec/draft_source.hpp"
#include "strata/common/spec/gated_proposal.hpp"
#include "strata/common/spec/window_rule.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::spec {

/// The make-up of one verify window, chosen by WindowPlanner.
struct Window {
    int T = 1;              ///< rows before the chained lookup tokens: the fed-back token, then the drafts
    bool first = false;     ///< the request's first window (no drafts of its own; not timed)
    bool mtp = false;       ///< the MTP drafter is loaded (it may only be DFlash2's fallback)
    bool drafted = false;   ///< a drafter (the MTP or DFlash2) proposed the drafts the window starts from
    bool lookup = false;    ///< rows 1 .. T - 1 are WindowPlanner::lookup_drafts, not the drafter's
    int proposed = 0;       ///< tokens in the suffix drafter's last proposal (WindowPlanner::lookup_drafts)
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
    /// Sets the MTP's widest window (--mtp-max-t) from the next round on (--draft auto: each drafter has its own).
    void set_mtp_rows(int mtp_rows) { mtp_rows_ = mtp_rows; }
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

    /// Begins a round at context position `pos` with the window `rule` keeps of the drafts (window_rule.hpp); the
    /// planner fills in the rule's cost-aware parts (`spec_cost`, `chain_rows`).  `rule.dprob` holds a value for each
    /// of the widest window's rows (the constructor's `rows`).
    Window begin(int64_t pos, WindowRuleIn rule) {
        policy.set_context(pos);
        rule.spec_cost = cost_aware;
        rule.chain_rows = chain_rows_;
        Window w;
        w.T = choose_window(rule);
        w.first = rule.first_window;
        w.mtp = rule.use_mtp;
        w.drafted = rule.draft_on;
        std::copy_n(rule.dprob, rows_, probs_.begin());
        return w;
    }
    /// Gets how many tokens the suffix drafter proposes for a lookup window of at most `look_rows` rows; they are
    /// lookup_drafts() until the next proposal, and their context match is sfx.last_match().
    int propose_lookup(int look_rows) { return sfx.propose(look_rows - 1, sbuf_.data()); }
    /// Takes a lookup window of at most `look_rows` rows instead where the policy expects it to pay and its first
    /// token agrees with the drafter's `drafts`.  `probabilistic` (the MTP's drafts are sampled, STRATA_SPEC_PROB):
    /// the choice must not depend on the draw, so `source` chooses for the MTP's widest window and the first tokens
    /// need not agree.
    void pick_lookup(Window& w, const int32_t* drafts, bool probabilistic, int look_rows) {
        if (suffix_draft_ <= 0 || w.first) return;
        w.proposed = propose_lookup(look_rows);
        w.match = sfx.last_match();
        if (w.proposed <= 0 || (w.drafted && !probabilistic && sbuf_[0] != drafts[0])) return;
        const DraftPolicy::Pick pick = probabilistic ? source.choose(mtp_rows_, w.proposed, w.match)
                                                     : policy.choose(w.T, w.proposed, w.match);
        w.lookup = pick.lookup;
        if (pick.lookup) w.T = pick.t;
    }
    /// Makes `w` the lookup window of `t` rows (of at most `look_rows`, context match `match`) that the policy chose
    /// before this round's draft, which was skipped for it (window_rule.hpp, skip_draft_for_lookup).
    void take_lookup(Window& w, int t, int match, int look_rows) {
        w.proposed = propose_lookup(look_rows);
        w.lookup = true;
        w.T = t;
        w.match = match;
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
    /// Appends lookup-chain tokens that pay after the drafter's `w.T - 1` drafts, keeping the window within `room`
    /// rows.
    void pick_chain(Window& w, const int32_t* drafts, int room) {
        static const bool fixed = std::getenv("STRATA_LOOKUP_CHAIN_FIXED") != nullptr;   // no policy gate
        if (lookup_chain_ <= 0 || w.first || w.lookup || w.T >= room) return;
        // the draft sources' context: its last tokens (at most 64), the drafter's pending drafts last
        const int pending = w.T - 1;
        const std::vector<int32_t>& h = sfx.history();
        ctail_.assign(h.end() - std::max(0, std::min((int) h.size(), 64 - pending)), h.end());
        ctail_.insert(ctail_.end(), drafts, drafts + pending);
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
        count(w, a);
        return a;
    }
    /// Adds `w`'s lookup and lookup-chain drafts to this request's tallies, given that its first `accepted` drafts
    /// held.
    void count(const Window& w, int accepted) {
        if (w.lookup) lookup.add(w.T - 1, accepted);
        if (w.chain > 0) chained.add(w.chain, std::max(0, accepted - (w.T - 1)));
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

}  // namespace strata::spec
