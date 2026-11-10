// include/strata/models/qwen38/conversation.hpp - one sequence on a loaded Qwen3.8 model: its prompt read in chunks and
// its verify windows (docs/MEMORY_MANAGER_PLAN.md, step M2c2).
//
// `strata`'s CLI path reads its prompt and decodes through it; `strata-q38` (step M2d) serves one conversation at a
// time with it.  The prompt path's loan of expert-cache slots, the elastic K/V and the adaptive tiers stay with the
// program, which owns those policies.
#pragma once

#include "strata/common/serve/request.hpp"
#include "strata/models/qwen38/model.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace strata::qwen38 {

/// A sequence's prompt path (`sp`, the batched prompt read) and verify windows (`ver`) over a session of a Model.
/// Both are public so that a caller can set their hooks and options before the matching init.
class Conversation {
public:
    strata::prefill::Prefill sp;
    strata::core::Verifier ver;

    /// Sets up the prompt path over session `ss` of `m`, in chunks of `m.o.prefill_chunk` tokens.  `borrow` (null:
    /// the prompt path allocates its own buffers) is `borrow_bytes` of expert-cache slots it may use meanwhile.
    bool init_prompt(Model& m, strata::core::SessionState& ss, void* borrow, uint64_t borrow_bytes, std::string& err);
    /// Sets up the verify windows over session `ss` of `m`, each at most `max_rows` tokens, with the engine's split and
    /// PCIe settings, and points the CPU pool's PCIe plan at them.
    bool init_decode(Model& m, strata::core::SessionState& ss, int max_rows, std::string& err);

    /// Reads `n` prompt tokens at positions `pos0`.. into the session (their K/V and recurrent state; no output).
    bool read_prompt(const int64_t* ids, int64_t n, int64_t pos0, std::string& err) {
        return sp.run(ids, n, pos0, err);
    }
    /// Runs one verify window: `tokens[0..T)` at positions `pos0`.., `out[t]` the pick after token t.  The caller then
    /// commits the tokens it keeps (`ver.commit`).  Fails also when the CPU expert pool failed during the window.
    /// `invariant`: the CPU experts' window-invariant arithmetic, which a probabilistic (STRATA_SPEC_PROB) decode needs
    /// so that a row's target distribution does not depend on its window (drive_pool_multi_invariant).
    bool run_window(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err, bool invariant = false);
    /// Decodes `tok` at `pos` and keeps it: `next` gets the token that follows it.
    bool decode(int32_t tok, int64_t pos, int32_t& next, std::string& err) {
        return run_window(1, &tok, pos, &next, err) && ver.commit(1, err);
    }
    /// Runs one window whose rows belong to the sessions given to `ver.init_slots`: row t is `tokens[t]` at `pos[t]` of
    /// session `sessions[t]` (Verifier::run_slot_rows), `out[t]` the pick after it.  The caller then commits
    /// (`ver.commit_slots`, `ver.commit_slot_prefixes`).  For strata-q38's step over several conversations.
    bool run_rows(const int* sessions, int S, const int32_t* tokens, const int64_t* pos, int32_t* out,
                  std::string& err);
    /// Hands the CPU expert pool's work in the windows to the layer split's stage that runs each layer (`split`
    /// outlives this object; null: one GPU).  For strata-q38 on several cards.
    void set_split_drive(SplitDrive* split) { split_ = split; }

private:
    /// Runs `window` (a verify window through the CPU expert pool) and fails also when the pool failed during it.
    bool run_through_pool(const std::function<bool()>& window, std::string& err);
    /// Gets the CPU expert pool's entry the windows call (`invariant`: run_window's), and its argument.
    strata::core::PoolMultiFn get_pool(bool invariant = false) const {
        if (split_ != nullptr)
            return invariant ? &drive_pool_split_invariant : &drive_pool_split;
        return invariant ? &drive_pool_multi_invariant : &drive_pool_multi;
    }
    void* get_pool_user() const { return split_ != nullptr ? (void*) split_ : (void*) &m_->drive; }

    Model* m_ = nullptr;   ///< set by init_decode
    SplitDrive* split_ = nullptr;
};

/// Gets the sampling of a request's verify windows from its options (greedy without a temperature; a seed of 0 is
/// taken from the clock).
strata::kernels::SamplerParams get_sampler_params(const strata::serve::Sampling& s);

}  // namespace strata::qwen38
