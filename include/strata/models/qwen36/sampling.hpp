#pragma once
// The Qwen3.6 engine's repeat / frequency / presence penalties on the host, header-only so a unit test
// (src/models/qwen36/sampling_test.cpp) can run them without a GPU.
#include "strata/common/serve/request.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace strata::qwen35 {

/// Applies the penalties to the logit row `lg` for the tokens of the last `s.penalty_last_n` (all when < 0) history
/// tokens.  A row can be shorter than the vocabulary (the MTP draft head picks among the first N token ids only,
/// --draft-vocab N): a history token that is not in the row (id < 0 or >= lg.size()) takes no penalty.
inline void apply_penalties(std::vector<float>& lg, const serve::Sampling& s, const std::vector<int32_t>& hist) {
    if (s.penalty_last_n != 0 && (s.penalty_repeat != 1.f || s.penalty_freq != 0.f || s.penalty_present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.penalty_last_n < 0 || (size_t) s.penalty_last_n >= hist.size()
                                ? 0 : hist.size() - (size_t) s.penalty_last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            if (id < 0 || (size_t) id >= lg.size())
                continue;
            float& v = lg[(size_t) id];
            if (s.penalty_repeat != 1.f) v = v > 0 ? v / s.penalty_repeat : v * s.penalty_repeat;
            v -= (float) c * s.penalty_freq + s.penalty_present;
        }
    }
}

}  // namespace strata::qwen35
