// include/strata/common/kernels/sampler_penalties.cuh - the penalties stage of the sampler chain on the device, shared
// by the plain draw (src/common/kernels/cuda/sampler.cu) and DeepSeek's speculative verify
// (src/models/deepseekv41/vendor/sampler.cu).
#pragma once

#include "strata/common/kernels/sampler.hpp"

namespace strata::kernels {

/// Counts how often token `v` occurs in the penalty window `h[0..n)`.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i)
        if (h[i] == v)
            ++c;
    return c;
}

/// Applies the repeat, frequency and presence penalties to a logit whose token occurs `count` times in the window,
/// transcribed from `llama_sampler_penalties_apply`.  The repeat penalty MULTIPLIES for non-positive logits and
/// DIVIDES for positive ones - dividing unconditionally is the natural reading of the source paper and it INVERTS the
/// penalty on half the vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

}  // namespace strata::kernels
