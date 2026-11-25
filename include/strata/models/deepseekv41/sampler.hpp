// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/models/deepseekv41/sampler.hpp - DeepSeek's sampling: the shared sampler chain
// (include/strata/common/kernels/sampler.hpp) plus speculative sampling for the NextN / DSpark drafts
// (src/models/deepseekv41/vendor/sampler.cu).
#pragma once

#include "strata/common/kernels/sampler.hpp"

namespace strata::dsv41k {

using kernels::SamplerParams;
using kernels::sample_tokens;

// SPECULATIVE SAMPLING (Leviathan et al. 2023, Chen et al. 2023) for the pipelined NextN decode - opt-in
// (DS4_DSPARK_SAMPLE=1).  The draft is DRAWN from the draft block's own distribution q (this chain on its logits,
// a Philox stream of its own) and its kept list is stored; the model's token is then the draft with probability
// min(1, p(d) / q(d)), else a draw from the residual max(0, p - q) renormalised - so it is distributed exactly as the
// plain draw from p, and a draft stands whenever the token is the draft (sum over x of min(p(x), q(x)) of the time
// instead of p(argmax q)).  Same (seed, counter): the same tokens - but not the plain draw's tokens.
struct SpecList {
    int n = 0;          // the kept list's length (1..64)
    int ids[64];        // descending, as the selection produced them
    double p[64];       // q over the kept list (sums to 1)
};

// draft rows: logits (n_tokens, n_vocab) -> out[t] drawn through this chain, q[t] its kept list.  DEVICE pointers,
// like sample_tokens; a sampled request without penalties only (greedy keeps the argmax draft).
void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out,
                  void* stream);
// verify rows: the model's logits (n_tokens, n_vocab), the draft draft[t] drawn from q[t] (same p.seed / p.counter as
// its sample_draft) -> out[t] under p.draft_rule (strata/common/kernels/draft_rule.hpp); the speculative rule's is
// distributed as sample_tokens' draw.  In speculative mode a draft absent from q[t] gets the plain draw.  With penalties
// (p.penalty_last_n > 0) `history` holds each row's tokens as in sample_tokens: p is then the penalized distribution.
void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q,
                   const int* draft, int* out, void* stream, const int* history = nullptr, int history_len = 0);

}  // namespace strata::dsv41k
