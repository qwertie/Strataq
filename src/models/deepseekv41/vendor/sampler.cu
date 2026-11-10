// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/models/deepseekv41/vendor/sampler.cu - speculative sampling for the NextN / DSpark drafts: sample_draft and
// sample_verify (see include/strata/models/deepseekv41/sampler.hpp).  The plain draw is the shared sample_tokens
// (src/common/kernels/cuda/sampler.cu); the draft rows go through the same chain, in llama.cpp's order:
//
//     top_k -> top_p -> min_p -> temperature -> pick
//
// (the penalties come first in that chain; these modes refuse them).  On Metal the three modes and the plain draw
// run on the host (`host_sample`).
#include "../ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "strata/models/deepseekv41/sampler.hpp"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif

#include <cuda_runtime.h>
#if defined(STRATA_METAL)
#include <strata/gpu/runtime.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::dsv41k {
namespace {

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

/// **ONE BLOCK PER ROW.**  The selection is the shared sampler's: the top_k list descending by logit, ties to the
/// lower index; `top_p`'s cut reads that order in double arithmetic, then min_p, temperature and the Philox draw.
// MODE 1 = speculative sampling's draft (sample_draft: the draw from the draft block's logits, on a Philox stream of
// its own, and its kept list in ql[t]); 2 = its verify (sample_verify: the draft draft[t] with probability
// min(1, p/q), else a draw from max(0, p - q)).
constexpr uint64_t kSpecDraftKey = 0x5bd1e9955bd1e995ull, kSpecAcceptKey = 0x9e3779b97f4a7c15ull,
                   kSpecResidKey = 0xc2b2ae3d27d4eb4full;
// one block of 1024 threads per row (the launches below): the bound keeps every mode at 64 registers a thread, so a
// block fits the 64K-register file - unbounded, sm_75 (Turing) compiled the draft mode to 68 and its launch failed
// ("too many resources requested for launch") wherever the NextN block sat on a Turing card
template <int MODE>
__global__ void __launch_bounds__(1024) sampler_kernel(const float* __restrict__ logits, int n_vocab, int n_tokens,
                                                       const SamplerParams p, int* __restrict__ out,
                                                       SpecList* __restrict__ ql, const int* __restrict__ draft) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist the kernel
    // keeps, 64.  Every row writes out[t]: a verify window reads all of them.
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    // ---- top_k: an exact RADIX SELECT, then a rank sort of the k survivors.  The order is the serial one:
    // descending by logit, ties to the lower index - i.e. descending by the composite key
    // (ordered(score) << 32) | ~index, which is distinct for every token.  So: find the k-th largest ordered score
    // T byte by byte (4 histogram passes over the vocabulary), keep every token above T and the lowest-indexed
    // ones equal to T, then rank the k of them by the composite key.  The k block-argmax rounds this replaces
    // re-scanned the taken list for every token in every round: 1.5 ms a sample at top_k 20 and 11 ms at 64 on an
    // RTX 4090 (the tail card's critical path in every sampled decode step).
    // THE COMMON CASE takes two passes: a coarse histogram of the top 11 bits finds the bin that holds the k-th
    // largest; every token in that bin or above (at most kCap) is copied to shared memory and ranked there by the
    // composite key.  A flat distribution with more than kCap tokens there takes the full radix select below.
    __shared__ int sel_ids[KMAX];
    __shared__ float sel_logit[KMAX];
    __shared__ unsigned int sh_prefix, sh_remaining, sh_ncand, sh_tie_base;
    // ONE shared pool, its parts reused phase by phase:
    //   [0, 2048)     the coarse histogram, then (once its bin is known) the candidates' scores and ids;
    //                 in the fallback, the byte histogram and the k candidates
    //   [2048, 3072)  the block scan
    constexpr int kBins = 2048, kCap = 1024;   // the coarse histogram: the ordered score's top 11 bits
    __shared__ unsigned int pool[kBins + 1024];
    unsigned int* coarse = pool;
    unsigned int* ccand_u = pool;
    int* ccand_i = (int*) pool + kCap;
    unsigned int* hist = pool;
    unsigned int* cand_u = pool + 256;
    int* cand_i = (int*) pool + 256 + KMAX;
    unsigned int* scan = pool + kBins;
    __shared__ int sh_bin;
    __shared__ unsigned int sh_ncoarse;
    // a score's ordered bits: larger float -> larger unsigned; NaN below everything (never kept before a number)
    auto ordered = [](float x) -> unsigned int {
        unsigned int b = __float_as_uint(x);
        if ((b & 0x7fffffffu) > 0x7f800000u) return 0u;
        if (b == 0x80000000u) b = 0u;   // -0 ties with +0, as the float comparison it replaces
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    auto score = [&](int v) -> float {
        return p.logit_bias ? l[v] + p.logit_bias[v] : l[v];
    };
    for (int b = threadIdx.x; b < kBins; b += blockDim.x) coarse[b] = 0u;
    if (threadIdx.x == 0) sh_ncoarse = 0u;
    __syncthreads();
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) atomicAdd(&coarse[ordered(score(v)) >> 21], 1u);
    __syncthreads();
    {   // the bin holding the k-th largest: suffix counts over the bins (2 bins a thread, a block scan)
        const int nt = (int) blockDim.x, per = (kBins + nt - 1) / nt;
        const int b_hi = kBins - 1 - (int) threadIdx.x * per;   // this thread's bins, from the top down
        unsigned int mine = 0u;
        for (int j = 0; j < per && b_hi - j >= 0; ++j) mine += coarse[b_hi - j];
        scan[threadIdx.x] = mine;
        __syncthreads();
        for (int off = 1; off < nt; off <<= 1) {
            const unsigned int add = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }
        unsigned int above = scan[threadIdx.x] - mine;   // tokens in the bins above this thread's
        if (above < (unsigned int) k && above + mine >= (unsigned int) k) {
            for (int j = 0; j < per && b_hi - j >= 0; ++j) {
                above += coarse[b_hi - j];
                if (above >= (unsigned int) k) { sh_bin = b_hi - j; sh_tie_base = above; break; }
            }
        }
        __syncthreads();
    }
    const int bin = sh_bin;
    const bool coarse_ok = sh_tie_base <= (unsigned int) kCap;   // the tokens in that bin and above fit
    if (coarse_ok) {
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            const unsigned int u = ordered(score(v));
            if ((int) (u >> 21) >= bin) {
                const unsigned int at = atomicAdd(&sh_ncoarse, 1u);
                if (at < (unsigned int) kCap) { ccand_u[at] = u; ccand_i[at] = v; }
            }
        }
        __syncthreads();
        const int nc = (int) min(sh_ncoarse, (unsigned int) kCap);
        for (int i = threadIdx.x; i < nc; i += blockDim.x) {
            const unsigned int ui = ccand_u[i];
            const int ii = ccand_i[i];
            int r = 0;
            for (int j = 0; j < nc && r < k; ++j) {
                const unsigned int uj = ccand_u[j];
                r += uj > ui || (uj == ui && ccand_i[j] < ii);
            }
            if (r < k) {
                sel_ids[r] = ii;
                sel_logit[r] = score(ii);
            }
        }
        __syncthreads();
    } else {
        if (threadIdx.x == 0) {
            sh_prefix = 0u;
            sh_remaining = (unsigned int) k;
            sh_ncand = 0u;
        }
        unsigned int mask = 0u;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0u;
            __syncthreads();
            const unsigned int prefix = sh_prefix;
            for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
                const unsigned int u = ordered(score(v));
                if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {   // the digit that holds the remaining-th largest, from the top
                unsigned int cum = 0u, rem = sh_remaining;
                int d = 255;
                for (; d > 0; --d) {
                    if (cum + hist[d] >= rem) break;
                    cum += hist[d];
                }
                sh_remaining = rem - cum;
                sh_prefix = prefix | ((unsigned int) d << shift);
            }
            mask |= 255u << shift;
            __syncthreads();
        }
        const unsigned int T = sh_prefix;      // the k-th largest ordered score
        const unsigned int need = sh_remaining; // how many tokens equal to T are kept (the lowest-indexed ones)
        // every token above T (fewer than k of them)
        for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
            const unsigned int u = ordered(score(v));
            if (u > T) {
                const unsigned int at = atomicAdd(&sh_ncand, 1u);
                if (at < (unsigned int) KMAX) { cand_u[at] = u; cand_i[at] = v; }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) sh_tie_base = sh_ncand;
        // the ties in index order: each thread counts its own contiguous range, an exclusive scan orders the ranges
        {
            const int nt = (int) blockDim.x;
            const int chunk = (n_vocab + nt - 1) / nt;
            const int lo = (int) threadIdx.x * chunk, hi = min(n_vocab, lo + chunk);
            unsigned int mine = 0u;
            for (int v = lo; v < hi; ++v) mine += ordered(score(v)) == T;
            scan[threadIdx.x] = mine;
            __syncthreads();
            for (int off = 1; off < nt; off <<= 1) {   // Hillis-Steele inclusive scan
                const unsigned int add = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0u;
                __syncthreads();
                scan[threadIdx.x] += add;
                __syncthreads();
            }
            unsigned int rank = scan[threadIdx.x] - mine;   // ties before this range
            for (int v = lo; v < hi && rank < need; ++v)
                if (ordered(score(v)) == T) {
                    const unsigned int at = sh_tie_base + rank;
                    if (at < (unsigned int) KMAX) { cand_u[at] = T; cand_i[at] = v; }
                    ++rank;
                }
        }
        __syncthreads();
        // rank sort: candidate i's place is the number of candidates with a larger composite key
        if ((int) threadIdx.x < k) {
            const unsigned int ui = cand_u[threadIdx.x];
            const int ii = cand_i[threadIdx.x];
            int r = 0;
            for (int j = 0; j < k; ++j) {
                const unsigned int uj = cand_u[j];
                r += uj > ui || (uj == ui && cand_i[j] < ii);
            }
            sel_ids[r] = ii;
            sel_logit[r] = score(ii);
        }
        __syncthreads();
    }

    // ---- top_p over the top_k list (descending as the selection produced them), then min_p, then temperature and
    // one Philox draw - llama.cpp's order (issue #53).
    // (one thread: the chain is serial and in double precision, which consumer cards run at 1/64 rate - 1024
    // threads computing it redundantly cost more than the whole selection above)
    if (threadIdx.x != 0) return;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = fmaxf(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + logf(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = fmaxf(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += exp((double) scaled(i) - (double) smx);
    double w[KMAX];   // the kept list's probabilities, as the plain draw computes them
    for (int i = 0; i < n_keep; ++i) w[i] = exp((double) scaled(i) - (double) smx) / sum;
    // the plain draw over w with a given uniform (the shared sample_tokens' pick)
    auto draw = [&](float u) -> int {
        double cum = 0.0;
        int pick = sel_ids[n_keep - 1];
        for (int i = 0; i < n_keep; ++i) {
            cum += w[i];
            if ((double) u < cum) { pick = sel_ids[i]; break; }
        }
        return pick;
    };
    if constexpr (MODE == 1) {
        SpecList* q = ql + t;
        q->n = n_keep;
        for (int i = 0; i < n_keep; ++i) { q->ids[i] = sel_ids[i]; q->p[i] = w[i]; }
        out[t] = draw(philox_uniform(p.seed ^ kSpecDraftKey, p.counter + (uint64_t) t));
    } else {
        const SpecList* q = ql + t;
        const int d = draft[t];
        const int qn = q->n < KMAX ? q->n : KMAX;
        auto qof = [&](int id) -> double {
            for (int j = 0; j < qn; ++j)
                if (q->ids[j] == id) return q->p[j];
            return 0.0;
        };
        double pd = 0.0;
        for (int i = 0; i < n_keep; ++i)
            if (sel_ids[i] == d) { pd = w[i]; break; }
        const double qd = qof(d);
        int pick = -1;
        if (qd <= 0.0) {
            pick = draw(philox_uniform(p.seed, p.counter + (uint64_t) t));   // not a draft from q: the plain draw
        } else if ((double) philox_uniform(p.seed ^ kSpecAcceptKey, p.counter + (uint64_t) t) * qd < pd) {
            pick = d;                                                        // kept: probability min(1, p(d)/q(d))
        } else {
            // rejected: a draw from max(0, p - q), renormalised - over p's kept list (p is 0 outside it)
            double r[KMAX], z = 0.0;
            for (int i = 0; i < n_keep; ++i) {
                const double x = w[i] - qof(sel_ids[i]);
                r[i] = x > 0.0 ? x : 0.0;
                z += r[i];
            }
            const float u = philox_uniform(p.seed ^ kSpecResidKey, p.counter + (uint64_t) t);
            if (z > 0.0) {
                double cum = 0.0;
                for (int i = 0; i < n_keep; ++i) {
                    if (r[i] <= 0.0) continue;
                    pick = sel_ids[i];   // the last one with residual mass, if rounding leaves cum below u
                    cum += r[i] / z;
                    if ((double) u < cum) break;
                }
            } else {
                pick = draw(u);   // p == q to the last bit: a rejection had probability 0 - never in practice
            }
        }
        out[t] = pick;
    }
}

#if defined(STRATA_METAL)
// ---- Metal: the sampled chain on the host (unified memory: the logits are read where the GPU wrote them).  The same
// function of (row, params, t) as the CUDA chain (sampler_kernel above; the shared sample_tokens), in the same order
// and the same double arithmetic: penalties -> top_k (descending by penalised logit, ties to the lower index) -> top_p
// -> min_p -> temperature -> one Philox draw; MODE 1 / 2 are the speculative draft and verify.  One row costs a
// partial sort of the vocabulary (~0.5 ms), against a ~70 ms decode step.
inline uint32_t host_umulhi(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32); }
inline float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32), c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = host_umulhi(0x9E3779B9u, c0), hi1 = host_umulhi(0xBB67AE85u, c2);
        const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
inline float host_penalised(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

// MODE 0 draw, 1 draft (kept list into ql), 2 verify (draft[t] against ql[t])
template <int MODE>
void host_sample(const float* dlogits, int n_tokens, int n_vocab, const int* dhistory, int history_len, const SamplerParams& p,
                 int* dout, SpecList* dql, const int* ddraft) {
    cudaDeviceSynchronize();   // (the GPU's logits are complete, in the shared memory)
    const float* logits = (const float*) strata::gpu::host_view((void*) dlogits);
    const int* history = dhistory ? (const int*) strata::gpu::host_view((void*) dhistory) : nullptr;
    int* out = (int*) strata::gpu::host_view(dout);
    SpecList* ql = dql ? (SpecList*) strata::gpu::host_view(dql) : nullptr;
    const int* draft = ddraft ? (const int*) strata::gpu::host_view((void*) ddraft) : nullptr;
    const float* bias = p.logit_bias ? (const float*) strata::gpu::host_view((void*) p.logit_bias) : nullptr;
    if (logits == nullptr || out == nullptr) {
        std::fprintf(stderr, "sample (host): no host view of the logits / output\n");
        std::exit(1);
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;
    std::vector<float> sc((size_t) n_vocab);
    std::vector<int> idx((size_t) n_vocab);
    for (int t = 0; t < n_tokens; ++t) {
        const float* l = logits + (size_t) t * n_vocab;
        const int* hrow = history ? history + (size_t) t * history_len : nullptr;
        int hlen = 0;
        if (hrow) {
            hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
            if (hlen < 0) hlen = 0;
            hrow += history_len - hlen;
        }
        for (int v = 0; v < n_vocab; ++v) {
            int cnt = 0;
            if (hlen > 0)
                for (int i = 0; i < hlen; ++i) cnt += hrow[i] == v;
            sc[(size_t) v] = host_penalised(bias ? l[v] + bias[v] : l[v], cnt, p);
        }
        // the ordered key: NaN below everything, -0 as +0 (the kernel's ordered())
        auto less_than = [&](int a, int b) {   // a ranks before b
            const float x = sc[(size_t) a], y = sc[(size_t) b];
            const bool nx = x != x, ny = y != y;
            if (nx != ny) return ny;
            if (nx && ny) return a < b;
            if (x != y) return x > y;
            return a < b;
        };
        for (int v = 0; v < n_vocab; ++v) idx[(size_t) v] = v;
        std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), less_than);
        std::sort(idx.begin(), idx.begin() + k, less_than);
        int sel_ids[64];
        float sel_logit[64];
        for (int i = 0; i < k; ++i) { sel_ids[i] = idx[(size_t) i]; sel_logit[i] = sc[(size_t) idx[(size_t) i]]; }
        int n_keep = k;
        float mx = sel_logit[0];
        for (int i = 1; i < k; ++i) mx = std::max(mx, sel_logit[i]);
        if (p.top_p < 1.0f) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += std::exp((double) sel_logit[i] - (double) mx);
            double cum = 0.0;
            int cut = k;
            for (int i = 0; i < k; ++i) {
                cum += std::exp((double) sel_logit[i] - (double) mx) / sum;
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
            if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
            n_keep = cut;
        }
        if (p.min_p > 0.0f) {
            const float thresh = sel_logit[0] + std::log(p.min_p);
            for (int i = 0; i < n_keep; ++i)
                if (sel_logit[i] < thresh) { n_keep = i; break; }
        }
        auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
        float smx = scaled(0);
        for (int i = 1; i < n_keep; ++i) smx = std::max(smx, scaled(i));
        double sum = 0.0;
        for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
        double w[64];
        for (int i = 0; i < n_keep; ++i) w[i] = std::exp((double) scaled(i) - (double) smx) / sum;
        auto draw = [&](float u) -> int {
            double cum = 0.0;
            int pick = sel_ids[n_keep - 1];
            for (int i = 0; i < n_keep; ++i) {
                cum += w[i];
                if ((double) u < cum) { pick = sel_ids[i]; break; }
            }
            return pick;
        };
        if (MODE == 0) {
            out[t] = draw(host_philox_uniform(p.seed, p.counter + (uint64_t) t));
        } else if (MODE == 1) {
            SpecList* q = ql + t;
            q->n = n_keep;
            for (int i = 0; i < n_keep; ++i) { q->ids[i] = sel_ids[i]; q->p[i] = w[i]; }
            out[t] = draw(host_philox_uniform(p.seed ^ kSpecDraftKey, p.counter + (uint64_t) t));
        } else {
            const SpecList* q = ql + t;
            const int d = draft[t];
            const int qn = q->n < KMAX ? q->n : KMAX;
            auto qof = [&](int id) -> double {
                for (int j = 0; j < qn; ++j)
                    if (q->ids[j] == id) return q->p[j];
                return 0.0;
            };
            double pd = 0.0;
            for (int i = 0; i < n_keep; ++i)
                if (sel_ids[i] == d) { pd = w[i]; break; }
            const double qd = qof(d);
            int pick = -1;
            if (qd <= 0.0) {
                pick = draw(host_philox_uniform(p.seed, p.counter + (uint64_t) t));
            } else if ((double) host_philox_uniform(p.seed ^ kSpecAcceptKey, p.counter + (uint64_t) t) * qd < pd) {
                pick = d;
            } else {
                double r[64], z = 0.0;
                for (int i = 0; i < n_keep; ++i) {
                    const double x = w[i] - qof(sel_ids[i]);
                    r[i] = x > 0.0 ? x : 0.0;
                    z += r[i];
                }
                const float u = host_philox_uniform(p.seed ^ kSpecResidKey, p.counter + (uint64_t) t);
                if (z > 0.0) {
                    double cum = 0.0;
                    for (int i = 0; i < n_keep; ++i) {
                        if (r[i] <= 0.0) continue;
                        pick = sel_ids[i];
                        cum += r[i] / z;
                        if ((double) u < cum) break;
                    }
                } else {
                    pick = draw(u);
                }
            }
            out[t] = pick;
        }
    }
}
#endif

void spec_check(const char* who, int n_tokens, int n_vocab, const SamplerParams& p) {
    if (p.greedy || p.temperature <= 0.0f || p.penalty_last_n > 0 || n_vocab <= 0) {
        std::fprintf(stderr, "%s: a sampled request without penalties only (greedy %d, temperature %g, penalty_last_n "
                             "%d, n_vocab %d)\n", who, (int) p.greedy, p.temperature, p.penalty_last_n, n_vocab);
        std::exit(1);
    }
    (void) n_tokens;
}
void spec_launched(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}
}  // namespace

void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out,
                  void* stream) {
    if (n_tokens <= 0) return;
    spec_check("sample_draft", n_tokens, n_vocab, p);
#if defined(STRATA_METAL)
    host_sample<1>(logits, n_tokens, n_vocab, nullptr, 0, p, out, q, nullptr);
    return;
#endif
    strata_ds4_launch(sampler_kernel<1>, (unsigned) n_tokens, 1024, 0, (cudaStream_t) stream, logits, n_vocab, n_tokens,
                      p, out, q, nullptr);
    spec_launched("sample_draft", stream);
}

void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q,
                   const int* draft, int* out, void* stream) {
    if (n_tokens <= 0) return;
    spec_check("sample_verify", n_tokens, n_vocab, p);
#if defined(STRATA_METAL)
    host_sample<2>(logits, n_tokens, n_vocab, nullptr, 0, p, out, const_cast<SpecList*>(q), draft);
    return;
#endif
    strata_ds4_launch(sampler_kernel<2>, (unsigned) n_tokens, 1024, 0, (cudaStream_t) stream, logits, n_vocab, n_tokens,
                      p, out, const_cast<SpecList*>(q), draft);
    spec_launched("sample_verify", stream);
}

#if defined(STRATA_METAL)
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void*) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (!(p.greedy || p.temperature <= 0.0f)) {
        host_sample<0>(logits, n_tokens, n_vocab, history, history_len, p, out, nullptr, nullptr);
        return;
    }
    if (::strata::dsv41::metal_host::sample_greedy(logits, n_tokens, n_vocab, history, history_len, p.penalty_last_n,
                                                   p.penalty_repeat, p.penalty_freq, p.penalty_present, p.logit_bias,
                                                   out))
        return;
    SamplerParams argmax = p;   // (sample_greedy could not run: the same chain cut to its first token)
    argmax.top_k = 1;
    argmax.top_p = 1.0f;
    argmax.min_p = 0.0f;
    argmax.temperature = 1.0f;
    host_sample<0>(logits, n_tokens, n_vocab, history, history_len, argmax, out, nullptr, nullptr);
}
#endif

}  // namespace strata::dsv41k
