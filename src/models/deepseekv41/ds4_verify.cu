// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_verify.cu - a window of up to kWin tokens through every layer (forward, run_dev, run_attention, run_moe), the
// head's logits, the DSpark drafter's taps, the check of a window of drafts (verify_rows, rollback) and the paged
// engine's calls for a step (step_conversations, prepare_group, launch_group, poll_group, finish_group, the joint
// rows).
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

// ---- a draft's check
bool Model::Impl::verify_rows(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k,
                              float min_p, uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err) {
    const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
    const int T = n + 1;
    for (int il = 0; il < g.nl; ++il)
        for (int m = 0; m < L[(size_t) il].nm; ++m)
            if (!init_verify_layer(member(L[(size_t) il], m), err))
                return false;
    const int64_t p0 = pos;
    int tk[kWin];
    tk[0] = y;
    for (int i = 0; i < n; ++i) tk[i + 1] = drafts[(size_t) i];
    spec_win = true;
    const double tv = now_ms();
    const bool ok = forward(tk, T, T, err, 0);   // (a window streams nothing - the lane is a decoded token's)
    st.verify_ms += now_ms() - tv;
    spec_win = false;
    if (!ok) return false;
    verified_p0 = p0;
    verified_rows = T;
    return pick_verified_rows(0, p0, drafts, temperature, top_p, top_k, min_p, seed, counter, picks, err);
}

bool Model::Impl::pick_verified_rows(int row0, int64_t p0, const std::vector<int>& drafts, float temperature,
                                     float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                                     std::vector<int>& picks, std::string& err) {
    const int n = (int) std::min<size_t>(drafts.size(), (size_t) kWin - 1);
    const int T = n + 1, V = g.vocab;
    Dev& hd = devs[(size_t) head_dev];
    cudaSetDevice(hd.id);
    if ((!d_ints && !dalloc(hd, d_ints, 64, err)) || (!d_specq && !dalloc(hd, d_specq, kWin, err)))
        return false;
    int m = 0;
    strata::dsv41k::SamplerParams sp;
    sp.greedy = temperature <= 0.0f;
    sp.temperature = temperature;
    sp.top_p = top_p;
    sp.top_k = top_k;
    sp.min_p = min_p;
    sp.seed = seed;
    sp.counter = counter;
    sp.draft_rule = draft_rule;
    // (penalties: row t counts the tokens up to its own position - y and the drafts before it, toks[0 .. p0 + 1 + t))
    if (sp.greedy || draft_rule.kind == strata::kernels::DraftRule::Exact) {   // each row's plain pick
        int h = 0;
        const int* hr = pen_rows(p0 + 1, T, sp, h);
        strata::dsv41k::sample_tokens(hd.logits + (size_t) row0 * V, T, V, hr, h, sp, d_ints + 16, hd.st);
        CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
        picks.assign(h_idx, h_idx + m + 1);
    } else {
        // row i keeps draft i or writes another token under the draft rule (draft_rule.hpp; speculative: with
        // probability min(1, p(d) / q(d)), else a draw from the residual max(0, p - q), each row a token
        // distributed as the model's own draw at that position), p through the request's chain - the top_k / top_p
        // / min_p cut included (the drafter's argmax can lie outside it: p(d) 0, never kept); past the first
        // rejection the rows do not count
        const bool own = drafts_q && dq_temp == temperature && dq_top_p == top_p && dq_top_k == top_k &&
                         dq_min_p == min_p && dq_seed == seed && dq_counter == counter;
        if (!own) {   // the drafter's argmax: q a point at each draft
            for (int i = 0; i < n; ++i) {
                h_specq[(size_t) i].n = 1;
                h_specq[(size_t) i].ids[0] = drafts[(size_t) i];
                h_specq[(size_t) i].p[0] = 1.0;
            }
            CK(cudaMemcpyAsync(d_specq, h_specq.data(), (size_t) n * sizeof(strata::dsv41k::SpecList),
                               cudaMemcpyHostToDevice, hd.st));
        }
        for (int i = 0; i < n; ++i) h_idx[i] = drafts[(size_t) i];
        CK(cudaMemcpyAsync(d_ints + 32, h_idx, (size_t) n * 4, cudaMemcpyHostToDevice, hd.st));
        int h = 0;
        const int* hr = pen_rows(p0 + 1, n, sp, h);
        strata::dsv41k::sample_verify(hd.logits + (size_t) row0 * V, n, V, sp, d_specq,
                                     d_ints + 32, d_ints + 16, hd.st, hr, h);
        CK(cudaMemcpyAsync(h_idx, d_ints + 16, (size_t) n * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        while (m < n && h_idx[m] == drafts[(size_t) m]) ++m;
        picks.assign(h_idx, h_idx + std::min(m + 1, n));
        if (m == n)
            picks.push_back(sample_row(row0 + n, temperature, top_p, top_k, min_p, seed,
                                       counter + (uint64_t) n, nullptr, p0 + 1 + n));
    }
    drafts_q = false;
    return true;
}

bool Model::Impl::keep_window(int64_t p0, int count, int keep, int row0, std::string& err) {
    const int n = count - 1, m = keep - 1;
    rollback(p0, count, keep, row0);
    if (!seed_drafter(p0, keep, std::max(row0, 0), err)) return false;
    st.drafts_offered += n;
    st.drafts_accepted += m;
    for (int i = 0; i < n && i <= m && i < 8; ++i) {
        ++st.draft_reached[i];
        if (i < m) ++st.draft_kept[i];
    }
    return true;
}

bool Model::Impl::init_verify_layer(Layer& l, std::string& err, bool joint) {
    Dev& d = devs[(size_t) l.dev];
    const size_t first = d.allocs.size(), before = d.scratch;
    if (!l.ring_bak && !dalloc(d, l.ring_bak, (size_t) kWin * win_rb(), err))
        return false;
    if (l.kv_src && l.ratio > 1 &&
        ((!l.c_snap && !dalloc(d, l.c_snap, (size_t) 2 * l.ratio * g.hd, err)) ||
         (!l.c_rows && !dalloc(d, l.c_rows, (size_t) 2 * kWin * g.hd, err))))
        return false;
    if (joint && l.c_snap && !l.joint_c_snap &&
        !dalloc(d, l.joint_c_snap, (size_t) 8 * 2 * l.ratio * g.hd, err))
        return false;
    l.dense_allocations.insert(l.dense_allocations.end(), d.allocs.begin() + first, d.allocs.end());
    l.dense_scratch_bytes += d.scratch - before;
    return true;
}

void Model::Impl::rollback(int64_t p0, int T, int keep, int row0) {
    if (keep >= T) return;
    // the ring rows: one launch a device for all its layers (a launch a layer and member was ~300 on 8-way tensor
    // parallel, a millisecond of the host's before the next draft)
    std::vector<RingBatch> rb(devs.size());
    for (int il = 0; il < g.nl; ++il)
        for (int m = 0; m < L[(size_t) il].nm; ++m) {
            Layer& l = member(L[(size_t) il], m);
            RingBatch& b = rb[(size_t) l.dev];
            if (b.n == kMaxRingBatch) {   // (more layers on a device than a batch takes: this one alone)
                cudaSetDevice(devs[(size_t) l.dev].id);
                ring_restore(l.win, l.ring_bak + (size_t) std::max(row0, 0) * win_rb(), p0, g.win, keep, T,
                             g.hd, devs[(size_t) l.dev].st);
                continue;
            }
            b.ring[b.n] = l.win;
            b.bak[b.n++] = l.ring_bak + (size_t) std::max(row0, 0) * win_rb();
        }
    for (size_t di = 0; di < devs.size(); ++di)
        if (rb[di].n > 0) {
            cudaSetDevice(devs[di].id);
            ring_restore_batch(rb[di], p0, g.win, keep, T, g.hd, devs[di].st);
        }
    for (int il = 0; il < g.nl; ++il)
        for (int m = 0; m < L[(size_t) il].nm; ++m) {
            Layer& l = member(L[(size_t) il], m);
            Dev& d = devs[(size_t) l.dev];
            if (l.c_snap != nullptr) {
                cudaSetDevice(d.id);
                const size_t ns = (size_t) l.ratio * g.hd * 4;
                const float* snap = row0 < 0 ? l.c_snap : l.joint_c_snap + (size_t) row0 * 2 * l.ratio * g.hd;
                const float* rows = l.c_rows + (size_t) std::max(row0, 0) * g.hd;
                cudaMemcpyAsync(l.c_kv, snap, ns, cudaMemcpyDeviceToDevice, d.st);
                cudaMemcpyAsync(l.c_score, snap + (size_t) l.ratio * g.hd, ns, cudaMemcpyDeviceToDevice, d.st);
                compress_win(rows, rows + (size_t) kWin * g.hd, l.c_kv, l.c_score, l.ratio, (int) p0, keep,
                             g.hd, l.c_norm, g.eps, d.latent, d.st);
            }
        }
    pos = p0 + keep;
    toks.resize((size_t) pos);
    ctoks.resize((size_t) pos);
}

// ---- a window
void Model::Impl::plan_window(int64_t p0, int T, WinPlan& wp, bool host_routed) {
    wp.par = !host_routed && par;
    wp.async = !host_routed && (par || (!sync_decode && T <= async_max_t));
    wp.vl = 0;
    for (int t = 0; t < T && !imgs.empty(); ++t)
        if (in_image(p0 + t)) wp.vl |= 1u << t;
    for (int il = 0; il < g.nl; ++il) {
        const Layer& l = L[(size_t) il];
        if (l.ratio > 0) {
            const int R = l.ratio;
            int n[kWin], k[kWin], clen_max = 0;
            for (int t = 0; t < T; ++t) {
                n[t] = (int) ((p0 + t + 1) / R);
                k[t] = std::min(g.ix_topk, n[t]);
                clen_max = std::max(clen_max, n[t]);
            }
            if (l.ix_src) {
                for (int t = 0; t < T; ++t) wp.cand[il][t] = cur_cand_n[t];
                if (clen_max > 0 && l.cand_src)
                    for (int t = 0; t < T; ++t) cur_cand_n[t] = std::min(g.cand_topk, (n[t] + g.cand_blk - 1) / g.cand_blk);
                for (int t = 0; t < T; ++t) cur_topk_n[t] = k[t];
            }
            for (int t = 0; t < T; ++t) wp.topk[il][t] = cur_topk_n[t];
        }
        if (wp.async) {
            wp.seq[il] = ++seq_issued;
            layer_of[wp.seq[il] % kRing].store(il, std::memory_order_release);
        }
    }
}

bool Model::Impl::compute_logits(Dev& d, const float* xn, int rows, std::string& err) {
    const int V = g.vocab, D = g.dim;
    cudaSetDevice(d.id);
    // (an EXL3 head: as many activations at a time as its split-K partials fit the scratch - 8 for a shared head, 4
    // for a whole one, each pass reading the whole share; a GGUF's: 8, so that every window the engine can build (rows
    // <= kWin = 8) walks the head once instead of twice - measured on the drafter's fixed T=5: 4.55 -> 2.83 ms, its
    // round 14.49 -> 13.46; the verify window's head 3.48 -> 2.68 ms, the same logits.  DS4_HEAD_PER overrides.)
    const int per = d.head.fmt != 0 ? (std::getenv("DS4_HEAD_PER") ? std::atoi(std::getenv("DS4_HEAD_PER")) :
                                                                               strata::ds4::head_per_default(defaults_026())
                                                                               )
                                    : std::max(1, x3::mv_cols(d.head));
    for (int c = 0; c < rows; c += per)
        if (!mv(d, {job(d.head, xn + (size_t) c * D, d.logits + (size_t) c * V, std::min(per, rows - c), D, V)},
                err))
            return false;
    return true;
}

bool Model::Impl::head_rows_dev(int di, int T, std::string& err) {
    Dev& d = devs[(size_t) di];
    cudaSetDevice(d.id);
    const int D = g.dim, H = g.hc;
    if (n_logits <= 0) return true;
    const int t0 = T - n_logits;
    hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
    return compute_logits(d, d.x, n_logits, err);
}

bool Model::Impl::tap(Dev& d, int il, int T, std::string& err) {
    const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
    if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
    const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
    const int D = g.dim;
    cudaSetDevice(d.id);
    hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
    CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
    CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                         cudaMemcpyDeviceToHost, d.st));
    return true;
}

bool Model::Impl::forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows) {
    if (T < 1 || T > kWin) {
        err = "a window of " + std::to_string(T) + " tokens";
        return false;
    }
    if (pos + T > o.max_context) {
        err = "the context is full";
        return false;
    }
    for (int t = 0; t < T; ++t)
        if (tk[t] < 0 || tk[t] >= g.vocab) {
            err = "token " + std::to_string(tk[t]) + " out of range";
            return false;
        }
    const int64_t p0 = pos;
    for (int t = 0; t < T; ++t) {
        toks.push_back(tk[t]);
        ctoks.push_back(ctok(pos + t, tk[t]));
    }
    // the Engram rows asked for now (the first Engram layer's first), collected right before their layer: what is
    // left of their reads by then is the window's Engram time
    const double te = now_ms();
    const int64_t eg_w0 = eg_wait_ns.load();
    for (int li = 0; li < pk.eg_layers; ++li) {
        EgBatch& b = *eg_win[(size_t) li];
        if (b.live) eg_finish(b);   // (a window that failed midway: its reads end before h_eg is reused)
        eg_issue(p0, T, li, nullptr, b);
    }
    st.engram_ms += now_ms() - te;
    const int D = g.dim;
    for (int t = 0; t < T; ++t)   // (h_emb is written again by the next window, after this one's last sync)
        std::memcpy(h_emb + (size_t) t * D, emb_row(p0 + t, tk[t]), (size_t) D * 2);
    plan_window(p0, T, wplan);
    prof_window = prof_on && T == 1;
    n_logits = std::min(logits_rows, T);
    const double tl = now_ms();
    if (par) {
        if (!forward_par(p0, T, err)) return false;
    } else if (!forward_serial(p0, T, err)) return false;
    st.layer_ms += now_ms() - tl;
    for (auto& b : eg_win)
        if (b->live) eg_finish(*b);
    st.engram_ms += (double) (eg_wait_ns.load() - eg_w0) / 1e6;
    eg_stats();
    Dev& d = devs[(size_t) head_dev];
    cudaSetDevice(d.id);
    const double th = now_ms();
    CK(cudaStreamSynchronize(d.st));
    st.head_ms += now_ms() - th;
    prof_collect();
    pos += T;
    st.tokens += T;
    if (seed_rows != 0 && !seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err)) return false;
    return true;
}

bool Model::Impl::step_conversations(const std::vector<int>& tokens, const std::function<void(int)>& select,
                                     const std::function<void(int)>& bind, const std::vector<int>& counts,
                                     std::string& err) {
    const int T = (int) tokens.size();
    if (T < 1 || T > 8 || std::any_of(groups.begin(), groups.end(),
                                     [](const Group& group) { return group.mem.size() != 1; })) {
        err = "independent rows require 1 to 8 rows and layer-parallel stages";
        return false;
    }
    std::fill(std::begin(row_counts), std::end(row_counts), 0);
    int first = 0;
    for (int count : counts) {
        if (count < 1 || first + count > T) {
            err = "invalid conversation window layout";
            return false;
        }
        row_counts[first] = count;
        first += count;
    }
    if (first != T) {
        err = "conversation windows do not cover the step";
        return false;
    }
    const bool verify = std::any_of(counts.begin(), counts.end(), [](int n) { return n > 1; });
    for (int il = 0; verify && il < g.nl; ++il)
        if (!init_verify_layer(L[(size_t) il], err, true))
            return false;
    if (!prepare_conversation_rows(tokens, select, h_emb, h_eg, eg_cap, row_positions, row_plans,
                                   err, row_counts))
        return false;
    plan_window(0, T, wplan);
    wplan.par = false;
    n_logits = T;
    prof_window = prof_on;
    eg_stats();
    select_row = bind;
    spec_win = verify;
    const double tl = now_ms();
    const bool ok = forward_serial(0, T, err);
    st.layer_ms += now_ms() - tl;
    select_row = {};
    spec_win = false;
    if (!ok)
        return false;
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        CK(cudaStreamSynchronize(d.st));
    }
    prof_collect();
    st.tokens += T;
    return true;
}

bool Model::Impl::prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
                                const std::function<void(int)>& bind, std::string& err) {
    if (tokens.empty() || tokens.size() > 8) {
        err = "pipeline groups require 1 to 8 independent rows";
        return false;
    }
    if (pipeline_groups.size() <= (size_t) group)
        pipeline_groups.resize((size_t) group + 1);
    PipelineGroup& pg = pipeline_groups[(size_t) group];
    Dev& head = devs[(size_t) head_dev];
    cudaSetDevice(head.id);
    const size_t per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
    if (pg.embeddings == nullptr &&
        (!halloc(pg.embeddings, (size_t) 8 * g.dim, err) ||
         !halloc(pg.engram, (size_t) std::max(1, pk.eg_layers) * 8 * per + 64, err) ||
         !halloc(pg.taps, (size_t) 8 * std::max<size_t>(1, g.d_targets.size()) * g.dim, err) ||
         !halloc(pg.hand, (size_t) 8 * (g.hc * g.dim + 8 + topk_stride() + cand_stride()) * 4, err) ||
         !dalloc(head, pg.logits, (size_t) 8 * g.vocab, err)))
        return false;
    if (pg.copies.empty()) {
        pg.copies.resize(srcs.size() * 2, nullptr);
        for (size_t i = 0; i < srcs.size(); ++i)
            if (!halloc(pg.copies[2 * i], (size_t) 8 * ckv_rb(), err) ||
                !halloc(pg.copies[2 * i + 1], (size_t) 8 * ik_rb(), err))
                return false;
    }
    if (pipeline_cpu_parts.empty()) {
        pipeline_cpu_parts.resize(devs.size(), nullptr);
        for (float*& part : pipeline_cpu_parts)
            if (!halloc(part, (size_t) 8 * g.dim, err))
                return false;
    }
    pg.count = (int) tokens.size();
    pg.bind = bind;
    if (!prepare_conversation_rows(tokens, select, pg.embeddings, pg.engram, 8, pg.positions, pg.plans, err))
        return false;
    plan_window(0, pg.count, pg.plan, true);
    pg.plan.par = false;
    return true;
}

bool Model::Impl::launch_group(int stage, int group, std::string& err) {
    PipelineGroup& pg = pipeline_groups[(size_t) group];
    Dev& d = devs[(size_t) stage];
    cudaSetDevice(d.id);
    const int T = pg.count;
    if (stage == 0) {
        CK(cudaMemcpyAsync(d.emb_rows, pg.embeddings, (size_t) T * g.dim * 2,
                           cudaMemcpyHostToDevice, d.st));
        embed(d.emb_rows, T, d.h, g.dim, g.hc, d.st);
        init_pre(d.pre, T, d.st);
    } else {
        Xfer x = hand_xfer(pg.hand, d, T, false);
        for (int i = 0; i < x.n; ++i)
            CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, d.st));
        strata::core::count_gpu_traffic(d.id, x.total(), 0);
    }
    std::swap(h_eg, pg.engram);
    std::swap(h_taps, pg.taps);
    const size_t old_cap = eg_cap;
    eg_cap = 8;
    active_pipeline = &pg;
    std::swap(h_part, pipeline_cpu_parts[(size_t) stage]);
    select_row = pg.bind;
    std::copy_n(pg.positions, T, row_positions);
    std::copy_n(pg.plans, T, row_plans);
    prof_window = false;
    // The CPU mailbox consumes one sequence in launch order, including across stages.
    pg.plan.async = par || (!sync_decode && T <= async_max_t);
    bool ok = true;
    for (Layer& l : L) {
        if (l.il >= g.nl || l.dev != stage)
            continue;
        if (pg.plan.async) {
            pg.plan.seq[l.il] = ++seq_issued;
            layer_of[pg.plan.seq[l.il] % kRing].store(l.il, std::memory_order_release);
        }
        if (!tap(d, l.il, T, err) || !layer(l, 0, 0, T, pg.plan, err)) {
            ok = false;
            break;
        }
    }
    select_row = {};
    std::swap(h_part, pipeline_cpu_parts[(size_t) stage]);
    active_pipeline = nullptr;
    eg_cap = old_cap;
    std::swap(h_taps, pg.taps);
    std::swap(h_eg, pg.engram);
    if (!ok)
        return false;
    cudaSetDevice(d.id);
    if (stage == head_dev) {
        n_logits = T;
        std::swap(d.logits, pg.logits);
        ok = head_rows_dev(stage, T, err);
        std::swap(d.logits, pg.logits);
        if (!ok)
            return false;
    } else {
        Xfer x = hand_xfer(pg.hand, d, T, true);
        for (int i = 0; i < x.n; ++i)
            CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, d.st));
        strata::core::count_gpu_traffic(d.id, 0, x.total());
    }
    CK(cudaEventRecord(d.ev_out, d.st));
    return true;
}

int Model::Impl::poll_group(int stage, std::string& err) {
    Dev& d = devs[(size_t) stage];
    cudaSetDevice(d.id);
    const cudaError_t result = cudaEventQuery(d.ev_out);
    if (result == cudaErrorNotReady)
        return 0;
    if (result == cudaSuccess)
        return 1;
    err = cudaGetErrorString(result);
    return -1;
}

bool Model::Impl::finish_group(int group, const std::function<void(int)>& select, std::string& err) {
    PipelineGroup& pg = pipeline_groups[(size_t) group];
    Dev& d = devs[(size_t) head_dev];
    cudaSetDevice(d.id);
    CK(cudaMemcpyAsync(d.logits, pg.logits, (size_t) pg.count * g.vocab * 4,
                       cudaMemcpyDeviceToDevice, d.st));
    std::swap(h_taps, pg.taps);
    bool ok = true;
    for (int row = 0; row < pg.count; ++row) {
        select(row);
        if (!seed_drafter(pg.positions[row], 1, row, err)) {
            ok = false;
            break;
        }
    }
    std::swap(h_taps, pg.taps);
    cudaSetDevice(d.id);
    CK(cudaStreamSynchronize(d.st));
    st.tokens += pg.count;
    return ok;
}

bool Model::Impl::prepare_conversation_rows(const std::vector<int>& tokens, const std::function<void(int)>& select,
                                           uint16_t* embeddings, uint8_t* engram, size_t capacity,
                                           int64_t* positions, WinPlan* plans, std::string& err,
                                           const int* window_counts) {
    const int T = (int) tokens.size();
    if (!window_counts) {
        std::fill(std::begin(row_counts), std::end(row_counts), 1);
        window_counts = row_counts;
    }
    for (int row = 0; row < T; ++row) {
        select(row);
        if (pos >= o.max_context || tokens[row] < 0 || tokens[row] >= g.vocab) {
            err = "independent row outside the context or vocabulary";
            return false;
        }
        const int64_t p0 = pos;
        positions[row] = p0;
        toks.push_back(tokens[row]);
        ctoks.push_back(ctok(p0, tokens[row]));
        pos = (int64_t) toks.size();
        std::memcpy(embeddings + (size_t) row * g.dim, emb_row(p0, tokens[row]), (size_t) g.dim * 2);
        const double te = now_ms();
        for (int li = 0; li < pk.eg_layers; ++li) {
            const size_t per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
            engram_rows(p0, 1, li, engram + ((size_t) li * capacity + row) * per);
        }
        st.engram_ms += now_ms() - te;
        if (window_counts[row] > 0)
            plan_window(p0, window_counts[row], plans[row], true);
    }
    return true;
}

void Model::Impl::dbg_nf(const char* tag, int il, const float* dev, size_t n) {
    const float* hp = (const float*) strata::gpu::host_view(dev);
    size_t nf = 0, first = 0; float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) { if (!nf) first = i; ++nf; } else if (std::fabs(v) > mx) mx = std::fabs(v); }
    std::fprintf(stderr, "DBG nf il=%d %s n=%zu nonfinite=%zu first=%zu max=%.4g\n", il, tag, n, nf, first, mx);
}

void Model::Impl::dbg_h(const char* tag, Dev& d, int T) {
    const float* hp = (const float*) strata::gpu::host_view(d.h);
    float mx = 0.0f; size_t nf = 0;
    const size_t n = (size_t) T * g.hc * g.dim;
    for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) ++nf; else if (std::fabs(v) > mx) mx = std::fabs(v); }
    std::fprintf(stderr, "DBG hmax %s T=%d max=%.4g nonfinite=%zu h0=%g\n", tag, T, mx, nf, hp[0]);
}

bool Model::Impl::forward_serial(int64_t p0, int T, std::string& err) {
    const int D = g.dim, H = g.hc;
    int cur = L[0].dev;
    if (ds4_dbg("DS4_SYNC_WIN")) cudaDeviceSynchronize();
    {
        Dev& d = devs[(size_t) cur];
        cudaSetDevice(d.id);
        CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, d.st));
        embed(d.emb_rows, T, d.h, D, H, d.st);
        if (ds4_dbg("DS4_DBG_EMB")) {
            float *hp = d.h;
            float hh[4] = {0,0,0,0};
            cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
        }
        init_pre(d.pre, T, d.st);
        if (ds4_dbg("DS4_DBG_EMB")) dbg_h("after init_pre", d, T);
    }
    std::vector<float> dump;
    if (dump_h != nullptr) dump.resize((size_t) T * g.nl * H * D);
    for (int il = 0; il < g.nl; ++il) {
        Layer& l = L[(size_t) il];
        if (l.dev != cur) {
            Dev& a = devs[(size_t) cur];
            Dev& b = devs[(size_t) l.dev];
            cudaSetDevice(a.id);
            const size_t nh = (size_t) T * H * D * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                         nc = (size_t) T * cand_stride() * 4;
            uint8_t* bb = a.bounce;
            CK(cudaMemcpyAsync(bb, a.h, nh, cudaMemcpyDeviceToHost, a.st));
            CK(cudaMemcpyAsync(bb + nh, a.pre, np, cudaMemcpyDeviceToHost, a.st));
            CK(cudaMemcpyAsync(bb + nh + np, a.topk, nk, cudaMemcpyDeviceToHost, a.st));
            CK(cudaMemcpyAsync(bb + nh + np + nk, a.cand, nc, cudaMemcpyDeviceToHost, a.st));
            CK(cudaEventRecord(a.ev_out, a.st));
            cudaSetDevice(b.id);
            CK(cudaStreamWaitEvent(b.st, a.ev_out, 0));
            CK(cudaMemcpyAsync(b.h, bb, nh, cudaMemcpyHostToDevice, b.st));
            CK(cudaMemcpyAsync(b.pre, bb + nh, np, cudaMemcpyHostToDevice, b.st));
            CK(cudaMemcpyAsync(b.topk, bb + nh + np, nk, cudaMemcpyHostToDevice, b.st));
            CK(cudaMemcpyAsync(b.cand, bb + nh + np + nk, nc, cudaMemcpyHostToDevice, b.st));
            strata::core::count_gpu_traffic(a.id, 0, nh + np + nk + nc);
            strata::core::count_gpu_traffic(b.id, nh + np + nk + nc, 0);
            cur = l.dev;
        }
        if (!tap(devs[(size_t) l.dev], il, T, err)) return false;
        if (il == 0 && ds4_dbg("DS4_DBG_EMB")) dbg_h("after tap", devs[(size_t) l.dev], T);
        if (!layer(l, 0, p0, T, wplan, err)) return false;
        if (dump_h != nullptr) {   // DS4_DUMP_H: every layer's residual stream, for tools/ds4/compare.py --layers
            Dev& dd = devs[(size_t) l.dev];
            std::vector<float> hv((size_t) T * H * D);
            cudaSetDevice(dd.id);
            CK(cudaMemcpyAsync(hv.data(), dd.h, hv.size() * 4, cudaMemcpyDeviceToHost, dd.st));
            CK(cudaStreamSynchronize(dd.st));
            for (int t = 0; t < T; ++t)
                std::memcpy(dump.data() + ((size_t) t * g.nl + l.il) * H * D, hv.data() + (size_t) t * H * D,
                            (size_t) H * D * 4);
        }
    }
    if (dump_h != nullptr) std::fwrite(dump.data(), 4, dump.size(), dump_h);
    if (cur != head_dev) {
        err = "the head is not on the last layer's GPU";
        return false;
    }
    return head_rows_dev(head_dev, T, err);
}

bool Model::Impl::forward_par(int64_t p0, int T, std::string& err) {
    wplan.win = ++win_seq;
    std::vector<std::string> errs(devs.size());
    for (size_t di = 0; di < devs.size(); ++di)
        workers[di]->submit([this, di, p0, T, &errs] {
            std::string e;
            if (!run_dev((int) di, p0, T, wplan, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
        });
    for (auto& w : workers) w->wait();
    for (const std::string& e : errs)
        if (!e.empty()) {
            err = e;
            return false;
        }
    return true;
}

Xfer Model::Impl::hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
    const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                 nc = (size_t) T * cand_stride() * 4;
    void* dv[4] = {d.h, d.pre, d.topk, d.cand};
    uint8_t* hv[4] = {host, host + nh, host + nh + np, host + nh + np + nk};
    const size_t nb[4] = {nh, np, nk, nc};
    Xfer x;
    x.n = 4;
    for (int i = 0; i < 4; ++i) {
        x.src[i] = put ? (const void*) dv[i] : (const void*) hv[i];
        x.dst[i] = put ? (void*) hv[i] : dv[i];
        x.bytes[i] = nb[i];
    }
    return x;
}

bool Model::Impl::run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
    Dev& d = devs[(size_t) di];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, H = g.hc;
    bool started = false;
    std::vector<char> got(srcs.size(), 0);
    for (int il = 0; il < g.nl; ++il) {
        Layer& L0 = L[(size_t) il];
        int mi = -1;
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == di) mi = m;
        if (mi < 0) continue;
        if (!started) {   // the window's input: the embedding rows, or the previous group's hand-off
            started = true;
            if (il == 0) {
                CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, s));
                embed(d.emb_rows, T, d.h, D, H, s);
                if (ds4_dbg("DS4_DBG_EMB")) {
                    float *hp = d.h;
                    float hh[4] = {0,0,0,0};
                    cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                }
                init_pre(d.pre, T, s);
            } else {
                const Dev& a = devs[(size_t) L[(size_t) il - 1].dev];
                xfer_get(hand_xfer(a.hand, d, T, false), a.hand_flag, wp.win, s, d.id);
            }
        }
        // the compressed KV / index keys another group's source layer wrote this window
        for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
            if (sidx < 0 || got[(size_t) sidx]) continue;
            got[(size_t) sidx] = 1;
            Src& sr = srcs[(size_t) sidx];
            if (L[(size_t) sr.layer].grp == L0.grp) continue;
            take_new_rows(sr, di, p0, T, sr.x_stage, sr.x_flag, wp.win, s);
        }
        if (mi == 0 && !tap(d, il, T, err)) return false;
        if (!layer(L0, mi, p0, T, wp, err)) return false;
        if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, il, p0, T);
        const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
        if (last && mi == 0 && il + 1 < g.nl)
            xfer_put(hand_xfer(d.hand, d, T, true), d.hand_flag, wp.win, d.xcnt, s, d.id);
    }
    if (d.grp == devs[(size_t) head_dev].grp) return head_rows_dev(di, T, err);
    return true;
}

bool Model::Impl::replicate(std::vector<KvRows>& copies, uint8_t* stage, cudaEvent_t ev, int64_t r0, int n, int from,
                            std::string& err) {
    if (active_pipeline != nullptr) {
        for (size_t i = 0; i < srcs.size(); ++i) {
            if (stage == srcs[i].h_ckv)
                stage = active_pipeline->copies[2 * i] + (size_t) active_row * ckv_rb();
            else if (stage == srcs[i].h_ik)
                stage = active_pipeline->copies[2 * i + 1] + (size_t) active_row * ik_rb();
        }
    }
    bool any = false;
    for (size_t od = 0; od < devs.size(); ++od) any = any || (copies[od] && (int) od != from);
    if (!any) return true;
    Dev& d = devs[(size_t) from];
    if (!copies[(size_t) from].for_each_run(r0, n, [&](size_t off, size_t bytes, const uint8_t* at) {
            CK(cudaMemcpyAsync(stage + off, at, bytes, cudaMemcpyDeviceToHost, d.st));
            strata::core::count_gpu_traffic(d.id, 0, bytes);
            return true;
        }))
        return false;
    CK(cudaEventRecord(ev, d.st));
    for (size_t od = 0; od < devs.size(); ++od) {
        if (!copies[od] || (int) od == from) continue;
        Dev& o2 = devs[od];
        cudaSetDevice(o2.id);
        CK(cudaStreamWaitEvent(o2.st, ev, 0));
        if (!copies[od].for_each_run(r0, n, [&](size_t off, size_t bytes, uint8_t* at) {
                CK(cudaMemcpyAsync(at, stage + off, bytes, cudaMemcpyHostToDevice, o2.st));
                strata::core::count_gpu_traffic(o2.id, bytes, 0);
                return true;
            }))
            return false;
    }
    cudaSetDevice(d.id);
    return true;
}

void Model::Impl::put_new_rows(const Src& sr, const Layer& l, int64_t g_first, int ng, uint8_t* stage, unsigned* flag,
                               unsigned seq, unsigned* cnt, cudaStream_t s) {
    std::vector<Xfer> xs;
    add_row_copies(xs, sr.ckv[(size_t) l.dev], g_first, ng, stage, true);
    if (l.owns_k && srcs[(size_t) l.ik_src].ik[(size_t) l.dev])
        add_row_copies(xs, srcs[(size_t) l.ik_src].ik[(size_t) l.dev], g_first, ng, stage + x_cap() * ckv_rb(),
                       true);
    // (the flag goes with the last part: the parts before it are earlier on the same stream)
    for (size_t i = 0; i < xs.size(); ++i)
        xfer_put(xs[i], i + 1 == xs.size() ? flag : nullptr, seq, cnt, s, devs[(size_t) l.dev].id);
}

void Model::Impl::take_new_rows(const Src& sr, size_t di, int64_t p0, int T, uint8_t* stage, const unsigned* flag,
                                unsigned seq, cudaStream_t s, bool host_first) {
    const int64_t g_first = p0 / sr.ratio;
    const int ng = (int) std::max<int64_t>(0, (p0 + T) / sr.ratio - g_first);
    std::vector<Xfer> xs;
    if (sr.ckv[di])
        add_row_copies(xs, sr.ckv[di], g_first, ng, stage, false);
    if (sr.ik[di])
        add_row_copies(xs, sr.ik[di], g_first, ng, stage + x_cap() * ckv_rb(), false);
    if (host_first && !xs.empty())
        host_wait_at_least(flag, seq);
    for (const Xfer& x : xs)
        xfer_get(x, flag, seq, s, devs[di].id);
}

void Model::Impl::add_row_copies(std::vector<Xfer>& xs, const KvRows& rows, int64_t r0, int n, uint8_t* stage,
                                 bool out) {
    rows.for_each_run(r0, n, [&](size_t off, size_t bytes, uint8_t* at) {
        if (xs.empty() || xs.back().n == 4)
            xs.emplace_back();
        Xfer& x = xs.back();
        x.src[x.n] = out ? (const void*) at : stage + off;
        x.dst[x.n] = out ? (void*) (stage + off) : at;
        x.bytes[x.n++] = bytes;
        return true;
    });
}

bool Model::Impl::run_attention(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
    Layer& l = member(L0, mi);
    Dev& d = devs[(size_t) l.dev];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn;
    if (ds4_dbg("DS4_TRACE_DISP")) std::fprintf(stderr, "TRACE layer %d p0=%lld T=%d\n", l.il, (long long) p0, T);
    if (ds4_dbg("DS4_DBG_H") && mi == 0 && (l.il < 2 || ds4_dbg("DS4_DBG_ALL"))) {   // DBG
        std::vector<float> hv((size_t) T * H * D);
        cudaSetDevice(d.id);
        {
            const float* hvp = (const float*) strata::gpu::host_view(d.h);
            if (hvp) std::memcpy(hv.data(), hvp, hv.size() * 4);
        }
        int nf = 0, nzv = 0;
        float mx = 0.0f;
        size_t firstnan = 0, lastnan = 0;
        for (size_t i = 0; i < hv.size(); ++i) {
            const float v = hv[i];
            if (!(v == v)) { if (nf == 0) firstnan = i; lastnan = i; ++nf; continue; }
            if (v != 0.0f) ++nzv;
            if (std::fabs(v) > mx) mx = std::fabs(v);
        }
        std::fprintf(stderr, "DBG h: il=%d dev=%d h=%p p0=%lld T=%d p0tok=%d zeros=%zu ninf=%d max=%.4g first=%.5g %.5g nan_range=%zu..%zu\n",
                     l.il, (int) l.dev, (void*) d.h, (long long) p0, T, toks.empty() ? -1 : toks[(size_t) p0],
                     hv.size() - (size_t) nzv - (size_t) nf, nf, mx, hv[0], hv.size() > 1 ? hv[1] : 0.0f, firstnan, lastnan);
    }
    prof_mark(d, 0);
    // Engram
    if (l.eg >= 0) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        const size_t per = (size_t) cols * (ehd + ehd / 32);
        eg_finish(*eg_win[(size_t) l.eg]);   // (asked for when the window began)
        uint8_t* src = h_eg + (size_t) l.eg * eg_cap * per;
        for (int t = 0; t < T; ++t)   // [token][rows | scales] -> [all rows][all scales]
            CK(cudaMemcpyAsync(d.eg_rows + (size_t) t * cols * ehd, src + (size_t) t * per, (size_t) cols * ehd,
                               cudaMemcpyHostToDevice, s));
        uint8_t* sc = d.eg_rows + (size_t) T * cols * ehd;
        for (int t = 0; t < T; ++t)
            CK(cudaMemcpyAsync(sc + (size_t) t * cols * (ehd / 32), src + (size_t) t * per + (size_t) cols * ehd,
                               (size_t) cols * (ehd / 32), cudaMemcpyHostToDevice, s));
        strata::core::count_gpu_traffic(d.id, (uint64_t) T * per, 0);
        engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
        dump_sub(d, "eg_rows", l.il, p0, T, d.eg_rows, (size_t) T * cols * ehd);
        dump_sub(d, "eg_x", l.il, p0, T, d.eg_x, (size_t) T * cols * ehd * 4);
        if (ds4_dbg("DS4_DBG_EG")) {
            dbg_nf("eg_x", l.il, d.eg_x, (size_t) T * cols * ehd);
            const float* xv = (const float*) strata::gpu::host_view(d.eg_x);
            for (size_t i = 0; i < (size_t) T * cols * ehd; ++i)
                if (!std::isfinite(xv[i])) {
                    const size_t t = i / ((size_t) cols * ehd), c = (i / ehd) % cols;
                    const uint8_t* dv = (const uint8_t*) strata::gpu::host_view(d.eg_rows) + (t * cols + c) * ehd;
                    const uint8_t* hv = src + t * per + c * ehd;
                    std::fprintf(stderr, "DBG eg bad x: t=%zu c=%zu e=%zu | dev/host row bytes:", t, c, i % ehd);
                    for (int b = 0; b < 80; ++b) std::fprintf(stderr, "%s%02x/%02x", b % 18 == 0 ? " |" : " ", dv[b], hv[b]);
                    std::fprintf(stderr, "\n");
                    break;
                }
        }
        if (!mv(d, {job(l.eg_wkv, d.eg_x, d.eg_kv, T, (size_t) cols * ehd, (size_t) (H + 1) * D)}, err)) return false;
        if (ds4_dbg("DS4_DBG_EG")) dbg_nf("eg_kv", l.il, d.eg_kv, (size_t) T * (H + 1) * D);
        dump_sub(d, "eg_kv", l.il, p0, T, d.eg_kv, (size_t) T * (H + 1) * D * 4);
        dump_sub(d, "eg_h0", l.il, p0, T, d.h, (size_t) T * H * D * 4);
        dump_sub(d, "eg_w", l.il, p0, 1, l.eg_w, (size_t) H * D * 4);
        engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);
        dump_sub(d, "eg_h", l.il, p0, T, d.h, (size_t) T * H * D * 4);
        if (ds4_dbg("DS4_DBG_EG")) dbg_nf("h_after_gate", l.il, d.h, (size_t) T * H * D);
    }
    dump_sub(d, "h_in", l.il, p0, T, d.h, (size_t) T * H * D * 4);
    dump_sub(d, "ring_in", l.il, p0, T, l.win, (size_t) g.win * kv_row_bytes(kKvFp8, HD));
    // attention
    hc_mix_split(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps,
                 d.a_pre, d.a_post, d.a_comb, s);
    dump_sub(d, "a_mix", l.il, p0, T, d.mix, (size_t) T * 32 * 4);
    dump_sub(d, "a_pre", l.il, p0, T, d.a_pre, (size_t) T * 8 * 4);
    dump_sub(d, "a_post", l.il, p0, T, d.a_post, (size_t) T * 8 * 4);
    dump_sub(d, "a_comb", l.il, p0, T, d.a_comb, (size_t) T * 64 * 4);
    hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);
    dump_sub(d, "pre_in", l.il, p0, T, d.pre, (size_t) T * 8 * 4);
    dump_sub(d, "attn_x", l.il, p0, T, d.x, (size_t) T * D * 4);
    {
        x3::MvJob j[4];
        int nj = 0;
        j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
        j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
        if (l.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
        if (l.kv_src && l.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
        if (!mv_n(d, j, nj, err)) return false;
    }
    dump_sub(d, "qr_raw", l.il, p0, T, d.qr, (size_t) T * g.q_lora * 4);
    dump_sub(d, "kvr_raw", l.il, p0, T, d.kvr, (size_t) T * HD * 4);
    Finish fq;
    fq.w = l.q_norm;
    fq.eps = g.eps;
    finish(d.qr, T, g.q_lora, g.q_lora, fq, s);
    dump_sub(d, "qr", l.il, p0, T, d.qr, (size_t) T * g.q_lora * 4);
    if (NHL > 0) {
        if (!mv(d, {job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)}, err)) return false;
    }
    if (!select_row) {
        if (!run_attention_state(L0, mi, p0, T, wp, err))
            return false;
    } else {   // a run of rows a conversation: select_row's
        float** fields[] = {&d.qr, &d.q, &d.kvr, &d.ckr, &d.cgr, &d.x, &d.o, &d.iq, &d.iw};
        const size_t strides[] = {(size_t) g.q_lora, (size_t) HQ, (size_t) HD, (size_t) HD, (size_t) HD,
                                  (size_t) D, (size_t) HQ, (size_t) g.ix_heads * g.ix_hd, 64};
        constexpr int kFields = (int) std::size(fields);
        float* bases[kFields];
        for (int i = 0; i < kFields; ++i)
            bases[i] = *fields[i];
        int* topk = d.topk;
        int* cand = d.cand;
        uint16_t* ext = d.ext;
        AttnRows batch_rows;
        AttnRows* const batch = row_counts[0] < T && NHL > 0 && !l.mtp &&
                                can_sparse_attn_unpack(T, (int) topk_stride(), d.attn_scr) ? &batch_rows : nullptr;
        int rpos[kWin];   // (batch) each row's position
        if (batch != nullptr) {   // every run's queries rotated (and their indexer queries made) at once
            for (int row = 0; row < T; row += row_counts[row])
                for (int i = 0; i < row_counts[row]; ++i)
                    rpos[row + i] = (int) row_positions[row] + i;
            finish(d.q, T * NHL, HD, HD, get_rope(l, d, l.ahn, 0, T, rpos), s);
            if (l.ratio > 0 && l.ix_src && !project_indexer_queries(l, d, 0, T, rpos, err))
                return false;
        }
        for (int row = 0; row < T;) {
            const int n = row_counts[row];
            active_row = row;
            select_row(row);
            for (int i = 0; i < kFields; ++i)
                *fields[i] = bases[i] + (size_t) row * strides[i];
            d.topk = topk + (size_t) row * topk_stride();
            d.cand = cand + (size_t) row * cand_stride();
            if (batch != nullptr)
                d.ext = ext + (size_t) row * ext_slice_rows() * HD;
            uint8_t* ring_bak = l.ring_bak;
            float* c_snap = l.c_snap;
            float* c_rows = l.c_rows;
            if (spec_win) {
                l.ring_bak += (size_t) row * win_rb();
                if (c_snap) {
                    l.c_snap = l.joint_c_snap + (size_t) row * 2 * l.ratio * HD;
                    l.c_rows += (size_t) row * HD;
                }
            }
            const bool ok = run_attention_state(L0, mi, row_positions[row], n, row_plans[row], err, 0, batch);
            l.ring_bak = ring_bak;
            l.c_snap = c_snap;
            l.c_rows = c_rows;
            for (int i = 0; i < kFields; ++i)
                *fields[i] = bases[i];
            d.topk = topk;
            d.cand = cand;
            d.ext = ext;
            if (!ok)
                return false;
            row += n;
        }
        if (batch != nullptr) {   // every run's attention in one launch, then their outputs rotated back
            sparse_attn_unpacked(d.q, d.ext, *batch, l.sink, NHL, HD, T, 1.0f / std::sqrt((float) HD), d.o,
                                 d.attn_scr, s);
            finish_attention(l, d, d.o, 0, T, rpos);
        }
    }
    if (NHL > 0) {
        dump_sub(d, "o_fin", l.il, p0, T, d.o, (size_t) T * HQ * 4);
        x3::MvJob j[16];
        const int per = HQ / OG;
        for (int gi = 0; gi < OG; ++gi)
            j[gi] = job(l.wo_a[gi], d.o + (size_t) gi * per, d.oa + (size_t) gi * g.o_lora,
                        T, HQ, (size_t) OG * g.o_lora);
        if (!mv_n(d, j, OG, err))
            return false;
        if (!mv(d, {job(l.wo_b, d.oa, d.ao, T, (size_t) OG * g.o_lora, D)}, err))
            return false;
    } else CK(cudaMemsetAsync(d.ao, 0, (size_t) T * D * sizeof(*d.ao), s));   // (no heads: nothing to add)
    dump_sub(d, "ao", l.il, p0, T, d.ao, (size_t) T * D * 4);
    hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);
    std::swap(d.h, d.h2);
    if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, 1000 + l.il, p0, T);   // (DS4_DUMP_L: after the attention)
    dump_sub(d, "h_attn", l.il, p0, T, d.h, (size_t) T * H * D * 4);
    prof_mark(d, 1);
    return true;
}

bool Model::Impl::run_attention_state(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err, int r,
                                      AttnRows* batch) {
    Layer& l = member(L0, mi);
    Dev& d = devs[(size_t) l.dev];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, HD = g.hd, NHL = l.ahn, HQ = NHL * HD;
    const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
    if (NHL > 0 && batch == nullptr)   // (batch: run_attention rotates every run's queries)
        finish(d.q, T * NHL, HD, HD, get_rope(l, d, l.ahn, p0, T, nullptr), s);
    if (NHL > 0) dump_sub(d, "q", l.il, p0, T, d.q, (size_t) T * HQ * 4);
    // the window rows: the ring's (positions p0 - win ..) then this window's, each query its own 128.  A DSpark
    // block's ring holds the trunk's main KV (positions <= p0 - 1, seeded); its block of drafts attends to all of
    // that ring and to the whole block, and leaves the ring as it was
    ring_to_ext(l.win, d.ext, l.mtp ? (int64_t) g.win : p0, g.win, HD, s);   // (a block's ring: its rows as they lie)
    Finish fk;
    fk.w = l.kv_norm;
    fk.eps = g.eps;
    fk.rd = g.rd;
    fk.pos = (int) p0;
    fk.freqs = freqs;
    fk.quant = 1;
    fk.dst = d.ext + (size_t) g.win * HD;
    fk.dst_stride = HD;
    finish(d.kvr, T, HD, HD, fk, s);
    if (!l.mtp) ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, spec_win ? l.ring_bak : nullptr);
    dump_sub(d, "kv_win", l.il, p0, T, d.ext + (size_t) g.win * HD, (size_t) T * HD * 4);
    dump_sub(d, "ext", l.il, p0, T, d.ext, ((size_t) g.win + T) * HD * 2);   // (BF16 rows: the ring's, then this window's)
    AttnRows ar;
    for (int t = 0; t < T; ++t) {
        if (l.mtp) {
            ar.win0[t] = 0;
            ar.nwin[t] = (int) std::min<int64_t>(g.win, p0);
            ar.nsel[t] = T;
            continue;
        }
        const int first = (int) std::max<int64_t>(g.win - p0, t + 1);   // ext row of position max(0, p - win + 1)
        ar.win0[t] = first;
        ar.nwin[t] = g.win + t - first + 1;
        ar.nsel[t] = 0;
    }
    KvRows ckv;
    int ckv_fmt = kKvFp4;
    if (l.ratio > 0) {
        Src& sr = srcs[(size_t) l.src];
        const int R = l.ratio;
        // the groups this window completes: g_first .. g_first + ng - 1
        const int64_t g_first = p0 / R;              // the first group whose last position is >= p0
        const int64_t g_end = (p0 + T) / R;          // groups completed by p0 + T - 1
        const int ng = (int) std::max<int64_t>(0, g_end - g_first);
        if (l.kv_src) {
            if (R == 1) {
                Finish fc;
                fc.w = l.c_norm;
                fc.eps = g.eps;
                finish(d.ckr, T, HD, HD, fc, s);
                CK(cudaMemcpyAsync(d.latent, d.ckr, (size_t) T * HD * 4, cudaMemcpyDeviceToDevice, s));
            } else {
                if (spec_win && l.c_snap != nullptr) {   // to replay the accepted rows if drafts are rejected
                    const size_t ns = (size_t) R * HD * 4, nr = (size_t) T * HD * 4;
                    CK(cudaMemcpyAsync(l.c_snap, l.c_kv, ns, cudaMemcpyDeviceToDevice, s));
                    CK(cudaMemcpyAsync(l.c_snap + (size_t) R * HD, l.c_score, ns, cudaMemcpyDeviceToDevice, s));
                    CK(cudaMemcpyAsync(l.c_rows, d.ckr, nr, cudaMemcpyDeviceToDevice, s));
                    CK(cudaMemcpyAsync(l.c_rows + (size_t) kWin * HD, d.cgr, nr, cudaMemcpyDeviceToDevice, s));
                }
                compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, d.latent, s);
                dump_sub(d, "ckr", l.il, p0, T, d.ckr, (size_t) T * HD * 4);
                dump_sub(d, "cgr", l.il, p0, T, d.cgr, (size_t) T * HD * 4);
            }
        }
        Rows lim;
        int clen_max = 0;
        for (int t = 0; t < T; ++t) {
            lim.n[t] = (int) ((p0 + t + 1) / R);
            lim.k[t] = std::min(g.ix_topk, lim.n[t]);
            clen_max = std::max(clen_max, lim.n[t]);
        }
        if (l.ix_src) {
            Src& ks = srcs[(size_t) l.ik_src];
            if (l.owns_k && ng > 0) {
                if (!mv(d, {job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)}, err)) return false;
                Finish fi;
                fi.w = l.ik_norm;
                fi.eps = g.eps;
                fi.rd = g.rd;
                fi.pos = (int) (g_first * R);
                fi.pos_step = R;
                fi.freqs = freqs;
                fi.quant = 3;
                fi.pdst = ks.ik[(size_t) l.dev];
                fi.pdst_row = g_first;
                finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
                if (!wp.par && !replicate(ks.ik, ks.h_ik + (size_t) r * ik_rb(), ks.ev_ik, g_first, ng, l.dev, err))
                    return false;
            }
            if (clen_max > 0) {
                // (batch: run_attention projects them)
                if (batch == nullptr && !project_indexer_queries(l, d, p0, T, nullptr, err))
                    return false;
                const long long ss = (long long) o.max_context + 64;
                bool restrict_ = false;
                for (int t = 0; t < T; ++t)
                    restrict_ = restrict_ || (lim.n[t] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                if (l.uses_cand && restrict_) {
                    Rows cl;   // the candidate positions of each query: its blocks x blk
                    int nmax = 0;
                    for (int t = 0; t < T; ++t) {
                        cl.n[t] = lim.n[t];
                        cl.k[t] = wp.cand[l.il][r + t];
                        nmax = std::max(nmax, wp.cand[l.il][r + t] * g.cand_blk);
                    }
                    index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, nmax, d.cand,
                                (int) cand_stride(), g.cand_blk, cl, d.scores, ss, s);
                    Rows tr;
                    for (int t = 0; t < T; ++t) {
                        tr.n[t] = wp.cand[l.il][r + t] * g.cand_blk;
                        tr.k[t] = lim.k[t];
                    }
                    topk(d.scores, ss, T, tr, d.cand, (int) cand_stride(), g.cand_blk, d.topk, (int) topk_stride(), s);
                } else {
                    index_score(d.iq, d.iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1,
                                lim, d.scores, ss, s);
                    if (l.cand_src) {
                        Rows cr;
                        for (int t = 0; t < T; ++t) {
                            cr.n[t] = lim.n[t];
                            cr.k[t] = g.cand_topk;
                        }
                        cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) bm_stride(), d.cand,
                                    (int) cand_stride(), s);
                    }
                    topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, (int) topk_stride(), s);
                }
            }
        }
        if (l.kv_src && ng > 0) {
            Finish fc;
            fc.rd = g.rd;
            fc.pos = (int) (g_first * R);
            fc.pos_step = R;
            fc.freqs = freqs;
            fc.quant = 2;
            fc.pdst = sr.ckv[(size_t) l.dev];
            fc.pdst_row = g_first;
            finish(d.latent, ng, HD, HD, fc, s);
            if (!wp.par && !replicate(sr.ckv, sr.h_ckv + (size_t) r * ckv_rb(), sr.ev_ckv, g_first, ng, l.dev, err))
                return false;
            if (wp.par && mi == 0 && sr.remote)   // the new rows for the other groups' readers
                put_new_rows(sr, l, g_first, ng, sr.x_stage, sr.x_flag, wp.win, d.xcnt, s);
        }
        if (ng > 0) dump_sub(d, "latent", l.il, p0, ng, d.latent, (size_t) ng * HD * 4);
        dump_sub(d, "topk", l.il, p0, T, d.topk, (size_t) T * topk_stride() * 4);
        for (int t = 0; t < T; ++t) ar.nsel[t] = lim.n[t] > 0 ? wp.topk[l.il][r + t] : 0;
        ckv = sr.ckv[(size_t) l.dev];
    }
    if (l.mtp) {   // the block's rows (BF16), after the ring's in ext, as every query's selection
        ckv = make_kv_rows((uint8_t*) (d.ext + (size_t) g.win * HD), kKvBf16, HD);
        ckv_fmt = kKvBf16;
        iota_rows(d.topk, T, T, (int) topk_stride(), s);
    }
    if (NHL > 0 && batch != nullptr) {
        const int r0 = active_row;
        sparse_attn_unpack(ckv, d.topk, (int) topk_stride(), ar, T, d.attn_scr, r0, s);
        for (int t = 0; t < T; ++t) {   // (batch's win0: rows of the whole ext, not of this run's slice)
            batch->win0[r0 + t] = ar.win0[t] + r0 * ext_slice_rows();
            batch->nwin[r0 + t] = ar.nwin[t];
            batch->nsel[r0 + t] = ar.nsel[t];
        }
    } else if (NHL > 0) {
        sparse_attn(d.q, d.ext, ckv, d.topk, (int) topk_stride(), ar, l.sink, NHL, HD, T,
                    1.0f / std::sqrt((float) HD), d.o, s, ckv_fmt, 0, &d.attn_scr);
        dump_sub(d, "o", l.il, p0, T, d.o, (size_t) T * HQ * 4);
        finish_attention(l, d, d.o, p0, T);
    }
    return true;
}

void Model::Impl::finish_attention(const Layer& l, Dev& d, float* o, int64_t p0, int T, const int* rpos) {
    Finish fo = get_rope(l, d, l.ahn, p0, T, rpos);
    fo.inverse = 1;
    fo.quant = 1;
    finish(o, T * l.ahn, g.hd, g.hd, fo, d.st);
}

bool Model::Impl::project_indexer_queries(const Layer& l, Dev& d, int64_t p0, int T, const int* rpos,
                                          std::string& err) {
    if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err))
        return false;
    Finish fi = get_rope(l, d, g.ix_heads, p0, T, rpos);
    fi.quant = 3;
    finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, d.st);
    gemv_f16_rows(l.i_wproj, d.x, (size_t) g.dim, g.ix_heads, g.dim, T,
                  1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, d.st);
    return true;
}

Finish Model::Impl::get_rope(const Layer& l, const Dev& d, int vpp, int64_t p0, int T, const int* rpos) const {
    Finish f;
    f.rd = g.rd;
    f.pos = (int) p0;
    f.vpp = vpp;
    f.freqs = l.ratio ? d.freqs_comp : d.freqs_win;
    if (rpos != nullptr) {
        f.n_rpos = T;
        std::copy_n(rpos, T, f.rpos);
    }
    return f;
}

bool Model::Impl::run_moe(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
    Dev& d = devs[(size_t) member(L0, mi).dev];
    Layer& l = member(L0, mi);
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, H = g.hc, FF = l.ffn;
    // the MoE
    hc_mix_split(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, d.hccnt, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps,
                 d.f_pre, d.f_post, d.f_comb, s);
    hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);
    dump_sub(d, "f_pre", l.il, p0, T, d.f_pre, (size_t) T * 8 * 4);
    dump_sub(d, "f_post", l.il, p0, T, d.f_post, (size_t) T * 8 * 4);
    dump_sub(d, "f_comb", l.il, p0, T, d.f_comb, (size_t) T * 64 * 4);
    dump_sub(d, "ffn_x", l.il, p0, T, d.x, (size_t) T * D * 4);
    if (wp.async) {
        // the routing goes to the service thread through the mailbox (a group's lead publishes it, the members
        // make the same plan) while the device runs the shared expert and the VRAM experts (from the plan the
        // router made), and waits for the answer only before the experts' sum
        gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
        const unsigned sq = wp.seq[l.il];
        const int slot = (int) (sq % kRing);
        const bool lead = mi == 0;
        route_publish(d.glog, l.gate_bias, l.n_exp, l.k_exp, l.k_model, T, g.route_scale, l.vram_tab, d.x, D,
                      d.ridx, d.rw, d.plan_ptr, lead ? reqs + slot : nullptr, sq, s, l.gate_bias_vl,
                      l.mtp ? 0u : wp.vl, 0, 0, get_route_prior(l));
        if (mi == 0 && !l.mtp)
            strata::kernels::count_routing(d.ridx, T, l.k_exp, kPicks, l.il, s);
        if (!mv(d, {job(l.s1, d.x, d.sg, T, D, FF), job(l.s3, d.x, d.su, T, D, FF)}, err)) return false;
        swiglu(d.sg, d.su, T * FF, g.lim, d.sh, s);
        if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, FF, D)}, err)) return false;
        prof_mark(d, 2);
        const x3::ExpertLayout& lay = l.lay;
        if (T == 1 && moe_token_path) {
            x3::MoeArgs ma;
            ma.plan_ptr = d.plan_ptr;
            ma.plan_w = d.rw;
            ma.cpu_part = lead ? resps[slot].part[0] : nullptr;
            ma.cpu_flag = lead ? &resps[slot].cpu[0] : nullptr;
            ma.lay = lay;
            ma.k = l.k_exp;
            ma.n_embd = D;
            ma.n_ff = FF;
            ma.limit = g.lim;
            const bool mx = lay.fmt == gg::kFmtMXFP4;
            // DS4_TLOG2: the expert stage's three host spans (gate/up issue, the CPU-lane wait, down issue)
            static const bool tlog2 = ds4_dbg("DS4_TLOG2");
            static double t_moe_gu = 0, t_moe_wait = 0, t_moe_dn = 0; static long t_moe_n = 0; static bool t_moe_reg = false;
            if (tlog2 && !t_moe_reg) {
                t_moe_reg = true;
                std::atexit([] {
                    std::fprintf(stderr, "tlog2: %ld stages | host ms a stage: gate_up %.2f  cpu_wait %.2f  down %.2f\n",
                                 t_moe_n, t_moe_n ? t_moe_gu / (double) t_moe_n : 0.0,
                                 t_moe_n ? t_moe_wait / (double) t_moe_n : 0.0,
                                 t_moe_n ? t_moe_dn / (double) t_moe_n : 0.0);
                });
            }
            const double m0 = tlog2 ? now_ms() : 0.0;
            if (mx) gg::moe_gate_up(ma, d.x, d.xg, s);
            else x3::moe_gate_up(ma, d.x, d.xg, s);
            const double m1 = tlog2 ? now_ms() : 0.0;
            prof_mark(d, 3);
            if (lead) wait_seq(&resps[slot].seq, sq, s);
            const double m2 = tlog2 ? now_ms() : 0.0;
            prof_mark(d, 4);
            if (mx) gg::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
            else x3::moe_down(ma, d.xg, d.sh_out, d.moe_out, s);
            if (tlog2) { t_moe_gu += m1 - m0; t_moe_wait += m2 - m1; t_moe_dn += now_ms() - m2; ++t_moe_n; }
        } else {   // each distinct expert read once for the window's tokens
            gf::MoeWin mw;
            mw.plan_ptr = d.plan_ptr;
            mw.T = T;
            mw.k = l.k_exp;
            mw.grp_ptr = d.grp_ptr;
            mw.grp_start = d.grp_start;
            mw.grp_ent = d.grp_ent;
            mw.n_groups = d.n_groups;
            gf::moe_win_group(mw, s);
            x3::WinArgs wa;
            wa.grp_ptr = d.grp_ptr;
            wa.grp_start = d.grp_start;
            wa.grp_ent = d.grp_ent;
            wa.n_groups = d.n_groups;
            wa.plan_ptr = d.plan_ptr;
            wa.plan_w = d.rw;
            wa.cpu_part = lead ? &resps[slot].part[0][0] : nullptr;
            wa.cpu_stride = kMaxDim;
            wa.cpu_flag = lead ? resps[slot].cpu : nullptr;
            wa.wait_ptr = lead ? &resps[slot].seq : nullptr;
            wa.wait_val = sq;
            wa.x = d.x;
            wa.x_stride = (size_t) D;
            wa.h = d.xg;
            wa.part = d.ed;
            wa.sh_out = d.sh_out;
            wa.out = d.moe_out;
            wa.out_stride = (size_t) D;
            wa.lay = lay;
            wa.T = T;
            wa.k = l.k_exp;
            wa.n_embd = D;
            wa.n_ff = FF;
            wa.limit = g.lim;
            if (!(lay.fmt == gg::kFmtMXFP4 ? gg::win_experts(wa, s) : x3::win_experts(wa, s))) {
                err = "exl3 win_experts failed";
                return false;
            }
        }
        if (lead) st.experts += (int64_t) T * l.k_exp;
        hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
        std::swap(d.h, d.h2);
        CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        prof_mark(d, 5);
        return true;
    }
    // (glog rows are 1024 apart: route reads them so)
    static const bool phases = ds4_dbg("DS4_PHASES");   // debugging: where a decode layer's host time goes
    static double ph[8] = {0}, ph_prev = 0; static long ph_n = 0;
    const double ph0 = phases ? now_ms() : 0.0;
    gemv_f32(l.gate_w, d.x, l.n_exp, D, T, d.glog, 1024, s);
    VlBits vb = {};
    vb.w[0] = l.mtp ? 0u : wp.vl;
    route(d.glog, l.gate_bias, l.n_exp, l.k_exp, l.k_model, T, g.route_scale, d.ridx, d.rw, s, l.gate_bias_vl,
          vb.w[0] != 0 ? &vb : nullptr, l.vram_tab, get_route_prior(l));
    if (mi == 0 && !l.mtp)
        strata::kernels::count_routing(d.ridx, T, l.k_exp, kPicks, l.il, s);
    CK(cudaMemcpyAsync(h_idx, d.ridx, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_w, d.rw, (size_t) T * kPicks * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_x, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, s));
    strata::core::count_gpu_traffic(d.id, 0, (uint64_t) T * D * 4);
    CK(cudaEventRecord(d.ev, s));
    // the shared expert while the host looks at the picks
    if (!mv(d, {job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)}, err)) return false;
    if (ds4_dbg("DS4_DBG_MOE")) { dbg_nf("x", l.il, d.x, (size_t) T * D); dbg_nf("sg", l.il, d.sg, (size_t) T * g.ff); dbg_nf("su", l.il, d.su, (size_t) T * g.ff);
        std::fprintf(stderr, "DBG s1 fmt=%d K=%d k=%d n=%d ld=%d raw=%p\n", l.s1.fmt, l.s1.K, l.s1.k, l.s1.n, l.s1.ld, (void*) l.s1.raw); }
    swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);
    if (ds4_dbg("DS4_DBG_MOE")) dbg_nf("sh", l.il, d.sh, (size_t) T * g.ff);
    if (!mv(d, {job(l.s2, d.sh, d.sh_out, T, g.ff, D)}, err)) return false;
    const double tw = now_ms();
    CK(cudaEventSynchronize(d.ev));
    st.gpu_wait_ms += now_ms() - tw;
    const double ph2 = phases ? now_ms() : 0.0;
    dump_sub(d, "glog", l.il, p0, T, d.glog, (size_t) T * 1024 * 4);
    dump_sub_host("ridx", l.il, p0, T, h_idx, (size_t) T * kPicks * 4);
    dump_sub_host("rw", l.il, p0, T, h_w, (size_t) T * kPicks * 4);
    dump_sub(d, "sh_out", l.il, p0, T, d.sh_out, (size_t) T * D * 4);
    // the picks by expert: the VRAM ones' rows (gathered inputs, one GEMV per matrix for all their tokens), the
    // CPU lane's entries
    Combine cm;
    Gather ga;
    uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
    int vex[kWin * kPicks], vrow0[kWin * kPicks], vn[kWin * kPicks];
    int nvex = 0, nrows = 0;
    std::vector<Ent> ents;
    std::vector<std::pair<int, float>> cpu_rows;   // slot -> (token, weight)
    std::vector<std::vector<std::pair<int, int>>> per((size_t) l.n_exp);   // expert -> (token, pick)
    bool tail[kWin * kPicks];
    mark_lane_skips(l, h_idx, h_w, T, tail);
    for (int t = 0; t < T; ++t) {
        cm.n[t] = 0;
        cm.cpu[t] = 0;
        for (int i = 0; i < l.k_exp; ++i) {
            const int e = h_idx[t * kPicks + i];
            if (e < 0 || e >= l.n_exp) {
                err = "router picked expert " + std::to_string(e) + " (layer " + std::to_string(l.il) + ", token " + std::to_string(t) + ", pick " + std::to_string(i) + ", of " + std::to_string(l.k_exp) + ")";
                return false;
            }
            // (a non-finite gate input: route_kernel then gave a valid pick but a NaN weight - the layer's output would be
            //  garbage; said here, where the picks are on the host anyway, instead of a request that answers nonsense)
            if (!std::isfinite(h_w[t * kPicks + i])) {
                err = "the router's weight is not finite (layer " + std::to_string(l.il) + ", token " + std::to_string(t) + ", pick " + std::to_string(i) + "): the gate's input went NaN or infinite";
                return false;
            }
            ++pr[e];
            count_usage(l, e);
            if (tail[t * kPicks + i]) {
                ++st.skipped;
                continue;
            }
            per[(size_t) e].push_back({t, i});
        }
    }
    int nslots = 0;
    for (int e = 0; e < l.n_exp; ++e) {
        const auto& pe = per[(size_t) e];
        if (pe.empty()) continue;
        const bool on_gpu = l.slot[(size_t) e] >= 0;
        if (on_gpu) {
            vex[nvex] = e;
            vrow0[nvex] = nrows;
            vn[nvex] = (int) pe.size();
            for (const auto& tp : pe) {
                ga.src[nrows] = tp.first;
                const int t = tp.first;
                cm.row[t][cm.n[t]] = nrows;
                cm.w[t][cm.n[t]] = h_w[t * kPicks + tp.second];
                ++cm.n[t];
                ++nrows;
            }
            ++nvex;
        } else {
            for (size_t a = 0; a < pe.size(); a += (size_t) ent_cap(l)) {
                Ent en;
                en.e = e;
                en.slot0 = nslots;
                en.n = (int) std::min<size_t>((size_t) ent_cap(l), pe.size() - a);
                for (int i = 0; i < en.n; ++i) {
                    en.tok[i] = pe[a + (size_t) i].first;
                    cpu_rows.push_back({pe[a + (size_t) i].first, h_w[pe[a + (size_t) i].first * kPicks + pe[a + (size_t) i].second]});
                    cm.cpu[en.tok[i]] = 1;
                }
                nslots += en.n;
                ents.push_back(en);
            }
        }
    }
    st.experts += (int64_t) T * l.k_exp;
    st.cpu_experts += nslots;
    if (nrows > 0) {
        gather_rows(d.x, ga, nrows, D, d.xe, s);
        dump_sub(d, "xe", l.il, p0, nrows, d.xe, (size_t) nrows * D * 4);
        if (dump_s != nullptr) { std::vector<int> vmap; for (int i = 0; i < nvex; ++i) { vmap.push_back(vex[i]); vmap.push_back(vrow0[i]); vmap.push_back(vn[i]); }
          dump_sub_host("vmap", l.il, p0, nvex, vmap.data(), vmap.size() * 4); }
        x3::MvJob j[2 * kWin * kPicks];
        int nj = 0;
        for (int i = 0; i < nvex; ++i) {
            const auto& vm = l.vm[(size_t) vex[i]];
            j[nj++] = job(vm[0], d.xe + (size_t) vrow0[i] * D, d.xg + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
            j[nj++] = job(vm[1], d.xe + (size_t) vrow0[i] * D, d.xu + (size_t) vrow0[i] * g.ff, vn[i], D, g.ff);
        }
        if (!mv_n(d, j, nj, err)) return false;
        dump_sub(d, "xg", l.il, p0, nrows, d.xg, (size_t) nrows * g.ff * 4);
        dump_sub(d, "xu", l.il, p0, nrows, d.xu, (size_t) nrows * g.ff * 4);
        swiglu(d.xg, d.xu, nrows * g.ff, g.lim, d.xhh, s);
        dump_sub(d, "xhh", l.il, p0, nrows, d.xhh, (size_t) nrows * g.ff * 4);
        nj = 0;
        for (int i = 0; i < nvex; ++i)
            j[nj++] = job(l.vm[(size_t) vex[i]][2], d.xhh + (size_t) vrow0[i] * g.ff, d.ed + (size_t) vrow0[i] * D,
                          vn[i], g.ff, D);
        if (!mv_n(d, j, nj, err)) return false;
        dump_sub(d, "ed", l.il, p0, nrows, d.ed, (size_t) nrows * D * 4);
    }
    const double ph3 = phases ? now_ms() : 0.0;
    if (!ents.empty()) {
        const double tc = now_ms();
        static const bool skip_cpu = ds4_dbg("DS4_SKIP_CPU");   // (timing only: the CPU lane's experts left out)
        if (!skip_cpu) cpu_experts(l, h_x, ents);
        else if (ds4_dbg("DS4_FAKE_CPU_US")) std::this_thread::sleep_for(std::chrono::microseconds(std::atoi(std::getenv("DS4_FAKE_CPU_US"))));   // (timing only)
        std::fill(h_part, h_part + (size_t) T * D, 0.0f);
        for (int sl = 0; sl < nslots; ++sl) {
            float* o = h_part + (size_t) cpu_rows[(size_t) sl].first * D;
            const float wv = cpu_rows[(size_t) sl].second;
            const float* y = c_yd.data() + (size_t) sl * D;
            for (int e = 0; e < D; ++e) o[e] += wv * y[e];
        }
        if (ds4_dbg("DS4_CPU_LOG")) { static double ms = 0; static long calls = 0, exps = 0; static bool reg = false;
          ms += now_ms() - tc; ++calls; exps += (long) ents.size();
          if (ds4_dbg("DS4_CPU_LOG") && !reg) { reg = true; std::atexit([] { std::fprintf(stderr, "cpu lane calls: %ld, experts %ld, %.1f ms (%.3f ms a call, %.3f ms an expert)%c", calls, exps, ms, ms / (double) calls, ms / (double) exps, 10); }); }
          (void) 0; (void) ms; }
        st.cpu_ms += now_ms() - tc;
        dump_sub_host("cpu_part", l.il, p0, T, h_part, (size_t) T * D * 4);
        CK(cudaMemcpyAsync(d.cpu_part, h_part, (size_t) T * D * 4, cudaMemcpyHostToDevice, s));
        strata::core::count_gpu_traffic(d.id, (uint64_t) T * D * 4, 0);
    }
    combine(d.ed, ents.empty() ? nullptr : d.cpu_part, d.sh_out, cm, T, D, d.moe_out, s);
    dump_sub(d, "moe_out", l.il, p0, T, d.moe_out, (size_t) T * D * 4);
    if (ds4_dbg("DS4_DBG_MOE")) {
        float hp = 0.0f;
        for (size_t i = 0; i < (size_t) T * D; ++i) hp = std::max(hp, std::fabs(h_part[i]));
        std::fprintf(stderr, "DBG moe il=%d T=%d vram_rows=%d cpu_slots=%d max|h_part|=%.4g", l.il, T, nrows, nslots, (double) hp);
        dbg_nf("moe_out", l.il, d.moe_out, (size_t) T * D);
        dbg_nf("sh_out", l.il, d.sh_out, (size_t) T * D);
    }
    hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);
    std::swap(d.h, d.h2);
    dump_sub(d, "h_out", l.il, p0, T, d.h, (size_t) T * H * D * 4);
    if (dump_l != nullptr && mi == 0) dump_layer(d, d.h, l.il, p0, T);   // (post-layer residual, as the chunk path writes it)
    CK(cudaMemcpyAsync(d.pre, d.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
    if (phases && T == 1) {
        const double ph4 = now_ms();
        ph[0] += ph2 - ph0; ph[1] += ph3 - ph2; ph[2] += ph4 - ph3; if (ph_prev > 0 && ph0 - ph_prev < 5.0) ph[3] += ph0 - ph_prev; ph_prev = ph4; ++ph_n;
        static bool reg = false;
        if (!reg) { reg = true; std::atexit([] { std::fprintf(stderr, "phases per layer call (%ld): route enqueue + shared expert + wait %.3f ms, routing analysis + GPU experts enqueue %.3f ms, CPU lane + combine + hc_post %.3f ms, attention enqueue %.3f ms%s", ph_n, ph[0] / ph_n, ph[1] / ph_n, ph[2] / ph_n, ph[3] / ph_n, "\n"); }); }
    }
    // (h_part / h_x are written again only after the next layer's route has synchronized, which follows these)
    return true;
}

bool Model::pick_joint_rows(int session, int row, const std::vector<int>& drafts, float t, float tp, int tk,
                            float mp, uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err) {
    Impl& p = *p_;
    const auto it = p.joint_q.find(session);
    p.drafts_q = it != p.joint_q.end() && it->second.size() >= drafts.size() && !drafts.empty();
    if (p.drafts_q) {
        Dev& d = p.devs[(size_t) p.head_dev];
        const strata::core::OnDevice on(d.id);
        if (!strata::core::gpu_copy(p.d_specq, it->second.data(), drafts.size() * sizeof(it->second[0]))) {
            err = "could not restore a conversation's proposal distribution";
            return false;
        }
        p.dq_temp = t;
        p.dq_top_p = tp;
        p.dq_top_k = tk;
        p.dq_min_p = mp;
        p.dq_seed = seed;
        p.dq_counter = counter;
    }
    return p.pick_verified_rows(row, p.row_positions[row], drafts, t, tp, tk, mp, seed, counter, picks, err);
}

bool Model::keep_joint_rows(int row, int count, int keep, std::string& err) {
    Impl& p = *p_;
    return p.keep_window(p.row_positions[row], count, keep, row, err);
}

bool Model::launch_group(int stage, int group, std::string& err) {
    if (p_->launch_group(stage, group, err))
        return true;
    std::string ignored;
    wait_state(ignored);
    return false;
}

bool Model::wait_state(std::string& err) {
    bool ok = true;
    for (size_t di = 0; di < p_->devs.size(); ++di)
        ok = wait_card((int) di, err) && ok;
    return ok;
}

bool Model::wait_card(int dev, std::string& err) {
    Dev& d = p_->devs[(size_t) dev];
    cudaSetDevice(d.id);
    const cudaError_t result = cudaStreamSynchronize(d.st);
    if (result == cudaSuccess)
        return true;
    err = cudaGetErrorString(result);
    return false;
}

}  // namespace strata::ds4
