// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_dspark.cu - the DSpark drafter (DeepSeek's DSparkBlock / forward_spec): its rings seeded from the trunk's taps,
// the drafts of one conversation or of several in one pass, and its weights loaded or freed
// (Model::switch_drafter_weights).
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

bool Model::Impl::seed_drafter(int64_t p0, int n, int row0, std::string& err) {
    if (g.n_mtp == 0 || n <= 0) return true;
    if (n > kWin * kPicks) {
        err = "seeding the drafter with " + std::to_string(n) + " rows";
        return false;
    }
    Dev& d = devs[(size_t) d_dev];
    cudaSetDevice(d.id);
    const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
    if (draft_taps != nullptr)
        for (int row = 0; row < n; ++row) {
            if (!strata::core::gpu_copy_2d_async(
                    draft_taps + (size_t) ((p0 + row) % g.win) * nt * D, 0,
                    h_taps + (size_t) (row0 + row) * nt * D, 0,
                    (size_t) nt * D * sizeof(float), 1, d.st)) {
                err = "retaining recent DSpark inputs failed";
                return false;
            }
        }
    if (!drafter_loaded)
        return true;
    if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
        for (Dev& e : devs) {
            cudaSetDevice(e.id);
            CK(cudaStreamSynchronize(e.st));
        }
        cudaSetDevice(d.id);
        const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
        std::fwrite(hdr, 8, 4, dump_d);
        std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
        std::fflush(dump_d);
    }
    for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
        const int na = std::min(kWin, n - a);
        CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                           cudaMemcpyHostToDevice, d.st));
        if (!mv(d, {job(d_main_proj, d.xe, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
    }
    CK(cudaEventRecord(ev_seed, d.st));
    Finish fm;
    fm.w = d_main_norm;
    fm.eps = g.eps;
    finish(d.ed, n, D, D, fm, d.st);
    for (int i = 0; i < g.n_mtp; ++i) {
        Layer& l = L[(size_t) (g.nl + i)];
        if (l.dev != d_dev) {
            err = "the DSpark blocks are split across devices";
            return false;
        }
        for (int a = 0; a < n; a += kWin) {
            const int na = std::min(kWin, n - a);
            if (!mv(d, {job(l.wkv, d.ed + (size_t) a * D, d.kvr, na, D, HD)}, err)) return false;
            Finish fk;
            fk.w = l.kv_norm;
            fk.eps = g.eps;
            fk.rd = g.rd;
            fk.pos = (int) (p0 + a);
            fk.freqs = d.freqs_win;
            fk.quant = 1;
            fk.dst = d.ext + (size_t) g.win * HD;
            fk.dst_stride = HD;
            finish(d.kvr, na, HD, HD, fk, d.st);
            ext_to_ring(d.ext, l.win, p0 + a, g.win, na, HD, d.st);
        }
    }
    return true;
}

bool Model::Impl::reseed_drafter_from_taps(int64_t position, std::string& err) {
    Dev& d = devs[(size_t) d_dev];
    const strata::core::OnDevice on_device(d.id);
    if (!strata::core::gpu_stream_synchronize(d.st)) {
        err = "waiting to restore DSpark rings failed";
        return false;
    }
    const size_t width = g.d_targets.size() * (size_t) g.dim;
    for (int64_t first = std::max<int64_t>(0, position - g.win); first < position;) {
        const int n = (int) std::min<int64_t>({position - first, kWin, g.win - first % g.win});
        if (!strata::core::gpu_copy(h_taps, draft_taps + (size_t) (first % g.win) * width,
                                    (size_t) n * width * sizeof(float))) {
            err = "reading recent DSpark inputs failed";
            return false;
        }
        if (!seed_drafter(first, n, 0, err))
            return false;
        first += n;
    }
    return true;
}

bool Model::Impl::draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature,
                        float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
    const double t0 = now_ms();
    // (with penalties the drafts are the drafter's argmax: its own draws know no history - verify() is as exact)
    drafts_q = temperature > 0.0f && sample_drafts && pen_n <= 0;
    dq_temp = temperature;
    dq_top_p = top_p;
    dq_top_k = top_k;
    dq_min_p = min_p;
    dq_seed = seed;
    dq_counter = counter;
    const bool ok = draft_(y, out, conf, err);
    if (!ok) drafts_q = false;
    if (max_drafts > 0 && (int) out.size() > max_drafts) {
        out.resize((size_t) max_drafts);
        conf.resize((size_t) max_drafts);
    }
    st.draft_ms += now_ms() - t0;
    ++st.drafts_rounds;
    return ok;
}

bool Model::Impl::draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
    out.clear();
    conf.clear();
    const int T = g.d_block, D = g.dim, H = g.hc;
    const int64_t p0 = pos;
    if (g.n_mtp == 0 || T < 1 || T > kWin || p0 + T > o.max_context) return true;
    int tk[kWin];
    tk[0] = y;
    for (int t = 1; t < T; ++t) tk[t] = g.d_noise;
    Dev& d = devs[(size_t) d_dev];
    cudaSetDevice(d.id);
    for (int t = 0; t < T; ++t) std::memcpy(h_emb + (size_t) t * D, embed_pinned + (size_t) tk[t] * D, (size_t) D * 2);
    CK(emb_upload(d.emb_rows, h_emb, (size_t) T * D * 2, d.st));
    embed(d.emb_rows, T, d.h, D, H, d.st);
    if (ds4_dbg("DS4_DBG_EMB")) {
        float *hp = d.h;
        float hh[4] = {0,0,0,0};
        cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
    }
    if (ds4_dbg("DS4_DBG_EMB")) {
        float hh[4] = {0, 0, 0, 0};
        cudaMemcpy(hh, d.h, sizeof(hh), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "DBG after embed: h[0..3] = %g %g %g %g\n", hh[0], hh[1], hh[2], hh[3]);
    }
    init_pre(d.pre, T, d.st);
    WinPlan dp;   // (window attention alone: the mailbox seqs)
    dp.async = !sync_decode && T <= async_max_t;
    for (int i = 0; i < g.n_mtp && dp.async; ++i) {
        const int il = g.nl + i;
        dp.seq[il] = ++seq_issued;
        layer_of[dp.seq[il] % kRing].store(il, std::memory_order_release);
    }
    const size_t hs = (size_t) T * H * D, rs = (size_t) g.win * g.hd;
    if (dump_d != nullptr && h_dump == nullptr &&
        (!halloc(h_dump, (size_t) g.n_mtp * kWin * H * D, err) || !halloc(h_dump_ring, (size_t) g.n_mtp * rs, err)))
        return false;
    for (int i = 0; i < g.n_mtp; ++i) {
        Layer& l = L[(size_t) (g.nl + i)];
        if (dump_d != nullptr) {   // (stream order: the ring the block reads - as BF16 rows - the stream it leaves)
            ring_to_ext(l.win, d.ext, g.win, g.win, g.hd, d.st);
            CK(cudaMemcpyAsync(h_dump_ring + (size_t) i * rs, d.ext, rs * 2, cudaMemcpyDeviceToHost, d.st));
        }
        if (!layer(l, 0, p0, T, dp, err)) return false;
        if (dump_d != nullptr) {
            cudaSetDevice(d.id);
            CK(cudaMemcpyAsync(h_dump + (size_t) i * hs, d.h, hs * 4, cudaMemcpyDeviceToHost, d.st));
        }
    }
    cudaSetDevice(d.id);
    hc_pre_norm(d.h, d.pre, nullptr, D, H, T, g.eps, d.x, d.st);   // (the confidence head reads it before the norm)
    Dev& hd = devs[(size_t) head_dev];
    if (d_dev != head_dev) {   // (through the host to the head's device)
        CK(cudaMemcpyAsync(d.bounce, d.x, (size_t) T * D * 4, cudaMemcpyDeviceToHost, d.st));
        CK(cudaEventRecord(d.ev_out, d.st));
        cudaSetDevice(hd.id);
        CK(cudaStreamWaitEvent(hd.st, d.ev_out, 0));
        CK(cudaMemcpyAsync(hd.x, d.bounce, (size_t) T * D * 4, cudaMemcpyHostToDevice, hd.st));
    }
    // the drafter's final norm and the trunk's head
    cudaSetDevice(hd.id);
    CK(cudaMemcpyAsync(hd.ao, hd.x, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, hd.st));
    Finish fn;
    fn.w = hd.d_norm;
    fn.eps = g.eps;
    finish(hd.ao, T, D, D, fn, hd.st);
    if (!compute_logits(hd, hd.ao, T, err)) return false;
    h_idx[0] = y;
    if (!sample_draft_rows(0, y, T, err))
        return false;
    if (!read_draft_rows(0, T, err))
        return false;
    int tok[kWin + 1];
    float cf[kWin];
    for (int i = 0; i <= T; ++i) tok[i] = h_idx[i];
    for (int i = 0; i < T; ++i) cf[i] = h_w[i];
    if (dump_d != nullptr) {   // 'D', the block's position, y, T, then the drafts, their confidences, the hidden
        std::vector<float> xs((size_t) T * D);   // rows before the final norm; then each block's ring and stream
        CK(cudaMemcpy(xs.data(), hd.x, xs.size() * 4, cudaMemcpyDeviceToHost));
        cudaSetDevice(d.id);
        CK(cudaStreamSynchronize(d.st));
        const int64_t hdr[4] = {'D', p0, y, T};
        std::fwrite(hdr, 8, 4, dump_d);
        std::fwrite(tok + 1, 4, (size_t) T, dump_d);
        std::fwrite(cf, 4, (size_t) T, dump_d);
        std::fwrite(xs.data(), 4, xs.size(), dump_d);
        for (int i = 0; i < g.n_mtp; ++i) {
            std::fwrite(h_dump_ring + (size_t) i * rs, 2, rs, dump_d);
            std::fwrite(h_dump + (size_t) i * hs, 4, hs, dump_d);
        }
        std::fflush(dump_d);
        cudaSetDevice(hd.id);
    }
    for (int i = 0; i < T; ++i) {
        out.push_back(tok[i + 1]);
        conf.push_back(cf[i]);
    }
    return true;
}

bool Model::Impl::draft_conversations(std::vector<Model::DraftRequest>& requests,
                                      const std::function<void(int)>& select,
                                      const std::function<void(int)>& bind, std::string& err) {
    if (requests.empty())
        return true;
    if (requests.size() == 1) {
        auto& r = requests.front();
        select(0);
        if (!draft(r.token, r.tokens, r.confidence, err, r.temperature, r.top_p, r.top_k, r.min_p,
                   r.seed, r.counter))
            return false;
        const int n = std::min(r.limit, (int) r.tokens.size());
        r.tokens.resize((size_t) n);
        r.confidence.resize((size_t) n);
        return save_joint_draft(r.session, n, err);
    }
    const int block = g.d_block, D = g.dim, H = g.hc;
    if (block < 1 || block > kWin || d_dev != head_dev) {
        err = "batched DSpark needs a layer-parallel drafter and head on the last stage";
        return false;
    }
    const double started = now_ms();
    Dev& d = devs[(size_t) d_dev];
    const strata::core::OnDevice on(d.id);
    const int per_tile = kWin / block;
    const size_t count = (requests.size() + per_tile - 1) / per_tile;
    draft_tiles.resize(std::max(draft_tiles.size(), count));
    struct Restore {
        Dev& d;
        float *h, *h2, *pre;
        std::function<void(int)>& bind;
        ~Restore() {
            d.h = h;
            d.h2 = h2;
            d.pre = pre;
            bind = {};
        }
    } restore{d, d.h, d.h2, d.pre, select_row};
    for (size_t index = 0; index < count; ++index) {
        auto& tile = draft_tiles[index];
        tile.first = (int) index * per_tile;
        tile.count = std::min(per_tile, (int) requests.size() - tile.first);
        if ((!tile.h && !dalloc(d, tile.h, (size_t) kWin * H * D, err)) ||
            (!tile.h2 && !dalloc(d, tile.h2, (size_t) kWin * H * D, err)) ||
            (!tile.pre && !dalloc(d, tile.pre, (size_t) kWin * 8, err)) ||
            (!tile.embeddings && !halloc(tile.embeddings, (size_t) kWin * D, err)))
            return false;
        const int T = tile.count * block;
        for (int row = 0; row < T; ++row) {
            const auto& r = requests[(size_t) tile.first + row / block];
            const int token = row % block == 0 ? r.token : g.d_noise;
            std::memcpy(tile.embeddings + (size_t) row * D, embed_pinned + (size_t) token * D,
                        (size_t) D * 2);
        }
        if (!strata::core::gpu_copy_2d_async(d.emb_rows, 0, tile.embeddings, 0,
                                            (size_t) T * D * 2, 1, d.st)) {
            err = "copying batched DSpark inputs failed";
            return false;
        }
        embed(d.emb_rows, T, tile.h, D, H, d.st);
        init_pre(tile.pre, T, d.st);
    }
    for (int layer_index = 0; layer_index < g.n_mtp; ++layer_index) {
        Layer& l = L[(size_t) g.nl + layer_index];
        for (size_t index = 0; index < count; ++index) {
            auto& tile = draft_tiles[index];
            const int T = tile.count * block;
            d.h = tile.h;
            d.h2 = tile.h2;
            d.pre = tile.pre;
            std::fill(std::begin(row_counts), std::end(row_counts), 0);
            for (int row = 0; row < T; ++row) {
                row_positions[row] = requests[(size_t) tile.first + row / block].pos + row % block;
                if (row % block == 0) {
                    row_counts[row] = block;
                    row_plans[row] = {};
                }
            }
            select_row = [&](int row) { bind(tile.first + row / block); };
            WinPlan plan;
            plan.async = !sync_decode;
            if (plan.async) {
                plan.seq[l.il] = ++seq_issued;
                layer_of[seq_issued % kRing].store(l.il, std::memory_order_release);
            }
            if (!layer(l, 0, 0, T, plan, err))
                return false;
            tile.h = d.h;
            tile.h2 = d.h2;
        }
    }
    select_row = {};
    for (size_t index = 0; index < count; ++index) {
        auto& tile = draft_tiles[index];
        const int T = tile.count * block;
        hc_pre_norm(tile.h, tile.pre, nullptr, D, H, T, g.eps, d.x, d.st);
        if (!strata::core::gpu_copy_2d_async(d.ao, 0, d.x, 0, (size_t) T * D * 4, 1, d.st)) {
            err = "copying batched DSpark head inputs failed";
            return false;
        }
        Finish norm;
        norm.w = d.d_norm;
        norm.eps = g.eps;
        finish(d.ao, T, D, D, norm, d.st);
        if (!compute_logits(d, d.ao, T, err))
            return false;
        bool sampled = false;
        for (int item = 0; item < tile.count; ++item) {
            const auto& r = requests[(size_t) tile.first + item];
            drafts_q = r.temperature > 0 && sample_drafts;
            sampled = sampled || drafts_q;
            dq_temp = r.temperature;
            dq_top_p = r.top_p;
            dq_top_k = r.top_k;
            dq_min_p = r.min_p;
            dq_seed = r.seed;
            dq_counter = r.counter;
            const int row = item * block, ints = item * (block + 1);
            if (!sample_draft_rows(row, r.token, block, err, ints) ||
                !strata::core::gpu_copy_2d_async(d.tok + row, 0, d_ints + ints, 0,
                                                (size_t) block * sizeof(int), 1, d.st)) {
                if (err.empty())
                    err = "packing batched DSpark confidence inputs failed";
                return false;
            }
        }
        confidence(d.x, d_mk_embed, d.tok, d_conf, T, D, g.d_rank, d_confv, d.st);
        if (!strata::core::gpu_copy_2d_async(h_idx, 0, d_ints, 0,
                (size_t) tile.count * (block + 1) * sizeof(int), 1, d.st) ||
            !strata::core::gpu_copy_2d_async(h_w, 0, d_confv, 0, (size_t) T * sizeof(float), 1, d.st) ||
            (sampled && !strata::core::gpu_copy_2d_async(h_specq.data(), 0, d_specq, 0,
                (size_t) T * sizeof(h_specq[0]), 1, d.st)) ||
            !strata::core::gpu_stream_synchronize(d.st)) {
            err = "reading batched DSpark outputs failed";
            return false;
        }
        for (int item = 0; item < tile.count; ++item) {
            auto& r = requests[(size_t) tile.first + item];
            const int row = item * block, ints = item * (block + 1);
            const int n = std::min({r.limit, block, max_drafts > 0 ? max_drafts : block});
            r.tokens.assign(h_idx + ints + 1, h_idx + ints + 1 + n);
            r.confidence.assign(h_w + row, h_w + row + n);
            auto& q = joint_q[r.session];
            q.clear();
            if (r.temperature > 0 && sample_drafts)
                q.assign(h_specq.begin() + row, h_specq.begin() + row + n);
        }
    }
    st.draft_ms += now_ms() - started;
    ++st.drafts_rounds;
    return true;
}

bool Model::Impl::sample_draft_rows(int row0, int y, int T, std::string& err, int int_offset) {
    Dev& hd = devs[(size_t) head_dev];
    const int V = g.vocab, R = g.d_rank;
    h_idx[int_offset] = y;
    int* ints = d_ints + int_offset;
    cudaSetDevice(hd.id);
    CK(cudaMemcpyAsync(ints, h_idx + int_offset, 4, cudaMemcpyHostToDevice, hd.st));
    strata::dsv41k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    if (drafts_q) {   // Proposal draws use the same counter positions as verification.
        sp.greedy = false;
        sp.temperature = dq_temp * (q_temp_scale > 0.0f ? q_temp_scale : 1.0f);
        sp.top_p = dq_top_p;
        sp.top_k = dq_top_k;
        sp.min_p = dq_min_p;
        sp.seed = dq_seed;
    }
    for (int i = 0; i < T; ++i) {   // the markov bias of the token before, then that position's draft
        float* row = hd.logits + (size_t) (row0 + i) * V;
        if (d_mk_head_q8 != nullptr)
            markov_add_q8(row, d_mk_embed, d_mk_head_q8, ints + i, V, R, hd.st);
        else
            markov_add(row, d_mk_embed, d_mk_head, ints + i, V, R, hd.st);
        if (drafts_q) {
            sp.counter = dq_counter + (uint64_t) i;
            strata::dsv41k::sample_draft(row, 1, V, sp, d_specq + row0 + i, ints + i + 1, hd.st);
        } else
            strata::dsv41k::sample_tokens(row, 1, V, nullptr, 0, sp, ints + i + 1, hd.st);
    }
    return true;
}

bool Model::Impl::read_draft_rows(int row0, int T, std::string& err) {
    Dev& hd = devs[(size_t) head_dev];
    confidence(hd.x + (size_t) row0 * g.dim, d_mk_embed, d_ints, d_conf, T, g.dim, g.d_rank,
               d_confv, hd.st);
    CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
    CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
    CK(cudaStreamSynchronize(hd.st));
    return true;
}

bool Model::Impl::save_joint_draft(int session, int n, std::string& err) {
    auto& q = joint_q[session];
    q.clear();
    if (!drafts_q || n <= 0)
        return true;
    q.resize((size_t) n);
    Dev& d = devs[(size_t) head_dev];
    const strata::core::OnDevice on(d.id);
    if (!strata::core::gpu_stream_synchronize(d.st) ||
        !strata::core::gpu_copy(q.data(), d_specq, q.size() * sizeof(q[0]))) {
        err = "could not preserve a conversation's proposal distribution";
        return false;
    }
    return true;
}

bool Model::switch_drafter_weights(bool on, std::string& err) {
    Impl& p = *p_;
    if (on == p.drafter_loaded)
        return true;
    if (on) {
        for (Dev& d : p.devs)
            d.draft_allocations.clear();
        for (size_t stage = 0; stage < p.devs.size(); ++stage)
            if (!p.load_dense((int) stage, err, true)) {
                p.devs[stage].loading_drafter = false;
                p.drafter_loaded = true;
                std::string ignored;
                switch_drafter_weights(false, ignored);
                return false;
            }
        p.drafter_loaded = true;
        return true;
    }
    for (Dev& d : p.devs) {
        const strata::core::OnDevice on_device(d.id);
        if (!strata::core::gpu_stream_synchronize(d.st)) {
            err = "waiting for DSpark's device work failed";
            return false;
        }
        for (auto& [address, bytes] : d.draft_allocations) {
            if (address == nullptr)
                continue;
            strata::core::gpu_free(address);
            d.allocs.erase(std::remove(d.allocs.begin(), d.allocs.end(), address), d.allocs.end());
            d.weights -= bytes;
            address = nullptr;
        }
    }
    p.drafter_loaded = false;
    return true;
}

}  // namespace strata::ds4
