// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_cpu_lane.cu - the CPU lane: the routed experts no GPU holds, computed by the host's worker pools (cpu_experts),
// the service thread that answers each layer's routing during the decode (serve_lane), the automatic switch-off of the
// SSD skip, the hop of other cards' experts (run_expert_hop) and Model::measure_cpu_expert_ms.
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

void Model::Impl::skip_auto_off(const char* why, double share) {
    SkipAuto& a = sk_auto;
    a.on.store(false);
    st.skip_on = 0;
    ++st.skip_switches;
    std::fprintf(stderr, "ds4: DISK_SKIP auto: off after %lld tokens (%s; cold share %.3f, the lowest %.3f)\n",
                 (long long) a.since, why, share, a.low < 1.5 ? a.low : share);
}

void Model::Impl::skip_auto_tick(int li, int T, int heavy, int cold, int heavy_cpu) {
    SkipAuto& a = sk_auto;
    // (the window in tokens, a window of T tokens as T of them: the same switch whatever the windows' size)
    const int64_t cap = (int64_t) disk_skip_window * std::max(1, g.nl - disk_skip_from);
    a.ring.push_back({heavy, cold, T, heavy_cpu});
    a.H += heavy, a.C += cold, a.Tn += T, a.P += heavy_cpu;
    while (a.Tn - a.ring.front()[2] >= cap) {
        a.H -= a.ring.front()[0], a.C -= a.ring.front()[1], a.Tn -= a.ring.front()[2], a.P -= a.ring.front()[3];
        a.ring.pop_front();
    }
    if (li != disk_skip_from) return;
    if (!a.on.load()) return;
    a.since += T;
    const double share = a.H > 0 ? (double) a.C / (double) a.H : 0.0;
    st.skip_cold = share;
    if (a.since >= disk_skip_budget) return skip_auto_off("its budget", share);
    if (a.Tn < cap || a.H <= 0) return;
    if (share < a.low - 0.05 * a.low) a.low = share, a.low_at = a.since;
    if (share <= 0.02) return skip_auto_off("under 2%", share);
    if (a.since - a.low_at >= disk_skip_settle) return skip_auto_off("no longer falling", share);
}

void Model::Impl::cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx,
                                 const int* ord) {
    if (!gg::mx_experts(l.lay)) return cpu_experts_iq(l, x, ents, ldx, ord);
    constexpr int RB = 64;   // rows a job (whole blocks of h: RB % 32 == 0)
    const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
    const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
    const size_t gr = gg::mx_row_bytes(E), dr = gg::mx_row_bytes(FF);
    const size_t xa = gg::mx_act_bytes(E), ha = gg::mx_act_bytes(FF);
    const bool band_q = FFh % 32 == 0;   // (each job's rows whole blocks of h)
    const float lim = g.lim;
    static_assert(kc::kExl3CpuMaxN <= gg::kMxCpuMaxN, "a CPU-lane entry's tokens in one call");
    // the int8 inputs in c_xh's bytes: token t's (t < kWin) at t xa, then the slots' h at (kWin + slot) ha; a token
    // index past the window (none on the decode or chunk paths) gets a row of its own after them
    uint8_t* xq = (uint8_t*) c_xh.data();
    uint8_t* hq = xq + (size_t) kWin * xa;
    uint8_t* xq_extra = hq + (size_t) kSlots * ha;
    auto x_act = [&](const Ent& en, int i) -> const void* {
        const int t = en.tok[i];
        return t < kWin ? xq + (size_t) t * xa : xq_extra + (size_t) (en.slot0 + i) * xa;
    };
    {
        bool done[kWin] = {};
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i) {
                const int t = en.tok[i];
                if (t < kWin && done[t]) continue;
                if (t < kWin) done[t] = true;
                gg::mx_quant(x + (size_t) t * ldx, E, (void*) x_act(en, i));
            }
    }
    std::function<void(int)> f[4];
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
            const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
            const void* ap[gg::kMxCpuMaxN];
            float yg[gg::kMxCpuMaxN][RB], yu[gg::kMxCpuMaxN][RB];
            float *gp[gg::kMxCpuMaxN], *up[gg::kMxCpuMaxN];
            for (int i = 0; i < en.n; ++i) {
                ap[i] = x_act(en, i);
                gp[i] = yg[i];
                up[i] = yu[i];
            }
            // (part X's rows, from where they are: a copy's part on node X, or the file's)
            gg::mx_rows(mx_part_ptr(l, en.e, 0, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, gp);
            gg::mx_rows(mx_part_ptr(l, en.e, 1, X), gr, E, r0 - X * FFh, r1 - X * FFh, ap, en.n, up);
            for (int i = 0; i < en.n; ++i) {
                float* h = band_q ? yg[i] : c_xhd.data() + (size_t) (en.slot0 + i) * FF + r0;
                for (int r = r0; r < r1; ++r) {
                    float gv = yg[i][r - r0], uv = yu[i][r - r0];
                    if (lim > 0.0f) {
                        gv = std::min(gv, lim);
                        uv = std::min(std::max(uv, -lim), lim);
                    }
                    h[r - r0] = gv / (1.0f + std::exp(-gv)) * uv;
                }
                if (band_q) gg::mx_quant_blocks(h, r0 / 32, r1 / 32, hq + (size_t) (en.slot0 + i) * ha, FF);
            }
        };
    run_nodes(ne * GB, f);
    if (!band_q)
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i)
                gg::mx_quant(c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF, hq + (size_t) (en.slot0 + i) * ha);
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
            const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
            const void* ap[gg::kMxCpuMaxN];
            float* yp[gg::kMxCpuMaxN];
            for (int i = 0; i < en.n; ++i) {
                ap[i] = hq + (size_t) (en.slot0 + i) * ha;
                yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + r0;
            }
            gg::mx_rows(mx_part_ptr(l, en.e, 2, X), dr, FF, r0 - X * Eh, r1 - X * Eh, ap, en.n, yp);
        };
    run_nodes(ne * DB, f);
}

void Model::Impl::cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx,
                                 const int* ord) {
    constexpr int RB = 64;   // rows a job
    const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH;
    const int GB = (FFh + RB - 1) / RB, DB = (Eh + RB - 1) / RB, ne = (int) ents.size();
    gg::IqCpu fm;
    std::string ferr;
    if (!gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, ferr) ||
        fm.act_bytes > (size_t) E * 4 || fm.h_bytes > (size_t) E * 4) {
        // (a layer that cannot be computed here and has no GPU kernel is refused at load: if one gets here anyway,
        // stop rather than answer zeros; Project Maya #620)
        std::fprintf(stderr, "ds4: layer %d's CPU-lane experts: %s (cannot compute them; aborting)\n", l.il,
                     ferr.empty() ? "activation rows too large" : ferr.c_str());
        std::fflush(stderr);
        std::abort();
    }
    static double t_pre = 0, t_q1 = 0, t_r1 = 0, t_q2 = 0, t_r2 = 0; static long n_calls = 0; static bool reg = false;
    if (ds4_dbg("DS4_CPU_LOG") && !reg) { reg = true; std::atexit([] { std::fprintf(stderr, "cpu_experts_iq phases (ms over %ld calls): prefetch %.1f, quant1 %.1f, gate_up %.1f, quant2 %.1f, down %.1f\n", n_calls, t_pre, t_q1, t_r1, t_q2, t_r2); }); }
    const bool cpulog = ds4_dbg("DS4_CPU_LOG");
    if (cpulog) ++n_calls;
    double tp0 = now_ms();
    const float lim = g.lim;
    double tp1 = now_ms(); if (cpulog) t_pre += tp1 - tp0;
    // the activations in c_xh's bytes, E 4 apart: token t's (t < kWin) at row t - quantized once, whatever number of
    // entries take it - then a row a slot (kWin + slot): a token past the window's input (none on the decode or
    // chunk paths), then (x no longer needed) the slot's h
    uint8_t* aq = (uint8_t*) c_xh.data();
    auto x_act = [&](const Ent& en, int i) -> uint8_t* {
        const int t = en.tok[i];
        return aq + (size_t) (t < kWin ? t : kWin + en.slot0 + i) * E * 4;
    };
    {
        bool done[kWin] = {};
        for (const Ent& en : ents)
            for (int i = 0; i < en.n; ++i) {
                const int t = en.tok[i];
                if (t < kWin && done[t]) continue;
                if (t < kWin) done[t] = true;
                gg::iq_cpu_quant(fm.gu_act, x + (size_t) t * ldx, E, x_act(en, i));
            }
    }
    double tq1 = now_ms(); if (cpulog) t_q1 += tq1 - tp1;
    std::function<void(int)> f[4];
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / GB] : j / GB)];
            const int r0 = X * FFh + (j % GB) * RB, r1 = std::min(X * FFh + FFh, r0 + RB);
            const void* ap[kEntMax];
            float* hp[kEntMax];
            for (int i = 0; i < en.n; ++i) {
                ap[i] = x_act(en, i);
                hp[i] = c_xhd.data() + (size_t) (en.slot0 + i) * FF + (size_t) X * FFh;   // (part X's rows of h)
            }
            // (part X's rows, from where they are: a copy's part on node X, or the file's - whole rows, so any
            // block size of the type)
            gg::iq_cpu_gate_up(fm, mx_part_ptr(l, en.e, 0, X), mx_part_ptr(l, en.e, 1, X), ap, en.n, hp,
                               r0 - X * FFh, r1 - X * FFh, lim);
        };
    run_nodes(ne * GB, f);
    double tr1 = now_ms(); if (cpulog) t_r1 += tr1 - tq1;
    for (const Ent& en : ents)
        for (int i = 0; i < en.n; ++i)
            gg::iq_cpu_quant(fm.d_act, c_xhd.data() + (size_t) (en.slot0 + i) * FF, FF,
                             aq + (size_t) (kWin + en.slot0 + i) * E * 4);
    double tq2 = now_ms(); if (cpulog) t_q2 += tq2 - tr1;
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (ord != nullptr ? ord[j / DB] : j / DB)];
            const int r0 = X * Eh + (j % DB) * RB, r1 = std::min(X * Eh + Eh, r0 + RB);
            const void* hp[kEntMax];
            float* yp[kEntMax];
            for (int i = 0; i < en.n; ++i) {
                hp[i] = aq + (size_t) (kWin + en.slot0 + i) * E * 4;
                yp[i] = c_yd.data() + (size_t) (en.slot0 + i) * E + (size_t) X * Eh;   // (part X's rows of it)
            }
            gg::iq_cpu_down(fm, mx_part_ptr(l, en.e, 2, X), hp, en.n, yp, r0 - X * Eh, r1 - X * Eh);
        };
    run_nodes(ne * DB, f);
    if (cpulog) t_r2 += now_ms() - tq2;
    {   // DS4_PROF: where the CPU lane's time goes (act prep + prefetch, gate/up, h quant, down), summed
        static const bool on = std::getenv("DS4_PROF") != nullptr;
        if (on) {
            static double acc[6] = {};
            const double te = now_ms();
            acc[0] += tq1 - tp0; acc[1] += tr1 - tq1; acc[2] += tq2 - tr1; acc[3] += te - tq2; acc[4] += 1; acc[5] += ne;
            if (((long long) acc[4]) % 1000 == 0)
                std::fprintf(stderr, "ds4 prof CPU lane (IQ rows) per call: n %.0f, entries/call %.2f, prep %.3f ms, gate+up %.3f ms, h-quant %.3f ms, down %.3f ms\n", acc[4], acc[5] / acc[4], acc[0] / acc[4], acc[1] / acc[4], acc[2] / acc[4], acc[3] / acc[4]);
        }
    }
}

bool Model::Impl::run_expert_hop(const Layer& l, const float* x, size_t ldx, std::vector<Ent>& cold,
                                 std::string& err) {
    if (hop_holders == nullptr || l.mtp || hop_holders[l.il] < 0)
        return true;
    ExpertHop& hop = *hops[(size_t) hop_holders[l.il]];
    const strata::core::OnDevice on(hop.d.id);
    Dev& d = hop.d;
    std::vector<Ent> remote;
    std::vector<float> input;
    Layer view;
    view.lay = l.lay;
    view.vm.resize((size_t) l.n_exp);
    for (const Ent& en : cold) {
        const uint8_t* blob = hop.find(l.il, en.e);
        if (blob == nullptr)
            continue;
        set_vm(view, en.e, blob);
        remote.push_back(en);
        for (int i = 0; i < en.n; ++i)
            input.insert(input.end(), x + (size_t) en.tok[i] * ldx, x + (size_t) en.tok[i] * ldx + g.dim);
    }
    if (remote.empty())
        return true;
    const double start = now_ms();
    CK(cudaMemcpyAsync(d.xe, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, d.st));
    std::vector<x3::MvJob> jobs;
    int rows = 0;
    for (const Ent& en : remote) {
        const auto& vm = view.vm[(size_t) en.e];
        jobs.push_back(job(vm[0], d.xe + (size_t) rows * g.dim, d.xg + (size_t) rows * g.ff,
                           en.n, g.dim, g.ff));
        jobs.push_back(job(vm[1], d.xe + (size_t) rows * g.dim, d.xu + (size_t) rows * g.ff,
                           en.n, g.dim, g.ff));
        rows += en.n;
    }
    if (!mv_n(d, jobs.data(), (int) jobs.size(), err))
        return false;
    swiglu(d.xg, d.xu, rows * g.ff, g.lim, d.xhh, d.st);
    jobs.clear();
    rows = 0;
    for (const Ent& en : remote) {
        jobs.push_back(job(view.vm[(size_t) en.e][2], d.xhh + (size_t) rows * g.ff,
                           d.ed + (size_t) rows * g.dim, en.n, g.ff, g.dim));
        rows += en.n;
    }
    if (!mv_n(d, jobs.data(), (int) jobs.size(), err))
        return false;
    std::vector<float> output((size_t) rows * g.dim);
    CK(cudaMemcpyAsync(output.data(), d.ed, output.size() * sizeof(float), cudaMemcpyDeviceToHost, d.st));
    strata::core::count_gpu_traffic(d.id, input.size() * sizeof(float), output.size() * sizeof(float));
    CK(cudaStreamSynchronize(d.st));
    rows = 0;
    for (const Ent& en : remote) {
        std::copy_n(output.data() + (size_t) rows * g.dim, (size_t) en.n * g.dim,
                    c_yd.data() + (size_t) en.slot0 * g.dim);
        rows += en.n;
    }
    cold.erase(std::remove_if(cold.begin(), cold.end(), [&](const Ent& en) {
        return hop.find(l.il, en.e) != nullptr;
    }), cold.end());
    hop.experts += remote.size();
    hop.rows += rows;
    hop.ms += now_ms() - start;
    return true;
}

void Model::Impl::cpu_experts_paged(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx) {
    std::vector<Ent> local = ents;
    std::string why;
    if (!run_expert_hop(l, x, ldx, local, why)) {
        std::fprintf(stderr, "strata-ds41: expert hop failed, computing on CPU: %s\n", why.c_str());
        local = ents;
    }
    if (local.empty())
        return;
    strata::core::ExpertSource& pool = *o.expert_pages;
    std::vector<int32_t> ids;
    for (const Ent& en : local)
        ids.push_back(en.e);
    pool.begin_layer(l.il, ids.data(), (int64_t) ids.size());
    for (size_t a = 0; a < local.size();) {
        std::vector<Ent> group, lost;
        std::vector<int64_t> want;
        for (; a < local.size(); ++a) {
            const bool known = std::find(want.begin(), want.end(), local[a].e) != want.end();
            if (!known && want.size() == (size_t) kCpuLaneExperts)
                break;
            if (!known)
                want.push_back(local[a].e);
            group.push_back(local[a]);
        }
        pool.prefetch(l.il, want.data(), (int64_t) want.size());
        for (const int64_t e : want)
            held[(size_t) e] = pool.blob(l.il, e);
        const auto is_lost = [&](const Ent& en) { return held[(size_t) en.e] == nullptr; };
        std::copy_if(group.begin(), group.end(), std::back_inserter(lost), is_lost);
        group.erase(std::remove_if(group.begin(), group.end(), is_lost), group.end());
        if (l.lay.fmt == gg::kFmtMXFP4)
            cpu_experts_mx(l, x, group, ldx, nullptr);
        else
            cpu_experts_exl3(l, x, group, ldx);
        for (const Ent& en : lost) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "ds4: layer %d expert %d could not be read: its rows are left out\n", l.il,
                             en.e);
            std::memset(c_yd.data() + (size_t) en.slot0 * g.dim, 0, (size_t) en.n * g.dim * 4);
        }
    }
}

void Model::Impl::cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx) {
    if (ldx == 0) ldx = (size_t) g.dim;
    // an expert with more tokens than an entry holds takes adjacent entries (ent_cap)
    int64_t count = 0;
    for (size_t i = 0; i < ents.size(); ++i)
        count += i == 0 || ents[i].e != ents[i - 1].e;
    cpu_expert_count += count;
    cpu_experts_paged(l, x, ents, ldx);
}

void Model::Impl::cpu_experts_exl3(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx) {
    const int E = g.dim, FF = g.ff, FFh = FF / NH, Eh = E / NH, GQ = E / 256, DQ = FF / 128;
    const int GB = FFh / 128, DB = Eh / 128, EB = E / 128, FB = FF / 128;
    const int ne = (int) ents.size();
    const float lim = g.lim;
    auto part = [&](int e, int r) {   // (every part X of an expert: its whole blob, NH 1)
        const uint8_t* b = held[(size_t) e];
        kc::Exl3Part p;
        p.trellis = (const uint32_t*) (b + l.hl.trellis[r]);
        p.suh = (const uint16_t*) (b + l.hl.suh[r]);
        p.svh = (const uint16_t*) (b + l.hl.svh[r]);
        p.k = l.hl.k[r];
        p.n = l.hl.n[r];
        p.K = l.lay.K[r];
        return p;
    };
    auto xh = [&](int X, int sl, int r) { return c_xh.data() + ((size_t) (X * kSlots + sl) * 2 + r) * E; };
    auto xsb = [&](int X, int sl, int r) { return c_xsb.data() + ((size_t) (X * kSlots + sl) * 2 + r) * EB; };
    auto acc = [&](int X, int sl, int r, int q) {
        return c_acc.data() + (((size_t) (X * kSlots + sl) * 2 + r) * GQ + q) * FFh;
    };
    auto xhd = [&](int sl) { return c_xhd.data() + (size_t) sl * FF; };
    auto xsd = [&](int sl) { return c_xsd.data() + (size_t) sl * FB; };
    auto accd = [&](int X, int sl, int q) { return c_accd.data() + ((size_t) (X * kSlots + sl) * DQ + q) * Eh; };
    std::function<void(int)> f[4];
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (j / (2 * GQ))];
            const int r = (j / GQ) % 2, q = j % GQ;
            const kc::Exl3Part p = part(en.e, r);
            const float* xp[kc::kExl3CpuMaxN];
            float* ap[kc::kExl3CpuMaxN];
            for (int i = 0; i < en.n; ++i) {
                const int sl = en.slot0 + i;
                const float* xt = x + (size_t) en.tok[i] * ldx;
                for (int b = 2 * q; b < 2 * q + 2; ++b)
                    xsb(X, sl, r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, xt + b * 128, xh(X, sl, r) + b * 128);
                xp[i] = xh(X, sl, r);
                ap[i] = acc(X, sl, r, q);
            }
            if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 16, (q + 1) * 16, ap[0]);
            else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 16, (q + 1) * 16, ap);
        };
    run_nodes(ne * 2 * GQ, f);
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (j / GB)];
            const int b = j % GB;
            const kc::Exl3Part pg = part(en.e, 0), pu = part(en.e, 1), pd = part(en.e, 2);
            for (int i = 0; i < en.n; ++i) {
                const int sl = en.slot0 + i;
                float y[2][128], a[128], hb[128];
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                    for (int q = 0; q < GQ; ++q) {
                        const float* aq = acc(X, sl, r, q) + b * 128;
                        for (int c = 0; c < 128; ++c) a[c] += aq[c];
                    }
                    double xs = 0.0;
                    for (int k = 0; k < EB; ++k) xs += xsb(X, sl, r)[k];
                    kc::exl3_cpu_band_out(r == 0 ? pg : pu, a, (float) xs, b * 128, y[r]);
                }
                for (int c = 0; c < 128; ++c) {
                    float gv = y[0][c], uv = y[1][c];
                    if (lim > 0.0f) {
                        gv = std::min(gv, lim);
                        uv = std::min(std::max(uv, -lim), lim);
                    }
                    hb[c] = gv / (1.0f + std::exp(-gv)) * uv;
                }
                const int gb = X * GB + b;
                xsd(sl)[gb] = kc::exl3_cpu_prepare_block(pd.suh + gb * 128, hb, xhd(sl) + gb * 128);
            }
        };
    run_nodes(ne * GB, f);
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (j / DQ)];
            const int q = j % DQ;
            const kc::Exl3Part p = part(en.e, 2);
            const float* xp[kc::kExl3CpuMaxN];
            float* ap[kc::kExl3CpuMaxN];
            for (int i = 0; i < en.n; ++i) {
                xp[i] = xhd(en.slot0 + i);
                ap[i] = accd(X, en.slot0 + i, q);
            }
            if (en.n == 1) kc::exl3_cpu_rows_acc(p, xp[0], q * 8, (q + 1) * 8, ap[0]);
            else kc::exl3_cpu_rows_acc_n(p, xp, en.n, q * 8, (q + 1) * 8, ap);
        };
    run_nodes(ne * DQ, f);
    for (int X = 0; X < NH; ++X)
        f[X] = [&, X](int j) {
            const Ent& en = ents[(size_t) (j / DB)];
            const int b = j % DB;
            const kc::Exl3Part pd = part(en.e, 2);
            for (int i = 0; i < en.n; ++i) {
                const int sl = en.slot0 + i;
                float a[128];
                for (int c = 0; c < 128; ++c) a[c] = 0.0f;
                for (int q = 0; q < DQ; ++q) {
                    const float* aq = accd(X, sl, q) + b * 128;
                    for (int c = 0; c < 128; ++c) a[c] += aq[c];
                }
                double xs = 0.0;
                for (int k = 0; k < FB; ++k) xs += xsd(sl)[k];
                kc::exl3_cpu_band_out(pd, a, (float) xs, b * 128, c_yd.data() + (size_t) sl * E + (size_t) X * Eh + b * 128);
            }
        };
    run_nodes(ne * DB, f);
}

void Model::Impl::serve_lane() {
    unsigned next = 1;
    auto last = std::chrono::steady_clock::now();
    std::vector<Ent> ents;
    while (!quit.load(std::memory_order_relaxed)) {
        Req& r = reqs[next % kRing];
        if (*(volatile unsigned*) &r.seq != next) {
            cpu_relax();
            // (a decode layer takes ~2.8 ms on a Strix Halo, longer than the old 2 ms spin window, so the thread slept
            // before every request and Windows' sleep_for(20 us) wakes after ~1 ms or more: the lane's round trip.  Now it
            // spins for DS4_LANE_SPIN_MS (default 250) after its last answer, i.e. through a whole decode token.)
            static const long spin_ms = std::getenv("DS4_LANE_SPIN_MS") ? std::atol(std::getenv("DS4_LANE_SPIN_MS")) : 250;
            if (std::chrono::steady_clock::now() - last > std::chrono::milliseconds(spin_ms)) {
#if defined(_WIN32)
                // DS4_LANE_NAP=sleep turns it off: a high-resolution waitable timer (Windows 10 1803+) instead of sleep_for's timer tick
                static const bool hires = !(std::getenv("DS4_LANE_NAP") != nullptr && std::string(std::getenv("DS4_LANE_NAP")) == "sleep");
                static HANDLE tmr = hires ? CreateWaitableTimerExW(nullptr, nullptr, 0x2 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS) : nullptr;
                if (tmr != nullptr) {
                    LARGE_INTEGER due;
                    due.QuadPart = -200;   // 20 us in 100 ns units
                    SetWaitableTimer(tmr, &due, 0, nullptr, nullptr, FALSE);
                    WaitForSingleObject(tmr, 5);
                } else
#endif
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const int li = layer_of[next % kRing].load(std::memory_order_acquire);
        Resp& rp = resps[next % kRing];
        const int T = std::max(1, std::min(kWin, r.T));
        for (int t = 0; t < T; ++t) rp.cpu[t] = 0;
        if (li >= 0 && li < (int) L.size()) {
            Layer& l = L[(size_t) li];
            uint32_t* pr = prof.data() + (size_t) li * g.n_exp;
            ents.clear();
            // the window's CPU-lane picks by expert (up to 8 tokens an entry), their weights by entry row
            float wv[kWin * kPicks];
            int tok_of[kWin * kPicks];
            int nslots = 0;
            std::vector<std::pair<int, int>> picks;   // (expert, t * kPicks + i), CPU-lane only
            if (trace_f != nullptr && !l.mtp) {   // DS4_TRACE=file: one record a route request: u8 layer, u8 T, T x k int16 expert ids
                static const bool with_ahead = std::getenv("DS4_TRACE_AHEAD") != nullptr;
                uint8_t hd[2] = {(uint8_t) li, (uint8_t) (T | (with_ahead ? 0x80 : 0))};
                std::fwrite(hd, 1, 2, trace_f);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < l.k_exp; ++i) {
                        const int16_t v = (int16_t) r.ids[t][i];
                        std::fwrite(&v, 2, 1, trace_f);
                    }
                if (with_ahead) {
                    const uint8_t na[2] = {0, 0};
                    std::fwrite(na, 1, 2, trace_f);
                }
            }
            const bool dsk_rule = disk_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
            // (DISK_SKIP_AUTO: skipping now; its counts only then - the routed experts computed either way, of them
            //  on the CPU lane and not in RAM, of them on the CPU lane)
            const bool aon = dsk_rule && disk_skip_auto && sk_auto.on.load(std::memory_order_relaxed);
            const bool dsk = dsk_rule && (!disk_skip_auto || aon);
            const bool rsk = ram_skip_w > 0.0f && !l.mtp && li >= disk_skip_from && li < g.nl;
            bool tail[kWin * kPicks];
            mark_lane_skips(l, &r.ids[0][0], &r.w[0][0], T, tail);
            int heavy = 0, cold = 0, heavy_cpu = 0;
            for (int t = 0; t < T; ++t) {
                float wsum = 0.0f;   // (DISK_SKIP_W: the token's route weights)
                if (dsk || rsk)
                    for (int i = 0; i < l.k_exp; ++i)
                        if (r.ids[t][i] >= 0 && r.ids[t][i] < l.n_exp) wsum += r.w[t][i];
                for (int i = 0; i < l.k_exp; ++i) {
                    const int e = r.ids[t][i];
                    if (e < 0 || e >= l.n_exp) continue;
                    ++pr[e];
                    count_usage(l, e);
                    const bool light = dsk && r.w[t][i] < disk_skip_w * wsum;
                    if (aon && !light) ++heavy;
                    if (l.slot[(size_t) e] < 0) {
                        if ((light && !in_ram(l, e)) || (rsk && r.w[t][i] < ram_skip_w * wsum) ||
                            tail[t * kPicks + i]) {
                            ++st.skipped;
                            continue;
                        }
                        if (aon && !light) {   // (cold: not in RAM now)
                            ++heavy_cpu;
                            cold += in_ram(l, e) ? 0 : 1;
                        }
                        picks.push_back({e, t * kPicks + i});
                    }
                }
            }
            if (aon) skip_auto_tick(li, T, heavy, cold, heavy_cpu);
            std::sort(picks.begin(), picks.end());
            for (size_t a = 0; a < picks.size();) {
                Ent en;
                en.e = picks[a].first;
                en.slot0 = nslots;
                en.n = 0;
                while (a < picks.size() && picks[a].first == en.e && en.n < ent_cap(l)) {
                    const int t = picks[a].second / kPicks, i = picks[a].second % kPicks;
                    en.tok[en.n++] = t;
                    tok_of[nslots] = t;
                    wv[nslots] = r.w[t][i];
                    ++nslots;
                    ++a;
                }
                ents.push_back(en);
            }
            static const bool lane_skip = std::getenv("DS4_LANE_SKIP") != nullptr;   // (timing oracle: the lane's compute left out; rp.cpu stays 0)
            if (!ents.empty() && lane_skip) { for (int t = 0; t < T; ++t) rp.cpu[t] = 0; }
            else if (!ents.empty()) {
                const double tc = now_ms();
                cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim);
                const int D = g.dim;
                for (int t = 0; t < T; ++t) {
                    bool any = false;
                    for (int sl = 0; sl < nslots; ++sl) any = any || tok_of[sl] == t;
                    if (!any) continue;
                    float* o = rp.part[t];
                    for (int e = 0; e < D; ++e) o[e] = 0.0f;
                    for (int sl = 0; sl < nslots; ++sl) {
                        if (tok_of[sl] != t) continue;
                        const float* y = c_yd.data() + (size_t) sl * D;
                        for (int e = 0; e < D; ++e) o[e] += wv[sl] * y[e];
                    }
                    rp.cpu[t] = 1;
                }
                st.cpu_ms += now_ms() - tc;
                st.cpu_experts += nslots;
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
        *(volatile unsigned*) &rp.seq = next;
        ++next;
        last = std::chrono::steady_clock::now();
    }
}

double Model::measure_cpu_expert_ms() {
    Impl& p = *p_;
    const Layer* layer = nullptr;
    Impl::Ent en;
    for (const Layer& candidate : p.L) {
        if (candidate.mtp)
            continue;
        for (int e = 0; e < candidate.n_exp; ++e)
            if (!p.o.expert_pages->transient(candidate.il, e)) {
                layer = &candidate;
                en.e = e;
                break;
            }
        if (layer != nullptr)
            break;
    }
    if (layer == nullptr)
        return 0;
    en.slot0 = 0;
    en.n = 1;
    en.tok[0] = 0;
    std::vector<float> input((size_t) p.g.dim, 0.0f);
    const std::vector<Impl::Ent> entries{en};
    p.cpu_experts_paged(*layer, input.data(), entries, (size_t) p.g.dim);
    const double start = now_ms();
    for (int i = 0; i < 3; ++i)
        p.cpu_experts_paged(*layer, input.data(), entries, (size_t) p.g.dim);
    return (now_ms() - start) / 3;
}

}  // namespace strata::ds4
