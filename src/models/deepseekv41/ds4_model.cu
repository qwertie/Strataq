// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
//
// The layer, for one token at position p (DeepSeek's inference/model.py, Block.forward with the lagged pre-mix):
//   [Engram, layers 1 and 14]  h += gate(h, key) * value from 24 n-gram rows of the FP8 tables
//   a_pre, a_post, a_comb = hc_mixes(h);  x = rmsnorm(sum_s pre[s] h[s])   (pre: the previous layer's f_pre)
//   attention(x): q from the low-rank wq_a / wq_b, one shared K = V row per token in a 128-slot window ring, plus
//                 (compress ratio > 0) up to 512 compressed rows the indexer picks; a learned sink per head; the
//                 output's rotary part rotated back, rounded to FP8, through the grouped wo_a and wo_b
//   h = post * attn + comb^T h;  f_* = hc_mixes(h);  x = rmsnorm(sum_s a_pre[s] h[s])
//   moe(x): 6 of 384 routed experts (sqrt-softplus scores, biased top-k) + the shared expert
//   h = post * moe + comb^T h;  pre = f_pre
// and after the last layer logits = head(rmsnorm(sum_s pre[s] h[s])).
//
// This file holds Model's calls into Model::Impl (ds4_model_impl.hpp), Impl's tear-down, the sampling, the profile and
// the debug dumps. The other ds4_*.cu files hold the rest of Impl, by job.
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

// ---- the profile, the debug dumps, the tear-down
cudaEvent_t Model::Impl::ek_ev() {
    int cur = 0;
    cudaGetDevice(&cur);
    std::vector<cudaEvent_t>& pl = ek_pool[cur];
    if (!pl.empty()) {
        cudaEvent_t e = pl.back();
        pl.pop_back();
        return e;
    }
    cudaEvent_t e = nullptr;
    cudaEventCreate(&e);
    return e;
}

void Model::Impl::prof_mark(const Dev& d, int stage) {
    if (ds4_dbg("DS4_TLOG")) {   // the host's own phase times (what the GPU meters do not see)
        static double t_last = 0, acc[5] = {0, 0, 0, 0, 0}; static long n = 0; static bool reg = false;
        if (!reg) {
            reg = true;
            std::atexit([] {
                std::fprintf(stderr,
                             "tlog: %ld layers | host ms: attn %.0f  moe_pre %.0f  gate_up %.0f  cpu_wait %.0f  down+post %.0f\n",
                             n, acc[0], acc[1], acc[2], acc[3], acc[4]);
            });
        }
        const double t = now_ms();
        if (t_last > 0) {
            const int b = stage == 0 ? 4 : stage - 1;
            acc[b >= 0 && b < 5 ? b : 4] += t - t_last;
        }
        t_last = t;
        if (stage == 3) ++n;
    }
    if (!prof_window && !prof_chunk) return;
    if (prof_window && !prof_chunk && prof_sync_on) {   // decode: the stream drained at each mark, host clock (events hang the driver on gfx1151 Windows)
        std::lock_guard<std::mutex> lk(prof_mu);
        cudaSetDevice(d.id);
        cudaStreamSynchronize(d.st);
        const double t = now_ms();
        if (prof_sync_t > 0) st.prof_ms[stage] += t - prof_sync_t;
        prof_sync_t = now_ms();
        return;
    }
    std::lock_guard<std::mutex> lk(prof_mu);
    if (prof_failed) return;
    if (d.id < 0 || (size_t) d.id >= prof_pool.size()) {
        std::fprintf(stderr, "ds4: profiling disabled: CUDA device %d is outside the event pool\n", d.id);
        prof_failed = true;
        return;
    }
    cudaError_t ce = cudaSetDevice(d.id);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "ds4: profiling disabled: cudaSetDevice(%d): %s\n", d.id, cudaGetErrorString(ce));
        prof_failed = true;
        return;
    }
    auto& pool = prof_pool[(size_t) d.id];
    cudaEvent_t e = nullptr;
    if (pool.empty()) {
        ce = cudaEventCreate(&e);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "ds4: profiling disabled: cudaEventCreate on CUDA%d: %s\n", d.id,
                         cudaGetErrorString(ce));
            prof_failed = true;
            return;
        }
    } else {
        e = pool.back();
        pool.pop_back();
    }
    ce = cudaEventRecord(e, d.st);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "ds4: profiling disabled: cudaEventRecord on CUDA%d: %s\n", d.id,
                     cudaGetErrorString(ce));
        (void) cudaEventDestroy(e);
        prof_failed = true;
        return;
    }
    prof_marks.push_back({d.id, e, stage});
}

void Model::Impl::prof_collect() {
    if (!prof_window && !prof_chunk) return;
    if (prof_window && !prof_chunk && prof_sync_on) {
        std::lock_guard<std::mutex> lk(prof_mu);
        prof_sync_t = 0;
        ++st.prof_tokens;
        return;
    }
    std::lock_guard<std::mutex> lk(prof_mu);
    const bool chunk_sample = prof_chunk;
    bool sample_ok = !prof_failed;
    double sample_ms[6] = {};
    std::vector<std::pair<int, cudaEvent_t>> last;
    int current_dev = -1;
    if (sample_ok) {
        for (const ProfMark& m : prof_marks) {
            if (m.dev < 0 || (size_t) m.dev >= prof_pool.size()) {
                std::fprintf(stderr, "ds4: profiling disabled: CUDA device %d is outside the event pool\n", m.dev);
                sample_ok = false;
                break;
            }
            if (current_dev != m.dev) {
                const cudaError_t ce = cudaSetDevice(m.dev);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "ds4: profiling disabled: cudaSetDevice(%d): %s\n", m.dev,
                                 cudaGetErrorString(ce));
                    sample_ok = false;
                    break;
                }
                current_dev = m.dev;
            }
            auto it = std::find_if(last.begin(), last.end(), [&](const auto& x) { return x.first == m.dev; });
            if (it != last.end()) {
                float ms = 0.0f;
                const cudaError_t ce = cudaEventElapsedTime(&ms, it->second, m.ev);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "ds4: profiling disabled: cudaEventElapsedTime on CUDA%d: %s\n", m.dev,
                                 cudaGetErrorString(ce));
                    sample_ok = false;
                    break;
                }
                sample_ms[m.stage] += ms;
                it->second = m.ev;
            } else {
                last.push_back({m.dev, m.ev});
            }
        }
    }
    if (sample_ok) {
        double* total = chunk_sample ? cprof_ms : st.prof_ms;
        const int stages = chunk_sample ? 4 : 6;
        for (int i = 0; i < stages; ++i) total[i] += sample_ms[i];
        for (const ProfMark& m : prof_marks) prof_pool[(size_t) m.dev].push_back(m.ev);
        prof_marks.clear();
        if (!chunk_sample) ++st.prof_tokens;
        return;
    }
    prof_failed = true;
    std::vector<ProfMark> retry;
    current_dev = -1;
    for (const ProfMark& m : prof_marks) {
        if (m.dev < 0 || (size_t) m.dev >= prof_pool.size()) continue;
        if (current_dev != m.dev) {
            if (cudaSetDevice(m.dev) != cudaSuccess) {
                retry.push_back(m);
                continue;
            }
            current_dev = m.dev;
        }
        (void) cudaEventDestroy(m.ev);
    }
    prof_marks = std::move(retry);
}

void Model::Impl::dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
    std::vector<float> v((size_t) T * g.hc * g.dim);
    cudaSetDevice(d.id);
    cudaMemcpyAsync(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost, d.st);
    cudaStreamSynchronize(d.st);
    std::lock_guard<std::mutex> lk(dump_mu);
    const int64_t hdr[3] = {il, p0, T};
    std::fwrite(hdr, 8, 3, dump_l);
    std::fwrite(v.data(), 4, v.size(), dump_l);
    std::fflush(dump_l);
}

void Model::Impl::dump_sub_host(const char* name, int il, int64_t p0, int T, const void* host, size_t nbytes) {
    if (!dump_s_on(il) || !dump_s_name(name)) return;
    std::lock_guard<std::mutex> lk(dump_mu);
    char nm[16] = {};
    std::strncpy(nm, name, 15);
    const int64_t hdr[4] = {il, p0, T, (int64_t) nbytes};
    std::fwrite(nm, 1, 16, dump_s);
    std::fwrite(hdr, 8, 4, dump_s);
    std::fwrite(host, 1, nbytes, dump_s);
    std::fflush(dump_s);
}

Model::Impl::~Impl() {
    quit.store(true);
    if (warm_bg.joinable()) warm_bg.join();   // (the warm-up uses the BLAS handles destroyed below)
    eg_io.stop();   // (its threads write into h_eg)
    for (auto& w : workers) w->stop();
    for (size_t i = 0; i < chunks.size(); ++i)
        if (chunks[i] && chunks[i]->blas) {
            cudaSetDevice(devs[i].id);
            cublasDestroy(chunks[i]->blas);
        }
    quit.store(true);
    if (service.joinable()) service.join();
    for (size_t di = 0; di < prof_pool.size(); ++di) {
        if (cudaSetDevice((int) di) != cudaSuccess) continue;
        for (cudaEvent_t e : prof_pool[di]) (void) cudaEventDestroy(e);
    }
    for (const ProfMark& m : prof_marks) {
        if (m.dev < 0 || (size_t) m.dev >= prof_pool.size()) continue;
        if (cudaSetDevice(m.dev) == cudaSuccess) (void) cudaEventDestroy(m.ev);
    }
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        for (void* p : d.allocs)
            strata::core::release_device(p);
        if (d.st) cudaStreamDestroy(d.st);
        if (d.ev) cudaEventDestroy(d.ev);
    }
    for (void* p : host_allocs)
        strata::core::release_host(p);
}

bool Model::Impl::router_nonfinite() {
    bool bad = false;
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        if (router_nonfinite_take(d.st)) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) {
                char where[32];
                std::snprintf(where, sizeof where, "CUDA%d", d.id);
                std::fprintf(stderr, "ds4: WARNING: the router met scores that are not numbers (NaN) on %s - the model's "
                                     "activations went non-finite; its picks are stand-ins and the answer is not reliable\n", where);
            }
            bad = true;
        }
    }
    return bad;
}

// ---- sampling
int Model::Impl::tok_back(Dev& d) {
    int t = -1;
    // (DS4_TEST_SAMPLE_FAIL=N: the Nth sample of the run fails as a CUDA error would - the test of #619's handling)
    static const long fail_at = std::getenv("DS4_TEST_SAMPLE_FAIL") ? std::atol(std::getenv("DS4_TEST_SAMPLE_FAIL")) : 0;
    static long calls = 0;
    if (fail_at > 0 && ++calls == fail_at) {
        std::fprintf(stderr, "ds4 sample: injected failure (DS4_TEST_SAMPLE_FAIL=%ld)\n", fail_at);
        return -1;
    }
    cudaError_t e = cudaMemcpyAsync(&t, d.tok, 4, cudaMemcpyDeviceToHost, d.st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(d.st);
    if (e == cudaSuccess) return t;
    std::fprintf(stderr, "ds4 sample: %s\n", cudaGetErrorString(e));
    return -1;
}

const int* Model::Impl::pen_rows(int64_t end0, int T, strata::dsv41k::SamplerParams& sp, int& h) {
    h = 0;
    if (pen_n <= 0 || T < 1 || T > kWin) return nullptr;
    Dev& d = devs[(size_t) head_dev];
    if (d_pen == nullptr) {
        std::string e;
        if (!halloc(h_pen, (size_t) kWin * kPenCap, e) || !dalloc(d, d_pen, (size_t) kWin * kPenCap, e)) {
            std::fprintf(stderr, "ds4: the penalty rows: %s - picks without the penalties\n", e.c_str());
            pen_n = 0;
            return nullptr;
        }
    }
    cudaSetDevice(d.id);
    h = std::min(pen_n, kPenCap);
    for (int t = 0; t < T; ++t) {
        int32_t* row = h_pen + (size_t) t * (size_t) h;
        const int64_t end = std::min<int64_t>(end0 + t, (int64_t) toks.size());
        const int take = (int) std::min<int64_t>(h, std::max<int64_t>(0, end));
        std::fill(row, row + (h - take), -1);
        std::copy(toks.begin() + (end - take), toks.begin() + end, row + (h - take));
    }
    cudaMemcpyAsync(d_pen, h_pen, (size_t) T * (size_t) h * sizeof(int32_t), cudaMemcpyHostToDevice, d.st);
    sp.penalty_last_n = h;
    sp.penalty_repeat = pen_repeat;
    sp.penalty_freq = pen_freq;
    sp.penalty_present = pen_present;
    return d_pen;
}

int Model::Impl::sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed,
                            uint64_t counter, const float* bias, int64_t hist_end) {
    Dev& d = devs[(size_t) head_dev];
    cudaSetDevice(d.id);
    strata::dsv41k::SamplerParams sp;
    sp.temperature = temperature;
    sp.greedy = temperature <= 0.0f;
    sp.top_p = top_p;
    sp.top_k = top_k;
    sp.min_p = min_p;
    sp.seed = seed;
    sp.counter = counter;
    sp.logit_bias = bias;
    int h = 0;
    const int* hr = pen_rows(hist_end, 1, sp, h);
    strata::dsv41k::sample_tokens(d.logits + (size_t) row * g.vocab, 1, g.vocab, hr, h, sp, d.tok, d.st);
    return tok_back(d);
}

int Model::Impl::sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
    Dev& d = devs[(size_t) head_dev];
    cudaSetDevice(d.id);
    strata::dsv41k::SamplerParams sp;
    sp.temperature = temperature;
    sp.greedy = temperature <= 0.0f;
    sp.top_p = top_p;
    sp.top_k = top_k;
    sp.min_p = min_p;
    sp.seed = seed;
    sp.counter = counter;
    int h = 0;
    const int* hr = pen_rows((int64_t) toks.size(), 1, sp, h);   // (the last fed token's row: every token consumed)
    strata::dsv41k::sample_tokens(d.logits + (size_t) std::max(0, n_logits - 1) * g.vocab, 1, g.vocab, hr, h,
                                   sp, d.tok, d.st);
    return tok_back(d);
}

// ---- the profile's file, the model's description
void Model::Impl::save_profile() const {
    if (o.profile.empty() || o.profile_readonly) return;   // (--profile-readonly: a test keeps the file it read)
    std::string err;
    if (!strata::core::write_expert_counts(o.profile, (int64_t) prof.size() / g.n_exp, g.n_exp, prof, err))
        std::fprintf(stderr, "ds4: %s\n", err.c_str());
}

std::vector<std::string> Model::Impl::info_lines() const {
    std::vector<std::string> v;
    int nh = 0;
    for (const Layer& l : L) nh += l.n_exp - l.n_vram;
    char b[512];
    // kv=fp4: DeepSeek's own cache layout (the compressed KV and the index keys FP4, the window FP8)
    std::snprintf(b, sizeof b, "INFO context=%lld kv=fp4 kv_resident=%lld expert_slots=%d engine_kind=ds4 "
                               "cpu_experts=%d ram_experts=%d engine=ds4-0.1",
                  (long long) o.max_context, (long long) o.max_context, g.nl * g.n_exp - nh, nh, 0);
    v.push_back(b);
    if (pk.eg_layers > 0) {   // the n-gram memory's (Engram's) table rows a token reads, for the Monitor
        std::snprintf(b, sizeof b, "INFO engram_layers=%d engram_lookups=%d", pk.eg_layers,
                      pk.eg_layers * (pk.eg_ngram - 1) * pk.eg_heads);
        v.push_back(b);
    }
    if (img_tok >= 0) {   // the server's image resize (DeepSeek's image_processor.py)
        std::snprintf(b, sizeof b,
                      "INFO vision=ds4 vision_patch=%d vision_merge=%d vision_min_pixels=%lld vision_max_tokens=%d "
                      "vision_max_wh_ratio=%lld image_token=%d vision_gpu=%d vision_mib=%lld vision_lazy=%d",
                      vg.patch, vg.merge, (long long) pk.num("vision.min_pixels", 0), vg.max_tokens,
                      (long long) pk.num("vision.max_wh_ratio", 0), img_tok, vt.device(),
                      (long long) (vt.device_bytes() >> 20), vt.lazy() ? 1 : 0);
        v.push_back(b);
    } else if (!vision_err.empty()) {   // (the server offers no pictures and says why)
        std::string why = vision_err;
        for (char& ch : why)
            if (ch == '\n' || ch == '\r') ch = ' ';
        v.push_back("NOTE images: OFF - the model's vision tower did not load (" + why + ")");
        v.push_back("INFO vision=off");
    }
    return v;
}

// ---- Model, which calls Impl
Model::Model() : p_(new Impl) {}
Model::~Model() = default;
bool Model::load(const Options& o, std::string& err) { return p_->load(o, err); }
bool Model::place_experts(std::string& err) { return p_->finish_load(err); }
std::vector<ExpertFileLayout> Model::get_expert_files() const { return p_->get_expert_files(p_->L.size()); }

std::vector<std::string> Model::get_file_paths() const {
    std::vector<std::string> paths;
    for (const std::string& f : p_->pk.files)
        paths.push_back(p_->pk.model_dir + "/" + f);
    return paths;
}
std::vector<int> Model::get_heat_order(int layer) const { return p_->get_heat_order(p_->L[(size_t) layer]); }
bool Model::set_expert_addresses(const std::vector<const uint8_t*>& at, std::string& err, int stage) {
    return p_->set_expert_addresses(at, err, stage);
}

std::vector<int64_t> Model::get_layer_cuts() const {
    std::vector<int64_t> cuts;
    for (int il = 1; il < p_->g.nl; ++il)
        if (p_->L[(size_t) il].dev != p_->L[(size_t) il - 1].dev)
            cuts.push_back(il);
    return cuts;
}

void Model::set_hop_holders(const int32_t* holders) { p_->hop_holders = holders; }
void Model::report_hops() const {
    for (const auto& hop : p_->hops)
        if (hop && hop->experts > 0)
            std::fprintf(stderr, "strata-ds41: so far gpu%d computed %lld experts of other cards' layers "
                                 "(%lld rows) in one hop each layer, %.1f ms waited for\n",
                         hop->d.id, (long long) hop->experts, (long long) hop->rows, hop->ms);
}
std::vector<float>& Model::get_usage() { return p_->usage; }
int64_t Model::get_cpu_experts() const { return p_->cpu_expert_count; }
uint64_t Model::get_drafter_weight_bytes(int stage) const {
    return p_->devs[(size_t) stage].drafter_bytes;
}

bool Model::reseed_drafter(int64_t position, std::string& err) {
    return p_->reseed_drafter_from_taps(position, err);
}

std::vector<uint64_t> Model::get_layer_dense_bytes() const {
    std::vector<uint64_t> bytes;
    for (int il = 0; il < p_->g.nl; ++il) {
        const Layer& layer = p_->L[(size_t) il];
        bytes.push_back(layer.weight_bytes + layer.dense_scratch_bytes +
                        (layer.slot_tab != nullptr ? align256((size_t) layer.n_exp * 8) : 0) +
                        (layer.vs_dev != nullptr ? align256((size_t) layer.n_exp * 4) : 0));
    }
    return bytes;
}
const Geometry& Model::geometry() const { return p_->g; }
std::vector<uint64_t> Model::get_expert_bytes() const {
    std::vector<uint64_t> bytes;
    for (const auto& l : p_->L)
        bytes.push_back(l.blob);
    return bytes;
}
std::vector<int> Model::get_devices() const {
    std::vector<int> ids;
    for (const Dev& d : p_->devs)
        ids.push_back(d.id);
    return ids;
}
void Model::use_state_memory(int dev, uint8_t* fixed, uint8_t* drafter, const ConversationRegions& at,
                             const std::vector<KvRows>& ckv, const std::vector<KvRows>& ik) {
    Impl& p = *p_;
    for (size_t s = 0; s < p.srcs.size(); ++s) {
        Src& src = p.srcs[s];
        src.ckv[(size_t) dev] = ckv[s];
        src.ik[(size_t) dev] = ik[s];
        Layer& l = p.L[(size_t) src.layer];
        if (l.dev == dev && at.group[s] != kNoOffset) {
            l.c_kv = (float*) (fixed + at.group[s]);
            l.c_score = l.c_kv + (size_t) l.ratio * p.g.hd;
        }
    }
    if (dev == p.d_dev && at.taps != kNoOffset)
        p.draft_taps = (float*) (drafter + at.taps);
    for (Layer& l : p.L)   // Paged stages are layer-parallel, with no tensor-parallel members.
        if (l.dev == dev)
            l.win = (l.mtp ? drafter : fixed) + at.ring[(size_t) l.il];
}
bool Model::step_conversations(const std::vector<int>& tokens, const std::function<void(int)>& select,
                               const std::function<void(int)>& bind, const std::vector<int>& counts,
                               std::string& err) {
    return p_->step_conversations(tokens, select, bind, counts, err);
}
bool Model::draft_conversations(std::vector<DraftRequest>& requests, const std::function<void(int)>& select,
                                const std::function<void(int)>& bind, std::string& err) {
    return p_->draft_conversations(requests, select, bind, err);
}
bool Model::save_joint_draft(int session, int n, std::string& err) {
    return p_->save_joint_draft(session, n, err);
}

bool Model::prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
                          const std::function<void(int)>& bind, std::string& err) {
    return p_->prepare_group(group, tokens, select, bind, err);
}
int Model::poll_group(int stage, std::string& err) { return p_->poll_group(stage, err); }
bool Model::finish_group(int group, const std::function<void(int)>& select, std::string& err) {
    return p_->finish_group(group, select, err);
}
void Model::set_draft_rule(strata::kernels::DraftRule rule) { p_->draft_rule = rule; }
int Model::get_route_tail_skip() const { return p_->route_tail_skip.load(std::memory_order_relaxed); }
void Model::set_route_tail_skip(int rank) { p_->route_tail_skip.store(std::max(rank, 0), std::memory_order_relaxed); }
int Model::get_route_prior_pct() const { return p_->route_prior_pct.load(std::memory_order_relaxed); }
void Model::set_route_prior_pct(int pct) { p_->route_prior_pct.store(std::max(pct, 0), std::memory_order_relaxed); }
int Model::sample_row(int row, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample_row(row, t, tp, tk, mp, seed, counter, nullptr, p_->row_positions[row] + 1);
}
bool Model::step(int token, bool logits, std::string& err) { return p_->step(token, logits, err); }
int Model::feed_unit() const { return p_->chunk > 0 ? std::max(p_->chunk, (int) p_->o.max_context) : kWin; }
void Model::set_progress(std::function<void(int64_t)> f) { p_->progress = std::move(f); }
void Model::set_cancel(std::function<bool()> f) { p_->cancel = std::move(f); }
void Model::set_yield(std::function<bool()> f) { p_->yield_to = std::move(f); }
int Model::fed() const { return p_->fed_n; }
int Model::prompt_chunk() const { return p_->chunk; }
bool Model::can_read_chunked(int rows) const { return p_->can_chunk(rows); }
uint64_t Model::get_prompt_bytes(int dev, int chunk, uint64_t& host) const {
    size_t host_bytes = 0;
    const uint64_t device = p_->lay_prompt(*p_->chunks[(size_t) dev], chunk, nullptr, nullptr, host_bytes);
    host = host_bytes;
    return device;
}
bool Model::lay_out_prompt(int dev, int chunk, uint8_t* device, uint8_t* host, std::string& err) {
    return p_->lay_out_prompt(dev, chunk, device, host, err);
}
bool Model::raise_prompt_chunk(int chunk, std::string& err) { return p_->raise_prompt_ceiling(chunk, err); }
bool Model::cancelled() const { return p_->halted; }
int Model::sample(float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter) {
    return p_->sample(t, tp, tk, mp, seed, counter);
}
void Model::set_penalties(int last_n, float repeat, float freq, float present) {
    const bool on = last_n > 0 && (repeat != 1.0f || freq != 0.0f || present != 0.0f);
    p_->pen_n = on ? last_n : 0;
    p_->pen_repeat = repeat;
    p_->pen_freq = freq;
    p_->pen_present = present;
}
void Model::set_penalties(const strata::serve::Sampling& s) {
    set_penalties(s.penalty_last_n < 0 ? INT_MAX : s.penalty_last_n,
                  std::isfinite(s.penalty_repeat) && s.penalty_repeat > 0.0f ? s.penalty_repeat : 1.0f,
                  std::isfinite(s.penalty_freq) ? s.penalty_freq : 0.0f,
                  std::isfinite(s.penalty_present) ? s.penalty_present : 0.0f);
}
bool Model::copy_logits_row(int row, float* out) {
    Impl& p = *p_;
    Dev& d = p.devs[(size_t) p.head_dev];
    cudaSetDevice(d.id);
    const size_t at = (size_t) (row < 0 ? std::max(0, p.n_logits - 1) : row) * (size_t) p.g.vocab;
    strata::core::count_gpu_traffic(d.id, 0, (uint64_t) p.g.vocab * 4);
    return cudaMemcpyAsync(out, d.logits + at, (size_t) p.g.vocab * 4, cudaMemcpyDeviceToHost, d.st) == cudaSuccess &&
           cudaStreamSynchronize(d.st) == cudaSuccess;
}
int Model::image_token() const { return p_->img_tok; }
int Model::dim() const { return p_->g.dim; }
bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err) {
    return p_->encode_image(rgb, width, height, span, err);
}
void Model::set_images(std::vector<Image> images) {
    std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) { return a.start < b.start; });
    p_->imgs = std::move(images);
}
int Model::draft_block() const { return p_->g.n_mtp > 0 ? p_->g.d_block : 0; }
// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the
// chance that it and every draft before it are kept stays at least this - a long window whose later drafts are likely
// lost is not verified whole.  0.5 on gfx1200 / gfx1201 (RDNA4) only: measured on 4x R9700, DeepSeek 2.25 bpw, six
// 128-token decodes an arm in one load - 71.7 -> 76.9 tok/s (medians), the same ids.  Project Maya ran 0.5 on a Radeon
// 8065S (Strix Halo: decode +6.5%, the same tokens), a single datapoint, so Strix keeps 0 until it is measured; CUDA
// measured flat, so CUDA keeps 0.
float Model::get_default_cum() const {
    static const char* const e = std::getenv("STRATA_DS4_DSPARK_CUM");
    return dspark_cum_value(e != nullptr, e != nullptr ? (float) std::atof(e) : 0.0f, p_->rdna4(), defaults_026());
}
bool Model::draft(int y, std::vector<int>& d, std::vector<float>& c, std::string& err, float t, float tp, int tk,
                  float mp, uint64_t seed, uint64_t counter) {
    return p_->draft(y, d, c, err, t, tp, tk, mp, seed, counter);
}
bool Model::verify_rows(int y, const std::vector<int>& d, float t, float tp, int tk, float mp, uint64_t seed,
                        uint64_t counter, std::vector<int>& picks, std::string& err) {
    return p_->verify_rows(y, d, t, tp, tk, mp, seed, counter, picks, err);
}
bool Model::keep_verified(int keep, std::string& err) { return p_->keep_verified(keep, err); }
int64_t Model::pos() const { return p_->pos; }
int64_t Model::max_context() const { return p_->o.max_context; }
int Model::vocab() const { return p_->g.vocab; }
void Model::reset() { p_->reset(); }
void Model::checkpoint(int slot) { p_->checkpoint(slot); }
void Model::swap_conversation(ConversationState& s) { p_->swap_conversation(s); }
std::string Model::get_state_problem(const ConversationState& s) const { return p_->get_state_problem(s); }
void Model::mark(bool shared) { p_->mark(shared); }
int Model::marks_cap() const { return Impl::marks_cap(); }
size_t Model::check_bytes() const { return p_->check_bytes(); }
bool Model::restore(int64_t limit) { return p_->restore(limit); }
void Model::pin() { p_->pin(); }
std::vector<int64_t> Model::check_positions() const { return p_->check_positions(); }
int Model::get_experts_per_token() const { return p_->L.empty() ? 0 : p_->L[0].k_exp; }
std::string Model::slot_identity() const { return p_->slot_identity(); }
bool Model::router_nonfinite() { return p_->router_nonfinite(); }
void Model::save_profile() const { p_->save_profile(); }
const Stats& Model::stats() const { return p_->st; }
std::vector<std::string> Model::info_lines() const { return p_->info_lines(); }

}  // namespace strata::ds4
