// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_prefill.cu - the prompt path: a prompt read in chunks of many tokens, layer by layer (forward_chunks), with its
// buffers (lay_chunk, alloc_chunks), the GEMMs of the experts many rows picked (run_set), the flow that streams a
// chunk's experts from RAM or the files, a long prompt's skip (yoco) and Model::feed and feed_mark. A prompt too short
// for a chunk goes through the decode's windows (feed_windows).
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

namespace {

// anonymous memory on one NUMA node (node < 0: wherever the first touch puts it), transparent huge pages
/// Allocates owned host expert storage; file mappings remain the operating system's cache.
void* node_map(size_t bytes, int node, int n_nodes,
               strata::core::MemoryLabel name = {"experts", "Host expert storage", "Layer weights"}) {
    void* p = platform::anon_alloc(bytes, node, n_nodes);
    strata::core::get_memory_manager().add(p, bytes, "ram", std::move(name));
    return p;
}
/// Releases host storage and the allocation labels recorded by node_map.
void free_node_memory(void* p, size_t bytes) {
    strata::core::get_memory_manager().remove(p);
    platform::anon_free(p, bytes);
}

// DS4_PRED_DUMP: a fixed +-1 random projection of the router input (D -> 256), one block a token
__global__ void pred_proj_k(const float* x, int D, float* out) {
    const int t = blockIdx.x, j = threadIdx.x;   // 256 threads
    const float* xr = x + (size_t) t * D;
    float a = 0.0f;
    for (int d = 0; d < D; ++d) {
        unsigned h = (unsigned) d * 2654435761u ^ ((unsigned) j + 1u) * 40503u;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        a += (h & 1u) ? xr[d] : -xr[d];
    }
    out[(size_t) t * 256 + j] = a * rsqrtf((float) D);
}

// DS4_UNPICKED_STAT=1: for each size of chunk a prompt runs, how many of a layer's experts none of the chunk's rows picked
// (the experts a chunk that streams all of them reads for nothing, Project Maya #691's "streams only the experts it
// picks"), printed at exit
struct UnpickedStat {
    std::mutex mu;
    std::map<int, std::array<double, 3>> by_rows;   // rows -> {chunk-layers, unpicked experts, experts}
    void add(int rows, int picked, int n_exp) {
        std::lock_guard<std::mutex> lk(mu);
        auto& v = by_rows[rows];
        v[0] += 1;
        v[1] += n_exp - picked;
        v[2] += n_exp;
    }
    ~UnpickedStat() {
        for (const auto& kv : by_rows)
            std::fprintf(stderr, "ds4 unpicked: chunks of %d rows: %.0f chunk-layers, %.2f%% of a layer's experts picked by none of them\n",
                         kv.first, kv.second[0], kv.second[2] > 0 ? 100.0 * kv.second[1] / kv.second[2] : 0.0);
    }
};
UnpickedStat g_unpicked_stat;
// DS4_SKIP_UNPICKED_ROWS: the streamed experts a skipping chunk copied and left out, printed at exit
struct SkipCount {
    std::atomic<size_t> skipped{0}, copied{0};
    ~SkipCount() {
        if (skipped + copied > 0)
            std::fprintf(stderr, "ds4 skip unpicked: %zu streamed experts copied, %zu left out (%.1f%%)\n", copied.load(), skipped.load(),
                         100.0 * (double) skipped / (double) (skipped + copied));
    }
};
SkipCount g_skip_count;

}  // namespace

// ---- a Stop or a yield
bool Model::Impl::chunk_go(int k) {
    std::lock_guard<std::mutex> lk(halt_mu);
    if (k >= halt_k) return false;
    halt_seen = std::max(halt_seen, k);
    return true;
}

bool Model::Impl::halt_end(int di, std::string& err) {
    Chunk& c = *chunks[(size_t) di];
    cudaSetDevice(devs[(size_t) di].id);
    if (c.flow)
        for (size_t i = c.base; i < c.n_read; ++i)
            while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
    if (c.cs != nullptr) cudaStreamSynchronize(c.cs);
    cudaStreamSynchronize(devs[(size_t) di].st);
    err = kHalted;
    return false;
}

// ---- the chunked prompt path
double Model::Impl::gpu_expert_share(double* exp_out, double* room_out, double* other_out) const {
    double exp_b = 0, other_b = 0, room = 0;
    for (const auto& kt : pk.t) {
        const std::string& n = kt.first;
        if (n.find(".engram.embed.") != std::string::npos || n.rfind("vision.", 0) == 0 || n == "embed.weight" ||
            (!o.dspark && n.rfind("mtp.", 0) == 0))
            continue;
        (n.find(".ffn.experts.") != std::string::npos ? exp_b : other_b) += (double) kt.second.bytes;
    }
    for (const Dev& d : devs)
        room += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20) - 1.5 * (1 << 30)) * o.vram_experts;
    if (exp_out) *exp_out = exp_b;
    if (room_out) *room_out = room;
    if (other_out) *other_out = other_b;
    return exp_b > 0 ? std::min(1.0, std::max(0.0, room - other_b) / exp_b) : 1.0;
}

double Model::Impl::link_h2d(Dev& d) {
    if (auto it = link_bw.find(d.id); it != link_bw.end()) return it->second;
    const size_t nb = (size_t) 32 << 20;
    void* h = nullptr;
    void* v = nullptr;
    const cudaStream_t s = 0;   // (the legacy stream)
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    double best = 0;
    cudaSetDevice(d.id);
    const strata::core::MemoryLabel label{"other", "Device allocations", "DeepSeek working buffers"};
    if (strata::core::allocate_host(&h, nb, cudaHostAllocPortable, label) == cudaSuccess &&
        strata::core::allocate_device(&v, nb, label) == cudaSuccess && cudaEventCreate(&e0) == cudaSuccess &&
        cudaEventCreate(&e1) == cudaSuccess) {
        std::memset(h, 0, nb);
        for (int rep = 0; rep < 4; ++rep) {
            float ms = 0;
            if (cudaEventRecord(e0, s) != cudaSuccess || cudaMemcpyAsync(v, h, nb, cudaMemcpyHostToDevice, s) != cudaSuccess ||
                cudaEventRecord(e1, s) != cudaSuccess || cudaEventSynchronize(e1) != cudaSuccess ||
                cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess)
                break;
            if (rep > 0 && ms > 0) best = std::max(best, (double) nb / (ms / 1000.0));   // (the first warms up)
        }
    }
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (v) strata::core::release_device(v);
    if (h) strata::core::release_host(h);
    cudaGetLastError();
    link_bw[d.id] = best;
    return best;
}

size_t Model::Impl::ring_mb_auto() const {
    double exp_b = 0, other_b = 0, room = 0;
    const double share = gpu_expert_share(&exp_b, &room, &other_b);
    // One rule for the staging ring (ds4_experts_rule.hpp: staging_ring_mb): 16 MB where the cards hold the experts, a small
    // ring from each device's room where they hold most of them (4 x R9700), the larger streamed ring from the summed room on a
    // unified-memory GPU (Strix Halo: 400 -> 1200 MB) and where the cards hold under half.  DS4_RING_MB overrides.
    bool unified = false;
    for (const Dev& d : devs) {
        cudaDeviceProp ip{};
        if (cudaGetDeviceProperties(&ip, d.id) == cudaSuccess && ip.integrated != 0) unified = true;
    }
    const size_t mb = (size_t) strata::ds4::staging_ring_mb(exp_b > 0 ? share : 0.0, room, (int) devs.size(), unified, rdna4(), defaults_026());
    return mb;
}

size_t Model::Impl::lay_chunk(Chunk& c, int C, uint8_t* base) const {
    size_t off = 0;
    auto put = [&](auto*& p, size_t n) {
        using T = std::remove_reference_t<decltype(*p)>;
        if (base != nullptr) p = (T*) (base + off);
        off += align256(n * sizeof(T));
    };
    const size_t Z = (size_t) C, D = (size_t) g.dim, H = (size_t) g.hc, HD = (size_t) g.hd, A = (size_t) c.A,
                 AI = (size_t) c.AI, ss = (size_t) o.max_context + 64, HQ = (size_t) c.HQ, OG = (size_t) c.OG,
                 FF = (size_t) c.FF, ES = 256, cols = (size_t) (pk.eg_ngram - 1) * pk.eg_heads,
                 ehd = (size_t) pk.eg_hd;
    {
        put(c.q, A * HQ), put(c.o, A * HQ), put(c.o16, A * HQ), put(c.oa, A * OG * g.o_lora);
        put(c.oa16, A * OG * g.o_lora), put(c.iq, AI * g.ix_heads * g.ix_hd), put(c.scores, AI * ss);
        put(c.bm, AI * bm_stride()), put(c.bounds, 2048), put(c.w16, c.w16_elems);
        if (c.has_eg) put(c.eg_x, ES * cols * ehd), put(c.eg_x16, ES * cols * ehd), put(c.eg_kv, ES * (H + 1) * D);
    }
    {
        put(c.h, Z * H * D), put(c.x, Z * D), put(c.acc, Z * D), put(c.x16, Z * D), put(c.qr, Z * g.q_lora);
        put(c.qr16, Z * g.q_lora), put(c.kvr, Z * HD), put(c.ckr, Z * HD), put(c.cgr, Z * HD);
        put(c.latent, (Z + 2) * HD), put(c.lat16, (Z + 2) * HD), put(c.ikr, (Z + 2) * g.ix_hd);
        put(c.ext, ((size_t) g.win + Z) * HD), put(c.iw, Z * 64), put(c.topk, Z * topk_stride());
        put(c.cand, Z * cand_stride()), put(c.mix, Z * 32), put(c.a_pre, Z * 8), put(c.a_post, Z * 8);
        put(c.a_comb, Z * 64), put(c.f_pre, Z * 8), put(c.f_post, Z * 8), put(c.f_comb, Z * 64), put(c.pre, Z * 8);
        put(c.glog, Z * 1024), put(c.ridx, Z * kPicks), put(c.rw, Z * kPicks), put(c.row_tok, Z * kPicks);
        put(c.pos, Z * kPicks), put(c.sg, Z * FF), put(c.su, Z * FF), put(c.sh16, Z * FF), put(c.h16, Z * FF);
        put(c.y, Z * D);
        if (c.has_first) put(c.emb, Z * D);
        if (c.has_eg) put(c.eg_rows, Z * cols * (ehd + ehd / 32));
    }
    return off;
}

size_t Model::Impl::lay_prompt(Chunk& c, int C, uint8_t* device, uint8_t* host, size_t& host_bytes) const {
    const size_t off = lay_chunk(c, C, device);
    if (device != nullptr) {
        c.ring = c.ring_n > 0 ? device + off : nullptr;
        c.C = C;
    }
    size_t at = 0;
    auto put = [&](auto*& p, size_t n) {
        using T = std::remove_reference_t<decltype(*p)>;
        if (host != nullptr) p = (T*) (host + at);
        at += align256(n * sizeof(T) + 64);   // (halloc's slack)
    };
    if (c.S > 0) {
        at = (size_t) (0 - (uintptr_t) host) % 4096;
        put(c.stage, (size_t) c.S * c.stage_stride);
    }
    const size_t cols = (size_t) (pk.eg_ngram - 1) * pk.eg_heads, ehd = (size_t) pk.eg_hd;
    for (int sl = 0; sl < 2; ++sl) {
        if (c.has_first) put(c.emb_host[sl], (size_t) C * g.dim);
        if (c.has_eg) put(c.eg_host[sl], (size_t) C * pk.eg_layers * cols * (ehd + ehd / 32));
    }
    host_bytes = at + (c.S > 0 && host == nullptr ? 4095 : 0);
    return off + (size_t) c.ring_n * c.ring_stride;
}

bool Model::Impl::lay_out_prompt(int di, int C, uint8_t* device, uint8_t* host, std::string& err) {
    std::lock_guard<std::mutex> lk(warm_mu);   // (the warm-up uses the chunk's buffers)
    Chunk& c = *chunks[(size_t) di];
    Dev& d = devs[(size_t) di];
    cudaSetDevice(d.id);
    cudaError_t e = cudaSuccess;
    if (C > 0) {
        size_t host_bytes = 0;
        e = cudaMemsetAsync(device, 0, lay_prompt(c, C, device, host, host_bytes), d.st);
    } else {
        c.C = 0;
        e = cudaStreamSynchronize(c.cs);
    }
    chunk = C > 0 ? C : prompt_ceiling;   // (forward_chunks reads a prompt in chunks of at most `chunk`)
    if (e == cudaSuccess) e = cudaStreamSynchronize(d.st);
    if (e == cudaSuccess) return true;
    err = "the prompt path on CUDA" + std::to_string(d.id) + ": " + cudaGetErrorString(e);
    return false;
}

bool Model::Impl::raise_prompt_ceiling(int C, std::string& err) {
    if (C > kMaxChunk) {
        err = "the prompt path's kernels take chunks of at most " + std::to_string(kMaxChunk) + " tokens";
        return false;
    }
    std::lock_guard<std::mutex> lk(warm_mu);   // (the warm-up uses the chunk's buffers)
    try {
        host_allocs.reserve(host_allocs.size() + 1);
    } catch (const std::exception& e) {
        err = std::string("the prompt path's host allocation records could not grow: ") + e.what();
        return false;
    }
    chunk = C;   // (x_cap and chunk_hand_bytes size by it)
    bool ok = regrow_host(h_emb, (size_t) std::max(kWin, C) * g.dim, false, err) &&
              regrow_host(h_eg, get_engram_bytes((size_t) C), false, err);
    for (Src& sr : srcs) {
        ok = ok && (!sr.remote || regrow_host(sr.x_stage, x_cap() * (ckv_rb() + ik_rb()), true, err));
        ok = ok && (!sr.x_stage1 || regrow_host(sr.x_stage1, x_cap() * (ckv_rb() + ik_rb()), true, err));
    }
    for (const std::unique_ptr<Chunk>& c : chunks)
        for (int sl = 0; sl < 2; ++sl)
            ok = ok && (!c->hand[sl] || regrow_host(c->hand[sl], chunk_hand_bytes(), true, err));
    if (ok) {
        eg_cap = (size_t) C;
        prompt_ceiling = C;
    }
    chunk = prompt_ceiling;
    return ok;
}

void Model::Impl::chunk_dims(Chunk& c, int nm) const {
    const int NHL = g.heads / head_members(nm), HQ = NHL * g.hd, OG = g.o_groups / head_members(nm);
    int FF = 0, ff0_ = 0;
    split128(g.ff, nm, 0, ff0_, FF);
    c.HQ = HQ, c.OG = OG, c.FF = FF;
    c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
    c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
    c.w16_elems = (size_t) 8 << 20;
}

bool Model::Impl::alloc_chunks(std::string& err) {
    slab_auto_grew_ = false;
    chunks.resize(devs.size());
    for (size_t di = 0; di < devs.size(); ++di) {
        Dev& d = devs[di];
        cudaSetDevice(d.id);
        chunks[di] = std::make_unique<Chunk>();
        Chunk& c = *chunks[di];
        chunk_dims(c, (int) groups[(size_t) d.grp].mem.size());
        // the base slab: STRATA_DS4_A_MB megabytes of q.  Every dense GEMM of a layer is issued once a slab, so the slab width divides
        // the prompt's dense weight traffic.  A unified-memory GPU (Strix Halo, main 3133160ed) takes 32 MB: measured 8 -> 32 MB took
        // its 32K prompt 325.89 -> 357.78 tok/s, 16K 315.01 -> 328.04, 8K +1.6%, 4K flat, decode unmoved.  A discrete card keeps 8 MB
        // (the 0.2.6 slab): its VRAM is the expert tier's, and on the RTX 5070 a bigger slab bought +2% prompt for -3% decode.
        // STRATA_DS4_A_MB overrides the base on any GPU (8 restores the old slab).
        bool unified = false;
        {
            cudaDeviceProp ip{};
            unified = cudaGetDeviceProperties(&ip, d.id) == cudaSuccess && ip.integrated != 0;
        }
        static const size_t a_mb_env = [] {
            const char* v = std::getenv("STRATA_DS4_A_MB");
            return (size_t) (v != nullptr ? std::max(1, std::atoi(v)) : 0);
        }();
        const size_t a_mb = strata::ds4::attn_slab_base_mb(unified, a_mb_env, defaults_026());
        c.A = std::max(64, (int) ((a_mb << 20) / (size_t) c.HQ) / 64 * 64);
        // DS4_ATTN_SLAB=N: the attention slab's rows.  The dense projections (q_b, wo_a/b) are dequantized and GEMM'd
        // once a slab, so a slab of 256 rows dequantizes the same weights 32 times for an 8K prompt (measured: the
        // dense dequant 916 ms of the 8K prompt, rocBLAS 867 ms); a larger slab costs VRAM for c.q/c.o and the oa
        // buffers (A * HQ x 4 bytes each).  0.2.7: unset = automatic (attn_slab_rows: up to 2048 rows where the buffers
        // take VRAM no expert would have used), 0 = the 0.2.6 rule above, N = that many rows.
        {
            int env = -1;
            if (const char* v = std::getenv("DS4_ATTN_SLAB")) env = std::max(0, std::atoi(v));
            if ((slab_force_base_ || unified || a_mb_env != 0) && env < 0) env = 0;   // (no automatic growth on a unified-memory GPU, or when STRATA_DS4_A_MB chose the base)
            // per row: c.q + c.o (float), c.o16 (half), c.oa (float), c.oa16 (half)
            const double bpr = (double) c.HQ * 10.0 + (double) c.OG * (double) g.o_lora * 6.0;
            double spare = 1e18;   // the expert tier off (--vram-experts 0): the buffers cost no expert
            double share = 0.0;    // the share of the routed experts the cards hold (0 with the tier off)
            if (o.vram_experts > 0.0) {
                double exp_b = 0, room = 0, other_b = 0;
                share = gpu_expert_share(&exp_b, &room, &other_b);
                spare = (room - other_b - exp_b) / (double) std::max<size_t>(1, devs.size());
            }
            const int base = c.A;
            // one rule (ds4_experts_rule.hpp: attn_slab_rows): the chunk's whole slab (<= 1024 rows) where the cards hold half the
            // experts or more (4 x R9700, +9% prompt), up to 2048 rows where the VRAM is spare, the base otherwise
            c.A = strata::ds4::attn_slab_rows(env, chunk, base, spare, bpr, rdna4() ? share : 0.0, defaults_026());
            if (env < 0 && c.A > base) {
                slab_auto_grew_ = true;
            }
        }
        // (the indexer's slab: its scores rows are the context wide - about 64 MB of them)
        c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
        c.w16_elems = (size_t) 8 << 20;
        bool has_eg = false, has_first = false;
        size_t stride = 0;
        for (const Layer& l : L)
            for (int m = 0; m < l.nm; ++m)
                if (member(l, m).dev == (int) di && !l.mtp) {
                    has_eg = has_eg || l.eg >= 0;
                    has_first = has_first || l.il == 0;
                    stride = std::max(stride, align256(unit(member(l, m))));
                }
        c.has_eg = has_eg, c.has_first = has_first;
        if (!halloc(c.bounds_pin, 2048, err, kPromptLabel)) return false;
        if (stride > 0) {   // DS4_RING_MB (400) of slots in kParts batches
            const char* rm = std::getenv("DS4_RING_MB");
            c.ring_stride = stride;
            if (rm != nullptr && std::strcmp(rm, "link") == 0) {   // opt-in (Project Maya #609): batches of ~10 ms of this device's link
                const double link = link_h2d(d);
                size_t fr = 0, tot = 0;
                cudaSetDevice(d.id);
                cudaMemGetInfo(&fr, &tot);
                const double room = std::max(0.0, (double) fr - (double) ((size_t) o.reserve_mib << 20)) * o.vram_experts;
                c.ring_n = strata::ds4::ring_slots(link, stride, room, Chunk::kParts);
                std::fprintf(stderr, "ds4: CUDA%d prompt ring (DS4_RING_MB=link): %d slots of %.1f MB, %d a batch (%.0f MB) for its %.1f GB/s link\n",
                             d.id, c.ring_n, stride / 1048576.0, c.ring_n / Chunk::kParts, (double) c.ring_n * stride / 1048576.0, link / 1e9);
            } else {
                const size_t mb = rm ? (size_t) std::atoi(rm) : ring_mb_auto();
                c.ring_n = std::max(Chunk::kParts, (int) ((mb << 20) / stride) / Chunk::kParts * Chunk::kParts);
            }
        }
        CK(cudaStreamCreateWithFlags(&c.cs, cudaStreamNonBlocking));
        for (int i = 0; i < Chunk::kParts; ++i) {
            CK(cudaEventCreateWithFlags(&c.filled[i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&c.freed[i], cudaEventDisableTiming));
        }
        if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(c.blas, d.st) != CUBLAS_STATUS_SUCCESS) {
            err = "cuBLAS on CUDA" + std::to_string(d.id);
            return false;
        }
        // (the hand-offs between groups: none where one group holds every layer - a chunk's h alone is 80 KB a row)
        const bool hands = d.mi == 0 && groups.size() > 1;
        for (int sl = 0; sl < 2; ++sl) {
            if (hands && (!hmap(c.hand[sl], chunk_hand_bytes(), err, kPromptLabel) ||
                          !hmap(c.hand_flag[sl], 32, err, kPromptLabel)))
                return false;
            CK(cudaEventCreateWithFlags(&c.emb_ev[sl], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&c.eg_ev[sl], cudaEventDisableTiming));
        }
        if (hands && !hmap(c.hand_ack, (size_t) 2 * 8 * 32, err, kPromptLabel)) return false;
    }
    return true;
}

bool Model::Impl::gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy,
                       float beta, std::string& err) {
    if (rows <= 0) return true;
#if defined(STRATA_USE_HIP)   // (wmma_gemm.cu is part of the AMD builds only: a CUDA build has no such kernel and keeps cuBLAS)
    static const bool pf = strata::ds4::dense_pfgemm_default(env_tri("STRATA_DENSE_PFGEMM"), defaults_026());
#endif
    const int cols = std::max(128, (int) std::min<size_t>((size_t) m.n, c.w16_elems / (size_t) m.k) / 128 * 128);
    for (int c0 = 0; c0 < m.n; c0 += cols) {
        const int nc = std::min(cols, m.n - c0);
        if (m.fmt != 0) gg::rows(nc == m.n ? m : gg::view(m, c0, nc), c.w16, false, d.st);   // (a GGUF's)
        else x3::reconstruct_rows(nc == m.n ? m : x3::view(m, c0, nc), c.w16, false, d.st);
#if defined(STRATA_USE_HIP)
        if (pf && rows >= 64 && ((uintptr_t) x16 & 15u) == 0 &&
            strata_pf_gemm_f16_ld(x16, ldx, c.w16, m.k, y + c0, rows, nc, m.k, ldy, beta, d.st))
            continue;
#endif
        const float one = 1.0f;
        const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nc, rows, m.k, &one, c.w16, CUDA_R_16F,
                                               m.k, x16, CUDA_R_16F, ldx, &beta, y + c0, CUDA_R_32F, ldy,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            err = "cublasGemmEx: status " + std::to_string((int) st);
            return false;
        }
    }
    return true;
}

// ---- the busy experts (run_set)
int Model::Impl::fat_rows() {
    static const int n = [] {
        const int v = std::getenv("DS4_FAT_ROWS") ? std::max(0, std::atoi(std::getenv("DS4_FAT_ROWS"))) : 0;
        return v > 0 ? std::max(v, 17) : 0;   // (the light kernels take the experts of 16 rows or fewer)
    }();
    return n;
}

bool Model::Impl::fat_ready(Dev& d, Chunk& c, int D, int FF) {
#if defined(STRATA_USE_HIP)   // (CUDA only: HIP is unmeasured)
    (void) d, (void) c, (void) D, (void) FF;
    return false;
#else
    if (fat_rows() <= 0 || c.fat_off) return false;
    if (c.fat_w != nullptr) return true;
    static const int R = std::getenv("DS4_FAT_PIECE") ? std::max(64, std::atoi(std::getenv("DS4_FAT_PIECE"))) : 1024;
    if (cudaSetDevice(d.id) != cudaSuccess || cudaMalloc(&c.fat_w, (size_t) 2 * FF * D * 2) != cudaSuccess ||
        cudaMalloc(&c.fat_x, (size_t) R * D * 2) != cudaSuccess ||
        cudaMalloc(&c.fat_gu, (size_t) R * 2 * FF * 4) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "ds4: CUDA%d: no VRAM for the busy experts' path (%.0f MB): the tiles take them\n", d.id,
                     ((double) 2 * FF * D * 2 + (double) R * D * 2 + (double) R * 2 * FF * 4) / 1e6);
        c.fat_off = true;
        return false;
    }
    c.fat_r = R;
    std::fprintf(stderr, "ds4: CUDA%d busy experts (%d rows or more) as FP16 GEMMs, %d rows a piece (%.0f MB)\n", d.id,
                 fat_rows(), R, ((double) 2 * FF * D * 2 + (double) R * D * 2 + (double) R * 2 * FF * 4) / 1e6);
    return true;
#endif
}

bool Model::Impl::fat_gate_up(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int row, int n, int a, int D,
                              int FF, cudaStream_t s, std::string& err) {
    const int tg = gg::expert_type(sa.lay, 0);
    if (!gg::iq::f16_rows(tg, blob + sa.lay.trellis[0], D, FF, c.fat_w, D, s) ||
        !gg::iq::f16_rows(tg, blob + sa.lay.trellis[1], D, FF, c.fat_w + (size_t) FF * D, D, s)) {
        err = "the busy experts' path: gate / up not decoded";
        return false;
    }
    const float one = 1.0f, zero = 0.0f;
    for (int p0 = 0; p0 < n; p0 += c.fat_r) {
        const int np = std::min(c.fat_r, n - p0);
        gg::iq::gather_f16(c.x16, D, c.row_tok + row + p0, np, D, c.fat_x, s);
        if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * FF, np, D, &one, c.fat_w, CUDA_R_16F, D, c.fat_x,
                         CUDA_R_16F, D, &zero, c.fat_gu, CUDA_R_32F, 2 * FF, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
            err = "the busy experts' path: cublasGemmEx (gate / up) failed";
            return false;
        }
        gg::iq::swiglu_f16(c.fat_gu, 2 * FF, np, FF, sa.limit, c.h16 + (size_t) (row - a + p0) * FF, FF, s);
    }
    return true;
}

bool Model::Impl::fat_down(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int lr0, int n, int D, int FF,
                           cudaStream_t s, std::string& err) {
    if (!gg::iq::f16_rows(gg::expert_type(sa.lay, 2), blob + sa.lay.trellis[2], FF, D, c.fat_w, FF, s)) {
        err = "the busy experts' path: down not decoded";
        return false;
    }
    const float one = 1.0f, zero = 0.0f;
    if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, D, n, FF, &one, c.fat_w, CUDA_R_16F, FF,
                     c.h16 + (size_t) lr0 * FF, CUDA_R_16F, FF, &zero, c.y + (size_t) lr0 * D, CUDA_R_32F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
        err = "the busy experts' path: cublasGemmEx (down) failed";
        return false;
    }
    return true;
}

bool Model::Impl::gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows,
                            float alpha, float* y, int ldy, std::string& err) {
    const float zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, W, CUDA_R_16F, k,
                                           x16, CUDA_R_16F, ldx, &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        err = "cublasGemmEx (F16 weights): status " + std::to_string((int) st);
        return false;
    }
    return true;
}

bool Model::Impl::hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes,
                               std::string& err) {
    const int hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, fn, hcd, h, hcd, &zero, mixes, 32) !=
        CUBLAS_STATUS_SUCCESS) {
        err = "cublasSgemm (mHC mixes)";
        return false;
    }
    hc_mix_scale(h, hcd, n_mix, T, g.eps, mixes, d.st);
    return true;
}

bool Model::Impl::stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                                const uint8_t* src) {
    auto cp = [&](uint8_t* d0, const uint8_t* s0, size_t n) {
        return cudaMemcpyAsync(d0, s0, n, cudaMemcpyHostToDevice, cs) == cudaSuccess;
    };
    bool ok = true;
    if (l.lay.fmt == gg::kFmtMXFP4) {   // the blob
        if (src != nullptr)
            ok = cp(dst, src, l.blob);
        else
            for (int r = 0; r < 3; ++r)
                ok = ok && cp(dst + l.lay.trellis[r], mx_mat(l, e, r), mx_part_bytes(l, r));
    } else {   // the whole expert in exllamav3's layout, its pieces at piece_src's offsets
        const uint8_t* p = src != nullptr ? src : file_part(l, e);
        for (int r = 0; r < 3; ++r)
            ok = ok && cp(dst + l.lay.trellis[r], p + l.piece_src[r][0], l.piece_bytes[r][0]) &&
                 cp(dst + l.lay.suh[r], p + l.piece_src[r][1], (size_t) l.hl.k[r] * 2) &&
                 cp(dst + l.lay.svh[r], p + l.piece_src[r][2], (size_t) l.hl.n[r] * 2);
    }
    if (ok)
        strata::core::count_gpu_traffic(devs[(size_t) member(l, m).dev].id, unit(member(l, m)),
                                        0, m == 0 ? 1 : 0);
    if (!ok) err = "streaming layer " + std::to_string(l.il) + " expert " + std::to_string(e) + " failed";
    return ok;
}

int Model::Impl::skip_unpicked_rows() {
    static const int n = [] {
        const char* v = std::getenv("DS4_SKIP_UNPICKED");
        if (v != nullptr && std::atoi(v) != 0) return INT_MAX;
        const char* r = std::getenv("DS4_SKIP_UNPICKED_ROWS");
        return r != nullptr ? std::max(0, std::atoi(r)) : 0;
    }();
    return n;
}

void Model::Impl::read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
    static const double cap = (std::getenv("DS4_READAHEAD_MB") ? std::atof(std::getenv("DS4_READAHEAD_MB")) : 2048.0) *
                              1048576.0;
    std::deque<std::pair<size_t, double>> issued;   // (batch, bytes) not yet copied
    double ahead = 0;
    for (size_t q = 0; q < S.b.size() && !stop.load(std::memory_order_relaxed);) {
        const size_t done = S.done.load(std::memory_order_relaxed);
        while (!issued.empty() && issued.front().first < done) {
            ahead -= issued.front().second;
            issued.pop_front();
        }
        if (ahead > cap) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }
        const Batch& bt = S.b[q];
        const Layer& l0 = L[(size_t) bt.il];
        double nb = 0;
        if (S.skips(bt.k)) {   // (the picked ones only are read, once the layer is routed)
            issued.push_back({q++, 0.0});
            continue;
        }
        if (l0.dev == di)
            for (int i = 0; i < bt.n; ++i) {
                const int e = l0.host_list[(size_t) (bt.j0 + i)];
                if (l0.lay.fmt == gg::kFmtMXFP4) {
                    for (int r = 0; r < 3; ++r)
                        platform::advise(pk.fds[(size_t) l0.mx_file[r]], (l0.mx_off[r] + (uint64_t) e * l0.mx_stride[r]), mx_part_bytes(l0, r), true);
                    nb += (double) l0.blob;
                    continue;
                }
                platform::advise(pk.fds[(size_t) l0.region_file[(size_t) e]], l0.region_off[(size_t) e], l0.region_bytes[(size_t) e], true);
                nb += (double) l0.region_bytes[(size_t) e];
            }
        issued.push_back({q++, nb});
        ahead += nb;
    }
}

bool Model::Impl::copy_ahead(Chunk& c, Streamer& S, int di, std::string& err) {
    const int P = Chunk::kParts;
    while (S.cp < S.b.size() && S.cp < S.kp + P) {
        const Batch& bt = S.b[S.cp];
        // a chunk that streams only its picks waits for its layer's routing
        const bool skip_unpicked = S.skips(bt.k);
        if (skip_unpicked && (bt.il != S.routed_layer || bt.k != S.routed_chunk)) break;
        const int half = (int) (S.cp % P);
        const Layer& l0 = L[(size_t) bt.il];
        const auto picked = [&](int i) {
            if (!skip_unpicked) return true;
            const size_t v = (size_t) l0.n_slots + (size_t) bt.j0 + (size_t) i;   // (the streamed ones' virtual slots)
            return S.routed_bounds[v + 1] > S.routed_bounds[v];
        };
        int mi = 0;
        for (int m = 0; m < l0.nm; ++m)
            if (member(l0, m).dev == di) mi = m;
        if (S.cp >= (size_t) P) CK(cudaStreamWaitEvent(c.cs, c.freed[half], 0));
        uint8_t* sb = nullptr;   // in place: the batch's regions in the part's staging slots
        if (c.stage != nullptr) {
            const double t_evt = now_ms();
            if (S.cp >= (size_t) P) CK(cudaEventSynchronize(c.filled[half]));   // (its last batch's copies out)
            c.dbg_evt_ms += now_ms() - t_evt;
            sb = c.stage + (size_t) half * (c.ring_n / P) * c.stage_stride;
            std::atomic<bool> bad{false};
            const double t_st = now_ms();
            c.readers->run(bt.n, [&](int i) {
                if (!picked(i)) return;   // (an expert no row of the chunk routed to)
                const int e = l0.host_list[(size_t) (bt.j0 + i)];
                if (l0.lay.fmt == gg::kFmtMXFP4 ? !read_mx_file(l0, e, sb + (size_t) i * c.stage_stride)
                                                : !pk.read_at(l0.region_file[(size_t) e], l0.region_off[(size_t) e],
                                                              l0.region_bytes[(size_t) e],
                                                              sb + (size_t) i * c.stage_stride))
                    bad.store(true);
            });
            c.dbg_stage_ms += now_ms() - t_st;
            if (bad.load()) {
                err = "reading layer " + std::to_string(l0.il) + "'s experts from the model's files failed";
                return false;
            }
        }
        const double t_dma = now_ms();
        for (int i = 0; i < bt.n; ++i) {
            if (!picked(i)) {
                ++S.skipped;
                g_skip_count.skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (skip_unpicked) {
                ++S.copied;
                g_skip_count.copied.fetch_add(1, std::memory_order_relaxed);
            }
            uint8_t* dst = c.ring + ((size_t) half * (c.ring_n / P) + i) * c.ring_stride;
            if (!stream_expert(l0, mi, l0.host_list[(size_t) (bt.j0 + i)], dst, c.cs, err,
                               sb != nullptr ? sb + (size_t) i * c.stage_stride : nullptr))
                return false;
            c.stream_bytes += (double) unit(member(l0, mi));
        }
        c.dbg_dma_ms += now_ms() - t_dma;
        CK(cudaEventRecord(c.filled[half], c.cs));
        ++S.cp;
        S.done.store(S.cp, std::memory_order_relaxed);
    }
    return true;
}

// ---- the flow (see Chunk)
void Model::Impl::flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
    c.items.emplace_back();
    Chunk::Item& it = c.items.back();
    it.il = l.il;
    it.mi = mi;
    it.k = k;
    it.e = e;
    it.v = v;
    // (the caller's pool holds it: copied into staging by io, then DMA'd - no read)
    const bool in_ram = !o.expert_pages->transient(l.il, e);
    const uint8_t* blob = in_ram ? o.expert_pages->blob(l.il, e) : nullptr;
    if (in_ram && o.expert_pages->pinned(l.il, e)) {   // (page-locked: DMA'd from where it lies)
        for (int r = 0; r < 3; ++r) it.src[r] = blob + l.lay.trellis[r];
        ++c.it_ram;
        return;
    }
    it.ram = blob;
    bool hit = false;
    if (!in_ram && c.lc != nullptr) {   // layer-major: from the layer cache, or kept for the chunks after
        const size_t i = c.base + c.items.size() - 1, stride = (l.blob + 63) & ~(size_t) 63;
        if (c.lc_il != l.il) {   // (a layer's first item: the slots the layer before used are its own again
            c.lc_il = l.il;      // once that layer's items are all read - `after` below)
            c.lc_used = 0;
            c.lc_n = (int) std::min<size_t>(c.lc_of.size(), c.lc_bytes / stride);
            c.lc_first = i;
            std::fill(c.lc_of.begin(), c.lc_of.end(), -1);
        }
        const int j = c.lc_of[(size_t) e];
        if (j >= 0) {   // (copied into staging by io once its reader is done, then DMA'd)
            it.ram = c.lc + (size_t) j * stride;
            it.after = c.lc_writer[(size_t) j];
            hit = true;
        } else if (k + 1 < (int) c.fl_T.size() && c.lc_used < c.lc_n) {   // (a later chunk may pick it)
            const int jn = c.lc_used++;
            c.lc_of[(size_t) e] = jn;
            c.lc_writer[(size_t) jn] = (int64_t) i;
            it.keep = c.lc + (size_t) jn * stride;
            it.after = (int64_t) c.lc_first - 1;
        }
    }
    it.stage = (int) (c.n_file++ % (uint64_t) c.S);
    uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
    for (int r = 0; r < 3; ++r) {
        const uint64_t off = l.mx_off[r] + (uint64_t) e * l.mx_stride[r];
        it.src[r] = sb + (in_ram || hit ? 0 : off % platform::kDirectAlign);
        sb += flow_region(r);
    }
    it.left.store(3, std::memory_order_relaxed);
    ++(in_ram ? c.it_ram : hit ? c.it_lc : c.it_file);
}

void Model::Impl::flow_spec(Chunk& c, int k, int il, int T) {
    static const double spec_q = std::getenv("DS4_SPEC_Q") ? std::atof(std::getenv("DS4_SPEC_Q")) : 0.9;
    Layer& l = L[(size_t) il];
    const int mi = flow_mi(c, l);
    int* vs = member(l, mi).vs_pin;
    c.spec_k = k;
    c.spec_il = il;
    c.spec_begin = c.base + c.items.size();
    // (the likely ones lead the layer's virtual slots, and the experts' outputs are summed in that order: from the
    //  profile as loaded, not the hits this process added since)
    const uint32_t* pr = prof_load.data() + (size_t) il * g.n_exp;
    double sum = 0;
    for (int e = 0; e < l.n_exp; ++e) sum += pr[e];
    std::vector<std::pair<double, int>> cand;
    const double picks = (double) l.k_exp * T;
    // (early reads only while they hit: once a run's likely ones went unpicked more often than DS4_SPEC_WASTE
    // (0.2) of the time, the rest of the run reads only what the routing picked - an unpicked one costs a whole
    // read, and how well the profile guesses depends on the PC's history and the prompt: V100, 64 tokens - 376
    // of 921 likely ones unpicked, 7 GB of 84 read for nothing; Radeon 8065S, 128 and 512 tokens - 44 of 945 and
    // 21 of 1568, the early reads 9-12% faster there)
    static const double waste = std::getenv("DS4_SPEC_WASTE") ? std::atof(std::getenv("DS4_SPEC_WASTE")) : 0.2;
    const bool early = c.it_spec < 2u * (uint64_t) c.ring_n || (double) c.it_spec_unused <= waste * (double) c.it_spec;
    for (int e : l.host_list) {
        if (!early) break;
        const double p = sum > 0 ? pr[e] / sum : 0.0;
        const double q = p >= 1.0 ? 1.0 : 1.0 - std::exp(picks * std::log1p(-p));
        if (q >= spec_q && spec_q <= 1.0) cand.push_back({q, e});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    });
    if (cand.size() > (size_t) (c.S + c.ring_n)) cand.resize((size_t) (c.S + c.ring_n));
    std::vector<char> is((size_t) l.n_exp, 0);
    int j = 0;
    for (const auto& x : cand) {
        vs[x.second] = l.pv_n + j++;
        is[(size_t) x.second] = 1;
    }
    for (int e : l.host_list)
        if (!is[(size_t) e]) vs[e] = l.pv_n + j++;
    for (const auto& x : cand) flow_push(c, l, mi, k, x.second, vs[x.second]);
    c.n_spec = (int) cand.size();
    c.it_spec += cand.size();
}

void Model::Impl::flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
    const Layer& l = L[(size_t) il];
    const int mi = flow_mi(c, l);
    const int* vs = member(l, mi).vs_pin;
    for (size_t i = c.spec_begin; i < c.spec_begin + (size_t) c.n_spec; ++i) {
        const int v = fit(c, i).v;
        if (b[(size_t) v + 1] == b[(size_t) v]) ++c.it_spec_unused;
    }
    for (int e : l.host_list) {   // (index order: their virtual slots ascending)
        const int v = vs[e];
        if (v >= l.pv_n + c.n_spec && b[(size_t) v + 1] > b[(size_t) v]) flow_push(c, l, mi, k, e, v);
    }
}

bool Model::Impl::flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
    while (flow_next1(c, k, il, k2, il2)) {
        if (c.fl_yoco < 0 || k2 >= c.fl_cut_k || il2 < c.fl_yoco) return true;
        k = k2;
        il = il2;
    }
    return false;
}

bool Model::Impl::flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const {
    const auto it = std::find(c.fl_layers.begin(), c.fl_layers.end(), il);
    const size_t i = (size_t) (it - c.fl_layers.begin());
    if (c.lm) {   // (layer-major: the layer's next chunk, then the next layer's first)
        if (k + 1 < (int) c.fl_T.size()) {
            k2 = k + 1;
            il2 = il;
            return true;
        }
        if (i + 1 >= c.fl_layers.size()) return false;
        k2 = 0;
        il2 = c.fl_layers[i + 1];
        return true;
    }
    if (i + 1 < c.fl_layers.size()) {
        k2 = k;
        il2 = c.fl_layers[i + 1];
        return true;
    }
    if (k + 1 < (int) c.fl_T.size() && !c.fl_layers.empty()) {
        k2 = k + 1;
        il2 = c.fl_layers[0];
        return true;
    }
    return false;
}

bool Model::Impl::flow_pump(Chunk& c, std::string& err) {
    const size_t end = c.base + c.items.size();
    IoQueue::Job jobs[6];
    while (c.n_read < end) {
        Chunk::Item& it = fit(c, c.n_read);
        if (it.after >= 0) {   // layer-major: a slot's expert copied out of it once its reader is done; a slot
            // written again once the layer before's items are all read (none still copies out of it)
            bool wait = false;
            if (it.keep == nullptr)
                wait = (size_t) it.after >= c.base && fit(c, (size_t) it.after).left.load(std::memory_order_acquire) != 0;
            else {
                while (c.lc_chk < c.n_read &&
                       (c.lc_chk < c.base || fit(c, c.lc_chk).left.load(std::memory_order_acquire) == 0))
                    ++c.lc_chk;
                wait = c.lc_chk <= (size_t) it.after;
            }
            if (wait) break;
        }
        if (it.stage >= 0) {
            const int64_t prev = c.stage_item[(size_t) it.stage];
            if (prev >= 0) {
                if ((size_t) prev >= c.n_dma) break;
                const cudaError_t q = cudaEventQuery(c.stage_ev[(size_t) it.stage]);
                if (q == cudaErrorNotReady) break;
                CK(q);
            }
            c.stage_item[(size_t) it.stage] = (int64_t) c.n_read;
            const Layer& l = L[(size_t) it.il];
            uint8_t* sb = c.stage + (size_t) it.stage * c.stage_stride;
            int nj = 0;
            for (int r = 0; r < 3; ++r) {
                IoQueue::Job& j = jobs[nj++];
                const platform::fd_t pf = pk.pfds.empty() ? platform::kBadFd : pk.pfds[(size_t) l.mx_file[r]];
                j.fd = pf != platform::kBadFd ? pf : pk.fds[(size_t) l.mx_file[r]];
                j.dfd = pk.dfds.empty() || c.buffered ? platform::kBadFd : pk.dfds[(size_t) l.mx_file[r]];
                j.off = l.mx_off[r] + (uint64_t) it.e * l.mx_stride[r];
                j.len = mx_part_bytes(l, r);
                j.dst = sb;
                j.left = &it.left;
                j.bad = &it.bad;
                j.mem = it.ram != nullptr ? it.ram + l.lay.trellis[r] : nullptr;
                j.keep = it.keep != nullptr ? it.keep + l.lay.trellis[r] : nullptr;
                sb += flow_region(r);
            }
            c.io->push(jobs, nj);
        }
        ++c.n_read;
    }
    const size_t R = (size_t) c.ring_n, dma0 = c.n_dma;
    struct Kick {   // (every way out: what was queued is submitted)
        cudaStream_t s;
        const size_t &a, b;
        ~Kick() {
            if (a != b) kick(s);
        }
    } kq{c.cs, c.n_dma, dma0};
    while (c.n_dma < c.n_read && c.n_dma < c.n_comp + R) {
        Chunk::Item& it = fit(c, c.n_dma);
        if (it.left.load(std::memory_order_acquire) != 0) break;
        if (it.bad.load(std::memory_order_acquire)) {
            err = "reading layer " + std::to_string(it.il) + " expert " + std::to_string(it.e) +
                  " from the model's files failed";
            return false;
        }
        const size_t r = c.n_dma % R;
        if (c.n_dma >= R) CK(cudaStreamWaitEvent(c.cs, c.ring_free[r], 0));
        const Layer& l = L[(size_t) it.il];
        uint8_t* dst = c.ring + r * c.ring_stride;
        for (int p = 0; p < 3; ++p)
            CK(cudaMemcpyAsync(dst + l.lay.trellis[p], it.src[p], mx_part_bytes(l, p), cudaMemcpyHostToDevice,
                               c.cs));
        CK(cudaEventRecord(c.ring_fill[r], c.cs));
        if (it.stage >= 0) CK(cudaEventRecord(c.stage_ev[(size_t) it.stage], c.cs));
        c.stream_bytes += (double) unit(l);
        strata::core::count_gpu_traffic(devs[(size_t) c.dev].id, unit(l), 0, it.mi == 0 ? 1 : 0);
        ++c.n_dma;
    }
    return true;
}

bool Model::Impl::flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b,
                             std::string& err) {
    cudaStream_t s = d.st;
    const Layer& vx = member(L0, mi);   // (a member's: its part of each VRAM expert, L0's slots)
    if (c.spec_k != k || c.spec_il != L0.il) {
        err = "the prompt path's expert stream is out of step (layer " + std::to_string(L0.il) + ")";
        return false;
    }
    flow_rest(c, k, L0.il, b);
    const size_t lb = c.spec_begin, le = c.base + c.items.size(), n_lay = le - lb;
    int k2 = 0, il2 = 0;
    if (flow_next(c, k, L0.il, k2, il2)) flow_spec(c, k2, il2, fl_rows(c, k2, il2));
    const size_t R = (size_t) c.ring_n, bmax = std::max<size_t>(1, R / 2);
    int v_done = 0;
    while (c.n_comp < le) {
        // a batch's bounds are fixed by the stream alone (bmax items, never across the ring's end), never by what
        // has landed: the experts' outputs are summed batch after batch (the VRAM ones' shares between them), so
        // the sums come out the same on every run whatever the reads' timing
        const size_t i0 = c.n_comp, i1 = std::min({le, i0 + bmax, (i0 / R + 1) * R});
        if (!flow_pump(c, err)) return false;
        if (c.n_dma < i1) {   // not all of the batch landed: wait for a read, or for a staging slot's copy
            const double tw = now_ms();
            kick(s);
            if (c.n_read > c.n_dma) c.io->wait_any(500);
            else {
                const Chunk::Item& it = fit(c, c.n_read);
                if (it.stage >= 0) CK(cudaEventSynchronize(c.stage_ev[(size_t) it.stage]));
            }
            c.wait_ms += now_ms() - tw;
            continue;
        }
        const int n = (int) (i1 - i0);
        int* bc = c.bc_pin + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
        int* bcd = c.bc_dev + (size_t) (c.n_batch % (uint64_t) c.bc_n) * c.bc_w;
        ++c.n_batch;
        // (the items' virtual slots ascend and the slots between them have no rows: their bounds side by side)
        for (int m = 0; m < n; ++m) bc[m] = b[(size_t) fit(c, i0 + (size_t) m).v];
        bc[n] = b[(size_t) fit(c, i1 - 1).v + 1];
        if (bc[n] > bc[0]) {
            CK(cudaStreamWaitEvent(s, c.ring_fill[(i1 - 1) % R], 0));
            CK(cudaMemcpyAsync(bcd, bc, (size_t) (n + 1) * 4, cudaMemcpyHostToDevice, s));
            if (!run_set(d, c, L0, c.ring + (i0 % R) * c.ring_stride, 0, c.ring_stride, bcd, bc, 0, n, T, err))
                return false;
        }
        for (size_t i = i0; i < i1; ++i) CK(cudaEventRecord(c.ring_free[i % R], s));
        kick(s);
        c.n_comp = i1;
        if (!flow_pump(c, err)) return false;
        const int v_to = (int) ((int64_t) L0.pv_n * (int64_t) (c.n_comp - lb) / (int64_t) std::max<size_t>(1, n_lay));
        if (!run_vram(d, c, L0, vx, b.data(), v_done, v_to, T, err)) return false;
        v_done = v_to;
    }
    if (!run_vram(d, c, L0, vx, b.data(), v_done, L0.pv_n, T, err)) return false;
    while (c.base < c.n_comp && c.items.size() > 4096) {   // (done with: read, copied and used)
        c.items.pop_front();
        ++c.base;
    }
    return true;
}

bool Model::Impl::run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride,
                          const int* d_bounds, const int* b, int i0, int i1, int T, std::string& err,
                          const unsigned long long* tab) {
    const int D = g.dim, FF = l.ffn;
    while (i0 < i1) {
        int j = i0 + 1;
        while (j < i1 && b[j + 1] - b[i0] <= c.C) ++j;
        const int a = b[i0], nrows = b[j] - a;
        int max_rows = 0;
        for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, b[i + 1] - b[i]);
        if (nrows > 0) {
            x3::SetArgs sa;
            if (tab != nullptr)
                sa.wtab = (const uint8_t* const*) (tab + (i0 - wb0));
            else
                sa.wbase = wbase + (size_t) (i0 - wb0) * stride;   // (wbase: set member wb0's blob)
            sa.stride = stride;
            sa.lay = l.lay;
            sa.n_exp = j - i0;
            sa.bounds = d_bounds + i0;
            sa.row0 = a;
            sa.max_rows = max_rows;
            sa.n_embd = D;
            sa.n_ff = FF;
            sa.limit = g.lim;
            // gate / up read the layer's F16 copy of the activations (the same rounding the tiles make for their
            // shared tile): with 18-36 column blocks an expert the F32 rows were read that many times
            sa.x16 = strata::ds4::gate_up_x16_default(defaults_026()) ? c.x16 : nullptr;
            // the busy experts (DS4_FAT_ROWS=N: those of N rows or more): their weights decoded to FP16 once and
            // multiplied by the library's GEMM - the tiles decode an expert's weights again for every 64 of its rows
            std::vector<int> fat;
            if (sa.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(sa.lay) && D % 256 == 0 && FF % 256 == 0 &&
                gg::expert_type(sa.lay, 1) == gg::expert_type(sa.lay, 0) &&
                gg::iq::f16_rows_ok(gg::expert_type(sa.lay, 0)) && gg::iq::f16_rows_ok(gg::expert_type(sa.lay, 2)) &&
                fat_ready(d, c, D, FF)) {
                int rest = 0;
                for (int i = i0; i < j; ++i) {
                    const int r = b[i + 1] - b[i];
                    if (r >= fat_rows()) fat.push_back(i);
                    else rest = std::max(rest, r);
                }
                if (!fat.empty()) {
                    sa.skip_ge = fat_rows();
                    sa.max_rows = rest;
                }
            }
            EkMark ek{};
            const bool ekp = prof_chunk;
            if (ekp) {
                std::lock_guard<std::mutex> lk(prof_mu);
                ek.e0 = ek_ev();
                ek.e1 = ek_ev();
                ek.e2 = ek_ev();
                ek.tg = gg::expert_type(sa.lay, 0);
                ek.td = gg::expert_type(sa.lay, 2);
                ek.nrows = nrows;
                ek.nexp = j - i0;
                ek.maxrows = max_rows;
                ek.ff = FF;
                cudaGetDevice(&ek.dev);
                cudaEventRecord(ek.e0, d.st);
            }
            if (sa.lay.fmt == gg::kFmtMXFP4) {
                gg::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                for (const int i : fat)
                    if (!fat_gate_up(c, sa, sa.wbase + (size_t) (i - i0) * stride, b[i], b[i + 1] - b[i], a, D, FF,
                                     d.st, err))
                        return false;
                if (ekp) cudaEventRecord(ek.e1, d.st);
                gg::set_down(sa, c.h16, FF, c.y, D, d.st);
                for (const int i : fat)
                    if (!fat_down(c, sa, sa.wbase + (size_t) (i - i0) * stride, b[i] - a, b[i + 1] - b[i], D, FF,
                                  d.st, err))
                        return false;
            } else {
                x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st);
                if (ekp) cudaEventRecord(ek.e1, d.st);
                x3::set_down(sa, c.h16, FF, c.y, D, d.st);
            }
            if (ekp) {
                cudaEventRecord(ek.e2, d.st);
                std::lock_guard<std::mutex> lk(prof_mu);
                ek_marks.push_back(ek);
            }
            if (dump_y != nullptr && (dump_y_layer < 0 || l.il == dump_y_layer)) {
                std::vector<int> bb(b + i0, b + j + 1);
                std::vector<uint16_t> hv((size_t) nrows * FF);
                std::vector<float> yv((size_t) nrows * D);
                CK(cudaStreamSynchronize(d.st));
                CK(cudaMemcpy(hv.data(), c.h16, hv.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(yv.data(), c.y, yv.size() * 4, cudaMemcpyDeviceToHost));
                std::lock_guard<std::mutex> lk(dump_mu);
                const int64_t hdr[6] = {l.il, a, nrows, j - i0, wbase == c.ring || (wbase > c.ring &&
                                        wbase < c.ring + (size_t) c.ring_n * c.ring_stride) ? 1 : 0, max_rows};
                std::fwrite(hdr, 8, 6, dump_y);
                std::fwrite(bb.data(), 4, bb.size(), dump_y);
                std::fwrite(hv.data(), 2, hv.size(), dump_y);
                std::fwrite(yv.data(), 4, yv.size(), dump_y);
                std::fflush(dump_y);
            }
            moe_combine(c.acc, c.y, c.pos, c.rw, T, l.k_exp, a, a + nrows, D, d.st);
        }
        i0 = j;
    }
    (void) err;
    return true;
}

// ---- a run of chunks through the layers
bool Model::Impl::layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                              bool kv_only) {
    Layer& l = member(L0, mi);
    Dev& d = devs[(size_t) l.dev];
    Chunk& c = *chunks[(size_t) l.dev];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    prof_mark(d, 0);
    const int D = g.dim, H = g.hc, HD = g.hd, NHL = l.ahn, HQ = NHL * HD, OG = l.ogn, FF = l.ffn;
    const float* freqs = l.ratio ? d.freqs_comp : d.freqs_win;
    // Engram, in slabs of 256 tokens
    if (l.eg >= 0) {
        const int cols = (pk.eg_ngram - 1) * pk.eg_heads, ehd = pk.eg_hd;
        const size_t per = (size_t) cols * (ehd + ehd / 32);
        const int slot = wp.eg_slot >= 0 ? wp.eg_slot : wp.k % 2;   // (read ahead by the run: two in flight)
        uint8_t* host = c.eg_host[slot] + (size_t) l.eg * c.C * per;
        CK(cudaMemcpyAsync(c.eg_rows, host, (size_t) T * per, cudaMemcpyHostToDevice, s));
        strata::core::count_gpu_traffic(d.id, (uint64_t) T * per, 0);
        CK(cudaEventRecord(c.eg_ev[slot], s));
        for (int s0 = 0; s0 < T; s0 += 256) {
            const int ns = std::min(256, T - s0);
            engram_dequant_tok(c.eg_rows + (size_t) s0 * per, per, cols, ns, ehd, c.eg_x, s, (int) eg_fmt[(size_t) l.eg]);
            x3::rows_f16(c.eg_x, cols * ehd, nullptr, ns, cols * ehd, c.eg_x16, s);
            if (!gemm(d, c, l.eg_wkv, c.eg_x16, cols * ehd, ns, c.eg_kv, (H + 1) * D, 0.0f, err)) return false;
            engram_gate(c.h + (size_t) s0 * H * D, c.eg_kv, l.eg_w, H, D, ns, g.eps, s);
        }
    }
    prof_mark(d, 4);
    // attention
    if (!hc_mix_chunk(d, c, c.h, l.hca_fn, T, c.mix, err)) return false;
    hc_split(c.mix, l.hca_scale, l.hca_base, H, g.sinkhorn, g.hc_eps, T, c.a_pre, c.a_post, c.a_comb, s);
    dump_sub(d, "mix0", l.il, p0, T, c.mix, (size_t) T * 32 * 4);
    hc_pre_norm(c.h, c.pre, l.attn_norm, D, H, T, g.eps, c.x, s);
    dump_sub(d, "a_mix", l.il, p0, T, c.mix, (size_t) T * 32 * 4);
    dump_sub(d, "a_pre", l.il, p0, T, c.a_pre, (size_t) T * 8 * 4);
    dump_sub(d, "a_post", l.il, p0, T, c.a_post, (size_t) T * 8 * 4);
    dump_sub(d, "a_comb", l.il, p0, T, c.a_comb, (size_t) T * 64 * 4);
    dump_sub(d, "attn_x", l.il, p0, T, c.x, (size_t) T * D * 4);
    x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
    prof_mark(d, 5);
    if (!kv_only && (!gemm(d, c, l.wq_a, c.x16, D, T, c.qr, g.q_lora, 0.0f, err) ||
                     !gemm(d, c, l.wkv, c.x16, D, T, c.kvr, HD, 0.0f, err)))
        return false;
    if (l.kv_src && !gemm(d, c, l.c_wkv, c.x16, D, T, c.ckr, HD, 0.0f, err)) return false;
    if (l.kv_src && l.ratio > 1 && !gemm(d, c, l.c_wgate, c.x16, D, T, c.cgr, HD, 0.0f, err)) return false;
    prof_mark(d, 6);
    const bool snap = !kv_only && wp.snap_n > 0 && mi == 0;
    if (!kv_only) {
        dump_sub(d, "qr_raw", l.il, p0, T, c.qr, (size_t) T * g.q_lora * 4);
        dump_sub(d, "kvr_raw", l.il, p0, T, c.kvr, (size_t) T * HD * 4);
        Finish fq;
        fq.w = l.q_norm;
        fq.eps = g.eps;
        finish(c.qr, T, g.q_lora, g.q_lora, fq, s);
        x3::rows_f16(c.qr, g.q_lora, nullptr, T, g.q_lora, c.qr16, s);
        // the window rows: the ring's (positions p0 - win ..) then the chunk's
        ring_to_ext(l.win, c.ext, p0, g.win, HD, s);
        Finish fk;
        fk.w = l.kv_norm;
        fk.eps = g.eps;
        fk.rd = g.rd;
        fk.pos = (int) p0;
        fk.freqs = freqs;
        fk.quant = 1;
        fk.dst = c.ext + (size_t) g.win * HD;
        fk.dst_stride = HD;
        finish(c.kvr, T, HD, HD, fk, s);
        // the states kept on the way (feed_mark): the ring as it was before each row snap_t - the old
        // ring, then the chunk's rows before it
        for (int i = 0; snap && i < wp.snap_n; ++i) {
            uint8_t* sw = L0.snap_win[wp.snap_ix[i]];
            if (sw == nullptr) continue;
            CK(cudaMemcpyAsync(sw, l.win, (size_t) g.win * win_rb(), cudaMemcpyDeviceToDevice, s));
            if (wp.snap_t[i] > 0) ext_to_ring(c.ext, sw, p0, g.win, wp.snap_t[i], HD, s, nullptr);
        }
        ext_to_ring(c.ext, l.win, p0, g.win, T, HD, s, nullptr);
    }
    KvRows ckv;
    const long long ss = (long long) o.max_context + 64;
    if (l.ratio > 0) {
        Src& sr = srcs[(size_t) l.src];
        const int R = l.ratio;
        const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
        const int ng = (int) std::max<int64_t>(0, g_end - g_first);
        const float* latent = c.latent;
        if (l.kv_src) {
            if (R == 1) {
                Finish fc;
                fc.w = l.c_norm;
                fc.eps = g.eps;
                finish(c.ckr, T, HD, HD, fc, s);
                latent = c.ckr;
            } else {
                for (int i = 0; snap && i < wp.snap_n; ++i) {   // (each slot: the latest row before snap_t that holds it)
                    float *skv = L0.snap_kv[wp.snap_ix[i]], *ssc = L0.snap_sc[wp.snap_ix[i]];
                    if (skv == nullptr) continue;
                    const int64_t ps = p0 + wp.snap_t[i];
                    for (int sl = 0; sl < R; ++sl) {
                        const int64_t q = ps - 1 - (((ps - 1 - sl) % R) + R) % R;
                        const float* kv = q >= p0 ? c.ckr + (size_t) (q - p0) * HD : l.c_kv + (size_t) sl * HD;
                        const float* sc = q >= p0 ? c.cgr + (size_t) (q - p0) * HD : l.c_score + (size_t) sl * HD;
                        CK(cudaMemcpyAsync(skv + (size_t) sl * HD, kv, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                        CK(cudaMemcpyAsync(ssc + (size_t) sl * HD, sc, (size_t) HD * 4, cudaMemcpyDeviceToDevice, s));
                    }
                }
                compress_win(c.ckr, c.cgr, l.c_kv, l.c_score, R, (int) p0, T, HD, l.c_norm, g.eps, c.latent, s);
            }
        }
        const int clen_max = (int) ((p0 + T) / R);
        if (l.ix_src) {
            Src& ks = srcs[(size_t) l.ik_src];
            if (l.owns_k && ng > 0) {
                x3::rows_f16(latent, HD, nullptr, ng, HD, c.lat16, s);
                if (!gemm(d, c, l.i_wk, c.lat16, HD, ng, c.ikr, g.ix_hd, 0.0f, err)) return false;
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
                finish(c.ikr, ng, g.ix_hd, g.ix_hd, fi, s);
            }
            if (clen_max > 0 && !kv_only) {
                if (!gemm_f16w(c, l.i_wproj, g.ix_heads, D, c.x16, D, T,
                               1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), c.iw, 64, err))
                    return false;
                for (int a0 = 0; a0 < T; a0 += c.AI) {
                    const int na = std::min(c.AI, T - a0);
                    const int64_t pa = p0 + a0;
                    if (!gemm(d, c, l.i_wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.iq,
                              g.ix_heads * g.ix_hd, 0.0f, err))
                        return false;
                    Finish fi;
                    fi.rd = g.rd;
                    fi.pos = (int) pa;
                    fi.vpp = g.ix_heads;
                    fi.freqs = freqs;
                    fi.quant = 3;
                    finish(c.iq, na * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);
                    Rows lim;
                    lim.gen = 1;
                    lim.p0 = pa;
                    lim.R = R;
                    lim.kcap = g.ix_topk;
                    lim.ccap = g.cand_topk;
                    lim.blk = std::max(1, g.cand_blk);
                    const int nlast = (int) ((pa + na) / R);   // the slab's widest query
                    int* tk = c.topk + (size_t) a0 * topk_stride();
                    int* cd = c.cand + (size_t) a0 * cand_stride();
                    const float* iw = c.iw + (size_t) a0 * 64;
                    if (nlast <= 0) continue;
                    if (l.uses_cand && (nlast + g.cand_blk - 1) / g.cand_blk > g.cand_topk) {
                        Rows cl = lim;
                        cl.gen = 2;
                        index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, g.cand_topk * g.cand_blk,
                                    cd, (int) cand_stride(), g.cand_blk, cl, c.scores, ss, s);
                        Rows tr = lim;
                        tr.gen = 3;
                        topk(c.scores, ss, na, tr, cd, (int) cand_stride(), g.cand_blk, tk, (int) topk_stride(), s);
                    } else {
                        index_score(c.iq, iw, ks.ik[(size_t) l.dev], g.ix_heads, g.ix_hd, na, nlast, nullptr, 0, 1,
                                    lim, c.scores, ss, s);
                        if (l.cand_src) {
                            Rows cr = lim;
                            cr.kcap = g.cand_topk;
                            cand_blocks(c.scores, ss, na, cr, g.cand_blk, c.bm, (long long) bm_stride(), cd,
                                        (int) cand_stride(), s);
                        }
                        topk(c.scores, ss, na, lim, nullptr, 0, 1, tk, (int) topk_stride(), s);
                    }
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
            finish(const_cast<float*>(latent), ng, HD, HD, fc, s);
            if (mi == 0 && sr.remote) {   // the new rows for the other groups' readers (slot k % 2, once
                const int slot = wp.k % 2;  // they took the chunk before last's)
                if (wp.k >= 2)
                    for (int r : sr.readers) {
                        host_wait_at_least(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, wp.win - 2);
                        wait_flags(sr.x_ack + ((size_t) slot * devs.size() + r) * 32, 1, 32, wp.win - 2, s);
                    }
                put_new_rows(sr, l, g_first, ng, sr.stage(slot), sr.flag(slot), wp.win, d.xcnt, s);
            }
        }
        ckv = sr.ckv[(size_t) l.dev];
    }
    if (kv_only) return true;
    prof_mark(d, 7);
    // the attention in slabs: q, the sparse attention, wo_a, wo_b (this member's share) into acc
    if (NHL == 0) CK(cudaMemsetAsync(c.acc, 0, (size_t) T * D * sizeof(*c.acc), s));   // (no heads)
    for (int a0 = 0; a0 < T && NHL > 0; a0 += c.A) {
        const int na = std::min(c.A, T - a0);
        if (!gemm(d, c, l.wq_b, c.qr16 + (size_t) a0 * g.q_lora, g.q_lora, na, c.q, HQ, 0.0f, err)) return false;
        Finish fr;
        fr.rd = g.rd;
        fr.pos = (int) (p0 + a0);
        fr.vpp = NHL;
        fr.freqs = freqs;
        finish(c.q, na * NHL, HD, HD, fr, s);
        prof_mark(d, 8);
        if (a0 == 0 && na == T) dump_sub(d, "q", l.il, p0, T, c.q, (size_t) T * HQ * 4);
        AttnRows ar;
        ar.gen = 1;
        ar.p0 = p0;
        ar.t0 = a0;
        ar.win = g.win;
        ar.R = l.ratio;
        ar.kcap = g.ix_topk;
        sparse_attn(c.q, c.ext, ckv, c.topk + (size_t) a0 * topk_stride(), (int) topk_stride(), ar, l.sink, NHL, HD,
                    na, 1.0f / std::sqrt((float) HD), c.o, s, kKvFp4, 0, &d.attn_scr);   // (a short last slab)
        prof_mark(d, 9);
        if (a0 == 0 && na == T) dump_sub(d, "o", l.il, p0, T, c.o, (size_t) T * HQ * 4);
        Finish fo;
        fo.rd = g.rd;
        fo.pos = (int) (p0 + a0);
        fo.vpp = NHL;
        fo.freqs = freqs;
        fo.inverse = 1;
        fo.quant = 1;
        finish(c.o, na * NHL, HD, HD, fo, s);
        if (a0 == 0 && na == T) dump_sub(d, "o_fin", l.il, p0, T, c.o, (size_t) T * HQ * 4);
        x3::rows_f16(c.o, HQ, nullptr, na, HQ, c.o16, s);
        const int per = HQ / OG;
        for (int gi = 0; gi < OG; ++gi)
            if (!gemm(d, c, l.wo_a[gi], c.o16 + (size_t) gi * per, HQ, na, c.oa + (size_t) gi * g.o_lora,
                      OG * g.o_lora, 0.0f, err))
                return false;
        x3::rows_f16(c.oa, OG * g.o_lora, nullptr, na, OG * g.o_lora, c.oa16, s);
        if (!gemm(d, c, l.wo_b, c.oa16, OG * g.o_lora, na, c.acc + (size_t) a0 * D, D, 0.0f, err)) return false;
        if (a0 == 0 && na == T) dump_sub(d, "ao", l.il, p0, T, c.acc, (size_t) T * D * 4);
    }
    dump_sub(d, "h_in", l.il, p0, T, c.h, (size_t) T * H * D * 4);
    hc_post_inplace(c.acc, c.h, c.a_post, c.a_comb, D, H, T, s);
    if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, 1000 + l.il, p0, T);   // (after the attention, as the window path)
    prof_mark(d, 1);
    // the MoE
    if (!hc_mix_chunk(d, c, c.h, l.hcf_fn, T, c.mix, err)) return false;
    hc_split(c.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn, g.hc_eps, T, c.f_pre, c.f_post, c.f_comb, s);
    hc_pre_norm(c.h, c.a_pre, l.ffn_norm, D, H, T, g.eps, c.x, s);
    x3::rows_f16(c.x, D, nullptr, T, D, c.x16, s);
    gemv_f32(l.gate_w, c.x, l.n_exp, D, T, c.glog, 1024, s);   // (F32 x: the token path's bits)
    for (int r0 = 0; r0 < T; r0 += kVlRows) {   // (route()'s image bits cover kVlRows rows a call)
        const int nr = std::min(kVlRows, T - r0);
        VlBits vb;
        const bool vl = l.gate_bias_vl != nullptr && vl_bits(p0 + r0, nr, vb);
        route(c.glog + (size_t) r0 * 1024, l.gate_bias, l.n_exp, l.k_exp, l.k_model, nr, g.route_scale,
              c.ridx + (size_t) r0 * kPicks, c.rw + (size_t) r0 * kPicks, s, l.gate_bias_vl, vl ? &vb : nullptr);
    }
    const int n_vs = L0.pv_n + (int) L0.host_list.size();
    CK(cudaMemcpyAsync(l.vs_dev, l.vs_pin, (size_t) l.n_exp * 4, cudaMemcpyHostToDevice, s));   // (this member's)
    moe_sort(c.ridx, T, l.k_exp, l.vs_dev, n_vs, c.bounds, c.row_tok, c.pos, s);
    CK(cudaMemcpyAsync(c.bounds_pin, c.bounds, (size_t) (n_vs + 1) * 4, cudaMemcpyDeviceToHost, s));
    prof_mark(d, 10);
    // the shared expert into acc while the host waits for the bounds
    if (!gemm(d, c, l.s1, c.x16, D, T, c.sg, FF, 0.0f, err) || !gemm(d, c, l.s3, c.x16, D, T, c.su, FF, 0.0f, err))
        return false;
    swiglu(c.sg, c.su, T * FF, g.lim, c.sg, s);
    x3::rows_f16(c.sg, FF, nullptr, T, FF, c.sh16, s);
    if (!gemm(d, c, l.s2, c.sh16, FF, T, c.acc, D, 0.0f, err)) return false;
    prof_mark(d, 2);
    // the expert stream's copies keep going while the device runs this layer's attention, routing and shared expert:
    // the next items (this layer's likely experts, queued when the layer before routed) go out as they are ready and
    // ring slots free, not only once the bounds are back (Project Maya #360; DS4_FLOW_PUMP_WAIT=0: wait as before)
    static const bool pump_wait = std::getenv("DS4_FLOW_PUMP_WAIT") == nullptr ||
                                  std::atoi(std::getenv("DS4_FLOW_PUMP_WAIT")) != 0;
    if (c.flow && pump_wait)
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (!flow_pump(c, err)) return false;
            std::this_thread::yield();
        }
    CK(cudaStreamSynchronize(s));
    std::vector<int> b(c.bounds_pin, c.bounds_pin + n_vs + 1);
    if (dump_r != nullptr && mi == 0) {
        const size_t nk = (size_t) T * l.k_exp, np = (size_t) T * kPicks;
        std::vector<int> ri(np), rt(nk);
        std::vector<float> gl((size_t) T * l.n_exp), gb((size_t) l.n_exp);
        CK(cudaMemcpy(ri.data(), c.ridx, np * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rt.data(), c.row_tok, nk * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy2D(gl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                        cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gb.data(), l.gate_bias, gb.size() * 4, cudaMemcpyDeviceToHost));
        std::lock_guard<std::mutex> lk(dump_mu);
        const int64_t hdr[8] = {l.il, p0, T, l.k_exp, l.n_exp, n_vs, L0.pv_n, kPicks};
        std::fwrite(hdr, 8, 8, dump_r);
        std::fwrite(ri.data(), 4, np, dump_r);
        std::fwrite(L0.vs_pin, 4, (size_t) l.n_exp, dump_r);
        std::fwrite(b.data(), 4, b.size(), dump_r);
        std::fwrite(rt.data(), 4, nk, dump_r);
        std::fwrite(gl.data(), 4, gl.size(), dump_r);
        std::fwrite(gb.data(), 4, gb.size(), dump_r);
        std::fflush(dump_r);
    }
    if (mi == 0 && !l.mtp && (pred_f != nullptr || xin_f != nullptr)) {
        std::lock_guard<std::mutex> lk(dump_mu);
        if (pred_f != nullptr) {
            float* dp = nullptr;
            CK(strata::core::allocate_device(&dp, (size_t) T * 256 * 4,
                {"other", "Device allocations", "DeepSeek working buffers"}));
            strata_ds4_launch(pred_proj_k, T, 256, 0, s, c.x, (int) D, dp);
            std::vector<float> hp((size_t) T * 256), hl((size_t) T * l.n_exp);
            CK(cudaMemcpyAsync(hp.data(), dp, hp.size() * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpy2DAsync(hl.data(), (size_t) l.n_exp * 4, c.glog, 1024 * 4, (size_t) l.n_exp * 4, (size_t) T,
                                 cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            CK(strata::core::release_device(dp));
            std::vector<uint16_t> o(hp.size() + hl.size());
            for (size_t i = 0; i < hl.size(); ++i) o[i] = strata::kernels::f16_from_f32(hl[i]);
            for (size_t i = 0; i < hp.size(); ++i) o[hl.size() + i] = strata::kernels::f16_from_f32(hp[i]);
            const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (T & 255), (uint8_t) (T >> 8)};
            std::fwrite(hd, 1, 3, pred_f);
            std::fwrite(o.data(), 2, o.size(), pred_f);
        }
        if (xin_f != nullptr && xin_got[(size_t) l.il] < xin_n) {   // up to 32 evenly spaced tokens of the chunk
            const int tk = std::min(std::min(T, 32), xin_n - xin_got[(size_t) l.il]);
            const int stride = std::max(1, T / tk);
            std::vector<float> hx((size_t) T * D);
            CK(cudaMemcpyAsync(hx.data(), c.x, hx.size() * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            const uint8_t hd[3] = {(uint8_t) l.il, (uint8_t) (tk & 255), (uint8_t) (tk >> 8)};
            std::fwrite(hd, 1, 3, xin_f);
            for (int i = 0; i < tk; ++i) std::fwrite(hx.data() + (size_t) std::min(T - 1, i * stride) * D, 4, (size_t) D, xin_f);
            xin_got[(size_t) l.il] += tk;
        }
    }
    if (mi == 0 && trace_f != nullptr && !l.mtp) {   // DS4_TRACE: a chunk's picks, u8 (layer | 0x80), u16 T, T x k int16 ids
        std::vector<int> ri((size_t) T * kPicks);
        CK(cudaMemcpy(ri.data(), c.ridx, ri.size() * 4, cudaMemcpyDeviceToHost));
        std::lock_guard<std::mutex> lk(dump_mu);
        const uint8_t hd[3] = {(uint8_t) (l.il | 0x80), (uint8_t) (T & 255), (uint8_t) (T >> 8)};
        std::fwrite(hd, 1, 3, trace_f);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < l.k_exp; ++i) {
                const int16_t v = (int16_t) ri[(size_t) t * kPicks + i];
                std::fwrite(&v, 2, 1, trace_f);
            }
    }
    if (mi == 0) {   // the profile (the CPU lane's experts: by the plan's order)
        uint32_t* pr = prof.data() + (size_t) l.il * g.n_exp;
        int n_picked = 0;
        // Swaps stay within a layer, so fully resident layers need no prompt counts.
        const bool lacking = !usage.empty() && !l.mtp &&
                             std::any_of(l.slot.begin(), l.slot.end(), [](int s) { return s < 0; });
        const float per = strata::core::get_prompt_routing_weight(strata::core::kPromptRoutings, l.k_exp,
                                                                  (int64_t) T * l.k_exp);
        for (int e = 0; e < l.n_exp; ++e) {
            const int v = L0.vs_pin[e];
            const int cnt = b[(size_t) v + 1] - b[(size_t) v];
            pr[e] += (uint32_t) cnt;
            if (cnt > 0) {
                ++n_picked;
                if (lacking)
                    count_usage(l, e, per * (float) cnt);
            }
        }
        static const bool ustat = std::getenv("DS4_UNPICKED_STAT") != nullptr;
        if (ustat && !l.mtp) g_unpicked_stat.add(T, n_picked, l.n_exp);
        st.experts += (int64_t) T * l.k_exp;
    }
    if (c.flow) {
        if (!flow_layer(d, c, L0, mi, wp.k, T, b, err)) return false;
        hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
        CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
        prof_mark(d, 3);
        return true;
    }
    // the streamed experts as their batches arrive (each batch's slots freed for the copies at once), the VRAM ones
    // in shares between them (the device works while the copies run)
    const int di = l.dev, P = Chunk::kParts;
    int nb = 0;
    for (size_t q = S.kp; q < S.b.size() && S.b[q].il == L0.il && S.b[q].k == wp.k; ++q) ++nb;
    if (S.skips(wp.k) && nb > 0) {   // (this layer routed: its picked experts are the ones to copy)
        S.routed_layer = L0.il;
        S.routed_chunk = wp.k;
        S.routed_bounds = b;
        if (!copy_ahead(c, S, di, err)) return false;
    }
    int v_done = 0;
    for (int bi = 0; bi < nb; ++bi) {
        const Batch& bt = S.b[S.kp];
        const int half = (int) (S.kp % P);
        CK(cudaStreamWaitEvent(s, c.filled[half], 0));
        const int v0 = L0.pv_n + bt.j0;
        if (!run_set(d, c, l, c.ring + (size_t) half * (c.ring_n / P) * c.ring_stride, v0, c.ring_stride, c.bounds,
                     b.data(), v0, v0 + bt.n, T, err))
            return false;
        CK(cudaEventRecord(c.freed[half], s));
        ++S.kp;
        if (!copy_ahead(c, S, di, err)) return false;
        const int v_to = (int) ((int64_t) L0.pv_n * (bi + 1) / nb);
        if (!run_vram(d, c, l, l, b.data(), v_done, v_to, T, err)) return false;
        v_done = v_to;
    }
    if (!run_vram(d, c, l, l, b.data(), v_done, L0.pv_n, T, err)) return false;
    hc_post_inplace(c.acc, c.h, c.f_post, c.f_comb, D, H, T, s);
    CK(cudaMemcpyAsync(c.pre, c.f_pre, (size_t) T * 8 * 4, cudaMemcpyDeviceToDevice, s));
    prof_mark(d, 3);
    return true;
}

void Model::Impl::plan_streams() {
    for (Layer& l : L) {
        if (l.mtp || l.vs_pin == nullptr) continue;
        l.host_list.clear();
        for (int e = 0; e < l.n_exp; ++e)
            if (l.slot[(size_t) e] < 0 || l.slot[(size_t) e] >= l.pv_n) l.host_list.push_back(e);
        for (int m = 0; m < l.nm; ++m) {   // (a group's members: each its own copy)
            Layer& x = member(l, m);
            int j = 0;
            for (int e = 0; e < l.n_exp; ++e)
                x.vs_pin[e] = l.slot[(size_t) e] >= 0 && l.slot[(size_t) e] < l.pv_n ? l.slot[(size_t) e] : l.pv_n + j++;
        }
    }
}

Xfer Model::Impl::chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
    const size_t nh = (size_t) T * g.hc * g.dim * 4, np = (size_t) T * 8 * 4, nk = (size_t) T * topk_stride() * 4,
                 nc = (size_t) T * cand_stride() * 4;
    void* dv[4] = {c.h, c.pre, c.topk, c.cand};
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

bool Model::Impl::run_layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                            const Run& run, int k, int il) {
    if (run.kv_only(k, il)) return layer_chunk(L0, mi, p0, T, wp, S, err, true);
    const int r0 = run.r0(k, il);
    if (r0 <= 0) return layer_chunk(L0, mi, p0, T, wp, S, err);
    if (il == run.yoco) {
        WinPlan w0 = wp;
        w0.snap_n = 0;
        if (!layer_chunk(L0, mi, p0, r0, w0, S, err, true)) return false;
    }
    Chunk& c = *chunks[(size_t) member(L0, mi).dev];
    float* const h0 = c.h;
    float* const pre0 = c.pre;
    c.h += (size_t) r0 * g.hc * g.dim;
    c.pre += (size_t) r0 * 8;
    WinPlan w1 = wp;
    w1.snap_n = 0;
    for (int i = 0; i < wp.snap_n; ++i)
        if (wp.snap_t[i] >= r0) {
            w1.snap_t[w1.snap_n] = wp.snap_t[i] - r0;
            w1.snap_ix[w1.snap_n++] = wp.snap_ix[i];
        }
    const bool ok = layer_chunk(L0, mi, p0 + r0, T - r0, w1, S, err);
    c.h = h0;
    c.pre = pre0;
    return ok;
}

bool Model::Impl::run_dev_layers(int di, const Run& run, std::string& err) {
    Dev& d = devs[(size_t) di];
    Chunk& c = *chunks[(size_t) di];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, H = g.hc, nk = (int) run.T.size();
    Streamer S;   // (none: the flow streams)
    const size_t hb = chunk_hand_bytes();
    // the Engram rows: (layer, chunk) after (layer, chunk) in the run's order, each read on a helper thread while
    // the one before runs (slot u % 2, once unit u - 2's copy of it is done)
    std::vector<std::pair<int, int>> egu;   // (Engram layer, chunk)
    for (int il : c.fl_layers)
        if (L[(size_t) il].eg >= 0)
            for (int k = 0; k < nk; ++k)
                if (run.runs(k, il)) egu.push_back({L[(size_t) il].eg, k});
    const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
    auto read_eg = [this, &c, &run, &egu, eg_per](size_t u) {
        if (u >= 2) cudaEventSynchronize(c.eg_ev[u % 2]);
        const int li = egu[u].first, k = egu[u].second;
        engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[u % 2] + (size_t) li * c.C * eg_per);
    };
    size_t eg_u = 0;
    std::future<void> eg_next;
    if (!egu.empty()) eg_next = std::async(std::launch::async, read_eg, (size_t) 0);
    for (size_t li = 0; li < c.fl_layers.size(); ++li) {
        const int il = c.fl_layers[li];
        Layer& L0 = L[(size_t) il];
        const bool last_l = li + 1 == c.fl_layers.size();
        for (int k = 0; k < nk; ++k) {
            if (!run.runs(k, il)) continue;   // (a skipped early chunk: nothing after the yoco layer)
            if (halt_now()) return halt_end(di, err);   // (a Stop: one GPU, this one)
            const int64_t p0 = run.p0[(size_t) k];
            const int T = run.T[(size_t) k], slot = k % 2;
            const bool last_k = k + 1 == nk;
            WinPlan wp;
            wp.par = true;
            wp.async = true;
            wp.win = run.seq[(size_t) k];
            wp.k = k;
            run.snaps(p0, T, wp);
            if (L0.eg >= 0) {   // this unit's rows ready, the next one's on their way
                const double te = now_ms();
                eg_next.get();
                st.engram_ms += now_ms() - te;
                wp.eg_slot = (int) (eg_u % 2);
                if (++eg_u < egu.size()) eg_next = std::async(std::launch::async, read_eg, eg_u);
            }
            if (li == 0) {   // the chunk's input: its embedding rows
                if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                for (int t = 0; t < T; ++t)
                    std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                (size_t) D * 2);
                CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                CK(cudaEventRecord(c.emb_ev[slot], s));
                embed(c.emb, T, c.h, D, H, s);
                if (ds4_dbg("DS4_DBG_EMB")) {
                    float *hp = c.h;
                    float hh[4] = {0,0,0,0};
                    cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                }
                init_pre(c.pre, T, s);
            } else {   // or its state after the layer before
                const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, false);
                for (int i = 0; i < x.n; ++i)
                    CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyHostToDevice, s));
                strata::core::count_gpu_traffic(d.id, x.total(), 0);
            }
            if (last_k && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                if (it != g.d_targets.end()) {
                    const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                    const int nr = std::min(T, tap_cap);
                    hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                    CK(cudaStreamWaitEvent(s, ev_seed, 0));
                    CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                         (size_t) nr, cudaMemcpyDeviceToHost, s));
                }
            }
            const bool kvo = run.kv_only(k, il);
            if (!run_layer(L0, 0, p0, T, wp, S, err, run, k, il)) return false;
            if (dump_l != nullptr) dump_layer(d, c.h, il, p0, T);
            if (kvo) continue;   // (no layer after it reads this chunk's state)
            if (!last_l) {   // the chunk's state kept for the next layer
                const Xfer x = chunk_xfer(c.lm_state + (size_t) k * hb, c, T, true);
                for (int i = 0; i < x.n; ++i)
                    CK(cudaMemcpyAsync(x.dst[i], x.src[i], x.bytes[i], cudaMemcpyDeviceToHost, s));
                strata::core::count_gpu_traffic(d.id, 0, x.total());
                continue;
            }
            // through every layer: the chunk's end, and the logits
            if ((size_t) k < chunk_done.size()) {
                CK(cudaEventRecord(chunk_done[(size_t) k], s));
                chunk_rec[(size_t) k].store(1, std::memory_order_release);
            }
            if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                for (int r0 = 0; r0 < T; r0 += kWin) {
                    const int nr = std::min(kWin, T - r0);
                    hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                    if (!compute_logits(d, d.x, nr, err)) return false;
                    CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                       (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                }
            } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                const int t0 = T - n_logits;
                hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                if (!compute_logits(d, d.x, n_logits, err)) return false;
            }
        }
        lm_layers.store((int) li + 1, std::memory_order_release);
    }
    return true;
}

bool Model::Impl::run_dev_chunks(int di, const Run& run, std::string& err) {
    Dev& d = devs[(size_t) di];
    Chunk& c = *chunks[(size_t) di];
    cudaSetDevice(d.id);
    cudaStream_t s = d.st;
    const int D = g.dim, H = g.hc, nk = (int) run.T.size();
    Streamer S;
    const int B = std::max(1, c.ring_n / Chunk::kParts);
    if (c.flow) {   // (the flow decides its streams as the layers route; nothing planned here)
        CK(cudaStreamSynchronize(s));
        CK(cudaStreamSynchronize(c.cs));
        c.items.clear();
        c.base = c.n_read = c.n_dma = c.n_comp = 0;
        c.n_file = 0;
        std::fill(c.stage_item.begin(), c.stage_item.end(), -1);
        c.fl_layers.clear();
        for (int il = 0; il < g.nl; ++il)   // (a group's layers: on each member)
            for (int m = 0; m < L[(size_t) il].nm; ++m)
                if (member(L[(size_t) il], m).dev == di) c.fl_layers.push_back(il);
        c.fl_T = run.T;
        c.fl_yoco = run.yoco;
        c.fl_cut_k = run.cut_k;
        c.fl_cut_r = run.cut_r;
        c.spec_k = c.spec_il = -1;
        c.wait_ms = 0;
        c.it_file = c.it_ram = c.it_spec = c.it_spec_unused = c.it_lc = 0;
        c.lc_il = -1;
        c.lc_used = 0;
        c.lc_first = c.lc_chk = 0;
        c.io_d0 = c.io->direct_bytes();
        c.io_b0 = c.io->buffered_bytes();
        c.io_busy0 = c.io->busy_ms();
        // (a short prompt's reads through the page cache, which keeps them for the decode after it - as the CPU
        // lane's reads in windows did: one V100, Maya-S from an NVMe, decode after 17 tokens 12.7 tokens/s read
        // past the cache, 14.0 after windows; a long one's past it, more than the cache holds - DS4_FLOW_BUFFERED,
        // 1024 tokens on one GPU, 0 on more: the longest run read through it)
        static const int buf_max = std::getenv("DS4_FLOW_BUFFERED") ? std::atoi(std::getenv("DS4_FLOW_BUFFERED"))
                                                                    : 0;   // (Strata: off until measured - Maya's 1024 on one GPU)
        int run_n = 0;
        for (int t : run.T) run_n += t;
        c.buffered = run_n <= buf_max;
        if (!c.fl_layers.empty()) {
            flow_spec(c, 0, c.fl_layers[0], run.T[0]);
            if (!flow_pump(c, err)) return false;
        }
        if (c.lm) return run_dev_layers(di, run, err);
    }
    for (int k = 0; k < nk && !c.flow; ++k)
        for (int il = 0; il < g.nl; ++il) {
            if (!run.runs(k, il) || run.kv_only(k, il)) continue;   // (a long prompt's skip: no experts there)
            const Layer& L0 = L[(size_t) il];
            bool here = false;
            for (int m = 0; m < L0.nm; ++m) here = here || member(L0, m).dev == di;
            if (!here) continue;
            for (int j0 = 0; j0 < (int) L0.host_list.size(); j0 += B)
                S.b.push_back({k, il, j0, std::min(B, (int) L0.host_list.size() - j0)});
        }
    if (skip_unpicked_rows() > 0) {   // (before the read-ahead thread starts: it reads skip_k)
        S.skip_k.assign((size_t) nk, 0);
        for (int k = 0; k < nk; ++k) S.skip_k[(size_t) k] = run.T[(size_t) k] <= skip_unpicked_rows();
    }
    if (!S.b.empty() && c.ring == nullptr) {
        err = "no slots to stream experts into on CUDA" + std::to_string(d.id);
        return false;
    }
    std::atomic<bool> ra_stop{false};
    std::thread ra;
    if (!S.b.empty()) ra = std::thread([this, &S, di, &ra_stop] { read_ahead(S, di, ra_stop); });
    struct Join {   // (every way out of the run: the read-ahead ends before S does)
        std::atomic<bool>& stop;
        std::thread& th;
        ~Join() {
            stop.store(true);
            if (th.joinable()) th.join();
        }
    } ra_join{ra_stop, ra};
    if (!copy_ahead(c, S, di, err)) return false;
    // this device's Engram layers: chunk k's rows read on a helper thread while chunk k - 1 runs (slot k % 2, once
    // chunk k - 2's copy of it is done)
    std::vector<int> egs;
    for (const Layer& l0 : L)
        for (int m = 0; m < l0.nm; ++m)
            if (member(l0, m).dev == di && !l0.mtp && l0.eg >= 0) egs.push_back(l0.eg);
    const size_t eg_per = (size_t) (pk.eg_ngram - 1) * pk.eg_heads * (pk.eg_hd + pk.eg_hd / 32);
    auto read_eg = [this, &c, &run, &egs, eg_per](int k) {
        if (k >= 2) cudaEventSynchronize(c.eg_ev[k % 2]);
        for (int li : egs)
            engram_rows(run.p0[(size_t) k], run.T[(size_t) k], li, c.eg_host[k % 2] + (size_t) li * c.C * eg_per);
    };
    std::future<void> eg_next;
    if (!egs.empty()) eg_next = std::async(std::launch::async, read_eg, 0);
    for (int k = 0; k < nk; ++k) {
        if (run_yields) {
            const int back = k - ((int) groups.size() + kYieldSlack);
            if (back >= 0)
                while (!(chunk_rec[(size_t) back].load(std::memory_order_acquire) &&
                         cudaEventQuery(chunk_done[(size_t) back]) == cudaSuccess)) {
                    if (past_halt(k)) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
        }
        if (!chunk_go(k)) return halt_end(di, err);   // (a Stop: the chunks every device reaches are done)
        const int64_t p0 = run.p0[(size_t) k];
        const int T = run.T[(size_t) k], slot = k % 2;
        const bool last_k = k + 1 == nk;
        if (!egs.empty()) {   // chunk k's rows ready, chunk k + 1's on their way
            const double te = now_ms();
            eg_next.get();
            if (d.mi == 0) st.engram_ms += now_ms() - te;
            if (k + 1 < nk) eg_next = std::async(std::launch::async, read_eg, k + 1);
        }
        WinPlan wp;
        wp.par = true;
        wp.async = true;
        wp.win = run.seq[(size_t) k];
        wp.k = k;
        run.snaps(p0, T, wp);
        bool started = false;
        std::vector<char> got(srcs.size(), 0);
        for (int il = 0; il < g.nl && run.runs(k, il); ++il) {
            if (halt_now()) return halt_end(di, err);   // (one GPU: after the layer it is on)
            Layer& L0 = L[(size_t) il];
            int mi = -1;
            for (int m = 0; m < L0.nm; ++m)
                if (member(L0, m).dev == di) mi = m;
            if (mi < 0) continue;
            if (!started) {   // the chunk's input: its embedding rows, or the previous group's hand-off
                started = true;
                if (il == 0) {
                    if (k >= 2) CK(cudaEventSynchronize(c.emb_ev[slot]));
                    for (int t = 0; t < T; ++t)
                        std::memcpy(c.emb_host[slot] + (size_t) t * D, emb_row(p0 + t, toks[(size_t) (p0 + t)]),
                                    (size_t) D * 2);
                    CK(cudaMemcpyAsync(c.emb, c.emb_host[slot], (size_t) T * D * 2, cudaMemcpyHostToDevice, s));
                    CK(cudaEventRecord(c.emb_ev[slot], s));
                    embed(c.emb, T, c.h, D, H, s);
                    if (ds4_dbg("DS4_DBG_EMB")) {
                        float *hp = c.h;
                        float hh[4] = {0,0,0,0};
                        cudaMemcpy(hh, hp, sizeof(hh), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "DBG at embed site %d: h[0..3] = %g %g %g %g\n", __LINE__, hh[0], hh[1], hh[2], hh[3]);
                    }
                    init_pre(c.pre, T, s);
                } else {
                    Chunk& a = *chunks[(size_t) L[(size_t) il - 1].dev];
                    host_wait_at_least(a.hand_flag[slot], wp.win);
                    xfer_get(chunk_xfer(a.hand[slot], c, T, false), a.hand_flag[slot], wp.win, s, d.id);
                    set_flag(a.hand_ack + ((size_t) slot * 8 + mi) * 32, wp.win, s);
                }
            }
            for (int sidx : {L0.src, L0.ix_src ? L0.ik_src : -1}) {
                if (sidx < 0 || got[(size_t) sidx]) continue;
                got[(size_t) sidx] = 1;
                Src& sr = srcs[(size_t) sidx];
                if (L[(size_t) sr.layer].grp == L0.grp) continue;
                take_new_rows(sr, di, p0, T, sr.stage(slot), sr.flag(slot), wp.win, s, true);
                set_flag(sr.x_ack + ((size_t) slot * devs.size() + di) * 32, wp.win, s);   // (taken, or none)
            }
            if (last_k && mi == 0 && g.n_mtp > 0) {   // the drafter's taps: the last chunk's last rows
                const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
                if (it != g.d_targets.end()) {
                    const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
                    const int nr = std::min(T, tap_cap);
                    hc_mean(c.h + (size_t) (T - nr) * H * D, nr, H, D, d.tap, D, s);
                    CK(cudaStreamWaitEvent(s, ev_seed, 0));
                    CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4,
                                         (size_t) nr, cudaMemcpyDeviceToHost, s));
                }
            }
            if (!run_layer(L0, mi, p0, T, wp, S, err, run, k, il)) return false;
            if (dump_l != nullptr && mi == 0) dump_layer(d, c.h, il, p0, T);
            const bool last = il + 1 == g.nl || L[(size_t) il + 1].grp != L0.grp;
            if (last && mi == 0 && il + 1 < g.nl) {   // the hand-off, once the next group took the chunk before last
                const int nnext = L[(size_t) il + 1].nm;
                if (k >= 2) {
                    for (int m = 0; m < nnext; ++m)
                        host_wait_at_least(c.hand_ack + ((size_t) slot * 8 + m) * 32, wp.win - 2);
                    wait_flags(c.hand_ack + (size_t) slot * 8 * 32, nnext, 32, wp.win - 2, s);
                }
                xfer_put(chunk_xfer(c.hand[slot], c, T, true), c.hand_flag[slot], wp.win, d.xcnt, s, d.id);
            }
        }
        if (di == head_dev && (size_t) k < chunk_done.size()) {
            CK(cudaEventRecord(chunk_done[(size_t) k], s));
            chunk_rec[(size_t) k].store(1, std::memory_order_release);
        }
        if (d.grp == devs[(size_t) head_dev].grp) {   // the head group: each member its share of the logits
            if (logits_all) {   // every row's logits (tests): the head over the chunk's rows, 16 at a time
                const size_t base = (size_t) (p0 - run.p0[0]) * g.vocab;
                for (int r0 = 0; r0 < T; r0 += kWin) {
                    const int nr = std::min(kWin, T - r0);
                    hc_pre_norm(c.h + (size_t) r0 * H * D, c.pre + (size_t) r0 * 8, d.norm_w, D, H, nr, g.eps, d.x, s);
                    if (!compute_logits(d, d.x, nr, err)) return false;
                    if (di != head_dev) continue;
                    CK(cudaMemcpyAsync(chunk_logits.data() + base + (size_t) r0 * g.vocab, d.logits,
                                       (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                    CK(cudaStreamSynchronize(s));
                }
            } else if (last_k && n_logits > 0) {   // (the last rows' logits, into the window path's buffers)
                const int t0 = T - n_logits;
                hc_pre_norm(c.h + (size_t) t0 * H * D, c.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, s);
                if (!compute_logits(d, d.x, n_logits, err)) return false;
            }
        }
    }
    return true;
}

int Model::Impl::pick_flow_min() {
    if (pick_min_ > 0) return pick_min_;
    pick_min_ = prompt_flow_min(std::getenv("DS4_PICK_MIN"));
    std::fprintf(stderr, "ds4: prompt path: the chunked pass from %d tokens, windows of 16 below\n", pick_min_);
    return pick_min_;
}

bool Model::Impl::is_pick_adaptive() {
    static const bool adaptive = [] {
        const char* e = std::getenv("DS4_PICK");
        return e != nullptr && std::strcmp(e, "adaptive") == 0;
    }();
    return adaptive;
}

bool Model::Impl::feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
    logits_all = false;
    for (int a = 0; a < n; a += kWin) {
        const int T = std::min(kWin, n - a);
        if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, err)) return false;
    }
    return true;
}

bool Model::Impl::lm_begin(const Run& run) {
    static const int asked = env_tri("DS4_LAYER_MAJOR");
    if (devs.size() != 1 || run.T.size() < 2 || asked == 0) return false;
    Chunk& c = *chunks[0];
    if (!c.flow) return false;
    size_t need = 0, n_exp = 0, blob_max = 0;
    double file_b = 0;
    for (int il = 0; il < g.nl; ++il) {
        const Layer& l = L[(size_t) il];
        const size_t ne = l.host_list.size();
        const size_t stride = (l.blob + 63) & ~(size_t) 63;
        need = std::max(need, ne * stride);
        n_exp = std::max(n_exp, (size_t) l.n_exp);
        blob_max = std::max(blob_max, stride);
        file_b += (double) ne * (double) l.blob;
    }
    int64_t rows = 0;
    for (int T : run.T) rows += T;
    const size_t hb = chunk_hand_bytes(), state_b = run.T.size() * hb;
    const double moved = 2.0 * (double) rows * ((double) hb / (double) chunk) * (double) g.nl;
    if (need == 0 || (asked < 0 && (double) (run.T.size() - 1) * file_b <= moved)) return false;
    size_t lc_bytes = need;
    if (const char* mb = std::getenv("DS4_LAYER_CACHE_MB")) lc_bytes = std::min(need, (size_t) std::atoll(mb) << 20);
    else {
        const uint64_t avail = ram_avail();
        if (avail != ~uint64_t{0})
            lc_bytes = std::min<uint64_t>(need, avail > kRamMargin + state_b ? avail - kRamMargin - state_b : 0);
    }
    if (lc_bytes < blob_max) return false;
    void* st_p = nullptr;
    c.lm_pinned = strata::core::allocate_host(&st_p, state_b, cudaHostAllocPortable,
        {"other", "Device allocations", "DeepSeek working buffers"}) == cudaSuccess;
    if (!c.lm_pinned) {
        cudaGetLastError();
        st_p = node_map(state_b, -1, 1, {"conversations", "Prompt processing", "Checkpoint state"});
        if (st_p == nullptr) return false;
    }
    c.lc = (uint8_t*) node_map(lc_bytes, -1, 1, {"other", "Prompt processing", "Layer cache"});
    if (c.lc == nullptr) {
        if (c.lm_pinned)
            strata::core::release_host(st_p);
        else free_node_memory(st_p, state_b);
        return false;
    }
    c.lm_state = (uint8_t*) st_p;
    c.lm_state_bytes = state_b;
    c.lc_bytes = lc_bytes;
    c.lc_of.assign(n_exp, -1);
    c.lc_writer.assign(n_exp, -1);
    c.lm = true;
    lm_layers.store(0);
    return true;
}

void Model::Impl::lm_end() {
    for (size_t di = 0; di < chunks.size(); ++di) {
        Chunk& c = *chunks[di];
        if (!c.lm) continue;
        cudaSetDevice(devs[di].id);
        cudaStreamSynchronize(devs[di].st);   // (nothing of the run's still reads or writes them: the device's
        cudaStreamSynchronize(c.cs);          // copies done, every read the run issued done - a failed run's too)
        for (size_t i = c.base; i < c.n_read; ++i)
            while (fit(c, i).left.load(std::memory_order_acquire) != 0) c.io->wait_any(500);
        if (c.lm_pinned)
            strata::core::release_host(c.lm_state);
        else free_node_memory(c.lm_state, c.lm_state_bytes);
        free_node_memory(c.lc, c.lc_bytes);
        c.lm_state = c.lc = nullptr;
        c.lm_state_bytes = c.lc_bytes = 0;
        c.lm = c.lm_pinned = false;
    }
}

int Model::Impl::yoco_mode() {
    static const int m = [] {
        const char* v = std::getenv("DS4_YOCO");
        if (v == nullptr) v = std::getenv("STRATA_DS4_PROMPT_SKIP");
        return v != nullptr ? std::atoi(v) : 1;
    }();
    return m;
}

int Model::Impl::yoco_keep(int yl) const {
    // (DS4_YOCO_KEEP=<rows>: fewer - DeepSeek's own serving replays the last 128 tokens through those layers
    //  (DeepInfer's "SWA bounded replay"), which shifts the answers a little; at least one window)
    static const int keep_env = std::getenv("DS4_YOCO_KEEP") ? std::atoi(std::getenv("DS4_YOCO_KEEP")) : 0;
    return yl <= 0 ? 0
           : keep_env > 0 ? std::max(keep_env, g.win)
           : yoco_mode() == 2 ? g.win : (g.nl - yl) * (g.win - 1) + g.win;
}

int Model::Impl::yoco_layer() const {
    if (yoco_mode() == 0 || chunk <= 0 || g.nl < 2 || (int) g.ratios.size() < g.nl) return -1;
    int yl = -1;
    for (int il = 0; il < g.nl && yl < 0; ++il)
        if (g.ratios[(size_t) il] == 1) yl = il;
    if (yl <= 0 || !L[(size_t) yl].kv_src) return -1;
    for (int il = 0; il < g.nl; ++il) {
        const Layer& l = L[(size_t) il];
        if (l.dev != L[0].dev) return -1;
        if (il > yl && (g.ratios[(size_t) il] != 1 || l.kv_src)) return -1;
    }
    return yl;
}

bool Model::Impl::forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot, int after) {
    if (n < 1) return true;
    flow_used.store(true);
    std::lock_guard<std::mutex> warm_lk(warm_mu);   // (the warm-up, if it is running, finishes first)
    if (pos + n > o.max_context) {
        err = "the context is full";
        return false;
    }
    for (int t = 0; t < n; ++t)
        if (tk[t] < 0 || tk[t] >= g.vocab) {
            err = "token " + std::to_string(tk[t]) + " out of range";
            return false;
        }
    // a long prompt's skip: the layers from yoco_layer() on (V4.1: 20-39) read the earlier positions only through
    // that layer's compressed KV and index keys, and each looks back one window (win - 1 positions) more than the
    // layer before - so they need their own work only on the last keep = (nl - yoco) (win - 1) + win rows for the
    // run's last rows and the rings the decode reads to be what the full work gives (up to the experts' summing
    // order).  The chunks stay as they are: the ones before the cut go through the layers before yoco and its
    // kv_only, the cut's chunk through those over all its rows and through the rest from the cut on (Run::r0, a
    // view of its buffers), the chunks after it through every layer - each layer's experts read as without the
    // skip.  One GPU (DS4_YOCO=1; 0: off); not with every row's logits (tests)
    const int yl = logits_rows > kWin ? -1 : yoco_layer();
    const int keep = yoco_keep(yl);
    // (a piece of a longer prompt, `after` more of its tokens in later feeds: the kept rows are the prompt's last, so
    //  a piece wholly before them is KV-only past yoco - as its rows are in the prompt read whole)
    const int skip = std::min(n - 1, n + after - keep);   // (at least the last row through every layer)
    const int a = yl > 0 && skip >= kYocoMin ? skip : 0;
    const int nc = (n + chunk - 1) / chunk, per = (n + nc - 1) / nc;
    if (std::getenv("DS4_CHUNK_LOG") != nullptr)
        std::fprintf(stderr, "forward_chunks n %d nc %d per %d yoco %d keep %d skip_rows %d (after %d)\n", n, nc,
                     per, yl, keep, a, after);
    Run run;
    for (int x = 0; x < n; x += per) {
        run.p0.push_back(pos + x);
        run.T.push_back(std::min(per, n - x));
        run.seq.push_back(++win_seq);
    }
    if (a > 0) {
        run.yoco = yl;
        run.cut_k = a / per;
        run.cut_r = a - run.cut_k * per;
    }
    logits_all = logits_rows > kWin;
    if (logits_all) chunk_logits.resize((size_t) n * g.vocab);
    for (int t = 0; t < n; ++t) {
        toks.push_back(tk[t]);
        ctoks.push_back(ctok(pos + t, tk[t]));
    }
    plan_streams();
    // the state the layers keep on the way: feed_mark's state before the run's last row
    run.snap_pos.clear();
    if (snap_slot >= 0) run.snap_pos.push_back(pos + n - 1);
    for (int il = 0; il < g.nl && !run.snap_pos.empty(); ++il) {
        Layer& l = L[(size_t) il];
        Dev& d = devs[(size_t) l.dev];
        for (size_t i = 0; i < run.snap_pos.size(); ++i) {
            if (l.snap_win[i] == nullptr && !dalloc(d, l.snap_win[i], (size_t) g.win * win_rb(), err)) return false;
            if (l.c_kv != nullptr && l.snap_kv[i] == nullptr &&
                (!dalloc(d, l.snap_kv[i], (size_t) l.ratio * g.hd, err) ||
                 !dalloc(d, l.snap_sc[i], (size_t) l.ratio * g.hd, err)))
                return false;
        }
    }
    const int T_last = run.T.back();
    n_logits = logits_all ? 0 : std::min(logits_rows, T_last);
    const double tl = now_ms();
    {   // each chunk's end on the head device, for the progress reports
        cudaSetDevice(devs[(size_t) head_dev].id);
        while (chunk_done.size() < run.T.size()) {
            cudaEvent_t e = nullptr;
            CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            chunk_done.push_back(e);
        }
        chunk_rec.reset(new std::atomic<int>[chunk_done.size()]);
        for (size_t i = 0; i < chunk_done.size(); ++i) chunk_rec[i].store(0);
    }
    prof_chunk = prof_on && !prof_sync_on;   // (DS4_PROF_SYNC: the decode profile only; the prompt events hang the driver)
    const bool lm = lm_begin(run);
    struct LmEnd {   // (every way out: its host memory freed once the device is done with it)
        Impl* p;
        bool on;
        ~LmEnd() {
            if (on) p->lm_end();
        }
    } lm_end_{this, lm};
    std::vector<std::string> errs(devs.size());
    halted = false;
    halt.store(false);
    yielding.store(false);
    {
        std::lock_guard<std::mutex> lk(halt_mu);
        halt_k = INT_MAX;
        halt_seen = -1;
    }
    bool halting = false, stopping = false;
    run_yields = yield_to != nullptr && !lm && !logits_all;
    for (size_t di = 0; di < devs.size(); ++di)
        workers[di]->submit([this, di, &run, &errs] {
            std::string e;
            if (!run_dev_chunks((int) di, run, e)) errs[di] = e.empty() ? "a device's thread failed" : e;
        });
    size_t reported = 0;
    for (;;) {
        bool all = true;
        for (auto& w : workers) all = all && w->idle();
        // a Stop, or a yield (a run layer by layer has no chunk to end at; every row's logits: tests): the devices end
        // the run early
        const bool stop = !all && !stopping && cancel && cancel();
        const bool yield = !all && !halting && !lm && !logits_all && yield_to && yield_to();
        if (stop || yield) {
            stopping = stopping || stop;
            yielding.store(!stopping, std::memory_order_release);
            halting = true;
            std::lock_guard<std::mutex> lk(halt_mu);
            if (halt_k == INT_MAX) halt_k = halt_seen + 1;
            halt.store(true, std::memory_order_release);
        }
        if (progress && lm) {   // (layer-major: the share of the layers every chunk went through)
            const int nd = lm_layers.load(std::memory_order_acquire);
            if (nd > 0 && nd < g.nl) progress((int64_t) n * nd / g.nl);
        } else if (progress) {   // chunks through every layer (the last layer's device past their event)
            while (reported < run.T.size() && chunk_rec[reported].load(std::memory_order_acquire) &&
                   cudaEventQuery(chunk_done[reported]) == cudaSuccess)
                ++reported;
            if (reported > 0 && reported < run.T.size())
                progress(run.p0[reported - 1] + run.T[reported - 1] - run.p0[0]);
        }
        if (all) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    for (const std::string& e : errs)   // (a device's failure first: a Stop's early end is not one)
        if (!e.empty() && e != kHalted) {
            err = e;
            return false;
        }
    const bool yielded = yielding.load(std::memory_order_acquire);
    for (const std::string& e : errs)
        if (e == kHalted && !yielded) {
            halted = true;
            err = "stopped while reading the prompt";
            return false;
        }
    // (a yield: the chunks before halt_k read, every device through them; none past it - all of them when every
    //  chunk had started)
    const int took = yielded && halt_k < (int) run.T.size() ? (int) (run.p0[(size_t) halt_k] - run.p0[0]) : n;
    fed_n = took;
    st.layer_ms += now_ms() - tl;
    eg_stats();
    Dev& d = devs[(size_t) head_dev];
    cudaSetDevice(d.id);
    const double th = now_ms();
    CK(cudaStreamSynchronize(d.st));
    st.head_ms += now_ms() - th;
    if (prof_chunk) {   // DS4_PROF: the run's GPU time by stage, every device's summed
        for (Dev& dv : devs) {
            cudaSetDevice(dv.id);
            CK(cudaStreamSynchronize(dv.st));
        }
        cudaSetDevice(d.id);
        prof_collect();
        std::fprintf(stderr,
                     "ds4 prompt GPU time, %d tokens: attention %.0f ms, routing + shared expert %.0f ms, experts %.0f ms "
                     "(with the waits for streamed ones), between layers %.0f ms\n",
                     n, cprof_ms[1], cprof_ms[2], cprof_ms[3], cprof_ms[0]);
        {   // the finer table (ms, % of the sum)
            static const char* nm_[16] = {"between layers", "attn rest (q_b, wo_a/b, hc_post)", "shared expert", "experts (incl waits)",
                                          "engram", "attn mHC (mix/split/norm)", "attn projections", "compressor + indexer",
                                          "q_b + rope", "sparse attention", "ffn mHC + router + sort", "", "", "", "", ""};
            double sum = 0;
            for (double v : cprof_ms) sum += v;
            for (int i = 0; i < 11; ++i)
                std::fprintf(stderr, "ds4 prof  %-34s %9.1f ms %5.1f%%\n", nm_[i], cprof_ms[i], sum > 0 ? 100.0 * cprof_ms[i] / sum : 0.0);
            struct Agg {
                double ms = 0, flops = 0;
                long calls = 0, rows = 0, experts = 0;
            };
            std::map<std::pair<int, int>, Agg> agg;   // (type, role 0 gate+up, 1 down)
            {
                std::lock_guard<std::mutex> lk(prof_mu);
                for (const EkMark& m : ek_marks) {
                    float a = 0, b = 0;
                    cudaSetDevice(m.dev);
                    cudaEventElapsedTime(&a, m.e0, m.e1);
                    cudaEventElapsedTime(&b, m.e1, m.e2);
                    Agg& x = agg[{m.tg, 0}];
                    x.ms += a;
                    x.flops += 2.0 * 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                    x.calls++, x.rows += m.nrows, x.experts += m.nexp;
                    Agg& y = agg[{m.td, 1}];
                    y.ms += b;
                    y.flops += 2.0 * m.nrows * (double) g.dim * (double) m.ff;
                    y.calls++, y.rows += m.nrows, y.experts += m.nexp;
                    ek_pool[m.dev].push_back(m.e0);
                    ek_pool[m.dev].push_back(m.e1);
                    ek_pool[m.dev].push_back(m.e2);
                }
                ek_marks.clear();
            }
            double tot = 0;
            for (auto& kv : agg) tot += kv.second.ms;
            for (auto& kv : agg)
                std::fprintf(stderr,
                             "ds4 prof  expert kernels type %-8s %-7s %9.1f ms  calls %5ld  experts %6ld (%.1f us each, %.1f a call)  rows/expert %6.1f  ~%5.1f TFLOPS\n",
                             gg::iq::name(kv.first.first), kv.first.second ? "down" : "gate+up", kv.second.ms, kv.second.calls,
                             kv.second.experts, kv.second.experts > 0 ? 1e3 * kv.second.ms / (double) kv.second.experts : 0.0,
                             kv.second.calls > 0 ? (double) kv.second.experts / (double) kv.second.calls : 0.0,
                             kv.second.experts > 0 ? (double) kv.second.rows / (double) kv.second.experts : 0.0,
                             kv.second.ms > 0 ? kv.second.flops / kv.second.ms / 1e9 : 0.0);
            std::fprintf(stderr, "ds4 prof  expert kernels total %.1f ms\n", tot);
        }
        for (double& v : cprof_ms) v = 0;
        prof_chunk = false;
    }
    if (std::getenv("DS4_FLOW_STATS") != nullptr) {
        const double ms = now_ms() - tl;
        for (size_t di = 0; di < chunks.size(); ++di) {
            const Chunk& c = *chunks[di];
            if (!c.flow) continue;
            const double db = (double) (c.io->direct_bytes() - c.io_d0) / 1e9;
            const double bb = (double) (c.io->buffered_bytes() - c.io_b0) / 1e9;
            const double fb = db + bb, rb = (double) c.it_ram * (double) L[0].blob / 1e9;   // (a member: its parts)
            char lcs[160] = "";
            if (c.lm)
                std::snprintf(lcs, sizeof lcs, ", %llu again from the layer cache (layer-major, %d chunks, %.1f GB "
                              "of RAM)", (unsigned long long) c.it_lc, (int) run.T.size(), (double) c.lc_bytes / 1e9);
            std::fprintf(stderr,
                         "ds4 prompt stream, CUDA%d: %d tokens in %.0f ms; %llu experts streamed: %llu from the files "
                         "(%.1f GB, %.2f GB/s: %.1f GB from the disk unbuffered, %.1f GB through the page cache), %llu "
                         "from RAM (%.1f GB)%s; %llu likely ones queued early, %llu of them unpicked; the compute waited "
                         "%.0f ms for reads\n",
                         devs[di].id, n, ms, (unsigned long long) (c.it_file + c.it_ram + c.it_lc),
                         (unsigned long long) c.it_file, fb, fb / std::max(1e-3, ms / 1000.0), db, bb,
                         (unsigned long long) c.it_ram, rb, lcs, (unsigned long long) c.it_spec,
                         (unsigned long long) c.it_spec_unused, c.wait_ms);
        }
    }
    if (logits_all) n_logits = (T_last - 1) % kWin + 1;   // (sample() takes the last row of d.logits)
    if (took < n) {   // (a yield: the rest of the tokens are a later feed's; no logits)
        toks.resize((size_t) (pos + took));
        ctoks.resize((size_t) (pos + took));
        n_logits = 0;
    }
    pos += took;
    st.tokens += took;
    if (lm_end_.on) {   // (the layer cache's RAM back)
        lm_end();
        lm_end_.on = false;
    }
    // (the drafter seeded from the run's last rows - when they went through every layer: not after a yield)
    const int nr = std::min({T_last, tap_cap, a > 0 ? n - a : n});   // (the rows that went through every layer)
    if (took == n && !seed_drafter(pos - nr, nr, 0, err)) return false;
    if (took < n) return true;   // (a yield: no checkpoint kept)
    // feed_mark's checkpoint `snap_slot`, from what the layers kept of it
    if (snap_slot >= 0 && !snap_check(ck[snap_slot & 1], run.snap_pos.back(), (int) run.snap_pos.size() - 1, err))
        return false;
    return true;
}

bool Model::Impl::snap_check(Check& ckp, int64_t at, int ix, std::string& err) {
    ckp.pos = at;
    ckp.win.resize(L.size());
    ckp.comp.resize(L.size() * 2);
    for (size_t i = 0; i < L.size(); ++i) {
        Layer& l = L[i];
        Dev& dv = devs[(size_t) l.dev];
        cudaSetDevice(dv.id);
        const bool kept = !l.mtp && l.snap_win[ix] != nullptr;
        ckp.win[i].resize(get_checkpoint_ring_bytes(i));
        CK(cudaMemcpyAsync(ckp.win[i].data(), kept ? l.snap_win[ix] : l.win, ckp.win[i].size(),
                           cudaMemcpyDeviceToHost, dv.st));
        if (l.c_kv) {
            ckp.comp[2 * i].resize((size_t) l.ratio * g.hd);
            ckp.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
            CK(cudaMemcpyAsync(ckp.comp[2 * i].data(), kept && l.snap_kv[ix] ? l.snap_kv[ix] : l.c_kv,
                               ckp.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, dv.st));
            CK(cudaMemcpyAsync(ckp.comp[2 * i + 1].data(), kept && l.snap_sc[ix] ? l.snap_sc[ix] : l.c_score,
                               ckp.comp[2 * i + 1].size() * 4, cudaMemcpyDeviceToHost, dv.st));
        }
    }
    for (Dev& dv : devs) {
        cudaSetDevice(dv.id);
        CK(cudaStreamSynchronize(dv.st));
    }
    return true;
}

bool Model::feed(const int* tokens, int n, int logits_rows, std::string& err, int after) {
    Impl& p = *p_;
    p.halted = false;
    p.fed_n = n;
    if (ds4_dbg("DS4_DBG_CHUNK")) std::fprintf(stderr, "feed n %d chunk %d chunk_min %d\n", n, p.chunk, p.chunk_min);
    return p.can_chunk(n, logits_rows) ? p.forward_chunks(tokens, n, logits_rows, err, -1, after)
                                       : p.feed_windows(tokens, n, logits_rows, err);
}

bool Model::feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err) {
    Impl& p = *p_;
    p.halted = false;
    p.fed_n = n;
    if (n >= 2 && p.can_chunk(n))   // (the last token rides along; its layers keep the state before it)
        return p.forward_chunks(tokens, n, logits_rows, err, slot);
    if (n > 1 && !p.feed_windows(tokens, n - 1, 0, err)) return false;
    p.checkpoint(slot);
    return n < 1 || p.feed_windows(tokens + n - 1, 1, logits_rows, err);
}

}  // namespace strata::ds4
