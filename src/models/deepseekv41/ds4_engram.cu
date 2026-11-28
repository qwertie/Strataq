// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_engram.cu - Engram: a window's rows of the n-gram tables, read from the files in the background (eg_issue) and
// collected right before their layer (eg_finish).
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

void Model::Impl::eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
    const int cols = (pk.eg_ngram - 1) * pk.eg_heads, hd = pk.eg_hd;
    const size_t per = (size_t) cols * (hd + hd / 32);
    b.clear();
    b.reads.reserve((size_t) T * cols);
    const std::string P = "layers." + std::to_string(g.eg_layers[(size_t) li]) + ".engram.embed.";
    const Tensor* tw = pk.get(P + "weight");
    const Tensor* ts = pk.get(P + "scale");
    const int fmt = eg_fmt[(size_t) li];
    const bool mx8 = fmt == 0;   // (a GGUF: the scales between the codes, one per 32)
    const bool q6k = fmt == 1;   // (a GGUF: one Q6_K superblock a 256 values, its scales inside)
    const bool iq4 = fmt == 2;   // (a GGUF: 18-byte blocks a 32 values, d and 16 nibble bytes, no scales)
    std::unordered_map<uint64_t, int> first;   // a row this batch reads already -> its read
    int64_t hits = 0;
    {
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            const std::vector<int32_t>& ct = ctoks;
            uint8_t* rd0 = dst != nullptr ? dst + (size_t) t * per : h_eg + ((size_t) li * eg_cap + t) * per;
            uint8_t* scales = rd0 + (size_t) cols * hd;
            if (ct[(size_t) p] < 0) {   // a picture's token: no rows (zeros: its gate adds nothing)
                std::memset(rd0, 0, per);
                continue;
            }
            int64_t prod[8];
            bool blocked = false;
            for (int s = 0; s < pk.eg_ngram; ++s) {   // (the look-back stops at the start and at a picture)
                blocked = blocked || p < s || ct[(size_t) (p - s)] < 0;
                const int64_t tk = blocked ? pk.eg_pad : ct[(size_t) (p - s)];
                prod[s] = tk * pk.eg_mult[(size_t) li * pk.eg_ngram + s];
            }
            int64_t rolling = prod[0];
            for (int n = 1; n < pk.eg_ngram; ++n) {
                rolling ^= prod[n];
                for (int h = 0; h < pk.eg_heads; ++h) {
                    const int c = (n - 1) * pk.eg_heads + h;
                    const int64_t id = rolling % pk.eg_primes[(size_t) li * cols + c] + pk.eg_off[(size_t) li * cols + c];
                    uint8_t* rd = rd0 + (size_t) c * hd;
                    uint8_t* sd = scales + (size_t) c * (hd / 32);
                    const uint64_t key = (uint64_t) id * 8 + (uint64_t) li;
                    if (const uint8_t* hit = eg_cache.find(key)) {
                        std::memcpy(rd, hit, (size_t) hd);
                        std::memcpy(sd, hit + hd, (size_t) hd / 32);
                        ++hits;
                        continue;
                    }
                    const auto fi = first.find(key);
                    if (fi != first.end()) {
                        b.dups.push_back({fi->second, {rd, sd}});
                        continue;
                    }
                    first.emplace(key, (int) b.reads.size());
                    EgRead r;
                    r.file = tw->file;
                    r.dst = rd;
                    r.scales = sd;
                    r.key = key;
                    if (mx8) {
                        r.mx8 = true;
                        r.n = (uint32_t) (hd / 32 * 33);
                        r.off = tw->off + (uint64_t) id * r.n;
                    } else if (q6k || iq4) {   // (into the row's code bytes; its scale bytes unused)
                        r.n = (uint32_t) (q6k ? hd / 256 * 210 : hd / 32 * 18);
                        r.off = tw->off + (uint64_t) id * r.n;
                    } else {
                        r.n = (uint32_t) hd;
                        r.off = tw->off + (uint64_t) id * hd;
                        r.file2 = ts->file;
                        r.n2 = (uint32_t) (hd / 32);
                        r.off2 = ts->off + (uint64_t) id * (hd / 32);
                    }
                    b.reads.push_back(r);
                }
            }
        }
    }
    eg_rows_n += hits + (int64_t) b.reads.size() + (int64_t) b.dups.size();
    eg_hits_n += hits + (int64_t) b.dups.size();
    eg_reads_n += (int64_t) b.reads.size();
    eg_io.submit(b);
}

void Model::Impl::eg_finish(EgBatch& b) {
    if (b.post.load(std::memory_order_acquire) == 2) return;
    const int64_t t0 = now_ns();
    eg_io.wait(b);
    int want = 0;
    if (b.post.compare_exchange_strong(want, 1, std::memory_order_acq_rel)) {
        const int hd = pk.eg_hd;
        for (const auto& d : b.dups) {
            const EgRead& r = b.reads[(size_t) d.first];
            std::memcpy(d.second.first, r.dst, (size_t) hd);
            std::memcpy(d.second.second, r.scales, (size_t) hd / 32);
        }
        if (ds4_dbg("DS4_DBG_EG") && pk.eg_hd >= 32) {   // DBG: a row whose IQ4_NL block scale is not finite
            for (size_t i = 0; i < b.reads.size(); ++i) {
                const EgRead& r = b.reads[i];
                for (int k = 0; k < hd / 32; ++k) {
                    uint16_t hb;
                    std::memcpy(&hb, r.dst + (size_t) k * 18, 2);
                    if ((hb & 0x7c00) == 0x7c00)
                        std::fprintf(stderr, "DBG eg bad scale: read %zu/%zu file=%d off=%llu n=%u block=%d half=%04x ok=%d\n", i, b.reads.size(),
                                     r.file, (unsigned long long) r.off, r.n, k, hb, (int) b.ok[i]);
                }
            }
        }
        {
            std::lock_guard<std::mutex> lk(eg_cache_mu);
            for (size_t i = 0; i < b.reads.size(); ++i) {
                const EgRead& r = b.reads[i];
                if (!b.ok[i]) {   // (a failed read: zeros, its gate adds nothing; not kept)
                    std::memset(r.dst, 0, (size_t) hd);
                    std::memset(r.scales, 0, (size_t) hd / 32);
                    continue;
                }
                if (uint8_t* c = eg_cache.put(r.key)) {
                    std::memcpy(c, r.dst, (size_t) hd);
                    std::memcpy(c + hd, r.scales, (size_t) hd / 32);
                }
            }
        }
        eg_io_ns += b.t1.load() - b.t0;
        b.live = false;
        b.post.store(2, std::memory_order_release);
    } else {
        while (b.post.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    }
    eg_wait_ns += now_ns() - t0;
}

void Model::Impl::eg_stats() {
    st.eg_rows = eg_rows_n.load();
    st.eg_hits = eg_hits_n.load();
    st.eg_reads = eg_reads_n.load();
    st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
    std::lock_guard<std::mutex> lk(eg_cache_mu);
    st.eg_cached = (int64_t) eg_cache.size();
    st.eg_cache_cap = (int64_t) eg_cache.capacity();
}

}  // namespace strata::ds4
