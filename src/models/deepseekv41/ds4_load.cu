// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_load.cu - loading the model: the pack's files, the dense weights on each GPU, where each layer's experts lie in
// the files and how their blobs are laid out, the experts' places for the paged engine (place_paged,
// set_expert_addresses), a layer's move to another card (Model::set_stage_layers) and the CPU lane's worker pools.
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

namespace {

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::vector<int> cpu_list(const std::string& s) {   // "0-3,8,10-11"
    std::vector<int> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty() || part == "\n") continue;
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
// each NUMA node's physical cores, each with its logical CPUs (the first one first) and efficiency class, in order -
// only the CPUs this process may run on (its affinity: taskset, a container's cpuset), so no worker is pinned to a
// CPU it cannot use
std::vector<std::vector<platform::Core>> node_threads() {
#if defined(_WIN32)
    std::vector<std::vector<platform::Core>> nodes = platform::node_core_threads();
#else
    std::vector<std::vector<platform::Core>> nodes;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_mask = sched_getaffinity(0, sizeof allowed, &allowed) == 0;
    auto ok = [&](int c) { return c >= 0 && (!have_mask || (c < CPU_SETSIZE && CPU_ISSET(c, &allowed))); };
#else   // (no CPU affinity mask to ask)
    auto ok = [&](int c) { return c >= 0; };
#endif
    // a hybrid Intel CPU's performance cores (the kernel's cpu_core PMU lists them; its E-cores are cpu_atom's)
    const std::vector<int> pcores = cpu_list(read_file("/sys/devices/cpu_core/cpus"));
    const bool hybrid = !pcores.empty() && !read_file("/sys/devices/cpu_atom/cpus").empty();
    for (int n = 0; n < 64; ++n) {
        const std::string cl = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cl.empty()) break;
        std::vector<platform::Core> cores;
        std::vector<int> seen;   // (each core once: by its first sibling)
        for (int c : cpu_list(cl)) {
            if (!ok(c)) continue;
            std::vector<int> sib = cpu_list(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty()) sib = {c};
            if (std::find(seen.begin(), seen.end(), sib[0]) != seen.end()) continue;
            seen.push_back(sib[0]);
            platform::Core core;
            for (int s : sib)
                if (ok(s)) core.cpus.push_back(s);
            if (core.cpus.empty()) continue;
            core.eff = hybrid && std::find(pcores.begin(), pcores.end(), core.cpus[0]) != pcores.end() ? 1 : 0;
            cores.push_back(core);
        }
        if (!cores.empty()) nodes.push_back(cores);
    }
#endif
    if (nodes.empty()) {
        std::vector<platform::Core> all;
        for (unsigned c = 0; c < std::max(1u, std::thread::hardware_concurrency() / 2); ++c) all.push_back({{(int) c}, 0});
        nodes.push_back(all);
    }
    return nodes;
}

void pin_self(int cpu) { platform::pin_self(cpu); }

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            u = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// DeepSeek's precompute_freqs_cis frequencies (YaRN when original_seq_len > 0), in F32 as torch computes them
std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast, double beta_slow) {
    std::vector<float> f((size_t) dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = (float) (2 * i) / (float) dim;
        f[(size_t) i] = 1.0f / std::pow((float) base, e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));   // (M_PI: not MSVC's)
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), (double) dim - 1);
        const float den = (float) std::max(high - low, 1e-3);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[(size_t) i] = f[(size_t) i] / (float) factor * (1.0f - smooth) + f[(size_t) i] * smooth;
        }
    }
    return f;
}

}  // namespace

// ---- weights
bool Model::Impl::upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
    std::vector<uint8_t> stage((size_t) std::min(t.bytes, artifact::kStageRound));
    cudaSetDevice(d.id);
    return artifact::for_each_round(t.bytes, [&](uint64_t done, uint64_t n) {
        if (!pk.read_at(t.file, t.off + done, n, stage.data())) {
            err = "read failed";
            return false;
        }
        CK(cudaMemcpy((uint8_t*) dst + done, stage.data(), n, cudaMemcpyHostToDevice));
        return true;
    });
}

bool Model::Impl::mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err,
                         size_t* acct, int part, int np, bool cols) {
    if (t.shape.size() != 2 || t.shape[1] % 256 != 0) {
        err = pre + ": a " + t.dtype + " tensor that is not a matrix of whole blocks";
        return false;
    }
    int n = (int) t.shape[0], k = (int) t.shape[1];
    const size_t rb = gg::type_row_bytes(ggt, k), bb = rb / ((size_t) k / 256);
    std::vector<uint8_t> w((size_t) n * rb);
    if (t.bytes != w.size() || !pk.read(t, w.data())) {
        err = pre + ": read failed";
        return false;
    }
    if (np > 1) {
        const int whole = cols ? n : k, wd = whole / np;
        if (whole % np != 0 || wd % (cols ? 1 : 256) != 0) {
            err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " + std::to_string(np) + " parts";
            return false;
        }
        std::vector<uint8_t> p;
        if (cols) {
            p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
            n = wd;
        } else {
            const size_t pb = (size_t) (wd / 256) * bb, off = (size_t) (part * wd / 256) * bb;
            p.resize((size_t) n * pb);
            for (size_t r = 0; r < (size_t) n; ++r) std::memcpy(p.data() + r * pb, w.data() + r * rb + off, pb);
            k = wd;
        }
        w.swap(p);
    }
    uint8_t* dp = nullptr;
    if (!dalloc(d, dp, w.size(), err, acct)) return false;
    cudaSetDevice(d.id);
    CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
    m = x3::Mat{};
    m.fmt = gg::kFmtGG;
    m.K = ggt;
    m.raw = dp;
    m.k = k;
    m.n = n;
    m.ld = (int) gg::type_row_bytes(ggt, k);
    return true;
}

bool Model::Impl::mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct,
                           int part, int np, bool cols) {
    const bool q8 = t.dtype == "Q8_0";
    // a ggml K-quant dense matrix (Q4_K / Q5_K / Q6_K: 256 values a block; the GEMV path of ds4_iq)
    const int ggt = (t.dtype == "Q4_K" || t.dtype == "Q5_K" || t.dtype == "Q6_K") ? gg_type_of(t.dtype) : -1;
    if (ggt > 0) return mat_gg(d, pre, t, ggt, m, err, acct, part, np, cols);
    if (t.shape.size() != 2 || (q8 && t.shape[1] % 32 != 0)) {
        err = pre + ": a " + t.dtype + " tensor that is not a matrix";
        return false;
    }
    int n = (int) t.shape[0], k = (int) t.shape[1];
    // a row's bytes, and the bytes of v of its values (BF16: 2 each; Q8_0: 34 every 32)
    auto bytes_of = [&](size_t v) { return q8 ? v / 32 * gg::kQ8Bytes : v * 2; };
    std::vector<uint8_t> w((size_t) n * bytes_of((size_t) k));
    if (t.bytes != w.size() || !pk.read(t, w.data())) {
        err = pre + ": read failed";
        return false;
    }
    if (np > 1) {
        const int whole = cols ? n : k, wd = whole / np;
        if (whole % np != 0 || wd % (q8 ? 32 : 8) != 0) {
            err = pre + ": " + std::to_string(whole) + (cols ? " outputs" : " inputs") + " do not split in " +
                  std::to_string(np) + " parts";
            return false;
        }
        const size_t rb = bytes_of((size_t) k);
        std::vector<uint8_t> p;
        if (cols) {
            p.assign(w.begin() + (long) ((size_t) part * wd * rb), w.begin() + (long) ((size_t) (part + 1) * wd * rb));
            n = wd;
        } else {
            const size_t pb = bytes_of((size_t) wd);
            p.resize((size_t) n * pb);
            for (size_t r = 0; r < (size_t) n; ++r)
                std::memcpy(p.data() + r * pb, w.data() + r * rb + bytes_of((size_t) part * wd), pb);
            k = wd;
        }
        w.swap(p);
    }
    uint8_t* dp = nullptr;
    if (!dalloc(d, dp, w.size(), err, acct)) return false;
    cudaSetDevice(d.id);
    CK(cudaMemcpy(dp, w.data(), w.size(), cudaMemcpyHostToDevice));
    m = x3::Mat{};
    m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
    m.raw = dp;
    m.k = k;
    m.n = n;
    m.ld = q8 ? (int) bytes_of((size_t) k) : k;
    return true;
}

std::vector<float> Model::Impl::q8_values(const uint8_t* q, size_t n) {
    std::vector<float> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* p = q + b * gg::kQ8Bytes;
        const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
        for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
    }
    return v;
}

bool Model::Impl::mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part, int np,
                      bool cols, int rot) {
    if (const Tensor* tw = pk.get(pre + ".weight"); tw != nullptr && (tw->dtype == "BF16" || tw->dtype == "Q8_0" || tw->dtype == "Q4_K" || tw->dtype == "Q5_K" || tw->dtype == "Q6_K"))
        return mat_bf16(d, pre, *tw, m, err, acct, part, np, cols);
    const Tensor* tr = pk.get(pre + ".trellis");
    const Tensor* su = pk.get(pre + ".suh");
    const Tensor* sv = pk.get(pre + ".svh");
    if (tr == nullptr || su == nullptr || sv == nullptr || tr->shape.size() != 3) {
        err = pre + ": not an EXL3 matrix in the pack";
        return false;
    }
    if (pk.get(pre + ".mul1") == nullptr) {
        err = pre + ": only the mul1 codebook is covered";
        return false;
    }
    m.k = (int) tr->shape[0] * 16;
    m.n = (int) tr->shape[1] * 16;
    m.K = (int) tr->shape[2] / 16;
    m.cb = 2;
    m.ld = 0;
    if (np > 1) {
        const int whole = cols ? m.n : m.k;
        int at = 0, w = 0;
        if (whole % 128 != 0 || whole / 128 < np) {
            err = pre + ": " + std::to_string(whole) + (cols ? " columns" : " rows") + " do not split in " +
                  std::to_string(np) + " parts of 128-blocks";
            return false;
        }
        split128(whole, np, part, at, w, rot);
        std::vector<uint8_t> t0(tr->bytes), u0(su->bytes), v0(sv->bytes);
        if (!pk.read(*tr, t0.data()) || !pk.read(*su, u0.data()) || !pk.read(*sv, v0.data())) {
            err = pre + ": read failed";
            return false;
        }
        const size_t tile = tr->bytes / ((size_t) (m.k / 16) * (m.n / 16));
        std::vector<uint8_t> t1, u1, v1;
        if (cols) {   // the tiles [c0 / 16, (c0 + w) / 16) of each 16-row slice, svh [c0, c0 + w)
            const size_t c0 = (size_t) at, nt = (size_t) m.n / 16, nth = (size_t) w / 16;
            t1.resize((size_t) (m.k / 16) * nth * tile);
            for (size_t sl = 0; sl < (size_t) m.k / 16; ++sl)
                std::memcpy(t1.data() + sl * nth * tile, t0.data() + (sl * nt + c0 / 16) * tile, nth * tile);
            u1 = u0;
            v1.assign(v0.begin() + (long) (c0 * 2), v0.begin() + (long) ((c0 + w) * 2));
            m.n = w;
        } else {      // the 16-row slices [r0 / 16, (r0 + w) / 16), suh [r0, r0 + w)
            const size_t r0 = (size_t) at, row = (size_t) (m.n / 16) * tile;
            t1.assign(t0.begin() + (long) (r0 / 16 * row), t0.begin() + (long) ((r0 + w) / 16 * row));
            u1.assign(u0.begin() + (long) (r0 * 2), u0.begin() + (long) ((r0 + w) * 2));
            v1 = v0;
            m.k = w;
        }
        uint32_t* dt = nullptr;
        uint16_t *ds = nullptr, *dv = nullptr;
        if (!dalloc(d, dt, t1.size() / 4, err, acct) || !dalloc(d, ds, u1.size() / 2, err, acct) ||
            !dalloc(d, dv, v1.size() / 2, err, acct))
            return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(dt, t1.data(), t1.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ds, u1.data(), u1.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v1.data(), v1.size(), cudaMemcpyHostToDevice));
        m.trellis = dt;
        m.suh = ds;
        m.svh = dv;
        return x3::supported(m) || (err = pre + ": a part not covered by the EXL3 kernels", false);
    }
    uint32_t* dt = nullptr;
    uint16_t *ds = nullptr, *dv = nullptr;
    if (!dalloc(d, dt, tr->bytes / 4, err, acct) || !dalloc(d, ds, su->bytes / 2, err, acct) ||
        !dalloc(d, dv, sv->bytes / 2, err, acct))
        return false;
    if (!upload(d, *tr, dt, err) || !upload(d, *su, ds, err) || !upload(d, *sv, dv, err)) return false;
    m.trellis = dt;
    m.suh = ds;
    m.svh = dv;
    if (!x3::supported(m)) {
        err = pre + ": k " + std::to_string(m.k) + " n " + std::to_string(m.n) + " K " + std::to_string(m.K) + " not covered by the EXL3 kernels";
        return false;
    }
    return true;
}

bool Model::Impl::f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want) {
    const Tensor* t = pk.get(name);
    if (t == nullptr) {
        err = name + ": not in the pack";
        return false;
    }
    std::vector<uint8_t> raw(t->bytes);
    if (!pk.read(*t, raw.data())) {
        err = name + ": read failed";
        return false;
    }
    size_t n = 0;
    std::vector<float> v;
    if (t->dtype == "F32") {
        n = t->bytes / 4;
        v.resize(n);
        std::memcpy(v.data(), raw.data(), t->bytes);
    } else if (t->dtype == "F16" || t->dtype == "BF16") {
        n = t->bytes / 2;
        v.resize(n);
        const uint16_t* s = (const uint16_t*) raw.data();
        for (size_t i = 0; i < n; ++i) v[i] = t->dtype == "F16" ? f16_to_f32(s[i]) : bf16_to_f32(s[i]);
    } else if (t->dtype == "Q8_0") {   // (a GGUF's router may come so)
        n = t->bytes / gg::kQ8Bytes * 32;
        v = q8_values(raw.data(), n);
    } else {
        err = name + ": dtype " + t->dtype + " is not a float";
        return false;
    }
    if (want != 0 && n != want) {
        err = name + ": " + std::to_string(n) + " values, expected " + std::to_string(want);
        return false;
    }
    if (!dalloc(d, p, n, err, acct)) return false;
    cudaSetDevice(d.id);
    CK(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
    return true;
}

bool Model::Impl::f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
    const Tensor* t = pk.get(name);
    if (t != nullptr && (t->dtype == "F32" || t->dtype == "BF16" || t->dtype == "Q8_0")) {   // (a GGUF's)
        const size_t n = t->dtype == "Q8_0" ? t->bytes / gg::kQ8Bytes * 32 : t->bytes / (t->dtype == "F32" ? 4 : 2);
        std::vector<uint8_t> raw(t->bytes);
        if (!pk.read(*t, raw.data())) {
            err = name + ": read failed";
            return false;
        }
        const std::vector<float> q = t->dtype == "Q8_0" ? q8_values(raw.data(), n) : std::vector<float>();
        std::vector<uint16_t> h(n);
        for (size_t i = 0; i < n; ++i) {
            float v;
            if (t->dtype == "F32") std::memcpy(&v, raw.data() + i * 4, 4);
            else if (t->dtype == "Q8_0") v = q[i];
            else v = bf16_to_f32(((const uint16_t*) raw.data())[i]);
            h[i] = strata::kernels::f16_from_f32(v);
        }
        if (!dalloc(d, p, n, err, acct)) return false;
        cudaSetDevice(d.id);
        CK(cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice));
        return true;
    }
    if (t == nullptr || t->dtype != "F16") {
        err = name + ": not an F16 tensor in the pack";
        return false;
    }
    if (!dalloc(d, p, t->bytes / 2, err, acct)) return false;
    return upload(d, *t, p, err);
}

bool Model::Impl::mk_head_q8(Dev& d, const std::string& name, uint8_t*& p, std::string& err, size_t* acct) {
    const Tensor* t = pk.get(name);
    if (t == nullptr || (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16")) {
        err = name + ": not an F32 / F16 / BF16 tensor";
        return false;
    }
    const size_t n = t->bytes / (t->dtype == "F32" ? 4 : 2);
    if (n == 0 || n % 32 != 0) {
        err = name + ": not a whole number of 32-value blocks";
        return false;
    }
    std::vector<uint8_t> raw(t->bytes);
    if (!pk.read(*t, raw.data())) {
        err = name + ": read failed";
        return false;
    }
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        if (t->dtype == "F32") std::memcpy(&v[i], raw.data() + i * 4, 4);
        else v[i] = t->dtype == "F16" ? f16_to_f32(((const uint16_t*) raw.data())[i])
                                      : bf16_to_f32(((const uint16_t*) raw.data())[i]);
    }
    const size_t nb = n / 32;
    std::vector<uint8_t> q(nb * 34);
    for (size_t b = 0; b < nb; ++b) {
        const float* s = v.data() + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(s[i]));
        const float dd = amax > 0.0f ? amax / 127.0f : 0.0f;
        const uint16_t dh = strata::kernels::f16_from_f32(dd);
        std::memcpy(q.data() + b * 34, &dh, 2);
        int8_t* o = (int8_t*) (q.data() + b * 34 + 2);
        for (int i = 0; i < 32; ++i) {
            const float x = dd > 0.0f ? s[i] / dd : 0.0f;
            o[i] = (int8_t) std::max(-127.0f, std::min(127.0f, std::round(x)));
        }
    }
    if (!dalloc(d, p, nb * 34, err, acct)) return false;
    cudaSetDevice(d.id);
    CK(cudaMemcpy(p, q.data(), nb * 34, cudaMemcpyHostToDevice));
    return true;
}

size_t Model::Impl::mat_bytes(const std::string& pre) const {
    if (const Tensor* t = pk.get(pre + ".weight")) return align256(t->bytes);
    size_t b = 0;
    for (const char* s : {".trellis", ".suh", ".svh"})
        if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
    return b;
}

// ---- expert regions and blob layout
bool Model::Impl::expert_layout_mx(Layer& l, std::string& err) {
    const std::string P = l.name + "ffn.experts.";
    const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
        const Tensor* t0 = pk.get(P + "0." + parts[r] + ".weight");
        const Tensor* t1 = l.n_exp > 1 ? pk.get(P + "1." + parts[r] + ".weight") : t0;
        const int ty = t0 != nullptr ? gg_type_of(t0->dtype) : -1;
        const size_t bytes = ty > 0 ? (size_t) n * gg::type_row_bytes(ty, k) : 0;
        if (t0 == nullptr || t1 == nullptr || bytes == 0 || t0->bytes != bytes) {
            err = P + "0." + parts[r] + ": not an MXFP4 or low-bit [" + std::to_string(n) + "][" + std::to_string(k) +
                  "] matrix the engine reads (" + (t0 != nullptr ? t0->dtype : std::string("missing")) + ")";
            return false;
        }
        l.mx_file[r] = t0->file;
        l.mx_off[r] = t0->off;
        l.mx_stride[r] = l.n_exp > 1 ? t1->off - t0->off : bytes;
        for (int e = 0; e < l.n_exp; ++e) {
            const Tensor* t = pk.get(P + std::to_string(e) + "." + parts[r] + ".weight");
            if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes || t->file != l.mx_file[r] ||
                t->off != l.mx_off[r] + (uint64_t) e * l.mx_stride[r]) {
                err = P + std::to_string(e) + "." + parts[r] + ": missing, or not where the layer's others are";
                return false;
            }
        }
        l.lay.trellis[r] = off;
        l.lay.suh[r] = l.lay.svh[r] = off;
        l.lay.K[r] = ty;
        off += align256(bytes);
    }
    l.lay.fmt = gg::kFmtMXFP4;
    if (!gg::mx_experts(l.lay) && (gg::expert_type(l.lay, 0) != gg::expert_type(l.lay, 1) ||
                                   gg::expert_type(l.lay, 0) == gg::kTypeMXFP4 ||
                                   gg::expert_type(l.lay, 2) == gg::kTypeMXFP4)) {
        err = P + ": gate / up / down of types " + gg::type_name(l.lay.K[0]) + " / " +
              gg::type_name(l.lay.K[1]) + " / " + gg::type_name(l.lay.K[2]) +
              " (gate and up must match; MXFP4 does not mix with the others)";
        return false;
    }
    l.blob = off;
    // a type the device kernels do not take (none of the i-quants now: gg::type_on_device lists them): the layer stays on the CPU lane, where
    // ggml-cpu's row dots read it - one warning, then the placement code keeps it out of VRAM
    for (int r = 0; r < 3; ++r)
        if (!gg::type_on_device(l.lay.K[r])) {
            l.cpu_only = true;
            std::fprintf(stderr,
                         "ds4: layer %d %s is %s: no GPU kernel for it, its experts stay on the CPU lane "
                         "(ggml-cpu's rows; VRAM tier: none)\n", l.il, r == 2 ? "down" : "gate/up",
                         gg::type_name(l.lay.K[r]));
            break;
        }
    // a layer the CPU lane cannot compute used to answer zeros for every pick it got there (Project Maya #620).  One
    // that has no GPU kernel can only be computed there: refuse it.  Any other may never reach the lane (all its
    // experts in VRAM): a warning, and the lane stops the engine rather than answer zeros (cpu_experts_*)
    std::string why;
    if (!lane_fmt_ok(l, why)) {
        const std::string msg = P + ": the CPU lane cannot compute experts of types " + gg::type_name(l.lay.K[0]) + " / " +
                                gg::type_name(l.lay.K[2]) + ": " + why;
        if (l.cpu_only) {
            err = msg;
            return false;
        }
        static bool said_lane = false;
        if (!said_lane) {
            said_lane = true;
            std::fprintf(stderr, "ds4: %s (fine while its experts stay in VRAM; the engine stops if one is routed to the CPU lane)\n",
                         msg.c_str());
        }
    }
    return true;
}

bool Model::Impl::lane_fmt_ok(const Layer& l, std::string& why) const {
    const int E = g.dim, FF = g.ff;
    if (gg::mx_experts(l.lay)) return true;   // (cpu_experts_mx's own kernels)
    gg::IqCpu fm;
    if (!gg::iq_cpu_fmt(gg::expert_type(l.lay, 0), gg::expert_type(l.lay, 2), E, FF, fm, why)) return false;
    const size_t row = (size_t) E * 4;
    if (fm.act_bytes > row || fm.h_bytes > row) {
        why = "activation rows too large";
        return false;
    }
    return true;
}

bool Model::Impl::read_mx_file(const Layer& l, int e, uint8_t* blob) const {
    for (int r = 0; r < 3; ++r)
        if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                        blob + l.lay.trellis[r]))
            return false;
    return true;
}

bool Model::Impl::expert_layout(Layer& l, std::string& err) {
    if (const Tensor* t = pk.get(l.name + "ffn.experts.0.w1.weight"); t != nullptr && gg_type_of(t->dtype) > 0)
        return expert_layout_mx(l, err);
    const std::string P = l.name + "ffn.experts.";
    const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
    const char* pieces[3] = {".trellis", ".suh", ".svh"};
    l.region_off.resize((size_t) l.n_exp);
    l.region_bytes.resize((size_t) l.n_exp);
    l.region_file.resize((size_t) l.n_exp);
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        const Tensor* tr = pk.get(P + "0." + parts[r] + ".trellis");
        if (tr == nullptr || pk.get(P + "0." + parts[r] + ".mul1") == nullptr) {
            err = P + "0." + parts[r] + ": not an EXL3 (mul1) matrix";
            return false;
        }
        if (tr->shape[2] % 16 != 0 || tr->shape[2] / 16 < 1 || tr->shape[2] / 16 > 8) {
            err = P + "0." + parts[r] + ": " + std::to_string(tr->shape[2]) +
                  " int16 a tile - the kernels take K 1..8 and 1.5 / 2.5 / 3.5 (mul1)";
            return false;
        }
        l.lay.K[r] = (int) tr->shape[2] / 16;
        l.lay.trellis[r] = off;
        off += align256(tr->bytes);
        l.lay.suh[r] = off;
        off += align256(pk.get(P + "0." + parts[r] + ".suh")->bytes);
        l.lay.svh[r] = off;
        off += align256(pk.get(P + "0." + parts[r] + ".svh")->bytes);
    }
    l.lay.cb = 2;
    l.blob = off;
    for (int e = 0; e < l.n_exp; ++e) {
        const std::string E = P + std::to_string(e) + ".";
        uint64_t lo = UINT64_MAX, hi = 0;
        int file = -1;
        for (int r = 0; r < 3; ++r)
            for (int q = 0; q < 3; ++q) {
                const Tensor* t = pk.get(E + parts[r] + pieces[q]);
                if (t == nullptr) {
                    err = E + parts[r] + pieces[q] + ": missing";
                    return false;
                }
                if (file < 0) file = t->file;
                if (t->file != file) {
                    err = E + ": split across files";
                    return false;
                }
                lo = std::min(lo, t->off);
                hi = std::max(hi, t->off + t->bytes);
                if (e == 0) l.piece_bytes[r][q] = t->bytes;
                else if (t->bytes != l.piece_bytes[r][q]) {
                    err = E + parts[r] + ": not the shape of expert 0's";
                    return false;
                }
            }
        l.region_off[(size_t) e] = lo;
        l.region_bytes[(size_t) e] = hi - lo;
        l.region_file[(size_t) e] = file;
        for (int r = 0; r < 3; ++r)
            for (int q = 0; q < 3; ++q) {
                const size_t src = pk.get(E + parts[r] + pieces[q])->off - lo;
                if (e == 0) l.piece_src[r][q] = src;
                else if (src != l.piece_src[r][q]) {
                    err = E + ": the pieces lie differently from expert 0's";
                    return false;
                }
            }
    }
    return true;
}

// ---- loading
bool Model::Impl::load(const Options& opt, std::string& err) {
    o = opt;
    if (!pk.load(o.pack, err)) return false;
    g.from(pk);
    if (g.dim == 0 || g.nl == 0 || (int) g.ratios.size() < g.nl) {
        err = o.pack + "/ds4.txt: incomplete geometry";
        return false;
    }
    if (g.hd != 512 || g.ix_hd != 128 || g.k_exp > 8 || g.hc > 8) {
        err = "geometry not covered (head_dim " + std::to_string(g.hd) + ", index_head_dim " +
              std::to_string(g.ix_hd) + ", experts per token " + std::to_string(g.k_exp) + ")";
        return false;
    }
    freqs_win = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow);
    freqs_comp = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow);

    // devices
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        err = "no CUDA device";
        return false;
    }
    std::vector<int> ids = o.gpus;
    if (ids.empty())
        for (int i = 0; i < ndev; ++i) ids.push_back(i);
    devs.resize(ids.size());
    gpu_set = strata::core::read_gpu_set_facts(ids);
    if (prof_on) prof_pool.resize((size_t) ndev);
    for (size_t i = 0; i < ids.size(); ++i) {
        Dev& d = devs[i];
        d.id = ids[i];
        CK(cudaSetDevice(d.id));
        CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&d.ev, cudaEventDisableTiming));
        if (!x3::prepare(d.st) || (pk.gguf && !gg::prepare(d.st))) {
            err = "exl3 scratch on CUDA" + std::to_string(d.id);
            return false;
        }
        CK(cudaMemGetInfo(&d.free0, &d.total));
        d.free0 = uma_free(d.id, d.free0);
    }
    // the groups' width: o.tp, less while the experts' blocks do not split in it (a member takes n_ff / tp of each
    // expert: the down's inputs in whole blocks - a GGUF's 256-value types split n_ff 2304 in 1, 3 or 9 parts)
    int tp = o.tp;
    if (pk.gguf && tp > 1) {
        int blk = 32;
        for (const auto& kv : pk.t) {
            const std::string& n = kv.first;
            if (n.find(".ffn.experts.") == std::string::npos || n.size() < 9 ||
                n.compare(n.size() - 9, 9, "w2.weight") != 0)
                continue;
            const int ty = gg_type_of(kv.second.dtype);
            if (ty >= 0) blk = std::max(blk, gg::type_block(ty));
        }
        while (tp > 1 && (g.ff % tp != 0 || (g.ff / tp) % blk != 0 || (g.ff / tp) % (std::getenv("DS4_TP_ALIGN") ? std::atoi(std::getenv("DS4_TP_ALIGN")) : 128) != 0 || g.heads % tp != 0 ||
                          g.o_groups % tp != 0))
            --tp;
        if (tp != o.tp)
            std::fprintf(stderr, "ds4: tensor parallel %d -> %d: the experts' %d-value blocks split n_ff %d no other way "
                                 "(each GPU its own layers)\n", o.tp, tp, blk, g.ff);
    }
    // the groups: runs of tp devices of one kind (their memory), a device unlike the next alone
    // - or, tp the number of GPUs, one group of all of them whatever their kind (its tightest member sets the
    // experts' room); threads mode for the windows (decode) where a group is 2 wide or more or --threads on, the
    // prompt path (its own thread a GPU) as asked for - also where the experts' blocks took the groups' width down
    // to 1 (a 256-value type's: Maya's quants), which left every prompt in windows of 16
    par = o.threads || tp > 1;
    // the chunked (layer-major) prompt path: with --tp / --threads as before; --prefill-chunk N (N > 0) or
    // kChunkedPromptDefault bring it to a single card's default run too (a host thread a GPU for the prompt only;
    // decode stays as asked).  --prefill-chunk 0 keeps the windows of 16 (DS4_WINDOWS=1 the same).  Off by default
    // until a prompt costs the decode nothing: its buffers hold VRAM the windowed path gives to experts
    prompt_par = par || o.tp > 1 ||
                 ((o.prefill_chunk > 0 || (o.prefill_chunk < 0 && kChunkedPromptDefault)) &&
                  std::getenv("DS4_WINDOWS") == nullptr);
    const bool tp_all = tp > 1 && tp == (int) devs.size();
    for (size_t i = 0; i < devs.size();) {
        Group G;
        G.mem.push_back((int) i);
        size_t j = i + 1;
        while ((int) G.mem.size() < tp && j < devs.size() && (tp_all || devs[j].total == devs[i].total))
            G.mem.push_back((int) j++);
        if ((int) G.mem.size() < tp && G.mem.size() > 1) {   // a short run at the end: devices of their own
            for (int m : G.mem) {
                Group one;
                one.mem.push_back(m);
                groups.push_back(one);
            }
        } else groups.push_back(G);
        i = j;
    }
    for (size_t gi = 0; gi < groups.size(); ++gi)
        for (size_t m = 0; m < groups[gi].mem.size(); ++m) {
            devs[(size_t) groups[gi].mem[m]].grp = (int) gi;
            devs[(size_t) groups[gi].mem[m]].mi = (int) m;
        }
    // The default prompt chunk, one rule (ds4_experts_rule.hpp: prefill_chunk_default).  Measured: the chunked pass streams the experts the GPUs
    // do not hold once per chunk, so where they hold less than half of them a larger chunk is fewer passes (1 x R9700 + 251 GB, 8K prompt:
    // 1024 200, 2048 277, 4096 353, 8192 431 tok/s; RTX 5070 / 3060 12 GB in place: 8192 244 / 228 against 1024 186 / 175); where they hold
    // most (4 x R9700: 8192 214, 1024 370) the small one wins; with the experts copied into RAM (--mmap-experts off) 4096 beats 8192 by 7-9%
    // (H100 32 GiB: 2048 502, 4096 538, 8192 495).
    const bool own_default = !(par || o.tp > 1);   // (a run with tp 1 and no --threads: `strata launch` writes --tp 1 for 4+ AMD cards, --tp 2 for 2-3 and nothing for one card = the engine's tp 2)
    const bool multi_tp = rdna4() && !own_default && !o.threads && devs.size() >= 2;   // (the 4-card default-tp run the 1024 chunk was also measured on; an explicit --prefill-chunk below wins over the rule)
    const bool gguf_copy = pk.gguf && o.mmap_experts == 0;
    uint64_t vmin_bytes = ~0ull;
    for (const Dev& dv : devs) vmin_bytes = std::min<uint64_t>(vmin_bytes, (uint64_t) dv.total);
    if (devs.empty()) vmin_bytes = 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const bool hip_build = true;
#else
    const bool hip_build = false;
#endif
    const int rule_chunk = prefill_chunk_default(own_default, gpu_expert_share(), gguf_copy, true, hip_build,
                                                 vmin_bytes, kVlRows, multi_tp, defaults_026());
    chunk = prompt_par ? (o.prefill_chunk >= 0 ? o.prefill_chunk : rule_chunk) : 0;
    if (prompt_par && o.prefill_chunk < 0 && rule_chunk == 4096 && gguf_copy)
        std::fprintf(stderr, "ds4: prompts in chunks of 4096 tokens (the experts are copied into RAM, %.0f GiB of graphics memory; --prefill-chunk N changes it)\n",
                     (double) vmin_bytes / (double) (1ull << 30));
    const char* dc = std::getenv("DS4_CHUNK");   // (tests: down to 16, a short prompt in several chunks)
    if (dc != nullptr) chunk = prompt_par ? std::atoi(dc) : 0;
    if (const char* v = std::getenv("DS4_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(v));
    if (chunk > 0) {
        chunk = std::min(kMaxChunk, std::max(dc != nullptr ? 16 : 256, chunk / 16 * 16));
        eg_cap = (size_t) chunk;
        tap_cap = std::max(kWin, 128);
    } else   // (only as asked for: --prefill-chunk 0, or --tp 1 with --threads off)
        std::fprintf(stderr,
                     "ds4: prompts run in windows of %d tokens (%s): long prompts prefill far slower than in\n"
                     "     chunks (a 7-GPU RTX 3090 rig, Maya-S read in place: 8K / 32K tokens 115 / 99 tokens/s,\n"
                     "     461 / 1010 in chunks of %d) - %s\n",
                     kWin, o.prefill_chunk == 0 ? "--prefill-chunk 0" : "--tp 1 with --threads off", kVlRows,
                     o.prefill_chunk == 0 ? "leave --prefill-chunk out for chunks"
                                          : "leave --tp 1 out, or pass --threads on, for chunks");
    // the vision tower on the first GPU, before the split: the layers get what it leaves
    if (o.vision && !load_vision(err)) {   // (a tower that cannot load leaves the model reading text: said loudly,
                                           // and the server is told pictures are off - info_lines)
        std::fprintf(stderr, "ds4: WARNING: the vision tower did not load (%s): pictures are OFF, the model reads "
                             "text only\n", err.c_str());
        vision_err = err.empty() ? "unknown reason" : err;
        err.clear();
        vt.release();
        img_tok = -1;
    }

    // layers: roles and the sources they read
    if (o.dspark_auto && o.dspark && g.n_mtp > 0) {   // (see Options::dspark_auto)
        double exp_b = 0, room_b = 0;
        for (const auto& kt : pk.t)
            if (kt.first.rfind("layers.", 0) == 0 && kt.first.find(".ffn.experts.") != std::string::npos) exp_b += (double) kt.second.bytes;
        for (const Dev& d : devs) room_b += std::max(0.0, (double) d.free0 - (double) ((size_t) o.reserve_mib << 20));
        if (exp_b > 1.5 * room_b) {
            o.dspark = false;
            std::fprintf(stderr, "ds4: DSpark not loaded: the routed experts (%.0f GB) are far more than the GPU holds (%.0f GB), where its verify "
                                 "rounds read 3.3x a token's bytes and gain nothing (--dspark on loads it anyway)\n", exp_b / 1e9, room_b / 1e9);
        }
    }
    if (!o.dspark) g.n_mtp = 0;
    if (o.dspark_block > 0 && std::getenv("DS4_DSPARK_MAX") == nullptr) max_drafts = o.dspark_block;   // (--dspark-block N: at most N drafts a round)
    L.resize((size_t) (g.nl + g.n_mtp));
    int last_src = -1;
    int experts_used = 1 << 20;
    for (int il = 0; il < g.nl + g.n_mtp; ++il) {
        Layer& l = L[(size_t) il];
        l.il = il;
        l.mtp = il >= g.nl;
        l.name = l.mtp ? "mtp." + std::to_string(il - g.nl) + "." : "layers." + std::to_string(il) + ".";
        l.n_exp = l.mtp ? g.d_n_exp : g.n_exp;
        if (l.mtp && l.n_exp > g.n_exp) {   // the profile keeps one row width (g.n_exp) for every block: a drafter with more experts than the model would write past it
            err = l.name + ": the drafter has " + std::to_string(l.n_exp) + " experts and the model " + std::to_string(g.n_exp) + ": not covered";
            return false;
        }
        l.k_exp = l.k_model = l.mtp ? g.d_k : g.k_exp;
        if (!l.mtp) {   // experts a word takes: --experts N / the faster-mode rule (ds4_experts_rule.hpp), DS4_TOPK=N over both (a test)
            if (il == 0) {
                uint64_t vram_all = 0;
                for (const Dev& dv : devs)
                    vram_all += (uint64_t) dv.total;
                const bool integrated = gpu_set.all_integrated;
                // two or more RDNA4 cards holding at least half of the experts (the 4 x R9700 class): Faster by default too
                const bool multi_rdna4 = rdna4() && devs.size() >= 2 && gpu_expert_share() >= 0.5;
                const bool single_rdna4 = rdna4() && devs.size() == 1;   // one R9700 / RX 9000 card: the CPU lane does most of the experts
                const int n = experts_test_override(experts_per_word(o.experts, vram_all, l.k_exp, integrated, multi_rdna4, defaults_026(), single_rdna4), l.k_exp);
                experts_used = n;
                if (n < l.k_exp)
                    std::fprintf(stderr, "ds4: experts per word: %d (%s; full quality: --experts %d)\n", n,
                                 o.experts >= 1 ? "as asked" : multi_rdna4 ? "faster mode: the cards hold most of the experts" :
                                 single_rdna4 && !integrated && vram_all > kFasterModeVramMax ? "faster mode: one RDNA4 card, the CPU lane does most of the experts" : "faster mode", l.k_exp);
                else if (o.experts >= 1)
                    std::fprintf(stderr, "ds4: experts per word: %d (full quality)\n", n);
            }
            l.k_exp = std::min(l.k_exp, experts_used);
        }
        l.ratio = g.ratios[(size_t) il];
        if (l.mtp) {   // window attention only: no compressor, indexer or Engram
            if (l.ratio != 0 || l.n_exp <= 0 || l.k_exp <= 0 || l.k_exp > kPicks) {
                err = l.name + ": a DSpark block of compress ratio " + std::to_string(l.ratio) + ", " +
                      std::to_string(l.n_exp) + " experts / " + std::to_string(l.k_exp) + " is not covered";
                return false;
            }
            if (!expert_layout(l, err)) return false;
            continue;
        }
        l.kv_src = std::find(g.kv_src.begin(), g.kv_src.end(), il) != g.kv_src.end();
        l.ix_src = std::find(g.ix_src.begin(), g.ix_src.end(), il) != g.ix_src.end();
        l.owns_k = l.kv_src;
        l.cand_src = il == g.cand_src;
        l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
        if (l.kv_src) {
            Src s;
            s.layer = il;
            s.ratio = l.ratio;
            s.cap = (o.max_context + l.ratio - 1) / l.ratio + 1;
            srcs.push_back(s);
            last_src = (int) srcs.size() - 1;
        }
        if (l.ratio > 0) {
            if (last_src < 0 || srcs[(size_t) last_src].ratio != l.ratio) {
                err = "layer " + std::to_string(il) + ": compress ratio " + std::to_string(l.ratio) +
                      " without a KV source of that ratio before it";
                return false;
            }
            l.src = last_src;
            l.ik_src = last_src;
        }
        auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
        if (eg != g.eg_layers.end()) l.eg = (int) (eg - g.eg_layers.begin());
        if (!expert_layout(l, err)) return false;
    }

    // the split: each layer's dense weights + its experts, the head on the last GPU
    for (Layer& l : L) {
        const std::string& P = l.name;
        size_t b = 0;
        if (l.mtp && l.il == g.nl) b += mat_bytes("mtp.0.main_proj");
        for (const char* m : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1",
                              "ffn.shared_experts.w2", "ffn.shared_experts.w3", "attn.compressor.wkv",
                              "attn.compressor.wgate", "attn.indexer.wq_b", "attn.indexer.wk", "engram.wkv"})
            b += mat_bytes(P + m);
        for (int gi = 0; gi < g.o_groups; ++gi) b += mat_bytes(P + "attn.wo_a.slice." + std::to_string(gi));
        b += tensor_bytes(P + "hc_attn_fn") + tensor_bytes(P + "hc_ffn_fn") + tensor_bytes(P + "ffn.gate.weight");
        b += (size_t) g.win * win_rb() + (2u << 20);
        l.dense_bytes = b;
    }
    const size_t head_b = mat_bytes("head") + (size_t) g.vocab * 4 + (16u << 20);
    {
        // the trunk's cuts (Options::plan_layers), the drafter's blocks on the last stage; without them (one GPU)
        // every layer on the first
        if (o.plan_layers) {
            std::vector<uint64_t> bytes;
            for (int il = 0; il < g.nl; ++il)
                bytes.push_back(L[(size_t) il].blob);
            std::vector<int64_t> cuts;
            if (!o.plan_layers(g, bytes, cuts, err) || cuts.size() + 1 != groups.size())
                return false;
            for (Layer& l : L)
                l.grp = l.mtp ? (int) groups.size() - 1 :
                    (int) (std::upper_bound(cuts.begin(), cuts.end(), l.il) - cuts.begin());
        }
        for (Layer& l : L) {
            l.dev = groups[(size_t) l.grp].mem[0];
            l.nm = l.mtp ? 1 : (int) groups[(size_t) l.grp].mem.size();   // (the drafter's blocks: on the lead alone)
        }
        head_dev = L[(size_t) g.nl - 1].dev;
        d_dev = g.n_mtp > 0 ? L[(size_t) g.nl].dev : 0;
    }
    // the dense weights (attention, the shared experts, the head) always live in VRAM - only the routed experts
    // can go to RAM, the CPU lane or the disk - so a GPU whose free memory cannot hold its layers' dense weights
    // cannot run them: said here, before anything loads, instead of a cudaMalloc failure part way in (Maya-XS on
    // an 8 GB card: 8.46 GB of dense weights).  A device with layers of a tensor parallel group holds a part of
    // them, so only the devices whose layers are whole are checked.
    for (size_t di = 0; di < devs.size(); ++di) {
        double need = di == (size_t) head_dev ? (double) head_b : 0.0;
        int first = -1, last = -1;
        bool whole = true;
        for (const Layer& l : L) {
            if (l.dev != (int) di) continue;
            whole = whole && groups[(size_t) l.grp].mem.size() == 1;
            need += (double) l.dense_bytes;
            if (!l.mtp) {
                if (first < 0) first = l.il;
                last = l.il;
            }
        }
        const Dev& d = devs[di];
        const double room = (double) d.free0 - (double) ((size_t) o.reserve_mib << 20);
        if (!whole || need <= 0 || need <= room) continue;
        // (past the free memory itself the cudaMalloc of these weights cannot succeed: said here, before anything loads.
        //  Between the free memory minus the reserve and the free memory they fit but leave less than --reserve-mib for the
        //  caches and scratch: a warning, the load goes on)
        const bool cannot = need > (double) d.free0;
        char msg[1024];
        std::snprintf(msg, sizeof msg,
                      "CUDA%d has %.2f GB free of %.2f GB, but the dense weights of %s%s need %.2f GB on it "
                      "(attention, shared experts%s): they always stay in VRAM - the CPU lane and --vram-experts "
                      "move only the routed experts. To run it: %s, or more GPUs (the layers are split across "
                      "them), or close other programs using this GPU",
                      d.id, (double) d.free0 / 1e9, (double) d.total / 1e9,
                      first < 0 ? "the drafter"
                                : ("layers " + std::to_string(first) + "-" + std::to_string(last)).c_str(),
                      di == (size_t) head_dev ? " and the head" : "", need / 1e9,
                      di == (size_t) head_dev ? ", the output head" : "",
                      (devs.size() == 1 ? "a GPU with about " +
                                              std::to_string((int) std::ceil((need + 2.5e9) / 1e9)) +
                                              " GB of VRAM or more"
                                        : std::string("GPUs with more free VRAM"))
                          .c_str());
        if (cannot) {
            err = msg;
            return false;
        }
        std::fprintf(stderr, "ds4: WARNING: CUDA%d has %.2f GB free and the dense weights of its layers need %.2f GB: they "
                             "fit, but leave less than the %lld MiB reserve for the caches and scratch (--reserve-mib); the "
                             "load goes on\n", d.id, (double) d.free0 / 1e9, need / 1e9, (long long) o.reserve_mib);
    }
    // each layer's FFN columns and heads: all of them
    for (Layer& l : L) {
        l.ff0 = 0;
        l.ffn = g.ff;
        l.ah0 = l.og0 = 0;
        l.ahn = g.heads;
        l.ogn = g.o_groups;
    }
    // which devices hold each source's compressed KV / index keys
    for (Src& s : srcs) {
        s.ckv.assign(devs.size(), make_kv_rows(nullptr, kKvFp4, g.hd));
        s.ik.assign(devs.size(), make_kv_rows(nullptr, kKvFp4p, g.ix_hd));
    }
    for (const Layer& l0 : L)
        for (int m = 0; m < l0.nm; ++m) {
            const Layer& l = member(l0, m);
            if (l.src >= 0) {
                srcs[(size_t) l.src].ckv[(size_t) l.dev].base = (uint8_t*) 1;
                if (l0.grp != L[(size_t) srcs[(size_t) l.src].layer].grp) srcs[(size_t) l.src].remote = true;
            }
            if (l.ix_src && l.ik_src >= 0) {
                srcs[(size_t) l.ik_src].ik[(size_t) l.dev].base = (uint8_t*) 1;
                if (l0.grp != L[(size_t) srcs[(size_t) l.ik_src].layer].grp) srcs[(size_t) l.ik_src].remote = true;
            }
        }

    // per device: the scratch, the caches, the dense weights
    for (size_t di = 0; di < devs.size(); ++di) {
        Dev& d = devs[di];
        const int D = g.dim, H = g.hc;
        const size_t W = kWin, P = (size_t) kWin * kPicks, cap = (size_t) o.max_context + 64;
        bool ok = dalloc(d, d.h, W * H * D, err) && dalloc(d, d.h2, W * H * D, err) && dalloc(d, d.x, W * D, err) &&
                  dalloc(d, d.qr, W * g.q_lora, err) && dalloc(d, d.q, W * g.heads * g.hd, err) &&
                  dalloc(d, d.kvr, W * g.hd, err) && dalloc(d, d.ckr, W * g.hd, err) &&
                  dalloc(d, d.cgr, W * g.hd, err) && dalloc(d, d.latent, W * g.hd, err) &&
                  dalloc(d, d.ikr, W * g.ix_hd, err) && dalloc(d, d.iq, W * g.ix_heads * g.ix_hd, err) &&
                  dalloc(d, d.iw, W * 64, err) && dalloc(d, d.scores, W * cap, err) &&
                  dalloc(d, d.bm, W * bm_stride(), err) && dalloc(d, d.o, W * g.heads * g.hd, err) &&
                  dalloc(d, d.oa, W * g.o_groups * g.o_lora, err) && dalloc(d, d.ao, W * D, err) &&
                  dalloc(d, d.mix, W * 32, err) && dalloc(d, d.a_pre, W * 8, err) && dalloc(d, d.a_post, W * 8, err) &&
                  dalloc(d, d.a_comb, W * 64, err) && dalloc(d, d.f_pre, W * 8, err) &&
                  dalloc(d, d.f_post, W * 8, err) && dalloc(d, d.f_comb, W * 64, err) && dalloc(d, d.pre, W * 8, err) &&
                  dalloc(d, d.glog, W * 1024, err) &&
                  dalloc(d, d.rw, W * kPicks, err) &&
                  dalloc(d, d.ridx, W * kPicks, err) &&
                  dalloc(d, d.moe_out, W * D, err) &&
                  dalloc(d, d.sg, W * g.ff, err) && dalloc(d, d.su, W * g.ff, err) && dalloc(d, d.sh, W * g.ff, err) &&
                  dalloc(d, d.sh_out, W * D, err) && dalloc(d, d.cpu_part, W * D, err) &&
                  dalloc(d, d.freqs_win, 64, err) && dalloc(d, d.freqs_comp, 64, err) &&
                  dalloc(d, d.topk, W * topk_stride(), err) && dalloc(d, d.cand, W * cand_stride(), err) &&
                  dalloc(d, d.tok, 16, err) && dalloc(d, d.emb_rows, W * D, err) &&
                  dalloc(d, d.ext, W * ext_slice_rows() * g.hd, err) && dalloc(d, d.xe, P * D, err) &&
                  dalloc(d, d.xg, P * g.ff, err) && dalloc(d, d.xu, P * g.ff, err) && dalloc(d, d.xhh, P * g.ff, err) &&
                  dalloc(d, d.ed, P * D, err) && dalloc(d, d.plan_ptr, P, err);
    if (ok && ds4_dbg("DS4_DBG_DEVH"))
        std::fprintf(stderr, "DBG devh: id=%d h=%p h2=%p\n", d.id, (void*) d.h, (void*) d.h2);
        if (!ok) return false;
        d.attn_scr.max_t = 8;   // (run_attention's batch of independent rows; sparse_attn's own: kAttnScratchT)
        d.attn_scr.cap = (int) topk_stride();
        if (!dalloc(d, d.attn_scr.rows, (size_t) d.attn_scr.max_t * topk_stride() * g.hd, err)) return false;
        if (!g.eg_layers.empty()) {
            const int cols = (pk.eg_ngram - 1) * pk.eg_heads;
            ok = dalloc(d, d.eg_rows, W * cols * (pk.eg_hd + pk.eg_hd / 32), err) &&
                 dalloc(d, d.eg_x, W * cols * pk.eg_hd, err) && dalloc(d, d.eg_kv, W * (g.hc + 1) * D, err);
            if (!ok) return false;
        }
        if ((int) di == head_dev && !dalloc(d, d.logits, (size_t) kWin * g.vocab, err)) return false;
        if (!halloc(d.bounce, bounce_bytes(), err)) return false;
        if (!dalloc(d, d.xcnt, 4, err) || !dalloc(d, d.hccnt, kWin, err) || !hmap(d.hand, bounce_bytes(), err) || !hmap(d.hand_flag, 32, err) ||
            !dalloc(d, d.grp_ptr, P, err) || !dalloc(d, d.grp_start, P + 1, err) || !dalloc(d, d.grp_ent, P, err) ||
            !dalloc(d, d.n_groups, 4, err))
            return false;
        if (g.n_mtp > 0 && !dalloc(d, d.tap, (size_t) tap_cap * D, err)) return false;
        if (g.n_mtp > 0 && (int) di == head_dev &&
            (!dalloc(d, d_confv, kWin, err) || !dalloc(d, d_ints, 64, err) || !dalloc(d, d_specq, kWin, err)))
            return false;
        CK(cudaEventCreateWithFlags(&d.ev_out, cudaEventDisableTiming));
        cudaSetDevice(d.id);
        CK(cudaMemcpy(d.freqs_win, freqs_win.data(), freqs_win.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.freqs_comp, freqs_comp.data(), freqs_comp.size() * 4, cudaMemcpyHostToDevice));
        for (Src& s : srcs) {
            if (s.h_ckv == nullptr) {   // (its events belong to the source layer's device: it records them)
                if (!halloc(s.h_ckv, (size_t) kWin * ckv_rb(), err) || !halloc(s.h_ik, (size_t) kWin * ik_rb(), err))
                    return false;
                CK(cudaSetDevice(devs[(size_t) L[(size_t) s.layer].dev].id));
                CK(cudaEventCreateWithFlags(&s.ev_ckv, cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&s.ev_ik, cudaEventDisableTiming));
                CK(cudaSetDevice(d.id));
            }
        }
    }
    for (Src& sr : srcs) {
        sr.remote = sr.remote || devs.size() > 1;
        if (sr.remote && (!hmap(sr.x_stage, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag, 32, err)))
            return false;
        if (sr.remote && chunk > 0 &&
            (!hmap(sr.x_stage1, x_cap() * (ckv_rb() + ik_rb()), err) || !hmap(sr.x_flag1, 32, err) ||
             !hmap(sr.x_ack, (size_t) 2 * devs.size() * 32, err)))
            return false;
        const int sg = L[(size_t) sr.layer].grp;
        for (size_t di = 0; di < devs.size(); ++di)
            if ((sr.ckv[di] || sr.ik[di]) && devs[di].grp != sg) sr.readers.push_back((int) di);
    }
    if (g.n_mtp > 0) {
        if (!halloc(h_taps, (size_t) tap_cap * g.d_targets.size() * g.dim, err)) return false;
        CK(cudaSetDevice(devs[(size_t) d_dev].id));
        CK(cudaEventCreateWithFlags(&ev_seed, cudaEventDisableTiming));
        CK(cudaEventRecord(ev_seed, devs[(size_t) d_dev].st));
    }
    // the dense weights, each device's layers by a thread of its own
    {
        std::vector<std::string> errs(devs.size());
        std::vector<std::thread> th;
        for (size_t di = 0; di < devs.size(); ++di)
            th.emplace_back([this, di, &errs] {
                std::string e;
                if (!load_dense((int) di, e)) errs[di] = e;
            });
        for (auto& t : th) t.join();
        for (auto& e : errs)
            if (!e.empty()) {
                err = e;
                return false;
            }
    }

    // the embedding table, pinned on the host
    {
        const Tensor* t = pk.get("embed.weight");
        const size_t nv = (size_t) g.vocab * g.dim;
        const bool q8 = t != nullptr && t->dtype == "Q8_0" && t->bytes == nv / 32 * gg::kQ8Bytes;
        // other block types (a GGUF whose dense tensors are Q4_K / Q5_K / Q6_K: the ISTA-DAS sets' Q5_K embed.weight):
        // each row through ggml's to_float, to the same BF16 the host table holds
        const int ety = t != nullptr ? gg_type_of(t->dtype) : -1;
        const size_t erb = ety > 0 ? gg::type_row_bytes(ety, g.dim) : 0;
        const bool qrow = t != nullptr && !(t->dtype == "BF16") && erb > 0 && t->bytes == (size_t) g.vocab * erb;
        if (t == nullptr || (!q8 && !qrow && (t->dtype != "BF16" || t->bytes != nv * 2))) {
            err = "embed.weight: not a BF16 (or Q8_0 / a ggml block type) [vocab][dim] table";
            return false;
        }
        if (!halloc(embed_pinned, nv, err)) return false;
        if (qrow) {   // (ggml's decode of each row, rounded to BF16 as the Q8_0 path rounds)
            std::vector<uint8_t> row(erb);
            for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                if (!pk.read_at(t->file, t->off + r * erb, erb, row.data())) {
                    err = "embed.weight: read failed";
                    return false;
                }
                if (!gg::gg_row_to_bf16(ety, row.data(), g.dim, embed_pinned + r * g.dim)) {
                    err = "embed.weight: ggml has no decode for " + t->dtype;
                    return false;
                }
            }
        } else if (q8) {   // (a GGUF's: each value rounded to BF16, a row at a time)
            std::vector<uint8_t> row((size_t) g.dim / 32 * gg::kQ8Bytes);
            for (size_t r = 0; r < (size_t) g.vocab; ++r) {
                if (!pk.read_at(t->file, t->off + r * row.size(), row.size(), row.data())) {
                    err = "embed.weight: read failed";
                    return false;
                }
                const std::vector<float> v = q8_values(row.data(), (size_t) g.dim);
                for (size_t c = 0; c < (size_t) g.dim; ++c) {
                    uint32_t u;
                    std::memcpy(&u, &v[c], 4);
                    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
                    embed_pinned[r * g.dim + c] = (uint16_t) (u >> 16);
                }
            }
        } else if (!pk.read(*t, embed_pinned)) {
            err = "embed.weight: read failed";
            return false;
        }
        if (ds4_dbg("DS4_DBG_EMB")) std::fprintf(stderr,
                "DBG emb pinned: %04x %04x %04x %04x  (row1: %04x %04x)\n",
                (unsigned) embed_pinned[0], (unsigned) embed_pinned[1],
                (unsigned) embed_pinned[2], (unsigned) embed_pinned[3],
                (unsigned) embed_pinned[(size_t) g.dim], (unsigned) embed_pinned[(size_t) g.dim + 1]);

    }
    // the host side of the CPU lane and the Engram rows
    if (!halloc(h_x, (size_t) kWin * g.dim, err) || !halloc(h_w, (size_t) kWin * kPicks, err) ||
        !halloc(h_part, (size_t) kWin * g.dim, err) || !halloc(h_idx, (size_t) kWin * kPicks, err) ||
        !halloc(h_emb, (size_t) std::max(kWin, chunk) * g.dim, err) ||
        !halloc(h_eg, get_engram_bytes(eg_cap), err))
        return false;
    if (pk.eg_layers > 0) {
        // the Engram rows' reader: threads enough to keep an SSD's queue busy (reads from the page cache need few);
        // the row cache a small share of the RAM free (DS4_ENGRAM_CACHE_MB: its size, 0 off; DS4_ENGRAM_THREADS)
        const size_t row = (size_t) pk.eg_hd + pk.eg_hd / 32;
#if defined(_WIN32)
        // (Windows: each thread's ReadFile is synchronous, so the threads are the queue depth - Project Maya: a 28K-token
        //  prompt in one chunk, 10 threads 17.1 s, 64 13.4 s on a 20-core PC.  Here (6 cores, shipped profile) 64 readers
        //  gave +4.5% on a 4K prompt but no decode gain in every pair: DS4_ENGRAM_THREADS=64 turns them on)
#endif
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int nth = (int) std::min(16u, std::max(4u, hw / 2));
        if (const char* e = std::getenv("DS4_ENGRAM_THREADS")) nth = std::max(1, std::atoi(e));
        const uint64_t avail = ram_avail();
        uint64_t mb = avail == ~uint64_t{0} ? 64 : std::min<uint64_t>(256, std::max<uint64_t>(16, (avail >> 20) / 64));
        if (const char* e = std::getenv("DS4_ENGRAM_CACHE_MB")) mb = std::strtoull(e, nullptr, 10);
        eg_cache.init((size_t) ((mb << 20) / row), row);
        std::vector<std::string> paths;
        for (const std::string& fn : pk.files) paths.push_back(pk.model_dir + "/" + fn);
        // Windows: the Engram tables' files that hold nothing else are unmapped and read unbuffered - an SSD's
        // queue kept by the reader threads instead of the page cache's (whose reads of one file from many threads
        // went ~255K rows a second here); unbuffered reads of a mapped file take a slow path, so only unmapped
        // ones (off by default: -8% on a 4K prompt here; DS4_EG_DIRECT=1 turns it on)
        std::vector<char> eg_direct(pk.files.size(), 0);
#if defined(_WIN32)
        if (pk.gguf && std::getenv("DS4_EG_DIRECT") != nullptr && std::atoi(std::getenv("DS4_EG_DIRECT")) != 0) {
            std::vector<char> eg_file(pk.files.size(), 0), other(pk.files.size(), 0);
            for (const auto& kt : pk.t)
                if (kt.second.file >= 0 && (size_t) kt.second.file < pk.files.size())
                    (kt.first.find(".engram.embed.") != std::string::npos ? eg_file : other)[(size_t) kt.second.file] = 1;
            for (size_t i = 0; i < pk.files.size(); ++i)
                if (eg_file[i] && !other[i]) {
                    if (i < pk.maps.size() && pk.maps[i].first != nullptr) {
                        platform::unmap(pk.maps[i].first, pk.maps[i].second);
                        pk.maps[i] = {nullptr, 0};
                    }
                    eg_direct[i] = 1;
                }
        }
#endif
        eg_io.start(paths, nth, eg_direct);
        for (int li = 0; li < pk.eg_layers; ++li) {
            eg_win.emplace_back(new EgBatch());
            const std::string id = std::to_string(g.eg_layers[(size_t) li]);
            const Tensor* tw = pk.get("layers." + id + ".engram.embed.weight");
            // The row layout the reader and the dequant take.  A type not here (or no tensor at all) would be read
            // as some other format's bytes - the old code then read the scale tensor of a pack that has none and
            // crashed in eg_issue - so refuse the pack once, with the type's name, before any token is read.
            int fmt;
            if (tw == nullptr) {
                std::fprintf(stderr, "ds4: Engram layer %s: engram.embed.weight is missing from the pack\n", id.c_str());
                std::exit(2);
            } else if (tw->dtype == "MXFP8") {
                fmt = 0;
            } else if (tw->dtype == "Q6_K") {
                fmt = 1;
            } else if (tw->dtype == "IQ4_NL") {
                fmt = 2;
            } else if (pk.get("layers." + id + ".engram.embed.scale") != nullptr) {
                fmt = 3;   // (an old pack: FP8 codes in one tensor, their E8M0 scales in another)
            } else {
                std::fprintf(stderr,
                             "ds4: Engram layer %s: engram.embed.weight type %s is not one the Engram reader handles "
                             "(MXFP8, Q6_K, IQ4_NL)\n",
                             id.c_str(), tw->dtype.c_str());
                std::exit(2);
            }
            if (fmt == 1 && pk.eg_hd % 256 != 0) {
                err = "Engram layer " + id + ": Q6_K rows of " + std::to_string(pk.eg_hd) +
                      " values (not whole superblocks of 256)";
                return false;
            }
            if (fmt == 2 && pk.eg_hd % 32 != 0) {
                err = "Engram layer " + id + ": IQ4_NL rows of " + std::to_string(pk.eg_hd) +
                      " values (not whole blocks of 32)";
                return false;
            }
            eg_fmt.push_back((uint8_t) fmt);
        }
    }
    prof.assign((size_t) (g.nl + g.n_mtp) * g.n_exp, 0u);
    std::error_code profile_ec;
    if (!o.profile.empty() && (std::filesystem::exists(o.profile, profile_ec) || profile_ec)) {
        std::vector<uint32_t> p;
        std::string perr;
        if (!strata::core::read_expert_counts(o.profile, g.n_exp, g.nl + kProfileExtraRows, p, perr)) {
            std::fprintf(stderr, "ds4: %s - the experts start unranked\n", perr.c_str());
        } else if (!profile_size_fits((uint64_t) p.size() * 4, g.nl, g.n_exp)) {
            std::fprintf(stderr, "ds4: %s holds %zu rows and this model's profile is %d layers of %d experts "
                                 "(and a few drafter rows): "
                                 "not read, the experts start unranked\n",
                         o.profile.c_str(), p.size() / g.n_exp, g.nl, g.n_exp);
        } else if (p.size() >= prof.size() || p.size() == (size_t) g.nl * g.n_exp) {
            std::copy_n(p.begin(), std::min(p.size(), prof.size()), prof.begin());
        } else {
            std::fprintf(stderr, "ds4: %s has incomplete drafter rows - the experts start unranked\n",
                         o.profile.c_str());
        }
    }
    prof_load = prof;
    return true;
}

bool Model::Impl::finish_load(std::string& err) {
    // the chunked prompt path's buffers (before the experts: they take what is left)
    if (chunk > 0) {
        // recommend, never force: a chunk the cards cannot hold in buffers is halved (down to 256 rows) and then the
        // prompt runs in windows of 16, said in one line - the load only fails for a reason that is not this
        std::vector<size_t> na0, sc0;
        for (Dev& d : devs) {
            na0.push_back(d.allocs.size());
            sc0.push_back(d.scratch);
        }
        const size_t nh = host_allocs.size();
        std::string aerr;
        while (chunk > 0 && !alloc_chunks(aerr)) {
            for (size_t di = 0; di < devs.size(); ++di) {
                Dev& d = devs[di];
                cudaSetDevice(d.id);
                cudaDeviceSynchronize();
                if (di < chunks.size() && chunks[di]) {
                    if (chunks[di]->blas) cublasDestroy(chunks[di]->blas);
                    if (chunks[di]->cs) cudaStreamDestroy(chunks[di]->cs);
                }
                while (d.allocs.size() > na0[di]) {
                    strata::core::release_device(d.allocs.back());
                    d.allocs.pop_back();
                }
                d.scratch = sc0[di];
            }
            while (host_allocs.size() > nh) {
                strata::core::release_host(host_allocs.back());
                host_allocs.pop_back();
            }
            chunks.clear();
            if (slab_auto_grew_) {   // the automatic slab did not fit: the 0.2.6 slab first, the chunk only if that does not fit either
                slab_auto_grew_ = false;
                slab_force_base_ = true;
                std::fprintf(stderr, "ds4: the automatic attention slab did not fit (%s): trying the 256-row slab\n", aerr.c_str());
                aerr.clear();
                continue;
            }
            const int was = chunk;
            chunk = chunk > 512 ? chunk / 2 / 16 * 16 : 0;
            if (chunk > 0) {
                eg_cap = (size_t) chunk;
                std::fprintf(stderr, "ds4: the prompt path's buffers for chunks of %d rows did not fit (%s): trying %d\n",
                             was, aerr.c_str(), chunk);
            } else {
                prompt_par = false;
                eg_cap = (size_t) kWin;
                tap_cap = kWin;
                std::fprintf(stderr, "ds4: the prompt path's buffers did not fit (%s): prompts run in windows of %d "
                                     "tokens (slower on long prompts; more free GPU memory or a lower --reserve-mib "
                                     "brings the chunks back)\n", aerr.c_str(), kWin);
            }
            aerr.clear();
        }
        if (chunk > 0 && aerr.empty()) err.clear();
    }
    prompt_ceiling = chunk;
    // the CPU lane: one part (the blob's layout) and one pool over every node's cores (the pages lie where the pool
    // put them)
    static const bool smt_cores = [] {   // DS4_CPU_SMT=1 (Windows): the second hardware thread of every core as well
#if defined(_WIN32)
        const char* v = std::getenv("DS4_CPU_SMT");
        return v != nullptr && std::atoi(v) != 0;
#else
        return false;
#endif
    }();
    core_threads = node_threads();
    cores.clear();
    for (const auto& n : core_threads) {
        cores.emplace_back();
        for (const platform::Core& c : n) {
            cores.back().push_back(c.cpus[0]);
            if (smt_cores)
                for (size_t k = 1; k < c.cpus.size(); ++k) cores.back().push_back(c.cpus[k]);
        }
    }
    make_pools();
    {
        const size_t E = (size_t) g.dim, FF = (size_t) g.ff, GQ = E / 256, DQ = FF / 128, S = kSlots;
        c_xh.resize((size_t) NH * S * 2 * E);
        c_xsb.resize((size_t) NH * S * 2 * (E / 128));
        c_acc.resize((size_t) NH * S * 2 * GQ * (FF / NH));
        c_xhd.resize(S * FF);
        c_xsd.resize(S * (FF / 128));
        c_accd.resize((size_t) NH * S * DQ * (E / NH));
        c_yd.resize(S * E);
    }
    if (!place_paged(err)) return false;
    if (chunk > 0) {
        // each device's staging for the streamed experts and its readers, each layer's virtual slots
        {   // (DS4_READERS, 8)
            size_t mr = 0;
            for (const Layer& l : L)
                if (!l.mtp) {
                    for (uint64_t r : l.region_bytes) mr = std::max<size_t>(mr, r);
                    if (l.lay.fmt == gg::kFmtMXFP4) {   // (a GGUF's: read as its blob)
                        mr = std::max(mr, l.blob);
                        for (int r = 0; r < 3; ++r) flow_part[r] = std::max(flow_part[r], mx_part_bytes(l, r));
                    }
                }
            const char* nr = std::getenv("DS4_READERS");
            const int n_readers = std::max(1, nr != nullptr ? std::atoi(nr) : 8);
            const bool flow_on = std::getenv("DS4_FLOW") == nullptr || std::atoi(std::getenv("DS4_FLOW")) != 0;
            for (size_t di = 0; di < chunks.size(); ++di) {
                auto& cp = chunks[di];
                if (cp == nullptr || cp->ring_n == 0) continue;
                cp->stage_stride = (mr + 4095) & ~(size_t) 4095;
                // the flow: every trunk layer of the device a GGUF's (alone or a tensor-parallel group's member)
                bool fl = flow_on;
                int n_exp_max = 0;
                for (const Layer& l : L)
                    if (!l.mtp)
                        for (int m = 0; m < l.nm; ++m)
                            if (member(l, m).dev == (int) di) {
                                fl = fl && l.lay.fmt == gg::kFmtMXFP4;
                                n_exp_max = std::max(n_exp_max, l.n_exp);
                            }
                if (fl)   // (each part read unbuffered from its offset rounded down: a page more each)
                    cp->stage_stride = std::max(cp->stage_stride, flow_stride());
                cp->S = cp->ring_n;
                cp->readers = std::make_unique<Pool>(n_readers - 1, std::vector<int>{}, 50);
                if (!fl) continue;
                cudaSetDevice(devs[di].id);
                cp->flow = true;
                cp->dev = (int) di;
                cp->io = std::make_unique<IoQueue>(n_readers);
                cp->stage_item.assign((size_t) cp->S, -1);
                cp->stage_ev.resize((size_t) cp->S);
                cp->ring_fill.resize((size_t) cp->ring_n);
                cp->ring_free.resize((size_t) cp->ring_n);
                for (auto* v : {&cp->stage_ev, &cp->ring_fill, &cp->ring_free})
                    for (cudaEvent_t& e : *v) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                // (a region is written again only bc_n batches later - more than a layer's items, so after that
                // layer's bounds wait, which every copy from it precedes)
                cp->bc_n = n_exp_max + 2;
                cp->bc_w = cp->ring_n + 1;
                if (!halloc(cp->bc_pin, (size_t) cp->bc_n * cp->bc_w, err, kPromptLabel) ||
                    !dalloc(devs[di], cp->bc_dev, (size_t) cp->bc_n * cp->bc_w, err, nullptr, kPromptLabel))
                    return false;
            }
            // every device with layers on the flow: a prompt reads only the experts it picks, once, so a short one
            // takes the chunked path too (one pass, where windows of 16 read the CPU lane's picks again for each
            // window, a page fault at a time) - DS4_CHUNK_MIN still overrides
            bool all_flow = true;
            for (const auto& cp : chunks) all_flow = all_flow && (cp == nullptr || cp->ring_n == 0 || cp->flow);
            if (all_flow && std::getenv("DS4_CHUNK_MIN") == nullptr) chunk_min = kFlowChunkMin;
        }
        for (Layer& l : L) {
            if (l.mtp) continue;
            for (int m = 0; m < l.nm; ++m) {   // (each member its own: the flow's members decide them each)
                Layer& x = member(l, m);
                if (!halloc(x.vs_pin, (size_t) l.n_exp, err, kPromptLabel)) return false;
                if (!dalloc(devs[(size_t) x.dev], x.vs_dev, (size_t) l.n_exp, err, nullptr, kPromptLabel)) return false;
            }
        }
    }
    {
        void* m = nullptr;
        if (strata::core::allocate_host(&m, sizeof(Req) * kRing, cudaHostAllocMapped | cudaHostAllocPortable,
            {"other", "Device allocations", "DeepSeek working buffers"}) != cudaSuccess) {
            err = "host-mapped memory for the decode path's requests";
            return false;
        }
        host_allocs.push_back(m);
        reqs = (Req*) m;
        if (strata::core::allocate_host(&m, sizeof(Resp) * kRing, cudaHostAllocMapped | cudaHostAllocPortable,
            {"other", "Device allocations", "DeepSeek working buffers"}) != cudaSuccess) {
            err = "host-mapped memory for the decode path's answers";
            return false;
        }
        host_allocs.push_back(m);
        resps = (Resp*) m;
        std::memset((void*) reqs, 0, sizeof(Req) * kRing);
        std::memset((void*) resps, 0, sizeof(Resp) * kRing);
        for (auto& x : layer_of) x.store(-1);
        const int core = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
        service = std::thread([this, core] {
            if (core >= 0) pin_self(core);
            serve_lane();
        });
    }
    for (Dev& d : devs) {
        cudaSetDevice(d.id);
        CK(cudaStreamSynchronize(d.st));
    }
    if (par || chunk > 0)   // (the windows' threads; the prompt path's, whatever mode the windows take)
        for (size_t di = 0; di < devs.size(); ++di) {
            workers.push_back(std::make_unique<Worker>());
            workers.back()->start();
        }
    if (chunk > 0) (void) pick_flow_min();   // (the startup line)
    reset();
    start_warm();
    return true;
}

void Model::Impl::warm_flow() {
    const int T = 16, D = g.dim, hcd = g.hc * g.dim, n_mix = (2 + g.hc) * g.hc;
    const float one = 1.0f, zero = 0.0f;
    for (size_t di = 0; di < chunks.size(); ++di) {
        if (!chunks[di] || chunks[di]->blas == nullptr) continue;
        const Chunk& c = *chunks[di];
        const Dev& d = devs[di];
        float *h = c.h, *mix = c.mix, *y = c.y;
        uint16_t *w16 = c.w16, *x16 = c.x16;
        float* own = nullptr;   // (the chunk has no buffers until a prompt's lend)
        cudaSetDevice(d.id);
        if (c.C < T) {
            const size_t n_h = (size_t) hcd * std::max(n_mix, T), n_f = n_h + 32 * T + 256 * T;
            own = (float*) strata::core::gpu_alloc(n_f * sizeof(float) + (size_t) (256 + T) * D * 2);
            if (own == nullptr) {
                (void) cudaGetLastError();
                continue;
            }
            h = own;
            mix = h + n_h;
            y = mix + 32 * T;
            w16 = (uint16_t*) (own + n_f);
            x16 = w16 + (size_t) 256 * D;
        }
        (void) cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n_mix, T, hcd, &one, h, hcd, h, hcd, &zero, mix, 32);
        const int ns[2] = {256, 32};
        for (int n : ns)
            (void) cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, D, &one, w16, CUDA_R_16F, D, x16, CUDA_R_16F, D, &zero,
                                y, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        (void) cudaStreamSynchronize(d.st);
        if (own != nullptr)
            strata::core::gpu_free(own);
        (void) cudaGetLastError();
    }
}

void Model::Impl::start_warm() {
    if (chunk <= 0 ||(std::getenv("DS4_WARM") && std::atoi(std::getenv("DS4_WARM")) == 0)) return;
    warm_bg = std::thread([this] {
        std::lock_guard<std::mutex> lk(warm_mu);
        if (flow_used.load() || quit.load()) return;
        warm_flow();
    });
}

// ---- the dense weights, the experts' places, the CPU lane's pools
bool Model::Impl::load_dense(int di, std::string& err, bool drafter_only, int layer_only) {
    Dev& d = devs[(size_t) di];
    cudaSetDevice(d.id);
    size_t* acct = &d.weights;
    for (Layer& l0 : L)
      for (int mi = 0; mi < l0.nm; ++mi) {
        Layer& l = member(l0, mi);
        if (l.dev != di || (drafter_only && !l.mtp) || (layer_only >= 0 && l.il != layer_only))
            continue;
        const size_t first_allocation = d.allocs.size();
        d.loading_drafter = l.mtp;
        const size_t before = d.weights, scratch_before = d.scratch;
        // (a group's member: its share of the heads (wq_b columns, the sinks), of wo_a's groups and wo_b's rows,
        // of the shared expert's FFN; the rest whole)
        const int nm = l0.nm, na = g.heads / std::max(1, l.ahn), ha = l.ah0 / std::max(1, l.ahn);
        const std::string& P = l.name;
        const std::string A = P + "attn.";
        bool ok = mat(d, A + "wq_a", l.wq_a, err, acct) &&
                  (l.ahn == 0 || mat(d, A + "wq_b", l.wq_b, err, acct, ha, na, true)) &&
                  mat(d, A + "wkv", l.wkv, err, acct) &&
                  (l.ahn == 0 || mat(d, A + "wo_b", l.wo_b, err, acct, ha, na, false)) &&
                  f32(d, A + "q_norm.weight", l.q_norm, err, acct, (size_t) g.q_lora) &&
                  f32(d, A + "kv_norm.weight", l.kv_norm, err, acct, (size_t) g.hd) &&
                  f32(d, A + "attn_sink", l.sink, err, acct, (size_t) g.heads) &&
                  f32(d, P + "attn_norm.weight", l.attn_norm, err, acct, (size_t) g.dim) &&
                  f32(d, P + "ffn_norm.weight", l.ffn_norm, err, acct, (size_t) g.dim) &&
                  f32(d, P + "hc_attn_fn", l.hca_fn, err, acct) && f32(d, P + "hc_attn_scale", l.hca_scale, err, acct) &&
                  f32(d, P + "hc_attn_base", l.hca_base, err, acct) && f32(d, P + "hc_ffn_fn", l.hcf_fn, err, acct) &&
                  f32(d, P + "hc_ffn_scale", l.hcf_scale, err, acct) &&
                  f32(d, P + "hc_ffn_base", l.hcf_base, err, acct) &&
                  f32(d, P + "ffn.gate.weight", l.gate_w, err, acct, (size_t) l.n_exp * g.dim) &&
                  f32(d, P + "ffn.gate.bias", l.gate_bias, err, acct, (size_t) l.n_exp) &&
                  (l.mtp || img_tok < 0 || pk.get(P + "ffn.gate.bias_vl") == nullptr ||
                   f32(d, P + "ffn.gate.bias_vl", l.gate_bias_vl, err, acct, (size_t) l.n_exp)) &&
                  mat(d, P + "ffn.shared_experts.w1", l.s1, err, acct, mi, nm, true, l.il % nm) &&
                  mat(d, P + "ffn.shared_experts.w2", l.s2, err, acct, mi, nm, false, l.il % nm) &&
                  mat(d, P + "ffn.shared_experts.w3", l.s3, err, acct, mi, nm, true, l.il % nm);
        if (!ok) return false;
        l.sink += (size_t) l.ah0;
        for (int gi = 0; gi < l.ogn; ++gi)
            if (!mat(d, A + "wo_a.slice." + std::to_string(l.og0 + gi), l.wo_a[gi], err, acct)) return false;
        if (l.kv_src) {
            const std::string C = A + "compressor.";
            if (!mat(d, C + "wkv", l.c_wkv, err, acct) || !f32(d, C + "norm.weight", l.c_norm, err, acct)) return false;
            if (l.ratio > 1 && !mat(d, C + "wgate", l.c_wgate, err, acct)) return false;
        }
        if (l.ix_src) {
            const std::string I = A + "indexer.";
            if (!mat(d, I + "wq_b", l.i_wq_b, err, acct) || !f16raw(d, I + "weights_proj.weight", l.i_wproj, err, acct))
                return false;
            if (l.owns_k && (!mat(d, I + "wk", l.i_wk, err, acct) || !f32(d, I + "k_norm.weight", l.ik_norm, err, acct)))
                return false;
        }
        if (l.eg >= 0) {
            if (!mat(d, P + "engram.wkv", l.eg_wkv, err, acct)) return false;
            if (pk.get(P + "engram.gate.weight") != nullptr) {   // a GGUF: q_weight * k_weight, folded already
                if (!f32(d, P + "engram.gate.weight", l.eg_w, err, acct, (size_t) g.hc * g.dim)) return false;
            } else {
                float *qw = nullptr, *kw = nullptr;
                if (!f32(d, P + "engram.q_weight", qw, err, acct) || !f32(d, P + "engram.k_weight", kw, err, acct))
                    return false;
                std::vector<float> a((size_t) g.hc * g.dim), b(a.size());
                CK(cudaMemcpy(a.data(), qw, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), kw, b.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];   // only ever used as a product
                CK(cudaMemcpy(qw, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
                l.eg_w = qw;
            }
        }
        if (g.n_mtp > 0 && !l.mtp && !init_verify_layer(l, err)) return false;
        d.loading_drafter = false;
        if (!drafter_only && !dalloc(d, l.vram_tab, (size_t) l.n_exp, err, acct)) return false;
        l.dense_scratch_bytes = d.scratch - scratch_before;
        l.weight_bytes = d.weights - before;
        l.dense_allocations.assign(d.allocs.begin() + first_allocation, d.allocs.end());
      }
    if (layer_only >= 0)
        return true;
    // The head, on the last layer's device.
    if (!drafter_only && d.grp == devs[(size_t) head_dev].grp) {
        if (!mat(d, "head", d.head, err, acct, 0, 1, true) ||
            !f32(d, "norm.weight", d.norm_w, err, acct, (size_t) g.dim))
            return false;
        d.loading_drafter = true;
        if (g.n_mtp > 0 && !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight", d.d_norm, err, acct, (size_t) g.dim))
            return false;
        d.loading_drafter = false;
    }
    d.loading_drafter = true;
    if (drafter_only && di == head_dev &&
        !f32(d, "mtp." + std::to_string(g.n_mtp - 1) + ".norm.weight",
             d.d_norm, err, acct, (size_t) g.dim))
        return false;
    if (di == head_dev) {
        head = d.head;
        norm_w = d.norm_w;
        d_norm = d.d_norm;
        if (g.n_mtp > 0) {
            const std::string Z = "mtp." + std::to_string(g.n_mtp - 1) + ".";
            if (!f32(d, Z + "confidence_head.proj.weight", d_conf, err, acct, (size_t) (g.dim + g.d_rank)))
                return false;
            const Tensor* te = pk.get(Z + "markov_head.embed.weight");
            const Tensor* th = pk.get(Z + "markov_head.head.weight");
            if (te == nullptr || th == nullptr || te->dtype != "BF16" || (th->dtype != "F16" && th->dtype != "BF16")) {
                err = Z + "markov_head: not the BF16 embed and F16 / BF16 head expected";
                return false;
            }
            if (!dalloc(d, d_mk_embed, te->bytes / 2, err, acct) || !upload(d, *te, d_mk_embed, err) ||
                (strata::ds4::markov_head_q8_default(defaults_026())
                     ? !mk_head_q8(d, Z + "markov_head.head.weight", d_mk_head_q8, err, acct)
                     : !f16raw(d, Z + "markov_head.head.weight", d_mk_head, err, acct)))   // (0.2.6: the head as F16)
                return false;
        }
    }
    if (g.n_mtp > 0 && di == d_dev) {
        if (!mat(d, "mtp.0.main_proj", d_main_proj, err, acct) ||
            !f32(d, "mtp.0.main_norm.weight", d_main_norm, err, acct, (size_t) g.dim))
            return false;
    }
    d.loading_drafter = false;
    if (!drafter_only)
        for (const auto& allocation : d.draft_allocations)
            d.drafter_bytes += allocation.second;
    return true;
}

void Model::Impl::set_vm(Layer& l, int e, const uint8_t* b) {
    if (l.lay.fmt == gg::kFmtMXFP4) {
        for (int r = 0; r < 3; ++r) {
            x3::Mat& m = l.vm[(size_t) e][(size_t) r];
            m = x3::Mat{};
            m.K = gg::expert_type(l.lay, r);   // (a low-bit type: its GEMV, ds4_iq's)
            m.fmt = m.K == gg::kTypeMXFP4 ? gg::kFmtMXFP4 : gg::kFmtGG;
            m.raw = b + l.lay.trellis[r];
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
        }
        return;
    }
    for (int r = 0; r < 3; ++r) {
        x3::Mat& m = l.vm[(size_t) e][(size_t) r];
        m.trellis = (const uint32_t*) (b + l.lay.trellis[r]);
        m.suh = (const uint16_t*) (b + l.lay.suh[r]);
        m.svh = (const uint16_t*) (b + l.lay.svh[r]);
        m.k = r < 2 ? g.dim : g.ff;
        m.n = r < 2 ? g.ff : g.dim;
        m.K = l.lay.K[r];
        m.cb = 2;
        m.ld = 0;
    }
}

std::vector<int> Model::Impl::get_heat_order(const Layer& l) const {
    std::vector<int> order((size_t) l.n_exp);
    std::iota(order.begin(), order.end(), 0);
    const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
    return order;
}

bool Model::Impl::place_paged(std::string& err) {
    size_t most = 0;
    for (Layer& l : L) {
        if (l.cpu_only) {
            err = "experts in page pools need GPU experts and no tensor parallelism";
            return false;
        }
        if (l.lay.fmt != gg::kFmtMXFP4) {
            for (int r = 0; r < 3; ++r) {
                l.hl.k[r] = r < 2 ? g.dim : g.ff;
                l.hl.n[r] = r < 2 ? g.ff : g.dim;
                l.hl.trellis[r] = l.lay.trellis[r];
                l.hl.suh[r] = l.lay.suh[r];
                l.hl.svh[r] = l.lay.svh[r];
            }
            l.hl.bytes = l.blob;
        }
        l.by_heat = get_heat_order(l);
        l.slot.assign((size_t) l.n_exp, -1);
        l.vm.assign((size_t) l.n_exp, {});
        l.page_at.assign((size_t) l.n_exp, 0);
        l.n_vram = l.n_slots = 0;
        if (!dalloc(devs[(size_t) l.dev], l.slot_tab, (size_t) l.n_exp, err))
            return false;
        most = std::max(most, (size_t) l.n_exp);
    }
    held.assign(most, nullptr);
    return true;
}

bool Model::Impl::set_expert_addresses(const std::vector<const uint8_t*>& at, std::string& err, int stage) {
    for (Layer& l : L) {
        if (l.slot_tab == nullptr || (stage >= 0 && l.dev != stage))
            continue;

        const uint8_t* const* a = at.data() + (size_t) l.il * g.n_exp;
        if (std::equal(l.page_at.begin(), l.page_at.end(), a,
                       [](unsigned long long was, const uint8_t* now) { return was == (unsigned long long) now; }))
            continue;
        std::vector<unsigned long long> slots;
        for (int e : l.by_heat) {
            l.page_at[(size_t) e] = (unsigned long long) a[e];
            l.slot[(size_t) e] = a[e] != nullptr ? (int) slots.size() : -1;
            if (a[e] != nullptr) {
                slots.push_back((unsigned long long) a[e]);
                set_vm(l, e, a[e]);
            }
        }
        l.pv_n = l.n_vram = l.n_slots = (int) slots.size();
        cudaSetDevice(devs[(size_t) l.dev].id);
        if (cudaMemcpy(l.vram_tab, l.page_at.data(), l.page_at.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(l.slot_tab, slots.data(), slots.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "writing layer " + std::to_string(l.il) + "'s expert tables failed";
            return false;
        }
    }
    return true;
}

std::vector<ExpertFileLayout> Model::Impl::get_expert_files(size_t n) const {
    std::vector<ExpertFileLayout> geo(n);
    for (size_t il = 0; il < n; ++il) {
        const Layer& l = L[il];
        ExpertFileLayout& gl = geo[il];
        gl.n_exp = l.n_exp;
        if (l.lay.fmt != gg::kFmtMXFP4) {
            gl.pieces.resize((size_t) l.n_exp);
            for (int e = 0; e < l.n_exp; ++e)
                for (int r = 0; r < 3; ++r) {
                    const size_t dst[] = {l.lay.trellis[r], l.lay.suh[r], l.lay.svh[r]};
                    for (int q = 0; q < 3; ++q)
                        gl.pieces[(size_t) e].push_back({l.region_file[(size_t) e],
                            l.region_off[(size_t) e] + l.piece_src[r][q], l.piece_bytes[r][q], dst[q]});
                }
        }
        for (int r = 0; r < 3 && gl.pieces.empty(); ++r) {
            gl.file[r] = l.mx_file[r];
            gl.off[r] = l.mx_off[r];
            gl.stride[r] = l.mx_stride[r];
            gl.bytes[r] = mx_part_bytes(l, r);
            gl.tr[r] = l.lay.trellis[r];
        }
        gl.blob = l.blob;
    }
    return geo;
}

std::vector<Model::Impl::LaneCfg> Model::Impl::lane_cfgs() const {
    std::vector<std::vector<platform::Core>> parts = core_threads;
    const bool one_over_all = parts.size() > 1;
    if (one_over_all)
        for (size_t Y = 1; Y < parts.size(); ++Y) parts[0].insert(parts[0].end(), parts[Y].begin(), parts[Y].end());
    parts.resize((size_t) std::max(1, std::min((int) parts.size(), std::max(NH, 1))));
    // a configuration from each part's cores that `keep` takes: their first CPUs, then (smt) their second ones, ..
    auto make = [&](const std::string& name, bool smt, const std::function<bool(const platform::Core&)>& keep) {
        LaneCfg c;
        c.name = name;
        for (const auto& p : parts) {
            std::vector<int> v;
            for (size_t k = 0; k < (smt ? 4u : 1u); ++k)
                for (const platform::Core& co : p)
                    if (keep(co) && k < co.cpus.size()) v.push_back(co.cpus[k]);
            c.cpus.push_back(v);
        }
        return c;
    };
    std::vector<LaneCfg> out = {make("cores", false, [](const platform::Core&) { return true; })};
    bool smt = false, mixed = false;
    int top = 0;
    for (const auto& p : parts)
        for (const platform::Core& co : p) {
            smt = smt || co.cpus.size() > 1;
            top = std::max(top, co.eff);
        }
    for (const auto& p : parts)
        for (const platform::Core& co : p) mixed = mixed || co.eff != top;
    if (smt) out.push_back(make("smt", true, [](const platform::Core&) { return true; }));
    if (mixed) out.push_back(make("pcores", false, [top](const platform::Core& co) { return co.eff == top; }));
    if (one_over_all && !core_threads.empty()) {
        LaneCfg c;
        c.name = "node";
        c.cpus.emplace_back();
        for (const platform::Core& co : core_threads[0]) c.cpus[0].push_back(co.cpus[0]);
        out.push_back(c);
    }
    return out;
}

std::vector<std::unique_ptr<Pool>> Model::Impl::pools_for(const LaneCfg& c) const {
    std::vector<std::unique_ptr<Pool>> out;
    const int svc = !cores.empty() && !cores[0].empty() ? cores[0][0] : -1;
    for (size_t X = 0; X < c.cpus.size(); ++X) {
        std::vector<int> v = c.cpus[X];
        if (X == 0) v.erase(std::remove(v.begin(), v.end(), svc), v.end());   // (the service thread's)
        // cores of the first pool left free beside the host thread's, for the thread that launches the decode's kernels
        // (and waits on the device) and whatever else the PC runs: with a worker spinning on every other logical CPU
        // that thread waits for a core (Project Maya #359, a 20-core Windows 11 PC: decode 28.3 -> 55.5 tok/s).  Only
        // when the lane would take every logical CPU and --cpu-threads is not given.
        // STRATA_DS4_LANE_FREE: the cores left free (2 by default, 0: as before)
        {
            static const int lane_free = [] {
                const char* e = std::getenv("STRATA_DS4_LANE_FREE");
                return e != nullptr ? std::max(0, std::atoi(e)) : 2;
            }();
            size_t lane_cpus = 0;
            for (const auto& cv : c.cpus) lane_cpus += cv.size();
            const bool lane_takes_all = lane_cpus >= (size_t) std::max(1u, std::thread::hardware_concurrency());
            if (X == 0 && o.cpu_threads <= 0 && lane_free > 0 && (int) v.size() > lane_free + 1 && lane_takes_all)
                v.resize(v.size() - (size_t) lane_free);
        }
        int nw = (int) v.size();
        if (o.cpu_threads > 0) nw = std::max(X == 0 ? 0 : 1, std::min(nw, o.cpu_threads / NH - (X == 0 ? 1 : 0)));
        out.push_back(std::make_unique<Pool>(nw, v, 20000));   // spin through a token's layers (20 ms)
    }
    return out;
}

void Model::Impl::make_pools() {
    const std::vector<LaneCfg> cfgs = lane_cfgs();
    size_t pick = 0;
    if (const char* v = std::getenv("STRATA_DS4_CPU_LANE"))
        for (size_t i = 0; i < cfgs.size(); ++i)
            if (cfgs[i].name == v) pick = i;
    pools = pools_for(cfgs[pick]);
    (void) gg::mx_tier();   // (measured here, not on the first token)
}

bool Model::set_stage_layers(int stage, int lb, int le, std::string& err) {
    Impl& p = *p_;
    for (int il = lb; il < le; ++il) {
        Layer& layer = p.L[(size_t) il];
        if (layer.dev == stage)
            continue;
        Dev& old = p.devs[(size_t) layer.dev];
        Dev& target = p.devs[(size_t) stage];
        std::vector<void*> allocations = layer.dense_allocations;
        if (layer.slot_tab != nullptr)
            allocations.push_back(layer.slot_tab);
        if (layer.vs_dev != nullptr)
            allocations.push_back(layer.vs_dev);
        const size_t layer_scratch = layer.dense_scratch_bytes +
            (layer.slot_tab != nullptr ? align256((size_t) layer.n_exp * 8) : 0) +
            (layer.vs_dev != nullptr ? align256((size_t) layer.n_exp * 4) : 0);
        if (old.id != target.id) {
            // The conversations are paused; reload before releasing the old copy so allocation failure is reversible.
            const Layer saved = layer;
            const size_t first = target.allocs.size(), weights = target.weights, scratch = target.scratch;
            layer.dev = stage;
            layer.grp = stage;
            layer.ring_bak = nullptr;
            layer.c_snap = layer.c_rows = layer.joint_c_snap = nullptr;
            bool ok = p.load_dense(stage, err, false, il);
            if (ok && saved.ring_bak != nullptr)
                ok = p.init_verify_layer(layer, err, saved.joint_c_snap != nullptr);
            if (ok && saved.slot_tab != nullptr)
                ok = p.dalloc(target, layer.slot_tab, (size_t) layer.n_exp, err);
            if (ok && saved.vs_dev != nullptr)
                ok = p.dalloc(target, layer.vs_dev, (size_t) layer.n_exp, err);
            if (!ok) {
                cudaSetDevice(target.id);
                for (size_t i = first; i < target.allocs.size(); ++i)
                    strata::core::release_device(target.allocs[i]);
                target.allocs.resize(first);
                target.weights = weights;
                target.scratch = scratch;
                layer = saved;
                return false;
            }
            cudaSetDevice(old.id);
            for (void* ptr : allocations) {
                strata::core::release_device(ptr);
                old.allocs.erase(std::remove(old.allocs.begin(), old.allocs.end(), ptr), old.allocs.end());
            }
            old.weights -= saved.weight_bytes;
            old.scratch -= layer_scratch;
            // Expert addresses must be republished even when the new owner's page numbers happen to match.
            std::fill(layer.page_at.begin(), layer.page_at.end(), ~0ull);
            std::fprintf(stderr, "strata-ds41: dense layer %d migrated: gpu%d -> gpu%d (%zu bytes)\n",
                         il, old.id, target.id, layer.weight_bytes);
        } else {
            old.weights -= layer.weight_bytes;
            target.weights += layer.weight_bytes;
            for (void* ptr : allocations) {
                old.allocs.erase(std::remove(old.allocs.begin(), old.allocs.end(), ptr), old.allocs.end());
                target.allocs.push_back(ptr);
            }
            old.scratch -= layer_scratch;
            target.scratch += layer_scratch;
            layer.dev = stage;
            layer.grp = stage;
        }
    }
    return true;
}

bool Model::attach_expert_hop(int stage, std::function<const uint8_t*(int, int)> find, std::string& err) {
    Impl& p = *p_;
    p.hops.resize(p.devs.size());
    auto hop = std::make_unique<ExpertHop>();
    Dev& d = hop->d;
    d.id = p.devs[(size_t) stage].id;
    const strata::core::OnDevice on(d.id);
    CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
    const size_t rows = kWin * kPicks;
    if (!p.dalloc(d, d.xe, rows * p.g.dim, err) || !p.dalloc(d, d.xg, rows * p.g.ff, err) ||
        !p.dalloc(d, d.xu, rows * p.g.ff, err) || !p.dalloc(d, d.xhh, rows * p.g.ff, err) ||
        !p.dalloc(d, d.ed, rows * p.g.dim, err))
        return false;
    hop->find = std::move(find);
    p.hops[(size_t) stage] = std::move(hop);
    return true;
}

}  // namespace strata::ds4
