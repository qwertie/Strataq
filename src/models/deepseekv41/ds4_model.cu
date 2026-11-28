#include "strata/base/gpu/managed_memory.hpp"
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.cu - DeepSeek-V4.1-Flash on Maya (see ds4_model.hpp).
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
#include "strata/common/kernels/routing_stats.hpp"
#include "strata/common/state/restore_points.hpp"
#include "strata/common/artifact/tensor_directory.hpp"
#include "ds4_ring_link.hpp"
#include "ds4_launch.h"   // strata_ds4_launch
#include "ds4_model.hpp"
#include "ds4_pick_rule.hpp"
#include "ds4_experts_rule.hpp"
#include "ds4_engram_io.hpp"

#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include "ds4_gguf_cpu.hpp"
#include "ds4_kernels.cuh"
#include "strata/models/deepseekv41/geometry.hpp"
#include "strata/base/gpu/gpu_set.hpp"
#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/env.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/traffic.hpp"
#include "strata/base/platform/os.hpp"
#include "strata/common/experts/expert_profile.hpp"
#include "strata/core/expert_source.hpp"
#include "ds4_pool.hpp"
#include "ds4_vision.cuh"
#include "strata/models/deepseekv41/exl3_cpu.hpp"
#include "strata/models/deepseekv41/exl3.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/experts/tail_skip.hpp"
#include "strata/common/kernels/f16_bits.hpp"
#include "strata/models/deepseekv41/moe_win.hpp"
#include "strata/models/deepseekv41/sampler.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/common/kernels/prefill/wmma_gemm.h"   // strata_pf_gemm_f16_ld (gemm() prefers it over hipBLAS on gfx11)
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif

namespace strata::ds4 {
namespace x3 = strata::dsv41k::exl3;
namespace kc = strata::dsv41k::cpu;
namespace gf = strata::dsv41k::glmf;
namespace gg = strata::ds4::gguf;
using strata::core::kMarksMax;
using strata::core::latest_point;
using strata::core::mark_victim;

#define CK(call)                                                                            \
    do {                                                                                    \
        cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) {                                                            \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                       \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

namespace {

using strata::platform::env_tri;

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// the embedding rows' upload: asynchronous on the device's stream
inline cudaError_t emb_upload(void* dst, const void* src, size_t n, cudaStream_t s) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
}

size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// A run of chunks' device-side waits for another group (xfer_get, wait_flags: flag >= seq) are queued only once the
// host sees the flag up, so the kernel finds it up and never spins.  A wait kernel spinning on a device while its
// thread goes on queueing the layer stalled that thread's first cuBLAS call behind it, and with it the other
// devices' (two H100s, the IQ2_XS pack on a layer split: both threads stopped in cublasSgemm, hc_mix_chunk).
void host_wait_at_least(const unsigned* flag, unsigned seq) {
    for (int spin = 0; (int) (*(volatile const unsigned*) flag - seq) < 0; ++spin) {
        if (spin < 1000) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

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
void pin_self(int cpu) { platform::pin_self(cpu); }
// the RAM this process can still get (~0: not known), cgroup limits included
uint64_t ram_avail() {
    strata::core::detail::HostMemory m;
    const bool known = strata::core::detail::host_available_memory(m);
    // Windows supplies commit capacity on a successful read even when no physical RAM remains.
    return known || m.commit != ~uint64_t{0} ? m.available : ~uint64_t{0};
}

#if !defined(_WIN32) && !defined(STRATA_USE_HIP)   // (uma_free's only)
// whether GPU `dev` shares the PC's memory (a Spark's GB10, a Strix Halo, an APU); false when the runtime cannot say
bool gpu_integrated(int dev) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return v != 0;
}
#endif
// the device memory the experts' placement may count on: cudaMemGetInfo's free, but on an integrated NVIDIA GPU on Linux
// (a DGX Spark's GB10, one memory with the CPU) get_uma_free_bytes's view of ram_avail: cudaMemGetInfo counts only
// MemFree there, and the model's files just read fill the page cache (measured on a GB10: 55 GB "free" with 115 GB
// available).  Not on HIP: a Strix Halo's GPU memory is its carve-out plus the GTT pool, which hipMemGetInfo already
// counts and MemAvailable may exceed
size_t uma_free(int dev, size_t fr) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    const uint64_t avail = gpu_integrated(dev) ? ram_avail() : ~uint64_t{0};
    if (avail != ~uint64_t{0})
        return strata::core::detail::get_uma_free_bytes(fr, avail);
#else
    (void) dev;
#endif
    return fr;
}

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

// ------------------------------------------------------------------ the pack
using Tensor = artifact::TensorEntry;

struct Pack {
    std::string dir, model_dir;
    std::unordered_map<std::string, std::string> kv;
    std::vector<std::string> files;
    std::vector<platform::fd_t> fds;
    std::unordered_map<std::string, Tensor> t;
    int eg_vocab = 0, eg_layers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    // format gguf (tools/ds4/pack_gguf.py): the files are a GGUF's shards, every one mapped at load (map())
    bool gguf = false;

    std::vector<std::pair<uint8_t*, size_t>> maps;   // file -> its read-only mapping (map(): the experts in place)
    std::vector<platform::fd_t> dfds;   // (gguf) file -> unbuffered (open_direct; kBadFd: buffered reads only)
    std::vector<platform::fd_t> pfds;   // (gguf) file -> for reads from many threads at once (platform::open_par)

    ~Pack() {
        for (auto& m : maps) platform::unmap(m.first, m.second);
        for (platform::fd_t fd : fds) platform::close_fd(fd);
        for (platform::fd_t fd : dfds) platform::close_fd(fd);
        for (platform::fd_t fd : pfds) platform::close_fd(fd);
    }
    // file i mapped whole, read-only (once; not thread-safe: called while loading)
    const uint8_t* map(int file) {
        if (maps.size() < fds.size()) maps.resize(fds.size(), {nullptr, 0});
        auto& m = maps[(size_t) file];
        if (m.first == nullptr) {
            const uint64_t size = platform::file_size(fds[(size_t) file]);
            void* p = platform::map_ro(fds[(size_t) file], size);
            if (p == nullptr) return nullptr;
            m = {(uint8_t*) p, (size_t) size};
        }
        return m.first;
    }

    bool load(const std::string& d, std::string& err) {
        dir = d;
        {
            std::ifstream f(d + "/ds4.txt");
            if (!f) {
                err = d + "/ds4.txt: not found (run tools/ds4/pack.py --model <model>)";
                return false;
            }
            std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp == std::string::npos) continue;
                kv[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir = str("model_dir");
        artifact::TensorDirectory toc;
        if (!toc.load(d + "/tensors.txt", err))
            return false;
        files = std::move(toc.files);
        t = std::move(toc.tensors);
        for (const std::string& fn : files) {
            const std::string p = model_dir + "/" + fn;
            const platform::fd_t fd = platform::open_ro(p);
            if (fd == platform::kBadFd) {
                err = p + ": cannot open";
                return false;
            }
            fds.push_back(fd);
        }
        gguf = str("format") == "gguf";
        if (gguf)
            for (size_t i = 0; i < fds.size(); ++i) {
                if (map((int) i) == nullptr) {
                    err = model_dir + "/" + files[i] + ": cannot map";
                    return false;
                }
                dfds.push_back(platform::open_direct(model_dir + "/" + files[i]));
                pfds.push_back(platform::open_par(model_dir + "/" + files[i]));
            }
        // the Engram tables
        std::FILE* e = std::fopen((d + "/engram.bin").c_str(), "rb");
        if (e == nullptr) {
            err = d + "/engram.bin: not found";
            return false;
        }
        char magic[4];
        int32_t hdr[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 &&
                  std::fread(hdr, 4, 6, e) == 6;
        if (ok) {
            eg_vocab = hdr[0], eg_layers = hdr[1], eg_ngram = hdr[2], eg_heads = hdr[3], eg_hd = hdr[4], eg_pad = hdr[5];
            const int cols = (eg_ngram - 1) * eg_heads;
            eg_map.resize((size_t) eg_vocab);
            eg_mult.resize((size_t) eg_layers * eg_ngram);
            eg_primes.resize((size_t) eg_layers * cols);
            eg_off.resize((size_t) eg_layers * cols);
            ok = std::fread(eg_map.data(), 4, eg_map.size(), e) == eg_map.size() &&
                 std::fread(eg_mult.data(), 8, eg_mult.size(), e) == eg_mult.size() &&
                 std::fread(eg_primes.data(), 8, eg_primes.size(), e) == eg_primes.size() &&
                 std::fread(eg_off.data(), 8, eg_off.size(), e) == eg_off.size();
        }
        std::fclose(e);
        if (!ok) {
            err = d + "/engram.bin: unreadable";
            return false;
        }
        return true;
    }

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v;
        std::stringstream ss(str(k));
        std::string x;
        while (std::getline(ss, x, ','))
            if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    const Tensor* get(const std::string& n) const {
        auto it = t.find(n);
        return it == t.end() ? nullptr : &it->second;
    }
    bool read_at(int file, uint64_t off, uint64_t n, void* dst) const {   // (any thread; side by side where pfds is)
        if ((size_t) file < pfds.size() && pfds[(size_t) file] != platform::kBadFd)
            return platform::pread_par(pfds[(size_t) file], dst, n, off);
        return platform::pread_full(fds[(size_t) file], dst, n, off);
    }
    bool read(const Tensor& x, void* dst) const { return read_at(x.file, x.off, x.bytes, dst); }
};

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

// ------------------------------------------------------------------ devices, layers
struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    size_t free0 = 0, total = 0, weights = 0, scratch = 0;
    std::vector<void*> allocs;
    /// Reloadable DSpark tensors; null addresses retain their size while unloaded.
    std::vector<std::pair<void*, size_t>> draft_allocations;
    bool loading_drafter = false; ///< Tags dense allocations while this device loads DSpark.
    uint64_t drafter_bytes = 0; ///< Full reload size, preserved after a failed reload.
    // a window's scratch, rows of up to kWin tokens (the routed experts' rows: kWin * kPicks)
    float *h = nullptr, *h2 = nullptr, *x = nullptr, *qr = nullptr, *q = nullptr, *kvr = nullptr, *ckr = nullptr,
          *cgr = nullptr, *latent = nullptr, *ikr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr,
          *bm = nullptr, *o = nullptr, *oa = nullptr, *ao = nullptr, *mix = nullptr, *a_pre = nullptr,
          *a_post = nullptr, *a_comb = nullptr, *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr,
          *glog = nullptr, *rw = nullptr, *moe_out = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *sh_out = nullptr, *cpu_part = nullptr, *eg_x = nullptr, *eg_kv = nullptr, *logits = nullptr,
          *freqs_win = nullptr, *freqs_comp = nullptr, *xe = nullptr, *xg = nullptr, *xu = nullptr, *xhh = nullptr,
          *ed = nullptr;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *tok = nullptr;
    unsigned long long* plan_ptr = nullptr;           // the decode path's VRAM experts (0: on the CPU lane)
    uint8_t* eg_rows = nullptr;
    uint16_t *emb_rows = nullptr, *ext = nullptr;   // the window's embedding rows; the window attention's rows
    AttnScratch attn_scr;   // a few queries' selected compressed rows decoded (sparse_attn)
    // the hand-off to the next device through pinned host memory (these cards copy peer to peer only through the
    // driver's staging, ~1.7 ms a copy): h | pre | topk | cand
    uint8_t* bounce = nullptr;
    cudaEvent_t ev_out = nullptr;
    float* tap = nullptr;   // a window's stream means before a DSpark target layer
    // threads: its group and its index in it, xfer_put's block counter, the hand-off to the next group (host-mapped:
    // h | pre | topk | cand, then its flag), the window MoE's grouping scratch
    int grp = 0, mi = 0;
    unsigned* xcnt = nullptr;
    unsigned* hccnt = nullptr;   // [kWin] hc_mix_split's per-token block counters (zeroed at allocation, left zero)
    // the head and the norms (the head's device: the last layer's)
    x3::Mat head;
    float *norm_w = nullptr, *d_norm = nullptr;
    uint8_t* hand = nullptr;
    unsigned* hand_flag = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int *grp_start = nullptr, *grp_ent = nullptr, *n_groups = nullptr;
};

/// Keeps DeepSeek hop launches independent of a holder's prompt and decoding scratch.
struct ExpertHop {
    Dev d;
    std::function<const uint8_t*(int, int)> find;
    int64_t experts = 0, rows = 0;
    double ms = 0;
    ~ExpertHop() {
        cudaSetDevice(d.id);
        if (d.st != nullptr)
            cudaStreamSynchronize(d.st);
        for (void* ptr : d.allocs)
            strata::core::release_device(ptr);
        if (d.st != nullptr)
            cudaStreamDestroy(d.st);
    }
};

// the chunked prompt path's buffers on a device: a chunk of C tokens through a layer at once (the attention in slabs of
// A queries, the indexer in slabs of AI), its GEMMs through FP16 reconstructions of the weights and cuBLAS
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

struct Chunk {
    int C = 0, A = 0, AI = 0;
    int HQ = 0, OG = 0, FF = 0;              // (lay_chunk: this device's heads' width, wo_a groups, FFN share)
    bool has_first = false, has_eg = false;
    float *h = nullptr, *x = nullptr, *acc = nullptr, *qr = nullptr, *kvr = nullptr, *ckr = nullptr, *cgr = nullptr,
          *latent = nullptr, *ikr = nullptr, *q = nullptr, *o = nullptr, *oa = nullptr, *iq = nullptr, *iw = nullptr,
          *scores = nullptr, *bm = nullptr, *mix = nullptr, *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr,
          *f_pre = nullptr, *f_post = nullptr, *f_comb = nullptr, *pre = nullptr, *glog = nullptr, *rw = nullptr,
          *sg = nullptr, *su = nullptr, *y = nullptr, *eg_x = nullptr, *eg_kv = nullptr;
    uint16_t *x16 = nullptr, *qr16 = nullptr, *o16 = nullptr, *oa16 = nullptr, *lat16 = nullptr, *sh16 = nullptr,
             *h16 = nullptr, *eg_x16 = nullptr, *ext = nullptr, *w16 = nullptr, *emb = nullptr;
    size_t w16_elems = 0;
    int *ridx = nullptr, *topk = nullptr, *cand = nullptr, *bounds = nullptr, *row_tok = nullptr, *pos = nullptr;
    uint8_t* eg_rows = nullptr;
    int* bounds_pin = nullptr;   // pinned: the sort's bounds, read back
    // the CPU lane's experts streamed in: ring_n VRAM slots of ring_stride bytes in kParts parts, a batch a part,
    // kParts batches in flight on the copy stream (filled / freed: each part's last copy, its last use)
    static constexpr int kParts = 4;
    uint8_t* ring = nullptr;
    size_t ring_stride = 0;
    int ring_n = 0;
    cudaStream_t cs = nullptr;
    cudaEvent_t filled[kParts] = {}, freed[kParts] = {};
    // streamed experts: each streamed expert's file region read into pinned staging (ring_n slots of
    // stage_stride bytes, a part's slots free again once its `filled` copies are done) by the readers, in parallel,
    // and DMA'd from there - a copy from the files' pages goes through the driver's staging, one thread at ~1.3 GB/s
    uint8_t* stage = nullptr;
    size_t stage_stride = 0;
    std::unique_ptr<Pool> readers;
    cublasHandle_t blas = nullptr;
    // the busy experts' path (run_set, DS4_FAT_ROWS): one expert's weights as FP16 (gate and up, then down), the rows
    // gathered for its GEMM, the GEMM's gate | up output - fat_r rows a piece; allocated at the first use
    uint16_t *fat_w = nullptr, *fat_x = nullptr;
    float* fat_gu = nullptr;
    int fat_r = 0;
    bool fat_off = false;
    // host-mapped: a chunk's hand-off to the next group (h | pre | topk | cand), two in flight (slot = chunk % 2), each
    // with its flag, and the next group's members' acknowledgements [slot][member] (32 apart)
    uint8_t* hand[2] = {nullptr, nullptr};
    unsigned* hand_flag[2] = {nullptr, nullptr};
    unsigned* hand_ack = nullptr;
    // pinned, two chunks in flight: the embedding rows (the first layer's device), the Engram rows (an Engram layer's)
    uint16_t* emb_host[2] = {nullptr, nullptr};
    uint8_t* eg_host[2] = {nullptr, nullptr};
    cudaEvent_t emb_ev[2] = {nullptr, nullptr}, eg_ev[2] = {nullptr, nullptr};
    double stream_bytes = 0;     // the experts streamed in, in all
    // the flow (a GGUF's in-place experts, every layer of the device on it alone; DS4_FLOW=0: the batches above): a
    // (chunk, layer)'s streamed experts are those it routes to - its likely ones (the profile's chance of a pick by the
    // chunk's tokens) queued once the layer before it routed, read while that one computes, the rest of its picks once
    // it routed itself; never an expert no token of the chunk picked unless it was a likely one.  Each item read by io
    // (unbuffered: the disk straight into its staging slot, every part of every item in flight at once - none waits
    // for a batch) or taken from the caller's pool as it lies, then DMA'd into the ring in order, item i into slot
    // i % ring_n; the compute takes the items as they land, in runs of what has.
    struct Item {
        int il = 0, k = 0, e = 0, v = 0, stage = -1;   // layer, chunk, expert, virtual slot, staging slot (-1: RAM)
        int mi = 0;                                    // the device's member index in the layer's group
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};   // where its gate / up / down (its parts) lie once read
        const uint8_t* ram = nullptr;                  // its blob in the caller's pool, copied into staging
        uint8_t* keep = nullptr;                       // (layer-major) its layer-cache slot: what is read kept there
        int64_t after = -1;                            // (layer-major) the items up to this one read first
        std::atomic<int> left{0};                      // parts still being read
        std::atomic<bool> bad{false};
    };
    bool flow = false;
    int dev = -1;                            // the device (index into devs) this chunk state belongs to
    std::unique_ptr<IoQueue> io;
    std::deque<Item> items;                  // items[0] is item `base`; the indices below are absolute
    size_t base = 0, n_read = 0, n_dma = 0, n_comp = 0;
    uint64_t n_file = 0;                     // file items so far (staging slot = ordinal % S)
    int S = 0;                               // staging slots, stage_stride apart
    std::vector<int64_t> stage_item;         // staging slot -> the item last read into it (-1: none)
    std::vector<cudaEvent_t> stage_ev, ring_fill, ring_free;   // a staging slot's DMA out; a ring slot's DMA in, its use
    int spec_k = -1, spec_il = -1, n_spec = 0;   // the likely items queued last: (chunk, layer), how many, from where
    size_t spec_begin = 0;
    std::vector<int> fl_layers, fl_T;        // the device's trunk layers in order; the run's chunk sizes
    int fl_yoco = -1, fl_cut_k = 0, fl_cut_r = 0;   // Run::yoco / cut_k / cut_r: what a layer's chunk streams for
    int bc_n = 0, bc_w = 0;                  // a batch's compacted bounds: bc_n regions of bc_w, by batch count
    int *bc_pin = nullptr, *bc_dev = nullptr;
    uint64_t n_batch = 0;
    double wait_ms = 0;                      // the compute waiting for a read
    // (DS4_PROF: where copy_ahead's time goes - the host staging copy, the DMA issues, the waits for a slot)
    double dbg_stage_ms = 0, dbg_dma_ms = 0, dbg_evt_ms = 0;
    uint64_t it_file = 0, it_ram = 0, it_spec = 0, it_spec_unused = 0;
    uint64_t io_d0 = 0, io_b0 = 0;           // io's unbuffered / buffered bytes when the run started
    double io_busy0 = 0;                     // io's busy time when the run started (DS4_PROF)
    bool buffered = false;                   // the run's reads all through the page cache (a short run: run_dev_chunks)
    // layer-major (a run of several chunks, every layer on this device): the layers one after another, each through
    // all the chunks; a chunk's state between its layers (h | pre | topk | cand, chunk k at k chunk_hand_bytes()) in
    // host memory, and the layer cache - RAM slots where the experts a layer's chunk reads from the files are kept for
    // its later chunks, so each is read once a run
    bool lm = false, lm_pinned = false;
    uint8_t* lm_state = nullptr;
    size_t lm_state_bytes = 0;
    uint8_t* lc = nullptr;                   // lc_bytes of slots, a layer's experts' blobs side by side
    size_t lc_bytes = 0;
    int lc_il = -1, lc_used = 0, lc_n = 0;   // the layer the cache holds, its slots taken, its slots in all
    size_t lc_first = 0, lc_chk = 0;         // the layer's first item; the items before lc_chk all read
    std::vector<int> lc_of;                  // expert -> its slot (-1: none)
    std::vector<int64_t> lc_writer;          // slot -> the item reading its expert into it
    uint64_t it_lc = 0;                      // items taken from the layer cache
};

// the states a run of chunks keeps on the way, at most: the checkpoint before its last row (feed_mark)
constexpr int kSnapN = 1;

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

struct Layer {
    int il = 0, dev = 0, ratio = 0;
    // its group (a group's members are copies of the device state in mates, member mi's share of the heads, of wo_a's
    // groups, of wo_b's rows and of the FFNs; every group the engine builds has one member, nm 1, mates empty)
    int grp = 0, nm = 1, mi = 0;
    std::vector<Layer> mates;
    int ff0 = 0, ffn = 0;     // this member's FFN columns [ff0, ff0 + ffn) (split128 of n_ff; the whole one alone)
    // this member's heads [ah0, ah0 + ahn) and wo_a groups [og0, og0 + ogn) - all of them alone; in a group of more
    // members than the heads split among evenly (head_members: 9, 7 or 6 over 64 heads), the first ones none: their
    // attention adds nothing to the sum
    int ah0 = 0, ahn = 0, og0 = 0, ogn = 0;
    // the chunked prompt path: each expert's virtual slot (its VRAM slot, or n_slots + its place among the CPU lane's
    // experts, which a chunk streams in that order) - pinned on member 0, a device copy on each member
    int* vs_pin = nullptr;
    int* vs_dev = nullptr;
    std::vector<int> host_list;
    std::string name;         // "layers.N." or, for the DSpark drafter's blocks, "mtp.N."
    int n_exp = 0, k_exp = 0; // routed experts, and how many a token takes (the drafter's: 128, 3)
    /// Holds the trained expert count so route() can preserve each pick's share when --experts lowers k_exp.
    int k_model = 0;
    std::vector<void*> dense_allocations; ///< Records tensors and verify scratch reloaded on a cut move.
    size_t dense_scratch_bytes = 0; ///< Reloadable scratch accounted separately from tensors.
    size_t weight_bytes = 0;  ///< Dense allocations accounted to the stage that currently runs this layer.
    bool mtp = false;         // a DSpark block (index stage = il - n_layers)
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1;      // the kv source whose compressed KV this layer reads (index into Impl::srcs)
    int ik_src = -1;   // the kv source whose index keys its indexer reads
    int eg = -1;       // engram layer index, or -1
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_bias = nullptr, *eg_w = nullptr,
          *gate_bias_vl = nullptr;   // (the image tokens' routing bias, with the vision tower)
    uint16_t* i_wproj = nullptr;
    float* gate_w = nullptr;   // the router's weights, F32 (DeepSeek's inference computes the gate in F32)
    uint8_t* win = nullptr;                     // [win] rows of kKvFp8 (kv_row_bytes)
    float *c_kv = nullptr, *c_score = nullptr;  // the compressor's partial group (ratio > 1)
    // a run of chunks' states on the way (feed_mark): the ring and the partial group as they were before a row, a set
    // for each
    uint8_t* snap_win[kSnapN] = {};
    float *snap_kv[kSnapN] = {}, *snap_sc[kSnapN] = {};
    // routed experts
    x3::ExpertLayout lay;
    size_t blob = 0;                            // bytes of one expert's blob
    int n_vram = 0;
    bool cpu_only = false;   // a role's type has no device kernel: every expert of this layer is the CPU lane's
    std::vector<int> slot;                      // expert -> VRAM slot or -1
    // the CPU lane keeps an expert as NH column parts, part X on NUMA node X: gate / up columns [X n_ff / NH, ..),
    // down columns [X dim / NH, ..), each a standalone EXL3 matrix (its own trellis, the full suh, its part of svh)
    struct Half {
        size_t trellis[3], suh[3], svh[3], bytes;
        int k[3], n[3];
    } hl;
    // the VRAM slots: n_slots; pv_n: those a run of chunks reads as VRAM ones (all of them), the virtual slots past it
    // the CPU lane's experts it streams in
    int n_slots = 0, pv_n = 0;
    // a speculative window's backups (trunk layers): the ring rows it replaced, and a compressor's state before it and
    // the window's raw rows (kv, then score) to replay the accepted ones
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
    /// Keeps compressor snapshots for each conversation's window in a shared pass.
    float* joint_c_snap = nullptr;
    unsigned long long* vram_tab = nullptr;     // device [n_exp]
    // (Options::expert_pages, set_expert_addresses') the experts hottest first by the profile at load, the order of
    // the VRAM slots; device [n_exp]: each slot's blob, a page-pool page, which the prompt's expert sets read; each
    // expert's page as last given (0: not on the GPU)
    std::vector<int> by_heat;
    unsigned long long* slot_tab = nullptr;
    std::vector<unsigned long long> page_at;
    std::vector<std::array<x3::Mat, 3>> vm;     // expert -> its gate, up, down as EXL3 views of its VRAM blob
    std::vector<uint64_t> region_off;           // expert -> file region (offset, bytes, file)
    std::vector<uint64_t> region_bytes;
    std::vector<int> region_file;
    size_t piece_src[3][3];                     // part (gate, up, down) x (trellis, suh, svh): offset in the region
    size_t piece_bytes[3][3];
    size_t dense_bytes = 0;
    // a GGUF's MXFP4 experts (lay.fmt == gg::kFmtMXFP4): matrix r (gate, up, down) of expert e lies at mx_off[r] +
    // e mx_stride[r] in file mx_file[r] - loaded from there, and read in place (the file's mapping) by the CPU lane
    int mx_file[3] = {-1, -1, -1};
    uint64_t mx_off[3] = {0, 0, 0}, mx_stride[3] = {0, 0, 0};
};

struct Src {
    int layer = 0, ratio = 1;
    int64_t cap = 0;
    // per device (empty where no layer reads it): rows of kKvFp4 (the compressed KV), kKvFp4p (the index keys)
    std::vector<KvRows> ckv, ik;
    uint8_t *h_ckv = nullptr, *h_ik = nullptr;   // pinned: a window's new rows, on their way to the other devices
    cudaEvent_t ev_ckv = nullptr, ev_ik = nullptr;
    // threads: whether devices outside the source's group read it, and the host-mapped hand-off of a window's new rows
    // (ckv rows, then ik rows at kWin * hd) and its flag
    bool remote = false;
    uint8_t* x_stage = nullptr;
    unsigned* x_flag = nullptr;
    // pipelined chunks: a second staging slot (slot 1; slot 0 is x_stage), its flag, and the readers' acknowledgements
    // [slot][device] (32 apart); readers: the devices outside the source's group that read it
    uint8_t* x_stage1 = nullptr;
    unsigned* x_flag1 = nullptr;
    unsigned* x_ack = nullptr;
    std::vector<int> readers;
    uint8_t* stage(int slot) const { return slot ? x_stage1 : x_stage; }
    unsigned* flag(int slot) const { return slot ? x_flag1 : x_flag; }
};

// a host thread that enqueues one device's work (threads mode)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    bool idle() {
        std::lock_guard<std::mutex> lk(mu);
        return !busy;
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
};

struct Group {
    std::vector<int> mem;   // device indices, member order (member 0 the lead)
};

constexpr int kMaxL = 64;   // layers (the trunk's and the drafter's) a window plan covers

// what a window's layers need from the host, decided before they run (threads enqueue the layers concurrently): the
// mailbox seq of each layer (async: the device publishes its routing), the indexer's selection sizes and the candidate
// blocks as each layer sees them, the window's hand-off seq
struct WinPlan {
    bool par = false, async = false;
    unsigned win = 0;
    int k = 0;              // (a run of chunks) the chunk's index: its hand-off slot k % 2
    // (a run of chunks) the chunk's rows whose states before them the layers keep: row snap_t[i] into the layers'
    // snapshot set snap_ix[i], in row order
    int snap_n = 0;
    int snap_t[kSnapN] = {}, snap_ix[kSnapN] = {};
    int eg_slot = -1;       // (a run of chunks, layer-major) the Engram rows' slot (-1: k % 2)
    unsigned vl = 0;        // (a window) bit t: token t is inside a picture's span (routed with the image bias)
    unsigned seq[kMaxL] = {};
    int topk[kMaxL][kWin] = {}, cand[kMaxL][kWin] = {};
};

}  // namespace

// ------------------------------------------------------------------ the model
struct Model::Impl {
    Options o;
    Pack pk;
    Geometry g;
    std::vector<Dev> devs;
    std::vector<Layer> L;
    std::vector<Src> srcs;
    std::vector<Group> groups;
    std::vector<std::unique_ptr<Worker>> workers;   // threads mode: one per device
    bool par = false;                 // threads mode (the windows: decode)
    strata::core::GpuSetFacts gpu_set;   // the traits every GPU in devs shares (read at the load)
    // Every GPU the engine uses is a gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) in a HIP build.  The DeepSeek engine's share-based defaults (the small
    // chunk for multi-card tp runs, the whole-chunk slab, the small staging ring, the pinned copy at share >= 0.5) and the drafter's cumulative cut are
    // the 4 x R9700 measurements, so only RDNA4 takes them until another AMD part is tested (Strix Halo next).
    bool rdna4() const { return gpu_set.all_gfx12; }
    bool prompt_par = false;          // the prompt path's chunks (a thread a GPU), as --tp / --threads asked
    unsigned win_seq = 0;             // the windows the threads ran (the hand-offs' seq)
    // the chunked prompt path (threads mode): chunk tokens a pass, runs shorter than chunk_min in windows
    int chunk = 0, chunk_min = 512;
    int prompt_ceiling = 0;           // the chunk capacity, which a prompt's lend may lower for its read
    static constexpr bool kChunkedPromptDefault = true;   // the layer-major prompt is the default (gate: KL 0.1685 vs 0.1686, 8K prompt 6x, decode within 2% - ds4 notes-dsv41-r97)
    static constexpr int kFlowChunkMin = 2;   // chunk_min with every device's layers on the flow (see load)
    // --prefill-chunk at most (route() runs in kVlRows-row slabs).  The default stays kVlRows: a 16384-token chunk
    // reads a 16K prompt's experts once instead of twice (V100: 461.8 -> 321.3 s, 736 -> 506 GB read) but its buffers
    // take ~1.9 GB of VRAM experts (745 -> 644 on a 32 GB V100) from every decoded token
    static constexpr int kMaxChunk = 4 * kVlRows;
    std::vector<std::unique_ptr<Chunk>> chunks;   // per device
    size_t eg_cap = kWin;             // tokens h_eg holds
    int tap_cap = kWin;               // rows h_taps holds (a chunk seeds the drafter with its last 128)
    static constexpr int kYocoMin = 1024;   // a long prompt's skip (forward_chunks): the fewest rows worth skipping
    bool logits_all = false;          // the last feed was a chunk with logits for every row (chunk_logits)
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;   // a run of chunks: the tokens done so far (the main thread, while it waits)
    // a Stop while a run of chunks goes through the layers (Model::set_cancel, asked while forward_chunks waits): one
    // GPU stops after the layer (or the layer's chunk) it is on (halt); several stop at a chunk every one of them
    // reaches - none starts chunk halt_k or later, halt_k one past the latest any started - so no device waits on a
    // hand-off that never comes (Project Maya v1.1.20)
    std::function<bool()> cancel;
    std::atomic<bool> halt{false};
    std::mutex halt_mu;
    int halt_k = INT_MAX, halt_seen = -1;
    bool halted = false;                      // the last run ended early on a Stop
    // a yield (Model::set_yield, asked while forward_chunks waits): the run ends at a chunk every device reaches, as on
    // a Stop, but the chunks before it stay read - feed() returns true with fed() the tokens it took.  One GPU too ends
    // at a chunk there, never mid-chunk (halt_now), unless a Stop follows (Project Maya #587)
    std::function<bool()> yield_to;
    std::atomic<bool> yielding{false};
    // a run a yield may end: no device starts chunk k before chunk k - (the stages + kYieldSlack) is through every
    // layer - the stages all still busy, and the chunks a yield waits for no more than that (a device's host otherwise
    // runs chunks ahead of its GPU, and its stage ahead of the next: a yield's cut came 5 chunks past the done ones)
    static constexpr int kYieldSlack = 1;
    bool run_yields = false;
    int fed_n = 0;   // the last feed's tokens read (fewer than it was given: a yield)
    static constexpr const char* kHalted = "stopped";
    bool past_halt(int k) {   // a Stop or yield came and chunk k is one no device starts
        if (!halt.load(std::memory_order_acquire)) return false;
        std::lock_guard<std::mutex> lk(halt_mu);
        return k >= halt_k;
    }
    bool chunk_go(int k) {   // chunk k may start on this device (several GPUs)
        std::lock_guard<std::mutex> lk(halt_mu);
        if (k >= halt_k) return false;
        halt_seen = std::max(halt_seen, k);
        return true;
    }
    bool halt_now() const {
        return devs.size() == 1 && halt.load(std::memory_order_acquire) && !yielding.load(std::memory_order_acquire);
    }
    // a run's early end on device di: the expert reads still in flight land first (they count down items the next run
    // clears), then its streams are idle
    bool halt_end(int di, std::string& err) {
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
    std::vector<cudaEvent_t> chunk_done;     // on the last layer's device: chunk k through every layer
    std::unique_ptr<std::atomic<int>[]> chunk_rec;   // chunk k's event recorded
    std::atomic<int> lm_layers{0};           // (layer-major) the layers every chunk went through
    WinPlan wplan;
    int head_dev = 0;
    x3::Mat head;
    float* norm_w = nullptr;          // on the head device
    // the drafter: main_proj / main_norm (on its first block's device), its final norm, the markov head and the
    // confidence head (on the head device: the drafts' logits come from the trunk's head)
    x3::Mat d_main_proj;
    float *d_main_norm = nullptr, *d_norm = nullptr, *d_conf = nullptr;
    uint16_t *d_mk_embed = nullptr, *d_mk_head = nullptr;   // BF16 [vocab][rank], F16 [vocab][rank]
    uint8_t* d_mk_head_q8 = nullptr;   // the markov head as Q8_0 blocks (the same bias, half the bytes a step)
    int d_dev = 0;                    // the drafter's first block's device
    /// The paged engine binds each row's pages without swapping host state or waiting on the stream.
    std::function<void(int)> select_row;
    /// Absolute token positions and indexer limits for the independent rows of a shared pass.
    int64_t row_positions[kWin] = {};
    WinPlan row_plans[kWin];
    /// Gives each conversation window's length at its first row; interior entries are zero.
    int row_counts[kWin] = {};
    /// Preserves each conversation's sampled proposal until joint verification.
    std::map<int, std::vector<strata::dsv41k::SpecList>> joint_q;
    /// Retains activations between layers when a drafter batch spans more than one kernel tile.
    struct DraftTile {
        float *h = nullptr, *h2 = nullptr, *pre = nullptr;
        uint16_t* embeddings = nullptr;
        int first = 0, count = 0;
    };
    std::vector<DraftTile> draft_tiles;
    /// Owns immutable inputs and retained outputs until the shared scheduler commits a group.
    struct PipelineGroup {
        int count = 0;
        uint16_t* embeddings = nullptr;
        uint8_t* engram = nullptr;
        float* taps = nullptr;
        float* logits = nullptr;
        uint8_t* hand = nullptr;
        std::vector<uint8_t*> copies;
        int64_t positions[8] = {};
        WinPlan plans[8];
        WinPlan plan;
        std::function<void(int)> bind;
    };
    std::vector<PipelineGroup> pipeline_groups;
    PipelineGroup* active_pipeline = nullptr; ///< Supplies stable cross-stage copies during a stage launch.
    /// Selects the independent row's copy slot, and its scratch slots when run_attention batches the attention.
    int active_row = 0;
    std::vector<float*> pipeline_cpu_parts; ///< Keeps each stage's asynchronous CPU-result upload alive.
    /// Gets the rows of the window buffer (Dev::ext) that run_attention sets aside for each window row when it batches
    /// the attention: a run (conversation) of n rows gets n of these, room for the ring's g.win rows and its own n.
    int ext_slice_rows() const { return g.win + 1; }

    bool spec_win = false;            // the window being run is a draft's verification (layer() keeps backups)
    float* h_taps = nullptr;          // pinned [kWin][targets][dim]: the last window's taps
    cudaEvent_t ev_seed = nullptr;    // the drafter's seeding read h_taps: the next window's taps wait for it
    float* d_confv = nullptr;         // on the head device
    int* d_ints = nullptr;            // on the head device: drafts in, picks out
    int64_t verified_p0 = 0;          // the first position of the window verify_rows ran, and its rows
    int verified_rows = 0;
    strata::kernels::DraftRule draft_rule;   // which drafts pick_verified_rows keeps for a sampled request
    // a sampled round's drafts: the distribution q each one was drawn from (its kept list, on the head device) for
    // verify()'s rule - a point at the draft for the drafter's argmax; and the chain they were drawn with (verify()
    // checks it is its own)
    strata::dsv41k::SpecList* d_specq = nullptr;
    std::vector<strata::dsv41k::SpecList> h_specq = std::vector<strata::dsv41k::SpecList>(kWin);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
    // the request's penalties (Model::set_penalties, llama.cpp's): every pick counts the last pen_n tokens the state has
    // consumed (toks) - a verify window's row t the tokens up to its own position, the drafts before it among them.
    // pen_n 0: none (no rows, the sampler calls as without).  The rows (kWin of at most kPenCap) are allocated with the
    // first request that has penalties; each pick waits for its stream, so the next one's rows never overwrite them
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    static constexpr int kPenCap = 4096;
    int32_t* h_pen = nullptr;
    int32_t* d_pen = nullptr;
    // (tests) DS4_DSPARK_SAMPLE=1: a sampled request's drafts drawn from the drafter's distribution through the
    // request's chain, as DeepSeek's reference does (else its argmax: the same acceptance on Uranus, 31-32% of 5
    // drafts a round either way); DS4_DSPARK_QTEMP=f: those draws at f x the request's temperature (any q keeps the
    // rule exact); DS4_DSPARK_MAX=n: at most n drafts a round; DS4_DSPARK_DUMP=file: the drafter's inputs and outputs
    // (tools/ds4/dspark_ref.py checks them against DeepSeek's checkpoint)
    bool sample_drafts = std::getenv("DS4_DSPARK_SAMPLE") != nullptr && std::atoi(std::getenv("DS4_DSPARK_SAMPLE")) != 0;
    float q_temp_scale = std::getenv("DS4_DSPARK_QTEMP") ? (float) std::atof(std::getenv("DS4_DSPARK_QTEMP")) : 1.0f;
    int max_drafts = std::getenv("DS4_DSPARK_MAX") ? std::atoi(std::getenv("DS4_DSPARK_MAX")) : 0;
    std::FILE* dump_d = std::getenv("DS4_DSPARK_DUMP") ? std::fopen(std::getenv("DS4_DSPARK_DUMP"), "wb") : nullptr;
    float* h_dump = nullptr;          // (the dump's pinned copies: each block's stream, each block's ring)
    uint16_t* h_dump_ring = nullptr;

    uint16_t* embed_pinned = nullptr;
    std::vector<float> freqs_win, freqs_comp;
    std::vector<std::unique_ptr<Pool>> pools;   // one per NUMA node, its workers on that node's cores
    std::vector<std::vector<int>> cores;        // each node's physical cores
    std::vector<std::vector<platform::Core>> core_threads;   // ... each with its logical CPUs and efficiency class
    // the CPU lane's parts of an expert, part X on NUMA node X: one, the whole blob the caller's pool holds (the CPU
    // lane's code still loops over the parts)
    static constexpr int NH = 1;
    static constexpr uint64_t kRamMargin = 8ull << 30;   // RAM left to the rest (the system, the page cache's churn)
    uint8_t* file_part(const Layer& l, int e) const {
        return pk.maps[(size_t) l.region_file[(size_t) e]].first + l.region_off[(size_t) e];
    }
    // host side of the CPU lane, a window's rows
    float *h_x = nullptr, *h_w = nullptr, *h_part = nullptr;
    int* h_idx = nullptr;
    uint8_t* h_eg = nullptr;          // the window's Engram rows: [engram layer][token] of [cols][hd] then the scales
    // the Engram rows' reader (its own threads and file handles), the rows already fetched, the window's batches (one
    // an Engram layer: asked for when the window's tokens are known, collected right before that layer)
    EgReader eg_io;
    EgRowCache eg_cache;
    std::mutex eg_cache_mu;
    std::vector<std::unique_ptr<EgBatch>> eg_win;
    // Engram layer li's embed rows, as the pack stores them: 0 MXFP8 ([scale, 32 codes] blocks, 33 bytes a 32), 1
    // ggml Q6_K (210 bytes a 256), 2 ggml IQ4_NL (18 bytes a 32: an fp16 d and 16 nibble bytes), 3 an old pack's FP8
    // codes with their E8M0 scales in a tensor of their own (see eg_issue)
    std::vector<uint8_t> eg_fmt;
    std::atomic<int64_t> eg_rows_n{0}, eg_hits_n{0}, eg_reads_n{0}, eg_wait_ns{0}, eg_io_ns{0};
    uint16_t* h_emb = nullptr;        // the window's embedding rows
    std::vector<float> c_xh, c_xsb, c_acc, c_xhd, c_xsd, c_accd, c_yd;
    // the state
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    using Check = Checkpoint;         // the state at a position: every window ring, the compressors' partial groups
    static constexpr int kChecks = kCheckpoints;
    Check ck[kChecks];                // the end of the last prompt, the position before it, and the anchor: the
                                      // position before the end of the first prompt of the run of prompts that each
                                      // continued the last in place (a chat template that drops earlier turns'
                                      // reasoning at a new user message re-renders the history from about there)
    // the restore points a request asked for (GEN marks= / shared=, restore_points.hpp; pin=N is shared=N): the start
    // of the last user message, a point near the prompt's tail, the end of a shared start - each a Check, taken where
    // the read stopped for it; at most marks_cap() a conversation (DS4_RESTORE_POINTS), none unless a request asks.
    // Like ck, every one is at or before pos and holds the state of the conversation's own tokens up to it
    std::vector<Check> mk;
    uint64_t mark_clock = 0;
    std::vector<uint32_t> prof;      // [nl][n_exp] routed-expert hits
    // ... as loaded: what the flow's early reads go by, so a prompt is streamed and summed in the same order whatever
    // this process answered before
    std::vector<uint32_t> prof_load;
    Stats st;
    int cur_topk_n[kWin] = {}, cur_cand_n[kWin] = {};   // what the latest indexer / candidate source left, per query
                                                        // (plan_window's: the windows' plans are made in order)
    int n_logits = 0;                 // the last forward's logit rows
    // the decode path (windows of one): the device publishes each layer's routing into reqs, the service thread
    // answers in resps with the CPU lane's part, and the device waits for the answer only before the expert sum
    Req* reqs = nullptr;
    Resp* resps = nullptr;
    unsigned seq_issued = 0;
    std::atomic<int> layer_of[kRing];
    std::thread service;
    std::atomic<bool> quit{false};
    bool sync_decode = std::getenv("DS4_SYNC") != nullptr;
    // a window of one through moe_gate_up / moe_down (DS4_MOE_TOKEN=0: win_experts, whose down rows run
    // before the wait for the CPU lane - the same bits, no faster)
    bool moe_token_path = std::getenv("DS4_MOE_TOKEN") == nullptr || std::atoi(std::getenv("DS4_MOE_TOKEN")) != 0;
    int async_max_t = std::getenv("DS4_ASYNC_T") ? std::atoi(std::getenv("DS4_ASYNC_T")) : 8;   // windows up to this (DS4_ASYNC_T=16 is Project Maya #608: windows of 9-16 rows on the async path too; not made the default - the logits test on p620 (phaseA-tests/maya-port/A2) still showed window-size dependence)
    static float env_f(const char* k, float d) { return (float) strata::platform::env_double(k, d); }
    size_t flow_part[3] = {0, 0, 0};   // the flow's staging room for each part: its largest over the trunk's layers
    // STRATA_DS4_DISK_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): on the small windows (decode, a
    // draft's check - never the prompt path) the CPU lane leaves out a routed expert whose share of its token's route
    // weights is below this and that the caller's pool does not hold now (in_ram), so a token waits on the SSD only for
    // the experts that weigh; the weight it carried is lost (no renormalisation).  On layers from
    // STRATA_DS4_DISK_SKIP_FROM (0) on.  An expert left out still counts as used (the profile).
    const float disk_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_DISK_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    // STRATA_DS4_RAM_SKIP_W (experimental opt-in, CHANGES THE ANSWERS; 0 = off): the same rule for every routed expert
    // VRAM does not hold - in RAM or not: a token goes without the light ones the link or the CPU lane would read.
    // Where the experts all sit in RAM the decode is bound by the RAM's bandwidth, and each one left out is a
    // streamed or CPU-lane expert's bytes not read; DISK_SKIP_W then does nothing
    const float ram_skip_w = [] {
        const char* v = std::getenv("STRATA_DS4_RAM_SKIP_W");
        return v != nullptr ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    const int disk_skip_from = std::max(0, strata::platform::env_int("STRATA_DS4_DISK_SKIP_FROM", 0));
    // --route-tail-skip (Model::set_route_tail_skip; 0 = off, CHANGES THE ANSWERS): on the trunk's layers the decode
    // leaves out a routed expert VRAM does not hold when no token of the window ranks it, by weight, among its top this
    // many (mark_lane_skips): the CPU lane does not read it
    std::atomic<int> route_tail_skip{0};
    // --route-prior-pct (Model::set_route_prior_pct; 0 = off, CHANGES THE ANSWERS): on the trunk's layers, the decode's
    // router picks experts as if the score of each one VRAM holds were this many percent higher (get_route_prior)
    std::atomic<int> route_prior_pct{0};
    /// Marks the picks of a window (ids and weights kPicks apart for T tokens) that route_tail_skip leaves out, by
    /// t * kPicks + i; never on the DSpark blocks.
    void mark_lane_skips(const Layer& l, const int* ids, const float* w, int T, bool* tail) const {
        const int rank = l.mtp || l.il >= g.nl ? 0 : route_tail_skip.load(std::memory_order_relaxed);
        strata::core::mark_tail_skips(ids, w, T, l.k_exp, kPicks, l.n_exp, rank,
                                      [&](int32_t e) { return l.slot[(size_t) e] < 0; }, tail);
    }
    /// Gets the router's `prior` for layer l's decode (route_prior_pct as a fraction; 0 on the DSpark blocks).
    float get_route_prior(const Layer& l) const {
        return l.mtp || l.il >= g.nl ? 0.0f : (float) route_prior_pct.load(std::memory_order_relaxed) / 100.0f;
    }
    static_assert(kWin * kPicks <= strata::core::kMaxTailSkipEntries, "a window has too many picks for the tail skip");
    // (Options::expert_pages) the blobs of the experts the CPU lane computes now, by expert, from the RAM's pool
    // (cpu_experts_paged); and each expert's routings ([layer][expert], count_usage; empty: not counted)
    std::vector<const uint8_t*> held;
    std::vector<float> usage;
    std::atomic<int64_t> cpu_expert_count{0};   // get_cpu_experts; the CPU lane's threads add to it
    bool in_ram(const Layer& l, int e) const {   // (DISK_SKIP_W) expert e's pages in RAM now
        return !o.expert_pages->transient(l.il, e);
    }
    // STRATA_DS4_DISK_SKIP_AUTO=1 (with DISK_SKIP_W; experimental, off by default): DISK_SKIP_W only while the model is
    // cold - from a start until VRAM and the page cache hold what they can, and again after an idle that let the page
    // cache lose what the last answer left there - so warm answers are the unmodified model's.
    //  - the cold share: of the routed experts heavy enough to be computed either way (VRAM ones too), those on the CPU
    //    lane and not in RAM when asked for, over the last STRATA_DS4_DISK_SKIP_WINDOW (32) tokens
    //  - off for good, mid-answer too, once that share stops falling (no new low by 5% for
    //    STRATA_DS4_DISK_SKIP_SETTLE (64) tokens: this PC's warm level, not a fixed bar) or falls under 2%, or after
    //    STRATA_DS4_DISK_SKIP_BUDGET (512) tokens of skipping
    const bool disk_skip_auto = disk_skip_w > 0.0f && env_f("STRATA_DS4_DISK_SKIP_AUTO", 0.0f) != 0.0f;
    const int disk_skip_window = std::max(4, (int) env_f("STRATA_DS4_DISK_SKIP_WINDOW", 32.0f));
    const int disk_skip_settle = std::max(4, (int) env_f("STRATA_DS4_DISK_SKIP_SETTLE", 64.0f));
    const int disk_skip_budget = std::max(1, (int) env_f("STRATA_DS4_DISK_SKIP_BUDGET", 512.0f));
    struct SkipAuto {
        std::atomic<bool> on{true};
        // a layer's (heavy routed experts, of them on the CPU lane and not in RAM, tokens, of them on the CPU lane)
        std::deque<std::array<int, 4>> ring;
        int64_t H = 0, C = 0, Tn = 0, P = 0, since = 0, low_at = 0;
        double low = 2.0;
    } sk_auto;
    void skip_auto_off(const char* why, double share) {   // (the serve thread)
        SkipAuto& a = sk_auto;
        a.on.store(false);
        st.skip_on = 0;
        ++st.skip_switches;
        std::fprintf(stderr, "ds4: DISK_SKIP auto: off after %lld tokens (%s; cold share %.3f, the lowest %.3f)\n",
                     (long long) a.since, why, share, a.low < 1.5 ? a.low : share);
    }
    // a layer's count into the window (the serve thread); once a token (its first skip layer) the switch off
    void skip_auto_tick(int li, int T, int heavy, int cold, int heavy_cpu) {
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
    std::vector<void*> host_allocs;
    std::FILE* dump_h = std::getenv("DS4_DUMP_H") ? std::fopen(std::getenv("DS4_DUMP_H"), "wb") : nullptr;
    // DS4_DUMP_L: (threads) every layer's residual streams as records [layer, p0, T, T x hc x dim floats], window or
    // chunk path (tests: where the two part)
    std::FILE* dump_l = std::getenv("DS4_DUMP_L") ? std::fopen(std::getenv("DS4_DUMP_L"), "wb") : nullptr;
    // DS4_DUMP_S: the sub-steps inside the layers named by DS4_DUMP_S_LAYERS (default 0,1,2,3), records
    // [char name[16], int64 il, p0, T, nbytes][nbytes raw bytes] (window path)
    std::FILE* dump_s = std::getenv("DS4_DUMP_S") ? std::fopen(std::getenv("DS4_DUMP_S"), "wb") : nullptr;
    std::vector<int> dump_s_layers = [] {
        std::vector<int> v;
        const char* e = std::getenv("DS4_DUMP_S_LAYERS");
        std::string str = e ? e : "0,1,2,3";
        size_t a = 0;
        while (a < str.size()) {
            size_t b = str.find(',', a);
            if (b == std::string::npos) b = str.size();
            if (b > a) v.push_back(std::atoi(str.substr(a, b - a).c_str()));
            a = b + 1;
        }
        return v;
    }();
    // DS4_DUMP_R: the chunk path's routing, a record a layer [layer, p0, T, k, n_exp, n_vs, n_slots, kPicks] (int64),
    // then the picks' experts [T kPicks] (the first k of each token's), each expert's virtual slot [n_exp], the
    // slots' row bounds [n_vs + 1], the sorted rows' tokens [T k] (int32), the gate's logits [T n_exp] and its bias
    // [n_exp] (F32)
    std::FILE* dump_r = std::getenv("DS4_DUMP_R") ? std::fopen(std::getenv("DS4_DUMP_R"), "wb") : nullptr;
    // DS4_DUMP_Y=<layer>:<file> (layer -1: every layer): its expert sets on the chunk path, a record a piece [layer,
    // row0, rows, n_exp, wbase kind (0 VRAM, 1 ring), max_rows] (int64), its bounds [n_exp + 1], its h rows [rows ff]
    // (F16) and y rows [rows dim] (F32)
    int dump_y_layer = std::getenv("DS4_DUMP_Y") ? std::atoi(std::getenv("DS4_DUMP_Y")) : -1;
    std::FILE* dump_y = [] {
        const char* v = std::getenv("DS4_DUMP_Y");
        const char* f = v != nullptr ? std::strchr(v, ':') : nullptr;
        return f != nullptr ? std::fopen(f + 1, "wb") : nullptr;
    }();
    std::mutex dump_mu;
    // DS4_PROF: the GPU's time a decoded token by stage - events in stream order at a layer's start, after its
    // attention, after the routing and the shared expert, after the experts' gate / up, after the wait for the CPU
    // lane, after the down pass and the post - read after the token's sync, each the time since the mark before it
    bool prof_on = std::getenv("DS4_PROF") != nullptr;
    // DS4_PRED_DUMP=file (predictor data, the chunked prompt path): per layer and chunk u8 layer, u16 T, T x 384 fp16 router
    // logits, T x 256 fp16 projection of the router input.  DS4_XIN_DUMP=file (expert similarity inputs): u8 layer, u16 T,
    // T x D fp32 router inputs, the first DS4_XIN_N (1024) tokens of each layer.
    std::FILE* pred_f = std::getenv("DS4_PRED_DUMP") != nullptr ? std::fopen(std::getenv("DS4_PRED_DUMP"), "wb") : nullptr;
    std::FILE* xin_f = std::getenv("DS4_XIN_DUMP") != nullptr ? std::fopen(std::getenv("DS4_XIN_DUMP"), "wb") : nullptr;
    int xin_n = std::getenv("DS4_XIN_N") ? std::atoi(std::getenv("DS4_XIN_N")) : 1024;
    std::array<int, 64> xin_got{};
    std::FILE* trace_f = std::getenv("DS4_TRACE") != nullptr ? std::fopen(std::getenv("DS4_TRACE"), "wb") : nullptr;
    struct ProfMark {
        int dev;
        cudaEvent_t ev;
        int stage;
    };
    std::mutex prof_mu;
    std::vector<ProfMark> prof_marks;
    std::vector<std::vector<cudaEvent_t>> prof_pool;   // CUDA events belong to the device that created them
    bool prof_failed = false;
    bool prof_window = false;   // (synchronous decode windows)
    const bool prof_sync_on = std::getenv("DS4_PROF_SYNC") != nullptr;
    double prof_sync_t = 0;     // DS4_PROF_SYNC: the host clock at the last mark
    // ... and a run of chunks' (prof_chunk): a layer's start, after its attention, after the routing and the shared
    // expert, after its experts (the streamed ones' waits for their reads in it); printed after the run
    bool prof_chunk = false;
    // DS4_PROF stages of the chunked prompt (finer ones than the decode's 0..5): 0 between layers, 1 attention rest (wq_b, wo,
    // hc_post), 2 router-side tail (shared expert), 3 experts (+ waits), 4 Engram, 5 attn mHC (mix, split, norm), 6 attention
    // projections (wq_a, wkv, compressor), 7 compressor + indexer, 8 q_b + rope, 9 sparse attention, 10 ffn mHC + router + sort
    double cprof_ms[16] = {};
    // DS4_PROF: the experts' kernels alone, by type (events around the gate/up and down calls of run_set)
    struct EkMark {
        cudaEvent_t e0, e1, e2;
        int tg, td, nrows, nexp, maxrows, dev, ff;
    };
    std::vector<EkMark> ek_marks;
    std::map<int, std::vector<cudaEvent_t>> ek_pool;   // by device
    cudaEvent_t ek_ev() {
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
    void prof_mark(const Dev& d, int stage) {
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
    void prof_collect() {
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
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T) {
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

    bool dump_s_on(int il) const {
        if (dump_s == nullptr) return false;
        for (int x : dump_s_layers)
            if (x == il) return true;
        return false;
    }
    bool dump_s_name(const char* name) const {   // DS4_DUMP_S_NAMES=a,b,c: only these points (default all)
        static const std::string names = [] { const char* e = std::getenv("DS4_DUMP_S_NAMES"); return std::string(e ? "," : "") + (e ? e : "") + (e ? "," : ""); }();
        return names.empty() || names.find(std::string(",") + name + ",") != std::string::npos;
    }
    void dump_sub(Dev& d, const char* name, int il, int64_t p0, int T, const void* dev, size_t nbytes) {
        if (!dump_s_on(il) || !dump_s_name(name)) return;
        std::vector<uint8_t> v(nbytes);
        cudaSetDevice(d.id);
        cudaMemcpyAsync(v.data(), dev, nbytes, cudaMemcpyDeviceToHost, d.st);
        cudaStreamSynchronize(d.st);
        dump_sub_host(name, il, p0, T, v.data(), nbytes);
    }
    void dump_sub_host(const char* name, int il, int64_t p0, int T, const void* host, size_t nbytes) {
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

    ~Impl() {
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
    bool drafter_loaded = true; ///< Whether DSpark projections can be used to seed or draft.
    float* draft_taps = nullptr; ///< Pool-owned recent trunk inputs for rebuilding DSpark rings.

    // the memory report's names of the buffers dalloc, hmap and halloc allocate: the chunked prompt path's, the others'
    static constexpr const char* kPromptLabel = "Prompt processing";
    static constexpr const char* kWorkLabel = "DeepSeek working buffers";
    template <class T>
    bool dalloc(Dev& d, T*& p, size_t n, std::string& err, size_t* acct = nullptr, const char* label = kWorkLabel) {
        void* v = nullptr;
        cudaSetDevice(d.id);
        const size_t b = align256(n * sizeof(T));
        if (strata::core::allocate_device(&v, b,
                {d.loading_drafter ? "drafter" : acct == &d.weights ? "weights" : "other", "Device allocations",
                 d.loading_drafter ? "Drafter" : label}) != cudaSuccess) {
            err = "cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        // zeroed before anything lands in it: the loads copy in on the default stream, which the non-blocking d.st
        // does not order against - a zeroing still queued there could wipe the weights just copied
        if (cudaMemsetAsync(v, 0, b, d.st) != cudaSuccess || cudaStreamSynchronize(d.st) != cudaSuccess) {
            strata::core::release_device(v);
            err = "zeroing " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(d.id) + " failed";
            return false;
        }
        d.allocs.push_back(v);
        if (d.loading_drafter && acct == &d.weights)
            d.draft_allocations.emplace_back(v, b);
        if (acct) *acct += b;
        else d.scratch += b;
        p = (T*) v;
        return true;
    }
    static Layer& member(Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    static const Layer& member(const Layer& l, int m) { return m == 0 ? l : l.mates[(size_t) m - 1]; }
    // bytes of a VRAM slot of l's experts (an expert's whole blob)
    static size_t unit(const Layer& l) { return l.blob; }
    // the members of a group of nm that take heads: the most the heads and the wo_a groups split among evenly (64
    // heads and 8 groups: 9 -> 8, 7 or 6 -> 4, 5 -> 4, 3 -> 2); the first members the rest, only their FFN share
    int head_members(int nm) const {
        int na = nm;
        while (na > 1 && (g.heads % na != 0 || g.o_groups % na != 0)) --na;
        return na;
    }
    // host-mapped memory (the same address on the host and every device), zeroed
    template <class T>
    bool hmap(T*& p, size_t n, std::string& err, const char* label = kWorkLabel) {
        void* v = nullptr;
        if (strata::core::allocate_host(&v, n * sizeof(T) + 64, cudaHostAllocMapped | cudaHostAllocPortable,
            {"other", "Device allocations", label}) != cudaSuccess) {
            err = "host-mapped memory (" + std::to_string((n * sizeof(T)) >> 20) + " MiB)";
            return false;
        }
        std::memset(v, 0, n * sizeof(T) + 64);
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err, const char* label = kWorkLabel) {
        void* v = nullptr;
        if (strata::core::allocate_host(&v, n * sizeof(T) + 64, cudaHostAllocPortable,
            {"other", "Device allocations", label}) != cudaSuccess) {
            err = "cudaHostAlloc " + std::to_string((n * sizeof(T)) >> 20) + " MiB failed";
            return false;
        }
        host_allocs.push_back(v);
        p = (T*) v;
        return true;
    }

    // ---- weights
    // Copies tensor t whole to dst on device d in staged rounds, so that a large tensor needs no host copy of itself.
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err) {
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
    // a dense matrix in a ggml block type (x3::Mat fmt kFmtGG, row = k / 256 blocks): whole, or part of its rows / inputs
    bool mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err, size_t* acct, int part,
                int np, bool cols) {
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
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols) {
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
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n) {
        std::vector<float> v(n);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = q + b * gg::kQ8Bytes;
            const float d = f16_to_f32((uint16_t) (p[0] | p[1] << 8));
            for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return v;
    }
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0) {
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
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0) {
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
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct) {
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

    // The markov bias's head as Q8_0 blocks of 32 (34 B: F16 d + 32 int8) instead of F16 values: the same bias sums with the
    // quantisation's error, half the bytes the drafter's five markov steps read (66 MB -> 33 MB) and half the memory.  The
    // pack is untouched: the F32 / F16 / BF16 tensor is quantised as it is loaded.
    bool mk_head_q8(Dev& d, const std::string& name, uint8_t*& p, std::string& err, size_t* acct) {
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

    size_t mat_bytes(const std::string& pre) const {
        if (const Tensor* t = pk.get(pre + ".weight")) return align256(t->bytes);
        size_t b = 0;
        for (const char* s : {".trellis", ".suh", ".svh"})
            if (const Tensor* t = pk.get(pre + s)) b += align256(t->bytes);
        return b;
    }    size_t tensor_bytes(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return t ? align256(t->bytes * (t->dtype == "F32" ? 1 : 2)) : 0;
    }

    // ---- expert regions and blob layout
    // a GGUF's MXFP4 experts: the blob is gate | up | down ([out][in / 32] blocks each, as the file holds them), every
    // expert's matrix r at the same stride from expert 0's in one file (the GGUF's [n_exp][out][in] tensor)
    // (the other block types: lay.fmt stays kFmtMXFP4 - read in place as these - and lay.K[r] is role r's ggml
    // type)
    static int gg_type_of(const std::string& dtype) {
        for (int t : {gg::kTypeMXFP4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 145, 146, 224, 225, 226, 227})
            if (dtype == gg::type_name(t)) return t;
        return -1;
    }
    bool expert_layout_mx(Layer& l, std::string& err) {
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
    // cpu_experts_iq / cpu_experts_mx take this layer's experts (the checks they make before computing); `why` if not
    bool lane_fmt_ok(const Layer& l, std::string& why) const {
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
    const uint8_t* mx_mat(const Layer& l, int e, int r) const {   // expert e's matrix r in cpu_experts_paged's blob
        return held[(size_t) e] + l.lay.trellis[r];
    }
    size_t mx_part_bytes(const Layer& l, int r) const {   // (its type's)
        return r < 2 ? (size_t) g.ff * gg::type_row_bytes(gg::expert_type(l.lay, r), g.dim)
                     : (size_t) g.dim * gg::type_row_bytes(gg::expert_type(l.lay, 2), g.ff);
    }
    // matrix r's rows, and the bytes of one of them
    int mx_n(int r) const { return r < 2 ? g.ff : g.dim; }
    size_t mx_row_b(const Layer& l, int r) const { return mx_part_bytes(l, r) / (size_t) mx_n(r); }
    // part X's rows [X n / NH, ..) of expert e's matrix r
    const uint8_t* mx_part_ptr(const Layer& l, int e, int r, int X) const {
        return mx_mat(l, e, r) + (size_t) X * (size_t) (mx_n(r) / NH) * mx_row_b(l, r);
    }
    // expert e (from the file) -> its blob
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const {   // (the movers: never a slot that may change)
        for (int r = 0; r < 3; ++r)
            if (!pk.read_at(l.mx_file[r], l.mx_off[r] + (uint64_t) e * l.mx_stride[r], mx_part_bytes(l, r),
                            blob + l.lay.trellis[r]))
                return false;
        return true;
    }

    bool expert_layout(Layer& l, std::string& err) {
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
    bool load(const Options& opt, std::string& err) {
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

    // the rest of the load (Model::place_experts): the prompt path's buffers, the CPU lane and the threads that serve
    // it, the layers' expert tables (set_expert_addresses fills them)
    bool finish_load(std::string& err) {
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

    // the caches' row bytes, packed as DeepSeek keeps them (ds4_kernels.cuh): the window ring's (FP8), the compressed
    // KV's (FP4 + E4M3 scales), the index keys' (FP4 + power-of-two scales)
    size_t win_rb() const { return kv_row_bytes(kKvFp8, g.hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, g.hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, g.ix_hd); }

    // The chunked prompt's first GEMM of each kind pays the BLAS library's start (its handle's heuristics, the code objects of its
    // kernels loaded on first use): ~1 s of hipModuleLoad plus ~2.5 s of library time on the R9700 (rocprofv3), 1-2 s on CUDA, once a
    // process - 3.6 s on the first prompt that went through the flow, measured as 4.9 s for a 25-token pass against 1.0 s for the
    // next one.  Paid here instead, at load: the chunk GEMMs on tiny inputs into the chunk's own scratch buffers, or
    // (before a prompt lends them) small ones of the warm-up's own (the mHC mix GEMM in F32, the FP16 GEMM the
    // projections and the indexer weights use) - no engine state is touched (heat, profile, stats, positions, KV), so the
    // answers are what they were.  DS4_WARM=0 turns it off.
    void warm_flow() {
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
    // The warm-up runs once the model is loaded, on a thread of its own: it cannot hide inside the load; after it, a
    // person typing gives it the seconds it needs.  A chunked pass that starts first waits for it (warm_mu) - never
    // later than that pass would have been without it - and a warm-up that finds the flow already used does nothing.
    // DS4_WARM=0 turns it off.
    std::mutex warm_mu;
    std::atomic<bool> flow_used{false};
    std::thread warm_bg;
    void start_warm() {
        if (chunk <= 0 ||(std::getenv("DS4_WARM") && std::atoi(std::getenv("DS4_WARM")) == 0)) return;
        warm_bg = std::thread([this] {
            std::lock_guard<std::mutex> lk(warm_mu);
            if (flow_used.load() || quit.load()) return;
            warm_flow();
        });
    }
    // ---- the chunked prompt path
    size_t x_cap() const { return (size_t) std::max<int>(kWin, chunk + 2); }   // a window's / chunk's new source rows
    // the bytes h_eg takes for `tokens` tokens' Engram rows (eg_cap of them)
    size_t get_engram_bytes(size_t tokens) const {
        return (size_t) std::max(1, pk.eg_layers) * tokens * (size_t) ((pk.eg_ngram - 1) * pk.eg_heads) *
               (size_t) (pk.eg_hd + pk.eg_hd / 32) + 64;
    }
    size_t chunk_hand_bytes() const {
        return (size_t) chunk * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }

    // the staging ring's size (MB) for the experts a chunk streams from the CPU lane: 400 where a good part of them are
    // streamed; 16 where the cards hold most of the experts (they are within 30% of fitting) (a 4-card box) - the ring's VRAM is then worth more
    // as expert slots, and the decode after a prompt keeps the windowed path's speed (R9700 x4: 29.6 -> 31.7 tok/s)
    // The share of the routed experts the GPUs can hold, from what is known at layout (their free memory less the reserve, the
    // prompt path's buffers and the dense weights; the trunk's experts only when no drafter is loaded): 1 = all of them.
    double gpu_expert_share(double* exp_out = nullptr, double* room_out = nullptr, double* other_out = nullptr) const {
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
    // this device's host-to-device link (bytes a second): 32 MB from pinned memory, the best of three (~10 ms on PCIe 3 x16), once a
    // device; 0: not measured.  The copies and their events on the legacy stream, as the ordinary load copies: timed on a stream of
    // their own they left decode ~20% slower for the whole run on 2x CMP 170HX (PCIe Gen2), though nothing used that stream again
    // (Project Maya #609, mw00).  Only DS4_RING_MB=link asks for it.
    std::map<int, double> link_bw;
    double link_h2d(Dev& d) {
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
    size_t ring_mb_auto() const {
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
    bool slab_auto_grew_ = false, slab_force_base_ = false;   // the automatic attention slab: grew this try / must not grow (it did not fit)
    // the chunk path's device buffers, laid out from base (nullptr: only their bytes counted): those of a fixed size
    // (the attention's and the indexer's slabs, the weights' FP16 reconstructions, Engram's slabs), then those of C
    // rows
    size_t lay_chunk(Chunk& c, int C, uint8_t* base) const {
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
    // the chunk path's buffers for chunks of C rows, laid out from `device` and `host` (nullptr: only their bytes
    // counted): on the GPU lay_chunk's and the ring the streamed experts land in; in page-locked memory the ring's
    // staging (once the load sized it: Chunk::S) at a 4 KiB bound for the unbuffered reads, and the embedding and
    // Engram rows of the two chunks in flight.  Gets the GPU's bytes; `host_bytes` the host's
    size_t lay_prompt(Chunk& c, int C, uint8_t* device, uint8_t* host, size_t& host_bytes) const {
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
    // lays device di's chunk path out for chunks of C rows (at most the load's chunk, prompt_ceiling) in
    // the lent `device` and `host` memory, the device's zeroed; C 0: waits until it no longer
    // reads them
    bool lay_out_prompt(int di, int C, uint8_t* device, uint8_t* host, std::string& err) {
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
    // raises prompt_ceiling to C rows, growing the page-locked buffers the load sized for its chunk:
    // the embedding and Engram rows, the remote sources' staging and the hand-offs.  Between
    // prompts (no chunk laid out).  False (and `err`) when they cannot grow; the old ceiling then stays, as the
    // buffers that did grow still hold it
    bool raise_prompt_ceiling(int C, std::string& err) {
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
    // replaces page-locked buffer `p` (halloc's, or with `mapped` hmap's) by one of `n` elements; false (and `err`)
    // keeping `p` when there is none to be had
    template <class T>
    bool regrow_host(T*& p, size_t n, bool mapped, std::string& err) {
        T* q = nullptr;
        if (!(mapped ? hmap(q, n, err, kPromptLabel) : halloc(q, n, err, kPromptLabel)))
            return false;
        if (p != nullptr) {
            host_allocs.erase(std::find(host_allocs.begin(), host_allocs.end(), (void*) p));
            strata::core::release_host(p);
        }
        p = q;
        return true;
    }

    // a device's chunk dimensions in a group of nm members: its heads' width, wo_a groups and FFN share (the widest
    // member's: member 0 takes split128's odd blocks), the attention's and the indexer's slabs (the indexer's scores
    // rows are the context wide - about 64 MB of them)
    void chunk_dims(Chunk& c, int nm) const {
        const int NHL = g.heads / head_members(nm), HQ = NHL * g.hd, OG = g.o_groups / head_members(nm);
        int FF = 0, ff0_ = 0;
        split128(g.ff, nm, 0, ff0_, FF);
        c.HQ = HQ, c.OG = OG, c.FF = FF;
        c.A = std::max(64, (int) (((size_t) 8 << 20) / (size_t) HQ) / 64 * 64);
        c.AI = std::max(32, std::min(128, (int) (((size_t) 64 << 20) / (((size_t) o.max_context + 64) * 4)) / 32 * 32));
        c.w16_elems = (size_t) 8 << 20;
    }
    bool alloc_chunks(std::string& err) {
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

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, then the product.
    // The product prefers the engine's own RDNA3 WMMA kernel (src/prefill/wmma_gemm.cu) over hipBLAS: on gfx1151 hipBLAS
    // measures 4.97 TFLOPS at this call convention while that kernel measures 20-40 (BM 128 x BN 256, grouped tile order), and
    // the chunked prompt's dense GEMM is 11.1 s of a 4K prompt at the library's rate.  Same operands and the same values (the
    // k order differs, so it is rounding-level: the quality gate and the acceptance gate cover it); STRATA_DENSE_PFGEMM=0 keeps
    // hipBLAS for an A/B.
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err) {
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
    // ---- the busy experts (run_set): an expert of many rows as FP16 GEMMs.  The tiles decode its weights again for
    // every 64 of its rows (a prompt's chunk gives a popular expert hundreds); decoded once, to the FP16 values the tiles
    // multiply (iq::f16_rows), the library's GEMM takes them at its own speed.  DS4_FAT_ROWS=N (default 0, off; N:
    // the experts of N rows or more) takes the experts of N rows or more; the same products, only the sums' order differ
    // (3x V100, Maya-XS: 64 rows +6-8% on 8K / 32K prompts; KL vs the reference 0.276, the stock path 0.285 - the same;
    //  RTX 5090, ISTA-DAS 2.25 bpw: a 13K-token prompt 1,400 -> 1,625 tok/s at 64 rows, decode unchanged)
    static int fat_rows() {
        static const int n = [] {
            const int v = std::getenv("DS4_FAT_ROWS") ? std::max(0, std::atoi(std::getenv("DS4_FAT_ROWS"))) : 0;
            return v > 0 ? std::max(v, 17) : 0;   // (the light kernels take the experts of 16 rows or fewer)
        }();
        return n;
    }
    bool fat_ready(Dev& d, Chunk& c, int D, int FF) {
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
    // h16[lr0 + r] = FP16 clamped SwiGLU of the expert's gate and up over x16[row_tok[row + r]], r < n
    bool fat_gate_up(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int row, int n, int a, int D, int FF,
                     cudaStream_t s, std::string& err) {
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
    // y[lr0 + r] = down(h16[lr0 + r]), r < n
    bool fat_down(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int lr0, int n, int D, int FF, cudaStream_t s,
                  std::string& err) {
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
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err) {
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
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err) {
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

    // expert e of l -> its VRAM copy at dst, in the GPU layout, by DMA on the copy stream from `src` (its file region
    // read into pinned staging; null: from where the CPU lane reads it)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr) {
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

    // a device's streamed experts: batches of ring_n / kParts of each of its layers' CPU-lane experts, chunk by chunk
    // in layer order; the copies run at most kParts batches ahead of the compute
    struct Batch {
        int k = 0, il = 0, j0 = 0, n = 0;   // chunk, layer, its host_list range
    };
    struct Streamer {
        std::vector<Batch> b;
        size_t cp = 0, kp = 0;        // the next batch to copy, to compute
        std::atomic<size_t> done{0};  // cp, for read_ahead's thread
        // DS4_SKIP_UNPICKED_ROWS / DS4_SKIP_UNPICKED (Project Maya #691): a chunk of few rows streams only the experts some
        // row of it picked - the copies of its layer wait for the layer's routing (routed_layer / routed_chunk, the
        // experts' row bounds in routed_bounds), then copy the picked ones into the same slots and batches
        std::vector<char> skip_k;     // chunk k streams only its picks
        bool skips(int k) const { return k >= 0 && (size_t) k < skip_k.size() && skip_k[(size_t) k]; }
        int routed_layer = -1, routed_chunk = -1;
        std::vector<int> routed_bounds;
        size_t skipped = 0, copied = 0;
    };
    // chunks of up to this many rows stream only the experts they pick (DS4_SKIP_UNPICKED_ROWS=N; DS4_SKIP_UNPICKED=1:
    // every chunk); 0 (default): every chunk streams every expert of its layers
    static int skip_unpicked_rows() {
        static const int n = [] {
            const char* v = std::getenv("DS4_SKIP_UNPICKED");
            if (v != nullptr && std::atoi(v) != 0) return INT_MAX;
            const char* r = std::getenv("DS4_SKIP_UNPICKED_ROWS");
            return r != nullptr ? std::max(0, std::atoi(r)) : 0;
        }();
        return n;
    }
    // streamed experts: the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const {
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
    bool copy_ahead(Chunk& c, Streamer& S, int di, std::string& err) {
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
    // Windows (WDDM: HIP's runtime and CUDA's driver alike) submits queued work only when the host waits on the GPU:
    // the flow's copies and the compute it queued between its waits are submitted at once (win_submit); elsewhere
    // nothing.  (CUDA too since #303: two RTX 3080s hung before the first layer, one GPU spinning on the other's flow)
    static void kick(cudaStream_t s) { win_submit(s); }
    size_t flow_region(int r) const {   // a part's room in a staging slot (read from its offset rounded down)
        constexpr size_t A = platform::kDirectAlign;
        return ((flow_part[r] + A - 1) & ~(A - 1)) + A;
    }
    size_t flow_stride() const { return flow_region(0) + flow_region(1) + flow_region(2); }
    static Chunk::Item& fit(Chunk& c, size_t i) { return c.items[i - c.base]; }
    // the device's member index in L0's group (0 alone)
    int flow_mi(const Chunk& c, const Layer& L0) const {
        for (int m = 0; m < L0.nm; ++m)
            if (member(L0, m).dev == c.dev) return m;
        return 0;
    }
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v) {
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
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T) {
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
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b) {
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
    // the next (chunk, layer) of the device after (k, il) in the run's order that streams experts (false: none) - a
    // long prompt's skip leaves out its early chunks' layers from Run::yoco on (kv_only there: no experts)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const {
        while (flow_next1(c, k, il, k2, il2)) {
            if (c.fl_yoco < 0 || k2 >= c.fl_cut_k || il2 < c.fl_yoco) return true;
            k = k2;
            il = il2;
        }
        return false;
    }
    // the rows chunk k runs layer il on (a long prompt's skip: the cut chunk's from cut_r on, from the yoco layer on)
    static int fl_rows(const Chunk& c, int k, int il) {
        const int T = c.fl_T[(size_t) k];
        return c.fl_yoco >= 0 && k == c.fl_cut_k && il >= c.fl_yoco ? T - c.fl_cut_r : T;
    }
    bool flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const {
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
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err) {
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
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err) {
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

    // a run's VRAM experts [i0, i1) (their blobs by member x's slot_tab), through run_set with layer l's layout
    bool run_vram(Dev& d, Chunk& c, const Layer& l, const Layer& x, const int* b, int i0, int i1, int T,
                  std::string& err) {
        return run_set(d, c, l, nullptr, 0, unit(x), c.bounds, b, i0, i1, T, err, x.slot_tab);
    }
    // the experts [i0, i1) of a sorted set (blobs at wbase + i * stride; bounds b on the host, d_bounds on the device)
    // over the chunk's rows, in pieces of at most C rows, each added into acc in pick order
    // (tab: device, the set's blobs from member wb0's on, in place of wbase + i * stride - the VRAM slots' of
    // Options::expert_pages, slot_tab; null: none)
    bool run_set(Dev& d, Chunk& c, const Layer& l, const uint8_t* wbase, int wb0, size_t stride, const int* d_bounds,
                 const int* b, int i0, int i1, int T, std::string& err, const unsigned long long* tab = nullptr) {
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

    std::vector<std::unique_ptr<ExpertHop>> hops; ///< Indexed by the shared engine's card index.
    const int32_t* hop_holders = nullptr; ///< Published only between decode windows.

    bool load_dense(int di, std::string& err, bool drafter_only = false, int layer_only = -1) {
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

    void set_vm(Layer& l, int e, const uint8_t* b) {
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
    /// Gets layer l's experts hottest first by the profile, then by index.
    std::vector<int> get_heat_order(const Layer& l) const {
        std::vector<int> order((size_t) l.n_exp);
        std::iota(order.begin(), order.end(), 0);
        const uint32_t* p = prof.data() + (size_t) l.il * g.n_exp;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
        return order;
    }
    /// Readies the layers (the trunk's and the DSpark blocks') for experts that set_expert_addresses places
    /// (Options::expert_pages): none on the GPU and no copy of the CPU lane's own, each layer's slot table allocated.
    bool place_paged(std::string& err) {
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
    /// Makes the GPU compute the experts at device addresses `at` ([layer][expert]; null: the CPU lane's) from the
    /// next window on: their slots in by_heat's order, the decode's table (vram_tab) and the prompt's (slot_tab).
    bool set_expert_addresses(const std::vector<const uint8_t*>& at, std::string& err, int stage) {
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

    // where the first `n` layers' experts lie in the files (the trunk's, then the DSpark blocks')
    std::vector<ExpertFileLayout> get_expert_files(size_t n) const {
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

    // The ways this host's CPUs can serve the CPU lane (make_pools takes one): a CPU list
    // per part X - NH of them, part X's pool on node X; with one part, one pool over every node's cores (the pages
    // lie where the pool put them) - in the order its workers take them.  The service thread (pinned to node 0's
    // first core) helps part 0's pool in place of a worker there.
    //   cores   a thread a physical core (as before; with --cpu-threads n, at most n)
    //   pcores  the performance cores alone (a hybrid CPU: Intel's P- and E-cores)
    //   node    node 0's cores alone (in place on two NUMA nodes or more: one pool spinning on both sockets syncs
    //           across them - GLM's lane on a 2-socket Xeon: 531 us an expert on 88 threads, 216 on one socket's 44)
    //   smt     a thread a logical CPU: every core's first CPU, then their siblings (a CPU with SMT) - by hand only
    //           (STRATA_DS4_CPU_LANE=smt): the lane's calls time the same on a core's two CPUs as on
    //           one (i5-12600T 1.84 / 1.86 ms, Ryzen AI Max+ 395 0.62 / 0.63), while decoding with the siblings busy
    //           was no faster and once slower - the siblings are left to the readers, as GLM's lane leaves them
    struct LaneCfg {
        std::string name;
        std::vector<std::vector<int>> cpus;
    };
    std::vector<LaneCfg> lane_cfgs() const {
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
    std::vector<std::unique_ptr<Pool>> pools_for(const LaneCfg& c) const {
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
    // the CPU lane's workers: lane_cfgs' first (cores), or the one STRATA_DS4_CPU_LANE=<name> asks for
    void make_pools() {
        const std::vector<LaneCfg> cfgs = lane_cfgs();
        size_t pick = 0;
        if (const char* v = std::getenv("STRATA_DS4_CPU_LANE"))
            for (size_t i = 0; i < cfgs.size(); ++i)
                if (cfgs[i].name == v) pick = i;
        pools = pools_for(cfgs[pick]);
        (void) gg::mx_tier();   // (measured here, not on the first token)
    }

    void reset() {
        pos = 0;
        toks.clear();
        ctoks.clear();
        for (Check& c0 : ck) c0.pos = -1;
        mk.clear();
        for (Layer& l0 : L)
            for (int m = 0; m < l0.nm; ++m) {
                Layer& l = member(l0, m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                if (l.win) cudaMemsetAsync(l.win, 0, (size_t) g.win * win_rb(), d.st);
                if (l.c_kv) cudaMemsetAsync(l.c_kv, 0, (size_t) l.ratio * g.hd * 4, d.st);
                if (l.c_score) cudaMemsetAsync(l.c_score, 0, (size_t) l.ratio * g.hd * 4, d.st);
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) { snap_into(ck[slot & 1]); }
    void snap_into(Check& c) {
        c.pos = pos;
        c.win.resize(L.size());
        c.comp.resize(L.size() * 2);
        for (size_t i = 0; i < L.size(); ++i) {
            Layer& l = L[i];
            Dev& d = devs[(size_t) l.dev];
            cudaSetDevice(d.id);
            c.win[i].resize(get_checkpoint_ring_bytes(i));
            cudaMemcpyAsync(c.win[i].data(), l.win, c.win[i].size(), cudaMemcpyDeviceToHost, d.st);
            if (l.c_kv) {
                c.comp[2 * i].resize((size_t) l.ratio * g.hd);
                c.comp[2 * i + 1].resize((size_t) l.ratio * g.hd);
                cudaMemcpyAsync(c.comp[2 * i].data(), l.c_kv, c.comp[2 * i].size() * 4, cudaMemcpyDeviceToHost, d.st);
                cudaMemcpyAsync(c.comp[2 * i + 1].data(), l.c_score, c.comp[2 * i + 1].size() * 4,
                                cudaMemcpyDeviceToHost, d.st);
            }
        }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
    }

    // the restore points a conversation keeps besides its checkpoints (MAYA_DS4_RESTORE_POINTS, 4; 0: a request's
    // marks= and shared= are not kept)
    static int marks_cap() {
        static const int n = [] {
            const char* v = std::getenv("STRATA_DS4_RESTORE_POINTS");
            if (v == nullptr) v = std::getenv("DS4_RESTORE_POINTS");
            return v != nullptr ? std::clamp(std::atoi(v), 0, kMarksMax) : 4;
        }();
        return n;
    }
    // the state at pos as a restore point a request asked for (shared: the end of a shared start, given up last); one
    // at this position already is kept (it holds the same state), else the one mark_victim() picks makes room - or this
    // one is not kept, when mark_victim() picks it
    void mark(bool shared) {
        if (Check* c = mark_room(pos, shared)) snap_into(*c);
    }
    // the restore point at position `at` to fill (nullptr: none kept, or one there already - in use again)
    Check* mark_room(int64_t at, bool shared) {
        return strata::core::make_restore_point_room(mk, mark_clock, at, shared, marks_cap(), mark_victim);
    }
    // checkpoint i of best_check()'s: ck, then the marks
    const Check& check_at(int i) const { return i < kChecks ? ck[i] : mk[(size_t) (i - kChecks)]; }

    // the checkpoint or mark at or before position limit (the state's tokens agree with a new prompt up to there), or -1
    int best_check(int64_t limit) const {
        std::vector<int64_t> at(kChecks + mk.size());
        for (size_t i = 0; i < at.size(); ++i) at[i] = check_at((int) i).pos;
        return latest_point(at, limit, pos);
    }

    // checkpoint 1 (the position before the last prompt's end) becomes the anchor too
    void pin() {
        if (ck[1].pos > 0) ck[2] = ck[1];
    }
    // the checkpoints the state can go back to (what a slot of it keeps), latest first
    std::vector<int64_t> check_positions() const {
        std::vector<int64_t> v;
        for (int k : slot_checks())
            if (std::find(v.begin(), v.end(), check_at(k).pos) == v.end()) v.push_back(check_at(k).pos);
        std::sort(v.rbegin(), v.rend());
        return v;
    }
    // the checkpoints a slot keeps, as best_check() numbers them (each with its role: 0, 1, the anchor, a mark)
    // (the marks least recently used first)
    std::vector<int> slot_checks(int64_t upto = -1) const {
        if (upto < 0) upto = pos;
        std::vector<int> v, m;
        for (int i = 0; i < kChecks + (int) mk.size(); ++i) {
            const Check& c = check_at(i);
            if (c.pos > 0 && c.pos <= upto && c.win.size() == L.size()) (i < kChecks ? v : m).push_back(i);
        }
        std::stable_sort(m.begin(), m.end(), [this](int a, int b) { return check_at(a).used < check_at(b).used; });
        v.insert(v.end(), m.begin(), m.end());
        return v;
    }
    /// Includes recent trunk inputs after the final DSpark ring in paged conversations.
    size_t get_checkpoint_ring_bytes(size_t layer) const {
        return (size_t) g.win * win_rb() + (draft_taps != nullptr && layer + 1 == L.size()
            ? (size_t) g.win * g.d_targets.size() * g.dim * sizeof(float) : 0);
    }
    size_t check_bytes() const {
        size_t n = 16;
        for (const Layer& l : L) n += (size_t) g.win * win_rb() + (l.c_kv ? 2 * (size_t) l.ratio * g.hd * 4 : 0);
        return n;
    }

    bool restore(int64_t limit) {
        const int b = best_check(limit);
        if (b < 0 || !get_check_problem(check_at(b)).empty()) return false;
        const Check& c = check_at(b);
        for (size_t i = 0; i < L.size(); ++i)
            for (int m = 0; m < L[i].nm; ++m) {   // (a group's members hold the same state: the lead's copy)
                Layer& l = member(L[i], m);
                Dev& d = devs[(size_t) l.dev];
                cudaSetDevice(d.id);
                cudaMemcpyAsync(l.win, c.win[i].data(), c.win[i].size(), cudaMemcpyHostToDevice, d.st);
                if (l.c_kv) {
                    cudaMemcpyAsync(l.c_kv, c.comp[2 * i].data(), c.comp[2 * i].size() * 4, cudaMemcpyHostToDevice, d.st);
                    cudaMemcpyAsync(l.c_score, c.comp[2 * i + 1].data(), c.comp[2 * i + 1].size() * 4,
                                    cudaMemcpyHostToDevice, d.st);
                }
            }
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        if (draft_taps != nullptr && drafter_loaded) {
            std::string err;
            if (!reseed_drafter_from_taps(c.pos, err))
                return false;
        }
        pos = c.pos;
        toks.resize((size_t) pos);
        ctoks.resize((size_t) pos);
        for (Check& o2 : ck)   // a later checkpoint saw tokens the state no longer has
            if (o2.pos > pos) o2.pos = -1;
        mk.erase(std::remove_if(mk.begin(), mk.end(), [this](const Check& x) { return x.pos > pos; }), mk.end());
        strata::core::touch_restore_point(mk, mark_clock, pos);
        return true;
    }


    // Makes the position and the Engram history follow `toks`, after another conversation's tokens took their place
    // (swap_conversation).
    void set_position() {
        pos = (int64_t) toks.size();
        ctoks.resize((size_t) pos);
        for (size_t i = 0; i < (size_t) pos; ++i)
            ctoks[i] = toks[i] < pk.eg_vocab ? pk.eg_map[(size_t) toks[i]] : 0;
    }

    // Model::get_state_problem: the first kept checkpoint of `s` whose rings or partial groups are not this model's
    std::string get_state_problem(const ConversationState& s) const {
        for (const Check& c : s.checks)
            if (c.pos >= 0)
                if (std::string why = get_check_problem(c); !why.empty()) return why;
        for (const Check& c : s.marks)
            if (std::string why = get_check_problem(c); !why.empty()) return why;
        return {};
    }
    // why checkpoint `c` cannot go back to the GPU: a layer count, ring or partial group not this model's ("": it can)
    std::string get_check_problem(const Check& c) const {
        if (c.win.size() != L.size() || c.comp.size() != 2 * L.size())
            return "a checkpoint of " + std::to_string(c.win.size()) + " layers, not " + std::to_string(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            const size_t group = L[i].kv_src && L[i].ratio > 1 ? (size_t) L[i].ratio * g.hd : 0;
            if (c.win[i].size() != get_checkpoint_ring_bytes(i) || c.comp[2 * i].size() != group ||
                c.comp[2 * i + 1].size() != group)
                return "a checkpoint whose layer " + std::to_string(i) + " is not this model's";
        }
        return {};
    }

    // Model::swap_conversation
    void swap_conversation(ConversationState& s) {
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.st);
        }
        toks.swap(s.tokens);
        for (int k = 0; k < kChecks; ++k)
            std::swap(ck[k], s.checks[(size_t) k]);
        mk.swap(s.marks);
        set_position();
    }

    std::string slot_identity() const {
        const auto size_of = [](const std::string& p) {
            std::error_code ec;
            const auto s = std::filesystem::file_size(p, ec);
            return ec ? 0ull : (unsigned long long) s;
        };
        std::ostringstream id;
        id << "strata-ds4-session-4 engine=" << STRATA_VERSION << " mtp=" << g.n_mtp << " dim=" << g.dim << " hd=" << g.hd
           << " win=" << g.win << " ratios=";
        for (int r : g.ratios) id << r << ",";
        id << " kv=";
        for (int v : g.kv_src) id << v << ",";
        id << " ix=";
        for (int v : g.ix_src) id << v << ",";
        id << " ds4.txt:" << size_of(pk.dir + "/ds4.txt") << " engram.bin:" << size_of(pk.dir + "/engram.bin")
           << " tensors.txt:" << size_of(pk.dir + "/tensors.txt");
        for (size_t i = 0; i < pk.files.size(); ++i)
            id << " " << pk.files[i] << ":" << platform::file_size(pk.fds[i]);
        return id.str();
    }

    size_t bounce_bytes() const {
        return (size_t) kWin * ((size_t) g.hc * g.dim * 4 + 8 * 4 + topk_stride() * 4 + cand_stride() * 4);
    }
    size_t topk_stride() const { return (size_t) std::max(g.ix_topk, 32); }
    size_t cand_stride() const { return (size_t) std::max(g.cand_topk, 32); }
    size_t bm_stride() const { return (size_t) (o.max_context + 64) / std::max(1, g.cand_blk) + 64; }
    static constexpr int kSlots = kWin * kPicks;   // the CPU lane's (token, expert) rows a window can need

    // ---- the vision tower (the checkpoint's vision.* / aligner.*) on the first GPU, its weights streamed from the
    // host; a prompt's pictures as their spans' rows (set_images)
    VisionTower vt;
    VisionGeo vg;
    int img_tok = -1;
    std::string vision_err;              // why the tower did not load (pictures off), or empty
    std::vector<uint16_t> img_mark[3];   // image_start, image_newline, image_end: BF16 [dim]
    std::vector<Image> imgs;             // the pictures of the prompt being fed, by start
    // a BF16 (or, rounded to nearest even, F32) tensor of n values into dst
    bool read_bf16(const Tensor& t, size_t n, void* dst) {
        if (t.dtype == "BF16") return t.bytes == n * 2 && pk.read(t, dst);
        if (t.dtype != "F32" || t.bytes != n * 4) return false;
        std::vector<float> f(n);
        if (!pk.read(t, f.data())) return false;
        uint16_t* o = (uint16_t*) dst;
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, &f[i], 4);
            o[i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        return true;
    }
    bool load_vision(std::string& err) {
        vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
        const int tok = (int) pk.num("image_token_id", -1);
        if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
        vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
        vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
        vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
        vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
        const char* mark[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const Tensor* t = pk.get(mark[i]);
            img_mark[i].resize((size_t) g.dim);
            if (t == nullptr || !read_bf16(*t, (size_t) g.dim, img_mark[i].data())) {
                err = std::string(mark[i]) + ": not a BF16 (or F32) row of the model's width";
                return false;
            }
        }
        Dev& d = devs[0];
        // the tower's tensors as BF16: the checkpoint's own, or a GGUF's mmproj (F32 norms / biases / marker rows, which
        // came from BF16, converted back exactly; mlp.w1 as the two files' gate and up, the halves of its rows)
        auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
            const Tensor* t = pk.get(n);
            if (t == nullptr && n.size() > 13 && n.compare(n.size() - 13, 13, "mlp.w1.weight") == 0) {
                const std::string pre = n.substr(0, n.size() - 13);
                const Tensor* tg = pk.get(pre + "mlp.gate.weight");
                const Tensor* tu = pk.get(pre + "mlp.up.weight");
                if (tg == nullptr || tu == nullptr || tg->shape.size() != 2 || tg->shape != tu->shape) return false;
                shape = {tg->shape[0] * 2, tg->shape[1]};
                const size_t half = (size_t) tg->shape[0] * (size_t) tg->shape[1];
                return dst == nullptr || (read_bf16(*tg, half, dst) && read_bf16(*tu, half, (uint16_t*) dst + half));
            }
            if (t == nullptr) return false;
            shape = t->shape;
            size_t n_el = 1;
            for (int64_t x : t->shape) n_el *= (size_t) x;
            return dst == nullptr || read_bf16(*t, n_el, dst);
        };
        if (!vt.load(vg, d.id, read, err)) return false;
        img_tok = tok;
        CK(cudaSetDevice(d.id));
        CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
        d.free0 = uma_free(d.id, d.free0);
        return true;
    }
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
        if (img_tok < 0) {
            err = "this model has no vision tower (or it is off)";
            return false;
        }
        std::vector<uint16_t> rows;
        int nh = 0, nw = 0;
        if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
        const size_t D = (size_t) g.dim;
        span.resize(((size_t) nh * (nw + 1) + 2) * D);
        uint16_t* o = span.data();
        auto put = [&](const uint16_t* r) {
            std::memcpy(o, r, D * 2);
            o += D;
        };
        put(img_mark[0].data());
        for (int i = 0; i < nh; ++i) {
            for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
            put(img_mark[1].data());
        }
        put(img_mark[2].data());
        return true;
    }
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const {
        if (imgs.empty()) return nullptr;
        auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
        if (it == imgs.begin()) return nullptr;
        const Image& im = *(it - 1);
        const int64_t r = p - im.start;
        if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
        return im.rows->data() + (size_t) r * g.dim;
    }
    // token tk's embedding row at position p: inside a picture's span, the picture's
    const uint16_t* emb_row(int64_t p, int tk) const {
        const uint16_t* r = tk == img_tok ? image_row(p) : nullptr;
        return r != nullptr ? r : embed_pinned + (size_t) tk * g.dim;
    }
    bool in_image(int64_t p) const { return img_tok >= 0 && toks[(size_t) p] == img_tok && image_row(p) != nullptr; }
    // the Engram's compressed id of token tk at p (-1: a picture's, which no n-gram spans)
    int32_t ctok(int64_t p, int tk) const {
        if (tk == img_tok && image_row(p) != nullptr) return -1;
        return tk < pk.eg_vocab ? pk.eg_map[(size_t) tk] : 0;
    }
    // a chunk's image rows (route()'s bits); false: none
    bool vl_bits(int64_t p0, int T, VlBits& b) const {
        if (imgs.empty()) return false;
        std::memset(&b, 0, sizeof b);
        bool any = false;
        for (int t = 0; t < T && t < kVlRows; ++t)
            if (in_image(p0 + t)) {
                b.w[t >> 5] |= 1u << (t & 31);
                any = true;
            }
        return any;
    }

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b) {
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
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b) {
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
    void eg_stats() {
        st.eg_rows = eg_rows_n.load();
        st.eg_hits = eg_hits_n.load();
        st.eg_reads = eg_reads_n.load();
        st.eg_io_ms = (double) eg_io_ns.load() / 1e6;
        std::lock_guard<std::mutex> lk(eg_cache_mu);
        st.eg_cached = (int64_t) eg_cache.size();
        st.eg_cache_cap = (int64_t) eg_cache.capacity();
    }
    // the chunked prompt path: rows of Engram layer `only` into dst, now (a helper thread, a chunk ahead)
    void engram_rows(int64_t p0, int T, int only, uint8_t* dst) {
        EgBatch b;
        eg_issue(p0, T, only, dst, b);
        eg_finish(b);
    }

    // ---- the CPU lane: a window's routed experts that live on the host, as entries (an expert and up to 8 of its
    // tokens, in rows slot0 ..) computed together: each node its column part of every matrix, from memory on that node,
    // in four rounds (the gate / up split-K pieces, each preparing its two input blocks of every token; each band's
    // sums / codebook / Hadamard / swiglu and that block of the down input, which both nodes then read whole; the down
    // pieces; the down bands).  yd[slot] gets each row's expert output.
    static constexpr int kEntMax = kWin > kc::kExl3CpuMaxN ? kWin : kc::kExl3CpuMaxN;
    struct Ent {
        int e = 0, n = 0, slot0 = 0;
        int tok[kEntMax] = {};
    };
    // tokens an entry holds: the i-quant GGUF experts take a whole window's tokens of an expert in one call (the CPU
    // lane's exact multi-token rows, 7+ tokens an expert: decoded once for all), the others (MXFP4, EXL3) up to 8
    static int ent_cap(const Layer& l) {
        return l.lay.fmt == gg::kFmtMXFP4 && !gg::mx_experts(l.lay) ? kEntMax : kc::kExl3CpuMaxN;
    }
    void run_nodes(int n, const std::function<void(int)>* f) {
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->start(n, f[X]);
        pools[0]->help();
        for (int X = 0; X < NH; ++X) pools[(size_t) X]->wait();
    }

    // a GGUF's MXFP4 experts on the CPU lane, read from the file's mapping or their copy in RAM (mx_part_ptr): each
    // node its rows of every matrix - the gate / up rows [X n_ff / NH, ..) of each entry (the swiglu into its h row),
    // then (h whole) the down rows [X dim / NH, ..) into yd.  A row's sum is the same whatever the split: the same bits
    // in a copy as in place.  The device's arithmetic (gg::mx_rows: int8 activations rounded as the device rounds
    // them, exact integer sums a block; every CPU tier the same bits): each token's input quantized once, whatever
    // number of entries take it, and each job its band of h straight into the down's int8 input.
    // (ord: the order the entries are computed in - their slots, so the sums, stay the same; null: as they are)
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
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
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord) {
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

    /// Computes the lent entries in one transfer each way, leaving the others for the CPU lane.
    bool run_expert_hop(const Layer& l, const float* x, size_t ldx, std::vector<Ent>& cold,
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

    // (Options::expert_pages) the CPU lane's entries from the caller's pool, kCpuLaneExperts experts at a time: the
    // pool keeps the blobs it gives for the layer in place while it gives no more than that many it does not hold.
    // Each group's entries computed as cpu_experts_mx computes them (an entry's rows do not depend on the others').
    // An expert the pool could not read: its rows are 0, said once.
    void cpu_experts_paged(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx) {
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
    // the CPU lane's entries of layer l (cpu_experts_paged) for the rows of x, ldx floats apart (0: g.dim)
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0) {
        if (ldx == 0) ldx = (size_t) g.dim;
        // an expert with more tokens than an entry holds takes adjacent entries (ent_cap)
        int64_t count = 0;
        for (size_t i = 0; i < ents.size(); ++i)
            count += i == 0 || ents[i].e != ents[i - 1].e;
        cpu_expert_count += count;
        cpu_experts_paged(l, x, ents, ldx);
    }
    /// Computes EXL3 experts held by the caller's RAM pool.
    void cpu_experts_exl3(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx) {
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

    // a device's router that had no pick to make (every score NaN: the gate's input went non-finite) since the last call: the
    // kernels took a stand-in expert so nothing read out of bounds, and this is how the request hears of it (the caller
    // polls it) - once per device in the log, and the caller ends the request with an error
    bool router_nonfinite() {
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

    // Adds weighted routings to the paged engine's adaptive counts, excluding MTP layers.
    void count_usage(const Layer& l, int e, float n = 1.0f) {
        if (!usage.empty() && !l.mtp)
            usage[(size_t) l.il * g.n_exp + e] += n;
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane() {
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

    // a GGUF's matrices (BF16 / MXFP4) to its own kernels, EXL3 ones to x3's
    static bool mv_any(const x3::MvJob* j, int n, cudaStream_t s) {
        bool g = false;
        for (int i = 0; i < n; ++i) g = g || j[i].m->fmt != 0;
        return g ? gg::mv(j, n, s) : x3::mv(j, n, s);
    }
    bool mv(Dev& d, std::initializer_list<x3::MvJob> jobs, std::string& err) {
        x3::MvJob j[x3::kMaxMvJobs];
        int n = 0;
        for (const x3::MvJob& x : jobs) j[n++] = x;
        if (!mv_any(j, n, d.st)) {
            err = "exl3 mv failed";
            return false;
        }
        return true;
    }
    bool mv_n(Dev& d, x3::MvJob* j, int n, std::string& err) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs)
            if (!mv_any(j + a, std::min(x3::kMaxMvJobs, n - a), d.st)) {
                err = "exl3 mv failed";
                return false;
            }
        return true;
    }
    // y[c] = x[c] W for the T activations x + c * x_cs (floats), into y + c * y_cs
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0,
                         float alpha = 1.0f) {
        x3::MvJob j;
        j.m = &m;
        j.xf = x;
        j.y = y;
        j.alpha = alpha;
        j.ncol = T;
        j.xf_cs = x_cs;
        j.y_cs = y_cs;
        return j;
    }

    // ---- the DSpark drafter (DeepSeek's DSparkBlock / forward_spec).  Its blocks' rings hold the trunk's main KV:
    // seed() turns a window's taps (the streams' mean entering layers 37..39) into it, through main_proj / main_norm
    // and each block's wkv / kv_norm / rope / FP8, for the positions the trunk kept.  draft() runs the block [y, noise x
    // (block - 1)] at positions pos .. through the blocks (each query attending to the ring and to the whole block),
    // the trunk's head, the markov bias token by token (greedy) and the confidence head.
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err) {
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

    /// Reconstructs the seated conversation after its dense weights or checkpoint returns.
    bool reseed_drafter_from_taps(int64_t position, std::string& err) {
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

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter) {
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
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err) {
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

    /// Runs one layer-major DSpark forward across eligible conversations, tiling only at kernel capacity.
    bool draft_conversations(std::vector<Model::DraftRequest>& requests,
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

    /// Samples one conversation's Markov chain from its rows in a shared head projection.
    bool sample_draft_rows(int row0, int y, int T, std::string& err, int int_offset = 0) {
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

    /// Retrieves the selected tokens and confidence head for one conversation's output rows.
    bool read_draft_rows(int row0, int T, std::string& err) {
        Dev& hd = devs[(size_t) head_dev];
        confidence(hd.x + (size_t) row0 * g.dim, d_mk_embed, d_ints, d_conf, T, g.dim, g.d_rank,
                   d_confv, hd.st);
        CK(cudaMemcpyAsync(h_idx, d_ints, (size_t) (T + 1) * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaMemcpyAsync(h_w, d_confv, (size_t) T * 4, cudaMemcpyDeviceToHost, hd.st));
        CK(cudaStreamSynchronize(hd.st));
        return true;
    }

    bool save_joint_draft(int session, int n, std::string& err) {
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

    // the token sample_tokens left in d.tok; -1 on a CUDA error, printed here (the callers reject a -1; Project Maya #619)
    int tok_back(Dev& d) {
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

    // the penalty rows of T picks (on the head device, its stream): row t the last h of toks[0 .. end0 + t), -1 in
    // front; the penalties into sp.  nullptr and h = 0 (sp untouched) for a request without penalties
    const int* pen_rows(int64_t end0, int T, strata::dsv41k::SamplerParams& sp, int& h) {
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

    // `hist_end`: the pick's penalty history ends there (toks[0 .. hist_end): the tokens up to its row's position)
    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias, int64_t hist_end) {
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

    // [y, drafts] at positions pos .. through the model: picks[i] gets the token that follows row i, for the rows up
    // to the first rejected draft (the rows after it get none), and the state holds every row until keep_verified.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify_rows(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                     uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err) {
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

    /// Picks the tokens of one conversation's verify rows from its logit rows, from row0 on, under draft_rule; p0 is
    /// the position of that window's first row (where the penalties' history of its picks ends).
    bool pick_verified_rows(int row0, int64_t p0, const std::vector<int>& drafts, float temperature, float top_p,
                            int top_k, float min_p, uint64_t seed, uint64_t counter, std::vector<int>& picks,
                            std::string& err) {
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

    // the window verify_rows ran keeps its first `keep` rows (at least its first): the rest rolled back, the drafter
    // seeded with the kept ones
    bool keep_verified(int keep, std::string& err) {
        return keep_window(verified_p0, verified_rows, keep, -1, err);
    }

    /// Commits a window's accepted prefix, using shared-pass scratch when row0 is nonnegative.
    bool keep_window(int64_t p0, int count, int keep, int row0, std::string& err) {
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

    /// Gives a trunk layer scratch for restoring state after rejected guesses, independently of DSpark weights.
    bool init_verify_layer(Layer& l, std::string& err, bool joint = false) {
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

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep, int row0 = -1) {
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

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) {
        return forward(&token, 1, want_logits ? 1 : 0, err);
    }

    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  kv_only (a long prompt's first pass, Run::stop): only what the layers
    // after it read of these rows - its compressed KV and index keys - and nothing of its own attention or experts
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool kv_only = false) {
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

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams() {
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

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const {
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

    // a run of chunks (start positions p0[k], sizes T[k], seqs seq[k]) on device di: its layers chunk after chunk, the
    // groups pipelined (a group takes chunk k + 1 while the next takes chunk k: the hand-offs and the sources' rows
    // two deep, each slot reused once its readers acknowledged it), the streamed experts' copies running ahead across
    // layers and chunks
    struct Run {
        std::vector<int64_t> p0;
        std::vector<int> T;
        std::vector<unsigned> seq;
        // the positions whose states before them the layers keep, in order, each into the layers' snapshot set of its
        // index: feed_mark's position before the run's last row
        std::vector<int64_t> snap_pos;
        // chunk [p0, p0 + T)'s rows among them, into wp
        void snaps(int64_t p0, int T, WinPlan& wp) const {
            wp.snap_n = 0;
            for (size_t i = 0; i < snap_pos.size() && wp.snap_n < kSnapN; ++i)
                if (snap_pos[i] >= p0 && snap_pos[i] < p0 + T) {
                    wp.snap_t[wp.snap_n] = (int) (snap_pos[i] - p0);
                    wp.snap_ix[wp.snap_n++] = (int) i;
                }
        }
        // a long prompt's skip (forward_chunks): the cut at row cut_r of chunk cut_k - the chunks before it go through
        // the layers before `yoco`, that layer as layer_chunk's kv_only and none after it; chunk cut_k's rows before
        // cut_r the same, its rows from cut_r on and the chunks after it through every layer (yoco -1: no skip)
        int yoco = -1, cut_k = 0, cut_r = 0;
        bool runs(int k, int il) const { return yoco < 0 || k >= cut_k || il <= yoco; }
        bool kv_only(int k, int il) const { return yoco >= 0 && k < cut_k && il == yoco; }
        int r0(int k, int il) const { return yoco >= 0 && k == cut_k && il >= yoco ? cut_r : 0; }   // the first row
    };
    // layer L0 over chunk k per a long prompt's skip (Run): its kv_only, or its rows from Run::r0 on - the chunk's
    // buffers seen from that row (the yoco layer: its kv_only over the rows before it first)
    bool run_layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
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

    // layer-major (c.lm: a run of several chunks, device di every trunk layer, the flow set up by run_dev_chunks):
    // layer after layer, each through all the chunks in order - a chunk's state to host memory after each of its
    // layers and back before the next - so a layer's experts are read from the files once for the whole run (the
    // layer cache keeps what a chunk read for the chunks after it), where chunk after chunk reads them once a chunk
    bool run_dev_layers(int di, const Run& run, std::string& err) {
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
    bool run_dev_chunks(int di, const Run& run, std::string& err) {
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

    // a prompt of n tokens that the flow could take (n >= chunk_min) takes the flow (can_chunk), else windows of 16.
    // The choice is made from the prompt's length alone (pick_flow_min): the same prompt takes the same path - and so
    // gives the same bits, the two paths round differently - in every session and on every restart.  Measured on 4
    // and 1 R9700 (first prompt of a run, windows against the flow, ms): 28 tokens 594 / 4966 (4 cards), 1094 / 5588
    // (1 card); 100: 1164 / 5375, 3279 / 7299; 300: 2831 / 5874, 9018 / 9803; 600: 5304 / 6808, 16969 / 11327; 1500:
    // 13325 / 8012, 41547 / 18187 - the flow pays a fixed ~5 s the first time (its buffers and GEMM libraries), then
    // wins from about 400 tokens up.
    // The length from which the flow is taken: one number for every box (ds4_pick_rule.hpp says why 96), DS4_PICK_MIN=N overrides.
    int pick_flow_min() {
        if (pick_min_ > 0) return pick_min_;
        pick_min_ = prompt_flow_min(std::getenv("DS4_PICK_MIN"));
        std::fprintf(stderr, "ds4: prompt path: the chunked pass from %d tokens, windows of 16 below\n", pick_min_);
        return pick_min_;
    }
    int pick_min_ = 0;
    // Gets whether DS4_PICK=adaptive lets every read the flow could take (n >= chunk_min) take it
    static bool is_pick_adaptive() {
        static const bool adaptive = [] {
            const char* e = std::getenv("DS4_PICK");
            return e != nullptr && std::strcmp(e, "adaptive") == 0;
        }();
        return adaptive;
    }
    // Gets whether a read of n tokens goes through the chunked path (DS4_PICK=adaptive: any the flow could take;
    // else those of pick_flow_min()'s length and up), and the paged
    // engine lends the path its buffers for them (Model::can_read_chunked). Keeping more than a window's logits
    // requires the chunked path even below the usual length threshold.
    bool can_chunk(int n, int logits_rows = 0) const {
        return chunk > 0 && n >= chunk_min && (logits_rows > kWin || is_pick_adaptive() || n >= pick_min_);
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err) {
        logits_all = false;
        for (int a = 0; a < n; a += kWin) {
            const int T = std::min(kWin, n - a);
            if (!forward(tokens + a, T, a + T == n ? logits_rows : 0, err)) return false;
        }
        return true;
    }
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never; any other number: whatever the cost; --prompt-order sets it); then its host memory:
    // the chunks' states (pinned, else pageable) and the layer cache, the largest layer's experts read from the files
    // if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB sets it) - at least one expert, or not layer-major
    bool lm_begin(const Run& run) {
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
    void lm_end() {
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

    // the first layer of a long prompt's second half (forward_chunks' two passes), or -1: the first with compress ratio 1
    // (V4.1: 20), every one after it ratio 1 too and owning no compressed KV or index keys of its own (they read its),
    // every trunk layer on one GPU.  The settings' MAYA_DS4_PROMPT_SKIP (STRATA_DS4_PROMPT_SKIP; DS4_YOCO for tests):
    // 1 (the default) the exact skip, 2 DeepSeek's own serving's (the last 128 rows only: faster, the answers shift a
    // little), 0 off
    static int yoco_mode() {
        static const int m = [] {
            const char* v = std::getenv("DS4_YOCO");
            if (v == nullptr) v = std::getenv("STRATA_DS4_PROMPT_SKIP");
            return v != nullptr ? std::atoi(v) : 1;
        }();
        return m;
    }
    // the rows of a long prompt's end that go through every layer (the skip leaves the others out past yoco)
    int yoco_keep(int yl) const {
        // (DS4_YOCO_KEEP=<rows>: fewer - DeepSeek's own serving replays the last 128 tokens through those layers
        //  (DeepInfer's "SWA bounded replay"), which shifts the answers a little; at least one window)
        static const int keep_env = std::getenv("DS4_YOCO_KEEP") ? std::atoi(std::getenv("DS4_YOCO_KEEP")) : 0;
        return yl <= 0 ? 0
               : keep_env > 0 ? std::max(keep_env, g.win)
               : yoco_mode() == 2 ? g.win : (g.nl - yl) * (g.win - 1) + g.win;
    }
    int yoco_layer() const {
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

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    // after: a piece of a longer prompt, its tokens still to come in later feeds (the skip's kept rows are the prompt's last).
    // snap_slot >= 0: the state before the last row is kept in checkpoint snap_slot (feed_mark).
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1, int after = 0) {
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

    // the state before position `at` from the layers' snapshot set ix (the drafter's blocks: as they are - the next
    // feed seeds them again from its own rows, past this position)
    bool snap_check(Check& ckp, int64_t at, int ix, std::string& err) {
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

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp, bool host_routed = false) {
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

    // the head over `rows` (<= 16) normed rows xn on the head's device d: each row's logits into d.logits, V apart
    bool compute_logits(Dev& d, const float* xn, int rows, std::string& err) {
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

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err) {
        Dev& d = devs[(size_t) di];
        cudaSetDevice(d.id);
        const int D = g.dim, H = g.hc;
        if (n_logits <= 0) return true;
        const int t0 = T - n_logits;
        hc_pre_norm(d.h + (size_t) t0 * H * D, d.pre + (size_t) t0 * 8, d.norm_w, D, H, n_logits, g.eps, d.x, d.st);
        return compute_logits(d, d.x, n_logits, err);
    }

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err) {
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

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1) {
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

    /// Verifies contiguous conversation windows in one layer-major pass.
    bool step_conversations(const std::vector<int>& tokens, const std::function<void(int)>& select,
                            const std::function<void(int)>& bind, const std::vector<int>& counts, std::string& err) {
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

    /// Captures row-local state before any pipeline stage may overwrite shared scratch.
    bool prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
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

    /// Queues a stage while each other stage retains its own device scratch.
    bool launch_group(int stage, int group, std::string& err) {
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

    /// Services the completion event used by the shared scheduler's stage owner.
    int poll_group(int stage, std::string& err) {
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

    /// Makes retained outputs available to sampling and the next single-conversation draft.
    bool finish_group(int group, const std::function<void(int)>& select, std::string& err) {
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

    /// Prepares attention positions and Engram inputs for synchronous windows or independent pipeline rows.
    bool prepare_conversation_rows(const std::vector<int>& tokens, const std::function<void(int)>& select,
                                  uint16_t* embeddings, uint8_t* engram, size_t capacity,
                                  int64_t* positions, WinPlan* plans, std::string& err,
                                  const int* window_counts = nullptr) {
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

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    void dbg_nf(const char* tag, int il, const float* dev, size_t n) {   // DBG: non-finite count of a device buffer
        const float* hp = (const float*) strata::gpu::host_view(dev);
        size_t nf = 0, first = 0; float mx = 0.0f;
        for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) { if (!nf) first = i; ++nf; } else if (std::fabs(v) > mx) mx = std::fabs(v); }
        std::fprintf(stderr, "DBG nf il=%d %s n=%zu nonfinite=%zu first=%zu max=%.4g\n", il, tag, n, nf, first, mx);
    }
    void dbg_h(const char* tag, Dev& d, int T) {   // DBG: max |h| of the window's residual stream, after the queue drains
        const float* hp = (const float*) strata::gpu::host_view(d.h);
        float mx = 0.0f; size_t nf = 0;
        const size_t n = (size_t) T * g.hc * g.dim;
        for (size_t i = 0; i < n; ++i) { const float v = hp[i]; if (!(v == v)) ++nf; else if (std::fabs(v) > mx) mx = std::fabs(v); }
        std::fprintf(stderr, "DBG hmax %s T=%d max=%.4g nonfinite=%zu h0=%g\n", tag, T, mx, nf, hp[0]);
    }
    bool forward_serial(int64_t p0, int T, std::string& err) {
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

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err) {
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

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const {
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

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err) {
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

    // a source's new cache rows [r0, r0 + n) on device `from` to the same rows of its other devices' copies, through
    // pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<KvRows>& copies, uint8_t* stage, cudaEvent_t ev, int64_t r0, int n, int from,
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

    // Queues the hand-off of the rows [g_first, g_first + ng) that layer l just wrote to source sr's compressed KV
    // (and, when l owns them, to its index keys): copies them to host-mapped `stage` for the devices outside l's group,
    // then sets *flag = seq.
    void put_new_rows(const Src& sr, const Layer& l, int64_t g_first, int ng, uint8_t* stage, unsigned* flag,
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

    // Queues device di's side of put_new_rows for the window or chunk of T tokens at p0: once *flag >= seq, copies
    // the rows of sr that di keeps from `stage`.  With `host_first` (the chunked prompt's threads) the host also waits
    // for the flag before it queues the device-side wait (see host_wait_at_least).
    void take_new_rows(const Src& sr, size_t di, int64_t p0, int T, uint8_t* stage, const unsigned* flag,
                       unsigned seq, cudaStream_t s, bool host_first = false) {
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

    // Appends to xs the copies of rows [r0, r0 + n) of `rows` to `stage` (out) or from it: one segment for each
    // block's part of them, in a new Xfer when the last one is full.
    static void add_row_copies(std::vector<Xfer>& xs, const KvRows& rows, int64_t r0, int n, uint8_t* stage, bool out) {
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

    /// Computes one layer for the shared window, with state selected independently for each attention row.
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        return run_attention(L0, mi, p0, T, wp, err) && run_moe(L0, mi, p0, T, wp, err);
    }

    /// Shares the dense projections across rows before and after their position-dependent attention.
    bool run_attention(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
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

    /// Updates the selected conversation's rings and compressors and reads its paged attention sources.  `r`: the row
    /// of the window (and of wp's per-row plans) that these T rows start at, past the rows of the runs before them.
    /// `batch`: if not null, run_attention rotates every run's queries, projects their indexer queries and runs the
    /// attention for every run at once; this decodes the run's selected rows into its scratch slots (from active_row)
    /// and puts its rows' AttnRows entries there.
    bool run_attention_state(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err, int r = 0,
                             AttnRows* batch = nullptr) {
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

    /// Rotates the attention output `o` of T rows at positions p0 .. (rpos: at rpos[t] each) back by their positions
    /// (DeepSeek's inverse RoPE on the output) and rounds it to FP8 as the reference does before layer member l's
    /// output projection.
    void finish_attention(const Layer& l, Dev& d, float* o, int64_t p0, int T, const int* rpos = nullptr) {
        Finish fo = get_rope(l, d, l.ahn, p0, T, rpos);
        fo.inverse = 1;
        fo.quant = 1;
        finish(o, T * l.ahn, g.hd, g.hd, fo, d.st);
    }

    /// Projects T rows' indexer queries (d.iq, rotated to positions p0, p0 + 1, .. or rpos[t] each, and rounded to
    /// FP4) and head weights (d.iw, rows 64 apart) from their q latents (d.qr) and attention inputs (d.x), for layer
    /// member l's top-k selection of compressed rows.
    bool project_indexer_queries(const Layer& l, Dev& d, int64_t p0, int T, const int* rpos, std::string& err) {
        if (!mv(d, {job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, (size_t) g.ix_heads * g.ix_hd)}, err))
            return false;
        Finish fi = get_rope(l, d, g.ix_heads, p0, T, rpos);
        fi.quant = 3;
        finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, d.st);
        gemv_f16_rows(l.i_wproj, d.x, (size_t) g.dim, g.ix_heads, g.dim, T,
                      1.0f / std::sqrt((float) g.ix_hd) / std::sqrt((float) g.ix_heads), d.iw, 64, d.st);
        return true;
    }

    /// Gets the Finish that rotates T rows of layer member l's vectors (vpp a row) by their positions: p0, p0 + 1, ..
    /// or, when rpos is given, rpos[t] each.
    Finish get_rope(const Layer& l, const Dev& d, int vpp, int64_t p0, int T, const int* rpos) const {
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

    /// Computes the shared and routed experts for all rows in this layer.
    bool run_moe(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
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

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter) {
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

    void save_profile() const {
        if (o.profile.empty() || o.profile_readonly) return;   // (--profile-readonly: a test keeps the file it read)
        std::string err;
        if (!strata::core::write_expert_counts(o.profile, (int64_t) prof.size() / g.n_exp, g.n_exp, prof, err))
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
    }

    std::vector<std::string> info_lines() const {
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

};

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

bool Model::prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
                          const std::function<void(int)>& bind, std::string& err) {
    return p_->prepare_group(group, tokens, select, bind, err);
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
    for (Dev& d : p_->devs) {
        cudaSetDevice(d.id);
        const cudaError_t result = cudaStreamSynchronize(d.st);
        if (result != cudaSuccess) {
            err = cudaGetErrorString(result);
            ok = false;
        }
    }
    return ok;
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
