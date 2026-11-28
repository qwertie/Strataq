// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_model_impl.hpp - the inside of ds4::Model (ds4_model.hpp), for the ds4_*.cu files of this folder only.
//
// Model::Impl holds everything the model keeps: the pack, the devices, the layers, the buffers and the conversation's
// state. Its methods are defined in files by job: ds4_load.cu loads the model and places its experts, ds4_prefill.cu
// reads a prompt in chunks, ds4_verify.cu runs a window of tokens through the layers (the decode, a draft's check and
// the paged engine's steps), ds4_cpu_lane.cu computes the experts the CPU holds, ds4_dspark.cu runs the DSpark
// drafter, ds4_engram.cu reads the Engram rows, ds4_vision_tower.cu runs the vision tower, ds4_checkpoints.cu keeps
// the state's checkpoints and restore points, and ds4_model.cu holds Model's calls, the sampling and the profile.
// The small helpers that run for every layer or token stay defined in the class, so that they are inlined as before.
#pragma once

#include "strata/base/gpu/managed_memory.hpp"
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

// The helpers and parts of the model that the ds4_*.cu files share.
namespace detail {

using strata::platform::env_tri;

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// the embedding rows' upload: asynchronous on the device's stream
inline cudaError_t emb_upload(void* dst, const void* src, size_t n, cudaStream_t s) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
}

inline size_t align256(size_t n) { return (n + 255) & ~(size_t) 255; }

// A run of chunks' device-side waits for another group (xfer_get, wait_flags: flag >= seq) are queued only once the
// host sees the flag up, so the kernel finds it up and never spins.  A wait kernel spinning on a device while its
// thread goes on queueing the layer stalled that thread's first cuBLAS call behind it, and with it the other
// devices' (two H100s, the IQ2_XS pack on a layer split: both threads stopped in cublasSgemm, hc_mix_chunk).
inline void host_wait_at_least(const unsigned* flag, unsigned seq) {
    for (int spin = 0; (int) (*(volatile const unsigned*) flag - seq) < 0; ++spin) {
        if (spin < 1000) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

// part p of np of `whole` (a multiple of 128) in 128-blocks, as even as they go (whole / 128 % np parts a block more:
// those from part rot on, cyclically - so a group's layers, each its own rot, even out what each member holds):
// tensor parallel's columns of an FFN or of the head - an even split is what whole / np gave before
inline void split128(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

// the RAM this process can still get (~0: not known), cgroup limits included
inline uint64_t ram_avail() {
    strata::core::detail::HostMemory m;
    const bool known = strata::core::detail::host_available_memory(m);
    // Windows supplies commit capacity on a successful read even when no physical RAM remains.
    return known || m.commit != ~uint64_t{0} ? m.available : ~uint64_t{0};
}

#if !defined(_WIN32) && !defined(STRATA_USE_HIP)   // (uma_free's only)
// whether GPU `dev` shares the PC's memory (a Spark's GB10, a Strix Halo, an APU); false when the runtime cannot say
inline bool gpu_integrated(int dev) {
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
inline size_t uma_free(int dev, size_t fr) {
#if !defined(_WIN32) && !defined(STRATA_USE_HIP)
    const uint64_t avail = gpu_integrated(dev) ? ram_avail() : ~uint64_t{0};
    if (avail != ~uint64_t{0})
        return strata::core::detail::get_uma_free_bytes(fr, avail);
#else
    (void) dev;
#endif
    return fr;
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

}  // namespace detail
using namespace detail;

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
    bool chunk_go(int k);   // chunk k may start on this device (several GPUs)
    bool halt_now() const {
        return devs.size() == 1 && halt.load(std::memory_order_acquire) && !yielding.load(std::memory_order_acquire);
    }
    // a run's early end on device di: the expert reads still in flight land first (they count down items the next run
    // clears), then its streams are idle
    bool halt_end(int di, std::string& err);
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
    void skip_auto_off(const char* why, double share);   // (the serve thread)
    // a layer's count into the window (the serve thread); once a token (its first skip layer) the switch off
    void skip_auto_tick(int li, int T, int heavy, int cold, int heavy_cpu);
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
    cudaEvent_t ek_ev();
    void prof_mark(const Dev& d, int stage);
    void prof_collect();
    void dump_layer(Dev& d, const float* h, int il, int64_t p0, int T);

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
    void dump_sub_host(const char* name, int il, int64_t p0, int T, const void* host, size_t nbytes);

    ~Impl();
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
    bool upload(Dev& d, const Tensor& t, void* dst, std::string& err);
    // a dense matrix in a ggml block type (x3::Mat fmt kFmtGG, row = k / 256 blocks): whole, or part of its rows / inputs
    bool mat_gg(Dev& d, const std::string& pre, const Tensor& t, int ggt, x3::Mat& m, std::string& err, size_t* acct, int part,
                int np, bool cols);
    // a GGUF's BF16 or Q8_0 matrix ([out][in]) onto the device, as mat() takes an EXL3 one: whole, or part `part` of its
    // columns (outputs: a range of its rows) or of its rows (inputs: a range of each row, whole Q8_0 blocks) - a
    // standalone [n][k] matrix
    bool mat_bf16(Dev& d, const std::string& pre, const Tensor& t, x3::Mat& m, std::string& err, size_t* acct, int part,
                  int np, bool cols);
    // a Q8_0 tensor's values as F32 (host)
    static std::vector<float> q8_values(const uint8_t* q, size_t n);
    // an EXL3 matrix onto the device: whole, or (np > 1) part `part` of its columns (cols) or of its rows - a standalone
    // matrix of its own (the rotations' 128-blocks stay whole)
    // (rot: split128's - an EXL3 matrix's uneven 128-block shares going round the members; a GGUF's BF16 / Q8_0 matrix
    // splits evenly, its groups only as wide as that goes)
    bool mat(Dev& d, const std::string& pre, x3::Mat& m, std::string& err, size_t* acct, int part = 0, int np = 1,
             bool cols = true, int rot = 0);
    // a float tensor (F32 / F16 / BF16) as F32 on the device
    bool f32(Dev& d, const std::string& name, float*& p, std::string& err, size_t* acct, size_t want = 0);
    bool f16raw(Dev& d, const std::string& name, uint16_t*& p, std::string& err, size_t* acct);

    // The markov bias's head as Q8_0 blocks of 32 (34 B: F16 d + 32 int8) instead of F16 values: the same bias sums with the
    // quantisation's error, half the bytes the drafter's five markov steps read (66 MB -> 33 MB) and half the memory.  The
    // pack is untouched: the F32 / F16 / BF16 tensor is quantised as it is loaded.
    bool mk_head_q8(Dev& d, const std::string& name, uint8_t*& p, std::string& err, size_t* acct);

    size_t mat_bytes(const std::string& pre) const;
    size_t tensor_bytes(const std::string& name) const {
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
    bool expert_layout_mx(Layer& l, std::string& err);
    // cpu_experts_iq / cpu_experts_mx take this layer's experts (the checks they make before computing); `why` if not
    bool lane_fmt_ok(const Layer& l, std::string& why) const;
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
    bool read_mx_file(const Layer& l, int e, uint8_t* blob) const;   // (the movers: never a slot that may change)

    bool expert_layout(Layer& l, std::string& err);

    // ---- loading
    bool load(const Options& opt, std::string& err);

    // the rest of the load (Model::place_experts): the prompt path's buffers, the CPU lane and the threads that serve
    // it, the layers' expert tables (set_expert_addresses fills them)
    bool finish_load(std::string& err);

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
    void warm_flow();
    // The warm-up runs once the model is loaded, on a thread of its own: it cannot hide inside the load; after it, a
    // person typing gives it the seconds it needs.  A chunked pass that starts first waits for it (warm_mu) - never
    // later than that pass would have been without it - and a warm-up that finds the flow already used does nothing.
    // DS4_WARM=0 turns it off.
    std::mutex warm_mu;
    std::atomic<bool> flow_used{false};
    std::thread warm_bg;
    void start_warm();
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
    double gpu_expert_share(double* exp_out = nullptr, double* room_out = nullptr, double* other_out = nullptr) const;
    // this device's host-to-device link (bytes a second): 32 MB from pinned memory, the best of three (~10 ms on PCIe 3 x16), once a
    // device; 0: not measured.  The copies and their events on the legacy stream, as the ordinary load copies: timed on a stream of
    // their own they left decode ~20% slower for the whole run on 2x CMP 170HX (PCIe Gen2), though nothing used that stream again
    // (Project Maya #609, mw00).  Only DS4_RING_MB=link asks for it.
    std::map<int, double> link_bw;
    double link_h2d(Dev& d);
    size_t ring_mb_auto() const;
    bool slab_auto_grew_ = false, slab_force_base_ = false;   // the automatic attention slab: grew this try / must not grow (it did not fit)
    // the chunk path's device buffers, laid out from base (nullptr: only their bytes counted): those of a fixed size
    // (the attention's and the indexer's slabs, the weights' FP16 reconstructions, Engram's slabs), then those of C
    // rows
    size_t lay_chunk(Chunk& c, int C, uint8_t* base) const;
    // the chunk path's buffers for chunks of C rows, laid out from `device` and `host` (nullptr: only their bytes
    // counted): on the GPU lay_chunk's and the ring the streamed experts land in; in page-locked memory the ring's
    // staging (once the load sized it: Chunk::S) at a 4 KiB bound for the unbuffered reads, and the embedding and
    // Engram rows of the two chunks in flight.  Gets the GPU's bytes; `host_bytes` the host's
    size_t lay_prompt(Chunk& c, int C, uint8_t* device, uint8_t* host, size_t& host_bytes) const;
    // lays device di's chunk path out for chunks of C rows (at most the load's chunk, prompt_ceiling) in
    // the lent `device` and `host` memory, the device's zeroed; C 0: waits until it no longer
    // reads them
    bool lay_out_prompt(int di, int C, uint8_t* device, uint8_t* host, std::string& err);
    // raises prompt_ceiling to C rows, growing the page-locked buffers the load sized for its chunk:
    // the embedding and Engram rows, the remote sources' staging and the hand-offs.  Between
    // prompts (no chunk laid out).  False (and `err`) when they cannot grow; the old ceiling then stays, as the
    // buffers that did grow still hold it
    bool raise_prompt_ceiling(int C, std::string& err);
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
    void chunk_dims(Chunk& c, int nm) const;
    bool alloc_chunks(std::string& err);

    // y[r][c] = x16[r] . W[:, c] (+ beta y) for r < rows: W reconstructed to FP16 in whole 128-column blocks, then the product.
    // The product prefers the engine's own RDNA3 WMMA kernel (src/prefill/wmma_gemm.cu) over hipBLAS: on gfx1151 hipBLAS
    // measures 4.97 TFLOPS at this call convention while that kernel measures 20-40 (BM 128 x BN 256, grouped tile order), and
    // the chunked prompt's dense GEMM is 11.1 s of a 4K prompt at the library's rate.  Same operands and the same values (the
    // k order differs, so it is rounding-level: the quality gate and the acceptance gate cover it); STRATA_DENSE_PFGEMM=0 keeps
    // hipBLAS for an A/B.
    bool gemm(Dev& d, Chunk& c, const x3::Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta,
              std::string& err);
    // ---- the busy experts (run_set): an expert of many rows as FP16 GEMMs.  The tiles decode its weights again for
    // every 64 of its rows (a prompt's chunk gives a popular expert hundreds); decoded once, to the FP16 values the tiles
    // multiply (iq::f16_rows), the library's GEMM takes them at its own speed.  DS4_FAT_ROWS=N (default 0, off; N:
    // the experts of N rows or more) takes the experts of N rows or more; the same products, only the sums' order differ
    // (3x V100, Maya-XS: 64 rows +6-8% on 8K / 32K prompts; KL vs the reference 0.276, the stock path 0.285 - the same;
    //  RTX 5090, ISTA-DAS 2.25 bpw: a 13K-token prompt 1,400 -> 1,625 tok/s at 64 rows, decode unchanged)
    static int fat_rows();
    bool fat_ready(Dev& d, Chunk& c, int D, int FF);
    // h16[lr0 + r] = FP16 clamped SwiGLU of the expert's gate and up over x16[row_tok[row + r]], r < n
    bool fat_gate_up(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int row, int n, int a, int D, int FF,
                     cudaStream_t s, std::string& err);
    // y[lr0 + r] = down(h16[lr0 + r]), r < n
    bool fat_down(Chunk& c, const x3::SetArgs& sa, const uint8_t* blob, int lr0, int n, int D, int FF, cudaStream_t s,
                  std::string& err);
    // y[r][c] = alpha x16[r] . W[c] for F16 weights W [n][k] (the indexer's weights_proj)
    bool gemm_f16w(Chunk& c, const uint16_t* W, int n, int k, const uint16_t* x16, int ldx, int rows, float alpha,
                   float* y, int ldy, std::string& err);
    // hc_mix over a chunk: the dots by an F32 GEMM, then the rms scale
    bool hc_mix_chunk(Dev& d, Chunk& c, const float* h, const float* fn, int T, float* mixes, std::string& err);

    // expert e of l -> its VRAM copy at dst, in the GPU layout, by DMA on the copy stream from `src` (its file region
    // read into pinned staging; null: from where the CPU lane reads it)
    bool stream_expert(const Layer& l, int m, int e, uint8_t* dst, cudaStream_t cs, std::string& err,
                       const uint8_t* src = nullptr);

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
    static int skip_unpicked_rows();
    // streamed experts: the batches' file regions read ahead of their copies - POSIX_FADV_WILLNEED, the kernel
    // reading them in the background in large requests, where the copies alone fault the pages in a few at a time -
    // at most DS4_READAHEAD_MB (2048) beyond the batch being copied; each region by its layer's first member alone
    void read_ahead(const Streamer& S, int di, const std::atomic<bool>& stop) const;
    bool copy_ahead(Chunk& c, Streamer& S, int di, std::string& err);

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
    void flow_push(Chunk& c, const Layer& l, int mi, int k, int e, int v);
    // the likely experts of (chunk k, layer il) for T tokens, first in its stream: the CPU lane's experts whose chance
    // to be picked at least once, 1 - (1 - share)^(picks), is at least DS4_SPEC_Q (0.9: the reads are what takes the
    // time, an unpicked one costs a whole read, and the routing is known a few ms after the layer before is done -
    // early reads pay where a chunk's long attention keeps the disk otherwise idle, and there nearly every expert is
    // that likely), likeliest first, at most what the staging and the ring hold (no more can be read before the layer
    // routes); the layer's virtual slots put them first, then its other CPU-lane experts in index order (the order the
    // rest of its picks are streamed in)
    // (a tensor-parallel group's members each decide their own stream and virtual slots - the same, from the same
    // profile and the same routing - and each reads only its part of every expert)
    void flow_spec(Chunk& c, int k, int il, int T);
    // (chunk k, layer il) routed (bounds b by virtual slot): the rest of its picks after its likely ones
    void flow_rest(Chunk& c, int k, int il, const std::vector<int>& b);
    // the next (chunk, layer) of the device after (k, il) in the run's order that streams experts (false: none) - a
    // long prompt's skip leaves out its early chunks' layers from Run::yoco on (kv_only there: no experts)
    bool flow_next(const Chunk& c, int k, int il, int& k2, int& il2) const;
    // the rows chunk k runs layer il on (a long prompt's skip: the cut chunk's from cut_r on, from the yoco layer on)
    static int fl_rows(const Chunk& c, int k, int il) {
        const int T = c.fl_T[(size_t) k];
        return c.fl_yoco >= 0 && k == c.fl_cut_k && il >= c.fl_yoco ? T - c.fl_cut_r : T;
    }
    bool flow_next1(const Chunk& c, int k, int il, int& k2, int& il2) const;
    // the reads (in order, each into its staging slot once that slot's last DMA is done) and the DMAs (in order, each
    // into its ring slot once that slot's last use is queued: the copy stream waits for it) as far as they can go now
    bool flow_pump(Chunk& c, std::string& err);
    // layer L0 of chunk k routed (bounds b, the device's copy d_bounds): its streamed experts as they land, the VRAM
    // ones in shares between them (the device works while the reads run)
    bool flow_layer(Dev& d, Chunk& c, Layer& L0, int mi, int k, int T, const std::vector<int>& b, std::string& err);

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
                 const int* b, int i0, int i1, int T, std::string& err, const unsigned long long* tab = nullptr);

    std::vector<std::unique_ptr<ExpertHop>> hops; ///< Indexed by the shared engine's card index.
    const int32_t* hop_holders = nullptr; ///< Published only between decode windows.

    bool load_dense(int di, std::string& err, bool drafter_only = false, int layer_only = -1);

    void set_vm(Layer& l, int e, const uint8_t* b);
    /// Gets layer l's experts hottest first by the profile, then by index.
    std::vector<int> get_heat_order(const Layer& l) const;
    /// Readies the layers (the trunk's and the DSpark blocks') for experts that set_expert_addresses places
    /// (Options::expert_pages): none on the GPU and no copy of the CPU lane's own, each layer's slot table allocated.
    bool place_paged(std::string& err);
    /// Makes the GPU compute the experts at device addresses `at` ([layer][expert]; null: the CPU lane's) from the
    /// next window on: their slots in by_heat's order, the decode's table (vram_tab) and the prompt's (slot_tab).
    bool set_expert_addresses(const std::vector<const uint8_t*>& at, std::string& err, int stage);

    // where the first `n` layers' experts lie in the files (the trunk's, then the DSpark blocks')
    std::vector<ExpertFileLayout> get_expert_files(size_t n) const;

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
    std::vector<LaneCfg> lane_cfgs() const;
    std::vector<std::unique_ptr<Pool>> pools_for(const LaneCfg& c) const;
    // the CPU lane's workers: lane_cfgs' first (cores), or the one STRATA_DS4_CPU_LANE=<name> asks for
    void make_pools();

    void reset();

    // the state at pos into checkpoint `slot` (the window rings and the compressors' partial groups: the rest of the
    // state is indexed by position, and what lies past a restored position is written again)
    void checkpoint(int slot) { snap_into(ck[slot & 1]); }
    void snap_into(Check& c);

    // the restore points a conversation keeps besides its checkpoints (MAYA_DS4_RESTORE_POINTS, 4; 0: a request's
    // marks= and shared= are not kept)
    static int marks_cap();
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
    std::vector<int64_t> check_positions() const;
    // the checkpoints a slot keeps, as best_check() numbers them (each with its role: 0, 1, the anchor, a mark)
    // (the marks least recently used first)
    std::vector<int> slot_checks(int64_t upto = -1) const;
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

    bool restore(int64_t limit);


    // Makes the position and the Engram history follow `toks`, after another conversation's tokens took their place
    // (swap_conversation).
    void set_position();

    // Model::get_state_problem: the first kept checkpoint of `s` whose rings or partial groups are not this model's
    std::string get_state_problem(const ConversationState& s) const;
    // why checkpoint `c` cannot go back to the GPU: a layer count, ring or partial group not this model's ("": it can)
    std::string get_check_problem(const Check& c) const;

    // Model::swap_conversation
    void swap_conversation(ConversationState& s);

    std::string slot_identity() const;

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
    bool read_bf16(const Tensor& t, size_t n, void* dst);
    bool load_vision(std::string& err);
    bool encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err);
    // the row of a picture's span at position p (null: p is in none)
    const uint16_t* image_row(int64_t p) const;
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
    bool vl_bits(int64_t p0, int T, VlBits& b) const;

    // ---- Engram: a window's rows of one table (Engram layer li), tokens p0 .. p0 + T - 1 into dst ([token] of [cols]
    // [hd] codes then the scales; nullptr: h_eg).  eg_issue serves the rows already fetched from the row cache (and a
    // row the window needs twice from one read) and hands the rest to the reader's threads; eg_finish waits for them,
    // copies the repeats and keeps the new rows in the cache.  The rows are the file's bytes either way.
    void eg_issue(int64_t p0, int T, int li, uint8_t* dst, EgBatch& b);
    // b's rows all in place (any thread; a group's members may ask at once: one finishes, the others wait for it)
    void eg_finish(EgBatch& b);
    void eg_stats();
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
    void cpu_experts_mx(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord);
    // ... the low-bit ones (ds4_iq's types): as above, through ggml-cpu's dots (gg::iq_cpu_*) - each token's input
    // quantized once to the weight type's activation (Q8_K) in its c_xh row's bytes, then (h whole) its h the same
    void cpu_experts_iq(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx, const int* ord);

    /// Computes the lent entries in one transfer each way, leaving the others for the CPU lane.
    bool run_expert_hop(const Layer& l, const float* x, size_t ldx, std::vector<Ent>& cold,
                        std::string& err);

    // (Options::expert_pages) the CPU lane's entries from the caller's pool, kCpuLaneExperts experts at a time: the
    // pool keeps the blobs it gives for the layer in place while it gives no more than that many it does not hold.
    // Each group's entries computed as cpu_experts_mx computes them (an entry's rows do not depend on the others').
    // An expert the pool could not read: its rows are 0, said once.
    void cpu_experts_paged(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx);
    // the CPU lane's entries of layer l (cpu_experts_paged) for the rows of x, ldx floats apart (0: g.dim)
    void cpu_experts(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx = 0);
    /// Computes EXL3 experts held by the caller's RAM pool.
    void cpu_experts_exl3(const Layer& l, const float* x, const std::vector<Ent>& ents, size_t ldx);

    // a device's router that had no pick to make (every score NaN: the gate's input went non-finite) since the last call: the
    // kernels took a stand-in expert so nothing read out of bounds, and this is how the request hears of it (the caller
    // polls it) - once per device in the log, and the caller ends the request with an error
    bool router_nonfinite();

    // Adds weighted routings to the paged engine's adaptive counts, excluding MTP layers.
    void count_usage(const Layer& l, int e, float n = 1.0f) {
        if (!usage.empty() && !l.mtp)
            usage[(size_t) l.il * g.n_exp + e] += n;
    }

    // the decode path's host side: each published routing in turn - its CPU lane experts (if any) into the answer's
    // part, then the answer's seq.  Spins while requests come (a token's 40 layers), naps after 2 ms without one.
    void serve_lane();

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
    bool seed_drafter(int64_t p0, int n, int row0, std::string& err);

    /// Reconstructs the seated conversation after its dense weights or checkpoint returns.
    bool reseed_drafter_from_taps(int64_t position, std::string& err);

    bool draft(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err, float temperature, float top_p,
               int top_k, float min_p, uint64_t seed, uint64_t counter);
    bool draft_(int y, std::vector<int>& out, std::vector<float>& conf, std::string& err);

    /// Runs one layer-major DSpark forward across eligible conversations, tiling only at kernel capacity.
    bool draft_conversations(std::vector<Model::DraftRequest>& requests,
                             const std::function<void(int)>& select,
                             const std::function<void(int)>& bind, std::string& err);

    /// Samples one conversation's Markov chain from its rows in a shared head projection.
    bool sample_draft_rows(int row0, int y, int T, std::string& err, int int_offset = 0);

    /// Retrieves the selected tokens and confidence head for one conversation's output rows.
    bool read_draft_rows(int row0, int T, std::string& err);

    bool save_joint_draft(int session, int n, std::string& err);

    // the token sample_tokens left in d.tok; -1 on a CUDA error, printed here (the callers reject a -1; Project Maya #619)
    int tok_back(Dev& d);

    // the penalty rows of T picks (on the head device, its stream): row t the last h of toks[0 .. end0 + t), -1 in
    // front; the penalties into sp.  nullptr and h = 0 (sp untouched) for a request without penalties
    const int* pen_rows(int64_t end0, int T, strata::dsv41k::SamplerParams& sp, int& h);

    // `hist_end`: the pick's penalty history ends there (toks[0 .. hist_end): the tokens up to its row's position)
    int sample_row(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter,
                   const float* bias, int64_t hist_end);

    // [y, drafts] at positions pos .. through the model: picks[i] gets the token that follows row i, for the rows up
    // to the first rejected draft (the rows after it get none), and the state holds every row until keep_verified.
    // Greedy: the drafts the trunk's argmax agrees with.  Sampled: the speculative-sampling rule over the request's own
    // chain (top_k, top_p, min_p, temperature): draft d stands with probability min(1, p(d) / q(d)), q the distribution
    // it was drawn from (the drafter's argmax: a point at d, so p(d)), a rejected one is replaced by a draw from the
    // residual max(0, p - q) - so each kept token is distributed as the trunk's own draw.
    bool verify_rows(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                     uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err);

    /// Picks the tokens of one conversation's verify rows from its logit rows, from row0 on, under draft_rule; p0 is
    /// the position of that window's first row (where the penalties' history of its picks ends).
    bool pick_verified_rows(int row0, int64_t p0, const std::vector<int>& drafts, float temperature, float top_p,
                            int top_k, float min_p, uint64_t seed, uint64_t counter, std::vector<int>& picks,
                            std::string& err);

    // the window verify_rows ran keeps its first `keep` rows (at least its first): the rest rolled back, the drafter
    // seeded with the kept ones
    bool keep_verified(int keep, std::string& err) {
        return keep_window(verified_p0, verified_rows, keep, -1, err);
    }

    /// Commits a window's accepted prefix, using shared-pass scratch when row0 is nonnegative.
    bool keep_window(int64_t p0, int count, int keep, int row0, std::string& err);

    /// Gives a trunk layer scratch for restoring state after rejected guesses, independently of DSpark weights.
    bool init_verify_layer(Layer& l, std::string& err, bool joint = false);

    // a verified window keeps its first `keep` positions: the ring rows and the compressors' state of the rest undone
    // (the compressed KV and the index keys past them are written again when those positions come)
    void rollback(int64_t p0, int T, int keep, int row0 = -1);

    // ---- a window: T tokens at positions pos .. pos + T - 1 through every layer; the logits of its last `logits` rows
    bool step(int token, bool want_logits, std::string& err) {
        return forward(&token, 1, want_logits ? 1 : 0, err);
    }

    // layer L0 for a chunk of T tokens at p0 as group member mi: layer()'s arithmetic over the chunk's rows (the
    // projections as GEMMs, the attention and the indexer in slabs, the routed experts as sorted rows: the VRAM ones in
    // place, the CPU lane's streamed in batches).  kv_only (a long prompt's first pass, Run::stop): only what the layers
    // after it read of these rows - its compressed KV and index keys - and nothing of its own attention or experts
    bool layer_chunk(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, Streamer& S, std::string& err,
                     bool kv_only = false);

    // the CPU lane's experts of every layer, in index order, and each expert's virtual slot (before a chunk)
    void plan_streams();

    // a chunk's hand-off between groups: h | pre | topk | cand of a chunk of T rows
    Xfer chunk_xfer(uint8_t* host, Chunk& c, int T, bool put) const;

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
                   const Run& run, int k, int il);

    // layer-major (c.lm: a run of several chunks, device di every trunk layer, the flow set up by run_dev_chunks):
    // layer after layer, each through all the chunks in order - a chunk's state to host memory after each of its
    // layers and back before the next - so a layer's experts are read from the files once for the whole run (the
    // layer cache keeps what a chunk read for the chunks after it), where chunk after chunk reads them once a chunk
    bool run_dev_layers(int di, const Run& run, std::string& err);
    bool run_dev_chunks(int di, const Run& run, std::string& err);

    // a prompt of n tokens that the flow could take (n >= chunk_min) takes the flow (can_chunk), else windows of 16.
    // The choice is made from the prompt's length alone (pick_flow_min): the same prompt takes the same path - and so
    // gives the same bits, the two paths round differently - in every session and on every restart.  Measured on 4
    // and 1 R9700 (first prompt of a run, windows against the flow, ms): 28 tokens 594 / 4966 (4 cards), 1094 / 5588
    // (1 card); 100: 1164 / 5375, 3279 / 7299; 300: 2831 / 5874, 9018 / 9803; 600: 5304 / 6808, 16969 / 11327; 1500:
    // 13325 / 8012, 41547 / 18187 - the flow pays a fixed ~5 s the first time (its buffers and GEMM libraries), then
    // wins from about 400 tokens up.
    // The length from which the flow is taken: one number for every box (ds4_pick_rule.hpp says why 96), DS4_PICK_MIN=N overrides.
    int pick_flow_min();
    int pick_min_ = 0;
    // Gets whether DS4_PICK=adaptive lets every read the flow could take (n >= chunk_min) take it
    static bool is_pick_adaptive();
    // Gets whether a read of n tokens goes through the chunked path (DS4_PICK=adaptive: any the flow could take;
    // else those of pick_flow_min()'s length and up), and the paged
    // engine lends the path its buffers for them (Model::can_read_chunked). Keeping more than a window's logits
    // requires the chunked path even below the usual length threshold.
    bool can_chunk(int n, int logits_rows = 0) const {
        return chunk > 0 && n >= chunk_min && (logits_rows > kWin || is_pick_adaptive() || n >= pick_min_);
    }
    // n tokens through windows of 16, the last one's logits_rows rows of logits
    bool feed_windows(const int* tokens, int n, int logits_rows, std::string& err);
    // a run layer-major? - one device holding every layer on the flow, several chunks, and what the chunks after the
    // first would read from the files again more than the chunks' states moved to host memory and back at every layer
    // (DS4_LAYER_MAJOR=0: never; any other number: whatever the cost; --prompt-order sets it); then its host memory:
    // the chunks' states (pinned, else pageable) and the layer cache, the largest layer's experts read from the files
    // if RAM allows (ram_avail() less kRamMargin; DS4_LAYER_CACHE_MB sets it) - at least one expert, or not layer-major
    bool lm_begin(const Run& run);
    void lm_end();

    // the first layer of a long prompt's second half (forward_chunks' two passes), or -1: the first with compress ratio 1
    // (V4.1: 20), every one after it ratio 1 too and owning no compressed KV or index keys of its own (they read its),
    // every trunk layer on one GPU.  The settings' MAYA_DS4_PROMPT_SKIP (STRATA_DS4_PROMPT_SKIP; DS4_YOCO for tests):
    // 1 (the default) the exact skip, 2 DeepSeek's own serving's (the last 128 rows only: faster, the answers shift a
    // little), 0 off
    static int yoco_mode();
    // the rows of a long prompt's end that go through every layer (the skip leaves the others out past yoco)
    int yoco_keep(int yl) const;
    int yoco_layer() const;

    // n tokens at positions pos .. in chunks of equal size (at most `chunk`), pipelined over the groups; the logits of
    // the last min(logits_rows, 16) rows (logits_rows > 16: a run of one chunk, every row's into chunk_logits)
    // after: a piece of a longer prompt, its tokens still to come in later feeds (the skip's kept rows are the prompt's last).
    // snap_slot >= 0: the state before the last row is kept in checkpoint snap_slot (feed_mark).
    bool forward_chunks(const int* tk, int n, int logits_rows, std::string& err, int snap_slot = -1, int after = 0);

    // the state before position `at` from the layers' snapshot set ix (the drafter's blocks: as they are - the next
    // feed seeds them again from its own rows, past this position)
    bool snap_check(Check& ckp, int64_t at, int ix, std::string& err);

    // the window's plan, in window order (the indexer's and the candidate source's sizes carry from window to window)
    void plan_window(int64_t p0, int T, WinPlan& wp, bool host_routed = false);

    // the head over `rows` (<= 16) normed rows xn on the head's device d: each row's logits into d.logits, V apart
    bool compute_logits(Dev& d, const float* xn, int rows, std::string& err);

    // the head's rows (the last n_logits of the window) on head group member di (its h / pre: the group's)
    bool head_rows_dev(int di, int T, std::string& err);

    // the DSpark drafter's taps: the streams' mean entering target layer il, on device d (after the last seeding read
    // h_taps)
    bool tap(Dev& d, int il, int T, std::string& err);

    bool forward(const int* tk, int T, int logits_rows, std::string& err, int seed_rows = -1);

    /// Verifies contiguous conversation windows in one layer-major pass.
    bool step_conversations(const std::vector<int>& tokens, const std::function<void(int)>& select,
                            const std::function<void(int)>& bind, const std::vector<int>& counts, std::string& err);

    /// Captures row-local state before any pipeline stage may overwrite shared scratch.
    bool prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
                       const std::function<void(int)>& bind, std::string& err);

    /// Queues a stage while each other stage retains its own device scratch.
    bool launch_group(int stage, int group, std::string& err);

    /// Services the completion event used by the shared scheduler's stage owner.
    int poll_group(int stage, std::string& err);

    /// Makes retained outputs available to sampling and the next single-conversation draft.
    bool finish_group(int group, const std::function<void(int)>& select, std::string& err);

    /// Prepares attention positions and Engram inputs for synchronous windows or independent pipeline rows.
    bool prepare_conversation_rows(const std::vector<int>& tokens, const std::function<void(int)>& select,
                                  uint16_t* embeddings, uint8_t* engram, size_t capacity,
                                  int64_t* positions, WinPlan* plans, std::string& err,
                                  const int* window_counts = nullptr);

    // one thread: the layers device after device, the hand-offs through pinned memory and events
    void dbg_nf(const char* tag, int il, const float* dev, size_t n);   // DBG: non-finite count of a device buffer
    void dbg_h(const char* tag, Dev& d, int T);   // DBG: max |h| of the window's residual stream, after the queue drains
    bool forward_serial(int64_t p0, int T, std::string& err);

    // threads: each device's thread enqueues its layers (a group's members each their share) at once, the
    // devices waiting on each other through host-mapped flags (the window's seq)
    bool forward_par(int64_t p0, int T, std::string& err);

    // the hand-off between groups: h | pre | topk | cand of device d <-> the host-mapped buffer
    Xfer hand_xfer(uint8_t* host, Dev& d, int T, bool put) const;

    bool run_dev(int di, int64_t p0, int T, const WinPlan& wp, std::string& err);

    // a source's new cache rows [r0, r0 + n) on device `from` to the same rows of its other devices' copies, through
    // pinned memory: the copy out on the source's stream, each copy in on its device's stream after it
    bool replicate(std::vector<KvRows>& copies, uint8_t* stage, cudaEvent_t ev, int64_t r0, int n, int from,
                   std::string& err);

    // Queues the hand-off of the rows [g_first, g_first + ng) that layer l just wrote to source sr's compressed KV
    // (and, when l owns them, to its index keys): copies them to host-mapped `stage` for the devices outside l's group,
    // then sets *flag = seq.
    void put_new_rows(const Src& sr, const Layer& l, int64_t g_first, int ng, uint8_t* stage, unsigned* flag,
                      unsigned seq, unsigned* cnt, cudaStream_t s);

    // Queues device di's side of put_new_rows for the window or chunk of T tokens at p0: once *flag >= seq, copies
    // the rows of sr that di keeps from `stage`.  With `host_first` (the chunked prompt's threads) the host also waits
    // for the flag before it queues the device-side wait (see host_wait_at_least).
    void take_new_rows(const Src& sr, size_t di, int64_t p0, int T, uint8_t* stage, const unsigned* flag,
                       unsigned seq, cudaStream_t s, bool host_first = false);

    // Appends to xs the copies of rows [r0, r0 + n) of `rows` to `stage` (out) or from it: one segment for each
    // block's part of them, in a new Xfer when the last one is full.
    static void add_row_copies(std::vector<Xfer>& xs, const KvRows& rows, int64_t r0, int n, uint8_t* stage, bool out);

    /// Computes one layer for the shared window, with state selected independently for each attention row.
    bool layer(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err) {
        return run_attention(L0, mi, p0, T, wp, err) && run_moe(L0, mi, p0, T, wp, err);
    }

    /// Shares the dense projections across rows before and after their position-dependent attention.
    bool run_attention(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err);

    /// Updates the selected conversation's rings and compressors and reads its paged attention sources.  `r`: the row
    /// of the window (and of wp's per-row plans) that these T rows start at, past the rows of the runs before them.
    /// `batch`: if not null, run_attention rotates every run's queries, projects their indexer queries and runs the
    /// attention for every run at once; this decodes the run's selected rows into its scratch slots (from active_row)
    /// and puts its rows' AttnRows entries there.
    bool run_attention_state(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err, int r = 0,
                             AttnRows* batch = nullptr);

    /// Rotates the attention output `o` of T rows at positions p0 .. (rpos: at rpos[t] each) back by their positions
    /// (DeepSeek's inverse RoPE on the output) and rounds it to FP8 as the reference does before layer member l's
    /// output projection.
    void finish_attention(const Layer& l, Dev& d, float* o, int64_t p0, int T, const int* rpos = nullptr);

    /// Projects T rows' indexer queries (d.iq, rotated to positions p0, p0 + 1, .. or rpos[t] each, and rounded to
    /// FP4) and head weights (d.iw, rows 64 apart) from their q latents (d.qr) and attention inputs (d.x), for layer
    /// member l's top-k selection of compressed rows.
    bool project_indexer_queries(const Layer& l, Dev& d, int64_t p0, int T, const int* rpos, std::string& err);

    /// Gets the Finish that rotates T rows of layer member l's vectors (vpp a row) by their positions: p0, p0 + 1, ..
    /// or, when rpos is given, rpos[t] each.
    Finish get_rope(const Layer& l, const Dev& d, int vpp, int64_t p0, int T, const int* rpos) const;

    /// Computes the shared and routed experts for all rows in this layer.
    bool run_moe(Layer& L0, int mi, int64_t p0, int T, const WinPlan& wp, std::string& err);

    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);

    void save_profile() const;

    std::vector<std::string> info_lines() const;

};

}  // namespace strata::ds4
