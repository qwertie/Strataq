#include "strata/base/gpu/managed_memory.hpp"
// src/models/qwen38/model.cpp - Model::load, Qwen3.8's model loader (it was inline in main() in
// src/program/generate.cpp), and the helpers it shares with that program's serve loop.
#include "strata/models/qwen38/model.hpp"
#include "strata/common/spec/window_planner.hpp"

#include "strata/base/platform/env.hpp"
#include "strata/base/platform/memory.hpp"
#include "strata/base/gpu/gfx_arch.hpp"
#include "strata/base/gpu/device.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/common/experts/expert_profile.hpp"
#include "strata/common/state/conversation_file.hpp"
#include "strata/common/state/conversation_memory.hpp"
#include "strata/common/kernels/coupled_draft.hpp"
#include "strata/common/kernels/expert_nibble.hpp"
#include "strata/common/experts/auto_cache.hpp"
#include "strata/base/gpu/gpu_report.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/models/qwen38/ple_residency.hpp"
#include "strata/models/qwen38/stage_reserve.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/base/gpu/pinned.hpp"
#include "strata/base/gpu/pin_budget.hpp"
#include "strata/common/experts/uma_cache.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/models/qwen38/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/models/qwen38/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/common/kernels/cpu/expert.hpp"
#include "strata/common/kernels/cpu/pool.hpp"
#include "strata/base/platform/aux_cpus.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/kernels/cpu/hadamard.hpp"
#include "strata/models/qwen38/kernels/ngram.hpp"
#include "strata/common/kernels/iq_kernels.hpp"
#include "strata/models/common/kernels/s2_expert_grouped.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/models/qwen38/kernels/shared_expert.hpp"
#include "strata/models/common/kernels/native_moe.hpp"
#include "strata/models/qwen38/kernels/native_gdn.hpp"
#include "strata/models/common/kernels/native_router.hpp"
#include "strata/models/qwen38/kernels/native_qsa.hpp"
#include "strata/models/qwen38/kernels/native_qsa_indexer.hpp"
#include "strata/models/qwen38/kernels/native_rope.hpp"
#include "strata/common/kernels/mrope.hpp"
#include "strata/models/qwen38/kernels/qsa.hpp"
#include "strata/models/qwen38/native_head.hpp"
#include "strata/models/qwen38/verify.hpp"
#include "strata/models/qwen38/mtp.hpp"
#if defined(STRATA_HAVE_DFLASH2)
#include "strata/models/qwen38/draft_cap.hpp"
#endif
#include "strata/models/qwen38/prefill/kva.hpp"
#include "strata/models/qwen38/prefill/prefill.hpp"
#include "strata/models/qwen38/prefill/share_rules.hpp"
#include "strata/common/experts/ram_tier.hpp"
#if defined(STRATA_PREFILL_MMQ)
#include "strata/common/kernels/prefill/moe_mmq.hpp"
#endif
#include "strata/models/qwen38/native_dense.hpp"
#include "strata/common/experts/layer_split.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/models/qwen38/batch_groups.hpp"
#include "strata/models/qwen38/kernels/cvec.hpp"
#include "strata/base/gpu/emulate.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/common/artifact/gguf_reader.hpp"
#include "strata/models/common/family_layout.hpp"
#include "strata/common/spec/draft_vocab.hpp"
#include "strata/common/kernels/prefill/gemm.hpp"
#include "strata/core/gfx12_chunk.hpp"
#include "strata/core/lookup_default.hpp"
#include "strata/core/pcie_share_rule.hpp"
#include "strata/core/spill_gate.hpp"
#include "strata/common/experts/split_walk.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#include <sys/stat.h>
#if !defined(__APPLE__)   // major() / minor(): <sys/types.h> on macOS
#include <sys/sysmacros.h>
#endif
#endif

#include <cuda_runtime.h>
#if defined(STRATA_GPU_METAL)
namespace strata::metal_engine {
void set_model_shard(const char* shard1);   // metal/src/engine/window.cpp
void set_batch_slots(int n);   // metal/src/engine/window.cpp: how many --batch slots the window keeps state for
std::string draft_vocab_default();   // metal/src/engine/window.cpp: the packaged default draft subset (data/draft_vocab.bin)
bool draft_vocab_whole_head();       // STRATA_METAL_DRAFT_VOCAB=full
}
#endif

#include <array>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iostream>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace strata::qwen38 {
namespace {
using Clock = std::chrono::steady_clock;
namespace nib = strata::core::nibble;
double g_last_h2d_gbps = -1.0;   // the last probe's reading, for the prompt planner (#2824)
// HIP: the thread that warms hipBLAS's libraries while the expert arena loads (see warm_gemm_libraries); joined at exit.
std::thread g_warm_thr;
struct WarmJoin { ~WarmJoin() { if (g_warm_thr.joinable()) g_warm_thr.join(); } } g_warm_join;

bool under_wddm();
std::string vram_batch_hint(const std::string& err, int batch);
void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out);
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary,
                          uint64_t& digest, std::string& err);
bool mtp_first_default_on();
bool pcie_auto_fork_arch_ok();
double probe_pcie_h2d_gbps(std::string* samples = nullptr);
double pcie_frac_for_gbps(double gbps, double base);
strata::kernels::cpu::PoolAffinity measure_pool_affinity(strata::core::ExpertSource* src, int workers, bool host_works,
                                                         int tasks);
double ple_direct_rows_per_s(strata::kernels::PleTable& t);
bool pcie_share_gfx12_default();
uint64_t nibble_max_slot_bytes();
std::string nibble_refusal(const Options& o);
void print_startup_gpu_report(const std::vector<int>& devices);
}  // namespace

int Model::load() {
    strata::core::set_coupled_draft(o.coupled_draft);
    {   // --host-core / STRATA_HOST_CORE, before the pool and the session pin any thread
        std::string hc = o.host_core;
        if (hc.empty())
            if (const char* e = std::getenv("STRATA_HOST_CORE")) hc = e;
        if (hc == "last") strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Last);
        if (hc == "sibling") strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Sibling);
    }
    {   // --pool-cores / STRATA_POOL_CORES (experimental, off by default), before the pool is built.  A list that cannot be read is
        // said and left out: the pool keeps its default placement (recommend, never force).
        std::string pc = o.pool_cores;
        if (pc.empty())
            if (const char* e = std::getenv("STRATA_POOL_CORES")) pc = e;
        if (!pc.empty()) {
            std::vector<int> cpus;
            std::string perr;
            if (strata::kernels::cpu::parse_pool_cores(pc, cpus, perr)) strata::kernels::cpu::set_pool_cores(cpus);
            else std::fprintf(stderr, "strata generate: --pool-cores: %s; the pool uses its default placement\n", perr.c_str());
        }
    }
    {   // --aux-cpus / STRATA_AUX_CPUS: read the CPUs this process may use now, before any thread is pinned; the plan
        // follows once the pool exists (it needs the workers' CPUs)
        std::string ac = o.aux_cpus;
        if (ac.empty())
            if (const char* e = std::getenv("STRATA_AUX_CPUS")) ac = e;
        std::string aerr;
        if (!strata::aux_cpus::configure(ac, aerr)) {
            std::fprintf(stderr, "strata generate: STRATA_AUX_CPUS: %s\n", aerr.c_str());
            return 2;
        }
    }
    strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    {   // the RAM guard (parking, session save/restore): --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
        int64_t mib = o.memory_limit_mib;
        if (mib < 0) {
            const char* env = std::getenv("STRATA_MEMORY_LIMIT_MIB");
            if (env != nullptr && env[0] != '\0') {
                const size_t len = std::strlen(env);
                const auto parsed = std::from_chars(env, env + len, mib);
                if (parsed.ec != std::errc{} || parsed.ptr != env + len || mib < 0 || mib > INT64_MAX / (1024 * 1024)) {
                    std::fprintf(stderr, "STRATA_MEMORY_LIMIT_MIB needs a nonnegative integer within range (got \"%s\")\n", env);
                    return 2;
                }
            } else mib = 0;
        }
        strata::core::set_memory_limit_bytes((uint64_t) mib << 20);
        if (o.serve || mib > 0) {
            constexpr double GiB = 1073741824.0;
            const auto mem = strata::core::sample_host_memory();
            if (!mem) {
                std::fprintf(stderr, "strata generate: memory guard: the available RAM cannot be read; parking and session "
                                     "save/restore are refused\n");
            } else {
                std::fprintf(stderr, "strata generate: memory guard: limit %.1f GiB (source: %s), current %.1f GiB, "
                                     "reclaimable cache credited %.1f GiB; available %.1f GiB (MemAvailable %.1f GiB%s), "
                                     "floor %lld MiB\n",
                             (double) mem->limit / GiB, strata::core::memory_source_name(mem->source),
                             (double) mem->current / GiB, (double) mem->credit / GiB, (double) mem->available / GiB,
                             (double) mem->mem_available / GiB,
                             mib > 0 ? (", --memory-limit-mib " + std::to_string((long long) mib)).c_str() : "",
                             (long long) o.conversation_cache_min_free_mib);
            }
        }
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // STRATA_PRIMARY_DEVICE: which VISIBLE device the engine's own session runs on.  The Windows HIP runtime
    // enumerates the cards in a fixed order no HIP_VISIBLE_DEVICES list can change (it only filters), so with a
    // slower card first a run could only put the model on it (RX 6600 before RX 6800).  0 (unset) is upstream's
    // behavior; the helper caches and the peer tier follow it, a layer split still assumes device 0 (see below).
    primary_device = [] {
        const char* v = std::getenv("STRATA_PRIMARY_DEVICE");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n > 0 ? n : 0;
    }();
    strata::core::set_primary_device(primary_device);
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  KV
    // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across
    // GPUs, not yet: control vectors, the helper caches (--expert-cache-remote), and lending cache slots to the
    // prompt path (each stage's prompt path has its own buffers).
    // --pcie-frac given: that one share is every stage's (the stages' own link probes are skipped), so a split over
    // a fast and a slow link cannot set the two apart from the command line (#485)
    pcie_given = o.pcie_frac >= 0.0;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = (o.serve || own_stages) && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto && !own_stages;
        if (split_same && o.conversation_cache_mib > 0 && o.conversation_cache_slots > 0 && o.prompt_cache > 0) {
            std::fprintf(stderr, "strata serve: conversation parking does not support a layer split on one GPU "
                                 "(--split-device 0); disable parking with --conversation-cache-mib 0\n");
            return 2;
        }
        // THE LAYER SPLIT'S SPILL WALK, STEP 6: FEWER CARDS.  Under `auto`, a later card with too little free memory to hold even a trimmed share (a card another program fills) is left out
        // of the split, said plainly; with none left the start is the single card's, with its own walk and the rest of the experts in RAM.  Only the cards the engine would have picked.
        if (ok && split_auto && !split_devs.empty() && std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu()) {
            std::vector<int> kept;
            for (const int d : split_devs) {
                size_t fb_c = 0, tb_c = 0;
                bool usable = true;
                {
                    const strata::core::OnDevice on_c(d);
                    usable = cudaMemGetInfo(&fb_c, &tb_c) != cudaSuccess || strata::program::split_walk::card_usable((int64_t) (fb_c >> 20));
                    (void) cudaGetLastError();
                }
                if (usable) { kept.push_back(d); continue; }
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d has only %.2f GiB free, too little to hold a share of the model beside its context, a working cache "
                                     "and the prompt buffers (%.1f GiB at least): the layer split runs without it\n",
                             d, (double) fb_c / 1073741824.0, (double) strata::program::split_walk::kMinCardMib / 1024.0);
            }
            if (kept.empty())
                std::fprintf(stderr, "strata generate: short of GPU memory: no other card has room, so the model runs on CUDA0 alone with the experts that do not fit in RAM\n");
            split_devs = kept;
        }
        if (ok && split_auto && split_devs.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: one GPU visible, so no split\n");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = (split_devs[i] > 0 || own_stages) && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i && !own_stages; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            std::fprintf(stderr, "strata generate: --layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU). K is the "
                                 "FIRST LAYER of each later GPU, not a count of layers per card: \"24,36,42\" for 4 "
                                 "GPUs, not \"24,12,6,6\" (got \"%s\")\n", n_dev - 1, o.layer_split.c_str());
            return 2;
        }
    }
    multi_gpu = !split_devs.empty() && !split_same;
    // STRATA_PRIMARY_DEVICE: a later stage's list (`--split-device` values must be > 0) still assumes the
    // primary is device 0 - refuse the combination rather than mis-place a stage.  The peer tier takes the
    // primary into account (peer_experts.cpp).
    if (primary_device != 0 && !o.layer_split.empty()) {
        std::fprintf(stderr, "strata generate: STRATA_PRIMARY_DEVICE %d is not supported with a layer split yet\n",
                     primary_device);
        return 2;
    }
    bool split_own_auto = false;   // #340: the split keeps own prompt buffers by its rule (not --no-prefill-borrow)
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        std::fprintf(stderr, "strata generate: --shared-expert-arena backs the resident arena and cannot be used with --mmap-experts\n");
        return 2;
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile\n");
        return 2;
    }
    remote_caches = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
                    o.expert_cache_remote[2] > 0;
    // A layer split keeps the resident RAM mode: every stage's GPU cache is left out of the RAM copy, and an adaptive
    // swap copies an evicted expert back from the card that owns its layer (resident_stage_swaps).
    // The helper-GPU caches are allowed beside the resident RAM mode too: their experts are left out of the RAM copy
    // like a stage's (below), the helpers' refills read through the source (the RAM copy when it holds the bytes),
    // and the two adaptive tiers already keep each other's experts out (helper_holds in the primary's candidates,
    // RemoteExpertOpt::adapt's resident table).  Measured on RX 6800 + RX 6600, Windows (docs/SECOND_GPU.md).
    // the helper-GPU expert caches (--expert-cache-remote, docs/SECOND_GPU.md): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (primary_device != 0) {
        // STRATA_PRIMARY_DEVICE: the helpers are the visible devices except the primary, in order
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int k = 0;
        for (int d = 0; d < n_vis && k < 3; ++d)
            if (d != primary_device) remote_dev[k++] = d;
        for (; k < 3; ++k) remote_dev[k] = -1;
    }
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            std::fprintf(stderr, "strata generate: a layer split across GPUs needs --expert-profile\n");
            return 2;
        }
        // A STAGE'S PROMPT PATH BORROWS FROM THAT STAGE'S OWN EXPERT CACHE.  This used to set
        // `no_prefill_borrow = true` - "each stage's prompt path has its own buffers" - which is true, but it is
        // a reason to give each stage its own LOAN, not a reason to make every stage withhold a chunk-sized
        // reserve from its cache for the whole session.  Forced on, it also collapsed `--prefill auto` to 2048
        // (below) and skipped the lend arm, so a stage paid for its prompt buffers twice over: once in VRAM it
        // never got back, once in the smaller chunk.  At `--prefill 5524` that reserve is 3.8 GiB per stage,
        // more than either 8 GB card had - which is how adding two GPUs to the two 12 GB ones lost 260K.
        // The loan is the tail of the stage's own cache (see `PfPart` in the serve block); outside the prompt
        // that tail is expert cache, so a large chunk costs a stage nothing permanent.
        // #340 - BUT A LOAN IS NOT FREE PER REQUEST: every stage streams the lent experts during the prompt and
        // copies them back after it, so on cards that hold (nearly) every expert of their layers a 2K prompt read
        // 37% slower than 0.1.29's own buffers (2x RX 9070 XT / R9700: 1570 -> 990 tok/s; 2x A5000 in #340: -35%).
        // So a split keeps 0.1.29's own buffers (2048-token chunks, priced into the split search) when they are a
        // small part of every card - at most 12% of its VRAM (16 GB and larger cards) - and borrows on smaller cards,
        // where the reserve would cost the cache (and the context) the paragraph above is about.  Measured with own
        // buffers on the 9070 XT + R9700: 2K 1588 tok/s, 16K 2006 (0.1.29 1568 / 1935, 0.1.31 993 / 1852).
        // OPT-IN (the owner's choice for 0.1.32): own buffers change which experts a full card keeps resident, so the
        // split's output differs from 0.1.31's; the default borrows as 0.1.31 did - with the concurrent refill and the
        // 96-blob split ring that alone gave 2K +27%, 16K +5% over 0.1.31, decode unchanged, output identical.
        // STRATA_SPLIT_OWN=auto: the 12% rule above; 1: own buffers on any cards; unset/0: borrow.
        // --split-skip-if-fits keeps the loans (it can fall back to one GPU, which must stay as it is).
        const char* own_env = std::getenv("STRATA_SPLIT_OWN");
        if (!o.no_prefill_borrow && !o.split_skip_if_fits && own_env != nullptr && own_env[0] != '0') {
            const char* v = std::string(own_env) == "auto" ? nullptr : own_env;
            bool own = v ? v[0] == '1' : true;
            // the buffers of a 2048-token chunk (or the --prefill one) + a 96-blob ring (as split_pf_mib prices them)
            const int64_t own_chunk = o.prefill_auto || o.prefill_chunk <= 0 ? 2048 : o.prefill_chunk;
            const int64_t reserve_mib = 160 + (own_chunk * 680) / 1024 + 96 * 4;
            std::string why;
            if (!v) {
                std::vector<int> devs = {0};
                for (const int d : split_devs) devs.push_back(d);
                for (const int d : devs) {
                    cudaDeviceProp prop{};
                    if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) { cudaGetLastError(); own = false; break; }
                    const int64_t total_mib = (int64_t) (prop.totalGlobalMem >> 20);
                    if (reserve_mib * 100 > 12 * total_mib) {
                        own = false;
                        why = "CUDA" + std::to_string(d) + " has " + std::to_string(total_mib) + " MiB";
                        break;
                    }
                }
            }
            if (own) {
                o.no_prefill_borrow = true;
                split_own_auto = true;
                std::fprintf(stderr, "strata generate: layer split: every stage keeps its own prompt buffers (~%lld MiB "
                                     "each, %lld-token chunks)%s\n", (long long) reserve_mib, (long long) own_chunk,
                             v ? " (STRATA_SPLIT_OWN=1)" : "");
            } else if (!why.empty()) {
                std::fprintf(stderr, "strata generate: layer split: the prompt paths borrow from the caches (%s)\n",
                             why.c_str());
            }
        }
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                std::fprintf(stderr, "strata generate: --expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)\n", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        std::fprintf(stderr, "strata generate: layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)\n",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
#if defined(STRATA_USE_HIP)
    {
        // every GPU this run uses must be an architecture the binary has code for (a gfx1100 build on a gfx1201
        // card would otherwise fail later with "invalid device function")
        std::vector<int> used{primary_device};   // STRATA_PRIMARY_DEVICE: the primary is not always 0
        if (multi_gpu) used.insert(used.end(), split_devs.begin(), split_devs.end());
        for (int r = 0; r < 3; ++r)
            if (o.expert_cache_remote[(size_t) r] > 0) used.push_back(remote_dev[r]);
        for (const int d : used) {
            if (const std::string why = strata::core::gpu_arch_problem(d); !why.empty()) {
                std::fprintf(stderr, "strata generate: %s\n", why.c_str());
                return 1;
            }
        }
    }
#endif
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        // 0.2.8 Strix Halo: apply_device_defaults' gfx1151 rule chose a chunk that fits this box; the "keep them small"
        // 2048 would throw the whole 8K prompt back into 4 chunks (measured: 2048 against 8192 costs ~26% of an 8K prompt
        // on this box).  Not measured with strata-q38, whose page pool pays for the larger buffers.
        o.prefill_chunk = o.prefill_own_keep_chunk && !is_paged() ? o.prefill_chunk : 2048;
    }
    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        std::fprintf(stderr, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
        return 2;
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        std::fprintf(stderr, "strata generate: --kv must be fp16, int8, q4_0 or k8v4\n");
        return 2;
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    // STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation --kv q4_0 already uses. Opt-in: first-token KL to
    // fp16 K/V improved on an NVFP4 pack (0.0066 -> 0.0051) but not on IQ2_XS (0.0022 -> 0.0054)
    const char* kv_rot = std::getenv("STRATA_KV_ROT");
    strata::core::qsa_set_kv_int8_rotate(kv_rot != nullptr && kv_rot[0] == '1');
    if (kv_rot != nullptr && kv_rot[0] == '1' && o.kv == "int8")
        std::fprintf(stderr, "strata generate: STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation (opt-in)\n");
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        std::fprintf(stderr, "strata generate: --kv-resident must be >= 0\n");
        return 2;
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    const auto window_rows = strata::spec::get_window_rows(o.spec, o.mtp_max_t, o.suffix_draft, o.lookup_chain);
    o.spec = window_rows.total;
    o.mtp_max_t = window_rows.model;
    // --lookup-max-t N: lookup windows up to N tokens, the MTP's stay at --spec.  Not given: 12 where the pr-nv2 runs measured it (serve, one GPU, the 8-row
    // build: rewrites and agent edit turns +11..14%, prose and code within noise); --lookup-max-t 0 or STRATA_LOOKUP_MAX_T=0 turns it off.  The wide
    // builds (HIP, 16 rows), a layer split, --batch, DFlash2 and --lookup-chain keep the old windows.  strata-q38's
    // shared step has no lookup window sizes of their own: off there.
    if (is_paged()) {
        o.lookup_max_t = 0;
    } else if (o.lookup_max_t < 0) {
        const char* lme = std::getenv("STRATA_LOOKUP_MAX_T");
        const bool env_off = lme != nullptr && lme[0] == '0';
        const bool narrow = strata::kernels::kVerifyMaxT == strata::kernels::kVerifyNarrowT;
        strata::core::LookupDefaultIn ld;
        ld.env_off = env_off, ld.narrow_window = narrow, ld.serve = o.serve, ld.multi_gpu = multi_gpu;
        ld.layer_split = !o.layer_split.empty(), ld.batch = o.batch > 1, ld.dflash2 = o.draft == "dflash2";
        ld.lookup_chain = o.lookup_chain > 0, ld.suffix_draft = o.suffix_draft > 0, ld.spec_at_least_2 = o.spec >= 2;
#if defined(STRATA_GPU_METAL)
        ld.metal_build = true;   // (Metal's windows hold 8 rows and it has no 12-row window: "metal window: window size out of range")
#endif
        o.lookup_max_t = strata::core::default_lookup_max_t(ld);
    }
    if (o.lookup_max_t > 0 && o.suffix_draft > 0 && o.spec >= 2) {
        if (o.mtp_max_t == 0) o.mtp_max_t = o.spec;
        o.spec = std::max(o.spec, std::min(o.lookup_max_t, strata::kernels::kVerifyLongT));
    }
    // A speculative (solo) window holds at most kVerifyNarrowT rows: the hand-offs, the penalty history and the drafters'
    // buffers are sized for that.  Only a plain --batch forms the wider windows of a build with kVerifyMaxT > 8.
    // (--lookup-max-t: a prompt-lookup window may be longer, up to kVerifyLongT; the MTP drafts stay at mtp_max_t.
    // strata-q38's --spec is its batch windows' room, up to kVerifyMaxT.)
    const int spec_cap = o.lookup_max_t > 0 && o.suffix_draft > 0 ? (int) strata::kernels::kVerifyLongT
                                                                  : (int) strata::kernels::kVerifyNarrowT;
    if (!is_paged() && o.spec > spec_cap) {
        std::fprintf(stderr, "strata generate: --spec %d: a speculative window holds 2..%d tokens\n", o.spec, spec_cap);
        return 1;
    }
    // dflash2-kern: right after the start the resident experts are the profile's, not this text's; a speculative round carries
    // 2-5 tokens, so the adaptive tier (one swap round per adapt_every VERIFY rounds) learned the text 2-5 times slower per
    // token.  For the first STRATA_SPEC_ADAPT_WARM rounds it swaps every STRATA_SPEC_ADAPT_EVERY rounds (set per card by
    // nv_defaults, where measured); later the normal interval.  An explicit --adapt-every is kept as it is.
    if (!o.adapt_every_given && o.spec >= 2 && o.adapt_every > 0) {
        const char* e = std::getenv("STRATA_SPEC_ADAPT_EVERY");
        const char* w = std::getenv("STRATA_SPEC_ADAPT_WARM");
        const int n = e ? std::atoi(e) : 0;
        const long long wr = w ? std::atoll(w) : 0;
        if (n > 0 && n < o.adapt_every && wr > 0) { o.adapt_warm_every = n; o.adapt_warm_rounds = wr; }
    }
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        try {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native).  A missing shard is an error
            // here: it used to be skipped, leaving a model with some tensors absent and a later error, or none.
            o.native_shards = strata::gguf_split_paths(o.native_preset);
            // a model family Strata recognises but has no engine path for yet (Qwen3.6-35B-A3B) is refused here,
            // with a message that says so, before anything else about the file is judged
            {
                const strata::GgufFile meta(o.native_shards.front());
                const std::string why = strata::model_family_not_supported(strata::model_family(meta));
                if (strata::model_family(meta) != strata::ModelFamily::Unknown && !why.empty()) {
                    std::fprintf(stderr, "strata generate: --native %s: %s\n", o.native_preset.c_str(), why.c_str());
                    return 2;
                }
            }
            // --ple-gguf defaults to the shard that holds the PLE table, found by name: shard 2 of the ISTA files
            // and of Unsloth's UD-Q4_K_XL, shard 1 of Swift's
            if (o.ple_gguf.empty() && !o.no_ple) {
                const strata::GgufModel model(o.native_shards);
                size_t at = 0;
                if (model.find("per_layer_token_embd.weight", &at) != nullptr) o.ple_gguf = o.native_shards[at];
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: --native %s: %s\n", o.native_preset.c_str(), e.what());
            return 2;
        }
        if (o.no_ple || o.ple_gguf.empty()) {
            std::fprintf(stderr, "strata generate: --native requires --ple-gguf (the PLE key is native too), and no "
                                 "shard of the model holds per_layer_token_embd.weight\n");
            return 2;
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = o.native_shards;
            // a PLE-only table (tools/ple_fp8_pack.py: architecture strata-ple) holds no projections
            bool ple_only = false;
            try {
                strata::GgufFile pg(o.ple_gguf);
                if (const strata::MetaValue* v = pg.get("general.architecture")) ple_only = v->s == "strata-ple";
            } catch (const std::exception&) {}
            if (!ple_only &&
                std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        // CUDA, a file that carries the hyper-connection projections as Q8_0 (Unsloth's UD-* files): the Q8_0 read is on by default (owner yes
        // 2026-11-07; was setup's unsloth_default_env, "closer to the file and faster").  The GSQ-RCO packs keep them BF16 and stay as they were.
        // STRATA_HC_Q8=0 turns it off; a value the user set is kept.  Before the first read of the switch (the dense load below).
        // Not measured with strata-q38: off there unless set.
        if (!is_paged() && strata::core::NativeDense::gguf_carries_q8_hc(o.native_dense_gguf))
            strata::platform::set_env("STRATA_HC_Q8", "1", false);
#endif
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        std::fprintf(stderr, "strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits\n");
        return 2;
    }
    if (o.no_ple && !o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --no-ple and --ple-gguf are mutually exclusive\n");
        return 2;
    }
    if (o.native_ple_postops && o.no_ple) {
        std::fprintf(stderr, "strata generate: --native-ple-postops requires PLE enabled\n");
        return 2;
    }
    if (!o.no_ple && o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --ple-gguf is required; --no-ple explicitly enables a diagnostic ablation\n");
        return 2;
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "strata generate: --max-context must be below 2^31\n");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    {
        using RST = strata::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            std::fprintf(stderr, "strata generate: --rope-scaling must be none, linear or yarn (got '%s')\n",
                         o.rope_scaling.c_str());
            return 2;
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "strata generate: a rope scaling knob is not a finite number (%g)\n", v);
                return 2;
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            std::fprintf(stderr, "strata generate: --rope-scale %g must be >= 1 (it extends the context, not shrinks it)\n",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            std::fprintf(stderr, "strata generate: --rope-freq-base must be a base above 1 (0 = the model's)\n");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            std::fprintf(stderr, "strata generate: invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)\n");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        std::fprintf(stderr, "strata generate: invalid sampling or resource parameter\n");
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        std::fprintf(stderr, "strata generate: --expert-cache-remote-placement must be stripe or layer\n");
        return 2;
    }
    if (o.remote_expert_opt && !o.serve) {
        std::fprintf(stderr, "strata generate: --remote-expert-opt requires --serve\n");
        return 2;
    }
    // the peer tier is the second card's only user: a layer split or a remote expert cache would put a second engine
    // part (and a second copy of the same experts) on it
    if (o.peer_device >= 1 && (!o.layer_split.empty() || o.expert_cache_remote[0] > 0)) {
        std::fprintf(stderr, "strata generate: --peer-device cannot be combined with %s\n",
                     !o.layer_split.empty() ? "--layer-split (use one or the other)"
                                            : "--expert-cache-device1..3 (the peer tier already caches experts there)");
        return 2;
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
        return 2;
    }
    if (o.native_flash_attn_short && (o.gpu_only_full || o.graph_only || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires the normal decode loop for status validation\n");
        return 2;
    }
    // The whole-model graph measurements replay every layer through CUDA0's session, which a layer split carves to
    // CUDA0's own range - the answer would read another stage's state.  Refused here rather than at the call, so
    // the reason is visible before 55 GB is loaded.
    if (multi_gpu && (o.gpu_only_full || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --gpu-only-full and --gpu-stages replay the whole model through one "
                             "session, which a layer split does not have; run them without --layer-split\n");
        return 2;
    }
    if (!o.native_head_gguf.empty()) {
        try {
            o.native_head_shards = o.native_head_gguf == o.native_preset ? o.native_shards
                                                                         : strata::gguf_split_paths(o.native_head_gguf);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: --native-head-gguf %s: %s\n", o.native_head_gguf.c_str(), e.what());
            return 2;
        }
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        std::fprintf(stderr, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
        return 2;
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
        return 2;
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled `/arch:AVX512`, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(STRATA_GPU_METAL)
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu in the release build, which the native pack's layout load initializes
    // next.  Refused here, by name, rather than an illegal instruction in the first expert.
    // The experimental older-CPU build (STRATA_ISA_FLOOR=avx|none, compiled on that PC; #394 #595 #623) has ggml-cpu
    // for that floor, so a native pack's experts run there on ggml-cpu (every AVX2 kernel is behind cpu_avx2_ok).
    const char* isa_floor = strata::kernels::cpu::isa_floor_build();
#if defined(STRATA_GPU_METAL)   // Apple Silicon: every expert is on the GPU; ggml-cpu (NEON) is the only CPU expert code
    strata::metal_engine::set_model_shard(o.native_preset.c_str());   // the decode window reads the model GGUF itself
    strata::metal_engine::set_batch_slots(o.batch);
    o.mmap_experts = true;   // ... experts included: no resident copy of them (the GGUF shards in place)
    if (false) {
#else
    if (!strata::kernels::cpu::cpu_avx2_ok() && isa_floor[0] == '\0') {
#endif
        std::fprintf(stderr, "strata generate: this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                             "CPU expert kernels need; Strata runs on Intel Haswell (2013), AMD Zen (2017) or newer. "
                             "Older CPUs are EXPERIMENTAL and slow: setup compiles an engine for them on this PC "
                             "(STRATA_ISA_FLOOR, see \"Older CPUs\" in docs/INSTALL.md)\n",
                     strata::kernels::cpu::cpu_name().c_str());
        return 2;
    }
    if (isa_floor[0] != '\0') {
        const bool avx_floor = std::strcmp(isa_floor, "avx") == 0;
        if (avx_floor ? !strata::kernels::cpu::cpu_avx1_ok() : !strata::kernels::cpu::cpu_sse42_ok()) {
            std::fprintf(stderr, "strata generate: this engine is the older-CPU build for %s, which this CPU (%s) does "
                                 "not have%s\n", avx_floor ? "AVX" : "SSE4.2 with POPCNT",
                         strata::kernels::cpu::cpu_name().c_str(),
                         avx_floor ? "; compile it with STRATA_ISA_FLOOR=none" : "");
            return 2;
        }
        if (!strata::kernels::cpu::cpu_avx2_ok())
            std::fprintf(stderr, "strata generate: EXPERIMENTAL older-CPU build (ggml-cpu for %s): this CPU has no AVX2, "
                                 "so the CPU's experts run on ggml-cpu's kernels - expect it to be slow\n", isa_floor);
        else
            std::fprintf(stderr, "strata generate: this is the older-CPU build (ggml-cpu for %s); this CPU has AVX2, "
                                 "and the normal build is faster on it\n", isa_floor);
    }
#else
    if (o.cpu_oracle_q8_0) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 is the x86 Q8_0 activation contract; this CPU uses the "
                             "scalar quantizer\n");
        return 2;
    }
#endif

    if (!o.native_head_gguf.empty() && !o.stream_token) {
        std::fprintf(stderr, "--native-head-gguf requires --stream-token\n");
        return 2;
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry& g0 = strata::core::kCompiledGeometry;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported down type used to exit from inside the first verify window, and
        // an unsupported dequant type left the prompt path's fp16 buffer unwritten.
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                std::fprintf(stderr, "strata generate: layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                                     "engine has no GPU kernels for\n", (long long) l,
                             strata::ggml_type_name((uint32_t) f.gu_type), strata::ggml_type_name((uint32_t) f.d_type),
                             f.gu_type, f.d_type);
                return 1;
            }
#if defined(STRATA_PREFILL_MMQ)
            // MODEL-UPDATE Phase 1: the K-quants' prompt kernels (STRATA_MMQ_KQ=1) are a library of their own on CUDA
            // (strata_kq): asking for the formats opens it now, before the expert cache takes the free VRAM
            (void) strata::prefill::mmq::supported(f.gu_type);
            (void) strata::prefill::mmq::supported(f.d_type);
#endif
        }
        // APR (prism.hadamard) packs: every expert path rotates its activations (CPU pool: native_quant_act / _h;
        // GPU verify, remote GPUs, prompt FP16 path: hadamard_rows).  Validated end to end on CUDA (sm_86, sm_89 with
        // a two-GPU layer split and peer decode, sm_120) and on HIP gfx1151.  Other HIP architectures run the same
        // kernels but have not been run end to end with such a pack: said here, not refused (the P2P prompt peer
        // says the same in Prefill::set_peer).  The former opt-in STRATA_APR is no longer read.
        if (strata::kernels::cpu::hadamard_spec().any()) {
            std::fprintf(stderr, "strata generate: APR pack: Hadamard-folded experts (block %d)\n",
                         strata::kernels::cpu::hadamard_spec().block);
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
            int dev = 0;
            cudaDeviceProp p{};
            if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
                std::string arch(p.gcnArchName);
                if (const size_t colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
                if (arch != "gfx1151")
                    std::fprintf(stderr, "strata generate: warning: Hadamard-folded experts are validated end to end "
                                         "on gfx1151 among AMD GPUs, not yet on %s\n", arch.c_str());
            } else {
                (void) cudaGetLastError();
            }
#endif
        }
    }
    // The expert arena copies every expert into RAM.  When the RAM available now cannot hold it with 1 GiB to spare
    // (platform::kArenaFitFloor), its read takes the RAM to zero and the start fails (a Strix Halo on Windows: 36.2 GiB available,
    // a 46.8 GiB arena, error 1453 at layer 21), so the resident low-RAM mode is taken instead (--resident-experts:
    // the GPU's cache keeps what it holds, only the rest is copied into RAM), or without a profile the experts are
    // read from the model's files (--mmap-experts).  Decided here, before anything that depends on the mode
    // (--kv-grow, the prompt chunk).  A start whose arena fits is unchanged; STRATA_ARENA_FIT=0 keeps the arena.
    if (!o.mmap_experts && o.shared_expert_arena.empty()) {
        const char* fit_env = std::getenv("STRATA_ARENA_FIT");
        const uint64_t arena_b = strata::kernels::cpu::expert_layout().total;
        const auto hm = strata::core::sample_host_memory();
        if (!(fit_env != nullptr && fit_env[0] == '0') && hm && hm->available > 0 &&
            !strata::platform::arena_fits_ram(hm->available, arena_b)) {
            const bool resident = !o.expert_profile.empty();
            if (resident) take_resident_experts(o);
            else o.mmap_experts = true;
            std::fprintf(stderr, "strata generate: the experts (%.1f GiB) do not fit in RAM available now (%.1f GiB, "
                                 "%.1f GiB kept free). Enabling `%s` mode to compensate.\n",
                         (double) arena_b / 1073741824.0, (double) hm->available / 1073741824.0,
                         (double) strata::platform::kArenaFitFloor / 1073741824.0,
                         resident ? "--resident-experts" : "--mmap-experts");
        }
    }
    native_pack = strata::kernels::cpu::expert_layout().native;
    // STRATA_MMQ_DIRECT: the room each sized expert slot keeps after its blob (0 unless it is on: the slot costs and
    // the arena are then today's, byte for byte)
    const int64_t slot_guard = native_pack ? (int64_t) strata::prefill::Prefill::slot_guard_bytes() : 0;
    // STRATA_PRIMARY_DEVICE: switch the engine's own thread to the chosen card BEFORE any GPU
    // context or allocation exists below (the helper preflights, the PCIe probe, the weights and the
    // arena all follow it).  Every other device switch in the engine saves and restores the current
    // device, so this single call is where the primary is picked.
    if (primary_device != 0) {
        int n_vis = 0;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || primary_device >= n_vis) {
            cudaGetLastError();
            std::fprintf(stderr, "strata generate: STRATA_PRIMARY_DEVICE=%d: no such visible device (%d visible)\n",
                         primary_device, n_vis);
            return 2;
        }
        if (cudaSetDevice(primary_device) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: STRATA_PRIMARY_DEVICE=%d: cudaSetDevice failed: %s\n",
                         primary_device, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        std::fprintf(stderr, "strata generate: primary device from STRATA_PRIMARY_DEVICE: CUDA%d\n", primary_device);
    }
    // ---- STRATA_EXPERT_NIBBLE=1 keeps the gate/up experts of IQ2_S / IQ3_XXS / IQ3_S layers as 4-bit codes in their VRAM
    // slots (strata/common/kernels/expert_nibble.hpp).  Unset or 0: nothing here runs, no table is set, and every slot size below is
    // the GGUF blob's, as before.  On: if anything would read a converted slot as GGUF blocks or write GGUF bytes into
    // one, the switch is ignored for this start with a warning (before anything is allocated); else the per-layer table
    // is set before the first slot estimate and the adaptive tier is off (it swaps GGUF blobs in and out of slots).
    // The residency, the loans and the prompt chunk are checked where they are known (those refuse the start).
    const bool nib_asked = nib::config().on || nib::config().error.rfind("STRATA_EXPERT_NIBBLE=", 0) == 0;
    nib_chunk_asked = o.prefill_chunk;
    if (nib_asked) {
        // (strata-q38's page pools size every expert's page by its GGUF blob)
        std::string why = is_paged() ? "strata-q38 keeps the GGUF slots" : nibble_refusal(o);
        if (why.empty()) {
            nib::set_layers(nib::plan_layers(strata::kernels::cpu::expert_layout(), why));
            if (why.empty() && !nib::any_on())
                why = "no layer of this pack is IQ2_S / IQ3_XXS / IQ3_S at 2560 x 640 among STRATA_EXPERT_NIBBLE_FMTS "
                      "and outside STRATA_EXPERT_NIBBLE_SKIP";
            if (why.empty()) (void) strata::prefill::Prefill::nibble_prompt_check(why);
        }
        if (!why.empty()) {
            // a warning, not a refusal: this start goes on as it would without the switch (the usual GGUF slots)
            std::fprintf(stderr, "strata generate: WARNING: STRATA_EXPERT_NIBBLE is off for this start: %s; the expert slots "
                                 "stay GGUF blocks\n", why.c_str());
            nib::set_layers({});
        } else {
            o.adapt_swaps = 0;
            o.peer_adapt_swaps = 0;
            const auto& lay = strata::kernels::cpu::expert_layout();
            int64_t n_on = 0, by_fmt[3] = {0, 0, 0};
            uint64_t more = 0;
            for (int64_t l = 0; l < lay.n_layers; ++l) {
                const nib::LayerVram& v = nib::layer(l);
                if (!v.on) continue;
                ++n_on;
                ++by_fmt[v.gu_type == nib::kIQ2_S ? 0 : v.gu_type == nib::kIQ3_XXS ? 1 : 2];
                more += (v.slot_bytes - (v.raw_bytes + 255) / 256 * 256) * (uint64_t) lay.n_expert;
            }
            std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: %lld of %lld layers keep their gate/up experts as 4-bit "
                                 "codes in VRAM (IQ2_S %lld, IQ3_XXS %lld, IQ3_S %lld), %.2f GiB more VRAM than their GGUF "
                                 "blobs for all %lld experts of each; the adaptive tier is off (--adapt-swaps 0)%s\n",
                         (long long) n_on, (long long) lay.n_layers, (long long) by_fmt[0], (long long) by_fmt[1],
                         (long long) by_fmt[2], (double) more / 1073741824.0, (long long) lay.n_expert,
                         nib::config().recheck ? "; STRATA_EXPERT_NIBBLE_RECHECK: every slot's hash after each request" : "");
        }
    }
    // STRATA_EARLY_REMOTE_CONTEXTS=1: create EVERY secondary context here, like CUDA1's.  Under WSL2 the driver's
    // pinned/mapped host budget (dxg gpadl, ~1 GiB) is spent by CUDA0's weights and MTP before the later loop runs,
    // and a new context then fails with cudaErrorMemoryAllocation (CUDA2: "cudaSetDevice(2) failed: out of memory").
    const char* early_env = std::getenv("STRATA_EARLY_REMOTE_CONTEXTS");
    const bool early_remote = early_env && early_env[0] == '1';
    // Creates the context of remote expert cache r's GPU; false after saying why it could not
    auto ready_remote_context = [&](int r) {
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
        return true;
    };
    // Keep CUDA1's proven startup order: initialise its context before
    // allocating GPU0 weights or mapping the large host expert arena.
    for (int r = 0; r < 3; ++r)
        if (o.expert_cache_remote[(size_t) r] > 0 && (r == 0 || early_remote) && !ready_remote_context(r)) return 1;
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack keeps 0.2 as its start value (0.1.43: PcieAuto refines it on CUDA).  `--calibrate` measures it outright.
    // NVFP4 packs: 0.25. Their 2.76 MB experts cross the link twice as long as the i-quants' the 0.55 was
    // measured on, and that copy sits on the GPU's critical path while the CPU pool runs beside it: decode
    // 103 -> 117 tok/s short, 105 -> 124 at 32K (RTX 5090, PCIe 5 x16, 9950X3D; 0.10 and 0.40 both slower).
    if (o.pcie_frac < 0.0) {
        const auto& lay0 = strata::kernels::cpu::expert_layout();
        const bool nvfp4_pack = native_pack && !lay0.fmt.empty() && lay0.fmt[0].gu_type == 40;
        const char* pfd = std::getenv("STRATA_PCIE_FRAC_DEFAULT");
        const bool pf_old = pfd != nullptr && std::strcmp(pfd, "old") == 0;
        // 0.2.7: gfx12 native packs start at 0.15 (see pcie_share_gfx12_default); other cards,
        // STRATA_PCIE_FRAC_DEFAULT=old and strata-q38 (not measured with it) keep the x16 share measured for them.
        const double base = strata::pcie_start_frac(nvfp4_pack, native_pack, pcie_share_gfx12_default() && !is_paged(),
                                                    pf_old, strata::qwen_defaults_026());
        std::string bursts;
#if !defined(STRATA_USE_HIP) && defined(_WIN32)
        // Windows: opt-in (STRATA_PCIE_AUTO_CANON=1); an RTX 5070 decoded a short Q2_0 answer 3% slower with it, code 2.5% faster
        const char* pac = std::getenv("STRATA_PCIE_AUTO_CANON");
        const bool pf_old_canon = pf_old || !(pac != nullptr && pac[0] == '1');
#elif !defined(STRATA_USE_HIP)
        const bool pf_old_canon = pf_old;
#else
        const bool pf_old_canon = true;   // the canonical pack's PcieAuto is CUDA only
#endif
        // the canonical pack keeps its 0.2 as the START value, but is probed too (CUDA, unless
        // STRATA_PCIE_FRAC_DEFAULT=old) so that PcieAuto can refine the share from the CPU pool's speed
        const double bw = (native_pack || !pf_old_canon) ? probe_pcie_h2d_gbps(&bursts) : -1.0;
        if (!native_pack && !(bw > 0.0)) {
            o.pcie_frac = base;
        } else if (!native_pack) {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s); the canonical pack "
                                 "starts at pcie_frac %.2f and refines it from the CPU pool's speed\n", bw, bursts.c_str(), base);
#if !defined(STRATA_USE_HIP)
            o.pcie_frac_auto = o.layer_split.empty();
            o.pcie_probe_gbps = bw;
#endif
        } else if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)\n", bw, bursts.c_str(), o.pcie_frac, base);
            // 0.1.43: split opt-in + gfx12 default-on (decode_defaults_arch_ok).  Other HIP parts: STRATA_PCIE_FRAC_DEFAULT=auto
            // turns the refinement on (pcie_auto_fork_arch_ok); the pool timings and miss counts it reads are engine-generic.
            // STRATA_PCIE_FRAC_DEFAULT=old keeps the start-up rule everywhere.
            o.pcie_frac_auto = (o.layer_split.empty() || pcie_auto_split_ok()) && !pf_old &&
                               (decode_defaults_arch_ok() || pcie_auto_fork_arch_ok());
            o.pcie_probe_gbps = bw;
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well
    if (!native_pack) {
        // #2347: an NVFP4 / i-quant / Unsloth pack IS a native pack and runs on an AVX2 CPU (its rows have AVX2
        // kernels).  This refusal is the canonical Q2_0 pack's, reached when the --pack folder has no
        // native_experts.txt - a hand-written config pointing --pack at the wrong folder reads as "this CPU is too
        // old".  Say so before the refusal, and where the file is when it sits one level down (a pack/ subfolder).
        if (!strata::kernels::cpu::cpu_features().usable()) {
            std::error_code ec;
            const std::filesystem::path pk(o.pack);
            std::fprintf(stderr, "strata generate: %s has no native_experts.txt, so it is read as the canonical Q2_0 "
                                 "pack, whose CPU kernel needs AVX-512.\n"
                                 "        An NVFP4, i-quant or Unsloth pack has that file in the folder given as --pack "
                                 "and runs on an AVX2 CPU as well;\n",
                         o.pack.c_str());
            if (std::filesystem::exists(pk / "pack" / "native_experts.txt", ec))
                std::fprintf(stderr, "        it is in %s: use that folder as --pack.\n", (pk / "pack").generic_string().c_str());
            else
                std::fprintf(stderr, "        if this is one of those packs, the folder is incomplete or the wrong one "
                                     "(setup writes it; run setup again).\n");
        }
        strata::kernels::cpu::cpu_require_expert_support();
    } else if (!strata::kernels::cpu::cpu_vnni_ok())
        std::fprintf(stderr, "strata generate: CPU expert kernels: %s\n",
                     !strata::kernels::cpu::cpu_avx2_ok()
                         ? (strata::kernels::cpu::cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr
                            ? "AVX1 (older-CPU build)"
                            : "ggml-cpu (older-CPU build)")
                     : std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu (STRATA_NO_IQ256 set)");
    // the compiled shape; a native model's file replaces it below with its own metadata (read_geometry), which a
    // pruned variant (GSQ-RCO Coder, fewer than 512 experts) differs from
    g = strata::core::kCompiledGeometry;
    strata::kernels::PleConsts ple_consts_run = strata::kernels::ple_artifact_consts();   // the model file's PLE hash constants when it has them
    K = g.n_expert_used;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any - the
        // artifact today ships none, so this is a no-op defaults channel for future fine-tunes.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        if (!o.native_preset.empty()) {
            // The geometry is the model file's (core::geometry_from_gguf: every "<arch>.*" shape key), checked
            // against the shapes this engine's kernels are built for.  The expert count and top-k are run-time
            // sizes: a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10.
            try {
                strata::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                std::string gerr;
                if (!strata::core::geometry_from_gguf(model_gguf, g, K, gerr)) {
                    std::fprintf(stderr, "strata generate: %s\n", gerr.c_str());
                    return 1;
                }
                const std::string mismatch = strata::core::compiled_geometry_mismatch(g);
                if (!mismatch.empty()) {
                    std::fprintf(stderr, "strata generate: %s: %s\n", o.native_preset.c_str(), mismatch.c_str());
                    return 1;
                }
                const std::string p = std::string(strata::model_family_arch(g.family)) + ".";
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.freq_base")) gguf_rope_base = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.type")) gguf_rope_type = v->s;
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
                // The PLE hash constants (multipliers, head sizes, head offsets) when the file carries them; the artifact's
                // own values stay the default for a file that does not.
                {
                    auto arr = [&](const char* k, std::vector<uint64_t>& out) {
                        const strata::MetaValue* v = model_gguf.get(p + k);
                        if (v == nullptr || v->type != strata::MetaType::ARRAY) return false;
                        for (const strata::MetaValue& e : v->items) out.push_back(e.u);
                        return true;
                    };
                    std::vector<uint64_t> pm, pv, po;
                    const bool a1 = arr("ple.layer_multipliers", pm), a2 = arr("ple.head_vocab_sizes", pv), a3 = arr("ple.head_offsets", po);
                    if (a1 || a2 || a3) {
                        std::string perr;
                        if (!(a1 && a2 && a3) || !strata::kernels::ple_consts_from_arrays(pm, pv, po, 0, ple_consts_run, perr)) {
                            std::fprintf(stderr, "strata generate: %s: the PLE hash constants in the model file are not usable: %s\n",
                                         o.native_preset.c_str(), (a1 && a2 && a3) ? perr.c_str() : "one of the three arrays is missing");
                            return 1;
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: reading the model's geometry from %s: %s\n",
                             o.native_preset.c_str(), e.what());
                return 1;
            }
        }
        using RST = strata::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                std::fprintf(stderr, "strata generate: %s carries rope.scaling.type '%s' - none, linear or yarn only\n",
                             o.native_preset.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = strata::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                std::fprintf(stderr, "strata generate: note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect\n");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = strata::kernels::rope_scaling_invalid(rope_cfg)) {
            std::fprintf(stderr, "strata generate: invalid rope scaling configuration: %s (type %s, factor %g, "
                                 "freq_scale %g, base %g, original context %g)\n",
                         why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                         rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        strata::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            std::fprintf(stderr,
                         "strata generate: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f\n",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                std::fprintf(stderr,
                             "strata generate: note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle\n",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                std::fprintf(stderr,
                             "strata generate: note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position\n",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                                 "and --prefill CHUNK\n", o.pack.c_str());
            return 2;
        }
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.embd_gguf.empty() ? o.native_shards : std::vector<std::string>{o.embd_gguf}, g.n_embd,
                               g.n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "%s %s (%.0f MiB, %.1f s)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     strata::ggml_type_name((uint32_t) native_embed.type()),
                     native_embed.in_vram() ? "in VRAM" : "in mapped host memory",
                     (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count());
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    std::set<std::string> skip;
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        // #326: a --compat-bf16 pack (OrcaRouter IQ3_XXS) keeps its BF16 key in the arena
        if (native_pack && !strata::core::NativeDense::keep_unquantized_ple_key(o.pack, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (native_pack) skip.insert("token_embd.weight");
        // STRATA_HC_Q8=1 (single GPU): the layers' hyper-connection down / up projections live only as Q8_0 (NativeDense
        // quantizes them from the pack file), so their bf16 copy is not loaded
        if (!o.native_dense_gguf.empty() && !multi_gpu && strata::core::NativeDense::hc_q8_requested())
            strata::core::NativeDense::add_hc_skip(o.pack, skip);
    }
    // Layer split: every GPU holds only the dense weights of ITS layers (the PLE tensors stay everywhere).
    // Without this each card keeps a full copy (~3.4 GB for the Coder) that its stage never reads - VRAM the
    // expert cache wants.  Opt-in, STRATA_STAGE_TRIM=1, on HIP and CUDA alike: measured on 2x MI50 (PR #639),
    // and the default waits for its author's re-run of the release branch on his cards.
    const std::set<std::string> skip_base = skip;
    // --trim-stage-weights (PR #559) is the same switch as STRATA_STAGE_TRIM=1 (PR #639)
    bool trim_asked = o.trim_stage_weights || [] {
        const char* v = std::getenv("STRATA_STAGE_TRIM");
        return v != nullptr && v[0] != 0 && std::string(v) != "0";
    }();
    // ---- THE LAYER SPLIT'S SPILL WALK, STEP 1 (include/strata/common/experts/split_walk.hpp), EARLY: THE TRIM FOR EVERY CARD, CUDA0 TOO.  Under `--layer-split auto` CUDA0's weights load before the
    // search has chosen the split, so it could not be trimmed.  When the cards in the split are short (two 8 GB cards: every card would carry a 3.5 GiB copy of the dense set beside its
    // session, head, drafter, a working cache and the prompt buffers, and 6 of 6 starts failed in 0.2.10), the split is chosen here, before anything is loaded, from what each card has
    // free once its fixed costs are paid, so every layer gets the same cache bytes on every card; it then runs as an explicit split with every card trimmed.  The dense set's size is
    // estimated from the pack (the native projections at the ratio the IQ3_S pack has); the check after CUDA0's load decides again from the exact figure.  Never reached by cards with
    // room; STRATA_STAGE_TRIM=0, STRATA_SPLIT_EARLY_OFF=1 or STRATA_SPILL_OFF=1 keeps the old path.
    if (multi_gpu && split_auto && !trim_asked && !split_devs.empty() && std::getenv("STRATA_SPILL_OFF") == nullptr &&
        !(std::getenv("STRATA_STAGE_TRIM") != nullptr && std::string(std::getenv("STRATA_STAGE_TRIM")) == "0") &&
        std::getenv("STRATA_SPLIT_EARLY_OFF") == nullptr && spill_discrete_gpu()) {
        namespace sw = strata::program::split_walk;
        sw::Model wm;
        uint64_t pool_est = 0;
        std::string perr;
        if (strata::core::WeightTable::pool_bytes(o.pack, pool_est, perr, skip.empty() ? nullptr : &skip)) {
            wm.dense_mib = (int64_t) ((pool_est + (o.native_dense_gguf.empty() ? 0 : pool_est * 138 / 100)) >> 20);
            std::vector<int> devs{primary_device};
            devs.insert(devs.end(), split_devs.begin(), split_devs.end());
            const int nd = (int) devs.size();
            wm.session_mib = (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, (g.n_layers + nd - 1) / nd) >> 20);
            if (!o.native_head_gguf.empty()) wm.head_mib = 410;   // the Q5_K head of this model family (0.41 GiB); the exact figure is read once the head loads
            wm.drafter_mib = (!o.mtp.empty() || o.draft == "dflash2") ? strata::program::stage_reserve::kDrafterMib : 0;
            wm.slot_mib = 3;
            const int64_t later_reserve = (o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib) + strata::program::stage_reserve::kWindowMib;
            wm.reserve_mib = later_reserve;
            std::vector<int64_t> free_mib((size_t) nd, 0), room_mib((size_t) nd, 0);
            bool short_any = false;
            int short_dev = 0;
            int64_t short_free = 0;
            for (int i = 0; i < nd; ++i) {
                const strata::core::OnDevice on_e(devs[(size_t) i]);
                size_t fb_e = 0, tb_e = 0;
                if (cudaMemGetInfo(&fb_e, &tb_e) != cudaSuccess) { (void) cudaGetLastError(); fb_e = 0; }
                free_mib[(size_t) i] = (int64_t) (fb_e >> 20);
                const bool last_e = i + 1 == nd;
                if (!short_any && sw::trim_needed(free_mib[(size_t) i], wm, last_e)) { short_any = true; short_dev = devs[(size_t) i]; short_free = free_mib[(size_t) i]; }
                room_mib[(size_t) i] = free_mib[(size_t) i] - sw::kContextMib - (i == 0 ? (int64_t) o.vram_reserve_mib : later_reserve) - sw::kStageLateAllocMib -
                                       (last_e ? wm.head_mib + wm.drafter_mib : 0);
            }
            // FEWER CARDS (step 6): when even the whole walk (the drafter and the head moved off) does not pay for every layer on these cards, the card with the least room is left
            // out, one at a time, said plainly; with only CUDA0 left the start is the single card's, with its own walk and the rest of the experts in RAM.
            const int64_t per_layer_f = (wm.dense_mib + (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20)) / std::max<int64_t>(g.n_layers, 1);
            auto base_room_of = [&](int i) {
                return free_mib[(size_t) i] - sw::kContextMib - (i == 0 ? (int64_t) o.vram_reserve_mib : later_reserve) - sw::kStageLateAllocMib;
            };
            while (short_any && devs.size() >= 2) {
                const int ndl = (int) devs.size();
                std::vector<int64_t> base_room((size_t) ndl);
                int64_t sum_room = 0;
                for (int i = 0; i < ndl; ++i) { base_room[(size_t) i] = base_room_of(i); sum_room += base_room[(size_t) i]; }
                if (sw::split_feasible(base_room.data(), ndl, g.n_layers, per_layer_f, wm.head_mib + wm.drafter_mib)) break;
                int worst = 1;
                for (int i = 2; i < ndl; ++i) if (base_room[(size_t) i] < base_room[(size_t) worst]) worst = i;
                std::fprintf(stderr, "strata generate: short of GPU memory: even with the drafter and the head moved off, %d cards with %.2f GiB of room in all cannot hold the %lld layers "
                                     "(%.2f GiB): the layer split runs without CUDA%d\n", ndl, (double) sum_room / 1024.0, (long long) g.n_layers,
                             (double) (per_layer_f * g.n_layers) / 1024.0, devs[(size_t) worst]);
                devs.erase(devs.begin() + worst);
                free_mib.erase(free_mib.begin() + worst);
            }
            if (short_any && devs.size() == 1) {
                std::fprintf(stderr, "strata generate: short of GPU memory: no other card has room, so the model runs on CUDA%d alone with the experts that do not fit in RAM\n", devs[0]);
                multi_gpu = false;
                split_auto = false;
                split_devs.clear();
                split_at.clear();
                o.layer_split.clear();
                short_any = false;
            }
            if (short_any) {
                const int nd = (int) devs.size();
                split_devs.assign(devs.begin() + 1, devs.end());
                // each remaining card's room; the last carries the head, and the drafter unless the room does not pay for it (the drafter step drops it then)
                room_mib.assign((size_t) nd, 0);
                int64_t sum_room = 0;
                for (int i = 0; i < nd; ++i) { room_mib[(size_t) i] = base_room_of(i); sum_room += room_mib[(size_t) i]; }
                room_mib[(size_t) nd - 1] -= wm.head_mib + (sum_room - wm.head_mib >= per_layer_f * g.n_layers + wm.drafter_mib ? wm.drafter_mib : 0);
                std::vector<int64_t> share((size_t) nd, 0);
                sw::proportional_layers(room_mib.data(), nd, g.n_layers, share.data());
                bool share_ok = true;
                for (const int64_t s : share) share_ok = share_ok && s >= 2;
                if (share_ok) {
                    std::vector<int64_t> at;
                    std::string ks;
                    int64_t run = 0;
                    for (int i = 0; i + 1 < nd; ++i) { run += share[(size_t) i]; at.push_back(run); ks += (ks.empty() ? "" : ",") + std::to_string(run); }
                    std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d has %.2f GiB free and a layer split card needs about %.2f GiB beside a full copy of the dense weights "
                                         "(%.2f GiB): the split is chosen from what each card has free (--layer-split %s) and every card loads only the dense weights of its own layers "
                                         "(STRATA_STAGE_TRIM=0 keeps the copies)\n",
                                 short_dev, (double) short_free / 1024.0,
                                 (double) sw::untrimmed_need_mib(wm, true) / 1024.0, (double) wm.dense_mib / 1024.0, ks.c_str());
                    split_at = at;
                    split_auto = false;
                    trim_asked = true;
                    o.layer_split = ks;
                }
            }
        }
    }
    // CUDA0'S ARENA IS BUILT BEFORE THE SPLIT IS KNOWN.  The prompt path's PLE tables are read out of it, here,
    // and `auto` does not choose its split points until 590 lines below - so CUDA0 can still only trim an
    // explicit --layer-split.  It is also the cheapest card to leave alone: it runs layers 0..split_at[0] on
    // the device with the most room, and on the 4-way IQ3_S rig the trim buys it 4598 -> 4608 slots.  The VRAM
    // is on the later cards (CUDA2 1241 -> 2485, CUDA3 533 -> 1772), and THEIR weights load after the search,
    // so they trim under `auto` too - see the block past it.
    const bool stage_trim = trim_asked && multi_gpu && !split_auto && !split_at.empty();
    if (o.trim_stage_weights && !multi_gpu)
        std::fprintf(stderr, "strata generate: --trim-stage-weights needs a layer split across GPUs (ignored)\n");
    // Only `--layer-split auto` WITH the trim asked loads the later stages' weights after the split search (they can
    // only be trimmed once the split is known).  Every other run keeps the 0.1.39 order: weights first, then the
    // search - so without the trim nothing about a split moves.
    bool late_weights = trim_asked && multi_gpu && split_auto;
    // THE LAYER SPLIT'S SPILL WALK, STEP 1 (include/strata/common/experts/split_walk.hpp): a card too short for the whole dense set loads only its own layers' weights.  Decided below,
    // once the dense set's size is known and before any later stage loads; `walk_trim` is that decision, `later_trim` the stages' trim (asked for, or the walk's).
    bool walk_trim = false;
    // keep_routers: CUDA0 with --mmap-experts keeps every layer's router (ffn_gate_inp, ~2.5 MiB a layer) - the file
    // tier's routing-aware prefetch (RouterLookahead, below) copies all of them from CUDA0's arena, and would
    // otherwise turn itself off
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out, bool keep_routers) {
        std::FILE* f = std::fopen((o.pack + "/index.txt").c_str(), "rb");
        if (!f) return;
        char line[1024], name[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
            const std::string n = name;
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) continue;
            if (keep_routers && n.ends_with(".ffn_gate_inp.weight")) continue;
            const int64_t l = std::atoll(name + 4);
            if (l < lb || l >= le) out.insert(n);
        }
        std::fclose(f);
    };
    if (stage_trim) {
        add_foreign(0, split_at[0], skip, o.mmap_experts);
        strata::core::NativeDense::set_layer_range(0, (int) split_at[0]);
        std::fprintf(stderr, "strata generate: layer split: CUDA0 loads the dense weights of layers 0-%lld only\n",
                     (long long) split_at[0] - 1);
    }
    // STRATA_HC_REQ8=1 (opt-in, changes the output): the hyper-connection projections are not loaded as BF16;
    // load_hc_q8 below requantizes them to int8 + an fp32 scale per 32 values (~0.56x the bytes), and the freed VRAM
    // goes to the expert cache like any other.  (Upstream's STRATA_HC_Q8 is a different thing: the GGUF's Q8_0 bytes.)
    const bool hc_req8_env = std::getenv("STRATA_HC_REQ8") != nullptr && std::atoi(std::getenv("STRATA_HC_REQ8")) != 0;
    if (hc_req8_env && multi_gpu)
        std::fprintf(stderr, "strata generate: WARNING: STRATA_HC_REQ8 does not support a layer split yet - the BF16 "
                             "projections are used\n");
    const bool hc_q8 = hc_req8_env && !multi_gpu;
    if (hc_q8) strata::core::WeightTable::hc_q8_names(g.n_layers, skip);
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    if (const cudaError_t ce = strata::core::allocate_device(&arena, pool_bytes,
        {"weights", "Device allocations", "Dense weight arena"}); ce != cudaSuccess) {
        // #486: the arena is the first large allocation and its size does not depend on the context, so what is
        // missing is held by something else: say how much was free
        cudaGetLastError();
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
        // HIP sees no device at all and /dev/kfd exists but this user cannot open it: the missing render / video group,
        // not another program holding the memory (0 MiB of 0 MiB would blame the wrong thing)
        {
            int n_hip = 0;
            if ((cudaGetDeviceCount(&n_hip) != cudaSuccess || n_hip < 1) && ::access("/dev/kfd", F_OK) == 0 &&
                ::access("/dev/kfd", R_OK | W_OK) != 0) {
                cudaGetLastError();
                std::fprintf(stderr, "strata generate: no GPU is visible (%s) and this user cannot open /dev/kfd "
                                     "(permission denied): it is not another program holding the GPU. Add your user to "
                                     "the render and video groups - sudo usermod -aG render,video $USER - then log out "
                                     "and in again\n", cudaGetErrorString(ce));
                return 1;
            }
        }
#endif
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "strata generate: cudaMalloc(%llu) for the weight arena failed (%s): %llu MiB of %llu "
                             "MiB VRAM free on this GPU. The arena is allocated first, before the KV and expert "
                             "caches: another program (or an engine that is still exiting) holds the rest - "
                             "nvidia-smi / rocm-smi lists them\n",
                     (unsigned long long) pool_bytes, cudaGetErrorString(ce), (unsigned long long) (free_b >> 20),
                     (unsigned long long) (total_b >> 20));
        {   // #3797: every card of the run, and who else is on it
            std::vector<int> report_devs;
            int cur = 0;
            if (cudaGetDevice(&cur) == cudaSuccess) report_devs.push_back(cur); else (void) cudaGetLastError();
            for (const int d : split_devs) report_devs.push_back(d);
            print_startup_gpu_report(report_devs);
        }
        return 1;
    }
    mem_mark("the CUDA context and the weight arena");
    auto load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    if (hc_q8) {
        uint64_t qb = 0;
        if (!wt.load_hc_q8(o.pack, qb, err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "strata generate: STRATA_HC_REQ8: hyper-connection projections as int8 + fp32/32 scales, "
                             "%llu MiB\n", (unsigned long long) (qb >> 20));
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s in %.1f s (%zu canonical tensors "
                         "skipped: served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), load_s(), skip.size());
    mem_mark("the pack weights");

    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights, in %.1f s\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }
    // ---- THE LAYER SPLIT'S SPILL WALK, STEP 1: THE TRIM.  CUDA0 holds the whole dense set now; every later card is about to load a copy of it (3.5 GiB for IQ3_S), which on a
    // card that is short (two 8 GB cards: 6 of 6 starts failed in 0.2.10) is what leaves its expert cache and its prompt path no room.  A card that cannot afford the copy
    // beside its session, its head, its drafter and a working cache loads only the dense weights of its own layers instead (--trim-stage-weights, bit-exact in itself:
    // the cache just gets the room).  A card with room never reaches this; STRATA_STAGE_TRIM=0 or STRATA_SPILL_OFF=1 keeps the old copy.  CUDA0 stays whole under
    // `--layer-split auto` (its weights load before the split is known); it runs the card's share of layers the search gives it.
    if (multi_gpu && !trim_asked && !split_devs.empty() && std::getenv("STRATA_SPILL_OFF") == nullptr &&
        !(std::getenv("STRATA_STAGE_TRIM") != nullptr && std::string(std::getenv("STRATA_STAGE_TRIM")) == "0") && spill_discrete_gpu()) {
        strata::program::split_walk::Model wm;
        wm.dense_mib = (int64_t) ((pool_bytes + (o.native_dense_gguf.empty() ? 0 : native_dense.weight_bytes())) >> 20);
        const int ns_w = (int) split_devs.size() + 1;
        wm.session_mib = (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, (g.n_layers + ns_w - 1) / ns_w) >> 20);
        if (const strata::core::WeightRef* wo_w = wt.find("output.weight"); wo_w != nullptr && !o.native_head_gguf.empty())
            wm.head_mib = (int64_t) (((uint64_t) wo_w->ne1 * (uint64_t) (g.n_embd / 256) * 176ull) >> 20);   // Q5_K: 176 bytes per 256 values
        wm.drafter_mib = (!o.mtp.empty() || o.draft == "dflash2") ? strata::program::stage_reserve::kDrafterMib : 0;
        wm.slot_mib = std::max<int64_t>(1, (int64_t) ((strata::kernels::cpu::expert_layout().max_blob + (1u << 20) - 1) >> 20));
        wm.reserve_mib = (o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib) + strata::program::stage_reserve::kWindowMib;
        for (size_t i = 0; i < split_devs.size(); ++i) {
            const strata::core::OnDevice on_w(split_devs[i]);
            size_t fb_w = 0, tb_w = 0;
            if (cudaMemGetInfo(&fb_w, &tb_w) != cudaSuccess) { (void) cudaGetLastError(); continue; }
            const bool last_w = i + 1 == split_devs.size();
            if (strata::program::split_walk::trim_needed((int64_t) (fb_w >> 20), wm, last_w)) {
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d has %.2f GiB free and a layer split card needs about %.2f GiB beside a full copy of the dense weights "
                                     "(%.2f GiB): each later card loads only the dense weights of its own layers (STRATA_STAGE_TRIM=0 keeps the copy)\n",
                             split_devs[i], (double) fb_w / 1073741824.0, (double) strata::program::split_walk::untrimmed_need_mib(wm, last_w) / 1024.0,
                             (double) wm.dense_mib / 1024.0);
                walk_trim = true;
                break;
            }
        }
        if (walk_trim && split_auto) late_weights = true;
    }
    // the stages' dense weights are trimmed to their own layers: asked for (an explicit split), or by the walk; under `auto` they load after the split search (late_weights)
    const bool later_trim = stage_trim || (walk_trim && !split_auto && !split_at.empty());
    stage_weights_trimmed = later_trim || late_weights;

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract; the per-stage
    // and dump measurements need the unfused layout of R, so they keep the old kernels.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && o.dump_layers.empty() &&
                                     o.dump_halves.empty() && !o.stage_timing);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/common/kernels/mrope.hpp).
    // It is the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (strata::core::allocate_device(&d_mrope, mrope_host.size() * sizeof(int32_t),
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot allocate the image position table\n");
            return 1;
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!load_control_vectors(o, g, cvec_summary, cvec_digest, ce)) {
            std::fprintf(stderr, "strata generate: control vector: %s\n", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    // **THE ENGINE RAN ON THE LEGACY DEFAULT STREAM, WHICH ON WDDM IS THE SLOW PATH.**  All four session
    // calls - `session_capture`, `session_replay`, `session_token` and `session_loop` - were handed `nullptr`,
    // i.e. stream 0.  `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: cannot create the main stream\n");
        return 1;
    }
    main_cs = (void*) main_stream;

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                  (uint64_t) g.n_head * g.head_dim +
                  (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        if (strata::core::allocate_host((void**) &half_stage, n * sizeof(float), cudaHostAllocDefault,
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the half-dump staging buffer\n");
            return 1;
        }
        ss.block.dump = half_stage;
    }

    if (strata::core::doorbell_init(g, K, db) == 0) {
        std::fprintf(stderr, "strata generate: doorbell_init failed\n");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    ple_emb_host.resize((size_t) strata::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    if (!o.ple_gguf.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" || o.ple_io == "ram" ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        // Keep the SSD awake while rows are being asked for (PleReader::set_keepalive): some SSDs stall the first
        // reads 50-150 ms after ~250 ms without a command.  STRATA_SSD_KEEPALIVE = ms without a read before one
        // page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the last row
        // request that this goes on (default 60; then the SSD may sleep until the next request).
        {
            const char* ka = std::getenv("STRATA_SSD_KEEPALIVE");
            const char* kw = std::getenv("STRATA_SSD_KEEPALIVE_WINDOW");
            pio.keepalive_ms = ka != nullptr && *ka ? std::clamp(std::atof(ka), 0.0, 10000.0) : 100.0;
            pio.keepalive_window_s = kw != nullptr && *kw ? std::clamp(std::atof(kw), 1.0, 86400.0) : 60.0;
            if (pio.mode != strata::kernels::PleIo::Direct || !pio.io_thread) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // 0.2.7 (R9700 measurement): keep the table resident in RAM when the PC can afford it.  The rows of a
        // prompt token are random 90 B reads in a 28.8 GB table, so even a cached direct read pays ~100 us a
        // token and a mapped read pays a minor fault a row.  Measured on p620 (R9700, IQ3_S, 8K prompt, 24576
        // expert pairs of which 13207 resident): PLE block 845 ms direct / 451 ms mmap / 132 ms resident,
        // prompt 2125 / 2368 / 2611 tok/s, and the answer md5 was identical in all three.  The RAM cost is the
        // table itself; the policy needs 8 GiB of headroom on top of it (available_host_bytes accounts for
        // cgroups and --memory-limit-mib), an explicit --ple-io always wins, and STRATA_PLE_AUTO_RAM=0 turns it off.
        // The expert arena is filled AFTER this point and wants its own RAM (51 GiB for IQ3_S), so the table is admitted
        // only when RAM holds table + arena + headroom (strata/models/qwen38/ple_residency.hpp).  Measured on the 64 GB RTX 5070 PC
        // (2026-10-26): table + 8 GiB alone admitted it, the table's lock took the lockable memory and the expert load
        // died with error 1453 (ERROR_WORKING_SET_QUOTA).
        bool ple_auto_ram = false;
        if (pio.mode == strata::kernels::PleIo::Direct && !o.ple_io_explicit) {
            const strata::kernels::PleIoOptions direct_pio = pio;
            const char* auto_ram = std::getenv("STRATA_PLE_AUTO_RAM");
            bool allowed = strata::core::ple_auto_ram_allowed(auto_ram, strata::qwen_defaults_026());
#if defined(__APPLE__)
            // Not on the Mac unless asked (STRATA_PLE_AUTO_RAM=1): unified memory with the experts mapped from the file cache.  M5 Max,
            // IQ3_S, 3 interleaved pairs: with the 26.8 GiB table locked, a cold 4K prompt after decode work is 10.6% slower (-10.7 -10.6
            // -10.0) than the same build with direct reads, the footprint is 42 GB against 15 and free memory dips to 19 GB against 46;
            // decode is the same (+0.2..0.6% median, inside the spread).  The Mac had direct reads until sample_host_memory learned
            // to read macOS RAM (the conversation cache fix), which let this policy fire there.
            if (!(auto_ram && std::string(auto_ram) == "1")) allowed = false;
#endif
            const uint64_t need = ple_table.table_bytes();
            const auto available = strata::core::available_host_bytes();
            // every routed expert's blob, the arena the engine copies into RAM at start (strata-q38: its RAM pool)
            uint64_t experts_ram = ram_pool_bytes >= 0 ? (uint64_t) ram_pool_bytes : 0;
            if (ram_pool_bytes < 0) {
                const auto& elay = strata::kernels::cpu::expert_layout();
                for (int64_t l = 0; l < elay.n_layers; ++l)
                    experts_ram += elay.blob_bytes(l) * (uint64_t) elay.n_expert;
            }
            if (allowed && need > 0 && strata::core::ple_resident_admit(available, need, experts_ram, 8ull << 30)) {
                ple_table.close();
                pio.mode = strata::kernels::PleIo::Mmap;
                pio.lock = true;
                pio.keepalive_ms = 0;
                std::string ram_err;
                if (ple_table.open(o.ple_gguf, ram_err, pio)) {
                    ple_auto_ram = true;
                    ple_lock_by_policy = ple_table.locked();
                    std::fprintf(stderr, "strata generate: --ple-io ram chosen by the residency policy (the table is "
                                         "%.1f GiB, the expert arena still to fill %.1f GiB and %.1f GiB of RAM is available; "
                                         "STRATA_PLE_AUTO_RAM=0 keeps the direct reads, --ple-io direct forces them)\n",
                                 (double) need / (1ull << 30), (double) experts_ram / (1ull << 30),
                                 (double) *available / (1ull << 30));
                } else {
                    // Back to what was asked for; a table that cannot be mapped must not stop the engine.
                    std::fprintf(stderr, "strata generate: resident PLE table failed (%s); keeping direct reads\n",
                                 ram_err.c_str());
                    pio = direct_pio;
                    if (!ple_table.open(o.ple_gguf, err, pio)) {
                        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                        return 1;
                    }
                }
            }
        }
        // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
        // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
        if (pio.mode == strata::kernels::PleIo::Direct && std::getenv("STRATA_PLE_PROBE") == nullptr) {
            constexpr double kSlowRowsPerS = 15000.0;   // ~60 MB/s of 4 KiB pages; a healthy NVMe is >100k
            const double rps = ple_direct_rows_per_s(ple_table);
            if (rps >= kSlowRowsPerS)
                std::fprintf(stderr, "strata generate: PLE direct read probe: %.0f rows/s (fine)\n", rps);
            if (rps > 0 && rps < kSlowRowsPerS) {
                // #1984 #1971 #2040: on no platform is the switch made on its own.  Windows: the mapped reads stalled prompts (the first chunk waits for 16 serial page faults
                // a token from a cold table with no progress mark).  Linux SATA SSD: the 15K rows/s bar is NVMe-calibrated, a SATA SSD
                // tops out at 10-14K with direct's 16 reads in flight and mapped reads were 7x slower.  The switch is only made when asked for
                // (STRATA_PLE_AUTO_MMAP=1, the 0.1.42 behaviour); otherwise the slow probe warns and direct is kept.
                bool auto_mmap = !o.ple_io_explicit;
                if (const char* am = std::getenv("STRATA_PLE_AUTO_MMAP"); am == nullptr || std::atoi(am) == 0) auto_mmap = false;
                if (auto_mmap) {
                    std::fprintf(stderr, "strata generate: --ple-io direct measures slow here (%.0f rows/s, about %.0f MB/s of "
                                         "random 4 KiB reads): using --ple-io mmap instead (the OS file cache holds the pages; "
                                         "--ple-io direct forces the old mode, --ple-io ram keeps the table in RAM)\n",
                                 rps, rps * 4096.0 / 1e6);
                    ple_table.close();
                    pio.mode = strata::kernels::PleIo::Mmap;
                    pio.keepalive_ms = 0;
                    if (!ple_table.open(o.ple_gguf, err, pio)) {
                        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                        return 1;
                    }
                } else {
                    std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                         "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 and trip the "
                                         "stall watchdog (#1425, #1629). --ple-io mmap was 40-100x faster on some DRAM-less NVMe drives but 7x slower on a SATA SSD (#2040), so it is not chosen for you; "
                                         "--ple-io ram needs the RAM for the table%s\n", rps, rps * 4096.0 / 1e6,
                                 o.ple_io_explicit ? "" : " (kept direct: pass --ple-io mmap or --ple-io ram to change it; in a config file's \"args\" they are two entries: \"--ple-io\", \"mmap\")");
                }
            }
        }
#if !defined(_WIN32)
        if (pio.mode == strata::kernels::PleIo::Direct) {
            // #605: --ple-io direct's random reads (up to --ple-inflight at once) are for SSDs; a rotational disk
            // can take longer than the stall watchdog's 60 s for one prompt chunk's rows.  Recommended, not forced.
            std::string dev;
            if (!under_wddm() && on_rotational_disk(o.ple_gguf, dev) == 1)   // (WSL's virtual disk says rotational)
                std::fprintf(stderr, "strata generate: WARNING: the n-gram table (%s) is on a rotational disk (%s): "
                                     "--ple-io direct reads it at random and can stall a prompt for minutes (#605). "
                                     "--ple-io ram (it then needs RAM for the table) or the model on an SSD is "
                                     "recommended\n", o.ple_gguf.c_str(), dev.c_str());
        }
#endif
        if (pio.lock)
            std::fprintf(stderr, "strata generate: PLE table %s (%s) in %.1f s\n",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         ple_auto_ram ? "--ple-io ram chosen by the residency policy" : "--ple-io ram",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        if (pio.keepalive_ms > 0)
            std::fprintf(stderr, "strata generate: the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request "
                                 "(STRATA_SSD_KEEPALIVE=0 turns it off)\n", pio.keepalive_ms, pio.keepalive_window_s);
        else if (pio.mode == strata::kernels::PleIo::Direct)
            std::fprintf(stderr, "strata generate: SSD keep-alive off: the SSD may fall asleep between reads\n");
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8) || !wk->native_q8_1) {
                std::fprintf(stderr, "strata generate: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        // The conv1d kernel reads F16, so this cast is a claim about the pack's storage.  A checkpoint that keeps
        // the tensor F32 (Q8_0, UD-Q4_K_XL) would hand the kernel the low halves of the f32 words - not an error,
        // a plausible wrong layer-1 routing.  tools/iq_pack.py narrows it (index kind 3); a pack that did not is
        // refused here (#255, gopinath87607).  F16 is index kind 5 or 3 (F16InF32) in a native pack and kind 0
        // (verbatim 2-byte F16) in the canonical Q2_0 pack; BF16 (kind 4) has the same size and is not F16.
        const bool f16 = wc->kind == strata::core::WeightKind::F16InF32 ||
                         (wc->kind == strata::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            std::fprintf(stderr, "strata generate: blk.1.ple_conv1d.weight is not stored as F16 (pack index kind %d, "
                                 "%llu B for %lld values); the PLE conv1d kernel reads F16 - repack with "
                                 "tools/iq_pack.py\n",
                         (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
            return 1;
        }
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = ple_consts_run;
        for (int h = 0; h < strata::kernels::PLE_N_HEADS; ++h) {   // the hash may only reach rows the table has
            if (ple_consts_run.offset[h] + ple_consts_run.vocab[h] > ple_table.rows()) {
                std::fprintf(stderr, "strata generate: PLE head %d reaches row %llu but the table has %llu rows (the model file's hash constants do not fit its PLE table)\n",
                             h, (unsigned long long) (ple_consts_run.offset[h] + ple_consts_run.vocab[h]), (unsigned long long) ple_table.rows());
                return 1;
            }
        }
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history - the session is
        // now allocated after the layer-split search (see the carve), and the wiring lands there
        ss.ple.emb_host = ple_emb_host.data();
        if (strata::core::allocate_device((void**) &ple_emb_dev, (size_t) strata::kernels::NG_N_EMBD * 4,
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &ple_scratch, strata::core::ple_run_scratch_bytes(),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the PLE buffers failed\n");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    if (strata::core::allocate_device(&d_parts, (size_t) K * g.n_embd * 4,
        {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the parts buffer failed\n");
        return 1;
    }

    // ---- --split-skip-if-fits: before any later stage loads, does CUDA0 alone hold every profiled pair?  What it
    // still has to allocate on one GPU is the whole session (the KV of every layer), the drafter and the head
    // (kDrafterMib below), the verify windows and the reserve; the prompt path borrows from the cache.  If the
    // profile's pairs fit in what is left, a split would only add the hand-offs: run on CUDA0 alone.
    if (multi_gpu && split_auto && o.split_skip_if_fits) {
        std::vector<std::pair<int32_t, int32_t>> prof;
        int64_t pslots = 0;
        std::string perr;
        if (remote_caches || o.expert_profile.empty() ||
            !strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, prof, pslots, perr)) {
            std::fprintf(stderr, "strata generate: --split-skip-if-fits: %s; the split stays\n",
                         remote_caches ? "remote expert caches are in use"
                                       : perr.empty() ? "no expert profile" : perr.c_str());
        } else {
            const auto& lay = strata::kernels::cpu::expert_layout();
            int64_t pairs_bytes = 0;
            for (const auto& pr : prof)
                pairs_bytes += native_pack ? strata::core::ExpertCache::slot_stride((int64_t) nib::slot_bytes(pr.first), slot_guard)
                                           : (int64_t) lay.max_blob;
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            const int64_t session = (int64_t) strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers);
            const int64_t held_back = session + (((int64_t) o.vram_reserve_mib + 1000 + 96) << 20);   // + drafter/head, windows
            const int64_t room = (int64_t) fb - held_back;
            cudaDeviceProp dp{};
            cudaGetDeviceProperties(&dp, 0);
            if (pairs_bytes <= room) {
                std::fprintf(stderr, "strata generate: layer split skipped (--split-skip-if-fits): CUDA0 (%s) holds all "
                                     "%zu profiled pairs (%.2f GiB) with the session (%.2f GiB, %lld-token context), "
                                     "the drafter and the reserve: %.2f GiB free, %.2f GiB to spare - one GPU\n",
                             dp.name, prof.size(), (double) pairs_bytes / 1073741824.0, (double) session / 1073741824.0,
                             (long long) o.max_context, (double) fb / 1073741824.0,
                             (double) (room - pairs_bytes) / 1073741824.0);
                multi_gpu = false;
                split_auto = false;
                split_devs.clear();
                split_at.clear();
                o.layer_split.clear();
            } else {
                std::fprintf(stderr, "strata generate: --split-skip-if-fits: CUDA0 (%s) would hold only %.2f of the "
                                     "profile's %.2f GiB (%.2f GiB free, %.2f GiB for the session, drafter and "
                                     "reserve): the split stays\n", dp.name,
                             (double) std::max<int64_t>(room, 0) / 1073741824.0, (double) pairs_bytes / 1073741824.0,
                             (double) fb / 1073741824.0, (double) held_back / 1073741824.0);
            }
        }
    }
    strata::core::Verifier::set_commit_async(!multi_gpu);   // see Verifier::set_commit_async
    // --draft-fallback none: the MTP layer is not loaded and every "is there an MTP" test below sees --mtp as absent; DFlash2 drafts alone.  The draft
    // vocabulary file is taken from the --mtp directory first.  Anything DFlash2 cannot do here is an error at start (below), never a silent MTP.
    if (o.draft == "dflash2" && o.draft_auto && o.draft_fallback == "none") {
        std::fprintf(stderr, "strata generate: --draft auto keeps both drafters (a request picks one); --draft-fallback none is for --draft dflash2\n");
        return 2;
    }
    if (o.draft == "dflash2" && !o.draft_auto && o.draft_fallback != "mtp") {   // (unset or none)
        if ((o.mtp.empty() && o.mtp_draft_vocab.empty()) || o.spec < 2 || o.draft_model.empty()) {   // --mtp-draft-vocab FILE alone is enough: no MTP folder is needed
            std::fprintf(stderr, "strata generate: --draft dflash2 drafts without the MTP layer (--draft-fallback mtp loads it too) and needs --draft-model PACK, --spec 2 or more and --mtp-draft-vocab FILE (or --mtp DIR, for DIR/draft_vocab.bin)\n");
            return 2;
        }
        if (!strata::core::Dflash2Drafter::head_native()) {   // without the vocabulary subset the DFlash2 would run its head over all of the target's rows (+0.7 ms a round measured on Metal) without a word.
            // Metal alone has a packaged default subset to fall back to (the Mac's MTP folder is empty by design); everywhere else a missing file is the error.
            std::string packaged;
            bool whole_head_ok = false;
#if defined(STRATA_GPU_METAL)
            packaged = strata::metal_engine::draft_vocab_default();
            whole_head_ok = strata::metal_engine::draft_vocab_whole_head();
#endif
            const strata::spec::Df2Vocab dv = strata::spec::df2_vocab_resolve(o.mtp, o.mtp_draft_vocab, packaged, whole_head_ok, [](const std::string& p) {
                std::error_code vec;
                return std::filesystem::exists(p, vec);
            });
            if (!dv.ok) {
                std::fprintf(stderr, "strata generate: %s\n", dv.why.c_str());
                return 2;
            }
            if (dv.packaged) std::fprintf(stderr, "strata generate: --draft dflash2: no draft vocabulary in the MTP folder: using the packaged default subset %s\n", dv.path.c_str());
            df2_vocab_file = dv.path;
        }
        o.mtp.clear();
        df2_solo = true;
    }
    // --pipeline-windows (opt-in): one conversation's windows with the two stages of a layer split overlapped.  Decided
    // here, before any stage sizes its expert cache (the second verifier per stage and the snapshots are allocated
    // after the caches, so their room is kept out of them).  What it does not support turns it off, said once.
    // 0.1.43: not asked for (-1) = on (2) for gfx12 layer splits of three or more stages, where it was measured (4x R9700);
    // with two stages it loses (the drafter's chain shares the last card with its window).  STRATA_PIPELINE_AUTO=0 keeps
    // it off, --pipeline-windows N asks for it as before.  Anything it does not support leaves it off, said nothing.
    // 0.1.44: the drafter can sit on the first stage's card (see mtp_first below) when its draft head is the subset file: the main head
    // stays on the last card and only the subset's rows are copied over once
    const bool mtp_first_ok = [&] {
        if (o.mtp.empty() || o.spec < 2) return false;
        if (const char* fh = std::getenv("STRATA_MTP_FULL_HEAD"); fh != nullptr && fh[0] == '1') return false;
        std::error_code ec;
        return o.mtp_draft_vocab.empty() ? std::filesystem::exists(std::filesystem::path(o.mtp) / "draft_vocab.bin", ec)
                                         : std::filesystem::exists(o.mtp_draft_vocab, ec);
    }();
    // --batch with the solo path pipelined (opt-in, STRATA_PW_WITH_BATCH=1).  With slots, a request that is alone runs the solo
    // path (GEN, with drafts; serve/server.py goes back to it when a slot is left alone, #857), and that path can keep
    // --pipeline-windows: both parities' verifiers share the stage's session (ver_b.init(.., gs.ss, ..)), so when the solo
    // request is STOPped for a newcomer, copy_to_slot reads a whole state from `ss` after the pipeline has drained.  Batch
    // windows never use the second verifiers.  With --batch-mtp the slot drafters go where the solo drafter is: the first
    // stage's card when mtp_first (they share its weights and draft head, and copy the last stage's rows over as it does).
    // Without this, --batch turns --pipeline-windows off and a request alone loses its overlap (2x RX 6900 XT, IQ3_S, 0.1.45:
    // 75 -> 66.5 tok/s at 32K with "parallel": 2).
    const bool pw_with_batch = [] {
        const char* e = std::getenv("STRATA_PW_WITH_BATCH");
        return e != nullptr && e[0] == '1';
    }();
    const bool batch_blocks_pw = o.batch != 0 && !pw_with_batch;
    const bool pw_auto = o.pipeline_windows < 0;
    if (pw_auto) {
        static const char* ae = std::getenv("STRATA_PIPELINE_AUTO");
        static const char* am = std::getenv("STRATA_PIPELINE_AUTO_MIN_STAGES");
        // two stages pay off only with the drafter on the first card (0.1.44, mtp_first): its chain no longer shares the last card with the window
        const size_t min_st = am != nullptr ? (size_t) std::max(2, std::atoi(am))
                                            : (mtp_first_ok && mtp_first_default_on() ? 2 : 3);
        bool on = o.serve && multi_gpu && !split_same && split_devs.size() + 1 >= min_st && !(ae != nullptr && std::atoi(ae) == 0);
#if defined(STRATA_USE_HIP)
        on = on && decode_defaults_arch_ok();
#else
        on = false;
#endif
        const bool df2_alone = o.draft == "dflash2" && !o.draft_auto && o.draft_fallback != "mtp";
        if (on && df2_alone)
            std::fprintf(stderr, "strata generate: --pipeline-windows is off by default here: DFlash2 drafts from the finished window's taps on "
                                 "every stage (--draft mtp keeps the pipeline)\n");
        on = on && !o.mtp.empty() && o.spec >= 2 && !batch_blocks_pw && o.peer_device < 0 && !remote_caches &&
             !o.adapt_async && !df2_alone;
        o.pipeline_windows = on ? 2 : 0;
        if (on)
            std::fprintf(stderr, "strata generate: --pipeline-windows 2 is on by default for a layer split of %zu stages "
                                 "(--pipeline-windows 0 or STRATA_PIPELINE_AUTO=0 turns it off)\n", split_devs.size() + 1);
    }
    if (o.pipeline_windows > 0) {
        const char* off = !o.serve ? "it needs --serve"
                        : split_same ? "it needs each stage of the layer split on its own GPU"
                        : !multi_gpu ? "it needs a layer split on two GPUs"
                        : split_devs.empty() ? "it needs a layer split into two or more stages"
                        : split_devs.size() > 1 && o.pipeline_windows == 1 ? "--pipeline-windows 1 needs exactly two stages"
                        : split_devs.size() > 1 && o.adapt_async ? "not with --adapt-async on three or more stages"
                        : o.draft == "dflash2" && !o.draft_auto && o.draft_fallback != "mtp"
                              ? "DFlash2 drafts from the finished window's taps on every stage, so the next window's drafts cannot exist while this one still runs (the speculative window the MTP's chain makes has no DFlash2 counterpart; see RESULTS-DF2MGPU.md)"
                        : o.mtp.empty() || o.spec < 2 ? "it needs the MTP drafter (--mtp, --spec)"
                        : batch_blocks_pw ? "not with --batch slots (STRATA_PW_WITH_BATCH=1 keeps it for the solo path)"
                        : o.peer_device >= 1 ? "not with --peer-device"
                        : remote_caches ? "not with the helper caches (--expert-cache-device1..3, --remote-expert-opt)"
                        : nullptr;
        if (off != nullptr) {
            std::fprintf(stderr, "strata generate: --pipeline-windows %d is off: %s\n", o.pipeline_windows, off);
            o.pipeline_windows = 0;
        } else if (o.batch != 0) {
            std::fprintf(stderr, "strata generate: --pipeline-windows %d stays on for the solo path beside %d batch slots "
                                 "(STRATA_PW_WITH_BATCH=1); batch windows run without it\n", o.pipeline_windows, o.batch);
        } else if (!pw_auto && o.pipeline_windows >= 2 && split_devs.size() == 1) {
            // #2637: the default (above) takes two stages only with the draft layer on the first card (mtp_first); asked for by
            // hand it was taken without a word.  Recommend, never force: it stays on as asked, and the line says what it costs.
            const char* e0 = std::getenv("STRATA_MTP_STAGE0");
            const bool first_card = mtp_first_ok && (e0 != nullptr ? std::atoi(e0) != 0 : mtp_first_default_on());
            if (!first_card)
                std::fprintf(stderr, "strata generate: note: --pipeline-windows %d on a layer split of two stages without the draft layer "
                                     "on the first card (%s): the draft chain then shares the last card with its window and the "
                                     "pipeline lost to the serial loop on 2x R9700 before 0.1.44 (docs/MULTI_GPU.md); kept as asked, "
                                     "--pipeline-windows 0 turns it off\n", o.pipeline_windows,
                             !mtp_first_ok ? "it needs --mtp with --spec 2 or more and the draft head's subset file (draft_vocab.bin)"
                                           : "STRATA_MTP_STAGE0=1 asks for it");
        }
    }
    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same WDDM reason)
    // One stage's dense weights on its device (the caller has made it current): `trim` = only layers [lb, le).
    auto load_stage_weights = [&](GpuStage& st, int64_t lb, int64_t le, bool trim) -> bool {
        std::set<std::string> skip_s = skip_base;
        uint64_t pool_s = pool_bytes;
        if (trim) {
            add_foreign(lb, le, skip_s, false);
            strata::core::NativeDense::set_layer_range((int) lb, (int) le);
            if (!strata::core::WeightTable::pool_bytes(o.pack, pool_s, err, &skip_s)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return false;
            }
            std::fprintf(stderr, "strata generate: layer split: CUDA%d loads the dense weights of layers %lld-%lld only\n",
                         st.dev, (long long) lb, (long long) le - 1);
        } else {
            skip_s = skip;
        }
        void* arena_s = nullptr;
        if (strata::core::allocate_device(&arena_s, pool_s,
            {"weights", "Device allocations", "Dense weight arena"}) != cudaSuccess ||
            !st.wt.load(o.pack, arena_s, pool_s, err, skip_s.empty() ? nullptr : &skip_s)) {
            cudaGetLastError();
            size_t free_b = 0, total_b = 0;   // #486: what that card had free
            cudaMemGetInfo(&free_b, &total_b);
            std::fprintf(stderr, "strata generate: layer split, CUDA%d weights: %s (%llu MiB needed, %llu MiB of %llu "
                                 "MiB free on that card)\n", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str(),
                         (unsigned long long) (pool_s >> 20), (unsigned long long) (free_b >> 20),
                         (unsigned long long) (total_b >> 20));
            {   // #3797: every card of the run, and who else is on it
                std::vector<int> report_devs;
                int cur = 0;
                if (cudaGetDevice(&cur) == cudaSuccess) report_devs.push_back(cur); else (void) cudaGetLastError();
                for (const int d : split_devs) report_devs.push_back(d);
                print_startup_gpu_report(report_devs);
            }
            return false;
        }
        if (!o.native_dense_gguf.empty() && !st.dense.load(o.native_dense_gguf, st.wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d native dense projections: %s\n", st.dev,
                         err.c_str());
            return false;
        }
        return true;
    };
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        double free_gib = 0;   // (own_stages: a stage on CUDA0, which the preflight takes for a helper's mistake)
        if (st.dev != primary_device && !strata::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::OnDevice on(st.dev);
        // Later stages' weights load here (0.1.39), except `--layer-split auto` with the trim asked: those load after
        // the split search (see late_weights), so `auto` can trim them too.
        if (!late_weights) {
            const int64_t lb = later_trim ? split_at[i] : 0;
            const int64_t le = later_trim ? (i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers) : 0;
            if (!load_stage_weights(st, lb, le, later_trim)) return 1;
        }
        // THE SESSION AND THE HEAD WAIT FOR THE SPLIT SEARCH.  `session_bytes` prices a stage's session by its
        // LAYER RANGE (the carve - every stage used to hold all 48 layers' state whatever it ran), so the
        // sessions are allocated after the search below has set `st.lb`/`st.le`; the last stage's head follows.
        if (cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st.adapt_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&st.adapt_ev, cudaEventDisableTiming) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: its streams failed\n", st.dev);
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!strata::kernels::cvec_replicate(err)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s\n", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (strata::core::allocate_device(&st.mrope, mrope_host.size() * sizeof(int32_t),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
                cudaMemcpy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d: the image position table failed\n", st.dev);
                return 1;
            }
            strata::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed).  A given
        // --pcie-frac is every stage's share and skips these probes (pcie_given); there is no per-stage setting yet.
        st.pcie_frac = o.pcie_frac;
        if (!pcie_given && native_pack) {
            std::string bursts;
            const double bw = probe_pcie_h2d_gbps(&bursts);
            if (bw > 0.0) st.pcie_frac = pcie_frac_for_gbps(bw, 0.55);
            st.link_gbps = bw;
            std::fprintf(stderr, "strata generate: layer split: CUDA%d PCIe probe %.1f GB/s (best of %s) -> pcie_frac "
                                 "%.2f\n", st.dev, bw, bursts.c_str(), st.pcie_frac);
        }
        if (!late_weights) {
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                                 "follows the split search)\n", st.dev, (double) fb / 1073741824.0);
        }
    }
    last_st = stages.empty() ? nullptr : stages.back().get();
    // 0.1.44: the drafter on the FIRST stage's card, for a two-stage pipelined decode.  Its chain then runs while that card is idle
    // (stage 0 has finished the window the chain follows), not beside the last stage's window on the last card, where the two
    // streams share the card and each takes ~2x as long (stage-1 window 11 -> 25 ms, chain 7 -> 21 ms, 2x R9700).  The chain's
    // source rows are copied over once per chain (stage_source_R), the draft head's rows once at load.
    mtp_first = [&] {
        if (!mtp_first_ok || o.pipeline_windows < 2 || stages.empty() || batch_blocks_pw) return false;
        const char* e = std::getenv("STRATA_MTP_STAGE0");
        if (e != nullptr) return std::atoi(e) != 0;   // 0 keeps the drafter on the last card, 1 asks for the first
        return mtp_first_default_on();
    }();
    // DFlash2 on a layer split: its card is the LAST stage's by default (the head rows, the embedding path and the Main role are there; only the
    // earlier stages' taps cross, a few KB a window), or the FIRST stage's with STRATA_DF2_STAGE=first (a measurement seam, not a default).
    df2_first = [&] {
        if (!multi_gpu || stages.empty() || o.draft != "dflash2" || o.draft_auto) return false;
        const char* e = std::getenv("STRATA_DF2_STAGE");
        return e != nullptr && std::string(e) == "first";
    }();

    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        // Multi-GPU: STRATA_PEER_HOT=<f> gives the peer card a share f of the HOT pairs, so both cards
        // compute routed experts every layer (the primary alone did ~25 of ~30 per layer-window).  Of the first
        // STRATA_PEER_HOT_AT (default 8700, ~ the primary's slots) ranks, every pair with floor((r+1)f) > floor(rf)
        // moves to just after that point: the primary fills past them, the peer (which takes what the primary does
        // not hold, in order) gets them first.
        if (o.peer_device >= 1) {   // default 0.45 (measured: 0.3-0.6 all better than 0; 0 = off, e.g. for the gate)
            const char* ph = std::getenv("STRATA_PEER_HOT");
            const double f = ph ? std::atof(ph) : 0.45;
            const char* pa = std::getenv("STRATA_PEER_HOT_AT");
            const size_t at = std::min(profile.size(), (size_t) (pa ? std::atoll(pa) : 8700));
            if (f > 0.0 && f < 1.0 && at > 0) {
                std::vector<std::pair<int32_t, int32_t>> keep, moved;
                for (size_t r = 0; r < at; ++r) {
                    const bool to_peer = (int64_t) ((double) (r + 1) * f) > (int64_t) ((double) r * f);
                    (to_peer ? moved : keep).push_back(profile[r]);
                }
                const size_t n_moved = moved.size();
                // the primary's share continues with the ranks after `at` until it is full; then the moved ones
                std::vector<std::pair<int32_t, int32_t>> out;
                out.reserve(profile.size());
                out.insert(out.end(), keep.begin(), keep.end());
                const size_t fill = std::min(profile.size(), at + n_moved);   // what the primary still takes
                out.insert(out.end(), profile.begin() + (long) at, profile.begin() + (long) fill);
                out.insert(out.end(), moved.begin(), moved.end());
                out.insert(out.end(), profile.begin() + (long) fill, profile.end());
                profile.swap(out);
                std::fprintf(stderr, "strata generate: STRATA_PEER_HOT %.2f: %zu of the first %zu ranked pairs moved "
                                     "behind rank %zu (the peer's)\n", f, n_moved, at, fill);
            }
        }
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // #477: the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only
    // CUDA0's pairs in `profile` below).  Empty without --expert-profile-save.
    if (!o.expert_profile_save.empty()) profile_loaded = profile;
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window, measured on
    // the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (0.33 ms on an RTX 5080, 0.50 on a
    //     3090: the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s);
    //   - an expert no cache holds costs ~190 ms per unit of routed mass: the CPU pool in decode and the PCIe stream
    //     in prompts (fitted: the sweep's best K, 26-28, is where one more layer on the faster card stops paying
    //     for the ~0.1% of the mass it pushes out of its cache);
    //     SCALED PER STAGE BY ITS PROBED LINK against the 20 GB/s x16 reference (pcie_frac_for_gbps' rule): the
    //     prompt path fetches a missed expert over ITS stage's link, so a stage on a narrow link pays
    //     proportionally more per missed unit.  Without this the search is blind to asymmetric cards:
    //     2026-10-04, 2x 4090 D with CUDA1 on a physical x1 link (1.6 GB/s), auto chose K=22 and read prompts
    //     at ~350 tok/s; with the scale (x12 on CUDA1) it chooses the fit-in-cache boundary K=38: ~1400 tok/s.
    //     With every scale 1.0 the objective is the formula the 190 ms was fitted with (unchanged).
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, 1 GiB for its windows and the drafter) is used;
    //     the routed mass a cache holds follows the COVERAGE CURVE below, not the (r+1)^-1.2 this used to use.
    // Up to 4 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // A FOUR-WAY PLACEMENT USED TO BE THE PROPORTIONAL GUESS AND NOTHING ELSE.  That formula reads only
    // `layer_ms`, so on four GPUs the free VRAM a placement leaves for expert caches - the whole point of
    // the search - never entered it.  It does now; see the ns == 4 branch below.
    // STRATA_SPLIT_MISS_MS tunes the miss cost (a slower CPU: higher).
    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache
    // is sized first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`),
    // so it withholds no VRAM of its own and this reserve is zero.  Only without borrowing - no profile to fill
    // a cache from, or --no-prefill-borrow - do the buffers take a reserve, and then this estimate stands in
    // for buffers that cannot be priced exactly yet because the sessions do not exist.  `plan_lend` uses the
    // exact `Prefill::bytes_needed` as soon as it can.
    pf_borrow = !o.no_prefill_borrow && !o.expert_profile.empty();   // (0.2.10: the auto cache sizing below can clear it, see AUTO NO-BORROW)
    // (#340: the estimate predates the streamed ring: from 1024-token chunks the prompt path also holds a ring of
    // whole expert blobs, which a split without borrowing sizes at 96 (Prefill::set_ring_override below) and books
    // here - without it a `--no-prefill-borrow` split filled the cards and the draft head no longer fit)
    const int64_t split_ring_mib =
        (multi_gpu && !pf_borrow && o.prefill_chunk >= strata::prefill::Prefill::stream_all_min_tokens())
            ? (int64_t) ((96ull * (uint64_t) strata::kernels::cpu::expert_layout().max_blob + (1ull << 20) - 1) >> 20)
            : 0;
    if (split_ring_mib > 0) strata::prefill::Prefill::set_ring_override(96);
    if (o.prefill_pipe >= 0.0) strata::prefill::Prefill::set_pipe(o.prefill_pipe);
    split_pf_mib =
        (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 + split_ring_mib : 0;
    // ---- WHAT A STAGE RESERVES, AND ON WHICH STAGE.  The flat 1 GiB this used to withhold from EVERY stage
    // after the first was booked "for its windows and the drafter", but the windows measure 75 MiB ("window up
    // to 6 tokens, 74.1 MiB of device buffers", on every boot) and the drafter is loaded on ONE stage - the
    // last one, which is also the only one that holds the head.  On the two identical 8 GB cards that GiB was
    // the entire difference between CUDA0's cache and CUDA1's: 814 slots against 188, 2026-09-30.  Both of
    // those allocations are already made before a stage's cache is sized, so what has to be held back here is
    // the windows and - only on the stage that carries them - the drafter and the head.
    // (the arithmetic is models/qwen38/stage_reserve.hpp's: 96 MiB for the verify windows - 75 MiB measured, rounded
    // up - and 1000 MiB for the MTP drafter and the head on the last stage only; pinned by
    // tests/program/stage_reserve_test.cpp)
    // --pipeline-windows: a second verifier on every stage (its arena alone is 70-75 MiB and its graphs come on top:
    // 64 MiB left two RTX 5070 Ti under Linux short of it) and, with the decode pipelined, two copies of the first
    // stage's GDN state (sized from its session where the first card's cache is sized; this is the split search's
    // estimate).  Allocated after the caches, so kept out of them.
    pipe_window_mib = o.pipeline_windows > 0 ? 160 : 0;
    const int64_t kPipeSnapMib = o.pipeline_windows >= 2 ? 96 : 0;   // ~3 MiB per GDN layer, twice, ~15 layers
    // #340: with the own prompt buffers chosen by the split's rule (not asked for with --no-prefill-borrow) the
    // boundary is searched as the borrowing configuration would (no reserve): the reserve then only makes the caches
    // smaller, which measured cost no decode (K=28 on 9070 XT + R9700: 58.4 tok/s own vs 58.5 borrowing), while a
    // search with the reserve moved the boundary to K=32 and decode to 54.8. STRATA_SPLIT_OWN_PLACE=reserve: the
    // search sees the reserve.
    // ---- `--layer-split auto` with the trim asked: a later stage's weights load after this search, so they are not in
    // `cudaMemGetInfo` yet and are priced from the pack for each candidate range [lb, le): `pool_bytes` of the trimmed
    // arena (reads the index, allocates nothing) and `NativeDense::weight_bytes_for` for the projections, which `load`
    // makes one cudaMalloc each (a 2 MiB granule per matrix of a MiB or more).  Priced as the full copy and without the
    // projections, a small card was over-credited and `auto` gave it layers it could not hold: P40 24 GB + RTX 3070
    // 8 GB picked K=2 where K=16 is right (#1238, #880).  Every other run has its weights loaded already, so its
    // `cap` is net of them and this is 0 (nothing about those placements moves); CUDA0's are loaded too.
    bool carve_ok = true;
    std::map<std::pair<int64_t, int64_t>, uint64_t> carve_memo;
    auto carve_bytes = [&](int64_t lb, int64_t le) -> uint64_t {
        if (!late_weights) return 0;
        const std::pair<int64_t, int64_t> key(lb, le);
        if (const auto it = carve_memo.find(key); it != carve_memo.end()) return it->second;
        std::set<std::string> sk = skip_base;
        add_foreign(lb, le, sk, false);
        uint64_t total = 0;
        if (!strata::core::WeightTable::pool_bytes(o.pack, total, err, sk.empty() ? nullptr : &sk)) {
            carve_ok = false;
            return 0;
        }
        if (!o.native_dense_gguf.empty()) {
            uint64_t dense = 0;
            if (!strata::core::NativeDense::weight_bytes_for(o.native_dense_gguf, wt, o.native_ple_key, lb, le, dense, err)) {
                carve_ok = false;
                return 0;
            }
            total += dense;
        }
        carve_memo.emplace(key, total);
        return total;
    };
    static const bool place_with_reserve = [] {
        const char* v = std::getenv("STRATA_SPLIT_OWN_PLACE");
        return v != nullptr && std::string(v) == "reserve";
    }();
    auto stage_room = [&](int dev, bool later, bool drafter, bool search = false) -> int64_t {
        const strata::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
            std::fprintf(stderr, "strata generate: layer split: CUDA%d free memory: %s\n", dev < 0 ? 0 : dev,
                         cudaGetErrorString(e));
        const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
        // what a later stage allocates after its cache and nothing prices (cuBLAS, the modules and the MMQ pool the prompt path warms up, ~250 MiB measured): the single card prices
        // it (kLateAllocMib, #4474); a stage did not, and two 16 GB cards ended a start with 131 MiB free and ran out of memory in the first long prompt.  Not under STRATA_DEFAULTS=026.
        const int64_t late_alloc = later && !strata::qwen_defaults_026() ? strata::program::split_walk::kStageLateAllocMib : 0;
        const int64_t base_reserve =
            (later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib) + late_alloc;
        strata::program::stage_reserve::Inputs rin;
        rin.base_reserve_mib = base_reserve;
        rin.prompt_buffers_mib = pf;
        rin.later = later;
        rin.drafter = drafter;
        rin.pipe_window_mib = pipe_window_mib;
        rin.pipe_snap_mib = kPipeSnapMib;
        return strata::program::stage_reserve::room_bytes((int64_t) fb, rin);
    };
    if (multi_gpu && split_auto) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int ns = (int) stages.size() + 1;
        std::vector<int64_t> cap((size_t) ns);
        // what stage i has left for experts if it runs [lb, le): its room, minus its session and (a later stage) its weights
        auto stage_left = [&](int i, int64_t lb, int64_t le) -> int64_t {
            return cap[(size_t) i] - (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le) -
                   (i == 0 ? 0 : (int64_t) carve_bytes(lb, le));
        };
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0, (mtp_first || df2_first) ? i == 0 : i + 1 == ns, true);
            int sms = 0, khz = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;
            cudaGetLastError();
            double speed = std::max(1.0, (double) sms * (double) khz / 1e6);   // SMs x GHz
            // STRATA_SPLIT_SPEED_SCALE="1,0.55": each stage's speed times this factor (a test seam: an unequal pair, e.g. the
            // 4060 Ti + 3060 of #2644, on two equal cards); unset = nothing changes.
            if (const char* sc = std::getenv("STRATA_SPLIT_SPEED_SCALE"); sc != nullptr && *sc) {
                const char* p = sc;
                for (int k = 0; k < i && p != nullptr; ++k) { p = std::strchr(p, ','); if (p) ++p; }
                const double f = p != nullptr ? std::atof(p) : 0.0;
                if (f > 0.0) speed = std::max(1.0, speed * f);
            }
            layer_ms[(size_t) i] = strata::program::split_rules::get_layer_ms(speed);
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB free before its session carve\n", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        // The per-stage link, measured here (the stage-setup probes are skipped when --pcie-frac is given, and the
        // search needs the reading either way).
        std::vector<double> link_gbps((size_t) ns, -1.0);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            const strata::core::OnDevice on(dev);
            std::string bursts;
            link_gbps[(size_t) i] = native_pack ? probe_pcie_h2d_gbps(&bursts) : -1.0;
        }
        // #2007: relative to the fastest stage's link and only from a 4x gap on (split_rules::link_miss_scale): a
        // symmetric rig, or a moderate gap, keeps the fitted formula.  STRATA_SPLIT_LINK_ABS=1: 0.1.41's 20 GB/s rule.
        // STRATA_SPLIT_UNIFORM=1: no per-link scaling at all (the pre-0.1.41 uniform miss cost), for A/B on links
        // where any scaling moves auto off the balanced placement.
        static const bool link_abs = [] { const char* v = std::getenv("STRATA_SPLIT_LINK_ABS"); return v && v[0] == '1'; }();
        static const bool split_uniform = [] { const char* v = std::getenv("STRATA_SPLIT_UNIFORM"); return v && v[0] == '1'; }();
        std::vector<double> link_scale = strata::program::split_rules::link_miss_scale(link_gbps);
        for (int i = 0; i < ns; ++i) {
            const double bw = link_gbps[(size_t) i];
            if (split_uniform) link_scale[(size_t) i] = 1.0;
            else if (link_abs) link_scale[(size_t) i] = bw > 0.0 ? std::clamp(20.0 / bw, 1.0, 64.0) : 1.0;
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d link %.1f GB/s -> miss cost x%.2f\n",
                         i == 0 ? 0 : stages[(size_t) i - 1]->dev, bw, link_scale[(size_t) i]);
        }
        const double miss_ms = std::getenv("STRATA_SPLIT_MISS_MS") ? std::atof(std::getenv("STRATA_SPLIT_MISS_MS")) : 190.0;
        // the profile's routed mass by rank (layer_split.cpp): up to 3 GPUs the 0.1.39 curve (r+1)^-1.2, so 2- and 3-GPU
        // placements do not move; the coverage curve was fitted on a 4-way rig.  STRATA_SPLIT_COVER_B asks for the
        // coverage curve at any GPU count, with that exponent (a model whose routing is sharper or flatter).
        strata::core::SplitProblem problem;
        problem.n_layers = g.n_layers;
        for (int i = 0; i < ns; ++i)
            problem.cards.push_back({layer_ms[(size_t) i], link_scale[(size_t) i]});
        for (const auto& pr : profile)
            problem.ranked_layers.push_back(pr.first);
        problem.expert_bytes = [&](int64_t l) -> int64_t {
            return native_pack ? strata::core::ExpertCache::slot_stride((int64_t) nib::slot_bytes(l), slot_guard)
                               : (int64_t) lay.max_blob;
        };
        problem.room = stage_left;
        problem.miss_ms = miss_ms;
        problem.cover_b = std::getenv("STRATA_SPLIT_COVER_B") ? std::atof(std::getenv("STRATA_SPLIT_COVER_B")) : 3.0;
        problem.coverage_curve = ns >= 4 || std::getenv("STRATA_SPLIT_COVER_B") != nullptr;
        problem.near_tie_pct =
            std::getenv("STRATA_SPLIT_NEAR_TIE_PCT") ? std::atof(std::getenv("STRATA_SPLIT_NEAR_TIE_PCT")) : 4.0;
        const int64_t kAutoLendPct = get_auto_lend_pct();
        // ---- LAYER SPLIT STARTABILITY GATE (LAYER 1).  The search must not price placements only by decode time:
        // a split is usable only if the prompt path can start, which requires (serve, `fits_one` 5625-5636 + own_fits
        // 5855-5869): every participant must lend a chunk, or else afford its own prompt buffers with the serve-time
        // 512 MiB headroom.  Here we check it for the last-resort chunk (512 tokens, bottom of kStepChunks 5849),
        // which is necessary and sufficient by monotonicity (both the slot test and `bytes_needed` grow with c).
        //
        // chunk_bytes uses a no-alloc fake SessionState.  qsa_state_init is NOT used: with base=nullptr it would
        // cudaMemcpy a null page table and kv_stream_reset a null map (CUDA "invalid argument") on a streamed
        // state.  bytes_needed only reads max_cells and kv_mode/n_pages/kv_int8/kv_q4/kv_hybrid, so we set those
        // fields directly, replicating kv_plan's arithmetic (layer.cpp:634) with the public getters.
        auto chunk_bytes = [&](int64_t lb, int64_t c) -> uint64_t {
            strata::core::qsa_set_kv_resident(o.kv_resident);
            const strata::kernels::QsaShapes s = strata::kernels::qsa_shapes(g);
            const int64_t pages = (o.max_context + s.page_size - 1) / s.page_size;
            int mode = 0;
            const int64_t kv_res = strata::core::qsa_kv_resident();
            if (kv_res > 0) {
                const int64_t r = (std::max(kv_res, strata::core::qsa_kv_resident_min()) + s.page_size - 1) / s.page_size;
                if (r < pages) mode = 1;
            }
            const bool q4 = strata::core::qsa_kv_q4();
            const bool hyb = strata::core::qsa_kv_hybrid();
            const bool i8 = strata::core::qsa_kv_int8() && !q4;
            std::vector<strata::core::QsaState> qs((size_t) g.n_qsa_layers());
            for (int64_t j = 0; j < g.n_qsa_layers(); ++j) {
                qs[(size_t) j].max_cells = o.max_context;
                qs[(size_t) j].kv_mode = mode;
                qs[(size_t) j].n_pages = pages;
                qs[(size_t) j].kv_int8 = i8;
                qs[(size_t) j].kv_q4 = q4;
                qs[(size_t) j].kv_hybrid = hyb;
            }
            strata::core::SessionState ss{};
            ss.max_cells = o.max_context;
            ss.qsa_states = qs.data();
            const int64_t I = std::max<int64_t>(g.qsa_interval, 1);
            ss.qsa_ord0 = (int) (lb / I);
            // no_ring: serve sizes the streamed ring to what the rig affords (scan's ring_room), so the gate's
            // discriminator is the MOE set against each cache's slots - the ring is a separate budget, not a
            // reason a stage cannot lend.  bytes_needed's default ring (384 slots) would inflate a 2-layer
            // stage's need past its cache and reject a split serve accepts.
            return strata::prefill::Prefill::bytes_needed_no_ring(g, ss, c);
        };
        auto can_start = [&](const std::vector<strata::core::SplitFill>& parts) -> bool {
            const int64_t kMaxBlob = (int64_t) nibble_max_slot_bytes();   // max_blob (or a larger 4-bit slot)
            auto slots_est_of = [&](int i, uint64_t cn) -> int64_t {
                if (parts[i].held <= 0) return -1;
                const uint64_t avg = (parts[i].used > 0)
                    ? (uint64_t) parts[i].used / (uint64_t) parts[i].held : (uint64_t) kMaxBlob;
                return (int64_t) ((cn + avg - 1) / avg);
            };
            auto fit_one = [&](int i, int64_t c, bool cap) -> bool {   // mirrors fits_one (5820-5824)
                const int64_t k = slots_est_of(i, chunk_bytes(parts[i].lb, c));
                if (k <= 0 || k + g_lend_keep > parts[i].held) return false;
                return !(cap && k * 100 > kAutoLendPct * parts[i].held);
            };
            auto all_fit = [&](int64_t c, bool cap) -> bool {
                for (int i = 0; i < ns; ++i) if (!fit_one(i, c, cap)) return false;
                return true;
            };
            // the chunk serve picks: the largest the WHOLE rig can lend (scan(nullptr), 5876-5895).  A stage
            // whose cache is smaller than its layers' chunk (CUDA1's 512 slots for one layer) lends nothing, so
            // the bisection returns 0 and the loan is off - the host then has to afford its own buffers.
            int64_t chunk = 0;
            if (o.prefill_auto) {
                int64_t lo = 1, hi = 8192 / 256, best = 0;
                while (lo <= hi) {
                    const int64_t mid = lo + (hi - lo) / 2;
                    if (all_fit(mid * 256, true)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
                }
                chunk = best;
            } else {
                for (int64_t c = o.prefill_chunk; c >= 256; c /= 2) if (all_fit(c, false)) { chunk = c; break; }
            }
            auto room_mib = [&](int i) -> int64_t { return (int64_t) ((parts[i].room - parts[i].used) >> 20); };
            static const bool gate_dbg = std::getenv("STRATA_LAYER_SPLIT_GATE") != nullptr;
            if (gate_dbg) {
                std::fprintf(stderr, "gate: K=%lld chunk=%lld", (long long) (ns == 2 ? parts[1].lb : 0), (long long) chunk);
                for (int i = 0; i < ns; ++i) {
                    const uint64_t cn = chunk_bytes(parts[i].lb, chunk > 0 ? chunk : 256);
                    std::fprintf(stderr, " | CUDA%d lb=%lld cnt=%lld used=%lldMiB cap=%lldMiB room=%lldMiB need=%lldMiB slots=%lld",
                                 i == 0 ? 0 : (int) stages[(size_t) i - 1]->dev, (long long) parts[i].lb,
                                 (long long) parts[i].held, (long long) (parts[i].used >> 20),
                                 (long long) (parts[i].room >> 20), (long long) room_mib(i),
                                 (long long) (cn >> 20), (long long) slots_est_of(i, cn));
                }
                std::fprintf(stderr, "\n");
            }
            if (chunk == 0) {   // no loan: every participant must afford its own buffers (serve 5960-5975)
                for (int i = 0; i < ns; ++i)
                    if (room_mib(i) < (int64_t) (chunk_bytes(parts[i].lb, 256) >> 20) + 1536) return false;
                return true;
            }
            for (int i = 0; i < ns; ++i) {
                const uint64_t cn = chunk_bytes(parts[i].lb, chunk);
                const int64_t k = slots_est_of(i, cn);
                const bool lends = k > 0 && k + g_lend_keep <= parts[i].held && k * 100 <= kAutoLendPct * parts[i].held;
                if (lends) continue;
                if (room_mib(i) >= (int64_t) (cn >> 20) + 1536) continue;   // affords its own buffers
                return false;
            }
            return true;
        };
        auto split_diagnosis = [&](const std::vector<int64_t>& at) {
            const int64_t c = 512;
            for (int i = 0; i < ns; ++i) {
                const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
                const int64_t le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                const uint64_t cn = chunk_bytes(lb, c);
                const int64_t room_mib = (int64_t) (stage_left(i, lb, le) >> 20);
                std::fprintf(stderr, "strata generate:   CUDA%d: a 512-token prompt needs %lld MiB of its own buffers "
                                     "(+512 MiB headroom); %lld MiB is left for them after its session and cache\n",
                             dev, (long long) (cn >> 20), (long long) room_mib);
            }
            std::fprintf(stderr, "strata generate: to make room: a smaller --max-context, --prefill 512, a smaller "
                                 "--vram-reserve-mib, an explicit --layer-split, or close other programs using the GPU\n");
        };
        problem.can_start = can_start;
        const char* const log_prefix = "strata generate: layer split auto: ";
        // THE LAYER SPLIT'S SPILL WALK, STEP 2: THE KV WINDOW DOWN.  No placement can start the prompt path with the window asked for: the sessions of the cards are smaller with a
        // smaller one (the rest of the K/V streams from pinned RAM), so the search runs again at 20480, 8192, 4096 cells a layer (the engine's smallest) and keeps the first window
        // that has a startable placement.  The context and the number of requests are not touched.  Never reached by a start that has a placement.
        const bool kv_walk = std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu();
        problem.choose_unchecked = !kv_walk;
        strata::core::SplitPlan plan = strata::core::plan_layer_split(problem, log_prefix);
        if (plan.cuts.empty() && kv_walk) {
            const int64_t kv_asked = o.kv_resident;
            for (int64_t w; plan.cuts.empty() &&
                            (w = strata::program::split_walk::next_kv_window(o.kv_resident, o.max_context, strata::core::qsa_kv_resident_min())) > 0;) {
                o.kv_resident = w;
                strata::core::qsa_set_kv_resident(w);
                plan = strata::core::plan_layer_split(problem, log_prefix);
                if (!plan.cuts.empty())
                    std::fprintf(stderr, "strata generate: short of GPU memory: the KV window is %lld cells a layer in VRAM instead of %s, the rest is read from RAM: the context "
                                         "(%lld tokens) is unchanged\n", (long long) w,
                                 kv_asked > 0 ? std::to_string((long long) kv_asked).c_str() : "the whole context", (long long) o.max_context);
            }
            if (plan.cuts.empty()) {
                o.kv_resident = kv_asked;
                strata::core::qsa_set_kv_resident(kv_asked);
                problem.choose_unchecked = true;
                plan = strata::core::plan_layer_split(problem, log_prefix);
            }
        }
        if (plan.start_unchecked && !plan.cuts.empty()) split_diagnosis(plan.cuts);
        if (plan.cuts.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: no valid split for %d GPUs (this model has %lld layers)\n",
                         ns, (long long) g.n_layers);
            return 2;
        }
        if (!carve_ok) {   // `carve_bytes` reports through `err`; it is called from the search, which cannot return
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        split_at = plan.cuts;
        strata::core::print_layer_split(plan, profile.size(), log_prefix);
    }
    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            std::fprintf(stderr, "strata generate: --layer-split: layer %lld is past the last (%lld)\n",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the whole ranking, hottest first, before each stage takes its own layers' share: the resident RAM copy ranks
    // the experts no GPU holds over ALL the layers (with CUDA0's share alone, a copy that did not fit kept only layers
    // 0..K-1's experts and read every later layer's from the files)
    if (multi_gpu) {
        profile_all = profile;
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }

    // ---- `--layer-split auto` with the trim asked: THE STAGES' WEIGHTS, NOW THAT THE SPLIT IS KNOWN.  Each later card
    // loads the dense weights of the layers it runs and nothing else (before this they loaded in the loop above, a
    // full copy per card: on the 4-way IQ3_S rig CUDA1 3650 expert slots, CUDA2 1241, CUDA3 533 and a 768-token prompt
    // chunk; carved 4608 / 2485 / 1772 and 6144).  The search priced these arenas from the pack index (`stage_pool`).
    if (late_weights)
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            const strata::core::OnDevice on(st.dev);
            size_t before = 0, total_b = 0;   // what this stage's load really costs, against what the search priced
            cudaMemGetInfo(&before, &total_b);
            if (!load_stage_weights(st, st.lb, st.le, true)) return 1;
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            {
                // under-pricing is what put layers on a card that could not hold them (#1238); a little over-pricing is
                // by design (a granule per matrix runs ~0.8% high).  Past 5% the pricing has drifted from the loading.
                const uint64_t took = before > fb ? (uint64_t) (before - fb) : 0;
                const uint64_t priced = carve_bytes(st.lb, st.le);
                if (priced < took || priced - took > took / 20)
                    std::fprintf(stderr, "strata generate: layer split, CUDA%d: WARNING the split search priced this "
                                         "stage's weights at %llu MiB, it took %llu MiB\n", st.dev,
                                 (unsigned long long) (priced >> 20), (unsigned long long) (took >> 20));
            }
            std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                                 "follows)\n", st.dev, (double) fb / 1073741824.0);
        }

    // ---- CUDA0's session, and the stages' sessions: sized to each device's own layer range (the carve).  A
    // stage that runs [lb, le) carves only those layers' GDN rows and QSA pools - before the carve every stage
    // held all 48 layers' state whatever layers it ran, which is the same disease the chunked-QSA-prefill PR
    // fixed in llama.cpp: allocation sized by the whole model instead of the device's own work.
    // the layer ranges a session is carved in, one per stage (every session - the engine's own and each --batch slot's -
    // is carved alike): LIST_CONVERSATIONS gives a session's size from them (session_bytes)
    {
        const strata::core::OnDevice on0(primary_device);   // STRATA_PRIMARY_DEVICE (upstream: a literal 0)
        const int64_t hi0 = multi_gpu ? split_at[0] : -1;
        // the elastic K/V (--kv-grow, see kvg_ensure): one GPU, the whole K/V in VRAM (no streaming), a profiled cache
        // that can give slots up, and every expert in RAM for the CPU to compute the ones it gives up
        {
            const char* ev = std::getenv("STRATA_KV_GROW");
            bool remote = false;
            for (const int r : o.expert_cache_remote) remote = remote || r > 0;
            const bool asked = ev != nullptr && ev[0] != '\0' ? ev[0] != '0' : o.kv_grow;
            const bool on = asked && !multi_gpu && o.kv_resident <= 0 &&
                            !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                            strata::core::vmm_available() &&
                            // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
                            o.batch == 0 && !o.vram_elastic && o.peer_device < 0;
            if (asked && !on)
                std::fprintf(stderr, "strata generate: --kv-grow is off (one GPU, a profile, the whole K/V in VRAM, "
                                     "every expert in RAM, no --batch, --vram-elastic or --peer-device)\n");
            const char* iv = std::getenv("STRATA_KV_GROW_INIT");
            strata::core::qsa_set_kv_elastic(on, iv != nullptr && std::atoll(iv) > 0 ? std::atoll(iv) : 16384);
            strata::core::ExpertCache::set_vmm(on);
        }
        session_layers.emplace_back(0, hi0);
        // strata-q38 moves the cut while it runs (session_set_layers), so its sessions are carved for every layer
        const int64_t carved_hi = own_stages ? -1 : hi0;
        if (strata::core::allocate_device(&sbuf, strata::core::session_bytes(g, o.max_context, K, 0, carved_hi),
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session state allocation failed\n");
            return 1;
        }
        if (strata::core::session_init(g, o.max_context, K, sbuf, ss, 0, carved_hi) == 0) {
            std::fprintf(stderr, "strata generate: session_init failed\n");
            return 1;
        }
        strata::core::session_set_layers(ss, g, 0, hi0);
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM\n",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // (#167) generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt
        // path or the verifier reads it (--serve zeroes it per request when nothing is reused)
        strata::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: zeroing the session state failed\n");
            return 1;
        }
        if (!o.ple_gguf.empty()) {
            if (!ss.ple.ready()) {
                std::fprintf(stderr, "strata generate: the PLE run is not ready after construction\n");
                return 1;
            }
            std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s\n",
                         (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
        }
    }
    mem_mark("the session (KV, GDN state, PLE history)");
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        const strata::core::OnDevice on(st.dev);
        void* sbuf_s = nullptr;
        session_layers.emplace_back(st.lb, st.le);
        const int64_t carved_lb = own_stages ? 0 : st.lb, carved_le = own_stages ? -1 : st.le;   // (as CUDA0's)
        if (strata::core::allocate_device(&sbuf_s, strata::core::session_bytes(g, o.max_context, K, carved_lb,
            carved_le),
            {"other", "Device allocations", "Model working buffers"}) !=
                cudaSuccess ||
            strata::core::session_init(g, o.max_context, K, sbuf_s, st.ss, carved_lb, carved_le) == 0) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: the session state failed\n", st.dev);
            return 1;
        }
        strata::core::session_set_layers(st.ss, g, st.lb, st.le);
        const bool last = i + 1 == stages.size();
        const strata::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !o.native_head_gguf.empty() && !st.head.load(o.native_head_shards, g.n_embd, wo_s->ne1, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d head: %s%s\n", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str(),
                         wo_s == nullptr ? "" : vram_free_note().c_str());
            return 1;
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights, session [%lld, %lld)%s; "
                             "%.2f GiB free\n", st.dev, (long long) st.lb, (long long) st.le,
                     last ? " and the head" : "", (double) fb / 1073741824.0);
    }

    // --batch: every stage carves o.batch more sessions like its own (same layer range and context), one per
    // slot of the batch windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.
    // Recommend, never force: a count the engine cannot run is said and adjusted (or batching left off) - the
    // engine still starts and serves one request at a time.
    // --batch-mtp / STRATA_BATCH_MTP=1 (opt-in): a slot also verifies one MTP proposal per window.  Recommend, never
    // force: when it cannot run it is said and left off (plain batching still starts); default off = exactly 0.1.39.
    // With a layer split the slots' drafters live on the LAST stage's GPU (where the solo drafter, the head and the final
    // residual rows are); the windows are the ones --batch runs, two rows per slot, through every stage.
    // --batch-groups: given as a number or auto, or (0.1.41) auto by default for a layer split with --batch >= 2.  The
    // pipelined slot groups run neither --batch-mtp's drafts nor the batch adaptive tier (--adapt-async 1), so when one of
    // them is asked for and the groups were not given, the default is one group (batch_groups.hpp).  Evaluated again
    // once the slots that fit are known, and BEFORE the --batch-mtp gate below, which must see the resolved number.
    auto batch_groups_for = [&](bool mtp_on) {
        strata::program::batch_groups::Input bin;
        bin.set = o.batch_groups_set;
        bin.auto_given = o.batch_groups_auto;
        bin.given = o.batch_groups;
        bin.batch = o.batch;
        bin.later_stages = (int) stages.size();
        bin.batch_mtp = mtp_on;
        bin.adapt_async = o.adapt_async != 0;
        { const char* ms = std::getenv("STRATA_BATCH_MTP_SERIAL"); bin.mtp_serial = ms != nullptr && ms[0] != '\0' && ms[0] != '0'; }
#if !defined(STRATA_USE_HIP)
        // NVIDIA, 2 cards: the pipelined groups by default (0.2.7); one serial group, the 0.2.1-0.2.6 default, with
        // STRATA_BATCH_GROUPS_NV_SERIAL=1 or STRATA_DEFAULTS=026 (batch_groups.hpp nv_serial_wanted)
        bin.nv_serial_faster = strata::program::batch_groups::nv_serial_wanted(std::getenv("STRATA_BATCH_GROUPS_NV_SERIAL"),
                                                                              strata::qwen_defaults_026());
#endif
        { const char* mp = std::getenv("STRATA_PIPED_MTP"); bin.mtp_piped = mp != nullptr && mp[0] != '\0' && mp[0] != '0'; }
        return strata::program::batch_groups::resolve(bin);
    };
    // DFlash2 slot drafts (the default for --draft dflash2 with --batch; STRATA_BATCH_DF2=0 turns them off): every batch slot gets its own drafter (the pack,
    // embedding and draft head shared, its own K/V ring and buffers).  A window carries each slot's token plus drafts only in the rows the active slots leave
    // spare, and a round with no spare row does no drafter work (the committed rows wait on the host and are replayed when spare rows return).  Measured on 2x
    // A6000 (RESULTS-DF2MGPU.md section 3): +25% / +9% over no drafts at 2 / 3 active slots, level at 4, within 1.5% at 8 and with users joining and leaving.
    // They run through the serial windows (one group); a layer split whose slots would be pipelined in groups by default keeps its groups and its undrafted
    // slots (the pipelined groups carry no drafts), and so does an explicit --batch-groups above 1.  One card: on with the explicit pair (STRATA_BATCH_DF2=1 forces them on).
    const bool batch_df2_wanted = [&] {
        const char* e = std::getenv("STRATA_BATCH_DF2");
        const bool off = e != nullptr && e[0] == '0';
        const bool forced = e != nullptr && e[0] != '0' && e[0] != '\0';
#if defined(STRATA_GPU_METAL)
        return false;   // (no per-slot drafters on Metal: one card, and the Metal drafter has no clones)
#endif
        if (off || !(o.draft == "dflash2" && !o.draft_auto && o.batch >= 2 && o.serve && !split_same)) return false;
        if (forced) return true;
        // an explicit --draft dflash2 with --batch N turns the slots' drafts on (the launcher sends that pair only when the user picks DFlash2; --draft auto and the MTP
        // draft layer are unchanged).  On one card that is all; on a layer split only where the slots would run as one group (the pipelined groups carry no drafts)
        return stages.empty() || batch_groups_for(false).groups <= 1;
    }();
    const char* batch_mtp_env = std::getenv("STRATA_BATCH_MTP");
    batch_mtp = o.batch_mtp || (batch_mtp_env != nullptr && batch_mtp_env[0] != '\0' && batch_mtp_env[0] != '0') || batch_df2_wanted;
    if (batch_mtp) {
        // (--pipeline-windows is already off with --batch slots)
        const char* why = o.batch < 2 ? "it needs --batch 2 or more" : o.mtp.empty() && !batch_df2_wanted ? "it needs --mtp"
                        : o.spec < 2 ? "it needs --spec T (T >= 2)" : split_same
                        ? "it needs each stage of a layer split on its own GPU"
                        : !o.serve ? "it needs --serve" : nullptr;
        // the groups it would run with (the default gives way to it: one group; a number or auto given pipelines;
        // STRATA_PIPED_MTP with one slot per group: the pipelined groups draft)
        if (why == nullptr && !stages.empty() && batch_groups_for(true).groups > 1 && !batch_groups_for(true).mtp_piped)
            why = batch_groups_for(true).mtp_dropped
                ? "a layer split with --batch pipelines the slots in groups by default and the pipelined groups run no slot "
                  "drafts (4 streams on 2 GPUs: 25-30% faster than the serial MTP windows); --batch-groups 1 or "
                  "STRATA_BATCH_MTP_SERIAL=1 keeps --batch-mtp with serial windows (fewer streams, or when prompts arrive "
                  "rarely, it can be the faster choice)"
                : "it does not combine with --batch-groups (pipelined slot groups) yet";
        // (a layer split with helper-GPU expert caches has not been run with slot drafters.  --remote-expert-opt alone does
        // not count: its object is only made with a helper cache on CUDA0 (`remote_opt` below), so without one it does nothing)
        if (why == nullptr && !stages.empty() && remote_caches)
            why = "it is not run with helper expert caches on a layer split";
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: WARNING: %s is off: %s\n", batch_df2_wanted ? "STRATA_BATCH_DF2 (the slots' DFlash2 drafts)" : "--batch-mtp", why);
            batch_mtp = false;
        }
    }
    // STRATA_PIPED_MTP (opt-in): --batch-mtp's proposals in the pipelined groups of a layer split, one slot per group;
    // each stage commits a group's window once the last stage has judged its proposals
    // (Verifier::batch_launch_rows / batch_commit_rows)
    piped_mtp = batch_mtp && !stages.empty() && batch_groups_for(true).mtp_piped;   // (final once the slots that fit are known, below)
    // STRATA_PIPED_MTP=N: up to N proposals per slot and window (at most --spec - 1); the first always, each further
    // one while the draft layer is at least --spec-min-p sure of it, as the solo path extends its windows
    if (piped_mtp) {
        pm_drafts = std::clamp(std::atoi(std::getenv("STRATA_PIPED_MTP")), 1,
                               std::max(1, std::min(o.spec - 1, (int) strata::kernels::kVerifyNarrowT - 1)));
    }
    // The most slots, up to n, whose windows hold at most kVerifyMaxT rows each: n itself up to that, more only as
    // pipelined groups on a layer split - the groups as resolved below (a number given, auto, or the default's one
    // group for --adapt-async 1), dividing the slots, each at most kVerifyMaxT, all at most one group's worth per
    // stage.  (--batch-mtp: n, as before.)
    auto slots_in_windows = [&](int n) {
        const int cap = strata::kernels::kVerifyMaxT;
        if (batch_mtp || n <= cap) return n;
        const int asked = o.batch;
        for (; n > cap; --n) {
            o.batch = n;
            const int gr = batch_groups_for(batch_mtp).groups;
            if (!stages.empty() && gr > 1 && n % gr == 0 && n / gr <= cap && n <= cap * ((int) stages.size() + 1))
                break;
        }
        o.batch = asked;
        return n;
    };
    if (o.batch != 0) {
        const int cap = strata::kernels::kVerifyMaxT;
        const char* off = !o.serve ? "it needs --serve" : o.batch < 2 ? "it needs 2 or more slots"
                        : split_same ? "it needs each stage of a layer split on its own GPU" : nullptr;
        if (off != nullptr) {
            std::fprintf(stderr, "strata generate: WARNING: --batch %d is off: %s\n", o.batch, off);
            o.batch = 0;
        } else if (o.batch > cap && !batch_mtp) {
            const int n = slots_in_windows(o.batch);
            if (n < o.batch) {
                std::fprintf(stderr, "strata generate: WARNING: --batch %d: a batch window holds at most %d rows, so %d "
                                     "slots\n", o.batch, cap, n);
                o.batch = n;
            }
        }
    }
    // --batch-groups auto: the pipeline needs one group per GPU stage to keep every card busy (4 x R9700, 8 clients:
    // 90 tok/s in one group, 136 in 2, 166 in 4).  The most groups, at most one per stage, that divide the slots.
    // The default (0.1.41: a layer split with --batch pipelines one group per stage; --batch-groups 1 opts out) gives
    // way to --batch-mtp and --adapt-async 1, which need the serial windows: one group, said once.
    int groups_said = 0;
    auto resolve_groups_auto = [&] {
        const auto r = batch_groups_for(batch_mtp);
        if (!r.from_auto) return;
        o.batch_groups = r.groups;
        if (groups_said == r.groups) return;   // (the second call, once the slots that fit are known, says nothing new)
        groups_said = r.groups;
        if (r.serial == strata::program::batch_groups::Serial::TwoCardsSerialFaster)
            std::fprintf(stderr, "strata generate: --batch-groups auto: 1 group of %d slot%s - %s; --batch-groups %d asks for the "
                                 "pipelined groups\n", o.batch, o.batch == 1 ? "" : "s",
                         strata::program::batch_groups::serial_what(r.serial), r.would_be);
        else if (r.serial != strata::program::batch_groups::Serial::None)
            std::fprintf(stderr, "strata generate: --batch-groups auto: 1 group of %d slot%s - %s does not run in pipelined "
                                 "groups; --batch-groups %d would pipeline them and turn it off\n", o.batch,
                         o.batch == 1 ? "" : "s", strata::program::batch_groups::serial_what(r.serial), r.would_be);
        else
            std::fprintf(stderr, "strata generate: --batch-groups auto: %d group%s of %d slot%s\n", r.groups,
                         r.groups == 1 ? "" : "s", o.batch / r.groups, o.batch / r.groups == 1 ? "" : "s");
    };
    if (o.batch > 0) resolve_groups_auto();
    if (o.batch > 0 && o.batch_groups > 1 && (stages.empty() || o.batch % o.batch_groups != 0)) {
        std::fprintf(stderr, "strata generate: WARNING: --batch-groups %d needs a layer split and to divide --batch %d; "
                             "one group\n", o.batch_groups, o.batch);
        o.batch_groups = 1;
    }
    if (o.batch > 0) {
        bslot_ss.resize(1 + stages.size());
        int fit = o.batch;   // the slots every stage could carve
        uint64_t bytes0 = 0;
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            const int dev = k == 0 ? 0 : stages[k - 1]->dev;
            const int64_t lo = k == 0 ? 0 : stages[k - 1]->lb;
            const int64_t hi = k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le;
            const strata::core::OnDevice on_k(dev);
            const uint64_t bytes = strata::core::session_bytes(g, o.max_context, K, lo, hi);
            if (k == 0) bytes0 = bytes;
            for (int b = 0; b < fit; ++b) {
                auto u = std::make_unique<strata::core::SessionState>();
                void* buf = nullptr;
                if (strata::core::allocate_device(&buf, bytes,
                    {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
                        strata::core::session_init(g, o.max_context, K, buf, *u, lo, hi) == 0) {
                    cudaGetLastError();
                    if (buf != nullptr)
                        strata::core::release_device(buf);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    std::fprintf(stderr, "strata generate: WARNING: --batch %d: slot %d's session does not fit on CUDA%d "
                                         "(%.2f GiB each, %.2f GiB free); %d slot%s\n", o.batch, b, dev,
                                 (double) bytes / 1073741824.0, (double) fb / 1073741824.0, b, b == 1 ? "" : "s");
                    fit = b;
                    break;
                }
                strata::core::session_zero(*u, g, nullptr, nullptr);
                bslot_ss[k].push_back(std::move(u));
            }
            cudaDeviceSynchronize();
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            std::fprintf(stderr, "strata generate: --batch: %d slot sessions on CUDA%d (%.2f GiB each); %.2f GiB free\n",
                         (int) bslot_ss[k].size(), dev, (double) bytes / 1073741824.0, (double) fb / 1073741824.0);
        }
        for (auto& v : bslot_ss) if ((int) v.size() > fit) v.resize((size_t) fit);   // the same count on every stage
        if (fit < 2) {
            std::fprintf(stderr, "strata generate: WARNING: --batch is off: not two slot sessions fit (a smaller "
                                 "--max-context or KV streaming, --kv-resident, makes them smaller)\n");
            bslot_ss.clear();
            o.batch = 0;
        } else {
            const int n = slots_in_windows(fit);   // (the slots that fit are held to kVerifyMaxT rows per group too)
            if (n < fit) {
                std::fprintf(stderr, "strata generate: WARNING: --batch: %d slot sessions fit, and a batch window holds "
                                     "at most %d rows, so %d slots\n", fit, (int) strata::kernels::kVerifyMaxT, n);
                for (auto& v : bslot_ss) v.resize((size_t) n);
                fit = n;
            }
            o.batch = fit;
            resolve_groups_auto();   // (the slots that fit may not be the ones asked for)
            if (o.batch_groups > 1 && o.batch % o.batch_groups != 0) o.batch_groups = 1;
            // the sessions' VRAM is the expert cache's: say what it costs (docs/BATCHING.md has the measured trade)
            std::fprintf(stderr, "strata generate: --batch %d: the slot sessions take %.2f GiB of VRAM on CUDA0 that the "
                                 "expert cache would otherwise hold\n", o.batch,
                         (double) bytes0 * o.batch / 1073741824.0);
        }
    }

    // STRATA_PIPED_MTP is decided again now that the slots that fit and the groups are final.  Recommend, never force: when
    // it cannot run it is said and left off, and --batch-mtp goes on as it would without it.
    if (piped_mtp) {
        const char* why = nullptr;
        char buf[160];
        const char* sc = std::getenv("STRATA_BATCH_SELF_COMMIT");
        if (o.batch < 2 || o.batch_groups <= 1) {
            std::snprintf(buf, sizeof buf, "it needs 2 or more pipelined groups and %d slot%s in %d group%s are left", o.batch,
                          o.batch == 1 ? "" : "s", o.batch_groups, o.batch_groups == 1 ? "" : "s");
            why = buf;
        } else if (sc != nullptr && sc[0] != '\0' && sc[0] != '0') {
            why = "it does not combine with STRATA_BATCH_SELF_COMMIT (the window would keep its rows itself)";
        } else if ((o.batch / o.batch_groups) * (1 + pm_drafts) > (int) strata::kernels::kVerifyNarrowT) {
            std::snprintf(buf, sizeof buf, "a group's window would hold %d rows (%d slots x %d), more than %d",
                          (o.batch / o.batch_groups) * (1 + pm_drafts), o.batch / o.batch_groups, 1 + pm_drafts,
                          (int) strata::kernels::kVerifyNarrowT);
            why = buf;
        }
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: WARNING: STRATA_PIPED_MTP is off: %s\n", why);
            piped_mtp = false;
        } else {
            std::fprintf(stderr, "strata generate: STRATA_PIPED_MTP: %d pipelined group%s of %d slot%s draft up to %d MTP "
                                 "proposal%s per slot each window\n", o.batch_groups, o.batch_groups == 1 ? "" : "s",
                         o.batch / o.batch_groups, o.batch / o.batch_groups == 1 ? "" : "s", pm_drafts,
                         pm_drafts > 1 ? "s" : "");
        }
    }
    // KVA (0.2.7: opt-in with STRATA_KVA=1; unset is the exact 0.2.6 path): what only the engine knows.  A layer split and
    // the batch slots read their prompts in pieces and in stages, which the one-device, one-cold-read rule of the
    // approximate prompt read cannot follow, so they stay on the exact path (and say so here, once).  The draft layer
    // (MTP) keeps its window, whatever it is: a session KVA approximated tells it the first cell built from a real row
    // (the cell floor, set per request below) and every session KVA did not approximate attends exactly as with
    // STRATA_KVA=0.
    {
        const strata::prefill::KvaConfig& kc = strata::prefill::kva_config();
        strata::prefill::KvaEngine& ke = strata::prefill::kva_engine();
        if (kc.explicit_on && !strata::prefill::kKvaBackend)
            std::fprintf(stderr, "strata generate: STRATA_KVA is set, but this build has no state-only late-layer path; "
                                 "every prompt is read exactly\n");
        if (kc.on) {
            // (strata-q38 has not been run with KVA: its conversations' cells and the draft layer's are pool pages)
            if (is_paged()) ke.blocked = "strata-q38";
            else if (multi_gpu) ke.blocked = "a layer split across GPUs";
            else if (o.batch > 0) ke.blocked = "batch slots (--batch)";
            else if (o.peer_device >= 0) ke.blocked = "a peer device (--peer-device)";
            if (!ke.blocked.empty()) {
                if (kc.explicit_on)
                    std::fprintf(stderr, "strata generate: KVA (the approximate late-layer prompt read) is off here: %s\n",
                                 ke.blocked.c_str());
            }
            mtp.enable_cell_floor(kc.wants_cell_floor(ke, !o.mtp.empty() && o.spec >= 2));
        }
    }
    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts.
    // In particular WDDM can refuse the draft weights after mapping tens of GiB of host pages.
    // SHORT OF VRAM: THE DRAFT LAYER IS THE LAST THING TO GO.  The layer (about 0.9 GiB), the native head, the draft head and a working cache beside the
    // prompt path's buffers must all fit after the slot sessions; on a card that cannot hold them (8 GB NVIDIA, 128K, 2 requests: 1.4 GiB free after
    // the sessions) the start used to stop at "no VRAM is left for the expert cache" or run out of memory in the head upload.  Without the layer
    // the model still answers (the verify window drafts from the prompt alone): slower decode, a start that works.  Not with DFlash2 (its own
    // block decides), a split, or STRATA_MTP_KEEP=1.
    // A LAYER SPLIT WALKS THE SAME STEP ON THE CARD THAT HOLDS THE DRAFTER (the last stage's, or CUDA0 with the drafter first): its free memory is read there, after its weights, session
    // and head are in.  DFlash2 (its pack, its ring and buffers, its draft head) is dropped by the same rule on that card; the single card keeps it as before (its own block decides).
    // Not in strata-q38: its page pool, sized after the load, decides what the drafter leaves.
    if (!is_paged()) {
        const bool split_walk_drafter = multi_gpu && !stages.empty() && std::getenv("STRATA_SPILL_OFF") == nullptr;
        if (!o.mtp.empty() && !use_df2 && (split_walk_drafter || (!multi_gpu && stages.empty())) && std::getenv("STRATA_MTP_KEEP") == nullptr && spill_discrete_gpu()) {
            std::error_code fec1, fec2;
            const uint64_t layer_b = (uint64_t) std::filesystem::file_size(std::filesystem::path(o.mtp) / "dense.bin", fec1) +
                                     (uint64_t) std::filesystem::file_size(std::filesystem::path(o.mtp) / "experts.bin", fec2);
            const strata::core::WeightRef* wo_mtp = wt.find("output.weight");
            const uint64_t head_b = (!o.native_head_gguf.empty() && wo_mtp != nullptr)
                                        ? (uint64_t) wo_mtp->ne1 * (uint64_t) (g.n_embd / 256) * 176ull : 0;   // Q5_K: 176 bytes per 256 values
            const int draft_dev = split_walk_drafter ? (last_st && !mtp_first ? last_st->dev : -1) : -1;
            uint64_t free_now = 0;
            {
                const strata::core::OnDevice on_draft(draft_dev);
                free_now = (uint64_t) strata::core::device_free_bytes();
            }
            const uint64_t need = strata::program::split_walk::draft_layer_need_bytes(layer_b, head_b);   // + the cache the loan needs (214 slots, 543 MiB) + what is allocated after it and the drafter beyond DFlash2 (measured: DFlash2 alone ends 391 MiB free, MTP 41 at the same cache)
            if (!fec1 && !fec2 && strata::program::split_walk::draft_layer_short(free_now, layer_b, head_b)) {
                const std::string card = split_walk_drafter ? "CUDA" + std::to_string(draft_dev < 0 ? 0 : draft_dev) + ": " : std::string();
                std::fprintf(stderr, "strata generate: short of GPU memory: %s%.2f GiB is free, and the draft layer (%.2f GiB), the head (%.2f GiB) and a working cache "
                                     "need about %.2f GiB: this start runs without the draft layer (decode is slower; STRATA_MTP_KEEP=1 keeps it)\n",
                             card.c_str(), (double) free_now / 1073741824.0, (double) layer_b / 1073741824.0, (double) head_b / 1073741824.0, (double) need / 1073741824.0);
                o.mtp.clear();
            }
        }
        // DFlash2 alone (--draft-fallback none, what Auto passes) on a stage that cannot hold it: no drafter, said in one line, instead of a start that ends
        if (split_walk_drafter && df2_solo && !use_df2 && std::getenv("STRATA_MTP_KEEP") == nullptr && spill_discrete_gpu()) {
            std::error_code fec3;
            const uint64_t pack_b = (uint64_t) std::filesystem::file_size(o.draft_model, fec3);
            const strata::core::WeightRef* wo_df = wt.find("output.weight");
            const uint64_t head_b = (!o.native_head_gguf.empty() && wo_df != nullptr)
                                        ? (uint64_t) wo_df->ne1 * (uint64_t) (g.n_embd / 256) * 176ull : 0;
            const int dev_df = last_st && !df2_first ? last_st->dev : -1;
            uint64_t free_df = 0;
            {
                const strata::core::OnDevice on_df(dev_df);
                free_df = (uint64_t) strata::core::device_free_bytes();
            }
            const uint64_t need_df = strata::program::split_walk::dflash2_need_bytes(pack_b, head_b / 2);
            if (!fec3 && free_df > 0 && free_df < need_df) {
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d: %.2f GiB is free, and DFlash2 (%.2f GiB with its buffers and draft head) and a working cache need about "
                                     "%.2f GiB: this start runs without a drafter (decode is slower; STRATA_MTP_KEEP=1 keeps it)\n",
                             dev_df < 0 ? 0 : dev_df, (double) free_df / 1073741824.0, (double) pack_b / 1073741824.0 + 0.45, (double) need_df / 1073741824.0);
                df2_solo = false;
                o.draft.clear();
                o.draft_model.clear();
            }
        }
    }
    if (!o.mtp.empty() || df2_solo) {   // (df2_solo: o.mtp is empty and no MTP layer is loaded; only the DFlash2 block below runs)
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        if (!o.mtp.empty() && !load_drafter(err)) {
            std::fprintf(stderr, "strata generate: %s%s%s\n", err.c_str(), vram_free_note().c_str(), vram_batch_hint(err, o.batch).c_str());
            return 1;
        }
        // the batch slots' drafters below go on the drafter's GPU too
        const strata::core::OnDevice on_mtp(get_drafter_stage() != nullptr ? get_drafter_stage()->dev : -1);
#if defined(STRATA_HAVE_DFLASH2)
        if (o.draft == "dflash2") {
            const char* why = o.mtp.empty() && !df2_solo ? "it needs --mtp (the fallback) and --spec 2 or more"
                              : !native_pack ? "it needs a native pack"
                              : o.draft_auto && (!split_devs.empty() || o.batch > 0) ? "--draft auto (both drafters) is for one card and one request at a time; --draft dflash2 runs on a layer split"
                              : o.pipeline_windows > 0 ? "it does not run beside --pipeline-windows (the MTP's speculative next window has no DFlash2 counterpart); --pipeline-windows 0"
                              : o.draft_model.empty() ? "it needs --draft-model PACK (tools/dflash2_pack.py)" : nullptr;
            if (why == nullptr) {
                if (df2.load(o.draft_model, multi_gpu && last_st != nullptr && !df2_first ? last_st->dev : 0, err)) {
                    use_df2 = true;
                    // Per card class, from the measured A/Bs (notes-dflash2-p1.md, RESULTS-CONF.md 11-08; the numbers are in core/draft_cap.hpp): Strix Halo 8, Blackwell cc 12.0 with
                    // at least 48 GB 8, other Blackwell 6; a 12 GB NVIDIA card is slower with DFlash2 than with the MTP at every cap but code, which only the user can know.
                    // Recommend, never force: the window is capped for the card unless --draft-cap says otherwise, and a small NVIDIA card gets one plain line.
                    cudaDeviceProp dp{};
                    if (cudaGetDeviceProperties(&dp, df2.device()) == cudaSuccess) {
#if defined(STRATA_USE_HIP)
                        const bool uma = strata::kernels::gfx_arch_is_gfx1151(dp.gcnArchName);
#else
                        const bool uma = false;
#endif
#if defined(STRATA_USE_HIP)
                        const int cc_major = 0;
#else
                        const int cc_major = dp.major;
#endif
#if defined(STRATA_USE_HIP)
                        const int cc_minor = 0;
                        const size_t card_bytes = 0;
#else
                        const int cc_minor = dp.minor;
                        const size_t card_bytes = dp.totalGlobalMem;
#endif
                        const int cap = o.draft_cap > 0 ? o.draft_cap : strata::core::dflash2_auto_cap(uma, cc_major, cc_minor, card_bytes);
                        if (cap > 0 && o.spec > cap) {
                            std::fprintf(stderr, "strata generate: --draft dflash2: windows capped at %d tokens (%s; --draft-cap N changes it)\n", cap,
                                         o.draft_cap > 0 ? "as asked" : "the width measured best on this card");
                            o.spec = cap;
                        }
#if !defined(STRATA_USE_HIP)
                        if (o.draft_cap == 0 && strata::core::dflash2_small_nvidia(true, dp.totalGlobalMem))
                            std::fprintf(stderr, "strata generate: --draft dflash2: on a 12 GB NVIDIA card the MTP drafter was faster in most measurements "
                                                 "(code prompts at a window of 6 gained 3%%); continuing as asked\n");
#endif
                    }
                } else if (df2_solo) {
                    std::fprintf(stderr, "strata generate: --draft dflash2 --draft-fallback none: %s. DFlash2 needs about 1.2 GiB of VRAM; there is no MTP to draft instead, so the start ends here%s\n",
                                 err.c_str(), vram_free_note().c_str());
                    return 1;
                } else {
                    std::fprintf(stderr, "strata generate: --draft dflash2: %s. DFlash2 needs about 1.2 GiB of VRAM (the pack 0.76 GiB, its buffers and K/V ring "
                                         "0.25 GiB, a draft head 0.2 GiB); start with --vram-reserve-mib 1900 to keep it (the expert cache then gives up about 2.5%% "
                                         "of its experts to the CPU), or a smaller --max-context. The MTP drafts instead.\n", err.c_str());
                }
            } else if (df2_solo) {
                std::fprintf(stderr, "strata generate: --draft dflash2 --draft-fallback none: DFlash2 is off here: %s; there is no MTP to draft instead, so the start ends here\n", why);
                return 1;
            } else {
                std::fprintf(stderr, "strata generate: --draft dflash2 is off: %s; the MTP drafts instead\n", why);
            }
            // DFlash2 did not start: the MTP drafts, with its own window (the wider --spec was DFlash2's), when --draft-mtp-spec names it
            if (!use_df2 && o.draft_mtp_spec_set && o.spec > o.draft_mtp_spec) o.spec = o.draft_mtp_spec;
        }
        if (batch_df2_wanted && batch_mtp) {   // the slots' DFlash2 drafters: streams and buffers now (the cache is sized around them); the head binds later
            if (use_df2) {
                std::string e2;
                for (int b = 0; b < o.batch; ++b) {
                    auto d = std::make_unique<strata::core::Dflash2Drafter>();
                    if (!d->clone_alloc(df2, e2)) {
                        std::fprintf(stderr, "strata generate: STRATA_BATCH_DF2: slot %d drafter: %s%s\n", b, e2.c_str(), vram_free_note().c_str());
                        return 1;
                    }
                    slot_df2.push_back(std::move(d));
                }
                batch_df2 = true;
                size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                std::fprintf(stderr, "strata generate: STRATA_BATCH_DF2: %d slot drafters on CUDA%d (%.0f MiB of private K/V ring and buffers each, sharing the "
                                     "solo drafter's pack); %.2f GiB free there\n", (int) slot_df2.size(), df2.device(),
                             (double) slot_df2.front()->vram_bytes() / 1048576.0, (double) fb / 1073741824.0);
            } else {
                std::fprintf(stderr, "strata generate: WARNING: STRATA_BATCH_DF2 is off: DFlash2 is not running\n");
                batch_mtp = false;
            }
        }
#endif
        if (batch_mtp && !batch_df2_wanted) {
            // Slot b's drafter lives where the solo drafter does (the guard above: the last stage's GPU), carved from the
            // LAST stage's slot-b session (bslot_ss.back(): stage 0's on one GPU): it borrows that session's context
            // length and RoPE table, which are on that device.  Its PLE history is the first stage's slot session: PLE
            // runs on the first stage, as for the solo drafter (load_drafter's `mtp.set_ple_session(&ss)`).
            for (int b = 0; b < o.batch; ++b) {
                auto d = std::make_unique<strata::core::MtpDrafter>();
                // STRATA_PIPED_MTP: the slot's proposal is drafted as an asynchronous chain (chain_launch), so the
                // pipeline goes on while it runs: the chain graphs carry the forcing kernel (a no-op here)
                if (piped_mtp) d->set_force_capture(true);
                // (mtp_first: the solo drafter is on the first stage's card, so slot b's is carved from its first-stage session)
                strata::core::SessionState& slot_s = mtp_first ? *bslot_ss[0][(size_t) b] : *bslot_ss.back()[(size_t) b];
                if (!d->load(o.mtp, draft_geometry, slot_s, o.spec, err, o.mtp_window, &mtp)) {
                    std::fprintf(stderr, "strata generate: batch MTP slot %d: %s%s\n", b, err.c_str(),
                                 vram_free_note().c_str());
                    return 1;
                }
                d->set_max_drafts(piped_mtp ? pm_drafts : 1);   // the serial windows verify one proposal per slot
                d->set_ple_session(bslot_ss[0][(size_t) b].get());
                slot_mtp.push_back(std::move(d));
            }
            if (!slot_mtp.empty()) {
                size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                std::fprintf(stderr, "strata generate: --batch-mtp: %d slot drafters on CUDA%d (%.0f MiB of private K/V "
                                     "state and buffers each, sharing the solo drafter's weights); %.2f GiB free there\n",
                             (int) slot_mtp.size(), mtp.device(), (double) slot_mtp.front()->vram_bytes() / 1048576.0,
                             (double) fb / 1073741824.0);
            }
        }
    }
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after
    // it and ate most of the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for
    // good at its first verify window.  Loaded first, the cache is sized around it.  #620: and before the expert
    // arena registers tens of GiB of host pages, like the drafter above - WDDM then refused the head's cudaMalloc on
    // a 16 GB card with the desktop on the iGPU ("native head upload: out of memory" with GiBs free).
    wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "strata generate: output.weight is missing\n"); return 1; }
    n_vocab = wo->ne1;
    if (!o.native_head_gguf.empty() && !multi_gpu) {   // a layer split's head is on its last stage
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_head_shards, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s%s%s\n", err.c_str(), vram_free_note().c_str(), vram_batch_hint(err, o.batch).c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes, in %.1f s\n",
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
#if defined(STRATA_HAVE_DFLASH2)
    // DF-05: with the MTP loaded and the same draft-head subset (the plain gather: no --mtp-q4 head, no repack) the DFlash2 shares the MTP's copy at bind
    // (STRATA_DF2_SHARE_HEAD_OFF=1: its own copy, the A/B), and the cache is sized without that second copy
    df2_expect_share = !df2_solo && !strata::core::Dflash2Drafter::head_native() && !o.mtp.empty() && std::getenv("STRATA_DF2_SHARE_HEAD_OFF") == nullptr && o.mtp_q4 != "all" && o.mtp_q4 != "head" &&
                       std::getenv("STRATA_MTP_FULL_HEAD") == nullptr && std::getenv("STRATA_Q6_PACKED") == nullptr && std::getenv("STRATA_MTP_Q8_SM60") == nullptr &&
                       std::filesystem::exists(get_df2_vocab_path());
#endif
    logits.resize((size_t) n_vocab);
    mem_mark("the native head");
    if (strata::core::allocate_device(&d_logits, (size_t) n_vocab * 4,
        {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata generate: the logits buffer failed%s\n", vram_free_note().c_str());
        return 1;
    }
    if (o.prefill_head_alias) {
        const char* why = !o.serve ? "it needs --serve"
                        : multi_gpu || !stages.empty() ? "it supports one GPU only"
                        : o.batch > 0 ? "it does not support --batch"
                        : o.pipeline_windows > 0 ? "it does not support --pipeline-windows"
                        : o.short_read > 0 ? "it needs --short-read 0"
                        : o.expert_cache != 1 && o.expert_cache >= 0 ? "it needs --expert-cache 1 or auto"
                        : o.mtp.empty() ? "it needs --mtp"
                        : !native_head.loaded() ? "it needs the native output head"
                        : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: --prefill-head-alias is unavailable: %s\n", why);
            return 2;
        }
        if (!mtp.bind(wt, &native_head, nullptr, err)) {
            std::fprintf(stderr, "strata generate: --prefill-head-alias: %s\n", err.c_str());
            return 1;
        }
        native_head.unload();
        prefill_head_alias_bound = true;
        std::fprintf(stderr, "strata generate: --prefill-head-alias: the %llu-byte output head is prompt workspace\n",
                     (unsigned long long) native_head.weight_bytes());
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r)
        if (o.expert_cache_remote[(size_t) r] > 0 && !early_remote && !ready_remote_context(r)) return 1;

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    {   // the card, and whether this build has code for it (a binary built for other GPUs fails at its first kernel
        // otherwise, after the whole expert arena has loaded) - before the arena starts loading
        int dev = 0;
        cudaDeviceProp p{};
        const bool named = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess;
        if (!named) cudaGetLastError();
        const char* name = named && p.name[0] ? p.name : "(an unnamed GPU)";
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr, "strata generate: GPU %d: %s (%s)\n", dev, name, named ? p.gcnArchName : "?");
        // the first prompt's one-time hipBLAS library loads (about 2.5 s), done on a thread while the expert arena loads
        if (named) g_warm_thr = std::thread([dev] { strata::prefill::warm_gemm_libraries(dev); });
        // (RDNA4's matrix-core prompt attention is the default since 148-amd: the engine says so once, at its first prompt;
        // STRATA_HIP_WMMA=0 turns it off.)
#if defined(_WIN32)
        // #468 #461: which HIP runtime was loaded - the bundled one beside the exe, or an AMD driver's System32 copy
        if (HMODULE h = GetModuleHandleA("amdhip64_7.dll")) {
            char path[MAX_PATH] = {};
            if (GetModuleFileNameA(h, path, MAX_PATH) > 0) {
                std::fprintf(stderr, "strata generate: HIP runtime %s\n", path);
                // #1261: the bundled runtime sits beside the exe, but Windows bound the import to another copy (the AMD
                // driver's, in System32, was seen with a different build): kernels of this engine can then fail
                // (hipErrorInvalidDeviceFunction in the prompt GEMMs).  Said, not changed - the import is bound before main.
                char exe[MAX_PATH] = {};
                if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
                    std::string beside(exe);
                    const size_t cut = beside.find_last_of("\\/");
                    beside = (cut == std::string::npos ? std::string() : beside.substr(0, cut + 1)) + "amdhip64_7.dll";
                    if (GetFileAttributesA(beside.c_str()) != INVALID_FILE_ATTRIBUTES &&
                        _stricmp(beside.c_str(), path) != 0) {
                        std::fprintf(stderr, "strata generate: WARNING: the HIP runtime in use is not the bundled one beside "
                                             "the engine (%s): if prompts fail with hipErrorInvalidDeviceFunction, tell the "
                                             "maintainers with this log (#1261)\n",
                                     beside.c_str());
                        // #461: why the bundled one lost.  Its imports (rocm_kpack.dll and the C++ runtime) must be beside the
                        // exe too; LoadLibraryEx on it with the search limited to its own folder says what is missing (126 =
                        // a dependency).  A diagnostic only: the copy it loads is released at once and nothing is forced.
                        SetLastError(0);
                        HMODULE probe = LoadLibraryExA(beside.c_str(), nullptr,
                                                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                        const DWORD le = probe != nullptr ? 0 : GetLastError();
                        if (probe != nullptr) FreeLibrary(probe);
                        std::fprintf(stderr, "strata generate: the bundled runtime beside the engine %s (GetLastError %lu)%s\n",
                                     probe != nullptr ? "loads on its own" : "does NOT load", (unsigned long) le,
                                     le == 126 ? ": a DLL it imports is missing from the engine's folder (rocm_kpack.dll, "
                                                 "msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll); run START-HERE.bat "
                                                 "again (0.1.40.3 and later copy them)" : "");
                    }
                }
            }
        }
#endif
#else
        std::fprintf(stderr, "strata generate: GPU %d: %s, compute capability %d.%d%s\n", dev, name,
                     strata::cc_major_of(p.major), strata::cc_minor_of(p.minor),
                     strata::emulated_cc() ? " (STRATA_EMULATE_CC: a test mode, the card is emulated)" : "");
#if !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)   // a CUDA toolkit mismatch; the gfx906 HIP build has no CUDART_VERSION
        {   // #542: a build whose libcudart is older than its headers (a CUDA 13 kit with a dangling libcudart.so that
            // CMake resolved to the system's CUDA 12 one) reads cudaDeviceProp shifted - silently, and slowly
            int rt = 0;
            if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt / 1000 != CUDART_VERSION / 1000)
                std::fprintf(stderr, "strata generate: WARNING: this engine was compiled with CUDA %d.%d headers but "
                                     "loaded a CUDA %d.%d runtime (libcudart): GPU properties can read wrong and some "
                                     "kernels go unused. Rebuild it against one toolkit (cmake -DCUDAToolkit_ROOT=<the "
                                     "toolkit>, with its libcudart.so present) (#542)\n",
                             CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10, rt / 1000, rt % 1000 / 10);
        }
#endif
#endif
        const std::string e = strata::core::device_code_error();
        if (!e.empty()) {
            std::fprintf(stderr, "strata generate: this engine has no code for %s (sm_%d%d): %s - rebuild it for this "
                                 "card (setup does: START-HERE.bat --setup)\n", name, p.major, p.minor, e.c_str());
            return 1;
        }
    }
    // #4846 (owner 11-01): the arena first, and when the RAM is still short after the wait and the give-back, the --mmap-experts source
    // for this run (o.mmap_experts is set here, before anything below sizes the experts' tier from it; nothing was sized for the arena
    // by now: the PLE table's lock, priced against it, was just given back).  A hand-set `--ple-io ram` and a shared arena file are kept
    // as asked (warn only), STRATA_ARENA_RAM_FALLBACK=0 is the old behaviour.  Two passes at most; the second only after the first chose
    // the file, and if the file cannot be opened the arena is tried again the old way (a start never ends in a refusal for this).
    const bool ram_fallback_ok = strata::core::arena_ram_fallback_allowed(
        std::getenv("STRATA_ARENA_RAM_FALLBACK"), o.ple_io_explicit && o.ple_io == "ram", !o.shared_expert_arena.empty());
    bool ram_fallback_taken = false, ram_fallback_failed = false, ram_fallback_resident = false;
    const bool ram_fallback_resident_off = [] { const char* v = std::getenv("STRATA_ARENA_FALLBACK_RESIDENT"); return v != nullptr && v[0] == '0'; }();
    // The RAM-copy fallback leaves max(--ram-keep-gib, the safety headroom: 8 GiB on Windows, where the page trim of the copy needs the
    // room, 4 GiB elsewhere) free; STRATA_RESIDENT_HEADROOM_GIB, when set, decides instead.  arena_ram.hpp.
#if defined(_WIN32)
    constexpr bool kWindowsHost = true;
#else
    constexpr bool kWindowsHost = false;
#endif
    const uint64_t ram_fallback_headroom = [&] {
        std::optional<uint64_t> env_headroom;
        if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
            env_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        return strata::core::fallback_headroom(env_headroom, o.ram_keep, strata::core::fallback_safety_headroom(kWindowsHost));
    }();
    for (bool again = true; again;) {
    again = false;
    if (o.mmap_experts) {
        // FileExpertSource maps the pack's experts.bin: a canonical pack has it; a native (IQ) pack has it when
        // built with `tools/iq_pack.py --experts-bin` (the per-layer blob sizes of its layout, PR #121).  The low-RAM
        // mode: the experts come from the file through the OS cache instead of a pinned copy in RAM, for a PC whose
        // GPU holds most of them but whose RAM cannot hold them all.
        // CS-T: a native pack without experts.bin maps the model's GGUF shards instead (native_experts.txt's spans,
        // checked against the files first) - no 30-77 GB copy of the experts on the disk
        src.set_gguf(o.native_preset);
        if (const char* v = std::getenv("STRATA_FETCH_THREADS"); v != nullptr && std::atoi(v) > 0)
            src.set_fetch_threads(std::atoi(v));
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            if (ram_fallback_taken) {   // #4846: the file cannot be opened either; the arena, loaded the old way, is what is left
                std::fprintf(stderr, "strata generate: the experts cannot be read from the file either (%s): loading the expert arena "
                                     "as before, short of RAM\n", err.c_str());
                o.mmap_experts = false;
                if (ram_fallback_resident) { o.resident_cpu_experts = o.resident_pin = o.resident_soft = ram_fallback_resident = false; o.resident_budget = 0; }
                ram_fallback_taken = false;
                ram_fallback_failed = true;
                again = true;
                continue;
            }
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts via mmap (%s; %s)\n",
                     ram_fallback_taken ? "not enough RAM for the expert arena: the same source as --mmap-experts" : "--mmap-experts",
                     src.gguf_mode() ? "the GGUF shards in place, no experts.bin" : "the A/B arm of R2.1");
        // #286: with a RAM budget the hottest experts live in it, and the rest are read from the drive unbuffered
        // when the file cache could not keep them beside the budget anyway (a 32 GB PC) - the mapped reads' page
        // faults are small requests on the critical path, and their pages take the RAM the budget was sized for
        // #773: without a RAM budget there is no RAM copy and the file cache IS the expert tier (measured on a
        // 32 GB, 2 x 16 GB rig: forcing unbuffered reads there re-read 163-629 GB from the drive and halved the
        // speed), so STRATA_UNBUFFERED_LOAD=1 is not honoured in that mode
        if (o.resident_budget > 0) {
            std::string why;
            const bool ub = src.set_unbuffered(o.resident_budget, why);
            std::fprintf(stderr, "strata generate: the file tier reads %s (%s)\n",
                         ub ? "unbuffered" : "through the file cache", why.c_str());
        } else if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0' &&
                   env[0] != '0') {
            std::fprintf(stderr, "strata generate: the file tier reads through the file cache (STRATA_UNBUFFERED_LOAD=%s "
                                 "ignored: without --resident-budget-gib the file cache holds the experts)\n", env);
        }
#if defined(_WIN32)
        // the GPU cache's fill and the RAM copy read unbuffered even so: through the mapping their pages sat in the
        // working set beside the copy (end_startup_unbuffered below, once the copy is built).  Windows, experts.bin
        if (o.resident_cpu_experts && !src.unbuffered()) {
            std::string why;
            if (src.begin_startup_unbuffered(why))
                std::fprintf(stderr, "strata generate: the GPU cache's fill and the RAM copy read unbuffered (%s)\n",
                             why.c_str());
            else if (why.find("STRATA_STARTUP_UNBUFFERED") == std::string::npos)
                std::fprintf(stderr, "strata generate: the GPU cache's fill and the RAM copy read through the mapping "
                                     "(%s)\n", why.c_str());
        }
#endif
        // Linux I/O path: whole-blob preads instead of page faults, and the predicted experts of the next layers read
        // ahead on I/O threads while the current layer computes (see FileExpertSource::set_io_prefetch).  0.2.0: on by
        // default on Linux, with the decode gate (STRATA_IO_PF_GATE) and zero-copy page-cached experts
        // (STRATA_COLD_OVERLAP=4); STRATA_IO_PREFETCH=0 / STRATA_IO_PF_GATE=0 / STRATA_COLD_OVERLAP=0 turn them off.
        // io prefetch + the gate: the GGUF read in place and (0.2.1) an experts.bin pack; zero-copy only for the GGUF in place
        const char* io_pf_env = std::getenv("STRATA_IO_PREFETCH");
        const bool io_pf_default = (io_pf_env == nullptr || io_pf_env[0] == '\0') && src.linux_io_defaults();
#if defined(__linux__)
        if ((io_pf_env == nullptr || io_pf_env[0] == '\0') && !io_pf_default && !src.linux_file_defaults() && !src.unbuffered())
            std::fprintf(stderr, "strata generate: file tier (experts.bin): io prefetch stays off by default here: the page cache has room "
                                 "for %.0f%% of the experts (it helps from 55%%: 32 GB-class RAM, decode +20..34%% measured); "
                                 "STRATA_IO_PREFETCH=1 turns it on anyway\n", src.cache_room_ratio() * 100.0);
#endif
        if (io_pf_default || (io_pf_env != nullptr && std::atoi(io_pf_env) != 0)) {
            const char* th = std::getenv("STRATA_IO_PF_THREADS");
            src.set_io_prefetch(true, th != nullptr ? std::atoi(th) : 0);
            io_pf_by_default = io_pf_default && src.io_prefetch();
            if (src.io_prefetch() && io_pf_default) {
                const char* cv = std::getenv("STRATA_COLD_OVERLAP");
                std::fprintf(stderr, "strata generate: file tier (Linux defaults, %s): page-cached experts "
                                     "computed from the mapping (STRATA_COLD_OVERLAP=%s), io prefetch with whole-blob preads on %s I/O threads, "
                                     "the decode's read-ahead %s; off: STRATA_COLD_OVERLAP=0, STRATA_IO_PREFETCH=0, "
                                     "STRATA_IO_PF_GATE=0\n",
                             src.linux_file_defaults() ? "the GGUF read in place" : "experts.bin: its pages are read from the mapping as they are",
                             cv != nullptr && cv[0] != '\0' ? cv : (src.linux_file_defaults() ? "4" : "0"), th != nullptr ? th : "8",
                             src.gate_on() ? "only while the page cache has room for it and it pays (STRATA_IO_PF_GATE=1)"
                                           : "always (STRATA_IO_PF_GATE=0)");
            } else if (src.io_prefetch())
                std::fprintf(stderr, "strata generate: file tier io prefetch ON (STRATA_IO_PREFETCH=1): whole-blob preads, "
                                     "the layer's uncached experts and the predicted ones read by %s I/O threads (STRATA_IO_PF_THREADS, default 8)%s\n",
                             th != nullptr ? th : "8", src.gate_on() ? "; the decode's read-ahead gated (STRATA_IO_PF_GATE=0: always)" : "");
            if (src.io_prefetch() && src.io_sched())
                std::fprintf(stderr, "strata generate: io prefetch scheduling ON (STRATA_IO_PF_SCHED=1): a deadline per read, "
                                     "confidence admission (STRATA_IO_PF_MINCONF), a per-window byte budget (STRATA_IO_PF_BUDGET_MB / _FRAC)\n");
            if (!src.io_prefetch() && !io_pf_default)
                std::fprintf(stderr, "strata generate: STRATA_IO_PREFETCH=1 is not available here (Linux mapped file tier "
                                     "only, not with unbuffered reads); the file tier reads as before\n");
            if (src.io_prefetch()) {   // recommend, never force: the engine runs as asked
                const char* st = std::getenv("STRATA_IO_PF_STAGE");
                if (st != nullptr && std::atoi(st) != 0)
                    std::fprintf(stderr, "strata generate: note: STRATA_IO_PF_STAGE=1 (staging) measured faster only on a 32 GB-class "
                                         "box (+25%% on an RTX 3060, 32 GB); it was slower at 16 GB and on an iGPU (docs/DETAILS.md, file tier I/O)\n");
#if defined(STRATA_USE_HIP)
                int igpu_dev = 0;
                cudaDeviceProp igpu_p{};
                if (cudaGetDevice(&igpu_dev) == cudaSuccess && cudaGetDeviceProperties(&igpu_p, igpu_dev) == cudaSuccess &&
                    igpu_p.integrated != 0 && o.adapt_every > 0 && o.adapt_swaps > 0)
                    std::fprintf(stderr, "strata generate: WARNING: STRATA_IO_PREFETCH=1 with the adaptive tier on an AMD iGPU (%s): "
                                         "GPU resets were seen with this combination under memory pressure (Radeon 780M); "
                                         "--adapt-every 100000 avoided them. Continuing as asked.\n", igpu_p.gcnArchName);
                else
                    (void) cudaGetLastError();
#endif
            }
        } else if (const char* sv = std::getenv("STRATA_IO_STATS"); sv != nullptr && std::atoi(sv) != 0) {
            src.set_io_stats(true);
        }
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        // Under WDDM (Windows, WSL2), a multi-GPU run (a layer split, or remote experts) starts with at most 8 GiB of
        // mapped host pages: pinning all of it into two contexts leaves WDDM refusing every later allocation -
        // measured on the 5080 + 3090 rig: cudaMemGetInfo and the next cudaMalloc fail.  Unregistered layers remain
        // in the resident arena; their streamed experts go through the pinned staging ring.  A Linux driver has no
        // such limit, so there the whole arena is pinned (#253).  STRATA_ARENA_PIN_GIB overrides both ways.
        const int pin_env = strata::core::arena_pin_cap_gib();   // -1 unset, -2 "auto" (#243, Windows sliced pin)
        const bool pin_wddm_cap = pin_env < 0 && (o.expert_cache_remote[0] > 0 || multi_gpu) && under_wddm();
        const uint64_t pin_limit = pin_env >= 0 ? (uint64_t) pin_env << 30 : pin_wddm_cap ? (8ull << 30) : 0;
        if (pin_env >= 0)
            std::fprintf(stderr, "strata generate: STRATA_ARENA_PIN_GIB=%d: %s\n", pin_env,
                         pin_env == 0 ? "the whole expert arena is pinned" : "the expert arena's pinning is capped");
        else if (pin_wddm_cap)
            std::fprintf(stderr, "strata generate: multi-GPU under WDDM: at most 8 GiB of the expert arena is pinned "
                                 "(STRATA_ARENA_PIN_GIB changes it)\n");
        // ---- STRATA_PIN_BY_NEED=1 (default off: the prefix pinning above, byte-identical).  WDDM does not let the
        // pin grow past ~RAM/2, and the prefix spends it on layers 0-31 whatever the GPU holds: layers 32-47 stay 100 %
        // CPU while ~10 GiB of pinned blobs sit in GPU-resident experts no DMA ever reads.  Pinning is registered
        // BEFORE the arena's pages are touched (pinned.cu), so the set must be chosen here, before arena_src.open: the
        // profile is already read, and the GPU cache's size - opened much later - is ESTIMATED from the same free-VRAM
        // reading its auto sizing uses (the weights, session and drafter are loaded by now, host registration takes no
        // VRAM): the first R ranked pairs are taken as GPU-resident, and the rest (then the unranked pairs, in layer
        // order) are what the arena pins, hottest first, until the driver refuses.  R is under-estimated a little
        // (x0.98): a resident pair pinned by mistake costs a few MiB, a hot non-resident one skipped costs DMA.
        // STRATA_PIN_BY_NEED_SKIP=<n> forces R.  Single GPU, profile given, no helper caches only.
        std::vector<std::pair<int32_t, int32_t>> pin_cand;   // the non-resident estimate, by decreasing rank, then the resident one
        size_t pin_nonres = 0;                               // (how many of them are the non-resident estimate)
        if (const char* pn = std::getenv("STRATA_PIN_BY_NEED"); pn != nullptr && pn[0] != '\0' && pn[0] != '0') {
            const char* off_why = profile.empty() ? "it needs --expert-profile (the ranking)"
                                : multi_gpu || o.peer_device >= 1 ? "not with a layer split or a peer GPU"
                                : remote_caches ? "not with helper caches on other GPUs"
                                : nullptr;
            if (off_why != nullptr) {
                std::fprintf(stderr, "strata generate: pin by need: off (%s); the prefix pinning stays\n", off_why);
            } else {
                const auto& pl = strata::kernels::cpu::expert_layout();
                size_t skip = profile.size();
                if (const char* sk = std::getenv("STRATA_PIN_BY_NEED_SKIP"); sk != nullptr && sk[0] != '\0') {
                    skip = std::min<size_t>(profile.size(), (size_t) std::max<long long>(std::atoll(sk), 0));
                } else {
                    // the bytes the GPU cache will hold: --expert-cache N slots of the largest blob, or what auto sizing
                    // finds free (the same reserve terms as "expert cache auto" below, without its small-card rescue,
                    // which only ever grants MORE slots)
                    uint64_t cap_bytes;
                    if (o.expert_cache > 0) {
                        cap_bytes = (uint64_t) o.expert_cache * nibble_max_slot_bytes();
                    } else {
                        const int64_t pf_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
                        int64_t mtp_b = (!o.mtp.empty() && native_head.loaded())
                                            ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
#if defined(STRATA_HAVE_DFLASH2)
                        if (use_df2 && native_head.loaded()) mtp_b += get_df2_head_bytes() + (48 << 20) + (o.serve ? (112 << 20) : 0);   // DFlash2's own draft head and logits (serve: and the prompt staging)
#endif
                        if (stages.empty())
                            for (const auto& d : slot_mtp) mtp_b += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
                        const int64_t pipe_b = o.pipeline_windows <= 0 ? 0
                            : (pipe_window_mib << 20) + (o.pipeline_windows >= 2 ? 2 * (int64_t) gdn_snapshot_bytes(ss) : 0);
                        const int64_t room = (int64_t) strata::core::device_free_bytes() -
                                             ((((int64_t) o.vram_reserve_mib + pf_mib) << 20) + mtp_b + pipe_b);
                        cap_bytes = room > 0 ? (uint64_t) room : 0;
                    }
                    cap_bytes = cap_bytes / 100 * 98;
                    uint64_t used = 0;
                    skip = 0;
                    for (const auto& pr : profile) {
                        const uint64_t b = pl.native ? (nib::slot_bytes(pr.first) + 255) / 256 * 256 : (uint64_t) pl.max_blob;
                        if (used + b > cap_bytes) break;
                        used += b;
                        ++skip;
                    }
                }
                std::vector<uint8_t> seen((size_t) (g.n_layers * g.n_expert), 0);
                for (const auto& pr : profile)
                    if (pr.first >= 0 && pr.first < g.n_layers && pr.second >= 0 && pr.second < g.n_expert)
                        seen[(size_t) (pr.first * g.n_expert + pr.second)] = 1;
                pin_cand.assign(profile.begin() + (long) skip, profile.end());
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (!seen[(size_t) (l * g.n_expert + e)]) pin_cand.emplace_back((int32_t) l, (int32_t) e);
                pin_nonres = pin_cand.size();   // (the report counts these)
                // what the cap leaves over goes to the pairs taken as GPU-resident (last in line): the adaptive tier
                // swaps them out as the conversation moves, and the prompt read streams whatever is not resident then
                pin_cand.insert(pin_cand.end(), profile.begin(), profile.begin() + (long) skip);
                std::fprintf(stderr, "strata generate: pin by need: %zu of the %zu ranked pairs taken as GPU-resident "
                                     "(STRATA_PIN_BY_NEED_SKIP=<n> sets it), %zu candidates to pin\n",
                             skip, profile.size(), pin_nonres);
                arena_src.set_pin_priority(pin_cand, pin_nonres);
            }
        }
        // The arena pins until the OS refuses; on Windows that leaves no page-locked memory for the small buffers the
        // engine allocates after it (the verify window's staging: "mapped staging allocation failed", exit 1).  Keep a
        // reserve while the arena pins and hand it back after (pin_budget.hpp).
        if (ple_lock_by_policy)   // #4846: the residency policy priced the table against the arena before the rest of the start took its RAM
            arena_src.give_back_ram = [&]() -> uint64_t {
                if (!ple_table.locked()) return 0;
                const uint64_t held = ple_table.table_bytes();
                if (!ple_table.unlock()) return 0;
                ple_lock_by_policy = false;   // (given back: the conversation cache's give-back has nothing left to do)
                return held;
            };
        strata::core::pin_reserve_take();
        arena_src.fall_back_when_ram_short = ram_fallback_ok && !ram_fallback_failed;
        arena_src.notices_on_stdout = o.serve;   // #5327: the RAM notices also on stdout in --serve mode only
        arena_src.ram_keep_bytes = o.ram_keep;
        arena_src.hard_limit_headroom_bytes = ram_fallback_headroom;
        if (o.ram_keep > 0)
            std::fprintf(stderr, "strata generate: --ram-keep-gib %.2f: the expert arena is used only if %.2f GiB of RAM stay available after it; "
                                 "otherwise the RAM copy of the hottest experts leaves %.2f GiB free (the larger of that and the %.0f GiB safety headroom, "
                                 "STRATA_RESIDENT_HEADROOM_GIB decides when set)\n",
                         (double) o.ram_keep / 1073741824.0, (double) o.ram_keep / 1073741824.0, (double) ram_fallback_headroom / 1073741824.0,
                         (double) strata::core::fallback_safety_headroom(kWindowsHost) / 1073741824.0);
        const bool arena_ok = arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                                             o.shared_expert_arena);
        strata::core::pin_note_arena(arena_src.pinned_arena_bytes());
        strata::core::pin_sim_report("after the expert arena");
        strata::core::pin_reserve_release_soft();
        if (!arena_ok && arena_src.ram_short()) {
            // one plain line, one next step; the numbers are in err.  With --serve STDOUT TOO, not only stderr: a service
            // setup (llama-swap and its like) keeps the engine's stdout in its log and never shows the stderr this line
            // lived on, so the 2.4x decode loss this fallback costs was invisible until it was measured (#5327).  A plain
            // generate run keeps its stdout (the token stream) clean: stderr only, as before.
            std::fprintf(stderr, "strata generate: not enough RAM for the expert arena (%s): this run reads the experts from the "
                                 "file, like --mmap-experts (slower; the answers are the same). To get the arena back, free that RAM "
                                 "and start again\n", err.c_str());
            if (o.serve) {
                std::printf("strata generate: not enough RAM for the expert arena (%s): this run reads the experts from the "
                            "file, like --mmap-experts (slower; the answers are the same). To get the arena back, free that RAM "
                            "and start again\n", err.c_str());
                std::fflush(stdout);
            }
            o.mmap_experts = true;
            ram_fallback_taken = true;
            // R5070: the plain file source has no RAM copy at all, and on the owner's 5070 PC (63.1 GB, 44.6 GiB available for a
            // 46.84 GiB arena) it decodes at 9-10 tok/s where a RAM copy of the hottest experts gives 27-28.  The fallback
            // takes the --resident-experts form instead (the hottest experts by the profile in RAM as far as the available
            // RAM allows, the rest read from the files); needs the static expert profile, STRATA_ARENA_FALLBACK_RESIDENT=0
            // keeps the plain file source.
            if (!o.expert_profile.empty() && !o.resident_cpu_experts && !ram_fallback_resident_off) {
                o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
                o.resident_headroom = ram_fallback_headroom;
                // the budget form (as --resident-budget-gib): the hottest experts in what is available minus the headroom, and the rest
                // read unbuffered, not through the mapped view (whose pages grow the working set until available RAM is near zero)
                if (const auto hm = strata::core::sample_host_memory(); hm)
                    o.resident_budget = strata::core::fallback_budget(hm->available, o.resident_headroom);
                ram_fallback_resident = true;
            }
            again = true;
            continue;
        }
        if (!arena_ok) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!pin_cand.empty()) {
            const auto& pl = strata::kernels::cpu::expert_layout();
            uint64_t cand_b = 0, pin_b = 0;
            for (size_t ci = 0; ci < pin_nonres; ++ci) {
                const auto& pr = pin_cand[ci];
                const uint64_t b = pl.blob_bytes(pr.first);
                cand_b += b;
                if (arena_src.pinned(pr.first, pr.second)) pin_b += b;
            }
            int64_t pcie_layers = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) pcie_layers += arena_src.pcie_layer(l) ? 1 : 0;
            std::fprintf(stderr, "strata generate: pin by need: %zu ranges, %.2f GiB pinned, %.1f %% of the non-resident "
                                 "experts (%lld of %lld layers have a PCIe share)\n",
                         arena_src.pin_ranges(), (double) pin_b / 1073741824.0,
                         cand_b > 0 ? 100.0 * (double) pin_b / (double) cand_b : 0.0, (long long) pcie_layers,
                         (long long) g.n_layers);
        }
        if (!arena_src.ram_warning().empty())   // #633: said before the load's numbers, which it explains
            std::fprintf(stderr, "strata generate: WARNING: %s\n", arena_src.ram_warning().c_str());
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        // A rate under ~0.2 GiB/s is not the hardware.  Task Scheduler / service contexts throttle this
        // read+fill about 24x (measured 0.05 vs 1.42 GiB/s for the same binary, args and cache state; the
        // scheduler's defaults - Below normal priority and a least-privilege token - were the only
        // difference between the runs).  Say so instead of letting the user blame the disk; see
        // docs/DETAILS.md, "Running it at startup (Task Scheduler)".
#ifdef _WIN32   // a Windows launch context; elsewhere a load this slow is the disk
        if (arena_src.load_gib_per_second() > 0.0 && arena_src.load_gib_per_second() < 0.2) {
            std::fprintf(stderr,
                         "strata generate: hint: ~24x below what this hardware streams from a normal "
                         "launch. If Strata is started by Task Scheduler or a service, register the task "
                         "with Priority 4 (Normal) and 'Run with highest privileges' - the scheduler's "
                         "defaults (Below normal + a least-privilege token) throttle the load. See "
                         "docs/DETAILS.md ('Running it at startup').\n");
        }
#endif
        srcp = &arena_src;
    }
    }   // the arena / file pass loop (#4846)
#if defined(__linux__)
    // 0.1.43.1: on by default (STRATA_ADAPT_CALLBACK_WARMUP=0 turns it off).  Linux threads inherit the creator's CPU mask, and
    // the host pins itself to one CPU later; a callback thread the driver creates lazily after that would run on the
    // host's CPU and fight the host's spin-wait (985 us instead of 4 us per callback, #1964 #1975).  Windows CPU Sets are
    // per thread and not inherited.
    if (const char* e = std::getenv("STRATA_ADAPT_CALLBACK_WARMUP"); !(e && e[0] == '0')) {
        // The driver creates its host-callback thread lazily. Create it before ExpertPool pins this thread.
        if (cudaLaunchHostFunc(nullptr, [](void*) {}, nullptr) != cudaSuccess ||
            cudaStreamSynchronize(nullptr) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata generate: could not warm up the GPU host-callback thread (continuing)\n");
        } else {
            std::fprintf(stderr, "strata generate: GPU host-callback thread started before host-core pinning\n");
        }
    }
#endif
    if (o.pool_measure)
        o.pool_affinity = measure_pool_affinity(srcp, o.pool_workers, !o.no_host_worker, o.pool_tasks);
    strata::kernels::cpu::ExpertPool& pool = cpu_pool.emplace(o.pool_workers, /*pin=*/true,
                                                             /*host_works=*/!o.no_host_worker,
                                                             o.pool_affinity, o.pool_tasks);
    std::fprintf(stderr, "strata generate: CPU pool tasks/phase: %d%s, participating threads: %d\n",
                 o.pool_tasks ? o.pool_tasks : 3 * (pool.workers() + (pool.host_works() ? 1 : 0)),
                 o.pool_tasks ? " (capped by rows)" : " (automatic)",
                 pool.workers() + (pool.host_works() ? 1 : 0));
    if (pool.plan_limit()[0] != '\0' && !pool.explicit_cores())   // the default count: what it was bounded by (a configured count is never changed; --pool-cores has its own line below)
        std::fprintf(stderr, "strata generate: CPU pool: %d workers by default (%d physical cores, %d logical processors; bounded by %s)\n",
                     pool.workers(), pool.plan_cores(), pool.plan_logical(), pool.plan_limit());
    if (srcp == &arena_src && strata::kernels::cpu::active_numa_split().on && !strata::kernels::cpu::active_numa_split().test)
        std::fprintf(stderr, "strata generate: NUMA split on, but this model's experts come from the arena path: memory is "
                             "not placed per node (known limitation)\n");
    if (pool.is_hybrid() && pool.affinity() != strata::kernels::cpu::PoolAffinity::All) {
        const char* aff_str = pool.affinity() == strata::kernels::cpu::PoolAffinity::PCores ? "p-cores" :
                              pool.affinity() == strata::kernels::cpu::PoolAffinity::All ? "all" : "auto";
        std::fprintf(stderr, "strata generate: hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s\n",
                     pool.p_cores(), pool.p_threads(), pool.e_cores(), pool.workers(), aff_str);
    }
    {   // where the threads run (--host-core)
        const strata::kernels::cpu::CpuTopology ht = strata::kernels::cpu::detect_cpu_topology(true, o.pool_affinity);
        // the pool's own order, not a fresh detection's: STRATA_NUMA_SPLIT interleaves the sockets
        const std::vector<int>& wc = pool.worker_cores();
        std::string on;
        for (int i = 0; i < pool.workers() && i < (int) wc.size(); ++i)
            on += (i > 0 ? "," : "") + std::to_string(wc[(size_t) i]);
        std::fprintf(stderr, "strata generate: %d pool workers on logical processors %s, the host thread on %d%s "
                             "(--host-core %s)\n", pool.workers(), on.c_str(), ht.host_core,
                     pool.host_works() ? " (draining too)" : "",
                     strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Last      ? "last"
                     : strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Sibling ? "sibling"
                                                                                                             : "first");
        // the same facts in plain words, and whether each worker has a physical core of its own
        std::fprintf(stderr, "strata generate: %s\n", pool.placement_line().c_str());
        // --aux-cpus: the plan, now the workers' CPUs are known; the threads that exist already (the CUDA driver's) move
        // now when it is on, the ones created later pin themselves (strata/base/platform/aux_cpus.hpp)
        const size_t nw = std::min<size_t>((size_t) pool.workers(), wc.size());
        std::fprintf(stderr, "%s\n",
                     strata::aux_cpus::set_topology(ht.host_core, std::vector<int>(wc.begin(), wc.begin() + (long) nw)).c_str());
    }
    if (o.no_ple_prefetch) strata::kernels::ple_prefetch_enable(false);
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena (#620)
    auto_cache = o.expert_cache < 0;
    // --pipeline-windows: what it allocates on CUDA0 after the cache (see pipe_window_mib)
    const int64_t pipe_first = o.pipeline_windows <= 0 ? 0
        : (pipe_window_mib << 20) + (o.pipeline_windows >= 2 ? 2 * (int64_t) gdn_snapshot_bytes(ss) : 0);
    if (pipe_first > 0)
        std::fprintf(stderr, "strata generate: --pipeline-windows %d: %lld MiB of CUDA0 kept out of the expert cache "
                             "(the second verifier%s)\n", o.pipeline_windows, (long long) (pipe_first >> 20),
                     o.pipeline_windows >= 2 ? ", the GDN snapshots" : "");
    // The MiB the prompt path's OWN buffers (no loan) take, which the cache sizing leaves out.  0.1.39's rule is
    // 160 + chunk * 680 / 1024 (~16 GiB at 24576 against ~2.4 GiB really allocated); STRATA_OWNED_PRICE=exact prices the
    // real allocation instead (Prefill::bytes_needed_owned: 2 MiB pages, the ring in one piece, as PR #796 measured)
    // and gives that cache the difference.  A recommendation to try, so it is not the default.
    auto owned_prefill_mib = [&]() -> int64_t {
        if (!(o.prefill_chunk > 0 && !pf_borrow)) return 0;
        static const bool exact = [] { const char* v = std::getenv("STRATA_OWNED_PRICE"); return v != nullptr && std::string(v) == "exact"; }();
        if (!exact) return 160 + (o.prefill_chunk * 680) / 1024;
        const int64_t mib = ((int64_t) strata::prefill::Prefill::bytes_needed_owned(g, ss, o.prefill_chunk, srcp != nullptr) + (1 << 20) - 1) >> 20;
        std::fprintf(stderr, "strata generate: STRATA_OWNED_PRICE=exact: the prompt path's own buffers for a %lld-token chunk: %lld MiB (the 0.1.39 rule: %lld MiB)\n",
                     (long long) o.prefill_chunk, (long long) mib, (long long) (160 + (o.prefill_chunk * 680) / 1024));
        return mib + 64;   // a margin for the allocator
    };
    if (o.expert_cache < 0) {
        size_t free_b = strata::core::device_free_bytes();
        if (free_b == 0 && o.prefill_head_alias) {
            (void) cudaGetLastError();
            free_b = strata::core::device_free_bytes();
        }
#if defined(STRATA_USE_HIP) && !defined(_WIN32)
        {   // An integrated GPU's "free VRAM" is system RAM (the BIOS carve-out plus the GTT pool, nearly all of the RAM when the kernel was told
            // ttm.pages_limit), and the cache copies experts into that same RAM on top of the arena: sized from the pool alone it can ask for
            // more than the OS has left, and the OOM killer takes the engine or the desktop.  So the AUTO size counts that RAM once: the
            // cache gets at most what the OS reports available less a reserve for the desktop
            // (include/strata/common/experts/uma_cache.hpp).
            // A recommendation: --expert-cache N is kept as asked, STRATA_UMA_RESERVE_GIB=0 gives the old sizing.
            cudaDeviceProp up{};
            int ud = 0;
            strata::core::detail::HostMemory uh{};
            if (cudaGetDevice(&ud) == cudaSuccess && cudaGetDeviceProperties(&up, ud) == cudaSuccess && up.integrated &&
                strata::core::detail::host_available_memory(uh)) {
                const uint64_t reserve = strata::core::uma_desktop_reserve_bytes(strata::platform::total_physical_memory());
                const uint64_t room = strata::core::uma_cache_room_bytes((uint64_t) uh.available, reserve);
                if (reserve > 0 && room < (uint64_t) free_b) {
                    std::fprintf(stderr, "strata generate: expert cache auto: this GPU shares system RAM: %.2f GiB free on the GPU, but the OS has "
                                         "%.2f GiB available; %.2f GiB stay for the desktop, so the cache is sized from %.2f GiB "
                                         "(STRATA_UMA_RESERVE_GIB changes the reserve, --expert-cache N sets the cache)\n",
                                 (double) free_b / 1073741824.0, (double) uh.available / 1073741824.0,
                                 (double) reserve / 1073741824.0, (double) room / 1073741824.0);
                    free_b = (size_t) room;
                }
            } else {
                (void) cudaGetLastError();
            }
        }
#endif
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead; `pf_borrow` is
        // the predicate a local `borrow` was here, hoisted above so both cache-size branches read the same one)
        // #1376: under WDDM the free figure read here runs high, and what is allocated after the cache (the verify
        // windows, the prompt path) comes out of the reserve.  0.1.40 ended a 20 GB RX 7900 XT with 2,170 MiB free
        // (its write-back check cut the cache); 0.1.40.2 passed that check with the default 700 MiB and ended with
        // 323 MiB, where Windows paged and decode fell from 52 to 21-25 tok/s.  --vram-reserve-mib 2600 fixed it.  So on
        // Windows HIP an auto cache on a card of 16 GB (15 GiB and up, #1709) keeps a floor of kWddmAutoReserveMib.  CUDA on Windows is
        // left as it was: the same 0.1.40 / 0.1.40.2 pair on an RTX 5070 ended with the same slots and free VRAM, and a
        // floor there would only cost 16 GB+ NVIDIA cards ~1.8 GiB of cache.  A reserve given on the command line is
        // kept as it is; a smaller card keeps its own sizing (#496 lowers the reserve further when the cache would not
        // fit); and this is said, never silent.  The cache size changes no output bit.
        constexpr int kWddmAutoReserveMib = 2560;
        if (!o.vram_reserve_given && o.vram_reserve_mib < kWddmAutoReserveMib) {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
            size_t fb_now = 0, tb_now = 0;
            // #1709: a 16 GB card (RX 9070 XT, 7800 XT) reports 15.9 GiB, just under 16: the floor is for 15 GiB and up
            // G16: and for a DISCRETE card only.  An integrated GPU (the device property, as the UMA sizing above reads it) shares the RAM with the budget's
            // page-locked complement, which takes the freed room straight back: on a Strix Halo the floor left UD-Q4_K_XL no room for the prompt path (wddm_auto_reserve_mib)
            cudaDeviceProp fl_prop{};
            int fl_dev = 0;
            const bool fl_integrated = cudaGetDevice(&fl_dev) == cudaSuccess && cudaGetDeviceProperties(&fl_prop, fl_dev) == cudaSuccess && fl_prop.integrated != 0;
            if (cudaMemGetInfo(&fb_now, &tb_now) == cudaSuccess &&
                strata::core::wddm_auto_reserve_mib(fl_integrated, (uint64_t) tb_now, o.vram_reserve_mib, kWddmAutoReserveMib) == kWddmAutoReserveMib) {
                std::fprintf(stderr, "strata generate: expert cache auto: Windows keeps %d MiB free after the cache on a card "
                                     "this size (the default %d MiB left 323 MiB on a 20 GB card and Windows paged, #1376); "
                                     "--vram-reserve-mib N sets it\n", kWddmAutoReserveMib, o.vram_reserve_mib);
                o.vram_reserve_mib = kWddmAutoReserveMib;
            }
#endif
        }
        const int64_t prefill_mib = owned_prefill_mib();
        // the draft layer's head and logits are allocated when it binds, after this: 0.1.27's CJK subset made them
        // ~110-180 MiB larger, and out of the reserve they left 16 GB cards below the stall line (#199)
        int64_t mtp_bind = (!o.mtp.empty() && native_head.loaded())
                               ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab)
                               : (mtp_first && last_st->head.loaded()) ? (int64_t) mtp.bind_bytes(last_st->head.row_bytes(), n_vocab) : 0;
        // (with a layer split the slot drafters bind on the last stage, whose cache is sized from what is free there)
        if (stages.empty())
            for (const auto& d : slot_mtp)
                mtp_bind += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2 && native_head.loaded()) mtp_bind += get_df2_head_bytes() + (48 << 20) + (o.serve ? (112 << 20) : 0);   // DFlash2's own draft head and logits (serve: and the prompt staging), bound after this
#endif
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + pipe_first;
        const int64_t blob = (int64_t) nibble_max_slot_bytes();   // max_blob (STRATA_EXPERT_NIBBLE: or a larger 4-bit slot)
        // #496: the least a working cache needs (the verify window cannot start without one, #174): a cache too small to lend the prompt path a
        // 256-token chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it allocate its own on top
        int64_t min_slots = (o.prefill_chunk > 0 && pf_borrow)
            ? ((int64_t) strata::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + g_lend_keep : 1;
        // #4474: what the load allocates after the cache that no setting prices (cuBLAS and the lazily loaded modules the prompt path and the
        // verifier warm up, ~245 MiB on every card measured) came out of the reserve, so a card ended 240 MiB under the reserve it was asked to
        // keep (459 MiB free at --vram-reserve-mib 700: under the 512 MiB line the serve check warns about, #2065).  The cache now leaves it too.
        // Not on a card that could not keep min_slots after it, and not under STRATA_DEFAULTS=026 (the v0.2.6 sizing).
        strata::core::AutoCacheSlots auto_slots = strata::core::auto_cache_slots(
            (int64_t) free_b, reserve, blob, min_slots, strata::qwen_defaults_026() ? 0 : strata::core::kLateAllocMib);
        int64_t slots = auto_slots.slots;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
#if defined(STRATA_GPU_METAL)
        // Metal: the decode window reads its experts from its own arena (metal/src/engine/window.cpp), and the device copies the shared host
        // code makes - this cache included - are given back before the arena is sized.  An auto cache of everything that is "free" (14.56 GiB on
        // a 32 GB Mac) only held the memory the arena and the prompt buffers need, left 85 MiB free at the first request, and was touched
        // again after it was given back.  1500 slots is the cache the identity gates run with; --expert-cache N sets another.
        constexpr int64_t kMetalAutoCacheSlots = 1500;
        if (slots > kMetalAutoCacheSlots) {
            std::fprintf(stderr, "strata generate: expert cache auto: %d slots, not %lld: on a Mac the engine reads its experts from its own arena and "
                                 "this cache would only hold memory the arena and the prompt buffers need (--expert-cache N sets it)\n",
                         (int) kMetalAutoCacheSlots, (long long) slots);
            slots = kMetalAutoCacheSlots;
        }
#endif
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head%s) -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20),
                     auto_slots.late_priced ? ", +256 MiB for what is allocated after the cache" : "", o.expert_cache);
#if defined(STRATA_USE_HIP) && !defined(_WIN32)
        // An APU's "VRAM" is system RAM: the device-free figure above counts the whole GPU-addressable pool and does not
        // subtract ordinary CPU allocations (the host expert arena), so a cache sized from it alone can ask the OOM killer
        // for nearly all RAM (PR #895). A warning, not a cap: the cache stays as asked, and --expert-cache N chooses.
        {
            cudaDeviceProp apu_prop{};
            int apu_dev = 0;
            strata::core::detail::HostMemory hm{};
            if (cudaGetDevice(&apu_dev) == cudaSuccess && cudaGetDeviceProperties(&apu_prop, apu_dev) == cudaSuccess &&
                apu_prop.integrated && strata::core::detail::host_available_memory(hm)) {
                const int64_t host_headroom = (int64_t) strata::core::uma_desktop_reserve_bytes(strata::platform::total_physical_memory());   // the OS, the desktop and request-time CPU work
                const int64_t cache_b = (int64_t) o.expert_cache * blob;
                if (cache_b > (int64_t) hm.available - host_headroom)
                    std::fprintf(stderr, "strata generate: WARNING: this GPU shares system RAM, and the cache above (%.2f GiB) is "
                                         "more than the host memory available now (%.2f GiB) less 4 GiB for the OS: the system may "
                                         "run out of memory; --expert-cache N sets the cache by hand\n",
                                 (double) cache_b / 1073741824.0, (double) hm.available / 1073741824.0);
            }
        }
#endif
        // SHORT OF VRAM: A LOAN THAT LEAVES LESS.  The prompt path borrows cache slots for its buffers and the cache keeps 128 behind; a card whose cache stays smaller
        // than that (8 GB NVIDIA, 128K: 112-212 slots) used to be squeezed by lowering the reserve to give the cache its 326 working slots, which left 100 MiB
        // free and stopped the first long prompt ("mmq: out of memory"), or gave the prompt path its own 600 MiB of buffers on top and stopped with "do not
        // fit".  The loan may leave 16 slots instead: the cache is smaller, prompts are read in smaller chunks and slower, the reserve stays, and the start works.
        if (o.prefill_chunk > 0 && pf_borrow && o.prefill_auto && o.expert_cache < min_slots && !o.vram_reserve_given &&
            std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu()) {
            constexpr int64_t kTightKeep = 16;
            const int64_t tight_min = ((int64_t) strata::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + kTightKeep;
            if (tight_min < min_slots) {
                std::fprintf(stderr, "strata generate: short of GPU memory: the cache (%d slots) is too small to lend the prompt path 256-token buffers and keep %lld more: "
                                     "the loan keeps %lld, prompts are read in smaller chunks, expect them to be slower\n",
                             o.expert_cache, (long long) min_slots, (long long) kTightKeep);
                min_slots = tight_min;
                g_lend_keep = kTightKeep;
                auto_slots = strata::core::auto_cache_slots((int64_t) free_b, reserve, blob, min_slots, strata::qwen_defaults_026() ? 0 : strata::core::kLateAllocMib);
                int64_t s3 = auto_slots.slots;
                if (!profile.empty()) s3 = std::min<int64_t>(s3, (int64_t) profile.size());
                o.expert_cache = (int) std::max<int64_t>(s3, 0);
            }
        }
        // #496: the verify window cannot start without a cache (#174), and a cache too small to lend the prompt path
        // a 256-token chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it
        // allocate its own on top - more than the reserve.  When the default reserve leaves less than that (a 6 GB
        // card), the reserve shrinks to what leaves exactly that cache, down to kSmallReserveMib: what is allocated
        // after the cache - the prompt path's own part, the verify buffers, the draft head - comes out of the reserve,
        // and below ~550 MiB a card ends with less than the 256 MiB the serve check calls LOW (IQ3_XXS, 32K, a 300 MiB
        // reserve: 5 MiB left), so the cache gets no more than it needs, and the serve check says so plainly when it
        // ends LOW (`reserve_adapted`).  A reserve given on the command line is kept.  No slot at all: the start
        // stops, saying what is short and what makes room.  A card the default reserve leaves that much is sized as
        // before.
        constexpr int kSmallReserveMib = 300;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - pipe_first - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                int64_t s2 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind + pipe_first)) / blob;
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                std::fprintf(stderr, "strata generate: expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots\n",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
                reserve_adapted = true;
            }
        }
        // SHORT OF VRAM: THE PROMPT PATH GETS ITS OWN SMALL BUFFERS, NOT A LOAN.  A cache below min_slots cannot lend the prompt path its chunk, so the
        // path allocates its own buffers after the cache, and nothing above priced them: the cache took what was left, and the first prompt (or the
        // verify arena, or DFlash2's staging) stopped with "do not fit" (8 GB NVIDIA, 128K, 2 requests: a 74-slot cache and 35 MiB left).  Price them
        // here instead, at the largest chunk that still leaves a cache (1024, 512, 256 tokens), and give the cache the rest: slower (a smaller
        // cache, smaller prompt chunks), never a refused start.  Not with a split, a cache or chunk given on the command line, or STRATA_SPILL_OFF=1.
        if (auto_cache && pf_borrow && o.prefill_auto && !o.vram_reserve_given && o.expert_cache < min_slots && !profile.empty() &&
            std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu()) {
            constexpr int64_t kSpillFloorMib = 100;   // the verify arena, the head check and the cuBLAS workspace beside the late allocations
            int64_t pick_chunk = 0, pick_slots = 0, pick_own = 0, pick_late = 0;
            for (const int64_t c : {(int64_t) 1024, (int64_t) 512, (int64_t) 256}) {
                const int64_t own = ((int64_t) strata::prefill::Prefill::bytes_needed_owned(g, ss, c, srcp != nullptr) >> 20) + 64;
                const int64_t late = strata::qwen_defaults_026() ? 0 : (int64_t) strata::core::kLateAllocMib;
                const int64_t room = (int64_t) free_b - mtp_bind - pipe_first - ((own + late + kSpillFloorMib) << 20);
                const int64_t s = room > 0 ? std::min<int64_t>(room / blob, (int64_t) profile.size()) : 0;
                if (s >= 64 || (s >= 1 && pick_slots < 64)) { pick_chunk = c; pick_slots = s; pick_own = own; pick_late = late; }
                if (s >= 64) break;
            }
            if (pick_slots >= 1) {
                std::fprintf(stderr, "strata generate: short of GPU memory: the cache cannot lend the prompt path its buffers, so the prompt path keeps its own "
                                     "(%lld MiB for %lld-token chunks) and the cache is %lld slots (%lld MiB): more is read from RAM and the SSD, expect it to be slower\n",
                             (long long) pick_own, (long long) pick_chunk, (long long) pick_slots, (long long) ((pick_slots * blob) >> 20));
                o.expert_cache = (int) pick_slots;
                o.vram_reserve_mib = (int) (pick_own + pick_late + kSpillFloorMib);
                o.no_prefill_borrow = true;
                pf_borrow = false;
                o.prefill_auto = false;
                o.prefill_chunk = pick_chunk;
                reserve_adapted = true;
            }
        }
        // SHORT OF VRAM: THE OUTPUT HEAD ON THE HOST, LAST.  When every step above left the cache no slot (4 requests at 128K on an 8 GB card: the sessions and the weights
        // leave 0.63 GiB after the head, and the cache needs 543 MiB beside a reserve), the head (0.41 GiB) moves to mapped pinned host memory and every logits row reads it
        // over PCIe: decode is about 3x slower (12.7 against 36.1 tok/s per user, 2 requests on a 12 GB mimic), but the start works.  Only a card that would otherwise
        // refuse; STRATA_HEAD_HOST_OFF=1 keeps the refusal.
        if (o.expert_cache == 0 && native_head.loaded() && !native_head.on_host() && !o.vram_reserve_given && !multi_gpu && stages.empty() &&
            std::getenv("STRATA_HEAD_HOST_OFF") == nullptr && std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu()) {
            std::string herr;
            if (native_head.to_host(herr)) {
                free_b = strata::core::device_free_bytes();
                strata::core::AutoCacheSlots a2 = strata::core::auto_cache_slots((int64_t) free_b, reserve, blob, min_slots, strata::qwen_defaults_026() ? 0 : strata::core::kLateAllocMib);
                int64_t s4 = a2.slots;
                if (!profile.empty()) s4 = std::min<int64_t>(s4, (int64_t) profile.size());
                if (s4 < min_slots && o.vram_reserve_mib > kSmallReserveMib) {   // the reserve lowered for the smaller cache, as above
                    const int64_t fit_mib = ((int64_t) free_b - mtp_bind - pipe_first - min_slots * blob) / (1 << 20) - prefill_mib;
                    if (fit_mib >= kSmallReserveMib) {
                        const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                        s4 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind + pipe_first)) / blob;
                        if (!profile.empty()) s4 = std::min<int64_t>(s4, (int64_t) profile.size());
                        o.vram_reserve_mib = r;
                        reserve_adapted = true;
                    }
                }
                o.expert_cache = (int) std::max<int64_t>(s4, 0);
                std::fprintf(stderr, "strata generate: short of GPU memory: the cache had no room, so the output head (%llu MiB) moves to pinned host memory and is read over PCIe for "
                                     "every logits row: decode is slower; the cache is now %d slots\n", (unsigned long long) (native_head.weight_bytes() >> 20), o.expert_cache);
            } else {
                std::fprintf(stderr, "strata generate: short of GPU memory: the output head stays in VRAM (%s)\n", herr.empty() ? "type or build not movable" : herr.c_str());
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = (((int64_t) at_reserve + prefill_mib) << 20) + mtp_bind + pipe_first + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib =
                (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            std::fprintf(stderr, "strata generate: no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, the English draft subset (setup --draft-vocab en; the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU\n",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str());
        }
    } else if (multi_gpu && o.expert_cache > 0) {
        // an explicit cache size leaves room for the prompt path's buffers and the reserve, or the first prompt
        // fails with "device buffers ... do not fit" (with borrowing - the default with a profile - the path lends
        // slots instead and `prefill_mib` is 0, so only the reserve is checked)
        size_t free_b = strata::core::device_free_bytes();
        const int64_t prefill_mib = owned_prefill_mib();
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + pipe_first;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) nibble_max_slot_bytes(), 0);
        if (o.expert_cache > fit) {
            // a WARNING that names the knob: the user asked for this size, and gets fewer slots
            std::fprintf(stderr, "strata generate: WARNING: layer split: --expert-cache %d leaves no room for the "
                                 "prompt path's buffers (%lld MiB) and the %d MiB reserve on CUDA0: %lld slots instead "
                                 "(a smaller --vram-reserve-mib leaves more of them)\n", o.expert_cache,
                         (long long) prefill_mib, o.vram_reserve_mib, (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so the same VRAM holds more experts than slots of
    // the largest blob would. The shared cache below sizes slots in profile order.
    // #369: that list does not match --expert-cache-per-layer. A range is layer * quota + n, so a size cut for
    // one layer can land in another layer's slot. Every expert in a layer is one size, so the per-layer list is
    // `quota` copies of that layer's own blob, in layer order. `--expert-cache N` stays the budget of N largest
    // blobs, and the quota is however many copies of every layer fit in it.
    std::vector<int64_t> sized_slots;
    uint64_t per_layer_bytes = 0;
    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {
        const size_t free_b = strata::core::device_free_bytes();   // the same reading the shared-cache sizing uses
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int asked = o.expert_cache;
        const uint64_t budget = (uint64_t) asked * nibble_max_slot_bytes();
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        uint64_t sum = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            sum += (uint64_t) strata::core::ExpertCache::slot_stride((int64_t) nib::slot_bytes(l), slot_guard);
        int64_t q = sum > 0 ? (int64_t) (cap / sum) : 0;
        if (q > g.n_expert) q = g.n_expert;
        if (q > 0) {
            per_layer_bytes = sum;
            sized_slots.reserve((size_t) q * (size_t) g.n_layers);
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const int64_t b = (int64_t) nib::slot_bytes(l);
                for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
            }
            o.expert_cache = (int) sized_slots.size();
            std::fprintf(stderr, "strata generate: per-layer slots use each layer's blob: %d uniform slots "
                                 "(%.2f GiB) -> %d slots, %lld per layer (%.2f GiB)\n",
                         asked, (double) budget / 1073741824.0, o.expert_cache, (long long) q,
                         (double) ((uint64_t) q * per_layer_bytes) / 1073741824.0);
        }
    } else if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = strata::core::device_free_bytes();
        const auto& lay = strata::kernels::cpu::expert_layout();
        // what the uniform sizing granted, in the 256-byte slots the loop below counts: the blob formats of 1280 / 2560
        // rows are whole multiples of 256 already; an NVFP4 blob's 16-byte scale tail is not, and the unrounded budget
        // held one slot fewer than asked (--expert-cache 1: none)
        const uint64_t budget = (uint64_t) o.expert_cache * ((nibble_max_slot_bytes() + 255) / 256 * 256);
        uint64_t used = 0;
        const size_t keep_free = ((size_t) o.vram_reserve_mib << 20) + (size_t) pipe_first;
        size_t free_room = free_b > keep_free ? free_b - keep_free : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (uint64_t) strata::core::ExpertCache::slot_stride((int64_t) nib::slot_bytes(pr.first), slot_guard);
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) nib::slot_bytes(pr.first));
        }
        // #831: the sized-slots loop walks the profile's pairs, so the profile's length is the ceiling; an explicit
        // --expert-cache above it used to be cut with no word.  A warning only (what is allocated does not change).
        if (!auto_cache && o.expert_cache > 0 && sized_slots.size() == profile.size() && (size_t) o.expert_cache > profile.size())
            std::fprintf(stderr, "strata generate: WARNING: --expert-cache %d is more than the %zu pairs of the profile %s: "
                                 "%zu slots (the profile is the ceiling on a native pack; a profile that ranks every "
                                 "(layer, expert) pair lifts it - tools/make_profile.py)\n", o.expert_cache,
                         profile.size(), o.expert_profile.c_str(), profile.size());
        o.expert_cache = (int) sized_slots.size();
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL) && !defined(_WIN32)
        // ---- AUTO NO-BORROW (0.2.10).  With batch slots on one NVIDIA card the prompt path lent cache slots to its own
        // buffers, and every lent slot's expert was read again from the host during each chunk.  When the card holds EVERY
        // expert of the profile and still has room for the path's own buffers (the chunk the loan would have used, 8192)
        // plus kNoBorrowKeepMib, the loan only costs time: the buffers are allocated instead, the chunk stays what it was, and
        // the arithmetic is the same (QUAL-PAR2: 12/12 identical, KL 0).  A card that cannot hold all experts keeps the loan -
        // its cache is worth more than the buffers (a 16 GB A4000 lost 2.5x on a long prompt and ran out of memory with them).
        // Not with an explicit --prefill N, a split, helper caches, STRATA_PREFILL_BORROW=1 or STRATA_AUTO_NOBORROW=0.
        if (auto_cache && pf_borrow && o.prefill_auto && o.batch >= 2 && !multi_gpu && stages.empty() && !profile.empty() &&
            sized_slots.size() == profile.size() && std::none_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](int n) { return n > 0; })) {
            const char* e1 = std::getenv("STRATA_AUTO_NOBORROW");
            const char* e2 = std::getenv("STRATA_PREFILL_BORROW");
            if (!(e1 != nullptr && e1[0] == '0') && !(e2 != nullptr && e2[0] == '1')) {
                constexpr int64_t kNoBorrowChunk = 8192, kNoBorrowKeepMib = 3072;
                uint64_t used_b = 0;
                for (const int64_t sb : sized_slots) used_b += (uint64_t) strata::core::ExpertCache::slot_stride(sb, slot_guard);
                const int64_t chunk = std::min<int64_t>(kNoBorrowChunk, std::max<int64_t>(256, o.prefill_auto_max));
                const int64_t own_mib = ((int64_t) strata::prefill::Prefill::bytes_needed_owned(g, ss, chunk, srcp != nullptr) >> 20) + 64;
                const size_t fb = strata::core::device_free_bytes();
                const int64_t room_mib = fb > used_b ? (int64_t) ((fb - used_b) >> 20) - o.vram_reserve_mib - (int64_t) (pipe_first >> 20) : 0;
                const bool fits = room_mib >= own_mib + kNoBorrowKeepMib;
                std::fprintf(stderr, "strata generate: every expert fits the cache; the prompt path's own buffers (%lld MiB for a %lld-token chunk) %s "
                                     "(%lld MiB left beside the %d MiB reserve, %lld MiB kept free)%s\n",
                             (long long) own_mib, (long long) chunk, fits ? "fit" : "do not fit", (long long) room_mib, o.vram_reserve_mib,
                             (long long) kNoBorrowKeepMib, fits ? ": no loan of cache slots" : ": the loan stays");
                if (fits) {
                    o.no_prefill_borrow = true;
                    pf_borrow = false;
                    o.prefill_auto = false;
                    o.prefill_chunk = chunk;
                }
            }
        }
#endif
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL) && !defined(_WIN32)
    // 0.2.10.1 (round0212 item 7): the FP16 weight cache is on when the engine runs batch slots on Linux CUDA.  Bit-exact against the uncached
    // form (gemm_w16_parity, greedy identity 12/12, first-token KL 0); +3.8% (c=1) .. +6.6% (c=8) total tokens/s on l620.  It takes only VRAM beyond
    // 3 GiB kept free; STRATA_PF_W16_CACHE_MIB=0 turns it off, =N is an explicit budget.
    if (o.batch >= 2) strata::prefill::Gemm::set_w16_auto_default(true);
#endif
    // #533: --vram-elastic: the cache in physical segments (the VRAM command resizes it between requests).  One GPU,
    // no helper caches, serve mode: anything else keeps the one cudaMalloc, said once.
    if (o.vram_elastic) {
        const char* why = !o.serve ? "it works between requests of --serve"
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : remote_caches ? "the helper caches on other GPUs"
                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
#if defined(STRATA_USE_HIP)
        if (why == nullptr) why = "it is NVIDIA-only for now";
#endif
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: --vram-elastic is off: %s\n", why);
            o.vram_elastic = false;
        } else {
            xcache.set_segment_bytes(o.vram_segment_mib << 20);
        }
    }
    // STRATA_MMQ_DIRECT: the guard after each sized slot (kept across the retry loop's close/open)
    xcache.set_slot_guard(slot_guard);
    if (slot_guard > 0 && o.expert_cache > 0) {
        if (!sized_slots.empty())
            std::fprintf(stderr, "strata generate: +%lld B guard per sized slot (STRATA_MMQ_DIRECT)\n", (long long) slot_guard);
        else   // xcache.open below: every slot is the largest blob's size, and no guard is added
            std::fprintf(stderr, "strata generate: uniform expert slots, no guard (STRATA_MMQ_DIRECT): an expert is read "
                                 "in its slot where the slot's slack covers the over-read\n");
    }
    if (o.expert_cache > 0 && open_expert_pages) {
        if (!open_expert_pages(o.expert_cache, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
    } else if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (per_layer_bytes > 0) {
                // Cutting the list short would put one layer's blob in the next layer's range.
                const auto& lay = strata::kernels::cpu::expert_layout();
                int64_t q = keep_bytes / (int64_t) per_layer_bytes;
                if (q > g.n_expert) q = g.n_expert;
                sized_slots.clear();
                if (q <= 0) { o.expert_cache = 0; return false; }
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    const int64_t b = (int64_t) nib::slot_bytes(l);
                    for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
                }
                o.expert_cache = (int) sized_slots.size();
                return true;
            }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() &&
                       used + strata::core::ExpertCache::slot_stride(sized_slots[keep], slot_guard) <= keep_bytes)
                    used += strata::core::ExpertCache::slot_stride(sized_slots[keep++], slot_guard);
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += strata::core::ExpertCache::slot_stride(s, slot_guard);
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
                // so with a small page file the cache's one big cudaMalloc fails while the VRAM is free.  An auto
                // cache then tries three quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
#if defined(_WIN32)
                MEMORYSTATUSEX ms{};
                ms.dwLength = sizeof ms;
                if (GlobalMemoryStatusEx(&ms))
                    std::snprintf(commit, sizeof commit, " (Windows has %.1f GiB of commit left: RAM + page file)",
                                  (double) ms.ullAvailPageFile / 1073741824.0);
#endif
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s%s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
#if defined(_WIN32)
                std::fprintf(stderr, "strata generate: on Windows the graphics card's memory also needs room in the page "
                                     "file: set it to \"System managed\" (System > About > Advanced system settings > "
                                     "Performance > Advanced > Virtual memory), or lower --expert-cache\n");
#endif
                return 1;
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = strata::core::device_free_bytes();
            const int64_t want = ((int64_t) o.vram_reserve_mib << 20) + pipe_first;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (under WDDM the free figure read before the allocation runs ~0.7 GiB high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.  (With open_expert_pages, strata-q38's, the pools are filled after READY
    // instead, by PagedEngine::fill_experts; so are a layer split's stages with open_stage_pages, below.)
    int64_t prefilled = 0;
    nib_st.reserve(stages.size());   // (a stage's fill holds a reference to its entry)
    if (!open_expert_pages && !profile.empty() && srcp != nullptr) {
        // #369 (dag08): per layer, a full layer skips only its own pairs - each layer takes its hottest experts until
        // its range is full (one full layer used to end the whole fill, leaving most layers empty)
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        // #286: an unbuffered file tier reads the pairs in batches of 64, the next batch while this one is copied
        std::future<void> ahead;
        auto read_batch = [&](int64_t at) { src.prefetch_pairs(profile.data() + at, std::min<int64_t>(64, want - at)); };
        // mapped reads: advise the next `fill_ahead` pairs so their reads overlap; STRATA_FILL_AHEAD=0 turns it off
        int64_t fill_ahead = 0;
        if (!per_layer && srcp == &src && !src.unbuffered()) {
            const char* v = std::getenv("STRATA_FILL_AHEAD");
            fill_ahead = v == nullptr ? 256 : std::max(0, std::atoi(v));
            if (fill_ahead > 0 && !src.advise_pairs(profile.data(), std::min<int64_t>(fill_ahead, want))) fill_ahead = 0;
            std::fprintf(stderr, "strata generate: the profile fill asks for %lld pairs ahead (STRATA_FILL_AHEAD)\n",
                         (long long) fill_ahead);
        }
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        if (nib::any_on()) {   // STRATA_EXPERT_NIBBLE: the converted layers' pairs go through the 4-bit fill instead
            int64_t conv = 0;
            for (int64_t i = 0; i < want; ++i) conv += nib::layer(profile[(size_t) i].first).on ? 1 : 0;
            if (conv > 0 && !nib_x.begin(xcache, conv, err)) {
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: %s\n", err.c_str());
                return 1;
            }
        }
        for (int64_t i = 0; i < want; ++i) {
            if (!per_layer && srcp == &src && src.unbuffered() && i % 64 == 0) {
                if (ahead.valid()) ahead.get();
                else read_batch(i);
                if (i + 64 < want) ahead = std::async(std::launch::async, [&read_batch](int64_t b) { strata::aux_cpus::pin_current_thread(); read_batch(b); }, i + 64);
            }
            if (fill_ahead > 0 && i + fill_ahead < want) (void) src.advise_pairs(profile.data() + i + fill_ahead, 1);
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            const uint8_t* b = srcp->blob_stable(profile[(size_t) i].first, profile[(size_t) i].second);
            const bool nib_pair = nib_x.active() && nib::layer(profile[(size_t) i].first).on;   // STRATA_EXPERT_NIBBLE
            if (b == nullptr || !(nib_pair ? nib_x.add(profile[(size_t) i].first, profile[(size_t) i].second, slot, b, err)
                                           : xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first)))) {
                std::fprintf(stderr, "strata generate: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            fill_bytes += strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            ++prefilled;
        }
        if (nib_x.active() && !nib_x.finish(err)) {
            std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA0: %s\n", err.c_str());
            return 1;
        }
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        // (STRATA_EXPERT_NIBBLE: a converted slot holds 4-bit codes, not the blob's bytes - the 4-bit check below is its check)
        if (prefilled > 0 && !nib::layer(profile[0].first).on &&
            !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob_stable(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (nib_x.active() && nib_x.n > 0) {
            // STRATA_EXPERT_NIBBLE: the hottest converted pair read back (its hash, its down rows); every item and every
            // rebuilt GGUF block of every converted slot was already checked on the device
            size_t c = 0;
            while (c < (size_t) want && !nib::layer(profile[c].first).on) ++c;
            if (!nib_x.check(profile[c].first, profile[c].second, srcp->blob_stable(profile[c].first, profile[c].second),
                             err)) {
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA0: %s\n", err.c_str());
                return 1;
            }
            std::set<int32_t> ls;
            for (const auto& pr : nib_x.pairs) ls.insert(pr.first);
            std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA0: %lld slots of %zu layers transcoded to 4-bit "
                                 "codes (%.2f s; every item round-tripped, every slot's GGUF blocks rebuilt and compared, "
                                 "slot hashes recorded); layer %d expert %d read back\n", (long long) nib_x.n, ls.size(),
                         nib_x.seconds, (int) profile[c].first, (int) profile[c].second);
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified\n",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
        if (std::getenv("STRATA_VERIFY_COMPLEMENT") != nullptr) {
            // tests: the filled slots' checksum (each slot's blob read back and hashed, then the hashes in slot order)
            std::vector<std::pair<int32_t, int64_t>> sl;   // (slot, blob bytes)
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (const int32_t s = xcache.slot_of(l, e); s >= 0)
                        sl.emplace_back(s, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
            std::sort(sl.begin(), sl.end());
            std::vector<uint64_t> sh(sl.size(), 0);
            std::atomic<size_t> next_slot{0};
            std::atomic<bool> copy_ok{true};
            auto hash_work = [&] {
                std::vector<uint8_t> hb;
                for (size_t i; (i = next_slot.fetch_add(1)) < sl.size();) {
                    hb.resize((size_t) sl[i].second);
                    if (cudaMemcpy(hb.data(), xcache.device_slot(sl[i].first), hb.size(), cudaMemcpyDeviceToHost) !=
                        cudaSuccess) copy_ok = false;
                    sh[i] = strata::core::fnv1a64(hb.data(), (uint64_t) hb.size());
                }
            };
            std::vector<std::thread> hth;
            for (int t = 1; t < 8; ++t) hth.emplace_back(hash_work);
            hash_work();
            for (auto& t : hth) t.join();
            uint64_t h = strata::core::fnv1a64(nullptr, 0);
            for (size_t i = 0; i < sl.size(); ++i) {   // field by field: a pair's padding is not part of it
                const int64_t v[2] = {(int64_t) sl[i].first, sl[i].second};
                h = strata::core::fnv1a64((const uint8_t*) v, sizeof v, h);
                h = strata::core::fnv1a64((const uint8_t*) &sh[i], sizeof sh[i], h);
            }
            std::fprintf(stderr, "strata generate: expert cache checksum %016llx (%zu slots%s)\n", (unsigned long long) h,
                         sl.size(), copy_ok ? "" : ", A READ-BACK FAILED");
        }
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now (they load above, before this), so what is left
        // to hold back is the windows - and `free_b` has already lost the drafter.
        int64_t room = open_stage_pages ? 0 : stage_room(st.dev, true, false);
        int64_t df2_after = 0;   // what DFlash2 binds on this card after its cache
#if defined(STRATA_HAVE_DFLASH2)
        // DFlash2's own draft head and logits (and, in serve, the prompt staging) are bound on its card AFTER the caches, as the single card prices them (mtp_bind above): without
        // this term the last stage ended its load with 130-400 MiB less than the plan kept (the serve check's LOW at 8 and 12 GB).  Not under STRATA_DEFAULTS=026.
        if (use_df2 && &st == last_st && !df2_first && !strata::qwen_defaults_026()) {
            df2_after = get_df2_head_bytes() + (48 << 20) + (o.serve ? (112 << 20) : 0);
            room = std::max<int64_t>(room - df2_after, 0);
        }
#endif
        // THE LAYER SPLIT'S SPILL WALK, STEP 5: THE OUTPUT HEAD ON THE HOST, LAST.  The card that holds the head (the last stage) has no room for a working cache beside it: the head
        // (0.41 GiB) moves to mapped pinned host memory and every logits row reads it over PCIe (decode is slower, the start works).  Only a card that would otherwise refuse.
        if (!open_stage_pages && &st == last_st && st.head.loaded() && !st.head.on_host() && !o.vram_reserve_given &&
            std::getenv("STRATA_HEAD_HOST_OFF") == nullptr && std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu() &&
            room < strata::program::split_walk::kWorkingCacheSlots * (int64_t) lay.max_blob) {
            std::string herr;
            const uint64_t head_mib = st.head.weight_bytes() >> 20;
            if (st.head.to_host(herr)) {
                room = std::max<int64_t>(stage_room(st.dev, true, false) - df2_after, 0);
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d: the cache would have no room, so the output head (%llu MiB) moves to pinned host memory and is read over "
                                     "PCIe for every logits row: decode is slower; the room for experts is now %.2f GiB\n",
                             st.dev, (unsigned long long) head_mib, (double) room / 1073741824.0);
            } else {
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d: the output head stays in VRAM (%s)\n", st.dev,
                             herr.empty() ? "type or build not movable" : herr.c_str());
            }
        }
        const strata::core::OnDevice on(st.dev);
        if (!open_stage_pages) {
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %.2f GiB free of %.2f, room for experts %.2f GiB\n",
                         st.dev, (double) fb / 1073741824.0, (double) tb / 1073741824.0, (double) room / 1073741824.0);
        }
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = native_pack ? strata::core::ExpertCache::slot_stride((int64_t) nib::slot_bytes(pr.first), slot_guard)
                                          : (int64_t) lay.max_blob;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) nib::slot_bytes(pr.first));
        }
        // #841: a stage's cache can fail to open with the VRAM free (the card's allocation is also charged to the
        // Windows commit, or the free figure ran high): it is tried again with 90% of the slots, up to 3 times, each
        // try said.  Nothing changes when the first open works.
        st.cache.set_slot_guard(slot_guard);   // STRATA_MMQ_DIRECT: the stages' prompt MMQ reads their own slots
        // (with open_stage_pages `sized` is empty: the hook opened the cache, or `err` says why not)
        bool stage_open = open_stage_pages && open_stage_pages(st, err);
        for (int attempt = 0; !sized.empty(); ++attempt) {
            stage_open = native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
                                     : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err);
            if (stage_open || attempt >= 3) break;
            (void) cudaGetLastError();
            const size_t keep = sized.size() * 9 / 10;
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s; trying %zu of %zu slots "
                                 "(try %d of 3)\n", st.dev, err.c_str(), keep, sized.size(), attempt + 1);
            if (keep == 0) break;
            st.cache.close();
            sized.resize(keep);
        }
        if (!stage_open) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         sized.empty() && !open_stage_pages ? "no room" : err.c_str());
            return 1;
        }
        if (!open_stage_pages) {
            int64_t filled = 0;
            strata::core::NibbleCache& nib_c = nib_st.emplace_back();   // STRATA_EXPERT_NIBBLE: this stage's converted slots
            if (nib::any_on()) {
                int64_t conv = 0;
                for (const auto& pr : st.profile) conv += nib::layer(pr.first).on ? 1 : 0;
                if (conv > 0 && !nib_c.begin(st.cache, conv, err)) {
                    std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA%d: %s\n", st.dev, err.c_str());
                    return 1;
                }
            }
            for (const auto& pr : st.profile) {
                if (filled >= st.cache.slots()) break;
                const int32_t slot = st.cache.admit(pr.first, pr.second);
                if (slot == strata::core::kNotResident) break;
                const uint8_t* b = srcp->blob_stable(pr.first, pr.second);
                const bool nib_pair = nib_c.active() && nib::layer(pr.first).on;   // STRATA_EXPERT_NIBBLE
                if (b == nullptr || !(nib_pair ? nib_c.add(pr.first, pr.second, slot, b, err)
                                               : st.cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first)))) {
                    std::fprintf(stderr, "strata generate: layer split, CUDA%d profile fill failed at pair %lld: %s\n",
                                 st.dev, (long long) filled, err.c_str());
                    return 1;
                }
                ++filled;
            }
            if (nib_c.active() && !nib_c.finish(err)) {
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA%d: %s\n", st.dev, err.c_str());
                return 1;
            }
            // (STRATA_EXPERT_NIBBLE: a converted slot holds 4-bit codes, not the blob's bytes - the 4-bit check below is its check)
            if (filled == 0 || (!nib::layer(st.profile[0].first).on &&
                                !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                      srcp->blob_stable(st.profile[0].first, st.profile[0].second), err,
                                                      (int64_t) lay.blob_bytes(st.profile[0].first)))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                             filled == 0 ? "nothing filled" : err.c_str());
                return 1;
            }
            if (nib_c.active() && nib_c.n > 0) {   // STRATA_EXPERT_NIBBLE: as CUDA0's above
                size_t c = 0;
                while (c < st.profile.size() && !nib::layer(st.profile[c].first).on) ++c;
                const auto& pc = st.profile[c];
                if (!nib_c.check(pc.first, pc.second, srcp->blob_stable(pc.first, pc.second), err)) {
                    std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA%d: %s\n", st.dev, err.c_str());
                    return 1;
                }
                std::set<int32_t> ls;
                for (const auto& pr : nib_c.pairs) ls.insert(pr.first);
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: CUDA%d: %lld slots of %zu layers transcoded to "
                                     "4-bit codes (%.2f s; every item round-tripped, every slot's GGUF blocks rebuilt and "
                                     "compared, slot hashes recorded); layer %d expert %d read back\n", st.dev,
                             (long long) nib_c.n, ls.size(), nib_c.seconds, (int) pc.first, (int) pc.second);
            }
            std::fprintf(stderr, "strata generate: layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                                 "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified\n",
                         st.dev, (long long) st.lb, (long long) (st.le - 1),
                         (long long) st.cache.slots(), st.cache.gib(),
                         (long long) filled, st.profile.size());
        }
    }
    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));
    // THE LAYER SPLIT'S SPILL WALK, STEP 4: THE LOAN KEEPS 16.  A stage whose cache is too small to lend the prompt path a 256-token chunk and keep 128 slots behind makes the path
    // allocate its own buffers on top (more than the reserve): the same short card as the single card's, which keeps 16 slots instead (the cache is smaller, prompts are read in
    // smaller chunks, the start works).  Never on a card with room: it only fires when a cache is below the usual loan.
    if (multi_gpu && !stages.empty() && g_lend_keep == 128 && o.prefill_chunk > 0 && pf_borrow && o.prefill_auto && !o.vram_reserve_given &&
        std::getenv("STRATA_SPILL_OFF") == nullptr && spill_discrete_gpu()) {
        for (const auto& stp : stages) {
            const int64_t blob_s = (int64_t) nibble_max_slot_bytes();
            const int64_t need_s = ((int64_t) strata::prefill::Prefill::bytes_needed(g, stp->ss, 256) + blob_s - 1) / blob_s + g_lend_keep;
            if ((int64_t) stp->cache.slots() < need_s) {
                std::fprintf(stderr, "strata generate: short of GPU memory: CUDA%d: the cache (%lld slots) is too small to lend the prompt path 256-token buffers and keep %lld more: "
                                     "the loan keeps 16, prompts are read in smaller chunks, expect them to be slower\n",
                             stp->dev, (long long) stp->cache.slots(), (long long) g_lend_keep);
                g_lend_keep = 16;
                break;
            }
        }
    }
    if (nib::any_on()) {
        // STRATA_EXPERT_NIBBLE: every expert of every layer must be in VRAM, in the cache of the card that runs the layer.
        // A converted layer's miss would be computed from the GGUF blob by the CPU or over PCIe, and a non-converted
        // layer's residency would differ from the switch-off run's (its smaller slots leave room for more), so the
        // replies' bits would too.  Then no slot is written again.
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const strata::core::ExpertCache* owner = &xcache;
            int owner_dev = 0;
            if (multi_gpu && !split_at.empty() && l >= split_at[0]) {
                owner = nullptr;
                for (const auto& stp : stages)
                    if (l >= stp->lb && l < stp->le) { owner = &stp->cache; owner_dev = stp->dev; }
            }
            int64_t missing = 0, first = -1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (owner == nullptr || owner->slot_of(l, e) == strata::core::kNotResident) {
                    if (first < 0) first = e;
                    ++missing;
                }
            if (missing > 0) {
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE refuses the start: layer %lld has %lld of its %lld "
                                     "experts out of CUDA%d's VRAM (expert %lld the first) - every expert must be resident "
                                     "(a profile that ranks every pair and room on the card; STRATA_EXPERT_NIBBLE_SKIP="
                                     "<layers> keeps some layers' smaller GGUF slots)\n", (long long) l, (long long) missing,
                             (long long) g.n_expert, owner_dev, (long long) first);
                return 1;
            }
        }
        xcache.seal();
        for (auto& stp : stages) stp->cache.seal();
        std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE: all %lld experts of all %lld layers are VRAM-resident; "
                             "the expert caches are sealed (nothing writes a slot from here on)\n",
                     (long long) g.n_expert, (long long) g.n_layers);
    }

    // ---- LAYER 2: exact startability probe on the live free figures.  Every session, cache and batch slot is
    // carved by now, so `cudaMemGetInfo` reads what serve's own_fits (5855-5869) will read.  Run serve's own
    // predicates on the chosen split: every participant must lend a 512-token chunk, or afford its own buffers
    // with the 512 MiB headroom.  Serve's live checks stay as the guard; this is the pre-flight confirmation.
    // STRATA_LAYER_SPLIT_PROBE=0 skips it.
    static const bool split_probe = [] {
        const char* v = std::getenv("STRATA_LAYER_SPLIT_PROBE");
        return v == nullptr || v[0] != '0';
    }();
    if (multi_gpu && split_auto && split_probe && !split_at.empty()) {
        const int ns = (int) stages.size() + 1;
        const int64_t kPct = get_auto_lend_pct();
        const int64_t c = 512;
        bool probe_ok = true;
        int probe_dev = 0;
        int64_t probe_need = 0, probe_fb = 0, probe_slots = 0, probe_est = 0;
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            const strata::core::OnDevice on(dev);
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            const strata::core::ExpertCache& xc = i == 0 ? xcache : stages[(size_t) i - 1]->cache;
            const uint64_t cn = strata::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[(size_t) i - 1]->ss, c);
            const int64_t slots = xc.slots();
            const uint64_t avg = slots > 0 ? xc.bytes() / (uint64_t) slots
                                           : (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
            const int64_t slots_est = (int64_t) (((cn + 255) / 256 * 256 + avg - 1) / avg);
            const bool lends = slots > 0 && slots_est > 0 && slots_est + g_lend_keep <= slots && slots_est * 100 <= kPct * slots;
            if (lends) continue;
            if ((int64_t) cn + (512ll << 20) > (int64_t) fb) {
                probe_ok = false; probe_dev = dev; probe_need = (int64_t) cn; probe_fb = (int64_t) fb;
                probe_slots = slots; probe_est = slots_est;
                break;
            }
        }
        if (probe_ok) {
            std::fprintf(stderr, "strata generate: layer split auto: L2 probe: the split is startable on the live free figures\n");
        } else {
            std::fprintf(stderr, "strata generate: layer split auto: L2 probe: CUDA%d cannot start the prompt path: "
                                 "a 512-token chunk needs %lld MiB of its own buffers (+512 MiB headroom) with %lld MiB "
                                 "free, and its %lld-slot cache cannot lend it (needs %lld slots)\n",
                         probe_dev, (long long) (probe_need >> 20), (long long) (probe_fb >> 20),
                         (long long) probe_slots, (long long) probe_est);
            std::fprintf(stderr, "strata generate: to make room: a smaller --max-context, --prefill 512, a smaller "
                                 "--vram-reserve-mib, an explicit --layer-split, or close other programs using the GPU\n");
        }
    }

    if (o.remote_expert_opt && o.expert_cache_remote[0] > 0) remote_opt = std::make_unique<strata::core::RemoteExpertOpt>();
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            std::fprintf(stderr, "strata generate: remote experts need --expert-profile, "
                                 "a CUDA0 expert cache and the expert pool\n");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote || std::any_of(o.expert_cache_remote_auto.begin(), o.expert_cache_remote_auto.end(),
                                        [](bool automatic) { return automatic; })) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            std::fprintf(stderr, "strata generate: remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3\n", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            std::fprintf(stderr, "strata generate: remote ranks %s across %d CUDA devices\n",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (remote_opt) remote_opt->attach(remote_experts[(size_t) r]);
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err,
                     o.expert_cache_remote_auto[(size_t) r])) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows\n", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    // ---- Multi-GPU: the second GPU's expert tier, filled with the ranked pairs the primary does not hold
    if (o.peer_device >= 1) {
        if (profile.empty() || srcp == nullptr || o.expert_cache <= 0) {
            std::fprintf(stderr, "strata generate: --peer-device needs --expert-profile and the expert cache\n");
            return 1;
        }
        const auto tp0 = Clock::now();
        if (!peer.open(o.peer_device, profile, xcache, *srcp, g.n_layers, g.n_expert, o.peer_reserve_mib, o.peer_slots,
                       err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: peer GPU %d: %lld experts, %.2f GiB (filled in %.1f s); with the primary's "
                             "%lld that is %lld of %lld on the GPUs\n", o.peer_device, (long long) peer.resident(), peer.gib(),
                     std::chrono::duration<double>(Clock::now() - tp0).count(), (long long) xcache.slots(),
                     (long long) (peer.resident() + xcache.slots()), (long long) (g.n_layers * g.n_expert));
    }

    if (remote_opt && !remote_opt->init(err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.peer = peer.valid() ? &peer : nullptr;
    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // P1.8 (PR 4972, re-checked as residency changes): when every routed expert sits in the GPU caches the decode reads nothing
    // from the file tier, and its Linux defaults (io prefetch, the decode gate) and the routing-aware prefetch only cost time.
    // Measured on 2x W7900D (gfx1100), IQ3_S experts.bin, a layer split holding 24576 of 24576: decode 57 -> 80 tok/s, the
    // output byte-identical.  Only what the defaults switched on is touched: STRATA_IO_PREFETCH=1 keeps io prefetch and the
    // look-ahead as before, STRATA_LOOKAHEAD=1 keeps the look-ahead.  The check runs here, again at every request's start and
    // after a VRAM resize (a smaller cache, a loan that was not refilled, an adaptive swap), so a later eviction turns them
    // back on.  STRATA_FILE_TIER_IDLE=0 turns the whole policy off.
    file_tier_idle_ok = [] { const char* v = std::getenv("STRATA_FILE_TIER_IDLE"); return v == nullptr || std::atoi(v) != 0; }();
    if (srcp == &src && file_tier_idle_ok) {
        const int64_t gpu_held_now = count_gpu_held();
        gpu_holds_all = gpu_held_now == g.n_layers * g.n_expert;
        if (gpu_holds_all && io_pf_by_default) {
            src.set_io_prefetch(false, 0);
            io_pf_suspended = true;
            const char* sv = std::getenv("STRATA_IO_STATS");
            src.set_io_stats(sv != nullptr && std::atoi(sv) != 0);
            std::fprintf(stderr, "strata generate: every expert is in the GPU caches (%lld of %lld): the file tier's Linux defaults "
                                 "(io prefetch, the decode gate) are off, the decode reads nothing from it (STRATA_IO_PREFETCH=1 keeps them)\n",
                         (long long) gpu_held_now, (long long) (g.n_layers * g.n_expert));
        }
    }
    const char* la_env = std::getenv("STRATA_LOOKAHEAD");
    la_asked = (la_env != nullptr && la_env[0] != '\0' && std::atoi(la_env) != 0) ||
               [] { const char* v = std::getenv("STRATA_IO_PREFETCH"); return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }();
    // CS-T: routing-aware prefetch of the file tier (the GGUF in place): the next layer's router on this layer's MoE
    // input predicts its experts and their pages are warmed meanwhile.  It only warms pages; STRATA_LOOKAHEAD=0 is
    // the A/B arm, STRATA_LOOKAHEAD_K the experts per token (default 10).
    // #1348, one prefetch design: with the Foresight swap space on (STRATA_FS_SLOTS) the look-ahead also runs on the pinned-RAM
    // tiers, where there are no pages to warm, and feeds its predictions to the swap space (STRATA_FS_AHEAD=0: not)
    fs_ahead = srcp != nullptr && [] { const char* v = std::getenv("STRATA_FS_SLOTS"); return v != nullptr && std::atoi(v) > 0; }() &&
               [] { const char* v = std::getenv("STRATA_FS_AHEAD"); return v == nullptr || std::atoi(v) != 0; }();
    // STRATA_LOOKAHEAD_STATS=1 (measurement only): run the predictor on any tier and score it against the real routing
    la_stats = srcp != nullptr && [] { const char* v = std::getenv("STRATA_LOOKAHEAD_STATS"); return v != nullptr && std::atoi(v) != 0; }();
    if (srcp != nullptr && ((srcp == &src && src.warms()) || fs_ahead || la_stats) &&
        (la_env == nullptr || std::atoi(la_env) != 0)) {
        std::vector<std::vector<uint16_t>> routers((size_t) g.n_layers);
        bool ok = true;
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const strata::core::WeightRef* w = wt.find("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
            ok = w != nullptr && w->kind == strata::core::WeightKind::Bf16InF32 &&
                 w->bytes == (uint64_t) (g.n_expert * g.n_embd) * 2;
            if (!ok) break;
            routers[(size_t) l].resize((size_t) (g.n_expert * g.n_embd));
            ok = cudaMemcpy(routers[(size_t) l].data(), w->data, (size_t) w->bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        const char* kv = std::getenv("STRATA_LOOKAHEAD_K");
        if (ok && lookahead.start(std::move(routers), g.n_embd, g.n_expert, kv ? std::atoi(kv) : 10, srcp, err, fs_ahead || la_stats)) {
            drive.d.lookahead = &lookahead;
            if (la_stats) lookahead.enable_stats(g.n_layers);
            if (const char* dv = std::getenv("STRATA_IO_PREFETCH_DEPTH"); dv != nullptr && std::atoi(dv) > 0)
                lookahead.set_depth(std::atoi(dv));
            else if (srcp == &src && src.io_prefetch())
                lookahead.set_depth(2);
            la_started = true;
            if (srcp == &src && gpu_holds_all && !la_asked && !fs_ahead && !la_stats) {   // started, but not attached while nothing is read
                drive.d.lookahead = nullptr;
                std::fprintf(stderr, "strata generate: routing-aware prefetch off (every expert is in the GPU caches; STRATA_LOOKAHEAD=1 runs it)\n");
            } else
                std::fprintf(stderr, "strata generate: routing-aware prefetch on (the next layer's router%s)\n",
                             fs_ahead ? ", feeding the Foresight swap space" : ", file tier");
        } else {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata generate: routing-aware prefetch off (%s)\n",
                         ok ? err.c_str() : "the routers are not BF16 in the arena");
            err.clear();
        }
    }
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (strata::core::allocate_device(&hit_scratch, (size_t) sb,
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &d_hit_slot, (size_t) K * sizeof(int32_t),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &d_hit_dst, (size_t) K * sizeof(int32_t),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34,
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement on
            // every chunk, and the reason enabling the cache changed the tokens.
            strata::core::allocate_device((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &d_hit_out, (size_t) K * g.n_embd * 4,
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the R4 hit path could not allocate its device buffers\n");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = open_expert_pages ? xcache.slot_bytes()
                                               : (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the hit path could not create its probe event\n");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // ---- P0.S8: the routing trace.  Only meaningful with the pool running, because the ids arrive through
    // the doorbell that the pool consumes - so `--no-pool` is refused rather than silently producing an empty
    // file that would read as "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently
    // dropped, so a captured graph cannot carry these events and the numbers would be zeros that read as
    // "every stage is free".  Refusing is the fix.
    if (o.stage_timing) {
        if (!o.no_capture) {
            std::fprintf(stderr, "strata generate: --stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.\n");
            return 2;
        }
        if (!strata::core::stage_timing_enable()) {
            std::fprintf(stderr, "strata generate: stage_timing_enable failed\n");
            return 1;
        }
        strata::core::stage_timing_name(0, "gr_read (attn)");
        strata::core::stage_timing_name(1, "attention block");
        strata::core::stage_timing_name(2, "gr_write (attn)");
        strata::core::stage_timing_name(3, "gr_read (ffn)");
        strata::core::stage_timing_name(4, "moe_route");
        strata::core::stage_timing_name(5, "moe_finish");
        strata::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals.  It is 36 of the 48 layers, 0.96 ms each, and its entire weight traffic
        // is ~26 MB - so ~0.11 ms at the measured read rate.  ~13 tiny latency-bound launches live in it and
        // a single "attention block" number cannot say which one costs anything.
        strata::core::stage_timing_name(8, "  gdn: quantize x");
        strata::core::stage_timing_name(9, "  gdn: qkv gemv");
        strata::core::stage_timing_name(10, "  gdn: conv+silu");
        strata::core::stage_timing_name(11, "  gdn: l2 norms");
        strata::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        strata::core::stage_timing_name(13, "  gdn: gdn_step");
        strata::core::stage_timing_name(14, "  gdn: z + out_norm");
        strata::core::stage_timing_name(15, "  gdn: out gemv");
    }
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            std::fprintf(stderr, "strata generate: --dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.\n");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    hit_fn = (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    pool_user = o.no_pool ? nullptr : (void*) &drive;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool.workers(),
                 pool.host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        // a layer split's CUDA0 session owns only [0, split_at[0]), so its graphs cover that range; the
        // whole-model replay paths (`session_loop`, the plain generate loop) refuse rather than read another
        // stage's state - a split runs its layers on the stages' verifiers (serve) or prefill stage chain
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/o.gpu_stages, 0,
                                           multi_gpu ? split_at[0] : -1)) {
            std::fprintf(stderr, "strata generate: session_capture: %s\n", err.c_str());
            return 1;
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        std::fprintf(stderr,
                     "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.\n");
        return 2;
    }
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.\n");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.\n");
        return 2;
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    return 0;
}

int Model::build_hit_tables(bool dumping_layers) {
    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph && !dumping_layers && half_dump == nullptr) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (strata::core::allocate_device((void**) &d_res, host_res.size() * sizeof(int32_t),
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            strata::core::allocate_device((void**) &d_hit_count, sizeof(int32_t),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            put_residency(d_res) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena (STRATA_ARENA_MMAP): the experts no GPU holds are the ones the CPU pool and the
        // prompt path will read - start reading them now instead of faulting them in 4 KB at a time mid-request
        // ... and the ones a GPU holds are handed back first (STRATA_ARENA_RELEASE=0 keeps them): the fill read the
        // whole arena, and left mapped and referenced it crowds every other allocation into swap
        {
            static const bool keep = std::getenv("STRATA_ARENA_RELEASE") && std::getenv("STRATA_ARENA_RELEASE")[0] == '0';
            rss_probe("before the release");
            uint64_t released = 0;
            if (!keep)
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] != strata::core::kNotResident)
                        released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                std::fprintf(stderr, "strata generate: expert arena: %.2f GiB of VRAM-held experts handed back to the OS\n",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            rss_probe("after the release");
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == strata::core::kNotResident)
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            rss_probe("after the prefetch hints");
        }
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const strata::core::OnDevice on(st->dev);
            if (strata::core::allocate_device((void**) &st->d_res, host_res.size() * sizeof(int32_t),
                {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
                put_residency(st->d_res) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split: CUDA%d residency table failed\n", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && !dumping_layers && half_dump == nullptr &&
        (hit_fn == nullptr || thits.on()) && !native_pack && !multi_gpu) {   // a split's token graph cannot span stages
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, thits.on() ? &thits : nullptr)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }
    return 0;
}

cudaError_t Model::put_residency(int32_t* dst) const {
    // The residency table is read by kernels on non-blocking streams, which do not wait for the legacy stream a plain
    // cudaMemcpy runs on, and a pageable copy can return before its DMA has landed: each upload waits for its own copy
    // (the current device's legacy stream, stream 0 - nothing else; HIP has no cudaStreamLegacy name, #550).
    const cudaError_t e = cudaMemcpy(dst, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return e != cudaSuccess ? e : cudaStreamSynchronize(nullptr);
}

int Model::get_batch_max_t() const {
    // (--batch-mtp and STRATA_PIPED_MTP draft with the draft layer's kernels, built for the narrow window: their windows
    // stay at most kVerifyNarrowT rows; only the plain batch windows are wide)
    if (batch_mtp && !piped_mtp) return (int) strata::kernels::kVerifyNarrowT;
    const int plain = std::max(o.spec, std::min(o.batch, (int) strata::kernels::kVerifyMaxT));
    const int group_rows = (piped_mtp && o.batch_groups > 1 && o.batch > 0)
        ? std::min<int>(strata::kernels::kVerifyNarrowT, (o.batch / o.batch_groups) * (1 + pm_drafts)) : 0;
    return std::max(plain, group_rows);
}

void Model::unload_drafter() {
    // MtpDrafter gives its memory back only in its destructor, and a new one is the unloaded state
    const strata::core::OnDevice on(mtp.device());   // (its destructor frees on the current device)
    mtp.~MtpDrafter();
    new (&mtp) strata::core::MtpDrafter();
}

bool Model::load_drafter(std::string& err) {
    // with a layer split across GPUs the drafter reads the last stage's residual: it lives on that device, its K/V
    // (unless in pages) beside that stage's session
    GpuStage* const st = get_drafter_stage();
    const strata::core::OnDevice on(st != nullptr ? st->dev : -1);
    mtp.set_prompt_len((int64_t) o.tokens.size());
    mtp.set_hnorm_per_stream(o.mtp_hnorm_stream);   // --mtp-hnorm stream (opt-in), before any capture
    mtp.set_q4(o.mtp_q4 == "all" || o.mtp_q4 == "proj", o.mtp_q4 == "all" || o.mtp_q4 == "head");   // before load
    if (!o.mtp_draft_vocab.empty()) mtp.set_draft_vocab(o.mtp_draft_vocab);
    // --pipeline-windows 2: 8 rows (the chain runs on past the window in flight: its drafts, its bonus token and
    // the next window's drafts), and the round/step graphs carry the forcing kernel.  The draft layer's kernels are
    // built for the narrow window; strata-q38's --spec is its batch windows' room, which a HIP build makes wider.
    const int mtp_t = o.pipeline_windows >= 2 ? strata::kernels::kVerifyNarrowT
                                              : std::min<int>(o.spec, strata::kernels::kVerifyNarrowT);
    if (o.pipeline_windows >= 2) mtp.set_force_capture(true);
    if (!mtp.load(o.mtp, draft_geometry, st != nullptr ? st->ss : ss, mtp_t, err, o.mtp_window))
        return false;
    mtp.set_ple_session(&ss);
    return true;
}

int64_t Model::count_gpu_held() const {
    int64_t held = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int st = multi_gpu ? stage_of(l) : 0;
        const strata::core::ExpertCache& c = st > 0 ? stages[(size_t) st - 1]->cache : xcache;
        const int64_t live = c.slots();
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const int32_t sl = c.slot_of(l, e);
            held += sl != strata::core::kNotResident && sl < live;
        }
    }
    return held;
}

void Model::refresh_file_tier() {
    if (srcp != &src || !file_tier_idle_ok) return;
    const int64_t total = g.n_layers * g.n_expert;
    const int64_t held = count_gpu_held();
    const bool all = held == total;
    if (all == gpu_holds_all) return;
    gpu_holds_all = all;
    bool acted = false;
    if (all) {
        if (io_pf_by_default && src.io_prefetch()) {
            src.set_io_prefetch(false, 0);
            io_pf_suspended = true;
            acted = true;
            const char* sv = std::getenv("STRATA_IO_STATS");
            src.set_io_stats(sv != nullptr && std::atoi(sv) != 0);
        }
        if (la_started && !la_asked && !fs_ahead && !la_stats && drive.d.lookahead != nullptr) { drive.d.lookahead = nullptr; acted = true; }
        if (acted)
            std::fprintf(stderr, "strata serve: every expert is in the GPU caches again (%lld of %lld): the file tier's Linux defaults are off\n",
                         (long long) held, (long long) total);
    } else {
        if (io_pf_suspended) {
            const char* th = std::getenv("STRATA_IO_PF_THREADS");
            src.set_io_prefetch(true, th != nullptr ? std::atoi(th) : 0);
            io_pf_suspended = !src.io_prefetch();
            acted = true;
        }
        if (la_started && drive.d.lookahead == nullptr) {
            drive.d.lookahead = &lookahead;
            if (!std::getenv("STRATA_IO_PREFETCH_DEPTH") && src.io_prefetch()) lookahead.set_depth(2);
            acted = true;
        }
        if (acted)
            std::fprintf(stderr, "strata serve: %lld of %lld experts are in the GPU caches now: the file tier's Linux defaults are on again\n",
                         (long long) held, (long long) total);
    }
}

std::string Model::get_df2_vocab_path() const {
    return df2_solo ? df2_vocab_file : o.mtp_draft_vocab.empty() ? o.mtp + "/draft_vocab.bin" : o.mtp_draft_vocab;
}

void Model::offer_df2_head() {
#if defined(STRATA_HAVE_DFLASH2)
    strata::core::MtpDrafter::DraftHeadView hv;
    if (!o.mtp.empty() && mtp.draft_head_view(&native_head, get_df2_vocab_path(), hv)) {
        strata::core::Dflash2Drafter::SharedHead sh;
        sh.rows = hv.rows; sh.vocab_dev = hv.vocab_dev; sh.vocab_host = hv.vocab_host; sh.n = hv.n; sh.type = hv.type;
        df2.share_head(sh);
    }
#endif
}

int64_t Model::get_df2_head_bytes() const {
    if (df2_expect_share) return 0;   // (its logits are in the 48 MiB the callers add)
#if defined(STRATA_HAVE_DFLASH2)
    if (strata::core::Dflash2Drafter::head_native()) return (int64_t) 7 * n_vocab * (int64_t) sizeof(float);   // STRATA_DF2_HEAD=native: no pruned copy, the logits of every row
#endif
    if (!df2_solo) return (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab);
    int64_t bytes = (int64_t) std::min(o.spec, (int) strata::kernels::kVerifyNarrowT) * n_vocab * (int64_t) sizeof(float);
    if (FILE* f = std::fopen(get_df2_vocab_path().c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fclose(f);
        if (size >= 4 && size % 4 == 0) bytes += (int64_t) (size / 4) * (int64_t) native_head.row_bytes() + (int64_t) size;
    }
    return bytes;
}

strata::core::VerifyHits Model::get_verify_hits() const {
    strata::core::VerifyHits vh;
    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
    vh.n_slots = xcache.slots();
    return vh;
}

strata::core::VerifyHits Model::get_verify_hits(const GpuStage& st) const {
    strata::core::VerifyHits vh;
    vh.d_res = st.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = st.cache.device_slot(0);
    vh.blob = thits.blob;
    vh.slot_off = st.cache.slot_offsets();
    vh.n_slots = st.cache.slots();
    return vh;
}

bool alloc_split_handoffs(const strata::core::ModelGeometry& g, int rows, std::vector<float*>& hand) {
    const size_t bytes = (size_t) rows * (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
    for (float*& h : hand) {
        float* host = nullptr;
        if (strata::core::allocate_host((void**) &host, bytes, cudaHostAllocMapped | cudaHostAllocPortable,
            {"other", "Device allocations", "Model working buffers"}) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &h, host, 0) != cudaSuccess)
            return false;
        std::memset(host, 0, bytes);
    }
    return true;
}

Model::~Model() {
    loop_scratch.free();
    strata::core::token_graph_free(tgraph);
}

bool Model::is_cpu_share_default_ok() {
    // 0.1.42 (H11): the default CPU share is armed only when the host has the memory for it.  One line says why not;
    // an explicit STRATA_PREFILL_CPU_SHARE (other than 0) is armed anyway, with a warning when memory is short.
    strata::prefill::share_rules::ShareMemory sm;
    strata::core::detail::HostMemory hm;
    if (strata::core::detail::host_available_memory(hm)) {
        sm.avail_ram = hm.available;
        sm.avail_commit = hm.commit == ~uint64_t{0} ? strata::prefill::share_rules::kUnknown : hm.commit;
    }
    sm.buffer_bytes = (uint64_t) (1024 + 8 * g.n_expert) * (uint64_t) g.n_embd * 4 + (256ull << 20);
    if (srcp == &src) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (xcache.slot_of(l, e) < 0 && src.resident_blob(l, e) == nullptr && !src.transient(l, e))
                    sm.inplace_bytes += (uint64_t) lay.blob_bytes(l);
    }
    const auto v = strata::prefill::share_rules::default_share_memory_verdict(sm);
    if (v.arm) return true;
    const char* ev = std::getenv("STRATA_PREFILL_CPU_SHARE");
    if (ev == nullptr)
        std::fprintf(stderr, "strata: the CPU prefill share is not armed by default: %s. STRATA_PREFILL_CPU_SHARE=auto "
                             "turns it on anyway\n", v.why.c_str());
    else if (std::strcmp(ev, "0") != 0)
        std::fprintf(stderr, "strata: WARNING STRATA_PREFILL_CPU_SHARE=%s is set and kept, but %s: the share can slow "
                             "prompts or fail on a machine this short of memory (STRATA_PREFILL_CPU_SHARE=0 turns it off)\n",
                     ev, v.why.c_str());
    return false;
}

void Model::arm_prompt_cpu_share() {
    // #1595: not by default when the cache holds every expert (nothing to take)
    strata::prefill::Prefill::arm_cpu_share(!multi_gpu && !o.no_pool,
        !multi_gpu && !o.no_pool && o.batch <= 0 &&
        strata::prefill::share_rules::default_share_has_work(open_expert_pages ? xcache.resident() : xcache.slots(),
                                                             (int64_t) (g.n_layers * g.n_expert)) &&
        is_cpu_share_default_ok());   // H11
}

int Model::pin_resident_experts(const std::vector<std::pair<int32_t, int32_t>>& stage_pairs, int64_t lend_from) {
    {   // the pack's and the MTP head's files: read at start, their cache is given back by the complement's drop
        std::vector<std::string> drop;
        for (const std::string& d : {o.pack, o.mtp}) {
            std::error_code ec;
            if (d.empty()) continue;
            if (std::filesystem::is_regular_file(d, ec)) { drop.push_back(d); continue; }
            for (const auto& f : std::filesystem::directory_iterator(d, ec))
                if (f.is_regular_file(ec)) drop.push_back(f.path().string());
        }
        src.set_extra_drop_files(std::move(drop));
    }
    const std::vector<std::pair<int32_t, int32_t>>& rank_all = profile_all.empty() ? profile : profile_all;
    bool resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, stage_pairs, lend_from,
                                                o.resident_headroom, o.resident_budget, &rank_all);
    std::string whole_err;
    if (!resident_ok && o.resident_soft) {
        // #467: the whole complement does not fit - keep what does, the hottest by the profile, through the #403
        // budget path (sized by available memory) instead of none: the misses outside it read the same file bytes
        // the mmap fallback reads, so the answers are unchanged.  Nothing pinned: the old fallback below.
        whole_err = err;
        resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, stage_pairs, -1, o.resident_headroom,
                                               strata::core::FileExpertSource::kResidentWhatFits, &rank_all);
        if (resident_ok)
            std::fprintf(stderr, "strata generate: WARNING: the whole resident RAM mode does not fit (%s); %.2f "
                                 "GiB of the experts the GPU does not hold, the hottest by the expert profile, are "
                                 "kept in RAM and the rest are read from the model folder through the OS file "
                                 "cache\n",
                         whole_err.c_str(), (double) src.resident_bytes() / 1073741824.0);
        else
            err = whole_err + "; " + err;
    }
    // the startup reads are done: the view of experts.bin back, the file tier as chosen above
    bool startup_kept = false;
    if (src.startup_unbuffered()) {
        std::string why;
        const auto end_t0 = std::chrono::steady_clock::now();
        startup_kept = !src.end_startup_unbuffered(why);
        if (startup_kept) std::fprintf(stderr, "strata generate: WARNING: %s\n", why.c_str());
        else
            std::fprintf(stderr, "strata generate: the startup reads are done: experts.bin mapped again (%.0f ms)\n",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - end_t0).count());
    }
    // #669 / #765: a big chunk lends many cache slots to the prompt path, and each lent slot's expert is read back
    // from the SSD whenever the RAM could not keep its copy (31 GB: auto:32768 read prompts ~3x slower than auto).
    // Said, never capped: --prefill is the user's, and with the RAM for them it is the faster one (+21-35%).
    // #1683: with a RAM budget the lend region is kept after the ranked experts, as far as the budget reaches, so
    // the advice there is a larger budget, not a smaller chunk (the budget used to keep none of it whatever the chunk)
    if (resident_ok && lend_from >= 0 && (o.prefill_chunk > 8192 || o.resident_budget > 0) && xcache.slots() > lend_from &&
        src.resident_lent_slots() < xcache.slots() - lend_from)
        std::fprintf(stderr, "strata generate: WARNING: a prompt chunk of %lld tokens lends %lld cache slots to the "
                             "prompt path, but only %lld of their experts fit in RAM: the others are read from the "
                             "SSD on every chunk, and prompts can read ~3x slower than with --prefill auto (#669). "
                             "%s keeps them all in RAM\n",
                     (long long) o.prefill_chunk, (long long) (xcache.slots() - lend_from),
                     (long long) src.resident_lent_slots(),
                     o.resident_budget > 0 ? "A larger --resident-budget-gib (the lend region comes after the ranked experts), or a smaller --prefill,"
                                           : "A smaller --prefill, or auto,");
    if (resident_ok) {
        // --adapt-async: twice the exchange buffers - the second half bounces the copies in whose source is not
        // page-locked (an asynchronous copy instead of the driver's staged, synchronous one)
        if (o.adapt_async && !src.reserve_exchanges(2 * std::min<int64_t>(o.adapt_swaps, 96), err,
                                                    std::min<int64_t>(o.adapt_swaps, 96)))
            turn_off_adapt_async(err.c_str());
        // pageable buffers would make every copy of a round synchronous (the driver stages them): that is the
        // blocking tier's cost, so the asynchronous tier is not worth running - said, and the blocking tier stays
        if (o.adapt_async && !src.exchange_pinned())
            turn_off_adapt_async("the exchange buffers could not be page-locked");
        if (o.adapt_every > 0 && o.adapt_swaps > 0 &&
            !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
            std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
            return 1;
        }
        {
            cudaDeviceProp ucp{};
            int ucd = 0;
            if (src.complement_pinned() && cudaGetDevice(&ucd) == cudaSuccess && cudaGetDeviceProperties(&ucp, ucd) == cudaSuccess && ucp.integrated)
                g_uma_pinned_complement.store(true);
        }
        std::fprintf(stderr, "strata generate: resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                             "cache; adaptive swaps %s\n",
                     (double) src.resident_bytes() / 1073741824.0,
                     src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                     (long long) xcache.resident(),
                     o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                            : "off");
    } else if (o.resident_soft) {
        std::fprintf(stderr, "strata generate: WARNING: the resident RAM mode does not fit (%s); the experts the "
                             "GPU does not hold are read from the model folder through the OS file cache "
                             "(--mmap-experts), which is slower when the RAM cannot keep them\n", err.c_str());
    } else if (o.resident_budget > 0) {
        // #403: a RAM budget that cannot be kept is not a reason to stop - the experts it would have held are
        // read from the files like the ones outside it (pin_cache_complement leaves nothing half-built)
        std::fprintf(stderr, "strata generate: WARNING: the RAM budget (--resident-budget-gib) cannot be kept (%s); "
                             "every expert the GPU does not hold is read from the model files %s "
                             "(--mmap-experts), which is slower\n", err.c_str(),
                     src.unbuffered() ? "with unbuffered I/O" : "through the OS file cache");
    } else {
        std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
        return 1;
    }
    // #577: the unbuffered choice above was made before the RAM copy existed, from the budget asked for; now the
    // copy is built, decide again from the RAM it really holds and the expert bytes outside it (on a 96 GB PC
    // the file cache keeps those, and every refill after a prompt read the drive instead).  Windows and Linux: the
    // unbuffered reads exist there
#if defined(_WIN32) || defined(__linux__)
    if (o.mmap_experts && o.resident_budget > 0 && !startup_kept) {
        std::string why;
        const bool was = src.unbuffered();
        const bool ub = src.recheck_unbuffered(why);
        std::fprintf(stderr, "strata generate: the file tier reads %s%s (%s)\n",
                     ub ? "unbuffered" : "through the file cache", ub == was ? "" : " (changed)", why.c_str());
        if (ub != was) {
            // the plain line (owner 11-06): printed once, when the mode changes to the file cache; the RAM figures are the ones the rule
            // used (the group's own limit and room, no operator cap).  STRATA_TRACE keeps the long technical line.
            constexpr double GiB = 1073741824.0;
            if (const auto mem = strata::core::sample_host_memory(strata::core::MemoryProbePaths{}, 0)) {
                if (!ub)
                    std::fprintf(stderr, "strata: expert reads now go through the file cache (%.1f GiB of RAM free for them, %.1f GiB limit)\n",
                                 (double) mem->available / GiB, (double) mem->limit / GiB);
                if (std::getenv("STRATA_TRACE") != nullptr)
                    std::fprintf(stderr, "strata trace: the file tier changed: RAM at the re-check: limit %.1f GiB (source: %s), current %.1f GiB, "
                                         "reclaimable cache credited %.1f GiB; available %.1f GiB (MemAvailable %.1f GiB)\n",
                                 (double) mem->limit / GiB, strata::core::memory_source_name(mem->source), (double) mem->current / GiB,
                                 (double) mem->credit / GiB, (double) mem->available / GiB, (double) mem->mem_available / GiB);
            }
        }
        // MODEL-UPDATE round 2 (opt-in STRATA_PROMPT_DIRECT=1): the prompt path's batched reads unbuffered while
        // decode keeps the file cache (z440, UD-Q4_K_XL, 39 GiB budget: all-unbuffered doubled the 6K prompt but
        // cost decode 9-13%).  The same bytes either way.
        if (!ub && [] { const char* v = std::getenv("STRATA_PROMPT_DIRECT"); return v != nullptr && v[0] == '1'; }()) {
            std::string pw;
            const bool pd = src.open_prompt_direct(pw);
            std::fprintf(stderr, "strata generate: the prompt path's expert reads %s\n",
                         pd ? "go unbuffered (STRATA_PROMPT_DIRECT=1)" : ("stay buffered: " + pw).c_str());
        }
    }
#endif
    return 0;
}

void Model::finish_expert_tiers() {
    // (the arena mode has no RAM copy to exchange with: its experts are always in RAM, the tier only copies them in)
    if (o.adapt_async && !src.complement_ready() && srcp != &arena_src) turn_off_adapt_async("the resident RAM mode is not running");
    // pp-opt: with the file tier unbuffered, the mapped view of experts.bin is only a fallback from here on - and while
    // it exists NTFS serves the unbuffered reads one at a time (drop_mapping).  STRATA_KEEP_MAPPING=1 keeps it (A/B).
    if (srcp == &src && src.unbuffered() && std::getenv("STRATA_KEEP_MAPPING") == nullptr) {
        std::string why;
        const bool dropped = src.drop_mapping(why);
        std::fprintf(stderr, "strata generate: the mapped view of the experts %s (%s)\n",
                     dropped ? "is closed" : "stays open", why.c_str());
        // --ple-io mmap / ram maps the PLE shard for the whole run; when that shard also holds experts, NTFS still runs
        // the unbuffered reads of that one file one at a time (the default --ple-io direct keeps no mapping)
        if (dropped && src.gguf_mode() && !o.ple_gguf.empty() && o.ple_io != "direct")
            std::fprintf(stderr, "strata generate: note: --ple-io %s keeps %s mapped; if that shard holds experts, their "
                                 "reads stay serialized (--ple-io direct, the default, keeps no mapping)\n",
                         o.ple_io.c_str(), o.ple_gguf.c_str());
    }
}

void Model::use_expert_source(strata::core::ExpertSource& source) {
    srcp = drive.d.src = &source;
    drive.d.lookahead = nullptr;
}

void Model::turn_off_adapt_async(const char* why) {
    std::fprintf(stderr, "strata generate: --adapt-async 1 is off (%s): the blocking adaptive tier\n", why);
    o.adapt_async = 0;
}

void Model::set_prompt_pinned_share() {
    if (srcp == nullptr || o.prefill_chunk <= 0) return;
    uint64_t pinned = 0, total = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const uint64_t b = lay.blob_bytes(l);
            total += b;
            if (srcp->pinned(l, e)) pinned += b;
        }
    double share = total ? (double) pinned / (double) total : 1.0;
    if (const char* t = std::getenv("STRATA_TEST_PINNED_SHARE")) share = std::atof(t);   // tests: the Windows-like ring rule
    strata::prefill::Prefill::set_pinned_share(share);
}

int Model::stage_of(int64_t l) const {
    int st = 0;
    while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
    return st;
}

size_t Model::gdn_snapshot_bytes(const strata::core::SessionState& s0) const {
    return (size_t) s0.gdn_alloc * ((size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                    (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1)) * sizeof(float);
}

int64_t g_lend_keep = 128;
std::atomic<bool> g_uma_pinned_complement{false};

bool spill_discrete_gpu() {
#if defined(STRATA_GPU_METAL)
    return strata::spill_allowed(true, false);
#else
    static const bool ok = [] {
        int d = 0;
        cudaDeviceProp pr{};
        if (cudaGetDevice(&d) != cudaSuccess || cudaGetDeviceProperties(&pr, d) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        return strata::spill_allowed(false, pr.integrated != 0);
    }();
    return ok;
#endif
}

// #620 #486: " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() {
    (void) cudaGetLastError();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}

// Gets the share (%) of an expert cache's slots the prompt path may borrow with --prefill auto.  At 8192-token chunks
// nearly every expert streams anyway, so a lent slot costs little: 90% when the copies are DMA from pinned RAM (Q2_0
// 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the limit (lending more only streams more through
// them).  A card too short for the usual loan (g_lend_keep below 128, see the cache sizing) lends what it must: 100%.
// Read when called: the pinned share and g_lend_keep are set once the RAM copy and the cache are known.
// STRATA_PREFILL_LEND_PCT overrides (tuning).
int64_t get_auto_lend_pct() {
    const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
    return v ? (int64_t) std::atoi(v)
             : g_lend_keep != 128 ? (int64_t) 100
                                  : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
}

void rss_probe(const char* where) {
    static const bool on = std::getenv("STRATA_RSS_TRACE") != nullptr;
    if (!on) return;
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    long long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "RssFile:", 8) == 0) { kb = std::atoll(line + 8); break; }
    std::fclose(f);
    std::fprintf(stderr, "strata rss: %s: RssFile %.2f GiB\n", where, (double) kb / (1024.0 * 1024.0));
    std::fflush(stderr);
}

/// One more launch with the prompt path's buffers in place: a 64 MiB allocation, a stream, a fill and a wait.  On an integrated GPU whose pool is full the driver cannot
/// build the command buffer ("PAL failed to submit CMD") and every later launch fails with "unspecified launch failure"; this finds it before the prompt is read.
#if defined(STRATA_USE_HIP)
static bool uma_launch_probe(std::string& why) {
    (void) cudaGetLastError();
    void* p = nullptr;
    if (cudaMalloc(&p, 64ull << 20) != cudaSuccess) { why = "a 64 MiB allocation failed"; (void) cudaGetLastError(); return false; }
    cudaStream_t st = nullptr;
    bool ok = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) why = "a stream could not be created";
    if (ok) ok = cudaMemsetAsync(p, 0, 64ull << 20, st) == cudaSuccess && cudaStreamSynchronize(st) == cudaSuccess && cudaGetLastError() == cudaSuccess;
    if (!ok && why.empty()) why = "a launch failed";
    if (st != nullptr) cudaStreamDestroy(st);
    cudaFree(p);
    if (!ok) (void) cudaGetLastError();
    return ok;
}
#endif

/// Integrated GPU (Strix Halo), the prompt path's own buffers, a chunk above the 16384 default that the person or the environment named: the unified pool reads 0 MiB free
/// on a start that works, so how much room is left cannot be read, and 32768 on a UD-Q4_K_XL cache that fills the pool takes the last of it (the driver then cannot create a
/// stream, and the context is unusable from then on: HIP 719 at the first launch, after the prompt was read).  So once the buffers are in place one stream, fill and wait are
/// tried; if they fail the engine says so in one line and ends, before any prompt is read, instead of dying after the prefill.  It cannot step down in the same process (the
/// failed context aborts on the first call that touches it).  Never for the 16384 default, a chunk that borrows cache slots, or a discrete GPU.  STRATA_PF_OWN_PROBE_OFF=1 skips it.
void uma_refuse_chunk_if_full(int64_t chunk, bool borrows, double cache_gib) {
#if !defined(STRATA_USE_HIP)
    (void) chunk; (void) borrows; (void) cache_gib;   // observed on the Strix Halo (HIP) only: CUDA and Metal builds are untouched
    return;
#else
    if (borrows || chunk <= 16384 || spill_discrete_gpu() || std::getenv("STRATA_PF_OWN_PROBE_OFF") != nullptr) return;
    std::string why;
    if (uma_launch_probe(why)) {
        if (std::getenv("STRATA_TRACE") != nullptr) std::fprintf(stderr, "strata trace: the launch probe passed with a %lld-token chunk\n", (long long) chunk);
        return;
    }
    const int64_t next = strata::core::gfx12_chunk_step_down(chunk);
    std::fprintf(stderr, "strata generate: a %lld-token prompt chunk does not fit beside the %.1f GiB expert cache on this GPU (%s): use --prefill auto:%lld, or a smaller --expert-cache\n",
                 (long long) chunk, cache_gib, why.c_str(), (long long) next);
    std::fflush(stderr);
    std::_Exit(2);
#endif
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const char* const env = std::getenv("STRATA_TRACE");
    if (env == nullptr) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
    // STRATA_TRACE=2: also wait for the device here and name the first step after which it reports an error.  A
    // "sticky" failure (#2013: unspecified launch failure surfacing in a later, unrelated call) is then pinned to
    // the step that ran the faulting work.  Only a diagnostic: it adds a synchronisation per mark.
    if (env[0] == '2') {
        const cudaError_t sync_e = cudaDeviceSynchronize();
        const cudaError_t last_e = cudaGetLastError();
        std::fprintf(stderr, "strata trace: device %s after %s (sync: %s, last error: %s)\n",
                     sync_e == cudaSuccess && last_e == cudaSuccess ? "ok" : "ERROR", where, cudaGetErrorString(sync_e),
                     cudaGetErrorString(last_e));
    }
}

/// 0.1.43: PcieAuto also refines a layer split's shares, each stage by its own probed link.  0.2.10.1: on by default for a discrete CUDA card
/// (strata/core/pcie_share_rule.hpp), STRATA_PCIE_AUTO_SPLIT=0 restores the fixed probe share; HIP stays opt-in (=1), as does an integrated GPU.
bool pcie_auto_split_ok() {
    static const bool on = [] {
#if defined(STRATA_USE_HIP)
        const bool hip = true;
#else
        const bool hip = false;
#endif
        return strata::pcie_auto_split_default(hip, spill_discrete_gpu(), strata::qwen_defaults_026(), std::getenv("STRATA_PCIE_AUTO_SPLIT"));
    }();
    return on;
}

// 0.1.43: the HIP defaults of the decode speed push (route tail skip, measured PCIe share) were measured on gfx1201 (Radeon AI PRO
// R9700) and apply on gfx1200 / gfx1201 only; every other AMD part (gfx11, gfx906, ...) keeps its earlier behaviour.  CUDA: always true.
bool decode_defaults_arch_ok() {
#if defined(STRATA_USE_HIP)
    int dev = 0;
    cudaDeviceProp p{};
    return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
           strata::kernels::gfx_arch_is_gfx12_wmma(p.gcnArchName);
#else
    return true;
#endif
}

double get_last_h2d_gbps() { return g_last_h2d_gbps; }

#if !defined(_WIN32)
// #605: whether `path` is on a rotational disk (1), not (0) or unknown (-1), from sysfs; `dev` is the disk's name
int on_rotational_disk(const std::string& path, std::string& dev) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return -1;
    char sys[64];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
    std::error_code ec;
    std::filesystem::path p = std::filesystem::canonical(sys, ec);
    for (int up = 0; !ec && up < 2; ++up, p = p.parent_path()) {   // a partition has its disk's queue
        std::ifstream q(p / "queue" / "rotational");
        int r = -1;
        if (q >> r) {
            dev = p.filename().string();
            return r;
        }
    }
    return -1;
}
#endif

/// Runs a verify window's pool call (drive_pool_multi's body) with the CPU experts' `arithmetic`.
void drive_pool_multi_impl(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                           int64_t layer, strata::kernels::cpu::NativeArithmetic arithmetic) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out, arithmetic);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < strata::core::kCompiledGeometry.n_layers) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

void drive_pool_multi(void* u, const float* x, const int32_t* ids, int64_t n, int64_t k, float* out, int64_t l) {
    drive_pool_multi_impl(u, x, ids, n, k, out, l, strata::kernels::cpu::NativeArithmetic::Default);
}
void drive_pool_multi_invariant(void* u, const float* x, const int32_t* ids, int64_t n, int64_t k, float* out, int64_t l) {
    drive_pool_multi_impl(u, x, ids, n, k, out, l, strata::kernels::cpu::NativeArithmetic::WindowInvariant);
}

/// Routes `layer`'s pool call to the stage that runs it (drive_pool_split's body, with `arithmetic`).
void drive_pool_split_impl(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                           int64_t layer, strata::kernels::cpu::NativeArithmetic arithmetic) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    if (s->holder != nullptr)
        d.d.peer = s->holder[layer] >= 0 ? s->hop[s->holder[layer]] : nullptr;
    drive_pool_multi_impl(s->base, x_f, ids, n_tok, k, out, layer, arithmetic);
}
void drive_pool_split(void* u, const float* x, const int32_t* ids, int64_t n, int64_t k, float* out, int64_t l) {
    drive_pool_split_impl(u, x, ids, n, k, out, l, strata::kernels::cpu::NativeArithmetic::Default);
}
void drive_pool_split_invariant(void* u, const float* x, const int32_t* ids, int64_t n, int64_t k, float* out, int64_t l) {
    drive_pool_split_impl(u, x, ids, n, k, out, l, strata::kernels::cpu::NativeArithmetic::WindowInvariant);
}

namespace {
// The plain reason for an out-of-memory in the loads that follow the request slots' sessions (the draft layer, the native head): with --batch N
// those N sessions came out of the card before the cache was sized, and a card that is short for them has nothing left for these.
std::string vram_batch_hint(const std::string& err, int batch) {
    if (batch < 2 || err.find("out of memory") == std::string::npos) return {};
    char buf[320];
    std::snprintf(buf, sizeof buf, "; this start keeps %d request slots (--batch %d), each a session of its own, so the card is too small for them at this context: "
                  "use a smaller --max-context or fewer --batch slots", batch, batch);
    return buf;
}

// Windows' WDDM driver model: native Windows, or WSL2 (its GPU goes through /dev/dxg to the Windows driver).  There,
// pinning a large arena into two CUDA contexts leaves WDDM refusing every later allocation (the 5080 + 3090 rig);
// a Linux driver has no such limit (#253: the 8 GiB cap cost a 4090 + 3060 split 3x of its prompt speed).
bool under_wddm() {
#ifdef _WIN32
    return true;
#else
    static const bool dxg = std::filesystem::exists("/dev/dxg");
    return dxg;
#endif
}

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= strata::core::kCompiledGeometry.n_layers) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports;
// `digest` identifies the uploaded tables, mode and range exactly (a session file is bound to it).
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary,
                          uint64_t& digest, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                std::fprintf(stderr, "strata generate: %s was made for '%s', not qwen4exp\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!strata::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    {
        strata::core::SessionIdentityBuilder b(0x43564543ull);   // "CVEC"
        b.i64("mode", o.cvec_mode);
        b.i64("first", first);
        b.i64("last", last);
        b.i64("single", single);
        b.bytes("dir", dir.data(), dir.size() * sizeof(float));
        b.bytes("scale", s.data(), s.size() * sizeof(float));
        digest = b.digest();
        if (digest == 0) digest = 1;   // 0 means "none loaded"
    }
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "strata generate: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

/// 0.1.44: the pipelined decode puts the drafter on the first stage's card by default where it was measured (gfx12; the HIP builds
/// only).  STRATA_MTP_STAGE0=0 / 1 decides either way.
[[maybe_unused]] bool mtp_first_default_on() {
#if defined(STRATA_USE_HIP)
    return decode_defaults_arch_ok();
#else
    return false;
#endif
}

/// 0.2.7: gfx12 native packs start their decode miss split at 0.15 instead of the x16 share 0.55.  Measured on an R9700
/// (p620, IQ3_S, 24576 expert pairs of which 13207 resident): --pcie-frac 0.15 decodes 94.6 tok/s at 8K against 86.1 and
/// 84.4 against 67.2 at 32K (rotated pairs; at 64K the window is attention-bound and it is +4.7%); under serve/server.py a
/// short chat decodes 93.2 against 84.4.  The CPU lane rounds a missing expert's FP32 rows differently from the GPU lane,
/// so this changes bits; the first-token KL of 0.0 against 0.55 is 0.0013 mean with the same top-1 in 12 of 12 prompts
/// (below the engine's own implementation-noise scale).  An explicit --pcie-frac always wins, and
/// STRATA_PCIE_FRAC_DEFAULT=old restores the old share; the HIP-only gate keeps every CUDA build on its own measured
/// values.
bool pcie_share_gfx12_default() {
#if defined(STRATA_USE_HIP)
    return decode_defaults_arch_ok();
#else
    return false;
#endif
}

// HIP outside gfx12: the same refinement, opt-in with STRATA_PCIE_FRAC_DEFAULT=auto (#2224).  Its author gate-measured it on an
// RX 7900 XTX (gfx1100) as "no regression" only (4 alternating starts at 64K, prefill 2189 base vs 2180 with it, decode 65.4 vs
// 65.4; under serve the share refines 0.55 -> 0.25 from the pool's 42-45 us/missed expert and the probed 28.4-28.5 GB/s link);
// a second report (2x RX 7900 GRE) found it never engages there (95-98% of the experts cached).  The gain is measured only on
// gfx1201 (where it is the default, decode_defaults_arch_ok), so the other parts keep the fixed share unless asked.
[[maybe_unused]] bool pcie_auto_fork_arch_ok() {
#if defined(STRATA_USE_HIP)
    const char* pfd = std::getenv("STRATA_PCIE_FRAC_DEFAULT");
    return pfd != nullptr && std::strcmp(pfd, "auto") == 0;
#else
    return false;
#endif
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// #485: a single timed burst read an x16 PCIe 4 link (RTX A3000 laptop) at 18.5, 6.9 and 5.8 GB/s in three starts -
// a link still in a low-power state, other DMA, a context not yet up to speed - and the low reading set the share.
// Such a disturbance only ever slows a copy: nothing makes a correctly timed copy faster than the link carries.  So
// the same 1 GiB is now copied as four bursts of 256 MiB, timed one by one, and the fastest is the link's figure (the
// median would still follow a disturbance that lasts through half the bursts).  Each burst takes ~10 ms on an x16
// PCIe 4 link, so the probe takes no longer than the one 1 GiB burst did.  `samples`, when given, gets every
// burst's reading for the log.
double probe_pcie_h2d_gbps(std::string* samples) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (strata::core::allocate_host(&h, kBytes, cudaHostAllocDefault,
        {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) return -1.0;
    if (strata::core::allocate_device(&d, kBytes,
        {"other", "Device allocations", "Model working buffers"}) != cudaSuccess) {
        strata::core::release_host(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    float ms[kBursts] = {};
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // Windows HIP: the events do not bracket the copies there (an RX 6800 read 3,300-26,000 GB/s, so every link kept
    // the x16 share), so each burst is timed on the host between two synchronizes: 256 MiB takes ~10 ms, so the
    // synchronize around it hardly matters
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    for (int b = 0; b < kBursts && ok; ++b) {
        const Clock::time_point t0 = Clock::now();
        cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
        ok = cudaDeviceSynchronize() == cudaSuccess;
        ms[b] = (float) std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
#else
    // the bursts run back to back on the stream, an event between each two
    cudaEvent_t ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts && cudaEventCreate(&ev[n_ev]) == cudaSuccess) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        cudaEventRecord(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
            cudaEventRecord(ev[b + 1]);
        }
        ok = cudaEventSynchronize(ev[kBursts]) == cudaSuccess;
        for (int b = 0; b < kBursts && ok; ++b) ok = cudaEventElapsedTime(&ms[b], ev[b], ev[b + 1]) == cudaSuccess;
    }
    for (int i = 0; i < n_ev; ++i) cudaEventDestroy(ev[i]);
#endif
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    if (!ok) (void) cudaGetLastError();
    strata::core::release_device(d);
    strata::core::release_host(h);
    if (bw > 0.0) g_last_h2d_gbps = bw;
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share measured on x16 links) from
// 20 GB/s up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about
// what the x16 share costs.  Continuous (#485): before, 19.9 GB/s gave 0.42 and 20.0 the full 0.55 (and 4.0 GB/s
// gave 0.08, 3.9 none), so a reading near either edge moved the share by a quarter of its range.  From 20 GB/s up
// (an x16 PCIe 4/5 link: ~26-28 GB/s) the share is unchanged.
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

// --pool-affinity measure: `all` against `p-cores` on a hybrid CPU, timed on this pack's own experts.
//
// The pool's default layout puts a worker on every physical core, and a layer's experts are done when the slowest
// worker is.  Measured on a Core i7-13700K (8 P + 8 E), IQ3_XXS, RTX 5090, same prompt and build: `all` (15 workers)
// spent 6.7 ms per verify window in the pool and decoded 148 tok/s, `p-cores` (7 workers) 2.2 ms and 188-191 tok/s;
// in an earlier run of the same day `all` took 2.4 ms, so the difference is not constant.  Hence a measurement at
// start and not a rule.
//
// What is timed is the decode's own call: two missed experts of four tokens per layer (run_split_multi_native on
// blobs from the arena), every layer in turn, with the host spinning ~150 us between two layers as it does on the
// GPU's flags.  The two layouts take turns in kBlocks blocks and the medians are compared; `p-cores` is taken
// unless `all` is more than 10% faster (fewer threads wait on fewer cores when the two are close).  The layout
// changes no result, only where threads run.  It is one reading at start and is not repeated: on the same PC one
// server start in four read `all` as the faster layout (6.6 ms against 8.6), where the others read it 2.4 to 4 times
// slower.
strata::kernels::cpu::PoolAffinity measure_pool_affinity(strata::core::ExpertSource* src, int workers, bool host_works,
                                                         int tasks) {
    namespace cpu = strata::kernels::cpu;
    using Clock = std::chrono::steady_clock;
    const cpu::ExpertLayout& lay = cpu::expert_layout();
    const cpu::CpuTopology topo = cpu::detect_cpu_topology(true, cpu::PoolAffinity::PCores);
    if (!topo.is_hybrid) {
        std::fprintf(stderr, "strata generate: --pool-affinity measure: this CPU is not a hybrid one; the layout stays all\n");
        return cpu::PoolAffinity::All;
    }
    constexpr int kExperts = 2, kTok = 4, kRounds = 10, kBlocks = 3;
    const int64_t nl = lay.n_layers;
    std::vector<const uint8_t*> blobs((size_t) nl * kExperts, nullptr);
    bool have = src != nullptr && lay.native && (int64_t) lay.fmt.size() >= nl && nl > 0;
    uint32_t seed = 12345u;
    for (int64_t l = 0; have && l < nl; ++l)
        for (int j = 0; j < kExperts; ++j) {
            seed = seed * 1664525u + 1013904223u;
            const uint8_t* b = src->blob(l, (int64_t) ((seed >> 8) % (uint32_t) lay.n_expert));
            if (b == nullptr) { have = false; break; }
            blobs[(size_t) l * kExperts + (size_t) j] = b;
        }
    if (!have) {
        std::fprintf(stderr, "strata generate: --pool-affinity measure needs a native pack with its experts in RAM; "
                             "the layout stays all\n");
        return cpu::PoolAffinity::All;
    }
    std::vector<uint8_t> nact((size_t) nl * kTok * cpu::kNativeActBytes);
    std::vector<cpu::ActQ> qact((size_t) nl * kTok);
    std::vector<float> x((size_t) cpu::H), out((size_t) kExperts * kTok * cpu::H);
    for (int64_t l = 0; l < nl; ++l)
        for (int t = 0; t < kTok; ++t) {
            for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t + 7 * (int) l) * 0.031f);
            const size_t at = (size_t) l * kTok + (size_t) t;
            if (cpu::q2_native_kernels(lay.fmt[(size_t) l].gu_type)) cpu::act_quant_any(x.data(), cpu::H, qact[at]);
            else cpu::native_quant_act(lay.fmt[(size_t) l], x.data(), nact.data() + at * cpu::kNativeActBytes);
        }
    int n_workers[2] = {0, 0};
    auto block = [&](cpu::PoolAffinity aff, int which) -> double {
        cpu::ExpertPool pool(workers, /*pin=*/true, host_works, aff, tasks);
        n_workers[which] = pool.workers();
        const int host = cpu::planned_host_core(aff);
        cpu::ThreadAffinity prev;
        if (host >= 0) prev = cpu::pin_current_thread(host);
        cpu::ExpertJobMulti jobs[kExperts];
        double ms = 0.0;
        for (int r = -1; r < kRounds; ++r)   // round -1 is not timed: the workers start and the rows are touched
            for (int64_t l = 0; l < nl; ++l) {
                for (int j = 0; j < kExperts; ++j) {
                    jobs[j].blob = blobs[(size_t) l * kExperts + (size_t) j];
                    jobs[j].nt = kTok;
                    for (int t = 0; t < kTok; ++t) {
                        const size_t at = (size_t) l * kTok + (size_t) t;
                        jobs[j].act[t] = &qact[at];
                        jobs[j].nact[t] = nact.data() + at * cpu::kNativeActBytes;
                        jobs[j].out[t] = out.data() + ((size_t) j * kTok + (size_t) t) * cpu::H;
                    }
                }
                const Clock::time_point t0 = Clock::now();
                pool.run_split_multi_native(lay.fmt[(size_t) l], jobs, kExperts);
                const Clock::time_point t1 = Clock::now();
                if (r >= 0) ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                while (std::chrono::duration<double, std::micro>(Clock::now() - t1).count() < 150.0) {}
            }
        if (host >= 0) cpu::restore_thread_affinity(prev);
        return ms / kRounds;
    };
    double all_ms[kBlocks], p_ms[kBlocks];
    for (int b = 0; b < kBlocks; ++b) {
        all_ms[b] = block(cpu::PoolAffinity::All, 0);
        p_ms[b] = block(cpu::PoolAffinity::PCores, 1);
    }
    std::sort(all_ms, all_ms + kBlocks);
    std::sort(p_ms, p_ms + kBlocks);
    const double a = all_ms[kBlocks / 2], p = p_ms[kBlocks / 2];
    const bool take_p = p <= a * 1.10;
    std::fprintf(stderr, "strata generate: --pool-affinity measure: %d experts x %d tokens per layer over %lld layers: "
                         "all (%d workers) %.2f ms, p-cores (%d workers) %.2f ms (medians of %d) -> %s\n",
                 kExperts, kTok, (long long) nl, n_workers[0], a, n_workers[1], p, kBlocks, take_p ? "p-cores" : "all");
    return take_p ? cpu::PoolAffinity::PCores : cpu::PoolAffinity::All;
}

// #1425 #1549 #1629: --ple-io direct (the default) does unbuffered random reads of the n-gram table; on some drives
// (DRAM-less NVMe, Windows unbuffered I/O) they run at 34 MB/s or 4-8 tok/s and a prompt reads layer 1 for minutes.
// Rows per second of a short cold probe: a few batches of random rows through the same batched reader a prompt chunk
// uses, stopped after ~0.6 s. Returns <= 0 when nothing could be measured.
double ple_direct_rows_per_s(strata::kernels::PleTable& t) {
    const uint64_t n = t.rows();
    if (n < 4096) return -1.0;
    constexpr size_t kTok = 4;   // 64 rows a batch
    std::vector<uint32_t> rows(kTok * 16);
    // one token's PLE rows are PLE_N_HEADS x PLE_HEAD_DIM floats = the compiled hidden size (2560 for Flash-Next, 4096 for q4_big): a
    // buffer of 2560 floats a token was a heap overflow of 24 KB at hidden 4096 (found by an ASan run; on Metal it landed in Apple's
    // shader-cache objects and the first pipeline of the first window died in MTLCompilerFSCache::getElementImpl)
    std::vector<float> out(kTok * (size_t) strata::kernels::PLE_N_HEADS * (size_t) strata::kernels::PLE_HEAD_DIM);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ n;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t done = 0;
    double el = 0;
    for (int b = 0; b < 64; ++b) {
        for (auto& r : rows) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r = (uint32_t) (x % n); }
        std::string e;
        if (!t.gather_batch(rows.data(), kTok, out.data(), e)) return -1.0;
        done += rows.size();
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > 0.6) break;
    }
    return el > 0 ? (double) done / el : -1.0;
}
/// Gets the largest slot an expert cache can hold for this pack: the GGUF `max_blob`, or a converted layer's 4-bit slot
/// when STRATA_EXPERT_NIBBLE makes that larger (the sizings that count slots of the largest blob would otherwise come
/// out short of a full cache and leave converted experts out of VRAM, which the switch refuses).
uint64_t nibble_max_slot_bytes() {
    const auto& lay = strata::kernels::cpu::expert_layout();
    uint64_t m = lay.max_blob;
    if (nib::any_on())
        for (int64_t l = 0; l < lay.n_layers; ++l) m = std::max<uint64_t>(m, nib::slot_bytes(l));
    return m;
}

/// With STRATA_EXPERT_NIBBLE=1, gets why this start cannot have 4-bit slots (empty: nothing here stops it): what the
/// build, the GPUs, the environment and the options decide, before anything is allocated.  The full residency, the
/// loans and the prompt chunk are checked where they are known.
std::string nibble_refusal(const Options& o) {
    const nib::Config& c = nib::config();
    if (!c.error.empty()) return c.error;
#if !STRATA_EXPERT_NIBBLE_BUILT || !defined(STRATA_NATIVE_EXPERTS)
    (void) o;
    return "the 4-bit slots and their kernels are in the CUDA build with STRATA_EXPERIMENTAL_SM60 and native experts only";
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        (void) cudaGetLastError();
        return "no CUDA device";
    }
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) {
            (void) cudaGetLastError();
            return "cannot read CUDA" + std::to_string(d) + "'s properties";
        }
        if (p.major != 6 || p.minor != 0)
            return "CUDA" + std::to_string(d) + " (" + p.name + ") is compute capability " + std::to_string(p.major) + "." +
                   std::to_string(p.minor) + ": the 4-bit slots are for compute capability 6.0 (GP100) only";
    }
    if (std::string e = nib::env_conflicts(); !e.empty()) return e;
    if (o.peer_device >= 1) return "--peer-device (its cache holds GGUF blobs)";
    for (size_t r = 0; r < o.expert_cache_remote.size(); ++r)
        if (o.expert_cache_remote[r] > 0 || o.expert_cache_remote_auto[r])
            return "remote expert caches (--expert-cache-device1..3 hold GGUF blobs)";
    if (o.kv_grow) return "--kv-grow (the elastic K/V moves slots at their GGUF size)";
    if (o.vram_elastic) return "--vram-elastic (it refills slots with GGUF blobs)";
    if (o.expert_profile.empty())
        return "no --expert-profile (every expert must be VRAM-resident from the profile fill)";
    if (o.expert_cache == 0) return "--expert-cache 0 (no VRAM slots to convert)";
    return {};
#endif
}

/// Prints the cards' free and total memory and the largest other programs that hold some, as a warning added to the
/// message that says the engine cannot start for lack of GPU memory (#3797, base/gpu/gpu_report.hpp). Best effort: a
/// card that cannot be asked, or a driver that does not list its programs, is left out or said so; it never changes the
/// failure and never throws. `devices`: the CUDA ordinals to report (duplicates are skipped).
void print_startup_gpu_report(const std::vector<int>& devices) {
    namespace gr = strata::program::gpu_report;
    std::vector<gr::Card> cards;
    int count = 0, prev = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { (void) cudaGetLastError(); return; }
    if (cudaGetDevice(&prev) != cudaSuccess) { (void) cudaGetLastError(); prev = 0; }
    std::vector<int> seen;
    for (const int d : devices) {
        if (d < 0 || d >= count || std::find(seen.begin(), seen.end(), d) != seen.end()) continue;
        seen.push_back(d);
        if (cudaSetDevice(d) != cudaSuccess) { (void) cudaGetLastError(); continue; }   // no context fits there either
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void) cudaGetLastError(); continue; }
        gr::Card c;
        c.index = d;
        c.free_mib = (uint64_t) (free_b >> 20);
        c.total_mib = (uint64_t) (total_b >> 20);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) c.name = prop.name;
        else (void) cudaGetLastError();
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        char bus[64] = {};
        if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, d) == cudaSuccess) c.processes_known = gr::query_other_processes(bus, c.processes);
        else (void) cudaGetLastError();
#endif
        cards.push_back(std::move(c));
    }
    if (prev >= 0 && prev < count) (void) cudaSetDevice(prev);
    (void) cudaGetLastError();
    if (cards.empty()) return;
#if defined(_WIN32)
    const uint32_t self = (uint32_t) GetCurrentProcessId();
#else
    const uint32_t self = (uint32_t) getpid();
#endif
    std::fprintf(stderr, "strata generate: GPU memory right now (best effort):\n%s\n",
                 gr::format_gpu_memory_report(cards, self).c_str());
}

}  // namespace
}  // namespace strata::qwen38
