#include "strata/common/serve/request_output.hpp"
#include "strata/common/sample/logit_bias.hpp"
// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "strata/base/platform/env.hpp"
#include "strata/base/platform/memory.hpp"
#include "strata/base/platform/io_priority.hpp"
#include "strata/models/qwen38/adapt_async_default.hpp"
#include "strata/core/arch_defaults.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/core/gfx12_chunk.hpp"
#include "strata/core/nv_chunk.hpp"
#include "strata/core/nv_defaults.hpp"
#include "strata/base/gpu/gfx_arch.hpp"
#include "strata/models/qwen38/batch_telemetry.hpp"
#include "strata/base/gpu/dma_batch.hpp"
#include "strata/program/adaptive_range.hpp"
#include "strata/core/run_manifest.hpp"
#include "strata/base/gpu/device.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/common/state/token_listing.hpp"
#include "strata/common/serve/loop.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/common/experts/expert_profile.hpp"
#include "strata/common/experts/expert_swaps.hpp"
#include "strata/common/kernels/expert_nibble.hpp"
#include "strata/models/qwen38/conversation_snapshot.hpp"
#include "strata/core/lookup_default.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/core/serve_stop_bound.hpp"
#endif
#include "strata/common/state/conversation_file.hpp"
#include "strata/common/state/session_save_reclaim.hpp"
#include "strata/common/state/conversation_memory.hpp"
#include "strata/models/qwen38/ple_residency.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/foresight_swap.hpp"
#include "strata/base/gpu/pinned.hpp"
#include "strata/base/gpu/pin_budget.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/base/gpu/memory_info.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/models/qwen38/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/models/qwen38/session.hpp"
#include "strata/models/qwen38/session_identity.hpp"
#include "strata/base/gpu/gfx_arch.hpp"
#include "strata/common/kernels/cpu/pool.hpp"
#include "strata/base/platform/aux_cpus.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/models/qwen38/kernels/ngram.hpp"
#include "strata/models/common/kernels/s2_expert_grouped.hpp"
#include "strata/common/kernels/sampler.hpp"
#include "strata/models/qwen38/kernels/verify_kernels.hpp"
#include "strata/common/kernels/elementwise.hpp"
#include "strata/models/common/kernels/native_moe.hpp"
#include "strata/models/qwen38/kernels/kv_q4.hpp"
#include "strata/models/qwen38/kernels/qsa.hpp"
#include "strata/models/qwen38/native_head.hpp"
#include "strata/core/spill_gate.hpp"
#include "strata/core/pcie_share_rule.hpp"
#include "strata/models/qwen38/verify.hpp"
#include "strata/models/qwen38/mtp.hpp"
#include "strata/common/spec/gated_proposal.hpp"
#if defined(STRATA_HAVE_DFLASH2)
#include "strata/models/qwen38/dflash2.hpp"
#endif
#include "strata/common/kernels/prefill/gemm.hpp"
#include "strata/models/qwen38/prefill/kva.hpp"
#include "strata/models/qwen38/prefill/kva_floor.hpp"
#include "strata/models/qwen38/prefill/prefill.hpp"
#include "strata/common/kernels/prefill/pf_fused_override.hpp"
#include "strata/base/gpu/sm120_series.hpp"
#if defined(STRATA_PREFILL_MMQ)
#include "strata/common/kernels/prefill/moe_mmq.hpp"
#endif
#include "strata/models/qwen38/conversation.hpp"
#include "strata/models/qwen38/model.hpp"
#include "strata/models/qwen38/prompt_images.hpp"
#include "strata/program/batch_read.hpp"
#include "strata/models/qwen38/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/common/state/prompt_checkpoint.hpp"
#include "strata/models/qwen38/stage_reserve.hpp"
#include "strata/base/gpu/gpu_report.hpp"
#include "strata/models/qwen38/prefill/share_rules.hpp"
#include "strata/base/platform/host_limits.hpp"
#include "strata/program/split_ring.hpp"
#include "strata/program/prefill_plan.hpp"
#include "strata/program/draft_kv_plan.hpp"
#include "strata/program/message_boundary.hpp"
#include "strata/common/spec/draft_vocab.hpp"
#include "strata/common/spec/window_rule.hpp"
#include "strata/common/spec/spec_start.hpp"
#include "strata/common/spec/window_planner.hpp"
#include "strata/models/qwen38/batch_groups.hpp"
#include "strata/program/par_counters.hpp"
#if defined(__linux__)
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif
#include "strata/program/pipe_probe.hpp"
#include "strata/models/qwen38/kernels/cvec.hpp"
#include "strata/common/state/progress.hpp"
#include "strata/base/gpu/device.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/common/artifact/gguf_reader.hpp"
#if defined(STRATA_HAVE_QWEN35)
#include "strata/models/qwen36/engine.hpp"
#endif
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <dbghelp.h>
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
// batch slot b <-> the main sequence: the running state (unless kv_only) and the first `upto` cells of K/V
bool window_slot_copy_main(bool into_slot, int slot, int64_t upto, bool kv_only, std::string& err);
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
#include <unordered_map>
#include <vector>

// STRATA_DF2_TAP_DUMP=FILE (research, off by default): every prompt tap batch the DFlash2 context takes is appended to FILE as
// int64 pos0, int64 rows, then rows x floats_per_row float32 (the tap rows [rows][taps][n_embd]), for the layer-major / chunk-major
// equivalence test (DF-08).  Nothing runs without the variable.
static void df2_tap_dump(const float* dev, int64_t rows, int64_t pos0, int64_t floats_per_row) {
    static const char* path = std::getenv("STRATA_DF2_TAP_DUMP");
    if (path == nullptr || path[0] == '\0' || rows <= 0) return;
    std::vector<float> h((size_t) (rows * floats_per_row));
    if (cudaMemcpy(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return;
    if (std::FILE* f = std::fopen(path, "ab")) {
        std::fwrite(&pos0, 8, 1, f);
        std::fwrite(&rows, 8, 1, f);
        std::fwrite(h.data(), sizeof(float), h.size(), f);
        std::fclose(f);
    }
}

// logprobs=K / allowed=...: one `LP` line for the token just emitted, from `ver`'s head
// logits for the window row it came from (before sampling, penalties and temperature).
// Format matches the SYCL port's LP line so one server-side reader serves both.
// Called only when the request asked (`req_logprobs >= 0`): a request without logprobs never reaches it, so the
// default path does no work for this and prints exactly what it printed before.
static void emit_lp_line(const strata::core::Verifier& ver, int t, int32_t token, int k,
                         const std::vector<int32_t>& allowed) {
    if (k < 0) return;
    static std::vector<float> row;
    row.resize((size_t) ver.vocab());
    if (!ver.copy_logits(t, row.data()) || token < 0 || (size_t) token >= row.size()) {
        strata::serve::print_response("LP nan\n");
        return;
    }
    float mx = -INFINITY;
    for (const float v : row) mx = std::max(mx, v);
    double se = 0.0;
    for (const float v : row) se += std::exp((double) v - (double) mx);
    const double lse = (double) mx + std::log(se);
    strata::serve::print_response("LP %.6f", (double) row[(size_t) token] - lse);
    if (!allowed.empty()) {
        // Only the ids the caller named; the server renormalises over them, which is what allowed_token_ids means.
        // The server refuses an id past the vocabulary with a 400 only when it can read the vocabulary size from the
        // tokenizer.  The engine's own vocab() is the authority: an id past it is left off the line, so the logits
        // are never read out of bounds.
        for (const int32_t id : allowed) {
            if (id == token || id < 0 || (int64_t) id >= ver.vocab()) continue;
            strata::serve::print_response(" %d:%.6f", (int) id, (double) row[(size_t) id] - lse);
        }
        strata::serve::print_response("\n");
        return;
    }
    if (k > 0) {
        static std::vector<int32_t> ord;
        ord.resize(row.size());
        for (size_t v = 0; v < row.size(); ++v) ord[v] = (int32_t) v;
        const int kk = std::min<int>(k, (int) row.size());
        std::partial_sort(ord.begin(), ord.begin() + kk, ord.end(),
                          [&](int32_t a, int32_t b) { return row[(size_t) a] > row[(size_t) b]; });
        for (int j = 0; j < kk; ++j)
            strata::serve::print_response(" %d:%.6f", ord[(size_t) j], (double) row[(size_t) ord[(size_t) j]] - lse);
    }
    strata::serve::print_response("\n");
}

namespace {
using strata::serve::print_response;
using namespace strata::qwen38;   // the model loader and the helpers it shares with the loops here
namespace nib = strata::core::nibble;
#if defined(__linux__)
// STRATA_CRASH_BT=1 (research): a fatal signal prints the call stack to stderr (the addresses go through addr2line -e <binary>)
char g_crash_path[512] = "";
void crash_bt_handler(int sig) {
    void* frames[48];
    const int n = backtrace(frames, 48);
    const char msg[] = "strata: fatal signal, call stack:\n";
    int fd = 2;
    if (g_crash_path[0] != '\0') {
        const int f = open(g_crash_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (f >= 0) fd = f;
    }
    (void) !write(fd, msg, sizeof msg - 1);
    backtrace_symbols_fd(frames, n, fd);
    signal(sig, SIG_DFL);
    raise(sig);
}
const bool crash_bt_installed = [] {
    const char* v = std::getenv("STRATA_CRASH_BT");
    if (v == nullptr || v[0] == '\0' || v[0] == '0') return false;
    if (v[0] != '1' || v[1] != '\0') std::snprintf(g_crash_path, sizeof g_crash_path, "%s", v);
    for (int sg : {SIGFPE, SIGSEGV, SIGBUS, SIGABRT}) signal(sg, crash_bt_handler);
    return true;
}();
#endif
// STRATA_PIPE_BYIELD_NODRAIN=1 (read once; off by default): with the pipelined groups, a BYIELD that reaches the main
// loop is dropped without draining the groups first.  A pipelined read never gives way (batch_read_piece), so the
// BYIELD is for a read that has ended, and the drain (and the refill after it) did nothing for it
bool pipe_byield_nodrain_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PIPE_BYIELD_NODRAIN");
        return v && v[0] != '0';
    }();
    return on;
}
// STRATA_ADMIT_MERGE_FIRST=1 (read once; off by default): a batch admission reads the last prompt tokens of its last
// windowed part (up to a window's max_t, the last prompt token included) and its first token in ONE window instead of
// the part's last windows and a window of the last token alone.  A row computes what it computes in any window while
// every expert is in VRAM, and row T-1 is the first window's position with its history and its Philox draw, so the
// token and the state are the same; the rows before it commit as the prompt windows commit them
bool admit_merge_first_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ADMIT_MERGE_FIRST");
        return v && v[0] != '0';
    }();
    return on;
}
// STRATA_CKPT_POOL=1 (read once; off by default): the serve loop keeps the running-state vectors of the conversation
// checkpoints it drops (evicted, pruned, a batch slot's old one) and saves the next checkpoint into them, so a save
// resizes nothing (no zero-fill of ~118 MB of fresh vectors) and a slot's copy reuses their pages.  Every byte is
// overwritten by the copy, so the checkpoints are the same
bool park_async_env() {   // STRATA_PARK_ASYNC (strata/core/conversation_snapshot.hpp)
    static const bool on = [] { const char* v = std::getenv("STRATA_PARK_ASYNC"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool ckpt_pool_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CKPT_POOL");
        return v && v[0] != '0';
    }();
    return on;
}
// STRATA_CKPT_POOL_READY=N (opt-in, N = 1..16, read once; only with the pool on, STRATA_CKPT_POOL): the pool keeps N ready
// checkpoints, each a running-state image with its pages in (~118 MB on Qwen3.8-Flash-Next), allocated on a background
// thread after any take.  Unset: the pool holds the dropped checkpoints only, as with STRATA_CKPT_POOL=1 before - that
// variable alone allocates nothing ahead of time and its value is not a count.
size_t ckpt_pool_ready_env() {
    static const size_t n = [] {
        const char* v = std::getenv("STRATA_CKPT_POOL_READY");
        const long k = v ? std::atol(v) : 0;
        return k >= 1 ? (size_t) std::min(16L, k) : (size_t) 0;
    }();
    return n;
}
// how many dropped checkpoints the pool keeps (a batch admission drops and takes about two): 2, or more when STRATA_CKPT_POOL_READY asks
size_t ckpt_pool_cap_env() { return std::max<size_t>(2, ckpt_pool_ready_env()); }
// STRATA_CKPT_THREADS=1 (read once; off by default): a layer split's checkpoint saves every stage's part at once, the
// first on the calling thread and each later stage's on a thread of its own device (kept for the process: the copies'
// private streams are per thread), instead of one stage after the other.  Each part is the same copy of the same state
bool ckpt_threads_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CKPT_THREADS");
        return v && v[0] != '0';
    }();
    return on;
}
// STRATA_SLOT_CKPT_MOVE=1 (read once; off by default): a batch admission MOVES its own turn checkpoint into the slot
// instead of copying it (~112 MB of host vectors).  The main session's chain no longer holds it, so the request path
// looks for it in the slots' checkpoints too: while the main session's cells still hold its tokens it is restored there
// as before (no K/V copy), else an active slot holding it gives it back like an idle one (a second request with the
// same history).  The same bytes are restored from another container, so the tokens are the same.  It keeps the
// dropped checkpoints' storage as STRATA_CKPT_POOL does (the slot's old checkpoint is the next save's)
bool slot_ckpt_move_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SLOT_CKPT_MOVE");
        return v && v[0] != '0';
    }();
    return on;
}
// STRATA_SLOT_COPY_BATCHSYNC=1 (read once; off by default): the direct slot copy (STRATA_SLOT_COPY_DIRECT) only queues
// its copies on each device's private copy stream and synchronizes every device once at the end, instead of waiting
// for each of its ~116 copies.  A streamed K/V state's residency refresh waits for its copies through an event.  The
// same copies in the same order on one stream, so the slot holds the same bytes
bool slot_copy_batchsync_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SLOT_COPY_BATCHSYNC");
        return v && v[0] != '0';
    }();
    return on;
}

// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
// A file-backed arena on AMD (STRATA_ARENA_MMAP; the engine raises ROCclr's pin-in-place threshold, see main): an
// adaptive swap copies one expert straight from the mapped file.  Through ROCclr's staging that copy blocks the host
// (decode windows of 150+ ms); instead the batch's pages are locked here for the async copies and unlocked when they
// have landed (unpin_blobs), so the release that follows can hand them back.  Neighbouring experts share a page and
// a copy from a range only PART of which is registered fails ("invalid argument"), so the page ranges are merged
// first and each merged span is registered once.  A span that does not register is copied through the staging
// path.  Portable: a layer split copies to either card.
// Whether pin_blobs locks anything (AMD on Linux with STRATA_ARENA_MMAP=1): the callers only gather the swaps' spans
// then - `blob()` is not free (it counts reads, and the file tier may assemble the blob), so the default runs skip it.
bool pin_blobs_on() {
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    static const bool on = std::getenv("STRATA_ARENA_MMAP") && std::getenv("STRATA_ARENA_MMAP")[0] == '1';
    return on;
#else
    return false;
#endif
}
// #884 (HIP, opt-in STRATA_HIP_ADAPT_KERNEL_COPY=1): an adaptive swap's host-to-device copy as a kernel on the refill stream
// instead of a copy-engine (SDMA) memcpy.  2x RX 6900 XT hung after a long prompt only with the MMQ prompt path, adaptive swaps
// AND SDMA together (any one off avoids it; the barrier packet that never passes is a ROCm-side problem, #884).  This keeps
// the swaps off SDMA without turning SDMA off for the prompt read (HSA_ENABLE_SDMA=0 costs 10-24% of it).  Needs the source to
// be mapped, 16-byte aligned memory (the registered arena); anything else takes the normal cudaMemcpyAsync.
cudaError_t adapt_copy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t st) {
#if defined(STRATA_USE_HIP)
    static const bool kernel = [] {
        const char* e = std::getenv("STRATA_HIP_ADAPT_KERNEL_COPY");
        const bool on = e != nullptr && e[0] == '1';
        if (on) std::fprintf(stderr, "strata: STRATA_HIP_ADAPT_KERNEL_COPY=1: adaptive swaps copy with a kernel, not SDMA (#884)\n");
        return on;
    }();
    if (kernel && bytes % 16 == 0 && ((uintptr_t) src & 15) == 0 && ((uintptr_t) dst & 15) == 0) {
        void* dsrc = nullptr;
        if (cudaHostGetDevicePointer(&dsrc, const_cast<void*>(src), 0) == cudaSuccess && dsrc != nullptr) {
            strata::kernels::copy_from_mapped((float*) dst, (const float*) dsrc, (int64_t) (bytes / 4), (void*) st);
            return cudaGetLastError();
        }
        (void) cudaGetLastError();   // not registered / not mapped: the ordinary copy
    }
#endif
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
}
int pin_blobs(std::vector<std::pair<uintptr_t, uintptr_t>> r, std::vector<void*>& live) {
    int failed = 0;   // ranges left pageable (a copy from them still works, through the driver's staging)
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    if (!pin_blobs_on() || r.empty()) return 0;
    for (auto& [a, e] : r) { a &= ~(uintptr_t) 4095; e = (e + 4095) & ~(uintptr_t) 4095; }
    std::sort(r.begin(), r.end());
    size_t k = 0;
    for (size_t i = 1; i < r.size(); ++i) {
        if (r[i].first <= r[k].second) r[k].second = std::max(r[k].second, r[i].second);
        else r[++k] = r[i];
    }
    r.resize(k + 1);
    for (const auto& [a, e] : r) {
        if (cudaHostRegister((void*) a, (size_t) (e - a), cudaHostRegisterReadOnly | cudaHostRegisterPortable) ==
            cudaSuccess)
            live.push_back((void*) a);
        else {
            (void) cudaGetLastError();
            ++failed;
        }
    }
#else
    (void) r; (void) live;
#endif
    return failed;
}
void unpin_blobs(std::vector<void*>& v) {
    for (void* p : v) (void) cudaHostUnregister(p);
    (void) cudaGetLastError();
    v.clear();
}

bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

using Clock = std::chrono::steady_clock;

// --pipeline-windows: every STRATA_PIPELINE_* test and tuning variable (THETA, FORCE_MISS, SWITCH, LOG, TRACE, DOOM_SKIP,
// LOOKUP_PON, PRESTAGE, AGREE, SNAP_OVERLAP, SPEC_DEPTH) is read only with STRATA_PIPELINE_DEBUG=1; without it the
// defaults apply.
static const char* pipe_dbg_env(const char* name) {
    static const bool on = [] { const char* v = std::getenv("STRATA_PIPELINE_DEBUG"); return v != nullptr && v[0] != 0 && v[0] != '0'; }();
    return on ? std::getenv(name) : nullptr;
}

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: the copies back are on its stream, and waited for before the refills are queued.
//
// With a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back must read the
// owning card's cache on that card's stream: `locate(layer)` names them (cache, stream, device; -1 = CUDA0).  Reading
// a later stage's slot number out of CUDA0's cache copied another expert's bytes into RAM - the resident mode with a
// split then computed some experts from the wrong weights after the first adaptive swaps (garbage output).
struct SwapHome {
    strata::core::ExpertCache* cache;
    cudaStream_t stream;
    int dev;
};
template <class Swap, class Locate>
bool resident_stage_swaps(strata::core::FileExpertSource& src, const std::vector<int32_t>& host_res,
                          int64_t n_expert, std::vector<Swap>& swaps, Locate locate) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const strata::core::OnDevice on(home.dev);
        nib::guard_raw(s.layer, "the adaptive tier's copy back (resident_stage_swaps)");
        if (const cudaError_t e = cudaMemcpyAsync(src.exchange_buffer(q, s.layer), home.cache->device_slot(slot),
                            (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), cudaMemcpyDeviceToHost,
                            home.stream); e != cudaSuccess) {
            std::fprintf(stderr, "strata: copy back of layer %d slot %d (device %d) failed: %s\n", (int) s.layer,
                         (int) slot, home.dev, cudaGetErrorString(e));
            return false;
        }
        bool seen = false;
        for (const SwapHome& u : used) seen = seen || u.stream == home.stream;
        if (!seen) used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& u : used) {
            const strata::core::OnDevice on(u.dev);
            if (const cudaError_t e = cudaStreamSynchronize(u.stream); e != cudaSuccess) {
                std::fprintf(stderr, "strata: copy back sync (device %d) failed: %s\n", u.dev, cudaGetErrorString(e));
                return false;
            }
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) {
                std::fprintf(stderr, "strata: stage_exchange refused layer %d in %d out %d\n", (int) x.layer,
                             (int) x.in, (int) x.out);
                return false;
            }
    }
    swaps.swap(kept);
    return true;
}

// what a refusal on the RAM guard says it had: "900 MiB available, source cgroup" (the smallest of MemAvailable, the
// cgroup limits above the engine and --memory-limit-mib; see strata/common/state/conversation_memory.hpp)
static std::string ram_note(const std::optional<strata::core::HostMemoryReading>& mem) {
    if (!mem) return "RAM telemetry unavailable";
    return std::to_string(mem->available >> 20) + " MiB available, source " + strata::core::memory_source_name(mem->source);
}

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in (#251)
// #1407: the file tier's activity as the OS counts it (Linux: major page faults and bytes read from storage). The #29
// watchdog treats a growing count as "the step is slow, not stuck": a prompt layer whose experts are re-read from the
// pack (low RAM, mmap experts) can take minutes while the heartbeat stands still. Zero where unavailable.
static uint64_t file_tier_activity() {
#if defined(__linux__)
    uint64_t total = 0;
    if (FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024] = {};
        const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        const char* p = std::strrchr(buf, ')');
        if (p != nullptr && n > 0) {
            // after "(comm)": state ppid pgrp session tty tpgid flags minflt cminflt majflt
            unsigned long long majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %llu", &majflt) == 1) total += majflt;
        }
    }
    if (FILE* f = std::fopen("/proc/self/io", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            unsigned long long v = 0;
            if (std::sscanf(line, "read_bytes: %llu", &v) == 1) total += v >> 20;   // MiB
        }
        std::fclose(f);
    }
    return total;
#elif defined(_WIN32)
    // #1984 #1971: Windows had no measure here, so a prompt waiting on mapped-file page-ins never got the I/O allowance.
    // Bytes read (MiB) plus page faults / 8 (Windows counts soft faults too; a stuck engine makes almost none).
    uint64_t total = 0;
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) total += io.ReadTransferCount >> 20;
    PROCESS_MEMORY_COUNTERS pmc{};
    // >> 3, not >> 6: a host that pages hard reads from the pagefile at ~200 faults a second (#2049: 190/s, one
    // thread waiting on one page at a time), which at >> 6 is 3 a second and never counted as busy
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) total += pmc.PageFaultCount >> 3;
    return total;
#else
    return 0;
#endif
}

std::string stage_text() {
    const strata::core::Progress& p = strata::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

#if !defined(_WIN32)
// #605: the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): 16 threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    std::map<std::string, int> where;
    int waiting = 0, threads = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        ++threads;
        std::ifstream st(e.path() / "stat");
        std::string line;
        std::getline(st, line);
        const size_t rp = line.rfind(')');
        if (rp == std::string::npos || rp + 2 >= line.size() || line[rp + 2] != 'D') continue;
        ++waiting;
        std::ifstream w(e.path() / "wchan");
        std::string wc;
        std::getline(w, wc);
        ++where[wc.empty() || wc == "0" ? "?" : wc];
    }
    if (threads == 0) return;
    std::string list;
    for (const auto& [name, n] : where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", waiting, threads, list.c_str(),
                 waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (#605: --ple-io ram), or a drive too slow for the reads asked of it" : "");
}
#endif

// #2824: what the drive under `path` reads at, GB/s, for the prompt planner (it prices the file set a chunk re-reads).  Linux:
// rotational disk 0.15, NVMe 2.5, other 0.45 (SATA); elsewhere 1.5.  STRATA_SSD_GBPS overrides.  `why` says where it came from.
double drive_read_gbps(const std::string& path, std::string& why) {
    if (const char* e = std::getenv("STRATA_SSD_GBPS"); e != nullptr && std::atof(e) > 0.0) {
        why = "STRATA_SSD_GBPS";
        return std::atof(e);
    }
#if !defined(_WIN32)
    std::string dev;
    const int rot = on_rotational_disk(path, dev);
    if (rot == 1) { why = dev + " is a rotational disk"; return 0.15; }
    if (rot == 0 && dev.rfind("nvme", 0) == 0) { why = dev + " is NVMe"; return 2.5; }
    if (rot == 0) { why = dev + " is a SATA-class SSD (assumed)"; return 0.45; }
#endif
    why = "assumed";
    return 1.5;
}

// --pipeline-windows: what the watchdog prints about the pipelined decode loop (set while it runs)
struct PlDiag {
    strata::core::Verifier* v[2][2] = {};
    std::atomic<int> a_seq{0}, a_bits{0}, b_bits{0}, d_bits{0}, chain_kind{0}, doomed{0}, ending{0}, chain_live{0};
    std::atomic<long long> iters{0};
};
PlDiag g_pl_diag;
void pl_diag_print(std::FILE* f) {
    PlDiag& d = g_pl_diag;
    auto bits = [](int b, char* o) {
        std::snprintf(o, 16, "%c%c%c%c%c%c", b & 1 ? 'R' : '-', b & 2 ? 'L' : '-', b & 4 ? 'F' : '-', b & 8 ? 'C' : '-',
                      b & 16 ? 'S' : '-', b & 32 ? 'V' : '-');
        return (const char*) o;
    };
    char ba[16], bb[16], bd[16];
    std::fprintf(f, "  pipelined loop: %lld iterations; A seq %d [%s] B [%s] D [%s] (Ready Launched Finished Committed "
                    "S1-launched S1-done); chain kind %d live %d; doomed %d ending %d\n", d.iters.load(), d.a_seq.load(),
                 bits(d.a_bits.load(), ba), bits(d.b_bits.load(), bb), bits(d.d_bits.load(), bd), d.chain_kind.load(),
                 d.chain_live.load(), d.doomed.load(), d.ending.load());
    const char* names[2][2] = {{"stage 0 even", "stage 0 odd"}, {"stage 1 even", "stage 1 odd"}};
    for (int st = 0; st < 2; ++st)
        for (int par = 0; par < 2; ++par)
            if (d.v[st][par] != nullptr) d.v[st][par]->diag_pipelined(f, names[st][par]);
}

/// #2065: the VRAM free when the current request began (-1 before the first); the stall report prints it, because a
/// card this full is the one measured cause of a window that never finishes (the driver pages GPU memory).
std::atomic<int64_t> g_req_vram_free_mib{-1};
constexpr int64_t kVramSafeMib =
#ifdef _WIN32
    1024;   // WDDM pages instead of failing (docs/DETAILS.md, the sysmem fallback): the line #2065 fell below
#else
    512;
#endif

#if defined(_WIN32)
// STRATA_STALL_STACKS=1: the call stack of every other thread, in the stall report.  The other threads are suspended while
// their stacks are unwound with the unwind tables (RtlLookupFunctionEntry / RtlVirtualUnwind: no heap, no loader lock, no
// stdio), into static storage; the threads run again before anything is looked up or printed.  Frames in strata.exe are then
// named from its PDB (dbghelp, when the PDB is beside the exe); other modules are printed as module+offset.
namespace stall_stacks {
constexpr int kMaxThreads = 256, kMaxFrames = 32, kMaxMods = 128;
struct Mod { uint64_t base; uint64_t size; char name[64]; };
struct Stack { DWORD tid; int n; uint64_t pc[kMaxFrames]; };
static Mod g_mods[kMaxMods];
static Stack g_stacks[kMaxThreads];
static int g_nmods = 0, g_nstacks = 0;

static void collect() {
    g_nmods = g_nstacks = 0;
    HMODULE hm[kMaxMods];
    DWORD need = 0;
    if (K32EnumProcessModules(GetCurrentProcess(), hm, sizeof hm, &need)) {
        const int n = std::min<int>(kMaxMods, (int) (need / sizeof(HMODULE)));
        for (int i = 0; i < n; ++i) {
            MODULEINFO mi{};
            char path[MAX_PATH] = {};
            if (!K32GetModuleInformation(GetCurrentProcess(), hm[i], &mi, sizeof mi)) continue;
            K32GetModuleFileNameExA(GetCurrentProcess(), hm[i], path, MAX_PATH);
            const char* base = path;
            for (const char* p = path; *p; ++p) if (*p == '\\' || *p == '/') base = p + 1;
            Mod& m = g_mods[g_nmods++];
            m.base = (uint64_t) (uintptr_t) mi.lpBaseOfDll;
            m.size = mi.SizeOfImage;
            std::snprintf(m.name, sizeof m.name, "%s", base);
        }
    }
    const DWORD me = GetCurrentThreadId(), pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te{};
    te.dwSize = sizeof te;
    static DWORD tids[kMaxThreads];
    int nt = 0;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid && te.th32ThreadID != me && nt < kMaxThreads) tids[nt++] = te.th32ThreadID;
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    for (int i = 0; i < nt; ++i) {
        HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tids[i]);
        if (th == nullptr) continue;
        if (SuspendThread(th) == (DWORD) -1) { CloseHandle(th); continue; }
        Stack& s = g_stacks[g_nstacks];
        s.tid = tids[i];
        s.n = 0;
        alignas(16) CONTEXT ctx;
        std::memset(&ctx, 0, sizeof ctx);
        ctx.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(th, &ctx)) {
            for (int k = 0; k < kMaxFrames && ctx.Rip != 0; ++k) {
                s.pc[s.n++] = ctx.Rip;
                DWORD64 base = 0;
                RUNTIME_FUNCTION* fe = RtlLookupFunctionEntry(ctx.Rip, &base, nullptr);
                if (fe == nullptr) {                       // a leaf: the return address is at the stack pointer
                    if (ctx.Rsp == 0) break;
                    ctx.Rip = *(const DWORD64*) ctx.Rsp;
                    ctx.Rsp += 8;
                } else {
                    PVOID hd = nullptr;
                    DWORD64 est = 0;
                    RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, ctx.Rip, fe, &ctx, &hd, &est, nullptr);
                }
            }
            ++g_nstacks;
        }
        ResumeThread(th);
        CloseHandle(th);
    }
}

static void print(std::FILE* f) {
    using SymInitializeFn = BOOL(WINAPI*)(HANDLE, const char*, BOOL);
    using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, DWORD64*, SYMBOL_INFO*);
    using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
    SymFromAddrFn sym_from = nullptr;
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        auto init = (SymInitializeFn) GetProcAddress(dbg, "SymInitialize");
        auto setopt = (SymSetOptionsFn) GetProcAddress(dbg, "SymSetOptions");
        sym_from = (SymFromAddrFn) GetProcAddress(dbg, "SymFromAddr");
        if (setopt) setopt(0x2 /*SYMOPT_UNDNAME*/ | 0x10 /*SYMOPT_DEFERRED_LOADS*/ | 0x20000 /*SYMOPT_NO_PROMPTS*/);
        if (init == nullptr || !init(GetCurrentProcess(), nullptr, TRUE)) sym_from = nullptr;
    }
    std::fprintf(f, "  thread stacks (STRATA_STALL_STACKS=1; innermost first; strata.exe frames named from its PDB when it is beside the exe):\n");
    for (int i = 0; i < g_nstacks; ++i) {
        const Stack& s = g_stacks[i];
        std::fprintf(f, "  thread %lu:\n", (unsigned long) s.tid);
        for (int k = 0; k < s.n; ++k) {
            const uint64_t pc = s.pc[k];
            const Mod* m = nullptr;
            for (int j = 0; j < g_nmods; ++j) if (pc >= g_mods[j].base && pc < g_mods[j].base + g_mods[j].size) { m = &g_mods[j]; break; }
            char nm[200] = {};
            if (m != nullptr && sym_from != nullptr && std::strcmp(m->name, "strata.exe") == 0) {
                alignas(8) char buf[sizeof(SYMBOL_INFO) + 160] = {};
                SYMBOL_INFO* si = (SYMBOL_INFO*) buf;
                si->SizeOfStruct = sizeof(SYMBOL_INFO);
                si->MaxNameLen = 160;
                DWORD64 disp = 0;
                if (sym_from(GetCurrentProcess(), pc, &disp, si)) std::snprintf(nm, sizeof nm, " %s+0x%llx", si->Name, (unsigned long long) disp);
            }
            if (m != nullptr) std::fprintf(f, "    %s+0x%llx%s\n", m->name, (unsigned long long) (pc - m->base), nm);
            else std::fprintf(f, "    0x%llx\n", (unsigned long long) pc);
        }
    }
}
}  // namespace stall_stacks
#endif

void stall_report(std::FILE* f, uint64_t layers_during) {
#if defined(_WIN32)
    const bool want_stacks = [] { const char* e = std::getenv("STRATA_STALL_STACKS"); return e != nullptr && e[0] == '1'; }();
    if (want_stacks) stall_stacks::collect();   // first: the stacks as they were when the stall was noticed
#endif
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "strata serve: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, stage_text().c_str(),
                 (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    if (const int64_t vf = g_req_vram_free_mib.load(); vf >= 0) {
        std::fprintf(f, "  VRAM free when this request began: %lld MiB%s\n", (long long) vf,
                     vf >= kVramSafeMib ? "" : " - LOW: a card this full can page GPU memory to system RAM (decode "
                     "falls to a few tok/s, then a window never finishes, #2065); a larger --vram-reserve-mib, a "
                     "smaller --max-context, or 'Prefer No Sysmem Fallback' for the engine in the NVIDIA control panel "
                     "(docs/DETAILS.md) leaves room");
    }
#if defined(_WIN32)
    if (want_stacks) stall_stacks::print(f);
#endif
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        if (auto fn = strata::core::diag_pipeline_fn().load()) fn(f);
#if !defined(_WIN32)
        disk_wait_threads(f);
#endif
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "strata-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000 | 0x100 | 0x4 | 0x800 | 0x40;   // ThreadInfo, ProcessThreadData, HandleData, FullMemoryInfo, IndirectlyReferencedMemory; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

/// #463's A/B: STRATA_ADAPT_NOWAIT=1 lets a verify window start before the adaptive tier's copies have landed (0.1.37)
bool adapt_nowait() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_NOWAIT"); return e && e[0] == '1'; }();
    return v;
}

/// #463 without the stall: a verify window takes the adaptive tier's previous swaps only once they are this many windows
/// old, and then waits for their copies (which have landed by then).  The window that first computes a swapped-in expert on
/// the GPU depends only on the window count, as with #463's wait, so the answers are as reproducible.  Default 1 = #463 and
/// 0.1.39 (the very next window waits for the copies: ~4.6 ms per window on an RTX 2060 SUPER over PCIe 3.0 x8); lag 2 is
/// opt-in (STRATA_ADAPT_LAG=2).  The batch paths, --pipeline-windows and the peer tier are not lagged.  (#764)
int adapt_lag() {
    static const int v = [] { const char* e = std::getenv("STRATA_ADAPT_LAG"); return e ? std::max(1, std::atoi(e)) : 1; }();
    return v;
}

/// STRATA_ADAPT_OVERLAP=1: the adaptive tier's copies land while the next decode window already runs.  The window
/// sees the swapped-in experts as resident from its first layer, exactly as after #463's wait (so the tokens are the same
/// ones); what changes is only when the bytes arrive: the copies go in layer order and the verify window holds a layer's
/// plan back until that layer's copies have landed (AdaptGate in strata/models/qwen38/verify.hpp).  The ~10 ms a
/// 96-swap round took at the next window's start (PCIe 3.0 x16, Q2_0) is then hidden behind the window.  Opt-in on
/// every backend (=1 turns it on); the startup log names the state, and why when it cannot run.
bool adapt_overlap() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_ADAPT_OVERLAP");
        // Opt-in on every backend (#2010).  Windows CUDA since 0.1.43: an RTX 5070 decoded Q2_0 2.6% slower with it
        // (story decode, -2.1% code), IQ3_XXS no faster.  Linux CUDA was on by default from 0.1.43 (same tokens, +5.5..8.3%
        // decode on an A4000, +4.4% on a P100) until an RTX 5090 Laptop (WSL2, Core Ultra 9 275HX, native IQ3_XXS pack,
        // --expert-cache-per-layer) decoded 21% slower with it: 84.6 vs 106.6 tok/s medians of 5 generations of 256
        // tokens from one build, the slowest of the fast run above the fastest of the slow one.
        return e && e[0] == '1';
    }();
    return v;
}

/// STRATA_ADAPT_FETCH (opt-in): 1 = the PCIe share also admits into the VRAM tier (ExpertDispatch::fetch_admit), beside
/// the tier's own copies from RAM; 2 = instead of them (the tier's counts still decay).  STRATA_ADAPT_FETCH_MIN and
/// STRATA_ADAPT_FETCH_MARGIN: the admission's thresholds (default 1 and 0.5).
int adapt_fetch() {
    static const int v = [] { const char* e = std::getenv("STRATA_ADAPT_FETCH"); return e ? std::max(0, std::min(2, std::atoi(e))) : 0; }();
    return v;
}

/// ExpertDispatch::evict_sync where the tier does not wait for its copies; STRATA_ADAPT_EVICT_SYNC=0 turns it off.
bool adapt_evict_sync() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_EVICT_SYNC"); return e == nullptr || std::atoi(e) != 0; }();
    return v;
}

/// STRATA_ADAPT_CHECKRES=1: before a verify window, the device's residency table read back into `back` and
/// compared with the host's (the pool's plan); the mismatched entries, the first ones reported.
int64_t res_check(const int32_t* d_res, const std::vector<int32_t>& host_res, std::vector<int32_t>& back,
                  int64_t n_expert, int64_t window) {
    if (cudaMemcpy(back.data(), d_res, host_res.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "strata: res check: the read-back failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    static int printed = 0;
    int64_t bad = 0;
    for (size_t i = 0; i < host_res.size(); ++i)
        if (back[i] != host_res[i]) {
            ++bad;
            if (printed < 20) {
                ++printed;
                std::fprintf(stderr, "strata: res check: window %lld layer %lld expert %lld: host %d, device %d\n",
                             (long long) window, (long long) (i / (size_t) n_expert), (long long) (i % (size_t) n_expert),
                             (int) host_res[i], (int) back[i]);
            }
        }
    return bad;
}

/// The pre-launch hook's patch: the changed entries of the host's residency table (`idx`) as (index, entry) pairs into
/// the mapped `pairs` (room for `cap`); their count, or -1 when they do not fit (then the whole table goes up).
int64_t res_patch_pairs(const std::vector<int32_t>& idx, const std::vector<int32_t>& host_res, int32_t* pairs, size_t cap) {
    if (pairs == nullptr || idx.size() > cap) return -1;
    for (size_t k = 0; k < idx.size(); ++k) {
        pairs[2 * k] = idx[k];
        pairs[2 * k + 1] = host_res[(size_t) idx[k]];
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return (int64_t) idx.size();
}

/// Mapped host memory for res_patch_pairs (portable: every card of a layer split reads it), or null.  The kernels read
/// it at the host's address: each of `devs` (-1: the current device) must map it there, else the whole table goes up.
int32_t* res_pairs_alloc(size_t cap, const std::vector<int>& devs) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, cap * 2 * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    for (int dev : devs) {
        const strata::core::OnDevice on(dev);
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess || d != p) {
            (void) cudaGetLastError();
            cudaFreeHost(p);
            return nullptr;
        }
    }
    return (int32_t*) p;
}

/// --adapt-async: one background job at a time (the asynchronous adaptive tier's copies and memcpys).  `post` hands
/// it over, `idle` says whether it has finished; the owner never posts while a job runs.
class JobThread {
public:
    JobThread() : th_([this] { loop(); }) {}
    ~JobThread() {
        { std::lock_guard<std::mutex> lk(mu_); quit_ = true; }
        cv_.notify_one();
        th_.join();
    }
    JobThread(const JobThread&) = delete;
    JobThread& operator=(const JobThread&) = delete;
    void post(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = std::move(f);
            has_ = true;
            done_.store(false, std::memory_order_release);
        }
        cv_.notify_one();
    }
    bool idle() const { return done_.load(std::memory_order_acquire); }
    void wait() const { while (!idle()) std::this_thread::yield(); }
private:
    // The second logical processor of the last physical core that has two: an SMT sibling no pool worker or host loop
    // is pinned to (the pool takes the first of every core but the first, the host the first core), or -1.
    static int smt_spare_cpu() {
#if defined(_WIN32)
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len == 0) return -1;
        std::vector<char> buf(len);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(),
                                              &len))
            return -1;
        int best = -1, core = 0;
        for (const char* p = buf.data(); p < buf.data() + len;
             p += ((const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size, ++core) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            const GROUP_AFFINITY& ga = e->Processor.GroupMask[0];
            int seen = 0;
            for (int bit = 0; bit < 64; ++bit)
                if ((ga.Mask & (1ull << bit)) && ++seen == 2 && core > 0) best = ga.Group * 64 + bit;
        }
        return best;
#else
        return -1;
#endif
    }
    void loop() {
        // the job's memcpys stay off the host's core and the pool's: by default the spare SMT sibling of the last
        // two-thread core (Windows); STRATA_ADAPT_JOB_CPU=<logical cpu>, or -1 for no pinning.  Without either, --aux-cpus
        // (off by default) puts the thread on the spare CPUs; an explicit STRATA_ADAPT_JOB_CPU is the user's word over it.
        strata::platform::name_this_thread("strata-adapt-job");
        const char* c = std::getenv("STRATA_ADAPT_JOB_CPU");
        const int cpu = c != nullptr ? std::atoi(c) : smt_spare_cpu();
        if (cpu >= 0) (void) strata::kernels::cpu::pin_current_thread(cpu);
        if (c != nullptr) strata::aux_cpus::note_owned_thread();
        else if (cpu < 0) strata::aux_cpus::pin_current_thread();
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return has_ || quit_; });
                if (!has_) return;
                f = std::move(job_);
                has_ = false;
            }
            f();
            done_.store(true, std::memory_order_release);
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool has_ = false, quit_ = false;
    std::atomic<bool> done_{true};
    std::thread th_;   // last: started once the members above exist
};

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = strata::core::ConversationImageKey;
using ConvCheckpoint = strata::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = strata::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    strata::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint save: %s\n", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// #3350: a conversation checkpoint is an optimization, like parking.  After a failed save: true when the failure was
/// only the copy's memory (a host or driver allocation, which does not stick), so the caller goes on without that
/// checkpoint; false when the GPU itself is unwell (a sticky error comes back from the synchronize), which still ends
/// the request as before.
bool checkpoint_failure_is_survivable() {
    (void) cudaGetLastError();   // clears a failed allocation; a sticky error is not cleared and is returned again below
    return cudaDeviceSynchronize() == cudaSuccess;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint restore: %s\n", error.c_str());
    return false;
}

/// STRATA_CKPT_POOL: the storage of dropped checkpoints, for the next save or slot copy to reuse.  `give` keeps a
/// checkpoint's vectors (its tokens, pictures and marks cleared), `take` hands one back (an empty checkpoint when none
/// is kept).  At most `cap` are kept (a batch admission drops and takes about two; STRATA_CKPT_POOL=N), the rest are
/// freed as before.
class CheckpointPool {
public:
    explicit CheckpointPool(bool on, size_t cap = 2) : on_(on), cap_(cap) {}
    bool on() const { return on_; }
    size_t cap() const { return cap_; }
    size_t held() { std::lock_guard<std::mutex> lk(mu_); return free_.size(); }
    void give(ConvCheckpoint&& c) {
        if (!on_ || c.gdn.empty()) return;
        strata::core::conversation_checkpoint_join(c.gdn.data());
        clear_marks(c);
        std::lock_guard<std::mutex> lk(mu_);
        if (free_.size() < cap_) free_.push_back(std::move(c));
    }
    ConvCheckpoint take() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!on_ || free_.empty()) return ConvCheckpoint{};
        ConvCheckpoint c = std::move(free_.back());
        free_.pop_back();
        return c;
    }
    /// v.erase(remove_if(v, drop)) with the dropped ones given to the pool (the kept ones keep their order)
    template <class Drop> void erase_if(std::vector<ConvCheckpoint>& v, Drop drop) {
        size_t w = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (drop(static_cast<const ConvCheckpoint&>(v[i]))) { give(std::move(v[i])); continue; }
            if (w != i) v[w] = std::move(v[i]);
            ++w;
        }
        v.erase(v.begin() + (std::ptrdiff_t) w, v.end());
    }
private:
    static void clear_marks(ConvCheckpoint& c) {
        c.ids.clear(); c.imgs.clear(); c.used = 0; c.pinned = false; c.dev_tag = 0;
        for (ConvCheckpoint& part : c.stage_parts) clear_marks(part);
    }
    bool on_;
    size_t cap_;
    std::mutex mu_;
    std::vector<ConvCheckpoint> free_;
};

/// STRATA_CKPT_THREADS: runs job(0) on the calling thread and job(k), k = 1..n-1, on persistent thread k at the same
/// time, and returns when all have.  The threads live as long as this object: a short-lived thread would leave its
/// private copy stream behind on every device it used (conversation_detail::copy_nonblocking).  A thread inherits the
/// host thread's one CPU (aux_cpus.hpp), where the copies would only take turns, so each first goes to the aux CPUs
/// (--aux-cpus) or else to every CPU the process may use.  One caller at a time; a job does not throw.
class StageThreads {
public:
    StageThreads() = default;
    StageThreads(const StageThreads&) = delete;
    StageThreads& operator=(const StageThreads&) = delete;
    ~StageThreads() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        go_.notify_all();
        for (std::thread& t : threads_) t.join();
    }
    void run(size_t n, const std::function<void(size_t)>& job) {
        if (n == 0) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            while (threads_.size() + 1 < n) {
                const size_t k = threads_.size() + 1;
                threads_.emplace_back([this, k] { loop(k); });
            }
            job_ = &job;
            n_ = n;
            pending_ = n - 1;
            ++round_;
        }
        go_.notify_all();
        job(0);
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk, [&] { return pending_ == 0; });
        job_ = nullptr;
    }
    static void spread_current_thread() {
#if defined(__linux__)
        // wide first: the mask the aux placement keeps as this thread's own is the one restore() puts back (a request
        // with aux_cpus=0), and the host's one CPU must not be it
        cpu_set_t all;
        CPU_ZERO(&all);
        for (int c = 0; c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
        (void) sched_setaffinity(0, sizeof all, &all);   // the kernel keeps the CPUs the process may use
        strata::aux_cpus::pin_current_thread();          // (a no-op while the feature is off)
#endif
    }
private:
    void loop(size_t k) {
        strata::platform::name_this_thread("strata-ckpt", (int) k);
        spread_current_thread();
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            go_.wait(lk, [&] { return stop_ || round_ != seen; });
            if (stop_) return;
            seen = round_;
            if (k >= n_) continue;   // a round with fewer jobs than threads
            const std::function<void(size_t)>* job = job_;
            lk.unlock();
            (*job)(k);
            lk.lock();
            if (--pending_ == 0) done_.notify_all();
        }
    }
    std::mutex mu_;
    std::condition_variable go_, done_;
    std::vector<std::thread> threads_;
    const std::function<void(size_t)>* job_ = nullptr;
    size_t n_ = 0, pending_ = 0;
    uint64_t round_ = 0;
    bool stop_ = false;
};


/// The ceiling of the CPU/PCIe miss share the auto rules use (PcieAuto, its start value and the later stages of a layer split) and the pool factor k
/// of the rule (strata/core/pcie_share_rule.hpp).  0.2.10.1 defaults on a discrete card: ceiling 0.95 and k falling with the pool rate (2 workers
/// 31 -> 63 tok/s, 4 workers 56 -> 79, a strong pool unchanged: k is 1.85 from 30 GB/s of expert bytes and the shares stay below 0.60); an integrated
/// GPU (Strix Halo and the other APUs, whose "link" is the shared memory) and STRATA_DEFAULTS=026 keep 0.60 and 1.85.  STRATA_PCIE_SHARE_MAX /
/// STRATA_PCIE_K_RULE=0 or 1 set either by hand.
static double pcie_share_max() {
    static const double v = strata::pcie_share_ceiling(spill_discrete_gpu(), strata::qwen_defaults_026(), std::getenv("STRATA_PCIE_SHARE_MAX"));
    return v;
}
static double pcie_pool_k(double r) {
    static const bool on = strata::pcie_k_rule_default(spill_discrete_gpu(), strata::qwen_defaults_026(), std::getenv("STRATA_PCIE_K_RULE"));
    return strata::pcie_pool_k(r, on);
}

// 0.1.42, CUDA single-GPU serve, no --pcie-frac: the share refined from what the CPU pool measures in the first decode
// windows.  The balance share of the missed experts is link / (link + CPU rate).  The link is the start-up probe; the CPU
// rate is the expert bytes over the pool's wall time per missed expert (ExpertDispatch::ms_run / multi_misses), which the
// engine already times.  The best shares measured with the route tail skip on (RTX 3060 0.15-0.20, Tesla P100 0.20, RTX
// A4000 flat from 0.15 to 0.40; the 0.1.41 rule gave 0.55, 0.29, 0.30) lie within one 0.05 step of link / (link + 1.85 x pool rate).
// Three updates at most (after 48, then 96 and 96 more windows), each rounded to 0.05, so a noisy window cannot move
// the share by less than a step and it stops moving after the third.  STRATA_PCIE_FRAC_DEFAULT=old keeps the 0.1.41 rule.
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;
    void mark(double ms_run, int64_t jobs) { mark_ms = ms_run; mark_jobs = jobs; }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double f = std::min(pcie_share_max(), std::max(0.10, std::round(link_gbps / (link_gbps + pcie_pool_k(r) * r) * 20.0) / 20.0));
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace

// STRATA_SPEC_PROB=1 with the default (pick) gate: a draft that fails the confidence gate is still a proposal attempt, and the
// verifier gets the gated measure (common/spec/gated_proposal.hpp).  That is exact only where the window choice cannot see a draw or a
// timing that the next rounds draw again.  Where it cannot be kept, say why and return false: the caller keeps the older rule
// (a warning, not a refusal).
static bool gated_proposal_usable(bool serial_single_device, int lookup_chain, int adapt_swaps, bool pcie_share, bool own_window_rule,
                                  int window, double min_p) {
    const char* why = nullptr;
#if defined(STRATA_GPU_METAL)
    why = "the Metal engine has no gated verifier";
#endif
    if (why == nullptr && !serial_single_device) why = "it needs serial decode on one device";
    if (why == nullptr && (lookup_chain != 0 || adapt_swaps != 0 || pcie_share))
        why = "it needs --lookup-chain 0, --adapt-swaps 0 and --pcie-frac 0";
    if (why == nullptr && own_window_rule) why = "STRATA_SPEC_COST, STRATA_SPEC_T_CYCLE, DFlash2 and --spec-corrupt change the window by their own rule";
    if (why == nullptr && window > strata::core::gated::kRows) why = "the window is wider than 8 rows";
    if (why == nullptr && (!std::isfinite(min_p) || min_p < 0.0)) why = "--spec-min-p must be a finite number >= 0";
    if (why == nullptr) return true;
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr, "strata: STRATA_SPEC_PROB=1 keeps the older acceptance rule in this run (%s). That rule's output "
                             "distribution is off when the confidence gate fails; STRATA_SPEC_PROB_GATE=top is exact here.\n", why);
    }
    return false;
}

// A draft at MTP cell c is sampled at counter c+1. The next target starts
// there and needs one final cell for its bonus. Cap BEFORE any proposal draw;
// a zero budget therefore means no attempt, never a discarded failed NULL.
class GatedDraftLimit {
public:
    GatedDraftLimit(strata::core::MtpDrafter& mtp, bool active, int configured_mtp,
                    int64_t next_target_pos, int64_t context_end)
        : mtp_(mtp), active_(active), configured_(configured_mtp - 1) {
        if (active_) {
            const int64_t room = std::max<int64_t>(0, context_end - next_target_pos);
            mtp_.set_max_drafts((int) std::min<int64_t>(configured_, std::max<int64_t>(0, room - 1)));
        }
    }
    ~GatedDraftLimit() { if (active_) mtp_.set_max_drafts(configured_); }
    GatedDraftLimit(const GatedDraftLimit&) = delete;
    GatedDraftLimit& operator=(const GatedDraftLimit&) = delete;
private:
    strata::core::MtpDrafter& mtp_;
    bool active_;
    int configured_;
};

// Prepare one gated window from the last successful producer result. Context room
// is fixed before consulting D; the producer was already capped before drawing.
// No max_new/output-cap shortening: the ordinary emission loop owns that limit.
// Sets `w`'s rows and draft source (MTP or lookup) to the packet's.
static bool prepare_gated_window(strata::core::gated::Packet& packet, strata::spec::Window& w,
                                 const strata::core::MtpDrafter::DraftResult& generation,
                                 strata::spec::WindowPlanner& wplan, int room, int vocab, float threshold,
                                 const int32_t* q, const int32_t* drafts, const float* scores, std::string& err) {
    if (room < 1 || generation.budget < 0 || generation.generated < 0 ||
        generation.generated > generation.budget || generation.budget >= room) {
        err = "gated proposal: invalid context room or producer result";
        return false;
    }
    const strata::core::gated::RootChoice root = wplan.choose_gated_root(w, room);
    if (!strata::core::gated::make_packet(packet, q, drafts, scores, generation.generated, generation.budget,
                                         room, vocab, threshold, w.first, root)) {
        err = std::string("gated proposal: ") + packet.error;
        return false;
    }
    w.T = packet.T;
    w.lookup = packet.from_sfx;
    return true;
}

// #4350: after a fatal GPU error (an illegal memory access, a launch failure) the device context cannot be used again, and unwinding the
// destructors on it can hang until the stall watchdog (#224).  A failed decode window / commit therefore ends the engine at once when the
// runtime holds a sticky error, with "ERR fatal GPU error: ..." as its last stdout line: the server fails the request in flight with a 503
// and starts the engine again on the next request, like any engine death.
// STRATA_INJECT_FATAL=N (a test hook, off when unset): the Nth decode window fails with the text of a real fatal error, the same way.
[[noreturn]] static void exit_gpu_fatal(const std::string& what) {
    std::fprintf(stderr, "strata serve: fatal GPU error (%s): the engine ends so that the server can start it again\n", what.c_str());
    std::fflush(stderr);
    print_response("ERR fatal GPU error: %s\n", what.c_str());
    std::fflush(stdout);
    std::_Exit(3);
}
static void exit_if_gpu_fatal(const std::string& err) {
    static const long inject = [] { const char* v = std::getenv("STRATA_INJECT_FATAL"); return v != nullptr ? std::atol(v) : 0L; }();
    static long windows = 0;
    if (inject > 0 && ++windows == inject) {
        // STRATA_INJECT_FATAL_ONCE=<file>: fires once across restarts (the file is made when it fires; while it exists nothing is injected)
        const char* once = std::getenv("STRATA_INJECT_FATAL_ONCE");
        bool fire = true;
        if (once != nullptr && *once) {
            if (std::FILE* t = std::fopen(once, "rb")) { std::fclose(t); fire = false; }
            else if (std::FILE* t2 = std::fopen(once, "wb")) std::fclose(t2);
        }
        if (fire) exit_gpu_fatal("injected (STRATA_INJECT_FATAL): an illegal memory access was encountered");
    }
    const cudaError_t sticky = cudaPeekAtLastError();
    if (sticky != cudaSuccess) {
        // (by the runtime's own text, so that CUDA, HIP and SYCL builds read it the same way)
        const std::string text = cudaGetErrorString(sticky);
        static const char* const gone[] = {"illegal memory access", "illegal address", "unspecified launch failure", "launch failure",
                                           "hardware exception", "illegal instruction", "misaligned address", "uncorrectable", "launch timed out",
                                           "device-side assert"};
        bool context_gone = false;
        for (const char* w : gone)
            if (text.find(w) != std::string::npos) context_gone = true;
        if (context_gone) exit_gpu_fatal(err.empty() ? std::string(cudaGetErrorString(sticky)) : err);
    }
}

#if defined(STRATA_USE_HIP) && defined(_WIN32)
static int engine_main(int argc, char** argv) {   // main() is at the end of this file (#3431)
#else
int main(int argc, char** argv) {
#endif
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    strata::platform::restore_normal_io_priority();   // #4733: not the low disk priority of the program that started this one
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    {   // 0.2.1.2: hipblas / libhipblaslt / rocblas are delay-loaded (CMakeLists.txt) and live in <exe folder>\rocm\bin, which only the
        // launcher puts on PATH; make the folder part of this process's DLL search, so `strata.exe --version` run straight from the
        // package folder works.  The HIP runtime and the C++ runtime sit beside the exe and are found without it.
        wchar_t exe[MAX_PATH * 2] = {};
        const DWORD n = GetModuleFileNameW(nullptr, exe, (DWORD) (sizeof(exe) / sizeof(exe[0])));
        if (n > 0 && n < sizeof(exe) / sizeof(exe[0])) {
            std::wstring dir(exe, n);
            const size_t cut = dir.find_last_of(L"\\/");
            if (cut != std::wstring::npos) {
                dir.resize(cut);
                dir += L"\\rocm\\bin";
                if (GetFileAttributesW(dir.c_str()) != INVALID_FILE_ATTRIBUTES) SetDllDirectoryW(dir.c_str());
            }
        }
    }
#endif
    // `strata --version` / `-V` (also `strata generate --version`): the version and nothing else, before any GPU is touched
    for (int i = 1; i < argc && i < 3; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
            std::printf("strata %s\n", STRATA_VERSION);
            return 0;
        }
        // `strata --families`: the GGUF architectures this build runs (setup asks before it installs one)
        if (std::strcmp(argv[i], "--families") == 0) {
#if defined(STRATA_HAVE_QWEN35)
            std::printf("qwen4exp qwen35moe\n");
#else
            std::printf("qwen4exp\n");
#endif
            return 0;
        }
        // `strata --geometry`: the name of the model shape the kernels are compiled for (the geometry header's kGeometryName: flash_next for the default build);
        // Control checks it against the shape of the package's signed manifest (strata_control/shape_engine.py)
        if (std::strcmp(argv[i], "--geometry") == 0) {
            std::printf("%s\n", strata::core::kGeometryName);
            return 0;
        }
    }
    prepare_process(argc, argv);
#if defined(STRATA_HAVE_QWEN35)
    // **THE FAMILY SWITCH.**  A Qwen3.6-35B-A3B file (GGUF architecture qwen35moe) runs on its own engine path
    // (src/models/qwen36), decided here from the file's header before any of this engine's state exists; every other file
    // goes on below exactly as before.  Only the architecture string is read (the header of --native's shard).
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--native") != 0 && std::strcmp(argv[i], "--model") != 0) continue;
        bool qwen35 = false;
        try {
            const strata::GgufFile meta(strata::gguf_split_paths(argv[i + 1]).front());
            qwen35 = strata::model_family(meta) == strata::ModelFamily::Qwen35Moe;
        } catch (const std::exception&) {
            // not readable here: the normal path below reports it
        }
        if (qwen35) return strata::qwen35::run(argc, argv);
        break;
    }
#endif
    prime_gpu_queues();
    Options o;
    bool have_tokens = false;
    if (const int status = parse_options(argc, argv, o, have_tokens); status >= 0) return status;
    if (const int status = apply_device_defaults(o, false); status >= 0) return status;
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "strata generate: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        print_usage();
        return 2;
    }
    strata::qwen38::Model model(o);
    if (const int status = model.load(); status != 0) return status;
    // the loaded model's parts, under the names the loops below use
    auto& primary_device = model.primary_device;
    auto& pcie_given = model.pcie_given;
    auto& split_at = model.split_at;
    auto& split_devs = model.split_devs;
    auto& split_same = model.split_same;
    auto& multi_gpu = model.multi_gpu;
    auto& remote_caches = model.remote_caches;
    auto& err = model.err;
    auto& native_pack = model.native_pack;
    auto& g = model.g;
    auto& K = model.K;
    auto& pool_bytes = model.pool_bytes;
    auto& arena = model.arena;
    auto& wt = model.wt;
    auto& native_dense = model.native_dense;
    auto& d_mrope = model.d_mrope;
    auto& mrope_host = model.mrope_host;
    auto& cvec_summary = model.cvec_summary;
    auto& ss = model.ss;
    auto& sbuf = model.sbuf;
    auto& main_stream = model.main_stream;
    auto& main_cs = model.main_cs;
    auto& half_stride = model.half_stride;
    auto& half_dump = model.half_dump;
    auto& half_stage = model.half_stage;
    auto& db = model.db;
    auto& ple_table = model.ple_table;
    auto& ple_lock_by_policy = model.ple_lock_by_policy;
    auto& split_pf_mib = model.split_pf_mib;
    auto& d_parts = model.d_parts;
    auto& stages = model.stages;
    auto& last_st = model.last_st;
    auto& mtp_first = model.mtp_first;
    auto& profile = model.profile;
    auto& profile_loaded = model.profile_loaded;
    auto& pf_borrow = model.pf_borrow;
    auto& profile_all = model.profile_all;
    auto& session_layers = model.session_layers;
    auto& batch_mtp = model.batch_mtp;
    auto& piped_mtp = model.piped_mtp;
    auto& pm_drafts = model.pm_drafts;
    const auto batch_max_t = [&] { return model.get_batch_max_t(); };
    auto& bslot_ss = model.bslot_ss;
    auto& mtp = model.mtp;
#if defined(STRATA_HAVE_DFLASH2)
    auto& df2 = model.df2;
    auto& slot_df2 = model.slot_df2;
#endif
    auto& use_df2 = model.use_df2;
    auto& batch_df2 = model.batch_df2;
    auto& df2_first = model.df2_first;
    auto& df2_solo = model.df2_solo;
    const auto df2_vocab_path = [&] { return model.get_df2_vocab_path(); };
    const auto df2_offer_head = [&] { model.offer_df2_head(); };
    const strata::core::ModelGeometry& draft_geometry = strata::qwen38::Model::draft_geometry;
    auto& slot_mtp = model.slot_mtp;
    auto& n_vocab = model.n_vocab;
    auto& native_head = model.native_head;
    auto& logits = model.logits;
    auto& d_logits = model.d_logits;
    auto& prefill_head_alias_bound = model.prefill_head_alias_bound;
    auto& prefill_head_alias_at = model.prefill_head_alias_at;
    auto& src = model.src;
    auto& arena_src = model.arena_src;
    auto& srcp = model.srcp;
    strata::kernels::cpu::ExpertPool& pool = *model.cpu_pool;
    auto& xcache = model.xcache;
    auto& auto_cache = model.auto_cache;
    auto& nib_chunk_asked = model.nib_chunk_asked;
    auto& nib_x = model.nib_x;
    auto& nib_st = model.nib_st;
    auto& reserve_adapted = model.reserve_adapted;
    auto& remote_experts = model.remote_experts;
    auto& remote_opt = model.remote_opt;
    auto& peer = model.peer;
    auto& drive = model.drive;
    auto& lookahead = model.lookahead;
    auto& routing = model.routing;
    auto& pool_fn = model.pool_fn;
    auto& hit_fn = model.hit_fn;
    auto& pool_user = model.pool_user;
    auto& gr = model.gr;
    // STRATA_PCIE_BALANCE=1 (opt-in): each verify layer's PCIe count from measured costs (Verifier::set_pcie_balance)
    // instead of the start-up probe's share - the CPU's speed sets the right share as much as the link's.  IQ2_XS,
    // one AVX2 pool worker (emulated on an RTX 5090 + 9950X3D): 16.8-17.2 -> 12.8-13.0 ms a round; the 9950X3D's own
    // 16 workers: 12.2 either way.  One GPU, no batch slots, no --pcie-frac.
    auto pcie_balance = [&]() {
        const char* e = std::getenv("STRATA_PCIE_BALANCE");
        return !pcie_given && native_pack && !multi_gpu && o.batch <= 0 && e != nullptr && std::strcmp(e, "1") == 0;
    };
    // #731 (opt-in, STRATA_DISJOINT_ADAPT=1): the adaptive tiers leave an expert a helper GPU holds out of the primary's
    // promotion candidates (it would sit in both caches).  Asked live, from the helper's own cache (RemoteExperts::holds,
    // #854), so an expert the helper's tier swaps in or out later is followed - never a copy taken at load.
    const bool disjoint_adapt = [] { const char* v = std::getenv("STRATA_DISJOINT_ADAPT"); return v != nullptr && std::atoi(v) != 0; }();
    auto helper_holds = [&](int64_t l, int32_t e) -> bool {
        if (!disjoint_adapt) return false;
        for (int r = 0; r < drive.d.remote_count; ++r)
            if (drive.d.remote[r]->holds(l, e)) return true;
        return false;
    };
    // The swaps an adaptive round would make, best first (strata::core::choose_expert_swaps).  `fetch_rule` (the
    // synchronous rounds): with STRATA_ADAPT_FETCH=2, where the PCIe share fetched since the last round
    // (drive.d.fetch_avail, which this clears), its fetchable (pinned) experts are the admission's; the tier keeps the
    // rest (no share there, or not pinned).
    using Swap = strata::core::ExpertSwap;
    auto choose_swaps = [&](const std::vector<float>& usage, const std::vector<int32_t>& res, float min_gain,
                            bool fetch_rule, auto&& held_elsewhere) {
        std::vector<uint8_t>& fetch_avail = drive.d.fetch_avail;
        std::vector<uint8_t> fetched;   // the layers the PCIe share fetched in (empty: none)
        if (fetch_rule && !fetch_avail.empty()) {
            if (drive.d.fetch_admit == 2)
                fetched = fetch_avail;
            std::fill(fetch_avail.begin(), fetch_avail.end(), 0);
        }
        return strata::core::choose_expert_swaps(usage, res, g.n_layers, g.n_expert, min_gain, [&](int64_t l, int32_t e) {
            return held_elsewhere(l, e) || (!fetched.empty() && fetched[(size_t) l] && srcp->pinned(l, e));
        });
    };
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return strata::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return strata::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the embedding buffer failed\n");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the sampler output buffer failed\n");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    void* token_stream = o.stream_token ? main_cs : nullptr;
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!strata::core::embed_row(wt, g, tok, d_emb, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        if (pos == 0) {
            strata::core::session_zero(ss, g, d_emb, token_stream);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (cudaMemcpyAsync(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t) token_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: the residual broadcast failed\n");
                    return false;
                }
        }
        return o.stream_token || cudaDeviceSynchronize() == cudaSuccess;
    };

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            std::fprintf(stderr, "strata generate: sampling greedy\n");
        else
            std::fprintf(stderr, "strata generate: sampling temperature=%g top_k=%d top_p=%g seed=%llu\n",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    // The logits header is written WITH THE FIRST ROW, not at open: a native pack never reaches the
    // per-token dump site, and a header promising rows that were never written is worse than no file.
    int32_t hdr[2] = {0, 0};
    bool hdr_written = false;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            std::fprintf(stderr, "strata generate: logits dump dimensions exceed int32\n");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_logits.c_str());
            return 1;
        }
        // **THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.**  The loop writes one row per
        // position from 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts
        // receiving at position `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows, and the
        // header used to claim 11.  A header that describes a different file from the one written is the same
        // class of defect as a self-check that verifies the wrong invariant: anything reading the count instead
        // of the size gets a wrong answer that looks authoritative.  `tools/logits_identical.py` caught it by
        // parsing the header and refusing the file.
        const int32_t n_rows = (int32_t) strata::qwen38::logits_selection::row_count(dump_positions, o.logits_stride);
        hdr[0] = n_vocab; hdr[1] = n_rows;
    }

    // ---- THE C1 ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_layers.c_str());
            return 1;
        }
        if (cudaHostAlloc((void**) &layer_stage, layer_floats * sizeof(float), cudaHostAllocDefault) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the layer-dump staging buffer\n");
            return 1;
        }
    }

    // ---- prefill is the DECODE PATH ONE TOKEN AT A TIME, which `phase-2-correct-engine.md:12-13` says is
    // fine here: "process the prompt through the decode-style graphs in small batches; a 19K-token prompt will
    // take minutes".  A real batched prefill is P2.S6's other half and is not this.
    //
    // **THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE
    // LAST *PROMPT* POSITION.**  The first version sampled only on the decode positions, so `produced` was
    // still empty when the first generated position asked for `produced.back()` - an out-of-bounds read on an
    // empty vector.  Teacher forcing below is what makes the distinction unnecessary to special-case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- R0.9: THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        strata::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            // #610: a native (IQ) pack captures no per-layer graphs; the serve path times its window's stages
            std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(),
                         native_pack ? " - a native (IQ) pack has no per-layer graphs to replay. Its decode window's GPU "
                                       "stages are printed per request by the serve path instead: STRATA_VERIFY_PROFILE=1 "
                                       "STRATA_DECODE_TIMING=1 with --serve (docs/DETAILS.md, \"Where a decode window's "
                                       "time goes\")" : "");
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 are GDN and 12 are QSA and a total cannot
        // separate them.  Round 309's uncaptured table put GDN at 10.88 ms for 36 layers against QSA's 4.56 for
        // 12, which would make the recurrence the largest single R3 target - and that table had `moe_finish`
        // wrong by 4x, so the ratio is re-derived here from the captured graph rather than inherited.
        {
            // **ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED.**  The first version called the
            // per-layer replay a single time and printed `gdn / reps`, which reported GDN at 0.480 ms/token
            // against a mixer total of 14.039 - a factor of exactly `reps`, and the tell was that
            // 0.480 + 0.220 = 0.700 = 14.039 / 20.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (strata::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ R0.11: THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // the same independent-restatement test that caught round 320's divide-by-reps bug, and it is the only
        // reason to believe a table built out of differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f   <- R3.3's target is <= 3 ms for all four passes\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ R3.5c: THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!strata::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                            return 1;
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        strata::core::session_graphs_free(gr);
        strata::core::doorbell_free(db);
        cudaFree(d_next);
        return 0;
    }

    if (o.graph_only) {
        strata::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            std::fprintf(stderr, "strata generate: session_replay warm: %s\n", err.c_str());
            return 1;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay warm faulted\n");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay: %s\n", err.c_str());
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay faulted: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- R0.3: THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the
    // shared expert, the combine and the second `gr_write` - and the LM head.  Everything this project published
    // as "39.8 ms pure GPU" came from that loop while being described as the whole GPU, which also made the
    // "host's share = 49.4 - 39.8 = 9.6 ms" figure wrong by however much the missing work costs.  See
    // Memory/ERRORS.md A4/A5.
    //
    // This flag is what "pure GPU" has to mean, and it replaces that number everywhere.  No pool runs, so
    // `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        strata::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!strata::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay_full: %s\n", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full layers faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full lm_head: %s\n", err.c_str());
                return 1;
            }
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full head faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    if (const int status = model.build_hit_tables(layer_dump != nullptr); status != 0) return status;
    auto& loop_scratch = model.loop_scratch;
    auto& tgraph = model.tgraph;
    auto& host_res = model.host_res;
    auto res_put = [&model](int32_t* dst) { return model.put_residency(dst); };
    auto& d_res = model.d_res;
    auto& thits = model.thits;
    const bool graph_hits = model.graph_hits;

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // **`--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.**  It never enters
    // this loop, so it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the
    // NaN scan or the sampler - and `--no-pool --stats` against that floor was being read as "the per-layer
    // round trip costs 12.4 ms" when an unknown part of it is per TOKEN, not per layer.  That is the same error
    // the review catalogued as A4/A5, one level down: a difference between two measurements attributed to a
    // mechanism that neither of them isolates.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  **The
    // boundary after the layer loop is the one that matters**: without it the head's interval swallows all 48
    // layers and the report reads as "head = 50 ms", which is not a thing that can happen to 1.4 ms of GPU
    // work.  That is not hypothetical - it is what the first version of this printed.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below; #471)
    //
    // `nonfinite` (#879): a window's head logits held a NaN or an infinity (Verifier::nonfinite_row).  None of that
    // window's tokens was written; every state that may carry the poison (the live session, the checkpoints, the
    // parked conversations, the idle batch slots) is dropped at the next request's start, which reads its prompt from
    // token 0.  A batch slot ends the same way: `BDONE <slot> <n> nonfinite`.
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    model.set_prompt_pinned_share();
    // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]),
    // which is the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage
    // borrowing the tail of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the
    // chunk, is laid out by `Prefill::relayout`, and is refilled before any window reads - so outside the
    // prompt the whole cache is expert cache.  THIS IS THE POINT OF THE STRUCT: the loan used to exist only
    // for CUDA0, and `no_prefill_borrow` made every stage instead withhold a chunk-sized reserve from its
    // cache for the entire session, which is what cost the 4-way its context (see the note at the top of the
    // layer-split block).  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose
    // slot values are indices into whichever cache owns the layer, so a loan that marked rows by slot number
    // alone would hand CUDA0 a slot belonging to another stage's cache.
    struct PfPart {
        strata::core::ExpertCache* cache = nullptr;
        const strata::core::SessionState* ses = nullptr;
        strata::prefill::Prefill* sp = nullptr;
        int dev = -1;                  // -1: leave the device alone (CUDA0)
        int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
        int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
        int32_t first_now = -1;        // where its buffers are laid out now
        int64_t lent_chunk = 0;
        std::vector<std::pair<int32_t, int32_t>> lent;
    };
    std::vector<PfPart> pf_parts;   // the serve path's, filled there
    // the slots chunk c's buffers borrow from one cache: exact when it knows its per-slot offsets (a native pack's
    // blobs differ per layer), otherwise max_blob each
    auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
        return strata::program::split_lend_slots(p.cache->slots(), p.cache->bytes(), p.cache->slot_offsets(),
                                                 (uint64_t) strata::kernels::cpu::expert_layout().max_blob,
                                                 strata::prefill::Prefill::bytes_needed(g, *p.ses, c, srcp != nullptr));
    };
    // the bytes of one cache's slots from `first` to its end
    auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
        const strata::core::ExpertCache& xc = *p.cache;
        return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                 : (uint64_t) (xc.slots() - first) *
                                       (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    const PfPart cuda0_part{&xcache, &ss};   // CUDA0's cache alone: what plan_lend and generate's prompt price
    auto lend_slots = [&](int64_t c) { return part_slots(cuda0_part, c); };
    // STRATA_REFILL_PARTS=1 (gfx1151's table): a lent slot of a GGUF read in place is refilled from the pieces in the mapping.
    // The default assembles the blob in a staging buffer first (a 4 MB copy on this thread), then copies that.
    static const bool refill_parts = [] { const char* v = std::getenv("STRATA_REFILL_PARTS"); return v != nullptr && std::atoi(v) != 0; }();
    // The two halves of giving a loan back: the rows `p` lent are refilled into the SAME slots of its own cache, on its
    // own device, and the residency table is restored (the caller uploads it).  Split so that a layer split can queue
    // every stage's copies before it waits for any (#340).
    auto refill_loan_issue = [&](PfPart& p, std::string& e) -> bool {
        const strata::core::OnDevice on(p.dev);
        for (const auto& [i, slot] : p.lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
            strata::core::BlobPart parts[3];
            const int np = refill_parts && !refill_blocking() ? srcp->blob_parts(i / g.n_expert, i % g.n_expert, parts) : 0;
            if (np > 0) {   // the GGUF read in place: the tensors' pieces go from the mapping to the slot, no assembled copy
                if (!p.cache->fill_slot_queued_parts(slot, parts, np, e)) return false;
                host_res[(size_t) i] = slot;
                continue;
            }
            const uint8_t* b = srcp->blob_stable(i / g.n_expert, i % g.n_expert);
            const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
            if (b == nullptr || !(refill_blocking() ? p.cache->fill_slot_blocking(slot, b, e, nb)
                                                    : p.cache->fill_slot_queued(slot, b, e, nb)))
                return false;
            if (!refill_blocking()) srcp->note_async_read(b, nullptr);   // #1237
            host_res[(size_t) i] = slot;
        }
        return true;
    };
    auto refill_loan_wait = [&](PfPart& p, std::string& e) -> bool {
        const strata::core::OnDevice on(p.dev);
        if (!p.cache->sync_queued(e)) return false;
#if defined(_WIN32)
        // The copies have landed: the file pages touched by the loan need not stay in the working set.
        for (const auto& [i, slot] : p.lent)
            srcp->release(i / g.n_expert, i % g.n_expert);
#endif
        p.lent.clear();
        p.lent_chunk = 0;
        return true;
    };
    auto request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // STRATA_PREFILL_EQUAL=1 (opt-in, PR #693): a segment of `tokens` reads in n = ceil(tokens / max_chunk) chunks of EQUAL
    // size (ceil(tokens / n) rounded up to 256) instead of full chunks and a short last one.  A chunk of
    // stream_all_min_tokens() or more streams every expert the GPU does not hold, whatever its length, so a prompt's cost
    // is its number of such chunks and equal chunks borrow no more slots than that count needs.  A last chunk below
    // stream_all_min_tokens() stays as it is (it moves only the experts its own tokens route to).  Used ONLY where a
    // segment's chunk is chosen (the serve lend, generate's prompt); `request_chunk` is unchanged because
    // STRATA_SPLIT_SMALL_OWN uses it too.  The chunk geometry changes the rounding, so it is off by default.
    static const bool equal_chunks = [] { const char* e = std::getenv("STRATA_PREFILL_EQUAL"); return e != nullptr && e[0] == '1'; }();
    auto equal_chunk = [&](int64_t tokens, int64_t max_chunk) -> int64_t {
        const int64_t plain = request_chunk(tokens, max_chunk);
        if (!equal_chunks || tokens <= 0 || max_chunk <= 0) return plain;
        const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0);
        const int64_t last = tokens - (n - 1) * max_chunk;
        if (n <= 1 || last < strata::prefill::Prefill::stream_all_min_tokens()) return plain;
        const int64_t per = tokens / n + (tokens % n != 0);
        const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // #2824: the chunk a request of `tokens` reads in when the host-aware plan chose `plan` (< cap, the buffers' chunk = what the VRAM
    // scan picked).  A prompt that fits one chunk of `cap` keeps ONE chunk, as before (a single pass reads each expert once, so the
    // page cache cannot thrash, and splitting it would only add a pass); a longer one reads in n = ceil(tokens / plan) chunks of
    // EQUAL size (no tiny tail), at most `cap`.
    auto planned_chunk = [&](int64_t tokens, int64_t cap, int64_t plan) -> int64_t {
        if (plan <= 0 || plan >= cap || tokens <= 0) return equal_chunk(tokens, cap);
        return strata::program::prefill_plan::balanced_chunk(tokens, cap, plan);
    };
    // The share (%) of a cache's slots the buffers of an auto chunk may borrow: a lent slot's expert is streamed during
    // the prompt and refilled after it (measured on a 12 GB card, 32K Q2_0 prompt: 4096 791 tok/s, 6144 878, 8192 973
    // with 69% of the slots lent).  A request lends only what its own prompt needs (Prefill::relayout), so a big chunk
    // costs short prompts nothing.  Every participant's loan obeys the same cap.
    const int64_t kAutoLendPct = get_auto_lend_pct();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the
    // context, but a bare `auto` always reaches 8192.  32768 and 16384 stay opt-in (#282): a 32K prompt with
    // IQ2_XS (RTX 5090, 64K context) read at 5,624 tok/s in 8192-token chunks and 6,465 in one 32768 chunk (40K
    // -> 18K experts streamed; an NVFP4 pack at 262K: 3,535 -> 5,201).
    int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    // 0.1.44: a layer split reads a prompt as a pipeline of chunks (the later card reads chunk c while the first reads c+1), so a
    // 32K prompt in two 16384 chunks fills and drains the pipe for most of its time.  8192 chunks, same tokens: 2x R9700
    // pp32k +14%, 4x +19%, pp64k +4% / +16%.  STRATA_SPLIT_CHUNK16K=1 keeps the gfx12 16384 ceiling on a split.
    if (multi_gpu && o.prefill_gfx12_16k) {
        static const char* keep16 = std::getenv("STRATA_SPLIT_CHUNK16K");   // (the NVIDIA rule shares this: a layer split reads 8192-token chunks)
        if (!(keep16 != nullptr && std::atoi(keep16) != 0)) {
            auto_ceiling = std::min<int64_t>(auto_ceiling, 8192);
            o.prefill_auto_max = std::min<int64_t>(o.prefill_auto_max, 8192);   // the 0.1.39 chunk list reads this one
            o.prefill_chunk = std::min<int64_t>(o.prefill_chunk, 8192);
        }
    }
    // #2090 (bdwilton): the gfx12 16384 ceiling lends the prompt path about twice the slots of 8192 (R9700, Swift 1.5
    // IQ3_XXS: 3,907 slots / 6.38 GiB against 2,068 / 3.37 GiB), and every lent slot's expert is read from the RAM side
    // during the read.  On a 30 GB PC with --resident-experts that was 3.4 GB available, 20 GB in swap, 1.5M major faults
    // and a killed engine on 38-47K prompts; 8192 ran them at 95 tok/s.  The ceiling is the engine's default, not the
    // user's pick, so when the RAM available now does not cover the extra loan plus 6 GiB (that PC thrashed with 3.4 GB left and
    // ran with ~6.4) the default stays at 8192.
    // STRATA_GFX12_CHUNK=1 keeps the default ceiling anyway.  The ceiling is 32768 now: the guard steps it down 32768 -> 16384 -> 8192 until the loan fits.
    for (int64_t cand = auto_ceiling; o.prefill_gfx12_16k && cand > 8192;) {
        static const char* force16 = std::getenv(std::strcmp(o.prefill_raise_who, "NVIDIA") == 0 ? "STRATA_NV_CHUNK" : "STRATA_GFX12_CHUNK");
        const int64_t extra_slots = std::max<int64_t>(0, lend_slots(cand) - lend_slots(8192));
        const uint64_t extra_mib = ((uint64_t) extra_slots * strata::kernels::cpu::expert_layout().max_blob) >> 20;
        const auto hm = strata::core::sample_host_memory();   // the cgroup / --memory-limit-mib cap counts too
        const uint64_t avail_mib = hm ? hm->available >> 20 : mem_sample().avail_mib;
        // --resident-experts pins its RAM copy after this point, and that copy also holds the lent slots' experts: what
        // has to fit is every expert the cache does not hold plus the 16384 loan (R9700 IQ3_XXS: 23.35 GiB at 16384)
        uint64_t pin_mib = 0;
        if (o.resident_cpu_experts) {
            const uint64_t total = strata::kernels::cpu::expert_layout().total;
            const uint64_t cached = (uint64_t) std::max<int64_t>(0, (int64_t) xcache.bytes());
            pin_mib = ((total > cached ? total - cached : 0) +
                       (uint64_t) std::max<int64_t>(0, lend_slots(8192)) * strata::kernels::cpu::expert_layout().max_blob) >> 20;
        }
        if (avail_mib > 0 && avail_mib < pin_mib + extra_mib + 6144 && !(force16 != nullptr && std::strcmp(force16, "1") == 0)) {
            const int64_t next = strata::core::gfx12_chunk_step_down(cand);
            auto_ceiling = std::min<int64_t>(auto_ceiling, next);
            o.prefill_auto_max = std::min<int64_t>(o.prefill_auto_max, next);
            o.prefill_chunk = std::min<int64_t>(o.prefill_chunk, next);
            std::fprintf(stderr, "strata generate: %s: prompt chunks stay at %lld: %lld would lend %lld more cache slots "
                                 "(%llu MiB more held in RAM during a prompt), and %llu MiB of RAM is available for %llu MiB "
                                 "of resident experts plus that (#2090; %s=1 takes the default ceiling anyway)\n",
                         o.prefill_raise_who, (long long) next, (long long) cand, (long long) extra_slots, (unsigned long long) extra_mib,
                         (unsigned long long) avail_mib, (unsigned long long) pin_mib,
                         std::strcmp(o.prefill_raise_who, "NVIDIA") == 0 ? "STRATA_NV_CHUNK" : "STRATA_GFX12_CHUNK");
            cand = next;
            continue;
        }
        break;
    }
    if (o.prefill_auto_cap > 0) auto_ceiling = std::min<int64_t>(auto_ceiling, o.prefill_auto_cap);   // #1630
    // The auto scan used to walk a fixed list of sizes - 32768, 16384, 8192, 6144, ... - and take the first that
    // fit.  That list is coarse exactly where a rig needs it: this one affords ~8,700 tokens and was handed 8192,
    // and 8704 is not on it.  `bytes_needed` is a sum of (T x positive constant) terms plus a max of such sums,
    // so it rises monotonically with T, and so does every test the scan applies - which makes the largest size
    // that fits a bisection on the 256-token grid the prompt path already works on (`request_chunk` rounds up to
    // it).  log2(32768/256) = 7 probes, against the list's 10, at one `bytes_needed` per probe.
    auto biggest_chunk = [](int64_t ceiling, auto&& ok) -> int64_t {
        if (ceiling < 256) return 0;
        int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
        while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
        }
        return best;
    };
    // Whether one participant can lend chunk c: a loan must leave the g_lend_keep-slot floor.  The percentage cap
    // (`cap`) is an AUTO-chunk rule and only the auto scan applies it - an explicit --prefill is the operator's
    // number, and a loan of it only has to fit.
    auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
        const int64_t k = part_slots(p, c);
        if (k <= 0 || k + g_lend_keep > p.cache->slots()) return false;
        return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
    };
    // `only`: one participant alone (CUDA0's for plan_lend, and #448: what one GPU would choose); null: every one
    auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
        if (only != nullptr) return fits_one(*only, c, cap);
        for (const PfPart& p : pf_parts)
            if (!fits_one(p, c, cap)) return false;
        return true;
    };
    // The chunk and the ring are one budget.  The ring used to be a fixed 384 slots taken off the top
    // before the chunk was considered: 506 MiB on a Q2_0 pack, where it was tuned, but 1,912 MiB on
    // Q8_0, more than half an 8 GB card's cache - which is what kept that card at a 1024-token chunk.
    // Size the chunk first and hand the ring what the smallest participant leaves over, up to the ring's
    // byte budget - but hand it ALL of it: the ring is the better buy per byte.  Measured on a 4-way IQ3_S rig
    // (2x RTX 3060 + 2x RTX 5060, CUDA3 lending at its 90% cap) on a 120K prompt: 8960 tokens with the 17-slot
    // ring that leaves reads at 963 tok/s, 8192/130 at 1,008, 7168 with the ring full at its 199-slot byte
    // budget at 1,000, and 5632/199 at 915.  A ring slot is worth ~0.53 tok/s there and a chunk token ~0.05.
    // So the chunk to take is the largest one that still leaves the ring full, not the largest one that leaves
    // it anything at all.  kRingMin is the fallback floor for a rig where no chunk can afford a full ring;
    // ring_slots() returns STAGE at or below it, which turns streaming off.
    constexpr int64_t kRingMin = 16;
    // The room is a BYTE budget: a ring slot is max_blob bytes (`carve` lays out one whole blob each), but a cache
    // slot holds its own layer's blob, which on the 4-way IQ3_S rig averages 0.85 of it.  In slots the ring looked
    // 18% cheaper than it is, and a chunk whose ring only fitted after that discount was refused outright - the scan
    // stopped one step short while the chunk it did take had room to spare for the cap-sized ring.
    const int64_t kBlob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
    // the ring slots (max_blob bytes each) that p's lend budget holds beside chunk c's other buffers
    auto part_room = [&](const PfPart& p, int64_t c) -> int64_t {
        // clamped at 0: a cache under g_lend_keep slots has a negative lend budget, and `slots - budget` is
        // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
        // rather than reading one entry past its offsets
        const int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - g_lend_keep,
                                                            kAutoLendPct * p.cache->slots() / 100));
        const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
        const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c, srcp != nullptr);
        return (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob);
    };
    auto ring_room = [&](int64_t c, const PfPart* only, int64_t cap) -> int64_t {
        if (only != nullptr)
            return std::min(cap, part_room(*only, c));
        int64_t room = cap;
        for (const PfPart& p : pf_parts)
            room = std::min(room, part_room(p, c));
        return room;
    };
    // The largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend with the
    // ring still at its full byte budget (ring_room is capped at ring_max, so that is room == ring_max),
    // under the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192, #282).  Failing that, the
    // largest chunk whose ring clears kRingMin.
    // 0.1.39's list with its ring: the chunk it would pick, and the size up to which a prompt keeps that ring
    auto old_pick = [&](const PfPart* only) -> int64_t {
        strata::prefill::Prefill::set_ring_budget(0, 0);
        static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
        for (const int64_t c : kAutoChunks) {
            // above 8192: only when asked for, and only when a prompt of the context can use it (#282)
            if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
            if (o.prefill_auto_cap > 0 && c > o.prefill_auto_cap) continue;   // #1630
            if (fits(c, true, only)) return c;
        }
        return 0;
    };
    auto scan = [&](const PfPart* only) -> int64_t {
        if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
            for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                if (fits(c, false, only)) return c;
            return 0;
        }
        if (!strata::prefill::Prefill::ring_bytes_enabled()) return old_pick(only);   // STRATA_RING_BYTES=0
        const int64_t small = old_pick(only);
        const int64_t cap = strata::prefill::Prefill::ring_cap_for(small);
        auto probe = [&](int64_t floor) -> int64_t {
            return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                const int64_t room = ring_room(t, only, cap);
                if (room < floor) return false;
                strata::prefill::Prefill::set_ring_budget((int) room, 0);
                // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which
                // case the chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                return fits(t, true, only);
            });
        };
        int64_t c = probe(cap);
        if (c == 0 && cap < strata::prefill::Prefill::ring_default_slots()) c = probe(kRingMin);
        return std::max(c, small);   // never a smaller chunk than 0.1.39's
    };
    // CUDA0's loan alone (the resident RAM copy's lend region, generate's prompt): sets `chunk` to the chunk `scan`
    // picks and the ring to what the serve path sets for it, and returns the slots that chunk borrows (0, with `chunk`
    // unchanged, when none fits).
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        const int64_t c = scan(&cuda0_part);
        if (o.prefill_auto && strata::prefill::Prefill::ring_bytes_enabled()) {
            // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed it)
            const int64_t small = old_pick(&cuda0_part);
            if (c > small)
                strata::prefill::Prefill::set_ring_budget(
                    (int) ring_room(c, &cuda0_part, strata::prefill::Prefill::ring_cap_for(small)), small);
        }
        if (c <= 0)
            return 0;
        chunk = c;
        return lend_slots(c);
    };
    // ---- #2824: THE PROMPT PLAN PRICES THE HOST.  The scans above ask whether a loan fits the GPU cache.  But a lent
    // slot's expert is not in VRAM for the length of the prompt: it comes from the RAM copy if the copy kept it, else from
    // the model file, and a chunk streams every expert the GPU does not hold once per chunk.  When that file set is bigger
    // than the RAM the OS keeps file pages in, it cycles through the page cache and each chunk reads most of it again
    // (2x RTX 3060, 32 GB, IQ2_XS, 29K prompt: 8192-token chunks read 28-45 GB, 4096 read ~1 GB).  So the auto pick is
    // refined by prefill_plan (include/strata/program/prefill_plan.hpp): the chunk x ring grid at or below the VRAM pick,
    // each priced with the expert bytes its loan displaces without a RAM backup, the RAM left for the page cache, the
    // measured PCIe rate and a drive-speed class, and the least predicted prompt time wins.  A set that fits the page cache
    // keeps today's pick untouched.  Advisory: --prefill N and STRATA_PREFILL_RING are the operator's, and
    // STRATA_PREFILL_HOST_PLAN=0 turns this off.  No weights or experts change; only the chunk shape (rounding).
    //   fits_vram(c)       whether every participant can lend chunk c at the ring currently set
    //   lend_of(c, bytes, slots)   the cache bytes and slots the loan of chunk c takes at the ring currently set
    //   ring_full(c)       the largest ring slot count on offer for chunk c
    //   restore()          puts the ring state back as the VRAM pick left it
    // Returns the chunk to use (== `chunk` when nothing changes); the ring state is set for it.
    auto host_plan_chunk = [&](int64_t chunk, uint64_t gpu_cache_bytes, int64_t cache_slots,
                               const std::function<bool(int64_t)>& fits_vram,
                               const std::function<void(int64_t, uint64_t&, int64_t&)>& lend_of,
                               const std::function<int64_t(int64_t)>& ring_full,
                               const std::function<void()>& restore, int64_t tokens_hint = 0) -> int64_t {
        namespace pp = strata::program::prefill_plan;
        const char* off = std::getenv("STRATA_PREFILL_HOST_PLAN");
        if (!o.prefill_auto || chunk <= 0 || srcp != &src || (off != nullptr && off[0] == '0') ||
            !strata::prefill::Prefill::ring_bytes_enabled())
            return chunk;
        const auto& lay = strata::kernels::cpu::expert_layout();
        uint64_t total = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) total += (uint64_t) lay.blob_bytes(l) * (uint64_t) g.n_expert;
        if (total == 0 || gpu_cache_bytes >= total) return chunk;   // the GPU holds every expert: no file tier
        const auto hm = strata::core::sample_host_memory();
        if (!hm) return chunk;
        const uint64_t slack = 1200ull << 20;   // MemAvailable overstates what the cache can hold (~1 GiB on the #2824 box)
        pp::Host host;
        host.page_cache = src.unbuffered() ? 0 : (hm->available > slack ? hm->available - slack : 0);
        host.h2d_gbps = get_last_h2d_gbps() > 0.0 ? get_last_h2d_gbps() : 12.0;
        std::string why;
        host.ssd_gbps = drive_read_gbps(o.native_preset.empty() ? o.pack : o.native_preset, why);   // where the experts live
        if (const char* e = std::getenv("STRATA_PREFILL_PLAN_TOKS"); e != nullptr && std::atof(e) > 0.0)
            host.compute_tok_s = std::atof(e);
        const uint64_t ram_copy = src.complement_ready() ? src.resident_bytes() : 0;
        const int64_t kept_slots = src.complement_ready() ? src.resident_lent_slots() : 0;
        const uint64_t slot_avg = cache_slots > 0 ? gpu_cache_bytes / (uint64_t) cache_slots : 0;
        const uint64_t kept_bytes = (uint64_t) std::max<int64_t>(0, kept_slots) * slot_avg;   // the copy's lend region
        pp::Tier tier;
        tier.experts_total = total;
        tier.gpu_cache = gpu_cache_bytes;
        const uint64_t ram_outside_gpu = ram_copy > kept_bytes ? ram_copy - kept_bytes : 0;
        tier.file_tier = total > gpu_cache_bytes + ram_outside_gpu ? total - gpu_cache_bytes - ram_outside_gpu : 0;
        tier.extra_file = o.ple_gguf.empty() ? 0 : (350ull << 20);   // PLE rows the prompt touches (#2824: ~0.35 GiB)
        const int64_t tokens = tokens_hint > 0 ? tokens_hint
                                               : std::max<int64_t>(chunk, std::min<int64_t>(std::max<int64_t>(o.max_context, 4096), 32768));
        std::vector<pp::Option> opts;
        auto add = [&](int64_t c, int64_t ring_req) {
            strata::prefill::Prefill::set_ring_budget((int) ring_req, 0);
            if (!fits_vram(c)) return;
            pp::Option op;
            op.chunk = c;
            op.ring = strata::prefill::Prefill::ring_slots_for(c);
            op.ring_full = ring_full(c);
            uint64_t lb = 0;
            int64_t ls = 0;
            lend_of(c, lb, ls);
            op.lend_bytes = lb;
            op.lend_kept = std::min<uint64_t>(lb, kept_bytes);
            for (const pp::Option& e : opts) if (e.chunk == op.chunk && e.ring == op.ring) return;
            opts.push_back(op);
        };
        // today's pick first, priced under the ring state it left
        restore();
        pp::Option base;
        base.chunk = chunk;
        base.ring = strata::prefill::Prefill::ring_slots_for(chunk);
        base.ring_full = ring_full(chunk);
        {
            uint64_t lb = 0;
            int64_t ls = 0;
            lend_of(chunk, lb, ls);
            base.lend_bytes = lb;
            base.lend_kept = std::min<uint64_t>(lb, kept_bytes);
        }
        opts.push_back(base);
        for (int64_t c = chunk; c >= 512; c -= 256) {
            const int64_t full = ring_full(c);
            add(c, (int) full);
            if (full > 32) add(c, (int) (full / 2));
            if (full > 16) add(c, 16);
        }
        pp::Pick pick = pp::choose(host, tier, tokens, opts, 0);
        std::fprintf(stderr, "%s\n", pp::describe(pick, host, tokens).c_str());
        std::fprintf(stderr, "strata: prefill plan inputs: %.1f GiB experts, %.1f GiB in the GPU caches, %.1f GiB RAM copy "
                             "(%lld lend slots kept), file tier %.1f GiB; %.1f GiB available for the page cache%s; PCIe %.1f GB/s; "
                             "drive %.2f GB/s (%s); a %lld-token prompt\n",
                     (double) total / pp::kGiB, (double) gpu_cache_bytes / pp::kGiB, (double) ram_copy / pp::kGiB,
                     (long long) kept_slots, (double) tier.file_tier / pp::kGiB, (double) host.page_cache / pp::kGiB,
                     src.unbuffered() ? " (the file tier reads unbuffered: none)" : "", host.h2d_gbps, host.ssd_gbps, why.c_str(),
                     (long long) tokens);
        if (!pick.changed) {
            restore();
            return chunk;
        }
        strata::prefill::Prefill::set_ring_budget((int) pick.best.opt.ring, 0);
        return pick.best.opt.chunk;
    };
    // ---- THE ELASTIC K/V (--kv-grow; vmm.hpp).  Allocated for the whole --max-context up front, the K/V of a 262K
    // context (3.35 GiB with the drafter's at int8) takes VRAM the expert cache could hold more experts in.  Now the K/V pools hold physical memory only for the cells
    // the requests reach.  When a request needs more, slots just below the prompt path's loan give up their experts
    // (the CPU computes those from RAM, like any miss) and their whole chunks are mapped into the K/V; a later short
    // request hands them back and the slots are refilled with the profile's hottest experts.  Neither side's
    // addresses move, so every captured graph stays valid.  The context is never smaller: --max-context cells always
    // fit, the cache simply holds fewer experts while a long one is live.
    struct KvGrow {
        bool on = false;
        int64_t top = 0, lo = 0;          // slots [lo, top) hold no expert; their whole chunks may be with the K/V
        int64_t floor = 128;              // the cache keeps at least these slots
        int64_t step = 8192;              // the K/V grows in whole steps of cells
        int64_t cells = 0;                // what every K/V pool holds now
        std::vector<strata::core::VmmChunk> spare;   // out of the cache, not (yet) in the K/V
        int64_t grows = 0, trims = 0, fresh = 0, evicted = 0, refilled = 0;
    } kvg;
    auto kvg_start = [&](int64_t top) {
        kvg.on = strata::core::qsa_kv_elastic() && xcache.vmm_range() != nullptr && d_res != nullptr &&
                 !host_res.empty() && srcp != nullptr && top > kvg.floor;
        if (!kvg.on) return;
        kvg.top = kvg.lo = top;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        if (const char* v = std::getenv("STRATA_KV_GROW_STEP"); v != nullptr && std::atoll(v) > 0) kvg.step = std::atoll(v);
        // tests: the cache keeps this many slots (its size: the K/V grows into new VRAM only)
        if (const char* v = std::getenv("STRATA_KV_GROW_FLOOR"); v != nullptr && std::atoll(v) > 0) kvg.floor = std::atoll(v);
        std::fprintf(stderr, "strata generate: elastic K/V: %lld of %lld cells in VRAM (%.2f of %.2f GiB); the expert "
                             "cache (%lld slots) gives the K/V room below slot %lld as the context grows\n",
                     (long long) kvg.cells, (long long) o.max_context,
                     (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (double) strata::core::qsa_kv_elastic_full_bytes() / 1073741824.0, (long long) xcache.slots(),
                     (long long) top);
    };
    // Room for `cells` cells (rounded up to a step).  `quiesce` must leave nothing running on the device and land
    // the adaptive tier's swaps (a swap in flight could still be writing a slot given up here).
    auto kvg_ensure = [&](int64_t cells, const std::function<void()>& quiesce) -> bool {
        if (!kvg.on || cells <= kvg.cells) return true;
        const int64_t target = std::min<int64_t>(o.max_context, (cells + kvg.step - 1) / kvg.step * kvg.step);
        const int64_t need = strata::core::qsa_kv_elastic_need(target);
        if (need == 0) { kvg.cells = strata::core::qsa_kv_elastic_cells(); return true; }
        quiesce();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        std::vector<int32_t> owner;   // slot -> residency index
        // The slots given up are the ones just below the prompt path's loan, and the loan is most of the cache: they
        // hold experts of middling heat.  So an expert there that is hotter than the coldest one in the loan region
        // moves into that one's slot (a device copy), and the coldest is given up instead - the cache then holds what
        // a smaller cache would.  Heat: the adaptive tier's routing counts, then the profile's rank.
        std::vector<int32_t> rank;
        std::vector<std::pair<float, int32_t>> cold;   // the loan region's residents, hottest first
        auto heat = [&](int32_t i) -> float {
            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) i];
            return u * 1048576.0f - (float) rank[(size_t) i];
        };
        const auto& lay = strata::kernels::cpu::expert_layout();
        int64_t gave = 0, moved = 0;
        auto freeable = [&]() -> int64_t {   // mapped chunks wholly inside [lo, top): a chunk a live slot shares stays
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            int64_t k = 0;
            for (int64_t c = c0; c < c1; ++c) k += r.mapped(c) ? 1 : 0;
            return k;
        };
        while ((int64_t) kvg.spare.size() + freeable() < need && kvg.lo > kvg.floor) {
            if (owner.empty()) {
                owner.assign((size_t) xcache.slots(), -1);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= 0 && host_res[i] < xcache.slots()) owner[(size_t) host_res[i]] = (int32_t) i;
                rank.assign(host_res.size(), (int32_t) profile.size());
                for (size_t r = 0; r < profile.size(); ++r)
                    rank[(size_t) profile[r].first * (size_t) g.n_expert + (size_t) profile[r].second] = (int32_t) r;
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= kvg.top) cold.emplace_back(heat((int32_t) i), (int32_t) i);
                std::sort(cold.begin(), cold.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            --kvg.lo;
            ++gave;
            if (const int32_t oi = owner[(size_t) kvg.lo]; oi >= 0) {
                const int64_t layer = oi / g.n_expert;
                int32_t vi = -1;
                if (!cold.empty() && cold.back().first < heat(oi)) {
                    vi = cold.back().second;
                    const int32_t vs = host_res[(size_t) vi];
                    nib::guard_raw(layer, "the elastic K/V's slot move (kvg_ensure)");
                    if (xcache.slot_offset(vs + 1) - xcache.slot_offset(vs) < lay.blob_bytes(layer) ||
                        cudaMemcpy(xcache.device_slot(vs), xcache.device_slot((int32_t) kvg.lo),
                                   (size_t) lay.blob_bytes(layer), cudaMemcpyDeviceToDevice) != cudaSuccess)
                        vi = -1;
                    else {
                        cold.pop_back();
                        host_res[(size_t) oi] = vs;
                        host_res[(size_t) vi] = strata::core::kNotResident;
                        ++moved;
                    }
                }
                if (vi < 0) host_res[(size_t) oi] = strata::core::kNotResident;
                ++kvg.evicted;
            }
        }
        // a device-to-device cudaMemcpy does not wait for the copy: the moves still read the slots given up, and a
        // chunk unmapped under a copy faults it (the error then sticks to whatever runs next)
        if (gave > 0 && cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata: moving experts for the K/V failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        {
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            for (int64_t c = c0; c < c1; ++c)
                if (r.mapped(c))
                    if (const strata::core::VmmChunk h = r.unmap(c)) kvg.spare.push_back(h);
        }
        if (gave > 0 &&
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "strata: the residency table upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        int64_t fresh = 0;
        auto grow_take = [&]() -> strata::core::VmmChunk {
            if (kvg.spare.empty()) { ++fresh; return 0; }   // the cache is at its floor: new memory
            const strata::core::VmmChunk h = kvg.spare.back();
            kvg.spare.pop_back();
            return h;
        };
        bool ok = strata::core::qsa_kv_elastic_grow(target, grow_take);
        // STRATA_PARK_VRAM_MIB: the parked conversations' VRAM copies of their K/V give way to the K/V itself
        if (!ok && strata::core::conversation_park_vram_release_all() > 0) {
            (void) cudaGetLastError();
            ok = strata::core::qsa_kv_elastic_grow(target, grow_take);
        }
        kvg.fresh += fresh;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        ++kvg.grows;
        std::fprintf(stderr, "strata: K/V grown to %lld cells (%.2f GiB); the expert cache gave %lld slots for it, "
                             "%lld hotter experts moved to colder ones' slots (%lld of %lld slots hold experts)%s\n",
                     (long long) kvg.cells, (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (long long) gave, (long long) moved, (long long) (kvg.lo + (xcache.slots() - kvg.top)),
                     (long long) xcache.slots(),
                     fresh > 0 ? " - and new VRAM, the cache being at its floor" : "");
        if (!ok)
            std::fprintf(stderr, "strata: the K/V could not grow to %lld cells (%s)\n", (long long) target,
                         cudaGetErrorString(cudaGetLastError()));
        return ok;
    };
    // A request that needs far fewer cells than the K/V holds gives the rest back: the slots refill with the
    // profile's hottest experts the GPU does not hold.  Run on a quiet device (as kvg_ensure's `quiesce`).
    auto kvg_trim = [&](int64_t cells, int64_t hold) -> bool {   // `hold`: the cells a parked conversation could need
        if (!kvg.on) return true;
        // #1128: a parked conversation comes back through kvg_ensure's grow, so a K/V that fell to a short request's
        // size is grown again (its experts shuffled again) after every short request in between: the K/V keeps the
        // longest parked conversation's cells while it is parked
        const int64_t want = std::max<int64_t>(cells, hold);
        const int64_t target = std::max<int64_t>(kvg.step, (want + kvg.step - 1) / kvg.step * kvg.step);
        if (kvg.cells < target + 2 * kvg.step || kvg.lo >= kvg.top) return true;
        strata::core::qsa_kv_elastic_shrink(target, [&](strata::core::VmmChunk h) { kvg.spare.push_back(h); });
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
        {   // one run from the lowest unmapped chunk, as long as the chunks handed back last (one access call)
            const int64_t c0 = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            int64_t cend = c0, need = (int64_t) kvg.spare.size();
            while (cend < c1 && need > 0) need -= r.mapped(cend++) ? 0 : 1;
            if (!r.map_range(c0, cend, [&]() -> strata::core::VmmChunk {
                    if (kvg.spare.empty()) return 0;
                    const strata::core::VmmChunk h = kvg.spare.back();
                    kvg.spare.pop_back();
                    return h;
                }))
                std::fprintf(stderr, "strata: the expert cache could not take its VRAM back from the K/V\n");
        }
        const int64_t lo0 = kvg.lo;
        while (kvg.lo < kvg.top) {   // a slot whose chunks are all mapped again holds an expert again
            const int64_t a = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            const int64_t b = (int64_t) ((xcache.slot_offset(kvg.lo + 1) + G - 1) / G);
            bool all = true;
            for (int64_t c = a; c < b && all; ++c) all = r.mapped(c);
            if (!all) break;
            ++kvg.lo;
        }
        // the refills on one stream and one sync (a blocking copy each was a driver round trip per expert)
        const auto& lay = strata::kernels::cpu::expert_layout();
        std::vector<std::pair<size_t, int32_t>> filled;   // (residency index, slot), resident once the copies landed
        size_t pi = 0;
        for (int64_t s = lo0; s < kvg.lo; ++s) {
            const uint64_t room = xcache.slot_offset(s + 1) - xcache.slot_offset(s);
            for (; pi < profile.size(); ++pi) {
                const int32_t l = profile[pi].first, e = profile[pi].second;
                const size_t i = (size_t) l * (size_t) g.n_expert + (size_t) e;
                if (host_res[i] >= 0 || lay.blob_bytes(l) > room) continue;
                const uint8_t* b = srcp->blob_stable(l, e);
                if (b == nullptr) continue;
                nib::guard_raw(l, "the elastic K/V's refill (kvg_trim)");
                cudaMemcpyAsync(xcache.device_slot((int32_t) s), b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice,
                                nullptr);
                filled.emplace_back(i, (int32_t) s);
                ++pi;
                break;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata: refilling the expert cache failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        for (const auto& [i, s] : filled) host_res[i] = s;
        kvg.refilled += (int64_t) filled.size();
        // res_put waits for its copy: the next reader can be a verify window (a non-blocking stream) right away
        if (res_put(d_res) != cudaSuccess) return false;
        for (const strata::core::VmmChunk h : kvg.spare) strata::core::vmm_chunk_free(h);   // new ones, if any
        kvg.spare.clear();
        ++kvg.trims;
        std::fprintf(stderr, "strata: K/V trimmed to %lld cells; %lld slots back to the expert cache, refilled from "
                             "the profile\n", (long long) kvg.cells, (long long) (kvg.lo - lo0));
        return true;
    };
    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from experts.bin into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    // --adapt-async (opt-in): where the asynchronous adaptive tier does not apply, the blocking tier runs, said once.
    // It exchanges experts between the GPU caches and the resident RAM copy only (the copy is checked once it is
    // built, below).  Beside the --batch slots it advances between their windows, never between a prompt's chunks;
    // not with --batch-groups (pipelined slot groups), a peer tier or the helper GPUs' tier, which pick their swaps
    // from the same table.  The pipelined decode loop (--pipeline-windows 2) has windows in flight at every tick:
    // there the steps that overwrite something a window may still read wait for those windows (see adapt_tick);
    // STRATA_PIPELINE_ADAPT_ASYNC=0 keeps the blocking tier beside it instead.
    // Where the asynchronous tier could run on its own (CUDA, one GPU, the expert arena, no batch slots), the engine measures
    // and chooses during the session: see include/strata/models/qwen38/adapt_async_default.hpp.  --adapt-async 0 or
    // STRATA_ADAPT_ASYNC=0 keep the blocking tier; --adapt-async 1 keeps its old meaning.
    if (o.adapt_async_given < 0) {
        strata::core::AdaptAsyncAuto aa;
        aa.flag = o.adapt_async_given;
        aa.env = strata::core::adapt_async_env();
#if !defined(STRATA_USE_HIP)
        aa.cuda = true;
#endif
        aa.serve = o.serve;
        aa.adaptive = o.adapt_every > 0 && o.adapt_swaps > 0;
        aa.arena = srcp == &arena_src;
        aa.multi_gpu = multi_gpu;
        aa.pipeline_windows = o.pipeline_windows >= 2;
        aa.batch = o.batch > 0;
        aa.peer = peer.valid();
        aa.remote_opt = remote_opt != nullptr;
        aa.adapt_overlap = adapt_overlap();
        o.adapt_async_self = strata::core::adapt_async_self_decides(aa);
    }
    if (o.adapt_async) {
        static const bool pl_async = [] {
            const char* v = std::getenv("STRATA_PIPELINE_ADAPT_ASYNC");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const char* why = !o.serve ? "it needs --serve"
                        : o.adapt_every <= 0 || o.adapt_swaps <= 0 ? "the adaptive tier is off"
                        : !o.resident_cpu_experts && srcp != &arena_src ? "it needs the resident RAM mode, --resident-experts, or the expert arena"
                        : multi_gpu && srcp == &arena_src ? "not with a layer split in the arena mode"
                        : o.pipeline_windows >= 2 && srcp == &arena_src ? "not with --pipeline-windows 2 in the arena mode"
                        : o.batch > 0 && o.batch_groups > 1 ? "not with --batch-groups (the pipelined slot groups)"
                        : peer.valid() ? "not with --peer-device"
                        : remote_opt ? "not with --remote-expert-opt"
                        : o.pipeline_windows >= 2 && !pl_async ? "not with --pipeline-windows 2 (STRATA_PIPELINE_ADAPT_ASYNC=0)"
                        : nullptr;
        if (why != nullptr) model.turn_off_adapt_async(why);
    }
    // #340: whether a layer split's stage whose card has room for the prompt path's buffers keeps them as its own
    // instead of borrowing cache slots.  STRATA_SPLIT_OWN_BUFFERS=0: every stage borrows (0.1.30/0.1.31).
    static const bool split_own_ok = [] {
        const char* v = std::getenv("STRATA_SPLIT_OWN_BUFFERS");
        return v == nullptr || v[0] != '0';
    }();
    // STRATA_PIPE_ADAPT_EVERY (batch_groups.hpp; the rounds run in the pipelined loop below): said once here, with the
    // other start-up lines, before READY
    if (o.serve && o.batch > 0 && o.batch_groups > 1 && multi_gpu) {
        const int every = strata::program::batch_groups::pipe_adapt_every(std::getenv("STRATA_PIPE_ADAPT_EVERY"));
        if (every > 0)
            std::fprintf(stderr, "strata batch: STRATA_PIPE_ADAPT_EVERY=%d: the pipelined groups run an adaptive-tier round "
                                 "every %d group windows (the pipeline empties for it)%s\n", every, every,
                         o.adapt_every <= 0 || o.adapt_swaps <= 0 ? " - but the adaptive tier is off" : "");
    }
    // #340 / STRATA_SPLIT_RING: a layer split's streamed-ring override, decided from the stage caches' residency.  It
    // has to be in place BEFORE the split's lend regions are sized (Prefill::bytes_needed counts the ring): they were
    // sized with the default ring and the override applied afterwards, so a ring of 384 borrowed 5396 slots and the RAM
    // copy kept 4662 (the rest streamed from the GGUF mmap on every chunk).  Called once; later calls do nothing.
    bool split_ring_decided = false;
    auto decide_split_ring = [&]() {
        if (split_ring_decided || !o.serve || !multi_gpu || host_res.empty()) return;
        split_ring_decided = true;
        int64_t res_n = 0;
        for (const int32_t r : host_res) res_n += r >= 0;
        const double res_share = (double) res_n / (double) host_res.size();
        const int ring = strata::program::split_ring_slots(std::getenv("STRATA_SPLIT_RING"), res_share);
        if (ring > 0) {
            strata::prefill::Prefill::set_ring_override(ring);
            std::fprintf(stderr, "strata serve: layer split: %.0f%% of the experts resident, the prompt path's "
                                 "streamed ring %d slots\n", 100.0 * res_share, ring);
        }
    };
    if (o.resident_cpu_experts) {
        int64_t lend_from = -1;
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && d_res != nullptr && xcache.slots() > 0) {
            int64_t chunk = o.prefill_chunk;
            const int64_t k = plan_lend(chunk);
            if (k > 0) lend_from = xcache.slots() - k;
        }
        // a layer split: the experts every later stage's cache holds are left out of the RAM copy, as CUDA0's are
        // (with them the copy keeps no lend region: pin_cache_complement turns the loan off)
        // (true of the single lend region only: the stage_lend_regions block below hands every part's
        // borrowed tail to the copy as well, so under a split the copy keeps a lend region per part again)
        std::vector<std::pair<int32_t, int32_t>> stage_pairs;
        for (auto& st : stages)
            for (int64_t l = st->lb; l < st->le; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (st->cache.slot_of(l, e) >= 0) stage_pairs.emplace_back((int32_t) l, (int32_t) e);
        // A helper GPU's experts are left out of the RAM copy as a stage's are - the RAM
        // that duplication would take goes to colder experts instead.
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (remote_experts[(size_t) r].holds(l, (int32_t) e)) stage_pairs.emplace_back((int32_t) l, (int32_t) e);
        }
        // The split's lend regions.  The serve scan lends the tail slots of EVERY part's cache - CUDA0's own
        // part first, then every stage (the `pf_parts` loop below) - and `lend_from` above is switched off in
        // pin_cache_complement the moment `additional_gpu_pairs` is non-empty, so until now none of those experts
        // were in the RAM copy: during a prompt they came back through page faults from the mapped files.
        // Measured 2026-10-06 on an RTX 2000 Ada + 5060 Ti rig (17/48 split, IQ3_S mapped from the GGUF,
        // --prefill auto, one 73K-token prompt): 3.7 GiB read from the NVMe for the one prompt (the arena
        // mode reads 1.1 GiB), and one 8192-token chunk spent 1.5 s of its 3.9 s on host staging (arena:
        // 0.37 s).  Size each part's region at the largest chunk the scan can ever pick - auto's ceiling, or an
        // explicit --prefill, which the scan only walks down - take its tail slots with the scan's own
        // arithmetic, and list the pairs from the highest slot down, as the borrowing takes them.  #340 is
        // honoured the same way (the same free-VRAM test, at this startup moment): a part whose card can
        // fund the bound chunk's buffers outright lends nothing at any pick, so it gets no region and no RAM
        // is spent on it.  The single-GPU loan is untouched: without a split the lend region stays in
        // pin_cache_complement's own path, and no regions are set.
        std::vector<std::vector<std::pair<int32_t, int32_t>>> stage_lends;
        decide_split_ring();   // the ring the prompt path really uses, BEFORE bytes_needed sizes the regions below
        if (pf_borrow && d_res != nullptr && o.prefill_chunk > 0 && multi_gpu && !stages.empty()) {
            const int64_t bound = o.prefill_auto ? auto_ceiling : o.prefill_chunk;
            for (size_t i = 0; i <= stages.size(); ++i) {   // the scan's part order: CUDA0, then every stage
                strata::core::ExpertCache& xc = i == 0 ? xcache : stages[i - 1]->cache;
                const strata::core::SessionState& ses = i == 0 ? ss : stages[i - 1]->ss;
                const int64_t dev = i == 0 ? -1 : stages[i - 1]->dev;
                const int64_t lb = i == 0 ? 0 : stages[i - 1]->lb;
                const int64_t le = i == 0 ? split_at[0] : stages[i - 1]->le;
                if (xc.slots() <= 0 || le <= lb) continue;
                const uint64_t need = strata::prefill::Prefill::bytes_needed(g, ses, bound);
                if (split_own_ok) {   // the card funds its buffers at any chunk the scan can pick: nothing is lent
                    const strata::core::OnDevice on(dev);
                    size_t fb = 0, tb = 0;
                    if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
                        if ((uint64_t) fb >= need + (3ull << 29)) continue;
                    } else (void) cudaGetLastError();
                }
                // its tail slots, as the scan's `part_slots` counts them
                int64_t k = strata::program::split_lend_slots(
                    xc.slots(), xc.bytes(), xc.slot_offsets(),
                    (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
                k = std::min(k, xc.slots() - g_lend_keep);                       // the scan's 128-slot floor
                if (o.prefill_auto) k = std::min(k, kAutoLendPct * xc.slots() / 100);   // the auto cap
                if (k <= 0) continue;
                std::vector<std::pair<int32_t, int32_t>> region;
                for (int64_t l = lb; l < le; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (xc.slot_of(l, e) >= xc.slots() - k) region.emplace_back((int32_t) l, (int32_t) e);
                std::sort(region.begin(), region.end(),
                          [&](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
                              return xc.slot_of(a.first, a.second) > xc.slot_of(b.first, b.second);
                          });
                if (!region.empty()) stage_lends.push_back(std::move(region));
            }
            src.stage_lend_regions(std::move(stage_lends));
        }
        if (const int status = model.pin_resident_experts(stage_pairs, lend_from); status != 0) return status;
    }
    model.finish_expert_tiers();
    if (o.serve) {
        if (o.spec < 2 || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            std::fprintf(stderr, "strata serve: needs --spec T and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
            return 2;
        }
        // STRATA_VOLTA_BUILD backport (see docs/ and ninfer-flash-next-v100-backport): --mtp is optional here,
        // mirroring the plain `generate` command's own use_mtp handling below. Without it, drafting falls back
        // to the suffix/prompt-lookup drafter (DraftPolicy still picks Lookup vs plain per round) or, when
        // neither fires, plain one-token-per-round decoding - slower, never wrong: the verify window still
        // confirms every emitted token against the real model regardless of where the draft came from.
        const bool use_mtp = !o.mtp.empty();
        // STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path may take a blob a helper's VRAM cache already
        // holds from that cache instead of reading the files again.  The helpers are idle while a prompt is read
        // (the serve path is serial: no batch slots), so the copy needs no ordering against their kernel streams.
        // Everything else - the fallback's batched unbuffered reads, its pinned RAM copy, the prompt path's own
        // decisions - is forwarded unchanged.  Default off: the prompt path's source is srcp, exactly as before.
        std::vector<strata::core::RemoteExperts*> prefill_helpers;
        for (int r = 0; r < drive.d.remote_count; ++r) prefill_helpers.push_back(drive.d.remote[r]);
        std::unique_ptr<strata::core::RemotePrefillSource> prefill_helper_src;
        strata::core::ExpertSource* prefill_src = srcp;
        uint64_t helper_blobs_before = 0, helper_bytes_before = 0;
        if (const char* hv = std::getenv("STRATA_PREFILL_HELPER_SOURCE"); hv != nullptr && std::atoi(hv) != 0 &&
            srcp != nullptr && !multi_gpu && stages.empty() && o.batch <= 0 && !prefill_helpers.empty()) {
            prefill_helper_src = std::make_unique<strata::core::RemotePrefillSource>(*srcp, prefill_helpers);
            prefill_src = prefill_helper_src.get();
            std::fprintf(stderr, "strata serve: the prompt path reads the experts a helper cache holds from that "
                                 "cache (STRATA_PREFILL_HELPER_SOURCE)\n");
        }
        strata::prefill::Prefill sp;
        // 149-prefetch STRATA_RAM_ADAPT2=1 (with STRATA_RAM_ADAPT=n, the promotions staged per window): see the window loop
        struct RamAdapt2 {
            bool on = false;
            float prompt_w = 0.25f, cached_w = 0.15f;
            int per_layer = 2;
            int skip_top = 2;
            int64_t installed = 0;
            std::vector<float> seed;
        } ram_adapt2;
        {
            const char* v = std::getenv("STRATA_RAM_ADAPT2");
            ram_adapt2.on = v != nullptr && std::atoi(v) != 0;
            if (const char* w = std::getenv("STRATA_RAM_ADAPT_PROMPT_W")) ram_adapt2.prompt_w = (float) std::max(0.0, std::atof(w));
            if (const char* w = std::getenv("STRATA_RAM_ADAPT_CACHED_W")) ram_adapt2.cached_w = (float) std::clamp(std::atof(w), 0.0, 1.0);
            if (const char* w = std::getenv("STRATA_RAM_ADAPT_PER_LAYER")) ram_adapt2.per_layer = std::max(1, std::atoi(w));
            if (const char* w = std::getenv("STRATA_RAM_ADAPT_SKIP_TOP")) ram_adapt2.skip_top = std::max(0, std::atoi(w));
        }
        // the pool is idle while a prompt is read unless batch slots decode between its parts; with
        // STRATA_PREFILL_CPU_SHARE the staged-chunk limit before the chunk below sizes the loans (bytes_needed reads it)
        // #1607: Windows, the commit limit (RAM + page file) is what ends the engine silently; compare what this run plans
        // to allocate on the host from here (the conversation cache, checkpoints) with the commit that is left
#if defined(_WIN32)
        {
            MEMORYSTATUSEX cs{};
            cs.dwLength = sizeof cs;
            if (GlobalMemoryStatusEx(&cs)) {
                const uint64_t planned = (o.prompt_cache > 0 ? (uint64_t) std::max<int64_t>(o.conversation_cache_mib, 0) << 20 : 0) +
                                         (2ull << 30);   // checkpoints and prompt staging
                const auto cv = strata::platform::host_limits::commit_verdict((uint64_t) cs.ullAvailPageFile,
                                                                              (uint64_t) cs.ullTotalPageFile, planned, 4ull << 30);
                std::fprintf(stderr, "strata serve: Windows commit: %.1f GiB left of a %.1f GiB limit (RAM + page file)\n",
                             (double) cs.ullAvailPageFile / 1073741824.0, (double) cs.ullTotalPageFile / 1073741824.0);
                if (cv.warn) std::fprintf(stderr, "strata serve: %s\n", cv.text.c_str());
            }
        }
#endif
        // STRATA_PREFILL_CPU_SHARE_BATCH=1 (opt-in): the prompt path gets the pool beside batch slots too; a read takes the
        // share only while no slot decodes (set_share_blocked before each read), since the slots' windows use the pool
        static const bool share_batch = [] { const char* v = std::getenv("STRATA_PREFILL_CPU_SHARE_BATCH"); return v != nullptr && v[0] == '1'; }();
        const bool share_pool = (o.batch <= 0 || share_batch) && !o.no_pool;
        if (share_batch && o.batch > 0 && !o.no_pool)
            std::fprintf(stderr, "strata serve: STRATA_PREFILL_CPU_SHARE_BATCH=1: the prompt path may use the CPU pool beside the "
                                 "%lld batch slots, for a read with no slot decoding\n", (long long) o.batch);
        bool share_has_work = strata::prefill::share_rules::default_share_has_work(
            (int64_t) xcache.slots(), (int64_t) (g.n_layers * g.n_expert));   // #1595
        if (!share_has_work)
            std::fprintf(stderr, "strata serve: every expert is in the GPU cache (%lld of %lld): the CPU prefill share has "
                                 "nothing to take, so it is not armed by default (STRATA_PREFILL_CPU_SHARE=auto forces it)\n",
                         (long long) xcache.slots(), (long long) (g.n_layers * g.n_expert));
        if (share_pool && stages.empty() && !multi_gpu && share_has_work && !model.is_cpu_share_default_ok()) share_has_work = false;   // H11
        strata::prefill::Prefill::arm_cpu_share(share_pool, share_pool && stages.empty() && !multi_gpu && share_has_work);
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        // #340: a layer split whose caches already hold most experts streams few of them through the prompt path, so
        // the 384-slot ring (sized for a card that streams nearly every expert of a chunk) only makes every stage's
        // loan bigger: 96 slots (0.1.30's ring here) when >= 75% of the (layer, expert) pairs are resident.  One GPU
        // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
        // (decided above, before the lend regions were sized: a no-op here unless the resident RAM mode skipped it)
        decide_split_ring();
        int64_t pf_plan_chunk = 0;   // #2824: the host-aware plan's chunk for prompts longer than o.prefill_chunk (0: none)
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
        // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
        // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
        if (pf_borrow && d_res != nullptr) {
            pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
            for (auto& st : stages)
                pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
            int64_t chunk = scan(nullptr);
            // The small-card prompt path (strata::small_card_prompt_path; owner's yes 2026-11-02): this chunk is what the DEFAULT prompt path got from the cache.
            // Below 8,192 tokens the default streams every expert again for every chunk, and the unfused experts plus (from two chunks) the layer-major walk
            // read the same prompt 9-50% sooner (RESULTS-DF2MEM.md); the chunk is then sized again for that layout.  One CUDA card, a native pack, an auto
            // chunk; STRATA_PF_FUSED / STRATA_PREFILL_LAYER_MAJOR set by the user win; STRATA_PF_SMALL=0 turns the rule off.
            if (chunk > 0 && o.prefill_auto && !multi_gpu && stages.empty() && !(std::getenv("STRATA_PF_SMALL") != nullptr && std::getenv("STRATA_PF_SMALL")[0] == '0')) {
#if defined(STRATA_USE_HIP)
                constexpr bool pf_small_hip = true;
#else
                constexpr bool pf_small_hip = false;
#endif
                const char* fe = std::getenv("STRATA_PF_FUSED");
                const char* le = std::getenv("STRATA_PREFILL_LAYER_MAJOR");
                const strata::SmallCardPromptPath pp = strata::small_card_prompt_path(
                    chunk, pf_small_hip, 1, strata::kernels::cpu::expert_layout().native, strata::qwen_defaults_026(),
                    fe != nullptr && fe[0] != '\0', le != nullptr && le[0] != '\0');
                if (pp.any()) {
                    const int64_t chunk_default = chunk;
                    if (pp.fused_off) strata::prefill::pf_fused_override().store(0);
                    if (pp.layer_major)
                        strata::platform::set_env("STRATA_PREFILL_LAYER_MAJOR", "1", false);
                    chunk = scan(nullptr);
                    if (chunk <= 0) {   // the unfused layout's buffers do not fit anywhere: keep the default path as it was
                        strata::prefill::pf_fused_override().store(-1);
                        if (pp.layer_major)
                            strata::platform::unset_env("STRATA_PREFILL_LAYER_MAJOR");
                        chunk = scan(nullptr);
                        std::fprintf(stderr, "strata serve: small expert cache: the unfused prompt path does not fit here; the default path stays (chunk %lld tokens)\n",
                                     (long long) chunk);
                    } else
                    std::fprintf(stderr, "strata serve: small expert cache: the default prompt path would read in chunks of %lld tokens (under %lld), "
                                         "so prompts read with the unfused experts%s (chunk %lld tokens); STRATA_PF_SMALL=0 keeps the default path, "
                                         "STRATA_PF_FUSED / STRATA_PREFILL_LAYER_MAJOR set by hand win\n",
                                 (long long) chunk_default, (long long) strata::kSmallCardChunkBelow,
                                 pp.layer_major ? " and, from two chunks, the layer-major walk" : "", (long long) chunk);
                }
            }
            // #448: a small card in a layer split caps every stage's chunk (an RTX 3080's 512-slot cache held a
            // 32 GB card's split to 512 tokens: prompts 6.2x slower, decode the same).  Named when it bites, so the
            // regression is one log line: each stage that cannot fund the chunk CUDA0 alone would read in.
            const int64_t alone = pf_parts.size() > 1 ? scan(&pf_parts[0]) : chunk;
            // both scans left the ring override on their own last trial: put it back on the chunk that won, which
            // is what `init` lays out and what the INFO line reports
            if (chunk > 0 && o.prefill_auto && strata::prefill::Prefill::ring_bytes_enabled()) {
                // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed
                // it); the byte-budget ring is for the chunks past it
                const int64_t small = old_pick(nullptr);
                strata::prefill::Prefill::set_ring_budget(
                    small >= chunk ? 0 : (int) ring_room(chunk, nullptr, strata::prefill::Prefill::ring_cap_for(small)),
                    small);
            }
            // #2824: the VRAM pick, priced against the host (RAM left for the page cache, the displaced experts with no RAM
            // backup, the drive): a smaller chunk / ring when the file set the prompt re-reads outgrows the page cache.
            // The RAM copy is built by now, so its bytes and the lend slots it kept are known.
            const int64_t chunk_vram = chunk;
            if (chunk > 0 && o.prefill_auto && strata::prefill::Prefill::ring_bytes_enabled()) {
                const int64_t small0 = old_pick(nullptr);
                const int64_t cap0 = strata::prefill::Prefill::ring_cap_for(small0);
                const int64_t ring0 = small0 >= chunk ? 0 : ring_room(chunk, nullptr, cap0);
                auto restore_ring = [&] { strata::prefill::Prefill::set_ring_budget((int) ring0, small0); };
                restore_ring();
                uint64_t gc_bytes = 0;
                int64_t gc_slots = 0;
                for (const PfPart& p : pf_parts) {
                    gc_bytes += (uint64_t) std::max<int64_t>(0, (int64_t) p.cache->bytes());
                    gc_slots += p.cache->slots();
                }
                chunk = host_plan_chunk(
                    chunk, gc_bytes, gc_slots, [&](int64_t c) { return fits(c, true, nullptr); },
                    [&](int64_t c, uint64_t& bytes, int64_t& slots) {
                        bytes = 0;
                        slots = 0;
                        for (const PfPart& p : pf_parts) {
                            const int64_t k = part_slots(p, c);
                            if (k <= 0) continue;
                            slots += k;
                            bytes += part_bytes(p, (int32_t) (p.cache->slots() - k));
                        }
                    },
                    [&](int64_t c) { return ring_room(c, nullptr, cap0); }, restore_ring);
            }
            // the buffers stay sized for today's chunk; the plan's smaller chunk applies to prompts longer than it (planned_chunk)
            if (chunk > 0 && chunk != chunk_vram) {
                pf_plan_chunk = chunk;
                chunk = chunk_vram;
            }
            if (chunk > 0 && alone > chunk_vram) {
                for (size_t i = 1; i < pf_parts.size(); ++i) {
                    const PfPart& p = pf_parts[i];
                    if (fits_one(p, alone, o.prefill_auto)) continue;
                    const int dev = p.dev < 0 ? 0 : p.dev;
                    cudaDeviceProp prop{};
                    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
                        (void) cudaGetLastError();
                        prop.name[0] = 0;
                    }
                    const std::string pct =
                        o.prefill_auto ? ", and lend at most " + std::to_string(kAutoLendPct) + "%" : "";
                    std::fprintf(stderr, "strata serve: WARNING: prompt chunk %lld tokens, not %lld: CUDA%d (%s) "
                                         "has %lld expert-cache slots, and a %lld-token chunk borrows %lld of them "
                                         "(it must keep 128%s) - prompts read slower than on CUDA0 alone (#448)\n",
                                 (long long) chunk_vram, (long long) alone, dev, prop.name,
                                 (long long) p.cache->slots(), (long long) alone,
                                 (long long) part_slots(p, alone), pct.c_str());
                    // the helper tiers start at CUDA1 without a split (and are enabled in order)
                    std::fprintf(stderr, "strata serve:   a card this small can serve as a helper expert cache "
                                         "instead of a split stage: without --layer-split, with %s "
                                         "(docs/SECOND_GPU.md)\n",
                                 dev == 1 ? "--expert-cache-device1 N" : "--expert-cache-device1..3 N, in order");
                }
            }
            if (chunk > 0) {
                if (o.prefill_auto)
                    // the RESOLVED ring, not the room offered: ring_slots() clamps to [16, RING_MAX] and a chunk
                    // under stream_all_min() gets STAGE, and STRATA_PREFILL_RING overrides both.  This is what
                    // `init` lays out, and what the INFO line reports to the Monitor tab.
                {
                    std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens, a %lld-slot ring\n",
                                 (long long) chunk,
                                 (long long) strata::prefill::Prefill::ring_slots_for(chunk));
                    if (pf_plan_chunk > 0)
                        std::fprintf(stderr, "strata serve: prompts longer than %lld tokens read in equal chunks of at most %lld "
                                             "(the host-aware plan); shorter ones in one chunk\n",
                                     (long long) chunk, (long long) pf_plan_chunk);
                }
                else if (chunk != o.prefill_chunk)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in "
                                         "every expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                for (PfPart& p : pf_parts) {
                    p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                    p.first_now = p.first;
                }
                // #340: a split stage whose card still has room for the chunk's buffers (its cache already holds
                // every expert of its layers, so auto stopped short of its VRAM) keeps them as its own instead of
                // borrowing cache slots: nothing of it is lent, streamed during the prompt or refilled after it.
                // Room = the buffers + 1.5 GiB (the verify windows, the draft head, hipBLAS/cuBLAS workspaces made
                // after this).  Layer splits only - one GPU keeps its loan exactly as before.
                if (pf_parts.size() > 1 && split_own_ok) {
                    for (PfPart& p : pf_parts) {
                        const strata::core::OnDevice on(p.dev);
                        size_t fb = 0, tb = 0;
                        if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                        const uint64_t need = strata::prefill::Prefill::bytes_needed(g, *p.ses, chunk, srcp != nullptr);
                        if ((uint64_t) fb >= need + (3ull << 29)) {
                            std::fprintf(stderr, "strata serve:   CUDA%d keeps its own prompt buffers (%.2f GiB of "
                                                 "%.2f GiB free): no loan\n", p.dev < 0 ? 0 : p.dev,
                                         (double) need / 1073741824.0, (double) fb / 1073741824.0);
                            p.first = -1;
                            p.first_now = -1;
                        }
                    }
                }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            } else if (pf_parts.size() > 1) {
                // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
                // here and so do we, rather than refusing to start - but say what every stage has, because a split
                // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
                // and `init` would otherwise report only that the buffers do not fit.
                std::fprintf(stderr, "strata serve: no stage can lend the prompt path its %lld-token buffers, so "
                                     "each stage allocates its own:\n", (long long) o.prefill_chunk);
                for (const PfPart& p : pf_parts)
                    std::fprintf(stderr, "strata serve:   CUDA%d has %lld slots, and a %lld-token chunk needs "
                                         "the last %lld of them\n", p.dev < 0 ? 0 : p.dev,
                                 (long long) p.cache->slots(), (long long) o.prefill_chunk,
                                 (long long) part_slots(p, o.prefill_chunk));
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
        }
        if (o.prefill_head_alias) {
            for (PfPart& p : pf_parts) {
                p.first = p.cache->slots() > 1 ? 1 : -1;
                p.first_now = p.first;
            }
            lend_first = pf_parts[0].first;
            borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
            borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            static constexpr int64_t candidates[] = {2048, 1536, 1024, 512};
            for (const int64_t candidate : candidates)
                if (strata::prefill::Prefill::bytes_needed(g, ss, candidate) + (512ull << 20) <=
                    free_b + borrow_bytes) {
                    o.prefill_chunk = candidate;
                    break;
                }
            std::fprintf(stderr, "strata serve: prompt chunk head alias: %lld tokens with %lld MiB borrowed and "
                                 "%lld MiB free\n",
                         (long long) o.prefill_chunk, (long long) (borrow_bytes >> 20), (long long) (free_b >> 20));
        }
#if defined(STRATA_GPU_METAL)
        // Metal: the prompt path's buffers are the Metal window's own (a region of its expert arena, metal/src/engine/window.cpp) - there is
        // no loan of expert-cache slots.  Taking one anyway made short prompts end in `Prefill::relayout` ("a chunk of at most 0": the Metal
        // Prefill has no borrowed layout) and long ones refill "lent" slots through a copy into device allocations the Metal window had given back.
        for (PfPart& p : pf_parts) { p.first = -1; p.first_now = -1; }
        lend_first = -1;
        borrow = nullptr;
        borrow_bytes = 0;
#endif
        // STRATA_MTP_ROOM_MIN=TOKENS (opt-in): a request that READS at least TOKENS prompt tokens (0: every request)
        // points the prompt path's ring AT the MTP drafter's 512 routed experts - 708 MB that only the DECODE reads
        // (record_rest; the prompt's own passes, record_front's K/V and the prompt path's Prefill::draft_kv, run the
        // layer's dense tensors and its K/V) - for as long as the prompt is read.  The block is lent IN PLACE: it is
        // never freed, never allocated again and never moves, so no prompt path is laid out again, the decode's
        // captured graphs stay valid (nothing is dropped), and what comes back before the first window is the
        // decode's own bytes (MtpDrafter::experts_reclaim).  TOKENS counts the tokens this request reads, not the
        // whole prompt: a turn that reuses a cached conversation prefix and reads a few hundred tokens gains nothing
        // from the room and does not take it.  Unset (the default): not one line below runs, and the prompt path is
        // laid out exactly as before.
        const int64_t mtp_room_min = [] {
            const char* v = std::getenv("STRATA_MTP_ROOM_MIN");
            if (v == nullptr || v[0] == '\0') return (int64_t) -1;
            char* end = nullptr;
            const long long n = std::strtoll(v, &end, 10);
            if (end == v || *end != '\0' || n < 0) return (int64_t) -2;   // not a count: say so, not "0: every request"
            return (int64_t) n;
        }();
        if (mtp_room_min == -2) {
            std::fprintf(stderr, "strata generate: STRATA_MTP_ROOM_MIN=%s is not a token count (0: every request)\n",
                         std::getenv("STRATA_MTP_ROOM_MIN"));
            return 2;
        }
        int mtp_room_slots = 0;
        if (mtp_room_min >= 0) {
            const char* why = !o.serve ? "it needs --serve"
                            : o.mtp.empty() ? "it needs --mtp"
                            : o.batch > 0 ? "it does not support --batch: its slot drafters share the drafter's weights"
                            : mtp.experts_bytes() == 0 ? "the drafter's 512 experts are not this drafter's to lend"
                            : nullptr;
            if (why != nullptr) {
                std::fprintf(stderr, "strata generate: STRATA_MTP_ROOM_MIN is unavailable: %s\n", why);
                return 2;
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            mtp_room_slots = blob > 0 ? (int) std::min<int64_t>(mtp.experts_bytes() / blob, 4096) : 0;
            std::fprintf(stderr, "strata serve: STRATA_MTP_ROOM_MIN=%lld: a request that reads %lld prompt tokens or "
                                 "more points the prompt path's ring on CUDA%d at the drafter's %llu MiB of experts "
                                 "(%d slots past its own) for as long as it is read, and the decode's own bytes are back "
                                 "before the first window\n",
                         (long long) mtp_room_min, (long long) mtp_room_min, mtp.device(),
                         (unsigned long long) (mtp.experts_bytes() >> 20), mtp_room_slots);
        }
        bool any_loan = borrow != nullptr;
        for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
        if (nib::any_on()) {
            // STRATA_EXPERT_NIBBLE: the prompt path keeps its own buffers (a loan is refilled with GGUF blobs) and reads in
            // the chunk it reads without the switch (the chunking is part of the replies' bits: auto's ceiling, or
            // --prefill N, and no host-aware plan of smaller chunks).  Otherwise refused, never a quietly different reply.
            std::string why;
            const int64_t expect = o.prefill_auto ? auto_ceiling : nib_chunk_asked;
            if (any_loan) why = "the prompt path borrows expert-cache slots (a loan is refilled with GGUF blobs)";
            else if (o.prefill_chunk != expect)
                why = "the prompt chunk is " + std::to_string(o.prefill_chunk) + " tokens, not the " + std::to_string(expect) +
                      " it is without the switch when nothing limits it (that would change the replies' bits; an explicit "
                      "--prefill N that fits keeps N)";
            else if (pf_plan_chunk > 0)
                why = "the host-aware plan reads long prompts in chunks of " + std::to_string(pf_plan_chunk) + " tokens";
            std::string room;
            for (size_t i = 0; why.empty() && i <= stages.size(); ++i) {
                const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                const strata::core::OnDevice on(dev);
                size_t fb = 0, tb = 0;
                if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                char b[64];
                std::snprintf(b, sizeof b, "CUDA%d %.2f GiB", dev < 0 ? 0 : dev, (double) fb / 1073741824.0);
                room += (room.empty() ? "" : ", ") + std::string(b);
            }
            if (!why.empty()) {
                std::fprintf(stderr, "strata serve: STRATA_EXPERT_NIBBLE refuses the start: %s (STRATA_EXPERT_NIBBLE_SKIP=<layers> "
                                     "keeps some layers' GGUF blocks and their smaller slots)\n", why.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata serve: STRATA_EXPERT_NIBBLE: prompt chunk %lld, no loan; free after the prompt "
                                 "buffers: %s\n", (long long) o.prefill_chunk, room.c_str());
        }
        if (any_loan) {
            if (borrow != nullptr)
                std::fprintf(stderr, "strata serve: the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)\n",
                             (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
            for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
                if (pf_parts[i].first >= 0) std::fprintf(stderr, "strata serve:   CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)\n",
                             pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                             (long long) pf_parts[i].cache->slots(),
                             (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
            // A split's RAM copy keeps the experts of the slots the prompt path borrows (the lend regions sized above):
            // every borrowed expert not kept is streamed from the GGUF mmap during every chunk.  Say both counts side
            // by side - they used to differ whenever a ring override (STRATA_SPLIT_RING) was set, because the regions
            // were sized with the default ring before the override applied.
            if (multi_gpu && src.complement_ready()) {
                int64_t borrowed = borrow != nullptr ? xcache.slots() - lend_first : 0;
                for (size_t i = 1; i < pf_parts.size(); ++i)
                    if (pf_parts[i].first >= 0) borrowed += pf_parts[i].cache->slots() - pf_parts[i].first;
                const int64_t kept = src.resident_lent_slots();
                std::fprintf(stderr, "strata serve: lend sizing: the prompt path borrows %lld slots (ring %lld slots at "
                                     "chunk %lld), %lld of them keep their experts in RAM too\n",
                             (long long) borrowed,
                             (long long) strata::prefill::Prefill::ring_slots_for(o.prefill_chunk),
                             (long long) o.prefill_chunk, (long long) kept);
                if (kept < borrowed)
                    std::fprintf(stderr, "strata serve: WARNING: %lld of the %lld borrowed slots have no RAM copy: "
                                         "their experts are streamed from the model file during every chunk\n",
                                 (long long) (borrowed - kept), (long long) borrowed);
            }
        } else {
            std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        }
        rss_probe("the prompt path set up");
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next.
        // THE CHUNK STEPS DOWN INSTEAD OF EXITING.  A split's stage caches are sized after the arena is registered, and
        // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
        // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
        // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
        // The path the room belongs to (set below): the one on the drafter's card, which is where the ring's slots
        // point into the drafter's expert block.  The room's LAYOUT is a start-up property - it costs nothing to
        // hold (no allocation, no VRAM: the slots ARE the drafter's bytes) - and a prompt takes it by moving that
        // path's ring count (set_ring_room_active), so any later rebuild here re-carves it the same way.
        strata::prefill::Prefill* room_path = nullptr;
        auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
            room_path = nullptr;
            for (size_t i = 0; i < stages.size(); ++i) {
                GpuStage& st = *stages[i];
                st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
                const strata::core::OnDevice on(st.dev);
                void* sb = nullptr;              // this stage's own loan, out of its own cache
                uint64_t sbb = 0;
                // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
                if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
                    sb = st.cache.device_slot(pf_parts[i + 1].first);
                    sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
                }
                // STRATA_MTP_ROOM_MIN: the drafter's card is the one whose prompt path is laid out with the room
                const bool room_here = mtp_room_slots > 0 && mtp.device() == st.dev;
                if (room_here) room_path = &st.sp;
                st.sp.set_ring_room(room_here ? mtp_room_slots : 0);
                st.sp.set_ring_room_base(room_here ? mtp.experts_base() : nullptr, room_here ? mtp.experts_bytes() : 0);
                if (share_pool) st.sp.set_cpu_pool(&pool);   // one stage at a time takes it for a chunk (prefill.cpp)
                if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream,
                                err, sb, sbb, o.prefill_head_alias)) {
                    err = "layer split, CUDA" + std::to_string(st.dev) + " prompt path: " + err;
                    return err.find("do not fit") != std::string::npos ? 2 : 1;
                }
            }
            if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
            if (share_pool) sp.set_cpu_pool(&pool);
            int cur_dev = 0;
            cudaGetDevice(&cur_dev);   // the device this path is about to run on (dev -1: the current one)
            const bool room_here = mtp_room_slots > 0 && mtp.device() == cur_dev;
            if (room_here) room_path = &sp;
            sp.set_ring_room(room_here ? mtp_room_slots : 0);
            sp.set_ring_room_base(room_here ? mtp.experts_base() : nullptr, room_here ? mtp.experts_bytes() : 0);
            if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
                         o.prefill_head_alias))
                return err.find("do not fit") != std::string::npos ? 2 : 1;
            if (mtp_room_slots > 0 && room_path == nullptr)   // the drafter's card holds none of the prompt paths
                std::fprintf(stderr, "strata serve: prompt room: no prompt path runs on CUDA%d, the drafter's card: the "
                                     "prompt reads with its own ring\n", mtp.device());
            return 0;
        };
        static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
        {
            // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave
            // headroom on that device (a chunk that fits to the last MiB left hipBLAS nothing: its GEMMs then
            // failed to launch on gfx1201 and the prompt hung).  `bytes_needed` is the same count `init` makes.
            const int64_t kHeadroom = (o.prefill_chunk <= 512 ? 128ll : 512ll) << 20;   // (a 512-token chunk or less is the tight-card path: its buffers are priced with a floor of their own)
            auto own_fits = [&](int64_t c, int& dev_out, int64_t& need_out, int64_t& free_out) -> bool {
                for (size_t i = 0; i <= stages.size(); ++i) {
                    const bool loan = i == 0 ? borrow != nullptr : (i < pf_parts.size() && pf_parts[i].first >= 0);
                    if (loan) continue;
                    const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                    const strata::core::OnDevice on(dev);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    const int64_t need = (int64_t) strata::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[i - 1]->ss, c, srcp != nullptr);
                    if (need + kHeadroom > (int64_t) fb) {
                        dev_out = dev < 0 ? 0 : dev; need_out = need; free_out = (int64_t) fb;
                        return false;
                    }
                }
                return true;
            };
            int dev = 0;
            int64_t need = 0, fb = 0;
            // gfx1151 (Strix Halo, Windows): `prefill_own_keep_chunk` is set above because the prompt path keeps its own buffers at the 16,384-token
            // chunk and the cache sizing already reserved them.  The "free VRAM" read here is the unified pool's figure after the pinned RAM complement
            // (which the pool counts twice), not a limit: with 860 MiB "free" the MTP engine allocates the 6.7 GiB of buffers anyway and reads prompts at
            // 955 tok/s, but this step-down found 1.7 GiB "free" in the DFlash2-alone engine (no MTP layer, a slightly larger cache), cut the chunk to
            // 1,536 and read them at 303 tok/s (RESULTS-DF2STRIX.md).  The step-down stays for a real allocation failure (the loop below).
            // STRATA_PF_OWN_CHECK=1 (tester) restores the arithmetic check on this card.
            static const bool own_check_restored = [] { const char* v = std::getenv("STRATA_PF_OWN_CHECK"); return v != nullptr && v[0] == '1'; }();
            const bool keep_chunk_uma = o.prefill_own_keep_chunk && !own_check_restored;
            if (!own_fits(o.prefill_chunk, dev, need, fb)) {
              if (keep_chunk_uma) {
                std::fprintf(stderr, "strata serve: a %lld-token chunk's prompt buffers need %lld MiB, %lld MiB are reported free: kept (this GPU's memory is the "
                                     "unified pool; STRATA_PF_OWN_CHECK=1 steps the chunk down by that figure)\n",
                             (long long) o.prefill_chunk, (long long) (need >> 20), (long long) (fb >> 20));
              } else {
                int64_t c = 0;
                for (const int64_t s : kStepChunks) {
                    int d2 = 0;
                    int64_t n2 = 0, f2 = 0;
                    if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
                }
                if (c > 0) {
                    std::fprintf(stderr, "strata serve: a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                         "%lld MiB free: %lld-token chunks\n", (long long) o.prefill_chunk,
                                 (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                    o.prefill_chunk = c;
                }
              }
            }
        }
        mem_mark("before the prompt path init");
        for (;;) {
            const int r = init_prompt_paths();
            if (r == 0) break;
            int64_t next = 0;
            for (const int64_t c : kStepChunks)
                if (c < o.prefill_chunk) { next = c; break; }
            if (r == 2 && next > 0) {
                std::fprintf(stderr, "strata serve: %s: trying a %lld-token chunk\n", err.c_str(), (long long) next);
                // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
                // error: it is sticky, and the next launch check would report "out of memory" for a kernel
                for (auto& stp : stages) {
                    const strata::core::OnDevice on(stp->dev);
                    stp->sp.reset();
                    (void) cudaGetLastError();
                }
                sp.reset();
                (void) cudaGetLastError();
                o.prefill_chunk = next;
                if (any_loan) {                  // smaller loans for the smaller chunk
                    for (PfPart& p : pf_parts)
                        if (p.first >= 0) {
                            p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                            p.first_now = p.first;
                        }
                    lend_first = pf_parts[0].first;
                    borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                    borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
                }
                err.clear();
                continue;
            }
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
                std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                     "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)\n");
            return 1;
        }
        uma_refuse_chunk_if_full(o.prefill_chunk, borrow != nullptr, xcache.gib());
        // The peer's streaming and a split's helpers live INSIDE the layout (`set_peer` / `set_stage_helper` build
        // them in the path's own Impl), so a path that is laid out again - the prompt room, below - has to be armed
        // again.  Both calls start by dropping what they built before, and both leave `err` clear on a decline.
        auto arm_prompt_helpers = [&]() {
            if (peer.valid() && o.peer_prefill_rows != 0) {
                const int64_t rows = o.peer_prefill_rows > 0 ? o.peer_prefill_rows : o.prefill_chunk * K / 2;
                if (!sp.set_peer(&peer, rows, err)) {
                    std::fprintf(stderr, "strata serve: %s - the prompt path stays on the primary GPU\n", err.c_str());
                    err.clear();
                    (void) cudaGetLastError();   // (as above)
                }
            }
            // a layer split: a one-chunk prompt runs the stages one after the other, so each stage's prompt path gets the
            // next stage's GPU (the last one the first's) to stream and compute a share of its experts meanwhile
            if (multi_gpu && !stages.empty()) {
                std::vector<std::pair<strata::prefill::Prefill*, int>> paths{{&sp, 0}};
                for (const auto& st : stages) paths.emplace_back(&st->sp, st->dev);
                for (size_t i = 0; i < paths.size(); ++i) {
                    const strata::core::OnDevice on(paths[i].second);
                    if (!paths[i].first->set_stage_helper(paths[(i + 1) % paths.size()].first, err)) {
                        std::fprintf(stderr, "strata serve: %s - stage %zu reads its prompts alone\n", err.c_str(), i);
                        err.clear();
                    }
                }
            }
        };
        arm_prompt_helpers();
        mem_mark("the head and the prompt path");
        // #340: STRATA_SPLIT_SMALL_OWN=S (tokens): on a layer split, every stage that borrows keeps the slots for an
        // S-token chunk's buffers for the whole session (0.1.29's own buffers, carved from the tail of its cache):
        // a request of at most S prompt tokens then lends, streams and refills nothing, a longer one lends (and
        // refills) only the slots above them.  Their experts stay non-resident (the CPU / PCIe path computes them).
        // STRATA_SPLIT_SMALL_MAX=M: a request of at most M prompt tokens reads in S-token chunks on those stages
        // (several chunks, still nothing lent) instead of borrowing for a bigger one.
        int64_t split_small = 0, split_small_max = 0;
        if (multi_gpu && !pf_parts.empty() && !host_res.empty()) {
            const char* v = std::getenv("STRATA_SPLIT_SMALL_OWN");
            const int64_t S = v ? request_chunk(std::atoll(v), o.prefill_chunk) : 0;
            const char* vm = std::getenv("STRATA_SPLIT_SMALL_MAX");
            split_small = S;
            split_small_max = S > 0 ? std::max<int64_t>(S, vm ? std::atoll(vm) : S) : 0;
            int64_t evicted = 0;
            for (PfPart& p : pf_parts) {
                if (S <= 0 || p.first < 0) continue;
                const int32_t first_s = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, S)));
                int64_t n = 0;
                for (int64_t l = p.lb; l < p.le; ++l)
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        int32_t& r = host_res[(size_t) (l * g.n_expert + ex)];
                        if (r >= first_s) { r = strata::core::kNotResident; ++n; }
                    }
                evicted += n;
                std::fprintf(stderr, "strata serve:   CUDA%d keeps %lld slots for %lld-token prompts (%lld experts "
                                     "no longer resident)\n", p.dev < 0 ? 0 : p.dev,
                             (long long) (p.cache->slots() - first_s), (long long) S, (long long) n);
            }
            if (evicted > 0) {
                if (d_res != nullptr)
                    res_put(d_res);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    res_put(st->d_res);
                }
            }
        }
        kvg_start(borrow != nullptr ? (int64_t) lend_first : xcache.slots());
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
        // neutral path (no upload, no buffer handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        // (the widest window the run can have: a batch of more than kVerifyNarrowT slots is one way to a wider one, a
        // prompt-lookup window longer than that (--lookup-max-t) the other)
        const size_t kHistSlots = (size_t) kPenaltyWindowCap *
            (size_t) (o.lookup_max_t > (int) strata::kernels::kVerifyNarrowT ? strata::kernels::kVerifyLongT
                      : o.batch > strata::kernels::kVerifyNarrowT ? strata::kernels::kVerifyMaxT : strata::kernels::kVerifyNarrowT);
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const strata::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: the penalty-history allocation failed\n");
            return 1;
        }
        strata::core::Verifier ver;
        strata::core::VerifyHits vh = model.get_verify_hits();
        // STRATA_ROUTE_RESIDENT_MTP=1 (EXPERIMENTAL, with STRATA_ROUTE_RESIDENT=<margin>; changes the output): the MTP
        // draft's router gets the same residency bias, with the last layer's residency row as its table (single GPU)
        if (use_mtp && stages.empty() && thits.d_res != nullptr && std::getenv("STRATA_ROUTE_RESIDENT") != nullptr &&
            std::getenv("STRATA_ROUTE_RESIDENT_MTP") != nullptr && std::atoi(std::getenv("STRATA_ROUTE_RESIDENT_MTP")) != 0) {
            const float margin = (float) std::atof(std::getenv("STRATA_ROUTE_RESIDENT"));
            int lo = 6, hi = 9;
            if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT_RANKS")) std::sscanf(v, "%d-%d", &lo, &hi);
            if (margin > 0.0f) {
                mtp.set_route_resident(thits.d_res + (size_t) (g.n_layers - 1) * (size_t) g.n_expert, margin, lo, hi);
                std::fprintf(stderr, "strata generate: STRATA_ROUTE_RESIDENT_MTP: the MTP draft's router is biased toward layer %lld's resident experts (margin %g, ranks %d-%d; experimental)\n",
                             (long long) (g.n_layers - 1), margin, lo, hi);
            }
        }
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        strata::core::Verifier ver_same;
        SplitDrive split_drive;
        auto stage_ver = [&](int st) -> strata::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        // a layer split's hand-off buffers (and its pipelined windows) hold kVerifyMaxT rows: --lookup-max-t stops there
        if (n_stages > 1 && o.spec > strata::kernels::kVerifyNarrowT) {
            std::fprintf(stderr, "strata serve: --lookup-max-t %d: a layer split's windows stop at %d tokens\n",
                         o.lookup_max_t, strata::kernels::kVerifyNarrowT);
            o.spec = strata::kernels::kVerifyNarrowT;
        }
        // --pipeline-windows (decided where the split was): two verifiers per stage, one for the even windows and one
        // for the odd ones, sharing one stream of their card, and a hand-off per parity - so stage 0 can run window
        // K+1 while stage 1 still reads window K's hand-off.  With more stages every stage but the last is a "front"
        // stage: window K+1 follows window K through them one card behind, and only the last stage waits for K's verdict
        const bool pipe = o.pipeline_windows > 0 && n_stages >= 2 && !split_same && (int) stages.size() == n_stages - 1;
        strata::core::Verifier ver_b;
        SplitDrive split_drive_b;
        std::vector<cudaStream_t> pl_stream(pipe ? (size_t) n_stages : 0, nullptr);
        // stage st's verifier of one parity (0: the even windows, 1: the odd ones)
        auto stage_ver_p = [&](int st, int par) -> strata::core::Verifier& {
            if (st == 0) return par == 0 ? ver : ver_b;
            return par == 0 ? stages[(size_t) st - 1]->ver : stages[(size_t) st - 1]->ver_b;
        };
        if (pipe) {
            bool ok_p = true;
            ok_p = cudaStreamCreateWithFlags(&pl_stream[0], cudaStreamNonBlocking) == cudaSuccess;   // the current card
            for (int st = 1; st < n_stages && ok_p; ++st) {
                const strata::core::OnDevice on(stages[(size_t) st - 1]->dev);
                ok_p = cudaStreamCreateWithFlags(&pl_stream[(size_t) st], cudaStreamNonBlocking) == cudaSuccess;
            }
            if (!ok_p) { std::fprintf(stderr, "strata serve: --pipeline-windows: stream creation failed\n"); return 1; }
            for (int st = 0; st < n_stages; ++st)
                for (int par = 0; par < 2; ++par) {
                    stage_ver_p(st, par).set_stream(pl_stream[(size_t) st]);
                    stage_ver_p(st, par).set_always_publish(true);
                }
        }
        if (n_stages > 1) {
            // pipelined slot groups write their rows at hbase = group * rows-per-group, so with more than
            // kVerifyNarrowT slots in groups the hand-off holds one row per slot (otherwise kVerifyNarrowT, as before; a
            // wide build holds more only for a batch of more than that: the buffer is not made larger for a run that
            // never forms a window of 9..16 rows)
            // (STRATA_PIPED_MTP: per slot its token and up to pm_drafts proposals)
            const int hrows = piped_mtp ? std::max<int>(strata::kernels::kVerifyNarrowT, (1 + pm_drafts) * o.batch)
                            : !batch_mtp && o.batch_groups > 1 ? std::max<int>(strata::kernels::kVerifyNarrowT, o.batch)
                            : !batch_mtp ? std::max<int>(strata::kernels::kVerifyNarrowT,
                                                         std::min<int>(o.batch, strata::kernels::kVerifyMaxT))
                                         : strata::kernels::kVerifyNarrowT;
            std::vector<float*> hand((size_t) n_stages - 1, nullptr);
            if (!alloc_split_handoffs(g, hrows, hand)) {
                std::fprintf(stderr, "strata serve: the layer-split hand-off allocation failed\n");
                return 1;
            }
            // STRATA_SPLIT_OVERLAP=1 (opt-in; measured on 2x RX 7900 XTX only): a ready word per hand-off, raised by the
            // writing stage's GPU, so the next stage's window is launched early and the host never syncs between
            // stages (Verifier::set_handoff_flags).  Not for --split-device 0 (both stages on one GPU).
            const bool overlap_asked = [] {
                const char* v = std::getenv("STRATA_SPLIT_OVERLAP");
                return v != nullptr && std::atoi(v) != 0;
            }();
            // --pipeline-windows (#859) runs its own overlapped loop with its own hand-offs: the two do not mix
            const bool overlap = overlap_asked && !split_same && o.pipeline_windows == 0;
            if (overlap_asked && o.pipeline_windows > 0)
                std::fprintf(stderr, "strata serve: STRATA_SPLIT_OVERLAP is off beside --pipeline-windows %d "
                                     "(that loop overlaps the stages itself)\n", o.pipeline_windows);
            std::vector<uint32_t*> hflag_h((size_t) n_stages - 1, nullptr), hflag_d((size_t) n_stages - 1, nullptr);
            if (overlap)
                for (size_t i = 0; i < hflag_h.size(); ++i) {
                    if (cudaHostAlloc((void**) &hflag_h[i], 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                        cudaHostGetDevicePointer((void**) &hflag_d[i], hflag_h[i], 0) != cudaSuccess) {
                        std::fprintf(stderr, "strata serve: the layer-split hand-off flag allocation failed\n");
                        return 1;
                    }
                    std::memset(hflag_h[i], 0, 64);
                }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
                if (hrows > strata::kernels::kVerifyNarrowT) stage_ver(st).set_handoff_rows(hrows);
#if defined(STRATA_HAVE_DFLASH2)
                if (use_df2 && st > 0) stage_ver(st).set_taps(df2.tap_layers());   // (stage 0 gets its taps below, as on one card)
                if (batch_df2 && st > 0) stage_ver(st).set_batch_taps(true);
#endif
                if (overlap)
                    stage_ver(st).set_handoff_flags(st == 0 ? nullptr : hflag_h[(size_t) st - 1],
                                                    st == 0 ? nullptr : hflag_d[(size_t) st - 1],
                                                    st + 1 < n_stages ? hflag_h[(size_t) st] : nullptr,
                                                    st + 1 < n_stages ? hflag_d[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = get_pcie_num(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max(o.spec, std::min(o.batch, (int) strata::kernels::kVerifyMaxT)), err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const strata::core::OnDevice on(gs.dev);
                    gs.ver.set_remote_expert_opt(remote_opt.get());
                    // --batch-mtp: every stage runs the same 2-rows-per-slot windows as the first (up to kVerifyMaxT
                    // rows, the hand-off's size) and bounds its captured batch layouts like the first (below)
                    if (batch_mtp) gs.ver.set_batch_graph_limit(64);
                    // (STRATA_PIPED_MTP: a group's window is its one slot's token and proposals, at most --spec rows,
                    // so the stage keeps the windows it has without --batch-mtp - the solo path's too)
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, model.get_verify_hits(gs), gs.head.loaded() ? &gs.head : nullptr,
                                       batch_max_t(), err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = get_pcie_num(gs.pcie_frac);
                }
                if (!ok_s) {
                    std::fprintf(stderr, "strata serve: layer split, stage %d: %s\n", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            std::fprintf(stderr, "strata serve: layer split: layers %s, one hand-off per window%s\n", plan_s.c_str(),
                         overlap ? " (overlapped: STRATA_SPLIT_OVERLAP=1)" : "");
        }
        // --pipeline-windows: the odd windows' verifiers and their hand-off (initialized before `ver`, which stays the
        // watchdog's verifier), and the drafter's own row buffer, which either parity's rows are copied into
        float* pl_mtp_R = nullptr;
        if (pipe) {
            std::vector<float*> hand_b((size_t) n_stages - 1, nullptr);   // the odd windows' hand-off per boundary
            if (!alloc_split_handoffs(g, strata::kernels::kVerifyNarrowT, hand_b)) {
                std::fprintf(stderr, "strata serve: --pipeline-windows: the second hand-off allocation failed\n");
                return 1;
            }
            ver_b.set_stage(0, split_at[0], nullptr, hand_b[0]);
            for (int st = 1; st < n_stages; ++st) {
                GpuStage& gs = *stages[(size_t) st - 1];
                const bool last = st + 1 == n_stages;
                gs.ver_b.set_stage(split_at[(size_t) st - 1], last ? -1 : split_at[(size_t) st], hand_b[(size_t) st - 1],
                                   last ? nullptr : hand_b[(size_t) st]);
                const strata::core::OnDevice on(gs.dev);
                if (!gs.ver_b.init(gs.wt, g, gs.ss, model.get_verify_hits(gs), gs.head.loaded() ? &gs.head : nullptr,
                                   o.spec, err)) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows: the second verifier on CUDA%d (stage %d): %s "
                                         "(raise --vram-reserve-mib by ~200, or --pipeline-windows 0)\n",
                                 gs.dev, st + 1, err.c_str());
                    return 1;
                }
                if (last && !mtp_first && cudaMalloc((void**) &pl_mtp_R, (size_t) strata::kernels::kVerifyNarrowT * (size_t) (g.hc * g.n_embd) *
                                                               sizeof(float)) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows: the drafter's row buffer does not fit\n");
                    return 1;
                }
            }
            if (mtp_first) {   // the drafter's row buffer on its own card (the first)
                const strata::core::OnDevice on_m(mtp.device());
                if (cudaMalloc((void**) &pl_mtp_R, (size_t) strata::kernels::kVerifyNarrowT * (size_t) (g.hc * g.n_embd) *
                                                       sizeof(float)) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows: the drafter's row buffer does not fit\n");
                    return 1;
                }
            }
            if (!ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
                std::fprintf(stderr, "strata serve: --pipeline-windows: the second verifier on CUDA0 (the first card): "
                                     "%s (raise --vram-reserve-mib by ~200, or --pipeline-windows 0)\n", err.c_str());
                return 1;
            }
            // the setters reach the later stages; the pipeline never chains run/commit
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver_p(st, 1).set_next(&stage_ver_p(st + 1, 1), &split_drive_b);
        }
        ver.set_remote_expert_opt(remote_opt.get());
        if (pcie_balance() && n_stages == 1) ver.set_pcie_balance(&drive.d.pcie_model);
        if (batch_mtp) ver.set_batch_graph_limit(64);   // --batch-mtp only: slot rotation makes many layouts (LRU); 0.1.39 keeps all
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2) ver.set_taps(df2.tap_layers());
        if (batch_df2) ver.set_batch_taps(true);
#endif
        if (!ver.init(wt, g, ss, vh, native_head.loaded() || prefill_head_alias_bound ? &native_head : nullptr,
                      batch_max_t(), err) ||
            (use_mtp && !prefill_head_alias_bound && !mtp.bind(last_st && !mtp_first ? last_st->wt : wt, last_st ? &last_st->head : &native_head,
                                  pipe ? pl_mtp_R : ver.final_R_all(), err))) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            return 1;
        }
        if (prefill_head_alias_bound) mtp.set_window_R(ver.final_R_all());
#if defined(STRATA_HAVE_DFLASH2)
#if defined(STRATA_HAVE_DFLASH2)
        // DFlash2 prompt taps on a layer split.  Every stage's prompt path taps the requested layers it runs into its own staging and hands them to its
        // on_taps callback (on that stage's thread).  The callback lands the rows in a landing [2048][taps][n_embd] on the drafter's card at their
        // position, and the drafter's context is updated for every position all stages have delivered (in position order, the solo path's per-chunk calls).
        struct Df2PromptLand {
            std::mutex mu;
            float* dev = nullptr;
            int dev_id = 0;
            int64_t first = 0, flushed = 0;
            std::vector<int64_t> reached;
            std::vector<int> slot0, nt;
            std::vector<cudaStream_t> cs;   // per stage: its copies into the landing (on the stage's own card)
            ~Df2PromptLand() { if (dev != nullptr) { cudaSetDevice(dev_id); cudaFree(dev); } }
        } dland;
        auto df2_stage_prefill = [&](int k) -> strata::prefill::Prefill& { return k == 0 ? sp : stages[(size_t) k - 1]->sp; };
        auto df2_stage_prompt_taps_reserve = [&](std::string& why) -> bool {
            const int64_t nl = g.n_layers;
            dland.nt.assign((size_t) n_stages, 0);
            dland.slot0.assign((size_t) n_stages, 0);
            dland.reached.assign((size_t) n_stages, 0);
            for (int k = 0; k < n_stages; ++k) {
                const int64_t lb = k == 0 ? 0 : split_at[(size_t) k - 1], le = k + 1 < n_stages ? split_at[(size_t) k] : nl;
                for (int l : df2.tap_layers()) {
                    if (l < 0 || l >= nl - 1) continue;
                    if (l < lb) ++dland.slot0[(size_t) k];
                    else if (l < le) ++dland.nt[(size_t) k];
                }
            }
            const strata::core::OnDevice on_d(df2.device());
            dland.dev_id = df2.device();
            if (cudaMalloc((void**) &dland.dev, (size_t) 2048 * (size_t) df2.n_taps() * (size_t) g.n_embd * 4) != cudaSuccess) {
                (void) cudaGetLastError();
                dland.dev = nullptr;
                why = "the prompt landing of the DFlash2 taps does not fit in VRAM";
                return false;
            }
            dland.cs.assign((size_t) n_stages, nullptr);
            for (int k = 0; k < n_stages; ++k) {
                if (dland.nt[(size_t) k] == 0) continue;
                const strata::core::OnDevice on_s(k == 0 ? 0 : stages[(size_t) k - 1]->dev);
                if (cudaStreamCreateWithFlags(&dland.cs[(size_t) k], cudaStreamNonBlocking) != cudaSuccess) { why = "the prompt tap copy stream"; return false; }
                if (!df2_stage_prefill(k).reserve_taps(df2.tap_layers(), 2048, g.n_embd, why)) return false;
            }
            return true;
        };
        auto df2_stage_prompt_taps_install = [&]() {
            for (int k = 0; k < n_stages; ++k) {
                if (dland.nt[(size_t) k] == 0) continue;
                df2_stage_prefill(k).on_taps = [&, k](const float* taps, int64_t rows, int64_t pos0, std::string& e) -> bool {
                    std::lock_guard<std::mutex> lk(dland.mu);
                    const size_t ntot = (size_t) df2.n_taps(), N = (size_t) g.n_embd;
                    const int64_t row = pos0 - dland.first;
                    if (row < 0 || row + rows > 2048) { e = "DFlash2 prompt taps: rows outside the landing"; return false; }
                    const size_t w = (size_t) dland.nt[(size_t) k] * N * 4;
                    cudaError_t ce = cudaMemcpy2DAsync(dland.dev + ((size_t) row * ntot + (size_t) dland.slot0[(size_t) k]) * N, ntot * N * 4, taps, w, w,
                                                       (size_t) rows, cudaMemcpyDefault, dland.cs[(size_t) k]);
                    if (ce == cudaSuccess) ce = cudaStreamSynchronize(dland.cs[(size_t) k]);
                    if (ce != cudaSuccess) { e = std::string("DFlash2 prompt taps: ") + cudaGetErrorString(ce); return false; }
                    if (dland.flushed < 0) dland.flushed = pos0;   // the first delivered row: a request that resumes reads from there on
                    dland.reached[(size_t) k] = pos0 + rows;
                    int64_t m = INT64_MAX;
                    for (int j = 0; j < n_stages; ++j) if (dland.nt[(size_t) j] > 0) m = std::min(m, dland.reached[(size_t) j]);
                    if (m > dland.flushed) {
                        if (!df2.context(dland.dev + (size_t) (dland.flushed - dland.first) * ntot * N, m - dland.flushed, dland.flushed, e)) return false;
                        dland.flushed = m;
                    }
                    return true;
                };
            }
        };
        // a request starts: the stages tap the last 2047 prompt rows (the solo path's window) and the landing starts over
        auto df2_stage_prompt_taps_begin = [&](int64_t first) {
            std::lock_guard<std::mutex> lk(dland.mu);
            dland.first = first;
            dland.flushed = -1;
            for (auto& r : dland.reached) r = -1;
            for (int k = 0; k < n_stages; ++k) df2_stage_prefill(k).set_taps(df2.tap_layers(), first, 2048);
        };
#endif
        if (use_df2) {   // DFlash2 in the serve loop: one card or a layer split (taps gathered onto the drafter's card), one request at a time
            std::string why_not;
            df2.set_max_drafts((o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec) - 1);   // the widest window - 1: the head and the walk run for that many rows
            df2_offer_head();
            // where the drafter's weights come from: the last stage's card by default (its head rows and embedding path), as the MTP's
            const bool df2_on_last = multi_gpu && last_st != nullptr && !df2_first;
            const bool df2_chain = n_stages > 1;   // the taps come from several verifiers (gather_taps lands them on the drafter's card)
            if (df2.device() != 0 && strata::core::native_embed() != nullptr && strata::core::native_embed()->in_vram()) why_not = "the embedding table sits in CUDA0's VRAM (STRATA_EMBED_VRAM), out of reach of the drafter's card";
            else if (!(df2_chain ? ver.tapping_chain() : ver.tapping())) why_not = "the verifier keeps no taps here";
            else if (!df2.bind(df2_on_last ? last_st->wt : wt, multi_gpu ? &last_st->head : &native_head, df2_vocab_path(), why_not, df2_first ? last_st->dev : -1) ||
                     !df2.prepare(df2_chain ? df2.landing() : ver.tap_dev(), why_not) ||
                     !(multi_gpu ? df2_stage_prompt_taps_reserve(why_not) : sp.reserve_taps(df2.tap_layers(), 2048, g.n_embd, why_not))) {
                if (why_not.empty()) why_not = "the drafter could not be set up";
            }
            if (!why_not.empty()) {
                std::fprintf(stderr, "strata serve: --draft dflash2: %s. It needs about 1.2 GiB of VRAM in all (the pack 0.76 GiB, buffers and K/V ring 0.25 GiB, "
                                     "a draft head 0.2 GiB, 0.1 GiB of prompt staging); --vram-reserve-mib 1900 keeps it. The MTP drafts instead.\n", why_not.c_str());
                if (df2_solo) {   // --draft-fallback none: there is no MTP to draft instead
                    std::fprintf(stderr, "strata serve: --draft-fallback none: no MTP is loaded, so the start ends here\n");
                    return 1;
                }
                use_df2 = false;
                if (o.draft_mtp_spec_set && o.spec > o.draft_mtp_spec) o.spec = o.draft_mtp_spec;   // the MTP drafts with its own window, not DFlash2's
            } else {
                if (multi_gpu) df2_stage_prompt_taps_install();
                else sp.on_taps = [&](const float* taps, int64_t rows, int64_t pos0, std::string& e) -> bool { df2_tap_dump(taps, rows, pos0, (int64_t) df2.tap_layers().size() * g.n_embd); return df2.context(taps, rows, pos0, e); };
                std::fprintf(stderr, "strata serve: DFlash2 drafter on (%s, graphs captured in %.0f ms); windows up to %d tokens; %s%s\n",
                             o.draft_model.c_str(), df2.ms_capture, o.spec, df2_solo ? "no MTP layer is loaded (--draft-fallback none)" : "the MTP layer stays as the fallback",
                             df2_chain ? (std::string("; layer split: the drafter is on CUDA") + std::to_string(df2.device()) + " (" + (df2_on_last ? "last" : "first") +
                                          " stage), the other stages' taps are gathered onto it").c_str() : "");
            }
        }
        // the taps of the window just run (rows 0..rows-1), where the drafter reads them: the verifier's own buffer on one card; on a layer split every
        // stage's share landed on the drafter's card (Verifier::gather_taps; the drafter's stream waits on the copies)
        auto df2_win_taps = [&](int rows, std::string& e) -> const float* {
            if (n_stages <= 1) return ver.tap_dev();
            if (!ver.gather_taps(df2.landing(), df2.device(), 0, rows, 0, df2.stream(), e)) return nullptr;
            return df2.landing();
        };
        if (batch_df2 && use_df2) {   // the slot drafters bind to the solo drafter's head and embedding and capture their graphs on their own landing
            for (size_t b = 0; b < slot_df2.size(); ++b) {
                auto& d = *slot_df2[b];
                if (!d.clone_bind(df2, err)) { std::fprintf(stderr, "strata serve: STRATA_BATCH_DF2: slot %zu: %s\n", b, err.c_str()); return 1; }
                d.set_max_drafts((o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec) - 1);
                if (!d.prepare(d.landing(), err)) { std::fprintf(stderr, "strata serve: STRATA_BATCH_DF2: slot %zu: %s\n", b, err.c_str()); return 1; }
            }
            std::fprintf(stderr, "strata serve: STRATA_BATCH_DF2: %zu slot drafters ready (batch windows carry each slot's token and its spare-row drafts)\n", slot_df2.size());
        } else if (batch_df2) {   // the solo drafter did not start (a fallback is loaded): the slots decode without drafts
            std::fprintf(stderr, "strata serve: STRATA_BATCH_DF2 is off: DFlash2 did not start\n");
            batch_df2 = false;
            batch_mtp = false;
        }
#endif
        const bool draft_on = use_mtp || use_df2;   // some drafter proposes (--draft-fallback none: DFlash2 alone, use_mtp false)
        if (pipe) mtp.set_source_R(ver.final_R_all());   // the serial decode's rows (the last stage's even verifier)
        auto free_slot_mtp_rows = [](float* p) { if (p != nullptr) (void) cudaFree(p); };
        std::vector<std::unique_ptr<float, decltype(free_slot_mtp_rows)>> slot_mtp_rows;
        if (batch_mtp && !batch_df2) {
            // like the solo drafter's `bind` above: the last stage's weights (embedding) and head, and the residual
            // buffers on the drafter's device (the last stage's GPU with a layer split)
            const strata::core::OnDevice on_slot_mtp(mtp.device());
            for (int b = 0; b < o.batch; ++b) {
                float* r = nullptr;
                const size_t bytes = (size_t) o.spec * (size_t) g.hc * (size_t) g.n_embd * sizeof(float);
                if (cudaMalloc((void**) &r, bytes) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: batch MTP slot %d residual buffer does not fit%s\n",
                                 b, vram_free_note().c_str());
                    return 1;
                }
                slot_mtp_rows.emplace_back(r, free_slot_mtp_rows);
                if (!slot_mtp[(size_t) b]->bind(last_st && !mtp_first ? last_st->wt : wt, last_st ? &last_st->head : &native_head, r, err,
                                                &mtp) ||
                    (piped_mtp && !slot_mtp[(size_t) b]->prepare_chain(err))) {   // STRATA_PIPED_MTP: the chain graphs
                    std::fprintf(stderr, "strata serve: batch MTP slot %d: %s%s\n", b, err.c_str(),
                                 vram_free_note().c_str());
                    return 1;
                }
            }
        }
        // each stage's verifier gets its stage's slot sessions
        if (o.batch > 0) {
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                std::vector<strata::core::SessionState*> ptrs;
                for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
                strata::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (!vk.init_slots(ptrs, err)) {
                    std::fprintf(stderr, "strata serve: --batch, stage %zu: %s\n", k + 1, err.c_str());
                    return 1;
                }
                // STRATA_PIPED_MTP: the rows a group's commit replays after the verdict, one hand-off row each
                if (piped_mtp && bslot_ss.size() > 1 &&
                    !vk.init_deferred_commit(std::max<int>(strata::kernels::kVerifyNarrowT, (1 + pm_drafts) * o.batch), err)) {
                    std::fprintf(stderr, "strata serve: STRATA_PIPED_MTP, stage %zu: %s\n", k + 1, err.c_str());
                    return 1;
                }
            }
        }
        for (int st = 0; st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (st > 0) {
                stage_ver(st).set_split(o.spec_split);
                stage_ver(st).set_pcie_mode(get_pcie_mode_code(o));
            }
        }
        if (pipe) {   // the odd windows' pool routing: the same as the even ones', with their plans
            split_drive_b = split_drive;
            for (int st = 0; st < n_stages; ++st) {
                split_drive_b.plan[st] = stage_ver_p(st, 1).plan_sink();
                stage_ver_p(st, 1).set_split(o.spec_split);
                stage_ver_p(st, 1).set_pcie_mode(get_pcie_mode_code(o));
            }
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const strata::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        const strata::core::PoolMultiFn invariant_win_pool_fn = n_stages > 1 ? &drive_pool_split_invariant : &drive_pool_multi_invariant;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        mem_mark("the verifier and the drafter's binding");
        ver.set_split(o.spec_split);
        ver.set_pcie_mode(get_pcie_mode_code(o));
        // --pipeline-windows 2: stage 0's GDN state before a speculative commit, restored when the guess was wrong.
        // Window W reads the state its snapshot must hold and no window writes it (only commits do), so it is copied
        // on a side stream while W runs, into the buffer of W's parity, and only W's speculative commit waits for it
        // (STRATA_PIPELINE_SNAP_OVERLAP=0: one buffer, copied on the critical path).
        // With three or more stages every front stage (all but the last) runs speculative windows, so each keeps its own
        // pair of snapshots, side stream and events.
        struct PlSnap {
            int dev = 0;
            size_t bytes = 0;
            void* state = nullptr;            // the stage's GDN state (its session's)
            float* buf[2] = {nullptr, nullptr};
            cudaStream_t side = nullptr;
            cudaEvent_t pre[2] = {nullptr, nullptr}, ev[2] = {nullptr, nullptr};
        };
        const int pl_nf = pipe ? n_stages - 1 : 0;   // the front stages
        std::vector<PlSnap> pl_snaps((size_t) pl_nf);
        bool pl_snap_ok = false, pl_snap_overlap = false;
        if (pipe && o.pipeline_windows >= 2) {
            pl_snap_ok = true;
            pl_snap_overlap = [] { const char* v = pipe_dbg_env("STRATA_PIPELINE_SNAP_OVERLAP"); return v == nullptr || std::atoi(v) != 0; }();
            for (int st = 0; st < pl_nf && pl_snap_ok; ++st) {
                PlSnap& sn = pl_snaps[(size_t) st];
                strata::core::SessionState& sst = st == 0 ? ss : stages[(size_t) st - 1]->ss;
                sn.dev = stage_ver_p(st, 0).device();
                sn.bytes = model.gdn_snapshot_bytes(sst);
                sn.state = (void*) sst.gdn_state;
                const strata::core::OnDevice on(sn.dev);
                if (cudaMalloc((void**) &sn.buf[0], sn.bytes) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows 2: the GDN snapshot of stage %d (%.1f MiB) does "
                                         "not fit; the decode stays serial\n", st + 1, (double) sn.bytes / 1048576.0);
                    cudaGetLastError();
                    sn.buf[0] = nullptr;
                    pl_snap_ok = false;
                    break;
                }
                if (pl_snap_overlap) {
                    bool ok_s = cudaMalloc((void**) &sn.buf[1], sn.bytes) == cudaSuccess &&
                                cudaStreamCreateWithFlags(&sn.side, cudaStreamNonBlocking) == cudaSuccess;
                    for (int i = 0; i < 2 && ok_s; ++i)
                        ok_s = cudaEventCreateWithFlags(&sn.pre[i], cudaEventDisableTiming) == cudaSuccess &&
                               cudaEventCreateWithFlags(&sn.ev[i], cudaEventDisableTiming) == cudaSuccess;
                    if (!ok_s) {
                        cudaGetLastError();
                        pl_snap_overlap = false;
                    }
                }
            }
        }
        if (pipe)
            std::fprintf(stderr, "strata serve: --pipeline-windows %d: two verifiers per stage, %d stages overlap across "
                                 "windows%s\n", o.pipeline_windows, n_stages,
                         !pl_snap_ok ? "" : pl_snap_overlap ? " (decode too; the GDN snapshots beside the window)"
                                                            : " (decode too; the GDN snapshot on the critical path)");
        // the verifiers by [stage][parity] - the first stage and the LAST (stage 1 of a two-stage split; the decode loop
        // reaches the middle stages through stage_ver_p) - their pool routing, and the drafter's events (in a pipelined
        // prompt read, a stage-1 window waits until the drafter has read the rows of the same parity's previous window)
        strata::core::Verifier* PV[2][2] = {{&ver, &ver_b}, {pipe ? &stages.back()->ver : nullptr, pipe ? &stages.back()->ver_b : nullptr}};
        SplitDrive* PSD[2] = {&split_drive, &split_drive_b};
        cudaEvent_t pl_mtp_ev[2] = {nullptr, nullptr};
        bool pl_mtp_live[2] = {false, false};
        bool pl_prepared = false;
        // every graph captured and the drafter's prompt path sized while nothing is in flight (a capture syncs its stream)
        auto pl_prepare = [&](std::string& e) -> bool {
            if (pl_prepared) return true;
            for (int st = 0; st < n_stages; ++st)
                for (int par = 0; par < 2; ++par) {
                    strata::core::Verifier& v = stage_ver_p(st, par);
                    const strata::core::OnDevice on(v.device());
                    if (!v.capture_all(e)) return false;
                }
            if (!mtp.prepare_prefill(e)) return false;
            if (o.pipeline_windows >= 2 && pl_snap_ok && !mtp.prepare_chain(e)) return false;
            const strata::core::OnDevice on(mtp.device());
            for (cudaEvent_t& ev : pl_mtp_ev)
                if (ev == nullptr && cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
                    e = "pipeline: event creation failed";
                    return false;
                }
            pl_prepared = true;
            return true;
        };
        // INVARIANT: nothing pipelined is in flight outside read_windows_pl and the pipelined decode loop - both end
        // here (a checkpoint, a park, the next request and the serial paths all find both stages and the drafter idle).
        // Every window in flight is served to its end (its picks discarded), then both stages and the drafter idle.
        // (--adapt-async: the decode loop then settles the asynchronous tier's round, a_pl_settle, see its end.)
        auto pl_drain = [&]() {
            std::string e2;
            for (int round = 0; round < 10000000; ++round) {
                bool any = false;
                for (int st = 0; st < n_stages && pipe; ++st)
                    for (int par = 0; par < 2; ++par) {
                        strata::core::Verifier& v = stage_ver_p(st, par);
                        if (!v.in_flight()) continue;
                        any = true;
                        if (v.service(&drive_pool_split, PSD[par], e2) < 0) return;
                        if (v.done(e2)) v.pl_finish(nullptr, e2);
                        else if (!e2.empty()) return;
                    }
                if (!any) break;
            }
            for (int st = 0; st < n_stages && pipe; ++st) {
                const strata::core::OnDevice on(stage_ver_p(st, 0).device());
                cudaStreamSynchronize(stage_ver_p(st, 0).stream());
            }
            const strata::core::OnDevice on(mtp.device());
            cudaStreamSynchronize(mtp.stream());
        };
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
        // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
        // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
        // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
        // chain's root is the deepest point every request so far shared (the end of the system prompt, in
        // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
        // shares that prefix mounts through it instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        // #879: a window met non-finite logits: the next request drops every cached state first and reads from token 0
        bool nan_flush = false;
        // #3465: the non-finite logits came in a window that reads the PROMPT's tail (the last few tokens, read through
        // the verify windows after the batched part).  The retry reads that tail batched (--short-read 0 for one request):
        // a reporter's prompts failed there every time and passed with --short-read 0.
        bool nf_prompt_windows = false, short_read_off_once = false;
        std::vector<ConvCheckpoint> checks;
        // KVA: which prefixes of token ids a KVA read approximated, so the draft layer knows, for whatever state a request
        // continues from, the first cell built from a real row (kva_floor.hpp).  Empty without KVA.
        strata::prefill::KvaFloorRegistry kva_floors;
        // STRATA_CKPT_POOL: the dropped checkpoints' storage, reused.  STRATA_SLOT_CKPT_MOVE turns it on too: a moved
        // checkpoint frees nothing at the next prune, so every save would fault in fresh pages (17 -> 40-60 ms)
        CheckpointPool ckpt_pool(ckpt_pool_env() || slot_ckpt_move_env(), ckpt_pool_cap_env());
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        int64_t tail_ckpt_len = -1;   // --prompt-cache-tail: the length of the one tail checkpoint alive (-1 = none)
        // --prefill-pipe: the plan of the prompt segment being read (read_part sets it around its runs), for the
        // checkpoints taken inside it
        ConvCheckpoint::PromptPlan seg_plan;
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
        // (the snapshots accept a null draft)
        // #3493: no --conversation-cache-mib given: size the cache for agent use from the RAM that is free now that the model is
        // loaded (an agent Harness alternates a long main conversation with shorter ones; with the cache off every request read
        // its whole prompt again, Reused 0 on 12 of 12).  An explicit value, 0 included, is never touched.
        if (o.conversation_cache_mib < 0) {
            if (o.serve && o.prompt_cache > 0 && o.conversation_cache_slots > 0) {
                auto cache_room = strata::core::available_host_bytes();
                // #4518: the residency policy locked the PLE table before it knew what the rest of the load would take; if the lock is what
                // leaves the automatic cache too little (or none), the lock is given back (the pages stay cached, the OS may reclaim them)
                // and the cache is sized again.  --ple-io ram asked for by hand and --conversation-cache-mib N are never touched.
                if (ple_lock_by_policy && ple_table.locked() &&
                    strata::core::ple_lock_blocks_cache(cache_room, ple_table.table_bytes(), o.conversation_cache_min_free_mib)) {
                    const auto before = cache_room;
                    if (ple_table.unlock()) {
                        cache_room = strata::core::available_host_bytes();
                        std::fprintf(stderr, "strata serve: PLE table lock given back: the lock left %.1f GiB of RAM for the conversation cache, "
                                             "%.1f GiB without it (the table stays cached; --conversation-cache-mib N and --ple-io ram are kept as asked)\n",
                                     before ? (double) *before / (1ull << 30) : 0.0, cache_room ? (double) *cache_room / (1ull << 30) : 0.0);
                    }
                }
                o.conversation_cache_mib = strata::core::auto_conversation_cache_mib(cache_room, o.conversation_cache_min_free_mib);
                if (o.conversation_cache_mib > 0)
                    std::fprintf(stderr, "strata serve: conversation cache sized from the free RAM: %lld MiB for up to %d parked "
                                         "conversations, so an agent's next turn reuses its prompt (--conversation-cache-mib N sets it, 0 turns it off)\n",
                                 (long long) o.conversation_cache_mib, o.conversation_cache_slots);
                else
                    std::fprintf(stderr, "strata serve: conversation cache off: not enough free RAM to park conversations "
                                         "(--conversation-cache-mib N asks for one anyway)\n");
            } else {
                o.conversation_cache_mib = 0;
            }
        }
        strata::core::ConversationCache conversations(
            o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
            (size_t) o.conversation_cache_slots, o.conversation_cache_min_tokens);
        // Disk sessions: what a session file is bound to.  The model fingerprint samples every model input this
        // engine loaded, by role (conversation_file.hpp), once; the config fingerprint covers the RESOLVED settings
        // that change what the saved bytes mean - the rope (K is cached post-RoPE), the loaded control vector, the
        // K/V format and the arithmetic switches.  Sampling, seeds, draft tuning and the expert tier are not in it.
        std::optional<uint64_t> model_fp, config_fp;
        auto session_identity = [&](strata::core::SessionFileIdentity& id, std::string& e,
                                    const std::function<void()>& next_file) -> bool {
            if (!model_fp) {
                uint64_t fp = 0;
                if (!strata::qwen38::get_session_model_fingerprint(model, srcp, fp, e, next_file))
                    return false;
                model_fp = fp;
            }
            if (!config_fp)
                config_fp = strata::qwen38::get_session_config_fingerprint(model);
            id.model = *model_fp;
            id.config = *config_fp;
            return true;
        };
        // What a park captures: the main session's conversation (park_current) or an idle batch slot's (park_slot:
        // the reply it decoded, before an admission overwrites it).  One budget / admission / snapshot path for both.
        struct ParkSource {
            const std::vector<int32_t>& ids;
            const std::vector<ImgKey>& imgs;
            std::vector<ConvCheckpoint>& checks;          ///< moved apart into stage parts for the capture, put back after
            bool cvec;
            strata::core::SessionState& session;          ///< stage 0's
            std::vector<strata::core::SessionState*> stage_sessions;   ///< the later stages', in stage order (stages[k]->dev)
            const strata::core::QsaState* draft0;         ///< the draft layer's K/V saved with stage 0's image (or none)
            const strata::core::QsaState* draft_last;     ///< ... with the last stage's image (a split: the drafter's GPU)
            bool with_reuse;                              ///< the retained K/V describes this source (the main session's)
            std::string what;                             ///< the log's words after the token count ("", " from slot 1")
        };
        // Save only on a switch/rewind, not on each continuing request. No graph
        // addresses change: all parked images live in ordinary host vectors.
        // STRATA_PARK_ASYNC: set by the request around its park - the live checkpoints it drops right after (they move
        // into the parked image instead of being copied); a background allocation then refills STRATA_CKPT_POOL
        std::function<bool(const ConvCheckpoint&)> park_drops;
        std::future<void> pool_refill;
        // STRATA_CKPT_POOL_READY=N (with STRATA_CKPT_POOL): the pool holds N checkpoints whose vectors already have the
        // session's sizes (the ones conversation_checkpoint_save resizes to, so a pooled one is never resized) with their
        // pages in: a fresh 118 MB vector costs ~40 ms of page faults at a save.  Topped up on a background thread, off the
        // host's core, whenever a take leaves it short; one top-up at a time.  Allocation only: a save overwrites every byte.
        // Not with a layer split (its checkpoints carry stage parts; saved as before).  Unset: no ready checkpoints, nothing
        // allocated ahead of time.
        auto pool_topup = [&] {
            if (ckpt_pool_ready_env() == 0 || !ckpt_pool.on() || !stages.empty()) return;
            const ConvStateSizes pool_z = conv_state_sizes(g, ss);
            if (pool_z.gdn == 0) return;
            if (pool_refill.valid() && pool_refill.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
            const size_t have = ckpt_pool.held();
            if (have >= ckpt_pool_ready_env()) return;
            pool_refill = std::async(std::launch::async, [&ckpt_pool, want = ckpt_pool_ready_env() - have, z = pool_z,
                                                          layers = (size_t) std::max<int64_t>(ss.qsa_alloc, 0),
                                                          ple = ss.ple_hist != nullptr] {
                StageThreads::spread_current_thread();   // (a thread the host starts inherits its one CPU: aux_cpus.hpp)
                for (size_t k = 0; k < want; ++k) {
                    ConvCheckpoint c;
                    c.gdn.resize(z.gdn); c.ple.resize(ple ? z.ple : 0); c.tails.resize(layers * z.tail);
                    c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
                    ckpt_pool.give(std::move(c));
                }
            });
        };
        pool_topup();   // the first request's checkpoints find the pool full
        auto park_source = [&](const ParkSource& src, size_t held) -> bool {
            // #2049: parking copies the outgoing conversation's running checkpoints and K/V into host memory, which
            // is a silent stretch of the "request" stage; when the OS has paged that memory out it takes minutes.
            // Named, so a stall report says "parking"; its copies beat (conversation_copy.hpp), so the watchdog
            // sees a park that moves and ends one that does not.
            struct ParkStage {
                ParkStage() { strata::core::progress_at("request: parking the previous conversation"); }
                ~ParkStage() { strata::core::progress_at("request"); strata::core::progress_beat(); }
            } park_stage;
            // --conversation-cache-min-tokens: a short conversation is not worth one of the slots.  Parking is a
            // switch-time decision, so a client's small side requests would otherwise park themselves and, once the
            // cache is full, evict the long conversation they were interleaved with (oldest first) - the reuse it
            // wanted.  Returning true is the same non-error skip the byte-budget and RAM-floor paths below take.
            if (!conversations.wants((int64_t) src.ids.size())) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (%zu tokens, below the %lld "
                                     "minimum; parked=%zu)\n",
                             src.ids.size(), (long long) o.conversation_cache_min_tokens, conversations.size());
                return true;
            }
            // #342: before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold
            // nothing the outgoing chain does not, apart from the tail this conversation rewrote)
            if (const size_t dropped = conversations.drop_superseded(src.ids, src.imgs, src.checks, src.cvec))
                std::fprintf(stderr, "strata serve: conversation cache: dropped %zu superseded cop%s of this "
                             "conversation; parked=%zu\n", dropped, dropped == 1 ? "y" : "ies", conversations.size());
            // A layer split parks one image per stage: the first stage's (with the draft layer's K/V) and one per
            // later stage (without it - the draft ring is saved once).  The checkpoints are MOVED apart into their
            // stage parts for the capture and put back together whichever way this ends (no running state copied).
            const size_t n_st = stages.size();
            // the draft layer's K/V lives on the last stage's GPU (the drafter is loaded there): with a split it is
            // saved with that stage's image, under its device; stage 0's image holds none
            const strata::core::QsaState* draft0 = src.draft0;   // no --mtp: no draft K/V (snapshots accept null)
            auto draft_of = [&](size_t k) -> const strata::core::QsaState* { return k + 1 == n_st ? src.draft_last : nullptr; };
            std::vector<ConvCheckpoint>& checks = src.checks;
            strata::core::ConversationCheckpointSplit cs;
            if (n_st > 0) cs = strata::core::conversation_checkpoints_split(std::move(checks), n_st);   // #752: two-phase; a bad_alloc leaves `checks` whole
            struct MergeBack {
                strata::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
                ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
            } merge_back{cs, checks, n_st > 0};
            const strata::core::ConversationView view{src.ids, src.imgs, n_st > 0 ? cs.stage0 : checks, src.cvec};
            const int64_t retained_n = src.with_reuse ? conversations.retained_tokens() : 0;
            // a batch slot's image carries no retained K/V (it describes the main session's conversation), and the
            // main session's stays retained for its own park
            auto reuse = src.with_reuse ? conversations.take_reuse_for(src.ids)   // never another conversation's K/V
                                        : strata::core::ConversationKvReuse{};
            if (retained_n > 0 && reuse.unchanged_tokens < retained_n)
                std::fprintf(stderr, "strata serve: conversation cache: retained K/V reused for %lld of its %lld tokens "
                                     "(the rest is not this conversation's)\n", (long long) reuse.unchanged_tokens,
                             (long long) retained_n);
            std::vector<strata::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
            stage_reuse.resize(n_st);
            reuse.stages.clear();
            // the later stages' estimates, each with its own retained K/V when it still fits
            size_t stage_estimate = 0, stage_retained = 0;
            for (size_t k = 0; k < n_st; ++k) {
                auto& st = stages[k];
                const strata::core::OnDevice on(st->dev);
                const strata::core::ConversationView view_k{src.ids, src.imgs, cs.parts[k], src.cvec};
                size_t b = 0;
                if (!strata::core::conversation_snapshot_bytes(view_k, *src.stage_sessions[k], g, draft_of(k), b, err)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (stage CUDA%d: %s)\n",
                                 st->dev, err.c_str());
                    err.clear();
                    return true;
                }
                if (!stage_reuse[k].kv.empty()) {
                    size_t r = 0;
                    if (strata::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, *src.stage_sessions[k], g,
                                                                          draft_of(k), r, err)) {
                        b = r;
                        stage_retained += stage_reuse[k].bytes();
                    } else {
                        stage_reuse[k] = {};
                        err.clear();
                    }
                }
                stage_estimate += b;
            }
            size_t estimate = 0;
            if (!strata::core::conversation_snapshot_bytes(view, src.session, g, draft0, estimate, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (%s)\n", err.c_str());
                err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
                return true;
            }
            const size_t fresh_estimate = estimate;
            if (!reuse.kv.empty() && !strata::core::conversation_snapshot_capture_bytes(
                    reuse, view, src.session, g, draft0, estimate, err)) {
                reuse = {};
                estimate = fresh_estimate;
                err.clear();
            }
            estimate += stage_estimate;
            // A park carrying its own retained K/V replaces memory the cache
            // already held, so capacity is make_room's call - it runs next either
            // way, and put()'s accounting still bounds the budget. The with-reuse
            // estimate must stay uncapped: it counts the retained buffers'
            // capacity and directories, and put() charges that same true size -
            // a capped figure would under-evict and overfill the budget.
            if (!conversations.make_room(estimate, held)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (snapshot %zu MiB exceeds available budget)\n",
                             estimate >> 20);
                return true;
            }
            const auto t0 = Clock::now();
            try {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t retained = reuse.bytes() + stage_retained;
                const size_t additional = estimate > retained ? estimate - retained : 0;
                // The physical-RAM gate: the incoming snapshot has to fit beside the floor.  make_room()
                // above only balanced the cache's own budget, so a full cache leaves this one short even
                // though every parked conversation could give its RAM back - and refusing here throws
                // away the whole prompt read that produced this snapshot (measured: 270k tokens, 96 s).
                // Evict the least recently active parked conversations until it fits, or until none are
                // left.  Each ConversationBuffer is a list of 16 MiB segments, each segment its own
                // allocation, so an evicted entry is back with the kernel before the next check reads
                // /proc/meminfo; no waiting is needed.  slots() bounds the loop, and the two lines
                // below say what it did either way.  The reading is the guard's (MemAvailable, the cgroup
                // limits above the engine, --memory-limit-mib); `mem` keeps the last one for the refusal line.
                size_t evicted = 0;
                std::optional<strata::core::HostMemoryReading> mem;
                auto admit = [&] {
                    mem = strata::core::sample_host_memory();
                    return strata::core::conversation_memory_admit(
                        mem ? std::optional<uint64_t>(mem->available) : std::nullopt, additional, floor);
                };
                while (!admit() && conversations.size() > 0 && evicted < conversations.slots() &&
                       conversations.evict_oldest())
                    ++evicted;
                if (!admit()) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, %s; evicted %zu, %zu still parked)\n",
                                 additional >> 20, (long long) o.conversation_cache_min_free_mib, ram_note(mem).c_str(),
                                 evicted, conversations.size());
                    return true;
                }
                if (evicted)
                    std::fprintf(stderr, "strata serve: conversation cache: evicted %zu parked conversation%s to admit this snapshot (%zu MiB plus %lld MiB floor)\n",
                                 evicted, evicted == 1 ? "" : "s", additional >> 20,
                                 (long long) o.conversation_cache_min_free_mib);
                strata::core::SavedConversation image;
                size_t reused_bytes = 0;
                // STRATA_CKPT_POOL: the running state and the checkpoint copies go into dropped checkpoints' vectors
                // (the ones this request drops refill the pool right after); what is left goes back
                std::vector<ConvCheckpoint> storage;
                if (ckpt_pool.on() && n_st == 0)
                    for (size_t k = 0; k <= view.checkpoints.size(); ++k) {
                        ConvCheckpoint c = ckpt_pool.take();
                        if (c.gdn.empty()) break;
                        storage.push_back(std::move(c));
                    }
                std::vector<ConvCheckpoint> moved;   // STRATA_PARK_ASYNC: the ones this request drops (view: the rest)
                if (park_drops && n_st == 0 && src.with_reuse) {   // (the main session's checkpoints only, not an idle slot's)
                    size_t w = 0;
                    for (size_t i = 0; i < checks.size(); ++i) {
                        if (park_drops(static_cast<const ConvCheckpoint&>(checks[i]))) { moved.push_back(std::move(checks[i])); continue; }
                        if (w != i) checks[w] = std::move(checks[i]);
                        ++w;
                    }
                    checks.erase(checks.begin() + (std::ptrdiff_t) w, checks.end());
                }
                const bool saved = strata::core::conversation_snapshot_save(image, view, src.session, g, draft0, err,
                        std::move(reuse), &reused_bytes, storage.empty() ? nullptr : &storage,
                        moved.empty() ? nullptr : &moved);
                for (ConvCheckpoint& c : storage) ckpt_pool.give(std::move(c));
                if (!saved) return false;
                pool_topup();   // what the park took (and what moved out) is allocated again, in the background
                for (size_t k = 0; k < n_st; ++k) {   // the later stages (the last one with the draft layer's K/V)
                    auto& st = stages[k];
                    const strata::core::OnDevice on(st->dev);
                    if (cudaDeviceSynchronize() != cudaSuccess) { err = "stage sync failed"; return false; }
                    const strata::core::ConversationView view_k{src.ids, src.imgs, cs.parts[k], src.cvec};
                    strata::core::SavedConversation part;
                    if (!strata::core::conversation_snapshot_save(part, view_k, *src.stage_sessions[k], g, draft_of(k), err,
                            std::move(stage_reuse[k]), &reused_bytes))
                        return false;
                    image.stage_images.push_back(std::move(part));
                }
                const auto mem_after = strata::core::sample_host_memory();
                if (!strata::core::conversation_memory_admit(
                        mem_after ? std::optional<uint64_t>(mem_after->available) : std::nullopt, 0, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (RAM floor after capture: %lld MiB needed, %s)\n",
                                 (long long) o.conversation_cache_min_free_mib, ram_note(mem_after).c_str());
                    return true;
                }
                const size_t snapshot_bytes = image.bytes();
                const bool stored = conversations.put(std::move(image), held);
                std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens%s in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu\n",
                             stored ? "parked" : "skipped", src.ids.size(), src.what.c_str(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes, reused_bytes);
            } catch (const std::bad_alloc&) {
                // The active state has not been touched. Continue with normal
                // prompt processing rather than killing a serving process.
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
            }
            return true;
        };
        auto park_current_body = [&](size_t held) -> bool {
            if (!conversations.enabled() || !live_ok || live.empty()) return true;
            const size_t n_st = stages.size();
            ParkSource src{live, live_imgs, checks, cvec_cached, ss, {},
                           (n_st > 0 && !mtp_first) || !use_mtp ? nullptr : &mtp.kv_state(),
                           use_mtp && !mtp_first && n_st > 0 ? &mtp.kv_state() : nullptr, true, ""};
            for (auto& st : stages) src.stage_sessions.push_back(&st->ss);
            return park_source(src, held);
        };
        // #752: the whole park (the checkpoint split included) is under one try, so a host OOM skips parking instead of
        // ending the server
        auto park_current = [&](size_t held) -> bool {
            try { return park_current_body(held); }
            catch (const std::bad_alloc&) {
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
                return true;
            }
        };
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        bool pp_tail_saved = false;
        // #471: the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled
        // mid-read reports as read, instead of the whole prompt
        int64_t pp_reached = 0;
        // #1620: the session holds exactly the tokens [0, pp_reached) of the prompt (no chunk or window is in flight, none
        // failed): a request cancelled here keeps them as the live conversation instead of dropping what it restored
        bool pp_exact = false;
        Clock::time_point pp_t0 = Clock::now();
        // #879: this request met a window whose head logits were not all finite (nonfinite_hit): it ends with
        // DONE ... nonfinite, emitting none of that window's tokens
        bool nf_hit = false;
        auto nonfinite_hit = [&](const strata::core::Verifier& v, const char* what) -> bool {
            const int row = v.nonfinite_row();
            if (row < 0) return false;
            std::fprintf(stderr, "strata serve: non-finite logits (#879) in a %s window at position %lld (row %d): its "
                                 "tokens are not emitted, the request ends (DONE nonfinite) and the next request reads "
                                 "its prompt from token 0\n", what, (long long) v.nonfinite_pos(), row);
            std::fflush(stderr);
            nf_hit = true;
            nan_flush = true;
            if (std::strcmp(what, "prompt") == 0) nf_prompt_windows = true;
            return true;
        };
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        // #613: why the last checkpoint_at failed - a failed device sync is the GPU itself (a hang Windows then resets),
        // not the checkpoint, and the error says so
        std::string ckpt_why;
        // STRATA_TRACE: what each checkpoint cost - the device syncs, the copies into RAM (the vectors' first fill
        // included), the eviction that frees one - one line per checkpoint; with the prompt's read parts summed per
        // request for a batch admission's line (below, at BADM)
        const bool ckpt_trace = std::getenv("STRATA_TRACE") != nullptr;
        struct ReqTimes {
            int ckpts = 0;
            double sync_ms = 0.0, save_ms = 0.0, evict_ms = 0.0, read_ms = 0.0, windows_ms = 0.0;
        } req_times;
        auto gpu_sync_ok = [&]() -> bool {
            const cudaError_t e = cudaDeviceSynchronize();
            if (e == cudaSuccess) return true;
            ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                       ") - a GPU hang; on Windows the driver is then reset";
            return false;
        };
        StageThreads ckpt_threads;   // STRATA_CKPT_THREADS: the later stages' parts of a checkpoint (started on first use)
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool as_tail = false) -> bool {
            ckpt_why.clear();
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) {
                    c.used = ++check_clock;
                    if (!as_tail && L == tail_ckpt_len) tail_ckpt_len = -1;   // a periodic one lands here: a normal item now
                    return true;
                }
            ConvCheckpoint c = parts == nullptr ? ckpt_pool.take() : ConvCheckpoint{};   // (STRATA_CKPT_POOL)
            if (parts == nullptr) pool_topup();   // the next save finds one ready
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                for (size_t i = 0; i < c.running_state().size(); ++i)
                    *c.running_state()[i] = std::move(*(*parts)[0].running_state()[i]);
                c.dev_tag = (*parts)[0].dev_tag;
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else if (ckpt_pool.on() || ckpt_threads_env()) {
                // STRATA_CKPT_POOL / STRATA_CKPT_THREADS: the parts go into the vectors a pooled checkpoint already has
                // (none resized), with the threads all at once; part k is stage k's session, synchronized first
                const size_t n = stages.size() + 1;
                const bool reused = !c.gdn.empty();   // (a pooled one)
                c.stage_parts.resize(stages.size());
                for (ConvCheckpoint& part : c.stage_parts) part.ids = c.ids;
                std::vector<char> synced(n, 0), saved(n, 0);
                std::vector<std::string> why(n);
                std::vector<double> sync_part(n, 0.0), part_ms(n, 0.0);
                std::vector<std::exception_ptr> thrown(n);   // (a failed allocation: thrown here, as without the threads)
                const std::function<void(size_t)> save_part = [&](size_t k) {
                    try {
                        const Clock::time_point p0 = Clock::now();
                        const strata::core::OnDevice on(k == 0 ? -1 : stages[k - 1]->dev);
                        const cudaError_t e = cudaDeviceSynchronize();
                        if (e != cudaSuccess) {
                            why[k] = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
                                     ") - a GPU hang; on Windows the driver is then reset";
                            return;
                        }
                        synced[k] = 1;
                        const Clock::time_point p1 = Clock::now();
                        saved[k] = checkpoint_save(k == 0 ? c : c.stage_parts[k - 1], k == 0 ? ss : stages[k - 1]->ss, g);
                        sync_part[k] = std::chrono::duration<double, std::milli>(p1 - p0).count();
                        part_ms[k] = std::chrono::duration<double, std::milli>(Clock::now() - p0).count();
                    } catch (...) {
                        thrown[k] = std::current_exception();
                    }
                };
                const Clock::time_point t0 = Clock::now();
                if (ckpt_threads_env()) {
                    ckpt_threads.run(n, save_part);
                } else {
                    for (size_t k = 0; k < n; ++k) {
                        save_part(k);
                        if (!synced[k] || !saved[k]) break;
                    }
                }
                for (size_t k = 0; k < n; ++k)
                    if (thrown[k]) std::rethrow_exception(thrown[k]);
                for (size_t k = 0; k < n; ++k) {
                    if (!synced[k]) { ckpt_why = why[k]; return false; }
                    if (!saved[k]) return false;
                }
                if (ckpt_trace) {   // sync: the first stage's (the others' are in their parts' times)
                    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    std::string each;
                    for (size_t k = 0; k < n; ++k) {
                        char b[32];
                        std::snprintf(b, sizeof b, " %.1f", part_ms[k]);
                        each += b;
                    }
                    ++req_times.ckpts;
                    req_times.sync_ms += sync_part[0];
                    req_times.save_ms += wall - sync_part[0];
                    std::fprintf(stderr, "strata trace: checkpoint at %lld: sync %.1f ms, save %.1f ms (%zu stage(s)%s, "
                                         "each%s ms, %s)\n", (long long) L, sync_part[0], wall - sync_part[0], n,
                                 ckpt_threads_env() ? " at once" : "", each.c_str(),
                                 reused ? "into a dropped checkpoint's vectors" : "new vectors");
                }
            } else {
                Clock::time_point t0 = Clock::now();
                auto lap = [&](double& acc) {   // (STRATA_TRACE only)
                    if (!ckpt_trace) return;
                    const Clock::time_point t1 = Clock::now();
                    acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    t0 = t1;
                };
                double sync_ms = 0.0, save_ms = 0.0;
                if (!gpu_sync_ok()) return false;
                lap(sync_ms);
                if (!checkpoint_save(c, ss, g)) return false;
                lap(save_ms);
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const strata::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    part.ids = c.ids;
                    if (!gpu_sync_ok()) return false;
                    lap(sync_ms);
                    if (!checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                    lap(save_ms);
                }
                if (ckpt_trace) {
                    ++req_times.ckpts;
                    req_times.sync_ms += sync_ms;
                    req_times.save_ms += save_ms;
                    std::fprintf(stderr, "strata trace: checkpoint at %lld: sync %.1f ms, save %.1f ms (%zu stage(s))\n",
                                 (long long) L, sync_ms, save_ms, stages.size() + 1);
                }
            }
            c.used = ++check_clock;
            if (seg_plan.chunk > 0 && L > seg_plan.from && L < seg_plan.to) c.plan = seg_plan;   // inside a planned segment
            if (as_tail) {   // only one tail checkpoint stays alive: the previous request's goes
                const int64_t old_tail = tail_ckpt_len;
                auto old = [&](const ConvCheckpoint& k) { return old_tail >= 0 && (int64_t) k.ids.size() == old_tail; };
                if (ckpt_pool.on())
                    ckpt_pool.erase_if(checks, old);
                else
                    checks.erase(std::remove_if(checks.begin(), checks.end(), old), checks.end());
                tail_ckpt_len = L;
            } else if (L == tail_ckpt_len) {
                tail_ckpt_len = -1;   // the flagged tail was dropped and a normal checkpoint takes its length
            }
            checks.push_back(std::move(c));
            const Clock::time_point te0 = Clock::now();
            const size_t before_evict = checks.size();
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                std::vector<char> is_tail;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) {
                    stamps.push_back(k.used);
                    is_tail.push_back(tail_ckpt_len >= 0 && (int64_t) k.ids.size() == tail_ckpt_len);
                }
                // the tail checkpoint serves only a branch of the last request: it leaves before a periodic one
                std::unique_ptr<bool[]> tail_flags(new bool[is_tail.size()]);
                for (size_t i = 0; i < is_tail.size(); ++i) tail_flags[i] = is_tail[i] != 0;
                // the pin=N prefix is never the victim (flags only when one is pinned: nothing changes without)
                std::unique_ptr<bool[]> pin_flags;
                for (const ConvCheckpoint& k : checks)
                    if (k.pinned) {
                        pin_flags.reset(new bool[checks.size()]);
                        for (size_t i = 0; i < checks.size(); ++i) pin_flags[i] = checks[i].pinned;
                        break;
                    }
                std::vector<int64_t> lens;
                for (const ConvCheckpoint& k : checks) lens.push_back((int64_t) k.ids.size());
                const size_t victim = o.prompt_cache_spread
                    ? strata::program::conv_cache::spread_victim(lens.data(), stamps.data(), stamps.size(),
                                                                 o.prompt_cache, tail_flags.get(), pin_flags.get())
                    : strata::program::conv_cache::eviction_victim(stamps.data(), stamps.size(), o.prompt_cache,
                                                                   tail_flags.get(), pin_flags.get());
                ckpt_pool.give(std::move(checks[victim]));   // (STRATA_CKPT_POOL; else nothing)
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            if (ckpt_trace && checks.size() < before_evict) {
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - te0).count();
                req_times.evict_ms += ms;
                std::fprintf(stderr, "strata trace: checkpoint at %lld: %zu evicted in %.1f ms\n", (long long) L,
                             before_evict - checks.size(), ms);
            }
            return true;
        };
        // STRATA_SPLIT_MTP_BATCH=1 (default off): the layer split's drafter is on the last stage, so E-9 (Prefill::draft_kv)
        // runs there, on the Prefill that ran the chunk, instead of the drafter's own per-8-row pass.  The drafts' K/V
        // come from Q8_1 x Q8_0 MMQ rather than mmvq: drafts may differ, the target's tokens never.
        static const bool split_mtp_batch = strata::program::draft_kv::split_env_on(std::getenv("STRATA_SPLIT_MTP_BATCH"));
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            // KVA: a chunk the late layers did not compute has no real final rows: its draft cells are neither built
            // nor read (the floor rises before the cells are built, so they are skipped)
            if (use_mtp && mtp.cell_floor_enabled() && sp.kva_floor() > mtp.cell_floor()) mtp.set_cell_floor(sp.kva_floor());
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            // E-9: batched through the prompt path when it can (a layer split: only with STRATA_SPLIT_MTP_BATCH=1)
            bool batched = false;
            const strata::program::draft_kv::Plan dkv = strata::program::draft_kv::plan(
                use_mtp, multi_gpu, split_mtp_batch && !mtp_first, strata::core::native_embed() != nullptr);
            if (dkv.try_batched || dkv.why != nullptr) {
                strata::prefill::Prefill& dsp = multi_gpu ? stages.back()->sp : sp;   // the drafter's stage
                const char* why = dkv.why;
                if (dkv.try_batched) batched = dsp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e, &why);
                if (multi_gpu) {   // once per process: what this env did
                    static bool said = false;
                    if (!said && (batched || (e.empty() && why != nullptr))) {
                        said = true;
                        if (batched)
                            std::fprintf(stderr, "strata serve: draft layer prompt K/V batched on CUDA%d (E-9 on the layer "
                                                 "split, K/V mode %d)\n", stages.back()->dev, mtp.kv_state().kv_mode);
                        else
                            std::fprintf(stderr, "strata serve: STRATA_SPLIT_MTP_BATCH=1 declined, the drafter's own pass "
                                                 "runs: %s\n", why);
                    }
                }
            }
            if (!e.empty() || (use_mtp && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
#if defined(STRATA_HAVE_DFLASH2)
            if (use_df2) {
                std::vector<int32_t> tk((size_t) T);
                for (int64_t t = 0; t < T; ++t) tk[(size_t) t] = (int32_t) cur[(size_t) (p0 + t)];
                df2.note_rows(tk.data(), T, p0);
            }
#endif
            if (use_mtp && std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr)
                std::fprintf(stderr, "strata serve: DRAFT_PREFILL path=%s mode=%d cells=%lld\n",
                             batched ? "batched" : "token", mtp.kv_state().kv_mode, (long long) T);
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            pp_reached = done;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            print_response("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
            strata::core::progress_at("reading the prompt (batched), done up to token", done);
            strata::core::progress_beat();
            std::fflush(stdout);
            // layer-major prefill: a sub-chunk before its super-chunk's end has the super-chunk's state - the
            // checkpoint is taken at that end instead
            const bool periodic_checkpoint = o.prompt_cache_every > 0 && done >= pp_next_check && !sp.layer_major_mid();
            // Keep the prefill's existing chunk geometry. One extra near-tail checkpoint
            // can serve a branch whose shared prefix ends before the final cached state.
            const bool tail_checkpoint = o.prompt_cache_tail && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
                !multi_gpu && !pp_tail_saved && pp_total - done > std::max<int64_t>(1, o.short_read) &&
                pp_total - done <= T &&   // this chunk's real size (--prefill auto chunks differ from o.prefill_chunk)
                !sp.layer_major_mid();
            if (periodic_checkpoint || tail_checkpoint) {
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done, nullptr, tail_checkpoint && !periodic_checkpoint);
                }
                if (!saved) {
                    // #3350: only a GPU that is unwell ends the request; a copy that ran out of memory skips this checkpoint
                    if (!ckpt_why.empty() || !checkpoint_failure_is_survivable()) {
                        e = "saving a conversation checkpoint failed" + ckpt_why;
                        return false;
                    }
                    std::fprintf(stderr, "strata serve: checkpoint at %lld skipped: its state could not be copied out (see "
                                         "above); the request goes on without it\n", (long long) done);
                }
                if (periodic_checkpoint) pp_next_check = done + o.prompt_cache_every;
                if (tail_checkpoint) pp_tail_saved = true;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                strata::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                strata::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, stage_ss, g)) {
                        // #3350: the copy ran out of memory (not the GPU): leave this part out. The set at `done` is then
                        // incomplete and the checkpoint is skipped below, as for a part that was never made
                        if (!checkpoint_failure_is_survivable()) { e = "saving a checkpoint part failed"; return false; }
                        std::fprintf(stderr, "strata serve: checkpoint at %lld skipped: stage %zu's state could not be copied "
                                             "out (see above); the request goes on without it\n", (long long) done, i);
                        std::lock_guard<std::mutex> lk(part_mu);
                        part_at[done].resize(stages.size() + 1);   // (marks the set as seen; its empty parts make it incomplete)
                        return true;
                    }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        drive.d.plan = ver.plan_sink();
        // Foresight swap space (STRATA_FS_SLOTS; unset = null and nothing changes)
        std::unique_ptr<strata::core::ForesightSwap> fs_swap;
        {
            int main_dev = 0;
            cudaGetDevice(&main_dev);
            std::vector<int> devs{main_dev};
            for (auto& st : stages) devs.push_back(st->dev);
            std::vector<int> card((size_t) g.n_layers, 0);
            for (int64_t l = 0; l < g.n_layers; ++l) card[(size_t) l] = multi_gpu ? model.stage_of(l) : 0;
            fs_swap.reset(strata::core::foresight_swap_from_env(g.n_layers, g.n_expert, card, devs));
            drive.d.fs = fs_swap.get();
            if (fs_swap && drive.d.lookahead != nullptr) lookahead.set_foresight(fs_swap.get());
            src.set_foresight_swap(fs_swap.get());   // #2134: the exchanges hold the experts whose bytes they move
        }
        // the look-ahead thread and the exchanges must not touch the swap space once it is gone (declared after it:
        // destroyed first)
        struct FsDetach {
            strata::core::RouterLookahead* la;
            strata::core::FileExpertSource* src;
            ~FsDetach() { la->set_foresight(nullptr); src->set_foresight_swap(nullptr); }
        } fs_detach{&lookahead, &src};
        // 0.2.10 (BW-MEASURE E26): PcieAuto starts from the pool size instead of the link-only 0.55 (0.15 on gfx12), so the first answers after a launch
        // do not run ~50 windows at a share the machine does not want.  The pool's expert-byte rate R is assumed to be min(65, 4.0 GB/s per worker)
        // (measured: 31-35 GB/s at 6-8 workers, 47-52 at 12, 55-60 at 16-30, 73-75 at 32) and the share is the refinement's own formula, link / (link +
        // 1.85 R) rounded to 0.05 and clamped 0.10..0.60, so a weak CPU starts HIGH (4 workers 0.50, 2 workers 0.60).  The converged share is unchanged.
        // STRATA_PCIE_START=old keeps the link-only start.  First answer on an RTX PRO 5000 mimic (32 workers): +4..+12%; 8 workers and the R9700: +-1%.
        if (o.pcie_frac_auto && o.pcie_probe_gbps > 0.0) {
            const char* ps = std::getenv("STRATA_PCIE_START");
#if defined(STRATA_USE_HIP)
            // HIP: the new start is opt-in (STRATA_PCIE_START=new).  On the R9700 it measured +-1% by the note above, and it flips near-tie tokens in the first
            // ~50 windows after a start (INT3/G2, p620, commit 06ebf2e4d: v0.2.9 vs the merged build with the other new defaults off 21/45 identical answers,
            // 45/45 with STRATA_PCIE_START=old), so AMD keeps v0.2.9's start share.  CUDA keeps the new start by default (STRATA_PCIE_START=old restores).
            const bool new_start = ps != nullptr && std::strcmp(ps, "new") == 0;
#else
            const bool new_start = ps == nullptr || std::strcmp(ps, "old") != 0;
#endif
            if (new_start) {
                const double r = std::min(65.0, 4.0 * (double) pool.workers());
                const double f0 = std::min(pcie_share_max(), std::max(0.10, std::round(o.pcie_probe_gbps / (o.pcie_probe_gbps + pcie_pool_k(r) * r) * 20.0) / 20.0));
                std::fprintf(stderr, "strata serve: PCIe share: start %.2f from %d pool workers (assumed %.0f GB/s of expert bytes) and the %.1f GB/s link (was %.2f)\n",
                             f0, pool.workers(), r, o.pcie_probe_gbps, o.pcie_frac);
                o.pcie_frac = f0;
            }
        }
        drive.d.pcie_num = get_pcie_num(o.pcie_frac);
        const bool all_experts_resident = !host_res.empty() &&
            std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
        // STRATA_ROUTE_TAIL_SKIP: ON BY DEFAULT on CUDA builds (owner decision, 0.1.42) and on HIP gfx1200/gfx1201 (0.1.43): a missed expert that every token of
        // a verify window routes at rank 7 or lower is neither copied nor computed (answers differ slightly from 0.1.41,
        // KL ~0.003-0.006; docs/DETAILS.md).  =0 restores 0.1.41's bytes.  Nothing to do, so no default and no line, when every
        // expert is in VRAM; off with --batch slots (a window then mixes requests, and one request's skip would depend on
        // another's tokens) and, as in the dispatch, with a peer or helper GPU.  A layer split is fine: each stage skips by itself.
        {
            const char* tse = std::getenv("STRATA_ROUTE_TAIL_SKIP");
            // #1884: with nearly every expert in the GPU cache (V100 32 GB, Coder IQ1_M: 94.8% of the experts, 99.9% decode hits) the few
            // misses are cheap and the skip costs more draft acceptance (70.1% -> 68.2%) than it saves: -1..2% decode.  No default then.
            size_t cached_experts = 0;
            for (int32_t r : host_res) cached_experts += r >= 0 ? 1 : 0;
            // (STRATA_TAIL_SKIP_MAX_CACHED=<percent>, default 90, moves the line; 101 = never).  0.1.43: 60 on a HIP layer split, where
            // 2x R9700 (5 pairs, skip on vs off): Q2_0 story -6.2% at 70% cached, -1.8% at 60%; IQ3_S +6..15% at 44-52%, about 0 at 60%.
#if defined(STRATA_USE_HIP)
            const double max_cached_default = n_stages > 1 ? 0.60 : 0.90;
#else
            const double max_cached_default = 0.90;
#endif
            const char* mcs = std::getenv("STRATA_TAIL_SKIP_MAX_CACHED");
            const double max_cached = mcs != nullptr && mcs[0] != 0 ? std::atof(mcs) / 100.0 : max_cached_default;
            const bool nearly_all_cached = !host_res.empty() && !all_experts_resident &&
                                           (double) cached_experts >= max_cached * (double) host_res.size();
            // NVFP4 experts (any layer): no default (sergqwer/strata-nvfp4 c83625c3).  On the fork's qwen-nvfp4-gptq pack the
            // skip moved the answers by teacher-forced KL 0.035 against it off - as much as the GPTQ quantization's own error
            // against an all-Q8_0 reference (0.038) - for +11.5% decode (RTX 5090).  STRATA_ROUTE_TAIL_SKIP=7 opts in.
            bool nvfp4_experts = false;
            if (native_pack)
                for (const auto& f : strata::kernels::cpu::expert_layout().fmt)
                    nvfp4_experts = nvfp4_experts || f.gu_type == strata::kernels::cpu::kNvfp4Type ||
                                    f.d_type == strata::kernels::cpu::kNvfp4Type;
            const bool tail_default_path = tse == nullptr && !all_experts_resident && !nearly_all_cached && o.batch == 0 &&
                                           decode_defaults_arch_ok() &&
                                           o.peer_device < 0;
            const bool tail_default_ok = tail_default_path && !nvfp4_experts;
            if (tail_default_path && nvfp4_experts)
                std::fprintf(stderr, "strata serve: STRATA_ROUTE_TAIL_SKIP is off by default with NVFP4 experts (it moved an NVFP4 "
                                     "pack's answers by about as much as its quantization does); STRATA_ROUTE_TAIL_SKIP=7 turns it on\n");
            if (tse == nullptr && nearly_all_cached && o.batch == 0 && o.peer_device < 0 && decode_defaults_arch_ok())
                std::fprintf(stderr, "strata serve: STRATA_ROUTE_TAIL_SKIP is off by default: %.1f%% of the experts are in the GPU cache, so "
                                     "few are missed and skipping them costs more draft acceptance than it saves; "
                                     "STRATA_ROUTE_TAIL_SKIP=7 turns it on\n", 100.0 * (double) cached_experts / (double) host_res.size());
            if (tail_default_ok) strata::core::set_tail_skip_rank(7);
            if (strata::core::tail_skip_rank() > 0 && !all_experts_resident)
                std::fprintf(stderr, "strata serve: STRATA_ROUTE_TAIL_SKIP=%d is ON%s: a missed expert that every token of a verify window routes at rank %d or "
                                     "lower is skipped (+10..20%% decode on 12-16 GB cards that miss experts, answers differ slightly from 0.1.41); "
                                     "STRATA_ROUTE_TAIL_SKIP=0 turns it off\n", strata::core::tail_skip_rank(),
                             tse == nullptr ? " (default)" : "", strata::core::tail_skip_rank());
        }
        if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
            drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        PcieAuto pcie_auto;
        if (o.pcie_frac_auto && !all_experts_resident && o.batch == 0 && o.peer_device < 0 && (n_stages == 1 || pcie_auto_split_ok())) {
            const auto& pl = strata::kernels::cpu::expert_layout();
            double sum = 0.0;
            for (int64_t l = 0; l < g.n_layers; ++l) sum += (double) pl.blob_bytes(l);
            pcie_auto.on = true;
            pcie_auto.link_gbps = o.pcie_probe_gbps;
            pcie_auto.blob_bytes = g.n_layers > 0 ? sum / (double) g.n_layers : 0.0;
        }
        // PcieAuto after a decode window (the serial loop, and 0.1.44 the pipelined one at every verdict): the share, refined from
        // the CPU pool's speed.  A layer split: every later stage takes the share its own probed link gives against the same pool
        // rate, on both parities' routing (the pipelined loop's second verifier).
        auto pcie_auto_step = [&](double& req_pcie_frac) {
            if (!pcie_auto.on) return;
            double tc_us = 0.0, rate = 0.0;
            const double nf = req_pcie_frac == o.pcie_frac
                                  ? pcie_auto.window(drive.d.ms_run, drive.d.multi_misses, o.pcie_frac, &tc_us, &rate)
                                  : (pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses), -1.0);
            if (nf < 0.0) return;
            std::fprintf(stderr, "strata serve: PCIe share: the CPU pool takes %.0f us per missed expert (%.1f GB/s of expert bytes), "
                                 "the link %.1f GB/s -> pcie_frac %.2f (was %.2f; --pcie-frac N fixes it, "
                                 "STRATA_PCIE_FRAC_DEFAULT=old keeps the start-up rule)\n",
                         tc_us, rate, pcie_auto.link_gbps, nf, o.pcie_frac);
            o.pcie_frac = nf;
            req_pcie_frac = nf;
            drive.d.pcie_num = get_pcie_num(nf);
            split_drive.pcie_num[0] = drive.d.pcie_num;
            split_drive_b.pcie_num[0] = drive.d.pcie_num;
            for (int st = 1; st < n_stages; ++st) {
                GpuStage& gs = *stages[(size_t) st - 1];
                if (!(gs.link_gbps > 0.0)) continue;
                gs.pcie_frac = std::min(pcie_share_max(), std::max(0.10, std::round(gs.link_gbps / (gs.link_gbps +
                                   pcie_pool_k(rate) * rate) * 20.0) / 20.0));
                split_drive.pcie_num[st] = get_pcie_num(gs.pcie_frac);
                split_drive_b.pcie_num[st] = split_drive.pcie_num[st];
                std::fprintf(stderr, "strata serve: PCIe share: stage %d (link %.1f GB/s) -> pcie_frac %.2f\n", st + 1,
                             gs.link_gbps, gs.pcie_frac);
            }
        };
        // the serial windows of one GPU, where the pool can admit (expert_pool_dispatch_multi's fetch_on)
        const char* fetch_off = d_res == nullptr ? "no residency table" : o.spec_split ? "--spec-split"
                              : multi_gpu || !stages.empty() ? "a layer split" : peer.valid() ? "--peer-device"
                              : remote_opt || drive.d.remote_count > 0 ? "helper GPUs" : o.adapt_async ? "--adapt-async"
                              : o.pipeline_windows > 0 ? "--pipeline-windows" : o.batch > 0 ? "--batch"
                              : adapt_overlap() ? "STRATA_ADAPT_OVERLAP=1" : nullptr;
        if (adapt_fetch() > 0) {   // STRATA_ADAPT_FETCH (opt-in)
            const char* off = drive.d.usage.empty() ? "the adaptive tier is off" : fetch_off != nullptr ? fetch_off
                            : o.pcie_mode == "dma" || o.pcie_mode == "direct" ? "--pcie-mode dma / direct"
                            : drive.d.plan == nullptr ? "no GPU plan"
                            : drive.d.pcie_num <= 0 ? "no PCIe share"
                            : src.complement_ready() ? "the resident RAM mode" : nullptr;
            if (off != nullptr) std::fprintf(stderr, "strata serve: STRATA_ADAPT_FETCH is off (%s)\n", off);
            else {
                drive.d.fetch_admit = adapt_fetch();
                drive.d.res_mut = host_res.data();
                drive.d.in_flight.assign(host_res.size(), 0);
                drive.d.fetch_avail.assign((size_t) g.n_layers, 0);
                if (const char* v = std::getenv("STRATA_ADAPT_FETCH_MIN")) drive.d.fetch_min = (float) std::atof(v);
                if (const char* v = std::getenv("STRATA_ADAPT_FETCH_MARGIN")) drive.d.fetch_margin = (float) std::atof(v);
                std::fprintf(stderr, "strata serve: STRATA_ADAPT_FETCH=%d: the PCIe share takes each layer's most-routed "
                                     "misses and admits those routed >= %.2f and %.2f more than a resident one%s\n",
                             drive.d.fetch_admit, drive.d.fetch_min, drive.d.fetch_margin,
                             drive.d.fetch_admit == 2 ? "; the tier's own copies are off" : "");
            }
        }
        // #477 --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays
        // empty and nothing below runs).  It needs the adaptive tier's counts and the residency table.
        std::vector<double> heat;
        if (!o.expert_profile_save.empty() && !all_experts_resident) {
            if (drive.d.usage.empty() || host_res.empty())
                std::fprintf(stderr, "strata serve: --expert-profile-save needs the adaptive tier (--adapt-every and "
                                     "--adapt-swaps above 0) and --expert-profile: nothing will be saved\n");
            else
                heat.assign(drive.d.usage.size(), 0.0);
        }
        Clock::time_point profile_saved_at = Clock::now();
        cudaStream_t adapt_stream = nullptr;
        if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        int pending_age = 0;   // the windows the pending swaps have waited (adapt_lag)
        std::vector<void*> pin_live;   // the swaps' locked arena pages (pin_blob), unlocked once they have landed
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        // 0.1.43 STRATA_ADAPT_OVERLAP=1 (adapt_overlap above): the single-GPU, plain-RAM-arena decode loop only
        strata::core::AdaptGate agate;
        bool aflipped = false;   // the pending swaps are in the tables and their copies may still be landing
        bool aov = false;
        int64_t aov_flips = 0;
        // #2010: the overlap is opt-in on every backend now, so the off state is the default and has to be visible too.
        // The predicates are side-effect free, so they are read in both states: they name why it could not run.
        // A peer tier (--peer-device) joins the overlap: measured on one rig so far (RTX 5060 Ti x8 + RTX 3060, IQ3_XXS),
        // where the blocking wait took 3-8 ms of a 27-43 ms window.
        const bool aov_want = adapt_overlap();
        const char* aov_why = multi_gpu || !stages.empty() ? "not on a layer split"
                              : remote_opt ? "not with --remote-expert-opt"
                              : o.adapt_async ? "not with --adapt-async"
                              : o.pipeline_windows > 0 ? "not with --pipeline-windows"
                              : pin_blobs_on() ? "not with page-locked swap sources"
                              : strata::core::dma_batch_mode() != 0 ? "not with STRATA_DMA_BATCH"
                              : adapt_lag() != 1 || adapt_nowait() ? "not with STRATA_ADAPT_LAG / STRATA_ADAPT_NOWAIT"
                              // the resident RAM mode is exact too (soft_commit_exchanges) but not a win everywhere: RTX A4000 +7.0%
                              // (5/5 pairs), RTX 3060 on a 46 GB box -2.6% (0/5; the CPU experts and the copies share the RAM bandwidth)
                              : src.complement_ready() && !(std::getenv("STRATA_ADAPT_OVERLAP_RESIDENT") &&
                                                            std::getenv("STRATA_ADAPT_OVERLAP_RESIDENT")[0] == '1')
                                    ? "not in the resident RAM mode unless STRATA_ADAPT_OVERLAP_RESIDENT=1"
                              : drive.d.usage.empty() ? "the adaptive tier is off"
                              : nullptr;
        if (aov_want) {
            if (aov_why == nullptr) {
                agate.ticks.resize((size_t) g.n_layers + 1);
                for (int64_t l = 0; l <= g.n_layers; ++l) agate.ticks[(size_t) l] = {&agate.landed, (int32_t) l};
                agate.has.assign((size_t) g.n_layers, 0);
                agate.peer_has.assign((size_t) g.n_layers, 0);
                if (!ver.set_adapt_gate(&agate)) aov_why = "the device-planned layers (STRATA_VERIFY_DEVICE_PLAN) are on";
                // #2188: on Windows the default is the host function.  Under WDDM a stream's work sits in the driver's
                // command buffer until something flushes it, and nothing does while the host only spins on the pinned
                // int: the first request stopped with "copies for layer 45 never landed".  STRATA_ADAPT_SIGNAL=copy asks
                // for the stream-ordered copy there too (the enqueue then flushes the stream), =host everywhere.
                const char* asig = std::getenv("STRATA_ADAPT_SIGNAL");
#if defined(_WIN32)
                const bool sig_copy = asig != nullptr && asig[0] == 'c';
#else
                const bool sig_copy = !(asig != nullptr && asig[0] == 'h');
#endif
                if (aov_why == nullptr && !sig_copy)
                    std::fprintf(stderr, "strata serve: adaptive overlap signal: host functions%s\n",
                                 asig == nullptr ? " (the Windows default, #2188; STRATA_ADAPT_SIGNAL=copy: a stream-ordered copy)" : "");
                if (aov_why == nullptr && sig_copy) {
                    // the landing of each layer's copies is told by a stream-ordered D2H copy of the layer number into pinned memory
                    // (STRATA_ADAPT_SIGNAL=host: a host function instead). No callback thread has to be scheduled while the pool's workers spin on every other CPU
                    int32_t* vals = nullptr; void* hp = nullptr;
                    std::vector<int32_t> iota((size_t) g.n_layers + 1);
                    for (size_t i = 0; i < iota.size(); ++i) iota[i] = (int32_t) i;
                    if (cudaMalloc((void**) &vals, iota.size() * 4) == cudaSuccess &&
                        cudaMemcpy(vals, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                        cudaHostAlloc(&hp, 64, cudaHostAllocPortable) == cudaSuccess) {
                        *(volatile int32_t*) hp = -1;
                        agate.hland = (volatile int32_t*) hp;
                        agate.dvals = vals;
                        std::fprintf(stderr, "strata serve: adaptive overlap signal: stream-ordered copies (STRATA_ADAPT_SIGNAL=host: a host function)\n");
                    } else { (void) cudaGetLastError(); }
                }
                // the peer tier's copies land over its own link and are gated the same way, with the same kind of signal
                if (aov_why == nullptr && peer.valid())
                    agate.peer_hland = peer.set_overlap_gate(&agate.peer_landed, &agate.peer_has, &agate.peer_last, sig_copy);
            }
            aov = aov_why == nullptr;
            if (aov)
                std::fprintf(stderr, "strata serve: adaptive overlap (STRATA_ADAPT_OVERLAP=0 turns it off): the adaptive tier's copies land while the next window runs "
                                     "(same tokens, a layer waits for its own copies)%s\n",
                             !peer.valid() ? "" : agate.peer_hland != nullptr ? "; the peer tier's copies too, over its own link (stream-ordered copies)"
                                                                              : "; the peer tier's copies too, over its own link (host functions)");
            else if (std::getenv("STRATA_ADAPT_OVERLAP") != nullptr)   // the env var was set: say what stopped it
                std::fprintf(stderr, "strata serve: STRATA_ADAPT_OVERLAP=1 is off: %s\n", aov_why);
        } else
            std::fprintf(stderr, "strata serve: adaptive overlap: off%s%s\n",
                         aov_why ? ": " : " (STRATA_ADAPT_OVERLAP=1 turns it on)", aov_why ? aov_why : "");
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        auto res_upload = [&]() -> cudaError_t {   // the first error, every table tried
            cudaError_t r = d_res != nullptr ? res_put(d_res) : cudaSuccess;
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                const cudaError_t e = res_put(st->d_res);
                if (r == cudaSuccess) r = e;
            }
            return r;
        };
        auto apply_pending = [&](bool wait) {
            if (peer.valid()) peer.apply_pending(wait);
            if (pending.empty()) {
                // a round where only the peer swapped: its copies have landed (or, without the wait, are still in
                // flight), and the gate closes with them
                if (!(peer.valid() && peer.has_pending())) {
                    agate.active.store(false, std::memory_order_release);
                    aflipped = false;
                }
                return;
            }
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) cudaEventSynchronize(st->adapt_ev);
                    else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            unpin_blobs(pin_live);
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
                if (!drive.d.in_flight.empty()) drive.d.in_flight[(size_t) i] = 0;
            }
            if (drive.d.evict_sync || drive.d.fetch_admit > 0) {   // a failed upload leaves them to the pre-launch hook
                for (const auto& pr : pending) drive.d.res_dirty_idx.push_back(pr.first);
                drive.d.res_dirty = true;
            }
            pending.clear();
            pending_age = 0;
            agate.active.store(false, std::memory_order_release);   // 0.1.43: the copies have landed (STRATA_ADAPT_OVERLAP)
            aflipped = false;
            if (res_upload() == cudaSuccess) {   // the whole table: the hook has nothing left
                drive.d.res_dirty = false;
                drive.d.res_dirty_idx.clear();
            }
        };
        // Where the tier does not wait for its copies (STRATA_ADAPT_NOWAIT=1, STRATA_ADAPT_LAG > 1), its evictions reach
        // the device's tables before the next window: adapt() marks them non-resident at once, apply_pending uploads
        // only when the round's copies land - a window or more later, and until then doorbell_publish_res skips the
        // activation of a layer whose routed experts all look resident, while the pool computes an evicted one from the
        // activation published before.  The hook runs at each verifier's launch; the serial windows (one at a time,
        // the batch windows included, a layer split's stages one after another) take it.  STRATA_ADAPT_FETCH's
        // admissions (the pool writes host_res during a window) go up the same way, whether the tier waits or not.
        //   --batch-groups with a layer split launches a group while another is in flight on the other stage, which
        //   waits on this thread: a synchronous upload there could wait on that window, so those verifiers publish
        //   every activation instead (set_always_publish; the device table then decides nothing a stale entry breaks).
        //   --pipeline-windows is exempt: its verifiers always publish, and the tier fences the windows in flight
        //   before it overwrites a slot (adapt_fence).  --adapt-async uploads at each of its steps.
        // STRATA_ADAPT_CHECKRES=1 reads the tables back at each launch and compares them with the host's.
        const bool piped_groups = o.batch > 0 && o.batch_groups > 1 && !stages.empty();
        const bool serial_windows = !pipe && !piped_groups;
        const bool tier_lags = !drive.d.usage.empty() && (adapt_nowait() || adapt_lag() > 1);
        drive.d.evict_sync = serial_windows && tier_lags && (adapt_evict_sync() || drive.d.fetch_admit > 0);
        if (piped_groups && tier_lags && adapt_evict_sync()) {
            ver.set_always_publish(true);
            for (auto& st : stages) st->ver.set_always_publish(true);
        }
        const bool res_checking = std::getenv("STRATA_ADAPT_CHECKRES") != nullptr && serial_windows;
        std::vector<int32_t> res_back;
        int64_t res_windows = 0, res_bad = 0, res_bad_windows = 0;
        if (res_checking) {
            res_back.assign(host_res.size(), 0);
            drive.d.dres_check = res_back.data();
        }
        // The changed entries go up by a kernel on each card's verifier stream, reading them from mapped memory: a
        // synchronous upload waits for the copy engine, which the tier's swaps of the round before still hold (~4 ms
        // on 33 swaps; measured +0.9 ms a chat round).  Too many changes, or no mapped memory: the whole table.
        constexpr size_t kResPatchCap = 16384;
        std::vector<int> res_devs{-1};
        for (auto& st : stages) res_devs.push_back(st->dev);
        int32_t* res_pairs = (serial_windows && (drive.d.fetch_admit > 0 || drive.d.evict_sync))
                                 ? res_pairs_alloc(kResPatchCap, res_devs) : nullptr;
        auto res_sync = [&](bool check, std::string& e) -> bool {
            if (drive.d.res_dirty) {
                const int64_t n = res_patch_pairs(drive.d.res_dirty_idx, host_res, res_pairs, kResPatchCap);
                cudaError_t r = cudaSuccess;
                if (n >= 0) {
                    if (d_res != nullptr) {
                        strata::kernels::res_patch(d_res, res_pairs, (int) n, ver.stream());
                        r = cudaGetLastError();
                    }
                    for (size_t k = 0; k < stages.size() && r == cudaSuccess; ++k) {
                        const strata::core::OnDevice on(stages[k]->dev);
                        strata::kernels::res_patch(stages[k]->d_res, res_pairs, (int) n, stages[k]->ver.stream());
                        r = cudaGetLastError();
                    }
                } else {
                    r = res_upload();
                }
                if (r != cudaSuccess) {
                    e = std::string("the residency table update failed: ") + cudaGetErrorString(r);
                    return false;
                }
                drive.d.res_dirty = false;
                drive.d.res_dirty_idx.clear();
            }
            if (!check) return true;
            cudaStreamSynchronize(ver.stream());   // the patches landed (the verifier streams do not block the read)
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                cudaStreamSynchronize(st->ver.stream());
            }
            int64_t bad = 0;
            for (auto& st : stages) {   // the later stages' copies first: res_back ends with CUDA0's (the pool's check)
                const strata::core::OnDevice on(st->dev);
                bad += res_check(st->d_res, host_res, res_back, g.n_expert, res_windows);
            }
            bad += res_check(d_res, host_res, res_back, g.n_expert, res_windows);
            ++res_windows;
            res_bad += bad;
            res_bad_windows += bad > 0;
            if ((res_windows & 255) == 0 || (bad > 0 && res_bad_windows <= 4))
                std::fprintf(stderr, "strata serve: res check: %lld windows, %lld with mismatches (%lld entries), "
                                     "%lld layer-windows with CPU experts and no activation published\n",
                             (long long) res_windows, (long long) res_bad_windows, (long long) res_bad,
                             (long long) drive.d.stale_x);
            return true;
        };
        struct PreLaunchOff {   // the hooks' captures die before the verifiers
            strata::core::Verifier& v;
            std::vector<std::unique_ptr<GpuStage>>& stages;
            strata::core::ExpertDispatch& d;
            int32_t* pairs;
            ~PreLaunchOff() {
                v.set_pre_launch(nullptr);
                for (auto& st : stages) st->ver.set_pre_launch(nullptr);
                d.dres_check = nullptr;
                if (pairs != nullptr) cudaFreeHost(pairs);
            }
        } pre_launch_off{ver, stages, drive.d, res_pairs};
        if (serial_windows && (drive.d.fetch_admit > 0 || drive.d.evict_sync || res_checking)) {
            ver.set_pre_launch([&](cudaStream_t, std::string& e) { return res_sync(res_checking, e); });
            for (auto& st : stages) st->ver.set_pre_launch([&](cudaStream_t, std::string& e) { return res_sync(false, e); });
            if (drive.d.evict_sync && adapt_evict_sync())
                std::fprintf(stderr, "strata serve: the adaptive tier's evictions reach the device's table before the next "
                                     "window (STRATA_ADAPT_EVICT_SYNC=0: when its copies land)\n");
        }
        // --pipeline-windows: set while the tier adapts beside windows in flight (see its use in adapt)
        std::function<void()> adapt_fence;
        int64_t adapt_nswaps = 0;   // the blocking tier's swaps chosen, cumulative (the decode timing line's swaps/window)
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        auto adapt = [&]() -> bool {
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
            std::vector<Swap> swaps = choose_swaps(drive.d.usage, host_res, o.adapt_min_gain, true,
                                                   [&](int64_t l, int32_t e) {
                return (peer.valid() && peer.has(l, e)) || (remote_opt && remote_opt->owns(l, e)) || helper_holds(l, e);
            });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            if (aov) {   // 0.1.43: the copies go in layer order and each layer's landing is told to the verify window
                std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.layer < b.layer; });
                agate.landed.store(-1, std::memory_order_release);
                if (agate.hland != nullptr) *agate.hland = -1;
                agate.has.assign((size_t) g.n_layers, 0);
                agate.last = -1;
                for (const Swap& s : swaps) { agate.has[(size_t) s.layer] = 1; agate.last = std::max(agate.last, s.layer); }
                // the peer's marks of the previous round go too (its adapt below sets this round's, if it swaps anything)
                agate.peer_landed.store(-1, std::memory_order_release);
                agate.peer_has.assign((size_t) g.n_layers, 0);
                agate.peer_last = -1;
            }
            // the copies back read the cache of the card that owns each layer (see resident_stage_swaps)
            auto home_of = [&](int64_t layer) -> SwapHome {
                const int stn = multi_gpu ? model.stage_of(layer) : 0;
                if (stn > 0) {
                    GpuStage* gs = stages[(size_t) stn - 1].get();
                    return {&gs->cache, gs->adapt_stream, gs->dev};
                }
                return {&xcache, adapt_stream, -1};
            };
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, home_of)) return false;
            adapt_nswaps += (int64_t) swaps.size();   // (read after the thread's join)
            if (pin_blobs_on()) {   // lock the batch's source pages (a file-backed arena on AMD; see pin_blobs)
                std::vector<std::pair<uintptr_t, uintptr_t>> spans;
                for (const Swap& s : swaps)
                    if (const uint8_t* b = srcp->blob_stable(s.layer, s.in))
                        spans.emplace_back((uintptr_t) b,
                                           (uintptr_t) b + strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                pin_blobs(std::move(spans), pin_live);
            }
            // --pipeline-windows: windows are in flight while the tier adapts.  The evicted experts stop being planned
            // for the GPU first (the pool computes them on the CPU from the rows every layer publishes, see
            // Verifier::set_always_publish); then every copy into their slots waits until all the work the stages'
            // streams hold now has completed: any window that may still read those slots was launched before this.
            std::vector<int32_t> fenced_slots;
            if (adapt_fence && !swaps.empty()) {
                for (const Swap& s : swaps) {
                    const size_t out = (size_t) s.layer * g.n_expert + s.out;
                    fenced_slots.push_back(host_res[out]);
                    host_res[out] = strata::core::kNotResident;
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                adapt_fence();
            }
            bool main_live = false;
            // STRATA_DMA_BATCH (F12): the main device's copies as one cudaMemcpyBatchAsync (mode 0: one each)
            const int dma_mode = multi_gpu ? 0 : strata::core::dma_batch_mode();
            std::vector<void*> cp_dst;
            std::vector<const void*> cp_src;
            std::vector<size_t> cp_bytes;
            for (size_t si = 0; si < swaps.size(); ++si) {
                const Swap& s = swaps[si];
                if (aov && (si == 0 || swaps[si - 1].layer != s.layer)) {
                    // test hook (STRATA_ADAPT_OVERLAP_SLOW_US=N): a slow link, to make the windows wait for the copies
                    static const int slow_us = [] { const char* e = std::getenv("STRATA_ADAPT_OVERLAP_SLOW_US"); return e ? std::atoi(e) : 0; }();
                    static int slow_arg = slow_us;
                    if (slow_us > 0)
                        cudaLaunchHostFunc(adapt_stream, [](void* p) { std::this_thread::sleep_for(std::chrono::microseconds(*(int*) p)); }, &slow_arg);
                }
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = fenced_slots.empty() ? host_res[out] : fenced_slots[si];
                const uint8_t* b = srcp->blob_stable(s.layer, s.in);
                nib::guard_raw(s.layer, "the adaptive tier's copy in");
                const int stn = multi_gpu ? model.stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const strata::core::OnDevice on(gs ? gs->dev : -1);
                if (dma_mode != 0 && slot >= 0 && b != nullptr) {
                    cp_dst.push_back(xcache.device_slot(slot));
                    cp_src.push_back(b);
                    cp_bytes.push_back((size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                    main_live = true;
                    host_res[out] = strata::core::kNotResident;
                    srcp->prefetch(s.layer, s.out);
                    pending.emplace_back((int32_t) in, slot);
                    if (!drive.d.in_flight.empty()) drive.d.in_flight[in] = 1;   // STRATA_ADAPT_FETCH leaves it alone
                    continue;
                }
                if (slot < 0 || b == nullptr ||
                    adapt_copy_h2d(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                                   (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                   gs ? gs->adapt_stream : adapt_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: adaptive swap copy failed (layer %d, slot %d, pinned %d): %s\n",
                                 (int) s.layer, (int) slot, pin_live.empty() ? 0 : 1, cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
                if (aov && (si + 1 == swaps.size() || swaps[si + 1].layer != s.layer) &&
                    (agate.hland != nullptr
                         ? cudaMemcpyAsync((void*) agate.hland, agate.dvals + s.layer, 4, cudaMemcpyDeviceToHost, adapt_stream)
                         : cudaLaunchHostFunc(adapt_stream, [](void* p) {
                        auto* t = (strata::core::AdaptGate::Tick*) p;
                        int32_t cur = t->counter->load(std::memory_order_relaxed);
                        while (cur < t->layer && !t->counter->compare_exchange_weak(cur, t->layer, std::memory_order_release)) {}
                    }, &agate.ticks[(size_t) s.layer])) != cudaSuccess) {
                    (void) cudaGetLastError();
                    cudaStreamSynchronize(adapt_stream);   // no callback: wait here, the gate opens at once
                    agate.landed.store((int32_t) g.n_layers, std::memory_order_release);
                }
                if (!drive.d.in_flight.empty()) drive.d.in_flight[in] = 1;   // STRATA_ADAPT_FETCH leaves it alone
            }
            if (drive.d.evict_sync && !swaps.empty()) {   // the evicted ones are misses now (see the pre-launch hook)
                for (const Swap& s : swaps) drive.d.res_dirty_idx.push_back((int32_t) (s.layer * g.n_expert + s.out));
                drive.d.res_dirty = true;
            }
            if (!cp_dst.empty() && strata::core::copy_blobs(cp_dst.data(), cp_src.data(), cp_bytes.data(), cp_dst.size(),
                                                            adapt_stream, dma_mode) != cudaSuccess) {
                std::fprintf(stderr, "strata serve: adaptive swap copy failed (batched): %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            // #2188: hand the copies and their landing signals to the GPU now.  WDDM batches a stream's work until a
            // driver call flushes it, and the verify window only spins on the pinned int (Linux submits at once: no-op)
            if (aov && agate.hland != nullptr && !swaps.empty()) (void) cudaStreamQuery(adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const strata::core::OnDevice on(st->dev);
                    cudaEventRecord(st->adapt_ev, st->adapt_stream);
                }
            // #477: the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its
            // decays: the sum is proportional to the routing itself) - only with --expert-profile-save, else `heat`
            // is empty
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            if (peer.valid()) {                   // multi-GPU: the peer takes the next most-routed CPU misses
                std::vector<int32_t> r0 = host_res;
                for (const auto& [i, slot] : pending) r0[(size_t) i] = slot;   // swapped into the primary already
                std::string perr;
                if (!peer.adapt(drive.d.usage.data(), r0.data(),
                                o.peer_adapt_swaps >= 0 ? o.peer_adapt_swaps : o.adapt_swaps, perr)) {
                    std::fprintf(stderr, "strata serve: %s\n", perr.c_str());
                    return false;
                }
            }
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            return true;
        };
        // ---- --adapt-async 1: the asynchronous adaptive tier (opt-in; the resident RAM mode, see the checks where
        // the RAM copy is built).  adapt() runs a whole round - choose, copy the evicted slots back with a stream sync,
        // copy the new experts in from the RAM copy - on a thread the decode loop joins after the draft: ~24 ms every
        // 4th window with ~90 swaps.  Here a round moves through four steps, each started by `adapt_tick` between
        // windows once the previous step's copies (events) and host work (the job thread) are done:
        //   1. job: choose (adapt()'s rule, over snapshots of the routing counts and the residency) and copy the
        //      evicted slots back into exchange buffers (D2H), each from the cache of the card that owns its layer;
        //   2. stage the exchanges and mark the evicted experts non-resident (the CPU reads them from the exchange
        //      buffers from the next window on); job: copy the new experts in (straight from a page-locked RAM copy,
        //      else through a pinned bounce buffer), on the owning card's refill stream;
        //   3. mark the new experts resident; job: move the evicted blobs into the RAM places the new ones left;
        //   4. flip the copy's offsets.
        // A slot is overwritten only after its old expert stopped being planned for the GPU (step 2, between windows,
        // with the device copies of the residency table uploaded as apply_pending does), and a RAM place only once its
        // expert is resident (step 3), so no window computes from memory that is changing.  A request starts with the
        // round drained (adapt_tick(true)), before its prompt lends any slot.  Not bit-exact run to run: which window
        // first computes a swapped-in expert on the GPU (which rounds differently from the CPU) depends on when its
        // copy lands.  `j`: the swap's exchange buffer (its copy back) and bounce buffer (a_cap + j).
        // --pipeline-windows 2: the pipelined decode loop has windows in flight on both cards at every tick (window K+1
        // on the first while the last verifies or commits K), each one waiting on this thread at every layer (its
        // doorbell) and planning every layer from the residency table as it is when that layer is served.  There the
        // tier keeps two rules (a_pl_idle, set by that loop alone; unset, nothing is in flight at a tick):
        //   - no CUDA call of the tier reaches a card while a window there waits on this thread.  A copy queued to a
        //     card whose window spins on a host flag can block in the driver until that window ends (issue #31, where
        //     the host waited forever), and a thread blocked in the driver can hold up this one, which the window
        //     waits for.  So the job thread does host work alone (the choice, the bounce copies, the RAM moves) and
        //     this thread queues each card's copies at that card's next gap - a moment its stage has no window in
        //     flight (a_gap: before a window is launched there, and at every tick) - from page-locked memory only;
        //   - a gap of a stage after a step is also the step's fence for that card: every window planned there before
        //     the step has completed.  So step 2's copies into the evicted slots (queued at a gap after step 2)
        //     overwrite no slot a window may still read, and step 3's RAM moves wait (AState::RamFence) for a gap after
        //     step 3 on every card whose layers exchanged (a window planned before may still pull an incoming expert
        //     over PCIe from the RAM place its evicted one takes).  The CPU's reads need no fence: the pool runs inside
        //     the loop's service calls.
        // The device copies of the residency table are not uploaded meanwhile (a synchronous copy; the pipelined
        // verifiers publish every row whatever they say): once, when the loop has drained.  The loop ticks once per
        // verdict (after the drafter's chain is launched) and, while a round is in flight, every 0.5 ms without
        // counting a window.
        // The arena mode (srcp == &arena_src, a_arena below): every expert is in the arena already (a pageable or
        // partly pinned copy that never moves), so a round has no exchange and no RAM move - choose, mark the evicted
        // experts non-resident, copy the new ones in, mark them resident (steps 1-3 reduced to the copy in; RamFence and
        // step 4 pass through).  By default (STRATA_ADAPT_PICK_INLINE) the choice runs on this thread in the tick that
        // starts the round, and the copies in are posted in the same tick; the next tick lands them
        // (STRATA_ADAPT_ASYNC_WAIT: it waits for the copies rather than polling, so the window at which an expert first
        // runs on the GPU does not depend on timing) and a new round may start at once.  The job thread's copies
        // read the arena (blob_at: no counter) and block it on pageable memory, never this thread;
        // STRATA_ADAPT_ASYNC_BATCH caps the copies in flight per tick (the copy engine is shared with the windows).
        // Invariant (the serial loop, and the batch loop between windows): at a tick nothing in flight reads a slot -
        // a window's run ends with cudaStreamSynchronize (verify.cpp), the draft is synchronous and the commit does not
        // read the expert cache - so a slot whose expert was marked non-resident here may be overwritten right away;
        // and the job thread never touches host_res or drive.d.usage (it works on the snapshots a_usage / a_res).
        // Anything that gives a slot to something else (kvg_ensure's quiesce, a prompt's lend) calls adapt_quiesce first.
        struct ASwap { int32_t layer, in, out, slot; int home; int64_t j; bool exchange; const uint8_t* from; };
        struct AHome { strata::core::ExpertCache* cache; cudaStream_t stream; cudaEvent_t ev; int dev; bool used; };
        enum class AState { Idle, Pick, CopyBack, SwapIn, RamFence, Commit };
        AState astate = AState::Idle;
        std::vector<ASwap> aswaps;
        std::vector<float> a_usage;
        std::vector<int32_t> a_res;
        int64_t a_win = 0, a_last = 0, a_rounds = 0, a_swapped = 0;
        double a_ms = 0;                  // wall time from a round's start to its flip, summed (the "ms/round" stat)
        Clock::time_point a_t0;
        std::atomic<bool> a_err{false};
        double a_pick_ms = 0;             // the arena's inline choice, summed (the stats line)
        strata::program::AdaptiveCopyRange a_rng;   // the batch being copied in (a_copy_in), and where the next one starts
        std::vector<AHome> ahomes;   // [0] = CUDA0's cache, [k] = layer split stage k's
        std::unique_ptr<JobThread> ajob;
        // the arena mode (srcp == &arena_src): every expert is in the arena already, so a round is a copy in only - no
        // exchange buffers, no RAM moves (see a_arena below)
        const bool a_arena = srcp == &arena_src;
        // as the blocking tier's exchange buffers; the arena has none, so its rounds are as large as the blocking tier's
        const int64_t a_cap = a_arena ? o.adapt_swaps : std::min<int64_t>(o.adapt_swaps, 96);
        const auto a_env = [](const char* name, int def) {
            const char* v = std::getenv(name);
            return v != nullptr && *v != 0 ? std::atoi(v) : def;
        };
        // arena only (the other modes keep their steps as they were): the choice on this thread (STRATA_ADAPT_PICK_INLINE),
        // the wait for the copies (STRATA_ADAPT_ASYNC_WAIT), the copies in flight per tick (STRATA_ADAPT_ASYNC_BATCH, 0 = all)
        const bool a_inline = a_arena && a_env("STRATA_ADAPT_PICK_INLINE", 1) != 0;
        const bool a_wait = a_arena && a_env("STRATA_ADAPT_ASYNC_WAIT", 1) != 0;
        const size_t a_batch = a_arena ? (size_t) std::max(0, a_env("STRATA_ADAPT_ASYNC_BATCH", 0)) : 0;
        // o.adapt_async_self (arena, one GPU, nothing explicit): the homes are set up, the job thread is created only when the
        // session's measurement turns the asynchronous tier on (asw_*, in the decode loop)
        const bool a_self = o.adapt_async_self && !o.adapt_async && a_arena && !drive.d.usage.empty();
        if ((o.adapt_async || a_self) && !drive.d.usage.empty() &&
            ((src.complement_ready() && src.exchange_capacity() >= 2 * a_cap) || a_arena)) {
            ahomes.push_back({&xcache, adapt_stream, adapt_ev, 0, false});
            for (auto& st : stages) ahomes.push_back({&st->cache, st->adapt_stream, st->adapt_ev, st->dev, false});
            if (o.adapt_async) {
                ajob = std::make_unique<JobThread>();
                std::fprintf(stderr, "strata serve: adaptive tier asynchronous (--adapt-async 1): up to %lld swaps per round, "
                                     "a round every %d windows%s\n", (long long) a_cap, o.adapt_every,
                             a_arena ? " (arena: copy-in only)" : "");
            } else {
                std::fprintf(stderr, "strata serve: adaptive tier: blocking, and the asynchronous tier is tried once the session "
                                     "shows the swaps hold the windows up (--adapt-async 0 or STRATA_ADAPT_ASYNC=0: never)\n");
            }
        }
        strata::core::AdaptAsyncSelf asw;   // the session's choice between the two tiers (adapt_async_default.hpp)
        if (a_self && !ahomes.empty()) asw.arm();
        // the pipelined loop's side (see above): whether stage `st` has no window in flight; per home, the copies back
        // and in to queue at its next gap, the RAM fence still to pass and the gaps seen
        std::function<bool(int)> a_pl_idle;
        std::vector<uint8_t> a_back_due(ahomes.size(), 0), a_in_due(ahomes.size(), 0), a_ram_due(ahomes.size(), 0);
        std::vector<uint64_t> a_gap_seq(ahomes.size(), 0), a_ram_mark(ahomes.size(), 0);
        bool a_res_dirty = false;
        auto a_due = [](const std::vector<uint8_t>& v) {
            return std::any_of(v.begin(), v.end(), [](uint8_t b) { return b != 0; });
        };
        auto a_mark = [&](AHome& h) {   // an event after the copies just queued on h's stream, submitted now (WDDM)
            const strata::core::OnDevice on(h.dev);
            if (cudaEventRecord(h.ev, h.stream) != cudaSuccess) a_err = true;
            (void) cudaStreamQuery(h.stream);
            h.used = true;
        };
        auto a_pick = [&]() {   // step 1's choice (host work)
            aswaps.clear();
            for (const Swap& c : choose_swaps(a_usage, a_res, o.adapt_min_gain, false, helper_holds)) {
                if ((int64_t) aswaps.size() >= a_cap) break;
                const int32_t slot = a_res[(size_t) c.layer * g.n_expert + c.out];
                if (slot < 0) continue;
                // an exchange (as resident_stage_swaps): `in` from the RAM copy, `out` held by its slot alone takes its
                // place there; otherwise `out` is in the copy already (the lend region) or `in` is not (a RAM budget)
                const bool x = !a_arena && src.has_resident(c.layer, c.in) && !src.has_resident(c.layer, c.out);
                aswaps.push_back({c.layer, c.in, c.out, slot, multi_gpu ? model.stage_of(c.layer) : 0, (int64_t) aswaps.size(), x,
                                  nullptr});
            }
        };
        auto a_back = [&](int hi) {   // step 1's copies back (D2H) from home hi's cache, then its event
            AHome& h = ahomes[(size_t) hi];
            bool any = false;
            {
                const strata::core::OnDevice on(h.dev);
                for (const ASwap& w : aswaps) {
                    if (w.home != hi || !w.exchange) continue;
                    nib::guard_raw(w.layer, "the async adaptive tier's copy back (a_back)");
                    if (const cudaError_t e = cudaMemcpyAsync(src.exchange_buffer(w.j, w.layer), h.cache->device_slot(w.slot),
                                                              (size_t) strata::kernels::cpu::expert_layout().blob_bytes(w.layer),
                                                              cudaMemcpyDeviceToHost, h.stream); e != cudaSuccess) {
                        std::fprintf(stderr, "strata serve: adaptive tier: copy back of layer %d slot %d (CUDA%d) failed: %s\n",
                                     (int) w.layer, (int) w.slot, h.dev, cudaGetErrorString(e));
                        a_err = true;
                    }
                    any = true;
                }
            }
            if (any) a_mark(h);
        };
        auto a_from = [&](const ASwap& w) -> const uint8_t* {   // step 2: where `in` is copied from (host work)
            if (a_arena) {   // the arena's blob, pinned or not (a pageable one blocks the job thread, not the host)
                const uint8_t* b = arena_src.blob_at(w.layer, w.in);   // (blob() counts reads_: not from this thread)
                if (b == nullptr) {
                    std::fprintf(stderr, "strata serve: adaptive tier: layer %d expert %d is not in the arena\n",
                                 (int) w.layer, (int) w.in);
                    a_err = true;
                }
                return b;
            }
            if (src.pinned(w.layer, w.in))
                if (const uint8_t* b = src.resident_blob(w.layer, w.in)) return b;
            // pageable, or not in the RAM copy: through a bounce buffer (an async copy)
            uint8_t* bb = src.exchange_buffer(a_cap + w.j);
            if (bb == nullptr || !src.copy_blob(w.layer, w.in, bb)) {
                std::fprintf(stderr, "strata serve: adaptive tier: layer %d expert %d could not be read\n", (int) w.layer,
                             (int) w.in);
                a_err = true;
                return nullptr;
            }
            return bb;
        };
        auto a_in_one = [&](const ASwap& w, const uint8_t* b) {   // step 2: one copy in (H2D), on its home's stream
            AHome& h = ahomes[(size_t) w.home];
            const strata::core::OnDevice on(h.dev);
            nib::guard_raw(w.layer, "the async adaptive tier's copy in (a_in_one)");
            if (const cudaError_t e = cudaMemcpyAsync(h.cache->device_slot(w.slot), b,
                                                      (size_t) strata::kernels::cpu::expert_layout().blob_bytes(w.layer),
                                                      cudaMemcpyHostToDevice, h.stream); e != cudaSuccess) {
                std::fprintf(stderr, "strata serve: adaptive tier: copy of layer %d into slot %d (CUDA%d) failed: %s\n",
                             (int) w.layer, (int) w.slot, h.dev, cudaGetErrorString(e));
                a_err = true;
            }
        };
        auto a_choose = [&]() {   // step 1 on the job thread (the serial loop)
            a_pick();
            for (AHome& h : ahomes) h.used = false;
            for (size_t hi = 0; hi < ahomes.size(); ++hi) a_back((int) hi);
        };
        auto a_copy_in = [&]() {   // step 2 on the job thread (the serial loop): each source read, then copied in
            std::vector<uint8_t> used(ahomes.size(), 0);
            for (size_t i = a_rng.lo; i < a_rng.hi; ++i) {   // (the batch; all the round's swaps unless the arena caps it)
                const ASwap& w = aswaps[i];
                if (const uint8_t* b = a_from(w)) {
                    a_in_one(w, b);
                    used[(size_t) w.home] = 1;
                }
            }
            for (size_t hi = 0; hi < ahomes.size(); ++hi)
                if (used[hi]) a_mark(ahomes[hi]);
        };
        auto a_post_batch = [&]() {   // step 2: the next batch of copies in goes to the job thread (all of them but for the arena's cap)
            a_rng.next_batch(aswaps.size(), a_batch);
            ajob->post(a_copy_in);
        };
        auto a_stage_in = [&]() {   // step 2's host part on the job thread (the pipelined loop: a_in queues the copies)
            for (ASwap& w : aswaps) w.from = a_from(w);
        };
        auto a_in = [&](int hi) {   // step 2's copies in to home hi (the pipelined loop, at its gap)
            bool any = false;
            for (const ASwap& w : aswaps)
                if (w.home == hi && w.from != nullptr) {
                    a_in_one(w, w.from);
                    any = true;
                }
            if (any) a_mark(ahomes[(size_t) hi]);
        };
        auto a_ready = [&](bool wait) -> bool {   // the job and its copies are done
            if (wait) ajob->wait();
            else if (!ajob->idle()) return false;
            for (AHome& h : ahomes)
                if (h.used) {
                    const strata::core::OnDevice on(h.dev);
                    if (wait) { if (cudaEventSynchronize(h.ev) != cudaSuccess) a_err = true; }
                    else if (const cudaError_t q = cudaEventQuery(h.ev); q == cudaErrorNotReady) return false;
                    else if (q != cudaSuccess) a_err = true;
                }
            return true;
        };
        // --pipeline-windows 2: stage `st` has no window in flight (see above) - the copies due on its card are queued
        // now, and the steps taken before are fenced for it.  On the loop's thread.
        auto a_gap = [&](int st) {
            if (!a_pl_idle || st < 0 || st >= (int) ahomes.size() || !a_pl_idle(st)) return;
            ++a_gap_seq[(size_t) st];
            if (a_back_due[(size_t) st]) {
                a_back_due[(size_t) st] = 0;
                a_back(st);
            }
            if (a_in_due[(size_t) st] && ajob->idle()) {   // once the job has read the sources
                a_in_due[(size_t) st] = 0;
                a_in(st);
            }
        };
        auto a_picked = [&]() {   // (pipelined) step 1's choice is made: each card's copies back at its next gap
            for (AHome& h : ahomes) h.used = false;
            for (const ASwap& w : aswaps)
                if (w.exchange) a_back_due[(size_t) w.home] = 1;
            astate = AState::CopyBack;
            for (int st = 0; st < (int) ahomes.size(); ++st) a_gap(st);
        };
        // between windows: `drain` finishes the round in flight and starts none; `count` false (the pipelined loop
        // between verdicts) moves it on without counting a window.  False when a copy failed.
        auto adapt_tick = [&](bool drain, bool count = true) -> bool {
            if (!ajob) return true;
            if (!drain && count) ++a_win;   // one tick per window
            const bool pl = (bool) a_pl_idle;
            if (pl)
                for (int st = 0; st < (int) ahomes.size(); ++st) a_gap(st);
            for (;;) {
                if (a_err.load()) return false;
                switch (astate) {
                case AState::Idle:
                    if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
                    a_last = a_win;
                    ++a_rounds;
                    a_t0 = Clock::now();
                    a_usage = drive.d.usage;
                    a_res = host_res;
                    // #477: the routing counted, as adapt() counts it (--expert-profile-save; else `heat` is empty)
                    for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
                    for (float& v : drive.d.usage) v *= o.adapt_decay;
                    if (a_inline) {   // the arena: choose here (no copy back to wait for), then straight on to step 2
                        const Clock::time_point tp0 = Clock::now();
                        a_pick();
                        a_pick_ms += std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
                        for (AHome& h : ahomes) h.used = false;
                        astate = AState::CopyBack;
                        continue;
                    }
                    if (pl) {
                        ajob->post(a_pick);
                        astate = AState::Pick;
                    } else {
                        ajob->post(a_choose);
                        astate = AState::CopyBack;
                    }
                    return true;
                case AState::Pick:   // (pipelined) step 1: the copies back wait for the choice, then for their card's gap
                    if (!ajob->idle()) {
                        if (!drain) return true;
                        ajob->wait();
                    }
                    a_picked();
                    continue;
                case AState::CopyBack: {   // step 2: the evicted experts leave the GPU's plan
                    if (a_due(a_back_due)) return true;   // (pipelined) a card not seen idle since the choice
                    if (!a_ready(drain)) return true;
                    if (a_err.load()) return false;
                    size_t k = 0;
                    for (size_t i = 0; i < aswaps.size(); ++i) {
                        const ASwap w = aswaps[i];
                        const size_t out = (size_t) w.layer * g.n_expert + w.out;
                        if (host_res[out] != w.slot) continue;   // the table moved since the snapshot: not this one
                        if (w.exchange && !src.stage_exchange(w.layer, w.in, w.out, w.j)) {
                            std::fprintf(stderr, "strata serve: adaptive tier: stage_exchange refused layer %d in %d out %d\n",
                                         (int) w.layer, (int) w.in, (int) w.out);
                            return false;
                        }
                        host_res[out] = strata::core::kNotResident;   // the CPU computes it from the next window on
                        srcp->prefetch(w.layer, w.out);
                        aswaps[k++] = w;
                    }
                    aswaps.resize(k);
                    if (aswaps.empty()) {   // nothing to move (converged, or the table moved since the snapshot): the round ends here
                        a_ms += std::chrono::duration<double, std::milli>(Clock::now() - a_t0).count();
                        astate = AState::Idle;
                        continue;
                    }
                    for (AHome& h : ahomes) h.used = false;
                    if (pl) {   // the copies in at each card's next gap: after this step, so fenced (see above)
                        a_res_dirty = true;
                        for (const ASwap& w : aswaps) a_in_due[(size_t) w.home] = 1;
                        // the whole round is one batch: when the pipelined loop drains before step 3, the next tick
                        // takes step 3 on the serial rule and marks [lo, hi), which an earlier round left otherwise
                        a_rng.whole_round(aswaps.size());
                        ajob->post(a_stage_in);
                    } else {
                        res_upload();
                        a_rng.clear();
                        a_post_batch();
                    }
                    astate = AState::SwapIn;
                    if (!drain) return true;
                    continue;
                }
                case AState::SwapIn:   // step 3: the new experts are in their slots
                    if (a_due(a_in_due)) return true;   // (pipelined) a card not seen idle since step 2
                    if (!a_ready(drain || a_wait)) return true;
                    if (a_err.load()) return false;
                    for (size_t i = a_rng.landed_begin(pl); i < a_rng.landed_end(pl, aswaps.size()); ++i) {   // (the batch just landed)
                        const ASwap& w = aswaps[i];
                        host_res[(size_t) w.layer * g.n_expert + w.in] = w.slot;
                        srcp->release(w.layer, w.in);
                    }
                    if (pl) a_res_dirty = true;
                    else res_upload();
                    a_swapped += (int64_t) (a_rng.landed_end(pl, aswaps.size()) - a_rng.landed_begin(pl));
                    if (!pl && a_rng.more(aswaps.size())) {   // the arena's cap: the next batch goes now
                        for (AHome& h : ahomes) h.used = false;
                        a_post_batch();
                        if (!drain) return true;
                        continue;
                    }
                    if (pl)   // the RAM moves wait for a gap after this step on every card whose layers exchanged
                        for (const ASwap& w : aswaps)
                            if (w.exchange) {
                                a_ram_due[(size_t) w.home] = 1;
                                a_ram_mark[(size_t) w.home] = a_gap_seq[(size_t) w.home];
                            }
                    astate = AState::RamFence;
                    continue;
                case AState::RamFence:   // step 3's RAM moves, once no window may still read the places they fill
                    if (a_arena) {   // no RAM copy to move
                        astate = AState::Commit;
                        continue;
                    }
                    if (pl)
                        for (size_t hi = 0; hi < ahomes.size(); ++hi)
                            if (a_ram_due[hi] && a_gap_seq[hi] <= a_ram_mark[hi]) return true;
                    std::fill(a_ram_due.begin(), a_ram_due.end(), (uint8_t) 0);
                    astate = AState::Commit;
                    if (src.exchange_rotation()) continue;   // STRATA_EXCHANGE_ROTATE: no bytes move, the flip is all there is
                    ajob->post([&] { src.commit_copies(); });
                    if (!drain) return true;
                    continue;
                case AState::Commit:   // step 4
                    if (!a_ready(drain)) return true;
                    if (a_err.load()) return false;
                    if (!a_arena) src.commit_flip();
                    a_ms += std::chrono::duration<double, std::milli>(Clock::now() - a_t0).count();
                    aswaps.clear();
                    astate = AState::Idle;
                    continue;   // a round due now starts at this same boundary
                }
            }
        };
        // kvg_ensure's `quiesce` (and any caller about to give cache slots away): nothing runs on the device, the blocking
        // tier's swaps have landed AND the asynchronous round in flight is finished - a swap still copying could write a
        // slot the K/V is lent.  A failed refill is surfaced as ERR at the next tick (a_err).
        auto adapt_quiesce = [&] {
            cudaDeviceSynchronize();
            apply_pending(true);
            if (!adapt_tick(true)) {   // a failed round still has copies in flight: they land before any slot moves
                a_err = true;
                if (ajob) ajob->wait();
                cudaDeviceSynchronize();
            }
        };
        // --pipeline-windows 2: the loop has drained (nothing in flight on either card): what waits for a gap goes now,
        // and the round in flight goes on unfenced (a_pl_idle unset) from the next tick, the next request's drain.  The
        // job's work is the host's alone, so the wait for it ends.
        auto a_pl_settle = [&]() {
            if (!ajob || !a_pl_idle) return;
            ajob->wait();
            bool idle = true;
            for (int st = 0; st < (int) ahomes.size(); ++st) idle = idle && a_pl_idle(st);
            if (idle) {
                if (astate == AState::Pick) a_picked();
                for (int st = 0; st < (int) ahomes.size(); ++st) a_gap(st);
                if (a_res_dirty) res_upload();
            } else {   // the loop did not drain (a window failed): the round cannot go on, the next tick says so
                std::fprintf(stderr, "strata serve: adaptive tier: a window is still in flight after the pipelined loop\n");
                a_err = true;
            }
            a_pl_idle = nullptr;
            a_res_dirty = false;
        };
        // #477: write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).
        // A swap still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
        auto save_profile = [&](const char* why) {
            if (heat.empty()) return;
            std::vector<uint8_t> resident(host_res.size(), 0);
            for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
            for (const auto& p : pending) resident[(size_t) p.first] = 1;
            if (astate == AState::SwapIn)   // --adapt-async: the same for a round past its step 2
                for (const ASwap& w : aswaps) resident[(size_t) w.layer * g.n_expert + w.in] = 1;
            std::string e;
            const auto ranked = strata::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                                   profile_loaded);
            if (strata::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                std::fprintf(stderr, "strata serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(),
                             why);
            else
                std::fprintf(stderr, "strata serve: the expert profile was not saved: %s\n", e.c_str());
            profile_saved_at = Clock::now();
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        strata::serve::Input input;
        sp.should_stop = [&] { return input.is_stop_requested(); };
        // 149-prefetch STRATA_RAM_ADAPT2: the prompt's routing seeds the RAM policy with a limited weight
        // (STRATA_RAM_ADAPT_PROMPT_W, default 0.25: an expert every prompt token routes to counts like ~5 decode routings),
        // fading by 5% a decode window; the decode's own counts and the GPU tier are untouched
        if (ram_adapt2.on && ram_adapt2.prompt_w > 0.0f) {
            ram_adapt2.seed.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
            sp.on_routes = [&](int64_t l, const int32_t* ids, int64_t n) {
                if (l < 0 || l >= g.n_layers || n <= 0) return;
                const float per =
                    strata::core::get_prompt_routing_weight(ram_adapt2.prompt_w * 20.0f, g.n_expert_used, n);
                float* sd = ram_adapt2.seed.data() + l * g.n_expert;
                for (int64_t i = 0; i < n; ++i)
                    if (ids[i] >= 0 && ids[i] < g.n_expert) sd[ids[i]] += per;
            };
        }
        // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
        const bool trace = std::getenv("STRATA_TRACE") != nullptr;
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            static Clock::time_point t0;   // process-wide: the stages run on different threads
            static bool t0_set = false;
            if (!t0_set) { t0 = Clock::now(); t0_set = true; }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::fprintf(stderr, "strata trace: %s %lld %lld %.1f ms\n", what, a, b, ms);
            std::fflush(stderr);
        };
        {
            // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // STRATA_VRAM_FREE=device: the whole card's figure when it is the smaller one (the same value otherwise)
            free_b = std::min(free_b, strata::core::device_free_bytes());
            // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
            // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= 256) {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
#ifdef _WIN32
                if (free_mib < 512 && o.prefill_auto)   // #1275: said, never changed (the chunk is the operator's to cap)
                    std::fprintf(stderr, "strata serve: that is little room for the verify windows' buffers under WDDM; if the "
                                         "engine exits right after this line, --prefill auto:8192 (a smaller prompt chunk "
                                         "borrows fewer cache slots) or a larger --vram-reserve-mib leaves more\n");
#endif
                rss_probe("serving");
            } else if (reserve_adapted) {
                // #496: the reserve was already lowered to make the cache fit - a bigger one would leave it no room
                std::fprintf(stderr, "strata serve: WARNING: %lld MiB of VRAM free with everything loaded - this card "
                                     "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert "
                                     "cache fits): requests may stall. Close other programs that use the GPU, or lower "
                                     "--max-context\n", (long long) free_mib, o.vram_reserve_mib);
            } else if (g_uma_pinned_complement.load()) {
                // Round 0216: said plainly, no reserve advice (a larger --vram-reserve-mib does not raise this figure here)
                std::fprintf(stderr, "strata serve: %lld MiB of graphics memory reported free. On this PC the graphics chip shares the system "
                                     "memory, so this figure reads low, and a larger --vram-reserve-mib won't raise it. If decode drops to a "
                                     "few tok/s, lower --max-context.\n", (long long) free_mib);
            } else {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                     " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                             (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
                // #781 / #831: an explicit --expert-cache N is a byte budget that is not checked against the VRAM once
                // the slots are written (auto is), so on a card it fills a smaller N is the cure: a few hundred slots
                // can be the difference between 14 and 100 tok/s under WDDM.  Said, never changed.
                if (!auto_cache)
                    std::fprintf(stderr, "strata serve: the expert cache was set with --expert-cache (%lld slots): a "
                                         "smaller one - or --expert-cache auto, which checks the free VRAM once the "
                                         "slots are written - leaves room; decode can be several times slower with the "
                                         "card this full\n", (long long) xcache.slots());
            }
            // #2192: the same check on a layer split's later cards.  The line above reads CUDA0 alone: on 2x RTX 2080 Ti
            // (WDDM, --batch 4) CUDA0 had 740 MiB free while CUDA1 started its window captures at 153 MiB and fell to
            // 52, and the first batch window on it never finished (the host stuck in cudaGraphLaunch: a page-in while the
            // other card's graph spins on a host flag).  Said, never changed.
            for (const auto& stp : stages) {
                const strata::core::OnDevice on(stp->dev);
                size_t sf = 0, stot = 0;
                if (cudaMemGetInfo(&sf, &stot) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                const int64_t sf_mib = (int64_t) (sf >> 20);
                const int later_reserve = o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
                if (sf_mib >= 512)
                    std::fprintf(stderr, "strata serve: CUDA%d: %lld MiB of VRAM free with everything loaded\n", stp->dev,
                                 (long long) sf_mib);
                else
                    std::fprintf(stderr, "strata serve: CUDA%d: %lld MiB of VRAM free with everything loaded - LOW: its window "
                                         "graphs are captured on first use and need room too, and %s; add "
                                         "--vram-reserve-later-mib %lld to the config's args (or lower --max-context)\n",
                                 stp->dev, (long long) sf_mib,
#ifdef _WIN32
                                 "under WDDM a full card pages instead of failing, which can stall a request for good (#2192)",
#else
                                 "a full card can stall a request",
#endif
                                 (long long) (later_reserve + 768 - sf_mib));
                if (sf_mib < 512) {   // #3702: how far the plan was off, so the reserve to ask for is not a guess
                    strata::program::stage_reserve::Inputs rin;
                    rin.base_reserve_mib = later_reserve;
                    rin.prompt_buffers_mib = split_pf_mib;
                    rin.later = true;
                    rin.pipe_window_mib = model.pipe_window_mib;
                    const std::string why = strata::program::gpu_report::planned_vs_free_text(
                        strata::program::stage_reserve::reserve_mib(rin), sf_mib);
                    if (!why.empty()) std::fprintf(stderr, "strata serve: CUDA%d: %s\n", stp->dev, why.c_str());
                }
            }
        }
        // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // "Experts in VRAM" is every tier, not one card's.  This used to report `xcache` alone, so a layer split
            // showed CUDA0's cache as if it were the whole GPU's: a 4-GPU 256K run read 3327 experts when the four
            // cards held 13320, and the Monitor tab was wrong by 4x for every multi-GPU config.  The tiers are
            // disjoint by construction (remote_experts.cpp skips any pair a stage already claimed), so they add.
            const int64_t slots_primary = (int64_t) xcache.slots();
            const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
            int64_t slots_all = slots_primary, mib_all = mib_primary;
            for (const auto& st : stages) {
                slots_all += (int64_t) st->cache.slots();
                mib_all += (int64_t) (st->cache.bytes() >> 20);
            }
            for (int r = 0; r < 3; ++r)
                if (o.expert_cache_remote[(size_t) r] > 0) {
                    slots_all += remote_experts[(size_t) r].resident();
                    mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
                }
            print_response(
                "INFO logit_bias=1 context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                        "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f conversation_cache_mib=%lld conversation_cache_slots=%d "
                        "conversation_cache_min_free_mib=%lld conversation_cache_min_tokens=%lld "
                        "tail_role_token=%lld vram_elastic=%d "
                        "prefill_cap=%lld prefill_stages=%d stream_all_min=%lld prefill_pipe=%g%s engine=" STRATA_VERSION "\n",
                        (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                         ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                        (long long) slots_all, (long long) mib_all,
                        (long long) slots_primary, (long long) mib_primary,
                        o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) ((o.mmap_experts ? src.resident_bytes() : strata::kernels::cpu::expert_layout().total) >> 20),
                        pool.workers(), o.pcie_frac,
                        o.spec_min_p, (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                        (long long) o.conversation_cache_min_free_mib, (long long) o.conversation_cache_min_tokens,
                        (long long) o.tail_role_token, xcache.segmented() ? 1 : 0,
                        // the prompt path: its chunk cap (the VRAM command's relend can lower it later), the stages a
                        // prompt flows through, the chunk size from which a chunk streams every expert, --prefill-pipe
                        (long long) o.prefill_chunk, sp.stage_count(),
                        (long long) strata::prefill::Prefill::stream_all_min_tokens(),
                        strata::prefill::Prefill::pipe_setting(),
                        o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                       " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                       " batch_groups=" + std::to_string(o.batch_groups)).c_str() : "");
            if (strata::core::manifest::enabled()) {   // STRATA_MANIFEST: engine choice and budgets (a diagnostic; off by default)
                auto& M = strata::core::manifest::instance();
                M.engine("version", STRATA_VERSION);
                M.engine("engine", stages.empty() ? "native-single-gpu" : "native-layer-split");
                M.engine("stages", std::to_string(stages.size() + 1));
                M.engine("kv_format", o.kv);
                M.engine("drafter", !o.mtp.empty() ? "mtp" : "none-or-lookup");
                M.engine("spec_window", std::to_string(o.spec));
                M.engine("pipeline_windows", std::to_string(o.pipeline_windows));
                M.engine("ple_io", o.ple_io);
                M.engine("expert_cache_request", o.expert_cache < 0 ? "auto" : std::to_string(o.expert_cache));
                M.budget("max_context", (int64_t) o.max_context);
                M.budget("kv_resident_cells", (int64_t) o.kv_resident);
                M.budget("expert_slots_all", slots_all);
                M.budget("expert_cache_all_mib", mib_all);
                M.budget("expert_slots_primary", slots_primary);
                M.budget("expert_cache_primary_mib", mib_primary);
                M.budget("vram_reserve_mib", (int64_t) o.vram_reserve_mib);
                M.budget("prefill_chunk_cap", (int64_t) o.prefill_chunk);
                M.budget("prefill_stages", (int64_t) sp.stage_count());
                M.budget("pool_workers", (int64_t) pool.workers());
                M.budget("memory_limit_mib", (int64_t) o.memory_limit_mib);
                M.budget("resident_budget_bytes", (int64_t) o.resident_budget);
                M.budget("batch_slots", (int64_t) o.batch);
                M.budget("conversation_cache_mib", (int64_t) o.conversation_cache_mib);
                M.path("decode", "route-tail-skip", std::getenv("STRATA_ROUTE_TAIL_SKIP") == nullptr || std::getenv("STRATA_ROUTE_TAIL_SKIP")[0] != '0',
                       std::getenv("STRATA_ROUTE_TAIL_SKIP") == nullptr ? "STRATA_ROUTE_TAIL_SKIP not set: the engine default applies (it skips tails only where the configuration allows)"
                                                                         : (std::string("STRATA_ROUTE_TAIL_SKIP=") + std::getenv("STRATA_ROUTE_TAIL_SKIP")));
                M.path("transfer", "dma-batch", strata::core::dma_batch_mode() != 0,
                       strata::core::dma_batch_mode() != 0 ? "STRATA_DMA_BATCH" : "mode 0: one copy per expert (the default loop)",
                       std::string("mode ") + std::to_string(strata::core::dma_batch_mode()));
            }
        }
        // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
        // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
        // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
        // part was doing (stall_report), so one occurrence says where the wait is.
        {
            const char* ws = std::getenv("STRATA_WATCHDOG_S");
            const int limit = ws ? std::atoi(ws) : 60;   // one step (a prompt layer, a verify window) takes seconds
            // #1407: while the OS is still reading the file tier (major faults / bytes read advance), a silent step is
            // slow, not stuck; it gets up to STRATA_WATCHDOG_IO_S (default 10 x the limit) of such time. 0 = no allowance.
            const char* wio = std::getenv("STRATA_WATCHDOG_IO_S");
            const int io_limit = wio ? std::atoi(wio) : limit * 10;
            if (limit > 0)
                std::thread([limit, io_limit] {
                    strata::platform::name_this_thread("strata-watchdog");
                    strata::aux_cpus::pin_current_thread();
                    strata::core::Progress& p = strata::core::progress();
                    uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                    auto since = std::chrono::steady_clock::now();
                    auto io_since = since;
                    uint64_t io_last = file_tier_activity();
                    auto io_active_at = since;   // the last second the file tier was busy
                    bool io_noted = false;
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t b = p.beats.load();
                        if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; io_since = now; io_active_at = now; io_noted = false; io_last = file_tier_activity(); continue; }
                        {   // sampled every second, also during the first `limit` seconds, so a busy file tier is seen
                            const uint64_t a = file_tier_activity();
                            if (a >= io_last + 16) io_active_at = now;   // 16 = faults + MiB read in a second: busy
                            io_last = a;
                        }
                        if (now - since < std::chrono::seconds(limit)) continue;
                        if (io_limit > limit && now - io_since < std::chrono::seconds(io_limit)) {
                            // a thrashing host has seconds with almost no reads (one thread waiting on one page while
                            // swap is written): busy within the last 10 s still counts as slow, not stuck (P2003, 46 GB)
                            if (now - io_active_at < std::chrono::seconds(10)) {
                                if (!io_noted) {
                                    io_noted = true;
                                    std::fprintf(stderr, "strata serve: no step finished for %d s, but the file tier is still being read "
                                                         "(slow storage or low RAM): waiting up to %d s in all (STRATA_WATCHDOG_IO_S) (#1407)\n",
                                                 limit, io_limit);
                                }
                                continue;
                            }
                        }
                        // a blocking step's explicit allowance (session files): still within it, not yet stuck
                        if (strata::core::progress_now_ms() < p.allow_until_ms.load()) continue;
                        std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s) - stopping "
                                             "the engine so the server starts it again (issue #29)%s\n",
                                     limit, stage_text().c_str(),
                                     is_launch_blocking() ? "; CUDA_LAUNCH_BLOCKING=1 is set, and no verify window can "
                                                         "finish with it (#1341)" : "");
                        stall_report(stderr, p.ticks.load() - ticks_at);
                        strata::core::release_gpu_waits(stderr);   // #267: no spin kernel outlives the process
                        std::fflush(stderr);
                        std::abort();
                    }
                }).detach();
        }
        strata::core::pin_sim_report("at READY");
        strata::core::pin_reserve_release_hard();   // the load is over: nothing else needs the headroom
        strata::core::manifest::instance().write("ready");   // STRATA_MANIFEST (a no-op unless it is set)
        // "stop": this engine honours STOP; "conversations": it answers LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS;
        // "memory": GET_MEMORY_USE
        strata::serve::configure_legacy_requests(o.batch);
        print_response("INFO request_capacity=%d\n", std::max(1, o.batch));
        print_response("READY %lld stop conversations memory requests\n", (long long) o.max_context);
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec;
        const int S_mtp_wide = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        // --draft auto: a request that drafts with the MTP layer has its own widest window (the MTP loses to itself above 4 on Blackwell), the DFlash2 requests S_mtp_wide
        const int S_mtp_narrow = o.draft_auto ? std::min(S_mtp_wide, std::max(2, o.draft_mtp_spec)) : S_mtp_wide;
        int S_mtp = S_mtp_wide;   // this request's (set when it starts)
        int S_look = S;   // --draft auto: the widest window a prompt-lookup draft may make for this request (the MTP requests keep the MTP's width, as without DFlash2)
        if (use_mtp && S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
        // --pipeline-windows 2 loads the drafter with 8 rows: the serial loop's chain keeps its own length
        else if (use_mtp && mtp.max_t() > S_mtp) mtp.set_max_drafts(S_mtp - 1);
        // the draft policy learns over the whole process
        strata::spec::WindowPlanner wplan(S, S_mtp, o.suffix_draft, o.lookup_chain, o.lookup_chain_min, o.max_context);
        if (wplan.cost_aware && use_mtp)
            mtp.set_chain_rule([&](int idx, float pj) { return wplan.keep_draft(idx, pj); });
        wplan.policy.set_sizes(lookup_size_mask(o));
        // a layer split's hand-off buffers hold kVerifyNarrowT rows: its windows stay at that
        if (n_stages > 1) wplan.policy.set_sizes(wplan.policy.sizes() & ((2u << strata::kernels::kVerifyNarrowT) - 1u));
        if (o.lookup_max_t > 0)
            std::fprintf(stderr, "strata serve: --lookup-max-t %d: MTP windows up to %d tokens, window sizes %s\n",
                         o.lookup_max_t, S_mtp, size_list(wplan.policy.sizes(), S).c_str());
        // --draft auto: the other drafter's learned costs, swapped into wplan.policy when a request uses that drafter
        strata::spec::DraftPolicy policy_idle = wplan.policy;
        bool policy_is_df2 = use_df2;   // which drafter's costs wplan.policy holds (the engine's first drafter at the start)
        struct DraftBandit { double tps[2] = {0, 0}; int n[2] = {0, 0}; int64_t reqs = 0; };   // [1] = DFlash2, [0] = MTP: decode tok/s of finished requests (--draft-rule learn)
        std::unordered_map<uint64_t, DraftBandit> bandit;
        static const bool draft_log = std::getenv("STRATA_DRAFT_LOG") != nullptr;   // one stderr line per request: drafter, tokens, rounds, ms, upkeep ms
        static const bool draft_keep_off = std::getenv("STRATA_DRAFT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DRAFT_KEEP")) == 0;   // diagnostic: do not keep the idle drafter current
        int64_t draft_req_count = 0;
        // --lookup-max-t: a lookup window's graph is captured before its round; a size whose graph finds no VRAM is
        // dropped for good (the policy never picks it again) and the round takes the next smaller allowed size.
        // -1: a capture failed for another reason (err says why).
        auto graph_fits = [&](int t) -> int {
            while (t > 1) {
                bool no_vram = false;
                if (ver.prepare(t, no_vram, err)) return t;
                if (!no_vram) return -1;
                wplan.policy.disable(t);
                std::fprintf(stderr, "strata serve: no VRAM for the %d-token window graph: that size is dropped, "
                                     "sizes left %s (a larger --vram-reserve-mib keeps it)\n", t,
                             size_list(wplan.policy.sizes(), S).c_str());
                t = wplan.policy.fit(t - 1);
            }
            return 1;
        };
        // Shrinks a window of `t` rows to a size whose graph fits (--lookup-max-t; otherwise every size does); false
        // when a capture failed for another reason (err says why).
        auto fit_rows = [&](int& t) -> bool {
            if (o.lookup_max_t > 0) t = graph_fits(t);
            return t >= 0;
        };
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        // ---- #533: VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache
        // until that much VRAM is free for other programs (a game, a CAD session), or grows it back towards its full
        // size when more than that is free.  Never on its own: only this command (the server's POST /v1/vram) moves
        // it.  A shrink gives back the cache's LAST segments: the experts in their slots become CPU misses, as any
        // expert outside the cache is (no output changes beyond what a smaller cache decodes), the prompt path's
        // loan moves down to the end of what is left (a smaller chunk when it no longer fits).  A grow maps them
        // again and puts each slot's expert back (or, when the adaptive tier already brought that one back, the
        // most-routed missing expert of the same layer); the adaptive tier carries on from there.
        std::vector<std::pair<int32_t, int32_t>> vram_evicted;   // (residency index, slot) the shrinks took
        const int64_t chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache
        auto relend = [&]() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
            if (pf_parts.empty() || pf_parts[0].first < 0) return;
            PfPart& p = pf_parts[0];
            const int64_t live = xcache.slots();
            auto fits = [&](int64_t c) {
                const int64_t k = part_slots(p, c);
                return k > 0 && k + g_lend_keep <= live && !(o.prefill_auto && k * 100 > kAutoLendPct * live);
            };
            int64_t c = chunk_full;
            while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
            o.prefill_chunk = c;
            p.first = (int32_t) std::max<int64_t>(0, live - part_slots(p, c));
            p.first_now = -1;   // laid out again at the next loan
        };
        auto vram_command = [&](const std::string& cmd, std::string& e) -> bool {
            // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
            char* end = nullptr;
            const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
            if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
            if (!xcache.segmented()) {
                e = "VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)";
                return false;
            }
            if (src.complement_ready()) {
                e = "VRAM: not with the resident low-RAM mode (the cache holds experts RAM does not)";
                return false;
            }
            if (peer.valid()) { e = "VRAM: not with --peer-device"; return false; }
            for (const PfPart& p : pf_parts)
                if (!p.lent.empty()) { e = "VRAM: a prompt's loan is still out"; return false; }
            if (!ver.wait_commit(e)) return false;
            adapt_quiesce();                         // the adaptive tier's swaps in flight land first (async too)
            if (cudaDeviceSynchronize() != cudaSuccess) {
                e = std::string("VRAM: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            const auto t0 = Clock::now();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
            const int64_t before = xcache.slots();
            // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
            const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
                g_lend_keep + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
            std::string note;
            if ((int64_t) free_b < want_free) {
                const int64_t floor_b = xcache.bytes_of(floor_slots);
                int64_t keep = mapped - (want_free - (int64_t) free_b);
                if (keep < floor_b) {
                    keep = floor_b;
                    note = " (the cache keeps its smallest size: the prompt path's buffers)";
                }
                if (!xcache.shrink(keep, e)) return false;
                const int64_t live = xcache.slots();
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= live) {
                        vram_evicted.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                res_upload();
            } else if (mapped < xcache.full_bytes()) {
                std::string gerr;
                if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
                const int64_t live = xcache.slots();
                std::string ferr;
                std::vector<std::pair<int32_t, int32_t>> keep_out;
                for (const auto& [i, slot] : vram_evicted) {
                    if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                    const int64_t layer = i / g.n_expert;
                    int64_t pick = i;
                    if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                        pick = -1;
                        float best = -1.0f;
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t j = (size_t) (layer * g.n_expert + ex);
                            if (host_res[j] >= 0) continue;
                            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                            if (u > best) { best = u; pick = (int64_t) j; }
                        }
                    }
                    if (pick < 0) continue;
                    const uint8_t* b = srcp->blob_stable(layer, pick % g.n_expert);
                    if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                            (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(layer))) {
                        e = "VRAM: refilling the cache failed: " + ferr;
                        return false;
                    }
                    srcp->note_async_read(b, nullptr);   // #1237: the queued copy reads a stage buffer; it is not recycled under it
                    host_res[(size_t) pick] = slot;
                }
                vram_evicted.swap(keep_out);
                if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
                res_upload();
            }
            relend();
            cudaMemGetInfo(&free_b, &total_b);
            std::fprintf(stderr, "strata serve: VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                                 "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s\n",
                         reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                         (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                         (long long) (free_b >> 20), (long long) o.prefill_chunk,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
            print_response("VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld expert_cache_mib=%lld "
                        "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld\n", reserve,
                        (long long) xcache.slots(), (long long) xcache.full_slots(),
                        (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                        (long long) (free_b >> 20), (long long) o.prefill_chunk);
            return true;
        };
        // ---- --batch: the slots of the batch windows
        struct BSlot {
            bool active = false, stop = false;
            int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
            std::array<int32_t, strata::kernels::kVerifyNarrowT> draft{}; ///< the slot's next MTP proposal
            bool draft_ready = false;
            std::array<float, strata::kernels::kVerifyNarrowT> dprob{};   ///< DFlash2 slot drafts: the selector's probability of each draft
            int n_drafts = 1;              ///< STRATA_PIPED_MTP: how many of `draft` its next window checks
            int64_t p = 0;                 ///< its position
            int64_t produced = 0, max_new = 0;
            Clock::time_point t0;
            // the slot's own conversation cache: the tokens its sessions hold (the prompt, then every token a window
            // fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from it
            // instead of reading its history again (see `slot_source` in the request path).
            std::vector<int32_t> ids;
            int64_t reply0 = 0;            ///< the index in `ids` where its penalty window starts (the last prompt token; 0 with STRATA_PENALTY_PROMPT=1)
            int pen_n = 0;                 ///< its request's penalty window (penalty_last_n) when it has a repeat / frequency / presence penalty, else 0
            bool cached = false;           ///< idle, and its sessions still hold `ids`
            bool cvec = true;              ///< the control vector setting `ids` were read with
            bool img = false;              ///< the conversation has pictures (never reused from the slot)
            // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's
            // header): a client's next turn renders the history again WITHOUT this reply's thinking, so it matches
            // the slot's tokens only up to there - the slot's K/V up to it plus this state continue from it
            std::vector<ConvCheckpoint> checks;
            // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
            // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0)
            bool partial = false, partial_from0 = false;
            // #879: it was running when non-finite logits dropped the cached state; its state may have started from
            // that state (a checkpoint, a parked conversation), so it is not kept as a cache when it ends
            bool tainted = false;
            ConvCheckpoint::PromptPlan partial_plan;   ///< --prefill-pipe: the segment it stopped in, read on in its chunks
        };
        std::vector<BSlot> bs((size_t) std::max(o.batch, 0));
        // STRATA_BATCH_DF2_DEFER=0: every round drafts for every slot, spare rows or not (the first measurement's behaviour, for the A/B)
        // STRATA_BATCH_DF2_CONF=P (default: --draft-conf, else 0.35; 0 off): a slot's draft rows stop where the product of the selector's probabilities falls below P;
        // STRATA_BATCH_DF2_MAXX=N: at most N draft rows a slot and window (default kDrafts)
        const double df2_conf = [&] { const char* e = std::getenv("STRATA_BATCH_DF2_CONF"); return e != nullptr && e[0] != '\0' ? std::atof(e) : (o.draft_conf > 0.0 ? o.draft_conf : 0.35); }();
        const int df2_cap = [] { const char* e = std::getenv("STRATA_BATCH_DF2_MAXX"); return e != nullptr && e[0] != '\0' ? std::max(0, std::atoi(e)) : 99; }();
        const bool batch_df2_defer = [] { const char* e = std::getenv("STRATA_BATCH_DF2_DEFER"); return !(e != nullptr && e[0] == '0'); }();
        // timing of the batch windows since the slots were last all idle (one stderr line then)
        double bt_run = 0, bt_commit = 0, bt_emit = 0;
        double bt_wait0 = 0, bt_pool0 = 0;
        int64_t bt_miss0 = 0, bt_hits0 = 0, bt_pcie0 = 0;
        int64_t bt_windows = 0, bt_rows = 0, bt_tokens = 0, bt_accepted = 0;   // bt_accepted: --batch-mtp proposals taken
        int64_t pipe_rows = 0;   // the pipelined groups' rows, never reset (bt_rows restarts when the slots go idle)
        // per stage (0 = CUDA0's verifier): the GPU-reach wait and pool time when the timed windows began; and the routed
        // (token, expert) entries the CPU and the PCIe share had served by then (bt_hits0 holds the VRAM entries)
        std::vector<double> bt_stage_wait0, bt_stage_pool0;
        int64_t bt_cpu_ent0 = 0, bt_off_ent0 = 0;
        // the adaptive VRAM tier in batch windows: windows seen since the engine started (--adapt-every counts them), and,
        // since the last timing line, its rounds, the experts it swapped in and the ms a window waited for it
        int64_t ba_window = 0, bt_adapt_rounds = 0, bt_adapt_swaps = 0;
        double bt_adapt_wait = 0;
        bool in_prompt_read = false;   // batch_step runs between a prompt's chunks: the adaptive tier stands still then
        double serve_adapt_drain_ms = 0.0;   // 149: the request's wait for the adaptive round in flight
        int64_t bt_a_rounds0 = 0, bt_a_swapped0 = 0;   // the asynchronous tier's counters when the timed windows began
        Clock::time_point bt_start = Clock::now();
        const char* batch_telemetry_env = std::getenv("STRATA_BATCH_TELEMETRY");
        const bool batch_telemetry_enabled = batch_telemetry_env != nullptr &&
                                             batch_telemetry_env[0] == '1' && batch_telemetry_env[1] == '\0';
        strata::core::BatchWindowSampler batch_telemetry(batch_telemetry_enabled);
        strata::program::ParCounters pc;   // STRATA_PAR_COUNTERS=1: counters only (par_counters.hpp); off: a branch
        auto pc_ex = [&]() {
            strata::program::ParCounters::Ex e;
            e.misses = drive.d.multi_misses; e.entries = drive.d.multi_entries; e.hits = drive.d.cache_hits;
            e.offload = drive.d.offload_entries; e.pcie = drive.d.pcie_experts; e.run_ms = drive.d.ms_run; e.jobs_ms = drive.d.ms_jobs;
            e.vwait_ms = ver.ms_wait; e.vpool_ms = ver.ms_pool;
            return e;
        };
        auto report_batch_window = [&](int processed_rows, int active_rows, bool idle, Clock::time_point started) {
            if (!batch_telemetry_enabled) return;
            const auto sample = batch_telemetry.complete(processed_rows, active_rows, idle, started, Clock::now());
            if (!sample) return;
            print_response("BSTAT %lld %lld %lld %lld\n", (long long) sample->windows,
                        (long long) sample->processed_rows, (long long) sample->active_rows,
                        (long long) sample->elapsed_ms);
            std::fflush(stdout);
        };
        int admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
        long long admit_max_new = 0;
        // BCO <slot> <ids> (research, PAR-H; STRATA_COALESCE_PREFILL on the server): prompts of waiting requests, read in
        // the SAME layer-major pass as the next BGEN's prompt into the named idle slots' sessions (the experts a layer
        // streams are shared), so their own admissions find the prompt in the slot and read nothing.  BCOK <slot> <tokens>
        // says a slot holds its prompt; a declined or failed co-read says nothing and costs nothing.
        struct CoReq { int slot = -1; std::vector<int64_t> ids; };
        std::vector<CoReq> co_pending, co_set;
        auto batch_on = [&] { for (const BSlot& b : bs) if (b.active) return true; return false; };
        // the main sessions and the slot sessions of a stage live on the same device and are carved alike, so
        // a slot copy can go device to device (the same bytes into the same places as the checkpoint/K-V image
        // round trip through host memory, which costs ~40-90 ms per admission on a 4-card split); each device is
        // synchronized once before and once after.  Opt-in: STRATA_SLOT_COPY_DIRECT=1 (default: the host path).
        static const bool slot_copy_direct = [] {
            const char* v = std::getenv("STRATA_SLOT_COPY_DIRECT");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        // synchronizes every device a slot session is on; false when one failed (the others are synchronized still)
        auto sync_all_slot_devices = [&]() {
            bool ok = true;
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                ok = cudaDeviceSynchronize() == cudaSuccess && ok;
            }
            return ok;
        };
        // the main sessions -> slot b's (`into_slot`; the session a request just left behind, its prompt `ids`) or
        // slot b's -> the main ones (a request that continues the conversation an idle slot holds reads on from
        // there), on every stage.  `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state
        // restored instead
        auto copy_slot = [&](bool into_slot, int b, const std::vector<int32_t>& ids, const ConvCheckpoint* at,
                             std::string& e) -> bool {
            // the main session's K/V is replaced by the slot's: K/V retained from a restore described the old one
            if (!into_slot)
                conversations.take_reuse();
            const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) ids.size();
            if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
#if defined(STRATA_GPU_METAL)
            // the running state and the K/V of the Metal window, not the session's device buffers
            if (!strata::metal_engine::window_slot_copy_main(into_slot, b, upto, at != nullptr, e)) return false;
            return at == nullptr || strata::core::conversation_checkpoint_restore(*at, ss, g, e);   // (the checkpoint's running state)
#endif
            const char* sync_failed = into_slot ? "batch admission: device sync failed"
                                                : "batch slot restore: device sync failed";
            const bool direct = slot_copy_direct && at == nullptr;
            // STRATA_SLOT_COPY_BATCHSYNC: every direct copy is only queued, and the sync after them is the one wait
            // per device
            const bool queued = direct && slot_copy_batchsync_env();
            if (direct && !sync_all_slot_devices()) { e = "batch slot copy: device sync failed"; return false; }
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                strata::core::SessionState& main_ss = k == 0 ? ss : stages[k - 1]->ss;
                strata::core::SessionState& slot_ss = *bslot_ss[k][(size_t) b];
                strata::core::SessionState& from = into_slot ? main_ss : slot_ss;
                strata::core::SessionState& to = into_slot ? slot_ss : main_ss;
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (direct) {
                    // a copy that fails part-way leaves earlier residency refreshes queued on the legacy stream:
                    // drain them before the caller reuses the sessions (its zeroing runs on non-blocking streams,
                    // not ordered against these)
                    bool ok = strata::core::conversation_checkpoint_copy(from, to, ids, g, e, queued);
                    for (int64_t j = 0; ok && j < from.qsa_alloc; ++j)
                        ok = strata::core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j],
                                                                to.qsa_states[to.qsa_ord0 + j], g, upto, true, e,
                                                                queued);
                    if (!ok) {
                        (void) sync_all_slot_devices();
                        return false;
                    }
                    continue;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) { e = sync_failed; return false; }
                if (at != nullptr) {
                    if (!strata::core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, g, e))
                        return false;
                } else {
                    strata::core::ConversationCheckpoint ck;
                    ck.ids = ids;
                    if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                        !strata::core::conversation_checkpoint_restore(ck, to, g, e))
                        return false;
                }
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    strata::core::ConversationKv img;
                    if (!strata::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                        return false;
                    if (cudaDeviceSynchronize() != cudaSuccess) { e = sync_failed; return false; }
                    if (!strata::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                        return false;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) { e = sync_failed; return false; }
            }
            if (direct && !sync_all_slot_devices()) { e = "batch slot copy: device sync failed"; return false; }
            if (batch_df2) {
                // the slot's drafter starts from an exact copy of the solo drafter's ring (its taps came with the
                // prompt read), and a returning solo request goes on from the slot's ring
#if defined(STRATA_HAVE_DFLASH2)
                if (!(into_slot ? slot_df2[(size_t) b]->copy_state_from(df2, e)
                                : df2.copy_state_from(*slot_df2[(size_t) b], e)))
                    return false;
#endif
            } else
            if (batch_mtp) {
                // Admission first builds the solo draft KV; copy it into the slot before drafting, and a returning solo
                // request resumes from its slot's.  Both drafters' K/V are on the drafter's device (the last stage's
                // GPU with a layer split): the copy's ring refill is a kernel launch there.
                strata::core::MtpDrafter& from = into_slot ? mtp : *slot_mtp[(size_t) b];
                strata::core::MtpDrafter& to = into_slot ? *slot_mtp[(size_t) b] : mtp;
                const strata::core::OnDevice on_mtp(mtp.device());
                strata::core::ConversationKv image;
                if (!from.idle(e) || !to.idle(e) ||
                    !strata::core::conversation_kv_save(image, from.kv_state(), draft_geometry, upto, false, e) ||
                    !strata::core::conversation_kv_restore(image, to.kv_state(), draft_geometry, upto, false, e) ||
                    cudaDeviceSynchronize() != cudaSuccess)
                    return false;
                to.set_prompt_len(upto);
            }
            return true;
        };
        auto stage_verifier = [&](int k) -> strata::core::Verifier& { return k == 0 ? ver : stages[(size_t) k - 1]->ver; };
        // What an idle slot holds (the conversation it decoded last: its sessions, its draft K/V and its checkpoints)
        // goes to the parked conversations before an admission overwrites it.  The main session was parked with that
        // conversation's prompt only (the admission's copy), so the reply the slot decoded lives nowhere else: without
        // this, the conversation's next turn finds its prompt-read state and reads the whole reply again (measured on
        // four 200K chats over two slots: 255 of a 256-token reply re-read, and a chat whose copies the cache had
        // evicted meanwhile read from token 0).  Nothing to do when the live main session or a parked conversation
        // already holds that state (or a continuation of it).
        // Parking is an optimization like a checkpoint (#3350): a skip and a failed snapshot (a host or device
        // allocation that ran out) are logged and the slot is overwritten as before.  Nothing here ends the request
        // or the engine.
        auto park_slot = [&](int b) {
            BSlot& sl = bs[(size_t) b];
            if (!conversations.enabled() || sl.active || !sl.cached || sl.img || sl.ids.empty()) return;
            auto prefix_of = [&](const std::vector<int32_t>& whole) {
                return whole.size() >= sl.ids.size() && std::equal(sl.ids.begin(), sl.ids.end(), whole.begin());
            };
            const bool in_main = live_ok && cvec_cached == sl.cvec && prefix_of(live) &&
                                 imgs_below(live_imgs, (int64_t) sl.ids.size()).empty();
            if (in_main || conversations.holds(sl.ids, {}, sl.cvec)) return;
            if (conversations.supersedes_richer(sl.ids, {}, sl.checks, sl.cvec)) {
                // its park would drop (as superseded) the main session's copy of this conversation, and with it the checkpoints the slot
                // does not hold - an agent's system-prompt root that the next requests borrow (#4420, as 0.2.6 did: the slot is overwritten)
                std::fprintf(stderr, "strata serve: conversation cache: slot %d not parked before its overwrite (a parked copy of its "
                                     "conversation holds checkpoints the slot does not; it stays)\n", b);
                return;
            }
            if (use_mtp && !batch_mtp) {   // no draft K/V of its own: an image without it would not restore beside the drafter
                std::fprintf(stderr, "strata serve: conversation cache: slot %d not parked before its overwrite (its %zu tokens "
                                     "have no per-slot draft K/V without --batch-mtp)\n", b, sl.ids.size());
                return;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr, "strata serve: conversation cache: slot %d not parked before its overwrite (device sync "
                                     "failed)\n", b);
                return;
            }
            std::string why;
            if (batch_mtp && !batch_df2 && !slot_mtp[(size_t) b]->idle(why)) {
                std::fprintf(stderr, "strata serve: conversation cache: slot %d not parked before its overwrite (%s)\n",
                             b, why.c_str());
                return;
            }
            const size_t n_st = stages.size();
            static const std::vector<ImgKey> no_imgs;
            ParkSource src{sl.ids, no_imgs, sl.checks, sl.cvec, *bslot_ss[0][(size_t) b], {},
                           (n_st > 0 && !mtp_first) || !batch_mtp || batch_df2 ? nullptr : &slot_mtp[(size_t) b]->kv_state(),
                           batch_mtp && !batch_df2 && !mtp_first && n_st > 0 ? &slot_mtp[(size_t) b]->kv_state() : nullptr, false,
                           " from slot " + std::to_string(b)};
            for (size_t k = 1; k < bslot_ss.size(); ++k) src.stage_sessions.push_back(bslot_ss[k][(size_t) b].get());
            try {
                if (park_source(src, 0)) return;
                // the snapshot failed (the copy's own memory, as in #3350): the slot is not touched, so go on without
                // the image; a non-sticky allocation error is cleared, a sticky one comes back from the next call
                std::fprintf(stderr, "strata serve: conversation cache: slot %d not parked before its overwrite (%s)\n",
                             b, err.empty() ? "the snapshot failed" : err.c_str());
                err.clear();
                (void) checkpoint_failure_is_survivable();
            } catch (const std::bad_alloc&) {
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking slot %d\n", b);
            }
        };
        // one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
        // so it keeps the conversation it holds
        auto batch_step = [&]() -> bool {
            int S = 0;
            int rows[strata::kernels::kVerifyMaxT] = {};
            int32_t tok[strata::kernels::kVerifyMaxT] = {}, outb[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            int first[strata::kernels::kVerifyMaxT] = {}, active[strata::kernels::kVerifyMaxT] = {};
            static size_t next_slot = 0;
            // Each MTP slot uses two rows; rotate slots when more than four are active.
            int A = 0;
            // (--batch-mtp: two rows a slot, at most kVerifyNarrowT; the plain windows hold up to kVerifyMaxT)
            const int win_cap = batch_mtp ? (int) strata::kernels::kVerifyNarrowT : (int) strata::kernels::kVerifyMaxT;
            int nrow[strata::kernels::kVerifyMaxT] = {};   // the rows slot a (the a-th active) holds: its token and its drafts
            // DFlash2 slots: the rows beyond one a slot are SPARE; they carry the slots' drafts, shared evenly in slot order (so the layout of a window
            // depends on which slots are active, not on the drafts: the captured batch graphs are reused).  More slots than rows: no drafts, the slots
            // take turns as in the plain windows.  A draft row a slot has no room for (its last tokens, the context's end) is not asked for.
            int df2_extra[64] = {};
            if (batch_df2) {
                int act_ids[64] = {}, nact = 0;
                for (size_t b = 0; b < bs.size() && nact < (int) (sizeof act_ids / sizeof act_ids[0]); ++b)
                    if (bs[b].active) act_ids[nact++] = (int) b;
                const int spare = std::max(0, win_cap - nact);
                for (int k = 0; k < nact && nact > 0; ++k) {
                    const BSlot& q = bs[(size_t) act_ids[k]];
                    int want = spare / nact + (k < spare % nact ? 1 : 0);
                    want = std::min(want, (int) strata::core::Dflash2Drafter::kDrafts);
                    want = std::min<int>(want, (int) std::max<int64_t>(0, q.max_new - q.produced - 1));
                    want = std::min<int>(want, (int) std::max<int64_t>(0, (int64_t) o.max_context - q.p - 2));
                    want = std::min(want, df2_cap);
                    if (q.draft_ready && df2_conf > 0.0) {   // a draft row is worth its cost where the chain so far is likely: the product of the selector's probabilities
                        double chain = 1.0;
                        int keep_n = 0;
                        while (keep_n < want && (chain *= (double) q.dprob[(size_t) keep_n]) >= df2_conf) ++keep_n;
                        want = keep_n;
                    }
                    if (act_ids[k] < (int) (sizeof df2_extra / sizeof df2_extra[0])) df2_extra[act_ids[k]] = want;
                }
#if defined(STRATA_HAVE_DFLASH2)
                // A slot that sat out earlier rounds (no spare rows: its drafter did no work, the committed rows' taps wait on the host) gets its ring brought
                // up to date and its first drafts now, before the window is built.  If that cannot be done it takes no drafts this round.
                for (int k = 0; k < nact; ++k) {
                    const int b = act_ids[k];
                    BSlot& q = bs[(size_t) b];
                    if (b >= (int) (sizeof df2_extra / sizeof df2_extra[0]) || df2_extra[b] == 0 || q.draft_ready) continue;
                    strata::core::Dflash2Drafter& d = *slot_df2[(size_t) b];
                    int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                    float pp[strata::core::Dflash2Drafter::kDrafts];
                    const size_t row_bytes = (size_t) d.n_taps() * (size_t) g.n_embd * 4;
                    const strata::core::OnDevice on_d(d.device());
                    if (!d.has_deferred() || d.deferred_hi() != q.p || !d.catch_up(q.p - 1, err) ||
                        cudaMemcpyAsync(d.landing(), d.deferred_row(q.p - 1), row_bytes, cudaMemcpyHostToDevice, d.stream()) != cudaSuccess ||
                        !d.draft(d.landing(), 1, q.p - 1, q.x, dd, pp, err)) {
                        if (err.empty()) err = "the deferred rows do not end at the slot's position";
                        std::fprintf(stderr, "strata batch: DFlash2 slot %d: catch-up failed (%s): no drafts for it this round\n", b, err.c_str());
                        err.clear();
                        df2_extra[b] = 0;
                        continue;
                    }
                    d.clear_deferred();
                    for (int j = 0; j < strata::core::Dflash2Drafter::kDrafts && j < (int) q.draft.size(); ++j) { q.draft[(size_t) j] = dd[j]; q.dprob[(size_t) j] = pp[j]; }
                    q.draft_ready = true;
                }
#endif
            }
            for (size_t offset = 0; offset < bs.size(); ++offset) {
                const int b = (int) ((next_slot + offset) % bs.size());
                if (!bs[(size_t) b].active) continue;
                const int extra = batch_df2 ? (b < (int) (sizeof df2_extra / sizeof df2_extra[0]) ? df2_extra[b] : 0) : batch_mtp ? 1 : 0;
                if (S + 1 + extra > win_cap) break;
                BSlot& q = bs[(size_t) b];
                if (batch_mtp && !q.draft_ready && !batch_df2) {
                    err = "batch MTP: a live slot has no draft";
                    return false;
                }
                first[A] = S;
                nrow[A] = 1 + extra;
                active[A++] = b;
                rows[S] = b;
                tok[S] = q.x;
                pos[S] = q.p;
                ++S;
                for (int j = 0; j < extra; ++j) {
                    rows[S] = b;
                    tok[S] = q.draft[(size_t) j];
                    pos[S] = q.p + 1 + j;
                    ++S;
                }
            }
            if (batch_mtp && A > 0) next_slot = ((size_t) active[A - 1] + 1) % bs.size();
            if (S == 0) return true;
            const bool was_busy = strata::core::progress().busy.load();
            strata::core::progress().busy.store(true);
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            apply_pending(false);
            // --adapt-async: the round in flight moves on a step when its job and copies are done (or a due round starts).
            // Never between a prompt's chunks (its loan of cache slots and its chunks read the tables); a request drains
            // the round before its prompt starts (adapt_tick(true) where the request begins).  The main thread's part is
            // small (the table changes and uploads); the copies are on the helper thread and the cards' refill streams.
            if (ajob && !in_prompt_read) {
                const Clock::time_point ta = Clock::now();
                if (!adapt_tick(false)) {
                    print_response("ERR an adaptive refill failed\n");
                    return false;
                }
                bt_adapt_wait += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            }
            if (bt_windows == 0) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
                bt_stage_wait0.assign(stages.size() + 1, 0.0);
                bt_stage_pool0.assign(stages.size() + 1, 0.0);
                for (size_t k = 0; k <= stages.size(); ++k) {
                    bt_stage_wait0[k] = stage_verifier((int) k).ms_wait;
                    bt_stage_pool0[k] = stage_verifier((int) k).ms_pool;
                }
                bt_cpu_ent0 = drive.d.multi_entries; bt_off_ent0 = drive.d.offload_entries;
                if (pc.on()) pc.begin(pc_ex(), (int) g.n_layers);
                bt_a_rounds0 = a_rounds; bt_a_swapped0 = a_swapped;
            }
            for (int t = 0; t < S; ++t) {   // each penalised slot's reply tail for its penalty window (the verifier builds the rows from it and this window's tokens)
                const BSlot& ps = bs[(size_t) rows[t]];
                if (ps.pen_n <= 0 || (t > 0 && rows[t - 1] == rows[t])) continue;
                const int64_t from = std::max<int64_t>(ps.reply0, (int64_t) ps.ids.size() - ps.pen_n);
                ver.set_slot_history(rows[t], ps.ids.data() + from, (int64_t) ps.ids.size() - from);
            }
            const Clock::time_point w0 = Clock::now();
            if (!ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
                print_response("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                return false;
            }
            const Clock::time_point w1 = Clock::now();
            // The adaptive VRAM tier, as the solo loop runs it: every --adapt-every windows, on a thread beside the commit and
            // the drafts.  Every stage's window has finished here (run_slot_rows synchronised each stage's stream), and
            // nothing in flight reads the cache slots a swap overwrites; the swapped-in experts are landed (apply_pending)
            // before this window ends, so no other path (a prompt chunk read between windows, a loan of cache slots, a KV
            // growth) ever meets swaps in flight.  Not between a prompt's chunks: a loan of cache slots marks its lent
            // experts missing, and they are put back when the loan ends.  (With --adapt-async the tier is ticked above.)
            struct AdaptRun {
                std::thread th;
                bool ok = true;
                ~AdaptRun() { if (th.joinable()) th.join(); }   // an error return below still joins it
            } adapt_run;
            if (!drive.d.usage.empty() && !ajob && !in_prompt_read && (++ba_window % o.adapt_every) == 0 && pending.empty())
                adapt_run.th = std::thread([&] { adapt_run.ok = adapt(); });
            // Accept the proposal only when the target picked it and there is room to emit both tokens.
            std::vector<int> keep(bs.size(), 0);
            for (int a = 0; a < A; ++a) {
                const int b = active[a], i = first[a];
                const BSlot& sl = bs[(size_t) b];
                const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outb[i]) != o.eos_ids.end();
                (void) eos;
                int kp = 1;   // the proposals the target picked, in order: row j + 1 stands when the target's pick after row j is its token and nothing ended
                while (batch_mtp && kp < nrow[a] && outb[i + kp - 1] == tok[i + kp] && !sl.stop &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outb[i + kp - 1]) == o.eos_ids.end() &&
                       sl.produced + kp + 1 <= sl.max_new && sl.p + kp + 2 <= o.max_context)
                    ++kp;
                keep[b] = kp;
            }
            if (!(batch_mtp ? ver.commit_slot_prefixes(keep.data(), err) : ver.commit_slots(err))) {
                print_response("ERR %s\n", err.c_str());
                return false;
            }
            const Clock::time_point w2 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            bt_run += msd(w0, w1);
            bt_commit += msd(w1, w2);
            ++bt_windows;
            for (int a = 0; a < A; ++a) {
                bt_rows += keep[active[a]];
                bt_accepted += keep[active[a]] - 1;
            }
            std::vector<int> df2_launched;   // batch DFlash2: slots whose draft graph is queued
            for (int t = 0; t < A; ++t) {
                const int b = active[t];
                BSlot& sl = bs[(size_t) b];
                bool nf = false;   // #879: a row of this slot held non-finite logits: the slot ends, nothing emitted
                for (int r = first[t]; r < first[t] + (batch_mtp ? nrow[t] : 1); ++r) nf = nf || ver.nonfinite_in(r);
                if (nf) {
                    std::fprintf(stderr, "strata batch: non-finite logits (#879) in slot %d at position %lld: it ends "
                                         "(BDONE nonfinite) and the next request reads its prompt from token 0\n", b,
                                 (long long) pos[first[t]]);
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                    print_response("BDONE %d %lld nonfinite %.1f\n", b, (long long) sl.produced, ms);
                    sl.active = false;
                    sl.cached = false;
                    sl.checks.clear();
                    nan_flush = true;
                    continue;
                }
                for (int j = 0; j < keep[b]; ++j) {
                    const int32_t y = outb[first[t] + j];
                    sl.ids.push_back(tok[first[t] + j]);
                    print_response("BT %d %d\n", b, (int) y);
                    ++sl.produced;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        print_response("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.cached = o.prompt_cache > 0 && !sl.img && !sl.tainted;
                        break;
                    }
                    sl.x = y;
                    sl.p += 1;
                }
#if defined(STRATA_HAVE_DFLASH2)
                if (batch_df2 && sl.active) {
                    // The slot's committed rows' taps (every stage's share, landed on the drafter's card) are its drafter's next context; its drafts follow
                    // from the last committed token.  The window's host sync is behind us, so the taps are final.
                    strata::core::Dflash2Drafter& d = *slot_df2[(size_t) b];
                    const int kp = keep[b];
                    int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                    float pp[strata::core::Dflash2Drafter::kDrafts];
                    d.note_rows(tok + first[t], kp, pos[first[t]]);
                    if (nrow[t] > 1 || !batch_df2_defer) {
                        // this window carried drafts for the slot: its drafter takes the committed rows' taps and drafts again
                        if (d.has_deferred() && !d.catch_up(pos[first[t]], err)) {
                            print_response("ERR batch DFlash2 slot %d: %s\n", b, err.c_str());
                            return false;
                        }
                        d.clear_deferred();
                        if (!ver.gather_taps(d.landing(), d.device(), first[t], kp, 0, d.stream(), err) ||
                            !d.draft_launch(d.landing(), kp, pos[first[t]], outb[first[t] + kp - 1], err)) {
                            print_response("ERR batch DFlash2 slot %d: %s\n", b, err.c_str());
                            return false;
                        }
                        df2_launched.push_back(b);   // finished after the loop: the slots' graphs run at the same time
                        (void) dd; (void) pp;
                    } else {
                        // no spare row for the slot: no draft, no drafter work; the rows' taps wait on the host until spare rows return
                        float* dst = d.defer_rows(pos[first[t]], kp, err);
                        if (dst == nullptr || !ver.gather_taps_host(dst, first[t], kp, err)) {
                            print_response("ERR batch DFlash2 slot %d: %s\n", b, err.c_str());
                            return false;
                        }
                        sl.draft_ready = false;
                    }
                }
#endif
                if (batch_mtp && !batch_df2 && sl.active) {
                    // The slot's two residual rows (the final residuals of the last stage: ver.final_R_all() follows the
                    // chain, so with a layer split they are on the last stage's GPU, where the drafters are).  The copy is
                    // queued on the drafter's own stream after the window's host sync, so the draft graph reads it.
                    const strata::core::OnDevice on_slot_mtp(slot_mtp[(size_t) b]->device());
                    const size_t stride = (size_t) g.hc * (size_t) g.n_embd;
                    if (cudaMemcpyAsync(slot_mtp_rows[(size_t) b].get(),
                                        ver.final_R_all() + (size_t) first[t] * stride,
                                        2 * stride * sizeof(float), cudaMemcpyDeviceToDevice,
                                        slot_mtp[(size_t) b]->stream()) != cudaSuccess ||
                        !slot_mtp[(size_t) b]->draft(2, outb + first[t], pos[first[t]], keep[b] - 1,
                                                    sl.draft.data(), err)) {
                        print_response("ERR batch MTP slot %d: %s\n", b, err.c_str());
                        return false;
                    }
                    sl.draft_ready = true;
                }
            }
#if defined(STRATA_HAVE_DFLASH2)
            for (const int b : df2_launched) {
                int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                float pp[strata::core::Dflash2Drafter::kDrafts];
                BSlot& sl = bs[(size_t) b];
                if (!slot_df2[(size_t) b]->draft_finish(dd, pp, err)) {
                    print_response("ERR batch DFlash2 slot %d: %s\n", b, err.c_str());
                    return false;
                }
                for (int j = 0; j < strata::core::Dflash2Drafter::kDrafts && j < (int) sl.draft.size(); ++j) { sl.draft[(size_t) j] = dd[j]; sl.dprob[(size_t) j] = pp[j]; }
                sl.draft_ready = true;
            }
#endif
            std::fflush(stdout);
            bt_emit += msd(w2, Clock::now());
            if (adapt_run.th.joinable()) {
                const Clock::time_point wa = Clock::now();
                adapt_run.th.join();
                if (!adapt_run.ok) {
                    print_response("ERR an adaptive refill failed\n");
                    return false;
                }
                ++bt_adapt_rounds;
                bt_adapt_swaps += (int64_t) pending.size();
                apply_pending(true);   // the copies land and the tables follow (the next window plans from them)
                bt_adapt_wait += msd(wa, Clock::now());
            }
            strata::core::progress().busy.store(was_busy);
            report_batch_window(S, A, !batch_on(), w0);
            pc.window(S, A);
            if (!batch_on() && bt_windows > 0) {
                const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
                pc.flush(wall, pc_ex());
                const double L = (double) g.n_layers;
                const double e_vram = (double) (drive.d.cache_hits - bt_hits0), e_pcie = (double) (drive.d.offload_entries - bt_off_ent0),
                             e_cpu = (double) (drive.d.multi_entries - bt_cpu_ent0);
                const double e_all = std::max(e_vram + e_pcie + e_cpu, 1.0);
                std::string per_stage;   // every stage's own host wait for its GPU and its pool time (run_slot_rows recurses)
                for (size_t k = 0; k <= stages.size(); ++k) {
                    char sb[96];
                    std::snprintf(sb, sizeof sb, "%sCUDA%d wait %.2f + pool %.2f", k == 0 ? "" : ", ", k == 0 ? 0 : stages[k - 1]->dev,
                                  (stage_verifier((int) k).ms_wait - bt_stage_wait0[k]) / w,
                                  (stage_verifier((int) k).ms_pool - bt_stage_pool0[k]) / w);
                    per_stage += sb;
                }
                char adapt_txt[160] = "";
                if (!drive.d.usage.empty() && ajob)
                    std::snprintf(adapt_txt, sizeof adapt_txt,
                                  "; adaptive tier (async): %lld rounds started, %lld experts swapped in",
                                  (long long) (a_rounds - bt_a_rounds0), (long long) (a_swapped - bt_a_swapped0));
                else if (!drive.d.usage.empty())
                    std::snprintf(adapt_txt, sizeof adapt_txt, "; adaptive tier: %lld rounds, %lld experts swapped in",
                                  (long long) bt_adapt_rounds, (long long) bt_adapt_swaps);
                std::fprintf(stderr, "strata batch: %lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach "
                                     "wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f + adapt wait %.2f; per stage "
                                     "(ms/window): %s; per layer-window: CPU experts %.2f, VRAM hits %.2f, PCIe %.2f; routed "
                                     "entries per window: VRAM %.1f (%.1f%%), PCIe %.1f (%.1f%%), CPU %.1f (%.1f%%)%s; %.1f rows/s "
                                     "over %.0f ms of wall time (admissions included)\n",
                             (long long) bt_windows, bt_rows / w, (bt_run + bt_commit + bt_emit + bt_adapt_wait) / w, bt_run / w,
                             (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                             bt_adapt_wait / w, per_stage.c_str(),
                             (drive.d.multi_misses - bt_miss0) / (w * L), (drive.d.cache_hits - bt_hits0) / (w * L),
                             (drive.d.pcie_experts - bt_pcie0) / (w * L), e_vram / w, 100.0 * e_vram / e_all, e_pcie / w,
                             100.0 * e_pcie / e_all, e_cpu / w, 100.0 * e_cpu / e_all, adapt_txt,
                             1000.0 * bt_rows / std::max(wall, 1e-9), wall);
                for (size_t k = 0; k <= stages.size(); ++k) {
                    const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
                    if (!pr.empty()) std::fprintf(stderr, "strata batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
                }
#if defined(STRATA_HAVE_DFLASH2)
                if (batch_df2) {
                    int64_t cu = 0, cr = 0, dr = 0;
                    double cm = 0;
                    for (const auto& dsl : slot_df2) { cu += dsl->n_catchups; cr += dsl->n_catchup_rows; dr += dsl->n_deferred_rows; cm += dsl->ms_catchup; }
                    std::fprintf(stderr, "strata batch: DFlash2 slots: %lld rows deferred (no spare window row), %lld catch-ups replayed %lld rows in %.1f ms (%.2f ms each)\n",
                                 (long long) dr, (long long) cu, (long long) cr, cm, cu > 0 ? cm / (double) cu : 0.0);
                }
#endif
                if (batch_mtp)   // each slot-window offers one proposal and yields 1 + (accepted) tokens
                    std::fprintf(stderr, "strata batch: MTP proposals accepted %lld of %lld (%.1f%%)\n",
                                 (long long) bt_accepted, (long long) (bt_rows - bt_accepted),
                                 100.0 * (double) bt_accepted / (double) std::max<int64_t>(bt_rows - bt_accepted, 1));
                bt_run = bt_commit = bt_emit = bt_adapt_wait = 0;
                bt_windows = bt_rows = bt_tokens = bt_accepted = bt_adapt_rounds = bt_adapt_swaps = 0;
            }
            return true;
        };
        // ---- --batch-groups: the slot groups pipelined through the stages (--batch-groups G > 1 with a layer split)
        const int n_pipe = (int) stages.size() + 1;
        const bool piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1;
        strata::core::StageInFlightSampler stage_in_flight(n_pipe, piped && batch_telemetry_enabled);
        const int GS = piped ? o.batch / o.batch_groups : o.batch;
        // STRATA_PIPED_MTP: the pipelined groups draft too - a group's window carries each of its slots' rows (the slot's
        // token and that slot's own proposals), so any number of slots per group drafts - and every stage saves the rows
        // their commit replays
        const int pmtp_rows_per_slot = 1 + pm_drafts;              // the slot's token + its proposals
        const int pmtp_rows_per_group = GS * pmtp_rows_per_slot;    // the group's slots
        const bool pmtp = piped && piped_mtp && ver.deferred_commit_on() &&
                          pmtp_rows_per_group <= batch_max_t();
        if (piped && batch_mtp && !pmtp)   // #1413: said, not silent
            std::fprintf(stderr,
                         "strata generate: WARNING: --batch-groups %d runs the pipelined path, which builds one row per slot: "
                         "--batch-mtp's per-slot MTP drafts do not run there (their drafters still hold VRAM, about 0.9 GB per "
                         "slot). Drop --batch-mtp, or use --batch-groups 1.\n", o.batch_groups);
        if (GS > strata::kernels::kVerifyMaxT && !batch_mtp) {   // (slots_in_windows keeps it so: a group's window is
            // PGroup's rows, and the serial windows reach only the first kVerifyMaxT slots)
            std::fprintf(stderr, "strata generate: --batch %d in %d group%s: more than %d rows per window\n",
                         o.batch, std::max(o.batch_groups, 1), o.batch_groups > 1 ? "s" : "",
                         (int) strata::kernels::kVerifyMaxT);
            return 1;
        }
        struct PGroup {
            bool inflight = false;
            int stage = 0;                  ///< the stage it runs on or waits for
            int32_t tok[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            int64_t since = 0;              ///< tick it started waiting (fairness)
            int S = 0;                      ///< slots in its window: up to its last active one (idle ones cost rows)
            int active = 0;                  ///< live slots at launch; S can also include idle padding rows
            Clock::time_point started;       ///< the group window's first stage launch
            // STRATA_PIPED_MTP (pmtp): S counts ROWS - each of the group's slots' token, then that slot's proposals - at
            // hand-off rows [hbase, hbase + S)
            int rows[strata::kernels::kVerifyMaxT] = {};
            int hbase = 0;
        };
        std::vector<PGroup> pg((size_t) (piped ? o.batch_groups : 0));
        std::vector<int> stage_group((size_t) n_pipe, -1);
        int64_t pipe_tick = 0, rr = 0;
        auto pipe_inflight = [&] { for (const PGroup& x : pg) if (x.inflight) return true; return false; };
        auto group_active = [&](int gi) {
            for (int t = 0; t < GS; ++t) if (bs[(size_t) (gi * GS + t)].active) return true;
            return false;
        };
        // STRATA_TRACE: the drain before the line in hand (an admission's first phase), and when the first rows came
        // after a batch admission (the pipeline's refill after BADM) or after a drain for another line (drain + refill)
        double line_drain_ms = 0.0, adm_drain_ms = 0.0;
        const char* rows_wait = nullptr;   // what the next rows are timed from (rows_wait_t)
        Clock::time_point rows_wait_t{};
        // STRATA_PIPED_CONCURRENT (read_part): while a prompt is read on a thread of its own, the slots' doorbells are
        // served on another (conc_serving); between windows that thread waits on bell_cv and a launch wakes it (a
        // timed sleep is a whole timer tick on Windows, up to 15.6 ms, and the window's first layer would wait for it)
        bool conc_serving = false;
        std::mutex bell_mu;
        std::condition_variable bell_cv;
        std::atomic<bool> bell_waiting{false};
        // STRATA_PIPED_MTP: a slot whose next proposals are drafting (its chain in flight on the drafter's stream), and
        // per slot the event after the copy of its rows on the last stage's stream, which the chain waits for
        std::vector<char> pm_drafting(bs.size(), 0);
        std::vector<cudaEvent_t> pm_copied(pmtp ? bs.size() : 0, nullptr);
        if (pmtp) {
            const strata::core::OnDevice on_last(stage_verifier(n_pipe - 1).device());
            for (cudaEvent_t& e : pm_copied)
                if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: STRATA_PIPED_MTP: event creation failed\n");
                    return 1;
                }
        }
        // the chain of slot b has landed: its proposals are the slot's next drafts (1), still drafting (0), or an error (-1)
        auto pmtp_poll_draft = [&](int b, bool wait) -> int {
            if (!pm_drafting[(size_t) b]) return 1;
            for (;;) {
                const int r = slot_mtp[(size_t) b]->chain_poll(err);
                if (r < 0) { print_response("ERR batch MTP slot %d: %s\n", b, err.c_str()); return -1; }
                if (r == 1) break;
                if (!wait) return 0;
            }
            pm_drafting[(size_t) b] = 0;
            // the first proposal always; another while the draft layer was at least --spec-min-p sure of the one before
            // (the solo path's rule, MtpDrafter::draft)
            BSlot& sl = bs[(size_t) b];
            const int32_t* tok = slot_mtp[(size_t) b]->chain_tok();
            const float* prob = slot_mtp[(size_t) b]->chain_prob();
            sl.n_drafts = 1;
            sl.draft[0] = tok[0];
            while (sl.n_drafts < pm_drafts && prob[sl.n_drafts - 1] >= (float) o.spec_min_p) {
                sl.draft[(size_t) sl.n_drafts] = tok[sl.n_drafts];
                ++sl.n_drafts;
            }
            sl.draft_ready = true;
            return 1;
        };
        // STRATA_PIPED_MTP: the last stage has picked for group gi's window (its slot's token and G.S - 1 proposals).
        // Judge the proposals, queue every stage's commit of the kept rows (each stage runs it before that group's next
        // window there: the same stream), emit them, and start drafting the next proposals.
        auto pmtp_verdict = [&](PGroup& G, strata::core::Verifier& vk, const int32_t* outb) -> bool {
            auto is_eos = [&](int32_t y) {
                return std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
            };
            // each slot of the group owns a contiguous block of the window's rows: its token, then its own proposals
            struct Blk { int b = 0, f = 0, n = 0, keep = 1; };
            std::vector<Blk> blks;
            std::vector<int> keep(bs.size(), 1);
            for (int t = 0; t < G.S;) {
                const int b = G.rows[t];
                const int f = t;
                while (t < G.S && G.rows[t] == b) ++t;
                BSlot& sl = bs[(size_t) b];
                // proposal j stands when the target picked it after the row before it (and every earlier one stood); the
                // group was started only with proposals there is room to emit
                Blk blk;
                blk.b = b;
                blk.f = f;
                blk.n = t - f;
                while (blk.keep < blk.n && !sl.stop && outb[f + blk.keep - 1] == G.tok[f + blk.keep] &&
                       !is_eos(outb[f + blk.keep - 1]))
                    ++blk.keep;
                keep[(size_t) b] = blk.keep;
                bt_accepted += blk.keep - 1;
                bt_tokens += blk.n - 1;   // (proposals checked)
                blks.push_back(blk);
            }
            for (int k = 0; k < n_pipe; ++k)
                if (!stage_verifier(k).batch_commit_rows(G.rows, G.S, G.hbase, keep.data(), G.tok, G.pos, err)) {
                    print_response("ERR %s\n", err.c_str());
                    return false;
                }
            // #879: a row with non-finite logits: the slot that owns it ends, nothing of it is emitted
            for (int t = 0; t < G.S; ++t)
                if (vk.nonfinite_in(t)) {
                    const int b = G.rows[t];
                    BSlot& sl = bs[(size_t) b];
                    std::fprintf(stderr, "strata batch: non-finite logits (#879) in slot %d at position %lld: it ends (BDONE "
                                         "nonfinite) and the next request reads its prompt from token 0\n", b, (long long) sl.p);
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                    print_response("BDONE %d %lld nonfinite %.1f\n", b, (long long) sl.produced, ms);
                    sl.active = false;
                    sl.cached = false;
                    sl.draft_ready = false;
                    sl.checks.clear();
                    nan_flush = true;
                }
            for (const Blk& blk : blks) {
                BSlot& sl = bs[(size_t) blk.b];
                if (!sl.active) continue;   // it ended above (non-finite logits) or before this window
                for (int j = 0; j < blk.keep; ++j) {
                    const int32_t y = outb[blk.f + j];
                    print_response("BT %d %d\n", blk.b, (int) y);
                    ++sl.produced;
                    ++bt_rows;
                    ++pipe_rows;
                    const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    sl.ids.push_back(G.tok[blk.f + j]);
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        print_response("BDONE %d %lld %s %.1f\n", blk.b, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.draft_ready = false;
                        // a cache again (#857), as on the pipelined path without drafts: final once its commits have run
                        // (pmtp_settle, before anything reads the slot's sessions)
                        sl.cached = o.prompt_cache > 0 && !sl.img && !sl.tainted;
                        break;
                    }
                    sl.x = y;
                    sl.p += 1;
                }
            }
            // the next proposal: each slot's drafter catches up on ITS OWN rows of this window and drafts from the last
            // kept one, as a chain on its own stream (pmtp_poll_draft collects it before the group's next window).  The
            // rows - the last stage's final residuals - are copied on THAT stage's stream, so its next window, which
            // rewrites them, runs after the copy; the chain waits for the copy.
            const size_t stride = (size_t) g.hc * (size_t) g.n_embd;
            for (const Blk& blk : blks) {
                BSlot& sl = bs[(size_t) blk.b];
                if (!sl.active) continue;   // it ended: no more drafts for it
                const strata::core::OnDevice on_slot_mtp(slot_mtp[(size_t) blk.b]->device());
                sl.draft_ready = false;
                const float* src = vk.final_R_all() + (size_t) blk.f * stride;
                if (cudaMemcpyAsync(slot_mtp_rows[(size_t) blk.b].get(), src, (size_t) blk.n * stride * sizeof(float),
                                    cudaMemcpyDeviceToDevice, vk.stream()) != cudaSuccess ||
                    cudaEventRecord(pm_copied[(size_t) blk.b], vk.stream()) != cudaSuccess ||
                    cudaStreamWaitEvent(slot_mtp[(size_t) blk.b]->stream(), pm_copied[(size_t) blk.b], 0) != cudaSuccess ||
                    !slot_mtp[(size_t) blk.b]->chain_launch(blk.n, outb + blk.f, G.pos[blk.f], blk.keep - 1, nullptr, 0,
                                                            pm_drafts, 0, err)) {
                    print_response("ERR batch MTP slot %d: %s\n", blk.b,
                                err.empty() ? "the proposal's chain did not start" : err.c_str());
                    return false;
                }
                pm_drafting[(size_t) blk.b] = 1;
            }
            return true;
        };
        // STRATA_PIPED_MTP: the commits queued after the verdicts have run on every stage, so whatever reads a slot's
        // sessions next (an admission, a parked conversation, the solo path taking a slot back) sees its final state
        auto pmtp_settle = [&]() -> bool {
            if (!pmtp) return true;
            for (int b = 0; b < (int) bs.size(); ++b)   // (a chain in flight would refuse the drafter's next use)
                if (pmtp_poll_draft(b, true) < 0) return false;
            for (int k = 0; k < n_pipe; ++k) {
                strata::core::Verifier& vk = stage_verifier(k);
                const strata::core::OnDevice on_k(vk.device());
                if (cudaStreamSynchronize(vk.stream()) != cudaSuccess) {
                    print_response("ERR a pipelined commit failed on stage %d\n", k + 1);
                    return false;
                }
            }
            return true;
        };
        // STRATA_PIPE_ADAPT_EVERY=N (opt-in; unset or 0: off, as before): the pipelined groups start no adaptive-tier round
        // of their own (batch_step starts one every --adapt-every windows; pump below never does), so under steady parallel
        // load the VRAM tier only moves when the slots go back to the solo path, and a fresh engine keeps its start-up
        // profile for minutes.  With N > 0, after N group windows the pipeline starts no new group, lets the ones in flight
        // finish, and runs one round there as batch_step runs it between serial windows (nothing in flight reads a cache
        // slot; the swapped-in experts land before the next launch), then goes on.  Never while a prompt is read beside
        // the slots (its loan of cache slots), nor with --adapt-async (its own ticks).
        // (the rule: batch_groups.hpp pipe_adapt_every / pipe_adapt_due, tested in batch_groups_test)
        const int pipe_adapt_every = strata::program::batch_groups::pipe_adapt_every(std::getenv("STRATA_PIPE_ADAPT_EVERY"));
        int64_t pa_windows = 0, pa_last = 0, pa_rounds = 0, pa_swaps = 0;
        double pa_ms = 0;
        auto pa_due = [&] {
            strata::program::batch_groups::PipeAdapt p;
            p.every = pipe_adapt_every;
            p.tier_on = !drive.d.usage.empty();
            p.async_tier = (bool) ajob;
            p.prompt_read = in_prompt_read;
            p.swaps_pending = !pending.empty();
            p.windows = pa_windows;
            p.last = pa_last;
            return strata::program::batch_groups::pipe_adapt_due(p);
        };
        // STRATA_P4_TIMING=1 / STRATA_P4_TRACE=PATH: where a pipelined window's time goes (pipe_probe.hpp; off: a branch)
        strata::program::PipeProbe p4(n_pipe, pg.size());
        // one turn: serve every running stage, hand finished groups on, start what can start
        auto pump = [&](bool may_start) -> bool {
            ++pipe_tick;
            const strata::program::PipeProbe::PumpScope p4_scope(p4);
            if (p4.timing()) {
                int nb = 0, nf = 0;
                for (int k = 0; k < n_pipe; ++k) if (stage_group[(size_t) k] >= 0) ++nb;
                for (const PGroup& x : pg) if (x.inflight) ++nf;
                p4.busy(nb, nf);
                // a stage free with no group waiting at it = starved by the stage before, else it was only idle a moment
                for (int k = 1; k < n_pipe; ++k) {
                    if (stage_group[(size_t) k] >= 0) continue;
                    bool ready = false;
                    for (size_t g2 = 0; g2 < pg.size() && !ready; ++g2) ready = pg[g2].inflight && pg[g2].stage == k;
                    p4.free_stage(k, ready);
                }
            }
            if (bt_windows == 0 && !pipe_inflight()) {
                bt_start = Clock::now();
                p4.span_start(bt_start);
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
                if (pc.on()) pc.begin(pc_ex(), (int) g.n_layers);
            }
            for (int k = 0; k < n_pipe; ++k) {
                const int gi = stage_group[(size_t) k];
                if (gi < 0) continue;
                strata::core::Verifier& vk = stage_verifier(k);
                const auto p4_p0 = p4.now();
                const int r = vk.batch_poll(win_pool_fn, win_pool_user, err);
                p4.polled(p4_p0);
                if (r < 0) { print_response("ERR %s\n", err.c_str()); return false; }
                if (r == 0) continue;
                p4.done(k, gi);
                if (batch_telemetry_enabled) stage_in_flight.complete(k, Clock::now());
                stage_group[(size_t) k] = -1;
                PGroup& G = pg[(size_t) gi];
                if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
                // the last stage: the group's picks
                const int32_t* outb = vk.batch_out();
                if (pmtp && !pmtp_verdict(G, vk, outb)) return false;
                const strata::program::PipeProbe::VerdictScope p4_verdict(p4);
                for (int t = 0; t < G.S && !pmtp; ++t) {
                    BSlot& sl = bs[(size_t) (gi * GS + t)];
                    if (!sl.active) continue;
                    if (vk.nonfinite_in(t)) {   // #879: the slot ends, nothing emitted
                        std::fprintf(stderr, "strata batch: non-finite logits (#879) in slot %d at position %lld: it "
                                             "ends (BDONE nonfinite) and the next request reads its prompt from token 0\n",
                                     gi * GS + t, (long long) sl.p);
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        print_response("BDONE %d %lld nonfinite %.1f\n", gi * GS + t, (long long) sl.produced, ms);
                        sl.active = false;
                        sl.cached = false;
                        sl.checks.clear();
                        nan_flush = true;
                        continue;
                    }
                    const int32_t y = outb[t];
                    print_response("BT %d %d\n", gi * GS + t, (int) y);
                    ++sl.produced;
                    ++bt_rows;
                    ++pipe_rows;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    sl.ids.push_back(sl.x);
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        print_response("BDONE %d %lld %s %.1f\n", gi * GS + t, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        // a pipelined slot is a cache again (#857: a request left alone in its slot goes back to the solo path with its
                        // drafts): its state is final once its last window has left the last stage.  A later group window whose pad
                        // row writes this slot clears the flag (below, where the group starts), so a stale state is never reused
                        sl.cached = o.prompt_cache > 0 && !sl.img && !sl.tainted;
                    } else {
                        sl.x = y;
                        sl.p += 1;
                    }
                }
                std::fflush(stdout);
                if (rows_wait != nullptr) {
                    std::fprintf(stderr, "strata trace: first rows %.1f ms after %s\n",
                                 std::chrono::duration<double, std::milli>(Clock::now() - rows_wait_t).count(), rows_wait);
                    rows_wait = nullptr;
                }
                ++bt_windows;
                ++pa_windows;
                G.inflight = false;
                report_batch_window(G.S, G.active, !pipe_inflight() && !batch_on(), G.started);
                pc.window(G.S, G.active);
            }
            // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
            for (int k = n_pipe - 1; k >= 0; --k) {
                if (stage_group[(size_t) k] >= 0) continue;
                int pick = -1;
                if (k > 0) {
                    for (int gi = 0; gi < (int) pg.size(); ++gi)
                        if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                            (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                            pick = gi;
                    if (pick < 0) continue;
                } else {
                    if (!may_start || pa_due()) continue;   // STRATA_PIPE_ADAPT_EVERY: a round is due, the pipeline empties first
                    for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                        const int gi = (int) ((rr + j) % (int64_t) pg.size());
                        if (pg[(size_t) gi].inflight || !group_active(gi)) continue;
                        if (pmtp) {   // its slots' proposal chains still running: another group goes first
                            bool ready = true;
                            for (int t = 0; t < GS; ++t) {
                                const int dr = pmtp_poll_draft(gi * GS + t, false);
                                if (dr < 0) return false;
                                if (dr == 0) { ready = false; break; }
                            }
                            if (!ready) continue;
                        }
                        pick = gi;
                    }
                    if (pick < 0) continue;
                    rr = pick + 1;
                    PGroup& G = pg[(size_t) pick];
                    G.S = 0;
                    if (pmtp) {   // the group's slots: each its token, then its own proposals - as many as could be emitted
                        G.hbase = pmtp_rows_per_group * pick;
                        G.S = 0;
                        G.active = 0;
                        for (int t = 0; t < GS; ++t) {
                            const int b = pick * GS + t;
                            const BSlot& sl = bs[(size_t) b];
                            if (!sl.active) continue;   // an idle slot carries no row (its state is left untouched)
                            ++G.active;
                            G.rows[G.S] = b;
                            G.tok[G.S] = sl.x;
                            G.pos[G.S] = sl.p;
                            ++G.S;
                            int nd = sl.draft_ready && !sl.stop ? sl.n_drafts : 0;
                            nd = (int) std::min<int64_t>(nd, std::min<int64_t>(sl.max_new - sl.produced - 1,
                                                                                o.max_context - sl.p - 2));
                            for (int j = 0; j < nd; ++j) {
                                G.rows[G.S] = b;
                                G.tok[G.S] = sl.draft[(size_t) j];
                                G.pos[G.S] = sl.p + 1 + j;
                                ++G.S;
                            }
                        }
                        if (G.S == 0) { G.inflight = false; continue; }   // every slot of the group went idle
                    } else {
                        G.active = 0;
                        for (int t = 0; t < GS; ++t) if (bs[(size_t) (pick * GS + t)].active) {
                            G.S = t + 1;
                            ++G.active;
                        }
                        for (int t = 0; t < G.S; ++t) {
                            BSlot& sl = bs[(size_t) (pick * GS + t)];
                            G.tok[t] = sl.active ? sl.x : 0;
                            G.pos[t] = sl.active ? sl.p : 0;
                            if (!sl.active) sl.cached = false;   // its pad row writes its state
                        }
                    }
                    G.inflight = true;
                    G.stage = 0;
                    if (batch_telemetry_enabled) G.started = Clock::now();
                    // (with the doorbells on their own thread, a failure it set for a window that left this stage is
                    // not cleared unseen)
                    if (conc_serving && drive.d.failed) {
                        print_response("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed");
                        return false;
                    }
                    drive.d.failed = false;
                    apply_pending(false);
                }
                PGroup& G = pg[(size_t) pick];
                strata::core::progress().busy.store(true);
                if (batch_telemetry_enabled) stage_in_flight.launch(k, Clock::now());
                if (k == 0) {   // the penalised slots' reply tails for the group's rows (the last stage draws them)
                    int grows[strata::kernels::kVerifyMaxT] = {};
                    for (int t = 0; t < G.S && t < (int) strata::kernels::kVerifyMaxT; ++t) grows[t] = pmtp ? G.rows[t] : pick * GS + t;
                    for (int t = 0; t < G.S && t < (int) strata::kernels::kVerifyMaxT; ++t) {
                        const BSlot& ps = bs[(size_t) grows[t]];
                        if (ps.pen_n <= 0 || (t > 0 && grows[t - 1] == grows[t])) continue;
                        const int64_t from = std::max<int64_t>(ps.reply0, (int64_t) ps.ids.size() - ps.pen_n);
                        ver.set_slot_history(grows[t], ps.ids.data() + from, (int64_t) ps.ids.size() - from);
                    }
                }
                const auto p4_l0 = p4.now();
                if (!(pmtp ? stage_verifier(k).batch_launch_rows(G.rows, G.S, G.hbase, G.tok, G.pos, err)
                           : stage_verifier(k).batch_launch(pick * GS, G.S, G.tok, G.pos, err))) {
                    print_response("ERR %s\n", err.c_str());
                    return false;
                }
                p4.launched(k, pick, p4_l0);
                stage_group[(size_t) k] = pick;
                if (conc_serving) {   // the doorbell thread may be waiting between windows: the launch above is seen
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    if (bell_waiting.load(std::memory_order_relaxed)) {
                        std::lock_guard<std::mutex> lk(bell_mu);
                        bell_cv.notify_one();
                    }
                }
            }
            if (drive.d.failed) {
                print_response("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed");
                return false;
            }
            if (pa_due() && !pipe_inflight()) {   // STRATA_PIPE_ADAPT_EVERY: the pipeline is empty - one round, as batch_step
                if (pmtp && !pmtp_settle()) return false;   // STRATA_PIPED_MTP: the queued commits and draft chains are done first
                const Clock::time_point ta = Clock::now();
                pa_last = pa_windows;
                if (!adapt()) {
                    print_response("ERR an adaptive refill failed\n");
                    return false;
                }
                ++pa_rounds;
                pa_swaps += (int64_t) pending.size();
                apply_pending(true);   // the copies land and the tables follow (the next launch plans from them)
                pa_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            }
            if (batch_telemetry_enabled) {
                const auto sample = stage_in_flight.sample(Clock::now(), !pipe_inflight() && !batch_on());
                if (sample) {
                    print_response("PSTAT %zu %lld", sample->in_flight_us.size(), (long long) sample->elapsed_us);
                    for (const int64_t busy : sample->in_flight_us) print_response(" %lld", (long long) busy);
                    print_response("\n");
                    std::fflush(stdout);
                }
            }
            // every slot done: the last commits have run before anything reads a slot (between windows nothing does)
            if (pmtp && !pipe_inflight() && !batch_on() && !pmtp_settle()) return false;
            if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: the timing line, as batch_step
                const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
                p4.span_end(wall);
                pc.flush(wall, pc_ex());
                char pa_txt[160] = "";
                if (pipe_adapt_every > 0)
                    std::snprintf(pa_txt, sizeof pa_txt, "; adaptive tier (STRATA_PIPE_ADAPT_EVERY): %lld rounds, %lld experts "
                                  "swapped in, %.0f ms", (long long) pa_rounds, (long long) pa_swaps, pa_ms);
                std::fprintf(stderr, "strata batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = "
                                     "%.1f rows/s (admissions included)%s", o.batch_groups, GS, (long long) bt_windows,
                             (long long) bt_rows, wall, 1000.0 * bt_rows / std::max(wall, 1e-9), pa_txt);
                if (pmtp)
                    std::fprintf(stderr, " (%lld of %lld MTP proposals kept)", (long long) bt_accepted, (long long) bt_tokens);
                std::fprintf(stderr, "\n");
                p4.flush();
                pa_rounds = pa_swaps = 0;
                pa_ms = 0;
                for (int k = 0; k < n_pipe; ++k) {
                    const std::string pr = stage_verifier(k).profile_report();
                    if (!pr.empty()) std::fprintf(stderr, "strata batch GPU stages, stage %d (ms/window):%s\n", k + 1, pr.c_str());
                }
                bt_windows = bt_rows = bt_tokens = bt_accepted = 0;
                // (not while a prompt is read in a request: its windows ran beside it, and the watchdog stays on for
                // the rest of the read, STRATA_PIPED_CONCURRENT; the request's end turns it off)
                if (!in_prompt_read) strata::core::progress().busy.store(false);
            }
            return true;
        };
        auto pipe_drain = [&]() -> bool {
            while (pipe_inflight())
                if (!pump(false)) return false;
            return pmtp_settle();
        };
        // BSTOP <slot> ends that slot at its next batch window (here, and in a prompt read that slots decode beside)
        auto stop_slot = [&](int b) {
            if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
        };
        // what the shared command loop (strata/common/serve/loop.hpp) asks of this engine
        strata::serve::EngineHooks hooks;
        hooks.is_busy = [&] { return batch_on() || (piped && pipe_inflight()); };
        hooks.step = [&] { return piped ? pump(true) : batch_step(); };
        hooks.before_line = [&](const std::string& l) {
            line_drain_ms = 0.0;
            strata::core::manifest::instance().flush_if_dirty("request");   // STRATA_MANIFEST: what the last request added (a no-op when off)
            // a request reads its prompt through every stage: the groups in flight finish first
            // (STRATA_PIPE_BYIELD_NODRAIN: not for a BYIELD, which is dropped)
            if (!piped || !(batch_on() || pipe_inflight()) || l.rfind("BSTOP ", 0) == 0 ||
                (pipe_byield_nodrain_env() && l.rfind("BYIELD", 0) == 0))
                return true;
            const int in_flight = (trace || pc.on()) ? (int) std::count_if(pg.begin(), pg.end(), [](const PGroup& G) {
                                              return G.inflight; }) : 0;
            const auto td = Clock::now();
            if (!pipe_drain()) return false;
            if (pc.on()) pc.drain(l, in_flight, std::chrono::duration<double, std::milli>(Clock::now() - td).count());
            if (trace) {   // STRATA_TRACE: what the drain before this line cost (BGEN: an admission's first phase)
                line_drain_ms = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                std::fprintf(stderr, "strata trace: drain for %.*s: %d groups in flight, %.1f ms\n",
                             (int) std::min(l.find(' '), l.size()), l.c_str(), in_flight, line_drain_ms);
                const bool timed = in_flight > 0 && l.rfind("BYIELD", 0) == 0;
                rows_wait = timed ? "the drain for BYIELD started" : nullptr;
                rows_wait_t = td;
                std::fflush(stderr);
            }
            return true;
        };
        hooks.stop_slot = stop_slot;
        hooks.on_late_yield = [&] {
            if (trace && piped)
                std::fprintf(stderr, "strata trace: BYIELD dropped, %d groups in flight\n",
                             (int) std::count_if(pg.begin(), pg.end(), [](const PGroup& G) { return G.inflight; }));
        };
        // the conversations the sessions hold and the parked ones, for the server's API monitor (token_listing.hpp):
        // between requests and between batch windows, changing nothing
        hooks.list_conversations = [&] {
            // a session's size for `cells` tokens: at --max-context the bytes it takes, at what it holds the bytes
            // a session sized for just those would take (its state and buffers, and their K/V)
            auto session_size = [&](int64_t cells) {
                uint64_t n = 0;
                for (const auto& [lo, hi] : session_layers)
                    n += strata::core::session_bytes(g, std::max<int64_t>(cells, 1), K, lo, hi);
                return n;
            };
            const uint64_t session_cap = session_size(o.max_context);
            std::vector<strata::core::HeldConversation> held;
            const bool main_held = live_ok && !live.empty();
            const size_t main_n = main_held ? live.size() : 0;
            held.push_back({"main", "vram", main_held ? "holding" : "empty", live.data(), main_n,
                            (uint64_t) o.max_context, session_cap, main_n ? session_size((int64_t) main_n) : 0});
            for (size_t b = 0; b < bs.size(); ++b) {
                const BSlot& sl = bs[b];
                const size_t n = sl.active || sl.partial || sl.cached ? sl.ids.size() : 0;
                held.push_back({"slot" + std::to_string(b), "vram",
                                sl.active ? "decoding" : sl.partial ? "reading" : sl.cached ? "holding" : "empty",
                                sl.ids.data(), n, (uint64_t) o.max_context, session_cap,
                                n ? session_size((int64_t) n) : 0});
            }
            conversations.for_each([&](uint64_t serial, const strata::core::SavedConversation& c) {
                held.push_back({"parked" + std::to_string(serial), "ram", "parked", c.live.ids.data(),
                                c.live.ids.size(), c.live.ids.size(), c.bytes(), c.bytes()});
            });
            return held;
        };
        // GET_MEMORY_USE (memory_report.hpp): each card's weights (the primary's pack arena; the native dense
        // projections and head), expert cache, sessions and drafters; the RAM's expert arena, streamed K/V and
        // conversation-cache budget
        hooks.describe_memory = [&] {
            strata::serve::MemoryReport r;
            for (size_t i = 0; i < session_layers.size(); ++i) {   // the primary card, then the stages' cards
                const int dev = i == 0 ? primary_device : stages[i - 1]->dev;
                const auto [lo, hi] = session_layers[i];
                strata::serve::Memory& m = r.gpu(dev);
                strata::core::read_gpu_memory(dev, m.name, m.total, m.free, m.unified);
                if (i == 0)
                    m.add("weights", pool_bytes + native_dense.allocated_bytes() + native_head.weight_bytes());
                else
                    m.add("weights", stages[i - 1]->dense.allocated_bytes() + stages[i - 1]->head.weight_bytes());
                m.add("experts", (uint64_t) (i == 0 ? xcache.bytes() : stages[i - 1]->cache.bytes()));
                m.add("conversations", (1 + bs.size()) * strata::core::session_bytes(g, o.max_context, K, lo, hi));
            }
            if (!o.mtp.empty()) {
                r.gpu(mtp.device()).add("drafter", mtp.vram_bytes());
                for (const auto& d : slot_mtp)
                    r.gpu(d->device()).add("drafter", d->vram_bytes());
            }
            strata::serve::Memory& ram = r.at("ram");
            ram.add("experts", o.mmap_experts ? src.resident_bytes() : strata::kernels::cpu::expert_layout().total);
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
                ram.add("conversations", strata::core::qsa_kv_host_bytes());
            if (conversations.enabled()) ram.paused_budget = (uint64_t) o.conversation_cache_mib << 20;
            r.settings = {{"context", o.max_context}, {"concurrent", std::max(1, o.batch)},
                          {"lookahead", o.spec >= 2 ? o.spec - 1 : 0}};
            return r;
        };
        strata::serve::Loop loop(input, hooks);
        while (loop.next(line)) {
            if (line.rfind("BCO ", 0) == 0) {   // BCO <slot> <id,id,...>: kept for the next BGEN
                CoReq cr;
                char* e1 = nullptr;
                cr.slot = (int) std::strtol(line.c_str() + 4, &e1, 10);
                for (const char* q = e1; q != nullptr && *q != '\0';) {
                    while (*q == ' ' || *q == ',') ++q;
                    if (*q == '\0') break;
                    char* e2 = nullptr;
                    cr.ids.push_back((int64_t) std::strtoll(q, &e2, 10));
                    if (e2 == q) { cr.ids.clear(); break; }
                    q = e2;
                }
                if (cr.slot >= 0 && cr.slot < (int) bs.size() && cr.ids.size() >= 3) co_pending.push_back(std::move(cr));
                continue;
            }
            if (line.rfind("VRAM", 0) == 0) {   // #533 (above): between requests, not a request
                std::string verr;
                if (batch_on()) verr = "VRAM: not while batch slots are decoding";
                else vram_command(line, verr);
                if (!verr.empty()) print_response("ERR %s\n", verr.c_str());
                std::fflush(stdout);
                continue;
            }
            // a command may touch the expert cache (a request lends slots, a K/V growth moves them, a session restore
            // quiesces it): a round of the asynchronous tier that the batch windows left in flight lands first
            if (!adapt_tick(true)) {
                print_response("ERR an adaptive refill failed\n");
                return 1;
            }
            admit_slot = -1;
            // BGEN <slot> <max_new> ... reads the request's prompt and first token as a GEN 1, then continues it in
            // that slot
            strata::serve::RequestLine req;
            const bool is_request = strata::serve::parse_request_line(line, req);
            if (is_request && req.batch) {
                if (bs.empty() || req.slot < 0 || req.slot >= (long) bs.size() || bs[(size_t) req.slot].active ||
                    req.max_new < 1) {
                    print_response("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
                    std::fflush(stdout);
                    continue;
                }
                admit_slot = (int) req.slot;
                admit_max_new = req.max_new;
                adm_drain_ms = line_drain_ms;
                co_set = std::move(co_pending);
                co_pending.clear();
                req.max_new = 1;
            }
            // #477: every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
            if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
                Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
                save_profile("periodic");
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() {
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress().busy.store(true);
                    strata::core::progress_at("request");
                }
                ~BusyScope() {
                    strata::core::progress().busy.store(false);
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress_at("idle");
                }
            } busy_scope;
            {   // #2065: said, never changed - the free VRAM now (the verify windows' captures and the batch layouts
                // eat into what startup reported), once per 128 MiB lower it gets
                size_t vf_b = 0, vt_b = 0;
                if (cudaMemGetInfo(&vf_b, &vt_b) == cudaSuccess) {
                    const int64_t vf = (int64_t) (vf_b >> 20);
                    g_req_vram_free_mib.store(vf);
                    static int64_t low_said_at = INT64_MAX;
                    static bool uma_said = false;
                    if (vf < kVramSafeMib && g_uma_pinned_complement.load()) {
                        // Round 0216: a note, once, no reserve advice (see g_uma_pinned_complement)
                        if (!uma_said) {
                            uma_said = true;
                            std::fprintf(stderr, "strata serve: Only %lld MiB of graphics memory reported free at the start of this request. "
                                                 "On this PC the figure reads low because the graphics chip shares the system memory. "
                                                 "If decode drops to a few tok/s, lower --max-context.\n", (long long) vf);
                        }
                    } else
                    if (vf < kVramSafeMib && vf + 128 <= low_said_at) {
                        low_said_at = vf;
                        std::fprintf(stderr, "strata serve: WARNING: only %lld MiB of VRAM free at the start of this request "
                                             "(below %lld MiB): when the card is this full the driver can page GPU memory "
                                             "to system RAM, decode falls to a few tok/s and a window may never finish "
                                             "(#2065). Add --vram-reserve-mib %lld to the config's args, or lower "
                                             "--max-context / --batch\n", (long long) vf, (long long) kVramSafeMib,
                                     (long long) ((o.vram_reserve_mib + (kVramSafeMib - vf) + 63) / 64 * 64));
                    }
                }
            }
            input.clear_stop();   // a STOP that arrived between requests is stale
            err.clear();
            // Disk sessions: SAVE <path> | RESTORE <path>, between requests (the path runs to the end of the line,
            // UTF-8).  The file holds what a parked conversation holds (conversation_file.hpp).  Answers: SAVED /
            // RESTORED <tokens> <bytes> <ms>; SERR <invalid|storage|memory|io> <published 0|1> <reason> for a refusal
            // or failure that leaves the engine in step (the live session as it was, unless published=1 says the new
            // file already replaced the old one); FATAL <reason> (and the engine exits) for a restore transfer that
            // failed after the device state was touched.  On the way: SESSION <done> <total> after every block of
            // the file moved (at most 16 MiB), and SWAIT <phase> <seconds> before a step that blocks in one call
            // (fingerprint file, state capture, file flush, rename + folder flush, validation, device transfer):
            // that step is allowed <seconds> (session_phase_limit_s) by the watchdog and by the server, no more.
            // (A verifier commit that fails before the command still answers ERR and exits, as for a request.)
            if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
                const bool save = line[0] == 'S';
                const std::string path = line.substr(save ? 5 : 8);
                const auto t0 = Clock::now();
                auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
                auto refuse = [&](const std::string& why, strata::core::SessionError kind = strata::core::SessionError::invalid,
                                  bool published = false) {
                    std::fprintf(stderr, "strata serve: session %s %s: %s\n", save ? "save" : "restore", path.c_str(),
                                 why.c_str());
                    std::string one = why;   // one protocol line
                    for (char& c : one) if (c == '\n' || c == '\r') c = ' ';
                    print_response(
                        "SERR %s %d %s\n", strata::core::session_error_name(kind), published ? 1 : 0, one.c_str());
                    std::fflush(stdout);
                    err.clear();
                };
                auto moving = [&](uint64_t done, uint64_t total) {
                    strata::core::progress_at(save ? "writing a session file, MiB" : "reading a session file, MiB",
                                              (int64_t) (done >> 20));
                    strata::core::progress_allow(0);   // a block moved: the ordinary watchdog limit again, and a beat
                    print_response("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
                    std::fflush(stdout);
                };
                // a step that blocks in one call: its explicit allowance, on both sides, announced before it starts
                auto blocking = [&](const char* phase, uint64_t bytes) {
                    const int64_t s = strata::core::session_phase_limit_s(bytes);
                    strata::core::progress_at(phase);
                    strata::core::progress_allow(s);
                    print_response("SWAIT %s %lld\n", phase, (long long) s);
                    std::fflush(stdout);
                };
                strata::core::progress_at(save ? "saving a session" : "restoring a session");
                // the background copy of the last turn checkpoint (STRATA_CKPT_ASYNC) lands before anything below copies, frees
                // or writes the checkpoints (a join inside the session file writer would match only the copy of a checkpoint)
                strata::core::conversation_checkpoint_join(nullptr);
                strata::core::conversation_park_join();   // (STRATA_PARK_ASYNC: the last park's K/V)
                if (path.empty()) { refuse("missing path"); continue; }
                if (!stages.empty() || multi_gpu) { refuse("session files do not support --layer-split"); continue; }
                if (o.peer_device >= 1) { refuse("session files do not support --peer-device"); continue; }
                if (o.batch > 0) { refuse("session files do not support --batch (parallel requests)"); continue; }
                if (o.prompt_cache <= 0) { refuse("session files need --prompt-cache > 0"); continue; }
                if (!ver.wait_commit(err)) {
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                strata::core::SessionFileIdentity id;
                try {
                    if (!session_identity(id, err, [&] { blocking("fingerprint", 2u << 20); })) {
                        refuse(err, strata::core::SessionError::io);
                        continue;
                    }
                } catch (const std::bad_alloc&) {
                    refuse("session identity: out of memory", strata::core::SessionError::memory);
                    continue;
                } catch (const std::exception& e) {
                    refuse(std::string("session identity: ") + e.what(), strata::core::SessionError::io);
                    continue;
                }
                if (save) {
                    if (!live_ok || live.empty()) { refuse("no complete session to save"); continue; }
                    size_t bytes = 0, kept = 0;
                    double capture_ms = 0;
                    try {
                        // RAM PREFLIGHT for the save (not a reservation), BEFORE anything is copied: the deepest
                        // checkpoint is chosen by reference, and the copies it will take (the list below and the one
                        // in `meta`), the copied live running state with its tokens and images, the K/V source
                        // directory and the 16 MiB write buffer are asked for above the parking floor.  Only the
                        // deepest checkpoint goes to disk (the next turn's resume point); the K/V is streamed from
                        // the authoritative pools straight into the file, without a full host capture.
                        strata::core::SessionSaveLive sl;
                        {
                            strata::core::ConversationStateSizes z;
                            std::string why;
                            uint64_t state = UINT64_MAX;   // unknown sizes: the preflight refuses rather than guesses
                            if (strata::core::conversation_session_sizes(g, ss, z, why)) {
                                const uint64_t q = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0);
                                const uint64_t per = (uint64_t) z.tail + z.dead + z.block_pos;
                                if (q == 0 || per <= (UINT64_MAX - z.gdn - z.ple) / q) state = z.gdn + z.ple + q * per;
                            }
                            sl.state_bytes = state;
                            sl.tokens = live.size();
                            sl.images = live_imgs.size();
                            sl.kv_layers = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0) + 1;
                        }
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        if (o.session_save_reclaim) {
                            const auto freed = strata::core::session_save_reclaim(
                                checks, conversations, sl, floor, strata::core::available_host_bytes);
                            if (freed.bytes || freed.checkpoints || freed.parked) {
                                // A removed tail must not mark a later checkpoint at this length as a tail.
                                if (tail_ckpt_len >= 0 && std::none_of(checks.begin(), checks.end(), [&](const auto& c) {
                                        return int64_t(c.ids.size()) == tail_ckpt_len;
                                    })) tail_ckpt_len = -1;
                                std::fprintf(stderr, "strata serve: save reclaimed %zu checkpoints, %zu parked, "
                                                     "%zu host bytes; checking RAM again\n",
                                             freed.checkpoints, freed.parked, freed.bytes);
                            }
                        }
                        auto admit = [&](uint64_t need, std::string& why) {
                            const auto mem = strata::core::sample_host_memory();
                            const std::optional<uint64_t> avail = mem ? std::optional<uint64_t>(mem->available) : std::nullopt;
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to save the session (" +
                                  (need == UINT64_MAX ? std::string("unknown") : std::to_string(need >> 20)) +
                                  " MiB plus a floor of " + std::to_string((long long) o.conversation_cache_min_free_mib) +
                                  " MiB needed, " + ram_note(mem) + ")";
                            return false;
                        };
                        std::vector<ConvCheckpoint> disk_checks;
                        std::string why;
                        if (!strata::core::session_save_checkpoints(checks, sl, admit, disk_checks, why)) {
                            refuse(why, strata::core::SessionError::memory);
                            continue;
                        }
                        kept = disk_checks.size();
                        const strata::core::ConversationView view{live, live_imgs, disk_checks, cvec_cached};
                        strata::core::SavedConversation meta;
                        std::vector<strata::core::SessionKvSource> sources;
                        // the live running state comes off the device in one synchronous copy
                        blocking("capture", sl.state_bytes == UINT64_MAX ? 0 : sl.state_bytes);
                        if (!strata::core::conversation_snapshot_sources(meta, sources, view, ss, g, mtp.kv_state(),
                                                                         err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        capture_ms = ms();
                        strata::core::SessionWriteOptions wo;
                        wo.min_free_bytes = (uint64_t) o.session_min_free_mib << 20;
                        wo.progress = moving;
                        wo.phase = blocking;
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_write(path, meta, sources, id, bytes, err, wo, &st)) {
                            refuse(err, st.error, st.published);
                            continue;
                        }
                        if (st.dir_flush_unsupported)
                            std::fprintf(stderr, "strata serve: session saved to %s; this filesystem cannot flush a "
                                                 "folder, so the new name is not flushed\n", path.c_str());
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to save the session", strata::core::SessionError::memory);
                        continue;
                    }
                    std::fprintf(stderr, "strata serve: session saved %zu tokens, %zu of %zu checkpoints, %zu bytes to %s "
                                 "in %.1f ms (state %.1f ms)\n", live.size(), kept, checks.size(), bytes,
                                 path.c_str(), ms(), capture_ms);
                    print_response("SAVED %zu %zu %.1f\n", live.size(), bytes, ms());
                } else {
                    strata::core::SavedConversation image;
                    size_t bytes = 0;
                    try {
                        // bounds this session can ever restore (geometry, layer range, tokens, checkpoints, every
                        // state and K/V part), checked as the file is parsed and before each array is allocated; the
                        // file size follows from them.  The RAM the parse needs at its peak is asked first.
                        strata::core::SessionReadLimits limits;
                        limits.progress = moving;
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        limits.admit = [floor, &o](uint64_t need, std::string& why) {
                            const auto mem = strata::core::sample_host_memory();
                            const std::optional<uint64_t> avail = mem ? std::optional<uint64_t>(mem->available) : std::nullopt;
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to read it (" + std::to_string(need >> 20) + " MiB plus a floor of " +
                                  std::to_string((long long) o.conversation_cache_min_free_mib) + " MiB needed, " +
                                  ram_note(mem) + ")";
                            return false;
                        };
                        if (!strata::core::conversation_session_read_limits(
                                limits, ss, g, mtp.kv_state(), (uint64_t) o.max_context,
                                (uint64_t) std::max(o.prompt_cache, 1), err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_read(path, id, image, bytes, err, limits, &st)) {
                            refuse(err, st.error);
                            continue;
                        }
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to read the session", strata::core::SessionError::memory);
                        continue;
                    }
                    const double read_ms = ms();
                    // the whole image against this engine, still without any device write
                    blocking("validate", bytes);
                    if (!strata::core::conversation_snapshot_validate(image, ss, g, mtp.kv_state(), err)) {
                        refuse(err);
                        continue;
                    }
                    // the elastic K/V (--kv-grow) maps only the cells it has grown to: make room for the file's cells
                    if (!kvg_ensure((int64_t) image.live.ids.size() + 256, adapt_quiesce)) {
                        refuse("the K/V cannot grow to the saved conversation: no VRAM is left", strata::core::SessionError::memory);
                        continue;
                    }
                    conversations.take_reuse();   // retained K/V described the outgoing session
                    live_ok = false;
                    // host -> device in synchronous copies of the whole state: one bounded allowance
                    blocking("transfer", bytes);
                    if (strata::core::conversation_snapshot_restore(image, ss, g, mtp.kv_state(), err) !=
                        strata::core::ConversationRestore::restored) {
                        // validated above: a failure here is a transfer failure, after device writes began - never
                        // decode from a partial state; the server starts the engine again
                        std::fprintf(stderr, "strata serve: session restore %s: transfer failed: %s\n", path.c_str(),
                                     err.c_str());
                        print_response(
                            "FATAL restoring the session file failed after the device state was changed: %s\n",
                                    err.c_str());
                        std::fflush(stdout);
                        return 1;
                    }
                    live = std::move(image.live.ids);
                    live_imgs = std::move(image.live.imgs);
                    checks = std::move(image.checkpoints);
                    for (const ConvCheckpoint& c : checks) check_clock = std::max(check_clock, c.used);
                    cvec_cached = image.cvec;
                    live_ok = true;
                    if (nan_flush) {   // #879: the file replaced the poisoned session; the parked ones go (no slots here)
                        nan_flush = false;
                        const size_t n_parked = conversations.clear();
                        std::fprintf(stderr, "strata serve: after non-finite logits (#879): the restored session "
                                             "replaces the live one; dropped %zu parked conversation%s\n", n_parked,
                                     n_parked == 1 ? "" : "s");
                    }
                    std::fprintf(stderr, "strata serve: session restored %zu tokens, %zu checkpoints, %zu bytes from %s "
                                 "in %.1f ms (read+check %.1f ms)\n", live.size(), checks.size(), bytes, path.c_str(),
                                 ms(), read_ms);
                    print_response("RESTORED %zu %zu %.1f\n", live.size(), bytes, ms());
                }
                std::fflush(stdout);
                continue;
            }
            if (is_request) model.refresh_file_tier();
            if (!is_request) {
                print_response("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n");
                continue;
            }
            const bool geni = req.pictures;
            const long long max_new = req.max_new;
            // optional keys between max_new and the ids: the sampling keys (strata::serve::Sampling; text requests
            // only) and the ones below.  Absent keys keep today's behavior: greedy, no penalties.
            strata::serve::Sampling smp;
            long long req_pen_from = -1;   // pen_from=N: this request continues one that began at token N (the last token of the original prompt): the penalty window starts there
            std::string req_bias_text;
            bool req_bias_present = false;
            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
            // logprobs=K (0..20): after each "T id" an "LP logprob id:logprob ..." line.
            // -1 (absent) = no LP lines, so a server that never asks sees today's output.
            int req_logprobs = -1;
            // allowed=a,b,c: with logprobs, print only these token ids (renormalized server-side).
            std::vector<int32_t> req_allowed;
            // ckpt=0: a one-shot call whose turn no later request extends.  No checkpoint at its last turn boundary
            // (so no split there) nor every --prompt-cache-every tokens, and its session is neither continued nor
            // parked after it.  It still resumes from a checkpoint it matches, and still saves the system-prompt root
            // when that reaches --prompt-cache-root.  Absent = checkpointed as before.
            int req_ckpt = 1;
            // prefill=1 (with max_new 0): split serving.  Read the prompt as a request would, up to the place where the first
            // verify window starts (the session holds n - 1 tokens, exactly what a next turn that continues it needs), and
            // stop: no token is made.  A SAVE after it is the hand-over to another engine (docs/SPLIT_SERVE.md).
            bool req_prefill_only = false;
            // pin=N: the first N prompt tokens are a shared read-only prefix (a long document that many short queries
            // follow).  The prompt is read in two parts at N, the checkpoint there is PINNED (retention never evicts it,
            // a parked conversation holding it stays parked) and a later request that resumes from it does not park the
            // branch it leaves: N queries cost one prefix, not N.  Absent = as before.
            int64_t req_pin = 0;
            // tuning keys (setup's calibration measures settings without restarting the engine): the PCIe share of
            // the missed experts and the draft-probability floor, for this request only
            double req_pcie_frac = o.pcie_frac, req_spec_min_p = o.spec_min_p;
            std::string req_draft;   // draft=mtp|dflash2 (--draft auto): which drafter this request uses
            bool req_min_p_set = false;
            double req_pipe = -1.0;   // --prefill-pipe for this request (setup's calibration of a layer split)
            int req_aux_cpus = -1;   // aux_cpus=0|1: --aux-cpus for this request (-1 = the start-up value)
            // #2228: img_keys=<hex>,<hex>,... (GENI): each picture's key from the server (its normalized bytes' hash),
            // in prompt order.  With them the conversation cache matches a picture by its key and grid, not by its
            // embedding floats, so a picture encoded again with other bits still matches.  Absent: the embeddings' hash.
            std::vector<uint64_t> req_img_keys;
            bool req_pcie_set = false;   // the request names its own share (the measured one holds meanwhile)
            for (const auto& [key, val] : req.keys) {   // (#75: GENI takes the same keys; image requests were greedy)
                if (strata::serve::read_sampling_key(key, val, smp)) continue;
                const float fv = std::strtof(val.c_str(), nullptr);
                if (key == "logit_bias") { req_bias_text = val; req_bias_present = true; }
                else if (key == "pen_from") req_pen_from = std::atoll(val.c_str());
                else if (key == "cvec") req_cvec = std::atoi(val.c_str());
                else if (key == "ckpt") req_ckpt = std::atoi(val.c_str()) != 0;
                else if (key == "prefill") req_prefill_only = std::atoi(val.c_str()) != 0;
                else if (key == "pin") req_pin = std::max<long long>(0, std::atoll(val.c_str()));
                else if (key == "logprobs") req_logprobs = std::atoi(val.c_str());
                else if (key == "allowed") {
                    size_t p = 0;
                    while (p < val.size()) {
                        const size_t c = val.find(",", p);
                        const std::string one = val.substr(p, c == std::string::npos ? std::string::npos : c - p);
                        if (!one.empty()) req_allowed.push_back((int32_t) std::atoi(one.c_str()));
                        if (c == std::string::npos) break;
                        p = c + 1;
                    }
                }
                else if (key == "pcie_frac") { req_pcie_frac = std::clamp((double) fv, 0.0, 1.0); req_pcie_set = true; }
                else if (key == "spec_min_p") { req_spec_min_p = std::clamp((double) fv, 0.0, 1.0); req_min_p_set = true; }
                else if (key == "draft") req_draft = val;
                else if (key == "pipe") req_pipe = std::max(0.0, (double) fv);
                else if (key == "aux_cpus") req_aux_cpus = std::atoi(val.c_str()) != 0 ? 1 : 0;
                else if (key == "img_keys") req_img_keys = strata::serve::parse_hex_list(val);
                // unknown keys are skipped
            }
            strata::prefill::Prefill::set_pipe_request(req_pipe);   // every request sets it: none carries over
            bool req_df2 = use_df2;   // --draft auto: this request's drafter, decided once the prompt is parsed (below)
            std::vector<float> req_bias;
            std::string bias_error;
            if (req_bias_present && !strata::core::parse_logit_bias(req_bias_text, (int) n_vocab, req_bias, bias_error)) {
                print_response("ERR logit_bias: %s\n", bias_error.c_str());
                continue;
            }
            const std::string& emb_path = req.file;
            std::vector<int64_t> ids;
            std::string pe;
            if ((max_new < 1 && !(max_new == 0 && req_prefill_only)) || (req_prefill_only && max_new != 0) ||
                (geni && emb_path.empty()) || !parse_i64_list(req.ids.c_str(), ids, pe)) {
                print_response("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            // --draft auto: this request's drafter.  Without the DFlash2 drafter loaded every request uses the MTP, as before.
            uint64_t bandit_key = 0;
            if (use_df2 && o.draft_auto) {
                const std::string& d = !req_draft.empty() ? req_draft : o.draft_rule;
                if (d == "mtp") req_df2 = false;
                else if (d == "dflash2" || d == "df2") req_df2 = true;
                else if (d == "learn") {   // what this kind of request (its first prompt tokens: the system prompt of the app) did best so far, with some exploring
                    bandit_key = 1469598103934665603ull;
                    for (int64_t i = 1; i < std::min<int64_t>(n, 33); ++i) bandit_key = (bandit_key ^ (uint64_t) ids[(size_t) i]) * 1099511628211ull;
                    DraftBandit& b = bandit[bandit_key];
                    const int64_t r = b.reqs++;
                    if (b.n[1] < 2) req_df2 = true;
                    else if (b.n[0] < 2) req_df2 = false;
                    else {
                        const bool df_better = b.tps[1] >= 0.99 * b.tps[0];   // DFlash2 unless the MTP is clearly ahead
                        const int every = std::fabs(b.tps[1] / b.tps[0] - 1.0) < 0.04 ? 6 : 12;   // the other one is tried now and then
                        req_df2 = (r % every == every - 1) ? !df_better : df_better;
                    }
                }
                if (!req_df2 && !req_min_p_set && o.draft_mtp_min_p >= 0.0) req_spec_min_p = o.draft_mtp_min_p;
                if (req_df2 != policy_is_df2) { std::swap(wplan.policy, policy_idle); policy_is_df2 = req_df2; }
                S_mtp = req_df2 ? S_mtp_wide : S_mtp_narrow;
                wplan.set_mtp_rows(S_mtp);
                S_look = req_df2 ? S : std::min(S, S_mtp_narrow);
                if (use_mtp) mtp.set_max_drafts(S_mtp - 1);
            }
            if (req_prefill_only && (geni || o.batch > 0 || n < 2)) {
                print_response("ERR prefill=1 needs a text prompt of at least 2 tokens and no --batch\n");
                continue;
            }
            req_imgs.clear();
            if (geni && !o.vision) { print_response("ERR this engine was started without --vision\n"); continue; }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    strata::qwen38::read_prompt_images(emb_path, ids, g.n_embd, req_img_keys, mrope_host,
                                                       img_rows, row_ptr, req_imgs, ve);
                }
                tr("positions built", (long long) img_rows.size());
                cudaDeviceSynchronize();
                tr("device idle");
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool {
                    bool ok = cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                         cudaMemcpyHostToDevice) == cudaSuccess;
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        cudaDeviceSynchronize();
                        ok = ok && cudaMemcpy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                              cudaMemcpyHostToDevice) == cudaSuccess;
                    }
                    return ok;
                };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    print_response("ERR %s\n", ve.c_str());
                    std::fflush(stdout);
                    continue;
                }
                mrope_identity = !geni;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                print_response("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            (long long) max_new, (long long) o.max_context);
                continue;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { print_response("ERR a token id is outside the vocabulary\n"); continue; }
            std::array<int64_t, 3> remote_before{};
            std::array<int64_t, 3> launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            // ms_begin/ms_wait are cumulative since boot; the log line used to print them next to per-request deltas,
            // so the host time read as if it belonged to this request.  Take deltas here like every other column.
            std::array<double, 3> begin_before{}, wait_before{};
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
                begin_before[(size_t) r] = remote_experts[(size_t) r].ms_begin();
                wait_before[(size_t) r] = remote_experts[(size_t) r].ms_wait();
            }
            cur = ids;
            const Clock::time_point r0 = Clock::now();
            tr("r0");
            strata::core::conversation_checkpoint_join(nullptr);   // STRATA_CKPT_ASYNC: the last save's copy has landed
            const int64_t streamed0 = sp.chain_experts_streamed();   // the DONE line's experts_streamed= is the difference
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                for (int64_t i = 0; i < L; ++i)
                    if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            const bool want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;
            // the last request's final commit may still be running on the verifier's stream (set_commit_async):
            // everything below reads, restores or zeroes the session from other streams and the host (the end of the
            // last request waited already; this covers a request that ended on an error path)
            if (!ver.wait_commit(err)) {
                print_response("ERR %s\n", err.c_str());
                return 1;
            }
            // STRATA_MTP_ROOM_MIN: this request points the prompt path's ring at the drafter's experts, if it reads
            // enough new tokens and the room applies here.  The open is below, once `read_from` is known: the
            // threshold counts the tokens the request reads, not the whole prompt (a reused conversation prefix buys
            // nothing from the room).  Nothing is laid out again - the room's slots are the drafter's own bytes
            // (Prefill::set_ring_room_base) - and `room_close` below gives the decode its contents back before the
            // first window.  A room left open by an earlier request (a cancelled prompt) is closed first.
            bool mtp_room_open = mtp_room_slots > 0 && room_path != nullptr && !mtp.experts_resident();
            auto head_restore = [&]() -> int {
                if (!o.prefill_head_alias || native_head.loaded()) return 0;
                sp.reset();
                const auto head_t0 = Clock::now();
                if (!native_head.load(o.native_head_shards, g.n_embd, n_vocab, err)) {
                    print_response("ERR restoring the native output head failed: %s\n", err.c_str());
                    return 1;
                }
                // the verify windows' captured graphs read the head at the address of their capture
                if (prefill_head_alias_at == nullptr) prefill_head_alias_at = native_head.weights();
                if (native_head.weights() != prefill_head_alias_at) {
                    print_response("ERR --prefill-head-alias: the output head was reloaded at another address, so the "
                                "captured verify graphs would read freed memory\n");
                    return 1;
                }
                std::fprintf(stderr, "strata serve: --prefill-head-alias: output head restored in %.1f ms before decode\n",
                             std::chrono::duration<double, std::milli>(Clock::now() - head_t0).count());
                return 0;
            };
            auto room_close = [&](const char* why) -> int {
                const auto t0 = Clock::now();
                if (room_path != nullptr) {
                    // the ring goes back to the rule's count FIRST: nothing after this can stage into the lender's
                    // bytes, and the streams are idle before they are written over (a prompt pass reads them)
                    if (!room_path->idle(err)) { print_response("ERR %s\n", err.c_str()); return 1; }
                    room_path->set_ring_room_active(false);
                }
                if (!mtp.experts_reclaim(err)) { print_response("ERR %s\n", err.c_str()); return 1; }
                mtp_room_open = false;
                std::fprintf(stderr, "strata serve: prompt room (%s): the drafter's experts are the decode's again in "
                                     "%.1f ms\n", why,
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                return 0;
            };
            if (mtp_room_open && room_close("left open") != 0) return 1;
            if (o.prefill_head_alias && native_head.loaded()) {
                const auto head_t0 = Clock::now();
                native_head.unload();
                sp.reset();
                if (init_prompt_paths() != 0) {
                    print_response("ERR rebuilding the head-alias prompt path failed: %s\n", err.c_str());
                    return 1;
                }
                arm_prompt_helpers();   // the rebuild above dropped them (they live inside the layout)
                std::fprintf(stderr, "strata serve: --prefill-head-alias: prompt workspace rebuilt in %.1f ms\n",
                             std::chrono::duration<double, std::milli>(Clock::now() - head_t0).count());
            }
            auto kv_quiesce = adapt_quiesce;
            if (kvg.on) {   // the elastic K/V: this prompt's cells (it gives back what it does not need below)
                // a failed growth may leave the tier half-changed: the engine stops (as for any CUDA failure)
                if (!kvg_ensure(n + 256, kv_quiesce)) {
                    print_response("ERR the K/V cannot grow to this prompt: no VRAM is left\n");
                    std::fflush(stdout);
                    return 1;
                }
            }
            nf_hit = false;
            // #879: an earlier window met non-finite logits.  The poison sits in a conversation's K/V and recurrent
            // state, so nothing that may hold it is continued: the live session, every checkpoint, every parked
            // conversation and every idle batch slot go, and this prompt is read from token 0 (resume == 0 below)
            if (nan_flush) {
                nan_flush = false;
                if (nf_prompt_windows) {
                    nf_prompt_windows = false;
                    short_read_off_once = o.short_read > 0;
                }
                const size_t n_checks = checks.size();
                live_ok = false;
                checks.clear();
                tail_ckpt_len = -1;
                const size_t n_parked = conversations.clear();
                size_t n_slots = 0, n_running = 0;
                for (BSlot& sl : bs)
                    if (sl.active) {   // a running slot finishes its reply, but is not kept as a cache after it
                        sl.tainted = true;
                        ++n_running;
                        // STRATA_SLOT_CKPT_MOVE reads a running slot's checkpoints (and a finished one's: lent_slot):
                        // they predate the flush, so they go now
                        if (slot_ckpt_move_env()) {
                            if (ckpt_pool.on())
                                for (ConvCheckpoint& c : sl.checks) ckpt_pool.give(std::move(c));
                            sl.checks.clear();
                        }
                    } else if (sl.cached) {
                        sl.cached = false;
                        sl.partial = false;
                        sl.checks.clear();
                        ++n_slots;
                    }
                std::fprintf(stderr, "strata serve: after non-finite logits (#879): dropped the live session, %zu "
                                     "checkpoint%s, %zu parked conversation%s and %zu idle slot%s (%zu running slot%s "
                                     "not kept after their reply); reading this prompt from token 0\n", n_checks,
                             n_checks == 1 ? "" : "s", n_parked, n_parked == 1 ? "" : "s", n_slots,
                             n_slots == 1 ? "" : "s", n_running, n_running == 1 ? "" : "s");
            }
            int64_t resume = 0;
            bool from_live = false;
            if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // STRATA_SLOT_CKPT_MOVE: a turn checkpoint a batch admission moved into its slot is still one of the main
            // session's while the main session holds its tokens (the cells below it are the ones it was taken over):
            // it is restored there as one of `checks` would be (lent_slot: the slot that keeps it).  moved_in: the length
            // of a slot's checkpoint this request moved into `checks` (from the slot it is admitted to), -1: none
            int lent_slot = -1;
            int64_t moved_in = -1;
            if (slot_ckpt_move_env() && o.prompt_cache > 0 && want_cvec == cvec_cached && live_ok)
                for (int b = 0; b < (int) bs.size(); ++b) {
                    if (bs[(size_t) b].cvec != cvec_cached || bs[(size_t) b].tainted) continue;   // (#879)
                    for (const ConvCheckpoint& c : bs[(size_t) b].checks) {
                        const int64_t L = (int64_t) c.ids.size();
                        if (L > resume && L <= (int64_t) live.size() && starts_with(c.ids, c.imgs) &&
                            std::equal(c.ids.begin(), c.ids.end(), live.begin()) && imgs_below(live_imgs, L) == c.imgs) {
                            resume = L;
                            from_live = false;
                            lent_slot = b;
                        }
                    }
                }
            // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a
            // source too - its sessions are copied back below, so only the new part is read.  STRATA_SLOT_CKPT_MOVE: an
            // active slot's checkpoints too (its windows write its cells from p on, so the ones below a checkpoint still
            // hold its prompt; the groups have finished before a request is read)
            int slot_source = -1;
            int64_t slot_tokens = 0;
            bool resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
            // --prefill-pipe: the plan of the segment a read goes on in (a BYIELD park, a periodic checkpoint), and
            // where it goes on; read_part reads the rest of that segment in its chunks
            ConvCheckpoint::PromptPlan resume_plan;
            int64_t resume_plan_at = -1, resume_plan_end = -1;   // where it goes on, and the segment end it needs
            const ConvCheckpoint* slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
            if (o.prompt_cache > 0 && req_imgs.empty())
                for (int b = 0; b < (int) bs.size(); ++b) {
                    const BSlot& sl = bs[(size_t) b];
                    const bool idle_source = !sl.active && sl.cached;
                    // (an active slot only where the drafter's rows come along or are not drafted from: the copy
                    // takes the main layers' cells, so the main drafter would keep another conversation's below it)
                    const bool active_source = sl.active && !sl.tainted && slot_ckpt_move_env() &&
                                               (!use_mtp || batch_mtp || admit_slot >= 0);
                    if (sl.cvec != want_cvec || !(idle_source || active_source)) continue;
                    if (idle_source && (int64_t) sl.ids.size() > std::max(resume, slot_tokens) && starts_with(sl.ids, {})) {
                        slot_source = b;
                        slot_tokens = (int64_t) sl.ids.size();
                        slot_ck = nullptr;
                    }
                    for (const ConvCheckpoint& c : sl.checks)
                        if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                            slot_source = b;
                            slot_tokens = (int64_t) c.ids.size();
                            slot_ck = &c;
                        }
                }
            // #2228: a checkpoint or the live state whose TOKENS this prompt starts with, deeper than what is reused, but
            // whose pictures differ (a picture is its grid and the hash of its embeddings: an image encoded again with
            // other bits - another encoder start, a CPU encoder that is not bit-reproducible - no longer matches).  The
            // reuse then falls back to a checkpoint before that picture; said once per request, so the log tells this
            // apart from a client that rewrote its history.
            if (o.prompt_cache > 0 && !req_imgs.empty()) {
                int64_t deep = 0;
                const std::vector<ImgKey>* deep_imgs = nullptr;
                auto tokens_match = [&](const std::vector<int32_t>& pre) {
                    const int64_t L = (int64_t) pre.size();
                    if (L < 1 || L > n - 1) return false;
                    for (int64_t i = 0; i < L; ++i)
                        if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                    return true;
                };
                auto consider = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) {
                    if ((int64_t) pre.size() > std::max(deep, resume) && tokens_match(pre) &&
                        imgs_below(req_imgs, (int64_t) pre.size()) != pre_imgs) {
                        deep = (int64_t) pre.size();
                        deep_imgs = &pre_imgs;
                    }
                };
                if (live_ok) consider(live, live_imgs);
                for (const ConvCheckpoint& c : checks) consider(c.ids, c.imgs);
                if (deep_imgs != nullptr) {
                    const std::vector<ImgKey> now = imgs_below(req_imgs, deep);
                    size_t k = 0;
                    while (k < now.size() && k < deep_imgs->size() && now[k] == (*deep_imgs)[k]) ++k;
                    std::fprintf(stderr, "strata serve: conversation cache: the state at %lld tokens has this prompt's "
                                         "tokens but picture %zu (at token %lld) is not the same as when it was saved "
                                         "(its embeddings changed): it cannot be reused (#2228)\n", (long long) deep,
                                 k + 1, (long long) (k < now.size() ? now[k].start : -1));
                }
            }
            const auto parked = conversations.best(ids, req_imgs, want_cvec);
            std::optional<strata::core::SavedConversation> incoming;
            // A checkpoint inside ANOTHER parked conversation (a sibling subagent's system prompt and tools, say) is
            // borrowed: its K/V up to there and the checkpoint come over, and that conversation stays parked for its
            // own next turn (ConversationCache::borrows).  Everything else is taken whole, as before.
            // (a layer split restores whole stage images only, so it never borrows; a batch slot that holds a longer
            // start of this prompt wins as before)
            bool borrow = stages.empty() && parked.tokens > std::max(resume, slot_tokens) &&
                          conversations.borrows(parked);
            if (parked.tokens > std::max(resume, slot_tokens) && !borrow)
                incoming.emplace(conversations.take(parked.index));
            if (incoming || borrow) slot_source = -1;
            // Reject the entire image before parking/overwriting the outgoing
            // state. Invalid entries can safely fall back to its existing prefix.
            if (incoming && incoming->stage_images.size() != stages.size()) {
                std::fprintf(stderr, "strata serve: conversation cache: discard snapshot (stage count)\n");
                incoming.reset();
            }
            if (incoming) for (size_t i = 0; i < stages.size(); ++i) {
                const strata::core::OnDevice on(stages[i]->dev);
                if (!strata::core::conversation_snapshot_validate(incoming->stage_images[i], stages[i]->ss, g,
                        use_mtp && !mtp_first && i + 1 == stages.size() ? &mtp.kv_state() : nullptr, err)) {   // the draft: the last stage's
                    std::fprintf(stderr, "strata serve: conversation cache: discard invalid stage snapshot (%s)\n",
                                 err.c_str());
                    incoming.reset();
                    err.clear();
                    break;
                }
            }
            if (incoming && !strata::core::conversation_snapshot_validate(*incoming, ss, g,
                    use_mtp && (stages.empty() || mtp_first) ? &mtp.kv_state() : nullptr, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: discard invalid snapshot (%s)\n", err.c_str());
                incoming.reset();
                err.clear();
            }
            // #2409: a restore writes into buffers that exist already. The one thing it can still allocate on a card is
            // this thread's private copy stream for it (a pooled one is reused), and an out-of-memory there is fatal
            // for the process (a failed restore must not be decoded from). So a card that would have to create one and
            // has less than kRestoreFirstUseMib free gets no restore: the snapshot goes back to the cache and this
            // prompt is read again. A card that already has its stream is never refused, however full it is.
            if (incoming) {
                std::vector<int> restore_devs{-1};
                for (const auto& stp : stages) restore_devs.push_back(stp->dev);
                std::vector<uint64_t> restore_free;
                std::vector<uint8_t> restore_ready;
                int tight = -1;
                bool sampled = true;
                for (const int rd : restore_devs) {
                    const strata::core::OnDevice on(rd);
                    int cur = 0;
                    size_t fb = 0, tb = 0;
                    if (cudaGetDevice(&cur) != cudaSuccess || cudaMemGetInfo(&fb, &tb) != cudaSuccess) {
                        (void) cudaGetLastError();
                        sampled = false;   // no sample: the restore goes ahead as it always did
                        break;
                    }
                    restore_free.push_back((uint64_t) fb);
                    restore_ready.push_back(strata::core::conversation_copy_stream_ready(cur) ? 1 : 0);
                }
                if (sampled &&
                    !strata::program::conv_cache::restore_fits(restore_free.data(), restore_ready.data(), restore_free.size(),
                                                               strata::program::conv_cache::kRestoreFirstUseMib << 20)) {
                    for (size_t k = 0; k < restore_free.size(); ++k)
                        if (!restore_ready[k] && restore_free[k] < (strata::program::conv_cache::kRestoreFirstUseMib << 20)) tight = (int) k;
                    std::fprintf(stderr, "strata serve: conversation cache: skip restore (%s has %lld MiB free and no copy stream "
                                         "yet, which needs up to %lld MiB); rereading the prompt\n",
                                 tight <= 0 ? "CUDA0" : ("stage " + std::to_string(tight)).c_str(),
                                 (long long) (tight >= 0 ? restore_free[(size_t) tight] >> 20 : 0),
                                 (long long) strata::program::conv_cache::kRestoreFirstUseMib);
                    auto retry = std::move(*incoming);
                    incoming.reset();
                    if (!conversations.put(std::move(retry)))
                        std::fprintf(stderr, "strata serve: conversation cache: could not return the skipped snapshot\n");
                    err.clear();
                }
            }
            if (borrow) {
                const strata::core::SavedConversation* donor = conversations.find(parked.serial);
                if (donor == nullptr ||
                    !strata::core::conversation_snapshot_validate(*donor, ss, g, use_mtp ? &mtp.kv_state() : nullptr,
                                                                  err)) {
                    std::fprintf(stderr, "strata serve: conversation cache: cannot borrow from a parked conversation "
                                         "(%s)\n", err.empty() ? "gone" : err.c_str());
                    borrow = false;
                    err.clear();
                }
            }
            // Preserve the outgoing branch before any checkpoint rewind, reset,
            // or incoming restore overwrites the positional state it requires.
            // pin=N: this request resumes exactly at the pinned shared prefix of the conversation the session holds
            // (a sibling query of the last one).  What the last query added past it is its private suffix - a few
            // dozen tokens, cheaper to read than the prefix's whole K/V is to copy to the host - so it is not parked:
            // N queries keep one prefix.  A request without pin=, or one that resumes anywhere else, parks as before.
            bool pin_sibling = false;
            if (req_pin > 0 && req_ckpt && o.prompt_cache >= 3 && !from_live && !incoming && slot_source < 0 &&
                resume == req_pin && live_ok)
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() == resume && c.pinned) pin_sibling = true;
            // --conversation-cache-park-gain N: a park now would supersede the entry parked_prefix_tokens() reports (the copy
            // this conversation wrote last), so a conversation that advanced fewer than N tokens since then keeps it:
            // it still holds this conversation's prefix, and re-parking writes the whole snapshot only for the next
            // turn to drop it again.  0 (the default) parks every turn, as before.
            const size_t park_held = o.conversation_cache_park_gain > 0
                                         ? conversations.parked_prefix_tokens(live, live_imgs, checks, cvec_cached)
                                         : 0;
            const bool park_worth = o.conversation_cache_park_gain <= 0 || park_held == 0 ||
                                    (int64_t) live.size() >= (int64_t) park_held + o.conversation_cache_park_gain;
            // STRATA_PARK_ASYNC: the checkpoints this request drops after its park - all of them when another state
            // comes in (a parked conversation, a borrow, a batch slot) or the cvec changes, else the ones off this
            // prompt's path (off_path below) - move into the parked image
            struct ParkDrops {
                std::function<bool(const ConvCheckpoint&)>& f;
                ~ParkDrops() { f = nullptr; }
            } park_drops_reset{park_drops};
            if (park_async_env()) {
                const bool drops_all = incoming.has_value() || borrow || slot_source >= 0 || want_cvec != cvec_cached;
                park_drops = [&, drops_all](const ConvCheckpoint& c) {
                    return drops_all || (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                };
            }
            if (borrow && !pin_sibling && park_worth) {
                conversations.guard(parked.serial);   // parking the outgoing conversation must not evict the donor
                const bool parked_ok = park_current(0);
                const bool blocked = conversations.guard_blocked();
                conversations.unguard();
                if (!parked_ok) {
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                if (blocked) {
                    // the outgoing conversation fits only in the donor's room: it keeps its place, the donor is
                    // taken whole as before (it was validated above)
                    std::fprintf(stderr, "strata serve: conversation cache: no room beside the parked conversation "
                                         "this request starts with; taking it instead of borrowing\n");
                    borrow = false;
                    if (const size_t at = conversations.index_of(parked.serial); at != SIZE_MAX)
                        incoming.emplace(conversations.take(at));
                    if (!park_current(incoming ? incoming->bytes() : 0)) {
                        print_response("ERR %s\n", err.c_str());
                        return 1;
                    }
                }
            } else if ((!from_live || incoming || slot_source >= 0) && !pin_sibling && park_worth &&
                       !park_current(incoming ? incoming->bytes() : 0)) {
                print_response("ERR %s\n", err.c_str());
                return 1;
            }
            if (slot_source >= 0) {
                lent_slot = -1;   // (a longer start than the main session's)
                const auto t0 = Clock::now();
                if (!copy_slot(false, slot_source, bs[(size_t) slot_source].ids, slot_ck, err)) {
                    // the main session may be half written: read this prompt from the start
                    std::fprintf(stderr, "strata batch: restoring slot %d failed (%s); reading the prompt\n",
                                 slot_source, err.c_str());
                    err.clear();
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    resume = 0;
                    from_live = false;
                } else {
                    checks.clear();   // the main session's checkpoints were of the conversation it held before
                    // a read that gave way, the same request again (into its own slot): on with it - its segments and
                    // checkpoints as if it had not stopped
                    if (slot_ck == nullptr && bs[(size_t) slot_source].partial && admit_slot == slot_source) {
                        resumed_from0 = bs[(size_t) slot_source].partial_from0;
                        resume_plan = bs[(size_t) slot_source].partial_plan;
                        resume_plan_at = slot_tokens;
                        resume_plan_end = resume_plan.to;
                        bs[(size_t) slot_source].partial = false;
                        bs[(size_t) slot_source].partial_plan = {};
                    }
                    // the slot's checkpoints come along (the one continued from, and any shorter one: prefixes of what
                    // was copied back, their K/V cells with it), so the chain keeps its branch points and the parked
                    // copies of this conversation a turn back - whose deepest checkpoint the chain holds again - go
                    // as superseded (#342) instead of taking a parked entry each until the cache evicts another chat.
                    // STRATA_SLOT_CKPT_MOVE: the slot this request continues in drops its checkpoints at the
                    // admission, so they move (the one continued from moves on into the slot then: moved_in)
                    const bool move_cks = slot_ckpt_move_env() && admit_slot == slot_source;
                    const int64_t ck_len = slot_ck != nullptr ? (int64_t) slot_ck->ids.size() : -1;
                    std::vector<ConvCheckpoint>& sc = bs[(size_t) slot_source].checks;
                    // a shorter checkpoint comes along only when it is a prefix of the tokens copied back (text only):
                    // its cells are then the ones the main session holds
                    const std::vector<int32_t> copied_back = slot_ck != nullptr ? slot_ck->ids : bs[(size_t) slot_source].ids;
                    auto prefix_of_copied = [&](const ConvCheckpoint& c) {
                        return c.imgs.empty() && c.ids.size() <= copied_back.size() &&
                               std::equal(c.ids.begin(), c.ids.end(), copied_back.begin());
                    };
                    // the one continued from first, exactly as before this change (its copy is the one the request needs)
                    const ConvCheckpoint* first = slot_ck;
                    if (first != nullptr) {
                        if (move_cks) {
                            checks.push_back(std::move(sc[(size_t) (first - sc.data())]));
                            sc.erase(sc.begin() + (first - sc.data()));
                            first = nullptr;   // gone from the slot
                        } else {
                            checks.push_back(*first);
                        }
                    }
                    // then the shorter ones: an optimization, so a host allocation that fails skips them
                    try {
                        for (auto it = sc.begin(); it != sc.end();) {
                            if ((int64_t) it->ids.size() > slot_tokens || &*it == first || !prefix_of_copied(*it)) {
                                ++it;
                            } else if (move_cks) {
                                checks.push_back(std::move(*it));
                                it = sc.erase(it);
                            } else {
                                checks.push_back(*it);
                                ++it;
                            }
                        }
                    } catch (const std::bad_alloc&) {
                        std::fprintf(stderr, "strata batch: slot %d: its shorter checkpoints were not all copied (out of "
                                             "memory); the rest are skipped\n", slot_source);
                    }
                    if (slot_ck != nullptr) {
                        for (ConvCheckpoint& c : checks)   // the main session's chain has it now
                            if ((int64_t) c.ids.size() == ck_len) {
                                live = c.ids;
                                c.used = ++check_clock;
                                slot_ck = &c;
                            }
                        if (move_cks) moved_in = ck_len;
                    } else {
                        live = bs[(size_t) slot_source].ids;
                    }
                    live_imgs.clear();
                    resume = slot_tokens;
                    from_live = true;
                    std::fprintf(stderr, "strata batch: slot %d gave back %lld tokens of this conversation (%s) in "
                                 "%.1f ms\n", slot_source, (long long) slot_tokens,
                                 slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
            }
            // the elastic K/V: a parked conversation is restored whole, and it may be longer than this prompt
            // (STRATA_PARK_VRAM_MIB: the parked K/V copies in VRAM go first, then one more try)
            if (incoming && !kvg_ensure((int64_t) incoming->live.ids.size() + 256, kv_quiesce) &&
                (strata::core::conversation_park_vram_release_all() == 0 ||
                 !kvg_ensure((int64_t) incoming->live.ids.size() + 256, kv_quiesce))) {
                print_response("ERR the K/V cannot grow to the parked conversation: no VRAM is left\n");
                return 1;
            }
            if (borrow) {
                const auto t0 = Clock::now();
                const strata::core::SavedConversation* donor = conversations.find(parked.serial);   // still parked
                if (donor == nullptr ||
                    strata::core::conversation_snapshot_restore_prefix(*donor, parked.tokens, ss, g,
                                                                       use_mtp ? &mtp.kv_state() : nullptr, err) !=
                        strata::core::ConversationRestore::restored) {
                    // validated above and guarded while parking: a failure here is fatal, as for a whole restore
                    print_response("ERR restoring a borrowed conversation prefix: %s\n",
                                err.empty() ? "gone" : err.c_str());
                    return 1;
                }
                // the donor's checkpoints on this prompt's path, copied: the donor keeps its own (the one at
                // parked.tokens is mounted below, as for any checkpoint resume)
                checks.clear();
                for (const ConvCheckpoint& c : donor->checkpoints)
                    if ((int64_t) c.ids.size() <= parked.tokens && starts_with(c.ids, c.imgs)) checks.push_back(c);
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() == parked.tokens) { live = c.ids; live_imgs = c.imgs; }
                cvec_cached = donor->cvec;
                resume = parked.tokens;
                from_live = false;
                conversations.count_borrow();
                std::fprintf(stderr, "strata serve: conversation cache: borrowed %lld tokens from a parked "
                             "conversation, which stays parked, in %.1f ms; parked=%zu bytes=%zu borrowed=%zu\n",
                             (long long) resume, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.borrowed());
            }
            if (incoming) {
                const auto t0 = Clock::now();
                if (strata::core::conversation_snapshot_restore(*incoming, ss, g,
                        use_mtp && (stages.empty() || mtp_first) ? &mtp.kv_state() : nullptr, err) !=
                    strata::core::ConversationRestore::restored) {
                    // Already prevalidated above: a failure here is fatal, never
                    // permission to decode from a partially restored session.
                    print_response("ERR restoring parked conversation: %s\n", err.c_str());
                    return 1;
                }
                for (size_t i = 0; i < stages.size(); ++i) {   // the later stages' parts
                    const strata::core::OnDevice on(stages[i]->dev);
                    if (strata::core::conversation_snapshot_restore(incoming->stage_images[i], stages[i]->ss, g,
                            use_mtp && !mtp_first && i + 1 == stages.size() ? &mtp.kv_state() : nullptr, err) !=
                        strata::core::ConversationRestore::restored) {
                        print_response("ERR restoring parked conversation (stage CUDA%d): %s\n", stages[i]->dev,
                                    err.c_str());
                        return 1;
                    }
                    if (cudaDeviceSynchronize() != cudaSuccess) { print_response("ERR stage sync\n"); return 1; }
                }
                if (use_mtp && std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr) {
                    uint64_t draft_hash = 0;
                    const auto& draft_kv = stages.empty() || mtp_first ? incoming->kv.back() : incoming->stage_images.back().kv.back();
                    const strata::core::OnDevice on_d(stages.empty() || mtp_first ? 0 : stages.back()->dev);
                    if (!strata::core::conversation_kv_verify(draft_kv, mtp.kv_state(), g,
                            int64_t(incoming->live.ids.size()), false, draft_hash, err)) {
                        print_response("ERR verifying restored draft KV: %s\n", err.c_str());
                        return 1;
                    }
                    std::fprintf(stderr, "strata serve: SNAPSHOT_VERIFY draft=%016llx cells=%lld mode=%d source=%s resident=%lld\n",
                                 (unsigned long long) draft_hash, (long long) draft_kv.cells,
                                 mtp.kv_state().kv_mode, "ram",
                                 (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                }
                live = std::move(incoming->live.ids);
                live_imgs = std::move(incoming->live.imgs);
                if (stages.empty()) {
                    checks = std::move(incoming->checkpoints);
                } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
                    strata::core::ConversationCheckpointSplit cs;
                    cs.stage0 = std::move(incoming->checkpoints);
                    for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
                    if (!strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
                }
                cvec_cached = incoming->cvec;
                resume = parked.tokens;
                from_live = parked.live;
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                    std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                    for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv), live);
                }
                incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                lent_slot = -1;   // (the main session holds the parked conversation now)
                std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                             (long long) resume, from_live ? "live" : "checkpoint",
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes());
            }
            if (want_cvec != cvec_cached) {
                live_ok = false;
                checks.clear();
                cvec_cached = want_cvec;
            }
            if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(want_cvec);
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            auto off_path = [&](const ConvCheckpoint& c) {
                return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
            };
            if (ckpt_pool.on())   // STRATA_CKPT_POOL: their storage is the next checkpoint's
                ckpt_pool.erase_if(checks, off_path);
            else
                checks.erase(std::remove_if(checks.begin(), checks.end(), off_path), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            // the elastic K/V: the outgoing session is parked and this request rewrites every cell from `resume` on,
            // so cells past this prompt's are no longer anyone's - far more than it needs go back to the cache
            if (kvg.on) {
                kv_quiesce();
                // STRATA_KV_GROW_HOLD=1 (opt-in): keep the K/V as long as the longest parked conversation; default: trim to this request alone, as 0.1.40
                static const bool hold_on = [] { const char* v = std::getenv("STRATA_KV_GROW_HOLD"); return v != nullptr && std::atoi(v) != 0; }();
                if (!kvg_trim(n + 256, hold_on ? conversations.longest_tokens() + 256 : 0)) {
                    print_response("ERR the K/V could not give its VRAM back to the expert cache\n");
                    std::fflush(stdout);
                    return 1;
                }
            }
            int64_t reread_to = -1;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
            if (resume == 0) {
                strata::core::session_zero(ss, g, nullptr, main_cs);
                cudaStreamSynchronize(main_stream);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    cudaStreamSynchronize(st->stream);
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                bool lent = false;   // STRATA_SLOT_CKPT_MOVE: the one a slot keeps (found above)
                if (c == nullptr && lent_slot >= 0)
                    for (ConvCheckpoint& k : bs[(size_t) lent_slot].checks)
                        if ((int64_t) k.ids.size() == resume && starts_with(k.ids, k.imgs)) { c = &k; lent = true; }
                if (!lent) lent_slot = -1;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
                if (reread && c != nullptr) {
                    // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
                    // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
                    // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
                    // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
                    // checkpoint missed shows up as a difference.
                    strata::core::session_zero(ss, g, nullptr, main_cs);
                    cudaStreamSynchronize(main_stream);
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                        cudaStreamSynchronize(st->stream);
                    }
                    reread_to = resume;
                    std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                         "restoring\n", (long long) resume);
                } else if (c == nullptr || !checkpoint_restore(*c, ss, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const strata::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    print_response("ERR restoring a conversation checkpoint failed\n");
                    return 1;
                }
                if (lent && lent_slot == admit_slot) {
                    // the slot this request continues in drops its checkpoints at the admission: this one moves back
                    // into the chain (and on into the slot then: moved_in)
                    std::vector<ConvCheckpoint>& sc = bs[(size_t) lent_slot].checks;
                    const auto it = sc.begin() + (c - sc.data());
                    checks.push_back(std::move(*it));
                    sc.erase(it);
                    moved_in = resume;
                    lent_slot = -1;
                }
                if (lent && trace)
                    std::fprintf(stderr, "strata trace: the checkpoint at %lld restored from slot %d's%s\n",
                                 (long long) resume, lent_slot >= 0 ? lent_slot : admit_slot,
                                 lent_slot >= 0 ? "" : " (moved back into the chain)");
            }
            // --prefill-pipe: going on from a periodic checkpoint taken inside a planned segment (whatever restored it:
            // a checkpoint of this session, of a slot or of a parked conversation), the rest of that segment is read
            // in its chunks.  STRATA_CKPT_REREAD reads [0, resume) again in the chunks the saving read had, which it
            // can only when that segment started at 0.  The BYIELD park set resume_plan above.
            if (resume_plan.chunk <= 0 && resume > 0 && !stages.empty())
                for (const ConvCheckpoint& k : checks)
                    if ((int64_t) k.ids.size() == resume && k.plan.chunk > 0) {
                        if (reread_to <= 0) {
                            resume_plan = k.plan;
                            resume_plan_at = resume;
                            resume_plan_end = k.plan.to;
                        } else if (k.plan.from == 0) {   // [0, resume) only: its end is the checkpoint
                            resume_plan = k.plan;
                            resume_plan_at = 0;
                            resume_plan_end = resume;
                        } else {
                            std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: the checkpoint is inside a "
                                                 "segment that started at %lld, so a read from 0 has other chunks\n",
                                         (long long) k.plan.from);
                        }
                        break;
                    }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (use_mtp && resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
            tr("request", n, geni ? 1 : 0);
            req_times = ReqTimes{};
            if (use_mtp) mtp.set_prompt_len(n);
            // KVA: the draft layer's first real cell for the state this request continues from (0 for a cold read - the
            // first segment raises it if KVA approximates it - and for any state no KVA read produced)
            if (use_mtp && mtp.cell_floor_enabled())
                mtp.set_cell_floor(resume > 0 && reread_to <= 0 ? kva_floors.floor_for(ids.data(), resume) : 0);
#if defined(STRATA_HAVE_DFLASH2)
            if (use_df2) {   // the drafter's ring goes on from the live conversation, or starts over (a parked or edited one)
                std::vector<int32_t> idv(ids.begin(), ids.end());
                bool cold = false;
                df2.begin_request(idv.data(), (int64_t) idv.size(), reread_to > 0 ? 0 : resume, &cold);
                if (df2.last_rebuilt > 0)
                    std::fprintf(stderr, "strata serve: DFlash2: %s%lld context rows of the window rebuilt from the drafter's log in %.1f ms%s\n",
                                 df2.last_swapped ? "a parked conversation came back: " : "", (long long) df2.last_rebuilt, df2.last_ms,
                                 df2.last_missing >= 64 ? " (the rest were never logged)" : "");
                if (cold && df2.last_missing >= 64)   // (a few rows missing are not worth a line: they are counted in the ring stats)
                    std::fprintf(stderr, "strata serve: DFlash2: %lld context rows of the window are not the drafter's to give (a conversation it has no log of: edited, "
                                         "from a session file, or its log was evicted): it drafts poorly for the first rounds (the answers are the same)\n",
                                 (long long) std::max<int64_t>(df2.last_missing, 1));
                if (multi_gpu) df2_stage_prompt_taps_begin(std::max<int64_t>(0, n - 2047));
                else sp.set_taps(df2.tap_layers(), std::max<int64_t>(0, n - 2047), 2048);
            }
#endif
            const int64_t read_from = reread_to > 0 ? 0 : resume;
            // STRATA_MTP_ROOM_MIN: opened here, once the reused prefix is known - the threshold counts the tokens this
            // request reads (an agent turn on a long conversation reads a few hundred and gains nothing from the room).
            // Before `lend()` below, so a relayout re-carves with the room (its slots are the drafter's own bytes).
            if (mtp_room_slots > 0 && room_path != nullptr && n - read_from >= mtp_room_min && mtp.experts_resident()) {
                const auto room_t0 = Clock::now();
                if (!mtp.experts_lend(err)) {
                    std::fprintf(stderr, "strata serve: prompt room: %s: this prompt reads as it always did\n", err.c_str());
                } else {
                    room_path->set_ring_room_active(true);   // before the loan: a relayout re-carves with the room
                    mtp_room_open = true;
                    std::fprintf(stderr, "strata serve: prompt room: %lld tokens to read, the prompt path's ring is the "
                                         "drafter's %llu MiB (%d slots past its own, lent in %.1f ms)\n",
                                 (long long) (n - read_from),
                                 (unsigned long long) (mtp.experts_bytes() >> 20), mtp_room_slots,
                                 std::chrono::duration<double, std::milli>(Clock::now() - room_t0).count());
                }
            }
            conversations.limit_reuse(read_from);
            pp_total = n;
            pp_from = read_from;
            pp_reached = read_from;
            pp_exact = true;   // a restore, a checkpoint mount or a zeroed session: the state is exactly at read_from
            pp_t0 = r0;
            pp_next_check = reread_to > 0 || !req_ckpt ? INT64_MAX : resume + o.prompt_cache_every;   // ckpt=0: none
            pp_tail_saved = reread_to > 0 || !req_ckpt;   // ckpt=0 (#861): no tail checkpoint either
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            print_response("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
            strata::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
            // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
            // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
            // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
            // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
            // read batched and its header still goes through the windows.  Picture rows need the batched path.
            // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
            static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
            int64_t req_short_read = o.short_read;   // STRATA_PIPELINE_SWITCH may set it per request (short_read=)
            if (short_read_off_once) {   // #3465: the request after non-finite logits in a prompt window
                short_read_off_once = false;
                req_short_read = 0;
                std::fprintf(stderr, "strata serve: after non-finite logits in a prompt window (#3465): this request reads its "
                                     "whole prompt through the batched path, none of it through the verify windows\n");
            }
            auto windows_ok = [&](int64_t a, int64_t b) -> bool {
                if (no_short || b - a > req_short_read) return false;
                if (sp.embd_rows != nullptr)
                    for (int64_t i = a; i < b; ++i)
                        if (sp.embd_rows[i] != nullptr) return false;
                return true;
            };
            // --pipeline-windows, testing: STRATA_PIPELINE_SWITCH=<file> is read at every request, "pw=<0|1|2>
            // theta=<f> force_miss=<k> short_read=<n>" - an A/B of the pipelined and the serial loops (and forced
            // rollbacks) on one server, with the same expert placement
            int pl_pw = pipe ? o.pipeline_windows : 0;
            float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
                return v ? (float) std::atof(v) : 0.2f;
            }();
            int pl_force_miss = [] {   // exactness test: every k-th speculative window gets a wrong first token
                const char* v = pipe_dbg_env("STRATA_PIPELINE_FORCE_MISS");
                return v ? std::max(0, std::atoi(v)) : 0;
            }();
            if (static const char* sw = pipe_dbg_env("STRATA_PIPELINE_SWITCH"); sw != nullptr && pipe) {
                if (std::FILE* f = std::fopen(sw, "r")) {
                    char buf[256] = {};
                    const size_t nr = std::fread(buf, 1, sizeof buf - 1, f);
                    std::fclose(f);
                    buf[nr] = 0;
                    if (const char* q = std::strstr(buf, "pw=")) pl_pw = std::max(0, std::min(2, std::atoi(q + 3)));
                    if (const char* q = std::strstr(buf, "theta=")) pl_theta = (float) std::atof(q + 6);
                    if (const char* q = std::strstr(buf, "force_miss=")) pl_force_miss = std::max(0, std::atoi(q + 11));
                    if (const char* q = std::strstr(buf, "short_read=")) req_short_read = std::max(0LL, std::atoll(q + 11));
                    std::fprintf(stderr, "strata pipeline switch: pw=%d theta=%.3f force_miss=%d short_read=%lld\n", pl_pw,
                                 pl_theta, pl_force_miss, (long long) req_short_read);
                }
            }
            // --pipeline-windows: tokens [a, b) through the windows with the stages overlapped.  The tokens are known,
            // so every window commits whole: stage 0 queues window K, its commit and window K+1 back to back (each
            // parity's verifier keeps its own window), and stage 1 runs window K once stage 0 is done with it.  Every
            // stage runs the same windows in the same order as read_windows below, so the session ends the same.
            auto read_windows_pl = [&](int64_t a, int64_t b, std::string& e) -> bool {
                strata::core::progress_at("reading the prompt (pipelined verify windows), from token", a);
                if (!pl_prepare(e)) return false;
                for (int par = 0; par < 2; ++par) PV[0][par]->set_head_sampling(false);   // reaches stage 1 (set_next)
                struct Restore {
                    strata::core::Verifier* v[2];
                    ~Restore() { v[0]->set_head_sampling(true); v[1]->set_head_sampling(true); }
                } restore{{PV[0][0], PV[0][1]}};
                struct Win { int T; int64_t q; };
                std::vector<Win> wins;
                for (int64_t q = a; q < b;) {
                    const int T = (int) std::min<int64_t>(S, b - q);
                    wins.push_back({T, q});
                    q += T;
                }
                const size_t N = wins.size();
                std::vector<int32_t> w0((size_t) S), w1((size_t) S), nx((size_t) S);
                size_t l0 = 0, f0 = 0, l1 = 0, f1 = 0;   // stage 0 launched / finished, stage 1 launched / finished
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                auto fail = [&](std::string why) -> bool {
                    pl_drain();
                    e = why.empty() ? std::string("a pipelined prompt window failed") : why;
                    return false;
                };
                // The drafter's prefill blocks this thread until its stream is idle (on HIP, MtpDrafter::prefill syncs
                // once per group whatever its `sync` says: prefill_per_group_sync, src/core/mtp.cpp).  With the drafter
                // on the first card (mtp_first) a stage-0 window still in flight spins on that GPU for doorbells only
                // this thread serves, so a prefill called beside it never returns: the prompt read stopped for good at
                // its first window (0.1.56, gfx1201 + gfx1030, --pipeline-windows 2, issue #5147).  Serve stage 0 to its
                // end first; the rest of the read keeps both stages overlapped.  The tokens are unchanged: the same
                // windows, in the same order, only served a little earlier.
                auto serve_stage0_to_end = [&]() -> bool {
                    while (f0 < l0) {
                        strata::core::Verifier& v0 = *PV[0][f0 & 1];
                        if (v0.service(win_pool_fn, PSD[f0 & 1], e) < 0) return false;
                        if (v0.done(e)) {
                            if (!v0.pl_finish(nullptr, e)) return false;
                            ++f0;
                        } else if (!e.empty()) return false;
                    }
                    return true;
                };
                while (f1 < N) {
                    if (input.is_stop_requested()) return fail("cancelled");
                    // stage 0: up to two windows queued; window K+2 (K's parity) only once stage 1 has finished window
                    // K, whose hand-off K+2 overwrites
                    while (l0 < N && l0 < f0 + 2 && l0 < f1 + 2) {
                        const Win& w = wins[l0];
                        for (int t = 0; t < w.T; ++t) w0[(size_t) t] = (int32_t) cur[(size_t) (w.q + t)];
                        strata::core::Verifier& v = *PV[0][l0 & 1];
                        if (!v.pl_launch(w.T, w0.data(), w.q, e) || !v.pl_commit_async(w.T, e)) return fail(e);
                        ++l0;
                    }
                    if (f0 < l0) {
                        strata::core::Verifier& v = *PV[0][f0 & 1];
                        if (v.service(win_pool_fn, PSD[f0 & 1], e) < 0) return fail(e);
                        if (v.done(e)) {
                            if (!v.pl_finish(nullptr, e)) return fail(e);
                            ++f0;
                        } else if (!e.empty()) return fail(e);
                    }
                    if (l1 < f0 && l1 == f1) {   // stage 1: the next window, its stage 0 done
                        const Win& w = wins[l1];
                        for (int t = 0; t < w.T; ++t) w1[(size_t) t] = (int32_t) cur[(size_t) (w.q + t)];
                        strata::core::Verifier& v = *PV[1][l1 & 1];
                        if (pl_mtp_live[l1 & 1]) {   // the drafter has read this parity's previous rows first
                            const strata::core::OnDevice on(mtp.device());
                            cudaStreamWaitEvent(v.stream(), pl_mtp_ev[l1 & 1], 0);
                        }
                        if (!v.pl_launch(w.T, w1.data(), w.q, e) || !v.pl_commit_async(w.T, e)) return fail(e);
                        ++l1;
                    }
                    if (f1 < l1) {
                        strata::core::Verifier& v = *PV[1][f1 & 1];
                        if (v.service(win_pool_fn, PSD[f1 & 1], e) < 0) return fail(e);
                        if (v.done(e)) {
                            if (!v.pl_finish(nullptr, e)) return fail(e);
                            if (nonfinite_hit(v, "prompt")) return fail("non-finite logits (#879)");
                            const Win& w = wins[f1];
                            for (int t = 0; t < w.T; ++t) nx[(size_t) t] = (int32_t) cur[(size_t) (w.q + t + 1)];
                            // the drafter shares stage 0's card: no stage-0 window may still be in flight when its
                            // prefill waits on the host (see serve_stage0_to_end)
                            if (mtp_first && !serve_stage0_to_end()) return fail(e);
                            if (!mtp.prefill(v.final_R(0), nx.data(), w.T, w.q, e, false)) return fail(e);
                            {
                                const strata::core::OnDevice on(mtp.device());
                                cudaEventRecord(pl_mtp_ev[f1 & 1], mtp.stream());
                            }
                            pl_mtp_live[f1 & 1] = true;
                            pp_reached = w.q + w.T;   // #471
                            ++f1;
                        } else if (!e.empty()) return fail(e);
                    }
                    if (drive.d.failed) return fail(drive.d.fail ? drive.d.fail : "the expert pool failed");
                }
                pl_drain();   // nothing in flight: both stages' commits and the drafter's rows have landed
                pl_mtp_live[0] = pl_mtp_live[1] = false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                print_response("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
            auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
                // (STRATA_LOGPOS reads every window's logits: the serial loop)
                // the pipelined prompt read stays two-stage (three or more stages read serially; only the decode overlaps)
                if (pipe && n_stages == 2 && pl_pw >= 1 && std::getenv("STRATA_LOGPOS") == nullptr) return read_windows_pl(a, b, e);
                strata::core::progress_at("reading the prompt (verify windows), from token", a);   // #217: not "batched"
                // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
                struct NoHeadSampling {
                    strata::core::Verifier& v;
                    explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
                pp_exact = true;   // between windows the session is exactly at `a`
                for (int64_t q = a; q < b;) {
                    if (input.is_stop_requested()) { e = "cancelled"; return false; }
                    pp_exact = false;   // until this window and the drafter's rows are in
                    int T = (int) std::min<int64_t>(S, b - q);
                    // --lookup-max-t: the prompt's windows take the allowed sizes too (each size is a graph of its own);
                    // one whose graph finds no VRAM is dropped as in decode
                    if (o.lookup_max_t > 0) {
                        T = graph_fits(wplan.policy.fit(T));
                        if (T < 0) { e = err; return false; }
                    }
                    for (int t = 0; t < T; ++t) {
                        win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                        nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                        return false;
                    }
                    if (nonfinite_hit(ver, "prompt")) {   // (its commit is not needed: the next request reads from 0)
                        e = "non-finite logits (#879)";
                        return false;
                    }
                    // STRATA_LOGPOS=<path>: the teacher-forced log-probability of every token read here (every
                    // token is committed and nxt[t] is the prompt's own next token), appended to <path>; the last
                    // column is the log-probability of STRATA_LOGPOS_EXTRA (default 248046, <|im_end|>), so the
                    // end-of-turn mass a chat model puts on raw text can be taken out of the measurement
                    static std::FILE* logpos = [] {
                        const char* p = std::getenv("STRATA_LOGPOS");
                        return p != nullptr ? std::fopen(p, "ab") : nullptr;
                    }();
                    static const int32_t logpos_extra = [] {
                        const char* p = std::getenv("STRATA_LOGPOS_EXTRA");
                        return p != nullptr ? (int32_t) std::atoi(p) : (int32_t) 248046;
                    }();
                    if (logpos != nullptr && !ver.window_logprobs(nxt.data(), T, q, logpos_extra, logpos, e))
                        return false;
                    if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
#if defined(STRATA_HAVE_DFLASH2)
                    if (use_df2) {
                        const float* tp = df2_win_taps((int) T, e);
                        if (tp == nullptr || !df2.context(tp, T, q, e)) return false;
                        df2.note_rows(win.data(), T, q);
                    }
#endif
                    q += T;
                    pp_reached = q;   // #471
                    pp_exact = true;
                }
                // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
                if (!ver.wait_commit(e)) return false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                print_response("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // the batched path's slots are lent just before its first run and given back (refilled) before a window
            // reads - so the windows always see the whole expert cache - or once the prompt is read
            // Every participant gives its loan back here: the rows it lent are refilled into the SAME slots from
            // the arena, the residency table is restored, and one upload puts it on every device.  A stage refills
            // through its own cache and its own device - a slot refilled into the wrong cache would leave that
            // stage's cache holding an expert it does not own, which is silent and produces plausible tokens.
            auto refill_issue = [&](PfPart& p, std::string& e) -> bool {
                tr("refill start", (long long) p.lent.size());
                return refill_loan_issue(p, e);
            };
            auto refill_one = [&](PfPart& p, std::string& e) -> bool {
                if (p.lent.empty()) return true;
                return refill_issue(p, e) && refill_loan_wait(p, e);
            };
            // #340: with several stages every stage's copies go out first (each card on its own link), then each is
            // waited for - the same copies into the same slots, so the same cache; one stage: refill_one exactly.
            // STRATA_REFILL_SERIAL=1: stage after stage, as 0.1.30/0.1.31.
            static const bool refill_serial = std::getenv("STRATA_REFILL_SERIAL") != nullptr;
            auto refill = [&](std::string& e) -> bool {
                const auto t_rf = Clock::now();
                int64_t n_lent = 0, n_parts = 0;
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
                if (n_parts > 1 && !refill_serial) {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_issue(p, e)) return false;
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_loan_wait(p, e)) return false;
                } else {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_one(p, e)) return false;
                }
                if (n_parts > 0) res_upload();
                if (trace && n_parts > 0) {
                    std::fprintf(stderr, "strata trace: refilled %lld slots on %lld stage(s) in %.1f ms\n",
                                 (long long) n_lent, (long long) n_parts,
                                 std::chrono::duration<double, std::milli>(Clock::now() - t_rf).count());
                    std::fflush(stderr);
                }
                return true;
            };
            // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
            // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
            // participant's own cache, and marking only that participant's own layers
            int64_t seg_lend = 0;   // the loan the last lend left (the largest participant's chunk; 0: none)
            auto lend = [&](int64_t tokens, std::string& e, int64_t exact) -> bool {
                seg_lend = 0;
                if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
                const auto t_ln = Clock::now();
                // what this segment needs, capped by the configured chunk: a request lends only what its own
                // prompt needs, so a large chunk costs a short prompt nothing
                const int64_t want_full = planned_chunk(tokens, o.prefill_chunk, pf_plan_chunk);
                // #340: a short enough request reads in the stages' own S-token chunks (nothing lent)
                const int64_t want_seg = o.prefill_head_alias ? o.prefill_chunk
                                       : split_small > 0 && tokens <= split_small_max ? std::min(want_full, split_small)
                                                                                     : want_full;
                // `exact`: a planned segment read on - the loan it was read with (what is left would size another)
                const int64_t want = exact > 0 ? std::min(exact, o.prefill_chunk) : want_seg;
                if (want <= 0) {
                    e = "prefill: cannot lend buffers for an empty request segment";
                    return false;
                }
                bool any = false;
                for (PfPart& p : pf_parts) {
                    if (p.first < 0) continue;
                    if (!p.lent.empty()) {
                        if (want <= p.lent_chunk) continue;            // its current loan already covers this
                        // ONLY this participant's loan goes back: `refill` would return the other participants'
                        // loans too, and their buffers are still laid out in their caches - marking those slots
                        // resident again would hand the next window a prompt buffer in place of an expert
                        if (!refill_one(p, e)) return false;
                    }
                    const strata::core::OnDevice on(p.dev);
                    int64_t want_p = want;
                    int32_t first = p.first_now;
                    // --prefill-head-alias: the loan laid out at start (slot 1 on) is kept; a hybrid layout cannot relayout.
                    // (STRATA_MTP_ROOM_MIN's room is not one: its slots are the drafter's own bytes, so a roomed
                    // request lends and is laid out exactly as any other - a relayout re-carves the room with it.)
                    if (!o.prefill_head_alias) {
                        // #2090: the loan a chunk needs is NOT monotonic in the chunk: a chunk above the scan's small_max
                        // lays out the byte-budget ring (199 slots at 5120), one at or below it 0.1.39's (384 at 4096), so
                        // a request's smaller chunk can need MORE slots than the auto chunk's region (p.first) holds.  The
                        // loan starts where the chunk itself needs it (the 128-slot floor aside); a chunk that cannot be
                        // lent even then is halved until it can, instead of ending the engine.
                        int64_t need_p = part_slots(p, want_p);
                        while ((need_p <= 0 || need_p + g_lend_keep > p.cache->slots()) && want_p > 256) {
                            want_p = std::max<int64_t>(256, want_p / 2 / 256 * 256);
                            need_p = part_slots(p, want_p);
                        }
                        if (need_p <= 0 || need_p + g_lend_keep > p.cache->slots()) {
                            e = "prefill: no chunk of this request fits the slots that can be lent (#2090)";
                            return false;
                        }
                        if (want_p != want)
                            std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens for this request: its "
                                         "buffers do not fit the slots that can be lent (#2090)\n", (long long) want,
                                         (long long) want_p);
                        if (need_p > p.cache->slots() - p.first)   // the case that ended the engine before (#2090)
                            std::fprintf(stderr, "strata serve: the loan for a chunk of %lld tokens (%lld slots) is larger than the "
                                         "auto chunk's (%lld slots): starting it earlier (#2090)\n", (long long) want_p,
                                         (long long) need_p, (long long) (p.cache->slots() - p.first));
                        first = (int32_t) (p.cache->slots() - need_p);
                    }
                    if (want_p != p.sp->chunk() || first != p.first_now) {
                        if (!p.sp->relayout(want_p, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                        p.first_now = first;
                    }
                    for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t i = (size_t) (l * g.n_expert + ex);
                            if (host_res[i] >= first) {
                                p.lent.emplace_back((int32_t) i, host_res[i]);
                                host_res[i] = strata::core::kNotResident;
                                any = true;
                            }
                        }
                    p.lent_chunk = want_p;
                }
                for (const PfPart& p : pf_parts)
                    if (p.first >= 0) seg_lend = std::max(seg_lend, p.lent_chunk);
                if (any) res_upload();
                if (trace) {
                    int64_t n_lent = 0;
                    for (const PfPart& p : pf_parts) n_lent += (int64_t) p.lent.size();
                    std::fprintf(stderr, "strata trace: lent %lld slots for %lld tokens in %.1f ms\n", (long long) n_lent,
                                 (long long) want, std::chrono::duration<double, std::milli>(Clock::now() - t_ln).count());
                    std::fflush(stderr);
                }
                return true;
            };
            // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
            // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
            // decode for a share of the time the chunk took (STRATA_BATCH_DECODE_SHARE, default 0.5), so a long
            // prompt does not stop the others.  The slots' windows then see the cache without the slots this prompt
            // borrowed (marked missing: those experts run on the CPU), never a prompt buffer.
            static const double decode_share = [] {
                const char* v = std::getenv("STRATA_BATCH_DECODE_SHARE");
                return v != nullptr ? std::max(0.0, std::atof(v)) : 0.5;
            }();
            bool batch_fatal = false;
            int64_t il_parts = 0;   // the slots' windows run between this prompt's chunks
            double il_ms = 0;
            int64_t yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
            int yielded_slot = -1;
            // `BYIELD <slot>` (the server: a shorter request is waiting): at the next chunk boundary the part read so
            // far is copied into <slot> (this admission's own, or a free one the server reserved for a solo request),
            // the request ends with `YIELDED <slot> <tokens>` + DONE cancel, and the server sends it again later: it
            // continues from the slot with the same chunks.  #656's cooperative preemption, with a slot as the park.
            int64_t req_chunk = 0, req_chunk_len = 0;   // the DONE line's chunk=: the longest batched segment's first chunk

            // STRATA_PREFILL_CPU_SHARE_BATCH: the read takes the CPU share only with no slot decoding (their windows use the pool)
            auto share_gate = [&]() {
                int busy = 0;
                for (const auto& x : bs) busy += x.active ? 1 : 0;
                strata::prefill::Prefill::set_share_blocked(busy > 0);
                static int said = 0;
                static const bool on = [] { const char* v = std::getenv("STRATA_PREFILL_CPU_SHARE_BATCH"); return v != nullptr && v[0] == '1'; }();
                if (on && said < 4) {
                    ++said;
                    std::fprintf(stderr, "strata batch: CPU share for this read: %s (%d slot(s) decoding)\n", busy > 0 ? "off" : "on", busy);
                }
            };
            auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
                // --prefill-pipe on a layer split: the segment's chunk is planned ONCE, on its whole length (or taken
                // from the read it goes on from: resume_plan), and every run of it reads in that chunk (force_chunk) -
                // the same chunks whether it is read in one run, in pieces beside decoding slots, or goes on after a
                // BYIELD or from a checkpoint.  Without the pipe nothing here changes: runs read in the buffers' chunk.
                ConvCheckpoint::PromptPlan plan;
                if (!stages.empty()) {
                    if (resume_plan.chunk > 0 && a0 == resume_plan_at) {
                        using strata::prefill::Prefill;
                        const ConvCheckpoint::PromptPlan& rp = resume_plan;
                        if (b0 == resume_plan_end && rp.from >= 0 && rp.from <= a0 && (a0 - rp.from) % rp.chunk == 0 &&
                            rp.chunk <= sp.chunk() && Prefill::pipe_key(rp.pipe) == Prefill::pipe_key(Prefill::pipe_setting())) {
                            plan = rp;
                        } else {
                            std::fprintf(stderr, "strata prefill: the read going on at %lld is not the segment "
                                                 "[%lld, %lld) it was planned in (%lld-token chunks): planned again\n",
                                         (long long) a0, (long long) rp.from, (long long) rp.to, (long long) rp.chunk);
                        }
                        resume_plan = {};   // the first segment only
                    }
                    if (plan.chunk <= 0) {
                        const int64_t c = sp.plan_chunk(b0 - a0);
                        if (c > 0) {
                            plan.chunk = c;
                            plan.from = a0;
                            plan.to = b0;
                            plan.lend = seg_lend;
                            plan.pipe = strata::prefill::Prefill::pipe_setting();
                            plan.single = b0 - a0 <= c;
                            plan.chunked = c < std::min(b0 - a0, sp.chunk());
                            if (plan.chunked) strata::prefill::Prefill::log_plan(b0 - a0, c, sp.stage_count());
                        }
                    }
                }
                const bool planned = plan.chunk > 0;
                const bool plan_single = planned && plan.single;
                if (b0 - a0 > req_chunk_len) {
                    req_chunk_len = b0 - a0;
                    req_chunk = planned ? std::min(plan.chunk, b0 - a0) : std::min(sp.chunk(), b0 - a0);
                }
                seg_plan = plan;   // the checkpoints taken inside it keep it; cleared when the segment is done
                struct SegPlanScope {
                    ConvCheckpoint::PromptPlan& p;
                    ~SegPlanScope() { p = {}; }
                } seg_plan_scope{seg_plan};
                // the first run of a BGEN with co-reads: the waiting prompts are read in the same layer-major pass
                bool co_tried = false;
                auto co_run = [&](int64_t q, int64_t r, std::string& ce) -> bool {
                    using PSeg = strata::prefill::Prefill::Seg;
                    std::vector<PSeg> segs;
                    PSeg lead;
                    lead.tokens = ids.data() + q;
                    lead.n = r - q;
                    lead.pos0 = q;
                    segs.push_back(lead);
                    std::vector<const CoReq*> used;
                    for (const CoReq& c : co_set) {
                        if (c.slot < 0 || c.slot >= (int) bs.size() || c.slot == admit_slot || bs[(size_t) c.slot].active ||
                            bs[(size_t) c.slot].partial)
                            continue;
                        bool dup = false;
                        for (const CoReq* u : used) dup = dup || u->slot == c.slot;
                        if (dup) continue;
                        PSeg sg;
                        sg.tokens = c.ids.data();
                        sg.n = (int64_t) c.ids.size() - 1;   // (the last token goes through the slot's own first window)
                        sg.pos0 = 0;
                        for (size_t k = 0; k < bslot_ss.size(); ++k) sg.stage_ss.push_back(bslot_ss[k][(size_t) c.slot].get());
                        segs.push_back(sg);
                        used.push_back(&c);
                    }
                    if (segs.size() < 2) return sp.run(lead.tokens, lead.n, lead.pos0, ce);
                    for (const CoReq* c : used) {
                        park_slot(c->slot);   // the conversation the slot holds, before this read overwrites it
                        for (size_t k = 0; k < bslot_ss.size(); ++k) {
                            const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                            strata::core::session_zero(*bslot_ss[k][(size_t) c->slot], g, nullptr, nullptr);
                            cudaDeviceSynchronize();
                        }
                        bs[(size_t) c->slot] = BSlot{};   // (it holds nothing until the read is done)
                    }
                    const auto tco = Clock::now();
                    std::fprintf(stderr, "strata batch: co-read starting: %zu more prompt(s) with this request's %lld tokens\n",
                                 used.size(), (long long) lead.n);
                    std::fflush(stderr);
                    if (!sp.run_multi(segs, ce)) {
                        if (ce.rfind("prefill: multi-prompt read declined", 0) == 0) {   // nothing was read: the usual run
                            std::fprintf(stderr, "strata batch: co-read of %zu prompt(s) declined (%s)\n", used.size(), ce.c_str());
                            ce.clear();
                            return sp.run(lead.tokens, lead.n, lead.pos0, ce);
                        }
                        return false;
                    }
                    for (const CoReq* c : used) {
                        BSlot& sl = bs[(size_t) c->slot];
                        sl = BSlot{};
                        sl.ids.assign(c->ids.begin(), c->ids.end() - 1);
                        sl.cached = true;
                        sl.cvec = want_cvec;
                        print_response("BCOK %d %lld\n", c->slot, (long long) sl.ids.size());
                    }
                    std::fflush(stdout);
                    std::fprintf(stderr, "strata batch: co-read: %zu more prompt(s) read in the same pass as this request's %lld tokens "
                                         "(%.0f ms)\n", used.size(), (long long) lead.n,
                                 std::chrono::duration<double, std::milli>(Clock::now() - tco).count());
                    return true;
                };
                bool conc_read = false;   // run_conc: the slots decode beside the read, so the share stays off for all of it
                auto run = [&](int64_t q, int64_t r) -> bool {
                    if (planned) sp.force_chunk(plan.chunk, plan_single, plan.chunked);
                    if (!conc_read) share_gate();   // STRATA_PREFILL_CPU_SHARE_BATCH: re-decided for every run (a piece beside decoding slots too)
                    bool run_ok;
                    if (!co_set.empty() && !co_tried && admit_slot >= 0 && req_imgs.empty()) {
                        co_tried = true;
                        run_ok = co_run(q, r, e);
                    } else {
                        run_ok = sp.run(ids.data() + q, r - q, q, e);
                    }
                    if (run_ok && q == 0 && sp.kva_floor() > 0) {   // KVA approximated the start of this prompt: remember which tokens
                        kva_floors.note(ids.data(), sp.kva_floor());
                        if (use_mtp && mtp.cell_floor_enabled()) mtp.set_cell_floor(sp.kva_floor());
                    }
                    return run_ok;
                };
                // STRATA_BATCH_CONCURRENT_READ=1 (opt-in, one GPU, --batch with --no-prefill-borrow): the prompt is read on a
                // thread of its own while this thread keeps the slots' batch windows going, so the decoding slots do not
                // stop for an admission.  The prompt writes the main session, the windows the slots' sessions; the prompt
                // borrows no cache slots (--no-prefill-borrow), so the windows' experts stay where they are.  The same
                // bytes are computed as in the serial form (a window's rows do not depend on what else is on the GPU).
                static const bool conc1 = [] {
                    const char* v = std::getenv("STRATA_BATCH_CONCURRENT_READ");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                auto run_conc = [&](int64_t q, int64_t r) -> bool {
                    if (!conc1 || piped || !stages.empty() || o.batch <= 0 || !o.no_prefill_borrow || !batch_on() ||
                        !req_imgs.empty() || kvg.on)
                        return run(q, r);
                    std::atomic<bool> done{false};
                    bool read_ok = false;
                    in_prompt_read = true;
                    conc_read = true;
                    strata::prefill::Prefill::set_share_blocked(true);
                    std::thread reader([&] {
                        try {   // (an exception leaving a std::thread ends the process: a failed read instead)
                            read_ok = run(q, r);
                        } catch (const std::exception& x) {
                            read_ok = false;
                            e = std::string("the prompt read failed: ") + x.what();
                        } catch (...) {
                            read_ok = false;
                            e = "the prompt read failed";
                        }
                        done.store(true, std::memory_order_release);
                        input.wake();   // (the loop below may be waiting for a line)
                    });
                    bool step_ok = true;
                    int64_t windows0 = bt_windows;
                    const auto tc = Clock::now();
                    while (!done.load(std::memory_order_acquire)) {
                        if (step_ok && batch_on()) {
                            step_ok = batch_step();
                        } else {
                            input.wait_for_line(std::chrono::milliseconds(2),
                                                [&] { return done.load(std::memory_order_acquire); });
                        }
                        // a BSTOP that came meanwhile ends its slot at its next window
                        input.take_slot_controls(stop_slot, [](int) { return false; });
                    }
                    reader.join();
                    in_prompt_read = false;
                    conc_read = false;
                    std::fprintf(stderr, "strata batch: read %lld tokens on a thread of its own in %.0f ms; the slots ran %lld windows meanwhile\n",
                                 (long long) (r - q), std::chrono::duration<double, std::milli>(Clock::now() - tc).count(),
                                 (long long) (bt_windows - windows0));
                    if (!step_ok) {
                        batch_fatal = true;
                        e = "a batch window failed";
                        return false;
                    }
                    return read_ok;
                };
                // (a layer split reads its stages as a pipeline over one run's chunks: beside slots a chunk at a time, with
                // none decoding in pieces of STRATA_BATCH_YIELD_CHUNKS chunks (default 8; 0 = one run) so a BYIELD is taken)
                static const int64_t yield_chunks = [] {
                    const char* v = std::getenv("STRATA_BATCH_YIELD_CHUNKS");
                    return v != nullptr ? (int64_t) std::max(0, std::atoi(v)) : (int64_t) 8;
                }();
                // a piece's chunk: the buffers' chunk, or with a plan as many whole planned chunks as that holds (the
                // stages still pipeline inside a piece, and every piece starts on the plan's grid)
                const int64_t C = planned ? std::max<int64_t>(1, sp.chunk() / plan.chunk) * plan.chunk
                                          : std::max<int64_t>(sp.chunk(), 1);
                const int64_t C_read = planned ? plan.chunk : C;   // what the yield test calls a chunk
                // BYIELD <ys> with the read stopped at q (nothing in flight): the part read waits in slot ys and the
                // read ends with e = "yield" (true); false when it cannot give way there (the read goes on).  Only
                // where the rest is read the same way after it (more than a chunk and more than a short read left),
                // text only, and into a slot that is not decoding.
                auto give_way = [&](int ys, int64_t q, std::string& err) -> bool {
                    const bool can = ys < (int) bs.size() && !bs[(size_t) ys].active &&
                                     (admit_slot < 0 || ys == admit_slot) && req_imgs.empty() && o.prompt_cache > 0 &&
                                     b0 - q > std::max<int64_t>(C_read, o.short_read);
                    std::string ye;
                    const auto ty = Clock::now();
                    std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
                    if (can) park_slot(ys);   // the conversation the slot holds, before this read overwrites it
                    if (can && copy_slot(true, ys, pre, nullptr, ye)) {
                        BSlot& sl = bs[(size_t) ys];
                        sl = BSlot{};
                        sl.ids = std::move(pre);
                        sl.cached = true;
                        sl.partial = true;
                        sl.partial_from0 = read_from == 0 || resumed_from0;
                        if (planned) sl.partial_plan = plan;   // the resend reads the rest in these chunks
                        sl.cvec = cvec_cached;
                        for (const ConvCheckpoint& c : checks)   // the root / periodic checkpoints of this read
                            if ((int64_t) c.ids.size() <= q && std::equal(c.ids.begin(), c.ids.end(), ids.begin(),
                                    [](int32_t x, int64_t y) { return (int64_t) x == y; }))
                                sl.checks.push_back(c);
                        yielded_at = q;
                        yielded_slot = ys;
                        std::fprintf(stderr, "strata batch: the prompt read gives way at %lld of %lld tokens; slot %d "
                                             "holds it (copied in %.1f ms)\n", (long long) q, (long long) n, ys,
                                     std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                        err = "yield";
                        return true;
                    }
                    if (can) bs[(size_t) ys].cached = false;   // half written
                    std::fprintf(stderr, "strata batch: BYIELD %d not taken at %lld of %lld%s%s\n", ys, (long long) q,
                                 (long long) n, ye.empty() ? "" : ": ", ye.c_str());
                    return false;
                };
                // Opt-in, experimental: STRATA_PIPED_CONCURRENT=1 reads the prompt on a thread of its own while
                // this thread keeps the slot groups decoding through the pipeline (pump), so they never stop for it
                // and the read has no drain inside it.  The prompt and the windows use their own streams and memory
                // (the prompt's K/V goes to the main sessions, the slots' to theirs; the PLE table is locked), and
                // the adaptive tier stands still (its round landed before the read).  The slots the prompt borrowed
                // were lent before this read with nothing in flight and are refilled after it, so the windows see
                // them missing and compute those experts on the CPU, as between pieces.  A BYIELD ends the run at
                // its next chunk (Prefill::stop_at_chunk): the pipeline is drained there and the part read gives
                // way as below, or the read goes on from there.
                // Not where a read's thread allocates or frees (the idle stage's help, STRATA_PREFILL_HELP; the batched
                // draft K/V, STRATA_SPLIT_MTP_BATCH: a free waits for the device), where serving a doorbell makes a
                // CUDA call (--pcie-mode dma, the Foresight swap space STRATA_FS_SLOTS, the helper caches), and where
                // the expert source recycles the buffers it hands out (the GGUF in place, an unbuffered source, the
                // I/O read-ahead: one thread at a time).  Pictures are read the usual way (their rows go up through
                // pinned slots of a size known only then).
                static const bool pr_conc = [&] {
                    const char* v = std::getenv("STRATA_PIPED_CONCURRENT");
                    if (v == nullptr || std::atoi(v) == 0) return false;
                    auto env_on = [](const char* k) {
                        const char* x = std::getenv(k);
                        return x != nullptr && std::atoi(x) != 0;
                    };
                    strata::program::ConcurrentReadSetup s;
                    s.prefill_help = env_on("STRATA_PREFILL_HELP");
                    s.split_mtp_batch = env_on("STRATA_SPLIT_MTP_BATCH");
                    s.fs_slots = env_on("STRATA_FS_SLOTS");
                    s.pcie_dma = o.pcie_mode == "dma";
                    // (--remote-expert-opt works only through a helper cache, so without one it does not count)
                    s.helper_caches = prefill_src != srcp;
                    for (int r = 0; r < 3; ++r) s.helper_caches |= o.expert_cache_remote[(size_t) r] > 0;
                    s.staged_source = srcp == &src && src.stages_blobs();
                    s.other_source = srcp != &src && srcp != &arena_src;
                    if (const char* why = strata::program::concurrent_read_off(s)) {
                        std::fprintf(stderr, "strata batch: STRATA_PIPED_CONCURRENT is off beside %s\n", why);
                        return false;
                    }
                    return true;
                }();
                // STRATA_PIPED_CONCURRENT_MIN=N (opt-in, with STRATA_PIPED_CONCURRENT=1): only a read of at least N tokens
                // runs beside the decoding slots on its own thread; a shorter one takes the usual path (drained windows,
                // no priority streams' cost on the short prompts' TTFT and throughput).  Unset or 0: every read, as before.
                static const int64_t pr_conc_min = [] {
                    const char* v = std::getenv("STRATA_PIPED_CONCURRENT_MIN");
                    return v != nullptr ? (int64_t) std::max(0LL, std::atoll(v)) : (int64_t) 0;
                }();
                if (piped && pr_conc && req_imgs.empty() && (b0 - a0) >= pr_conc_min) {   // (with no slot decoding it only watches for a BYIELD)
                    // the drafter's prompt pass captures its graphs and grows its records at first use, and the
                    // prompt path's pinned upload slots grow at their first larger upload: done here, with nothing in
                    // flight (on the reader's thread a capture, an allocation or a free takes the driver's lock and
                    // can wait for the device, #31)
                    if (use_mtp && !mtp.reserve_prefill(C, e)) return false;
                    if (!sp.reserve_uploads(C))
                        std::fprintf(stderr, "strata batch: the prompt path's pinned upload slots could not be "
                                             "reserved (they grow at first use)\n");
                    // where the read's thread and the doorbell thread run: on Linux a thread inherits its creator's
                    // CPUs, and the host thread is pinned to the one core the pool leaves it, so both would share that
                    // core with the host thread that pumps the windows - every window then waits its turn there for
                    // the doorbell of each layer (4x RTX 5080, 4 stages: 6 rows/s during a read).  They move off the
                    // host's core and the pool workers' CPUs instead (--aux-cpus places them, when it is on).
                    // STRATA_PIPED_CONCURRENT_UNPIN=0 keeps the inherited CPUs (an A/B); STRATA_TEST_CONC_ON_HOST_CPU=1
                    // (Windows, a test) puts them on the host thread's CPUs first, as Linux does.
                    static const bool conc_unpin = [] {
                        const char* v = std::getenv("STRATA_PIPED_CONCURRENT_UNPIN");
                        return v == nullptr || std::atoi(v) != 0;
                    }();
#if defined(_WIN32)
                    static const bool conc_on_host = std::getenv("STRATA_TEST_CONC_ON_HOST_CPU") != nullptr;
                    std::vector<ULONG> host_cpu_sets;
                    if (conc_on_host) {
                        const HANDLE self = GetCurrentThread();
                        ULONG count = 0;
                        if (GetThreadSelectedCpuSets(self, nullptr, 0, &count) ||
                            GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                            host_cpu_sets.resize(count);
                            if (count && !GetThreadSelectedCpuSets(self, host_cpu_sets.data(), count, &count))
                                host_cpu_sets.clear();
                        }
                        static bool said = false;
                        if (!said) {
                            said = true;
                            std::fprintf(stderr, "strata batch: test: the read's and the doorbell threads start on the "
                                                 "host thread's %zu CPU set(s)%s\n", host_cpu_sets.size(),
                                         conc_unpin ? ", then leave them (the fix)"
                                                    : " and stay there (as on Linux)");
                        }
                    }
#endif
                    // the threads that already inherited the host's core (the prompt path's copy threads, started by
                    // the first prompt on the host thread) work during the read too: off that core as well (Linux)
                    if (conc_unpin && !strata::aux_cpus::enabled())
                        if (const int moved = strata::aux_cpus::release_host_inheritors(); moved > 0)
                            std::fprintf(stderr, "strata batch: %d threads that ran on the host thread's core moved off "
                                                 "it, for reads beside the slots\n", moved);
                    auto conc_thread_cpus = [&] {
                        strata::aux_cpus::pin_current_thread();
#if defined(_WIN32)
                        const HANDLE self = GetCurrentThread();
                        if (conc_on_host && !host_cpu_sets.empty())
                            SetThreadSelectedCpuSets(self, host_cpu_sets.data(), (ULONG) host_cpu_sets.size());
                        if (conc_unpin && conc_on_host) SetThreadSelectedCpuSets(self, nullptr, 0);
#else
                        if (conc_unpin && !strata::aux_cpus::enabled()) (void) strata::aux_cpus::place_off_host();
#endif
                    };
                    // the slots' doorbells are served on a thread of their own for the read, which makes no CUDA
                    // call: a call here or on the read's threads can wait for the driver's lock while its holder
                    // waits for the GPU (a free; a module the first kernel of a shape loads), and a window on that
                    // GPU waits for its doorbell (#31).  Ends (and hands the service back to pump) on every exit.
                    struct Doorbells {
                        std::atomic<bool> on{true};
                        std::thread th;
                        std::function<void()> wake, off;
                        ~Doorbells() {
                            on.store(false, std::memory_order_release);
                            if (wake) wake();
                            if (th.joinable()) th.join();
                            if (off) off();
                        }
                    } bells;
                    for (int k = 0; k < n_pipe; ++k) stage_verifier(k).set_external_serve(true);
                    conc_serving = true;
                    bells.off = [&] {
                        for (int k = 0; k < n_pipe; ++k) stage_verifier(k).set_external_serve(false);
                        conc_serving = false;
                    };
                    bells.wake = [&] {
                        std::lock_guard<std::mutex> lk(bell_mu);
                        bell_cv.notify_one();
                    };
                    bells.th = std::thread([&] {
                        conc_thread_cpus();
                        auto any_busy = [&] {
                            for (int k = 0; k < n_pipe; ++k)
                                if (stage_verifier(k).batch_busy()) return true;
                            return false;
                        };
                        while (bells.on.load(std::memory_order_acquire)) {
                            bool busy = false;
                            for (int k = 0; k < n_pipe; ++k) {
                                auto& vk = stage_verifier(k);
                                if (vk.batch_busy()) {
                                    busy = true;
                                    vk.batch_serve(win_pool_fn, win_pool_user);
                                }
                            }
                            if (busy) continue;
                            // STRATA_TEST_BELL_SLEEP_US=N (a test, an A/B of the wait below): sleep N us instead, as
                            // the first form of this thread did (a whole timer tick on Windows)
                            static const int test_sleep_us = [] {
                                const char* t = std::getenv("STRATA_TEST_BELL_SLEEP_US");
                                return t != nullptr ? std::max(0, std::atoi(t)) : 0;
                            }();
                            if (test_sleep_us > 0) {
                                std::this_thread::sleep_for(std::chrono::microseconds(test_sleep_us));
                                continue;
                            }
                            // no window in flight: wait for pump's next launch (it notifies when this is set; the
                            // fences pair the flag with batch_launch's running flag, so one of the two sides sees the
                            // other), or for the end of the read
                            std::unique_lock<std::mutex> lk(bell_mu);
                            bell_waiting.store(true, std::memory_order_relaxed);
                            std::atomic_thread_fence(std::memory_order_seq_cst);
                            bell_cv.wait_for(lk, std::chrono::milliseconds(2),
                                             [&] { return !bells.on.load(std::memory_order_acquire) || any_busy(); });
                            bell_waiting.store(false, std::memory_order_relaxed);
                        }
                    });
                    for (int64_t q0 = a0; q0 < b0;) {
                        const auto tc = Clock::now();
                        const int64_t rows0 = pipe_rows;
                        // per stage, for the read's log line: its windows and the host time that set them up (the
                        // PLE gather included, which waits while the prompt holds the table) - both this thread's
                        std::vector<std::pair<int64_t, double>> st0((size_t) n_pipe);
                        for (int k = 0; k < n_pipe; ++k)
                            st0[(size_t) k] = {stage_verifier(k).windows, stage_verifier(k).ms_host};
                        std::atomic<bool> done{false};
                        std::atomic<int> want_yield{-1};
                        bool read_ok = true;
                        std::string read_err;
                        in_prompt_read = true;
                        sp.stop_at_chunk = [&] { return want_yield.load(std::memory_order_acquire) >= 0; };
                        share_gate();   // on this thread, before the reader starts
                        std::thread reader([&] {
                            conc_thread_cpus();
                            try {   // (an exception leaving a std::thread ends the process: a failed read instead)
                                if (planned) sp.force_chunk(plan.chunk, plan_single, plan.chunked);
                                read_ok = sp.run(ids.data() + q0, b0 - q0, q0, read_err);
                            } catch (const std::exception& x) {
                                read_ok = false;
                                read_err = std::string("the prompt read failed: ") + x.what();
                            } catch (...) {
                                read_ok = false;
                                read_err = "the prompt read failed";
                            }
                            done.store(true, std::memory_order_release);
                            input.wake();   // (the loop below may be waiting for a line)
                        });
                        struct ReadEnd {   // on every way out of this round (an exception too): the reader joined first
                            std::thread& t;
                            std::function<bool()>& hook;
                            bool& reading;
                            ~ReadEnd() {
                                if (t.joinable()) t.join();
                                hook = nullptr;
                                reading = false;
                            }
                        } read_end{reader, sp.stop_at_chunk, in_prompt_read};
                        bool pump_ok = true;
                        while (!done.load(std::memory_order_acquire)) {
                            const bool decoding = pump_ok && (batch_on() || pipe_inflight());
                            if (decoding) pump_ok = pump(true);
                            else   // nothing to pump: until the read ends or a line comes (both notify)
                                input.wait_for_line(std::chrono::milliseconds(2),
                                                    [&] { return done.load(std::memory_order_acquire); });
                            // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD ends the read at
                            // its next chunk
                            input.take_slot_controls(stop_slot, [&](int ys) {
                                if (want_yield.load() >= 0) return false;
                                want_yield.store(ys, std::memory_order_release);
                                return true;
                            });
                        }
                        reader.join();   // (a failed window leaves the read to finish: it is not stopped mid-run)
                        sp.stop_at_chunk = nullptr;
                        in_prompt_read = false;
                        if (!pump_ok || !pipe_drain()) {
                            batch_fatal = true;
                            e = "a pipelined batch window failed";
                            return false;
                        }
                        const int64_t q = read_ok && sp.stopped_at() >= 0 ? q0 + sp.stopped_at() : b0;
                        const double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - tc).count();
                        std::string per;
                        for (int k = 0; k < n_pipe; ++k) {
                            const auto& v = stage_verifier(k);
                            const int64_t w = v.windows - st0[(size_t) k].first;
                            char b[96];
                            const double host = w > 0 ? (v.ms_host - st0[(size_t) k].second) / (double) w : 0.0;
                            std::snprintf(b, sizeof b, "%sstage %d: %lld windows, %.1f ms host each", k ? "; " : "",
                                          k + 1, (long long) w, host);
                            per += b;
                        }
                        std::fprintf(stderr, "strata batch: read %lld tokens on a thread of its own in %.0f ms "
                                             "(concurrent, to %lld of %lld); the slots decoded %lld rows meanwhile "
                                             "(%.1f rows/s; %s)\n",
                                     (long long) (q - q0), read_ms, (long long) q, (long long) n,
                                     (long long) (pipe_rows - rows0),
                                     read_ms > 0.0 ? 1000.0 * (double) (pipe_rows - rows0) / read_ms : 0.0,
                                     per.c_str());
                        if (!read_ok) {
                            e = read_err;
                            return false;
                        }
                        if (q < b0 && give_way(want_yield.load(), q, e)) return false;
                        q0 = q;   // done, or the give-way was not taken there: the rest goes on the same way
                    }
                    return true;
                }
                // Opt-in (#1906): on the pipelined slot groups a read goes in pieces of STRATA_PIPED_READ_PIECE tokens
                // (whole chunks).  A piece's end has the pipeline drained, so a waiting shorter request can take
                // over there (BYIELD); with slots decoding, the groups then decode through the pipeline (pump) for
                // STRATA_BATCH_DECODE_SHARE x the piece's time, capped / floored by STRATA_PIPED_DECODE_MAX_MS /
                // _MIN_MS, and are drained (pipe_drain) before the next piece, as before any read.  Unset: one run.
                static const int64_t pr_piece = [] {
                    const char* v = std::getenv("STRATA_PIPED_READ_PIECE");
                    return v != nullptr ? (int64_t) std::max(0, std::atoi(v)) : (int64_t) 0;
                }();
                static const double pr_cap = [] {
                    const char* v = std::getenv("STRATA_PIPED_DECODE_MAX_MS");
                    return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
                }();
                static const double pr_floor = [] {
                    const char* v = std::getenv("STRATA_PIPED_DECODE_MIN_MS");
                    return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
                }();
                const bool pr_on = piped && pr_piece > 0;
                auto piece_now = [&]() -> int64_t {
                    return pr_on ? strata::program::piped_read_piece(pr_piece, C)
                                 : strata::program::batch_read_piece(o.batch > 0, piped, !stages.empty(), batch_on(), C,
                                                                     yield_chunks);
                };
                // the slots decode through the pipeline for budget_ms, drained before the next read
                auto piped_decode = [&](double budget_ms, std::string& err) -> bool {
                    const auto td = Clock::now();
                    in_prompt_read = true;
                    do {
                        if (!pump(true)) {
                            in_prompt_read = false; batch_fatal = true; err = "a pipelined batch window failed";
                            return false;
                        }
                    } while (batch_on() &&
                             std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget_ms);
                    if (!pipe_drain()) {
                        in_prompt_read = false; batch_fatal = true; err = "a pipelined batch window failed";
                        return false;
                    }
                    in_prompt_read = false;
                    ++il_parts;
                    il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                    return true;
                };
                // prompts arriving back to back: one decode budget before a read that follows a piecewise read
                static Clock::time_point pr_prev_end{};
                static bool pr_have_prev = false;
                static double pr_prev_piece_ms = 0.0;   // the previous read's last full piece
                if (pr_on) {
                    const double lead =
                        pr_prev_piece_ms > 0.0
                            ? strata::program::piped_decode_budget_ms(pr_prev_piece_ms, decode_share, pr_cap, pr_floor)
                            : (pr_cap > 0.0 ? pr_cap : std::max(pr_floor, 200.0));
                    const double since =
                        pr_have_prev ? std::chrono::duration<double, std::milli>(Clock::now() - pr_prev_end).count()
                                     : -1.0;
                    if (strata::program::piped_read_lead_in(true, batch_on(), since, lead)) {
                        const int64_t rows0 = pipe_rows;
                        const auto tl = Clock::now();
                        if (!piped_decode(lead, e)) return false;
                        std::fprintf(stderr, "strata batch: a read right after a read: the slots decoded %lld rows in "
                                             "%.0f ms first\n", (long long) (pipe_rows - rows0),
                                     std::chrono::duration<double, std::milli>(Clock::now() - tl).count());
                    }
                }
                int64_t piece = piece_now();
                if (piece <= 0) return run(a0, b0);
                for (int64_t q = a0; q < b0;) {
                    const int64_t r = std::min(b0, q + piece);
                    const auto tq = Clock::now();
                    if (!run_conc(q, r)) return false;
                    const double piece_ms = std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                    if (pr_on && r < b0) pr_prev_piece_ms = piece_ms;
                    const int64_t piece_tokens = r - q;
                    q = r;
                    if (q >= b0) continue;
                    int ys = -1;
                    // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                    input.take_slot_controls(stop_slot, [&](int y) {
                        ys = y;
                        return true;
                    });
                    if (ys >= 0 && give_way(ys, q, e)) return false;
                    // (on the pipelined path a piece's end has the pipeline drained - nothing runs during a piece and
                    // every decode budget ends in pipe_drain - so the part read can give way there as off the pipeline)
                    if (pr_on) {   // the pipelined pieces: the slots decode through the pipeline, then it is drained
                        if (!batch_on()) {
                            std::fprintf(stderr, "strata batch: piped read %lld tokens in %.0f ms (to %lld of %lld)\n",
                                         (long long) piece_tokens, piece_ms, (long long) q, (long long) n);
                            continue;
                        }
                        const double budget =
                            strata::program::piped_decode_budget_ms(piece_ms, decode_share, pr_cap, pr_floor);
                        const int64_t rows0 = pipe_rows;
                        const auto td = Clock::now();
                        if (!piped_decode(budget, e)) return false;
                        std::fprintf(stderr, "strata batch: piped read %lld tokens in %.0f ms (to %lld of %lld), "
                                             "then the slots decoded %lld rows in %.0f ms\n", (long long) piece_tokens,
                                     piece_ms, (long long) q, (long long) n, (long long) (pipe_rows - rows0),
                                     std::chrono::duration<double, std::milli>(Clock::now() - td).count());
                        continue;
                    }
                    if (decode_share <= 0.0 || !batch_on()) continue;
                    const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                    const auto td = Clock::now();
                    in_prompt_read = true;
                    do {
                        if (!batch_step()) { in_prompt_read = false; batch_fatal = true; e = "a batch window failed"; return false; }
                    } while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
                    in_prompt_read = false;
                    ++il_parts;
                    il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                    if (trace) {
                        std::fprintf(stderr, "strata trace: prompt chunk to %lld, then the slots decoded %.0f ms\n",
                                     (long long) q, std::chrono::duration<double, std::milli>(Clock::now() - td).count());
                        std::fflush(stderr);
                    }
                }
                if (pr_on) { pr_prev_end = Clock::now(); pr_have_prev = true; }   // for the next read's lead-in
                return true;
            };
            apply_pending(true);
            const auto drain_t0 = Clock::now();
            const bool drained_ok = adapt_tick(true);
            // 149: how long the request waited for the adaptive round still in flight (in the prompt's time)
            serve_adapt_drain_ms = std::chrono::duration<double, std::milli>(Clock::now() - drain_t0).count();
            if (!drained_ok) {   // --adapt-async: the round in flight lands before the prompt lends any slot
                print_response("ERR an adaptive refill failed\n");
                return 1;
            }
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            const strata::kernels::SamplerParams req_sp = get_sampler_params(smp);
            if (!ver.set_logit_bias(req_bias, err) || (pipe && !ver_b.set_logit_bias(req_bias, err))) {
                print_response("ERR %s\n", err.c_str());
                return 1;
            }
            ver.set_sampling(req_sp);
            if (pipe) ver_b.set_sampling(req_sp);   // (reaches the later stage's odd verifier)
            if (use_mtp) mtp.set_draft_sampling(req_sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
            const bool gated_proposal = use_mtp && mtp.prob() && strata::core::spec_gate_pick() &&
                gated_proposal_usable(o.pipeline_windows < 2 && o.batch == 0 && n_stages == 1 && !multi_gpu && drive.d.remote_count == 0,
                                      o.lookup_chain, o.adapt_swaps, req_pcie_frac != 0.0 || o.pcie_frac_auto, wplan.cost_aware || req_df2, S,
                                      req_spec_min_p);
            {   // aux_cpus=0|1: where the non-pool, non-host threads run for this request (-1 = the start-up --aux-cpus); moves
                // threads only.  An enabled placement is swept at every request: it catches threads created since.
                static const bool aux_start = strata::aux_cpus::enabled();
                const bool was_on = strata::aux_cpus::enabled();
                const bool want = req_aux_cpus < 0 ? aux_start : req_aux_cpus != 0;
                strata::aux_cpus::set_enabled(want);
                if (want != was_on)
                    std::fprintf(stderr, "strata serve: aux cpus %s for this request%s\n", strata::aux_cpus::enabled() ? "ON" : "off",
                                 want && !strata::aux_cpus::enabled() ? " (asked for, but no CPU is spare: see the start-up line)" : "");
            }
            ver.pause_pcie_balance(req_pcie_set);   // the measured costs keep across requests
            drive.d.pcie_num = get_pcie_num(req_pcie_frac);
            // a layer split: CUDA0's share as asked; a later GPU keeps its own (its link) unless the request sets one
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same || req_pcie_frac != o.pcie_frac)
                                               ? drive.d.pcie_num : get_pcie_num(stages[(size_t) st - 1]->pcie_frac);
            if (pipe) for (int st = 0; st < split_drive.n; ++st) split_drive_b.pcie_num[st] = split_drive.pcie_num[st];
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            // #4590: the repeat / frequency / presence penalties count the tokens this reply has written, not the prompt's last
            // tokens (the prompt's own numbers and quoted answers were penalised, and a conversation whose last turn asks for
            // something new could get its previous answer back).  Earlier assistant turns inside the prompt are not counted
            // either.  STRATA_PENALTY_PROMPT=1 brings the old window back (the last penalty_last_n tokens of prompt + reply).
            // The history is rebuilt from the request's prompt ids every time, so a conversation restored from the cache or a
            // saved slot seeds it exactly as a cold read does.
            static const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            if (pipe) ver_b.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            bool cancelled = false;
            tr("prompt start", n - 1);
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            int64_t turn_at = o.prompt_cache > 0 ? strata::core::find_prompt_turn_boundary(
                ids, resume, o.turn_token, o.tail_role_token) : -1;
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
            // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
            // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
            // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
            // is cheaper to read again than the extra part costs (~0.3 s).
            const int64_t last_turn = turn_at;   // ckpt=0 drops the split there, the root is still looked for before it
            if (!req_ckpt) turn_at = -1;
            const int64_t root_at = o.prompt_cache > 0 && (read_from == 0 || resumed_from0)
                ? strata::core::find_prompt_root_boundary(ids, last_turn, o.turn_token, o.prompt_cache_root) : -1;
            static const bool message_checkpoint = [] {
                const char* e = std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");
                return e != nullptr && std::atoi(e) != 0;
            }();
            // Only add a snapshot; the existing token and image checks still decide reuse.
            const int64_t message_at = message_checkpoint && !multi_gpu && o.prompt_cache > 0
                ? strata::program::message_checkpoint_boundary(ids, resume, turn_at, o.turn_token) : -1;
            // pin=N (opt-in): one more place the prompt is read to and checkpointed.  One pinned prefix at a time: a
            // new pin replaces the old; it needs room for the root, the pin and a rotating checkpoint
            int64_t pin_at = -1;
            auto pin_checkpoint = [&](int64_t L) {
                bool found = false;
                for (ConvCheckpoint& c : checks) {
                    c.pinned = (int64_t) c.ids.size() == L;
                    found = found || c.pinned;
                }
                return found;
            };
            if (req_pin > 0) {
                if (o.prompt_cache < 3 || !req_ckpt)
                    std::fprintf(stderr, "strata serve: pin=%lld ignored: %s\n", (long long) req_pin,
                                 req_ckpt ? "it needs --prompt-cache 3 or more" : "ckpt=0 keeps no checkpoint");
                else if (req_pin >= n - 1)
                    std::fprintf(stderr, "strata serve: pin=%lld ignored: the prefix must be shorter than the prompt "
                                 "(%lld tokens)\n", (long long) req_pin, (long long) n);
                else if (req_pin > read_from) pin_at = req_pin;
                else if (!pin_checkpoint(req_pin))
                    std::fprintf(stderr, "strata serve: pin=%lld: the cache has no checkpoint there to pin (it resumed from "
                                 "%lld)\n", (long long) req_pin, (long long) read_from);
            }
            std::vector<int64_t> cuts = {reread_to, root_at, message_at, turn_at, n - 1};
            if (pin_at >= 0) {
                cuts.push_back(pin_at);
                std::sort(cuts.begin(), cuts.end());   // the skipped -1s first, n - 1 still last
            }
            int64_t at = read_from;
            // STRATA_ADMIT_MERGE_FIRST: the last part, when it goes through the windows, leaves its last tokens (up to
            // max_t with the last prompt token) to the first window, which starts at merge_at (-1: the first window
            // holds the last prompt token alone).  Only where every window row is the same in any window: a batch
            // admission's first token, every expert of every stage in VRAM (the zero-doorbell graphs, no loan, no swap)
            // and the serial loop with nothing else reading its windows
            int64_t merge_at = -1;
            auto merge_from = [&](int64_t a0, int64_t to) -> int64_t {
                if (!admit_merge_first_env() || admit_slot < 0 || max_new != 1 || batch_mtp || pipe) return -1;
                if (hist_n > 0 && !pen_prompt) return -1;   // #4590: the merged window's prompt rows would join the reply-only penalty window
                if (to != n - 1 || to == turn_at || to == root_at || to == message_at || to == pin_at) return -1;
                if (!req_imgs.empty() || kvg.on || peer.valid() || remote_caches || remote_opt || fs_swap) return -1;
                if (std::getenv("STRATA_LOGPOS") != nullptr) return -1;
                if (host_res.empty() || !pending.empty() ||
                    std::any_of(host_res.begin(), host_res.end(), [](int32_t r) { return r < 0; }))
                    return -1;
                int mt = strata::kernels::kVerifyMaxT;
                for (int k = 0; k < n_pipe; ++k) {
                    if (!stage_verifier(k).ar_live()) return -1;
                    mt = std::min(mt, stage_verifier(k).max_t());
                }
                if (mt < 2) return -1;
                return std::max(a0, n - mt);
            };
            for (const int64_t to : cuts) {
                if (to <= at) continue;
                err.clear();
                const bool win = windows_ok(at, to);
                if (win && !refill(err)) {
                    print_response("ERR refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                // a read going on in a planned segment gets the loan it was read with (what is left would size
                // another, and the loan decides which experts are read in place: bits)
                const int64_t plan_lend = resume_plan.chunk > 0 && at == resume_plan_at && to == resume_plan_end
                                              ? resume_plan.lend : 0;
                if (!win && !lend(to - at, err, plan_lend)) {
                    print_response("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                    return 1;
                }
                if (win) merge_at = merge_from(at, to);   // (after the refill: the loans are back)
                const int64_t read_to = merge_at >= 0 ? merge_at : to;   // (merge_at > at: the part's windows before it)
                const auto tsp = Clock::now();
                pp_exact = false;   // a batched or pipelined part is in flight: a cancel may leave the state past pp_reached
                const bool sp_ok = read_to <= at || (win ? read_windows(at, read_to, err) : read_part(at, to, err));
                if (sp_ok) pp_exact = true;
                if (trace) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - tsp).count();
                    (win ? req_times.windows_ms : req_times.read_ms) += ms;
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (read_to - at),
                                 win ? "windows" : "batched", ms);
                    if (merge_at >= 0)
                        std::fprintf(stderr, "strata trace: the first window takes the last %lld prompt tokens\n",
                                     (long long) (n - merge_at));
                    std::fflush(stderr);
                }
                if (!sp_ok && nf_hit) {   // #879: DONE nonfinite (the lent slots are refilled below)
                    cancelled = true;
                    break;
                }
                if (sp_ok && merge_at >= 0) {   // the server's progress reaches the prompt's end, as after its last window
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                    print_response("PP %lld %lld %.0f %.1f\n", (long long) to, (long long) pp_total, ms,
                                ms > 0.0 ? 1000.0 * (double) (to - pp_from) / ms : 0.0);
                    std::fflush(stdout);
                }
                if (!sp_ok) {
                    if (yielded_at < 0 && (batch_fatal || !input.is_stop_requested())) {
                        std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                        print_response("ERR %s\n", err.c_str());
                        // #224: a CUDA fault (an illegal address) poisons the context for the whole process, and
                        // unwinding the destructors on it could hang until the 60 s watchdog: leave at once
                        if (cudaPeekAtLastError() != cudaSuccess) {
                            std::fflush(stdout);
                            std::fflush(stderr);
                            std::_Exit(1);
                        }
                        return 1;
                    }
                    cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                    break;
                }
                at = to;
                if ((to == turn_at || to == root_at || to == message_at || to == pin_at) &&
                    !checkpoint_at(to, nullptr, to == message_at && to != turn_at && to != root_at && to != pin_at)) {   // the tail kind leaves first
                    if (!ckpt_why.empty() || !checkpoint_failure_is_survivable()) {   // #3350: only an unwell GPU ends the request
                        print_response("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                        return 1;
                    }
                    std::fprintf(stderr, "strata serve: checkpoint at %lld skipped: its state could not be copied out (see "
                                         "above); the request goes on without it\n", (long long) to);
                }
                if (to == pin_at && !pin_checkpoint(pin_at))
                    std::fprintf(stderr, "strata serve: pin=%lld: its checkpoint was not kept\n", (long long) pin_at);
                if (trace && to == message_at)
                    std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                 (long long) message_at, (long long) (turn_at - message_at));
            }
            if (!refill(err)) {
                print_response("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            if (mtp_room_open && room_close("prompt done") != 0) return 1;
            if (head_restore() != 0) return 1;
            tr("prompt done (slots refilled)");
            if (il_parts > 0)
                std::fprintf(stderr, "strata batch: the prompt was read in %lld parts, the slots decoding %.0f ms between "
                                     "them\n", (long long) il_parts + 1, il_ms);
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            print_response("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token alone (STRATA_ADMIT_MERGE_FIRST: the prompt
            // tokens from merge_at on, the last one's output its first token)
            const int merge_T = merge_at >= 0 && !cancelled ? (int) (n - merge_at) : 0;
            int64_t p = merge_T > 0 ? merge_at : n - 1;
            int32_t x = (int32_t) ids[(size_t) p];
            // (the drafter writes max_t - 1 drafts: 8 rows with --pipeline-windows 2)
            const size_t DS = (size_t) std::max(S, use_mtp ? mtp.max_t() : 0);
            const size_t WS = (size_t) std::max(S, merge_T);
            std::vector<int32_t> drafts(DS, 0), window(WS), outv(WS);
            std::vector<float> dprob(DS, 0.0f);
            wplan.start(ids);
            // STRATA_SPEC_TRACE=<file> in serve (bench only, off by default): per round "R p T a sfx ms k match | probs | drafts | lookup draft | committed tokens"
            static std::FILE* const df2_trace = [] { const char* tf = std::getenv("STRATA_SPEC_TRACE"); return tf != nullptr && tf[0] != '\0' ? std::fopen(tf, "a") : nullptr; }();
            static const bool trace_nopick = std::getenv("STRATA_SPEC_TRACE_NOPICK") != nullptr;   // the lookup draft is logged but never taken
            std::array<int32_t, 8> tr_d{};
            std::array<int32_t, 16> tr_s{};
            int tr_k = 0, tr_match = 0;
            // This request's source planner: the live policy at its start, then only committed outcomes (no round
            // times, no rows past the accepted ones: their Philox counters are drawn again by the next rounds).
            wplan.source = wplan.policy;
            strata::core::MtpDrafter::DraftResult generation;
            strata::core::gated::Packet proposal; // immutable through the synchronous verifier call
            bool first_window = true;
            int64_t produced_n = 0;
            // STRATA_FOLLOW=<file> (token forcing, measurement only, off by default; tools/force_check.py): one line per request in the order the
            // engine decodes them, comma-separated token ids (an empty line: that request is not forced).  The serial decode windows below
            // take the line's tokens in place of the head's argmax, so drafts are accepted exactly when they equal the forced continuation, and
            // STRATA_FOLLOW_LOGPOS=<file> gets one STRATA_LOGPOS row (FP32 head logits, STRATA_LOGPOS_TOPK) for every row whose logits are valid
            // (each accepted row plus the one that follows): "position, forced token, logp(forced), argmax, logp(argmax), match, ..."
            static const std::vector<std::vector<int32_t>> fol_sets = [] {
                std::vector<std::vector<int32_t>> v;
                if (const char* f = std::getenv("STRATA_FOLLOW"); f != nullptr && f[0] != '\0') {
                    std::ifstream in(f);
                    std::string line;
                    while (std::getline(in, line)) {
                        std::vector<int32_t> ids;
                        for (const char* q = line.c_str(); *q != '\0';) {
                            if (*q >= '0' && *q <= '9') { ids.push_back((int32_t) std::atoi(q)); while (*q >= '0' && *q <= '9') ++q; }
                            else ++q;
                        }
                        v.push_back(std::move(ids));
                    }
                }
                return v;
            }();
            static size_t fol_next = 0;
            static std::FILE* const fol_lp = [] { const char* p = std::getenv("STRATA_FOLLOW_LOGPOS"); return p != nullptr && p[0] != '\0' ? std::fopen(p, "ab") : nullptr; }();
            const std::vector<int32_t>* fol = nullptr;
            if (!fol_sets.empty()) {
                if (fol_next < fol_sets.size() && !fol_sets[fol_next].empty()) fol = &fol_sets[fol_next];
                ++fol_next;
            }
            // Gets how many of a window's `a` accepted drafts (the target's picks `picks`) to commit: on HIP none whose
            // output this request cannot emit (past max_new or an end-of-text token: core/serve_stop_bound.hpp), so they
            // are not cached.  On by default where it was measured (HIP: 4 x R9700, Qwen3.8-Flash-Next IQ3_S, 0 of 16
            // continuations fell back to the older checkpoint against 4 of 16 without it, #4677);
            // STRATA_SERVE_STOP_BOUND=0 turns it off.
            auto stop_bounded = [&](int a, const int32_t* picks) -> int {
#if defined(STRATA_USE_HIP)
                static const bool stop_bound = [] {
                    const char* v = std::getenv("STRATA_SERVE_STOP_BOUND");
                    return v == nullptr || std::strcmp(v, "0") != 0;
                }();
                if (stop_bound)
                    return strata::core::serve_stop_bound(a + 1, max_new - produced_n, picks, [&](int32_t y) {
                        return std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    }) - 1;
#endif
                (void) picks;
                return a;
            };
            int64_t t2_rej[8] = {}, t2_hit[8] = {};   // STRATA_MTP_TOP2: rejections inside the MTP drafts by depth, runner-up hits
            int64_t dp_tested[8] = {}, dp_ok[8] = {};   // STRATA_SPEC_DEPTH=1: MTP drafts tested / accepted by depth
            int64_t draft_offered = 0, draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t> consumed;
            consumed.reserve((size_t) (n + max_new + S));
            double keep_ms = 0;   // --draft auto: host time spent queueing the idle drafter's upkeep
            bool keep_pending = false;
            auto keep_sync = [&]() -> bool {   // the queued upkeep is done: the verifier may rewrite the taps / residual rows now
                if (!keep_pending) return true;
                keep_pending = false;
#if defined(STRATA_HAVE_DFLASH2)
                if (use_df2 && !df2.context_wait(err)) return false;
#endif
                if (use_mtp && cudaStreamSynchronize(mtp.stream()) != cudaSuccess) { err = "the MTP upkeep failed"; return false; }
                return true;
            };
            const int64_t rounds_at_start = rounds;
            for (int64_t i = 0; i < p; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            // #4590: where the penalty window may start in `consumed` (see pen_prompt above): the end of the prompt without its
            // last token, which joins the window as the fed-back head `x`.
            size_t pen_gen0 = strata::kernels::penalty_window_from(consumed.size(), pen_prompt);
            if (req_pen_from >= 0 && !pen_prompt) pen_gen0 = std::min<size_t>(pen_gen0, (size_t) req_pen_from);   // a continued request: its reply began earlier
            const char* finish = "length";
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
            struct DecSnap {
                double wait, pool, host, plan, actq, jobs, run;
                int64_t misses, entries, hits, pcie;
            };
            auto dec_snap = [&]() {
                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                               drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                               drive.d.pcie_experts};
            };
            const DecSnap ds0 = dec_snap();
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            double dt_adapt_wait = 0, dt_adapt_join = 0;   // the tier's wait before a window / join after the draft (ms, summed)
            const int64_t dt_sw0 = ajob ? a_swapped : adapt_nswaps;
            int64_t dec_windows = 0, dec_T = 0;
            const int64_t decode_hits0 = drive.d.cache_hits;
            // CS-T: the RAM and file tiers of this request (the mmap source; 0 with the arena)
            const int64_t ram0 = src.ram_reads(), files0 = src.file_reads();
            const uint64_t file_bytes0 = src.file_read_bytes();
            const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t offload0 = drive.d.offload_entries;   // #588
            pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses);   // the prompt's windows are not the decode's
            if (cancelled) finish = nf_hit ? "nonfinite" : "cancel";
            // ======== --pipeline-windows 2: the decode windows with the two cards overlapped ========
            // Stage 0 (the first card) runs window K+1 while stage 1 verifies window K, on the guess that K is accepted
            // whole and that its bonus token is the drafter's.  The drafter, teacher-forced through K+1's drafts,
            // guesses K+1's bonus and drafts K+2 while stage 1 runs K+1.  A wrong guess costs the speculative window:
            // stage 0's GDN state comes back from a snapshot and K's commit is replayed with its real count (the
            // window's own commit inputs are intact in its parity's verifier).  Stage 1 never runs a speculative window
            // and a window row computes what it would in any other window, so every emitted token is the serial loop's
            // (with STRATA_IQ_MT_MIN=1 bit for bit; sampled requests with the same Philox draw per position).
            bool pl_ran = false;
            const bool pl_want = pipe && pl_pw >= 2 && pl_snap_ok;
            const char* pl_serial = !pl_want ? nullptr
                                  : hist_n > 0 ? "repetition penalties (penalty_last_n)"
                                  : mtp.coupled() ? "coupled draft sampling" : nullptr;
            if (pl_serial != nullptr) {   // said once per reason
                static std::set<std::string> said;
                if (said.insert(pl_serial).second)
                    std::fprintf(stderr, "strata serve: --pipeline-windows: a request with %s decodes serially\n",
                                 pl_serial);
            }
            if (pl_want && pl_serial == nullptr && !cancelled && produced_n < max_new) {
                pl_ran = true;
                if (!pl_prepare(err)) {
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                struct PW {
                    int T = 0, seq = 0;
                    int64_t p = 0;
                    int32_t tok[8] = {};
                    float prob[8] = {};    // prob[i]: the drafter's probability of tok[i + 1]
                    float p_on = 0.0f;     // the chain's estimate that the window before is accepted whole, tok[0] its bonus
                    // launched: on stage 0; finished: done on every front stage (all but the last); s1: on the last
                    bool ready = false, launched = false, finished = false, s1 = false, s1_done = false;
                    int fr = 0;            // front stages done
                    bool fl = false;       // in flight on front stage `fr`
                    uint32_t com = 0;      // the front stages this window was committed on speculatively (bit per stage)
                    bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                    bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                    bool sfx = false;      // a lookup window (the suffix drafter's drafts)
                    int sfx_match = 0;
                };
                // front stage `st`'s verifier of w's parity, and the last stage's
                auto VF = [&](int st, const PW& w) -> strata::core::Verifier& { return stage_ver_p(st, w.seq & 1); };
                auto V1 = [&](const PW& w) -> strata::core::Verifier& { return *PV[1][w.seq & 1]; };
                auto SDf = [&](const PW& w) { return (void*) PSD[w.seq & 1]; };
                const float theta = pl_theta;
                const int force_miss = pl_force_miss;
                static const bool pl_log = pipe_dbg_env("STRATA_PIPELINE_LOG") != nullptr;
                // the snapshot of the GDN state window `seq` reads on front stage `st` (overlapped: before its launch
                // there, on the stage's side stream)
                auto snap_take = [&](int st, int seq) -> bool {
                    if (!pl_snap_overlap) return true;
                    PlSnap& sn = pl_snaps[(size_t) st];
                    const strata::core::OnDevice on(sn.dev);
                    cudaStream_t cs = pl_stream[(size_t) st];
                    const int q = seq & 1;
                    return cudaEventRecord(sn.pre[q], cs) == cudaSuccess &&
                           cudaStreamWaitEvent(sn.side, sn.pre[q], 0) == cudaSuccess &&
                           cudaMemcpyAsync(sn.buf[q], sn.state, sn.bytes, cudaMemcpyDeviceToDevice, sn.side) == cudaSuccess &&
                           cudaEventRecord(sn.ev[q], sn.side) == cudaSuccess &&
                           (cudaStreamQuery(sn.side), true);
                };
                PW A, B, D;              // A: the oldest window not verified; B: the next; D: a wrong speculative window
                bool doomed = false;     // D still runs: when it finishes, stage 0 goes back to undo_w's real commit
                PW undo_w;
                int undo_keep = 0;
                int32_t undo_ple[2] = {-1, -1};
                int chain_kind = 0;      // 1: after an on-path verdict (A's drafts forced); 2: fresh (A's drafts from it)
                bool early_used = false;
                bool b_done = false;     // B has been made from the chain in flight
                int chain_n = 0;
                int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
                std::vector<int32_t> outp(8, 0);
                bool ending = false;
                A.T = 1;
                A.p = p;
                A.tok[0] = x;
                A.ready = true;
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // the adaptive tier beside the pipeline: the serial loop's rule and adapt() on a thread of its own (its
                // copies wait for the windows in flight, which this loop keeps serving meanwhile).  --adapt-async: the
                // asynchronous tier's steps instead, its copies queued here at each card's gaps (a_pl_idle, a_gap)
                std::thread pl_adapt_thr;
                std::atomic<bool> pl_adapt_done{true};
                bool pl_adapt_ok = true;
                // an error ends the engine, as the serial loop's do: every GPU wait is released first (#267), so no
                // window in flight keeps spinning on a flag nobody raises, and the tier's thread can finish (the
                // asynchronous tier's job does host work alone: it ends)
                auto die = [&](const std::string& m) -> int {
                    std::fprintf(stderr, "strata serve: pipelined decode: %s\n", m.c_str());
                    strata::core::diag_pipeline_fn().store(nullptr);
                    strata::core::release_gpu_waits(stderr);
                    if (pl_adapt_thr.joinable()) pl_adapt_thr.join();
                    if (ajob) {
                        ajob->wait();
                        a_pl_idle = nullptr;
                    }
                    print_response("ERR %s\n", m.c_str());
                    std::fflush(stdout);
                    return 1;
                };
                // STRATA_PIPELINE_TRACE=<file>: every event of the loop with its time (appended per request)
                static const char* pl_trace_path = pipe_dbg_env("STRATA_PIPELINE_TRACE");
                std::string pl_trace;
                const Clock::time_point pl_tr0 = Clock::now();
                auto tre = [&](const char* ev, int seq, int x0 = 0, int x1 = 0) {
                    if (pl_trace_path == nullptr) return;
                    char b[96];
                    std::snprintf(b, sizeof b, "%.3f %s %d %d %d\n",
                                  std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                    pl_trace += b;
                };
                auto pump0 = [&](PW& w) -> bool {   // w's front stage in flight
                    if (!w.fl) return true;
                    strata::core::Verifier& v = VF(w.fr, w);
                    // a doomed window (its guess was wrong) gets empty plans: no expert work on the CPU or the GPU for
                    // its remaining layers, so it frees stage 0 and the pool sooner (its rows are discarded and the
                    // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                    static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                    const bool skip = doom_skip && doomed && &w == &D;
                    if (v.service(skip ? nullptr : &drive_pool_split, SDf(w), err) < 0) return false;
                    if (v.done(err)) {
                        if (!v.pl_finish(nullptr, err)) return false;
                        w.fl = false;
                        if (++w.fr == pl_nf) w.finished = true;
                        tre("F0", w.seq, w.fr, w.spec);
                    }
                    return err.empty();
                };
                // the serial loop's window size over the chain's outputs from index `from` (`avail` of them)
                auto t_rule = [&](const float* pr, int from, int avail) {
                    int T = S_mtp;
                    if (req_spec_min_p > 0.0) {
                        T = 1;
                        while (T < S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) req_spec_min_p) ++T;
                    }
                    return std::max(1, std::min(T, 1 + avail - from));
                };
                // B from the chain's outputs: its row 0 is the guess at index `base`, its drafts follow
                auto make_b = [&](const int32_t* oc, const float* op, int base, int avail, float pa) {
                    B = PW{};
                    if (base >= avail) return;
                    B.seq = A.seq + 1;
                    B.p = A.p + A.T;
                    B.tok[0] = oc[base];
                    B.T = t_rule(op, base + 1, avail);
                    for (int i = 1; i < B.T; ++i) {
                        B.tok[i] = oc[base + i];
                        B.prob[i - 1] = op[base + i];
                    }
                    B.p_on = pa * op[base];
                    if (force_miss > 0 && ++pl_fm % force_miss == 0) B.tok[0] = B.tok[0] == 0 ? 1 : 0;
                    B.ready = true;
                    B.made = true;
                };
                // events after all the work both stages' streams hold now, and whether they have passed
                std::vector<cudaEvent_t> fence_ev((size_t) n_stages, nullptr), exch_ev((size_t) n_stages, nullptr);
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::OnDevice on(stage_ver_p(st, 0).device());
                    cudaEventCreateWithFlags(&fence_ev[(size_t) st], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&exch_ev[(size_t) st], cudaEventDisableTiming);
                }
                auto stage_record = [&](std::vector<cudaEvent_t>& ev) {   // after the work every stage's stream holds now
                    for (int st = 0; st < n_stages; ++st) {
                        const strata::core::OnDevice on(stage_ver_p(st, 0).device());
                        cudaEventRecord(ev[(size_t) st], pl_stream[(size_t) st]);
                        (void) cudaStreamQuery(pl_stream[(size_t) st]);
                    }
                };
                auto stage_passed = [&](std::vector<cudaEvent_t>& ev) {
                    for (cudaEvent_t e : ev)
                        if (cudaEventQuery(e) == cudaErrorNotReady) return false;
                    return true;
                };
                // adapt()'s fence, on the tier's thread: it waits on the HOST.  A device-side wait (the refill streams
                // waiting for the stages' events) would queue copies behind windows that need this loop, beside it.
                adapt_fence = [&]() {
                    stage_record(fence_ev);
                    while (!stage_passed(fence_ev)) std::this_thread::yield();
                };
                src.prepare_overrides();
                // The swaps land in two steps.  Once their copies are done, the incoming experts are resident (the plans
                // from then on use their slots).  What gives their RAM back - the resident RAM mode's exchange, the
                // released or unpinned pages - waits until every window that may still pull them over PCIe (a plan
                // published before that point) has completed: both stages' streams are fenced there.
                bool exch_wait = false;
                std::vector<int32_t> pl_landed;
                auto pl_release = [&]() {
                    unpin_blobs(pin_live);
                    src.commit_exchanges();
                    for (const int32_t i : pl_landed) srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    pl_landed.clear();
                    exch_wait = false;
                };
                auto pl_apply = [&]() {
                    if (exch_wait) {
                        if (stage_passed(exch_ev)) pl_release();
                        return;
                    }
                    if (pending.empty() || cudaEventQuery(adapt_ev) != cudaSuccess) return;
                    for (auto& st : stages)
                        if (st->adapt_live && cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
                    for (auto& st : stages) st->adapt_live = false;
                    for (const auto& [i, slot] : pending) {
                        host_res[(size_t) i] = slot;
                        pl_landed.push_back(i);
                    }
                    pending.clear();
                    pending_age = 0;
                    res_upload();
                    stage_record(exch_ev);
                    exch_wait = true;
                };
                // --adapt-async: the asynchronous tier beside the windows (see adapt_tick).  A stage is idle when neither
                // of its verifiers has a window in flight (a commit or a snapshot copy waits on nothing of the host's)
                if (ajob) a_pl_idle = [&](int st) { return !PV[st][0]->in_flight() && !PV[st][1]->in_flight(); };
                double a_poll_ms = 0.0;
                auto pl_adapt = [&]() -> bool {   // the blocking tier (--adapt-async 0)
                    if (pl_adapt_thr.joinable() && pl_adapt_done.load()) {
                        pl_adapt_thr.join();
                        if (!pl_adapt_ok) return false;
                    }
                    if (pl_adapt_thr.joinable()) return true;   // still adapting
                    pl_apply();
                    if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_iv(rounds)) == 0 && pending.empty() &&
                        !exch_wait) {
                        pl_adapt_done = false;
                        pl_adapt_thr = std::thread([&] { strata::platform::name_this_thread("strata-adapt"); strata::aux_cpus::pin_current_thread(); pl_adapt_ok = adapt(); pl_adapt_done = true; });
                    }
                    return true;
                };
                // the serial loop's prompt lookup for a fresh window: where the text repeats an earlier stretch and the
                // drafter's own first guess agrees, the policy may take the lookup's continuation (up to S rows)
                std::vector<int32_t> pl_sbuf((size_t) (2 * S), 0);
                // a lookup window's continuation goes on past it: the next window (B) is taken from the same lookup and
                // speculated behind it (a copied stretch is mostly accepted whole).  STRATA_PIPELINE_LOOKUP_PON: B's
                // estimate (default 0.75; 0 = no lookup B)
                static const float pl_lookup_pon = [] {
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_LOOKUP_PON");
                    return v ? (float) std::atof(v) : 0.75f;
                }();
                auto pick_lookup = [&](PW& w, const int32_t* chain0) {
                    if (o.suffix_draft <= 0 || w.seq == 0) return;
                    const int k = wplan.sfx.propose(2 * S - 1, pl_sbuf.data());
                    const int match = wplan.sfx.last_match();
                    if (k <= 0 || pl_sbuf[0] != chain0[0]) return;
                    const strata::spec::DraftPolicy::Pick pk = wplan.policy.choose(w.T, std::min(k, S - 1), match);
                    if (!pk.lookup) return;
                    w.T = pk.t;
                    for (int i = 1; i < w.T; ++i) {
                        w.tok[i] = pl_sbuf[(size_t) i - 1];
                        w.prob[i - 1] = 1.0f;
                    }
                    w.sfx = true;
                    w.sfx_match = match;
                    const int rest = k - (w.T - 1);   // lookup tokens past w: B's row 0 (w's bonus guess) and its drafts
                    if (&w == &A && rest >= 1 && pl_lookup_pon > 0.0f && !b_done) {
                        B = PW{};
                        B.seq = A.seq + 1;
                        B.p = A.p + A.T;
                        B.T = std::min(S, rest);
                        for (int i = 0; i < B.T; ++i) B.tok[i] = pl_sbuf[(size_t) (A.T - 1 + i)];
                        for (int i = 1; i < B.T; ++i) B.prob[i - 1] = 1.0f;
                        B.p_on = pl_lookup_pon;
                        B.ready = true;
                        B.made = true;
                        b_done = true;   // the chain in flight makes no B for this A
                        tre("CD", A.seq + 1, 3, (int) (1000.0f * B.p_on));
                    }
                };
                const Clock::time_point pl_t0 = Clock::now();
                for (int st = 0; st < 2; ++st)
                    for (int par = 0; par < 2; ++par) g_pl_diag.v[st][par] = PV[st][par];
                strata::core::diag_pipeline_fn().store(&pl_diag_print);
                auto pw_bits = [](const PW& w) {
                    return (w.ready ? 1 : 0) | (w.launched ? 2 : 0) | (w.finished ? 4 : 0) | (w.com != 0 ? 8 : 0) |
                           (w.s1 ? 16 : 0) | (w.s1_done ? 32 : 0);
                };
                // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                // the previous verdict, the tokens it emitted, how many
                double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
                int64_t cal_n[10] = {}, cal_on[10] = {};   // per p_on decile: windows scored, on the path
                double last_verdict = 0.0;
                int64_t pl_disagree = 0, pl_late = 0;
                static const bool pl_prestage = [] {   // STRATA_PIPELINE_PRESTAGE=0: B staged at its launch
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_PRESTAGE");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                static const bool pl_agree = [] {   // STRATA_PIPELINE_AGREE=0: only the old chain's probabilities
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_AGREE");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
                while (true) {
                    g_pl_diag.iters.fetch_add(1, std::memory_order_relaxed);
                    g_pl_diag.a_seq.store(A.seq, std::memory_order_relaxed);
                    g_pl_diag.a_bits.store(pw_bits(A), std::memory_order_relaxed);
                    g_pl_diag.b_bits.store(pw_bits(B), std::memory_order_relaxed);
                    g_pl_diag.d_bits.store(pw_bits(D), std::memory_order_relaxed);
                    g_pl_diag.chain_kind.store(chain_kind, std::memory_order_relaxed);
                    g_pl_diag.chain_live.store(mtp.chain_live() ? 1 : 0, std::memory_order_relaxed);
                    g_pl_diag.doomed.store(doomed ? 1 : 0, std::memory_order_relaxed);
                    g_pl_diag.ending.store(ending ? 1 : 0, std::memory_order_relaxed);
                    // ---- --adapt-async: the round in flight moves on between verdicts too (at most every 0.5 ms)
                    if (ajob && astate != AState::Idle) {
                        const double t = ms_now();
                        if (t - a_poll_ms >= 0.5) {
                            a_poll_ms = t;
                            if (!adapt_tick(false, false)) return die("an adaptive refill failed");
                        }
                    }
                    // ---- every window in flight served
                    if ((doomed && !pump0(D)) || !pump0(A) || !pump0(B)) return die(err);
                    if (A.s1 && !A.s1_done) {
                        strata::core::Verifier& v = V1(A);
                        if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                        if (v.done(err)) {
                            if (!v.pl_finish(outp.data(), err)) return die(err);
                            A.s1_done = true;
                            tre("F1", A.seq, A.T);
                        } else if (!err.empty()) return die(err);
                    }
                    if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
                    // ---- a wrong speculative window has stopped (it is launched on no further stage once doomed):
                    // every front stage it ran on back to the verified window's real commit
                    if (doomed && !D.fl) {
                        for (int st = 0; st < pl_nf; ++st) {
                            if ((undo_w.com & (1u << st)) == 0) continue;
                            PlSnap& sn = pl_snaps[(size_t) st];
                            {
                                const strata::core::OnDevice on(sn.dev);
                                if (cudaMemcpyAsync(sn.state, sn.buf[pl_snap_overlap ? (undo_w.seq & 1) : 0], sn.bytes,
                                                    cudaMemcpyDeviceToDevice, pl_stream[(size_t) st]) != cudaSuccess)
                                    return die("restoring the GDN snapshot failed");
                            }
                            if (st == 0) {
                                ss.ple_prev[0] = undo_ple[0];
                                ss.ple_prev[1] = undo_ple[1];
                            }
                            if (!VF(st, undo_w).pl_commit_async(undo_keep, err)) return die(err);
                        }
                        doomed = false;
                        ++pl_undo;
                        tre("U", undo_w.seq, undo_keep);
                    }
                    if (ending) {
                        if (!doomed) break;
                        continue;
                    }
                    // ---- the drafter's chain, read as its outputs land (an event after each): a fresh A once its size
                    // is decided, then B once ITS size is decided - its bonus guess and a draft below spec_min_p (or
                    // S_mtp - 1 drafts) - so stage 0 need not wait for the whole chain
                    if (chain_kind != 0) {
                        const int k_ready = mtp.chain_outputs_ready(err);
                        if (k_ready < 0) return die(err);
                        if (chain_kind == 2 && !early_used) {
                            // A's size is decided by its first outputs: a draft below spec_min_p ends it (the serial
                            // loop's rule), so A goes to stage 0 as soon as that is known, not after all S_mtp - 1
                            const int k = k_ready;
                            bool decided = k >= S_mtp - 1;
                            if (!decided && req_spec_min_p > 0.0)
                                for (int j = 0; j < k; ++j)
                                    if (mtp.chain_prob()[j] < (float) req_spec_min_p) { decided = true; break; }
                            if (decided) {
                                A.T = t_rule(mtp.chain_prob(), 0, std::min(k, S_mtp - 1));
                                for (int i = 1; i < A.T; ++i) {
                                    A.tok[i] = mtp.chain_tok()[i - 1];
                                    A.prob[i - 1] = mtp.chain_prob()[i - 1];
                                }
                                pick_lookup(A, mtp.chain_tok());
                                A.ready = true;
                                early_used = true;
                                tre("CE", A.seq, A.T);
                            }
                        }
                        const int r = mtp.chain_poll(err);
                        if (r < 0) return die(err);
                        const int k = r == 1 ? chain_n : k_ready;
                        const int32_t* oc = mtp.chain_tok();
                        const float* op = mtp.chain_prob();
                        if (r == 1 && chain_kind == 2 && !early_used) {
                            A.T = t_rule(op, 0, std::min(chain_n, S_mtp - 1));
                            for (int i = 1; i < A.T; ++i) {
                                A.tok[i] = oc[i - 1];
                                A.prob[i - 1] = op[i - 1];
                            }
                            pick_lookup(A, oc);
                            A.ready = true;
                            early_used = true;
                        }
                        if (!b_done && (chain_kind == 1 || early_used)) {
                            const int base = A.T - 1;                    // B's row 0: the guess of A's bonus
                            bool decided = r == 1 || k >= base + S_mtp;
                            if (!decided && k > base + 1 && req_spec_min_p > 0.0)
                                for (int j = base + 1; j < k; ++j)
                                    if (op[j] < (float) req_spec_min_p) { decided = true; break; }
                            if (decided) {
                                float pa = 1.0f;
                                if (chain_kind == 1) {
                                    // forced through A's drafts: its outputs there are the drafter's own picks from
                                    // A's verified predecessor, a better estimate of A's acceptance than the chain that
                                    // drafted A; a disagreement means A is unlikely to be accepted whole
                                    bool agree = true;
                                    for (int i = 0; i + 1 < A.T; ++i) {
                                        if (oc[i] != A.tok[i + 1]) agree = false;
                                        pa *= pl_agree ? op[i] : A.prob[i];
                                    }
                                    if (pl_agree && !agree) { pa = 0.0f; ++pl_disagree; }
                                } else {
                                    for (int i = 0; i + 1 < A.T; ++i) pa *= A.prob[i];
                                }
                                if (!A.sfx) make_b(oc, op, base, k, pa);   // a lookup window's bonus is not the chain's
                                // stage B now (its PLE rows from the tokens before it as they will be once A is
                                // committed whole), so its launch behind A is only the graph launch (never while a
                                // rollback is pending: B's verifier is the one the undo commit reads)
                                if (B.ready && B.p_on >= theta && pl_prestage && !VF(0, B).in_flight() && !doomed) {
                                    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
                                    for (int i = 0; i < A.T; ++i) { prev[0] = prev[1]; prev[1] = A.tok[i]; }
                                    if (!VF(0, B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                }
                                b_done = true;
                                tre("CD", A.seq, chain_kind, (int) (1000.0f * B.p_on));
                                if (pl_log)
                                    std::fprintf(stderr, "strata pipeline: chain %d for p=%lld T=%d: B T=%d p_on %.3f (%d "
                                                         "outputs)\n", chain_kind, (long long) A.p, A.T, B.T, B.p_on, k);
                            }
                        }
                        if (r == 1) chain_kind = 0;
                    }
                    // ---- the front stages: A on its next one (never while a wrong window still holds a front stage's
                    // state)
                    if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                        const int st = A.fr;
                        if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                        if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
                        if (!snap_take(st, A.seq)) return die("the GDN snapshot failed");
                        if (!VF(st, A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                        A.fl = true;
                        if (st == 0) A.launched = true;
                        tre("L0", A.seq, A.T, st);
                    }
                    // ---- the front stages: B, speculatively, one stage behind A.  On each stage A is committed there
                    // (its snapshot taken first) right before B launches, as stage 0 alone did with two stages
                    // STRATA_PIPELINE_SPEC_DEPTH=k (debug): B runs speculatively on the first k front stages only
                    static const int pl_spec_depth = [] {
                        const char* v = pipe_dbg_env("STRATA_PIPELINE_SPEC_DEPTH");
                        return v ? std::max(1, std::atoi(v)) : 1 << 20;
                    }();
                    if (B.ready && !B.fl && B.fr < pl_nf && B.fr < pl_spec_depth && A.fr > B.fr && !doomed) {
                        const int st = B.fr;
                        if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                            if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
                            PlSnap& sn = pl_snaps[(size_t) st];
                            cudaStream_t cs = pl_stream[(size_t) st];
                            {
                                const strata::core::OnDevice on(sn.dev);
                                if (pl_snap_overlap ? cudaStreamWaitEvent(cs, sn.ev[A.seq & 1], 0) != cudaSuccess
                                                    : cudaMemcpyAsync(sn.buf[0], sn.state, sn.bytes,
                                                                      cudaMemcpyDeviceToDevice, cs) != cudaSuccess)
                                    return die("the GDN snapshot failed");
                            }
                            if (st == 0) {
                                undo_ple[0] = ss.ple_prev[0];
                                undo_ple[1] = ss.ple_prev[1];
                            }
                            if (!VF(st, A).pl_commit_async(A.T, err) || !snap_take(st, B.seq) ||
                                !VF(st, B).pl_launch(B.T, B.tok, B.p, err))
                                return die(err.empty() ? std::string("the GDN snapshot failed") : err);
                            A.com |= 1u << st;
                            B.fl = true;
                            if (st == 0) {
                                B.launched = true;
                                B.spec = true;
                                ++pl_spec;
                            }
                            tre("L0", B.seq, B.T, 10 + st);
                        } else {
                            B.ready = false;   // not worth it: A's verdict decides the next window
                            ++pl_gate;
                        }
                    }
                    // ---- the last stage: A, once every front stage is done
                    if (A.finished && !A.s1) {
                        if (chain_kind == 1 && !B.ready) ++pl_late;   // stage 0 idles until the chain has B
                        if (ajob) a_gap(1);   // --adapt-async: stage 1 is idle until this launch
                        if (!V1(A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                        A.s1 = true;
                        tre("L1", A.seq, A.T);
                    }
                    // ---- A's verdict
                    if (!A.s1_done) continue;
                    // the drafter must be idle before the next chain.  The chain needs nothing from the host, so this
                    // wait is bounded even with B (made from its first outputs) in flight on stage 0; in practice the
                    // chain ended long before a verdict that needs a whole stage-1 window.
                    while (chain_kind != 0 && mtp.chain_live())
                        if (mtp.chain_poll(err) < 0) return die(err);
                    if (chain_kind != 0 && !B.launched && !B.made) B = PW{};
                    chain_kind = 0;
                    const int a = stop_bounded(strata::spec::accepted_prefix(A.tok + 1, outp.data(), A.T - 1), outp.data());
                    if (!V1(A).pl_commit_async(a + 1, err)) return die(err);
                    for (int i = 0; i <= a; ++i) consumed.push_back(A.tok[i]);
                    draft_offered += A.T - 1;
                    draft_accepted += a;
                    ++rounds;
                    if (drive.d.fs) drive.d.fs->completed.fetch_add(1, std::memory_order_release);
                    ++dec_windows;
                    dec_T += A.T;
                    const bool nf = nonfinite_hit(V1(A), "decode");   // #879: none of its tokens goes out
                    bool eos = false;
                    for (int i = 0; !nf && i <= a && produced_n < max_new && !eos; ++i) {
                        print_response("T %d\n", (int) outp[(size_t) i]);
                        if (req_logprobs >= 0) emit_lp_line(V1(A), i, outp[(size_t) i], req_logprobs, req_allowed);
                        strata::core::progress_beat();
                        ++produced_n;
                        if (o.suffix_draft > 0) wplan.sfx.append(outp[(size_t) i]);
                        eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outp[(size_t) i]) != o.eos_ids.end();
                    }
                    std::fflush(stdout);
                    if (nf) finish = "nonfinite";
                    else if (eos) finish = "stop";
                    else if (input.is_stop_requested()) finish = "cancel";
                    const bool last = nf || eos || produced_n >= max_new || input.is_stop_requested();
                    const bool on = !nf && B.launched && a == A.T - 1 && B.tok[0] == outp[(size_t) A.T - 1];
                    if (B.made) {   // the gate's calibration: would B have been on the path, by its estimate p_on
                        const bool would = a == A.T - 1 && B.tok[0] == outp[(size_t) A.T - 1];
                        const int bin = std::min(9, std::max(0, (int) (B.p_on * 10.0f)));
                        cal_n[bin] += 1;
                        cal_on[bin] += would ? 1 : 0;
                    }
                    tre("V", A.seq, a, on ? 1 : (B.launched ? -1 : 0));
                    {
                        const double now = ms_now();
                        const int c = A.spec ? 1 : 0;
                        if (A.sfx) wplan.lookup.add(A.T - 1, a);
                        if (A.seq > 0 && !last) wplan.policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                        cls_ms[c] += now - last_verdict;
                        cls_tok[c] += a + 1;
                        cls_n[c] += 1;
                        last_verdict = now;
                    }
                    if (pl_log)
                        std::fprintf(stderr, "strata pipeline: verdict p=%lld T=%d a=%d B %s%s (B front %d%s, A com %x) out",
                                     (long long) A.p, A.T, a, B.launched ? "launched" : B.ready ? "ready" : "none",
                                     on ? " ON" : "", B.fr, B.fl ? "+" : "", A.com);
                    if (pl_log) {
                        for (int i = 0; i <= a; ++i) std::fprintf(stderr, " %d", (int) outp[(size_t) i]);
                        std::fprintf(stderr, "\n");
                    }
                    x = outp[(size_t) a];
                    p = A.p + a + 1;
                    pcie_auto_step(req_pcie_frac);   // 0.1.44: the measured share in the pipelined loop too (STRATA_PCIE_AUTO_SPLIT=1 on a split)
                    if (!last && !ajob && !pl_adapt()) return die("an adaptive refill failed");
                    if (on && !last) {
                        ++pl_on;
                        // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                        mtp.set_source_R(V1(A).final_R(0));
                        chain_n = std::min(strata::kernels::kVerifyNarrowT - 1, B.T + S_mtp - 1);
                        if (!mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err))
                            return die(err);
                        chain_kind = 1;
                        tre("CL", A.seq, 1);
                        early_used = false;
                        b_done = false;
                        // A accepted whole: its commit on the front stages B has not reached yet (B, now A, runs there
                        // after it)
                        for (int st = 0; st < pl_nf; ++st)
                            if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(A.T, err)) return die(err);
                        A = B;
                        B = PW{};
                    } else {
                        // the front stages B never ran on take A's real commit now; those it did run on are undone
                        // (snapshot restored, the commit replayed) once it has stopped
                        for (int st = 0; st < pl_nf; ++st)
                            if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(a + 1, err)) return die(err);
                        if (B.launched) {   // a wrong guess (or the end): undo once it has stopped
                            doomed = true;
                            D = B;
                            undo_w = A;
                            undo_keep = a + 1;
                        }
                        // B is D now (or dropped): never pump it twice (whichever copy saw the window complete first
                        // would finish it, and the other would wait for it forever)
                        B = PW{};
                        if (!last) {
                            mtp.set_source_R(V1(A).final_R(0));
                            chain_n = strata::kernels::kVerifyNarrowT - 1;
                            if (!mtp.chain_launch(A.T, outp.data(), A.p, a, nullptr, 0, chain_n, chain_n, err))
                                return die(err);
                            chain_kind = 2;
                            tre("CL", A.seq, 2);
                            early_used = false;
                            b_done = false;
                            PW nA;
                            nA.seq = A.seq + 1;
                            nA.p = A.p + a + 1;
                            nA.tok[0] = outp[(size_t) a];
                            A = nA;
                        }
                    }
                    // --adapt-async: the verdict's tick, once the drafter's chain is launched (off its critical path)
                    if (!last && ajob && !adapt_tick(false)) return die("an adaptive refill failed");
                    if (last) ending = true;
                }
                pl_drain();   // nothing in flight: both stages' commits and the drafter have landed
                strata::core::diag_pipeline_fn().store(nullptr);
                if (pl_adapt_thr.joinable()) pl_adapt_thr.join();   // its copies waited only for finished work
                adapt_fence = nullptr;
                if (exch_wait) pl_release();   // every window has completed (drained)
                // --adapt-async: both stages idle now: the copies still due are queued and the round in flight goes on
                // unfenced (the INVARIANT at pl_drain: a checkpoint, a park, the serial loop and the next request find
                // nothing pipelined in flight) at the next tick, the next request's drain
                a_pl_settle();
                for (cudaEvent_t ev : fence_ev) if (ev) cudaEventDestroy(ev);
                for (cudaEvent_t ev : exch_ev) if (ev) cudaEventDestroy(ev);
                if (pl_trace_path != nullptr && !pl_trace.empty()) {
                    if (std::FILE* f = std::fopen(pl_trace_path, "a")) {
                        std::fprintf(f, "# request p0=%lld\n", (long long) (n - 1));
                        std::fwrite(pl_trace.data(), 1, pl_trace.size(), f);
                        std::fclose(f);
                    }
                }
                // the odd windows' counters and GPU profiles into the verifiers that report them
                for (int st = 0; st < n_stages; ++st) stage_ver_p(st, 0).absorb_stats(stage_ver_p(st, 1));
                if (!pl_adapt_ok) {
                    print_response("ERR an adaptive refill failed\n");
                    return 1;
                }
                while (mtp.chain_live() && mtp.chain_poll(err) == 0) {}
                mtp.set_source_R(ver.final_R_all());   // the serial loop's rows again
                const double pl_ms = std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count();
                dt_run += pl_ms;
                if (dec_timing) {
                    auto avg = [](double s_, double n_) { return n_ > 0 ? s_ / n_ : 0.0; };
                    std::fprintf(stderr, "strata pipeline: %lld windows in %.0f ms (%.2f ms/window): %lld speculative, %lld on "
                                         "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                 (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                 (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
                    std::fprintf(stderr, "strata pipeline classes: fresh %.0f windows %.2f ms %.2f tok | speculative %.0f "
                                         "windows %.2f ms %.2f tok | forced-chain disagreements %lld, chain late %lld\n",
                                 cls_n[0], avg(cls_ms[0], cls_n[0]), avg(cls_tok[0], cls_n[0]), cls_n[1],
                                 avg(cls_ms[1], cls_n[1]), avg(cls_tok[1], cls_n[1]), (long long) pl_disagree,
                                 (long long) pl_late);
                    std::fprintf(stderr, "strata pipeline: drafter on CUDA%d: %.3f ms of host time per chain launch (%.3f ms of it "
                                         "staging the source rows, %lld times so far)\n", mtp.device(),
                                 avg(mtp.ms_draft, (double) mtp.rounds), avg(mtp.ms_src, (double) mtp.n_src), (long long) mtp.n_src);
                    std::string cal;
                    char b[48];
                    for (int i = 0; i < 10; ++i)
                        if (cal_n[i] > 0) {
                            std::snprintf(b, sizeof b, " %.1f:%lld/%lld", i / 10.0, (long long) cal_on[i], (long long) cal_n[i]);
                            cal += b;
                        }
                    std::fprintf(stderr, "strata pipeline calibration (p_on decile: on/scored):%s\n", cal.c_str());
                }
            }
#if defined(STRATA_HAVE_DFLASH2)
#if defined(STRATA_GPU_METAL)
            const int df2_skip_min = strata::spec::df2_lookup_skip_min(8);
#else
            const int df2_skip_min = strata::spec::df2_lookup_skip_min(0);
#endif
            bool df2_deferred = false;
            int dfr_n = 0;
            int64_t dfr_pos = 0;
            int32_t dfr_anchor = 0;
            // a long suffix match: the next round's DFlash2 draft is deferred to that round's start (spec/window_rule.hpp skip_draft_for_lookup)
            auto defer_df2 = [&](int a_) -> bool {
                if (o.suffix_draft <= 0 || df2_skip_min <= 0 || !req_df2) return false;
                const int ks = wplan.propose_lookup(S_look);
                if (!strata::spec::skip_draft_for_lookup(true, df2_skip_min, ks, wplan.sfx.last_match())) return false;
                df2_deferred = true; dfr_n = a_ + 1; dfr_pos = p; dfr_anchor = outv[(size_t) a_];
                return true;
            };
            // the next round's drafts from the committed rows' taps (DFlash2), as the CLI loop does
            auto draft_df2 = [&](int a_) -> bool {
                int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                float pp[strata::core::Dflash2Drafter::kDrafts];
                const float* tp = df2_win_taps(a_ + 1, err);
                if (tp == nullptr || !df2.draft(tp, a_ + 1, p, outv[(size_t) a_], dd, pp, err)) return false;
                for (size_t i = 0; i < drafts.size() && i < (size_t) strata::core::Dflash2Drafter::kDrafts; ++i) { drafts[i] = dd[i]; dprob[i] = pp[i]; }
                if (ss.ple.ready() && ss.ple.table != nullptr) {   // the PLE rows of the next window's tokens, read ahead (as the MTP's draft does)
                    int n_pf = 0;
                    const int cap = std::min<int>(S_mtp, (int) drafts.size() + 1);
                    for (int t = 1; t < cap && (req_spec_min_p <= 0.0 || dprob[(size_t) t - 1] >= (float) req_spec_min_p); ++t) ++n_pf;
                    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
                    auto pf = [&](int32_t tok) {
                        uint32_t rows16[strata::kernels::PLE_N_HEADS];
                        strata::kernels::ngram_rows(&tok, prev, 1, ss.ple.consts, rows16);
                        prev[0] = prev[1];
                        prev[1] = tok;
                        ss.ple.table->prefetch_rows(rows16);
                    };
                    pf(outv[(size_t) a_]);
                    for (int i = 0; i < n_pf; ++i) pf(drafts[(size_t) i]);
                }
                return true;
            };
#else
            auto draft_df2 = [&](int) -> bool { return false; };
#endif
            if (df2_trace != nullptr) std::fprintf(df2_trace, "# request prompt %lld df2 %d spec %d\n", (long long) n, req_df2 ? 1 : 0, S_mtp);
            while (!pl_ran && !cancelled && produced_n < max_new) {
                bool dfr_lookup = false;
                int dfr_T = 0, dfr_match = 0;
#if defined(STRATA_HAVE_DFLASH2)
                if (df2_deferred) {   // the draft skipped for a long lookup match: the lookup if the policy takes it, else the draft now (same taps and rows)
                    df2_deferred = false;
                    const int kd = wplan.propose_lookup(S_look);
                    const int md = wplan.sfx.last_match();
                    const strata::spec::DraftPolicy::Pick pk = kd > 0 ? wplan.policy.choose(S_mtp, kd, md) : strata::spec::DraftPolicy::Pick{};
                    if (kd > 0 && pk.lookup) {
                        dfr_lookup = true; dfr_T = pk.t; dfr_match = md;
                        if (!df2.context(ver.tap_dev(), dfr_n, dfr_pos, err)) { print_response("ERR %s\n", err.c_str()); return 1; }
                    } else {
                        int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                        float pp[strata::core::Dflash2Drafter::kDrafts];
                        if (!df2.draft(ver.tap_dev(), dfr_n, dfr_pos, dfr_anchor, dd, pp, err)) { print_response("ERR %s\n", err.c_str()); return 1; }
                        for (size_t i = 0; i < drafts.size() && i < (size_t) strata::core::Dflash2Drafter::kDrafts; ++i) { drafts[i] = dd[i]; dprob[i] = pp[i]; }
                    }
                }
#endif
                // no drafter: one token a round unless a lookup draft fires.  The window rule is that of this request's
                // ACTIVE source (window_rule.hpp): the MTP chain's cost rule and probabilities do not govern DFlash2's
                // candidates, though the MTP may stay loaded beside it
                strata::spec::Window w = wplan.begin(p, {.use_mtp = use_mtp, .draft_on = draft_on, .df2_active = req_df2,
                                                         .first_window = first_window, .cap = S_mtp,
                                                         .start = draft_on ? S_mtp : 1, .spec_min_p = req_spec_min_p,
                                                         .draft_conf = o.draft_conf, .dprob = dprob.data()});
                if (first_window) w.T = merge_T > 0 ? merge_T : 1;
                const bool merged = first_window && merge_T > 0;   // the prompt's last tokens and the first token
                // a repeat of earlier context (prompt lookup) where the drafter's own first guess agrees, or the lookup
                // chained after the drafts: taken where their expected tokens per ms beat the drafter's window
                const int room = (int) std::min<int64_t>(S, o.max_context - p);
                if (gated_proposal) {
                    wplan.source.apply_costs(p);   // held round times whose draws are all committed now
                    if (!prepare_gated_window(proposal, w, generation, wplan, room, (int) ver.vocab(),
                                              (float) req_spec_min_p, mtp.spec_q(), drafts.data(), dprob.data(), err)) {
                        print_response("ERR %s\n", err.c_str()); return 1;
                    }
                    if (first_window && merge_T > 0) w.T = merge_T;   // STRATA_ADMIT_MERGE_FIRST: the prompt's last tokens ride in this window
                } else {
                    const int T_drafted = w.T;
                    // DFlash2's candidates are not draws from the MTP's q: only the MTP's are probabilistic
                    wplan.pick_lookup(w, drafts.data(), use_mtp && !req_df2 && mtp.prob(), S_look);
                    if (df2_trace != nullptr) {
                        tr_k = std::min(w.proposed, 16);
                        tr_match = w.match;
                        std::copy_n(wplan.lookup_drafts(), tr_k, tr_s.begin());
                    }
                    if (w.lookup && trace_nopick) {
                        w.lookup = false;
                        w.T = T_drafted;
                    }
                    if (w.lookup && !fit_rows(w.T)) { print_response("ERR %s\n", err.c_str()); return 1; }
                }
                if (dfr_lookup) {   // the lookup window chosen before the draft (the draft was skipped)
                    wplan.take_lookup(w, dfr_T, dfr_match, S_look);
                    if (!fit_rows(w.T)) { print_response("ERR %s\n", err.c_str()); return 1; }
                }
                if (df2_trace != nullptr) std::copy_n(drafts.begin(), std::min<size_t>(drafts.size(), 8), tr_d.begin());
                wplan.pick_chain(w, drafts.data(), S_look);
                if (w.chain > 0) {
                    int rows = w.rows();
                    if (!fit_rows(rows)) { print_response("ERR %s\n", err.c_str()); return 1; }
                    w.chain = std::max(0, rows - w.T);
                }
                const int T_mtp = w.T, T = w.rows();
                const Clock::time_point round0 = Clock::now();
                if (p + T > o.max_context) break;
                window[0] = x;
                for (int i = 1; i < T_mtp; ++i)
                    window[(size_t) i] = merged     ? (int32_t) ids[(size_t) (p + i)]
                                         : w.lookup ? wplan.lookup_drafts()[i - 1]
                                                    : drafts[(size_t) i - 1];
                std::copy_n(wplan.chain_drafts(), w.chain, window.begin() + T_mtp);
                if (!keep_sync()) { print_response("ERR %s\n", err.c_str()); return 1; }
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
                // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
                // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
                const Clock::time_point taw0 = Clock::now();
                const bool asw_skip = first_window || merged;   // the session's tier choice counts steady decode windows only
                double asw_wait_ms = 0;
                if (adapt_nowait()) apply_pending(false);
                else if (aov && (!pending.empty() || (peer.valid() && peer.has_pending())) && !aflipped) {
                    // 0.1.43 STRATA_ADAPT_OVERLAP: the tables take the swaps now, exactly where the wait would have ended;
                    // the copies still landing are waited for per layer inside the window (AdaptGate)
                    // (resident RAM mode: the staged evicted experts look committed to the planner at once, the RAM
                    // copy's places stay as they are until the copies have landed; see soft_commit_exchanges)
                    if (src.complement_ready() && !src.soft_commit_exchanges()) {
                        apply_pending(true);   // not possible here: the plain wait
                    } else {
                        for (const auto& [i, slot] : pending) host_res[(size_t) i] = slot;
                        if (peer.valid()) peer.flip_pending();   // its copies land over its own link, gated the same way
                        res_upload();
                        aflipped = true;
                        ++aov_flips;
                        agate.active.store(true, std::memory_order_release);
                    }
                }
                else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
                // STRATA_RAM_ADAPT=<n> (opt-in, the resident RAM mode on one GPU): the RAM copy is chosen once, at
                // start, so an expert it does not hold is read from the file every time the conversation routes to
                // it.  With this, before a window and while no adaptive round is in flight, up to n experts the windows
                // keep routing to (the adaptive tier's decayed count, at least its own candidate floor of 2) take the
                // RAM places of experts of their layer that are not routed to (count below 0.5), one pair per layer,
                // the largest difference first.  One blob read each, on this thread; the RAM stays the same size, and
                // the routing, the weights and the VRAM tier's own rule are untouched.
                static const int ram_adapt_n = [] {
                    const char* e = std::getenv("STRATA_RAM_ADAPT");
                    return e != nullptr ? std::max(0, std::atoi(e)) : 0;
                }();
                if (ram_adapt_n > 0 && ram_adapt2.on && !first_window && !drive.d.usage.empty() && pending.empty() &&
                    (!ajob || astate == AState::Idle) && src.complement_ready() && !multi_gpu && !peer.valid() &&
                    !remote_opt && o.batch == 0) {
                    // 149-prefetch STRATA_RAM_ADAPT2=1: install the promotions whose reads have landed (window boundary:
                    // nothing reads the RAM copy now), then stage the next ones, ranked by the stall they are expected to
                    // save per byte: (decayed decode count + the prompt's seed) x the miss's cost (a cold expert costs
                    // a drive read, one in the page cache STRATA_RAM_ADAPT_CACHED_W of that), against what the evicted
                    // one is expected to cost; up to STRATA_RAM_ADAPT_PER_LAYER pairs a layer.  The reads run on the
                    // source's I/O thread, not here.
                    ram_adapt2.installed += src.ram_install(host_res.data(), g.n_expert);
                    for (float& x : ram_adapt2.seed) x *= 0.95f;
                    struct Cand { float score; int32_t e; };
                    struct RamSwap2 { float gain; int32_t layer, in, out; };
                    std::vector<RamSwap2> swaps;
                    std::vector<Cand> ins, outs;
                    for (int64_t l = 0; l < g.n_layers; ++l) {
                        const float* u = drive.d.usage.data() + l * g.n_expert;
                        const float* sd = ram_adapt2.seed.empty() ? nullptr : ram_adapt2.seed.data() + l * g.n_expert;
                        const int32_t* r = host_res.data() + l * g.n_expert;
                        ins.clear();
                        outs.clear();
                        for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                            if (r[e] >= 0) continue;   // in VRAM
                            const float ue = u[e] + (sd != nullptr ? sd[e] : 0.0f);
                            if (src.has_resident(l, e)) outs.push_back({ue, e});
                            else if (ue >= 2.0f) ins.push_back({ue, e});
                        }
                        // the GPU tier's own next picks are left to it: its round takes the layer's most used experts
                        // outside VRAM (about 2 a layer a round), and a RAM place spent on one of them is skipped at the
                        // install (measured: 60% of the promotions skipped before this rule) - STRATA_RAM_ADAPT_SKIP_TOP
                        if (ram_adapt2.skip_top > 0 && !ins.empty()) {
                            std::vector<float> du;
                            for (const Cand& c : ins) du.push_back(u[c.e]);
                            const size_t k = std::min(du.size(), (size_t) ram_adapt2.skip_top);
                            std::nth_element(du.begin(), du.begin() + (std::ptrdiff_t) (k - 1), du.end(), std::greater<float>());
                            const float top = du[k - 1];
                            size_t dropped = 0;
                            ins.erase(std::remove_if(ins.begin(), ins.end(), [&](const Cand& c) {
                                          if (dropped < k && u[c.e] >= top) { ++dropped; return true; }
                                          return false;
                                      }), ins.end());
                        }
                        if (ins.empty() || outs.empty()) continue;
                        const size_t P = (size_t) ram_adapt2.per_layer;
                        const size_t ni = std::min(ins.size(), 2 * P), no = std::min(outs.size(), P);
                        std::partial_sort(ins.begin(), ins.begin() + (std::ptrdiff_t) ni, ins.end(),
                                          [](const Cand& a, const Cand& b) { return a.score > b.score; });
                        std::partial_sort(outs.begin(), outs.begin() + (std::ptrdiff_t) no, outs.end(),
                                          [](const Cand& a, const Cand& b) { return a.score < b.score; });
                        ins.resize(ni);
                        for (Cand& c : ins) {   // the miss cost of the few best: cold on the drive, or in the page cache
                            if (src.ram_queued(l, c.e)) { c.score = -1.0f; continue; }
                            c.score *= src.tier_of(l, c.e) == 1 ? ram_adapt2.cached_w : 1.0f;
                        }
                        std::sort(ins.begin(), ins.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
                        const double bytes = (double) std::max<uint64_t>(1, src.blob_bytes(l));
                        size_t oi = 0;
                        for (size_t k = 0; k < ins.size() && oi < no; ++k) {
                            while (oi < no && src.ram_queued(l, outs[oi].e)) ++oi;
                            if (oi >= no) break;
                            const float gain = ins[k].score - outs[oi].score;   // the evicted one would cost a drive read
                            if (ins[k].score <= 0.0f || gain < 1.0f) break;
                            swaps.push_back({(float) (gain / bytes * 1048576.0), (int32_t) l, ins[k].e, outs[oi].e});
                            ++oi;
                        }
                    }
                    std::sort(swaps.begin(), swaps.end(), [](const RamSwap2& a, const RamSwap2& b) { return a.gain > b.gain; });
                    int staged_n = 0;
                    for (const RamSwap2& x : swaps) {
                        if (staged_n >= ram_adapt_n) break;
                        if (src.ram_stage(x.layer, x.in, x.out)) ++staged_n;
                    }
                } else
                if (ram_adapt_n > 0 && !first_window && !drive.d.usage.empty() && pending.empty() &&
                    (!ajob || astate == AState::Idle) && src.complement_ready() && !multi_gpu && !peer.valid() &&
                    !remote_opt && o.batch == 0) {
                    // up to kRamAdaptPerLayer pairs a layer: the layer's most routed experts outside the copy against its least
                    // routed ones in it, paired in that order while the outgoing one is still unused
                    constexpr float kRamAdaptRouted = 2.0f, kRamAdaptUnused = 0.5f;
                    constexpr size_t kRamAdaptPerLayer = 4;
                    struct RamSwap { float gain; int32_t layer, in, out; };
                    std::vector<RamSwap> swaps;
                    std::vector<int32_t> ins, outs;
                    for (int64_t l = 0; l < g.n_layers; ++l) {
                        const float* u = drive.d.usage.data() + l * g.n_expert;
                        const int32_t* r = host_res.data() + l * g.n_expert;
                        ins.clear();
                        outs.clear();
                        for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                            if (r[e] >= 0) continue;   // in VRAM: the RAM copy does not hold it
                            if (src.has_resident(l, e)) outs.push_back(e);
                            else if (u[e] >= kRamAdaptRouted) ins.push_back(e);
                        }
                        const size_t ni = std::min(ins.size(), kRamAdaptPerLayer), no = std::min(outs.size(), kRamAdaptPerLayer);
                        std::partial_sort(ins.begin(), ins.begin() + (std::ptrdiff_t) ni, ins.end(),
                                          [&](int32_t a, int32_t b) { return u[a] > u[b]; });
                        std::partial_sort(outs.begin(), outs.begin() + (std::ptrdiff_t) no, outs.end(),
                                          [&](int32_t a, int32_t b) { return u[a] < u[b]; });
                        for (size_t i = 0; i < ni && i < no && u[outs[i]] < kRamAdaptUnused; ++i)
                            swaps.push_back({u[ins[i]] - u[outs[i]], (int32_t) l, ins[i], outs[i]});
                    }
                    std::sort(swaps.begin(), swaps.end(),
                              [](const RamSwap& a, const RamSwap& b) { return a.gain > b.gain; });
                    // the incoming expert was usually just read for the window before: a pair whose `in` still sits in that
                    // read's stage buffer moves with a copy out of it (no file read); n caps only the pairs that need the file read
                    int file_left = ram_adapt_n;
                    for (const RamSwap& x : swaps) {
                        if (src.replace_resident_staged(x.layer, x.in, x.out)) continue;
                        if (file_left > 0) { --file_left; src.replace_resident(x.layer, x.in, x.out); }
                    }
                }
                if (!adapt_tick(false)) {   // --adapt-async: the round in flight moves on a step when it can
                    print_response("ERR an adaptive refill failed\n");
                    return 1;
                }
                asw_wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - taw0).count();
                dt_adapt_wait += asw_wait_ms;
                if (hist_n > 0) {
                    // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
                    // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
                    // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
                    // was staged, and the drafted rows read unwritten slots.)
                    strata::kernels::penalty_rows(consumed.data() + pen_gen0, (int64_t) (consumed.size() - pen_gen0), window.data(), T,
                                                  hist_n, hist_stage.data());
                    const strata::core::OnDevice on_h(hist_dev);
                    cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                               cudaMemcpyHostToDevice);
                }
                // the elastic K/V: the window's cells and the drafter's beyond them
                if (kvg.on && p + T + 64 > kvg.cells &&
                    !kvg_ensure(p + T + 64, adapt_quiesce)) {
                    print_response("ERR the K/V cannot grow: no VRAM is left\n");
                    return 1;
                }
                tr("window", p, T);
                const Clock::time_point tw0 = Clock::now();
                // STRATA_SPEC_PROB: the MTP drafts' distributions q, judged by rejection sampling (core/spec_prob.hpp);
                // a suffix window and the lookup chain's tail are point masses and keep the exact-match rule
                // (never on a first window: a merged one holds prompt tokens, not drafts)
                if (gated_proposal) ver.set_spec_proposal(proposal.n_q ? proposal.mu.data() : nullptr, proposal.n_q);
                else if (strata::spec::mtp_q_applies(use_mtp, req_df2, use_mtp && mtp.prob(), w.lookup, first_window, T_mtp))
                    ver.set_spec_q(mtp.spec_q(), T_mtp - 1);   // only candidates the MTP drew
                const auto decode_pool_fn = use_mtp && mtp.prob() ? invariant_win_pool_fn : win_pool_fn;
                exit_if_gpu_fatal("");   // (the test hook counts windows here)
                if (!ver.run(T, window.data(), p, decode_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                    exit_if_gpu_fatal(drive.d.failed && drive.d.fail ? drive.d.fail : err);
                    print_response("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                if (aflipped) apply_pending(true);   // 0.1.43: every layer's copies were waited for in the window; the bookkeeping now
                // Diagnostics (as in generate): STRATA_DUMP_FIRST_LOGITS=<path> writes the logits of each request's first
                // window to <path>.<n>, so the prompt paths (a peer card's share, a split) can be compared by KL
                if (first_window) {
                    static const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS");
                    static int fl_n = 0;
                    if (fl != nullptr) {
                        std::vector<float> row((size_t) ver.vocab());
                        const std::string path = std::string(fl) + "." + std::to_string(fl_n++);
                        std::FILE* f = ver.copy_logits(merged ? T - 1 : 0, row.data()) ? std::fopen(path.c_str(), "wb") : nullptr;
                        if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                            std::fprintf(stderr, "strata serve: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", path.c_str());
                        if (f) std::fclose(f);
                    }
                }
                if (fol != nullptr) {   // STRATA_FOLLOW: the forced continuation stands in for the argmax (row i answers continuation index produced_n + i)
                    for (int i = 0; i < T; ++i) {
                        const int64_t fi = produced_n + i - (merged ? T - 1 : 0);
                        if (fi >= 0 && fi < (int64_t) fol->size()) outv[(size_t) i] = (*fol)[(size_t) fi];
                    }
                }
                // a merged window commits every row, as the prompt windows do
                int a = merged ? T - 1 : strata::spec::accepted_prefix(window.data() + 1, outv.data(), T - 1);
                if (fol != nullptr && fol_lp != nullptr) {   // rows 0..a have valid logits (their inputs are the accepted tokens)
                    std::vector<int32_t> fol_tg((size_t) a + 1, -1);
                    for (int i = 0; i <= a; ++i) {
                        const int64_t fi = produced_n + i - (merged ? T - 1 : 0);
                        fol_tg[(size_t) i] = merged && i < T - 1 ? window[(size_t) i + 1]
                                           : (fi >= 0 && fi < (int64_t) fol->size()) ? (*fol)[(size_t) fi] : -1;
                    }
                    std::string fe;
                    if (!ver.window_logprobs(fol_tg.data(), a + 1, p, 248046, fol_lp, fe))
                        std::fprintf(stderr, "strata serve: STRATA_FOLLOW_LOGPOS: %s\n", fe.c_str());
                }
                if (!merged) {   // (a merged window's prompt rows must all be committed)
                    a = stop_bounded(a, outv.data());
                    wplan.count(w, a);
                }
                if (!w.lookup && !first_window)
                    for (int d = 0; d < T_mtp - 1 && d < 8 && d <= a; ++d) { ++dp_tested[d]; dp_ok[d] += d < a; }
                if (strata::core::MtpDrafter::top2_env() && !w.lookup && !first_window && a < T_mtp - 1 && a < 8) {
                    ++t2_rej[a];
                    if (mtp.top2(a) == outv[(size_t) a]) ++t2_hit[a];
                }
                const Clock::time_point tw1 = Clock::now();
                std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
                bool adapt_ok = true;
                // (--adapt-async 1: the asynchronous tier above instead, ticked before the window)
                if (!drive.d.usage.empty() && !ajob && ((rounds + 1) % o.adapt_iv(rounds)) == 0)
                    adapt_thr = std::thread([&] { strata::platform::name_this_thread("strata-adapt"); strata::aux_cpus::pin_current_thread(); adapt_ok = adapt(); });
                if (!ver.commit(a + 1, err)) {
                    exit_if_gpu_fatal(err);
                    if (adapt_thr.joinable()) adapt_thr.join();
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                if (nonfinite_hit(ver, first_window ? "first decode" : "decode")) {   // #879: none of its tokens goes out
                    if (adapt_thr.joinable()) adapt_thr.join();
                    if (!adapt_ok) {
                        print_response("ERR an adaptive refill failed\n");
                        return 1;
                    }
                    finish = "nonfinite";
                    break;
                }
                // a merged window: the drafter's rows of the prompt tokens in it, as the prompt windows give them
                if (merged && use_mtp && !mtp.prefill(ver.final_R_all(), window.data() + 1, T - 1, p, err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
#if defined(STRATA_HAVE_DFLASH2)
                if (use_df2) df2.note_rows(window.data(), a + 1, p);
                // --draft auto: the drafter this request does not use keeps up with the commit (its context / its K/V), so a later request can use it
                if (use_df2 && o.draft_auto && !merged && !draft_keep_off) {
                    const Clock::time_point tk0 = Clock::now();
                    // queued on the idle drafter's own stream, overlapped with the other's draft; keep_sync() below waits before the next window rewrites the taps / rows
                    if (!req_df2) {
                        if (!df2.context_async(ver.tap_dev(), a + 1, p, err)) { if (adapt_thr.joinable()) adapt_thr.join(); print_response("ERR %s\n", err.c_str()); return 1; }
                        keep_pending = true;
                    } else if (use_mtp) {
                        if (!mtp.prefill(ver.final_R_all(), outv.data(), a + 1, p, err, false)) { if (adapt_thr.joinable()) adapt_thr.join(); print_response("ERR %s\n", err.c_str()); return 1; }
                        keep_pending = true;
                    }
                    keep_ms += std::chrono::duration<double, std::milli>(Clock::now() - tk0).count();
                }
#endif
                if (!merged) {   // (a merged window's rows are prompt tokens, not drafts)
                    draft_offered += T - 1;
                    draft_accepted += a;
                }
                first_window = false;
                bool eos = false;
                for (int i = merged ? a : 0; i <= a && produced_n < max_new && !eos; ++i) {   // (merged: row T-1 only)
                    print_response("T %d\n", (int) outv[(size_t) i]);
                    if (req_logprobs >= 0) emit_lp_line(ver, i, outv[(size_t) i], req_logprobs, req_allowed);
                    strata::core::progress_beat();
                    ++produced_n;
                    wplan.append(outv[(size_t) i]);
                    eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                std::fflush(stdout);
                ++rounds;
                if (drive.d.fs) drive.d.fs->completed.fetch_add(1, std::memory_order_release);
                const Clock::time_point tw2 = Clock::now();
                // coupled drafts with penalties: the next window's row-0 history (`consumed` holds this window's
                // commit, outv[a] is its row 0) - the drafts extend it on the device as the verify rows will
                if (use_mtp && hist_n > 0 && mtp.coupled() && !eos && produced_n < max_new)
                    mtp.set_draft_history(consumed.data() + pen_gen0, (int64_t) (consumed.size() - pen_gen0), outv[(size_t) a]);
                generation = {};   // a skipped or failed production must not expose an older q view
#if defined(STRATA_HAVE_DFLASH2)
                // The request ends here, so no draft follows and the window's committed rows would never reach the drafter's ring or its log: every turn would
                // leave a hole of a + 1 positions behind it, which a later turn that resumes after this one finds blank (and cannot rebuild). Update them now.
                static const bool df2_end_update = std::getenv("STRATA_DF2_END_UPDATE") == nullptr || std::getenv("STRATA_DF2_END_UPDATE")[0] != '0';   // =0: the pre-fix behaviour (diagnostics)
                if (df2_end_update && use_df2 && req_df2 && !merged && (eos || produced_n >= max_new)) {
                    const float* tp_end = df2_win_taps(a + 1, err);   // (on a layer split: every stage's taps gathered onto the drafter's card)
                    if (tp_end == nullptr || !df2.context(tp_end, a + 1, p, err)) { if (adapt_thr.joinable()) adapt_thr.join(); print_response("ERR %s\n", err.c_str()); return 1; }
                }
#endif
                const GatedDraftLimit draft_limit(mtp, gated_proposal, S_mtp, p + a + 1, o.max_context);
                const bool drafted = !draft_on || eos || produced_n >= max_new ||
                                     (req_df2 ? ((!merged && defer_df2(a)) || draft_df2(a)) : mtp.draft_long(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p,
                                                                               nullptr, gated_proposal ? &generation : nullptr));
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += merged ? 1 : T;   // a merged window's prompt rows are not decode rows
                }
                pcie_auto_step(req_pcie_frac);   // the default share, refined from the CPU pool's speed (PcieAuto); a request's own share is left alone
                if (adapt_thr.joinable()) {
                    const Clock::time_point taj0 = Clock::now();
                    adapt_thr.join();
                    const double asw_join = std::chrono::duration<double, std::milli>(Clock::now() - taj0).count();
                    dt_adapt_join += asw_join;
                    asw_wait_ms += asw_join;
                }
                if (!adapt_ok) {
                    print_response("ERR an adaptive refill failed\n");
                    return 1;
                }
                // the session's choice of the adaptive tier (adapt_async_default.hpp): ~64 windows on the blocking tier, then
                // ~64 on the asynchronous one if the swaps held the windows up, kept only if the decode got faster
                if (!asw_skip && (asw.phase() == strata::core::AdaptAsyncSelf::Phase::Measure || asw.trialing())) {
                    const bool was_trial = asw.trialing();
                    const double asw_ms = std::chrono::duration<double, std::milli>(Clock::now() - taw0).count();
                    const auto act = asw.window(asw_ms, asw_wait_ms, a + 1);
                    if (act == strata::core::AdaptAsyncSelf::Action::ToAsync) {
                        ajob = std::make_unique<JobThread>();
                        std::fprintf(stderr, "strata serve: adaptive tier: the swaps held the windows up by %.1f%% (%.2f ms per "
                                             "token): trying the asynchronous tier for a while\n",
                                     100.0 * asw.wait_fraction(), asw.measure_ms_per_token());
                    } else if (act == strata::core::AdaptAsyncSelf::Action::ToBlocking) {
                        adapt_quiesce();
                        ajob.reset();
                        std::fprintf(stderr, "strata serve: adaptive tier: the asynchronous tier was slower (%.2f against "
                                             "%.2f ms per token): back to the blocking tier for this session\n",
                                     asw.trial_ms_per_token(), asw.measure_ms_per_token());
                    } else if (was_trial && asw.kept()) {
                        std::fprintf(stderr, "strata serve: adaptive tier: the asynchronous tier stays on (%.2f against %.2f ms "
                                             "per token)\n", asw.trial_ms_per_token(), asw.measure_ms_per_token());
                    }
                }
                if (!drafted) {
                    print_response("ERR %s\n", err.c_str());
                    return 1;
                }
                const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - round0).count();
                if (gated_proposal && !eos && !w.first) {
                    wplan.source.observe_committed(w.lookup, w.lookup ? T : std::min(S_mtp, room), a, w.match);
                    // after the next draft: its counters too
                    wplan.source.defer_cost(std::max<int64_t>(p + T - 1, p + a + 1 + S_mtp), T, round_ms);
                }
                if (df2_trace != nullptr && !w.first) {
                    std::fprintf(df2_trace, "R %lld %d %d %d %.3f %d %d |", (long long) p, T, a, w.lookup ? 1 : 0,
                                 round_ms, tr_k, tr_match);
                    for (int i = 0; i < 7; ++i) std::fprintf(df2_trace, " %.4f", i < S ? (double) wplan.window_probs()[i] : 0.0);
                    std::fprintf(df2_trace, " |");
                    for (int i = 0; i < 7; ++i) std::fprintf(df2_trace, " %d", (int) tr_d[(size_t) i]);
                    std::fprintf(df2_trace, " |");
                    for (int i = 0; i < tr_k; ++i) std::fprintf(df2_trace, " %d", (int) tr_s[(size_t) i]);
                    std::fprintf(df2_trace, " |");
                    for (int i = 0; i <= a; ++i) std::fprintf(df2_trace, " %d", (int) outv[(size_t) i]);
                    std::fprintf(df2_trace, "\n");
                    tr_k = 0;
                }
                wplan.observe(w, a, round_ms, eos);
                if (eos) { finish = "stop"; break; }
                if (input.is_stop_requested()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            if (!keep_sync()) { print_response("ERR %s\n", err.c_str()); return 1; }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            if (use_df2 && o.draft_auto && bandit_key != 0 && produced_n >= 48 && decode_ms > 0.0) {
                DraftBandit& b = bandit[bandit_key];
                const int w = req_df2 ? 1 : 0;
                const double tps = 1000.0 * (double) produced_n / decode_ms;
                b.tps[w] = b.n[w] == 0 ? tps : 0.65 * b.tps[w] + 0.35 * tps;
                ++b.n[w];
            }
            if (draft_log && use_df2)
                std::fprintf(stderr, "strata draft: request %lld drafter=%s window<=%d tokens=%lld rounds=%lld decode_ms=%.1f keep_ms=%.1f (%.3f per round)\n",
                             (long long) ++draft_req_count, req_df2 ? "dflash2" : "mtp", S_mtp, (long long) produced_n, (long long) (rounds - rounds_at_start),
                             decode_ms, keep_ms, rounds > rounds_at_start ? keep_ms / (double) (rounds - rounds_at_start) : 0.0);
            // the last commit (set_commit_async): the session is complete before anything reads or copies it
            if (!ver.wait_commit(err)) {
                exit_if_gpu_fatal(err);
                print_response("ERR %s\n", err.c_str());
                return 1;
            }
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                if (!pl_ran) {
                    const double dt_other = decode_ms - dt_adapt_wait - dt_run - dt_commit - dt_draft - dt_adapt_join;
                    std::fprintf(stderr,
                                 "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                 "adapt wait %.2f + verify %.2f (GPU-reach wait %.2f + per-layer host %.2f "
                                 "[plan %.2f actq %.2f jobs %.2f CPU %.2f] + stage %.2f) + commit/emit %.2f + "
                                 "draft %.2f + adapt join %.2f + other %.2f; per layer-window: CPU experts "
                                 "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f; %.1f swaps/window\n",
                                 (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_adapt_wait / w,
                                 dt_run / w, (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w,
                                 (d1.plan - ds0.plan) / w, (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w,
                                 (d1.run - ds0.run) / w, (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w,
                                 dt_adapt_join / w, dt_other / w, (d1.misses - ds0.misses) / (w * L),
                                 (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L),
                                 (d1.pcie - ds0.pcie) / (w * L), (double) ((ajob ? a_swapped : adapt_nswaps) - dt_sw0) / w);
                } else {   // Keep the pipeline's existing timing format.
                    std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                         "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                         "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                         "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f; adapt wait %.2f join %.2f, %.1f swaps/window\n",
                                 (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                                 (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                                 (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                                 (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                                 (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L),
                                 dt_adapt_wait / w, dt_adapt_join / w, (double) ((ajob ? a_swapped : adapt_nswaps) - dt_sw0) / w);
                }
                if (aov)
                    std::fprintf(stderr, "strata decode adapt overlap: %lld windows took swaps still landing, a layer waited %.2f ms in all so far\n",
                                 (long long) aov_flips, ver.ms_gate);
                for (int st = 0; st < n_stages; ++st) {   // every stage's GPU profile, not only the first card's
                    const std::string pr = stage_ver(st).profile_report();
                    if (pr.empty()) continue;
                    if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                    else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                }
            }
            if (drive.d.fs) std::fprintf(stderr, "strata serve: %s\n", drive.d.fs->report().c_str());
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0 && req_ckpt;   // ckpt=0: nothing to continue or park (#830)
                if (nf_hit) live_ok = false;   // #879 (the next request's start drops the rest)
            } else if (!nf_hit && pp_exact && pp_reached > 0 && o.prompt_cache > 0 && req_ckpt &&
                       pp_reached <= (int64_t) cur.size()) {
                // #1620: stopped between two windows (or before the first token was read): the session is exactly at
                // pp_reached, so it stays the live conversation - the next request continues from it or parks it.
                // Without this a cancelled request that had restored a parked conversation dropped it.
                live.assign(cur.begin(), cur.begin() + pp_reached);
                live_imgs = imgs_below(req_imgs, pp_reached);
                live_ok = true;
                std::fprintf(stderr, "strata serve: cancelled request keeps its conversation: %lld tokens live\n",
                             (long long) pp_reached);
            }
            if (nib::config().recheck && nib::any_on() && !cancelled) {
                // STRATA_EXPERT_NIBBLE_RECHECK=1 (debug): every converted slot's hash again on its own device, against the
                // hash recorded at load - a path that wrote a slot after the seal is found here
                std::string rerr;
                int64_t bad_layer = -1;
                const auto rt0 = Clock::now();
                bool ok = nib_x.recheck(rerr, bad_layer);
                int64_t checked = nib_x.n;
                for (size_t i = 0; ok && i < nib_st.size(); ++i) {
                    ok = nib_st[i].recheck(rerr, bad_layer);
                    checked += nib_st[i].n;
                }
                if (!ok) {
                    std::fprintf(stderr, "strata serve: STRATA_EXPERT_NIBBLE_RECHECK: %s\n", rerr.c_str());
                    if (bad_layer >= 0) nib::fatal(bad_layer, "STRATA_EXPERT_NIBBLE_RECHECK", "a slot changed after the load");
                    std::fflush(stderr);
                    std::abort();
                }
                std::fprintf(stderr, "strata serve: STRATA_EXPERT_NIBBLE_RECHECK: %lld 4-bit slots hash as at load (%.0f ms)\n",
                             (long long) checked, std::chrono::duration<double, std::milli>(Clock::now() - rt0).count());
            }
            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
            if (state_hash && !cancelled && o.prompt_cache > 0) {   // every finished request, ckpt=0 too (parity gates)
                // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
                // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    print_response("ERR synchronizing state fingerprint\n");
                    return 1;
                }
                const int64_t L = (int64_t) live.size();
                const strata::kernels::QsaShapes qs = strata::kernels::qsa_shapes(g);
                bool hash_ok = true;
                std::array<uint8_t, 65536> hash_buffer;
                auto hash_dev = [&](const void* p, size_t bytes, uint64_t h) {
                    for (size_t offset = 0; hash_ok && offset < bytes;) {
                        const size_t n = std::min(hash_buffer.size(), bytes - offset);
                        // VRAM or a streamed host copy, with fixed diagnostic workspace.
                        if (cudaMemcpy(hash_buffer.data(), static_cast<const uint8_t*>(p) + offset, n, cudaMemcpyDefault) != cudaSuccess) {
                            hash_ok = false;
                            break;
                        }
                        h = fnv1a(hash_buffer.data(), n, h);
                        offset += n;
                    }
                    return h;
                };
                // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
                auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
                    const int64_t ps = qs.page_size;
                    for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                        for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                            const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                            const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                            h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                        }
                    return h;
                };
                const ConvStateSizes z = conv_state_sizes(g, ss);
                uint64_t h_gdn = hash_dev(ss.gdn_state, z.gdn, 1469598103934665603ull);
                if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr && ss.gdn_alloc > 0) {   // per GDN layer: which one differs first
                    const size_t per = z.gdn / (size_t) ss.gdn_alloc;
                    std::string s;
                    char b[8];
                    for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
                        std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                        s += b;
                    }
                    std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
                }
                uint64_t h_ple = hash_dev(ss.ple_hist, ss.ple_hist ? z.ple : 0, 1469598103934665603ull);
                uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
                // pooled= keeps its 0.1.29 meaning: the completed rows [0, L / idx_block) only.  pooled_full= adds the
                // spare row at L / idx_block (the `dead` key the next block completion overwrites), which a
                // conversation restore writes back; dead= is the spare key itself
                uint64_t h_dead = h_tail, h_pool_full = h_tail;
                const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
                // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
                auto kv_arrays = [&](const strata::core::QsaState& st) {
                    const bool h = st.kv_mode != 0;
                    std::vector<std::pair<const void*, int64_t>> a;
                    if (st.kv_q4) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
                    } else if (st.kv_hybrid) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                             {h ? st.host.k_scale : st.k_scale, scb}};
                    } else if (st.kv_int8) {
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                             {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
                    } else {
                        a = {{h ? st.host.k_pool : st.k_pool, qs.head_dim * 2},
                             {h ? st.host.v_pool : st.v_pool, qs.head_dim * 2}};
                    }
                    return a;
                };
                const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                           ss.max_cells);   // = the primary state's max_cells
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {   // this session's owned QSA ordinals only
                    const strata::core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
                    h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
                    h_dead = hash_dev(st.idx_dead, z.dead, h_dead);
                    h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
                    h_pool_full = hash_dev(st.idx_pooled, (size_t) (L > 0 ? L / qs.idx_block + 1 : 0) * qs.idx_dim * 4,
                                           h_pool_full);
                    // KV streaming: the host copy is the identity layout and holds every cell
                    for (const auto& [pool, w] : kv_arrays(st)) {
                        h_kv = hash_cells(pool, w, 0, L, h_kv);
                        h_stale = hash_cells(pool, w, L, end_cell, h_stale);
                    }
                }
                uint64_t h_mtp = 1469598103934665603ull;
                if (use_mtp) {
                    const strata::core::QsaState& ms = mtp.kv_state();
                    const int64_t mL = std::min<int64_t>(L, ms.max_cells);
                    for (const auto& [pool, w] : kv_arrays(ms))
                        if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
                }
                if (!hash_ok) {
                    print_response("ERR reading state fingerprint\n");
                    return 1;
                }
                std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                                     "kv=%016llx mtp=%016llx stale=%016llx dead=%016llx pooled_full=%016llx ple_prev=%d,%d\n", (long long) L,
                             (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                             (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                             (unsigned long long) h_stale, (unsigned long long) h_dead,
                             (unsigned long long) h_pool_full, ss.ple_prev[0], ss.ple_prev[1]);
            }
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            const int64_t req_offload = drive.d.offload_entries - offload0;
            // #471: the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a
            // cancel stopped it part-way (a cancelled request used to be logged and counted as having read them all)
            const int64_t fresh = n - resume;
            const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            //      [RAM blobs] [file blobs] [file MB]   (CS-T tiers; appended, so an older server reads the rest)
            //      [prompt tokens read]   (#471: fewer than <prompt> - <reused> when a cancel stopped the read)
            //      [offloaded]   (#588: the decode's routed experts the GPU read over PCIe or another GPU computed;
            //                    not in [lookups])
            //      [lookup-chain accepted] [lookup-chain offered] [suffix accepted] [suffix offered] [windows]
            //      (only with --lookup-chain)
            //      chunk=<tokens> experts_streamed=<blobs>   (key=value, last: the first chunk of the request's longest
            //      batched segment, 0 when it read none; the expert blobs its prompt read copied host -> device on
            //      every stage - not the CPU share's, the stage helper's or a --peer-device peer's)
            char chain_txt[128] = "";
            if (o.lookup_chain > 0)
                std::snprintf(chain_txt, sizeof(chain_txt), " %lld %lld %lld %lld %lld",
                              (long long) wplan.chained.ok, (long long) wplan.chained.drafts,
                              (long long) wplan.lookup.ok, (long long) wplan.lookup.drafts, (long long) dec_windows);
            if (yielded_at >= 0) print_response("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
            print_response("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s chunk=%lld "
                        "experts_streamed=%lld\n",
                        (long long) produced_n,
                        (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                        (long long) resume, (long long) req_hits, (long long) req_look,
                        (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                        (long long) req_offload, chain_txt, (long long) req_chunk,
                        (long long) (sp.chain_experts_streamed() - streamed0));
            std::fflush(stdout);
            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                bool cont = !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                            (int64_t) live.size() == p;
                const auto tc0 = Clock::now();
                if (cont) park_slot(admit_slot);   // the conversation the slot holds, before this one overwrites it
                if (cont && !copy_slot(true, admit_slot, live, nullptr, err)) {
                    std::fprintf(stderr, "strata serve: batch admission failed: %s\n", err.c_str());
                    bs[(size_t) admit_slot].cached = false;   // its sessions may be half written
                    cont = false;
                }
                const auto tc1 = Clock::now();
                if (cont) {
                    ver.set_slot_sampling(admit_slot, req_sp);   // the request's own sampling, row by row
                    if (!ver.set_slot_logit_bias(admit_slot, req_bias, err)) {   // and its own logit bias
                        print_response("ERR %s\n", err.c_str());
                        return 1;
                    }
                    BSlot& sl = bs[(size_t) admit_slot];
                    if (ckpt_pool.on())
                        for (ConvCheckpoint& c : sl.checks) ckpt_pool.give(std::move(c));
                    sl = BSlot{};
                    const auto tc2 = Clock::now();   // (the slot's old checkpoints freed)
                    sl.active = true;
                    sl.x = x;
                    sl.p = p;
                    sl.produced = 1;
                    sl.max_new = admit_max_new;
                    sl.t0 = Clock::now();
                    sl.ids = live;
                    {   // the penalty window of the slot's decode: the reply, from the last prompt token (the first window fed it), as the solo loop counts it
                        const bool slot_pen = req_sp.penalty_last_n > 0 && (req_sp.penalty_repeat != 1.0f || req_sp.penalty_freq != 0.0f || req_sp.penalty_present != 0.0f);
                        sl.pen_n = slot_pen ? std::min(req_sp.penalty_last_n, kPenaltyWindowCap) : 0;
                        sl.reply0 = (int64_t) strata::kernels::penalty_window_from(live.size() > 0 ? live.size() - 1 : 0, pen_prompt);
                        if (req_pen_from >= 0 && !pen_prompt) sl.reply0 = std::min<int64_t>(sl.reply0, req_pen_from);   // a continued request (prompt + what it generated): the reply began earlier
                    }
#if defined(STRATA_HAVE_DFLASH2)
                    if (batch_df2) {
                        // the slot's first drafts: the last prompt row's taps (still in the verifiers from the admission's window) and the token it produced
                        strata::core::Dflash2Drafter& d = *slot_df2[(size_t) admit_slot];
                        int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                        float pp[strata::core::Dflash2Drafter::kDrafts];
                        if (!ver.gather_taps(d.landing(), d.device(), 0, 1, 0, d.stream(), err) ||
                            !d.draft(d.landing(), 1, sl.p - 1, sl.x, dd, pp, err)) {
                            std::fprintf(stderr, "strata batch: DFlash2 admission for slot %d failed: %s\n", admit_slot, err.c_str());
                            return 1;
                        }
                        for (int j = 0; j < strata::core::Dflash2Drafter::kDrafts && j < (int) sl.draft.size(); ++j) { sl.draft[(size_t) j] = dd[j]; sl.dprob[(size_t) j] = pp[j]; }
                        sl.draft_ready = true;
                    } else
#endif
                    if (batch_mtp) {
                        int nd = 1;   // (STRATA_PIPED_MTP: up to its proposals per window, as --spec-min-p allows)
                        if (!slot_mtp[(size_t) admit_slot]->draft_first(1, ver.final_R_all(), sl.x,
                                                                        sl.p - 1, sl.draft.data(), err, nullptr,
                                                                        piped_mtp ? (float) o.spec_min_p : 0.0f, &nd)) {
                            std::fprintf(stderr, "strata batch: MTP admission for slot %d failed: %s\n",
                                         admit_slot, err.c_str());
                            return 1;
                        }
                        sl.n_drafts = std::max(1, nd);
                        sl.draft_ready = true;
                    }
                    sl.cvec = cvec_cached;
                    sl.img = !live_imgs.empty();   // pictures: not matched again by tokens alone, so not cached
                    const ConvCheckpoint* best = nullptr;
                    size_t best_at = 0;   // its place in `checks` (STRATA_SLOT_CKPT_MOVE)
                    for (size_t i = 0; i < checks.size(); ++i) {
                        const ConvCheckpoint& c = checks[i];
                        if (c.ids.size() < live.size() && (best == nullptr || c.ids.size() > best->ids.size()) &&
                            std::equal(c.ids.begin(), c.ids.end(), live.begin())) {
                            best = &c;
                            best_at = i;
                        }
                    }
                    bool best_lent = false;
                    // STRATA_SLOT_CKPT_MOVE: the checkpoint this request mounted from another slot (that slot keeps it)
                    if (lent_slot >= 0 && lent_slot != admit_slot && (best == nullptr || (int64_t) best->ids.size() < resume))
                        for (const ConvCheckpoint& c : bs[(size_t) lent_slot].checks)
                            if ((int64_t) c.ids.size() == resume && c.ids.size() < live.size() &&
                                std::equal(c.ids.begin(), c.ids.end(), live.begin())) {
                                best = &c;
                                best_lent = true;
                            }
                    // STRATA_SLOT_CKPT_MOVE: this request's own turn checkpoint (or the one it moved out of this slot)
                    // moves into the slot; a pinned prefix, a root and any other the chain shares are copied as before
                    const bool move_best = slot_ckpt_move_env() && best != nullptr && !best_lent && !sl.img &&
                                           !best->pinned && ((turn_at > 0 && (int64_t) best->ids.size() == turn_at) ||
                                            (int64_t) best->ids.size() == moved_in);
                    const auto tc3 = Clock::now();
                    if (move_best) {
                        sl.checks.push_back(std::move(checks[best_at]));
                        checks.erase(checks.begin() + (std::ptrdiff_t) best_at);
                    } else if (best != nullptr && !sl.img) {
                        if (ckpt_pool.on()) {   // a copy into a dropped checkpoint's vectors (their pages are in)
                            ConvCheckpoint c = ckpt_pool.take();
                            c = *best;
                            sl.checks.push_back(std::move(c));
                        } else {
                            sl.checks.push_back(*best);
                        }
                    }
                    const auto tc4 = Clock::now();
                    std::fprintf(stderr, "strata batch: slot %d takes %lld tokens (copied in %.1f ms)\n", admit_slot,
                                 (long long) live.size(), std::chrono::duration<double, std::milli>(tc4 - tc0).count());
                    pc.admission((long long) live.size(), prompt_ms);
                    if (trace) {   // STRATA_TRACE: the admission's phases (ms), from the drain before BGEN to this copy
                        auto span = [](Clock::time_point a, Clock::time_point b) {
                            return std::chrono::duration<double, std::milli>(b - a).count();
                        };
                        const double ck = req_times.sync_ms + req_times.save_ms + req_times.evict_ms;
                        std::fprintf(stderr, "strata trace: admission of slot %d: drain %.1f, prompt %.1f (read %.1f, "
                                             "windows %.1f, %d checkpoint(s) %.1f = sync %.1f + save %.1f + evict %.1f, "
                                             "rest %.1f), first window %.1f, slot copy %.1f, slot freed %.1f, "
                                             "checkpoint copy %.1f%s\n", admit_slot, adm_drain_ms, prompt_ms,
                                     req_times.read_ms, req_times.windows_ms, req_times.ckpts, ck, req_times.sync_ms,
                                     req_times.save_ms, req_times.evict_ms,
                                     prompt_ms - req_times.read_ms - req_times.windows_ms - ck, decode_ms, span(tc0, tc1),
                                     span(tc1, tc2), span(tc3, tc4), move_best ? " (moved)" : "");
                    }
                }
                print_response("BADM %d %d\n", admit_slot, cont ? 1 : 0);
                if (trace && piped && cont) {
                    rows_wait = "the admission";
                    rows_wait_t = Clock::now();
                }
                adm_drain_ms = 0.0;
                std::fflush(stdout);
                admit_slot = -1;
            }
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            // "12288 of 98179" when cancelled mid-read (#471), the rate from what was read
            char read_txt[64];
            if (cancelled)
                std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
            else
                std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
            std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                         (long long) n, (long long) resume, read_txt, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            if (serve_adapt_drain_ms >= 1.0)
                std::fprintf(stderr, "strata serve: the request first waited %.0f ms for the adaptive tier's round in flight\n",
                             serve_adapt_drain_ms);
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
            // to read (--pcie-frac) are in neither count - #588: so that share is said beside it (raising --pcie-frac
            // raises the hit rate while the PCIe reads may make the decode slower)
            if (req_look > 0) {
                char off[160] = "";
                if (req_offload > 0)
                    std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                                  "all %lld routed)", (long long) req_offload,
                                  100.0 * (double) req_offload / (double) (req_look + req_offload),
                                  (long long) (req_look + req_offload));
                std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s\n",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look, off);
            }
            if (drive.d.tail_experts > 0)   // STRATA_ROUTE_TAIL_SKIP, cumulative
                std::fprintf(stderr, "strata serve: route tail skip: %lld missed experts skipped (%lld entries) since the "
                                     "start\n", (long long) drive.d.tail_experts, (long long) drive.d.tail_skipped);
            // the resident RAM mode, cumulative: experts read from experts.bin since the copy was made (what the plain
            // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
            if (src.complement_ready())
                std::fprintf(stderr, "strata serve: resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                     "VRAM tier, %lld blob reads from the file\n",
                             (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                             (long long) src.file_reads());
            if (ram_adapt2.on)   // 149-prefetch: the staged promotions, cumulative
                std::fprintf(stderr, "strata serve: resident RAM (STRATA_RAM_ADAPT2): %lld promotions read on the I/O thread, "
                                     "%lld installed, %lld skipped at the install (%lld of them: the GPU tier took one of the pair)\n",
                             (long long) src.ram_staged_reads(),
                             (long long) ram_adapt2.installed, (long long) (src.ram_install_skips() + src.ram_install_gpu()), (long long) src.ram_install_gpu());
            if (src.ram_adapts() > 0)   // STRATA_RAM_ADAPT, cumulative
                std::fprintf(stderr, "strata serve: resident RAM: %lld experts moved in in place of unused ones, %lld of "
                                     "them out of a stage buffer (STRATA_RAM_ADAPT)\n", (long long) src.ram_adapts(),
                                     (long long) src.ram_staged());
            if (ajob)   // --adapt-async, cumulative (a round may still be in flight: it lands before the next request)
                std::fprintf(stderr, "strata serve: asynchronous adaptive tier: %lld rounds, %lld experts swapped in, "
                                     "%.1f ms per round (start to flip, between windows)",
                             (long long) a_rounds, (long long) a_swapped, a_rounds > 0 ? a_ms / (double) a_rounds : 0.0);
            if (ajob) {
                if (a_arena)
                    std::fprintf(stderr, " (arena, copy-in only, batch %zu, wait %d, pick %s %.2f ms/round)\n", a_batch,
                                 (int) a_wait, a_inline ? "inline" : "on the job", a_rounds > 0 ? a_pick_ms / (double) a_rounds : 0.0);
                else
                    std::fprintf(stderr, "\n");
            }
            if (src.exchange_rotation())
                std::fprintf(stderr, "strata serve: exchange rotation: %llu blocks, %llu host memcpy bytes avoided (cumulative payload)\n",
                             (unsigned long long)src.rotated_exchanges(),
                             (unsigned long long)src.avoided_exchange_copy_bytes());
            // CS-T: the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
            if (srcp == &src) {
                std::fprintf(stderr, "strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                                     "%.1f MB read%s\n", (long long) req_hits, (long long) src.ram_reads(),
                             (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6,
                             src.gguf_mode() ? " (the GGUF in place)" : "");
                // what the OS did for this request (since the previous one): the drive's share of the file tier's reads
                // against the page cache's, and what the I/O path did about it
                static strata::platform::ProcIo io_prev;
                static strata::core::FileExpertSource::IoCounters ic_prev;
                static uint64_t fb_prev = 0;
                const strata::platform::ProcIo io_now = strata::platform::proc_io_sample();
                const strata::core::FileExpertSource::IoCounters ic = src.io_counters();
                const uint64_t fb = src.file_read_bytes();
                if (io_now.valid) {
                    std::fprintf(stderr, "strata serve: file tier I/O this request: the OS read %.1f MB from storage (%llu major faults) for "
                                         "%.1f MB of expert reads", (double) (io_now.read_bytes - io_prev.read_bytes) / 1e6,
                                 (unsigned long long) (io_now.major_faults - io_prev.major_faults), (double) (fb - fb_prev) / 1e6);
                    if (src.io_stats())
                        std::fprintf(stderr, "; page cache held %.1f MB at the read, not %.1f MB",
                                     (double) (ic.cached_bytes - ic_prev.cached_bytes) / 1e6,
                                     (double) (ic.uncached_bytes - ic_prev.uncached_bytes) / 1e6);
                    if (src.io_prefetch())
                        std::fprintf(stderr, "; io prefetch: %llu read ahead (%.1f MB by pread), %llu used, %llu unused, %llu skipped "
                                             "(cached), %llu dropped; decode waited %.1f ms in %llu fetches",
                                     (unsigned long long) (ic.pf_read_blobs - ic_prev.pf_read_blobs),
                                     (double) (ic.pread_bytes - ic_prev.pread_bytes) / 1e6,
                                     (unsigned long long) (ic.pf_used - ic_prev.pf_used),
                                     (unsigned long long) (ic.pf_unused - ic_prev.pf_unused),
                                     (unsigned long long) (ic.pf_resident_skips - ic_prev.pf_resident_skips),
                                     (unsigned long long) (ic.pf_dropped - ic_prev.pf_dropped),
                                     (double) (ic.crit_us - ic_prev.crit_us) / 1e3,
                                     (unsigned long long) (ic.crit_n - ic_prev.crit_n));
                    if (src.io_prefetch() && src.gate_on())   // 149 STRATA_IO_PF_GATE: the decode's held share, the gate
                        std::fprintf(stderr, "; decode gate: cache room %.2f of the experts, read-ahead used %.0f%%, page cache "
                                             "held %.0f%% of the decode's reads (EWMA), %s, closed in %lld windows so far, %lld flips",
                                     src.gate_ratio(), 100.0 * src.gate_useful(), 100.0 * std::max(0.0, src.gate_held()),
                                     src.gate_open() ? "open" : "closed", (long long) src.gate_closed_windows(),
                                     (long long) src.gate_flips());
                    if (src.io_prefetch())   // 149-prefetch: bytes read ahead, of them used before their demand; the scheduler's drops
                        std::fprintf(stderr, "; read ahead %.1f MB, %.1f MB of it used; %llu late, %llu not admitted (confidence), "
                                             "%llu (budget %.0f MB/window)",
                                     (double) (ic.pf_bytes - ic_prev.pf_bytes) / 1e6,
                                     (double) (ic.pf_used_bytes - ic_prev.pf_used_bytes) / 1e6,
                                     (unsigned long long) (ic.pf_late - ic_prev.pf_late),
                                     (unsigned long long) (ic.pf_rej_conf - ic_prev.pf_rej_conf),
                                     (unsigned long long) (ic.pf_rej_budget - ic_prev.pf_rej_budget),
                                     (double) ic.pf_budget_bytes / 1048576.0);
                    std::fprintf(stderr, "\n");
                    io_prev = io_now;
                    ic_prev = ic;
                    fb_prev = fb;
                }
            }
            static const bool cold_log = std::getenv("STRATA_COLD_OVERLAP") != nullptr || std::getenv("STRATA_IO_STATS") != nullptr;
            if (drive.d.cold_mode != 0 && cold_log) {   // 149-cold STRATA_COLD_OVERLAP: this request's layers that had a blob to read
                static strata::core::ExpertDispatch cp;
                const strata::core::ExpertDispatch& c = drive.d;
                const double nl = (double) std::max<int64_t>(1, c.cold_layers - cp.cold_layers);
                std::fprintf(stderr, "strata serve: cold reads (STRATA_COLD_OVERLAP=%d) this request: %lld experts from the mapping "
                                     "directly; %lld layers with a blob to "
                                     "read (%lld such experts, %lld ready beside them, %lld pool batches); per such layer: "
                                     "R %.3f ms, C_r %.3f ms, C_c %.3f ms, the decode waited %.3f ms; total waited %.1f ms, min(R, C_r) %.1f ms\n",
                             c.cold_mode, (long long) (c.cold_zc - cp.cold_zc), (long long) (c.cold_layers - cp.cold_layers),
                             (long long) (c.cold_pending - cp.cold_pending), (long long) (c.cold_ready - cp.cold_ready),
                             (long long) (c.cold_batches - cp.cold_batches),
                             (c.ms_cold_r - cp.ms_cold_r) / nl, (c.ms_cold_cr - cp.ms_cold_cr) / nl,
                             (c.ms_cold_cc - cp.ms_cold_cc) / nl, (c.ms_cold_wait - cp.ms_cold_wait) / nl,
                             c.ms_cold_wait - cp.ms_cold_wait, c.ms_cold_hide - cp.ms_cold_hide);
                cp.cold_layers = c.cold_layers; cp.cold_pending = c.cold_pending; cp.cold_ready = c.cold_ready; cp.cold_zc = c.cold_zc;
                cp.cold_batches = c.cold_batches; cp.ms_cold_r = c.ms_cold_r; cp.ms_cold_cr = c.ms_cold_cr;
                cp.ms_cold_cc = c.ms_cold_cc; cp.ms_cold_wait = c.ms_cold_wait; cp.ms_cold_hide = c.ms_cold_hide;
            }
            // STRATA_SPLIT_TIMING: where each verify stage's host time went, cumulative per window since the start
            // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
            if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing)
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::Verifier& v = stage_ver(st);
                    const double w = v.windows > 0 ? (double) v.windows : 1.0;
                    std::fprintf(stderr, "strata serve: stage %d: %lld windows; per window: wait for the GPU %.3f ms, "
                                         "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n", st,
                                 (long long) v.windows, v.ms_wait / w, v.ms_pool / w, v.ms_host / w, v.ms_commit / w);
                }
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                // (this device's owned ordinals; a split's other stages hold theirs)
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                    const strata::kernels::KvStreamCounters c =
                        strata::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (wplan.lookup.windows > 0)
                std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                             (long long) wplan.lookup.windows, (long long) wplan.lookup.ok,
                             (long long) wplan.lookup.drafts);
            if (o.lookup_chain > 0)
                std::fprintf(stderr, "strata serve: lookup chain: %lld of %lld windows, %lld of %lld chained drafts accepted\n",
                             (long long) wplan.chained.windows, (long long) dec_windows,
                             (long long) wplan.chained.ok, (long long) wplan.chained.drafts);
            if (std::getenv("STRATA_SPEC_DEPTH") != nullptr)
                std::fprintf(stderr, "strata serve: spec depth: accepted/tested %lld/%lld %lld/%lld %lld/%lld %lld/%lld %lld/%lld\n",
                             (long long) dp_ok[0], (long long) dp_tested[0], (long long) dp_ok[1], (long long) dp_tested[1],
                             (long long) dp_ok[2], (long long) dp_tested[2], (long long) dp_ok[3], (long long) dp_tested[3],
                             (long long) dp_ok[4], (long long) dp_tested[4]);
            if (strata::core::MtpDrafter::top2_env())
                std::fprintf(stderr, "strata serve: mtp top2: rejected/runner-up by depth %lld/%lld %lld/%lld %lld/%lld %lld/%lld "
                                     "of %lld windows\n", (long long) t2_rej[0], (long long) t2_hit[0], (long long) t2_rej[1],
                             (long long) t2_hit[1], (long long) t2_rej[2], (long long) t2_hit[2], (long long) t2_rej[3],
                             (long long) t2_hit[3], (long long) dec_windows);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                std::fprintf(stderr, "strata serve: CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it in this request\n", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                             remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);
            if (prefill_helper_src) {
                const uint64_t blobs = prefill_helper_src->helper_blobs(), bytes = prefill_helper_src->helper_bytes();
                std::fprintf(stderr, "strata serve: prefill helper source: %lld blobs / %.1f MiB taken from a helper's "
                                     "cache in this request\n",
                             (long long) (blobs - helper_blobs_before), (double) (bytes - helper_bytes_before) / 1048576.0);
                helper_blobs_before = blobs;
                helper_bytes_before = bytes;
            }
        }
        if (loop.is_failed()) return 1;
        save_profile("exit");   // #477: QUIT, or the server closed stdin
#if defined(STRATA_GPU_METAL)
        // the stdin reader (detached above) may still take its mutex when the server closes the pipe after QUIT; on
        // macOS locking that mutex once this scope has destroyed it throws (the engine ended with an abort): end here
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
#endif
        return 0;
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = strata::spec::kSpecNotStarted;   // plan v0.3 P6: where the speculative loop starts (0 is a real position: a 1-token prompt on a native pack)
    Conversation conv;   // the prompt path and, with --spec, the verify windows
    strata::prefill::Prefill& prefill = conv.sp;
    model.arm_prompt_cpu_share();   // before the chunk below sizes the loan
    bool kvg_started = false;   // the elastic K/V took this run's cells
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    PfPart loan{&xcache, &ss};   // CUDA0's slots lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!o.no_prefill_borrow && !host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            const int64_t chunk_in = chunk;
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            if (k > 0 && o.prefill_auto && n_batched > 1024) {   // #2824: priced against the host, for THIS prompt's length
                const int64_t vram_chunk = chunk;
                chunk = host_plan_chunk(
                    chunk, (uint64_t) std::max<int64_t>(0, (int64_t) xcache.bytes()), xcache.slots(),
                    [&](int64_t c) { return fits_one(cuda0_part, c, true); },
                    [&](int64_t c, uint64_t& bytes, int64_t& slots) {
                        slots = lend_slots(c);
                        bytes = part_bytes(cuda0_part, (int32_t) (xcache.slots() - slots));
                    },
                    [&](int64_t) { return strata::prefill::Prefill::ring_cap_for(4096); },
                    [&] { int64_t c2 = chunk_in; (void) plan_lend(c2); }, n_batched);
                if (chunk != vram_chunk) {
                    // the whole prompt fits today's chunk: one pass, keep it (the plan's ring state goes back with it)
                    if (request_chunk(n_batched, std::numeric_limits<int64_t>::max() / 2) <= vram_chunk) {
                        chunk = vram_chunk;
                        int64_t c2 = chunk_in;
                        (void) plan_lend(c2);
                    } else {
                        chunk = std::min(vram_chunk, planned_chunk(n_batched, vram_chunk, chunk));   // equal chunks, no tail
                    }
                    k = lend_slots(chunk);
                }
            }
            const int64_t request_sized = equal_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = lend_slots(chunk);
                if (k + g_lend_keep > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : request_chunk(n_batched, 1024);
                std::fprintf(stderr, "strata generate: prompt chunk auto: %lld tokens\n", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            if (k > 0 && nib::any_on()) {   // STRATA_EXPERT_NIBBLE: the lent slots would be refilled with GGUF blobs
                std::fprintf(stderr, "strata generate: STRATA_EXPERT_NIBBLE refuses the start: the prompt path would borrow %lld "
                                     "cache slots (--no-prefill-borrow keeps its own buffers)\n", (long long) k);
                return 1;
            }
#if defined(STRATA_GPU_METAL)
            k = 0;   // Metal: no loan of expert-cache slots (the prompt buffers are the Metal window's own); the refill after it copied into given-back allocations
#endif
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                // the elastic K/V first: its hot experts move into slots of the loan, which are refilled after it
                kvg_started = true;
                kvg_start(first);
                if (!kvg_ensure(n_prompt + o.max_new + 64, [] { cudaDeviceSynchronize(); })) return 1;
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        loan.lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                res_put(d_res);
                borrow = xcache.device_slot(first);
                borrow_bytes = part_bytes(cuda0_part, first);
                std::fprintf(stderr, "strata generate: prompt path borrows %lld cache slots (%.2f GiB)\n", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            std::fprintf(stderr, "strata generate: prompt path allocates its own buffers (no cache slots to borrow)\n");
        if (!conv.init_prompt(model, ss, borrow, borrow_bytes, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        uma_refuse_chunk_if_full(o.prefill_chunk, borrow != nullptr, xcache.gib());
        mem_mark("the prompt path's buffers");
        if (std::getenv("STRATA_TRACE") != nullptr) {   // what the buffers of this chunk take, and (Windows) the commit left
            std::fprintf(stderr, "strata trace: the prompt path's own buffers for a %lld-token chunk: %lld MiB (bytes_needed_owned)", (long long) o.prefill_chunk,
                         (long long) (strata::prefill::Prefill::bytes_needed_owned(g, ss, o.prefill_chunk, srcp != nullptr) >> 20));
#if defined(_WIN32)
            MEMORYSTATUSEX ms_t{};
            ms_t.dwLength = sizeof ms_t;
            if (GlobalMemoryStatusEx(&ms_t))
                std::fprintf(stderr, "; commit left %.1f GiB, RAM available %.1f GiB", (double) ms_t.ullAvailPageFile / 1073741824.0, (double) ms_t.ullAvailPhys / 1073741824.0);
#endif
            std::fprintf(stderr, "\n");
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                if (mtp.cell_floor_enabled() && prefill.kva_floor() > mtp.cell_floor()) mtp.set_cell_floor(prefill.kva_floor());   // KVA
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // E-9
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2) {   // the drafter's context: the last 2047 prompt rows' taps
            df2.reset();
            prefill.set_taps(df2.tap_layers(), std::max<int64_t>(0, n_prompt - 2047), 2048);
            prefill.on_taps = [&](const float* taps, int64_t rows, int64_t pos0, std::string& e) -> bool { df2_tap_dump(taps, rows, pos0, (int64_t) df2.tap_layers().size() * g.n_embd); return df2.context(taps, rows, pos0, e); };
        }
#endif
        if (!kvg_started) {   // the elastic K/V: every cell this run can reach (no loan)
            kvg_started = true;
            kvg_start(xcache.slots());
            if (!kvg_ensure(n_prompt + o.max_new + 64, [] { cudaDeviceSynchronize(); })) return 1;
        }
        const Clock::time_point tp0 = Clock::now();
        if (!conv.read_prompt(o.tokens.data(), n_batched, 0, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!loan.lent.empty()) {
            const Clock::time_point tr = Clock::now();
            const size_t n_lent = loan.lent.size();
            if (!refill_loan_issue(loan, err) || !refill_loan_wait(loan, err)) {
                std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                return 1;
            }
            res_put(d_res);
            std::fprintf(stderr, "strata generate: %zu lent slots refilled in %.1f ms\n", n_lent,
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
#if defined(STRATA_HAVE_DFLASH2)
        prefill.clear_taps();
#endif
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        mem_mark("the prompt was read");
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const strata::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld, on the CPU %lld (share %.2f); PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident,
                     (long long) ps.experts_cpu, ps.cpu_share, ps.ms_ple);
        if (ps.experts_direct > 0)
            std::fprintf(stderr, "strata generate: prompt MMQ read %lld of the %lld resident experts in their slots "
                                 "(STRATA_MMQ_DIRECT)\n", (long long) ps.experts_direct, (long long) ps.experts_resident);
    }

    if (!kvg_started) {   // the elastic K/V of a prompt that was not read in batches
        kvg_started = true;
        kvg_start(xcache.slots());
        if (!kvg_ensure(n_prompt + o.max_new + 64, [] { cudaDeviceSynchronize(); })) return 1;
    }
    for (int64_t pos = pos_start;; ++pos) {
        // plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
        if (native_pack) { spec_pos = pos; break; }
        if (pos >= o.max_context) {
            std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) pos);
            return 2;
        }
        // **THE TOKEN TIMER STARTS HERE, BEFORE ANY OF THE TOKEN'S WORK (A7).**  It used to start after
        // `put_input`/`embed_row`, which excluded the embedding and the PLE window advance from the reported
        // rate, and it stopped before the NaN scan, the logits dump and the sampler.  The published tok/s
        // figure is a WALL-CLOCK rate: everything one token costs, PLE advance to sampled id.  A rate that
        // excludes real per-token work is not a rate anyone can plan against.
        const Clock::time_point t0 = Clock::now();
        // **THE PLE'S TOKEN WINDOW ADVANCES HERE, ONCE PER TOKEN, AND `ple_stage_token` RUNS OUTSIDE THE
        // CAPTURE.**  Both are the driver's job: `ngram_rows` is a host hash over the last three tokens and the
        // table gather is a host read, so either one inside a captured graph would run once at capture time and
        // replay forever.  `ple_prev` is OLDEST FIRST and `-1` means "no predecessor", which `ngram_rows`
        // treats as the EOS cut - a sequence boundary.
        ss.ple_token = (int32_t) tok;
        Clock::time_point tp = Clock::now();
        // Plan v0.3 P2: the 16 SSD reads start here and complete while the embedding is staged; `ms_ple` is
        // the issue plus the time still spent WAITING afterwards, i.e. the part the embedding did not hide.
        if (ss.ple.ready() && !strata::core::ple_issue_token(ss.ple, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (!put_input(tok, pos)) return 1;
        {
            const Clock::time_point n = Clock::now();
            ms_embed += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        if (pos % 256 == 0 || pos + 1 >= n_prompt - 1)
            std::fprintf(stderr, "strata generate: position %lld, token %lld%s\n", (long long) pos, (long long) tok,
                         pos < n_prompt ? " (prompt)" : "");
        // **`d.layers` IS THE BLOB'S LAYER AXIS, NOT A COUNTER.**  The adapter uses it to index
        // `experts.bin` as `layer * n_expert + expert`, so it MUST restart at 0 for every token.  Leaving it
        // running across tokens asks for layer 48 of a 48-layer file on the second token - which
        // `FileExpertSource` REFUSES rather than wrapping into layer 0's experts, and that refusal is the only
        // reason this was a clean error instead of a silently wrong second token.
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        err.clear();
        if (o.no_capture) {
            if (!strata::core::session_token(wt, g, pos, /*pos_base=*/0, ss, d_parts, main_cs,
                                             o.sync_every_layer, err)) {
                std::fprintf(stderr, "strata generate: session_token: %s\n", err.c_str());
                return 1;
            }
        } else {
            strata::core::doorbell_reset(db);
            if (tgraph.captured) {
                if (!strata::core::session_run_token(g, pos, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                     loop_scratch.y_miss, main_cs, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
            } else if (!strata::core::session_loop(g, pos, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user, /*overlap=*/true, main_cs,
                                        err, layer_stage, &loop_scratch)) {
                std::fprintf(stderr, "strata generate: session_loop: %s\n", err.c_str());
                return 1;
            }
        }
        if (final_r != nullptr) {
            cudaMemcpy(final_r_host.data(), ss.R, final_r_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const int64_t posrec[2] = {pos, tok};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
        }
        if (drive.d.failed) {
            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                         drive.d.fail ? drive.d.fail : "(no message)");
            return 1;
        }
        {
            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**
            const Clock::time_point n = Clock::now();
            ms_layers += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // ---- the ladder for THIS position, in the order `session_loop` filled it: layer 0 first.
        if (layer_dump != nullptr) std::fwrite(layer_stage, sizeof(float), layer_floats, layer_dump);
        if (half_dump != nullptr) {
            std::fwrite(half_stage, sizeof(float), (size_t) g.n_layers * (size_t) half_stride, half_dump);
        }
        if (!run_head(token_stream)) {
            std::fprintf(stderr, "strata generate: lm_head: %s\n", err.c_str());
            return 1;
        }
        // **A CHECKPOINT AFTER THE HEAD, BECAUSE AN ASYNC FAULT IS STICKY AND LIES ABOUT WHERE IT HAPPENED.**
        // Measured, and it cost an hour: without this, `embed_row`'s D2H on the NEXT token reported "an illegal
        // memory access" at a plane offset that has nothing to do with the fault, and the layer that actually
        // faulted had completed its own error checks successfully - because its kernels had not run yet.  A
        // sticky error surfaces at the next SYNCHRONISING call, which is whatever happens to come next.
        if (!o.stream_token && cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device faulted in lm_head at position %lld: %s\n",
                         (long long) pos, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_head += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        const bool emit_logits = dump != nullptr &&
            strata::qwen38::logits_selection::selected(pos, dump_positions, o.logits_stride);
        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;
        if (read_logits && (cudaMemcpyAsync(logits.data(), d_logits, (size_t) n_vocab * 4,
                                           cudaMemcpyDeviceToHost, (cudaStream_t) token_stream) != cudaSuccess ||
                            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess)) {
            std::fprintf(stderr, "strata generate: reading the logits back failed\n");
            return 1;
        }
        int bad = 0;
        if (read_logits) for (float v : logits) if (!std::isfinite(v)) ++bad;
        if (bad != 0) {
            std::fprintf(stderr, "strata generate: %d of %lld logits are not finite at position %lld\n", bad,
                         (long long) n_vocab, (long long) pos);
            return 1;
        }
        // the header with the first row (see `hdr`): a run that dumps nothing leaves an empty file
        if (emit_logits && !hdr_written && std::fwrite(hdr, sizeof hdr, 1, dump) != 1) {
            std::fprintf(stderr, "strata generate: cannot write logits header\n");
            std::fclose(dump);
            return 1;
        }
        if (emit_logits) hdr_written = true;
        if (emit_logits && std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            std::fclose(dump);
            return 1;
        }
        {
            // **993 KB OF SYNCHRONOUS D2H AND A 248,320-FLOAT HOST SCAN, EVERY TOKEN.**  (The review's notes
            // say 151,936 floats; the artifact's `output.weight` is 248,320 rows, so the real figure is 1.6x
            // that - a number nobody had checked because nothing measured this term.)  R2.6 asks for the dump
            // and the scan to be behind flags; round 36 did exactly that and measured it SLOWER, because on
            // this driver a large blocking readback is also what flushes the pipeline for the sampler that
            // follows.  Timed so the claim can be re-checked rather than remembered.
            const Clock::time_point n = Clock::now();
            ms_readback += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        int next = 0;
        // The draw is Philox(seed, position), as in a verify window (row t at pos0 draws pos0 + t): a seed gives
        // the same text whether a token comes from this path or from the speculative loop below.
        sp.counter = (uint64_t) pos;
        strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, sp, d_next, token_stream);
        if (cudaMemcpyAsync(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost,
                            (cudaStream_t) token_stream) != cudaSuccess ||
            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading the sampled token back failed: %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        // The sampled-token synchronization also completes every captured QSA
        // status readback. Retain one status per layer so a later layer cannot
        // hide an earlier failure; no extra synchronization or token allocation.
        if (o.native_flash_attn_short) for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            const int64_t i = ss.qsa_ord0 + j;   // the global ordinal, as the session's carve names it
            const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
            if (status != 0) {
                std::fprintf(stderr, "strata generate: native attention status %d at QSA layer %lld, position %lld\n",
                             status, (long long) i, (long long) pos);
                return 1;
            }
        }
        if (next < 0 || next >= n_vocab) {
            std::fprintf(stderr, "strata generate: the sampler returned %d, outside 0..%lld\n", next,
                         (long long) (n_vocab - 1));
            return 1;
        }
        {
            // **TWO DEVICE-WIDE SYNCS FOR FOUR BYTES.**  `sample_tokens(nullptr)` ends in
            // `cudaDeviceSynchronize()` (`sampler.cu:245`) and the blocking 4-byte read below is the second.
            const Clock::time_point n = Clock::now();
            ms_sample += std::chrono::duration<double, std::milli>(n - tp).count();
            ++phase_tokens;
        }
        // CHARGED HERE, AFTER THE SAMPLER, so the wall-clock rate covers the whole token including the embedding,
        // the NaN scan, the logits readback and the sample (A7).  Only DECODE positions count; prefill is
        // measured separately.
        if (pos >= n_prompt - 1) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        else prefill_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (pos == n_prompt - 1) ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        // **THE PREDICTION AT THE LAST PROMPT POSITION *IS* THE FIRST GENERATED TOKEN.**  Sampling on every
        // position and recording only from `n_prompt - 1` onward is what keeps the two cases from needing
        // separate handling - and the version that "obviously" only samples after the prompt loses exactly one
        // token's worth of conditioning.
        if (pos >= n_prompt - 1) produced.push_back(next);
        if ((int64_t) produced.size() >= o.max_new) break;
        if (o.stop_eos && pos >= n_prompt - 1 &&
            std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) break;
        // TEACHER FORCING while the prompt lasts: the next input is the prompt's own next token, not the
        // model's guess.  Feeding the guess would make the run depend on the model's own errors from position
        // 1, which is a different (and worse) measurement of the same prompt.
        // and the window advances: the token just decoded becomes the newest predecessor.
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tok;
        tok = (pos + 1 < n_prompt) ? o.tokens[(size_t) (pos + 1)] : next;
        // Plan v0.3 P6: from the first generated token on, the speculative loop below takes over.
        if (o.spec > 0 && pos >= n_prompt - 1) { spec_pos = pos + 1; break; }
    }

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (strata::spec::spec_loop_runs(spec_pos, (int64_t) produced.size(), o.max_new, ended)) {
        auto read_ids = [](const std::string& path, std::vector<int64_t>& ids) {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            return (bool) in && parse_i64_list(text.c_str(), ids, e);
        };
        std::vector<int64_t> oracle, follow;
        if (!o.spec_oracle.empty() && !read_ids(o.spec_oracle, oracle)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-oracle %s\n", o.spec_oracle.c_str());
            return 2;
        }
        if (!o.spec_follow.empty() && !read_ids(o.spec_follow, follow)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-follow %s\n", o.spec_follow.c_str());
            return 2;
        }
        // --spec-follow: the run ends with the continuation
        const int64_t max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new, (int64_t) follow.size());
        int64_t follow_differ = 0, follow_emitted = 0;
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "strata generate: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        strata::core::Verifier& ver = conv.ver;
        if (pcie_balance()) ver.set_pcie_balance(&drive.d.pcie_model);
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2) ver.set_taps(df2.tap_layers());
#endif
        if (!conv.init_decode(model, ss, o.spec, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2 && !ver.tapping()) {
            std::fprintf(stderr, "strata generate: --draft dflash2: the verifier keeps no taps here; the MTP drafts instead\n");
            use_df2 = false;
        }
        if (use_df2) df2_offer_head();
        if (use_df2 && !df2.bind(wt, &native_head, df2_vocab_path(), err)) {
            std::fprintf(stderr, "strata generate: --draft dflash2: %s. It needs about 1.2 GiB of VRAM in all (the pack 0.76 GiB, buffers and K/V ring "
                                 "0.25 GiB, a draft head 0.2 GiB); --vram-reserve-mib 1900 keeps it (the expert cache gives up about 2.5%% of its experts "
                                 "to the CPU). The MTP drafts instead.\n", err.c_str());
            use_df2 = false;
        }
        if (use_df2) df2.set_max_drafts((o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec) - 1);   // the widest window - 1: the head and the walk run for that many rows
        if (use_df2 && !df2.prepare(ver.tap_dev(), err)) {
            std::fprintf(stderr, "strata generate: --draft dflash2: %s. It needs about 1.2 GiB of VRAM in all (the pack 0.76 GiB, buffers and K/V ring "
                                 "0.25 GiB, a draft head 0.2 GiB); --vram-reserve-mib 1900 keeps it (the expert cache gives up about 2.5%% of its experts "
                                 "to the CPU). The MTP drafts instead.\n", err.c_str());
            use_df2 = false;
        }
        if (use_df2) std::fprintf(stderr, "strata generate: DFlash2 drafter on (%s, graphs captured in %.0f ms), %s\n", o.draft_model.c_str(), df2.ms_capture, df2_solo ? "no MTP layer is loaded (--draft-fallback none)" : "the MTP layer stays as the fallback");
        else if (o.draft == "dflash2" && o.draft_mtp_spec_set && o.spec > o.draft_mtp_spec) o.spec = o.draft_mtp_spec;   // fell back: the MTP's own window
#endif
        if (df2_solo && !use_df2) {   // --draft-fallback none: the reason is printed above; there is no MTP to draft instead
            std::fprintf(stderr, "strata generate: --draft-fallback none: DFlash2 did not start and no MTP is loaded, so the run ends here\n");
            return 1;
        }
        const bool draft_on = use_mtp || use_df2;   // some drafter proposes (--draft-fallback none: DFlash2 alone, use_mtp false)
        mem_mark("the verifier and the drafter's binding");
        ver.set_sampling(sp);   // the CLI's own sampling (until 0.1.19 this loop was always greedy); no penalties here
        if (use_mtp) mtp.set_draft_sampling(sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
        const int64_t pcie0 = drive.d.pcie_experts;
        // CS-T: the file tier since the decode began (the prompt path's copies are before this)
        const int64_t files0 = src.file_reads(), ram0 = src.ram_reads();
        const uint64_t fbytes0 = src.file_blob_bytes(), fall0 = src.file_read_bytes();
        const double fms0 = src.file_ms();
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        const char* fetch_off = d_res == nullptr ? "no residency table" : o.spec_split ? "--spec-split"
                              : multi_gpu ? "a layer split" : drive.d.peer != nullptr ? "a peer tier"
                              : drive.d.remote_count > 0 ? "helper GPUs" : nullptr;
        if (adapt_fetch() > 0) {   // STRATA_ADAPT_FETCH (opt-in)
            const char* off = drive.d.usage.empty() ? "the adaptive tier is off" : fetch_off != nullptr ? fetch_off
                            : o.pcie_mode == "dma" || o.pcie_mode == "direct" ? "--pcie-mode dma / direct"
                            : drive.d.plan == nullptr ? "no GPU plan"
                            : drive.d.pcie_num <= 0 ? "no PCIe share"
                            : src.complement_ready() ? "the resident RAM mode" : nullptr;
            if (off != nullptr) std::fprintf(stderr, "strata generate: STRATA_ADAPT_FETCH is off (%s)\n", off);
            else {
                drive.d.fetch_admit = adapt_fetch();
                drive.d.res_mut = host_res.data();
                drive.d.in_flight.assign(host_res.size(), 0);
                drive.d.fetch_avail.assign((size_t) g.n_layers, 0);
                if (const char* v = std::getenv("STRATA_ADAPT_FETCH_MIN")) drive.d.fetch_min = (float) std::atof(v);
                if (const char* v = std::getenv("STRATA_ADAPT_FETCH_MARGIN")) drive.d.fetch_margin = (float) std::atof(v);
                drive.d.fetch_log_on = std::getenv("STRATA_ADAPT_FETCH_VERIFY") != nullptr;
                std::fprintf(stderr, "strata generate: STRATA_ADAPT_FETCH=%d: the PCIe share takes each layer's most-routed "
                                     "misses and admits those routed >= %.2f and %.2f more than a resident one%s\n",
                             drive.d.fetch_admit, drive.d.fetch_min, drive.d.fetch_margin,
                             drive.d.fetch_admit == 2 ? "; the tier's own copies are off" : "");
            }
        }
        // before every window: the device's table follows the host's (the tier's evictions while its copies are in
        // flight, STRATA_ADAPT_FETCH's admissions - see the serve path); the hook's captures live in this block, so it
        // is taken off the verifier when the block ends
        const bool res_checking = std::getenv("STRATA_ADAPT_CHECKRES") != nullptr && d_res != nullptr;
        std::vector<int32_t> res_back;
        int64_t res_windows = 0, res_bad = 0, res_bad_windows = 0;
        if (res_checking) {
            res_back.assign(host_res.size(), 0);
            drive.d.dres_check = res_back.data();
        }
        drive.d.evict_sync = d_res != nullptr && !drive.d.usage.empty() && (adapt_nowait() || adapt_lag() > 1) &&
                             (adapt_evict_sync() || drive.d.fetch_admit > 0);
        // the changed entries by a kernel on the verifier's stream (see the serve path)
        constexpr size_t kResPatchCap = 16384;
        int32_t* res_pairs = (drive.d.fetch_admit > 0 || drive.d.evict_sync) ? res_pairs_alloc(kResPatchCap, {-1}) : nullptr;
        struct PreLaunchOff {
            strata::core::Verifier& v;
            strata::core::ExpertDispatch& d;
            int32_t* pairs;
            ~PreLaunchOff() {
                v.set_pre_launch(nullptr);
                d.dres_check = nullptr;
                if (pairs != nullptr) cudaFreeHost(pairs);
            }
        } pre_launch_off{ver, drive.d, res_pairs};
        if (drive.d.fetch_admit > 0 || drive.d.evict_sync || res_checking)
            ver.set_pre_launch([&](cudaStream_t cs, std::string& e) -> bool {
                if (drive.d.res_dirty) {
                    const int64_t n = res_patch_pairs(drive.d.res_dirty_idx, host_res, res_pairs, kResPatchCap);
                    cudaError_t r;
                    if (n >= 0) {
                        strata::kernels::res_patch(d_res, res_pairs, (int) n, cs);
                        r = cudaGetLastError();
                    } else {
                        r = res_put(d_res);
                    }
                    if (r != cudaSuccess) {
                        e = std::string("the residency table update failed: ") + cudaGetErrorString(r);
                        return false;
                    }
                    drive.d.res_dirty = false;
                    drive.d.res_dirty_idx.clear();
                }
                if (!res_checking) return true;
                cudaStreamSynchronize(cs);   // the patch landed
                const int64_t bad = res_check(d_res, host_res, res_back, g.n_expert, res_windows);
                ++res_windows;
                res_bad += bad;
                res_bad_windows += bad > 0;
                return true;
            });
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        cudaStream_t adapt_stream = nullptr;
        if (!drive.d.usage.empty() && cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        int pending_age = 0;   // the windows the pending swaps have waited (adapt_lag)
        std::vector<void*> pin_live;   // the swaps' locked arena pages (pin_blob), unlocked once they have landed
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        int64_t adapt_rounds = 0;   // counted here: `rounds` is declared below the adapt lambda
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            // STRATA_TRACE_ADAPT=1: whether a round's copies had landed when the next window read the table
            static const bool trace_pending = std::getenv("STRATA_TRACE_ADAPT") != nullptr;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) {
                if (trace_pending)
                    std::fprintf(stderr, "strata: PENDING not landed, %zu stay non-resident this window\n",
                                 pending.size());
                return;
            }
            if (trace_pending)
                std::fprintf(stderr, "strata: PENDING landed, %zu experts become resident\n", pending.size());
            unpin_blobs(pin_live);
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
                if (!drive.d.in_flight.empty()) drive.d.in_flight[(size_t) i] = 0;
            }
            if (drive.d.evict_sync || drive.d.fetch_admit > 0) {   // a failed upload leaves them to the pre-launch hook
                for (const auto& pr : pending) drive.d.res_dirty_idx.push_back(pr.first);
                drive.d.res_dirty = true;
            }
            pending.clear();
            pending_age = 0;
            if (d_res != nullptr && res_put(d_res) == cudaSuccess) {   // the whole table: the hook has nothing left
                drive.d.res_dirty = false;
                drive.d.res_dirty_idx.clear();
            }
        };
        // Plan v0.3 P6: the VRAM tier follows the conversation.  Candidates are missing experts routed at least
        // twice (decayed); each is paired with its layer's least-routed resident expert and swapped when it was
        // routed clearly more often.  Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            ++adapt_rounds;
            // STRATA_TRACE_ADAPT: why an adapt round did or did not swap.  Default off, one getenv, and it
            // reports the only thing that can make an adapt round a coin flip: whether the PREVIOUS round's
            // asynchronous expert copies had landed by the time this round started.
            static const bool trace_adapt = std::getenv("STRATA_TRACE_ADAPT") != nullptr;
            if (!pending.empty()) {
                if (trace_adapt)
                    std::fprintf(stderr, "strata: ADAPT round=%lld SKIPPED, %zu swaps still in flight\n",
                                 (long long) adapt_rounds, pending.size());
                return true;   // the previous swaps are still in flight
            }
            if (trace_adapt) std::fprintf(stderr, "strata: ADAPT round=%lld considering\n", (long long) adapt_rounds);
            std::vector<Swap> swaps = choose_swaps(drive.d.usage, host_res, 1.5f, true, helper_holds);
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            // this (non-serve) path refills CUDA0's cache only, so its copies back read CUDA0's cache too
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps,
                                      [&](int64_t) -> SwapHome { return {&xcache, adapt_stream, -1}; })) {
                std::fprintf(stderr, "strata generate: an adaptive refill failed (copying evicted experts back)\n");
                return false;
            }
            if (pin_blobs_on()) {   // lock the batch's source pages (a file-backed arena on AMD; see pin_blobs)
                std::vector<std::pair<uintptr_t, uintptr_t>> spans;
                for (const Swap& s : swaps)
                    if (const uint8_t* b = srcp->blob_stable(s.layer, s.in))
                        spans.emplace_back((uintptr_t) b,
                                           (uintptr_t) b + strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                pin_blobs(std::move(spans), pin_live);
            }
            // STRATA_DMA_BATCH (F12): the round's copies as one cudaMemcpyBatchAsync; mode 0 is the loop of one copy each
            const int dma_mode = strata::core::dma_batch_mode();
            std::vector<void*> cp_dst;
            std::vector<const void*> cp_src;
            std::vector<size_t> cp_bytes;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob_stable(s.layer, s.in);
                nib::guard_raw(s.layer, "the adaptive refill's copy in");
                const size_t blob_n = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                if (dma_mode != 0) {
                    cp_dst.push_back(xcache.device_slot(slot));
                    cp_src.push_back(b);
                    cp_bytes.push_back(blob_n);
                } else if (adapt_copy_h2d(xcache.device_slot(slot), b, blob_n, adapt_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
                if (!drive.d.in_flight.empty()) drive.d.in_flight[in] = 1;   // STRATA_ADAPT_FETCH leaves it alone
            }
            if (drive.d.evict_sync && !swaps.empty()) {   // the evicted ones are misses now (see the pre-launch hook)
                for (const Swap& s : swaps) drive.d.res_dirty_idx.push_back((int32_t) (s.layer * g.n_expert + s.out));
                drive.d.res_dirty = true;
            }
            if (!cp_dst.empty() && strata::core::copy_blobs(cp_dst.data(), cp_src.data(), cp_bytes.data(), cp_dst.size(),
                                                            adapt_stream, dma_mode) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                return false;
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            if (trace_adapt)
                std::fprintf(stderr, "strata: ADAPT round=%lld swapped %zu of %d slots, usage decayed\n",
                             (long long) adapt_rounds, swaps.size(), o.adapt_swaps);
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // STRATA_SPEC_T_CYCLE=1,2,3,4,5,6,7,8 (bench only, off by default): the window size cycles through the list, so one run
        // measures the cost of each verify width (with --spec-oracle drafts every row is accepted); the first round of a
        // width (its graph capture) is not counted
        std::vector<int> t_cycle;
        if (const char* tc = std::getenv("STRATA_SPEC_T_CYCLE"); tc != nullptr && tc[0] != '\0') {
            for (const char* q = tc; *q != '\0';) {
                const int v = std::atoi(q);
                if (v >= 1 && v <= o.spec) t_cycle.push_back(v);
                while (*q != '\0' && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        // STRATA_SPEC_TRACE=<file> (bench only, off by default): one line per round, "T a sfx ms p1 p2 ..." (window rows, drafts accepted,
        // lookup window, round ms, the MTP draft probabilities), for offline window-length policy studies
        std::FILE* spec_trace = nullptr;
        if (const char* tf = std::getenv("STRATA_SPEC_TRACE"); tf != nullptr && tf[0] != '\0') spec_trace = std::fopen(tf, "w");
        std::vector<double> t_cycle_ms((size_t) o.spec + 1, 0.0);
        std::vector<int64_t> t_cycle_n((size_t) o.spec + 1, 0), t_cycle_seen((size_t) o.spec + 1, 0);
        // plan v0.3 P6: with a native pack the first window is the last prompt token alone (it produces the first
        // generated token and the MTP's first cell); otherwise the token loop already did that.
        bool first_window = native_pack;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
#if defined(STRATA_HAVE_DFLASH2)
        const bool spec_uses_df2 = use_df2;
#else
        const bool spec_uses_df2 = false;   // no DFlash2 drafter in this build (Metal)
#endif
        strata::spec::WindowPlanner wplan(o.spec, S_mtp, o.suffix_draft, o.lookup_chain, o.lookup_chain_min,
                                          o.max_context);
        const bool gated_proposal = use_mtp && mtp.prob() && strata::core::spec_gate_pick() &&
            gated_proposal_usable(o.pipeline_windows < 2 && o.batch == 0 && !multi_gpu && drive.d.remote_count == 0,
                                  o.lookup_chain, o.adapt_swaps, o.pcie_frac != 0.0 || o.pcie_frac_auto,
                                  wplan.cost_aware || spec_uses_df2 || !t_cycle.empty() || o.spec_corrupt > 0, o.spec, o.spec_min_p);
        if (gated_proposal && !follow.empty())
            std::fprintf(stderr, "strata generate: --spec-follow replaces outputs before acceptance (teacher-forced diagnostic, not sampling-law evidence)\n");
        strata::core::MtpDrafter::DraftResult generation;
        strata::core::gated::Packet proposal;
        {   // Non-native prefill already emitted x; draft_first(cell=p-1) samples at p.
            const GatedDraftLimit draft_limit(mtp, gated_proposal, S_mtp, p, o.max_context);
            if (use_mtp && !first_window &&
                !mtp.draft_first(o.spec, ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p,
                                 nullptr, gated_proposal ? &generation : nullptr)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
        }
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<uint8_t> fdiff((size_t) o.spec, 0);
        std::FILE* hashes = o.window_hashes.empty() ? nullptr : std::fopen(o.window_hashes.c_str(), "w");
        if (!o.window_hashes.empty() && hashes == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.window_hashes.c_str());
            return 1;
        }
        std::vector<float> hash_rows;
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        wplan.source = wplan.policy;   // this generation's: committed outcomes only (observe_committed)
        if (const char* tab = std::getenv("STRATA_SPEC_COST_TABLE");
            wplan.cost_aware && tab != nullptr && tab[0] != '\0') {
            // "T1,T2,..;T1,T2,..;T1,..": the row costs of the 3 context buckets (< 6K, < 24K, longer); one list seeds the first
            double ms[strata::spec::DraftPolicy::kMaxT] = {};
            int n = 0, bucket = 0;
            for (const char* q = tab; *q != '\0' && bucket < strata::spec::DraftPolicy::kCtxBuckets;) {
                if (n < strata::spec::DraftPolicy::kMaxT) ms[n++] = std::atof(q);
                while (*q != '\0' && *q != ',' && *q != ';') ++q;
                const bool end_list = *q == ';' || *q == '\0';
                if (*q != '\0') ++q;
                if (end_list) {
                    wplan.policy.seed_costs(ms, n, 6.0, bucket++);
                    n = 0;
                }
            }
        }
        if (wplan.cost_aware && use_mtp)
            mtp.set_chain_rule([&](int idx, float pj) { return wplan.keep_draft(idx, pj); });
        wplan.policy.set_sizes(lookup_size_mask(o));
        wplan.start(o.tokens);
        for (int64_t t : produced) wplan.append((int32_t) t);
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        // STRATA_SPEC_WATCH=<seconds> (diagnostics, off by default): the serve watchdog for this loop.  A run whose verify
        // windows and prompt chunks stop beating for that long prints the stage it is in and the stall report, then ends.
        if (const char* sw = std::getenv("STRATA_SPEC_WATCH"); sw != nullptr && std::atoi(sw) > 0) {
            const int limit = std::atoi(sw);
            strata::core::progress().busy.store(true);
            std::thread([limit] {
                strata::core::Progress& pr = strata::core::progress();
                uint64_t last = pr.beats.load(), ticks_at = pr.ticks.load();
                auto since = std::chrono::steady_clock::now();
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    const auto now = std::chrono::steady_clock::now();
                    const uint64_t b = pr.beats.load();
                    if (b != last) { last = b; ticks_at = pr.ticks.load(); since = now; continue; }
                    if (now - since < std::chrono::seconds(limit)) continue;
                    std::fprintf(stderr, "strata generate: no progress for %d s (%s)\n", limit, stage_text().c_str());
                    stall_report(stderr, pr.ticks.load() - ticks_at);
                    strata::core::release_gpu_waits(stderr);
                    std::fflush(stderr);
                    std::abort();
                }
            }).detach();
        }
        // DFlash2 with a long suffix match: the draft of the next round is deferred (spec/window_rule.hpp)
#if defined(STRATA_GPU_METAL)
        const int df2_skip_min = strata::spec::df2_lookup_skip_min(8);
#else
        const int df2_skip_min = strata::spec::df2_lookup_skip_min(0);
#endif
        bool df2_deferred = false;
        int dfr_n = 0;
        int64_t dfr_pos = 0;
        int32_t dfr_anchor = 0;
        // the next round's drafts after a commit: the MTP chain, or the DFlash2 block from the committed rows' taps
        auto draft_next = [&](int T_, int a_) -> bool {
#if defined(STRATA_HAVE_DFLASH2)
            if (use_df2) {
                if (o.suffix_draft > 0 && !first_window) {   // a long suffix match: no draft now (spec/window_rule.hpp skip_draft_for_lookup); the next round decides
                    const int ks = wplan.propose_lookup(o.spec);
                    if (strata::spec::skip_draft_for_lookup(true, df2_skip_min, ks, wplan.sfx.last_match())) {
                        df2_deferred = true; dfr_n = a_ + 1; dfr_pos = p; dfr_anchor = outv[(size_t) a_];
                        return true;
                    }
                }
                static const bool wait_commit_first = std::getenv("STRATA_DF2_WAITCOMMIT") != nullptr;   // measurement: no overlap with the async commit
                if (wait_commit_first && !ver.wait_commit(err)) return false;
                int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                float pp[strata::core::Dflash2Drafter::kDrafts];
                if (!df2.draft(ver.tap_dev(), a_ + 1, p, outv[(size_t) a_], dd, pp, err)) return false;
                for (size_t i = 0; i < drafts.size() && i < (size_t) strata::core::Dflash2Drafter::kDrafts; ++i) { drafts[i] = dd[i]; dprob[i] = pp[i]; }
                // what the MTP does while it drafts: the PLE rows of the tokens the next window will hold are read ahead
                // (the window's gather would otherwise wait for them: 1.8 ms a round on strix)
                if (ss.ple.ready() && ss.ple.table != nullptr) {
                    int n_pf = 0;
                    const int cap = std::min<int>(S_mtp, (int) drafts.size() + 1);
                    double chain = 1.0;
                    for (int t = 1; t < cap; ++t) {
                        if (o.draft_conf > 0.0) { if ((chain *= (double) dprob[(size_t) t - 1]) < o.draft_conf) break; }
                        else if (o.spec_min_p > 0.0 && dprob[(size_t) t - 1] < (float) o.spec_min_p) break;
                        ++n_pf;
                    }
                    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
                    auto pf = [&](int32_t tok) {
                        uint32_t rows16[strata::kernels::PLE_N_HEADS];
                        strata::kernels::ngram_rows(&tok, prev, 1, ss.ple.consts, rows16);
                        prev[0] = prev[1];
                        prev[1] = tok;
                        ss.ple.table->prefetch_rows(rows16);
                    };
                    pf(outv[(size_t) a_]);
                    for (int i = 0; i < n_pf; ++i) pf(drafts[(size_t) i]);
                }
                return true;
            }
#endif
            return mtp.draft_long(T_, outv.data(), p, a_, drafts.data(), err, dprob.data(), (float) o.spec_min_p,
                                  nullptr, gated_proposal ? &generation : nullptr);
        };
        int win_cap = o.spec;   // the widest window the VRAM allows (lowered on an out-of-memory capture, see below)
        while ((int64_t) produced.size() < max_new) {
            const Clock::time_point t0 = Clock::now();
            bool dfr_lookup = false;
            int dfr_T = 0, dfr_match = 0;
#if defined(STRATA_HAVE_DFLASH2)
            if (df2_deferred) {   // the draft this round skipped for a long lookup match: the lookup if the policy takes it, else the draft now (same taps, same rows)
                df2_deferred = false;
                const int kd = wplan.propose_lookup(o.spec);
                const int md = wplan.sfx.last_match();
                const strata::spec::DraftPolicy::Pick pk = kd > 0 ? wplan.policy.choose(S_mtp, kd, md) : strata::spec::DraftPolicy::Pick{};
                if (kd > 0 && pk.lookup) {
                    dfr_lookup = true; dfr_T = pk.t; dfr_match = md;
                    if (!df2.context(ver.tap_dev(), dfr_n, dfr_pos, err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
                } else {
                    int32_t dd[strata::core::Dflash2Drafter::kDrafts];
                    float pp[strata::core::Dflash2Drafter::kDrafts];
                    if (!df2.draft(ver.tap_dev(), dfr_n, dfr_pos, dfr_anchor, dd, pp, err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
                    for (size_t i = 0; i < drafts.size() && i < (size_t) strata::core::Dflash2Drafter::kDrafts; ++i) { drafts[i] = dd[i]; dprob[i] = pp[i]; }
                }
            }
#endif
        retry_round:
            strata::core::progress_at("spec loop: choosing the window", (int64_t) produced.size());
            // the window rule of the round's ACTIVE source (window_rule.hpp): the MTP chain's cost rule and probabilities
            // do not govern DFlash2's candidates, though the MTP may stay loaded beside it as the fallback (use_mtp)
            strata::spec::Window w = wplan.begin(p, {.use_mtp = use_mtp, .draft_on = draft_on, .df2_active = spec_uses_df2,
                                                     .first_window = first_window, .cap = S_mtp, .start = S_mtp,
                                                     .spec_min_p = o.spec_min_p, .draft_conf = o.draft_conf,
                                                     .dprob = dprob.data()});
            const int room = (int) std::min<int64_t>(win_cap, o.max_context - p);
            if (gated_proposal) {
                wplan.source.apply_costs(p);   // held round times whose draws are all committed now
                if (!prepare_gated_window(proposal, w, generation, wplan, room, (int) ver.vocab(), (float) o.spec_min_p,
                                          mtp.spec_q(), drafts.data(), dprob.data(), err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1;
                }
            } else {
                if (first_window) w.T = 1;
                if (!t_cycle.empty() && !first_window) w.T = t_cycle[(size_t) (rounds % (int64_t) t_cycle.size())];
                // DFlash2's candidates are not draws from the MTP's q: only the MTP's are probabilistic
                wplan.pick_lookup(w, drafts.data(), use_mtp && !spec_uses_df2 && mtp.prob(), o.spec);
            }
            if (dfr_lookup)   // the lookup window chosen before the draft (the draft was skipped)
                wplan.take_lookup(w, dfr_T, dfr_match, o.spec);
            // The window's rows sit at p .. p + T - 1, and past the context there is no cell to verify them in: near
            // its end the window takes only the rows that fit (the admission checked the prompt and max_new, which a
            // full window can overrun)
            w.T = std::min(w.T, room);
            if (draft_on) wplan.pick_chain(w, drafts.data(), room);
            const int T_mtp = w.T, T = w.rows();
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = w.lookup ? wplan.lookup_drafts()[i - 1] : draft_on ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (i >= T_mtp) d = wplan.chain_drafts()[i - T_mtp];
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
            // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
            // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
            if (adapt_nowait()) apply_pending(false);
                else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
            if (!drive.d.fetch_log.empty()) {   // STRATA_ADAPT_FETCH_VERIFY: each admitted slot read back
                static int64_t checked = 0, bad = 0;
                cudaDeviceSynchronize();
                for (const auto& [i, slot] : drive.d.fetch_log) {
                    std::string verr;
                    const int64_t l = i / g.n_expert;
                    const bool same = host_res[(size_t) i] != slot ||   // evicted again since: nothing to check
                        xcache.verify_slot(slot, srcp->blob_stable(l, i % g.n_expert), verr,
                                           (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
                    ++checked;
                    if (!same) {
                        ++bad;
                        std::fprintf(stderr, "strata: FETCH VERIFY layer %lld expert %lld slot %d: %s\n", (long long) l,
                                     (long long) (i % g.n_expert), (int) slot, verr.c_str());
                    }
                }
                drive.d.fetch_log.clear();
                if ((checked & 1023) < 64)
                    std::fprintf(stderr, "strata: fetch verify: %lld admitted slots read back, %lld differ\n",
                                 (long long) checked, (long long) bad);
            }
            if (gated_proposal) ver.set_spec_proposal(proposal.n_q ? proposal.mu.data() : nullptr, proposal.n_q);
            // the MTP's q lists judge only candidates the MTP drew: DFlash2's are a deterministic pick (exact-match path)
            else if (strata::spec::mtp_q_applies(use_mtp, spec_uses_df2, use_mtp && mtp.prob(), w.lookup, first_window, T_mtp) &&
                     o.spec_corrupt <= 0)
                ver.set_spec_q(mtp.spec_q(), T_mtp - 1);
            if (!conv.run_window(T, window.data(), p, outv.data(), err, use_mtp && mtp.prob())) {
                // The graph of a window size is captured the first time it is used, and the draft source took VRAM: when the capture runs
                // out of memory the window gets one row narrower (the widths below it are already captured) instead of the run failing.
                if (T > 2 && !first_window && !gated_proposal && err.find("out of memory") != std::string::npos) {
                    win_cap = T - 1;
                    --window_hist[(size_t) T];
                    std::fprintf(stderr, "strata generate: not enough VRAM left for a %d-token verify window: windows are %d tokens at most from here\n", T, win_cap);
                    err.clear();
                    goto retry_round;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (hashes != nullptr) {   // FNV-1a over the window's final residual rows (a test path: a plain copy)
                const size_t n_floats = (size_t) T * (size_t) (g.hc * g.n_embd);
                hash_rows.resize(n_floats);
                cudaMemcpy(hash_rows.data(), ver.final_R_all(), n_floats * sizeof(float), cudaMemcpyDeviceToHost);
                uint64_t h = 1469598103934665603ull;
                const auto* bytes = (const uint8_t*) hash_rows.data();
                for (size_t i = 0; i < n_floats * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ull;
                std::fprintf(hashes, "%lld %lld %d %016llx", (long long) rounds, (long long) p, T, (unsigned long long) h);
                for (int i = 0; i < T; ++i) std::fprintf(hashes, " %d", (int) outv[(size_t) i]);
                std::fprintf(hashes, "\n");
            }
            if (!follow.empty()) {   // the continuation stands in for the argmax
                const size_t k0 = produced.size();
                for (int i = 0; i < T && k0 + (size_t) i < follow.size(); ++i) {
                    fdiff[(size_t) i] = outv[(size_t) i] != (int32_t) follow[k0 + (size_t) i];
                    outv[(size_t) i] = (int32_t) follow[k0 + (size_t) i];
                }
            }
            const int a = wplan.accept(w, window.data(), outv.data());
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                // Diagnostics: the first window runs the prompt's last token over the state the prompt path left
                // (keys, values, recurrent state), so its logits carry whatever that path did. A native pack never
                // runs the per-token loop --dump-logits reads; this is where prompt paths can be compared by output.
                if (const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS")) {
                    std::vector<float> row((size_t) ver.vocab());
                    std::FILE* f = ver.copy_logits(0, row.data()) ? std::fopen(fl, "wb") : nullptr;
                    if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                        std::fprintf(stderr, "strata generate: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", fl);
                    if (f) std::fclose(f);
                }
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_iv(rounds)) == 0)
                adapt_thr = std::thread([&] { strata::platform::name_this_thread("strata-adapt"); strata::aux_cpus::pin_current_thread(); adapt_ok = adapt(); });
            strata::core::progress_at("spec loop: commit", (int64_t) T);
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                wplan.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                if (!follow.empty()) { follow_differ += fdiff[(size_t) i]; ++follow_emitted; }
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            generation = {};
            const GatedDraftLimit draft_limit(mtp, gated_proposal, S_mtp, p + a + 1, o.max_context);
            const bool drafted = !draft_on || (int64_t) produced.size() >= max_new || draft_next(T, a);
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (gated_proposal && !w.first) {
                wplan.source.observe_committed(w.lookup, w.lookup ? T : std::min(S_mtp, room), a, w.match);
                // p has advanced: the window was p - a - 1 ..
                wplan.source.defer_cost(std::max<int64_t>(p - a - 2 + T, p + S_mtp), T, round_ms);
            }
            if (spec_trace != nullptr && timed_round) {
                std::fprintf(spec_trace, "%d %d %d %.3f", T, a, w.lookup ? 1 : 0, round_ms);
                for (int i = 0; i < o.spec - 1; ++i)
                    std::fprintf(spec_trace, " %.4f", (double) wplan.window_probs()[i]);
                std::fprintf(spec_trace, "\n");
            }
            if (!t_cycle.empty() && timed_round && T <= o.spec && t_cycle_seen[(size_t) T]++ > 0) {
                t_cycle_ms[(size_t) T] += round_ms;
                ++t_cycle_n[(size_t) T];
            }
            wplan.observe(w, a, round_ms, false);
            if (rounds % 64 == 0)
                std::fprintf(stderr, "strata generate: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        // the last commit (set_commit_async) before anything reads the session again
        if (!ver.wait_commit(err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (hashes != nullptr) std::fclose(hashes);
        if (spec_trace != nullptr) std::fclose(spec_trace);
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (!follow.empty())
            std::printf("%-24s %lld of %lld emitted tokens differ from the argmax\n", "follow",
                        (long long) follow_differ, (long long) follow_emitted);
        if (o.spec_min_p > 0.0 || o.draft_conf > 0.0 || wplan.cost_aware) {   // (any rule that sets the width)
            std::printf("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
            std::printf("  (min draft probability %.2f, draft-conf %.2f, cost rule %d)\n", o.spec_min_p, o.draft_conf, wplan.cost_aware ? 1 : 0);
        }
        if (ver.spec_windows > 0)   // STRATA_SPEC_PROB: windows judged against the MTP's q lists (only an MTP-drawn window should be)
            std::printf("%-24s %lld windows judged against q lists (source: %s)\n", "probabilistic accept", (long long) ver.spec_windows,
                        spec_uses_df2 ? "DFlash2" : "MTP");
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts",
                        (long long) wplan.lookup.windows, (long long) wplan.lookup.ok, (long long) wplan.lookup.drafts);
        if (o.lookup_chain > 0)
            std::printf("%-24s %lld windows, chained drafts accepted %lld of %lld\n", "lookup chain",
                        (long long) wplan.chained.windows, (long long) wplan.chained.ok,
                        (long long) wplan.chained.drafts);
        if (wplan.cost_aware) std::printf("%-24s %s\n", "spec cost table", wplan.policy.cost_table_text().c_str());
        if (!t_cycle.empty()) {
            std::printf("%-24s", "window cost (ms/round)");
            for (size_t i = 1; i < t_cycle_ms.size(); ++i)
                if (t_cycle_n[i] > 0) std::printf(" T%zu:%.2f(n=%lld)", i, t_cycle_ms[i] / (double) t_cycle_n[i], (long long) t_cycle_n[i]);
            std::printf("\n");
            {   // STRATA_VERIFY_PROFILE / STRATA_ROUTE_STATS: the same split per window width (bench only)
                const std::string pt = ver.profile_by_t_report();
                if (!pt.empty()) std::printf("GPU stages per window width (ms):\n%s", pt.c_str());
                const std::string rs = ver.route_stats_report();
                if (!rs.empty()) std::printf("routed experts per window width:\n%s", rs.c_str());
            }
        }
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0 && pool.numa_slices() == 2)   // STRATA_NUMA_SPLIT: each slice's drain, publish to its last task
            std::printf("; numa slices: s0 %.3f ms  s1 %.3f ms", pool.numa_slice_ms(0) / rounds, pool.numa_slice_ms(1) / rounds);
        if (rounds > 0) std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (rounds > 0 && drive.d.fetch_admit > 0)
            std::printf("%-24s %lld experts admitted from the PCIe share (%.2f a round)\n", "fetch admission",
                        (long long) drive.d.fetch_admitted, (double) drive.d.fetch_admitted / (double) rounds);
        if (res_checking)
            std::printf("%-24s %lld windows, %lld with mismatches (%lld entries), %lld layer-windows with CPU experts and "
                        "no activation published\n", "residency check", (long long) res_windows,
                        (long long) res_bad_windows, (long long) res_bad, (long long) drive.d.stale_x);
        if (src.complement_ready())
            std::printf("%-24s %.2f GiB of experts in RAM, %lld exchanged with the VRAM tier, %lld blob reads from "
                        "the file\n", "resident RAM", (double) src.resident_bytes() / 1073741824.0,
                        (long long) src.exchanges(), (long long) src.file_reads());
        if (srcp == &src) {  // CS-T: the RAM and file tiers (the GPU cache's share is the hit rate above)
            const double fms = src.file_ms() - fms0, fmb = (double) (src.file_blob_bytes() - fbytes0) / 1e6;
            std::printf("%-24s decode: RAM %lld blobs, files %lld blobs, %.1f MB read from the files (%.2f MB/round, "
                        "%.1f ms/round of reading, %.2f GB/s per reading thread)%s; prompt copies %.1f MB\n",
                        "expert tiers", (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        fmb, rounds > 0 ? fmb / rounds : 0.0, rounds > 0 ? fms / rounds : 0.0,
                        fms > 0 ? fmb / fms : 0.0, src.gguf_mode() ? " (the GGUF in place)" : "",
                        (double) (fall0 - fbytes0) / 1e6);
            if (src.io_stats()) {
                const strata::core::FileExpertSource::IoCounters ic = src.io_counters();
                std::printf("%-24s file reads: page cache held %.1f MB, not %.1f MB (mincore at the read); io prefetch %s: "
                            "%llu read ahead, %llu used, %llu unused, %llu skipped (cached), %llu dropped; pread %.1f MB in %llu calls "
                            "(%.2f ms each); decode waited %.1f ms/round in the fetches\n", "io path",
                            (double) ic.cached_bytes / 1e6, (double) ic.uncached_bytes / 1e6, src.io_prefetch() ? "on" : "off",
                            (unsigned long long) ic.pf_read_blobs, (unsigned long long) ic.pf_used,
                            (unsigned long long) ic.pf_unused, (unsigned long long) ic.pf_resident_skips,
                            (unsigned long long) ic.pf_dropped, (double) ic.pread_bytes / 1e6,
                            (unsigned long long) ic.pread_n, ic.pread_n > 0 ? (double) ic.pread_us / 1e3 / (double) ic.pread_n : 0.0,
                            rounds > 0 ? (double) ic.crit_us / 1e3 / rounds : 0.0);
            }
            if (drive.d.lookahead != nullptr)
                std::printf("%-24s %lld experts warmed, %lld of the file tier's %lld blob reads had been warmed (%.1f%%); "
                            "predictor %.1f ms/round on its thread, %lld layers skipped (still busy)\n", "routing prefetch",
                            (long long) src.warmed(), (long long) src.warmed_hits(),
                            (long long) (src.file_reads() - files0),
                            src.file_reads() > files0 ? 100.0 * (double) src.warmed_hits() / (double) (src.file_reads() - files0) : 0.0,
                            rounds > 0 ? drive.d.lookahead->busy_ms() / rounds : 0.0, (long long) drive.d.lookahead->skipped());
        }
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        if (rounds > 0 && ver.pcie_balance_on()) {
            const strata::core::PcieModel& pm = drive.d.pcie_model;
            std::printf("%-24s CPU %.0f us + %.0f an expert + %.0f a PCIe one, GPU %.0f us + %.0f a PCIe expert (the "
                        "PCIe count per layer from these)\n", "pcie model", 1000 * pm.a, 1000 * pm.c, 1000 * pm.d,
                        1000 * pm.g0, 1000 * pm.g);
        }
        (void) pool_ms0;
#if defined(STRATA_HAVE_DFLASH2)
        if (use_df2 && df2.rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds, GPU %.3f), %.0f MiB of VRAM\n", "dflash2", df2.ms_draft / (double) df2.rounds,
                        (long long) df2.rounds, df2.ms_gpu / (double) df2.rounds, (double) df2.vram_bytes() / 1048576.0);
        else
#endif
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        std::fprintf(stderr, "strata generate: cannot finish logits dump\n");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        cudaFreeHost(layer_stage);
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        cudaFreeHost(half_stage);
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) strata::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (cudaMemcpy(mx.data(), ss.block.mixed, mx.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading mixed back failed\n");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (cudaMemcpy(R.data(), ss.R, R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading R back failed\n");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // **THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.**  That measurement never enters the
        // token loop, so it excludes all six of these.  On the 78-token fixture + 200 generated tokens the six
        // sum to ~9 ms of non-layer work against ~1.5 ms of actual head GPU work - 16% of the token, and it is
        // not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  Three phases per layer and they were one number, which cannot
        // tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION - opposite fixes.
        // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
        // question is whether the time is in the drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (kvg.on)
            std::printf("%-24s %lld cells in VRAM, %lld grows, %lld trims, %lld experts given up, %lld refilled\n",
                        "  elastic K/V", (long long) kvg.cells, (long long) kvg.grows, (long long) kvg.trims,
                        (long long) kvg.evicted, (long long) kvg.refilled);
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // **THE ROUND TRIP, SPLIT AT THE RING.**  `ring latency` is the first half and stops when the ring
            // is seen; this is the second half - the driver calls after it, during which the GPU is IDLE
            // because `post[l]` has not been launched yet.  `--no-pool` is the arm that isolates it: 38.73
            // ms/token against a 26.32 ms pure-GPU floor is 12.4 ms of round trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can
            // be added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    cudaFree(d_next);
    cudaFree(d_logits);
    cudaFree(d_emb);
    cudaFree(d_parts);
    cudaFree(sbuf);
    cudaFree(arena);
    return 0;
}

#if defined(STRATA_USE_HIP) && defined(_WIN32)
// #3431: no stack, heap block or mapping the engine uses may lie under the GPU alias of a pinned expert-arena range (see
// claim_alias in src/core/pinned.cu: such a range and the ones after it are left unpinned, which is right but slower).  The
// aliases sit low in the address space (seen between 8 and 300 GiB), where Windows also places stacks and heaps, and in about one
// start in ten an alias covered the engine thread's stack.  So the free address space below 1 TiB is reserved here, for the life
// of the process (address space only, nothing committed), and the engine runs on a new thread: its stack, and everything the
// process allocates later, is placed above.  The aliases are not moved by it; one inside the reservation is clear.  What existed
// before main stays low: this first thread's stack (it only waits from here on, so an alias may cover it) and a few blocks of the
// process, which claim_alias still checks.  STRATA_STACK_ISOLATE=0 runs the engine on the first thread with nothing reserved.
namespace {
struct EngineArgs {
    int argc;
    char** argv;
    int rc;
};
DWORD WINAPI engine_thread(LPVOID p) {
    EngineArgs* a = static_cast<EngineArgs*>(p);
    a->rc = engine_main(a->argc, a->argv);
    return 0;
}
// every free piece below `limit` that a reservation can start in (64 KiB-aligned), reserved; the start of each piece
std::vector<uint64_t> reserve_low_address_space(uint64_t limit) {
    std::vector<uint64_t> held;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;
    uint64_t at = (uint64_t) si.lpMinimumApplicationAddress;
    MEMORY_BASIC_INFORMATION mbi;
    while (at < limit && VirtualQuery((void*) at, &mbi, sizeof mbi) == sizeof mbi) {
        const uint64_t end = (uint64_t) mbi.BaseAddress + mbi.RegionSize;
        if (mbi.State == MEM_FREE) {
            const uint64_t b = (at + gran - 1) & ~(gran - 1), e = std::min(end, limit);
            if (b < e) {
                if (VirtualAlloc((void*) b, (SIZE_T) (e - b), MEM_RESERVE, PAGE_NOACCESS) != nullptr) held.push_back(b);
            }
        }
        if (end <= at) break;
        at = end;
    }
    return held;
}
}  // namespace

int main(int argc, char** argv) {
    const char* iso = std::getenv("STRATA_STACK_ISOLATE");
    if (iso == nullptr || iso[0] != '0') {
        ULONG_PTR lo = 0, hi = 0;
        GetCurrentThreadStackLimits(&lo, &hi);
        strata::core::set_idle_thread_stack(lo, hi);
        strata::core::set_held_address_space(reserve_low_address_space(1ull << 40));
        EngineArgs a{argc, argv, 1};
        HANDLE h = CreateThread(nullptr, 0, engine_thread, &a, 0, nullptr);   // 0: the exe's own stack size, as the first thread's
        if (h != nullptr) {
            WaitForSingleObject(h, INFINITE);
            CloseHandle(h);
            return a.rc;
        }
        strata::core::set_idle_thread_stack(0, 0);   // the engine runs on this thread after all
    }
    return engine_main(argc, argv);
}
#endif
