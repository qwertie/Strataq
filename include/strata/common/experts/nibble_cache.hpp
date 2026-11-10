// include/strata/common/experts/nibble_cache.hpp - STRATA_EXPERT_NIBBLE=1 (strata/common/kernels/expert_nibble.hpp):
// the 4-bit fill and the checks of one expert cache's converted slots.  Used by Qwen3.8's loader (Model::load) and
// by the old engine's serve loop (STRATA_EXPERT_NIBBLE_RECHECK); nothing here runs with the switch off.
#pragma once

#include "strata/base/gpu/on_device.hpp"
#include "strata/common/kernels/cpu/expert_layout.hpp"
#include "strata/common/kernels/expert_nibble.hpp"
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

#if STRATA_EXPERT_NIBBLE_BUILT
/// One expert cache's converted slots.  The startup fill hands a converted layer's pairs to `add` instead of
/// `fill_slot_blocking`; per layer, `kBatch` GGUF blobs at a time go H2D into one staging buffer (the bytes
/// fill_slot_blocking would copy, from the source's `blob_stable`, which stays valid) and, per batch, the
/// transcoder writes the codes, scales and d into the slots and round-trips every item, the GGUF down rows are copied
/// behind them, the GGUF blocks are rebuilt from every slot and compared with the blob (both count against the layer
/// in `d_bad`), and each slot's hash is recorded for STRATA_EXPERT_NIBBLE_RECHECK.  One stream, so the staging
/// buffer's reuse is ordered without events.  The device arrays live as long as the process (RECHECK reads them);
/// `finish` frees the staging buffer.
struct NibbleCache {
    static constexpr int64_t kBatch = 64;
    struct Pending { int32_t expert, slot; const uint8_t* blob; };
    struct Batch { int32_t layer; int64_t at, n; };
    strata::core::ExpertCache* cache = nullptr;
    int dev = 0;                                         ///< the cache's device
    std::vector<std::vector<Pending>> pending;           ///< per layer: queued, at most kBatch
    std::vector<Batch> batches;                          ///< what was transcoded and hashed, in that order
    std::vector<std::pair<int32_t, int32_t>> pairs;      ///< (layer, expert) of each converted slot, in that order
    std::vector<int32_t> slots;                          ///< and its slot
    unsigned long long* h_ptrs = nullptr;                ///< pinned: the slots' device addresses, in that order
    unsigned long long* d_ptrs = nullptr;                ///< the same on the device (the transcoder's and hash's input)
    unsigned long long* d_ref = nullptr;                 ///< each slot's hash as loaded
    unsigned long long* d_now = nullptr;                 ///< RECHECK's
    unsigned long long* d_bad = nullptr;                 ///< per layer: items that did not round-trip, words that differ
    uint8_t* staging = nullptr;
    uint64_t raw_max = 0;
    cudaStream_t stream = nullptr;
    int64_t cap = 0, n = 0;
    double seconds = 0;

    bool active() const { return cache != nullptr; }

    static std::string cuda_fail(const char* what, cudaError_t e) { return std::string(what) + ": " + cudaGetErrorString(e); }

    /// For `c` on the current device, `want` converted pairs at most.
    bool begin(strata::core::ExpertCache& c, int64_t want, std::string& err) {
        if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
        cache = &c;
        cap = want > 0 ? want : 1;
        const auto& lay = strata::kernels::cpu::expert_layout();
        pending.assign((size_t) lay.n_layers, {});
        for (int64_t l = 0; l < lay.n_layers; ++l)
            if (nibble::layer(l).on) raw_max = std::max<uint64_t>(raw_max, nibble::layer(l).raw_bytes);
        cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaMallocHost((void**) &h_ptrs, (size_t) cap * 8);
        if (e == cudaSuccess) e = cudaMalloc((void**) &d_ptrs, (size_t) cap * 8);
        if (e == cudaSuccess) e = cudaMalloc((void**) &d_ref, (size_t) cap * 8);
        if (e == cudaSuccess) e = cudaMalloc((void**) &d_now, (size_t) cap * 8);
        if (e == cudaSuccess) e = cudaMalloc((void**) &d_bad, (size_t) lay.n_layers * 8);
        if (e == cudaSuccess) e = cudaMemsetAsync(d_bad, 0, (size_t) lay.n_layers * 8, stream);
        if (e == cudaSuccess) e = cudaMalloc((void**) &staging, (size_t) ((uint64_t) kBatch * raw_max));
        if (e != cudaSuccess) {
            err = cuda_fail("the 4-bit fill's buffers", e);
            return false;
        }
        if (const int rc = nibble::prepare((void*) stream); rc != 0) {
            err = std::string("the 4-bit fill's GGUF codebooks: ") +
                  (rc == (int) cudaErrorInvalidValue ? "a grid entry is not unique" : cudaGetErrorString((cudaError_t) rc));
            return false;
        }
        return true;
    }

    bool add(int32_t layer, int32_t expert, int32_t slot, const uint8_t* blob, std::string& err) {
        auto& q = pending[(size_t) layer];
        q.push_back({expert, slot, blob});
        return (int64_t) q.size() < kBatch || flush(layer, err);
    }

    bool flush(int32_t layer, std::string& err) {
        auto& q = pending[(size_t) layer];
        if (q.empty()) return true;
        const auto t0 = std::chrono::steady_clock::now();
        const nibble::LayerVram& L = nibble::layer(layer);
        const int64_t k = (int64_t) q.size();
        if (n + k > cap) {
            err = "the 4-bit fill was given more converted pairs than it counted";
            return false;
        }
        cudaError_t e = cudaSuccess;
        for (int64_t i = 0; i < k && e == cudaSuccess; ++i) {
            uint8_t* dst = cache->device_slot(q[(size_t) i].slot);
            if (dst == nullptr || cache->slot_room(q[(size_t) i].slot) < L.hashed) {   // a slot sized for the GGUF blob
                err = "the 4-bit fill: slot " + std::to_string(q[(size_t) i].slot) +
                      (dst == nullptr ? " is outside the cache" : " is smaller than a 4-bit slot");
                return false;
            }
            h_ptrs[n + i] = (unsigned long long) (uintptr_t) dst;
            e = cudaMemcpyAsync(staging + (uint64_t) i * L.raw_bytes, q[(size_t) i].blob, (size_t) L.raw_bytes,
                                cudaMemcpyHostToDevice, stream);
        }
        if (e == cudaSuccess) e = cudaMemcpyAsync(d_ptrs + n, h_ptrs + n, (size_t) k * 8, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) {
            err = cuda_fail("the 4-bit fill's copies in", e);
            return false;
        }
        auto m2 = [&](const char* what, int rc) -> bool {   // iq_kernels.cu's: a CUDA error code
            if (rc == 0) return true;
            err = std::string("the 4-bit ") + what + ": " + cudaGetErrorString((cudaError_t) rc);
            return false;
        };
        if (!m2("transcoder", nibble::transcode(L, staging, k, d_ptrs + n, d_bad + layer, (void*) stream))) return false;
        for (int64_t i = 0; i < k && e == cudaSuccess; ++i)
            e = cudaMemcpyAsync((uint8_t*) (uintptr_t) h_ptrs[n + i] + nibble::kDownOff,
                                staging + (uint64_t) i * L.raw_bytes + L.raw_down_off, (size_t) L.down_bytes,
                                cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) {
            err = cuda_fail("the 4-bit fill's down rows", e);
            return false;
        }
        if (!m2("GGUF check", nibble::check_gguf(L, staging, k, d_ptrs + n, d_bad + layer, (void*) stream))) return false;
        if (!m2("slot hash", nibble::hash_slots(L, d_ptrs + n, k, d_ref + n, (void*) stream))) return false;
        // the next batch's copies in land in the same staging bytes: they wait for this batch's kernels (one stream)
        for (const Pending& p : q) {
            pairs.emplace_back(layer, p.expert);
            slots.push_back(p.slot);
        }
        batches.push_back({layer, n, k});
        n += k;
        q.clear();
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }

    /// The rest of the queue, then waits: false (with `err` naming every layer) when an item did not round-trip or a
    /// rebuilt GGUF block differs from its blob.
    bool finish(std::string& err) {
        for (int32_t l = 0; l < (int32_t) pending.size(); ++l)
            if (!flush(l, err)) return false;
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t e = cudaStreamSynchronize(stream);
        std::vector<unsigned long long> bad(pending.size(), 0);
        if (e == cudaSuccess) e = cudaMemcpy(bad.data(), d_bad, bad.size() * 8, cudaMemcpyDeviceToHost);
        if (staging != nullptr) (void) cudaFree(staging);
        staging = nullptr;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (e != cudaSuccess) {
            err = cuda_fail("the 4-bit fill", e);
            return false;
        }
        for (size_t l = 0; l < bad.size(); ++l)
            if (bad[l] != 0) {
                char b[192];
                std::snprintf(b, sizeof b, "%slayer %zu (%s): %llu items or words do not round-trip", err.empty() ? "" : "; ",
                              l, nibble::fmt_name(nibble::layer((int64_t) l).gu_type), bad[l]);
                err += b;
            }
        if (!err.empty()) {
            err += " - the 4-bit code does not hold these weights (STRATA_EXPERT_NIBBLE_SKIP=<layers> keeps a layer's "
                   "GGUF blocks)";
            return false;
        }
        return true;
    }

    /// The end-to-end check of the fill as it is wired, on converted pair (layer, expert) with `raw` its GGUF blob: the
    /// slot read back, its hash against expert_nibble.hpp's host reference of what the device recorded, its down rows
    /// against the blob's.  (Every item and every GGUF block of every slot was checked on the device by `finish`.)
    bool check(int32_t layer, int32_t expert, const uint8_t* raw, std::string& err) const {
        int64_t at = -1;
        for (size_t i = 0; i < pairs.size() && at < 0; ++i)
            if (pairs[i].first == layer && pairs[i].second == expert) at = (int64_t) i;
        if (at < 0 || raw == nullptr) {
            err = "the 4-bit check: no converted slot or no GGUF blob for layer " + std::to_string(layer) + " expert " +
                  std::to_string(expert);
            return false;
        }
        const nibble::LayerVram& L = nibble::layer(layer);
        std::vector<uint8_t> got((size_t) L.hashed);
        unsigned long long h = 0;
        cudaError_t e = cudaMemcpy(got.data(), cache->device_slot(slots[(size_t) at]), got.size(), cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(&h, d_ref + at, 8, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            err = cuda_fail("the 4-bit check's read back", e);
            return false;
        }
        const char* why = nibble::hash_host(got.data(), L.hashed) != h ? "the device slot hash differs from the host reference"
                        : std::memcmp(got.data() + nibble::kDownOff, raw + L.raw_down_off, (size_t) L.down_bytes) != 0
                            ? "the down rows differ from the GGUF blob's"
                            : nullptr;
        if (why != nullptr) {
            err = "layer " + std::to_string(layer) + " expert " + std::to_string(expert) + " (" + nibble::fmt_name(L.gu_type) +
                  "): " + why;
            return false;
        }
        return true;
    }

    /// STRATA_EXPERT_NIBBLE_RECHECK: every converted slot's hash again, on the device, against the one recorded at load.
    /// False with the first slot that differs (or a CUDA error) in `err`; `bad_layer` its layer (-1: an error).
    bool recheck(std::string& err, int64_t& bad_layer) {
        bad_layer = -1;
        if (n == 0) return true;
        const strata::core::OnDevice on(dev);
        for (const Batch& b : batches) {
            const int rc = nibble::hash_slots(nibble::layer(b.layer), d_ptrs + b.at, b.n, d_now + b.at, (void*) stream);
            if (rc != 0) {
                err = cuda_fail("the slot hash", (cudaError_t) rc);
                return false;
            }
        }
        std::vector<unsigned long long> ref((size_t) n), now((size_t) n);
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e == cudaSuccess) e = cudaMemcpy(ref.data(), d_ref, (size_t) n * 8, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(now.data(), d_now, (size_t) n * 8, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            err = cuda_fail("the slot hash read back", e);
            return false;
        }
        for (int64_t i = 0; i < n; ++i)
            if (ref[(size_t) i] != now[(size_t) i]) {
                char b[224];
                std::snprintf(b, sizeof b, "CUDA%d: layer %d expert %d (slot %d) hashes %016llx, %016llx at load", dev,
                              (int) pairs[(size_t) i].first, (int) pairs[(size_t) i].second, (int) slots[(size_t) i],
                              now[(size_t) i], ref[(size_t) i]);
                err = b;
                bad_layer = pairs[(size_t) i].first;
                return false;
            }
        return true;
    }
};
#else
// A build without the 4-bit kernels (STRATA_EXPERT_NIBBLE_BUILT 0: HIP, Metal, a CUDA build without
// STRATA_EXPERIMENTAL_SM60): nibble_refusal refuses the switch, so no cache is ever converted and none of this is entered.
struct NibbleCache {
    int64_t n = 0;
    double seconds = 0;
    std::vector<std::pair<int32_t, int32_t>> pairs;
    bool active() const { return false; }
    bool begin(strata::core::ExpertCache&, int64_t, std::string& err) { err = "the 4-bit slots are not in this build"; return false; }
    bool add(int32_t, int32_t, int32_t, const uint8_t*, std::string& err) { err = "the 4-bit slots are not in this build"; return false; }
    bool finish(std::string&) { return true; }
    bool check(int32_t, int32_t, const uint8_t*, std::string& err) const { err = "the 4-bit slots are not in this build"; return false; }
    bool recheck(std::string&, int64_t& bad_layer) { bad_layer = -1; return true; }
};
#endif

}  // namespace strata::core
