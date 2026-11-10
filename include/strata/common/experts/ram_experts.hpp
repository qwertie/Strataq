// include/strata/common/experts/ram_experts.hpp - the RAM's page pool (docs/MEMORY_MANAGER_PLAN.md, step M5): the
// experts a GPU's pool does not hold, copies of some it does while room allows, and paused conversations, in one pool
// of page-locked memory that the GPU can read in place.  The experts in neither pool are read from the SSD without
// the OS file cache (the file source's unbuffered reads), so that the cache does not take the RAM back.
#pragma once

#include "strata/common/experts/expert_pages.hpp"
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

namespace strata::core {

/// The experts the CPU computes and a GPU's pool loads, as an ExpertSource: from the RAM's page pool when it holds
/// them, else read from the SSD.  An expert a step reads from the SSD (blob, prefetch) takes the place of the least
/// recently used one of its size, unless the current layer has used that one, so the pool keeps the experts the
/// conversations use; else it is read into a ring of staging buffers.  The pool's conversations are paused ones
/// (pool_copy.hpp); they take the least recently used experts' pages, and `refill` puts those back.
///
/// Threads: the engine's thread makes every call but blob, prefetch, copy_blob and copy_blobs, which the prompt path's
/// threads make too while the engine's thread reads a prompt chunk and makes no other call (a layer split's next card
/// reads its part of a chunk on a thread of its own while the first card reads the next chunk).  The prompt path asks
/// blob only for experts the pool holds (transient() false), which takes no page and no staging buffer, so a pointer
/// it got stays valid while the prompt is read.
class RamExperts : public ExpertSource {
public:
    RamExperts() = default;
    ~RamExperts() override;
    RamExperts(const RamExperts&) = delete;
    RamExperts& operator=(const RamExperts&) = delete;

    /// Makes the pool in `bytes` of page-locked RAM, less a staging ring of `ring` buffers of the largest expert (as
    /// many as the experts a step's layer may read from the SSD), with the conversations' `layout` and `config`'s
    /// limits; `config.page_bytes` is the GPU pool's, so that a conversation's blocks are alike in both, and the pool
    /// takes its memory in chunks of `config.chunk_bytes` (at most the pool).  The pool opens empty; `refill` brings
    /// in the experts of `ranking` in order, as far as they fit.  The experts of layer l are `blob_bytes[l]` bytes,
    /// `n_expert` per layer.  `files` reads the experts from the SSD and must outlive this object.
    bool open(ExpertSource& files, uint64_t bytes, int ring, const StateLayout& layout, PagePoolConfig config,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, const std::vector<ExpertKey>& ranking,
              std::string& err);
    /// Puts up to `max` of the experts the pool wants into free pages (ExpertPages::refill), reading them from the
    /// SSD in one batch of requests.  Gets how many, or -1 (with `err`) when the read failed; then none of them stays.
    int refill(int max, uint64_t& wait_ms, std::string& err);
    /// Takes or gives back chunks so that the pool and the staging ring hold `bytes` (ExpertPages::resize).
    PoolResize resize(uint64_t bytes) { return pages_.resize(bytes > ring_bytes() ? bytes - ring_bytes() : 0); }
    /// Copies expert `key` in from `bytes` (any memory: a GPU's pages it is leaving) when the pool does not hold it
    /// and has room for it among colder experts of its size; tells whether it did.  An expert of a layer past the
    /// pool's (DeepSeek's drafter, whose experts only the GPU's pool holds) is never copied.
    bool keep_copy(ExpertKey key, const uint8_t* bytes);
    /// Lends a run at the top of the pool for a prompt path's host buffers (ExpertPages::lend); a moved expert is read
    /// from its new pages.  Gets the run's address, or null (`lend.refusal` says why).
    uint8_t* lend(uint64_t bytes, PoolLend& lend) {
        const std::unique_lock lock(mu_);
        return pages_.lend(bytes, lend);
    }
    /// Gives back the run `lend` lent.
    void give_back() {
        const std::unique_lock lock(mu_);
        pages_.give_back();
    }
    /// Tells whether the pool holds expert (`layer`, `expert`).
    bool holds(int64_t layer, int64_t expert) const { return first_page_[index_of(layer, expert)] != kNotResident; }
    /// Gets the experts in the pool, which the paused conversations share.
    ExpertPages& pages() { return pages_; }
    const ExpertPages& pages() const { return pages_; }
    /// Tells whether the pool's memory is page-locked, which the GPU reads in place (device_alias).
    bool is_page_locked() const { return page_locked_; }
    /// Gets the bytes of the staging ring (outside the pool).
    uint64_t ring_bytes() const { return ring_slots_ * slot_bytes_; }
    /// Gets the RAM the GPU maps: the pool and the staging ring when they are page-locked, else 0.
    uint64_t get_mapped_bytes() const {
        return page_locked_ ? pages_.pool().report().total.bytes + ring_bytes() : 0;
    }
    /// Times reading a staging ring's worth of experts (one of every layer in turn) from the SSD, in GB/s; each
    /// `round` reads other experts, so that no read finds the drive's own cache warm.
    double time_ssd_reads(int round);
    /// Times copying `n` bytes from `bytes` (a GPU's memory) into the staging ring, in GB/s.
    double time_copy_in(const uint8_t* bytes, uint64_t n);

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override { return is_page_locked() && holds(layer, expert); }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t) const override { return is_page_locked(); }
    bool transient(int64_t layer, int64_t expert) const override { return !holds(layer, expert); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;

private:
    /// Experts to read from the SSD, and where each goes.
    struct Reads {
        std::vector<int32_t> layers, experts;
        std::vector<uint8_t*> dst;
        void add(int64_t layer, int64_t expert, uint8_t* to) {
            layers.push_back((int32_t) layer);
            experts.push_back((int32_t) expert);
            dst.push_back(to);
        }
    };

    size_t index_of(int64_t layer, int64_t expert) const { return (size_t) (layer * n_expert_ + expert); }
    uint8_t* get_page(int32_t first) const { return pages_.pool().page_address((PageId) first); }
    bool read_from_ssd(const Reads& reads, uint64_t most = UINT64_MAX);
    bool read_in(const Reads& reads);
    uint8_t* take_place(int64_t layer, int64_t expert);
    uint8_t* take_slot(int64_t layer, int64_t expert);
    const uint8_t* find_staged(int64_t layer, int64_t expert) const;

    uint8_t* alloc(uint64_t bytes);
    void free(uint8_t* p);

    /// While `refill` runs, the batch it reads from the SSD after admitting its experts (the `load` hook adds them);
    /// null otherwise, when the hook reads each expert at once.
    Reads* refill_reads_ = nullptr;
    ExpertSource* files_ = nullptr;
    /// The memory is page-locked (the GPU reads it in place), else pageable: decided by the first allocation.
    bool page_locked_ = true;
    std::map<const uint8_t*, uint8_t*> views_;   ///< each page-locked allocation's start -> its address on the GPU
    ExpertPages pages_;   ///< (after views_: destroying the pool gives its chunks back through them)
    std::vector<uint64_t> blob_bytes_;
    int64_t n_expert_ = 0;
    std::vector<int32_t> first_page_;      ///< per expert, its first page in the pool, or kNotResident
    uint8_t* ring_ = nullptr;
    uint64_t slot_bytes_ = 0;              ///< a staging buffer: the largest expert, rounded up to 4 KiB
    size_t ring_slots_ = 0, next_slot_ = 0;
    std::vector<int64_t> ring_owner_;      ///< per staging buffer, the expert it holds (index_of), or -1
    /// The pool's clock when the CPU began the current layer (begin_layer): no expert used since gives its pages to
    /// one read from the SSD, as the layer's work may still read it (the previous layers' work is done).  0 before
    /// the first layer: only lower-ranked experts give way.
    uint64_t layer_t0_ = 0;
    /// Held by blob, prefetch and refill, which change the pool and the ring; shared by copy_blobs, which only reads
    /// them
    mutable std::shared_mutex mu_;
};

}  // namespace strata::core
