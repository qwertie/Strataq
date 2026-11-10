// include/strata/common/experts/expert_pages.hpp - experts in the pages of a page pool that conversations take pages
// from too: a GPU's expert cache, or the RAM's copy of the experts (docs/MEMORY_MANAGER_PLAN.md, steps M2e, M2f and
// M5).
//
// For a GPU, the cache's arena is opened as uniform slots of one page each and is the pool's first chunk; the other
// chunks are allocated above it at whole pages from its start (on a GPU with virtual memory, right after it), so that
// every page is a slot: the kernels find an expert at the arena plus its slot times the page size, as in any cache of
// uniform slots, and an expert takes several consecutive pages (ExpertCache::set_expert_bytes).  Experts come in
// through ExpertCache::admit, in the expert profile's order, and leave when a growing conversation takes their pages,
// a prompt borrows them for its buffers (`lend`) or a shrink gives back their chunk.  The RAM's pool (RamExperts) is
// made of page-locked host memory the same way, without a cache, its slots the pool's page numbers.  Before an
// expert's pages go to another use, the `leaving` hook may copy it elsewhere (a GPU's expert back to RAM).
#pragma once

#include "strata/common/state/page_pool.hpp"
#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// The memory a pool's chunks come from (a GPU's, or page-locked RAM), given by the engine.
struct ChunkMemory {
    /// Gets `bytes` of the memory; null when it has not that much.
    std::function<uint8_t*(uint64_t bytes)> alloc;
    /// Gives back what `alloc` gave.
    std::function<void(uint8_t*)> free;
    /// Copies `bytes` inside the memory and waits for the copy (a shrink moves conversation pages out of a chunk);
    /// false when it failed.
    std::function<bool(uint8_t* to, const uint8_t* from, uint64_t bytes)> copy;
    /// Each chunk `alloc` gives starts where the last one ended (one virtual range), so runs may cross chunks.
    bool contiguous = false;
};

/// Experts in the pages of a page pool made over one arena: a GPU's expert cache (whose slots are the pages) or host
/// memory.  The pool's conversations take the least recently used experts' pages as they grow, and `refill` puts
/// those experts back once pages are free again.  An expert's last use is its rank in the ranking `open` is given
/// (an expert outside it ranks after it, in the order it is first admitted) until the engine counts a use
/// (note_use), so the coldest leave first.  Not thread-safe: the engine's step loop is its only caller.
class ExpertPages {
public:
    /// The engine's side of the pool: copying an expert in, publishing where experts are, and an expert's last look.
    struct Hooks {
        /// Copies an expert's bytes into its pages (from `first`) and waits for the copy; false (and `err`) when it
        /// could not.
        std::function<bool(ExpertKey key, int32_t first, std::string& err)> load;
        /// Tells the engine where an expert is now: its first page, or kNotResident when it left the pool.  A GPU's
        /// engine updates the residency tables its kernels read before its next step.
        std::function<void(ExpertKey key, int32_t first)> moved;
        /// Called with an expert's bytes (in the pool's memory) before its pages go to another use; may be empty.
        std::function<void(ExpertKey key, const uint8_t* bytes)> leaving;
    };

    /// Makes the pool over `cache`'s arena (open, with uniform slots of the page size and no expert yet), which is
    /// its first chunk, and chunks of `memory` up to `bytes` (whole chunks), with the conversations' `layout` and
    /// `config`'s limits; `config`'s page and chunk sizes are taken from the cache.  The experts of layer l are
    /// `blob_bytes[l]` bytes, `n_expert` per layer; at most `max_experts` experts are admitted.  `ranking` is the
    /// experts hottest first (the expert profile's order): their ranks, and the pool gives each expert size room for
    /// the ones of the first `max_experts` that fit (PagePoolConfig::expert_ranking).  From then on the cache's
    /// `admit` takes pages from the pool.  The pool opens without experts: `refill` brings in those of the first
    /// `max_experts` that fit (is_refill_wanted).
    bool open(ExpertCache& cache, ChunkMemory memory, uint64_t bytes, const StateLayout& layout,
              PagePoolConfig config, const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
              const std::vector<ExpertKey>& ranking, std::string& err);
    /// The same in chunks of `config.chunk_bytes` of `memory` only, without a cache: `admit` gives experts their
    /// pages, and a slot is a page's number.
    bool open(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
              const std::vector<ExpertKey>& ranking, std::string& err);
    /// Takes or gives back chunks so that the pool holds `bytes` (PagePool::resize, whose refusal it gets).  After a
    /// growth, `refill` brings in the experts of the first `max_experts` that the pool lacks.
    PoolResize resize(uint64_t bytes);
    /// Makes `ranking` (hottest first) the experts the pool holds, at most `max_experts` of them, as `open` does: the
    /// experts it holds that are not among the first `max_experts` leave (the `leaving` and `moved` hooks), and
    /// `refill` brings in the others.  strata-q38 gives a layer split's card its new layers' experts when the cut
    /// moves.
    void set_ranking(const std::vector<ExpertKey>& ranking, int64_t max_experts);
    /// Gets the experts in rank order, hottest first: set_ranking's, then those admit took from outside it.
    const std::vector<ExpertKey>& get_ranking() const { return ranked_; }
    /// Lends a run at the top of the pool for a prompt's buffers (PagePool::lend): `bytes` in whole pages, leaving no
    /// free pages for conversations (the prompt's own took theirs first); an expert it moves stays in the pool (the
    /// `moved` hook), one it evicts leaves it (`leaving` and `moved`) until `refill` brings it back.  Gets the run's
    /// address, or null (`lend.refusal` says why).
    uint8_t* lend(uint64_t bytes, PoolLend& lend);
    /// Tells whether `lend(bytes, ...)` may succeed now (PagePool::can_lend).
    bool can_lend(uint64_t bytes) const;
    /// Gives back the run `lend` lent.
    void give_back() { pool_->give_back(); }
    /// Tells whether the pool's chunks are back to back in one virtual range (ChunkMemory::contiguous).
    bool is_contiguous() const { return memory_.contiguous; }
    /// Gets the slot of a pool's page: what the cache's residency and the kernels use.
    int32_t get_slot(PageId page) const;
    void set_hooks(Hooks hooks) { hooks_ = std::move(hooks); }
    /// Gets the page pool that the experts share with the memory's conversations.
    PagePool& pool() { return *pool_; }
    const PagePool& pool() const { return *pool_; }

    /// Gets pages for an expert: the free run nearest its size's home, or the run of the least recently used expert
    /// of its size when that one was last used before `used_at` (the pool's clock), which also becomes the new
    /// expert's last use; 0 stands for the time of the expert's rank, so that only lower-ranked experts give way.
    /// Gets its first page, or kNotResident when it may not come in or there is no room.  Its bytes are the caller's
    /// to copy in.
    int32_t admit(int64_t layer, int64_t expert, uint64_t used_at = 0);
    /// Brings expert `in` of `layer` into free pages while the pool has some for it; else into the pages of expert
    /// `out` of the same layer (none when `out` is negative), which leaves as it would for a conversation (the
    /// `leaving` and `moved` hooks) but is not put back by `refill`.  Copies `in`'s bytes in.  Gets 1 when `in` came
    /// in; 0 when nothing changed: the pool already holds `in`, or has no free pages for it and does not hold `out`;
    /// -1 (and `err`) when the copy failed: then `in` is not in the pool (nor `out`, when it gave way).
    int replace(int32_t layer, int32_t out, int32_t in, std::string& err);
    /// Counts a use of an expert the pool holds, so that a conversation evicts it after the experts used less
    /// recently (nothing when the pool does not hold it).
    void note_use(ExpertKey key) { (void) pool_->use_expert(key); }
    /// Gets the pool's clock now (the times admit takes).
    uint64_t get_time() const { return now_(); }
    /// Tells whether experts that conversations evicted wait for pages that may be free for them now or soon.
    bool is_refill_wanted() const;
    /// Counts the experts of the ranking's first `max_experts` that the pool lacks, including those that wait for
    /// pages conversations hold.
    size_t get_missing_count() const { return evicted_.size(); }
    /// Takes the experts for which `is_held_elsewhere` is true off the list `refill` works through; they still come in
    /// when a caller admits them.  The RAM's pool uses it so that its start-up fill skips the GPUs' experts.
    void stop_wanting_if(const std::function<bool(ExpertKey)>& is_held_elsewhere);
    /// Puts back up to `max` evicted experts, hottest first, into free pages experts may take; it evicts no expert,
    /// since one there now may be one that decoding uses more (the adaptive tier's).  Gets how many (-1 when a copy
    /// failed, with `err`); `wait_ms` gets 0, or the milliseconds until more pages open to experts when the next
    /// expert found no room before then.
    int refill(int max, uint64_t& wait_ms, std::string& err);

private:
    bool open_pool(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
                   const std::vector<ExpertKey>& ranking, std::string& err);
    uint8_t* alloc_chunk(uint64_t bytes);
    void free_chunk(uint8_t* chunk);
    void on_evicted(ExpertKey key, PageId first);
    void on_relocated(ExpertKey key, PageId to);
    void want_missing();
    int32_t admit_free(ExpertKey key);
    size_t index_of(ExpertKey key) const { return (size_t) key.layer * (size_t) n_expert_ + (size_t) key.expert; }
    /// Gets the clock time that stands for the last use of the expert of rank `rank`.
    uint64_t get_used_at(int32_t rank) const { return t0_ - (uint64_t) rank; }

    ExpertCache* cache_ = nullptr;       ///< the GPU's cache whose slots are the pages; null for host memory
    uint8_t* arena_ = nullptr;           ///< the cache's arena: its first chunk, and slot 0's address
    ChunkMemory memory_;
    std::map<uint8_t*, uint8_t*> raw_;   ///< each chunk `memory_` gave, by its start: the allocation it starts in
    std::unique_ptr<PagePool> pool_;
    Hooks hooks_;
    std::function<uint64_t()> now_;      ///< the pool's clock
    std::vector<uint32_t> pages_;        ///< per layer, the pages one of its experts takes
    int64_t n_expert_ = 0, max_experts_ = 0;
    uint64_t t0_ = 0;                    ///< the clock time given to the hottest expert's last use
    std::vector<int32_t> rank_;          ///< per expert, its rank; -1 while it has none
    std::vector<ExpertKey> ranked_;      ///< the experts in rank order
    std::set<int32_t> evicted_;          ///< the ranks of the experts `refill` brings in: evicted, or not yet in
    /// The pool's free pages when `refill` last found no room and no page about to open; refills wait until that
    /// changes.  UINT64_MAX: not waiting.
    uint64_t blocked_at_free_ = UINT64_MAX;
    /// While admit_free runs: `admit` takes free pages only (a GPU's pool reaches it through the cache's admitter)
    bool free_only_ = false;
};

}  // namespace strata::core
