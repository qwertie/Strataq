// include/strata/common/state/page_pool.hpp - one memory (a GPU's, or the RAM the engine uses) cut into equal pages
// that experts and conversations trade while the engine runs.  docs/MEMORY_MANAGER_PLAN.md gives the design;
// src/common/state/README.md says how a conversation's state regions (state_layout.hpp) map onto pages.
#pragma once

#include "strata/common/state/state_layout.hpp"
#include "strata/base/platform/memory_manager.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace strata::core {

/// Index of a page in its pool.  It stays valid until a shrink gives back the chunk that holds it.
using PageId = uint32_t;
constexpr PageId no_page = UINT32_MAX;
/// Stands for the pool's clock at the call, as PagePool::add_expert's `used_at`.
constexpr uint64_t clock_now = UINT64_MAX;
/// The engine's handle on a conversation.
using ConversationId = uint64_t;

/// Names one expert's weights: its layer, and its index among that layer's experts.
struct ExpertKey {
    int32_t layer = 0, expert = 0;
    bool operator<(const ExpertKey& o) const { return layer != o.layer ? layer < o.layer : expert < o.expert; }
};

/// Sizes and limits of a pool.  The defaults are the plan's starting points, not measured ones.
struct PagePoolConfig {
    uint64_t page_bytes = 2ull << 20;      ///< an engine sets it from its experts' sizes (choose_expert_page_bytes)
    uint64_t chunk_bytes = 256ull << 20;   ///< the amount taken from and given back to the allocator at once
    /// Conversation tokens whose pages experts may not take, so that a new conversation never waits.  Counted as
    /// one conversation of that many tokens.
    int64_t tokens_min = 0;
    /// The most tokens all the pool's conversations may hold together.
    int64_t tokens_max = INT64_MAX;
    /// Clock units during which a page that a conversation gave back stays closed to experts, so that the two do not
    /// take the same pages back and forth.
    uint64_t hysteresis = 2000;
    /// The pages of each expert the pool may hold, hottest first (an expert profile's order).  The ones that fit tell
    /// how much room each expert size should get, which places each size's home point (PagePool::get_home).
    std::vector<uint32_t> expert_ranking;
    std::string label = "Expert and conversation pool";   ///< the name of this pool in allocation details
    std::vector<uint64_t> expert_bytes;   ///< actual weight bytes per layer, before rounding to pages
    /// The allocator's chunks may be one memory: a chunk it gives where the previous one ended (a GPU's virtual range,
    /// ChunkMemory::contiguous) is joined to it, so that runs may cross their bound.  Off, each run stays in its chunk.
    bool join_chunks = false;
    /// Supplies engine-specific category and purpose labels; empty uses the layer and expert size.
    std::function<MemoryLabel(ExpertKey, uint64_t)> label_expert;
};

/// What the pool calls outside itself.  The engine passes a device allocator (base/gpu) and a page copy on its
/// stream; tests pass host memory and a hand-moved clock.
struct PagePoolHooks {
    std::function<void*(size_t)> alloc;   ///< gets a chunk; nullptr when the memory has not that much
    std::function<void(void*)> free;      ///< gives a chunk back
    /// Gets the time for the experts' recency and the hysteresis: milliseconds by default.
    std::function<uint64_t()> now = [] {
        return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    /// Tells the expert's owner that its pages (from the given first one) go to another use, before the new user gets
    /// them, so that the owner can copy the expert to RAM or drop it.
    std::function<void(ExpertKey, PageId)> evicted = [](ExpertKey, PageId) {};
    /// Copies `pages` consecutive pages inside one chunk (or joined ones), when a shrink or a lend moves a
    /// conversation's block or an expert; returns false when it cannot.
    std::function<bool(PageId to, PageId from, uint32_t pages)> copy = [](PageId, PageId, uint32_t) { return false; };
    /// Tells the expert's owner that a lend copied its pages to `to` (PagePool::lend), so that it reads them there.
    std::function<void(ExpertKey, PageId to)> relocated = [](ExpertKey, PageId) {};
};

/// A pool's pages by use, for the GET_MEMORY_USE command.
struct PagePoolReport {
    struct Use {
        uint64_t pages = 0, bytes = 0;
    };
    Use experts;
    Use active;    ///< the growing regions (K/V) of active conversations
    Use fixed;     ///< the fixed regions of active conversations: recurrent state, the drafter's state
    Use paused;    ///< all pages of paused conversations
    Use lent;      ///< the run lent to a prompt's buffers (PagePool::lend)
    Use free;      ///< the rest, with the pages of conversations' holes that no block holds yet
    Use total;
    Use allocated;    ///< all but the free runs: experts, blocks, and the pages of holes no block holds yet
    /// The free pages in runs shorter than the smallest expert of the ranking: free memory no expert can use.
    Use fragmented;
    int active_conversations = 0, paused_conversations = 0;
    int64_t tokens = 0;   ///< held by all the pool's conversations
};

/// Outcome of PagePool::resize.
struct PoolResize {
    std::string refusal;   ///< why nothing (or, for a failed allocation or copy, not all) was done; empty when done
    /// Conversations holding pages in the chunks a shrink would give back, when the rest of the pool has no room for
    /// those pages: pausing (and removing) some of them, or any others, makes the room.
    std::vector<ConversationId> to_pause;
};

/// Outcome of PagePool::lend.
struct PoolLend {
    PageId first = no_page;   ///< the lent run's first page; no_page when it lent nothing
    std::string refusal;      ///< why it lent nothing; empty when it lent
    /// Conversations whose blocks it copied elsewhere (read their conversation_blocks again)
    std::vector<ConversationId> moved;
    uint32_t experts_moved = 0, experts_evicted = 0;   ///< the experts it copied elsewhere, and those it evicted
};

/// Gets the page size for experts whose sizes are `blob_bytes` (one per layer; the experts of a layer are all one
/// size): their greatest common divisor, so that every expert is a whole number of pages and no page is part empty
/// (25,600 bytes for Qwen3.8 IQ3_S, 92,160 for DeepSeek's 2.00 bpw set).  A page must be a multiple of 256 bytes,
/// the alignment the cache gives every slot, and at least 16 KiB, so that a pool has no more pages than its slot
/// tables can count; when the divisor is not, the multiple of 256 bytes from 16 KiB to 512 KiB that wastes the least
/// of the pages the experts take.
uint64_t choose_expert_page_bytes(const std::vector<uint64_t>& blob_bytes);

/// Names an expert size class consistently in GPU pools, RAM pools and model files.
std::string get_expert_memory_label(int32_t layer, uint64_t bytes);

/// Gets how many consecutive pages of `page_bytes` make one block of region `region`: a fixed region's whole size;
/// for a growing region the fewest pages whose waste (the bytes past the last whole unit of tokens) is at most 1%, up
/// to 1 MiB of block (or one unit, when that is larger), else within that limit the count that wastes least.  An
/// engine sizing a pool before making it calls this too.
uint32_t choose_block_pages(const StateRegion& region, uint64_t page_bytes);

/// The pages of one memory, in runs of consecutive pages: each run is free, holds one expert, holds one block of a
/// conversation, is a hole (an evicted expert's pages that one conversation fills with its blocks) or is lent to a
/// prompt's buffers.  A run stays inside one chunk, unless the chunks are joined (PagePoolConfig::join_chunks).
/// Pages may be small (every expert a whole number of them), so a pool may hold millions; the pool keeps runs, not
/// pages, and never walks the pages.
///
/// The bottom of the pool is the conversations' end: their blocks take the lowest free run that fits, so the token
/// minimum's reserve, which experts leave free, is filled first.  Each expert size has a home point, the middle of
/// the room the expert ranking gives that size, the rooms one after another from the reserve up, so that pages no
/// expert needs are left at the top, where `lend` lends a prompt's buffers.  An expert takes the free run of its size
/// nearest its home, so experts of one size sit together and the pages an expert leaves fit the next one of its size.
/// When none fits, the least-used expert of its size (used before it) gives up its run: exactly one eviction.  A
/// conversation past its free runs evicts the least-used expert of any size and fills that hole with its blocks
/// before it takes another; the hole goes back whole when its last block leaves.  A fixed region (a conversation's
/// recurrent state) needs one long run: the lowest free one, else the lowest stretch of experts it evicts.  Not
/// thread-safe: the engine's step loop is its only caller.
class PagePool {
public:
    /// Throws std::invalid_argument when a chunk holds no page, or a fixed region or one block of a growing region is
    /// larger than a chunk.
    PagePool(const StateLayout& layout, PagePoolConfig config, PagePoolHooks hooks);
    ~PagePool();
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;

    /// Takes or gives back chunks so that the pool holds `bytes` (rounded down to whole chunks), and places the
    /// experts' home points for the new size.  A shrink gives back the last chunks: their experts are evicted and
    /// their conversation blocks copied below (so read conversation_blocks again).  It is refused, changing nothing,
    /// while a run is lent or when an active conversation or the token minimum would not fit, and stops (having moved
    /// some blocks) when a block has no room.
    PoolResize resize(uint64_t bytes);
    /// Lends the top `pages` pages of the pool (one run, inside the joined chunks at its top) for a prompt's buffers
    /// until give_back.  Makes them free first: an expert there is copied into free pages elsewhere (above the
    /// reserve, else in the reserve's free pages) while that leaves `keep_free` free pages besides what the
    /// conversations' blocks there need, else evicted; a conversation's block there is copied elsewhere, evicting
    /// experts as its growth would.  Refused, lending nothing (but keeping the moves made), when a run is lent already,
    /// the joined chunks at the top are fewer pages, or a block has no room.
    PoolLend lend(uint32_t pages, uint64_t keep_free);
    /// Tells whether `lend(pages, ...)` may succeed: the joined chunks at the top hold them and the conversations'
    /// blocks there fit elsewhere (a fixed region may still find no run long enough).
    bool can_lend(uint32_t pages) const;
    /// Frees the lent run; experts may take its pages at once.
    void give_back();
    /// Gets the pages of the lent run (0: none).
    uint32_t lent_pages() const { return lent_pages_; }
    /// Changes the token limits; returns why not (changing nothing) or an empty string.
    std::string set_token_limits(int64_t min, int64_t max);
    /// Gives the conversations the regions of `layout` from now on, while the pool holds none (strata-q38 moves a
    /// layer split's cut, which changes the layers each card keeps state for).  Returns why not (changing nothing):
    /// it holds conversations, a fixed region or a block is larger than a chunk, or the token minimum does not fit.
    std::string set_layout(const StateLayout& layout);
    /// Replaces PagePoolConfig::expert_ranking, which places the expert sizes' home points (ExpertPages::set_ranking).
    void set_expert_ranking(std::vector<uint32_t> ranking);

    /// Gives conversation `id` (added if new) the pages for `tokens` tokens; fewer tokens than it has keep its
    /// pages.  Returns why not (changing nothing) or an empty string.
    std::string grow_conversation(ConversationId id, int64_t tokens);
    /// Frees the blocks of a conversation's growing regions past its first `tokens` tokens (a new prompt that
    /// shares only that much with it); its fixed regions keep their pages, so their addresses stay the same.
    void truncate_conversation(ConversationId id, int64_t tokens);
    /// Marks a conversation paused (it may not grow, and a resize or the limits need not fit it) or active again.
    /// Its pages stay until remove_conversation.
    void set_paused(ConversationId id, bool paused);
    /// Frees a conversation's pages; experts may take them after the hysteresis.
    void remove_conversation(ConversationId id);
    /// Gets the first page of each block of one region of a conversation, in order; a block is block_bytes(region)
    /// of consecutive pages.
    const std::vector<PageId>& conversation_blocks(ConversationId id, size_t region) const;

    /// Gets the first of an expert's `pages` consecutive pages: the ones it has, the free run nearest its size's home
    /// point (open to experts, and above the token minimum's reserve; in the reserve's pages only past the reserve's
    /// count and when no free page waits to open), or else, `may_evict`, the run of the least-used expert of its size,
    /// when that was last used before `used_at`.  `used_at` is also the new expert's last use, so that experts
    /// added from a ranking (hottest first) can be given earlier times for colder ones.  no_page when there is no
    /// room.
    PageId add_expert(ExpertKey key, uint32_t pages, uint64_t used_at = clock_now, bool may_evict = true);
    /// Counts a use of an expert for eviction and gets its first page; no_page when the pool does not hold it.
    PageId use_expert(ExpertKey key);
    /// Gets an expert's first page without counting a use; no_page when the pool does not hold it.
    PageId find_expert(ExpertKey key) const;
    /// Frees an expert's pages at its owner's request (without calling `evicted`).
    void remove_expert(ExpertKey key);
    /// Gives the pages of expert `out` to expert `in`, which must take as many pages and not be in the pool, at its
    /// owner's request (without calling `evicted`); `in`'s use counts from now.  Gets the run's first page, or
    /// no_page (changing nothing) when the pool does not hold `out` or already holds `in`.
    PageId replace_expert(ExpertKey out, ExpertKey in);
    /// Gets the earliest clock time at which a free page that is still closed to experts opens to them; 0 when no
    /// page is closed.
    uint64_t get_next_opening() const;
    /// Gets the page experts of `pages` pages gather around: the middle slot of the room the ranking gives their size
    /// (the middle of the experts' part of the pool for a size the ranking lacks).
    PageId get_home(uint32_t pages) const;

    /// Gets where a page starts in the pool's memory (a device address for a GPU's pool).
    uint8_t* page_address(PageId page) const;
    /// Gets the size of each page.
    uint64_t page_bytes() const { return config_.page_bytes; }
    /// Gets the size of each chunk: the steps in which resize takes and gives back memory.
    uint64_t chunk_bytes() const { return config_.chunk_bytes; }
    /// Gets the token limits (set_token_limits).
    int64_t tokens_min() const { return config_.tokens_min; }
    int64_t tokens_max() const { return config_.tokens_max; }
    /// Gets the bytes of one block of a region: its consecutive pages (all of a fixed region's).
    uint64_t block_bytes(size_t region) const { return block_pages_[region] * config_.page_bytes; }
    /// Gets the regions every conversation of the pool has.
    const StateLayout& layout() const { return layout_; }
    /// Gets the pages one conversation of `tokens` tokens takes.
    uint64_t pages_for(int64_t tokens) const;
    PagePoolReport report() const;
    /// Gets disjoint block totals with their category, pool and purpose, including unused pool space.
    std::vector<MemoryBlock> get_memory_blocks() const;
    /// Names a conversation's blocks; a missing name is shown as its numeric handle.
    void set_conversation_label(ConversationId id, std::string label);

private:
    /// Free runs of consecutive pages inside a hole, each as long as it can be.
    struct FreeRuns {
        std::map<PageId, uint32_t> runs;   ///< first page -> pages
        uint64_t pages = 0;
        PageId take(uint32_t n, PageId end);
        void give(PageId first, uint32_t n);
    };
    enum class RunUse : uint8_t { free, expert, conversation, hole, lent };
    /// Consecutive pages with one use, inside one chunk or joined ones.  A free run is as long as it can be: a run released next to
    /// a free one joins it.
    struct Run {
        uint32_t pages = 0;
        RunUse use = RunUse::free;
        uint64_t open_at = 0;              ///< free: when experts may take it (the latest of the runs it joined)
        uint64_t last_use = 0, uses = 0;   ///< expert
        ExpertKey expert;
        ConversationId owner = 0;          ///< conversation, hole
        FreeRuns room;                     ///< hole: the pages no block holds
    };
    using Runs = std::map<PageId, Run>;   ///< by first page
    using UseOrder = std::set<std::tuple<uint64_t, uint64_t, PageId>>;   ///< experts by (last use, uses, first page)
    struct Conversation {
        std::string label;
        int64_t tokens = 0;
        bool paused = false;
        std::vector<std::vector<PageId>> blocks;   ///< per region of the layout, each block's first page
        std::set<PageId> holes;                    ///< its holes with room for its smallest block
    };

    bool is_fixed(size_t region) const { return layout_.regions[region].kind == RegionKind::fixed; }
    /// Tells whether `page` begins a chunk that is not joined to the one before, where every run must begin.
    bool is_chunk_start(PageId page) const {
        const size_t c = page / pages_per_chunk_;
        return page % pages_per_chunk_ == 0 && (c >= joined_.size() || !joined_[c]);
    }
    PageId page_count() const { return (PageId) (chunks_.size() * pages_per_chunk_); }
    PageId get_top_stretch() const;
    std::string place_blocks();
    void place_homes();
    PoolResize shrink(size_t chunks);
    std::string check_limits(uint64_t pages, int64_t min, int64_t max) const;
    std::string check_growth(const Conversation& c, int64_t tokens) const;
    uint64_t region_blocks(size_t region, int64_t tokens) const;
    PageId get_minimum_pages() const;
    uint64_t reserved_pages() const;
    void open_due();
    bool is_open(const Run& r) const { return r.open_at <= now_; }
    PageId find_expert_place(uint32_t pages, uint64_t keep) const;
    PageId find_near(uint32_t pages, PageId home, PageId floor) const;
    PageId find_lowest(uint32_t pages, PageId end) const;
    PageId find_stretch(uint32_t pages, PageId end) const;
    PageId place_block(ConversationId id, uint32_t pages, PageId end);
    PageId place_fixed(ConversationId id, uint32_t pages, PageId end);
    void release_block(PageId first, uint32_t pages, uint64_t open_at);
    void drop_blocks(Conversation& c, const std::vector<size_t>& keep, uint64_t open_at);
    PageId take(PageId first, Run content);
    void evict(PageId first, uint64_t open_at);
    void free_expert(PageId first, uint64_t open_at);
    void list_expert(const Run& r, PageId first);
    void unlist_expert(const Run& r, PageId first);
    void release(PageId first, uint64_t open_at);
    void claim_free(PageId from, PageId end);
    void release_lent(PageId from, PageId end);
    bool is_lending(PageId first) const;
    /// Gets the pages conversations' blocks hold in run `r` (a block's run, or a hole's used pages).
    static uint32_t get_block_pages(const Run& r) { return r.use == RunUse::hole ? r.pages - r.room.pages : r.pages; }
    void index_free(Runs::iterator it);
    void unindex_free(Runs::iterator it);
    Runs::iterator get_run_at(PageId page);
    std::string gib(uint64_t pages) const;

    StateLayout layout_;
    PagePoolConfig config_;
    PagePoolHooks hooks_;
    const uint64_t pages_per_chunk_;
    std::vector<uint32_t> block_pages_;      ///< per region, the pages of one block
    uint32_t smallest_block_ = UINT32_MAX;   ///< the pages of the smallest block of a growing region
    uint32_t smallest_expert_ = 0;           ///< the pages of the ranking's smallest expert (0: no ranking)
    std::vector<void*> chunks_;
    std::vector<bool> joined_;   ///< per chunk: it starts where the previous one ended
    PageId lent_first_ = no_page;
    uint32_t lent_pages_ = 0;
    /// While `lend` clears a run: its first page, so that a block moved out takes no hole of the run (no_page:
    /// not lending)
    PageId lending_from_ = no_page;
    Runs runs_;   ///< tiles every chunk
    std::set<PageId> free_runs_;                          ///< every free run's first page
    std::set<std::pair<uint32_t, PageId>> open_runs_;     ///< the free runs open to experts, by (pages, first page)
    std::set<std::pair<uint64_t, PageId>> opening_;       ///< the closed ones by (open_at, first page)
    uint64_t now_ = 0;   ///< the clock when the current call began (open_due)
    uint64_t free_pages_ = 0;
    uint64_t fragmented_pages_ = 0;   ///< free pages in runs shorter than smallest_expert_
    std::map<uint32_t, PageId> homes_;           ///< expert size in pages -> its home point
    std::map<ExpertKey, PageId> experts_;        ///< each expert's first page
    UseOrder by_use_;                            ///< every expert, least used first
    std::map<uint32_t, UseOrder> size_by_use_;   ///< per expert size, its experts, least used first
    uint64_t expert_pages_ = 0;
    std::map<ConversationId, Conversation> conversations_;
};

}  // namespace strata::core
