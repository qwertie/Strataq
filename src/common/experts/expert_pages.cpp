// src/common/experts/expert_pages.cpp - a GPU's expert cache in the pages of its page pool (expert_pages.hpp).
#include "strata/common/experts/expert_pages.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {

/// The allocations a chunk of a cache's pool may take before one lands above the cache's arena.
constexpr int kChunkTries = 4;

}  // namespace

bool ExpertPages::open(ExpertCache& cache, ChunkMemory memory, uint64_t bytes, const StateLayout& layout,
                       PagePoolConfig config, const std::vector<uint64_t>& blob_bytes, int64_t n_expert,
                       int64_t max_experts, const std::vector<ExpertKey>& ranking, std::string& err) {
    const uint64_t page = (uint64_t) cache.slot_bytes(), pages = (uint64_t) cache.slots();
    if (!cache.valid() || cache.slot_offsets() != nullptr || cache.resident() != 0 || page == 0) {
        err = "the expert pages need an open, empty cache of uniform slots";
        return false;
    }
    config.page_bytes = page;
    config.chunk_bytes = pages * page;
    cache_ = &cache;
    arena_ = cache.device_slot(0);
    if (!open_pool(std::move(memory), std::max(bytes, config.chunk_bytes), layout, config, blob_bytes, n_expert,
                   max_experts, ranking, err))
        return false;
    cache.set_admitter([this](int64_t layer, int64_t expert) { return admit(layer, expert); });
    want_missing();
    return true;
}

bool ExpertPages::open(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
                       const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
                       const std::vector<ExpertKey>& ranking, std::string& err) {
    return open_pool(std::move(memory), bytes, layout, config, blob_bytes, n_expert, max_experts, ranking, err);
}

PoolResize ExpertPages::resize(uint64_t bytes) {
    const uint64_t had = pool_->report().total.pages;
    PoolResize r = pool_->resize(bytes);
    if (pool_->report().total.pages > had)   // the experts that did not fit before may come in now
        want_missing();
    return r;
}

void ExpertPages::set_ranking(const std::vector<ExpertKey>& ranking, int64_t max_experts) {
    std::vector<int32_t> rank(rank_.size(), -1);
    std::vector<uint32_t> sizes;
    for (size_t i = 0; i < ranking.size(); ++i) {
        rank.at(index_of(ranking[i])) = (int32_t) i;
        if ((int64_t) i < max_experts)
            sizes.push_back(pages_.at((size_t) ranking[i].layer));
    }
    for (size_t i = 0; i < rank.size(); ++i) {   // the experts it holds that the new ranking leaves out leave
        const ExpertKey key{(int32_t) (i / (size_t) n_expert_), (int32_t) (i % (size_t) n_expert_)};
        if (const PageId first = pool_->find_expert(key);
            first != no_page && (rank[i] < 0 || rank[i] >= max_experts)) {
            on_evicted(key, first);
            pool_->remove_expert(key);
        }
    }
    rank_ = std::move(rank);
    ranked_ = ranking;
    max_experts_ = max_experts;
    evicted_.clear();   // (the old ranks on_evicted kept: want_missing finds the new ones)
    pool_->set_expert_ranking(std::move(sizes));
    want_missing();
}

/// Marks the experts of the first max_experts_ ranks that the pool does not hold for `refill` to bring in.
void ExpertPages::want_missing() {
    for (int32_t rank = 0; rank < (int32_t) ranked_.size() && rank < max_experts_; ++rank)
        if (pool_->find_expert(ranked_[(size_t) rank]) == no_page)
            evicted_.insert(rank);
    blocked_at_free_ = UINT64_MAX;
}

uint8_t* ExpertPages::lend(uint64_t bytes, PoolLend& lend) {
    const uint64_t page = pool_->page_bytes();
    lend = pool_->lend((uint32_t) ((bytes + page - 1) / page), 0);
    return lend.first == no_page ? nullptr : pool_->page_address(lend.first);
}

bool ExpertPages::can_lend(uint64_t bytes) const {
    const uint64_t page = pool_->page_bytes();
    return pool_->can_lend((uint32_t) ((bytes + page - 1) / page));
}

int32_t ExpertPages::get_slot(PageId page) const {
    if (cache_ == nullptr)
        return (int32_t) page;
    return (int32_t) ((uint64_t) (pool_->page_address(page) - arena_) / pool_->page_bytes());
}

/// Makes the pool (open's shared part) with the conversations' `layout` and `config`'s limits, in chunks of `memory`
/// (the cache's arena first) up to `bytes`.
bool ExpertPages::open_pool(ChunkMemory memory, uint64_t bytes, const StateLayout& layout, PagePoolConfig config,
                            const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t max_experts,
                            const std::vector<ExpertKey>& ranking, std::string& err) {
    const uint64_t page = config.page_bytes;
    config.expert_bytes = blob_bytes;
    config.join_chunks = memory.contiguous;
    memory_ = std::move(memory);
    n_expert_ = n_expert;
    max_experts_ = max_experts;
    pages_.clear();
    for (const uint64_t b : blob_bytes)
        pages_.push_back((uint32_t) ((b + page - 1) / page));
    rank_.assign(blob_bytes.size() * (size_t) n_expert, -1);
    ranked_ = ranking;
    config.expert_ranking.clear();
    for (size_t i = 0; i < ranking.size(); ++i) {
        rank_.at(index_of(ranking[i])) = (int32_t) i;
        if ((int64_t) i < max_experts)
            config.expert_ranking.push_back(pages_.at((size_t) ranking[i].layer));
    }
    PagePoolHooks hooks;
    hooks.alloc = [this](size_t n) -> void* { return alloc_chunk(n); };
    hooks.free = [this](void* chunk) { free_chunk((uint8_t*) chunk); };
    hooks.evicted = [this](ExpertKey key, PageId first) { on_evicted(key, first); };
    hooks.copy = [this, page](PageId to, PageId from, uint32_t pages) {
        return memory_.copy(pool_->page_address(to), pool_->page_address(from), pages * page);
    };
    hooks.relocated = [this](ExpertKey key, PageId to) { on_relocated(key, to); };
    now_ = hooks.now;
    t0_ = std::max<uint64_t>(now_(), rank_.size() + 1);
    try {
        pool_ = std::make_unique<PagePool>(layout, config, std::move(hooks));
    } catch (const std::exception& e) {
        err = std::string("the page pool: ") + e.what();
        return false;
    }
    if (const PoolResize r = pool_->resize(bytes); !r.refusal.empty() || pool_->report().total.pages == 0) {
        err = "the page pool: " + (r.refusal.empty() ? std::string("no room for a chunk") : r.refusal);
        return false;
    }
    return true;
}

bool ExpertPages::is_refill_wanted() const {
    return !evicted_.empty() && pool_->report().free.pages != blocked_at_free_;
}

void ExpertPages::stop_wanting_if(const std::function<bool(ExpertKey)>& is_held_elsewhere) {
    std::erase_if(evicted_, [&](int32_t rank) { return is_held_elsewhere(ranked_[(size_t) rank]); });
}

int ExpertPages::refill(int max, uint64_t& wait_ms, std::string& err) {
    wait_ms = 0;
    int loaded = 0;
    while (loaded < max && !evicted_.empty()) {
        const ExpertKey key = ranked_[(size_t) *evicted_.begin()];
        const int32_t slot = admit_free(key);
        if (slot == kNotResident) {
            const uint64_t opening = pool_->get_next_opening();
            if (opening > 0)
                wait_ms = opening - std::min(opening, now_());
            else
                blocked_at_free_ = pool_->report().free.pages;
            break;
        }
        if (!hooks_.load(key, slot, err))
            return -1;
        hooks_.moved(key, slot);
        ++loaded;
    }
    return loaded;
}

int32_t ExpertPages::admit(int64_t layer, int64_t expert, uint64_t used_at) {
    const ExpertKey key{(int32_t) layer, (int32_t) expert};
    int32_t& rank = rank_[index_of(key)];
    const int32_t r = rank >= 0 ? rank : (int32_t) ranked_.size();   // an expert outside the ranking ranks after it
    if (r >= max_experts_)
        return kNotResident;
    const PageId page =
        pool_->add_expert(key, pages_[(size_t) layer], used_at != 0 ? used_at : get_used_at(r), !free_only_);
    if (page == no_page)
        return kNotResident;
    if (rank < 0) {
        rank = r;
        ranked_.push_back(key);
    }
    evicted_.erase(r);
    blocked_at_free_ = UINT64_MAX;
    return get_slot(page);
}

int ExpertPages::replace(int32_t layer, int32_t out, int32_t in, std::string& err) {
    const ExpertKey gone{layer, out}, key{layer, in};
    if (pool_->find_expert(key) != no_page)
        return 0;
    if (const int32_t free_slot = admit_free(key); free_slot != kNotResident) {
        if (!hooks_.load(key, free_slot, err)) {
            pool_->remove_expert(key);
            if (cache_ != nullptr)
                cache_->evict(layer, in);
            return -1;
        }
        hooks_.moved(key, free_slot);
        note_use(key);
        return 1;
    }
    if (out < 0)
        return 0;
    const PageId first = pool_->replace_expert(gone, key);
    if (first == no_page)
        return 0;
    if (hooks_.leaving)
        hooks_.leaving(gone, pool_->page_address(first));
    hooks_.moved(gone, kNotResident);
    if (int32_t& rank = rank_[index_of(key)]; rank >= 0) {
        evicted_.erase(rank);
    } else {   // ranked after the ranking, as admit ranks an expert outside it
        rank = (int32_t) ranked_.size();
        ranked_.push_back(key);
    }
    const int32_t slot = get_slot(first);
    if (!hooks_.load(key, slot, err)) {
        pool_->remove_expert(key);
        if (cache_ != nullptr)
            cache_->evict(layer, out);
        return -1;
    }
    if (cache_ != nullptr)
        cache_->replace(layer, out, in);
    hooks_.moved(key, slot);
    return 1;
}

/// Brings `key` into free pages without evicting anything, for `refill` and `replace`: through the cache when there
/// is one, so that the cache records the slot.  Gets the slot of its first page, or kNotResident.
int32_t ExpertPages::admit_free(ExpertKey key) {
    free_only_ = true;
    const int32_t slot = cache_ != nullptr ? cache_->admit(key.layer, key.expert) : admit(key.layer, key.expert);
    free_only_ = false;
    return slot;
}

/// Gets a chunk of `bytes` for the pool (its alloc hook): the cache's arena first, then memory_'s.  A cache's chunk
/// must start above the arena at a whole number of pages from it, as the kernels find a page at the arena plus its
/// slot times the page size; so it is allocated a page larger and starts at the first such place in it.  An
/// allocation below the arena is kept while the next one is tried (so that the allocator gives other memory) and
/// given back after.
uint8_t* ExpertPages::alloc_chunk(uint64_t bytes) {
    if (cache_ != nullptr && raw_.empty() && (pool_ == nullptr || pool_->report().total.pages == 0))
        return arena_;
    const uint64_t page = pool_->page_bytes(), extra = cache_ != nullptr && !memory_.contiguous ? page : 0;
    std::vector<uint8_t*> below;
    uint8_t* chunk = nullptr;
    for (int tries = 0; chunk == nullptr && tries < kChunkTries; ++tries) {
        uint8_t* raw = memory_.alloc(bytes + extra);
        if (raw == nullptr)
            break;
        if (cache_ == nullptr) {
            chunk = raw;
        } else if (raw < arena_ || (uint64_t) (raw - arena_ + bytes) / page + 1 > (uint64_t) INT32_MAX) {
            below.push_back(raw);
            continue;
        } else {
            chunk = raw + (page - (uint64_t) (raw - arena_) % page) % page;
        }
        raw_[chunk] = raw;
    }
    for (uint8_t* raw : below)
        memory_.free(raw);
    if (chunk != nullptr && cache_ != nullptr) {
        const auto end = (int64_t) ((uint64_t) (chunk - arena_ + bytes) / page);
        cache_->set_slot_reach(std::max(cache_->get_reach(), end));
    }
    return chunk;
}

/// Gives back a chunk alloc_chunk gave (the cache's arena stays: the cache owns it).
void ExpertPages::free_chunk(uint8_t* chunk) {
    if (const auto it = raw_.find(chunk); it != raw_.end()) {
        memory_.free(it->second);
        raw_.erase(it);
    }
}

/// Tells the cache and the engine that a lend copied an expert to the pages from `to`.
void ExpertPages::on_relocated(ExpertKey key, PageId to) {
    const int32_t slot = get_slot(to);
    if (cache_ != nullptr)
        cache_->relocate(key.layer, key.expert, slot);
    if (hooks_.moved)
        hooks_.moved(key, slot);
}

/// Forgets an expert whose pages (from `first`) the pool gives to another use, and keeps it for `refill`.
void ExpertPages::on_evicted(ExpertKey key, PageId first) {
    if (hooks_.leaving)
        hooks_.leaving(key, pool_->page_address(first));
    if (cache_ != nullptr)
        cache_->evict(key.layer, key.expert);
    if (const int32_t rank = rank_[index_of(key)]; rank < max_experts_)   // (admit refuses the others)
        evicted_.insert(rank);
    if (hooks_.moved)
        hooks_.moved(key, kNotResident);
}

}  // namespace strata::core
