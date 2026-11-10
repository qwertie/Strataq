// src/common/state/page_pool.cpp - the page pool (page_pool.hpp).  It keeps runs of pages, not pages: a pool may hold
// millions of small pages, and a call costs time in the runs it looks at, which ordered indexes keep few.
#include "strata/common/state/page_pool.hpp"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace strata::core {

uint64_t choose_expert_page_bytes(const std::vector<uint64_t>& blob_bytes) {
    constexpr uint64_t align = 256, smallest = 16 << 10, largest = 512 << 10;
    uint64_t gcd = 0;
    for (const uint64_t b : blob_bytes)
        gcd = std::gcd(gcd, b);
    if (gcd % align == 0 && gcd >= smallest)
        return gcd;
    uint64_t best = smallest, best_used = UINT64_MAX;
    for (uint64_t page = smallest; page <= largest; page += align) {
        uint64_t used = 0;
        for (const uint64_t b : blob_bytes)
            used += (b + page - 1) / page * page;
        if (used < best_used) {
            best = page;
            best_used = used;
        }
    }
    return best;
}

uint32_t choose_block_pages(const StateRegion& region, uint64_t page_bytes) {
    if (region.kind == RegionKind::fixed)
        return (uint32_t) ((region.bytes_for(0) + page_bytes - 1) / page_bytes);
    const uint64_t unit = region.rows * region.unit_bytes;
    if (unit == 0)
        return 1;
    const uint64_t most = (std::max<uint64_t>(1ull << 20, unit) + page_bytes - 1) / page_bytes;
    uint32_t best = 0;
    uint64_t best_waste = 0, best_bytes = 1;
    for (uint64_t k = (unit + page_bytes - 1) / page_bytes; k <= most; ++k) {
        const uint64_t bytes = k * page_bytes, waste = bytes % unit;
        if (waste * 100 <= bytes)
            return (uint32_t) k;
        if (best == 0 || waste * best_bytes < best_waste * bytes) {
            best = (uint32_t) k;
            best_waste = waste;
            best_bytes = bytes;
        }
    }
    return best;
}

PagePool::PagePool(const StateLayout& layout, PagePoolConfig config, PagePoolHooks hooks)
    : layout_(layout), config_(std::move(config)), hooks_(std::move(hooks)),
      pages_per_chunk_(config_.chunk_bytes / config_.page_bytes) {
    if (pages_per_chunk_ == 0)
        throw std::invalid_argument("a chunk of the page pool holds no page");
    if (const std::string why = place_blocks(); !why.empty())
        throw std::invalid_argument(why);
    if (!config_.expert_ranking.empty())
        smallest_expert_ = *std::min_element(config_.expert_ranking.begin(), config_.expert_ranking.end());
}

std::string PagePool::set_layout(const StateLayout& layout) {
    if (!conversations_.empty())
        return "the pool holds conversations";
    const StateLayout old = std::exchange(layout_, layout);
    std::string why = place_blocks();
    if (why.empty())
        why = check_limits(page_count(), config_.tokens_min, config_.tokens_max);
    if (!why.empty()) {
        layout_ = old;
        (void) place_blocks();
        return why;
    }
    place_homes();
    return {};
}

void PagePool::set_expert_ranking(std::vector<uint32_t> ranking) {
    config_.expert_ranking = std::move(ranking);
    smallest_expert_ = config_.expert_ranking.empty()
                           ? 0
                           : *std::min_element(config_.expert_ranking.begin(), config_.expert_ranking.end());
    fragmented_pages_ = 0;
    for (const PageId first : free_runs_)
        if (const uint32_t pages = runs_.at(first).pages; pages < smallest_expert_)
            fragmented_pages_ += pages;
    place_homes();
}

/// Sets block_pages_ and smallest_block_ from the layout; gets why a block does not fit a chunk, or an empty string.
std::string PagePool::place_blocks() {
    block_pages_.clear();
    smallest_block_ = UINT32_MAX;
    for (size_t r = 0; r < layout_.regions.size(); ++r) {
        block_pages_.push_back(choose_block_pages(layout_.regions[r], config_.page_bytes));
        if (block_pages_.back() > pages_per_chunk_)
            return is_fixed(r) ? "a fixed region of a conversation is larger than a chunk of the page pool"
                               : "a block of a growing region is larger than a chunk of the page pool";
        if (!is_fixed(r))
            smallest_block_ = std::min(smallest_block_, block_pages_.back());
    }
    return {};
}

PagePool::~PagePool() {
    for (void* chunk : chunks_)
        hooks_.free(chunk);
}

PoolResize PagePool::resize(uint64_t bytes) {
    open_due();
    const size_t chunks = bytes / config_.chunk_bytes;
    if (chunks < chunks_.size())
        return shrink(chunks);
    std::string refusal;
    while (chunks_.size() < chunks) {
        void* chunk = hooks_.alloc(config_.chunk_bytes);
        if (chunk == nullptr) {
            refusal = "the memory had room for " + std::to_string(chunks_.size()) + " of " +
                      std::to_string(chunks) + " chunks";
            break;
        }
        get_memory_manager().mark_pool_backing(chunk, config_.chunk_bytes, config_.label);
        const PageId first = page_count();
        joined_.push_back(config_.join_chunks && !chunks_.empty() &&
                          chunk == (uint8_t*) chunks_.back() + config_.chunk_bytes);
        chunks_.push_back(chunk);
        runs_[first].pages = (uint32_t) pages_per_chunk_;
        runs_[first].use = RunUse::lent;   // (released at once: it joins a free run that ends where it starts)
        release(first, 0);
    }
    place_homes();
    return {refusal, {}};
}

std::string PagePool::set_token_limits(int64_t min, int64_t max) {
    std::string why = check_limits(page_count(), min, max);
    if (why.empty()) {
        config_.tokens_min = min;
        config_.tokens_max = max;
        place_homes();
    }
    return why;
}

std::string PagePool::grow_conversation(ConversationId id, int64_t tokens) {
    open_due();
    auto [it, added] = conversations_.try_emplace(id);
    Conversation& c = it->second;
    if (added)
        c.blocks.resize(layout_.regions.size());
    std::string why = check_growth(c, tokens);
    std::vector<size_t> had;
    for (const std::vector<PageId>& blocks : c.blocks)
        had.push_back(blocks.size());
    for (const bool fixed : {true, false})   // a fixed region first, while the pool has the most room in one piece
        for (uint32_t r = 0; why.empty() && r < c.blocks.size(); ++r)
            while (why.empty() && is_fixed(r) == fixed && c.blocks[r].size() < region_blocks(r, tokens)) {
                const PageId first = fixed ? place_fixed(id, block_pages_[r], no_page)
                                           : place_block(id, block_pages_[r], no_page);
                if (first != no_page)
                    c.blocks[r].push_back(first);
                else
                    why = fixed ? "the pool has no room in one piece for the conversation's fixed state"
                                : "the pool has no room for a block of the conversation's tokens";
            }
    if (!why.empty()) {
        drop_blocks(c, had, now_ + config_.hysteresis);
        if (added)
            conversations_.erase(id);
        return why;
    }
    c.tokens = std::max(c.tokens, tokens);
    return {};
}

void PagePool::truncate_conversation(ConversationId id, int64_t tokens) {
    open_due();
    Conversation& c = conversations_.at(id);
    std::vector<size_t> keep;
    for (size_t r = 0; r < c.blocks.size(); ++r)
        keep.push_back(is_fixed(r) ? c.blocks[r].size()
                                   : std::min<size_t>(c.blocks[r].size(), region_blocks(r, tokens)));
    drop_blocks(c, keep, now_ + config_.hysteresis);
    c.tokens = std::min(c.tokens, tokens);
}

void PagePool::set_paused(ConversationId id, bool paused) {
    conversations_.at(id).paused = paused;
}

void PagePool::remove_conversation(ConversationId id) {
    open_due();
    Conversation& c = conversations_.at(id);
    drop_blocks(c, std::vector<size_t>(c.blocks.size(), 0), now_ + config_.hysteresis);
    conversations_.erase(id);
}

const std::vector<PageId>& PagePool::conversation_blocks(ConversationId id, size_t region) const {
    return conversations_.at(id).blocks.at(region);
}

PageId PagePool::add_expert(ExpertKey key, uint32_t pages, uint64_t used_at, bool may_evict) {
    if (const PageId held = use_expert(key); held != no_page)
        return held;
    if (used_at == clock_now)
        used_at = now_;
    if (pages == 0 || pages > pages_per_chunk_)
        return no_page;
    PageId first = find_expert_place(pages, reserved_pages());
    if (const auto same = size_by_use_.find(pages); first == no_page && may_evict && same != size_by_use_.end() &&
                                                    !same->second.empty() &&
                                                    std::get<0>(*same->second.begin()) < used_at) {
        first = std::get<2>(*same->second.begin());
        evict(first, 0);
    }
    if (first == no_page)
        return no_page;
    Run content;
    content.pages = pages;
    content.use = RunUse::expert;
    content.expert = key;
    content.last_use = used_at;
    content.uses = 1;
    list_expert(runs_.at(take(first, std::move(content))), first);
    return first;
}

PageId PagePool::use_expert(ExpertKey key) {
    open_due();
    const auto it = experts_.find(key);
    if (it == experts_.end())
        return no_page;
    const PageId first = it->second;
    Run& r = runs_.at(first);
    const auto before = std::make_tuple(r.last_use, r.uses, first);
    r.last_use = now_;
    ++r.uses;
    const auto after = std::make_tuple(r.last_use, r.uses, first);
    for (UseOrder* order : {&by_use_, &size_by_use_.at(r.pages)}) {
        auto node = order->extract(before);
        node.value() = after;
        order->insert(std::move(node));
    }
    return first;
}

PageId PagePool::find_expert(ExpertKey key) const {
    const auto it = experts_.find(key);
    return it == experts_.end() ? no_page : it->second;
}

void PagePool::remove_expert(ExpertKey key) {
    open_due();
    if (const auto it = experts_.find(key); it != experts_.end())
        free_expert(it->second, 0);
}

PageId PagePool::replace_expert(ExpertKey out, ExpertKey in) {
    open_due();
    const auto it = experts_.find(out);
    if (it == experts_.end() || experts_.count(in) != 0)
        return no_page;
    const PageId first = it->second;
    Run& r = runs_.at(first);
    unlist_expert(r, first);
    r.expert = in;
    r.last_use = now_;
    r.uses = 1;
    list_expert(r, first);
    return first;
}

uint64_t PagePool::get_next_opening() const {
    const auto it = opening_.upper_bound({hooks_.now(), no_page});
    return it == opening_.end() ? 0 : it->first;
}

PageId PagePool::get_home(uint32_t pages) const {
    if (const auto it = homes_.find(pages); it != homes_.end())
        return it->second;
    const PageId reserve = get_minimum_pages();
    return reserve + (page_count() - reserve) / 2;
}

uint8_t* PagePool::page_address(PageId page) const {
    return (uint8_t*) chunks_[page / pages_per_chunk_] + page % pages_per_chunk_ * config_.page_bytes;
}

uint64_t PagePool::pages_for(int64_t tokens) const {
    uint64_t n = 0;
    for (size_t r = 0; r < layout_.regions.size(); ++r)
        n += region_blocks(r, tokens) * block_pages_[r];
    return n;
}

PagePoolReport PagePool::report() const {
    PagePoolReport out;
    uint64_t active = 0, fixed = 0, paused = 0;
    for (const auto& [id, c] : conversations_) {
        ++(c.paused ? out.paused_conversations : out.active_conversations);
        out.tokens += c.tokens;
        for (size_t r = 0; r < c.blocks.size(); ++r)
            (c.paused ? paused : is_fixed(r) ? fixed : active) += c.blocks[r].size() * block_pages_[r];
    }
    const auto use = [&](uint64_t pages) { return PagePoolReport::Use{pages, pages * config_.page_bytes}; };
    out.experts = use(expert_pages_);
    out.active = use(active);
    out.fixed = use(fixed);
    out.paused = use(paused);
    out.lent = use(lent_pages_);
    out.free = use(page_count() - expert_pages_ - active - fixed - paused - lent_pages_);
    out.total = use(page_count());
    out.allocated = use(page_count() - free_pages_);
    out.fragmented = use(fragmented_pages_);
    return out;
}
std::vector<MemoryBlock> PagePool::get_memory_blocks() const {
    std::vector<MemoryBlock> out;
    for (const auto& [page, run] : runs_) {
        if (run.use != RunUse::expert)
            continue;
        const uint64_t bytes = run.pages * config_.page_bytes;
        const uint64_t weight = (size_t) run.expert.layer < config_.expert_bytes.size() ?
            config_.expert_bytes[(size_t) run.expert.layer] : bytes;
        MemoryLabel name = config_.label_expert ? config_.label_expert(run.expert, weight) :
            MemoryLabel{"experts", config_.label, get_expert_memory_label(run.expert.layer, weight)};
        name.pool = config_.label;
        const std::string group = "expert/" + name.category + "/" + std::to_string(run.expert.layer);
        add_memory_block(out, std::move(name), bytes, group, {(uint32_t) run.expert.expert});
    }
    for (const auto& [id, c] : conversations_)
        for (size_t r = 0; r < c.blocks.size(); ++r) {
            const StateRegion& region = layout_.regions[r];
            const std::string label = (c.label.empty() ? "Conversation " + std::to_string(id) : c.label) + ": " +
                (region.label.empty() ? (is_fixed(r) ? "Fixed state" : "Context") : region.label);
            add_memory_block(out, {c.paused ? "paused" : "conversations", config_.label, label},
                             c.blocks[r].size() * block_pages_[r] * config_.page_bytes);
        }
    add_memory_block(out, {"prompt_buffers", config_.label, "Prompt buffers"}, lent_pages_ * config_.page_bytes);
    add_memory_block(out, {"pool_free", config_.label, "Unused pages"}, report().free.bytes);
    return out;
}
void PagePool::set_conversation_label(ConversationId id, std::string label) {
    conversations_.at(id).label = std::move(label);
}

bool PagePool::can_lend(uint32_t pages) const {
    const PageId end = page_count();
    if (lent_pages_ > 0 || pages == 0 || pages > end - get_top_stretch())
        return false;
    uint64_t blocks = 0, room = free_pages_ + expert_pages_;   // the pages the blocks there may go to
    for (auto it = std::prev(runs_.upper_bound(end - pages)); it != runs_.end(); ++it) {
        const Run& r = it->second;
        if (r.use == RunUse::free || r.use == RunUse::expert)
            room -= r.pages;
        else
            blocks += get_block_pages(r);
    }
    return blocks <= room;
}

PoolLend PagePool::lend(uint32_t pages, uint64_t keep_free) {
    open_due();
    PoolLend out;
    if (!can_lend(pages)) {
        out.refusal = lent_pages_ > 0 ? "a run is lent already"
                      : pages > page_count() - get_top_stretch()
                          ? "the pool's top holds " + gib(page_count() - get_top_stretch()) + " in one piece"
                          : "the conversations' blocks there have no room elsewhere";
        return out;
    }
    const PageId end = page_count(), from = end - pages;
    lending_from_ = from;
    claim_free(from, end);
    uint64_t blocks = 0;   // the conversations' pages there, which experts moving out leave room for
    std::vector<PageId> experts;
    for (auto it = get_run_at(from); it != runs_.end() && it->first < end; ++it)
        if (it->second.use == RunUse::expert)
            experts.push_back(it->first);
        else if (it->second.use != RunUse::lent)
            blocks += get_block_pages(it->second);
    for (const PageId first : experts) {
        const Run expert = runs_.at(first);
        if (const PageId to = find_expert_place(expert.pages, keep_free + blocks);
            to != no_page && hooks_.copy(to, first, expert.pages)) {
            unlist_expert(expert, first);
            list_expert(runs_.at(take(to, expert)), to);
            release(first, 0);
            hooks_.relocated(expert.expert, to);
            ++out.experts_moved;
        } else {
            evict(first, 0);
            ++out.experts_evicted;
        }
        claim_free(from, end);
    }
    for (auto& [id, c] : conversations_) {
        bool moved = false;
        for (uint32_t r = 0; r < c.blocks.size(); ++r)
            for (PageId& block : c.blocks[r]) {
                if (const auto run = get_run_at(block); run->first >= end || run->first + run->second.pages <= from)
                    continue;
                const PageId to = is_fixed(r) ? place_fixed(id, block_pages_[r], no_page)
                                              : place_block(id, block_pages_[r], no_page);
                if (to == no_page || !hooks_.copy(to, block, block_pages_[r])) {
                    if (to != no_page)
                        release_block(to, block_pages_[r], 0);
                    if (moved)
                        out.moved.push_back(id);
                    release_lent(from, end);
                    lending_from_ = no_page;
                    out.refusal = "a conversation's block there has no room elsewhere";
                    return out;
                }
                release_block(block, block_pages_[r], 0);
                block = to;
                moved = true;
                claim_free(from, end);
            }
        if (moved)
            out.moved.push_back(id);
    }
    lending_from_ = no_page;
    for (auto it = get_run_at(from); it != runs_.end() && it->first < end; ++it)
        if (it->second.use != RunUse::lent) {
            release_lent(from, end);
            out.refusal = "the run could not be cleared";
            return out;
        }
    for (auto it = runs_.lower_bound(from); it != runs_.end() && it->first < end;)
        it = runs_.erase(it);
    Run lent;
    lent.pages = pages;
    lent.use = RunUse::lent;
    runs_.emplace(from, lent);
    lent_first_ = out.first = from;
    lent_pages_ = pages;
    return out;
}

void PagePool::give_back() {
    if (lent_pages_ == 0)
        return;
    open_due();
    release(lent_first_, 0);
    lent_first_ = no_page;
    lent_pages_ = 0;
}

/// Places each expert size's home point: above the token minimum's reserve, the sizes in order (smallest lowest),
/// each given room for its experts among the ranking's hottest that fit, and its home at the slot in the middle of
/// that room, so that experts placed outwards from it tile the room exactly.
void PagePool::place_homes() {
    const PageId reserve = get_minimum_pages();
    std::map<uint32_t, uint64_t> held;   // size -> experts of the ranking's prefix that fits
    for (const uint32_t pages : config_.expert_ranking)
        held.try_emplace(pages, 0);
    uint64_t room = page_count() - reserve;
    for (const uint32_t pages : config_.expert_ranking) {
        if (pages > room)
            break;
        room -= pages;
        ++held[pages];
    }
    homes_.clear();
    PageId at = reserve;
    for (const auto& [pages, n] : held) {
        homes_[pages] = at + (PageId) (n / 2 * pages);
        at += (PageId) (n * pages);
    }
}

/// Gives back all chunks past the first `chunks`, after moving the conversation blocks there (the largest first) and
/// evicting the experts there.
PoolResize PagePool::shrink(size_t chunks) {
    const PageId end = (PageId) (chunks * pages_per_chunk_);
    if (lent_pages_ > 0)
        return {"a prompt's buffers are lent from the pool's top", {}};
    if (std::string why = check_limits(end, config_.tokens_min, config_.tokens_max); !why.empty())
        return {why, {}};
    uint64_t moving = 0, kept = 0;
    std::set<ConversationId> holders;
    // pages, conversation, region, index
    std::vector<std::tuple<uint32_t, ConversationId, uint32_t, uint32_t>> blocks;
    for (const auto& [id, c] : conversations_)
        for (uint32_t r = 0; r < c.blocks.size(); ++r)
            for (uint32_t i = 0; i < c.blocks[r].size(); ++i)
                if (const auto run = get_run_at(c.blocks[r][i]); run->first + run->second.pages > end) {
                    moving += block_pages_[r];
                    holders.insert(id);
                    blocks.emplace_back(block_pages_[r], id, r, i);
                } else
                    kept += block_pages_[r];
    if (moving > end - kept)
        return {"the conversations' pages need " + gib(moving - (end - kept)) + " more than the smaller pool keeps",
                {holders.begin(), holders.end()}};
    std::stable_sort(blocks.begin(), blocks.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    for (const auto& [pages, id, r, i] : blocks) {
        const PageId to = is_fixed(r) ? place_fixed(id, pages, end) : place_block(id, pages, end);
        if (to == no_page)
            return {"the conversations' blocks have no room in the smaller pool", {holders.begin(), holders.end()}};
        PageId& from = conversations_.at(id).blocks[r][i];
        if (!hooks_.copy(to, from, pages)) {
            release_block(to, pages, 0);
            return {"a conversation's page could not be copied", {}};
        }
        release_block(from, pages, 0);
        from = to;
    }
    std::vector<PageId> experts;
    for (auto it = get_run_at(end); it != runs_.end(); ++it)
        if (it->second.use == RunUse::expert)
            experts.push_back(it->first);
    for (const PageId first : experts)
        evict(first, 0);
    if (const auto across = get_run_at(end); across->first < end) {   // a free run across into a joined chunk
        unindex_free(across);
        free_pages_ -= across->first + across->second.pages - end;
        across->second.pages = end - across->first;
        index_free(across);
    }
    for (auto it = runs_.lower_bound(end); it != runs_.end(); it = runs_.erase(it)) {
        free_pages_ -= it->second.pages;
        unindex_free(it);
    }
    for (size_t c = chunks; c < chunks_.size(); ++c)
        hooks_.free(chunks_[c]);
    chunks_.resize(chunks);
    joined_.resize(chunks);
    place_homes();
    return {};
}

/// Gets why a pool of `pages` pages could not keep the token limits `min` and `max` and the active conversations,
/// or an empty string.
std::string PagePool::check_limits(uint64_t pages, int64_t min, int64_t max) const {
    if (min < 0 || min > max)
        return "the token minimum must be from 0 to the maximum";
    if (min > 0 && pages_for(min) > pages)
        return "the token minimum needs " + gib(pages_for(min)) + ", the pool has " + gib(pages);
    for (const auto& [id, c] : conversations_) {
        if (c.paused)
            continue;
        if (c.tokens > max)
            return "an active conversation holds " + std::to_string(c.tokens) + " tokens";
        if (pages_for(c.tokens) > pages)
            return "an active conversation needs " + gib(pages_for(c.tokens)) + ", the pool has " + gib(pages);
    }
    return {};
}

/// Gets why conversation `c` cannot grow to `tokens` tokens, or an empty string.
std::string PagePool::check_growth(const Conversation& c, int64_t tokens) const {
    if (c.paused)
        return "the conversation is paused";
    int64_t all = tokens - c.tokens;
    for (const auto& [id, other] : conversations_)
        all += other.tokens;
    if (tokens > c.tokens && all > config_.tokens_max)
        return "conversations would hold " + std::to_string(all) + " tokens, above the maximum of " +
               std::to_string(config_.tokens_max);
    uint64_t needed = 0;
    for (size_t r = 0; r < c.blocks.size(); ++r)
        needed += (std::max<uint64_t>(region_blocks(r, tokens), c.blocks[r].size()) - c.blocks[r].size()) *
                  block_pages_[r];
    const PagePoolReport now = report();
    const uint64_t room = now.free.pages + now.experts.pages;
    if (needed > room)
        return "the conversation needs " + gib(needed) + " more, the pool can free " + gib(room);
    return {};
}

/// Gets how many blocks region `region` of a conversation of `tokens` tokens takes: a fixed region one (none when
/// it is empty).
uint64_t PagePool::region_blocks(size_t region, int64_t tokens) const {
    if (is_fixed(region))
        return block_pages_[region] > 0 ? 1 : 0;
    const uint64_t per_block = (uint64_t) layout_.regions[region].tokens_in(block_bytes(region));
    return ((uint64_t) tokens + per_block - 1) / per_block;
}

/// Gets the pages of the token minimum's reserve (none without a minimum), at most the pool's.
PageId PagePool::get_minimum_pages() const {
    return config_.tokens_min == 0 ? 0 : (PageId) std::min<uint64_t>(page_count(), pages_for(config_.tokens_min));
}

/// Gets how many free pages experts must leave, so that the conversations can reach the token minimum.
uint64_t PagePool::reserved_pages() const {
    uint64_t held = 0;
    for (const auto& [id, c] : conversations_)
        for (size_t r = 0; r < c.blocks.size(); ++r)
            held += c.blocks[r].size() * block_pages_[r];
    return std::max<uint64_t>(get_minimum_pages(), held) - held;
}

/// Reads the clock for the call and opens to experts the free runs whose hysteresis has passed.
void PagePool::open_due() {
    now_ = hooks_.now();
    while (!opening_.empty() && opening_.begin()->first <= now_) {
        const PageId first = opening_.begin()->second;
        opening_.erase(opening_.begin());
        open_runs_.insert({runs_.at(first).pages, first});
    }
}

/// Gets where an expert of `pages` pages goes in free pages open to experts, leaving `keep` free pages: nearest its
/// size's home, above the token minimum's reserve; in the reserve (the conversations' end, overflow) only when no free
/// page above it is waiting to open, so that a closed run does not send an expert low.  no_page when there is no room.
PageId PagePool::find_expert_place(uint32_t pages, uint64_t keep) const {
    if (free_pages_ < keep + pages)
        return no_page;
    PageId first = find_near(pages, get_home(pages), get_minimum_pages());
    if (first == no_page && opening_.empty())
        first = find_near(pages, get_home(pages), 0);
    return first;
}

/// Gets the first page of the joined chunks at the pool's top: the most `lend` may lend.
PageId PagePool::get_top_stretch() const {
    size_t c = chunks_.size();
    while (c > 1 && joined_[c - 1])
        --c;
    return (PageId) ((c == 0 ? 0 : c - 1) * pages_per_chunk_);
}

/// Gets where `pages` pages for an expert whose size's home is `home` go, at or above page `floor`: in the open free
/// run nearest the home (searching outwards from it), at the place in it nearest the home; no_page when no open run
/// has that much room above `floor`.
PageId PagePool::find_near(uint32_t pages, PageId home, PageId floor) const {
    if (open_runs_.lower_bound({pages, 0}) == open_runs_.end())
        return no_page;
    auto up = free_runs_.lower_bound(home);
    auto down = std::make_reverse_iterator(up);
    for (;;) {
        const PageId below_end = down != free_runs_.rend() ? *down + runs_.at(*down).pages : 0;
        if (below_end < floor + pages)   // neither that run nor any below it has room above the floor
            down = free_runs_.rend();
        if (up == free_runs_.end() && down == free_runs_.rend())
            return no_page;
        const bool upwards = down == free_runs_.rend() ||
                             (up != free_runs_.end() && *up - home < home - std::min(home, below_end));
        const PageId first = upwards ? *up++ : *down++;
        const Run& r = runs_.at(first);
        if (const PageId low = std::max(first, floor); is_open(r) && first + r.pages >= low + pages)
            return std::clamp<PageId>(home, low, first + r.pages - pages);
    }
}

/// Gets the lowest free run before `end` (open to experts or not) that holds `pages` pages; no_page when there is
/// none.
PageId PagePool::find_lowest(uint32_t pages, PageId end) const {
    for (const PageId first : free_runs_) {
        if (first >= end)
            break;
        if (runs_.at(first).pages >= pages && (uint64_t) first + pages <= end)
            return first;
    }
    return no_page;
}

/// Gets the lowest place before `end` where `pages` consecutive pages inside one chunk (or joined ones) are free or
/// hold experts; no_page when there is none.
PageId PagePool::find_stretch(uint32_t pages, PageId end) const {
    PageId start = no_page;
    uint64_t length = 0;
    for (const auto& [first, r] : runs_) {
        if (first >= end)
            break;
        if (is_chunk_start(first) || (r.use != RunUse::free && r.use != RunUse::expert))
            start = no_page;
        if (r.use != RunUse::free && r.use != RunUse::expert)
            continue;
        if (start == no_page) {
            start = first;
            length = 0;
        }
        if ((length += std::min<uint64_t>(r.pages, end - first)) >= pages)
            return start;
    }
    return no_page;
}

/// Gets the first of `pages` pages before `end` for a block of conversation `id`: in one of its holes, else the
/// lowest free run that holds them, else in a new hole, the run of the least-used expert of any size (evicted);
/// no_page when there is none.
PageId PagePool::place_block(ConversationId id, uint32_t pages, PageId end) {
    Conversation& c = conversations_.at(id);
    for (const PageId hole : c.holes) {
        Run& h = runs_.at(hole);
        if (const PageId first = (uint64_t) hole + h.pages <= end && !is_lending(hole) ? h.room.take(pages, end)
                                                                                        : no_page;
            first != no_page) {
            if (h.room.pages < smallest_block_)
                c.holes.erase(hole);
            return first;
        }
    }
    Run content;
    content.use = RunUse::conversation;
    content.owner = id;
    if (const PageId first = find_lowest(pages, end); first != no_page) {
        content.pages = pages;
        return take(first, std::move(content));
    }
    for (const auto& [last_use, uses, first] : by_use_) {
        const uint32_t n = runs_.at(first).pages;
        if (n < pages || (uint64_t) first + n > end)
            continue;
        const PageId hole = first;   // a copy: evict erases the entry `first` refers to
        evict(hole, 0);
        content.use = RunUse::hole;
        content.pages = n;
        content.room.give(hole, n);
        take(hole, std::move(content));
        c.holes.insert(hole);
        return place_block(id, pages, end);
    }
    return no_page;
}

/// Gets the first of `pages` consecutive pages before `end` for a fixed region of conversation `id`: the lowest free
/// run that holds them, else the lowest stretch of free runs and experts, whose experts are evicted; no_page when
/// there is none.
PageId PagePool::place_fixed(ConversationId id, uint32_t pages, PageId end) {
    PageId first = find_lowest(pages, end);
    if (first == no_page && (first = find_stretch(pages, end)) != no_page) {
        std::vector<PageId> experts;
        for (auto it = get_run_at(first); it != runs_.end() && it->first < first + pages; ++it)
            if (it->second.use == RunUse::expert)
                experts.push_back(it->first);
        for (const PageId e : experts)
            evict(e, now_ + config_.hysteresis);
    }
    if (first == no_page)
        return no_page;
    Run content;
    content.pages = pages;
    content.use = RunUse::conversation;
    content.owner = id;
    return take(first, std::move(content));
}

/// Frees a conversation's block: its run, or its pages in a hole, which goes back whole (closed to experts until
/// `open_at`) once no block is left in it.
void PagePool::release_block(PageId first, uint32_t pages, uint64_t open_at) {
    const auto it = get_run_at(first);
    if (it->second.use != RunUse::hole) {
        release(first, open_at);
        return;
    }
    Run& h = it->second;
    Conversation& c = conversations_.at(h.owner);
    h.room.give(first, pages);
    if (h.room.pages == h.pages) {
        c.holes.erase(it->first);
        release(it->first, open_at);
    } else if (h.room.pages >= smallest_block_)
        c.holes.insert(it->first);
}

/// Frees each region's blocks past its first `keep[region]`; experts may take the pages from clock time `open_at`.
void PagePool::drop_blocks(Conversation& c, const std::vector<size_t>& keep, uint64_t open_at) {
    for (size_t r = 0; r < c.blocks.size(); ++r)
        for (; c.blocks[r].size() > keep[r]; c.blocks[r].pop_back())
            release_block(c.blocks[r].back(), block_pages_[r], open_at);
}

/// Puts `content` at `first`, inside a free run whose pages before and after it stay free; returns `first`.
PageId PagePool::take(PageId first, Run content) {
    const auto it = get_run_at(first);
    const PageId at = it->first;
    const Run free = it->second;
    unindex_free(it);
    runs_.erase(it);
    free_pages_ -= content.pages;
    const auto keep = [&](PageId from, PageId end) {
        if (from == end)
            return;
        Run part = free;
        part.pages = end - from;
        index_free(runs_.emplace(from, part).first);
    };
    keep(at, first);
    keep(first + content.pages, at + free.pages);
    runs_.emplace(first, std::move(content));
    return first;
}

/// Tells the owner that the expert whose run starts at `first` leaves the pool, and frees its pages; experts may take
/// them from clock time `open_at`.
void PagePool::evict(PageId first, uint64_t open_at) {
    hooks_.evicted(runs_.at(first).expert, first);
    free_expert(first, open_at);
}

/// Frees the pages of the expert whose run starts at `first`; experts may take them from clock time `open_at`.
void PagePool::free_expert(PageId first, uint64_t open_at) {
    unlist_expert(runs_.at(first), first);
    release(first, open_at);
}

/// Records that expert run `r` at `first` holds its expert (its place and its use).
void PagePool::list_expert(const Run& r, PageId first) {
    experts_[r.expert] = first;
    by_use_.insert({r.last_use, r.uses, first});
    size_by_use_[r.pages].insert({r.last_use, r.uses, first});
    expert_pages_ += r.pages;
}

/// Forgets what list_expert recorded.
void PagePool::unlist_expert(const Run& r, PageId first) {
    experts_.erase(r.expert);
    by_use_.erase({r.last_use, r.uses, first});
    size_by_use_[r.pages].erase({r.last_use, r.uses, first});
    expert_pages_ -= r.pages;
}

/// Frees the run that starts at `first`, joining it to a free run on either side in its chunk or a joined one; experts
/// may take its pages from clock time `open_at` (a joined run's from the later of the two times).
void PagePool::release(PageId first, uint64_t open_at) {
    auto it = runs_.find(first);
    Run run;
    run.pages = it->second.pages;
    run.open_at = open_at;
    free_pages_ += run.pages;
    runs_.erase(it);
    if (const PageId end = first + run.pages; !is_chunk_start(end))
        if (const auto after = runs_.find(end); after->second.use == RunUse::free) {
            run.pages += after->second.pages;
            run.open_at = std::max(run.open_at, after->second.open_at);
            unindex_free(after);
            runs_.erase(after);
        }
    if (!is_chunk_start(first))
        if (const auto before = std::prev(runs_.lower_bound(first)); before->second.use == RunUse::free) {
            unindex_free(before);
            before->second.pages += run.pages;
            before->second.open_at = std::max(before->second.open_at, run.open_at);
            index_free(before);
            return;
        }
    index_free(runs_.emplace(first, run).first);
}

/// Lends (as runs of their own) the free pages in [from, end), the run `lend` clears.
void PagePool::claim_free(PageId from, PageId end) {
    for (auto it = get_run_at(from); it != runs_.end() && it->first < end;) {
        if (it->second.use != RunUse::free) {
            ++it;
            continue;
        }
        const PageId first = std::max(it->first, from), last = std::min(it->first + it->second.pages, end);
        Run lent;
        lent.pages = last - first;
        lent.use = RunUse::lent;
        it = std::next(runs_.find(take(first, lent)));
    }
}

/// Frees the lent runs in [from, end): the pages a `lend` that failed had cleared.
void PagePool::release_lent(PageId from, PageId end) {
    for (PageId at = from;;) {
        const auto it = runs_.lower_bound(at);
        if (it == runs_.end() || it->first >= end)
            return;
        at = it->first + it->second.pages;
        if (it->second.use == RunUse::lent)
            release(it->first, 0);
    }
}

/// Tells whether the run that starts at `first` reaches into the run `lend` is clearing.
bool PagePool::is_lending(PageId first) const {
    return lending_from_ != no_page && first + runs_.at(first).pages > lending_from_;
}

/// Adds a free run to the indexes (of all free runs, and of the open ones or the closed ones) and the count of
/// fragmented pages.
void PagePool::index_free(Runs::iterator it) {
    free_runs_.insert(it->first);
    if (it->second.pages < smallest_expert_)
        fragmented_pages_ += it->second.pages;
    if (is_open(it->second))
        open_runs_.insert({it->second.pages, it->first});
    else
        opening_.insert({it->second.open_at, it->first});
}

/// Removes a free run from the indexes (before it changes or leaves).
void PagePool::unindex_free(Runs::iterator it) {
    free_runs_.erase(it->first);
    if (it->second.pages < smallest_expert_)
        fragmented_pages_ -= it->second.pages;
    if (open_runs_.erase({it->second.pages, it->first}) == 0)
        opening_.erase({it->second.open_at, it->first});
}

/// Gets the run that holds `page`.
PagePool::Runs::iterator PagePool::get_run_at(PageId page) {
    return std::prev(runs_.upper_bound(page));
}

/// Gets the size of `pages` pages as text, in GiB.
std::string PagePool::gib(uint64_t pages) const {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", (double) (pages * config_.page_bytes) / (1ull << 30));
    return text;
}

/// Takes the first run that holds `n` pages before `end`, from its start; no_page when there is none.
PageId PagePool::FreeRuns::take(uint32_t n, PageId end) {
    for (auto it = runs.begin(); it != runs.end() && it->first < end; ++it)
        if (it->second >= n && it->first + n <= end) {
            const PageId first = it->first;
            if (it->second > n)
                runs.emplace(first + n, it->second - n);
            runs.erase(it);
            pages -= n;
            return first;
        }
    return no_page;
}

/// Frees `n` pages from `first`, joining the runs on either side.
void PagePool::FreeRuns::give(PageId first, uint32_t n) {
    pages += n;
    auto next = runs.lower_bound(first);
    if (next != runs.end() && next->first == first + n) {
        n += next->second;
        next = runs.erase(next);
    }
    if (next != runs.begin())
        if (const auto before = std::prev(next); before->first + before->second == first) {
            before->second += n;
            return;
        }
    runs.emplace(first, n);
}

std::string get_expert_memory_label(int32_t layer, uint64_t bytes) {
    char label[96];
    std::snprintf(label, sizeof label, "Layer %d (%.2f MiB experts)", layer, (double) bytes / 1048576.0);
    return label;
}

}  // namespace strata::core
