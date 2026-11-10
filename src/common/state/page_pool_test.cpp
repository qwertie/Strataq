// The page pool (page_pool.hpp) on host memory with a hand-moved clock: conversations growing across blocks and
// truncated, fixed regions and blocks in consecutive pages, experts tiling the room of their size around its home
// point and evicting one of their size, an expert replaced in place, conversations filling the holes of experts they
// evict and giving them back whole, the allocated and fragmented pages, the token limits and the experts' overflow
// into the reserve, the hysteresis, growing and shrinking, the report, long random trading without fragmentation, a
// new layout and expert ranking (a layer split's cut moved), joined chunks (one virtual range) and the run a prompt
// borrows from the top (lend), and a pool of two million pages (Qwen3.8 IQ3_S's experts in pages of 25,600 bytes).
#include "strata/common/state/page_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

using namespace strata::core;
using Pages = std::vector<PageId>;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

constexpr uint64_t page = 64, chunk = 4 * page;

/// Gets the test's conversation: a 100-byte recurrent state (2 pages) and K/V of 2 heads x 8 bytes a token (4 tokens
/// a page), so a conversation of N tokens takes 2 + ceil(N / 4) pages.
StateLayout get_layout() {
    StateLayout layout;
    layout.add(RegionKind::fixed, 100);
    layout.add(RegionKind::per_token, 8, 2);
    return layout;
}

/// Gets a conversation of K/V only: one page a block of 4 tokens.
StateLayout get_kv_layout() {
    StateLayout layout;
    layout.add(RegionKind::per_token, 8, 2);
    return layout;
}

PagePoolConfig get_config(int64_t tokens_min = 0, int64_t tokens_max = INT64_MAX) {
    PagePoolConfig config;
    config.page_bytes = page;
    config.chunk_bytes = chunk;
    config.tokens_min = tokens_min;
    config.tokens_max = tokens_max;
    config.hysteresis = 100;
    return config;
}

/// Gets get_config's settings for chunks that are one range (PagePoolConfig::join_chunks).
PagePoolConfig get_joined_config() {
    PagePoolConfig config = get_config();
    config.join_chunks = true;
    return config;
}

/// A pool of `chunks` chunks (12 pages by default) on host memory, and what it called.
struct Fixture {
    uint64_t clock = 0;
    int chunks = 0, max_chunks = 100;
    std::vector<ExpertKey> evicted;
    std::vector<std::pair<ExpertKey, PageId>> relocated;
    /// With join_chunks, the chunks' buffer: they are back to back in it (one virtual range), else each is an
    /// allocation of its own.  (Before `pool`, which gives its chunks back when it goes.)
    std::vector<uint8_t> range;
    PagePool pool;

    explicit Fixture(PagePoolConfig config = get_config(), const StateLayout& layout = get_layout(), int n = 3)
        : range(config.join_chunks ? 64 * config.chunk_bytes : 0), pool(layout, config, get_hooks()) {
        check(pool.resize(n * config.chunk_bytes).refusal.empty() &&
                  pool.report().total.bytes == n * config.chunk_bytes,
              "the pool takes its chunks");
    }
    PagePoolHooks get_hooks() {
        PagePoolHooks hooks;
        hooks.alloc = [this](size_t bytes) -> void* {
            if (chunks == max_chunks)
                return nullptr;
            ++chunks;
            return range.empty() ? std::calloc(1, bytes) : range.data() + (chunks - 1) * bytes;
        };
        hooks.free = [this](void* p) {
            --chunks;
            if (range.empty())
                std::free(p);
        };
        hooks.now = [this] { return clock; };
        hooks.evicted = [this](ExpertKey key, PageId) { evicted.push_back(key); };
        hooks.copy = [this](PageId to, PageId from, uint32_t pages) {
            std::memmove(pool.page_address(to), pool.page_address(from), pages * page);
            return true;
        };
        hooks.relocated = [this](ExpertKey key, PageId to) { relocated.emplace_back(key, to); };
        return hooks;
    }
    /// Adds experts 0 to `count` - 1 of layer 0, of one page each, expert n at clock time n.
    void add_experts(int count) {
        for (int n = 0; n < count; clock = ++n)
            pool.add_expert({0, n}, 1);
    }
};

/// Gets the experts' numbers in `keys` (all of layer 0).
std::vector<int> get_experts(const std::vector<ExpertKey>& keys) {
    std::vector<int> out;
    for (const ExpertKey& key : keys)
        out.push_back(key.expert);
    return out;
}
void test_labels() {
    Fixture f;
    f.pool.add_expert({3, 0}, 1);
    f.pool.add_expert({3, 1}, 1);
    check(f.pool.grow_conversation(7, 1).empty(), "labelled conversation fits");
    f.pool.set_conversation_label(7, "Chat with spaces");
    const auto blocks = f.pool.get_memory_blocks();
    uint64_t bytes = 0, experts = 0;
    bool conversation = false;
    for (const MemoryBlock& b : blocks) {
        bytes += b.bytes;
        if (b.name.category == "experts") {
            experts += b.bytes;
            check(b.name.label.find("Layer 3 (") == 0, "expert labels retain their layer and size class");
            check(b.identity_group == "expert/experts/3" &&
                  std::set<uint32_t>(b.ids.begin(), b.ids.end()) == std::set<uint32_t>({0, 1}),
                  "layer groups retain logical expert IDs across memories");
        }
        conversation = conversation || b.name.label.find("Chat with spaces:") == 0;
    }
    check(experts == 2 * page && conversation && bytes == f.pool.report().total.bytes,
          "labels partition the pool without counting backing or holes twice");
    f.pool.set_paused(7, true);
    uint64_t paused = 0;
    for (const MemoryBlock& b : f.pool.get_memory_blocks())
        if (b.name.category == "paused")
            paused += b.bytes;
    check(paused == f.pool.report().paused.bytes, "paused blocks follow their conversation's lifetime");
}

/// Exercises the aligned views returned by ExpertPages rather than exact allocation addresses.
void test_aligned_backing() {
    constexpr size_t prefix = 13, padding = 20;
    const std::string memory = "aligned-pool-test";
    auto& manager = get_memory_manager();
    PagePoolHooks hooks;
    hooks.alloc = [&](size_t bytes) -> void* {
        auto* raw = (uint8_t*) std::calloc(1, bytes + padding);
        manager.add(raw, bytes + padding, memory, {"other", "Device allocations", "Engine buffers"});
        return raw ? raw + prefix : nullptr;
    };
    hooks.free = [&](void* view) {
        auto* raw = (uint8_t*) view - prefix;
        manager.remove(raw);
        std::free(raw);
    };
    {
        PagePool pool(get_layout(), get_config(), std::move(hooks));
        check(pool.resize(2 * chunk).refusal.empty(), "aligned pool grows to two allocations");
        const auto blocks = manager.get_blocks(memory);
        check(blocks.size() == 1 && blocks[0].name.label == "Alignment padding" &&
                  blocks[0].bytes == 2 * padding,
              "aligned backing is excluded while prefix and suffix padding remain visible");
        uint64_t total = blocks[0].bytes;
        for (const MemoryBlock& block : pool.get_memory_blocks())
            total += block.bytes;
        check(total == 2 * (chunk + padding), "pool and ledger totals partition physical backing exactly");
        check(pool.resize(chunk).refusal.empty() && manager.get_blocks(memory).at(0).bytes == padding,
              "shrinking an aligned pool releases the original allocation and its padding");
    }
    check(manager.get_blocks(memory).empty(), "destroying an aligned pool leaves no backing allocations");
}

void test_allocation() {
    Fixture f;
    check(f.pool.report().total.pages == 12 && f.pool.report().free.pages == 12, "a new pool's pages are free");
    check(f.pool.pages_for(0) == 2 && f.pool.pages_for(5) == 4, "pages_for: the fixed pages and the blocks");
    check(f.pool.grow_conversation(1, 5).empty(), "a conversation of 5 tokens");
    check(f.pool.conversation_blocks(1, 0) == Pages{0} && f.pool.block_bytes(0) == 2 * page,
          "the fixed region is one block of whole pages, at the bottom");
    check(f.pool.conversation_blocks(1, 1) == Pages{2, 3}, "5 tokens take 2 blocks");
    check(f.pool.grow_conversation(1, 8).empty() && f.pool.conversation_blocks(1, 1).size() == 2,
          "tokens 6 to 8 fit the second block");
    check(f.pool.grow_conversation(1, 9).empty() && f.pool.conversation_blocks(1, 1).back() == 4,
          "token 9 takes a third block, the lowest free page");
    check(f.pool.grow_conversation(1, 3).empty() && f.pool.report().tokens == 9, "fewer tokens keep the pages");
    check(f.pool.page_address(1) == f.pool.page_address(0) + page && f.pool.page_address(4) != nullptr,
          "pages of a chunk are adjacent");
    PagePoolConfig tiny = get_config();
    tiny.page_bytes = 8;
    check(PagePool(get_layout(), tiny, f.get_hooks()).block_bytes(1) == 16,
          "a page smaller than one token's K/V makes a block of several pages");
    bool threw = false;
    try {
        tiny.chunk_bytes = 8;
        PagePool pool(get_kv_layout(), tiny, f.get_hooks());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a block larger than a chunk is refused");
}

void test_runs() {
    Fixture g;   // conversations 1 and 2 grow in turns, so that their blocks alternate from page 6 on
    for (int64_t tokens = 4; tokens <= 16; tokens += 4)
        check(g.pool.grow_conversation(1, tokens).empty() && g.pool.grow_conversation(2, tokens).empty(), "growth");
    check(g.pool.conversation_blocks(2, 0) == Pages{4} && g.pool.conversation_blocks(1, 1) == Pages{2, 6, 8, 10},
          "a fixed region skips a chunk that has no room for it in one piece");
    g.pool.truncate_conversation(1, 4);
    check(g.pool.conversation_blocks(1, 1) == Pages{2} && g.pool.conversation_blocks(1, 0) == Pages{0} &&
              g.pool.report().free.pages == 3 && g.pool.report().tokens == 20,
          "a truncated conversation gives back its last blocks and keeps its fixed pages");
    check(g.pool.grow_conversation(3, 4) == "the pool has no room in one piece for the conversation's fixed state" &&
              g.pool.report().free.pages == 3 && g.pool.report().active_conversations == 2,
          "three free pages apart do not hold a fixed region of two");

    Fixture h;   // conversation 4's pages are in the last chunk, and the first chunk has one free page left
    for (ConversationId id = 1; id <= 4; ++id)
        check(h.pool.grow_conversation(id, 4).empty(), "four conversations fill the pool");
    check(h.pool.conversation_blocks(4, 0) == Pages{9}, "conversation 4's fixed region on pages 9 and 10");
    std::memset(h.pool.page_address(10), 0x77, page);
    h.pool.remove_conversation(2);
    h.pool.remove_conversation(3);
    check(h.pool.resize(2 * chunk).refusal.empty() && h.pool.conversation_blocks(4, 0) == Pages{4} &&
              h.pool.conversation_blocks(4, 1) == Pages{3} && h.pool.page_address(5)[0] == 0x77,
          "a shrink moves a fixed region in one piece");
    bool threw = false;
    try {
        StateLayout big;
        big.add(RegionKind::fixed, chunk + 1);
        PagePool pool(big, get_config(), h.get_hooks());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a fixed region larger than a chunk is refused");
}

/// Gets a pool of one chunk of 16 pages whose ranking is 2 experts of 2 pages and 4 of 3 pages (their rooms pages 0
/// to 3 and 4 to 15, their homes 2 and 10), filled with them: ranking place r holds expert r of layer 0, used at
/// 100 - r.
PagePoolConfig get_tiling_config() {
    PagePoolConfig config = get_config();
    config.chunk_bytes = 16 * page;
    config.expert_ranking = {3, 2, 3, 2, 3, 3};
    return config;
}
Pages fill_tiling(Fixture& f) {
    f.clock = 100;
    Pages at;
    for (int r = 0; r < 6; ++r)
        at.push_back(f.pool.add_expert({0, r}, get_tiling_config().expert_ranking[(size_t) r], 100 - r));
    return at;
}

void test_experts_tile() {
    Fixture f(get_tiling_config(), get_kv_layout(), 1);
    check(f.pool.get_home(2) == 2 && f.pool.get_home(3) == 10, "each size's home is the middle slot of its room");
    check(fill_tiling(f) == Pages{10, 2, 7, 0, 4, 13} && f.pool.report().free.pages == 0,
          "experts fill their rooms outwards from their homes, without a page left over");
    check(f.pool.add_expert({1, 0}, 3, 50) == no_page && f.evicted.empty(),
          "a full pool keeps the experts used after a new one");
    check(f.pool.add_expert({1, 0}, 3) == 13 && get_experts(f.evicted) == std::vector<int>{5},
          "a new expert evicts exactly one: the least-used of its size");
    f.pool.remove_expert({0, 3});
    check(f.pool.add_expert({1, 1}, 3) == 4 && get_experts(f.evicted) == std::vector<int>{5, 4} &&
              f.pool.report().free.pages == 2,
          "the free pages of a smaller size do not hold it");
    check(f.pool.add_expert({1, 2}, 2) == 0, "and stay for an expert of their size");
}

void test_replace() {
    Fixture f;
    f.add_experts(12);
    check(f.pool.report().free.pages == 0, "twelve experts fill the pool");
    const PageId first = f.pool.use_expert({0, 5});
    check(f.pool.replace_expert({0, 5}, {0, 20}) == first && f.pool.use_expert({0, 5}) == no_page &&
              f.pool.use_expert({0, 20}) == first && f.evicted.empty(),
          "a replaced expert's pages go to the new one, without an eviction");
    check(f.pool.replace_expert({0, 5}, {0, 21}) == no_page && f.pool.replace_expert({0, 1}, {0, 2}) == no_page,
          "an expert the pool does not hold is not replaced, nor one by an expert it holds");
    f.pool.replace_expert({0, 0}, {0, 22});
    check(f.pool.add_expert({1, 0}, 1, 1) == no_page && f.pool.add_expert({1, 0}, 1) != no_page &&
              get_experts(f.evicted) == std::vector<int>{1},
          "the new expert's use counts from the replacement");
}

/// Checks that routing refreshes both eviction orders without changing expert ownership or page totals.
void test_use_expert() {
    Fixture f(get_config(), get_kv_layout());
    f.add_experts(12);
    const PageId first = f.pool.find_expert({0, 0});
    f.clock = 20;
    check(f.pool.use_expert({0, 0}) == first && f.pool.use_expert({0, 0}) == first &&
              f.pool.find_expert({0, 0}) == first && f.pool.report().experts.pages == 12,
          "repeated routing preserves the expert's place and page count");
    check(f.pool.add_expert({0, 12}, 1) != no_page && get_experts(f.evicted) == std::vector<int>{1},
          "a same-size replacement protects the recently routed expert");
    check(f.pool.grow_conversation(1, 4).empty() && get_experts(f.evicted) == std::vector<int>{1, 2} &&
              f.pool.find_expert({0, 0}) == first && f.pool.report().experts.pages == 11,
          "a conversation protects the same recently routed expert");
}

void test_holes() {
    PagePoolConfig config = get_tiling_config();
    Fixture f(config, get_kv_layout(), 1);
    fill_tiling(f);
    check(f.pool.grow_conversation(1, 4).empty() && f.pool.conversation_blocks(0 + 1, 0) == Pages{13} &&
              get_experts(f.evicted) == std::vector<int>{5},
          "a conversation evicts the least-used expert of any size and takes its hole");
    check(f.pool.grow_conversation(1, 12).empty() && f.pool.conversation_blocks(1, 0) == Pages{13, 14, 15} &&
              f.evicted.size() == 1,
          "it fills the hole before it takes another");
    check(f.pool.grow_conversation(1, 16).empty() && f.pool.conversation_blocks(1, 0).back() == 4 &&
              get_experts(f.evicted) == std::vector<int>{5, 4} && f.pool.report().experts.pages == 10 &&
              f.pool.report().free.pages == 2,
          "the next block takes the next least-used expert's hole; its unused pages count as free");
    f.pool.truncate_conversation(1, 12);
    check(f.pool.report().free.pages == 3 && f.pool.get_next_opening() == 200,
          "a hole goes back whole when its last block leaves, closed to experts for the hysteresis");
    f.pool.remove_conversation(1);
    f.clock = 200;
    check(f.pool.add_expert({0, 4}, 3, 96) == 4 && f.pool.add_expert({0, 5}, 3, 95) == 13 &&
              f.pool.report().free.pages == 0,
          "the evicted experts come back to their own places");
}

void test_allocated_and_fragmented() {
    Fixture f(get_tiling_config(), get_kv_layout(), 1);
    check(f.pool.report().allocated.pages == 0 && f.pool.report().fragmented.pages == 0, "an empty pool");
    fill_tiling(f);
    f.pool.remove_expert({0, 3});
    check(f.pool.report().allocated.pages == 14 && f.pool.report().fragmented.pages == 0,
          "a free run as long as the smallest expert is not fragmented");
    check(f.pool.grow_conversation(1, 4).empty() && f.pool.conversation_blocks(1, 0) == Pages{0} &&
              f.pool.report().allocated.pages == 15 && f.pool.report().fragmented.pages == 1 &&
              f.pool.report().fragmented.bytes == page,
          "a block in it leaves a page that no expert can use");
    f.pool.remove_conversation(1);
    check(f.pool.report().allocated.pages == 14 && f.pool.report().fragmented.pages == 0,
          "the block's page joins the run again");
}

/// A layer split's cut moves: the conversations' regions change while the pool holds none, and the experts' ranking.
void test_new_layout() {
    Fixture f(get_config(8));   // 8 tokens of the reserve: 4 pages
    check(f.pool.grow_conversation(1, 4).empty() && !f.pool.set_layout(get_kv_layout()).empty() &&
              f.pool.pages_for(4) == 3,
          "no new layout while the pool holds a conversation");
    f.pool.remove_conversation(1);
    check(f.pool.set_layout(get_kv_layout()).empty() && f.pool.pages_for(4) == 1 &&
              f.pool.grow_conversation(2, 4).empty() && f.pool.conversation_blocks(2, 0).size() == 1,
          "an empty pool's conversations take the new regions");
    f.pool.remove_conversation(2);
    StateLayout too_large;   // a fixed region of 5 pages, and a chunk of 4
    too_large.add(RegionKind::fixed, 5 * page);
    StateLayout too_wide;    // 2 pages a token: the reserve of 8 tokens would take 16 of the 12 pages
    too_wide.add(RegionKind::per_token, 8, 16);
    check(!f.pool.set_layout(too_large).empty() && !f.pool.set_layout(too_wide).empty() && f.pool.pages_for(4) == 1,
          "a layout whose fixed region exceeds a chunk or whose reserve does not fit is refused, the old one stays");
    f.pool.set_expert_ranking({5});
    check(f.pool.report().fragmented.pages == 12, "with a ranking of 5-page experts, every 4-page run is fragmented");
    f.pool.set_expert_ranking({1, 1});
    check(f.pool.report().fragmented.pages == 0 && f.pool.get_home(1) == 3,
          "with one of 1-page experts none is, and their home is above the reserve of the new layout");
}

void test_minimum() {
    PagePoolConfig config = get_config(8);   // 8 tokens: 4 pages
    config.expert_ranking.assign(8, 1);
    Fixture f(config);
    f.add_experts(8);
    check(f.pool.report().free.pages == 4 && f.evicted.empty() && f.pool.get_home(1) == 8,
          "experts leave the minimum's pages free, at the bottom");
    check(f.pool.add_expert({1, 0}, 1) != no_page && f.evicted.size() == 1, "past the minimum an expert replaces one");
    check(f.pool.grow_conversation(1, 8).empty() && f.evicted.size() == 1 &&
              f.pool.conversation_blocks(1, 0) == Pages{0},
          "a conversation takes the kept pages");
    check(f.pool.grow_conversation(1, 9).empty() && f.evicted.size() == 2, "and experts' pages past the minimum");
}

void test_overflow() {
    PagePoolConfig config = get_config(8);   // the reserve: pages 0 to 3
    config.expert_ranking.assign(6, 1);
    Fixture f(config);
    f.add_experts(6);
    check(f.pool.grow_conversation(1, 4).empty() && f.pool.grow_conversation(2, 4).empty() &&
              f.pool.conversation_blocks(2, 0) == Pages{10} && f.pool.conversation_blocks(2, 1) == Pages{3} &&
              f.pool.report().free.pages == 0,
          "a fixed region past the reserve takes the free run above the experts");
    f.pool.remove_conversation(1);
    f.clock = 50;
    const PageId closed = f.pool.add_expert({1, 0}, 1);
    check(closed >= 4 && closed < 10 && f.evicted.size() == 1,
          "while the reserve's free pages are closed, a new expert evicts one of its size instead");
    f.clock = 200;
    check(f.pool.add_expert({1, 1}, 1) == 2 && f.pool.add_expert({1, 2}, 1) == 1 && f.evicted.size() == 1,
          "once they open, experts overflow into the reserve's pages past what it must keep");
    check(f.pool.add_expert({1, 3}, 1) != no_page && f.evicted.size() == 2 && f.pool.report().free.pages == 1,
          "and leave the rest free");
}

void test_maximum() {
    Fixture f(get_config(0, 10));
    check(f.pool.grow_conversation(1, 8).empty(), "8 tokens fit a maximum of 10");
    check(!f.pool.grow_conversation(2, 4).empty() && f.pool.report().active_conversations == 1,
          "a conversation past the maximum is refused and not added");
    check(!f.pool.set_token_limits(0, 6).empty(), "a maximum below an active conversation is refused");
    check(!f.pool.set_token_limits(9, 4).empty(), "a minimum above the maximum is refused");
    check(!f.pool.set_token_limits(41, 100).empty(), "a minimum larger than the pool is refused");
    check(f.pool.set_token_limits(0, 100).empty() && f.pool.grow_conversation(2, 4).empty(), "a raised maximum");
    check(f.pool.grow_conversation(3, 40).find("the conversation needs") == 0, "a full pool refuses a conversation");
}

void test_hysteresis() {
    Fixture f;
    check(f.pool.grow_conversation(1, 40).empty() && f.pool.report().free.pages == 0, "a conversation fills the pool");
    f.pool.remove_conversation(1);
    f.clock = 50;
    check(f.pool.add_expert({0, 0}, 1) == no_page, "experts wait for the hysteresis");
    check(f.pool.grow_conversation(2, 4).empty() && f.pool.conversation_blocks(2, 0)[0] == 0,
          "conversations do not");
    f.pool.set_paused(2, true);
    check(f.pool.grow_conversation(2, 5) == "the conversation is paused", "a paused conversation does not grow");
    f.clock = 100;
    check(f.pool.add_expert({0, 0}, 1) == 6, "after the hysteresis experts take the pages (of no size's room: the "
                                             "middle of the pool)");
}

void test_resize() {
    Fixture f;
    check(f.pool.grow_conversation(1, 12).empty() && f.pool.grow_conversation(2, 4).empty(), "pages 0 to 7");
    f.add_experts(4);
    check(f.pool.report().experts.pages == 4 && f.pool.report().free.pages == 0, "experts on pages 8 to 11");
    const std::vector<PageId> kv = f.pool.conversation_blocks(2, 1);
    std::memset(f.pool.page_address(kv[0]), 0x5a, page);
    f.pool.remove_conversation(1);
    check(f.pool.resize(chunk + 100).refusal.empty() && f.chunks == 1, "the pool shrinks to its first chunk");
    check(f.evicted.size() == 4, "the given-back chunks' experts are evicted");
    const std::vector<PageId> moved = f.pool.conversation_blocks(2, 1);
    check(moved[0] < 4 && f.pool.page_address(moved[0])[page - 1] == 0x5a, "a moved page keeps its contents");
    check(f.pool.report().total.pages == 4 && f.pool.report().free.pages == 1, "the smaller pool's pages");
    check(f.pool.resize(3 * chunk).refusal.empty() && f.pool.report().free.pages == 9, "and grows again");
    f.max_chunks = 4;
    check(f.pool.resize(5 * chunk).refusal == "the memory had room for 4 of 5 chunks" && f.chunks == 4,
          "a growth the memory has no room for keeps the chunks it got");
}

void test_shrink_refusals() {
    Fixture f;
    check(f.pool.grow_conversation(1, 24).empty(), "a conversation of 8 pages");
    check(f.pool.resize(chunk).refusal.find("an active conversation needs") == 0 && f.chunks == 3,
          "a pool too small for the biggest active conversation is refused");
    check(f.pool.grow_conversation(2, 4).empty(), "a second conversation on pages 8 to 10");
    const PoolResize r = f.pool.resize(2 * chunk);
    check(!r.refusal.empty() && r.to_pause == std::vector<ConversationId>{2} && f.chunks == 3,
          "conversation pages with nowhere to go name the conversations to pause");
    f.pool.set_paused(2, true);
    check(f.pool.resize(2 * chunk).to_pause.size() == 1, "a paused conversation keeps its pages until removed");
    f.pool.remove_conversation(2);
    check(f.pool.resize(2 * chunk).refusal.empty() && f.chunks == 2, "once removed the pool shrinks");
}

/// Tells whether a conversation's blocks of a region stay out of pages [from, end).
bool is_outside(const Pages& blocks, PageId from, PageId end) {
    return std::none_of(blocks.begin(), blocks.end(), [&](PageId b) { return b >= from && b < end; });
}

/// Joined chunks (one virtual range): runs cross their bounds, and a shrink cuts a run across the new end.
void test_joined() {
    Fixture f(get_joined_config(), get_layout(), 4);   // 16 pages in one range
    check(f.pool.page_address(4) == f.pool.page_address(3) + page && f.pool.lend(10, 0).first == 6,
          "joined chunks are one run: a run may cross their bounds");
    f.pool.give_back();
    check(f.pool.grow_conversation(1, 20).empty() && f.pool.grow_conversation(2, 1).empty() &&
              f.pool.conversation_blocks(2, 0) == Pages{7},
          "a fixed region across a chunk's bound");
    f.pool.remove_conversation(1);
    check(f.pool.resize(2 * chunk).refusal.empty() && f.chunks == 2 && f.pool.conversation_blocks(2, 0) == Pages{0} &&
              f.pool.report().free.pages == 5,
          "a shrink moves a block across the new end and cuts the free run across it");
    check(f.pool.resize(4 * chunk).refusal.empty() && f.pool.report().free.pages == 13, "and grows again");
}

/// PagePool::lend: a free run at the top as it is, else experts moved (leaving the pages the prompt's K/V needs) or
/// evicted, stray conversation blocks moved, never across chunks that are not joined, everything back after give_back.
void test_lend() {
    {   // a free run, which nothing else takes while it is lent
        Fixture f(get_config(), get_layout(), 3);   // three chunks of 4 pages, not joined
        const PoolLend l = f.pool.lend(3, 0);
        check(l.first == 9 && l.refusal.empty() && l.experts_moved == 0 && f.pool.lent_pages() == 3 &&
                  f.pool.report().lent.pages == 3 && f.pool.report().free.pages == 9,
              "a free run at the pool's top is lent as it is");
        check(f.pool.lend(1, 0).refusal == "a run is lent already" && !f.pool.can_lend(1) &&
                  f.pool.resize(2 * chunk).refusal.find("lent") != std::string::npos,
              "one run at a time, and no shrink while it is lent");
        check(f.pool.grow_conversation(1, 28).empty() && is_outside(f.pool.conversation_blocks(1, 1), 9, 12) &&
                  !f.pool.grow_conversation(1, 29).empty(),
              "a growing conversation takes every page but the lent ones");
        uint64_t prompt = 0;
        for (const MemoryBlock& b : f.pool.get_memory_blocks())
            prompt += b.name.category == "prompt_buffers" ? b.bytes : 0;
        check(prompt == 3 * page, "the lent pages are their own use in the memory report");
        f.pool.give_back();
        check(f.pool.lent_pages() == 0 && f.pool.report().free.pages == 3 && f.pool.grow_conversation(1, 32).empty(),
              "given back, the pages are free again");
    }
    for (const uint64_t keep : {0, 1}) {   // an expert there moves while the free pages left are enough
        Fixture f(get_joined_config(), get_layout(), 2);   // 8 pages in one range
        check(f.pool.add_expert({0, 0}, 2) == 4 && f.pool.add_expert({1, 0}, 2) == 2,
              "experts outside the ranking sit round the middle: pages 2 to 5");
        std::memset(f.pool.page_address(4), 0x5a, 2 * page);
        const PoolLend l = f.pool.lend(4, keep);   // pages 4 to 7
        check(l.first == 4 && f.pool.report().lent.pages == 4, "the run is the pool's top");
        if (keep == 0)
            check(l.experts_moved == 1 && f.pool.find_expert({0, 0}) == 0 && f.relocated.size() == 1 &&
                      f.relocated[0].second == 0 && f.pool.page_address(1)[page - 1] == 0x5a && f.evicted.empty(),
                  "an expert in the run is copied to free pages elsewhere, and its owner told");
        else
            check(l.experts_evicted == 1 && get_experts(f.evicted) == std::vector<int>{0} && f.relocated.empty() &&
                      f.pool.report().free.pages == 2,
                  "it is evicted when moving it would leave fewer free pages than the prompt's K/V needs");
        f.pool.give_back();
        check(f.pool.report().lent.pages == 0 && f.pool.report().free.pages == (keep == 0 ? 4u : 6u) &&
                  (keep == 0 || f.pool.add_expert({0, 0}, 2) != no_page),
              "given back, the pages are free again and an evicted expert comes back");
    }
    {   // a conversation's blocks there move out, keeping their contents
        Fixture f(get_joined_config(), get_layout(), 3);   // 12 pages in one range
        check(f.pool.grow_conversation(1, 24).empty() && f.pool.grow_conversation(2, 4).empty() &&
                  f.pool.conversation_blocks(2, 0) == Pages{8} && f.pool.conversation_blocks(2, 1) == Pages{10},
              "a conversation's pages at the pool's top");
        f.pool.remove_conversation(1);
        std::memset(f.pool.page_address(10), 0x33, page);
        const PoolLend l = f.pool.lend(3, 0);
        check(l.first == 9 && l.moved == std::vector<ConversationId>{2} &&
                  f.pool.conversation_blocks(2, 0) == Pages{0} && f.pool.conversation_blocks(2, 1) == Pages{2} &&
                  f.pool.page_address(2)[0] == 0x33,
              "stray blocks (a fixed region reaching into the run too) are copied out of it");
    }
    {   // a block in a hole there moves out, and the hole goes
        Fixture f(get_joined_config(), get_kv_layout(), 3);   // 12 pages in one range
        for (int n = 0; n < 6; ++n)
            f.pool.add_expert({0, n}, 2);
        check(f.pool.find_expert({0, 5}) == 10 && f.pool.report().free.pages == 0, "six experts fill the pool");
        f.clock = 10;
        for (int n = 0; n < 5; ++n)
            f.pool.use_expert({0, n});
        check(f.pool.grow_conversation(1, 4).empty() && f.pool.conversation_blocks(1, 0) == Pages{10},
              "a conversation's block in the least-used expert's hole, at the pool's top");
        const PoolLend l = f.pool.lend(2, 0);
        check(l.first == 10 && l.moved == std::vector<ConversationId>{1} &&
                  f.pool.conversation_blocks(1, 0) == Pages{0} && get_experts(f.evicted) == std::vector<int>{5, 4},
              "its block moves to the hole of another least-used expert, outside the run");
    }
    {   // refusals: a run across chunks that are not joined, and blocks with nowhere to go
        Fixture f(get_config(), get_layout(), 3);
        check(!f.pool.can_lend(5) && f.pool.lend(5, 0).refusal.find("in one piece") != std::string::npos &&
                  f.pool.lend(4, 0).first == 8,
              "a lent run stays inside the top chunk when the chunks are not joined");
        f.pool.give_back();
        Fixture g(get_joined_config(), get_layout(), 2);
        check(g.pool.grow_conversation(1, 24).empty() && g.pool.report().free.pages == 0, "a conversation fills it");
        check(!g.pool.can_lend(3) && !g.pool.lend(3, 0).refusal.empty() && g.pool.lent_pages() == 0 &&
                  g.pool.report().free.pages == 0,
              "a run whose blocks have nowhere to go is refused, changing nothing");
    }
}

void test_report() {
    Fixture f;
    f.pool.grow_conversation(1, 5);
    f.pool.grow_conversation(2, 4);
    f.pool.set_paused(2, true);
    f.pool.add_expert({0, 0}, 1);
    f.pool.add_expert({0, 1}, 1);
    const PagePoolReport r = f.pool.report();
    check(r.active.pages == 2 && r.fixed.pages == 2 && r.paused.pages == 3 && r.experts.pages == 2 &&
              r.free.pages == 3,
          "the pages by use");
    check(r.active.pages + r.fixed.pages + r.paused.pages + r.experts.pages + r.free.pages == r.total.pages,
          "the uses add up to the pool");
    check(r.total.bytes == 12 * page && r.paused.bytes == 3 * page, "bytes are pages times the page size");
    check(r.active_conversations == 1 && r.paused_conversations == 1 && r.tokens == 9, "the conversations");
}

void test_block_runs() {
    StateLayout layout;   // K/V of 2 rows x 48 bytes a token: a token's 96 bytes take 2 pages, 2 tokens 3 exactly
    layout.add(RegionKind::per_token, 48, 2);
    PagePoolConfig config = get_config();
    config.chunk_bytes = 8 * page;
    Fixture f(config, layout);
    check(f.pool.block_bytes(0) == 3 * page && f.pool.pages_for(5) == 9, "a block of 3 pages holds 2 tokens");
    check(f.pool.grow_conversation(1, 4).empty() && f.pool.conversation_blocks(1, 0) == Pages{0, 3},
          "each block is 3 consecutive pages");
    check(f.pool.grow_conversation(1, 6).empty() && f.pool.conversation_blocks(1, 0).back() == 8,
          "a block does not cross a chunk's end");
    StateRegion int8;   // Qwen3.8's int8 K/V unit: 4 tokens of 2 heads in 4,736 bytes
    int8.kind = RegionKind::per_block;
    int8.unit_bytes = 4736;
    int8.block = 4;
    check(choose_block_pages(int8, 25600) == 5 && choose_block_pages(int8, 2 << 20) == 1,
          "pages of 25,600 bytes make blocks of 5 (27 units, 0.1% unused)");
}

void test_random_trading() {
    // experts of 1 to 4 pages and conversations trading pages at random in 4 chunks of 16 pages, as an engine trades
    // them: experts leave only when evicted, and come back (hottest first) between steps.  No two of them ever share
    // a page, the free pages are the rest, and once the conversations have left the experts fill the pool as well as
    // they fill an empty one.
    PagePoolConfig config = get_config(8);
    config.chunk_bytes = 16 * page;
    std::vector<uint32_t> sizes;
    for (int r = 0; r < 30; ++r)
        sizes.push_back(1 + (r * 7) % 4);
    config.expert_ranking = sizes;
    const uint64_t t0 = 1000000;
    std::map<int, std::pair<PageId, uint32_t>> held;   // expert -> first page, pages
    const auto refill = [&](Fixture& f) {
        for (const ExpertKey& key : f.evicted)
            held.erase(key.expert);
        f.evicted.clear();
        for (int r = 0; r < (int) sizes.size(); ++r)
            if (held.count(r) == 0)
                if (const PageId first = f.pool.add_expert({0, r}, sizes[(size_t) r], t0 - r); first != no_page)
                    held[r] = {first, sizes[(size_t) r]};
    };
    Fixture fresh(config, get_layout(), 4);
    refill(fresh);
    const uint64_t fits = fresh.pool.report().experts.pages;
    held.clear();
    Fixture f(config, get_layout(), 4);
    f.clock = t0;
    refill(f);
    std::set<ConversationId> live;
    uint64_t seed = 1;
    const auto next = [&](uint64_t n) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33) % n;
    };
    for (int step = 0; step < 20000; ++step) {
        f.clock += next(60);
        const ConversationId id = 1 + next(3);
        switch (next(4)) {
        case 0:
            refill(f);
            break;
        case 1:
            if (f.pool.grow_conversation(id, (int64_t) next(60)).empty())
                live.insert(id);
            break;
        default:
            if (live.count(id) == 0)
                break;
            if (next(2) == 0)
                f.pool.truncate_conversation(id, (int64_t) next(30));
            else {
                f.pool.remove_conversation(id);
                live.erase(id);
            }
        }
        for (const ExpertKey& key : f.evicted)
            held.erase(key.expert);
        f.evicted.clear();
        std::vector<int> owner(64, 0);
        uint64_t used = 0;
        const auto claim = [&](PageId first, uint64_t pages) {
            for (PageId p = first; p < first + pages; ++p) {
                check(p / 16 == first / 16 && owner[p] == 0, "every run is inside one chunk and on pages of its own");
                owner[p] = 1;
            }
            used += pages;
        };
        for (const auto& [e, at] : held)
            claim(at.first, at.second);
        for (const ConversationId c : live)
            for (size_t r = 0; r < 2; ++r)
                for (const PageId first : f.pool.conversation_blocks(c, r))
                    claim(first, f.pool.block_bytes(r) / page);
        const PagePoolReport rep = f.pool.report();
        check(rep.experts.pages + rep.fixed.pages + rep.active.pages + rep.paused.pages == used &&
                  rep.free.pages + used == 64,
              "the free pages are the ones nothing holds");
    }
    for (const ConversationId c : live)
        f.pool.remove_conversation(c);
    f.clock += 1000;
    refill(f);
    std::printf("page_pool_test: after 20,000 random trades the experts fill %llu pages, in an empty pool %llu\n",
                (unsigned long long) f.pool.report().experts.pages, (unsigned long long) fits);
    check(f.pool.report().experts.pages == fits, "the trading leaves the experts as much room as before");
}

void test_many_pages() {
    // Qwen3.8 IQ3_S: 48 layers of 512 experts, 7 sizes from 59 to 104 pages of 25,600 bytes; a pool of 1.9M pages
    // (in chunks of 10,000) holds 96% of them.  A conversation of 112 MiB of fixed state and 12 layers of K/V units
    // grows to 204,800 tokens, then leaves.
    constexpr uint64_t big_page = 25600, chunk_pages = 10000, chunks = 190;
    const uint32_t sizes[] = {59, 67, 73, 77, 85, 91, 104};
    StateLayout layout;
    layout.add(RegionKind::fixed, 112ull << 20);
    for (int l = 0; l < 12; ++l)
        layout.add(RegionKind::per_block, 4736, 1, 4);
    std::vector<ExpertKey> ranked;   // the experts hottest first, a layer's experts spread over the ranking
    for (int32_t e = 0; e < 512; ++e)
        for (int32_t l = 0; l < 48; ++l)
            ranked.push_back({l, (e * 37 + l * 11) % 512});
    PagePoolConfig config;
    config.page_bytes = big_page;
    config.chunk_bytes = chunk_pages * big_page;
    config.tokens_min = 8192;
    for (const ExpertKey& k : ranked)
        config.expert_ranking.push_back(sizes[k.layer % 7]);
    uint64_t clock = 1000000;
    std::vector<ExpertKey> evicted;
    PagePoolHooks hooks;
    hooks.alloc = [](size_t) { return (void*) 4096; };   // nothing reads the pages
    hooks.free = [](void*) {};
    hooks.now = [&] { return clock; };
    hooks.evicted = [&](ExpertKey key, PageId) { evicted.push_back(key); };
    PagePool pool(layout, config, hooks);
    check(pool.resize(chunks * config.chunk_bytes).refusal.empty() && pool.report().total.pages == 1900000,
          "1.9M pages");
    const auto start = std::chrono::steady_clock::now();
    const auto seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    const uint64_t t0 = clock;   // the hottest expert's last use; a colder one's is earlier, as ExpertPages gives
    const auto add = [&](size_t rank) {
        const ExpertKey k = ranked[rank];
        return pool.add_expert(k, sizes[k.layer % 7], t0 - 1 - rank) != no_page;
    };
    size_t held = 0;
    while (held < ranked.size() && add(held))
        ++held;
    const uint64_t expert_pages = pool.report().experts.pages;
    const double fill_s = seconds();
    check(held > 23000 && pool.report().free.pages < pool.pages_for(8192) + 104 * 7 + chunks * 104,
          "the experts fill the pool down to the token minimum's reserve");
    for (int64_t tokens = 2048; tokens <= 204800; tokens += 2048, clock += 10)
        check(pool.grow_conversation(1, tokens).empty(), "the conversation grows by evicting experts");
    const size_t gone = evicted.size();
    check(gone > 0, "the conversation took experts' pages");
    const double grow_s = seconds() - fill_s;
    pool.remove_conversation(1);
    clock += 10000;
    std::set<ExpertKey> left(evicted.begin(), evicted.end());
    size_t back = 0;
    for (size_t rank = 0; rank < held; ++rank)
        if (left.count(ranked[rank]) != 0 && add(rank))
            ++back;
    const double refill_s = seconds() - fill_s - grow_s;
    std::printf("page_pool_test: 1.9M pages: %zu experts (%llu pages) in %.3f s; a 204,800-token conversation "
                "evicted %zu in %.3f s; %zu came back in %.3f s, %llu expert pages held\n",
                held, (unsigned long long) expert_pages, fill_s, gone, grow_s, back, refill_s,
                (unsigned long long) pool.report().experts.pages);
    check(back == gone && pool.report().experts.pages == expert_pages,
          "every evicted expert comes back once the conversation leaves");

    // three conversations of up to 100,000 tokens growing, shrinking and ending at random for 300 steps, the
    // evicted experts put back after each step, as the engine puts them back between requests
    uint64_t seed = 5;
    const auto next = [&](uint64_t n) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33) % n;
    };
    std::set<ConversationId> live;
    std::map<ExpertKey, size_t> rank_of;
    for (size_t r = 0; r < held; ++r)
        rank_of[ranked[r]] = r;
    std::set<size_t> out;   // the ranks of the evicted experts
    evicted.clear();
    for (int step = 0; step < 300; ++step) {
        clock += 1000;
        const ConversationId id = 1 + next(3);
        if (next(3) > 0) {
            if (pool.grow_conversation(id, 1 + (int64_t) next(100000)).empty())
                live.insert(id);
        } else if (live.count(id) != 0 && next(2) == 0)
            pool.truncate_conversation(id, (int64_t) next(50000));
        else if (live.count(id) != 0) {
            pool.remove_conversation(id);
            live.erase(id);
        }
        for (const ExpertKey& key : evicted)
            out.insert(rank_of.at(key));
        evicted.clear();
        clock += 3000;
        for (auto it = out.begin(); it != out.end() && add(*it);)
            it = out.erase(it);
    }
    for (const ConversationId c : live)
        pool.remove_conversation(c);
    clock += 10000;
    for (auto it = out.begin(); it != out.end() && add(*it);)
        it = out.erase(it);
    std::printf("page_pool_test: then 300 random steps of 3 conversations; afterwards %llu expert pages held, "
                "%zu experts out (%.3f s in all)\n",
                (unsigned long long) pool.report().experts.pages, out.size(), seconds());
    check(out.empty() && pool.report().experts.pages == expert_pages,
          "after random trading every expert comes back");
    check(seconds() < 10, "two million pages take seconds, not minutes");
}
}  // namespace

int main() {
    test_labels();
    test_aligned_backing();
    test_allocation();
    test_runs();
    test_experts_tile();
    test_replace();
    test_use_expert();
    test_holes();
    test_allocated_and_fragmented();
    test_new_layout();
    test_minimum();
    test_overflow();
    test_maximum();
    test_hysteresis();
    test_resize();
    test_shrink_refusals();
    test_report();
    test_joined();
    test_lend();
    test_block_runs();
    test_random_trading();
    test_many_pages();
    std::printf("page_pool_test: %d checks passed\n", checks);
    return 0;
}
