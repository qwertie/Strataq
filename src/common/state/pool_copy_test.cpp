// A conversation's pages copied between two page pools on host memory and through a file (pool_copy.hpp): the bytes
// arrive in the other pool's blocks, a place with fewer blocks takes the first ones, and one with more is refused; the
// pause files a process that no longer runs left are deleted; one card's regions of a layer split go to their place in
// a pool that holds every card's regions, and back; and a card's share of a fixed region goes to its byte range of a
// larger one, to a file, and back.
#include "strata/common/state/pool_copy.hpp"

#include "strata/base/platform/os.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

/// A pool of 64 pages of 64 bytes on host memory, for a conversation of a 100-byte state and 16 bytes a token, those
/// regions `parts` times (a layer split's cards one after another).
struct Pool {
    std::vector<std::unique_ptr<uint8_t[]>> chunks;
    std::unique_ptr<PagePool> pool;
    explicit Pool(int parts = 1) {
        StateLayout layout;
        for (int i = 0; i < parts; ++i) {
            layout.add(RegionKind::fixed, 100);
            layout.add(RegionKind::per_token, 8, 2);
        }
        PagePoolConfig config;
        config.page_bytes = 64;
        config.chunk_bytes = 64 * 64;
        PagePoolHooks hooks;
        hooks.alloc = [this](size_t bytes) { return chunks.emplace_back(new uint8_t[bytes]()).get(); };
        hooks.free = [](void*) {};
        pool = std::make_unique<PagePool>(layout, config, hooks);
        pool->resize(config.chunk_bytes);
    }
    /// Gets each byte of conversation `id`'s blocks, region by region.
    std::vector<uint8_t> get_bytes(ConversationId id) const {
        std::vector<uint8_t> bytes;
        for (size_t r = 0; r < pool->layout().regions.size(); ++r)
            for (const PageId p : pool->conversation_blocks(id, r))
                bytes.insert(bytes.end(), pool->page_address(p), pool->page_address(p) + pool->block_bytes(r));
        return bytes;
    }
};

using File = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

/// Writes conversation `id`'s section of `pool` to a new file at `path`.
bool save(const PagePool& pool, ConversationId id, const std::string& path, std::string& err, RegionSpan span = {}) {
    const File f(std::fopen(path.c_str(), "wb"), &std::fclose);
    return f != nullptr && write_conversation_section(f.get(), pool, id, err, span);
}

/// Reads the file `path` of one section into conversation `id` of `pool`.
bool load(const std::string& path, const PagePool& pool, ConversationId id, std::string& err, RegionSpan span = {}) {
    const File f(std::fopen(path.c_str(), "rb"), &std::fclose);
    return f != nullptr && read_conversation_section(f.get(), pool, id, err, span);
}
}  // namespace

int main() {
    Pool a, b, c;
    check(a.pool->grow_conversation(1, 40).empty() && b.pool->grow_conversation(7, 40).empty() &&
              c.pool->grow_conversation(3, 12).empty(),
          "the conversations get their pages");
    uint8_t n = 1;
    for (size_t r = 0; r < 2; ++r)
        for (const PageId p : a.pool->conversation_blocks(1, r))
            for (uint64_t i = 0; i < a.pool->block_bytes(r); ++i)
                a.pool->page_address(p)[i] = n++;
    std::string err;
    check(copy_conversation(*a.pool, 1, *b.pool, 7, err) && b.get_bytes(7) == a.get_bytes(1),
          "a copy into as many blocks has every byte");
    check(copy_conversation(*a.pool, 1, *c.pool, 3, err), "a copy into fewer blocks is done");
    const std::vector<uint8_t> want = a.get_bytes(1), got = c.get_bytes(3);
    const size_t fixed = a.pool->block_bytes(0);
    check(std::equal(got.begin(), got.begin() + (long) fixed, want.begin()) &&
              std::equal(got.begin() + (long) fixed, got.end(), want.begin() + (long) fixed),
          "a copy into fewer blocks takes the fixed region and the first blocks");
    check(!copy_conversation(*c.pool, 3, *a.pool, 1, err), "a copy into more blocks is refused");

    const std::string path = (std::filesystem::temp_directory_path() / "pool_copy_test.bin").string();
    Pool d, e;
    check(save(*a.pool, 1, path, err), "the file is written");
    check(d.pool->grow_conversation(5, 40).empty() && load(path, *d.pool, 5, err) &&
              d.get_bytes(5) == a.get_bytes(1),
          "the file gives back every byte");
    check(e.pool->grow_conversation(5, 12).empty() && load(path, *e.pool, 5, err) &&
              e.get_bytes(5) == c.get_bytes(3),
          "a place with fewer blocks reads the first ones");
    {
        std::FILE* file = std::fopen(path.c_str(), "w+b");
        check(file != nullptr && write_conversation_section(file, *a.pool, 1, err) &&
                  write_conversation_section(file, *c.pool, 3, err),
              "two page sections share a stream");
        std::rewind(file);
        check(read_conversation_section(file, *d.pool, 5, err) && d.get_bytes(5) == a.get_bytes(1) &&
                  read_conversation_section(file, *e.pool, 5, err) && e.get_bytes(5) == c.get_bytes(3),
              "each page section is read in order");
        std::fclose(file);
    }
    check(save(*a.pool, 1, path, err), "the larger section is written");
    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);
    check(!load(path, *e.pool, 5, err), "truncation in skipped blocks is refused");
    std::filesystem::remove(path);
    check(!load(path, *d.pool, 5, err), "a missing file is refused");

    Pool both(2), back, want20;
    const RegionSpan second{2, 2};
    check(both.pool->grow_conversation(9, 20).empty() &&
              copy_conversation(*a.pool, 1, *both.pool, 9, err, {}, second) &&
              back.pool->grow_conversation(4, 20).empty() && want20.pool->grow_conversation(4, 20).empty() &&
              copy_conversation(*both.pool, 9, *back.pool, 4, err, second) &&
              copy_conversation(*a.pool, 1, *want20.pool, 4, err) && back.get_bytes(4) == want20.get_bytes(4),
          "a card's regions go to their place among every card's and come back");
    check(!copy_conversation(*both.pool, 9, *back.pool, 4, err), "a copy of every card's regions into one card's is "
          "refused");
    check(save(*both.pool, 9, path, err, second) && e.pool->grow_conversation(6, 12).empty() &&
              load(path, *e.pool, 6, err) && e.get_bytes(6) == c.get_bytes(3),
          "a card's regions go to a file of their own");
    std::filesystem::remove(path);

    // the share: 40 bytes of a's fixed region at byte 50 of both's first fixed region, then a's K/V at both's second
    const RegionSpan card_part{1, 1, 0, 0, 40}, whole_part{3, 1, 0, 50, 40};
    Pool again;
    check(copy_conversation(*a.pool, 1, *both.pool, 9, err, card_part, whole_part) &&
              std::equal(a.pool->page_address(a.pool->conversation_blocks(1, 0)[0]),
                         a.pool->page_address(a.pool->conversation_blocks(1, 0)[0]) + 40,
                         both.pool->page_address(both.pool->conversation_blocks(9, 0)[0]) + 50),
          "a share of a fixed region goes to its byte range of another");
    check(again.pool->grow_conversation(2, 20).empty() &&
              copy_conversation(*both.pool, 9, *again.pool, 2, err, whole_part, card_part) &&
              save(*both.pool, 9, path, err, whole_part) && e.pool->grow_conversation(7, 20).empty() &&
              load(path, *e.pool, 7, err, card_part),
          "it comes back, and through a file");
    const std::vector<uint8_t> want_back = again.get_bytes(2), file_back = e.get_bytes(7);
    const std::vector<uint8_t> share = a.get_bytes(1), kv = want20.get_bytes(4);
    check(std::equal(want_back.begin(), want_back.begin() + 40, share.begin()) &&
              std::equal(file_back.begin(), file_back.begin() + 40, share.begin()) &&
              std::equal(want_back.begin() + (long) fixed, want_back.end(), kv.begin() + (long) fixed) &&
              std::equal(file_back.begin() + (long) fixed, file_back.end(), kv.begin() + (long) fixed),
          "the share's bytes and the K/V's are the card's");
    check(!copy_conversation(*a.pool, 1, *both.pool, 9, err, card_part, {3, 1, 0, 50, 30}),
          "shares of different sizes are refused");
    std::filesystem::remove(path);

    RegionSpan packed_from, packed_to;
    packed_from.count = packed_to.count = 0;
    packed_from.ranges = {{0, 4, 12}, {0, 50, 20}};
    packed_to.ranges = {{0, 10, 12}, {0, 70, 20}};
    check(copy_conversation(*a.pool, 1, *both.pool, 9, err, packed_from, packed_to),
          "two packed ranges copy independently within one fixed region");
    for (size_t i = 0; i < packed_from.ranges.size(); ++i) {
        const auto& from = packed_from.ranges[i];
        const auto& to = packed_to.ranges[i];
        const uint8_t* src = a.pool->page_address(a.pool->conversation_blocks(1, 0)[0]) + from.offset;
        const uint8_t* dst = both.pool->page_address(both.pool->conversation_blocks(9, 0)[0]) + to.offset;
        check(std::equal(src, src + from.bytes, dst), "each packed range keeps its own bytes");
    }
    check(save(*both.pool, 9, path, err, packed_to) &&
              load(path, *again.pool, 2, err, packed_from),
          "packed ranges round-trip through a pause file");
    packed_to.ranges = {{0, 120, 20}};
    check(!copy_conversation(*a.pool, 1, *both.pool, 9, err, packed_from, packed_to),
          "a packed range outside its block is refused");
    packed_to.ranges = {{99, 0, 20}};
    check(!save(*both.pool, 9, path, err, packed_to), "an absent region is refused");
    std::filesystem::remove(path);

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pool_copy_test";
    std::filesystem::create_directories(dir);
    const std::string mine = get_pause_file_prefix(dir.string()) + "1.bin";
    const std::filesystem::path gone = dir / "paused-2147483000-1-1.bin", other = dir / "paused-notes.txt";
    for (const std::filesystem::path& f : {std::filesystem::path(mine), gone, other})
        std::fclose(std::fopen(f.string().c_str(), "wb"));
    std::vector<std::string> running;
    check(find_stale_pause_files(dir.string(), &running) == std::vector<std::string>{gone.string()} &&
              running == std::vector<std::string>{mine} && std::filesystem::exists(mine) &&
              std::filesystem::exists(gone) && std::filesystem::exists(other),
          "only files of a dead process are offered for adoption, without deletion; a live one's are named");
#if defined(_WIN32)
    // a file of this process's id from before it began: a process that ran earlier with the same id wrote it
    const std::string mine_id = "paused-" + std::to_string(strata::platform::get_process_id());
    const std::filesystem::path reused = dir / (mine_id + "-1-1.bin"), unstamped = dir / (mine_id + "--1.bin");
    for (const std::filesystem::path& f : {reused, unstamped})
        std::fclose(std::fopen(f.string().c_str(), "wb"));
    std::vector<std::string> stale{gone.string(), reused.string()};
    std::sort(stale.begin(), stale.end());
    check(find_stale_pause_files(dir.string()) == stale,
          "a file whose process id was reused by a later process is offered for adoption; an unstamped one is not");
#endif
    check(find_stale_pause_files((dir / "missing").string()).empty(),
          "a missing pause directory does not fail discovery");
    std::filesystem::remove_all(dir);
    std::printf("pool_copy_test: %d checks passed\n", checks);
    return 0;
}
