// A paused conversation's file (pause_file.hpp): the header and one page section per card come back as written, also
// into another cut's places; a file of another model or settings, a damaged or cut-short one, one with more tokens than
// the engine holds and one that is no pause file are refused; and a failed write leaves the file that was there.
#include "strata/common/state/pause_file.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <utility>
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

/// A pool of 64 pages of 64 bytes on host memory, for a conversation of a 100-byte state and 16 bytes a token.
struct Pool {
    std::vector<std::unique_ptr<uint8_t[]>> chunks;
    std::unique_ptr<PagePool> pool;
    Pool() {
        StateLayout layout;
        layout.add(RegionKind::fixed, 100);
        layout.add(RegionKind::per_token, 8, 2);
        PagePoolConfig config;
        config.page_bytes = 64;
        config.chunk_bytes = 64 * 64;
        PagePoolHooks hooks;
        hooks.alloc = [this](size_t bytes) { return chunks.emplace_back(new uint8_t[bytes]()).get(); };
        hooks.free = [](void*) {};
        pool = std::make_unique<PagePool>(layout, config, hooks);
        pool->resize(config.chunk_bytes);
    }
    /// Gives conversation `id` pages for `tokens` tokens, filled with bytes that start at `first`.
    void fill(ConversationId id, int64_t tokens, uint8_t first) {
        check(pool->grow_conversation(id, tokens).empty(), "a conversation gets its pages");
        for (size_t r = 0; r < 2; ++r)
            for (const PageId p : pool->conversation_blocks(id, r))
                for (uint64_t i = 0; i < pool->block_bytes(r); ++i)
                    pool->page_address(p)[i] = first++;
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
}  // namespace

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / "pause_file_test.bin").string();
    const SessionFileIdentity identity{11, 22};
    PauseHeader head;
    head.live = {5, 6, 7, 8};
    head.points = {2, 4};
    head.state.assign(1u << 20, 7);
    head.cut = {12};
    head.images = {{1, 0x123456789abcdef0ull}, {3, 42}};
    head.last_tokens_at = 1700000000;
    Pool first, second;
    first.fill(1, 40, 1);
    second.fill(3, 12, 100);
    std::string err;
    check(write_pause_file(path, identity, head, {{first.pool.get(), 1}, {second.pool.get(), 3}}, err),
          "a file of two cards is written");

    PauseStream file;
    PauseHeader got;
    check(open_pause_file(path, identity, 100, file, got, err) == SessionError::none && got.live == head.live &&
              got.points == head.points && got.state == head.state && got.cut == head.cut &&
              got.images == head.images && got.last_tokens_at == head.last_tokens_at,
          "the header comes back as written");
    Pool first_back, second_back;
    check(first_back.pool->grow_conversation(5, 40).empty() && second_back.pool->grow_conversation(5, 12).empty() &&
              read_conversation_section(file.get(), *first_back.pool, 5, err) &&
              read_conversation_section(file.get(), *second_back.pool, 5, err) &&
              first_back.get_bytes(5) == first.get_bytes(1) && second_back.get_bytes(5) == second.get_bytes(3),
          "each card's pages follow the header in order");
    file.reset();

    check(open_pause_file(path, {11, 23}, 100, file, got, err) == SessionError::invalid && file == nullptr &&
              err.find("another model") != std::string::npos,
          "a file of other settings is refused");
    check(open_pause_file(path, {12, 22}, 100, file, got, err) == SessionError::invalid, "and of another model");
    check(open_pause_file(path, identity, 3, file, got, err) == SessionError::invalid,
          "a file with more tokens than the engine holds is refused");
    check(open_pause_file(path + ".none", identity, 100, file, got, err) == SessionError::io, "a missing file is io");

    const std::string copy = path + ".copy";
    check(copy_pause_file(path, copy, err) && copy_pause_file(path, copy, err) &&
              open_pause_file(copy, identity, 100, file, got, err) == SessionError::none,
          "a durable copy can replace an existing named session");
    file.reset();
    std::filesystem::remove(copy);

    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);
    check(open_pause_file(path, identity, 100, file, got, err) == SessionError::invalid,
          "a file cut short is refused before restoration");
    file.reset();

    for (const auto& [images, activity] : std::vector<std::pair<std::vector<ConversationImageKey>, uint64_t>>{
             {head.images, 0}, {{}, head.last_tokens_at}, {{}, 0}}) {
        head.images = images;
        head.last_tokens_at = activity;
        check(write_pause_file(path, identity, head, {{first.pool.get(), 1}}, err),
              "optional header fields are written");
        check(open_pause_file(path, identity, 100, file, got, err) == SessionError::none &&
                  got.images == images && got.last_tokens_at == activity,
              "image-only, activity-only and legacy headers remain readable");
        file.reset();
    }

    head.images.clear();
    head.last_tokens_at = 0;
    check(write_pause_file(path, identity, head, {{first.pool.get(), 1}}, err), "the file is written again");
    {
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        std::fseek(f, 20, SEEK_SET);
        std::fputc(0x55, f);
        std::fclose(f);
    }
    check(open_pause_file(path, identity, 100, file, got, err) == SessionError::invalid &&
              err.find("damaged") != std::string::npos,
          "a damaged header is refused");

    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        std::fputs("not a pause file at all, only some text", f);
        std::fclose(f);
    }
    check(open_pause_file(path, identity, 100, file, got, err) == SessionError::invalid, "another file is refused");

    check(write_pause_file(path, identity, head, {{first.pool.get(), 1}}, err), "a file is replaced");
    const auto kept = std::filesystem::file_size(path);
    RegionSpan absent;
    absent.ranges = {{99, 0, 20}};
    check(!write_pause_file(path, identity, head, {{first.pool.get(), 1, absent}}, err) &&
              std::filesystem::file_size(path) == kept && !std::filesystem::exists(path + ".tmp"),
          "a failed write keeps the file that was there and leaves no other");
    std::filesystem::remove(path);
    std::printf("pause_file_test: %d checks passed\n", checks);
    return 0;
}
