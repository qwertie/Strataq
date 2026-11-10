// src/common/serve/paged_start.cpp - what every model's start does before the paged engine runs it (paged_engine.hpp,
// "a model's start"): the engine's own options, the runtime's environment, the cards' links, the pools' sizes.
#include "strata/common/serve/paged_engine.hpp"

#include "strata/base/gpu/links.hpp"
#include "strata/base/gpu/on_device.hpp"
#include "strata/base/gpu/runtime.hpp"
#include "strata/base/platform/env.hpp"
#include "strata/common/experts/card_rules.hpp"
#include "strata/common/experts/split_rules.hpp"
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace strata::serve {

/// The bytes each way that measure a GPU's link to RAM at start: the old engine's probe's burst (probe_pcie_h2d_gbps
/// in src/models/qwen38/model.cpp), about 10 ms on PCIe 4.0 x16 and 90 ms on 3.0 x4.
constexpr size_t kLinkProbeBytes = 256ull << 20;

int take_paged_options(int& argc, char** argv, const std::string& program, PagedSettings& s) {
    struct Own {
        const char* name;
        int64_t* value;
        std::string* text = nullptr;   ///< instead of `value`, for a word
    };
    int64_t rows = s.step.rows, lookahead = -1, concurrent = -1, mtp_rows = 0;   // -1: chosen after parsing
    std::string drafter = "on", save_on_shutdown = "off", gpu_mib, popularity_decay;
    const Own own[] = {{"--tokens-min", &s.tokens_min},       {"--tokens-max", &s.tokens_max},
                       {"--conversation-tokens", &s.conversation_context},
                       {"--vram-conversations", &s.vram_conversations},
                       {"--conversation-idle-seconds", &s.conversation_idle_seconds},
                       {"--decode-share-percent", &s.decode_share_percent},
                       {"--prompt-turn-ms", &s.prompt_turn_ms},
                       {"--gpu-mib", nullptr, &gpu_mib},      {"--ram-mib", &s.ram_mib},
                       {"--ssd-pause-mib", &s.ssd_pause_mib}, {"--rows", &rows},
                       {"--mtp-max-t", &mtp_rows},            {"--lookup-chain", &s.lookup_chain},
                       {"--lookup-chain-min", &s.lookup_chain_min},
                       {"--suffix-draft", &s.suffix_draft},    {"--lookahead", &lookahead},
                       {"--concurrent", &concurrent}, {"--batch-groups", nullptr, &s.batch_groups},
                       {"--pause-dir", nullptr, &s.pause_dir}, {"--copy-back", nullptr, &s.copy_back},
                       {"--drafter", nullptr, &drafter},      {"--remote-experts", nullptr, &s.remote_experts},
                       {"--save-on-shutdown", nullptr, &save_on_shutdown},
                       {"--expert-popularity-decay", nullptr, &popularity_decay},
                       {"--step-profile", &s.step_profile}};
    const char* const name = program.c_str();
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const Own* o = std::find_if(std::begin(own), std::end(own),
                                    [&](const Own& x) { return std::strcmp(argv[i], x.name) == 0; });
        if (o == std::end(own)) {
            argv[out++] = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s: %s needs a value\n", name, o->name);
            return 2;
        }
        if (o->text != nullptr)
            *o->text = argv[++i];
        else
            *o->value = std::atoll(argv[++i]);
    }
    argc = out;
    if (s.step_profile != 0 && s.step_profile != 1) {
        std::fprintf(stderr, "%s: --step-profile takes 0 or 1\n", name);
        return 2;
    }
    s.step.rows = (int) std::clamp<int64_t>(rows, 0, INT32_MAX);
    s.step.lookahead = (int) std::clamp<int64_t>(lookahead, -1, INT32_MAX);
    s.step.concurrent = (int) std::clamp<int64_t>(concurrent, -1, INT32_MAX);   // -1: --batch, set once it is read
    s.drafter = drafter == "on";
    s.save_on_shutdown = save_on_shutdown == "on";
    s.step.model_lookahead = mtp_rows > 0 ? (int) std::min<int64_t>(mtp_rows - 1, 7) : -1;
    s.suffix_draft = std::clamp<int64_t>(s.suffix_draft, 0, INT32_MAX);
    s.decode_share_percent = std::clamp<int64_t>(s.decode_share_percent, 0, INT32_MAX);
    s.prompt_turn_ms = std::clamp<int64_t>(s.prompt_turn_ms, 0, INT32_MAX);
    s.lookup_chain = std::clamp<int64_t>(s.lookup_chain, 0, 7);
    s.lookup_chain_min = std::clamp<int64_t>(s.lookup_chain_min, 3, INT32_MAX);
    if (s.batch_groups != "auto" && (s.batch_groups.empty() ||
        s.batch_groups.find_first_not_of("0123456789") != std::string::npos ||
        std::strtoull(s.batch_groups.c_str(), nullptr, 10) < 1 ||
        std::strtoull(s.batch_groups.c_str(), nullptr, 10) > INT32_MAX)) {
        std::fprintf(stderr, "%s: --batch-groups takes auto or a positive integer\n", name);
        return 2;
    }
    if (s.copy_back != "auto" && s.copy_back != "on" && s.copy_back != "off") {
        std::fprintf(stderr, "%s: --copy-back takes auto, on or off\n", name);
        return 2;
    }
    if (!popularity_decay.empty() &&
        !((s.popularity_decay = std::strtod(popularity_decay.c_str(), nullptr)) > 0 && s.popularity_decay <= 1)) {
        std::fprintf(stderr, "%s: --expert-popularity-decay takes a factor above 0 and at most 1\n", name);
        return 2;
    }
    if (s.remote_experts != "auto" && s.remote_experts != "on" && s.remote_experts != "off") {
        std::fprintf(stderr, "%s: --remote-experts takes auto, on or off\n", name);
        return 2;
    }
    for (const auto& [flag, word] : {std::pair{"--drafter", &drafter}, {"--save-on-shutdown", &save_on_shutdown}})
        if (*word != "on" && *word != "off") {
            std::fprintf(stderr, "%s: %s takes on or off\n", name, flag);
            return 2;
        }
    if (std::string err; !gpu_mib.empty() && !parse_i64_list(gpu_mib.c_str(), s.gpu_mib, err)) {
        std::fprintf(stderr, "%s: --gpu-mib takes MiB, one per card of a layer split: %s\n", name, err.c_str());
        return 2;
    }
    return -1;
}

void set_paged_runtime_env(const PagedSettings& s, bool hops_possible) {
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // ROCclr on Windows keeps freed memory in its resource cache and gives none of it back (on the Strix Halo, 8 GiB
    // freed with hipHostFree or hipFree stayed taken from Windows' RAM), so the pools could not shrink
    strata::platform::set_env("GPU_RESOURCE_CACHE_SIZE", "0", false);
#endif
    // A hop needs a free queue beside the owner's spinning verify window. Extra queues are requested only for
    // separate GPUs: same-GPU stages cannot hop (get_hop_refusal). Whether separate NVIDIA cards need the extra
    // connections remains for the NVIDIA session's explicit-default-queues check to measure.
    if (s.remote_experts != "off" && hops_possible) {
        strata::platform::set_env("GPU_MAX_HW_QUEUES", "16", false);
        strata::platform::set_env("CUDA_DEVICE_MAX_CONNECTIONS", "32", false);
    }
}

void measure_card(PagedCard& card, const std::string& program, uint64_t expert_bytes) {
    const strata::core::LinkSpeed link = card.link = strata::core::measure_host_link(card.dev, kLinkProbeBytes);
    card.refill_batch = strata::core::get_refill_batch(link, expert_bytes, kRefillBudgetMs, kRefillBatch);
    std::fprintf(stderr, "%s: %s's link to RAM: %.1f GB/s to the GPU, %.1f GB/s back, %.3f ms a round trip\n",
                 program.c_str(), card.id.c_str(), link.to_gbps, link.from_gbps, link.latency_ms);
    card.layer_ms = strata::program::split_rules::get_layer_ms(strata::core::get_card_speed(card.dev));
}

strata::core::ChunkMemory get_gpu_chunk_memory(int dev, bool zeroed) {
    strata::core::ChunkMemory memory;
    memory.alloc = [dev, zeroed](uint64_t n) {
        const strata::core::OnDevice on(dev);
        // Adapters initialize pages on use; touching free runs can page active weights out under WDDM.
        return (uint8_t*) (zeroed ? strata::core::gpu_alloc_zeroed((size_t) n) :
                                   strata::core::gpu_alloc((size_t) n));
    };
    memory.free = [dev](uint8_t* p) {
        const strata::core::OnDevice on(dev);
        strata::core::gpu_free(p);
    };
    memory.copy = [dev](uint8_t* to, const uint8_t* from, uint64_t n) {
        const strata::core::OnDevice on(dev);
        return strata::core::gpu_copy(to, from, (size_t) n);
    };
    return memory;
}

uint64_t get_gpu_pool_bytes(int dev, uint64_t keep, int64_t gpu_mib, uint64_t most, uint64_t& room) {
    room = get_gpu_room(dev, keep, 0);
    return gpu_mib >= 0 ? std::min((uint64_t) gpu_mib << 20, room) : std::min(room, most);
}

uint64_t get_conversation_bytes(const strata::core::StateLayout& layout, int64_t tokens, uint64_t page_bytes) {
    uint64_t bytes = layout.bytes_for(tokens);
    for (const strata::core::StateRegion& r : layout.regions)
        bytes += strata::core::choose_block_pages(r, page_bytes) * page_bytes;
    return bytes;
}

uint64_t get_chunk_bytes(uint64_t pool_bytes, uint64_t page_bytes) {
    return std::max<uint64_t>(std::min(kChunkBytes, pool_bytes) / page_bytes, 1) * page_bytes;
}

void say_gpu_pool(const std::string& program, const PagedCard& card, uint64_t room, uint64_t chunk_bytes) {
    const strata::core::PagePool& pool = card.experts.pool();
    std::fprintf(stderr, "%s: the page pool of %s (layers %lld-%lld): %.2f GiB (of %.2f GiB it could take) in chunks "
                         "of %.0f MiB, pages of %llu bytes, for the experts and the conversations, %lld tokens of it "
                         "kept for the conversations\n",
                 program.c_str(), card.id.c_str(), (long long) card.lb, (long long) card.le - 1,
                 (double) pool.report().total.bytes / 1073741824.0, (double) room / 1073741824.0,
                 (double) chunk_bytes / 1048576.0, (unsigned long long) pool.page_bytes(),
                 (long long) pool.tokens_min());
}

uint64_t get_ram_pool_bytes(const PagedSettings& s, uint64_t headroom, uint64_t most) {
    return s.ram_mib < 0 ? std::min(get_ram_room(headroom), most) : (uint64_t) s.ram_mib << 20;
}

strata::core::PagePoolConfig get_ram_pool_config(const PagedCard& card) {
    strata::core::PagePoolConfig config;
    config.page_bytes = card.experts.pool().page_bytes();
    config.chunk_bytes = kChunkBytes / config.page_bytes * config.page_bytes;
    return config;
}

std::vector<strata::core::ExpertKey> get_ram_ranking(const std::vector<strata::core::ExpertKey>& profile,
                                                     const std::vector<uint64_t>& blob_bytes, int64_t n_expert) {
    const auto at = [&](int64_t l, int64_t e) { return (size_t) (l * n_expert + e); };
    std::vector<strata::core::ExpertKey> ranking = profile;
    std::vector<uint8_t> ranked(blob_bytes.size() * (size_t) n_expert, 0);
    for (const strata::core::ExpertKey& k : profile)
        ranked[at(k.layer, k.expert)] = 1;
    for (int32_t l = 0; l < (int32_t) blob_bytes.size(); ++l)
        for (int32_t e = 0; e < (int32_t) n_expert; ++e)
            if (!ranked[at(l, e)])
                ranking.push_back({l, e});
    return ranking;
}

bool open_ram_pool(const std::string& program, const PagedSettings& s, strata::core::ExpertSource& files,
                   uint64_t headroom, int ring, const strata::core::StateLayout& layout, const PagedCard& card,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert,
                   const std::vector<strata::core::ExpertKey>& profile, const uint8_t* gpu_bytes,
                   strata::core::RamExperts& ram, bool& copy_back, std::string& err) {
    uint64_t every = 0;
    for (const uint64_t b : blob_bytes)
        every += b * (uint64_t) n_expert;
    const uint64_t bytes = get_ram_pool_bytes(
        s, headroom, every + (uint64_t) ring * *std::max_element(blob_bytes.begin(), blob_bytes.end()));
    const auto open_t0 = std::chrono::steady_clock::now();
    if (!ram.open(files, bytes, ring, layout, get_ram_pool_config(card), blob_bytes, n_expert,
                  get_ram_ranking(profile, blob_bytes, n_expert), err))
        return false;
    const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - open_t0).count();
    double ssd = 0, back = 0;
    // (no more than `gpu_bytes`' chunk holds: a small GPU pool's chunk is smaller than the staging ring)
    const uint64_t timed = std::min(ram.ring_bytes(), card.experts.pool().chunk_bytes());
    // the best of three: on the Strix Halo single copies measured 3.3 to 57 GB/s, single SSD reads 0.3 to 2.4 GB/s
    for (int round = 0; round < 3; ++round) {
        ssd = std::max(ssd, ram.time_ssd_reads(round));
        back = std::max(back, ram.time_copy_in(gpu_bytes, timed));
    }
    copy_back = s.copy_back == "on" || (s.copy_back == "auto" && strata::core::is_copy_back_worth(back, ssd));
    std::fprintf(stderr, "%s: the RAM's page pool: %.2f GiB%s, opened in %.1f s; it takes experts while the engine "
                         "serves, and an expert it does not hold is read from the SSD without the file cache.  An "
                         "expert the GPU evicts is %s (copying it back: %.1f GB/s, reading the SSD: %.1f GB/s; "
                         "--copy-back %s)\n",
                 program.c_str(), (double) (ram.pages().pool().report().total.bytes) / 1073741824.0,
                 ram.is_page_locked() ? ", page-locked" : "", open_s, copy_back ? "copied back to RAM" : "dropped",
                 back, ssd, s.copy_back.c_str());
    return true;
}

}  // namespace strata::serve
