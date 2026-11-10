// include/strata/common/serve/paged_engine.hpp - the engine whose conversations live in page pools, several decoding
// at once, for any model that implements PagedModel (paged_model.hpp; docs/MEMORY_MANAGER_PLAN.md, steps M2d to M9
// and D3).  strata-q38 runs it for Qwen3.8 (include/strata/models/qwen38/paged_model.hpp), strata-ds41 for DeepSeek
// V4.1 (include/strata/models/deepseekv41/paged_model.hpp).
//
// It answers the server's line protocol (`--serve` is implied) like `strata --serve --batch N`: GEN, BGEN, STOP,
// BSTOP, QUIT, LIST_CONVERSATIONS, GET_CONTEXT_WINDOWS, GET_MEMORY_USE and CHANGE_SETTINGS (BYIELD is dropped: a
// prompt read never gives way).  Each conversation's state takes page-pool pages as it grows; the experts are in the
// same pool (ExpertPages): a growing conversation takes the least-used experts' pages, and the adaptive tier
// fills the pages conversations left with experts.  Up to `concurrent` conversations decode together, each step one
// pass per group through the model with a row per conversation and the rows left over shared among their guesses
// (strata::spec::Step). Opt-in --batch-groups overlaps groups on separate stages; every wave drains before the
// serve loop can admit, pause, restore or change settings. A prompt is read in passes of its own, a chunk at a time
// between the others' steps. A request continues the conversation that holds the longest start of its prompt,
// reading only the rest; a conversation the prompt leaves before its end goes back to a point it keeps
// (PagedModel::get_rewind_points), and stays paused whole
// as well. A request's `priority=N` key (default 0) selects conversations for a step, higher first; equal priorities
// retain request order. BGEN replies with its admission's T lines, DONE and BADM <slot> <continues>. A request
// waiting outside the selected set may reply with DONE 0 and BADM <slot> 1 before emitting any token; subsequent
// output uses BT <slot> <token> and ends with BDONE as usual. Selected requests keep the first-token handoff.
//
// The RAM is a second page pool (RamExperts): the experts the GPU's pool does not hold, then copies of those it does
// while room allows, and the paused conversations.  The experts in neither are read from the SSD without the file
// cache.  A conversation that leaves its place on the GPU (another request takes the place, or a growing conversation
// needs its pages) is paused, its pages copied to the RAM's pool or to a file on the SSD, and comes back when a
// request continues it.
//
// On a layer split each card runs a range of layers and has a page pool of its own, with the experts of its layers
// and each conversation's state for them.  A paused conversation is one conversation in the RAM's pool (each card's
// regions one after another) or one file per card.  A card with pages to spare holds experts of other cards' layers
// that their cards have no room for, and computes them in one hop (`remote_experts`; auto: where a hop over the card's
// link costs less than the CPU computing an expert).  When a card's memory setting changes, the cut moves if the
// planner says so: between two steps every conversation is paused to RAM, the cards take their new layers, and the
// conversations come back.
//
// CHANGE_SETTINGS (memory_report.hpp) changes, between two steps, each pool's size (`memory_mib.gpu<N>`, per card
// `memory_mib.gpu<N>.<card>` for a later card on the GPU of an earlier one,
// `memory_mib.ram`: a pool takes and gives back memory in chunks of 256 MiB; a shrink evicts the experts of the
// chunks it gives back and moves or pauses the conversations there), `tokens_min`, `tokens_max`, `context`,
// `ssd_pause_mib`, `drafter` (off gives its weights back), `concurrent`, `vram_conversations`, `rows` and `lookahead`.
// The GPU residency count includes idle caches and their fixed regions, and cannot fall below `concurrent`.
// Pool shrinks must accommodate the biggest decoding conversation at the end of its request and the token minimum.
// The combined GPU token budget is independent of page padding. The per-conversation limit cannot exceed it or the
// loaded capacity; reducing it must still leave each active request room for its next token. Conflicting edits
// change nothing.
#pragma once

#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/spec/step.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::serve {

/// The conversation tokens whose pages experts leave free when --tokens-min is not given.
constexpr int64_t kDefaultTokensMin = 8192;
/// The SSD space paused conversations may take when --ssd-pause-mib is not given.
constexpr int64_t kDefaultSsdPauseMib = 8192;
/// The most experts the start-up fill loads into a pool in one step of the serve loop, so that a request waits
/// little for it.
constexpr int kRefillBatch = 16;
/// The time the experts the start-up fill loads into a GPU's pool in one step of the serve loop may take on its link:
/// a slow link takes fewer (get_refill_batch).
constexpr double kRefillBudgetMs = 10.0;
/// The step in which the pools take memory and give it back (PagePoolConfig::chunk_bytes).  An expert may not cross a
/// chunk's end, so each chunk ends in a run too short for any expert: 0.36% of the Strix Halo's GPU pool in chunks of
/// this size (docs/MEMORY_MANAGER_PLAN.md, M2f).
constexpr uint64_t kChunkBytes = 256ull << 20;

/// The paged engine's own options.
struct PagedSettings {
    int64_t tokens_min = kDefaultTokensMin;        ///< --tokens-min
    int64_t tokens_max = -1;                       ///< combined GPU logical tokens; -1: --max-context
    int64_t conversation_context = -1;             ///< --conversation-tokens; -1: loaded capacity or tokens_max
    int64_t vram_conversations = -1;               ///< --vram-conversations; -1: model-aware cached-place allowance
    int64_t conversation_idle_seconds = 66;        ///< --conversation-idle-seconds: age eligible under memory pressure
    /// --decode-share-percent: after each turn of a prompt read, the conversations that decode take steps for this
    /// percentage of the turn's time, so that a long prompt slows them down instead of stopping them; 0: they wait
    /// for the whole prompt.  The old engine's STRATA_BATCH_DECODE_SHARE, in percent.
    int64_t decode_share_percent = 50;
    /// --prompt-turn-ms: the least time one turn of a prompt read lasts (it ends after a whole chunk); 0: one chunk
    /// per turn.  Longer turns switch less often between reading and decoding.
    int64_t prompt_turn_ms = 0;
    /// --gpu-mib, per card of a layer split (one value: every card's); empty: what each GPU has free less what it keeps
    std::vector<int64_t> gpu_mib;
    int64_t ram_mib = -1;                          ///< --ram-mib; -1: get_ram_room
    int64_t ssd_pause_mib = kDefaultSsdPauseMib;   ///< --ssd-pause-mib
    std::string pause_dir;                         ///< --pause-dir; empty: the temporary folder's <program name>
    std::string copy_back = "auto";                ///< --copy-back
    std::string batch_groups = "1";             ///< --batch-groups: live-concurrency divisor or auto.
    std::string remote_experts = "auto";           ///< --remote-experts
    /// --expert-popularity-decay: the factor by which each decode step multiplies the adaptive tier's routing counts;
    /// 0: --adapt-decay's (a factor per 4 steps) to the power 0.25
    double popularity_decay = 0;
    int64_t lookup_chain = 0, lookup_chain_min = 3; ///< Existing lookup-chain length and match floor.
    int64_t suffix_draft = 0; ///< --suffix-draft: minimum repeated context; 0 disables prompt lookup.
    bool drafter = true;                           ///< --drafter on|off: its weights loaded (with --lookahead)
    /// --save-on-shutdown on|off: when the engine ends, its conversations in the GPUs' and the RAM's pools go to the
    /// SSD (by request priority, then the most recently used, up to --ssd-pause-mib) and stay there for the next
    /// start to adopt.
    bool save_on_shutdown = false;
    int64_t step_profile = 0; ///< --step-profile 1 reports host elapsed times for each decode step.
    strata::spec::StepSettings step;               ///< --rows, --lookahead and --concurrent (default --batch)
};

/// Runs the engine over `model`, whose start made `cards` (the GPUs' pools) and `ram` (the RAM's pool: experts and
/// paused conversations, its conversation layout the whole model's regions): carves each conversation's state on each
/// card, then answers the server's commands until QUIT or the end of stdin.  `copy_back`: an expert a GPU's pool
/// evicts is copied to `ram`. Gets the exit status.
int serve_paged_model(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, bool copy_back,
                      const PagedSettings& settings);

// ---- a model's start (paged_start.cpp): what every model's start does before serve_paged_model

/// Removes the engine's own options (PagedSettings: --tokens-min, --gpu-mib, --rows, ...) from `argv` and reads them
/// into `s`; what is left is the model's.  `program` starts its error messages.  Gets -1 to go on, else the exit
/// status.
int take_paged_options(int& argc, char** argv, const std::string& program, PagedSettings& s);
/// Sets the GPU runtime's environment for the engine, before the runtime starts (an explicit setting wins): on HIP on
/// Windows, no resource cache, so that the pools can shrink; when `hops_possible` (stages on separate GPUs) and
/// `s.remote_experts` is not off, more hardware queues.
void set_paged_runtime_env(const PagedSettings& s, bool hops_possible);
/// Measures `card`'s link to RAM (PagedCard::link, and refill_batch for experts of up to `expert_bytes`) and predicts
/// its speed from its multiprocessors and clock (layer_ms); says the link on stderr after `program`.  Call before the
/// card's pool takes the memory the measuring copies need.
void measure_card(PagedCard& card, const std::string& program, uint64_t expert_bytes);
/// Gets GPU `dev`'s chunk allocator. Disable `zeroed` only when the adapter initializes every page before use.
strata::core::ChunkMemory get_gpu_chunk_memory(int dev, bool zeroed = true);
/// Gets the bytes a pool on GPU `dev` takes: `gpu_mib` (--gpu-mib), but no more than `room`; -1: `room` but no more
/// than `most`.  `room` gets what the pool could take: get_gpu_room(dev, keep, 0).
uint64_t get_gpu_pool_bytes(int dev, uint64_t keep, int64_t gpu_mib, uint64_t most, uint64_t& room);
/// Gets the most bytes one conversation of `tokens` tokens in `layout` takes in pages of `page_bytes`: its regions'
/// bytes and each region's last block, which it may fill only in part.
uint64_t get_conversation_bytes(const strata::core::StateLayout& layout, int64_t tokens, uint64_t page_bytes);
/// Gets the chunks' size of a pool of `pool_bytes` in pages of `page_bytes`: kChunkBytes, or the pool when smaller,
/// in whole pages.
uint64_t get_chunk_bytes(uint64_t pool_bytes, uint64_t page_bytes);
/// Says on stderr after `program` what `card`'s open pool took: of `room` it could take, its chunks of `chunk_bytes`,
/// its pages and the conversation tokens kept for the conversations.
void say_gpu_pool(const std::string& program, const PagedCard& card, uint64_t room, uint64_t chunk_bytes);
/// Gets the bytes the RAM's pool and its staging ring take: --ram-mib (`s`), by default get_ram_room(`headroom`) but
/// no more than `most`.
uint64_t get_ram_pool_bytes(const PagedSettings& s, uint64_t headroom, uint64_t most);
/// Gets the RAM's pool's config: the page size of `card`'s pool (a conversation's blocks are alike in both) and chunks
/// of kChunkBytes in whole pages.
strata::core::PagePoolConfig get_ram_pool_config(const PagedCard& card);
/// Gets the order in which the RAM's pool takes experts: those of `profile` (hottest first), then the other experts
/// of the `blob_bytes.size()` layers of `n_expert`.  Its start-up fill skips the experts the GPUs' pools took
/// (ExpertPages::stop_wanting_if), so that RAM holds what the GPUs lack.
std::vector<strata::core::ExpertKey> get_ram_ranking(const std::vector<strata::core::ExpertKey>& profile,
                                                     const std::vector<uint64_t>& blob_bytes, int64_t n_expert);
/// Opens the RAM's page pool `ram` over `files` (the experts on the SSD) and gets whether an expert a GPU's pool evicts
/// is copied back to it (`copy_back`: --copy-back; auto by the speeds of copying from `gpu_bytes`, the start of
/// `card`'s pool's first chunk, and of reading the SSD, measured here).  The pool takes --ram-mib, by default
/// get_ram_room(`headroom`) but no more than every expert and its staging ring of `ring` buffers take; its
/// conversations are `layout`'s, its pages `card`'s (get_ram_pool_config).  It opens empty: the engine's start-up
/// fill brings in the experts in get_ram_ranking's order as far as they fit.  Says on stderr after `program` what it
/// took.
bool open_ram_pool(const std::string& program, const PagedSettings& s, strata::core::ExpertSource& files,
                   uint64_t headroom, int ring, const strata::core::StateLayout& layout, const PagedCard& card,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert,
                   const std::vector<strata::core::ExpertKey>& profile,
                   const uint8_t* gpu_bytes, strata::core::RamExperts& ram, bool& copy_back, std::string& err);

/// Gets why the engine cannot decode with `s` on a model whose steps run at most `max_rows` rows (empty: it can).
std::string get_step_settings_problem(const strata::spec::StepSettings& s, int max_rows);
/// Gets the memory a pool on device `dev` may take besides what it holds: what the GPU has free less `keep`.  On
/// Windows, for a GPU whose memory is the system RAM, what is free is what is free of the share the BIOS gives it
/// (get_share_room, `mapped` the RAM's page-locked pool), not the runtime's free memory: that is the share and what
/// the GPU may borrow beyond it, and HIP takes page-locked host memory off it twice (on the Strix Halo, 8 GiB of
/// hipHostMalloc took 16 GiB off it).  Its memory counts in Windows' commit charge, so no more than get_commit_room
/// either.  Cards on one GPU each see what is free of it.
uint64_t get_gpu_room(int dev, uint64_t keep, uint64_t mapped);
/// Gets the RAM a pool may take by default besides what the pools hold: what is available less `headroom`, and no more
/// than get_commit_room.
uint64_t get_ram_room(uint64_t headroom);

}  // namespace strata::serve
