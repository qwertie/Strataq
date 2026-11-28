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
// `ssd_pause_mib`, `drafter` (off gives its weights back), `draft_rule` (draft_rule.hpp's text), `concurrent`,
// `vram_conversations`, `rows`, `lookahead` and `vram_reserve_mib` (PagedSettings::vram_reserve_mib).
// The GPU residency count includes idle caches and their fixed regions, and cannot fall below `concurrent`.
// Pool shrinks must accommodate the biggest decoding conversation at the end of its request and the token minimum.
// The combined GPU token budget is independent of page padding. The per-conversation limit cannot exceed it or the
// loaded capacity; reducing it must still leave each active request room for its next token. Conflicting edits
// change nothing.
#pragma once

#include "strata/common/experts/expert_files.hpp"
#include "strata/common/experts/layer_split.hpp"
#include "strata/common/experts/ram_experts.hpp"
#include "strata/common/serve/paged_model.hpp"
#include "strata/common/spec/step.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
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
/// The GPU memory in MiB the pools leave unallocated on each GPU for other programs when --vram-reserve-mib is
/// absent. Prompt buffers come from the pools; PagedModel::get_gpu_keep covers model buffers outside them.
/// One pool chunk; measured only on the Strix Halo (docs/ENGINE_NOTES.md).
constexpr int64_t kDefaultVramReserveMib = 256;
/// The smallest prompt chunk the engine lends buffers for, and the step between the chunks it tries
/// (lend_prompt_buffers in paged_engine.cpp).
constexpr int64_t kPromptChunkStep = 256;

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
    /// (live: prefill_chunk) Caps the chunk chosen when borrowing prompt buffers, initially
    /// PagedModel::get_prompt_ceiling() (--prefill-chunk, also spelled --prefill); 0 disables borrowing.
    int64_t prompt_chunk = 0;
    /// --gpu-mib, per card of a layer split (one value: every card's); empty: what each GPU has free less what it keeps
    std::vector<int64_t> gpu_mib;
    int64_t ram_mib = -1;                          ///< --ram-mib; -1: get_ram_room
    /// --vram-reserve-mib (also --reserve-mib, strata-dsv41's name for it): the memory each GPU's pool leaves
    /// unallocated, for other programs and for the engine's own allocations outside the pools (PagedModel::get_gpu_keep
    /// adds what the model knows of these).  A change applies to the next growth (which keeps this much of an integrated
    /// GPU's dedicated memory free, PoolGrowth::spare) and to the largest pool a change may ask for; it resizes nothing
    /// by itself.
    int64_t vram_reserve_mib = kDefaultVramReserveMib;
    int64_t ssd_pause_mib = kDefaultSsdPauseMib;   ///< --ssd-pause-mib
    std::string pause_dir;                         ///< --pause-dir; empty: the temporary folder's <program name>
    std::string copy_back = "auto";                ///< --copy-back (live: copy_back): auto, on or off
    /// --batch-groups (live: batch_groups): how many groups a step's conversations split into, each group running on
    /// a different card of a layer split at once; 0 (also --batch-groups auto): the largest divisor of live
    /// concurrency no greater than the stage count.  A count that does not divide live concurrency runs one group.
    int64_t batch_groups = 1;
    std::string remote_experts = "auto";           ///< --remote-experts
    /// --prompt-order auto|layer|chunk: a prompt of several chunks is read layer by layer over all its chunks, chunk
    /// by chunk through every layer, or as each model's own rule picks (set_paged_runtime_env passes it on).
    std::string prompt_order = "auto";
    /// --expert-popularity-decay: the factor by which each decode step multiplies the adaptive tier's routing counts;
    /// 0: --adapt-decay's (a factor per 4 steps) to the power 0.25.  --expert-half-life (live: expert_half_life)
    /// sets it as the decode steps that halve a count (get_popularity_decay).
    double popularity_decay = 0;
    /// --route-tail-skip (live: route_tail_skip): the routing rank from which an expert missing from the GPUs is
    /// skipped when every token of a verify window ranks it that low (PagedModel::set_route_tail_skip); 0: never;
    /// -1: the model's default.
    int64_t route_tail_skip = -1;
    /// --route-prior-pct (live: route_prior_pct): how much more likely, in percent, picking a token's experts treats
    /// those already on the GPU as being (a logit bonus of ln(1 + pct/100); PagedModel::set_route_prior; changes the
    /// answers); 0: none; -1: the model's default.
    int64_t route_prior_pct = -1;
    /// --draft-confidence-pct and --draft-cumulative-pct (live: draft_confidence_pct, draft_cumulative_pct): the
    /// least confidence, in percent, of each drafted token and of the drafted tokens together for the step to verify
    /// them (spec::DraftLimits); -1: the model's.  A change applies from the next request.
    int64_t draft_confidence_pct = -1, draft_cumulative_pct = -1;
    int64_t lookup_chain = 0, lookup_chain_min = 3; ///< Existing lookup-chain length and match floor.
    int64_t suffix_draft = 0; ///< --suffix-draft: minimum repeated context; 0 disables prompt lookup.
    bool drafter = true;                           ///< --drafter on|off: its weights loaded (with --lookahead)
    /// --draft-rule: which drafts a sampled request's verify rows keep; unset: the model's (PagedModelFacts)
    std::optional<strata::kernels::DraftRule> draft_rule;
    /// --save-on-shutdown on|off: when the engine ends, its conversations in the GPUs' and the RAM's pools go to the
    /// SSD (by request priority, then the most recently used, up to --ssd-pause-mib) and stay there for the next
    /// start to adopt.
    bool save_on_shutdown = false;
    /// --step-profile on|off (also 1|0; live: step_profile): says on stderr how long each decode step's host work
    /// took, by part.
    bool step_profile = false;
    strata::spec::StepSettings step;               ///< --rows, --lookahead and --concurrent (default --batch)

    /// Gets vram_reserve_mib in bytes.
    uint64_t get_vram_reserve_bytes() const { return (uint64_t) vram_reserve_mib << 20; }
    /// Gets card `card`'s --gpu-mib (the last value serves the cards beyond it), -1 when not given.
    int64_t get_gpu_mib(size_t card) const {
        return gpu_mib.empty() ? -1 : gpu_mib[std::min(card, gpu_mib.size() - 1)];
    }
};

/// Runs the engine over `model`, whose start made `cards` (the GPUs' pools) and `ram` (the RAM's pool: experts and
/// paused conversations, its conversation layout the whole model's regions): carves each conversation's state on each
/// card, then answers the server's commands until QUIT or the end of stdin.  `copy_back_gain`: how many times faster
/// copying an expert a GPU's pool evicts to `ram` is than reading it from the SSD (open_ram_pool), which decides
/// whether the copy is made under --copy-back auto (get_copy_back_gain_for). Gets the exit status.
int serve_paged_model(PagedModel& model, PagedCards& cards, strata::core::RamExperts& ram, double copy_back_gain,
                      const PagedSettings& settings);

// ---- a model's start (paged_start.cpp): what every model's start does before serve_paged_model

/// Removes the engine's own options (PagedSettings: --tokens-min, --gpu-mib, --rows, ...) from `argv` and reads them
/// into `s`; what is left is the model's.  `program` starts its error messages.  Gets -1 to go on, else the exit
/// status.
int take_paged_options(int& argc, char** argv, const std::string& program, PagedSettings& s);
/// Sets the GPU runtime's environment for the engine, before the runtime starts (an explicit setting wins): on HIP on
/// Windows, no resource cache, so that the pools can shrink; when `hops_possible` (stages on separate GPUs) and
/// `s.remote_experts` is not off, more hardware queues; the models' prompt-order switches for `s.prompt_order`.
void set_paged_runtime_env(const PagedSettings& s, bool hops_possible);
/// Adds a stage running [lb, le), named gpu<dev> with .<index> for repeated GPUs. Measures the link before pool
/// allocation, reports it after `program`, and sets speed and refill_batch for `expert_bytes`-sized experts.
/// Successful link measurements are reused per GPU.
PagedCard& add_paged_card(PagedCards& cards, int dev, int64_t lb, int64_t le, const std::string& program,
                          uint64_t expert_bytes);
/// Holds allocation limits and diagnostics for a GPU page pool.
struct CardPoolPlan {
    strata::core::PagePoolConfig config;   ///< page and chunk sizes, tokens_min; other fields keep their defaults
    uint64_t bytes = 0;                    ///< what the pool takes
    uint64_t room = 0;                     ///< what the pool could take (get_gpu_room)
    int64_t gpu_mib = -1;                  ///< the card's --gpu-mib, -1 when not given
};
/// Sizes a pool using --gpu-mib, capped by get_gpu_room(dev, keep). The default also caps it at one max_context
/// conversation plus ranking's first max_experts experts, rounded to pages; blob_bytes is indexed by layer.
CardPoolPlan plan_card_pool(const PagedCard& card, const PagedSettings& s, const strata::core::StateLayout& layout,
                            int64_t max_context, uint64_t page_bytes, const std::vector<uint64_t>& blob_bytes,
                            const std::vector<strata::core::ExpertKey>& ranking, int64_t max_experts, uint64_t keep);
/// Gets the memory report's names of a pool's experts (PagePoolConfig::label_expert) for a model of `n_layers` layers,
/// whose drafter's experts, kept as the layers from `n_layers` on, are named as the drafter's.
std::function<strata::core::MemoryLabel(strata::core::ExpertKey, uint64_t)> get_expert_labels(int64_t n_layers);
/// Plans and reports DeepSeek's initial cuts using PagedEngine's planner, one tokens_min conversation, and experts
/// in hottest-first ranked_layers order. Repeated GPUs share free memory unless --gpu-mib sets each stage's pool.
/// Returns no cuts if planning fails.
std::vector<int64_t> plan_start_split(const std::vector<int>& devices, const PagedSettings& s, uint64_t keep,
                                      const strata::core::LayerSizes& sizes, std::vector<int32_t> ranked_layers,
                                      const std::string& program);
/// Gets GPU `dev`'s allocator of chunks of `chunk_bytes` (get_chunk_bytes).  On a GPU with virtual memory (vmm.hpp)
/// the chunks are mapped back to back in one address range as large as the GPU's memory (ChunkMemory::contiguous), so
/// that a pool's runs may cross them; else each is an allocation of its own.  On Windows, an integrated GPU's chunks
/// tell whether they went to shared memory instead of the dedicated memory (ChunkMemory::is_outside), and with virtual
/// memory a chunk that does not fit there whole is mapped from smaller pieces (take_chunk_in_pieces).  Disable
/// `zeroed` only when the adapter initializes every page before use.
strata::core::ChunkMemory get_gpu_chunk_memory(int dev, uint64_t chunk_bytes, bool zeroed = true);
/// Gets how full integrated GPU `dev`'s dedicated memory (the share the BIOS gives it) was, every program's bytes in
/// it together, when one of get_gpu_chunk_memory's chunks last went to shared memory instead; before any did, the
/// share's size.  No figure Windows reports gives the real level, and it depends on the allocations' sizes: on the
/// Strix Halo (Windows 11, a 63.83 GiB share, 2026-11-21, two rounds) a test program held 54.50 GiB with 256 MiB
/// pieces and 58.06-58.08 GiB with 256 then 16 MiB pieces. No value for another GPU, or on Linux.
std::optional<uint64_t> get_share_top(int dev);
/// Reports whether a chunk from get_gpu_chunk_memory has reached integrated GPU `dev`'s shared memory,
/// establishing a measured dedicated-memory limit for get_share_top.
bool has_found_share_end(int dev);
/// Estimates the room below get_share_top after all programs' current use of integrated GPU `dev`'s dedicated memory.
/// Returns zero when full, or no value if the Windows statistics are unavailable.
std::optional<uint64_t> get_dedicated_room(int dev);
/// While it lives, holds `bytes` (rounded up to kPieceBytes) of integrated GPU `dev`'s free dedicated memory, or as
/// much as fits, with placeholder allocations (those that land there, until one does not), so that a pool growing into shared memory
/// (PoolGrowth::outside_ok) leaves that much of the dedicated memory to the rest instead of filling it first.  Holds
/// nothing on another GPU, or on Linux.  Used by PagedEngine::resize_gpu.
class DedicatedHold {
public:
    DedicatedHold(int dev, uint64_t bytes);
    ~DedicatedHold();
    DedicatedHold(const DedicatedHold&) = delete;
    DedicatedHold& operator=(const DedicatedHold&) = delete;

private:
    int dev_;
    std::vector<void*> pieces_;   ///< the placeholders, each kPieceBytes
};
/// Gets the most bytes one conversation of `tokens` tokens in `layout` takes in pages of `page_bytes`: its regions'
/// bytes and each region's last block, which it may fill only in part.
uint64_t get_conversation_bytes(const strata::core::StateLayout& layout, int64_t tokens, uint64_t page_bytes);
/// Gets the chunks' size of a pool of `pool_bytes` in pages of `page_bytes`: kChunkBytes, or the pool when smaller, in
/// whole pages and, with virtual memory, whole units of its granularity.
uint64_t get_chunk_bytes(uint64_t pool_bytes, uint64_t page_bytes);
/// Reports the open pool's allocation and conversation reserve after `program`, warning if --gpu-mib exceeds room.
void say_gpu_pool(const std::string& program, const PagedCard& card, const CardPoolPlan& plan);
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
/// Opens the RAM's page pool `ram` over `files` (the experts on the SSD) and gets how many times faster copying an
/// expert a GPU's pool evicts back to it is than reading it from the SSD (`copy_back_gain`, get_copy_back_gain: by the
/// speeds of copying from `gpu_bytes`, the start of `card`'s pool's first chunk, and of reading the SSD, measured
/// here, whatever --copy-back says).  The pool takes --ram-mib, by default
/// get_ram_room(`headroom`) but no more than every expert and its staging ring of `ring` buffers take; its
/// conversations are `layout`'s, its pages `card`'s (get_ram_pool_config), its chunks at least the `prompt_bytes`
/// the model's prompt paths borrow from it at most (PagedModel::get_prompt_buffers).  It opens with one chunk, and
/// serve_paged_model grows it to the rest after the model's own memory is taken.  It opens
/// empty: the engine's start-up fill brings in the experts in get_ram_ranking's order as far as they fit.  Says on
/// stderr after `program` what it took.
bool open_ram_pool(const std::string& program, const PagedSettings& s, strata::core::ExpertFiles& files,
                   uint64_t headroom, int ring, const strata::core::StateLayout& layout, const PagedCard& card,
                   const std::vector<uint64_t>& blob_bytes, int64_t n_expert,
                   const std::vector<strata::core::ExpertKey>& profile, const uint8_t* gpu_bytes,
                   uint64_t prompt_bytes, strata::core::RamExperts& ram, double& copy_back_gain, std::string& err);
/// Gets the gain is_copy_back_worth weighs under --copy-back `mode`: huge for on, 0 for off, the `measured` one
/// (open_ram_pool's) for auto.
double get_copy_back_gain_for(const std::string& mode, double measured);
/// Gets the factor per decode step that halves the experts' routing counts in `half_life` steps; 0 gets 1, which
/// keeps them.  PagedSettings::popularity_decay's other unit.
double get_popularity_decay(int64_t half_life);
/// Gets the decode steps in which the factor `decay` per step halves a routing count, rounded to the live protocol's
/// range (1 to INT32_MAX); 0 for 1, when counts never fade.
int64_t get_half_life(double decay);

/// Gets why the engine cannot decode with `s` on a model whose steps run at most `max_rows` rows (empty: it can).
std::string get_step_settings_problem(const strata::spec::StepSettings& s, int max_rows);
/// Gets the memory a pool on device `dev` may take besides what it holds: what the GPU has free less `keep`.  On
/// Windows, for a GPU whose memory is the system RAM, what is free is an estimate of what its dedicated memory still
/// takes (get_share_top less every program's bytes there now), not the runtime's free memory: that is the share and
/// what the GPU may borrow beyond it, more slowly, and HIP takes page-locked host memory off it twice (on the Strix
/// Halo, 8 GiB of hipHostMalloc took 16 GiB off it); with `past_dedicated`, the runtime's free memory, shared memory
/// included, so HIP's double counting can underestimate room in that case. GPU capacity is treated independently
/// of Windows' commit charge, even on the Strix Halo where GPU allocations count toward it and can exhaust it.
/// Cards on one GPU each see what is free of it.
uint64_t get_gpu_room(int dev, uint64_t keep, bool past_dedicated = false);
/// Gets the RAM a pool may take by default besides what the pools hold: what is available less `headroom`, and no more
/// than get_commit_room.
uint64_t get_ram_room(uint64_t headroom);

}  // namespace strata::serve
