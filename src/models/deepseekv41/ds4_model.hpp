// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_model.hpp - DeepSeek-V4.1-Flash on Maya: an EXL3 model (tools/ds4/pack.py's pack beside its files) over
// the machine's GPUs, layer-split, with the routed experts that do not fit in VRAM on the CPU lane.
//
// Windows of up to 16 tokens through every layer (a prompt in windows, a decode step a window of one): each query of a
// window sees exactly what the reference's decode path gives it (DeepSeek's inference/model.py at start_pos > 0) -
// its own 128-token window, the compressed groups completed by its position.  The state a token leaves is the
// 128-token window ring of every layer, the compressors' partial groups, the compressed KV and index keys of the KV
// source layers (append-only), and the Engram n-gram history; checkpoint() / restore() keep the first two (the rest is
// truncated by position) so a conversation can go back to the end of its last prompt.
#pragma once

#include "strata/common/experts/expert_files.hpp"
#include "strata/common/kernels/draft_rule.hpp"
#include "strata/common/serve/memory_report.hpp"
#include "strata/common/serve/request.hpp"
#include "strata/models/deepseekv41/conversation_layout.hpp"
#include "strata/models/deepseekv41/conversation_state.hpp"
#include "strata/models/deepseekv41/kv_rows.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
class ExpertSource;
}

namespace strata::ds4 {

/// The experts the CPU lane reads at a time from the caller's pool (Options::expert_pages): the staging buffers that
/// pool must have for the experts it does not hold.
constexpr int kCpuLaneExperts = 8;

struct Options {
    std::string pack;                 // <model>/maya-ds4
    int64_t max_context = 32768;
    std::vector<int> gpus;            // empty: every visible GPU
    int64_t reserve_mib = 600;        // VRAM left free on each GPU
    double vram_experts = 1.0;        // the share of the expert budget to use (tests: less, so the CPU lane works)
    // the CPU lane's threads: 0 a thread a physical core, or where the host offers it the performance cores or one
    // NUMA node's cores when measured faster at load; n: at most n, a thread a core
    int cpu_threads = 0;
    std::string profile;              // expert hit counts (read to place the hottest experts in VRAM, written back)
    // --profile-readonly: read the profile to place the experts, never write it back.  Without this the run's routing
    // counts go into the file, so the next run places experts differently and greedy output / KL move between runs.
    bool profile_readonly = false;
    bool expert_cache = true;         // --no-expert-cache: false (strata-ds41 then makes no adaptive swaps)
    // --mmap-experts on / off / auto: 1, 0, -1.  The model reads it only for the prompt chunk's default (0 on a GGUF
    // pack: prefill_chunk_default's copied experts); the caller's pool holds the experts either way
    int mmap_experts = -1;
    // dspark_auto: the drafter is loaded only where the routed experts fit the GPUs (more than 1.5x what they have, the verify rounds
    // read 3.3x the bytes of a token and measured no faster than plain decoding, and the drafter's 7 GB of experts take the VRAM the
    // trunk's hottest would have had); --dspark on / off / none says it outright
    // experts a word takes: -1 the rule (5 where the cards hold 16 GiB or less in all, else the model's 6), N the router's top N at
    // their weights among all 6, not renormalised (--experts 6: full quality); ds4_experts_rule.hpp
    int experts = -1;
    bool dspark_auto = true;
    bool dspark = true;              // load the DSpark drafter (the checkpoint's mtp.* blocks) for speculative decoding
    int dspark_block = 0;             // at most this many drafts a round (--dspark-block N; 0: the pack's block size)
    float dspark_conf = 0.5f;         // a draft is verified while its confidence (sigmoid) is at least this
    // tensor parallelism: runs of tp GPUs of one kind (the same memory) take the same layers, each a share of the
    // heads, of the output projection and of every expert's FFN, summed through host-mapped memory (a GPU unlike the
    // next stays alone); tp > 1 runs with threads (--tp 1 --threads off: one host thread, layer split only)
    int tp = 2;
    bool threads = false;             // a host thread per GPU enqueues its layers (the hand-offs host-mapped)
    // prompts in chunks of this many tokens through each layer at once (threads): the GEMMs on the tensor cores, the
    // CPU lane's experts streamed to the GPUs a chunk at a time instead of computed on the host; 0: windows of 16;
    // -1: 2048, or 8192 with the experts read in place (every chunk streams them all from the files)
    int prefill_chunk = -1;
    bool vision = true;               // the vision tower (the checkpoint's vision.* / aligner.*), when it has one
    /// Holds the routed experts (the memory manager's pools; required): the GPU computes those set_expert_addresses
    /// names, the CPU lane and the prompt path read the others from this pool (a RAM page pool over the SSD).
    /// Requires layer-parallel stages, one GPU each.
    strata::core::ExpertSource* expert_pages = nullptr;
    /// Chooses trunk cuts from the geometry and expert bytes before loading dense weights; false sets the error.
    /// strata-ds41 supplies it with a layer split; empty on one GPU (every layer on it).
    std::function<bool(const Geometry&, const std::vector<uint64_t>&, std::vector<int64_t>&, std::string&)> plan_layers;
};

using strata::core::ExpertFileLayout;

/// Reads comma- or space-separated integers from `s`, stopping at the first that is not one.
std::vector<int> parse_ids(const std::string& s);
/// Reads option `name` of the model's own (those Options holds: --pack, --gpus, --max-context, --dspark, ...) into
/// `o`, its value from `next` (which exits when there is none); false when `name` is not one of them.  `drafting`
/// gets whether --dspark lets the drafter draft (off: loaded, not used).
bool read_option(const std::string& name, const std::function<std::string(const char*)>& next, Options& o,
                 bool& drafting);

/// A picture in the prompt: its span's rows (every position of the span carries the image token), BF16 [span][dim]
struct Image {
    int64_t start = 0;
    std::shared_ptr<const std::vector<uint16_t>> rows;
};

struct Stats {
    int64_t tokens = 0;
    int64_t experts = 0, cpu_experts = 0;   // routed experts run, and how many of them on the CPU lane
    double cpu_ms = 0.0, gpu_wait_ms = 0.0;   // the CPU lane's time; the host's waits for the routing
    double layer_ms = 0.0, head_ms = 0.0, engram_ms = 0.0;   // the layers' host time in all; the head's wait; Engram reads
    int64_t eg_rows = 0, eg_hits = 0, eg_reads = 0;   // Engram rows asked for, served without a read, read from the files
    double eg_io_ms = 0.0;                            // the windows' Engram reads, issue to last done (mostly hidden)
    int64_t eg_cached = 0, eg_cache_cap = 0;          // the row cache's rows held, its capacity
    int64_t skipped = 0;              // routed experts STRATA_DS4_DISK_SKIP_W / RAM_SKIP_W or the tail skip left out
    int skip_on = -1;                 // DISK_SKIP_AUTO: 1 skipping, 0 off (warm), -1 not in use
    int64_t skip_switches = 0;        // ... the times it switched
    double skip_cold = 0.0;           // ... the cold share over its window (while skipping)
    double prof_ms[6] = {};          // DS4_PROF: the GPU's time in decode by stage (between layers, attention, routing +
    int64_t prof_tokens = 0;         // shared expert, experts' gate / up, the wait for the CPU lane, down + post)
    int64_t drafts_offered = 0, drafts_accepted = 0, drafts_rounds = 0;
    // per draft position (the 1st .. 8th of a round): verified with all the drafts before it accepted, and accepted
    int64_t draft_reached[8] = {}, draft_kept[8] = {};
    double draft_ms = 0.0, verify_ms = 0.0;   // the drafter's time, the verification windows' time
};

/// Formats model-wide work per token fed between snapshots for PagedDeepSeek's request-end log. Concurrent
/// conversations, including their prompts, contribute too; these are not timings of an individual request.
inline std::string describe_decode_counters(const Stats& a, const Stats& b) {
    const double n = (double) std::max<int64_t>(1, b.tokens - a.tokens);
    char text[640];
    int at = std::snprintf(text, sizeof text,
        "decode counters (model-wide interval, includes concurrent work): %lld tokens; "
        "ms a token: layers %.2f (routing waits %.2f, CPU lane %.2f), head wait %.2f, "
        "Engram %.2f; experts a token %.1f, CPU lane %.2f",
        (long long) (b.tokens - a.tokens), (b.layer_ms - a.layer_ms) / n, (b.gpu_wait_ms - a.gpu_wait_ms) / n,
        (b.cpu_ms - a.cpu_ms) / n, (b.head_ms - a.head_ms) / n, (b.engram_ms - a.engram_ms) / n,
        (double) (b.experts - a.experts) / n, (double) (b.cpu_experts - a.cpu_experts) / n);
    if (const double pt = (double) (b.prof_tokens - a.prof_tokens); pt > 0 && at > 0 && at < (int) sizeof text)
        std::snprintf(text + at, sizeof text - (size_t) at, "; GPU (DS4_PROF) between %.2f attention %.2f routing %.2f "
                      "gate/up %.2f CPU wait %.2f down %.2f", (b.prof_ms[0] - a.prof_ms[0]) / pt,
                      (b.prof_ms[1] - a.prof_ms[1]) / pt, (b.prof_ms[2] - a.prof_ms[2]) / pt,
                      (b.prof_ms[3] - a.prof_ms[3]) / pt, (b.prof_ms[4] - a.prof_ms[4]) / pt,
                      (b.prof_ms[5] - a.prof_ms[5]) / pt);
    return text;
}

class Model {
public:
    Model();
    ~Model();
    /// Loads the dense weights, the embedding table and the Engram's reader, without the conversation's state
    /// (use_state_memory) or the routed experts (set_expert_addresses), which take the GPU memory left after it.
    bool load(const Options& o, std::string& err);
    /// Allocates the prompt path's buffers and the layers' expert tables (set_expert_addresses fills them) and starts
    /// the CPU lane's threads.  The model runs from then on.
    bool place_experts(std::string& err);

    /// Gets the model's shapes (load's).
    const Geometry& geometry() const;
    /// Gets the resolved number of routed experts used by a trunk token.
    int get_experts_per_token() const;
    /// Gets the bytes of one routed expert of each layer: the trunk's, then the DSpark blocks' when they are loaded.
    std::vector<uint64_t> get_expert_bytes() const;
    /// Gets where each layer's routed experts (get_expert_bytes' layers) lie in the model's files (GGUF or EXL3).
    std::vector<ExpertFileLayout> get_expert_files() const;
    /// Gets the paths of the model's files, which ExpertFileLayout::file numbers.
    std::vector<std::string> get_file_paths() const;
    /// Gets layer `layer`'s experts (a trunk layer's or a DSpark block's) hottest first by the expert profile's
    /// routings so far, then by index.
    std::vector<int> get_heat_order(int layer) const;
    /// (Options::expert_pages) Makes the GPU compute the experts of every layer (get_expert_bytes') at device addresses
    /// `at` (layer * n_exp + expert, n_exp the trunk's; null: the CPU lane's) from the next feed on; false (and `err`)
    /// when the tables could not be written. `stage` selects one device's layers, -1 all layers. The DSpark blocks'
    /// experts must all have an address when it drafts.
    bool set_expert_addresses(const std::vector<const uint8_t*>& at, std::string& err, int stage = -1);
    /// Gets each stage's trunk interval, in device order.
    std::vector<int64_t> get_layer_cuts() const;
    /// Changes a stage's interval, reloading moved dense tensors when the physical GPU changes.
    bool set_stage_layers(int stage, int lb, int le, std::string& err);
    /// Gets each trunk layer's device memory that set_stage_layers moves with it: its dense tensors and scratch.
    std::vector<uint64_t> get_layer_dense_bytes() const;
    /// Opens a stage's compute buffers for experts lent by the shared paged engine.
    bool attach_expert_hop(int stage, std::function<const uint8_t*(int, int)> find, std::string& err);
    /// Publishes the shared planner's per-layer destination card, or null when hopping is off.
    void set_hop_holders(const int32_t* holders);
    /// Prints counts of work completed on the other cards.
    void report_hops() const;
    /// Measures one routed expert from RAM for the shared auto-hop decision.
    double measure_cpu_expert_ms();
    /// (Options::expert_pages) Gets each trunk expert's routings in windows since the caller last changed them (layer
    /// * n_exp + expert), counted once the caller sized it to that.
    std::vector<float>& get_usage();
    /// Gets the experts the CPU lane has computed so far (cpu_experts' calls), each counted once a call.
    int64_t get_cpu_experts() const;
    /// Gets DSpark's reloadable dense allocation size on a stage, including while unloaded.
    uint64_t get_drafter_weight_bytes(int stage) const;
    /// Releases or reloads DSpark's dense tensors, retaining pool-owned conversation state.
    bool switch_drafter_weights(bool on, std::string& err);
    /// Rebuilds the seated conversation's DSpark rings from its recent trunk inputs.
    bool reseed_drafter(int64_t position, std::string& err);
    /// Gets the CUDA ids of the GPUs the model runs on, in its order (use_state_memory's `dev` is an index here).
    std::vector<int> get_devices() const;
    /// Points the conversation's state on GPU `dev` at memory the caller keeps: the window rings of the layers it runs
    /// and the partial groups of its KV sources in `fixed`, the DSpark blocks' rings in `drafter` (null without them),
    /// where `at` puts them (describe_conversation), and each KV source's compressed rows and index keys at `ckv[s]`
    /// and `ik[s]` where the GPU keeps them.  Call again whenever the memory moves.
    void use_state_memory(int dev, uint8_t* fixed, uint8_t* drafter, const ConversationRegions& at,
                          const std::vector<KvRows>& ckv, const std::vector<KvRows>& ik);


    /// Supplies one paged conversation's inputs and receives its DSpark proposal.
    struct DraftRequest {
        int session = 0, token = 0, limit = 0;
        int64_t pos = 0;
        float temperature = 0, top_p = 1, min_p = 0;
        int top_k = 0;
        uint64_t seed = 1, counter = 0;
        std::vector<int> tokens;
        std::vector<float> confidence;
    };
    /// Runs DSpark's layers once across the paged engine's eligible conversation windows.
    bool draft_conversations(std::vector<DraftRequest>& requests, const std::function<void(int)>& select,
                             const std::function<void(int)>& bind, std::string& err);
    /// Runs contiguous conversation windows through one layer-major pass. The counts partition tokens;
    /// select seats host state and bind selects GPU pages without waiting on the stream.
    bool step_conversations(const std::vector<int>& tokens, const std::function<void(int)>& select,
                            const std::function<void(int)>& bind, const std::vector<int>& counts, std::string& err);
    /// Preserves sampled proposal distributions for the paged engine's joint verification; n=0 discards them.
    bool save_joint_draft(int session, int n, std::string& err);
    /// Samples a conversation's window using its preserved proposal distribution.
    bool pick_joint_rows(int session, int row, const std::vector<int>& drafts, float t, float tp, int tk, float mp,
                         uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err);
    /// Restores rejected state and seeds the selected conversation from its accepted tap rows.
    bool keep_joint_rows(int row, int count, int keep, std::string& err);
    /// Stores independent row inputs for the paged engine's shared pipeline scheduler.
    bool prepare_group(int group, const std::vector<int>& tokens, const std::function<void(int)>& select,
                       const std::function<void(int)>& bind, std::string& err);
    /// Queues one stage using the prepared group's inputs and conversation pages.
    bool launch_group(int stage, int group, std::string& err);
    /// Completes outstanding writes before pages can be reused, including after a failed launch.
    bool wait_state(std::string& err);
    /// Waits for the work queued on GPU `dev` (an index into get_devices, which is also its pipeline stage).
    bool wait_card(int dev, std::string& err);
    /// Checks the stage's completion event without waiting for its stream.
    int poll_group(int stage, std::string& err);
    /// Restores the group's logits and retains its DSpark inputs after every stage drains.
    bool finish_group(int group, const std::function<void(int)>& select, std::string& err);
    /// Sets which drafts a sampled request's verify rows keep (verify_rows, pick_joint_rows); speculative by default.
    void set_draft_rule(strata::kernels::DraftRule rule);
    /// Gets the routing rank from which the decode leaves out an expert VRAM does not hold, when every token of a
    /// window ranks it that low or lower by weight (0: never; strata::core::mark_tail_skips).
    int get_route_tail_skip() const;
    /// Changes get_route_tail_skip's rank for the windows after this.
    void set_route_tail_skip(int rank);
    /// Gets how many percent higher the decode's router counts the score of an expert VRAM holds when it picks a
    /// token's experts (0: no bias; the weights stay the router's).
    int get_route_prior_pct() const;
    /// Changes get_route_prior_pct's percent for the steps after this.
    void set_route_prior_pct(int pct);
    /// Samples one row of the last shared pass with the request's sampling settings.
    int sample_row(int row, float t, float tp, int tk, float mp, uint64_t seed, uint64_t counter);
    /// Feeds `token` at pos(); with logits, leaves the next token's scores on the head's device.
    bool step(int token, bool logits, std::string& err);
    /// Feeds n tokens (windows of up to 16, or a long run in chunks of prefill_chunk); the last `logits_rows` of them get
    /// logits (logits_rows > 16: only a run of one chunk or window).
    bool feed(const int* tokens, int n, int logits_rows, std::string& err, int after = 0);
    /// feed() of a prompt's end that also keeps the state before its last token in checkpoint `slot` (a prompt's
    /// last token then rides along in the run of chunks instead of a window of its own); the same as
    /// feed(n - 1), checkpoint(slot), feed(1)
    bool feed_mark(const int* tokens, int n, int logits_rows, int slot, std::string& err);
    /// The next token from the last feed's last logit row (greedy when temperature <= 0).
    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The request's repetition / frequency / presence penalties (llama.cpp's) for every pick after this - sample() and
    /// verify_rows() alike: over the last `last_n` tokens the state has consumed (at most 4096; a verify window's row
    /// counts the drafts before it).  last_n 0 (or every penalty neutral): none, the picks exactly as without.
    void set_penalties(int last_n, float repeat, float freq, float present);
    /// set_penalties() from a request's sampling keys (penalty_last_n -1: the whole context; a repeat penalty of 0 or
    /// below, or a value that is not a number: neutral).
    void set_penalties(const strata::serve::Sampling& s);
    /// Copies row `row` of the head's logits (the one sample_row reads; -1: the one sample() reads) to `out`, vocab
    /// floats on the host, for strata-ds41's LP lines.
    bool copy_logits_row(int row, float* out);

    /// The DSpark drafter (speculative decoding): its block size (0: not loaded), drafts for the positions after `y`
    /// (y: the token at pos(), not fed yet) with their confidences, and the verification of [y, drafts]: how many of the
    /// drafts stand and the token after them (the state then holds y and the accepted drafts).  The drafts are the
    /// drafter's argmax; a sampled request passes its chain (the same temperature / top_p / top_k / min_p / seed /
    /// counter as its verify_rows()) so that with DS4_DSPARK_SAMPLE=1 they are drawn from the drafter's distribution,
    /// as DeepSeek's reference does.  verify_rows() keeps a draft d with probability min(1, p(d) / q(d)) over the
    /// request's chain - the speculative-sampling rule, so every token is still distributed as the model's own draw.
    int draft_block() const;
    bool draft(int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err, float temperature = 0.0f,
               float top_p = 1.0f, int top_k = 0, float min_p = 0.0f, uint64_t seed = 0, uint64_t counter = 0);
    /// The verification in two parts, for a caller that decides what to keep (strata-ds41's step): verify_rows runs [y,
    /// drafts] and gets in `picks` the token after each row up to the first rejected draft (the rows after it get
    /// none: picks has accepted + 1 tokens), the state holding every row; keep_verified then keeps the first `keep`
    /// rows (1 to picks' size) and rolls the rest back.  Nothing else may run the model between the two.
    bool verify_rows(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                     uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err);
    bool keep_verified(int keep, std::string& err);
    /// Gets how many of `conf` (the drafts' confidences, draft() order), at most `most`, a round verifies: while
    /// each is at least `conf_min` and their product at least `cum_min` (dspark_cum; 0: no such cut).
    static int count_confident(const std::vector<float>& conf, int most, float conf_min, float cum_min);
    /// Gets the drafts' cut on the product of their confidences when a request names none: STRATA_DS4_DSPARK_CUM,
    /// else 0.5 where every card is a gfx1200 / gfx1201 and 0 (off) elsewhere (dspark_cum_value).
    float get_default_cum() const;


    /// Tokens a feed call takes at a time to use the chunked prompt path (all of a prompt: its chunks run as a
    /// pipeline; 16 without it).
    int feed_unit() const;
    /// Called from feed() while a run of chunks goes through the layers, with the tokens done so far.
    void set_progress(std::function<void(int64_t)> f);
    /// Asked while a run of chunks goes through the layers: true (a Stop) ends it early - one GPU after the layer it is
    /// on, several at a chunk all of them reach - and feed() returns false with cancelled() true.  The state past the
    /// prompt's start is then not usable: restore() a checkpoint kept before the feed, or reset().
    void set_cancel(std::function<bool()> f);
    bool cancelled() const;   // the last feed() / feed_mark() ended early on a Stop
    /// Asked while a run of chunks goes through the layers: true (a yield) ends it at a chunk every GPU reaches, as a
    /// Stop does, but keeps the chunks read - feed() / feed_mark() return true having taken fed() of their tokens (the
    /// rest a later feed's; feed_mark's checkpoint then not kept).  Not on a run that goes layer by layer (Project Maya #587).
    void set_yield(std::function<bool()> f);
    int fed() const;   // the tokens the last feed() / feed_mark() took (all of them unless a yield)
    /// The chunked prompt path's chunk (0: windows).
    int prompt_chunk() const;
    /// (Options::expert_pages) The paged engine lends each GPU's chunked prompt path its buffers for each prompt
    /// (strata/common/serve/paged_model.hpp: get_prompt_ceiling, get_prompt_buffers, lay_out_prompt).  Gets whether a
    /// read of `rows` tokens may take the chunked path (else windows of 16, which borrow nothing).
    bool can_read_chunked(int rows) const;
    /// Gets the bytes GPU `dev`'s chunked prompt path borrows on the GPU for chunks of `chunk` tokens; `host` gets
    /// those it borrows in page-locked host memory.
    uint64_t get_prompt_bytes(int dev, int chunk, uint64_t& host) const;
    /// Lays GPU `dev`'s chunked prompt path out for chunks of `chunk` tokens (at most prompt_chunk()'s at load, or
    /// raise_prompt_chunk's) in the lent `device` and `host` memory (get_prompt_bytes'); `chunk` 0: waits until it no
    /// longer reads them.
    bool lay_out_prompt(int dev, int chunk, uint8_t* device, uint8_t* host, std::string& err);
    /// Lets lay_out_prompt take chunks of up to `chunk` tokens, growing the page-locked buffers the load sized for
    /// its chunk; between prompts.  False (and `err`, the old limit kept) when they cannot grow.
    bool raise_prompt_chunk(int chunk, std::string& err);
    int64_t pos() const;
    int64_t max_context() const;
    int vocab() const;
    int dim() const;

    void reset();
    /// true once if the router met non-finite scores on any device since the last call (see router_nonfinite_take)
    bool router_nonfinite();
    /// Keep the state at the current position in checkpoint slot 0 or 1 (the end of a prompt, the position before it);
    /// restore(limit) goes back to the latest one at or before limit (-1: none).
    void checkpoint(int slot = 0);
    bool restore(int64_t limit);
    /// Checkpoint 1 also becomes the anchor (kept until a restore goes before it): the position before the end of the
    /// first prompt of a run that continued in place, where a re-rendered history (earlier reasoning dropped) splits.
    void pin();
    /// Keep the state at the current position as a restore point a request asked for (GEN marks= / shared=, pin=: the
    /// start of the last user message, a point near the prompt's tail, the end of a shared start - shared: given up
    /// last), besides the three above: restore() goes back to these too.  At most marks_cap() a conversation
    /// (DS4_RESTORE_POINTS, 4; 0: none kept), the least recently used giving way; each check_bytes() of host RAM.
    void mark(bool shared = false);
    int marks_cap() const;
    size_t check_bytes() const;
    /// The positions restore() can go back to, latest first.
    std::vector<int64_t> check_positions() const;
    /// Exchanges the tokens and checkpoints of the conversation the model computes for `s`'s, once the GPUs' work
    /// ends.  For strata-ds41, which holds several conversations: the rest of each one's state is in its pages, which
    /// use_state_memory points the model at.
    void swap_conversation(ConversationState& s);
    /// Gets why `s`'s checkpoints cannot be this model's (their layers, rings or partial groups are of other sizes), or
    /// "" when they can: swap_conversation takes only such state (strata-ds41 checks a paused conversation's).
    std::string get_state_problem(const ConversationState& s) const;

    /// What a saved conversation must match: this engine's version, this model's geometry and files and the drafter's
    /// blocks - a saved conversation naming anything else is not this model's state.
    std::string slot_identity() const;

    /// The vision tower: the token every position of an image's span carries (-1: no tower).
    int image_token() const;
    /// An image's pixels (RGB rows, width x height, both multiples of the patch size: the server's resize) -> the
    /// rows of its span, BF16 [n_h * (n_w + 1) + 2][dim]: image_start, n_h rows of (n_w aligner rows, image_newline),
    /// image_end.
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& span, std::string& err);
    /// The pictures of the prompt about to be fed ({}: none): positions inside a span read its rows (and route with
    /// the image bias, and take no part in the Engram n-grams); the rest the embedding table's.
    void set_images(std::vector<Image> images);

    void save_profile() const;
    const Stats& stats() const;
    /// INFO / VRAM lines for the server's Monitor.
    std::vector<std::string> info_lines() const;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
