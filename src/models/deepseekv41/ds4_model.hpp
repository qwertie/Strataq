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
// truncated by position) so a conversation can go back to the end of its last prompt.  A conversation that leaves the
// model altogether (another chat answered in between) is a slot: slot_save / slot_load take the same state - the
// compressed KV and index keys included, which a checkpoint leaves in place - through a file or a host buffer.
#pragma once

#include "strata/common/serve/memory_report.hpp"
#include "strata/models/deepseekv41/conversation_layout.hpp"
#include "strata/models/deepseekv41/conversation_state.hpp"
#include "strata/models/deepseekv41/kv_rows.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
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
    bool vram_experts_auto = true;    // (Metal: no --vram-experts given: the share the anonymous-memory cap leaves - metal_expert_cap)
    double vram_experts = 1.0;        // the share of the expert budget to use (tests: less, so the CPU lane works)
    // the CPU lane's threads: 0 a thread a physical core, or where the host offers it the performance cores or one
    // NUMA node's cores when measured faster at load; n: at most n, a thread a core
    int cpu_threads = 0;
    std::string profile;              // expert hit counts (read to place the hottest experts in VRAM, written back)
    // --profile-readonly: read the profile to place the experts, never write it back.  Without this the run's routing
    // counts go into the file, so the next run places experts differently and greedy output / KL move between runs.
    bool profile_readonly = false;
    bool expert_cache = true;         // experts move between VRAM and the CPU lane as they are used
    // the CPU lane's experts read in place from the model's files, the page cache keeping what fits, instead of
    // copied into RAM: 1 on, 0 off, -1 (auto) where the RAM this process can get does not hold them
    int mmap_experts = -1;
    // the experts the VRAM leaves over are cached in RAM by a store of their own (ds4_store.hpp): whole experts read unbuffered into
    // slots, a replacement policy (0 lru, 1 lfu, 2 lfu then lru), filled on demand and by the lookahead's guesses.  GB of RAM at
    // most (never past the RAM the process can still get less 8 GB); 0: off, the OS page cache as before
    double ssd_cache_gb = -1.0;   // < 0: auto (the RAM the process can get less a floor, at most 48 GB; off below 4 GB), 0: off
    int ssd_policy = 0;
    // dspark_auto: the drafter is loaded only where the routed experts fit the GPUs (more than 1.5x what they have, the verify rounds
    // read 3.3x the bytes of a token and measured no faster than plain decoding, and the drafter's 7 GB of experts take the VRAM the
    // trunk's hottest would have had); --dspark on / off / none says it outright
    // experts a word takes: -1 the rule (5 where the cards hold 16 GiB or less in all, else the model's 6), N the router's top N with
    // their weights renormalised (--experts 6: full quality); ds4_experts_rule.hpp
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
    bool vision_only = false;         // (tests) load the vision tower alone: encode_image() and nothing else
    // --serve on a discrete GPU (--vision-resident: not): the tower waits in host memory and takes its ~320 MiB of graphics memory
    // for a picture turn only, out of what --reserve-mib keeps free (it stays on the card when the reserve is too small for it)
    bool vision_lazy = false;
    bool verbose = false;
    /// strata-ds41 (the memory manager's pools): the routed experts are the caller's - the GPU computes those
    /// set_expert_addresses names, the CPU lane and the prompt path read the others from this pool (a RAM page pool
    /// over the SSD) - instead of the old engine's VRAM slots, RAM store, files' mapping and moves. Requires
    /// layer-parallel stages. Null: the old engine's own.

    strata::core::ExpertSource* expert_pages = nullptr;
    /// Chooses trunk cuts from the geometry and expert bytes before loading dense weights; false sets the error.
    /// The paged engine supplies it; empty for the old engine.
    std::function<bool(const Geometry&, const std::vector<uint64_t>&, std::vector<int64_t>&, std::string&)> plan_layers;
    // --parallel: conversations held at once (1-8), each its own state - its position, tokens, checkpoints, window rings,
    // the compressors' partial groups, the compressed KV and the index keys (a full context's each); 1: as before
    int seqs = 1;
};

/// Where one trunk layer's routed experts lie in the model's files (Model::get_file_paths) and where each matrix goes
/// in an expert's blob, the layout the GPU's kernels and the CPU lane read: matrix r (gate, up, down) of expert e is
/// bytes[r] bytes at off[r] + e * stride[r] in file file[r], at tr[r] in a blob of `blob` bytes.
struct ExpertFileLayout {
    int n_exp = 0;
    int file[3] = {-1, -1, -1};
    uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};
    uint64_t bytes[3] = {0, 0, 0};
    uint64_t tr[3] = {0, 0, 0};
    uint64_t blob = 0;
    /// EXL3 stores nine separately placed pieces per expert, in the pack's file order.
    struct Piece {
        int file = -1;
        uint64_t off = 0, bytes = 0, dst = 0;
    };
    std::vector<std::vector<Piece>> pieces;
};

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
    int64_t stream_experts = 0;             // ... and how many of the others streamed to the GPU (the PCIe lane)
    double cpu_ms = 0.0, gpu_wait_ms = 0.0;   // the CPU lane's time; the host's waits for the routing
    double layer_ms = 0.0, head_ms = 0.0, engram_ms = 0.0;   // the layers' host time in all; the head's wait; Engram reads
    int64_t eg_rows = 0, eg_hits = 0, eg_reads = 0;   // Engram rows asked for, served without a read, read from the files
    double eg_io_ms = 0.0;                            // the windows' Engram reads, issue to last done (mostly hidden)
    int64_t eg_cached = 0, eg_cache_cap = 0;          // the row cache's rows held, its capacity
    int64_t swaps = 0;                // experts the cache moved into VRAM
    int64_t skipped = 0;              // routed experts STRATA_DS4_DISK_SKIP_W left out (light, not in RAM)
    int skip_on = -1;                 // DISK_SKIP_AUTO: 1 skipping, 0 off (warm), -1 not in use
    int64_t skip_switches = 0;        // ... the times it switched
    double skip_cold = 0.0;           // ... the cold share over its window (while skipping)
    int64_t ram_experts = 0, ram_moves = 0;   // CPU-lane experts read from the RAM tier; experts moved into it
    double prof_ms[6] = {};          // DS4_PROF: the GPU's time in decode by stage (between layers, attention, routing +
    int64_t prof_tokens = 0;         // shared expert, experts' gate / up, the wait for the CPU lane, down + post)
    // lookahead (decode windows of one): the picks of layers guessed a layer before (the next layer's gate on this
    // layer's FFN input), of them in the guess's top k / in all its guesses / in the guess two layers before; the picks
    // read from the files, of them prefetched early; experts prefetched early
    int64_t ahead_picks = 0, ahead_hit_k = 0, ahead_hit = 0, ahead_hit2 = 0;
    int64_t ahead_cold = 0, ahead_cold_hit = 0, ahead_pf = 0;
    int64_t ahead_rank_n[16] = {}, ahead_rank_hit[16] = {};   // the next-layer guesses by rank, and how many were picked
    // the RAM store (--ssd-cache), totals since the load: reads asked for and found there (hits), missed (misses), started ahead
    // of use (prefetched) of which used / given up unused, slots given up, bytes read, ms the CPU lane waited for the reads
    int64_t store_hits = 0, store_misses = 0, store_prefetched = 0, store_pf_used = 0, store_pf_wasted = 0, store_evictions = 0;
    int64_t store_bytes = 0, store_fails = 0;
    int64_t store_zc = 0, store_zc_visible = 0;   // experts the GPU computed from the store (zero-copy over PCIe); slots it may read
    double store_wait_ms = 0.0, store_busy_ms = 0.0;
    int64_t store_slots = 0, store_slots_max = 0;
    int64_t drafts_offered = 0, drafts_accepted = 0, drafts_rounds = 0;
    // per draft position (the 1st .. 8th of a round): verified with all the drafts before it accepted, and accepted
    int64_t draft_reached[8] = {}, draft_kept[8] = {};
    double draft_ms = 0.0, verify_ms = 0.0;   // the drafter's time, the verification windows' time
};

class Model {
public:
    Model();
    ~Model();
    /// Loads the dense weights, the embedding table and the Engram's reader, without the conversation's state
    /// (reserve_caches) or the routed experts (place_experts), which take the GPU memory left after it.
    bool load(const Options& o, std::string& err);
    /// Gives the conversation its state for the whole context in GPU memory of its own (the old engine's caches);
    /// strata-ds41 gives it page-pool pages instead (use_state_memory).
    bool reserve_caches(std::string& err);
    /// Allocates the prompt path's buffers, places the routed experts in the GPU memory left (the rest on the CPU
    /// lane and in the RAM store) and starts the threads that serve them.  The model runs from then on.
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
    /// Gets the GPU memory of the weights on GPU `dev` (an index into get_devices).
    uint64_t get_weight_bytes(int dev) const;
    /// Gets DSpark's reloadable dense allocation size on a stage, including while unloaded.
    uint64_t get_drafter_weight_bytes(int stage) const;
    /// Releases or reloads DSpark's dense tensors, retaining pool-owned conversation state.
    bool switch_drafter_weights(bool on, std::string& err);
    /// Rebuilds the seated conversation's DSpark rings from its recent trunk inputs.
    bool reseed_drafter(int64_t position, std::string& err);
    /// Gets the GPU memory of the buffers on GPU `dev` that are not the weights, the experts or the caches (the
    /// windows' and the prompt path's).
    uint64_t get_scratch_bytes(int dev) const;
    /// Gets the CUDA ids of the GPUs the model runs on, in its order (use_state_memory's `dev` is an index here).
    std::vector<int> get_devices() const;
    /// Points the conversation's state on GPU `dev` at memory the caller keeps, in place of reserve_caches': the
    /// window rings of the layers it runs and the partial groups of its KV sources in `fixed`, the DSpark blocks'
    /// rings in `drafter` (null without them), where `at` puts them (describe_conversation), and each KV source's
    /// compressed rows and index keys at `ckv[s]` and `ik[s]` where the GPU keeps them.  Call again whenever the
    /// memory moves.
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
    /// Checks the stage's completion event without waiting for its stream.
    int poll_group(int stage, std::string& err);
    /// Restores the group's logits and retains its DSpark inputs after every stage drains.
    bool finish_group(int group, const std::function<void(int)>& select, std::string& err);
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
    /// feed_mark() that also keeps the restore points `inner` on the way (positions inside the read - past its first
    /// token, short of its last - in order, with mark()'s `shared`), each the state a read up to it leaves: the same as
    /// feed() up to each and mark() there, then feed_mark() of the rest - but a run of chunks keeps them in one run
    /// instead of a run up to each.  marks_inside(): whether n tokens read so, the points at rows `rel` of them, cost
    /// at most a chunk's rows more than stopping at each (a long prompt's skip: the one read runs the rows before its
    /// first point through every layer, where the reads after it would have skipped theirs) - else the caller stops the
    /// read at the points, as before.  feed()'s `after`: the prompt's tokens still to come in later feeds (a piece of it)
    bool feed_marks(const int* tokens, int n, int logits_rows, int slot,
                    const std::vector<std::pair<int64_t, bool>>& inner, std::string& err);
    bool marks_inside(int n, const std::vector<int64_t>& rel) const;
    /// The next token from the last feed's last logit row (greedy when temperature <= 0).
    int sample(float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The request's repetition / frequency / presence penalties (llama.cpp's) for every pick after this - sample() and
    /// verify() alike: over the last `last_n` tokens the state has consumed (at most 4096; a verify window's row counts
    /// the drafts before it).  last_n 0 (or every penalty neutral): none, the picks exactly as without.
    void set_penalties(int last_n, float repeat, float freq, float present);
    /// The last feed's logit rows on the host (rows x vocab floats).
    bool logits(std::vector<float>& out);
    /// Copies row `row` of the head's logits (the one sample_row reads; -1: the one sample() reads) to `out`, vocab
    /// floats on the host, for strata-ds41's LP lines.
    bool copy_logits_row(int row, float* out);

    /// The DSpark drafter (speculative decoding): its block size (0: not loaded), drafts for the positions after `y`
    /// (y: the token at pos(), not fed yet) with their confidences, and the verification of [y, drafts]: how many of the
    /// drafts stand and the token after them (the state then holds y and the accepted drafts).  The drafts are the
    /// drafter's argmax; a sampled request passes its chain (the same temperature / top_p / top_k / min_p / seed /
    /// counter as its verify()) so that with DS4_DSPARK_SAMPLE=1 they are drawn from the drafter's distribution, as
    /// DeepSeek's reference does.  verify() keeps a draft d with probability min(1, p(d) / q(d)) over the request's
    /// chain - the speculative-sampling rule, so every token is still distributed as the model's own draw.
    int draft_block() const;
    /// every GPU the engine uses is a gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) in a HIP build, false anywhere else (decided once)
    bool gpus_are_gfx12() const;
    bool draft(int y, std::vector<int>& drafts, std::vector<float>& conf, std::string& err, float temperature = 0.0f,
               float top_p = 1.0f, int top_k = 0, float min_p = 0.0f, uint64_t seed = 0, uint64_t counter = 0);
    bool verify(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err);
    /// verify() in two parts, for a caller that decides what to keep (strata-ds41's step): verify_rows runs [y,
    /// drafts] and gets in `picks` the token after each row up to the first rejected draft (the rows after it get
    /// none: picks has accepted + 1 tokens), the state holding every row; keep_verified then keeps the first `keep`
    /// rows (1 to picks' size) and rolls the rest back.  Nothing else may run the model between the two.
    bool verify_rows(int y, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                     uint64_t seed, uint64_t counter, std::vector<int>& picks, std::string& err);
    bool keep_verified(int keep, std::string& err);
    /// Gets how many of `conf` (the drafts' confidences, draft() order), at most `most`, the old engine verifies: while
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
    /// The chunked prompt path's chunk (0: windows), and the split's pipeline stages (its groups of GPUs, one after
    /// another): a read of that many chunks keeps them all busy - --parallel's prompt pieces are that many chunks.
    int prompt_chunk() const;
    int pipeline_stages() const;
    /// A request starts (before its prompt): DISK_SKIP_AUTO may switch back on here, never mid-answer.
    void begin_request();
    int64_t pos() const;
    int64_t max_context() const;
    int vocab() const;
    int dim() const;
    const std::vector<int32_t>& tokens() const;   // what the state has consumed, in order

    /// --parallel's conversations (Options::seqs): each its own state.  seq_bind(s): every call of this class acts on
    /// conversation s from here - feed, step, draft / verify, sample, checkpoints, slots, reset, tokens (0 at load).
    int seq_count() const;
    void seq_bind(int s);
    int seq_bound() const;
    int64_t seq_pos(int s) const;
    const std::vector<int32_t>& seq_tokens(int s) const;
    /// One token into each of n conversations (distinct, n <= 16, each at its own next position) through the model
    /// together: the dense weights and the experts the rows share read once for all of them, each row's attention on
    /// its own conversation's rings and KV.  Row i's logits stay for sample_multi(i, ...), with the penalties
    /// set_penalties() set over row i's conversation's tokens.  No drafts; each conversation's drafter rings are
    /// seeded as its own step would seed them, so it drafts on alone later.  The bound conversation stays bound.
    bool step_multi(const int* seqs, const int* tokens, int n, std::string& err);
    int sample_multi(int row, float temperature, float top_p, int top_k, float min_p, uint64_t seed, uint64_t counter);
    /// The drafter beside --parallel (Project Maya #611): n conversations' verify windows through the model together -
    /// window i is conversation seqs[i]'s token y and its drafts (each conversation's own draft(), bound to it),
    /// toks[i][0 .. lens[i]) with toks[i][0] = y - as step_multi's rows (distinct conversations, at most verify_max()
    /// rows in all), a window's rows attending to each other as verify()'s.  Then verify_multi(i, ...) once for each
    /// window i: its verdict as verify()'s (the drafts as points: the drafter's argmax), with the penalties
    /// set_penalties() set over its conversation's tokens; the conversation keeps y and the accepted drafts, and its
    /// drafter rings are seeded with them.  The bound conversation stays bound.
    bool step_multi_spec(const int* seqs, const int* const* toks, const int* lens, int n, std::string& err);
    /// The drafter for n conversations (2 or more, distinct) in one pass: draft()'s drafts and confidences for each
    /// conversation seqs[i] after its token ys[i] (the drafter's argmax), the blocks' rows through the DSpark blocks
    /// together.  can_draft_multi: whether these conversations fit one pass (their blocks in a window, each its
    /// block's room in its context); else each drafts alone with draft().
    bool can_draft_multi(const int* seqs, int n) const;
    bool draft_multi(const int* seqs, const int* ys, int n, std::vector<std::vector<int>>& drafts,
                     std::vector<std::vector<float>>& conf, std::string& err);
    bool verify_multi(int i, const std::vector<int>& drafts, float temperature, float top_p, int top_k, float min_p,
                      uint64_t seed, uint64_t counter, int& accepted, int& next, std::string& err);
    int verify_max() const;   // the rows a window or a multi step takes at most

    void reset();
    /// The post-prompt cache settle run on a thread of its own while the server waits for its next request (WP-8's option 1):
    /// the swap rounds of cache_settle without the store preload, stopped early when `cancel` goes true.  `done` is set when
    /// the thread has nothing left to do.  Only one such run at a time; the caller joins the thread after setting cancel.
    void settle_idle(const std::atomic<bool>& cancel);
    /// true once if the router met non-finite scores on any device since the last call (see router_nonfinite_take)
    bool router_nonfinite();
    /// Keep the state at the current position in checkpoint slot 0 or 1 (the end of a prompt, the position before it);
    /// restore(limit) goes back to the latest one at or before limit (-1: none).
    void checkpoint(int slot = 0);
    bool restore(int64_t limit);
    int64_t restorable(int64_t limit) const;
    /// Checkpoint 1 also becomes the anchor (kept until a restore goes before it): the position before the end of the
    /// first prompt of a run that continued in place, where a re-rendered history (earlier reasoning dropped) splits.
    void pin();
    /// Keep the state at the current position as a restore point a request asked for (GEN marks= / shared=, pin=: the
    /// start of the last user message, a point near the prompt's tail, the end of a shared start - shared: given up
    /// last), besides the three above: restore() and a slot go back to these too.  At most marks_cap() a conversation
    /// (DS4_RESTORE_POINTS, 4; 0: none kept), the least recently used giving way; each check_bytes() of host RAM.
    void mark(bool shared = false);
    int marks_cap() const;
    size_t check_bytes() const;
    /// The positions restore() can go back to, latest first (a slot keeps them: it can be taken back at any of them).
    std::vector<int64_t> check_positions() const;
    /// Exchanges the tokens and checkpoints of the conversation the model computes for `s`'s, once the GPUs' work
    /// ends.  For strata-ds41, which holds several conversations: the rest of each one's state is in its pages, which
    /// use_state_memory points the model at.
    void swap_conversation(ConversationState& s);
    /// Gets why `s`'s checkpoints cannot be this model's (their layers, rings or partial groups are of other sizes), or
    /// "" when they can: swap_conversation takes only such state (strata-ds41 checks a paused conversation's).
    std::string get_state_problem(const ConversationState& s) const;

    /// The conversation in the model into `out` (slot_save): its position, its tokens, every window ring (the DSpark
    /// blocks' included), the compressors' partial groups and each KV source's compressed KV and index keys up to it,
    /// the pieces the state a prompt continues is made of.  0 and err: nothing was set aside (it holds pictures, there
    /// is no conversation, or the conversation is already set aside - unchanged since).
    /// `at` >= 0: the conversation as it stood at its restore point `at` (a mark kept by mark()), though it was read on past it.
    uint64_t slot_save(std::vector<uint8_t>& out, std::string& err, int64_t at = -1);
    /// `data` (a slot's bytes) back into the model: the state at the slot's position n_pos, as slot_save took it -
    /// what a prompt that continues the slot's tokens reads on from (the caller checked the tokens are a prefix).
    bool slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err);
    /// What a saved conversation must match: this engine's version, this model's geometry and files and the drafter's
    /// blocks - a slot naming anything else is not this model's state (a file's .meta says so; slot_load refuses).
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
    /// The picture turn is over: a tower that lives in host memory between turns gives its graphics memory back (no-op otherwise).
    void vision_idle();
    /// True when the tower holds no graphics memory between picture turns.
    bool vision_lazy() const;

    void save_profile() const;
    const Stats& stats() const;
    std::string describe() const;
    /// INFO / VRAM lines for the server's Monitor.
    std::vector<std::string> info_lines() const;
    /// Describes the weights, experts, drafter, caches and scratch on each GPU, the experts in RAM and on the SSD
    /// and the Engram table for GET_MEMORY_USE (the parked conversations are added by the caller's loop).
    strata::serve::MemoryReport describe_memory() const;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
