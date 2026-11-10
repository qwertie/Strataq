// include/strata/models/qwen38/model.hpp - the Qwen3.8 engine's loaded model: the weights, the head, the per-layer
// embeddings, the session, the drafter, the expert sources and cache, the CPU pool and the layer split's stages.
//
// `Model::load` is the loader that used to be inline in `strata`'s main() (src/program/generate.cpp); the old engine
// constructs a Model, loads it and keeps working on its members under their old names. Its members are the loader's
// former local variables: what the serve and generate loops use is public, so a second program can be built on the same
// loader instead of a copy of it (docs/MEMORY_MANAGER_PLAN.md, step M2d).
#pragma once

#include "strata/models/qwen38/options.hpp"
#include "strata/common/experts/hit_hook.hpp"
#include "strata/common/experts/nibble_cache.hpp"
#include "strata/common/kernels/cpu/pool.hpp"
#include "strata/common/kernels/rope_scaling.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/weights.hpp"
#include "strata/models/qwen38/kernels/ngram.hpp"
#include "strata/models/qwen38/layer.hpp"
#include "strata/models/qwen38/mtp.hpp"
#include "strata/models/qwen38/native_dense.hpp"
#include "strata/models/qwen38/native_head.hpp"
#include "strata/models/qwen38/prefill/prefill.hpp"
#include "strata/models/qwen38/session.hpp"
#include "strata/models/qwen38/verify.hpp"
#if defined(STRATA_HAVE_DFLASH2)
#include "strata/models/qwen38/dflash2.hpp"
#endif

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::qwen38 {
/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

/// Plan v0.3 P6: the pool for a verify window, with the CPU experts' Default arithmetic as every decode has always run.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer);
/// The same with the window-invariant CPU arithmetic that a probabilistic (STRATA_SPEC_PROB) decode uses, so a row's
/// target distribution does not depend on its window.
void drive_pool_multi_invariant(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out, int64_t layer);

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
    /// strata-q38: per layer, the stage whose pool holds the layer's experts that its own stage's does not, which
    /// computes them in one hop (`hop` of that stage), or -1; null: the pool's peer (`Drive`'s) stays as it is
    const int32_t* holder = nullptr;
    strata::core::PeerExperts* hop[kMax] = {};
};
/// Layer split: the pool for a verify window, routed to the stage that runs the layer (`user` is a SplitDrive).
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer);
/// drive_pool_split with the window-invariant CPU arithmetic (drive_pool_multi_invariant).
void drive_pool_split_invariant(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out, int64_t layer);

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    double link_gbps = 0.0;   ///< this card's probed host->device bandwidth (PcieAuto's per-stage share)
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    cudaStream_t stream = nullptr;
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::core::Verifier ver_b;                        ///< --pipeline-windows: the odd windows' verifier
    strata::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

/// Qwen3.8's model as `strata` loads it from its settings: weights, head, per-layer embeddings, session, drafter,
/// expert sources, expert cache and CPU pool, and the stages of a layer split. Owns them; the GPU buffers held as raw
/// pointers (`arena`, `sbuf`, `d_parts`, `d_logits`, ...) are freed by the program that loaded it, at its end.
class Model {
public:
    /// Keeps `o`, which load() reads and adjusts (for example the expert cache's size when it is `auto`).
    explicit Model(Options& o) : o(o) {}
    /// Frees the token path's scratch and its captured graph.
    ~Model();

    /// Applies the process-wide settings in `o`, checks them, and loads everything this class holds. Returns 0 when the
    /// model is up, otherwise the exit status for main() (the reason has been printed).
    int load();

    /// Fills the expert residency table from the loaded caches and uploads it (the hit path the token graph and the
    /// verify windows decide on the device), sets up the token path's scratch, and captures the one-graph token when
    /// nothing needs a host step inside it (`dumping_layers`: --dump-layers asks for one per layer).  Call once,
    /// after load().  Returns 0, otherwise the exit status for main() (the reason has been printed).
    int build_hit_tables(bool dumping_layers);
    /// strata-q38's drafter switch: gives back the MTP drafter's weights and buffers.  ConversationDrafter::init binds
    /// a drafter loaded again (load_drafter).
    void unload_drafter();
    /// Sets the MTP drafter's options from --mtp-* and loads it from --mtp on its card (get_drafter_stage), for load()
    /// and again after unload_drafter; false (and `err`) when it could not.
    bool load_drafter(std::string& err);
    /// Gets the layer split's stage whose card runs the MTP drafter, which binds that stage's weights and head; null
    /// when the drafter is on CUDA0 (no split, or mtp_first).
    GpuStage* get_drafter_stage() const { return last_st != nullptr && !mtp_first ? last_st : nullptr; }
    /// Gets the draft vocabulary file DFlash2 binds its head from: the subset df2_vocab_resolve chose with DFlash2 alone
    /// (`df2_solo`), else the MTP's (--mtp-draft-vocab, or <mtp>/draft_vocab.bin).
    std::string get_df2_vocab_path() const;
    /// Lets DFlash2 bind the MTP drafter's copy of the draft head instead of its own (`df2_expect_share`); call after
    /// mtp.bind and before df2.bind.
    void offer_df2_head();
    /// Gets the VRAM DFlash2's own draft head and logits take when it binds (0 when it shares the MTP's head).
    int64_t get_df2_head_bytes() const;
    /// Gets how many (layer, expert) pairs a live slot of their stage's expert cache holds (a slot past the live ones
    /// was given back).
    int64_t count_gpu_held() const;
    /// P1.8: turns the file tier's Linux defaults (io prefetch, the decode gate) and the routing-aware prefetch off
    /// while every routed expert is in the GPU caches, and back on when one is not.  Call at a quiet point (a request's
    /// start, a VRAM resize): nothing in flight.  Only what the policy turned off is turned on again.
    void refresh_file_tier();
    /// Copies `host_res` to the device table `dst` (CUDA0's or a stage's) and waits for the copy.
    cudaError_t put_residency(int32_t* dst) const;
    /// Gets what a verify window on CUDA0 knows of the expert cache: the residency tables and the slots.
    strata::core::VerifyHits get_verify_hits() const;
    /// Gets what a layer split's later stage `st`'s verify windows know of its own expert cache.
    strata::core::VerifyHits get_verify_hits(const GpuStage& st) const;

    /// Tells whether the host has the memory for the prompt path's default CPU share (and says why not on stderr).
    bool is_cpu_share_default_ok();
    /// Arms the prompt path's CPU share as a one-conversation run does: by default when one GPU's cache leaves experts
    /// for the CPU and the host has the memory for it.
    void arm_prompt_cpu_share();

    /// Tells the prompt path the share of the expert bytes that are pinned in RAM (it sizes its streamed ring and the
    /// slots it may borrow by it).
    void set_prompt_pinned_share();
    /// --resident-experts: copies the experts no GPU cache holds into RAM (page-locked when it can), except
    /// `stage_pairs` (the (layer, expert) pairs other GPUs hold); the experts of the slots the prompt path may borrow
    /// (from slot `lend_from` on; -1: none) it keeps as far as RAM allows.  Returns 0, otherwise the exit status for
    /// main().
    int pin_resident_experts(const std::vector<std::pair<int32_t, int32_t>>& stage_pairs, int64_t lend_from);
    /// Settles the expert tiers once the RAM copy is decided: the asynchronous adaptive tier off without one, and the
    /// mapped view of the experts closed when they are read unbuffered.
    void finish_expert_tiers();
    /// strata-q38: makes `source` the one the CPU pool and the prompt path read experts from (it outlives the model),
    /// in place of the files or the arena, and stops the file tier's routing-aware prefetch.  Before the prompt path
    /// is made.
    void use_expert_source(strata::core::ExpertSource& source);
    /// Turns --adapt-async off and says why.
    void turn_off_adapt_async(const char* why);

    /// Gets the stage that runs layer `l` with a layer split (0: CUDA0's).
    int stage_of(int64_t l) const;
    /// Gets the rows a verifier's window holds: the solo path's drafts (--spec rows), the plain batch windows (the
    /// slots) and, with STRATA_PIPED_MTP, a group's rows (its slots, each its token and its proposals).  Final only once
    /// load() has fitted the slots and the groups, at the verifiers' init.
    int get_batch_max_t() const;
    /// Tells whether strata-q38 loads this model: its page pools hold the experts and the conversations
    /// (`open_expert_pages` is set), so the old engine's features that size slots or state by their own rules stay off.
    bool is_paged() const { return (bool) open_expert_pages; }
    /// Gets the bytes of the GDN state that one snapshot of session `s0` copies.
    size_t gdn_snapshot_bytes(const strata::core::SessionState& s0) const;

    Options& o;                      ///< the engine's settings, owned by the caller
    int primary_device = 0;          ///< STRATA_PRIMARY_DEVICE: the visible device the engine's own session runs on
    bool pcie_given = false;         ///< --pcie-frac was given: every stage takes that share, no probe
    std::vector<int64_t> split_at;   ///< --layer-split: the first layer of each later stage
    std::vector<int> split_devs;     ///< the device of each later stage
    bool split_auto = false;         ///< --layer-split auto: the split points are searched once the weights are in
    bool split_same = false;         ///< a split on one GPU (--split-device 0) whose stages share CUDA0's state
    /// strata-q38: a layer split without --serve, each stage with weights, an expert cache and a session of its own
    /// on its card, CUDA0 too (--split-device 0,..: several stages on one GPU, the test of a split on one card)
    bool own_stages = false;
    /// --trim-stage-weights took effect: the layer split's cards hold the dense weights of their own layers only
    /// (CUDA0 too, unless the split was searched), so strata-q38 does not move its cut
    bool stage_weights_trimmed = false;
    /// A layer split across GPUs; cleared by --split-skip-if-fits before any stage loads
    bool multi_gpu = false;
    bool remote_caches = false;      ///< --expert-cache-remote asks for a helper GPU's expert cache
    strata::kernels::RopeScaling rope_cfg;   ///< the context extension, resolved from the flags and the model file
    std::string err;                 ///< the last failure's message
    bool native_pack = false;        ///< the pack holds native (IQ / Q-K) experts, run by verify windows only
    strata::core::ModelGeometry g;
    int64_t K = 0;                   ///< experts per token
    uint64_t pool_bytes = 0;         ///< the size of `arena`
    void* arena = nullptr;           ///< the device memory the pack's dense weights live in
    strata::core::WeightTable wt;
    strata::core::NativeDense native_dense;
    int32_t* d_mrope = nullptr;       ///< --vision: each cell's (t, h, w) for the rope kernels, on the device
    std::vector<int32_t> mrope_host;  ///< ... and its host copy
    std::string cvec_summary = "0";   ///< the loaded control vectors as INFO reports them
    uint64_t cvec_digest = 0;         ///< identifies the loaded control vectors, for session files (0: none)
    strata::core::SessionState ss;    ///< the engine's own session
    void* sbuf = nullptr;             ///< `ss`'s memory, sized to CUDA0's own layer range after the layer-split search
    cudaStream_t main_stream = nullptr;
    void* main_cs = nullptr;          ///< `main_stream` as the session functions take it
    uint64_t half_stride = 0;         ///< --dump-halves: the floats of one layer's record
    std::FILE* half_dump = nullptr;   ///< --dump-halves: the file
    float* half_stage = nullptr;      ///< --dump-halves: the pinned staging buffer the graphs copy into
    strata::core::Doorbell db;        ///< the GPU's hand-off to the CPU expert pool
    strata::kernels::PleTable ple_table;   ///< the per-layer embedding table (in the model's second GGUF shard)
    /// The residency policy, not --ple-io ram, locked `ple_table` in RAM: the conversation cache may give the lock back
    /// (#4518)
    bool ple_lock_by_policy = false;
    /// strata-q38: the bytes its RAM page pool will take (--ram-mib), which the PLE table's residency rule counts in
    /// place of the expert arena; < 0: the arena (every expert's blob)
    int64_t ram_pool_bytes = -1;
    float* d_parts = nullptr;         ///< the CPU pool's expert outputs, K rows, on the device
    std::vector<std::unique_ptr<GpuStage>> stages;   ///< a layer split's later stages, in layer order
    GpuStage* last_st = nullptr;      ///< the last of `stages` (it holds the head), or null without a split
    bool mtp_first = false;           ///< --pipeline-windows: the drafter runs on the first stage's card
    bool df2_first = false;           ///< a layer split's DFlash2 runs on the first stage's card (STRATA_DF2_STAGE=first)
    std::vector<std::pair<int32_t, int32_t>> profile;          ///< CUDA0's (layer, expert) pairs, hottest first
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;   ///< --expert-profile-save: the whole ranking as loaded
    bool pf_borrow = false;           ///< the prompt path borrows expert-cache slots instead of owning its buffers
    int64_t split_pf_mib = 0;         ///< a layer split's own prompt buffers per stage, MiB (0: none or borrowed)
    int64_t pipe_window_mib = 0;      ///< --pipeline-windows: the second verifier each stage allocates, MiB
    std::vector<std::pair<int32_t, int32_t>> profile_all;      ///< a layer split's whole ranking, before stages' shares
    /// The layer ranges a session is carved in, one per stage (the engine's own session and each batch slot's alike)
    std::vector<std::pair<int64_t, int64_t>> session_layers;
    bool batch_mtp = false;           ///< --batch-mtp or STRATA_BATCH_MTP: batch slots draft with the MTP layer
    /// STRATA_PIPED_MTP (opt-in): --batch-mtp's proposals in the pipelined groups of a layer split, one slot per group
    bool piped_mtp = false;
    int pm_drafts = 0;                ///< STRATA_PIPED_MTP=N: the most proposals per slot and window (0: off)
    /// --batch: each slot's session, one per stage
    std::vector<std::vector<std::unique_ptr<strata::core::SessionState>>> bslot_ss;
    strata::core::MtpDrafter mtp;
#if defined(STRATA_HAVE_DFLASH2)
    strata::core::Dflash2Drafter df2;   ///< --draft dflash2
    bool use_df2 = false;
    /// STRATA_BATCH_DF2: one drafter per batch slot (the solo drafter's pack, each its own ring)
    std::vector<std::unique_ptr<strata::core::Dflash2Drafter>> slot_df2;
    bool batch_df2 = false;           ///< the slots draft with DFlash2 (set once the slot drafters are allocated)
#else
    static constexpr bool use_df2 = false;
    static constexpr bool batch_df2 = false;
#endif
    /// --draft dflash2 without --draft-fallback mtp: no MTP layer is loaded (o.mtp is cleared) and DFlash2 drafts alone
    bool df2_solo = false;
    std::string df2_vocab_file;       ///< the draft vocabulary DFlash2 alone binds its head from (get_df2_vocab_path)
    bool df2_expect_share = false;    ///< DFlash2 will share the MTP drafter's draft head (offer_df2_head)
    /// The draft layer's geometry (the canonical model's MTP head), `static` because MtpDrafter keeps a reference; the
    /// batch slots' draft-KV copies use this one too
    static inline const strata::core::ModelGeometry draft_geometry = strata::core::kCompiledGeometry;
    std::vector<std::unique_ptr<strata::core::MtpDrafter>> slot_mtp;   ///< --batch-mtp: each batch slot's drafter
    const strata::core::WeightRef* wo = nullptr;   ///< the canonical head's weight (output.weight)
    int64_t n_vocab = 0;
    strata::core::NativeHead native_head;
    std::vector<float> logits;          ///< one row of logits on the host
    float* d_logits = nullptr;          ///< one row of logits on the device
    bool prefill_head_alias_bound = false;          ///< --prefill-head-alias: the prompt path's head is the decode's
    const void* prefill_head_alias_at = nullptr;    ///< where the head came back the first time
    strata::core::FileExpertSource src;   ///< the experts read from the files in place
    strata::core::ArenaExpertSource arena_src;   ///< the experts in a RAM arena
    strata::core::ExpertSource* srcp = nullptr;   ///< the one of the two the engine reads experts from (null: none)
    /// The CPU expert pool. In an optional because it starts its threads when constructed, after the expert source
    std::optional<strata::kernels::cpu::ExpertPool> cpu_pool;
    strata::core::ExpertCache xcache;   ///< CUDA0's expert cache (VRAM slots)
    /// strata-q38: opens `xcache` in the pages of the GPU's page pool, for at most the given number of experts, in
    /// place of the cache's own sizing; load() calls it when it would open the cache, before the profile fills it.
    /// Empty (the `strata` program): the cache's own slots.
    std::function<bool(int64_t max_experts, std::string& err)> open_expert_pages;
    /// strata-q38: opens a layer split's later stage's expert cache `st.cache` in the pages of its card's page pool,
    /// in place of load()'s sizing of it from `st.profile` (load() then fills it as before).  Empty: load() sizes it.
    std::function<bool(GpuStage& st, std::string& err)> open_stage_pages;
    bool auto_cache = false;            ///< --expert-cache auto: the cache was sized to the free VRAM
    /// STRATA_EXPERT_NIBBLE: the prompt chunk before load() adjusted it (an explicit --prefill N must stay N)
    int64_t nib_chunk_asked = 0;
    /// STRATA_EXPERT_NIBBLE: CUDA0's expert cache's converted slots (STRATA_EXPERT_NIBBLE_RECHECK hashes them again)
    strata::core::NibbleCache nib_x;
    std::vector<strata::core::NibbleCache> nib_st;   ///< ... and each layer-split stage's, in `stages` order
    bool reserve_adapted = false;       ///< #496: the auto sizing lowered the reserve so a small card's cache fits
    std::array<strata::core::RemoteExperts, 3> remote_experts;   ///< --expert-cache-remote: the helper GPUs' caches
    std::unique_ptr<strata::core::RemoteExpertOpt> remote_opt;   ///< --remote-expert-opt
    strata::core::PeerExperts peer;     ///< --peer-device: the second GPU's expert tier
    Drive drive;                        ///< the CPU pool's adapter, which the layers call with their missed experts
    strata::core::RouterLookahead lookahead;   ///< routing-aware prefetch of the file tier
    std::FILE* routing = nullptr;       ///< --dump-routing: the file
    strata::core::PoolFn pool_fn = nullptr;   ///< what a layer calls for its missed experts (null: --no-pool)
    strata::core::HitFn hit_fn = nullptr;     ///< what a layer calls for its cached experts (null: no cache)
    void* pool_user = nullptr;                ///< `pool_fn`'s argument
    strata::core::SessionGraphs gr;     ///< the captured layer graphs (a canonical pack only)

    // build_hit_tables():
    strata::core::SessionLoopScratch loop_scratch;   ///< the token path's pinned staging, reused by every token
    strata::core::TokenGraph tgraph;    ///< the whole token as one graph (not captured: `captured` false)
    /// Which VRAM slot holds each (layer, expert), `layer * n_expert + expert` (kNotResident: none); empty when the
    /// hits are decided on the host.  With a layer split a slot of the cache of the stage that owns the layer.
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;           ///< `host_res` on CUDA0
    int32_t* d_hit_count = nullptr;     ///< the token graph's count of hits on the device
    strata::core::TokenHits thits;      ///< the token graph's device hit path (`on()` false: none)
    /// The experts' hits can be decided on the device: a cache filled from the profile, and the CPU pool on
    bool graph_hits = false;

private:
    // the file tier's idle policy (refresh_file_tier); STRATA_FILE_TIER_IDLE=0 turns it off
    bool file_tier_idle_ok = false;
    bool gpu_holds_all = false;       ///< every routed expert was in the GPU caches at the last check
    bool io_pf_by_default = false;    ///< the Linux file-tier defaults turned io prefetch on (no STRATA_IO_PREFETCH given)
    bool io_pf_suspended = false;     ///< the policy turned io prefetch off
    bool la_started = false;          ///< the routing-aware prefetch (`lookahead`) runs
    bool la_asked = false;            ///< STRATA_LOOKAHEAD=1 or STRATA_IO_PREFETCH=1: the look-ahead stays on regardless
    bool fs_ahead = false;            ///< the look-ahead feeds the Foresight swap space (STRATA_FS_SLOTS)
    bool la_stats = false;            ///< STRATA_LOOKAHEAD_STATS=1: the look-ahead scores its predictions
    strata::core::NativeEmbed native_embed;   ///< the native pack's embedding table, which set_native_embed points to
    std::vector<float> ple_emb_host;          ///< the per-layer embedding row `ss.ple` stages on the host
};

/// Allocates the hand-offs of a layer split's verify windows (`hand`: one per boundary between stages, of `rows` rows
/// of geometry `g`), zeroed, in mapped page-locked memory that a stage on any card reads; false when one did not fit.
bool alloc_split_handoffs(const strata::core::ModelGeometry& g, int rows, std::vector<float*>& hand);
/// How many cache slots a prompt-path loan leaves behind: 128 on a card with room; the cache sizing lowers it to 16 on
/// a card too short for that (get_auto_lend_pct then lends 100%).
extern int64_t g_lend_keep;
/// Round 0216 (STRIX-NEXT): true on an integrated GPU (a Strix Halo) whose expert complement is page-locked in RAM.  There the free figure the
/// serve checks read counts that pinned memory about twice (the figure is the driver's hipMemGetInfo, src/core/device.cu only lowers it by the DXGI
/// budget gap: 12,244 MiB after the cache, then 2.0 x the complement + 691 MiB less once it is pinned; RESULTS-STRIX2.md), so it reads 0 on a start
/// that works and a larger --vram-reserve-mib only moves experts from the cache into the complement.  The checks say so instead of advising a reserve.
/// Set by Model::pin_resident_experts; read by the serve loop's VRAM checks.
extern std::atomic<bool> g_uma_pinned_complement;
/// Tells whether a start short of VRAM may be moved off the GPU (strata/core/spill_gate.hpp): a discrete card only,
/// not Metal and not an integrated GPU (Strix Halo and the other APUs share the host's RAM).
bool spill_discrete_gpu();
/// Gets " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the CUDA error first).
std::string vram_free_note();
/// Gets the share (%) of an expert cache's slots the prompt path may borrow with --prefill auto.
int64_t get_auto_lend_pct();
/// Ends the process with one line when an integrated GPU (HIP) cannot launch beside the prompt path's own buffers for a
/// `chunk` above the 16384 default (`borrows`: the chunk borrows cache slots instead; `cache_gib`: the expert cache, for
/// the message).  Called once the buffers are in place, before any prompt is read.
void uma_refuse_chunk_if_full(int64_t chunk, bool borrows, double cache_gib);
/// Prints the process's mapped file pages (the arena) at a step of the startup when STRATA_RSS_TRACE is set (Linux).
void rss_probe(const char* where);
/// Prints the VRAM left after a step of the startup when STRATA_TRACE is set (and checks the device with 2).
void mem_mark(const char* where);
/// Tells whether PcieAuto also refines a layer split's shares (strata/core/pcie_share_rule.hpp: on for a discrete
/// CUDA card, opt-in STRATA_PCIE_AUTO_SPLIT=1 on HIP and an integrated GPU).
bool pcie_auto_split_ok();
/// Tells whether the HIP defaults of the decode speed push apply to the current card (gfx12; always on CUDA).
bool decode_defaults_arch_ok();
/// Gets the host-to-device bandwidth the last PCIe probe measured, GB/s (<= 0: none ran).
double get_last_h2d_gbps();
#if !defined(_WIN32)
/// Tells whether `path` is on a rotational disk (1), not (0) or unknown (-1), from sysfs; `dev` gets the disk's name.
int on_rotational_disk(const std::string& path, std::string& dev);
#endif

}  // namespace strata::qwen38
