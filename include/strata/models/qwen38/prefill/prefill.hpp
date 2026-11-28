// include/strata/models/qwen38/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include <array>

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/models/qwen38/layer.hpp"
#include "strata/models/qwen38/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>

namespace strata::core { class PeerExperts; }
namespace strata::kernels::cpu { class ExpertPool; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_direct = 0;     ///< ...of which MMQ read in their cache slots, not gathered (STRATA_MMQ_DIRECT=1)
    int64_t experts_cpu = 0;        ///< ...computed on the CPU pool instead of streamed (STRATA_PREFILL_CPU_SHARE)
    double cpu_share = 0;           ///< ...the share of the streamed ones it took last (measured by default)
    double ms_ple = 0;
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

/// HIP: one tiny BF16 and one tiny FP16 product on a private handle, so the first prompt does not pay hipBLAS's one-time
/// library loads (about 2.5 s on an R9700); once per device, later calls wait for the first and return.  Safe to call from a
/// thread before the expert arena loads.  STRATA_PREFILL_WARM=0 makes it a no-op.  Nothing on CUDA.
void warm_gemm_libraries(int device);


class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    /// `why` (optional): on a decline (false, `err` empty) the reason, a static string.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err, const char** why = nullptr);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
              bool borrow_spill = false);
    /// Reads the next prompts into `ss` instead of `init`'s session; `ss` is carved like it (the same layers and
    /// context).  strata-q38 reads each conversation's prompt into the conversation's own session.
    void set_session(core::SessionState& ss);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s, or
    /// raise_chunk_limit's) in `borrow` - a request lends only the slots its prompt needs - and, with
    /// set_host_borrowed, the host buffers in `host` (host_bytes_needed's bytes).  The stream must be idle (between
    /// prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err, void* host = nullptr,
                  uint64_t host_bytes = 0);
    /// With device and host buffers borrowed, between prompts: lets `relayout` take chunks of up to `chunk` tokens,
    /// growing the buffers of this path's own that `init` sized for its chunk.  False (and `err`, the old limit
    /// kept) when they cannot grow.  strata-q38 calls it when the live prompt chunk is raised.
    bool raise_chunk_limit(int64_t chunk, std::string& err);
    /// Before `init`: the host buffers that grow with the chunk (the PLE rows' uploads, a layer split's hand-offs)
    /// and the stager's ring come from `relayout`'s host region instead of page-locked allocations of their own, so
    /// that they exist only while a prompt is read.  strata-q38 lends them from the RAM's page pool.
    void set_host_borrowed(bool on) { host_borrowed_ = on; }
    /// Host bytes `relayout` takes from its host region with set_host_borrowed for chunks of `chunk` tokens, on a
    /// stage that hands its rows to another (`hand_off`) or not.
    static uint64_t host_bytes_needed(int64_t chunk, bool hand_off);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The chunk size from which a chunk streams every expert the GPU does not hold, routed or not; a smaller one
    /// stages only the experts it routes to.  1024, or STRATA_PREFILL_STREAM_MIN, or the CPU share's limit once that
    /// is armed (3072 when STRATA_PREFILL_CPU_SHARE is set).
    static int64_t stream_all_min_tokens();
    /// --prefill-pipe (opt-in): the chunk a layer split's first stage reads an `n`-token prompt in, for `stages`
    /// stages, chunks of at most `cap` (the buffers' size) and a stage-chunk cost of b + a C with b/a = `ratio` tokens
    /// (0: unknown - STRATA_PREFILL_EQUAL's rule: the same chunk count as `cap` gives, evened out when its last chunk
    /// has stream_all_min tokens or more): the chunk count with the pipeline's least time, see the definition. `cap`
    /// when that is today's split or not 3% better than it. A function of its arguments only.
    static int64_t pipeline_chunk(int64_t n, int64_t cap, int stages, double ratio);
    /// --prefill-pipe for this run: 0 off (the buffers' chunks), 1 today's chunk count evened out (STRATA_PREFILL_EQUAL's
    /// rule), a number > 1 the rig's b/a in tokens (setup's --calibrate measures it). Default: STRATA_PREFILL_PIPE, else 0.
    static void set_pipe(double value);
    /// The stages of the prompt path from this one on (1 without a layer split).
    int stage_count() const;
    /// --prefill-pipe on a layer split's first stage: the chunk an `n`-token segment is read in when it is read whole
    /// (pipeline_chunk over chunk(), the stage count and pipe_setting()), or 0 when the pipe is off here (one stage,
    /// value 0, or a later stage) and the buffers' chunk applies.  The engine plans a segment once, on its whole
    /// length, and reads every part of it with force_chunk, so its chunks do not depend on how it is cut into runs.
    int64_t plan_chunk(int64_t n) const;
    /// One log line for a planned segment (the same as a whole read logs when its plan differs from chunk()).
    static void log_plan(int64_t n, int64_t chunk, int stages);
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);

    int ring_room() const { return ring_room_; }

    /// True when every stream of this path has finished.  The room's close needs it: what a prompt staged into those
    /// slots is written over by the lender's own bytes, and a prompt pass still reading them would take the DMA.
    bool idle(std::string& err);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).  `src`: init gets an
    /// ExpertSource (carve picks the fused MoE layout only then).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 bool src = true);

    /// The same without the streamed ring: what the chunk's own buffers cost.  The auto chunk scan sizes the chunk
    /// first and hands the ring what the chunk leaves over, so it needs the chunk priced on its own.
    /// What `init` allocates when the prompt path OWNS its buffers: each cudaMalloc rounded up to a 2 MiB page and the
    /// ring as one allocation (the startup sizing of a cache without a loan).
    static uint64_t bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                       bool src = true);
    static uint64_t bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                         bool src = true);

    /// STRATA_MMQ_DIRECT: the guard each sized expert-cache slot keeps after its blob (ExpertCache::set_slot_guard),
    /// so the prompt MMQ can read a resident expert in its slot.  0 unless STRATA_MMQ_DIRECT is 1 or 2 on a CUDA
    /// build with the pointer-table MMQ (STRATA_MMQ_TAB).
    static size_t slot_guard_bytes();

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;
    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;
    /// The first stage only, checked before every chunk but the first: true ends the run there as if the prompt
    /// ended there.  `run` returns true once every stage has read the tokens before that chunk, and `stopped_at()`
    /// says how many of the run's tokens that is (-1 when the run read them all).  A read beside the batch slots
    /// (STRATA_PIPED_CONCURRENT) gives way to a waiting shorter request there.
    std::function<bool()> stop_at_chunk;
    /// Gets a layer's routed expert ids (T x k, -1 = none) once the router chose them, for a policy that counts the
    /// experts' use: the old engine's STRATA_RAM_ADAPT2 seed and the paged engine's adaptive tier.  Setting it costs
    /// one download of the ids per layer on the path that groups them on the GPU (STRATA_PF_FUSED).
    std::function<void(int64_t layer, const int32_t* ids, int64_t n)> on_routes;
    /// KVA: the end of the positions the last run approximated (they are a prefix: chunks from the run's first position
    /// on), i.e. the first position whose final rows are real; 0 when that run approximated nothing.  Raised chunk by
    /// chunk, before on_chunk reports the chunk, so the draft layer can skip the chunk's cells.  Reset by every run.
    int64_t kva_floor() const { return kva_floor_; }
    /// Grows this stage's (and the later stages') pinned upload slots to what a text chunk of `T` tokens uploads, so
    /// a run of such chunks allocates no pinned memory (STRATA_PIPED_CONCURRENT: an allocation or free on the read's
    /// thread takes the driver's lock).  Before the run, with nothing of this path in flight.  False: a slot could
    /// not be had (it then grows at first use, as before).
    bool reserve_uploads(int64_t T);

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

    /// The CPU expert pool (decode's, idle while a prompt is read). With STRATA_PREFILL_CPU_SHARE set, a chunk below
    /// stream_all_min() tokens - an agent's tool output - hands it the non-resident experts routed by at most MAXT of
    /// its tokens, fewest first, up to a share of the experts it would stream (`auto`: measured, where both sides end
    /// together). The CPU reads them from RAM (the arena, the page-locked copy, or the mapped experts.bin's pages)
    /// while the rest come over PCIe; their rows go to Dm's tail, as a peer's do. Not bit-identical to the GPU's rows
    /// (the CPU's own activation format). Unset (default) or null: every expert on the GPU. Only for a pool no other
    /// thread runs meanwhile (no batch slots); on a layer split every stage may have it - one stage at a time takes it
    /// for a chunk. Set before `init`.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// With STRATA_PREFILL_CPU_SHARE set and `applies` (a pool will be set): the chunks staged after their routing - the
    /// only ones the share applies to - go up to STRATA_PREFILL_CPU_SHARE_MAX tokens (default 3072) instead of 1024.
    /// Before any `bytes_needed` (the ring and the small-chunk buffers follow the limit); `init` also sets it.
    /// `by_default`: this path is one the share was measured on (CUDA, one GPU, no batch slots), so with the variable unset
    /// the share is on for chunks below 1024 tokens (STRATA_PREFILL_CPU_SHARE=0 turns it off).
    static void arm_cpu_share(bool applies, bool by_default = false);

    /// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): true while `on_chunk` reports a sub-chunk before the end of
    /// its super-chunk.  The session state is already at the super-chunk's end then, so no checkpoint may be taken.
    /// On a layer split it is the last stage's (the one that reports the rows), asked through the first.
    /// One prompt of a multi-prompt read (run_multi): its tokens, the position it starts at and, per stage of a layer split
    /// (index 0 = the first stage), the session it reads into (nullptr / missing: that stage's own main session).
    struct Seg {
        const int64_t* tokens = nullptr;
        int64_t n = 0;
        int64_t pos0 = 0;
        std::vector<core::SessionState*> stage_ss;
    };

    /// Tells whether a run of two chunks or more goes layer by layer (STRATA_PREFILL_LAYER_MAJOR, or the default
    /// here), unless the walk declines it (lm_prepare: an image prompt, no VRAM for it, ...).
    bool is_layer_major_on() const;

private:
    static uint64_t bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool owned_pages, bool src);
    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;   ///< set_cpu_pool
    // Layer-major prefill: the super-chunk's buffers for an n-token prompt (its size, 0 = not this prompt: `why`),
    // the run, and the buffers given back after it.  `cap` (> 0): at most that many tokens (a layer split's stages
    // share one size); lm_prepare_chain sizes every stage of a split.
    int64_t lm_prepare(int64_t n, std::string& why, int64_t cap = 0);
    int64_t lm_prepare_chain(int64_t n, std::string& why);
    void lm_release_chain();
    bool run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    struct LmSub { int seg = 0; int64_t c0 = 0, T = 0, t0 = 0; };   ///< a sub-chunk: prompt, first token, tokens, row in the super-chunk
    bool lm_run(const std::vector<Seg>& segs, int depth, const std::vector<LmSub>* given_super, std::string& err);
    void lm_release();
    bool lm_mid_ = false;
    bool lm_force_ = false;        ///< run_multi: the walk even for a short total
    bool lm_by_default_ = false;   ///< the walk is on because nothing was set (a wider VRAM margin, the measured super-chunk size)
    bool host_borrowed_ = false;                     ///< set_host_borrowed (survives `reset`)
    int ring_room_ = 0;                              ///< set_ring_room (this instance's, and it survives `reset`)
    void* ring_room_base_ = nullptr;                 ///< set_ring_room_base: the bytes the room's slots are
    uint64_t ring_room_bytes_ = 0;
    bool ring_room_active_ = false;                  ///< the room is this prompt's (survives `reset` with it)
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    Prefill* helper_ = nullptr;         ///< set_stage_helper
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    int64_t stopped_at_ = -1;           ///< stop_at_chunk ended the last run after this many tokens (-1: it read all)
    int64_t kva_floor_ = 0;             ///< kva_floor()
    bool pipe_chunked_ = false;         ///< a later stage: the chunks came from a --prefill-pipe plan (set likewise)
    int64_t force_chunk_ = 0;           ///< force_chunk, for the next run (0: none)
    bool force_single_ = false, force_planned_ = false;
    bool bind_stage_helper(int64_t T);  // binds the helper's buffers for a one-chunk prompt of T tokens
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    /// STRATA_HANDOFF_PIPE: the previous stage's per-piece "landed in hand_in_" events (cudaEvent_t[hand_pieces_],
    /// on its device) when it copies the chunk in pieces without a host wait; null with the one-copy hand-off
    const void* hand_ev_ = nullptr;
    int hand_pieces_ = 0;

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    /// Makes the buffers of this path's own that hold one chunk's tokens (outside carve's: the token ids, the step
    /// records, the routing tables, the PLE rows) hold chunks of `T` tokens, the largest `relayout` then takes;
    /// leaves them alone when they hold that already.  `init` and raise_chunk_limit use it.
    bool reserve_chunk(std::size_t T, std::string& err);
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
