// include/strata/models/qwen38/options.hpp - the settings of the Qwen3.8 engines: parse_options (options.cpp) fills
// them from the command line, and the model loader (model.hpp) and the engines' serve loops read them.
#pragma once

#include "strata/common/kernels/coupled_draft.hpp"
#include "strata/common/kernels/cpu/pool.hpp"
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::qwen38 {

/// The engine's settings, filled from its command line (print_usage() lists the flags).
struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    /// THE CONTEXT EXTENSION (rope scaling, rope_scaling.hpp).  These knobs resolve to ONE process config,
    /// set once before `session_init` builds the rope table and the graphs capture the kernels.  There is
    /// deliberately no per-request form: K sits in the cache POST-RoPE, so one cache must never mix two
    /// scalings.  Precedence: an EXPLICIT flag over the model file's rope keys over the struct defaults -
    /// `none` and `1` are explicit values (the opt-outs), the absent flag is not.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn (llama.cpp's names); empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F: the extension factor; 0 = the flag is absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N: 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N: 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Every shard of --native's model (strata::gguf_split_paths: the metadata shard first; a missing shard is an
    /// error) and of --native-head-gguf's (the same list unless that names another model).
    std::vector<std::string> native_shards, native_head_shards;
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// The token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py: BF16 as shipped)
    std::string embd_gguf;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    bool ple_io_explicit = false;      ///< --ple-io was given: the engine only warns when direct measures slow
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 256;   // the prompt path reads a chunk's rows at once: 64 left the SSD half idle (32K: 303 -> 189 ms)
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    bool kv_grow = false;              ///< the elastic K/V (--kv-grow, STRATA_KV_GROW=1; kvg_ensure)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    int pool_tasks = 0;           ///< Batched CPU expert tasks per phase; 0 keeps the existing policy
    /// #272: the pool's core layout; `all` (the default) is the layout it always had, auto / p-cores are opt-in
    strata::kernels::cpu::PoolAffinity pool_affinity = strata::kernels::cpu::PoolAffinity::All;
    /// --pool-affinity measure (opt-in, hybrid CPUs, native packs): `all` and `p-cores` are timed on this pack's
    /// own experts at start and the faster layout is taken (measure_pool_affinity)
    bool pool_measure = false;
    /// --host-core first|last|sibling (STRATA_HOST_CORE): the host thread's core (see HostCore in pool.hpp)
    std::string host_core;
    /// --pool-cores LIST (STRATA_POOL_CORES; experimental, off by default): the logical processors the pool's workers are pinned to
    std::string pool_cores;
    /// --aux-cpus off|auto|LIST (STRATA_AUX_CPUS): where the non-pool, non-host threads run (strata/base/platform/aux_cpus.hpp)
    std::string aux_cpus;
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool coupled_draft = strata::core::coupled_draft_env(); ///< Coupled draft sampling for MTP drafter under sampling
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    std::string shared_expert_arena; ///< Linux: optional file backing for the resident arena shared by processes
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// `--resident-experts` (the low-RAM PC's resident mode, chosen by setup): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), 4 GiB of RAM headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    bool resident_cpu_explicit = false;   ///< #384: --resident-cpu-experts given by itself (not only implied)
    /// CS-T `--resident-budget-gib N`: the resident mode with a RAM budget - the N GiB of experts the GPU cache does
    /// not hold that the expert profile ranks hottest are copied into RAM, the rest are read from the files in place
    /// (the GGUF shards when the pack has no experts.bin).  0 = the whole complement (--resident-experts).
    uint64_t resident_budget = 0;
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/models/common/kernels/cuda/s2_expert_grouped.cu`), it is wired at line
    /// ~660 via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::array<bool, 3> expert_cache_remote_auto{};
    bool remote_expert_opt = false; ///< optional optimization of the existing remote-expert decode path
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// Multi-GPU: a second expert tier on CUDA device `peer_device` (-1 = off), `peer_reserve_mib` left free
    /// there, `peer_slots` caps its size (0 = as many as fit), `peer_adapt_swaps` per adaptive round
    /// (-1 = adapt_swaps).
    int peer_device = -1;
    int peer_reserve_mib = 600;
    int64_t peer_slots = 0;
    int peer_adapt_swaps = -1;
    /// the prompt path's rows per layer the peer computes: -1 = half of chunk x top-k, 0 = the prompt path stays on
    /// the primary
    int64_t peer_prefill_rows = -1;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// #477 (--serve, opt-in): where to save what the adaptive tier learned, as a profile `--expert-profile` reads
    /// (the resident experts first, then the routing counted since the start); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (#496: no smaller automatic reserve)
    /// #533 (opt-in): the expert cache in physical segments (CUDA virtual memory management), so the serve loop's
    /// `VRAM <reserve_mib>` command can give part of it back to other programs and take it again.  Off: one cudaMalloc.
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    /// `--vram-reserve-later-mib N`: the reserve on a layer split's later cards (default: the same as the first).
    /// A card that drives no display needs less than the one the monitors are on.
    int vram_reserve_later_mib = -1;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  A chunk of
    /// stream_all_min tokens or more (1024 by default) streams every expert the GPU does not hold once, routed or not
    /// (a smaller one only the experts it routes to), so a bigger chunk streams fewer bytes per token (the "ubatch"
    /// effect).
    bool prefill_auto = false;
    /// #282, opt-in: the largest chunk `--prefill auto` may take - 8192 by default; `--prefill auto:16384` or
    /// `auto:32768` (or STRATA_PREFILL_AUTO_MAX) lets it go further, never past the context
    int64_t prefill_auto_max = 8192;
    /// #1630: a bare `--prefill auto` stops here on a Windows HIP build (0: no cap). A chunk the user names is not capped.
    int64_t prefill_auto_cap = 0;
    /// a bare `--prefill auto` (no ":N", no STRATA_PREFILL_AUTO_MAX): the gfx12 Linux rule below may raise its ceiling
    bool prefill_auto_bare = false;
    bool prefill_gfx12_16k = false;   ///< the gfx12 default raised the auto chunk ceiling to 16384 (a layer split takes 8192 back, 0.1.44)
    /// --prefill-pipe (opt-in): a layer split's prompt chunks sized for its pipeline (Prefill::pipeline_chunk): 0 off,
    /// 1 today's chunk count evened out, a number > 1 the rig's b/a in tokens (setup's --calibrate measures it).
    /// < 0: not given - STRATA_PREFILL_PIPE, else off.
    double prefill_pipe = -1.0;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
#if defined(STRATA_GPU_METAL)
    int spec = 2;   // Metal: a window of 2 by default (measured best on the M5 Max; an explicit --spec wins)
#else
    int spec = 0;
#endif
    /// --serve runs up to this many sequences at once, one token each per batch window (0 = off).  A request
    /// sent as `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual path, then
    /// continues in slot <slot> of the batch windows (`BT <slot> <id>` lines, then `BDONE <slot> ...`).
    int batch = 0;
    /// With an explicit --layer-split, every GPU loads only its own layers' dense weights instead of the whole
    /// model's: the VRAM they took goes back to the expert cache (opt-in)
    bool trim_stage_weights = false;
    /// The --batch slots in this many groups pipelined through the stages of a layer split (stage k
    /// runs one group while stage k+1 runs another).  1 = every slot in one window, stage after stage.
    int batch_groups = 1;
    bool batch_groups_set = false;    ///< --batch-groups was given (a number or auto): no default then
    bool batch_groups_auto = false;   ///< --batch-groups auto (#417 stage 1): one group per stage of a layer split, if it divides --batch
    bool batch_mtp = false;      ///< --batch-mtp / STRATA_BATCH_MTP=1 (opt-in): one MTP proposal per batch slot
    std::string spec_oracle;
    int spec_corrupt = 0;
    /// Benchmarks and A/B checks (eddoursul's fork, F19): the run emits this continuation (token ids) instead of
    /// the window's argmax, and a draft is accepted when it matches it, so runs with different speculation
    /// settings process the same text.
    std::string spec_follow;
    /// Per verify window: a line with its size, a hash of its final residual rows and its argmaxes.
    std::string window_hashes;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    bool pcie_frac_auto = false;     ///< single-GPU serve (CUDA + HIP/gfx1100), no --pcie-frac: the share is refined in the first decode windows (PcieAuto)
    double pcie_probe_gbps = 0.0;    ///< the start-up probe's reading (for PcieAuto)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    bool adapt_every_given = false;   ///< --adapt-every was on the command line (the per-card warm-up interval below does not override it)
    int adapt_warm_every = 0;         ///< dflash2-kern: > 0: the interval for the first adapt_warm_rounds verify rounds of a speculative run
    int64_t adapt_warm_rounds = 0;
    int adapt_iv(int64_t rounds) const { return adapt_warm_every > 0 && rounds < adapt_warm_rounds ? adapt_warm_every : adapt_every; }   ///< the interval in force at verify round `rounds`
    /// the usage counts are multiplied by this after each adaptation (--adapt-decay); strata-q38 reads it as a factor
    /// per 4 decode steps (PagedSettings::popularity_decay)
    float adapt_decay = 0.7f;
    /// an expert is swapped into the VRAM tier only when its decayed routing count beats the one it replaces by more than
    /// this (--adapt-min-gain; both the blocking and the asynchronous tier)
    float adapt_min_gain = 1.5f;
    /// --adapt-async 1 (--serve with the resident RAM mode; opt-in): the adaptive tier's rounds advance between verify
    /// windows on a helper thread instead of one window waiting for a whole round (see the tier in the --serve block).
    /// Not bit-exact run to run.  0 = the blocking tier.
    int adapt_async = 0;
    int adapt_async_given = -1;      ///< --adapt-async on the command line (-1: not given; then the engine may decide, adapt_async_default.hpp)
    bool adapt_async_self = false;   ///< the engine measures and chooses the asynchronous tier during the session (adapt_async_default.hpp)
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: ~7% slower; not bit-exact against no-split, the batch shape of its windows differs)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// --split-skip-if-fits (opt-in; the server passes it for a config's "split_skip_if_fits"): with
    /// --layer-split auto, run on CUDA0 alone when it holds every profiled expert pair plus the whole session (KV),
    /// the drafter and the reserve - a split then only adds hand-offs (two RDNA4 cards: 1,384 vs 1,794 tok/s at 4K)
    bool split_skip_if_fits = false;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    int64_t conversation_cache_mib = -1; // host RAM for independent conversations; -1: not given, --serve sizes it from the free RAM (#3493), 0: off
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    /// --serve: park only conversations of at least this many tokens.  Every parked conversation costs one of
    /// conversation_cache_slots whatever its length, so without it a client that interleaves short side requests
    /// with one long conversation evicts the long one (oldest first).  0 = park every conversation.
    int64_t conversation_cache_min_tokens = 0;
    /// --serve: park only after the conversation has advanced this many tokens past the copy it would supersede.
    /// A conversation that moved a few hundred tokens writes a full snapshot again, and the turn after that drops it
    /// (drop_superseded) after writing the same bytes once more, so on a long single conversation every turn pays a
    /// whole-snapshot copy for nothing.  0 = park every turn, as before.
    int64_t conversation_cache_park_gain = 0;
    /// --memory-limit-mib N / STRATA_MEMORY_LIMIT_MIB: the total RAM this engine's container or cgroup may use, for a
    /// container that cannot see its real limit (0 or unset = none; the flag wins over the variable)
    int64_t memory_limit_mib = -1;
    /// --serve SAVE: disk space a session file must leave free where it is written (MiB; 0 = no check)
    int64_t session_min_free_mib = 4096;
    bool session_save_reclaim = false;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    bool prompt_cache_tail = false;   // optional extra checkpoint at an existing near-tail chunk boundary
    bool prompt_cache_spread = false;   ///< --prompt-cache-spread: keep checkpoints spread back from the end, not the latest
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve (#458, opt-in): the role token of a short turn the server puts right before the new assistant turn
    /// (the "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>, as before).
    int64_t tail_role_token = -1;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    bool prefill_head_alias = false;   // lend the native output head's VRAM to prompt buffers between request phases
    /// --pipeline-windows N (--serve, a layer split on exactly two GPUs; opt-in): 1 = a prompt's short reads through
    /// the verify windows run with the stages overlapped (stage 0 reads window K+1 while stage 1 reads window K);
    /// 2 = the decode windows too: stage 0 runs window K+1 speculatively while stage 1 verifies window K (rolled back
    /// when K is not accepted as guessed).  0 = off (the stages take turns).
    /// -1 (not asked for) = 2 on gfx12 (HIP) layer splits of 3 or more stages, else 0; --pipeline-windows N asks for it.
    int pipeline_windows = -1;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/common/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// --lookup-chain K (opt-in, 0 = off): prompt lookup CHAINED after the MTP's proposal.  The suffix drafter matches
    /// the context followed by the MTP's drafts and appends up to K of the tokens that followed that match to the same
    /// verify window (the window grows to at most 8).  Greedy verification keeps the output; its own counts are kept.
    int lookup_chain = 0;
    int lookup_chain_min = 3;   ///< --lookup-chain-min M: the shortest match (in tokens, drafts included) it extends on
    /// --mtp-hnorm stream (opt-in): the draft layer's pre_fc_norm_hidden normalizes each hyper-connection stream on
    /// its own (llama.cpp's qwen4exp MTP graph) instead of one RMS over all four (the default, tools/mtp_probe.py).
    bool mtp_hnorm_stream = false;
    /// --mtp-q4 (opt-in): the draft layer's projections and draft head from 4-bit copies (fewer bytes per draft step)
    std::string mtp_q4;   ///< "" (off), all, proj (the projections only) or head (the draft head only)
    /// --mtp-draft-vocab FILE (opt-in): the draft head's token subset from FILE instead of <mtp>/draft_vocab.bin
    std::string mtp_draft_vocab;
    /// --draft dflash2 --draft-model PACK (opt-in, CUDA / HIP): the DFlash2 block drafter for Qwen3.8-Flash-Next (tools/dflash2_pack.py) drafts instead
    /// of the MTP layer, which stays loaded as the fallback.  Needs --mtp, --spec and a native pack; a missing pack falls back to the MTP.
    std::string draft, draft_model;
    double draft_conf = 0.0;   ///< --draft-conf P: with --draft dflash2 the window ends where the product of the selector's probabilities falls below P (replaces --spec-min-p)
    int draft_cap = 0;   ///< --draft-cap N: the widest verify window with --draft dflash2 (0 = by card class, see where the drafter loads)
    /// A control vector on the residual stream (strata/models/qwen38/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

// `--resident-experts`, and the start-up switch to it when the expert arena cannot fit the RAM (arena_fits_ram)
inline void take_resident_experts(Options& o) {
    o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
    o.resident_headroom = 4ull << 30;
    // A/B arms: STRATA_RESIDENT_PIN=0 keeps the copy pageable (the --resident-cpu-experts form);
    // STRATA_RESIDENT_HEADROOM_GIB=N leaves N GiB of the available RAM free instead of 4
    if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
        o.resident_pin = false;
    if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
        o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
}

/// Gets Verifier::set_pcie_mode's code for `--pcie-mode`: 0 DMA, 1 direct, 2 the copy kernel (auto, kernel).  Auto is
/// the copy kernel for every pack: DMA (the native packs' default until 0.1.13) has the host call cudaMemcpyAsync +
/// cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise, and issue #31's thread dumps
/// show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy kernel needs no host CUDA call
/// there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
inline int get_pcie_mode_code(const Options& o) { return o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2; }
/// Gets ExpertDispatch::pcie_num, the 256ths of a window's expert misses that the GPU reads over PCIe, for a share
/// `frac` (clamped to 0..1).
inline int get_pcie_num(double frac) { return std::clamp((int) (frac * 256.0 + 0.5), 0, 256); }

/// Prepares the process before any GPU call: the runtime's switches (eager kernel loading, AMD's pinned copies of a
/// mapped arena) and the GPUs `--gpu` names.  Warns about CUDA_LAUNCH_BLOCKING.
void prepare_process(int argc, char** argv);
/// Tells whether CUDA_LAUNCH_BLOCKING is on (never on HIP), with which no verify window can finish.
bool is_launch_blocking();
/// Opens the GPU's hardware queues in a fixed order on RDNA4 (see options.cpp); nothing elsewhere.  Call on the main
/// thread before any other thread exists.
void prime_gpu_queues();
/// Prints the command line's help (every option of `strata` for a Qwen3.8 model) to stderr.
void print_usage();
using strata::serve::parse_i64_list;
/// Reads the command line into `o`; `have_tokens` tells whether it gave the prompt (--tokens, --tokens-file).  Returns
/// -1 to go on, otherwise the exit status for main() (--help, or a bad option, whose reason has been printed).
int parse_options(int argc, char** argv, Options& o, bool& have_tokens);
/// Turns on the switches that are exact and faster on the GPUs found (gfx1151, gfx12, sm_86 ...), keeping the user's
/// own settings; call after parse_options, before the model loads.  Returns -1 to go on, otherwise the exit status
/// (a gfx906 build on a card it cannot run).
int apply_device_defaults(Options& o);

}  // namespace strata::qwen38
