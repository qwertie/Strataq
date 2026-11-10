#include "strata/base/gpu/managed_memory.hpp"
// src/models/qwen38/options.cpp - the Qwen3.8 engines' command line: the help text, the option parser and the
// per-device defaults applied before any engine code reads its switches (options.hpp).
#include "strata/base/platform/env.hpp"
#include "strata/models/qwen38/options.hpp"

#include "strata/base/gpu/gfx_arch.hpp"
#include "strata/base/gpu/pinned.hpp"
#include "strata/base/platform/aux_cpus.hpp"
#include "strata/base/platform/host_limits.hpp"
#include "strata/common/kernels/cpu/pool.hpp"
#include "strata/core/arch_defaults.hpp"
#include "strata/core/nv_defaults.hpp"
#include "strata/models/qwen38/adapt_decay.hpp"
#include "strata/models/qwen38/logits_selection.hpp"

#include <cuda_runtime.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace strata::qwen38 {

void print_usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --gpu LIST           run on these GPUs, comma-separated nvidia-smi/PCI indices (default: every\n"
                 "                       visible one); sets CUDA_VISIBLE_DEVICES/CUDA_DEVICE_ORDER inside the engine,\n"
                 "                       so a caller never has to export environment variables\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard); with --native, the model's\n"
                 "                       shard that holds per_layer_token_embd.weight when not given\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm;\n"
                 "                       ram: the whole table locked in RAM at start (VirtualLock on Windows,\n"
                 "                       mlock elsewhere), so no SSD read sits on the prompt or token path\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 256)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); streams with --kv-resident too\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --kv-grow            the K/V takes VRAM only for the cells the requests reach and the expert\n"
                 "                       cache holds the rest, giving slots back as the context grows (one GPU, with\n"
                 "                       --expert-profile; STRATA_KV_GROW=1/0 also). Default: the whole --max-context\n"
                 "                       allocated at start\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --embd-gguf PATH     the token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py:\n"
                 "                       BF16 as the checkpoint ships it; mapped host memory, no VRAM)\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear\n"
                 "                       (position interpolation) or yarn - llama.cpp's types and names.\n"
                 "                       Default: the model file's rope keys, else none. Fixed at startup:\n"
                 "                       K in the cache is post-RoPE, so one run one scaling\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's\n"
                 "                       factor, else 1 = off)\n"
                 "  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and\n"
                 "  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default: sized from the free RAM, up to 8192; 0 = off)\n"
                 "  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)\n"
                 "  --conversation-cache-min-free-mib N  --serve: RAM floor (under MemAvailable, the cgroup limits and\n"
                 "                       --memory-limit-mib) when parking, saving or restoring a session (default 2560)\n"
                 "  --memory-limit-mib N  --serve: the total RAM (MiB) this engine's container or cgroup may use, when\n"
                 "                       the container cannot see its real limit (also STRATA_MEMORY_LIMIT_MIB; the flag\n"
                 "                       wins; 0 = none). Parking and session save/restore keep their floor under it\n"
                 "  --conversation-cache-min-tokens N  --serve: park only conversations of at least N tokens, so\n"
                 "                       short side requests do not take the slots a long one needs (default 0 = park any)\n"
                 "  --conversation-cache-park-gain N  --serve: park only after the conversation advanced N tokens past\n"
                 "                       the copy it would supersede, so a turn that grew by a few hundred tokens does not\n"
                 "                       write a whole snapshot that the next turn drops (default 0 = park every turn)\n"
                 "  --session-min-free-mib N  --serve: disk space a SAVE must leave free (default 4096; 0 = no check)\n"
                 "  --session-save-reclaim  --serve: reclaim unpinned host caches if SAVE needs RAM (opt-in)\n"
                 "  --vram-elastic       --serve (#533, opt-in, NVIDIA): the expert cache in segments (--vram-segment-mib,\n"
                 "                       default 512), so the command `VRAM <reserve_mib>` (the server's POST /v1/vram) can\n"
                 "                       give VRAM back to other programs between requests and take it back later\n"
                 "  --batch N / --slots N  --serve (opt-in): up to N requests decode together in batch slots (2..8\n"
                 "                       normally; --batch-mtp waves more through eight-row windows),\n"
                 "                       each slot with its own session (VRAM like the main one); docs/BATCHING.md\n"
                 "  --batch-mtp          --batch (opt-in, needs --mtp and --spec): each slot also verifies one MTP proposal per\n"
                 "                       window (STRATA_BATCH_MTP=1 does the same); works on one GPU and on a layer split (the\n"
                 "                       slot drafters sit on the last stage's GPU; not with --batch-groups); needs VRAM per slot\n"
                 "  --batch-groups G     --batch with a layer split: the slots in G groups pipelined through the GPUs\n"
                 "  --pipeline-windows N --serve with a layer split on two GPUs (on by default for 2+ stages on AMD gfx12;\n"
                 "                       STRATA_PIPELINE_AUTO=0 or N=0 turns it off): one conversation's verify windows\n"
                 "                       with the GPUs overlapped - 1 = the prompt's short reads, 2 = decode as well\n"
                 "                       (stage 0 runs the next window while stage 1 verifies this one); docs/MULTI_GPU.md\n"
                 "  --trim-stage-weights an explicit --layer-split: each GPU loads only its own layers' dense weights\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing\n"
                 "                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)\n"
                 "  --prompt-cache-spread  --serve: keep the checkpoints spread back from the conversation's end (dense\n"
                 "                       near it, thinner further back) instead of the most recently used: a client that\n"
                 "                       rewrites its history N tokens back reads about N again, not the whole conversation\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of\n"
                 "                       the conversation checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --prefill-head-alias --serve: unload the native output head while each fresh prompt is read,\n"
                 "                       then release prompt buffers and reload it before decode\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --spec N             the MTP drafter's verify window: how many tokens it proposes per check (setup\n"
                 "                       writes 4; --suffix-draft lets it grow by 2, up to 8, where a repeat is likely)\n"
                 "  --spec-min-p P       how sure the draft layer must be to extend a verify window by another guess\n"
                 "                       (setup writes 0.5; --calibrate measures it on this PC, see docs/DETAILS.md)\n"
                 "  --lookup-chain K     opt-in: after the MTP's drafts, add up to K prompt-lookup drafts that continue\n"
                 "                       them (the window grows to at most 8; default 0 = off)\n"
                 "  --lookup-chain-min M  the shortest context match --lookup-chain extends on (default 3)\n"
                 "  --mtp-hnorm pooled|stream  the draft layer's hidden-input norm: one RMS over all four streams\n"
                 "                       (default) or one per stream (llama.cpp's MTP graph)\n"
                 "  --mtp-q4 all|proj|head  opt-in: the draft layer's projections and/or draft head as 4-bit copies\n"
                 "  --mtp-draft-vocab FILE  opt-in: the draft head's token subset (default <mtp>/draft_vocab.bin)\n"
                 "  --draft mtp|dflash2  the draft source (default mtp); dflash2 needs --draft-model PACK (tools/dflash2_pack.py)\n"
                 "                       and keeps the MTP layer loaded as the fallback (opt-in, Flash-Next, CUDA / HIP)\n"
                 "  --draft-conf P       with --draft dflash2: end the window where the product of the selector's probabilities falls below P (default 0 = --spec-min-p)\n"
                 "  --draft-cap N        the widest verify window with --draft dflash2 (default by card: 5 on Strix Halo / iGPU, --spec as given elsewhere)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend;\n"
                 "                       auto:16384 / auto:32768 (or STRATA_PREFILL_AUTO_MAX) allow bigger ones\n"
                 "  --prefill-pipe R     opt-in, a layer split: prompt chunks sized so the stages read at once - R is\n"
                 "                       this PC's b/a in tokens (setup's --calibrate measures it); 1 only evens out\n"
                 "                       the buffers' chunk count (STRATA_PREFILL_EQUAL's rule); 0 off (default; or\n"
                 "                       STRATA_PREFILL_PIPE)\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the argmax and\n"
                 "                       accept the drafts that match it, so speculation settings compare on the\n"
                 "                       same text (with --spec; not --serve)\n"
                 "  --window-hashes PATH per verify window, a line with its index, position, size, a 64-bit hash of\n"
                 "                       its final residual rows and its argmaxes: two builds that do the same\n"
                 "                       arithmetic write the same file (use with --spec-follow, --adapt-every 0)\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-profile-save P  --serve, #477: save what the adaptive tier learned (the experts in\n"
                 "                       VRAM, then the routing counted since the start) as a profile at P, on\n"
                 "                       QUIT and every --expert-profile-save-every MIN minutes (default 10;\n"
                 "                       0 = on QUIT only) between requests; start from it with --expert-profile P\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-device1 N|auto  fill CUDA1 with N experts or as many as fit\n"
                 "  --expert-cache-device2 N|auto  fill CUDA2 with more experts\n"
                 "  --expert-cache-device3 N|auto  fill CUDA3 with more experts\n"
                 "  --remote-expert-opt   enable remote-expert decode optimizations (serve mode)\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --peer-device N      a second GPU as an adaptive expert-cache tier (rows over P2P; it also\n"
                 "                       computes its experts' prompt rows).  Not with --layer-split or\n"
                 "                       --expert-cache-device1..3.  Default output unchanged without it.\n"
                 "  --peer-reserve-mib N  VRAM the peer tier leaves free on its card (default 600)\n"
                 "  --peer-slots N       expert slots on the peer (default: what fits)\n"
                 "  --peer-adapt-swaps N  peer cache swaps per adaptation step (default: --adapt-swaps)\n"
                 "  --peer-prefill-rows N  prompt rows per layer the peer computes (default half of chunk x top-k;\n"
                 "                       0 = the prompt path stays on the primary)\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.  On a native pack, N is still the budget of N\n"
                 "                       largest blobs; each layer's slots are that layer's own blob.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on (with --pool-affinity auto or\n"
                 "                       p-cores on a hybrid CPU: P-cores minus 1).  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --pool-tasks N       Batched CPU expert tasks per GU/Down phase (0..4096). Default 0 =\n"
                 "                       3 per participating thread; positive counts are capped by row count.\n"
                 "  --host-core WHERE    The host thread's core: first (the default: the first physical core, as always) or\n"
                 "                       last (the last physical core, the workers on the others).  Windows sends a GPU's\n"
                 "                       interrupts to one logical processor, usually the first, and every copy that lands\n"
                 "                       raises one: a spinning host there waits for them.  Moves threads only, never a\n"
                 "                       result.  Not on hybrid CPUs.  sibling: the first core's other hardware thread\n"
                 "                       (SMT), so the interrupts keep the first logical processor and the workers keep\n"
                 "                       every core; hybrid CPUs too, first without SMT (STRATA_HOST_CORE sets it too).\n"
                 "  --pool-cores LIST    Experimental, off by default: pin the CPU workers to exactly these logical\n"
                 "                       processors (0,2,4,6 or 2-14:2; the numbers Task Manager and lscpu show).  The\n"
                 "                       default is one worker per physical core, which the start-up line names.  Two\n"
                 "                       entries on one core, or the host's own, are kept with a warning; processors the\n"
                 "                       process may not use are left out.  The worker count is the list's length unless\n"
                 "                       --pool-workers says otherwise.  Moves threads only, never a result.\n"
                 "                       STRATA_POOL_CORES sets it too.\n"
                 "  --aux-cpus SET       Where the threads that are neither pool workers nor the host thread run (the\n"
                 "                       adaptive tier's job thread, the prefill helpers, the CUDA driver's threads, ...):\n"
                 "                       off (the default: wherever they were created), auto (the SMT siblings of the host's\n"
                 "                       core, then physical cores with no worker; never a CPU that shares a core with a\n"
                 "                       worker; nothing is done when none is spare) or a CPU list such as 24,26 (Linux; the\n"
                 "                       host's and the workers' CPUs are dropped from it).  Moves threads only, never a\n"
                 "                       result.  STRATA_AUX_CPUS sets it too; a request's strata_tune aux_cpus=0|1 beats it.\n"
                 "  --pool-affinity MODE Worker CPU affinity: all (default: one worker per physical core, as\n"
                 "                       always), auto (hybrid CPUs: P-cores first, then their SMT siblings,\n"
                 "                       then E-cores) or p-cores (P-cores and their siblings only).  measure (opt-in,\n"
                 "                       hybrid CPUs, native packs): all and p-cores are timed on the pack's own experts\n"
                 "                       at start (about a second) and the faster one is taken\n"
                 "  --coupled-draft      enable coupled draft sampling for MTP drafter under sampling (STRATA_SPEC_COUPLED)\n"
                 "  --no-coupled-draft   disable coupled draft sampling (propose argmax drafts)\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --shared-expert-arena FILE  Linux: back the resident arena with one MAP_SHARED file.\n"
                 "                       Put this file on /dev/shm, not ordinary SSD storage.\n"
                 "                       A small header binds an existing backing file to the same pack.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not\n"
                 "                       hold resident in ordinary RAM (and the prompt path's lend region as far as\n"
                 "                       RAM allows); adaptive swaps exchange them, so none is read from the file again.\n"
                 "  --resident-experts   the low-RAM PC's resident mode (setup): --mmap-experts --resident-cpu-experts\n"
                 "                       with the copy page-locked when possible, 4 GiB headroom, plain mmap if it\n"
                 "                       does not fit.  Same answers as --mmap-experts for the same placement.\n"
                 "  --adapt-min-gain F   the adaptive tier swaps an expert in only when its decayed routing count beats\n"
                 "                       the one it replaces by more than F (default 1.5; higher = fewer, surer swaps)\n"
                 "  --adapt-async 0|1    --serve with the resident RAM mode or the arena (one GPU): the adaptive\n"
                 "                       tier's swaps advance between verify windows instead of a window waiting for a\n"
                 "                       whole round (the arena: copy-in only).  Not bit-exact run to run.  Without the flag,\n"
                 "                       the expert arena on one CUDA GPU (no --batch) starts blocking and tries it for a\n"
                 "                       while when the swaps hold the windows up, keeping it only if it is faster; 0 = the\n"
                 "                       blocking tier always (also STRATA_ADAPT_ASYNC=0); 1 = always on.\n");
}

int parse_options(int argc, char** argv, Options& o, bool& have_tokens) {
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (a == "--gpu") { (void) next("--gpu"); }   // applied at startup, before any CUDA call
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!strata::qwen38::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") { o.ple_io = next("--ple-io"); o.ple_io_explicit = true; }
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--kv-grow") o.kv_grow = true;
        else if (a == "--no-kv-grow") o.kv_grow = false;
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--embd-gguf") o.embd_gguf = next("--embd-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--pool-tasks") {
            const char* v = next("--pool-tasks");
            char* end = nullptr;
            const long tasks = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || tasks < 0 || tasks > strata::kernels::cpu::ExpertPool::kMaxTasks) {
                std::fprintf(stderr, "strata generate: --pool-tasks expects an integer in 0..4096 (0 = automatic)\n");
                return 2;
            }
            o.pool_tasks = (int) tasks;
        }
        else if (a == "--host-core") {
            o.host_core = next("--host-core");
            if (o.host_core != "first" && o.host_core != "last" && o.host_core != "sibling") {
                std::fprintf(stderr, "strata generate: unknown --host-core value '%s' (expected first, last or sibling)\n",
                             o.host_core.c_str());
                return 2;
            }
        }
        else if (a == "--pool-cores") o.pool_cores = next("--pool-cores");
        else if (a == "--aux-cpus") {
            o.aux_cpus = next("--aux-cpus");
            std::string aerr;
            strata::aux_cpus::Config acfg;
            if (!strata::aux_cpus::parse_spec(o.aux_cpus, acfg, aerr)) {
                std::fprintf(stderr, "strata generate: %s\n", aerr.c_str());
                return 2;
            }
        }
        else if (a == "--pool-affinity") {
            const std::string v = next("--pool-affinity");
            if (v == "auto") o.pool_affinity = strata::kernels::cpu::PoolAffinity::Auto;
            else if (v == "p-cores" || v == "pcores") o.pool_affinity = strata::kernels::cpu::PoolAffinity::PCores;
            else if (v == "all") o.pool_affinity = strata::kernels::cpu::PoolAffinity::All;
            else if (v == "measure") o.pool_measure = true;
            else {
                std::fprintf(stderr, "strata generate: unknown --pool-affinity value '%s' (expected auto, p-cores, all or measure)\n", v.c_str());
                return 2;
            }
        }
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else if (a == "--coupled-draft") o.coupled_draft = true;
        else if (a == "--no-coupled-draft") o.coupled_draft = false;
        else parsed = false;
        // The chain continues here in a second statement: one chain of 120+ `else if` passed MSVC's limit of 128
        // nested blocks (C1061).  The order of the tests and what each does are unchanged.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--expert-cache-device1" || a == "--expert-cache-device2" || a == "--expert-cache-device3") {
            const size_t r = (size_t) (a.back() - '1');
            const std::string v = next(a.c_str());
            o.expert_cache_remote_auto[r] = v == "auto";
            o.expert_cache_remote[r] = v == "auto" ? std::numeric_limits<int>::max() : std::atoi(v.c_str());
        }
        else if (a == "--remote-expert-opt") o.remote_expert_opt = true;
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--vram-reserve-later-mib")
            o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
            // #282: auto:N (N = 16384 or 32768) or STRATA_PREFILL_AUTO_MAX lets auto take chunks above 8192
            const char* env_max = std::getenv("STRATA_PREFILL_AUTO_MAX");
            const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5)
                                     : (o.prefill_auto && env_max != nullptr ? std::atoll(env_max) : 8192);
            o.prefill_auto_max = want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
            o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
            {   // #1630: Windows HIP, RX 7900 XTX: a first long prompt at 8192-token chunks charged ~70 GB of commit and
                // killed the engine; 7500 and below were flat.  A bare auto stops at 6144; a chunk the user names stays.
#if defined(STRATA_USE_HIP) && defined(_WIN32)
                constexpr bool win_hip = true;
#else
                constexpr bool win_hip = false;
#endif
                const bool bare_auto = v == "auto" && !(o.prefill_auto && env_max != nullptr);
                o.prefill_auto_bare = bare_auto && o.prefill_auto;
                o.prefill_auto_cap = strata::platform::host_limits::auto_chunk_cap(win_hip, bare_auto && o.prefill_auto);
                if (o.prefill_auto_cap > 0) o.prefill_chunk = std::min(o.prefill_chunk, o.prefill_auto_cap);
                if (strata::platform::host_limits::warn_user_chunk(win_hip, bare_auto, o.prefill_chunk))
                    std::fprintf(stderr, "strata: WARNING --prefill %s: prompt chunks of 8192 tokens or more crashed the engine "
                                         "on Windows HIP (RX 7900 XTX: the commit charge jumped ~70 GB, exit code 0xC0000409, "
                                         "#1630); 7500 and below were stable. Kept as asked; a bare --prefill auto stops at %lld\n",
                                 v.c_str(), (long long) strata::platform::host_limits::kWinHipAutoChunkCap);
            }
        }
        else if (a == "--prefill-pipe") o.prefill_pipe = std::max(0.0, std::atof(next("--prefill-pipe")));
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--vram-elastic") o.vram_elastic = true;
        else if (a == "--vram-segment-mib") o.vram_segment_mib = std::atoll(next("--vram-segment-mib"));
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--batch") o.batch = std::atoi(next("--batch"));
        else if (a == "--slots") o.batch = std::atoi(next("--slots"));   // the same as --batch
        else if (a == "--trim-stage-weights") o.trim_stage_weights = true;
        else if (a == "--batch-groups") {
            const char* bgv = next("--batch-groups");
            o.batch_groups_set = true;
            if (std::strcmp(bgv, "auto") == 0) o.batch_groups_auto = true;
            else o.batch_groups = std::atoi(bgv);
        }
        else if (a == "--batch-mtp") o.batch_mtp = true;
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--window-hashes") o.window_hashes = next("--window-hashes");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") { o.adapt_every = std::atoi(next("--adapt-every")); o.adapt_every_given = true; }
        else if (a == "--adapt-decay") o.adapt_decay = strata::program::adapt_decay_checked(std::atof(next("--adapt-decay")));   // #1284: outside 0..1 -> 0.7, said
        else if (a == "--adapt-min-gain") o.adapt_min_gain = (float) std::atof(next("--adapt-min-gain"));
        else if (a == "--adapt-async") { o.adapt_async = std::atoi(next("--adapt-async")) != 0 ? 1 : 0; o.adapt_async_given = o.adapt_async; }
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--split-skip-if-fits") o.split_skip_if_fits = true;
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--session-save-reclaim") o.session_save_reclaim = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib" || a == "--conversation-cache-min-tokens" ||
                 a == "--conversation-cache-park-gain" ||
                 a == "--session-min-free-mib" || a == "--memory-limit-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = (a == "--conversation-cache-slots" || a == "--conversation-cache-min-tokens" ||
                                   a == "--conversation-cache-park-gain")
                                      ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                std::fprintf(stderr, "%s needs a nonnegative integer within range\n", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = number;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else if (a == "--conversation-cache-min-tokens") o.conversation_cache_min_tokens = number;
            else if (a == "--conversation-cache-park-gain") o.conversation_cache_park_gain = number;
            else if (a == "--session-min-free-mib") o.session_min_free_mib = number;
            else if (a == "--memory-limit-mib") o.memory_limit_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-tail") o.prompt_cache_tail = true;
        else if (a == "--prompt-cache-spread") o.prompt_cache_spread = true;
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--prefill-head-alias") o.prefill_head_alias = true;
        else if (a == "--pipeline-windows") o.pipeline_windows = std::max(0, std::min(2, std::atoi(next("--pipeline-windows"))));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--lookup-chain") o.lookup_chain = std::clamp(std::atoi(next("--lookup-chain")), 0, 7);
        else if (a == "--lookup-chain-min") o.lookup_chain_min = std::max(3, std::atoi(next("--lookup-chain-min")));
        else if (a == "--mtp-hnorm") {
            const std::string v = next("--mtp-hnorm");
            if (v != "pooled" && v != "stream") {
                std::fprintf(stderr, "strata generate: --mtp-hnorm takes pooled or stream\n");
                return 2;
            }
            o.mtp_hnorm_stream = v == "stream";
        }
        else if (a == "--mtp-q4") {
            o.mtp_q4 = next("--mtp-q4");
            if (o.mtp_q4 != "all" && o.mtp_q4 != "proj" && o.mtp_q4 != "head") {
                std::fprintf(stderr, "strata generate: --mtp-q4 takes all, proj or head\n");
                return 2;
            }
        }
        else if (a == "--mtp-draft-vocab") o.mtp_draft_vocab = next("--mtp-draft-vocab");
        else if (a == "--draft") {
            o.draft = next("--draft");
            if (o.draft != "dflash2" && o.draft != "mtp") {
                std::fprintf(stderr, "strata generate: --draft takes mtp or dflash2\n");
                std::exit(2);
            }
        }
        else if (a == "--draft-model") o.draft_model = next("--draft-model");
        else if (a == "--draft-conf") o.draft_conf = std::clamp(std::atof(next("--draft-conf")), 0.0, 1.0);
        else if (a == "--draft-cap") o.draft_cap = std::max(0, std::atoi(next("--draft-cap")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--peer-device") o.peer_device = std::atoi(next("--peer-device"));
        else if (a == "--peer-reserve-mib") o.peer_reserve_mib = std::atoi(next("--peer-reserve-mib"));
        else if (a == "--peer-slots") o.peer_slots = std::atoll(next("--peer-slots"));
        else if (a == "--peer-adapt-swaps") o.peer_adapt_swaps = std::atoi(next("--peer-adapt-swaps"));
        else if (a == "--peer-prefill-rows") o.peer_prefill_rows = std::atoll(next("--peer-prefill-rows"));
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
        else if (a == "--resident-experts") take_resident_experts(o);
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { std::fprintf(stderr, "strata generate: --resident-budget-gib needs N > 0\n"); return 2; }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
            o.resident_headroom = 4ull << 30;
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage();
            return 2;
        }
        }
    }
    return -1;
}


int apply_device_defaults(Options& o) {
#if defined(STRATA_USE_HIP)
    {   // gfx1151 (Strix Halo): the switches that are exact there are on by default (strata/core/arch_defaults.hpp); before
        // any engine code reads its switches (they are read at first use), and the user's own settings are kept
        cudaDeviceProp ad{};
        if (cudaGetDeviceProperties(&ad, 0) == cudaSuccess) {
            // an integrated Radeon (UMA / APU: the device property, not the name): the matrix-core prompt set is on by
            // default.  Only when EVERY visible GPU is integrated and of the same architecture, so a box that also holds a
            // discrete card (whichever the engine ends up using) keeps the old defaults.
            bool all_igpu = ad.integrated != 0;
            int ndev = 0;
            if (all_igpu && cudaGetDeviceCount(&ndev) == cudaSuccess) {
                for (int d = 1; d < ndev && d < 64; ++d) {
                    cudaDeviceProp od{};
                    if (cudaGetDeviceProperties(&od, d) != cudaSuccess || od.integrated == 0 ||
                        std::strncmp(od.gcnArchName, ad.gcnArchName, 7) != 0) { all_igpu = false; (void) cudaGetLastError(); break; }
                }
            }
            strata::core::apply_arch_defaults(ad.gcnArchName, all_igpu);
#if defined(_WIN32)
            const bool win_host = true;
#else
            const bool win_host = false;
#endif
            // UMA mode (021-minipc): the Radeon 780M's firmware (gc 11.0.1 MES) resets the GPU when the kernel evicts the queues of a
            // process that has a dispatch in a barrier loop, and a registered host arena is what gets it evicted (notes-021-minipc.md).
            // So its expert arena stays unregistered.  STRATA_UMA_REGISTER=1 (or STRATA_ARENA_PIN_GIB=0) registers it as before.
            if (strata::core::igpu_uma_arena_default(ad.gcnArchName, all_igpu, win_host))
                strata::core::arena_uma_no_register_flag().store(true);
            // gfx1151 on Windows (#1630 capped a bare `--prefill auto` at 6144 for a 7900 XTX's commit charge): the Strix Halo has its
            // 8192 chunk back.  IQ3_S, 8K prompt: 691 -> 787 tok/s (one chunk streams 1,785 experts instead of 2,782), 64K 713 -> 748;
            // 4K is one chunk already.  STRATA_GFX1151_CHUNK=0 (or STRATA_GFX1151_DEFAULTS=0) keeps 6144.
            if (win_host && all_igpu && o.prefill_auto_bare && o.prefill_auto_cap > 0 && std::strncmp(ad.gcnArchName, "gfx1151", 7) == 0) {
                const char* c1 = std::getenv("STRATA_GFX1151_CHUNK");
                const char* d1 = std::getenv("STRATA_GFX1151_DEFAULTS");
                if (!(c1 != nullptr && c1[0] == '0') && !(d1 != nullptr && d1[0] == '0')) {
                    o.prefill_auto_cap = 0;
                    // 0.2.8: 16,384 rather than 8,192.  The prompt reads every expert once per chunk, so a bigger chunk
                    // reads them fewer times: measured on strix (engine 0.1.52, own prompt buffers, IQ3_S) 32K 1,260.0 ->
                    // 1,307.9 tok/s and 64K 1,191.0 -> 1,238.2 (+3.8% / +4.0%); at 4K and 8K the chunk is the prompt, so
                    // nothing changes there.  The buffers' reserve grows to ~10.9 GiB and the cache keeps all 24,576 slots
                    // (the box reports 84.7 GiB free at a 64K context).  STRATA_GFX1151_CHUNK=0 keeps 6,144.
                    o.prefill_chunk = std::max<int64_t>(o.prefill_chunk, 16384);
                }
            }
            // 0.2.8: the Strix Halo prompt path keeps its own buffers instead of lending cache slots.  A lent slot's expert
            // is re-read from the host arena during every chunk, and on this box that copy runs at ~0.5 GB/s (the arena is
            // unregistered in UMA mode): measured on strix, engine 0.1.52, IQ3_S, 2026-10-27 - 823 slots = 2.75 s of a
            // 3.3 s 4K prompt, 1,408 slots = 5.5 s of a 6.1 s 8K prompt.  Without lending every expert stays in the cache
            // (24,576 slots) and the prompt reads VRAM only; the buffers' VRAM is reserved up front instead (the cache
            // still fits: 46.84 GiB of experts against a 63.6 GB carve-out).  STRATA_PREFILL_BORROW=1 keeps 0.2.7's
            // lending, STRATA_GFX1151_DEFAULTS=0 turns the whole gfx1151 block off.
            if (win_host && all_igpu && std::strncmp(ad.gcnArchName, "gfx1151", 7) == 0) {
                const char* pb = std::getenv("STRATA_PREFILL_BORROW");
                const char* pd = std::getenv("STRATA_GFX1151_DEFAULTS");
                if (!(pb != nullptr && pb[0] == '1') && !(pd != nullptr && pd[0] == '0')) {
                    o.no_prefill_borrow = true;
                    o.prefill_own_keep_chunk = true;
                }
            }
        } else (void) cudaGetLastError();
    }
#endif
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
    {   // CUDA sm_86 / sm_60: the switches that are exact AND faster there are on by default (strata/core/nv_defaults.hpp), when every
        // visible card has that one compute capability; before any engine code reads its switches, the user's own settings are kept
        int nd = 0;
        if (cudaGetDeviceCount(&nd) == cudaSuccess && nd > 0) {
            int m0 = 0, n0 = 0;
            bool same = cudaDeviceGetAttribute(&m0, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess &&
                        cudaDeviceGetAttribute(&n0, cudaDevAttrComputeCapabilityMinor, 0) == cudaSuccess;
            for (int d = 1; same && d < nd; ++d) {
                int m = 0, n = 0;
                same = cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, d) == cudaSuccess &&
                       cudaDeviceGetAttribute(&n, cudaDevAttrComputeCapabilityMinor, d) == cudaSuccess && m == m0 && n == n0;
            }
            if (same) { strata::core::apply_nv_defaults(m0, n0); strata::core::apply_spec_adapt_default(m0, n0); }
        }
        (void) cudaGetLastError();
    }
#endif
#if defined(STRATA_USE_HIP) && !defined(_WIN32)
    if (o.prefill_auto_bare && o.prefill_auto_cap == 0) {
        // gfx1200 / gfx1201 (RDNA4, Linux): a bare `--prefill auto` may reach 16384 instead of 8192 (R9700, IQ3_S / Q2_0, 16K / 32K /
        // 64K prompts: +0.5 .. +5.8%, residuals identical to 8192).  The fit scan still picks a smaller chunk when the expert cache cannot lend
        // the buffers.  An explicit --prefill N, auto:N or STRATA_PREFILL_AUTO_MAX is never touched (prefill_auto_bare is false then).
        // STRATA_GFX12_DEFAULTS=0 or STRATA_GFX12_CHUNK=0 keeps 8192.
        const char* c0 = std::getenv("STRATA_GFX12_CHUNK");
        const char* d0 = std::getenv("STRATA_GFX12_DEFAULTS");
        cudaDeviceProp cp{};
        if (!(c0 != nullptr && c0[0] == '0') && !(d0 != nullptr && d0[0] == '0') &&
            cudaGetDeviceProperties(&cp, 0) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(cp.gcnArchName)) {
            o.prefill_auto_max = 16384;
            o.prefill_chunk = 16384;
            o.prefill_gfx12_16k = true;
            std::fprintf(stderr, "strata generate: gfx12: --prefill auto may take chunks up to 16384 (STRATA_GFX12_CHUNK=0 keeps 8192)\n");
        }
        (void) cudaGetLastError();
    }
#endif
#if defined(STRATA_HIP_GFX906)
    {   // #1728: the gfx906 build's kernels assume 64-lane waves; on a wave32 card they would fault or give wrong
        // numbers, so say so before the first launch (the ready-made wave32 engine is the one for those cards)
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) == cudaSuccess) {
            for (int d = 0; d < n_dev; ++d) {
                cudaDeviceProp wp{};
                if (cudaGetDeviceProperties(&wp, d) == cudaSuccess && wp.warpSize != 64) {
                    std::fprintf(stderr, "strata generate: this engine was built for gfx906 (wave64, STRATA_HIP_GFX906), but GPU %d "
                                         "(%s) has %d-lane waves: use the ready-made engine for that card, or hide it "
                                         "(HIP_VISIBLE_DEVICES)\n", d, wp.name, wp.warpSize);
                    return 2;
                }
            }
        } else (void) cudaGetLastError();
    }
#endif
    (void) o;
    return -1;
}

// #1341 #964: CUDA_LAUNCH_BLOCKING=1 in the environment.  A verify window's graph waits on the GPU for flags the host
// raises once cudaGraphLaunch has returned, and a blocking launch returns only when the graph has finished, so the
// first window never ends (the batched prompt path has no such wait and runs).  HIP's own switches are not checked.
bool is_launch_blocking() {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    return false;
#else
    const char* v = std::getenv("CUDA_LAUNCH_BLOCKING");
    return v != nullptr && std::atoi(v) != 0;
#endif
}

void prepare_process(int argc, char** argv) {
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    // AMD, a file-backed arena (STRATA_ARENA_MMAP): ROCclr copies a pageable source of 1 MiB or more by locking its
    // pages in place (a GPU userptr), and keeps them - so every expert the VRAM fill copied from the mapped
    // experts.bin stayed resident, the release of the VRAM-held ones was undone by the driver at once, and a
    // 32 GB PC ran with ~0.3 GB free: the kernel swapped, and a turn's checkpoint took 25-80 s.  Above this size
    // ROCclr stages through its own pinned buffers instead; decode is unchanged (49.06 vs 49.16 ms/window).
    if (std::getenv("STRATA_ARENA_MMAP") && std::getenv("STRATA_ARENA_MMAP")[0] == '1')
        strata::platform::set_env("GPU_PINNED_MIN_XFER_SIZE", "1000000", false);   // MiB; an explicit setting wins
#endif
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    strata::platform::set_env("CUDA_MODULE_LOADING", "EAGER", false);
    if (is_launch_blocking())
        std::fprintf(stderr, "warning: CUDA_LAUNCH_BLOCKING=1: the verify windows cannot run with blocking launches, so "
                             "the first one after a prompt hangs until the watchdog ends the engine (#1341). Unset it; "
                             "STRATA_PF_STEP_SYNC=1 narrows down a failing prompt step without it\n");
    // --gpu LIST pins the visible GPUs (nvidia-smi/PCI order) from the command line instead of the caller's
    // environment: CUDA_VISIBLE_DEVICES is read at the first CUDA call, so this has to happen here, before
    // anything else.  The value itself is consumed again by the option loop below.
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--gpu") continue;
        // The caller's own environment is overridden, so say so (a stale CUDA_VISIBLE_DEVICES from a launcher
        // would otherwise silently lose to the flag).
        const char* envname = "CUDA_VISIBLE_DEVICES";
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
        envname = "HIP_VISIBLE_DEVICES";   // the AMD runtime ignores CUDA_VISIBLE_DEVICES (strata_server/server.py sets this one)
#endif
        const char* prev = std::getenv(envname);
        if (prev && prev[0] && std::string(prev) != argv[i + 1])
            std::fprintf(stderr, "warning: --gpu %s overrides %s=%s from the environment\n", argv[i + 1], envname, prev);
        strata::platform::set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        strata::platform::set_env(envname, argv[i + 1]);
        break;
    }
}

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
// The R9700 / RX 9000 "bimodal prompt speed" (0.2.0 known limit).  The runtime opens the process's hardware queues
// lazily, as the first stream of each kind is used, and the kernel's scheduler (MES) puts a queue on one of the
// card's compute pipes when the queue first gets work.  The loader threads reach their streams in a different order
// every start, so in about half of the starts the prompt path's two busy queues (the compute stream and the expert
// staging stream) landed on the SAME pipe: every one of the ~30,000 hand-offs of a prompt then waits for the pipe to
// switch queues, and a 4K prompt reads 13% slower (p620, 1,980 vs 2,290 tok/s).  Opening the queues here, one after
// the other on the main thread and before any other thread exists, makes the placement the same in every start (and
// it is the fast one).  Streams are destroyed again; the runtime keeps the queues.
//   Two queues are enough (R9700, "nn": 0 slow of 32 fresh starts, unprimed 10 of 32); five (nnnnh) fixed the starts too but cost
//   decode 2 to 3.6% and two expert slots, four normal ones did not even fix the starts.
//   STRATA_HIP_QUEUE_PRIME unset: gfx1200 / gfx1201 on Linux only, "nn" (not when GPU_MAX_HW_QUEUES is 1 or 2, #3480); "0": off; any other string is a spec
//   ('n' = a stream at the default priority, 'h' = at the highest) for experiments on other cards.
void prime_gpu_queues() {
    const char* env = std::getenv("STRATA_HIP_QUEUE_PRIME");
    if (env != nullptr && std::strcmp(env, "0") == 0) return;
    const char* pv = std::getenv("STRATA_PRIMARY_DEVICE");
    const int dev = pv != nullptr && std::atoi(pv) > 0 ? std::atoi(pv) : 0;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || dev >= n_dev) { (void) cudaGetLastError(); return; }
    const char* spec = env != nullptr && env[0] != 0 ? env : nullptr;
    if (spec == nullptr) {
#if defined(_WIN32)
        return;
#else
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || !strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName)) {
            (void) cudaGetLastError();
            return;
        }
        // #3480: with the runtime capped to two hardware queues (GPU_MAX_HW_QUEUES=1 or 2, the user's own setting) the two
        // primed queues are all of them: the high-priority draft stream and the side stream share them and a 4K prompt
        // reads 39% slower (RX 9060 XT, 681-753 against 1188-1238 tok/s).  Not primed then; a spec in
        // STRATA_HIP_QUEUE_PRIME still is.
        const char* cap = std::getenv("GPU_MAX_HW_QUEUES");
        if (cap != nullptr && std::atoi(cap) >= 1 && std::atoi(cap) <= 2) {
            std::fprintf(stderr, "strata generate: GPU queues not primed: GPU_MAX_HW_QUEUES=%s leaves none for the draft stream "
                                 "(STRATA_HIP_QUEUE_PRIME=nn primes them anyway)\n", cap);
            return;
        }
        spec = "nn";
#endif
    }
    if (dev != 0 && cudaSetDevice(dev) != cudaSuccess) { (void) cudaGetLastError(); return; }
    void* buf = nullptr;
    if (strata::core::allocate_device(&buf, 256,
        {"other", "Device allocations", "options buffers"}) != cudaSuccess) { (void) cudaGetLastError(); return; }
    int lo = 0, hi = 0;
    const bool have_prio = hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess;
    std::vector<cudaStream_t> ss;
    for (const char* c = spec; *c != 0; ++c) {
        cudaStream_t s = nullptr;
        const bool ok = (*c == 'h' && have_prio) ? hipStreamCreateWithPriority(&s, hipStreamNonBlocking, hi) == hipSuccess
                                                 : cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
        if (!ok) continue;
        cudaMemsetAsync(buf, 0, 256, s);
        cudaStreamSynchronize(s);
        ss.push_back(s);
    }
    for (cudaStream_t s : ss) cudaStreamDestroy(s);
    strata::core::release_device(buf);
    (void) cudaGetLastError();
    std::fprintf(stderr, "strata generate: opened %zu GPU queues in a fixed order (STRATA_HIP_QUEUE_PRIME=0 turns it off)\n", ss.size());
}
#else
void prime_gpu_queues() {}
#endif

}  // namespace strata::qwen38
