# The memory manager

A plan for a new Qwen3.8 engine whose memory and parallelism can be seen and changed while it runs. It is built on the
shared code of the refactor ([REFACTOR_PLAN.md](REFACTOR_PLAN.md)) and adds to it, so that Qwen3.6 and DeepSeek can
move onto it later. Status: M0 to M7 done, M8 in part (strata-q38 on a layer split), M9 in a first version (the cut moves while it
decodes; 2026-11-01), DeepSeek's D1 to D7 done (its steps are D1 to D10; 2026-11-02), branch
`memory-manager`.

## What the user gets

A panel in the server's web page shows, for each memory, what it holds now, and lets the user move the boundaries
while the model runs:

```
RTX 3080  | Expert cache                 | Conversations | Drafter | Free     |
20 GiB    | 15.51 GiB                    | 2.19 GiB      | 1.0 GiB | 1.30 GiB |
RTX 3060  | Expert cache         | Conversations | Free |
12 GiB    | 11.08 GiB            | 0.82 GiB      | 0.10 |
System RAM | Expert cache                         | Paused conversations: 2 | Free     |
48 GB      | 28.88 GiB                            | 2.00 GiB (55% used)     | 17 GB    |
SSD        | Experts not loaded | Engram    | Paused conversations: 0 |
           | 0 GiB              | 26.55 GiB | 4 GiB (0% used)         |
Conversations at once: 2 of 3 | Rows per step: 8 | Lookahead: 4
```

- Sizes change with the model running. The one limit: the biggest active conversation must still fit.
- Each change is either for this session only or saved as the default for the next start.
- On a PC whose graphics memory is system RAM (Strix Halo), the bars follow the BIOS's division: the GPU's bar is the
  share the BIOS gives the GPU, and what the GPU borrows beyond it is a segment of the RAM's bar.

## The settings

| Setting | What it limits | Changes live |
|---|---|---|
| Memory used, per GPU and in RAM | the total the engine takes from each memory; it gives memory back or takes more | yes |
| Combined GPU token budget | the total logical tokens of all GPU-resident conversations; paused RAM and SSD conversations do not count | yes |
| Per-conversation token limit | the most tokens one conversation may use; cannot exceed the combined GPU budget or the loaded model capacity | yes |
| Conversation tokens, minimum | the space kept free of experts in each pool for conversation state | yes |
| SSD space for paused conversations | how much of the disk paused conversations may take | yes |
| Drafter on/off (per GPU) | its weights: about 1 GiB for Qwen3.8 | yes (unloading frees the memory) |
| Conversations at once | how many conversations decode in the same step | yes |
| Conversations held in VRAM | how many active or cached conversations keep their GPU state; at least Conversations at once | yes |
| Idle conversation eviction age | how old idle GPU state must be to give way under memory pressure; 66 seconds by default | yes |
| Decoding while a prompt is read | after each turn of a prompt read, the decoding conversations step for this percentage of the turn's time; 50 by default, 0 makes them wait | yes |
| Save conversations on shutdown | when the engine ends, its conversations in VRAM and RAM go to the SSD, the most recent first, within the SSD allowance, for the next start to adopt; off by default | yes |
| Prompt reading turn | the least time one turn of a prompt read lasts, in whole chunks, before the decoding conversations get their share; 0 (one chunk) by default | yes |
| Rows per step | how many tokens one pass through the model computes (at most 8 today) | yes |
| Lookahead | the most guesses one conversation may add in a step; 0 turns drafting off | yes |

Conversations held in VRAM defaults to one more than Conversations at once. Qwen adds a further cached place on
cards with at least 10 GiB each: with one decoder the default is two on 8 GiB and three on 10 GiB or more. DeepSeek
keeps the smaller default to leave more VRAM for experts. Explicit settings remain in effect. The panel reports
fixed GPU memory per conversation before its KV cache; checkpoint snapshots are allocated separately in RAM.

Idle conversation eviction age defaults to 66 seconds (`--conversation-idle-seconds`). Completed prompt chunks and
decode steps update the activity time. Old idle state can give way when experts or another conversation need pages;
age alone never removes it. In-progress prompt reads and decoding requests are protected. The hard conversation
count may reclaim younger idle state, and priorities can pause requests safely between steps.

Qwen keeps only live recurrent state in its GPU pool. Rewind snapshots grow in RAM as captured and are freed on
checkpoint eviction. DeepSeek already keeps its ring and compressor checkpoints in RAM. Shared host-state readers,
RAM accounting and disk pause/restore carry both families' checkpoints. Active checkpoint heap RAM is reported
separately from the RAM page pool; paused host payloads count against its pause allowance. Qwen's layout identity is
4, so files from the former GPU-checkpoint layout cannot be resumed by this build. No migration is provided.

Rows per step must be at least the conversations at once. An edit that would put the per-conversation limit above
the combined GPU token budget is rejected with an explanation; neither value is adjusted automatically. Raising
the loaded model capacity still requires a model reload. The existing RAM pause-cache allowance is derived from
the maximum setting independently; RAM and SSD caches do not consume the GPU token budget.

There is no expert-cache setting: experts get every page the conversations are not using, the most-used on the GPUs,
then in RAM, and the rest are read from the SSD as needed. So a small conversation leaves almost all the memory to
experts, and a large one takes pages from them up to the maximum. There is no separate low-RAM mode either: when the
memory used is too small for all the experts, some stay on the SSD, read without the operating system's file cache
(`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux), so the cache does not take the RAM back. That replaces
today's `--resident-budget-gib`, `--mmap-experts` and the file tier's options.

### How rows are shared (the owner's rule)

Each step gives every active conversation one row for its next token. The rows left over go to guesses, split as
evenly as possible, and no conversation gets more than the lookahead:

| Conversations at once (3), rows (8), active now | Rows for guesses | Split |
|---|---:|---|
| 3 | 5 | 2, 2, 1 |
| 2 | 6 | 3, 3 |
| 1 | 7 | 7, or the lookahead if lower |

- A share is the most a conversation may use: its drafter may stop earlier when it is unsure (the confidence floor of
  today's `--spec-min-p`). Rows left unused stay empty.
- The extra row of an uneven split goes to the highest-priority conversation, and rotates between equals.
- Reading a prompt is not counted in these rows: it reads chunks of hundreds of tokens in passes of its own.
- Later, a share may be spent as a tree of guesses instead of a chain (see [Later](#later)).

Whether guesses touch fewer experts than other conversations' tokens (and so cost less when experts are in RAM) is
measured once this works, by comparing settings such as 1Ã—8 rows against 3Ã—8 and 3Ã—3. No setting is chosen from that
assumption beforehand.

## How it works

### One page pool per memory

Each GPU's memory and the RAM the engine uses are each one pool of equal-size pages. A page holds an expert, or a
block of a conversation's tokens, or is free.

- The page size is the greatest common divisor of the model's expert sizes, so every expert is a whole number of
  pages and no page is part empty (step M2f). The experts of one layer are all one size; sizes differ by layer. Qwen3.8
  IQ3_S has 7 sizes from 1,510,400 to 2,662,400 bytes: pages of 25,600 bytes, 59 to 104 per expert. DeepSeek's 2.00
  bpw set has 8 sizes from 7,741,440 to 10,229,760 bytes: pages of 92,160 bytes, 84 to 111 per expert. A page is a
  multiple of 256 bytes and at least 16 KiB; a model whose divisor is not gets the size from 16 to 512 KiB that wastes
  least.
- So a pool has millions of pages (about 2 million for all of Qwen3.8 IQ3_S's experts). The pool keeps runs of
  consecutive pages (free, one expert, one block of a conversation, or a conversation's hole), not pages, and finds room
  through ordered indexes.
- Each expert size has a home point: the middle of the room the expert profile's hottest experts of that size need,
  the sizes in order above the conversations' reserve. An expert takes the free run nearest its home, so the experts
  of one size sit together and an evicted one leaves a hole that fits the next one of its size. When there is no free
  run, evicting the least-used expert of the same size always makes room. Nothing moves experts to defragment: the
  unit tests trade pages at random (20,000 steps in a small pool, 300 steps of three conversations of up to 100,000
  tokens in a pool of 1.9 million pages) and the experts afterwards fill the pool as well as they fill an empty one.
- The conversations' end is the bottom of the pool: the `--tokens-min` reserve, which experts leave free, and the
  conversations' fixed regions (a recurrent state of 112 MiB takes one run; when no free run is long enough, the
  lowest stretch of experts is evicted for it). A growing conversation past its free runs evicts the least-used expert
  of any size and fills that hole with its blocks before it takes another; the hole goes back whole when its last
  block leaves.
- A conversation's growing region takes blocks of several consecutive pages when one page would waste much: Qwen3.8's
  int8 K/V unit (an indexer block of 4 tokens, 2 KV heads) is 4,736 bytes, so a page of 25,600 bytes would hold 5 of
  them and waste 7.5%; a block of 5 pages holds 27 and wastes 0.1%. (f16: 8,704 bytes, 32% of a page wasted, 0.8% in
  blocks of 12 pages; q4: 2,816 bytes, 1%; k8v4: 3,776 bytes, 11.5%, 0.4% in blocks of 4. The drafter's K/V has the
  main K/V's format, int8 under k8v4.)
- `GET_MEMORY_USE` gives a pool's pages, how many are allocated, and how many are free in runs shorter than the
  smallest expert (fragmented: free memory no expert can use); the panel shows both as shares of the pool.
- A pool takes its memory in chunks of about 256 MiB (on strata-q38's GPU the first chunk is the expert cache's
  arena). Growing the memory used adds chunks; shrinking it first empties the top chunks (their experts are evicted,
  conversation pages are copied below or paused) and then gives them back. On a GPU with virtual memory (CUDA, and
  HIP from ROCm 6: measured on the Strix Halo in step M10) the chunks are mapped back to back in one reserved address
  range, so a run may cross their bounds; without it each chunk is an allocation of its own and a run stays inside
  one. Neither needs the old engine's `VRAM` command.
- A prompt borrows its buffers from the top of the pool for the length of its read (step M10); the engines keep no
  prompt buffers of their own.
- Conversations and experts trade pages without a setting, up to the conversation maximum. Experts take pages back
  only after a conversation ends or is paused, and only after a short wait, so that the two do not thrash.
- On the SSD, an expert is read with its sector-aligned surroundings into a staging buffer, because reads without the
  file cache must be aligned. DeepSeek's Engram reader (its own threads reading the SSD) is the model for this.
- A conversation's memory is described with the existing state descriptor
  (`include/strata/common/state/state_layout.hpp`): fixed regions (the GDN state, the drafter's state) take whole
  pages once; per-token regions (K/V) take a block of consecutive pages per group of tokens as the conversation
  grows. The pool holds the blocks; a conversation keeps a list of its blocks.
- Weights that are always needed (the dense layers, the head) are outside the pools.
- On a PC whose graphics memory is system RAM (Strix Halo), the GPU's pool is the share the BIOS gives the GPU, not
  what the runtime reports as the GPU's size. The runtime reports the share plus what the GPU may borrow from the
  operating system's RAM (on the test PC, 107.6 GiB: a 63.8 GiB share and 47.6 GiB that may be borrowed, while Windows
  sees 63.6 GiB of RAM). Once the share is full, GPU allocations spill into that RAM, so any spill counts against the
  RAM pool and its setting. Otherwise the manager would silently take Windows' memory; DeepSeek reaching Windows'
  commit limit on this PC is likely this.

### Tiers and pausing

- A conversation is **active** (decoding, its pages on a GPU), **paused in RAM** or **paused on the SSD**. A paused
  conversation is copied page by page; coming back, it is copied up again before its next step.
- Every conversation has a priority (an API field, else the order it arrived). When memory is short, the
  lowest-priority paused conversation moves down a tier first, then the lowest-priority active one is paused. A
  running conversation may be paused for a higher-priority one.
- The SSD tier replaces today's separate mechanisms: Qwen3.8's RAM conversation cache, the session files, and
  DeepSeek's RAM/disk slots.

### Guesses in the page pool

- A guess writes its K/V into its conversation's last page; a rejected guess's cells are simply not counted. No
  separate session is needed for drafting.
- The drafter's own state per conversation is one more region of the conversation.

### The step

The engine's loop is a model-neutral step: pick the active conversations (by priority), share the rows by the rule
above, ask the drafter for each conversation's guesses, run the rows through the model with a parent table (which row
each row follows), accept, and update the pages. The model provides only:

- running a set of rows with their parent table (a chain today, a tree later);
- its state regions (the descriptor);
- its drafter's "propose up to N guesses".

For GDN models the verify kernel keeps one running state per row for rollback, as today; a tree needs each row to
start from its parent's state instead of the previous row's.

### Experts leaving a memory

RAM normally holds the experts the GPUs do not (the complement), so together they hold more different experts. An
expert evicted from a GPU is therefore copied back to RAM, in the background between steps, instead of being dropped
and read from the SSD again later.

- While RAM has room, it also keeps copies of experts that are on a GPU; evicting those costs nothing. RAM holds
  only the complement when it runs short.
- An autodetected setting drops the expert instead unless copying it back from the card is clearly faster (by
  default at least twice as fast) than reading it from the SSD again: when the two are close, dropping wins, because
  an evicted expert may not be needed again for a while. Both speeds are measured at start (a PCIe 3.0 x4 card, about
  3.2 GB/s, is close to a fast NVMe drive, and a Thunderbolt 4 enclosure, about 2.5 GB/s, is slower than one). The
  user may override it.

### Several GPUs

Today Qwen3.8 has a layer split, Qwen3.6 runs on one GPU, and DeepSeek has its own split with optional tensor
parallelism ([MULTI_GPU.md](MULTI_GPU.md)). The new step has one shared way:

- **A layer split for every model.** Each card owns a range of layers; its pool holds their conversation pages and
  state regions (the descriptor's one layout per device), and their experts by default. The hand-off between cards
  goes through pinned RAM, so cards without peer-to-peer access (GeForce on Windows) work. The planner that picks the
  cut (most of the expert profile on the GPUs, the faster card last) moves from `generate.cpp` to shared code; models
  give per-layer sizes (dense weights, experts, K/V bytes per token).
- **Experts may live on any card.** A card with free pages may hold another card's experts and compute them there,
  all of one layer's remote experts in one hop. This replaces today's helper caches and peer tier with one rule.
- **Links are measured, not assumed.** At start, each card's round-trip latency and its bandwidth each way are
  measured (the engine already probes each card's PCIe share). Remote experts are used on a card only when one hop
  costs less than the CPU computing those experts from RAM, and refills onto a slow card are paced by its bandwidth (a
  2 MB expert takes about 0.6 ms over PCIe 3.0 x4).
- **Memory per card.** The conversation token limits stay global; each card's share follows from its layers. The
  biggest active conversation must fit on every card.
- **Moving the cut while running.** When a card's memory setting changes, the planner may move the cut. The experts of
  moved layers stay where they are (now remote) until use moves them. A layer's dense weights (about 56 MB on
  Qwen3.8) and every conversation's pages for it are copied in the background, the tokens added meanwhile copied
  after, and ownership switches between two steps, one layer at a time.
- **Keeping cards busy.** With several conversations, the step splits them into groups, so that one card runs group
  A while the other runs group B (`--batch-groups auto`). The rows settings apply per group.
  Implemented for Qwen3.8 (3 November 2026): `spec::run_stage_groups` prepares all groups, then services each
  stage without waiting for another stage's GPU. `PagedModel` exposes prepare/launch/poll/finish/keep operations;
  Qwen's verifier preserves each group's deferred commit inputs, and the adapter preserves its picture coordinates,
  sampling histories and final residuals for MTP. Each group gets the row-sharing rule and the existing drafting
  options. Snapshot copies are nonblocking; sampling waits until every stage drains. A wave drains before
  acceptance and before control returns to the serve loop: admissions, RAM/SSD
  pauses, restores, pool changes and cut moves cannot change pages used by an in-flight group. The scheduler also
  drains on launch or poll failure. This first version overlaps groups within a wave; it does not keep the next
  token's wave in flight across control commands.
  `--batch-groups auto` selects the largest divisor of live concurrency no greater than the stage count; an explicit
  count must divide live concurrency, otherwise it uses one group. `1` keeps the synchronous path; omitting the
  option keeps that existing default, so pipeline groups remain opt-in. `--batch` only
  seeds concurrency; raising it live can grow the groups. The Python launcher forwards the existing option.
  DeepSeek now exposes stage launch/poll through that same scheduler. Its pass separates immutable group inputs,
  row-local attention binding, stage execution and deferred sampling. Group-owned compressed-row copy buffers and
  DSpark taps survive until the wave drains; stage ownership follows live cuts. Its one-group path remains separate
  and unchanged in execution. The NVIDIA session's `memory-groups-ds41-*` checks compare groups 1 and auto with
  1, 2, 3 and 4 conversations and DSpark on and off, in both card orders. Real two-card results remain unmeasured here.
  DeepSeek's local same-GPU checks passed groups 1/2/auto at two to four conversations, a live cut move with
  pauses/restores and pool changes, and DSpark concurrency 1 -> 2 -> 1. The setup and timing limits are in
  [the local report](../bench/results/ds41-groups-20261104/README.md). Real two-card speed and hops remain open.
  For Qwen3.8, validation remains incomplete: [DETAILS.md](DETAILS.md) records the same-GPU measurements
  (2.216 vs 2.220
  aggregate tokens/s, groups 1 vs auto), passed checks and answer-identity failures. In particular, suffix-only
  pictures during live concurrency changes still need investigation. This is implemented and opt-in, not an
  unconditional identity or speed claim. Real two-card gains remain unmeasured here.
- **Left out:** tensor parallelism (an exchange between cards at every layer, slow over PCIe without peer-to-peer
  access; DeepSeek keeps its `--tp` until there are cards with a fast link to measure), and one model on cards of two
  makes.

### The panel's bars

A memory smaller than 32 GiB is drawn as a bar of its size divided by 32 GiB of the full width (a 16 GiB card fills half);
larger memories fill the width. So RAM is seen to be bigger than a typical card, and cards do not shrink to slivers
beside a lot of RAM.

### Changing sizes by dragging

The boundaries between a bar's segments can be dragged (step M6).

- While dragging, the bar shows the new boundary and its numbers, and nothing is sent to the engine. A size the
  engine would refuse (for example below what the biggest active conversation needs) shows in red with the reason.
- The change is sent (`CHANGE_SETTINGS`) only when the mouse button is released. Escape cancels; dropping it where it
  started changes nothing; releasing on a refused size changes nothing.
- After a change, a small prompt offers "This session" (the default) or "Save as default", and an Undo stays for a
  few seconds.
- Easy to find: each draggable boundary has a visible grip and a resize cursor, and a tooltip says what dragging it
  changes. A one-line hint ("Drag the dividers to change sizes") shows under the panel until the first change.
- Other ways to do the same: clicking a segment opens its size as a number field; a divider can take keyboard focus
  and move with the arrow keys (Enter applies); touch works like the mouse.

## The protocol

New engine commands (between and during requests, like `LIST_CONVERSATIONS`):

- `GET_MEMORY_USE` reports every pool: per memory, its size and its pages by use (experts, active conversations, paused
  conversations, drafter, free), and the settings above. The old engines answer it too, read-only, from what they know
  (sizes from their startup, `LIST_CONVERSATIONS` bytes), so the panel shows them.
- `CHANGE_SETTINGS <name>=<value> [...]` changes settings live: `memory_mib.<memory>`, `tokens_min`, `tokens_max`,
  `ssd_pause_mib`, `drafter on|off`, `concurrent`, `rows`, `lookahead`. The engine answers with the new
  `GET_MEMORY_USE` report or `ERR <why>` (for example "the biggest active conversation needs 3.1 GiB").

The server forwards both through one endpoint (`GET/POST /v1/memory`); "save as default" writes the run config
(`strata-<model>.json`), the one file the web page, the app and the Terminal all read.

## Where the code goes

| Part | Folder |
|---|---|
| Page pool, page lists, tiers, priorities | `src/common/state/` |
| Step: picking conversations, sharing rows, accept | `src/common/spec/` and `src/common/serve/` |
| `GET_MEMORY_USE`/`CHANGE_SETTINGS` parsing and answers | `src/common/serve/` |
| Expert pages (evict, load, count use) | `src/common/experts/` |
| Copies between pool, RAM and SSD | `src/base/gpu/` (the runtime interface), `src/base/platform/` (files) |
| Qwen3.8 rows with parent table, its regions, its drafter | `src/models/qwen38/` |
| DeepSeek's rows with parent table, its regions, its drafter | `src/models/deepseekv41/` |
| `/v1/memory`, saving defaults | `strata_server/`, `strata_common/` (run config) |
| The panel | `strata_server/web/` |

## Steps

| Step | Work | Checked by |
|---|---|---|
| M0 | `GET_MEMORY_USE` in the three existing engines (read-only), `/v1/memory`, the panel without controls | the panel matches the engines' own numbers on Qwen3.8, Qwen3.6, DeepSeek. Done: `dffd3a25` |
| M0b | The desktop app shows the same panel: the panel is one framework-free module that the web page uses directly and the app (`apps/host`, `NativeMonitor`) wraps | the app and the web page draw the same bars from `/v1/memory`. Done: `strata_server/web/memory-panel.js` (and `memory-panel.css`) draws the bars, the conversations the engine holds (strata-q38's list: state, tokens, memory) and the settings; the web page imports it, and the app's Monitor screen wraps it (`MemoryPanel.tsx`; `NativeMonitor.tsx` is shown by no screen), reading the chosen model's `/v1/memory` through the host (`/api/native/memory`) and bundling the same two files. `apps/host/tests/memory-panel.test.mjs` draws strata-q38's own answer (three conversations, the Strix Halo) |
| M1 | The page pool and page lists: chunks, growing and shrinking, experts and conversations trading pages, with unit tests (CPU). Done: `src/common/state/page_pool.cpp` | unit tests: allocation, growth, emptying a chunk, a full pool, no thrashing (`page_pool_test`) |
| M2a | The pool holds Qwen3.8's state: fixed regions in consecutive pages (its GDN state is 112 MiB), truncation, `describe_conversation`. Done: `45eb5a07` | unit tests |
| M2b | Qwen3.8's attention kernels read K/V and the indexer keys from pool pages: a page stride per array (today one row index serves four packed arrays), in the decode, prompt, append and gather kernels. Done: `c4d1578a` (NVIDIA-only variants compiled, not run: run `kv_units_parity` in the NVIDIA session) | a parity test (pool layout bit-identical to today's layout); the old engine's tokens and speed unchanged |
| M2c | The Qwen3.8 model loader and a one-conversation step move out of `generate.cpp`'s `main()` (about 6,200 lines of loading and the decode lambdas) into `src/models/qwen38`; the old engine calls them | the old engine's tokens and speed unchanged Loader done: `7a698d73` (`strata::qwen38::Model::load()`) |
| M2c2 | The one-conversation step: the hit tables and verify hits into `Model`, and a `Conversation` (prompt read, one decode step) wrapping `Prefill` and `Verifier`, called by the old CLI path | the old engine's tokens and speed unchanged. Done (the serve path still has its own step) |
| M2d | The new engine (`strata-q38`, a thin `main()` over `src/models/qwen38` and the shared serve loop): one conversation in the pool, no drafting | token-identical at temperature 0 against the old engine; speed; the pool's pages grow with the conversation. Done: identical tokens with one-row windows; through the server on the Strix Halo, prompts 884-1,108 tok/s against 798-1,042 and decode 26-28 against 25-27; conversation memory 145 MB at 66 tokens and 245 MB at 8.3K, against 3,092 MB reserved by the old engine |
| M2e | Experts in the pool's pages: the expert cache's slots go through the pool, evicted experts go to the RAM tier or are dropped | the same tokens; conversations take expert pages and give them back. Done (`ExpertPages`, `src/common/experts/expert_pages.cpp`): one pool on the GPU, pages of 332,800 bytes for Qwen3.8 IQ3_S (the size per model that wastes least, 4.4% here); on a GPU whose memory is the system RAM the pool stays within the BIOS share (Windows: DXGI's dedicated memory); an evicted expert is dropped (the CPU reads it from the RAM copy or the files) and comes back between requests; `--tokens-min` (default 8192) replaces `--conversation-mib`. Copying evicted experts back to RAM is left for M5, which gives the RAM copy room of its own |
| M2f | Exact-fit pages: the page size is the greatest common divisor of the expert sizes; a pool of millions of pages; experts placed by size around home points; conversations fill expert holes; the pool's allocated and fragmented shares in `GET_MEMORY_USE` and the panel | the same tokens; no fragmentation in random trading; experts that leave come back. Done (`src/common/state/page_pool.cpp`): pages of 25,600 bytes for Qwen3.8 IQ3_S (0% of the experts' pages unused, against 4.4% with M2e's 332,800), the pool keeps runs of pages and ordered indexes, each expert size has a home point and an expert takes the free run nearest it or evicts one of its size, a conversation's K/V blocks are runs of 5 pages (27 int8 units, 0.1% unused, against 7.5% in one page). On the Strix Halo the same 7,846 experts take 15.97 GB instead of 16.69 GB; with the trading test's pool (18.6 GiB) 9,719 experts fit instead of 9,298, and after an 8.5K-token conversation all of them came back (M2e's build stayed 34 MB short). Fragmented share 0% (0.36% with 256 MiB chunks, so the GPU's pool is one chunk). Speed against the M4 build, 4 alternating rounds at 204,800 context, medians: 2K prompt 878 -> 862 tok/s, decode 24.65 -> 24.35; 8K prompt 1,062 -> 1,057, decode 24.13 -> 24.58 (noise). Unit tests: 1.9M pages, 23,960 experts placed in 0.05 s, 300 random steps of three conversations in 0.7 s, every expert back afterwards |
| M3 | Several conversations, the step with rows per step and conversations at once | answers identical per conversation; speed against `--batch`. Done (`strata::spec::Step`, `src/common/spec/step.cpp`): `strata-q38 --batch N --rows R --lookahead L` speaks the old engine's batch protocol (BGEN, BADM, BT, BDONE, BSTOP), so the server's "parallel" works unchanged; each step is one verify window with a row per conversation, each reading its own pages; `share_guess_rows` (the 2, 2, 1 rule) is ready for M4. A conversation takes pages when it gets its first request, nothing per slot at start: on the Strix Halo at 204,800 tokens, four held conversations took 0.49 GB (112 MiB of GDN state each) against the old engine's 12.4 GB, at the same or better speed (aggregate decode, 3 alternating rounds: 1, 2, 3 conversations 26.1, 40.9, 49.5 tok/s against 24.8, 39.7, 48.0). A conversation decoded beside others gives its tokens alone up to a late token in some prompts (mid: token 24 to 63 of 96), also with every expert on the GPU and in the old engine's `--batch`: the multi-row windows round differently |
| M4 | Drafting with the row-sharing rule and the lookahead | identical answers; speed against today's MTP. Done (`ConversationDrafter`, `src/models/qwen38/conversation_drafter.cpp`): `strata-q38 --mtp DIR --lookahead N` loads the MTP drafter (only with a lookahead) and each step asks it for each conversation's share of guesses, stopping below `--spec-min-p`; the drafter's K/V is one more region of each conversation in the pool (801 MiB of drafter against 1,034 MiB with its own K/V ring), and it reads a step's kept rows one step late, in the same pass as its next guesses. A request's `priority=N` orders a step's conversations; the extra rows go to the highest priority and rotate between equals. Answers (5 prompts, 128 tokens): with one guess a step, the same as without drafting up to the token where the old engine's own MTP also drifts (short 66, mid 119; two answers whole); with up to 3 guesses (the server's windows) they drift earlier (14, 24), at near-ties (the top candidates within 0.1-0.3 logits). On the Strix Halo (204,800 context, 3 alternating rounds, medians, tok/s): one conversation, a story / code: old MTP 34.4 / 44.7 (53% / 70% of drafts right), rows 4 lookahead 3 36.6 / 44.5 (58% / 70%), 1x3 35.3 / 46.5, 1x8 31.4 / 44.8 (40% / 56%), no drafting 25.7 / 24.1; three conversations: 3x8 54.7, 3x3 49.8, no drafting 49.7, the old engine's `--batch` 48.9 |
| M5 | Pausing to RAM and SSD, priorities; experts read from the SSD without the file cache | pause and resume give the same answers; files saved before load; with too little memory for the experts, the same answers as with enough, and RAM use stays at the setting. Done, but for pausing a running conversation (`RamExperts`, `src/common/experts/ram_experts.cpp`; `pool_copy`, `src/common/state/pool_copy.cpp`): strata-q38 keeps a second page pool in page-locked RAM (`--ram-mib`; by default what RAM and the commit limit have free less the resident headroom, 39-43 GiB on the Strix Halo) with the GPU's page size and conversation layout. At start it takes the experts the GPU's pool does not hold (the profile's order), then copies of the GPU's; an expert decoding reads from the SSD (unbuffered: `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`, through the file source) takes the place of the least recently used one of its size, never one the current layer used, else goes to a staging ring. This replaces strata-q38's RAM copy, arena and file tier (`--resident-experts` and `--mmap-experts` do nothing there; `--resident-budget-gib` stands for `--ram-mib`). An expert the GPU evicts is copied back to RAM when that measures at least twice as fast as an SSD read (`--copy-back auto`, best of three at start; on the Strix Halo 29-57 GB/s against 2.3-2.6 GB/s, so it is copied). A conversation that loses its place on the GPU (another request takes the place, or a growing one needs the pages: the lowest priority, then the least recently used, first) is paused: its blocks are copied to the RAM's pool while the conversations paused there take no more than one conversation of `--tokens-max` tokens (default `--max-context`), the lowest-priority one moving down to the SSD first; else into a file, written and flushed before its pages are given away (`--pause-dir`, at most `--ssd-pause-mib`, default 8192). A request whose prompt starts with a paused conversation resumes it; a seat whose GDN state comes back in other pages drops the verify windows' captured graphs. `GET_MEMORY_USE` reports the RAM's pool (experts, free, staging ring, allocated and fragmented pages, the paused budget) and the SSD (experts in neither pool, paused conversations, their budget). On the Strix Halo: answers identical to M2f (5 prompts, now with pauses between them); a 2,000-token conversation paused behind another and continued gives the same 48 tokens paused in RAM (140 MiB in 26 ms, back in 28 ms), on the SSD (135 ms, back in 65 ms) and moved from RAM down to the SSD; with a 4.4 GiB GPU pool, a 1 GiB RAM setting gives the same tokens as a 42 GiB one (with `--pcie-frac 0` and the prompt's CPU share off, as which experts the GPU reads over the bus depends on what RAM holds), the working set 2.9-3.8 GiB against 44.4-45.3 GiB. Against the old engine's low-RAM mode (`--resident-budget-gib`), the same 3000-slot GPU cache, two stories of 160 tokens, 2 rounds: 8 GiB of RAM decodes 9.0-10.3 tok/s against 7.4-10.5 (2.9-3.5 before the pool kept what decoding reads), working set 11.1 against 10.0 GiB; 1 GiB decodes 3.1-3.3 against 5.3-8.5, working set 4.1 against 2.8 GiB: there the old engine's adaptive tier moves the conversation's experts into VRAM, and strata-q38's GPU pool keeps the profile's. Left: the GPU pool taking the most-used experts while decoding; pausing a running conversation for a higher-priority one; the copy-back runs when the expert is evicted (between steps), not on a thread of its own |
| M5b | The GPU pool takes the experts decoding uses; a running conversation paused for a higher priority; stale pause files; the RAM outside the pools | the same tokens with the adaptive tier off; the low-RAM settings at least as fast as the old engine's; a preempted and a preempting answer complete. Done: strata-q38 counts each expert's routings while decoding (the old engine's counts, `--adapt-every`, `--adapt-swaps`, `--adapt-min-gain`, `--adapt-decay`) and every 4 steps moves the most-routed experts the GPU's pool lacks into the pages of the least-routed ones of their layer, by the old engine's rule, now shared (`choose_expert_swaps`, `include/strata/common/experts/expert_swaps.hpp`; its margin of 1.5 routings keeps experts from trading back and forth); the experts routed since the last round count as used, so a growing conversation evicts the least recently used ones, not the profile's coldest (M2e's open issue). A request's `priority` (an API field) can preempt: a BGEN of a higher priority to a busy slot ends that slot's request with `BDONE <slot> <n> preempted`, its conversation is paused, and the server sends it again with its tokens so far once a slot is free, which resumes it (`READY ... preempt`; the server picks the slot of the lowest priority below the new one's). Pause files carry the engine's process id, and a starting engine deletes those of engines no longer running. The RAM outside the pools was mostly the file source's read buffers, which each reading thread keeps at the size of its largest read: the prompt path's 32 threads kept 16-28 MiB each and the engine's thread 156 MiB from the start-up fill; reads on the prompt path now take at most 8 MiB. On the Strix Halo (same 3000-slot GPU cache, two stories of 160 tokens, 3 alternating rounds, medians, tok/s): 1 GiB of RAM 5.96 / 7.93 against the old engine's 5.25 / 7.91 and M5's 3.02 / 3.12; 8 GiB 9.58 / 11.13 against 7.39 / 10.58 and M5's 9.92 / 10.52; all the RAM 15.00 / 14.04 against 14.58 / 13.88. The moves cost 0.65-1.0 s per 160 tokens. Working set at 1 GiB: 3.76 GiB (M5 4.13, the old engine 2.83); what is left above the old engine is the page-locked region (2.36 GiB, fully resident, against 1.7 GiB) and the prompt path's smaller read buffers. Copy-back stays inline: it took at most 5.3 ms per 160 tokens (the RAM's pool rarely has room for an evicted expert colder than what it holds), so a thread of its own would save nothing measurable. With the adaptive tier off (`--adapt-swaps 0`) the answers are M5's (5 prompts); with it on, a 1 GiB RAM setting gives the same tokens as 42 GiB (with `--pcie-frac 0` and the prompt's CPU share off, 1,145 experts moved). One batch slot, temperature 0, tier off: a story preempted after 20 tokens by a request of priority 5 and continued gives the same 96 tokens as uninterrupted, and the preempting answer is the same too (with the tier on, the preempting answer differed after its first sentence: the GPU's experts then are another set than in the uninterrupted run). Left: the copies into the GPU's pool wait for each expert (a queued copy could overlap them, about 1 s per 160 tokens at most); a preempting request that fails after its BGEN can leave the slot's bookkeeping in the server to the other request's BDONE wait; a pause file whose process id a new process took stays. |
| M6 | Live `CHANGE_SETTINGS`, the panel's controls (web page and app), "this session" and "save as default" through the server; the app's own memory and parallelism settings (`parallel`, `mtp`, `vram_reserve_mib`, ...) give way to these, so the server is the only writer | resize while decoding; the biggest-conversation limit refuses. Done: strata-q38 answers `CHANGE_SETTINGS` (READY says `settings`) with the new `GET_MEMORY_USE` report or `ERR <why>`, which changes nothing. Both pools now take their memory in chunks of 256 MiB (the GPU's: the cache's arena is the first chunk, the others are allocated a page larger and start above it at a whole number of pages, so that the kernels' arena-plus-slot addressing holds; no virtual-memory API, so it works on AMD), and `memory_mib.<id>` grows a pool by chunks or shrinks it by emptying chunks: their experts are evicted, the conversations' blocks there are copied down (the K/V tables and the GDN state's address are rewritten, `ConversationPages::refresh`), and when they do not fit, idle conversations are paused, then the lowest-priority batch slot is preempted. A shrink is refused only below what the token minimum and the biggest decoding conversation at the end of its request need (`smallest=` on the `MEMORY` line, with `setting=` and `largest=`). Also live: `tokens_min`, `tokens_max` (the RAM's paused budget, as in M5), `ssd_pause_mib`, `drafter on|off` (off gives the weights back: 1.25 GiB more free on the Strix Halo; on loads them again in 0.4 s), `concurrent` (up to `--batch`; the waiting conversations take turns), `rows` (the verify windows are now sized for 8) and `lookahead`. New flags for the defaults: `--gpu-mib`, `--concurrent`, `--drafter`. The server: `POST /v1/memory {"set", "save": "session"|"default"}` (409 with the engine's reason; "default" writes the args of `strata-<model>.json` and keeps the .bak); `GET` adds `can_change`, `defaults`, `config`. The panel (`memory-panel.js`) drags the end of each pool and of the SSD's paused room (preview, red with the reason outside the range, nothing sent until release, Escape, arrow keys), has controls for the other settings, the session/default choice, Undo and the refusal message; the app forwards it (`POST /api/native/memory`), and its settings save sends `parallel` (as `concurrent`) and `mtp` (as `drafter`) through the server when the running engine can change them. On the Strix Halo: decoding a story while the GPU's pool went from 2 GiB to one chunk (an idle 8,500-token conversation paused, the story's pages moved down), the RAM's pool to half, both back, gave the same 96 tokens as without the changes, and the paused conversation continued to the same tokens (`--pcie-frac 0`, prompt CPU share off); each resize took 0.2-0.4 s with the story decoding, 2-16 ms on the engine's side through the server; `concurrent` 1 and back gave both answers unchanged; without changes the 5 prompts give M5b's answers (with `--pcie-frac 0` and the prompt CPU share off, as the RAM's pool now ends at a whole chunk). Left: more `concurrent` than `--batch` needs a restart; Windows counts the RAM's page-locked pool in the BIOS share's use (81 GiB of 63.8), so growth after start is limited by the free memory, not the share, and the server's Strix Halo bars read the share as full; a drafter switched on mid-conversation guesses from K/V it did not write until the next new conversation |
| M6b | M6's open issues: the Strix Halo's share counted with the RAM's pool in it, a resize that gets less than asked answered as success, the verify windows sized for 8 rows | the bars and the pools' ranges match Windows' own counters; a shortfall reaches the panel. Done: Windows counts page-locked host memory in the GPU's local use (DXGI: 4 GiB of hipHostMalloc raised it by 4 GiB) and in the process's shared GPU use (PDH), not its dedicated use, and HIP takes it off its free memory twice (8 GiB: 16 GiB). `get_share_room` takes the RAM pool's page-locked bytes off, and on a GPU whose share is known the GPU pool's room is the share's, not the runtime's free memory. The RAM's `MEMORY` line says `mapped=` (the page-locked pool), and the server takes it off what the GPU borrows: on the Strix Halo the RAM's bar counted the 20.95 GiB pool twice (as RAM experts and as 22.08 GiB borrowed by the GPU); now 1.05-1.13 GiB is borrowed (the engine's other page-locked buffers), and `in_share` (58.3 GiB) is what the engine holds in the 63.8 GiB share. Found on the way: ROCclr on Windows keeps freed memory in its resource cache and gives none back, so M6's shrinks freed nothing (a RAM pool shrink from 21.45 to 7.95 GiB left Windows' free RAM at 22.6 GiB); strata-q38 sets `GPU_RESOURCE_CACHE_SIZE=0` (an explicit setting wins), and the same shrink raised free RAM from 22.4 to 36.0 GiB, a GPU shrink from 49.75 to 40 GiB the share's free part from 2.2 to 12.0 GiB. Also found: since M6 the GPU's pool held only the experts of its first chunk's slot count (10,485) at start and left the rest of the pool free (on the Strix Halo 32 GiB of a 49.75 GiB pool); the others now come in by `refill` once the engine is idle (all of them within 20 s). And the default sizes let Windows' commit charge reach its limit (the GPU's memory counts in it on the Strix Halo: 8 GiB of hipMalloc raised it by 8.3 GiB; the RAM pool took what was left less 4 GiB, 23.9 GiB, and the PC froze): the pools now leave a tenth of the commit limit, at least 8 GiB, uncommitted (`get_commit_room`). A resize that gets less than asked: `CHANGE_SETTINGS`' answer has a `SHORTFALL <what>` line, `POST /v1/memory` a `"shortfall"`, and the panel says "Changed ... But it got less than asked: ...". The verify windows' buffers: 85.8 MiB at 8 rows against about 46 MiB at 1 (5.6 MiB a row, from the logs of 2- to 8-row windows), 0.06% of the share, so they stay sized for 8 |
| M7 | Measurements: guesses against other conversations' tokens, by where experts live; defaults chosen from them | numbers in DETAILS.md, measured on the Strix Halo and, if possible, the NVIDIA test PC. Done on the Strix Halo ([DETAILS.md](DETAILS.md#rows-per-step-lookahead-and-conversations-at-once-strata-q38), `tools/measure_rows.py`; the NVIDIA PC's run is prepared in `tools/nvidia_session/README.md`): 8 configurations from 1x1 to 3x8 with story, code and chat prompts, every expert on the GPU, most in RAM, about half on the SSD, 3 alternating rounds. A guess routes to fewer experts than another conversation's token (295-353 against 462 a row, of 480 for a token alone), but is kept only 50-72% of the time. One conversation: 3 guesses a step (4 rows) as fast as 2 and faster than 7 for prose (GPU: story 32.4 tok/s, 21.2 without guesses); several: 8 rows beat one row each (three conversations 52.0 against 40.8 tok/s on the GPU, 32.9 against 24.9 in RAM, 14.2 against 13.5 on the SSD). Defaults: rows 8 (kept), lookahead 3 with `--mtp` (it was 0 unless given), concurrent all the slots, `--tokens-min` 8192 and `--copy-back auto` kept (neither measurably costs or saves here) |
| M8 | Several GPUs: the shared layer split and planner, experts on any card, measured links, groups | tokens identical to one GPU; speed against today's layer split on the NVIDIA test PC (3060 + 3080, x16 and x4 if possible) In part (2026-11-01): the planner and the rules are shared code, strata-q38 runs on a layer split, and a card's spare pages hold other cards' experts, computed there in one hop; groups were added on 3 November 2026 with the validation limits above. Part 1: the layer split's planner (`plan_layer_split`, `src/common/experts/layer_split.cpp`) moved out of the old engine's loader, which calls it (its search, ties and retries unchanged); a model gives it per-layer sizes (`LayerSizes`: dense weights, one expert, fixed state, bytes per token) and each card its memory, speed and link (`make_paged_split_problem`). Links are measured (`src/base/gpu/links.cu`: each way and a small copy's round trip, to the host and between two cards); the rules (`src/common/experts/card_rules.cpp`): a hop's cost against the CPU's, refills paced by a card's link, copy-back only when at least twice as fast as an SSD read, and `place_experts` (each card its own layers' experts, then a card with room left the hottest experts of another card's layers where one hop is cheaper than the CPU, one card per layer). strata-q38 measures its GPU's link at start and paces refills and copy-back by these rules. Unit tests (`layer_split_test`, `card_rules_test`): a 3060 12 GB on PCIe 3.0 x4 and a 3080 20 GB on x16 cut at K=16 (14,080 of 24,576 experts on the cards); the 3080 at 9 GiB moves the cut to K=26; four equal cards 12,24,36; a Thunderbolt enclosure (2.5 GB/s) and an x4 card drop evicted experts (an NVMe drive reads as fast), an x16 card copies them back; one hop of 8 rows costs 0.13 ms over x4 and 0.23 ms over Thunderbolt, so it pays from 2 and 4 routed experts against 0.1 ms each on the CPU. On the Strix Halo (one GPU): strata-q38's 5 prompts give the M7 build's tokens (`--gpu-mib 14336 --ram-mib 8192 --pcie-frac 0`); its link measured 68.5-76.0 GB/s to the GPU, 61.1 back and 0.08-0.18 ms a round trip in two starts. Part 2: `strata-q38 --layer-split K[,K2..]|auto` gives each card (a `Card` in `paged_engine.cpp`) a page pool of its own (`Model::open_stage_pages` opens a stage's cache in it), with the experts of its layers (with `--expert-cache N`, the profile's first N, each on the card of its layer, so the cards hold what one GPU would) and each conversation's state for its layers (`ConversationPages` takes a session carved for a range of layers); the verify windows and the prompt path hand their rows from card to card as the old engine's split does (the hand-offs and a stage's hits are now `alloc_split_handoffs` and `Model::get_verify_hits(stage)`, which the old engine calls too); `GET_MEMORY_USE` has a memory per card and `CHANGE_SETTINGS memory_mib.<card>` resizes each; a paused conversation is one conversation in the RAM's pool, whose layout is each card's regions one after another (`RegionSpan` in `pool_copy.hpp`), or one file per card. `--split-device 0,..` puts several cards on one GPU, each with its own weights and pool (`Model::own_stages`): the test of a split here. Found on the way: the next card reads its part of a prompt chunk on a thread of its own, so `RamExperts`' blob and prefetch now take a lock (3 of 4 two-card runs crashed in the second chunk of an 8.3K prompt before, 0 of 3 after). On the Strix Halo, two stages on its one GPU (`--layer-split 24 --split-device 0 --trim-stage-weights`, 9 GiB pools, `--expert-cache 6000 --pcie-frac 0`, `STRATA_HC_Q8=0`, prompt CPU share off) give one card's tokens (14 GiB pool, same settings) for 5 prompts (24 to 8,478 tokens) and a conversation paused across both cards and resumed, in 4 runs, and also with every card's pool shrunk by 1 GiB and grown back before each request; one card's tokens are the M8 part 1 build's. Speed, two stages against one card (a busy PC, 6,000 experts on the GPU, the rest on the CPU): 8.3K prompt 154-199 against 114-125 tok/s, decode 5-13 against 4-8 tok/s (noise; not a measurement of a split, which needs two cards). Part 3: a card with spare pages holds the experts of another card's layers that their card has no room for, and computes them in one hop (`--remote-experts auto|on|off`, default auto). `PagedEngine::place_card_experts` runs `place_experts` on the profile's first `--expert-cache` with each card's pages less those kept for `--tokens-min`, and gives each card's pool the experts placed on it as its ranking (`ExpertPages::set_ranking`); a lent expert's slot goes to its holder's `Card::lent`, not to the residency table the owner's windows read, so the owner's windows treat it as a CPU row, and the CPU pool's dispatch hands the layer's lent experts to the holder (`SplitDrive::holder`, set per layer by `drive_pool_split`), whose `PeerExperts` computes them: the old engine's `--peer-device` path (one launch a layer: the layer's input to the holder, the same q8_1 and grouped kernels as a GPU's hits, the rows back into the CPU's mapped rows), here opened over the holder's pool (`PeerExperts::attach`) instead of a cache of its own. Auto lends where one hop over the holder's link to RAM costs less than the CPU computing one expert (0.035 ms: the Strix Halo's CPU pool, 70,349 experts in 2,449 ms); a pool resize places the experts again, a cut move first takes the lent experts back and places them after. Found on the way: on HIP the hop's stream waited for ever behind a verify window spinning on the GPU for its rows (the runtime puts streams on 4 hardware queues a priority; 2 of 2 runs hung, also with the hop's stream at the highest priority), so with a layer split strata-q38 asks for 16 queues (`GPU_MAX_HW_QUEUES`, and 32 CUDA connections; an explicit setting wins). On the Strix Halo (two stages on its GPU, `--expert-cache 6000 --pcie-frac 0`, M8's prompts, pools of 3 and 12 GiB, `--remote-experts on`): the second stage holds 2,272 of the first stage's experts and computed 25,962 of them in 6 requests, the same tokens in two runs; against the same experts on the GPUs with the lent ones held by the first stage (its hit path), the same tokens in 3 of 4 independent answers, the 8.5K-token answer differing at its 64th token (and the two continuations of it); the CPU computing them instead (`--remote-experts off`) differs from the hop in 3 of 4 (at tokens 64, 64 and 5). The 64-token continuation took 7.4 s with the hop, 7.6 on the CPU, 6.9 held by the first stage (one GPU: no measure of a hop between cards). Here auto lends nothing: a hop measures 0.12-0.14 ms. Part 4: drafting on a split: the MTP drafter runs on the last card (`Model::get_drafter_stage`, which `load()` and the live `drafter on` share), only that card's pool has the drafter's K/V; on the Strix Halo's two stages the drafter works and switches off and on, 269 of 392 guesses right against one card's 267, and the tokens are one card's in 2 of 6 answers: the split's final residuals are not bit-identical to one card's (a 23-token prompt's checksum 2496.38 against 2502.45), which drafting's near-ties show and greedy decoding without drafting did not; unexplained yet. Checked: the prompt path's threads ask `RamExperts::blob` only for experts its pool holds, so no staging buffer or page changes hands while a prompt is read (the lock of part 2 keeps the use order); the page tables are freed on their own GPU; the panel's per-card sizes save as `--gpu-mib`'s list. Groups now use the shared stage scheduler described above, with its reported identity failures. Left: why a split's residuals differ from one card's, the prompt path using the lent experts (it treats them as experts its card lacks), `--layer-split auto` for strata-q38 is the old engine's search with its cache sizes (not run here: one GPU). The NVIDIA session's `memory` phase now runs the two-card checks (`tools/nvidia_session/README.md` lists what it must confirm) |
| M9 | Moving the layer cut while running | a cut moved during decoding gives the same tokens | A first version (2026-11-01). When `CHANGE_SETTINGS` changes a card's memory, strata-q38 runs the planner again (`plan_layer_split` on `make_paged_split_problem` with the cards' pool sizes, each layer's expert, GDN state and K/V sizes, each card's predicted speed and measured link, and room for the decoding conversations) and moves the cut between two steps (`PagedEngine::move_cut`): every conversation is paused into the RAM's pool, each card's pool takes the regions of its new layers (`PagePool::set_layout`) and their experts (`ExpertPages::set_ranking`: a moved layer's experts leave the card it left, so the residency table names only the owner's slots), the sessions, now carved for every layer at the start, are described for the new range (`session_set_layers`), the verify windows and prompt paths take their new ranges (their graphs dropped), the decoding conversations come back, and each card fills its pool before the next step, so that the steps compute the same experts on the GPUs. The RAM pool's layout no longer depends on the cut: it has the whole model's regions, and a card's part is its share of the GDN state (a byte range of one region, `RegionSpan::part`) and its QSA layers' regions; a paused conversation's running state is kept apart from the cut too, and files written at another cut come back through the RAM's pool. A card keeps to the layers whose dense weights it holds: with `--trim-stage-weights` the cut does not move. One stderr line per move (`strata-q38: the layer cut moved: K=24 -> K=11 (...) in N ms`). On the Strix Halo (two stages on its GPU without the trim, 9 GiB pools, `--expert-cache 3000 --pcie-frac 0 --batch 1`): a story decoded in a batch slot while gpu0's pool went from 9 to 1.75 GiB after 24 tokens (K=24 -> 11 in 3.4 s, a conversation paused and back), a second conversation came back from RAM at the moved cut, both went to the SSD, the pool went back (K=11 -> 24 in 1.4 s) and both came back from the SSD files of the other cut: the same tokens as with no change (5 answers of 16 to 96 tokens); M8's two-stage check (with the trim and resizes) and its one-card run give M8's tokens. Faster (not done): copy only the moved layers' state from card to card in the background, the tokens added meanwhile after, one layer at a time, instead of every conversation through RAM (needs a pool layout per layer or per conversation); load a moved layer's dense weights, so that trimmed cards can move too; fill the moved experts in the background (the CPU computes them meanwhile, so the tokens may differ). The NVIDIA session's `memory-cut-move` check now moves the cut (`tools/nvidia_session/README.md`) |
| M9b | The first NVIDIA session's findings (`tools/nvidia_session`, 2026-11-01) | a pool grows back after a cut move on real cards | Done (2026-11-02). A pool that had shrunk could not grow back to its size (an RTX 3060: `ERR gpu0 has room for 1.00 GiB of pool, not 2.25 GiB`): the room a pool may grow into kept the prompt path's own buffers (and the drafter's head) free on top of the GPU's free memory after they were allocated, so they counted twice; after the start strata-q38 keeps only `--vram-reserve-mib` and an unbound drafter head (on the Strix Halo, two stages: the largest pool 31.14 -> 32.48 GB). The link measurement (`links.cu`) read an RTX 3080 on x16 at 3.4-6.8 GB/s against the old probe's 13.4: it now copies 256 MiB (as the old probe), its host memory touched first, after one copy each way (likely a link idling at a lower PCIe generation; on the Strix Halo 68.6-80.3 GB/s to the GPU, as before). The kit's two-card identity checks run without drafting, with 400 experts (the 3060 as second card has room for a pool under 1 GiB) and without the prompt's CPU share; `memory-rows` uses the kit's own run config (the copied one named the Strix Halo's paths). Refills take free pages only and evict no expert, so the experts that early prompts made hot stay. Tokens: strata-q38's 5 prompts and a pause on one card and on two stages with resizes, and M9's run give the build before's (and M9's reference) tokens. A live setting for the cut (`layer_cut`) and `--fill background` (the RAM's pool filled between requests) were tried and removed: the engines take no settings beyond the old engines' and the table above |
| M10 | No permanent prompt buffers: each prompt borrows its buffers from the page pools for the length of its read and gives them back. strata-q38 held them beside its pool for the whole run: on the owner's IQ3_S setup 1,003.6 MiB of VRAM and 365.3 MiB of page-locked RAM, an eighth of an 8 GB card. The owner's decisions: (1) conversations take the bottom of a pool (the `--tokens-min` reserve), the experts sit above by size around home points, and pages no expert needs collect at the top; the pool grows and shrinks at its top. (2) A prompt gets one run at the pool's top (`PagePool::lend`): free pages as they are; else the experts there are copied into free pages elsewhere (above the reserve, else the reserve's free pages), else evicted; conversation blocks there are copied out and their tables rewritten (`PagedPart::refresh`). The prompt's own conversation takes its pages first (`make_room`), so the lend leaves it what it needs. Giving the run back frees it, and the experts it evicted come back by the start-up fill's rounds. GET_MEMORY_USE shows the run as `prompt_buffers` ("Prompt buffers", its own colour in the panel). (3) The prompt path's host buffers come from the RAM pool's top the same way. (4) In the new engines `--prefill N` is the largest chunk: each prompt reads in the largest chunk up to it whose buffers the pools can lend then; `auto` and no flag mean 8,192. When not even 256 tokens' buffers can be lent, the request fails with an ERR that says so (waiting could wait for ever on a pool that is too small). The old engines' `--prefill` is unchanged. (5) The engine's side is shared: a model gives only "bytes a chunk needs" and "lay the buffers out here" (`PagedModel::get_prompt_buffers`, `lay_out_prompt`). (6) A first version kept a start-up arena sized for the largest chunk, which no shrink gave back; the owner rejected it, because small cards must still start: on the 6 GiB-pool run 5,385 MiB of VRAM is outside the pool, so an 8 GB card has about 2.3 GiB of pool with the drafter on. So the GPU pool is one virtual range instead | identical greedy tokens at the same chunk; GPU virtual memory under HIP against hipMalloc; prompt and decode speed against the build before; an 8 GB card's pool; the memory report during and after a prompt | Done for strata-q38 (2026-11-10). **Virtual memory under HIP** (a test program on the Strix Halo, Windows, HIP 7.1): a reserved 64 GiB range with 160 chunks of 256 MiB mapped back to back, 8 unmapped and freed from the top (free memory rose by 1.98 GiB) and mapped again; a kernel wrote and read 10.7 billion words across 160 chunk bounds with none wrong; per chunk hipMemCreate 5.0 ms, hipMemMap 4.3 ms, set access under 0.01 ms, unmap 0.3 ms; bandwidth through the range equal to hipMalloc's (6 GiB: copy 204 / 203 GB/s, read 231 / 227, write 212 / 209, strided 24.9 / 24.6, random 27.7 / 28.4). So the shared VMM code (`vmm.hpp`, now for CUDA and HIP with chunks of any whole number of the granularity) gives each GPU's pool one range as large as the GPU's memory: its chunks (about 255 MiB, whole pages and whole granules) are mapped back to back, and a run may cross them (`PagePool`'s joined chunks); a shrink unmaps and frees chunks from the top. There is no start-up arena for the prompt buffers, so no shrink floor for them. A GPU without virtual memory gets chunks of their own, and a lent run then stays inside the top chunk. The RAM pool is page-locked host memory without virtual memory, so its chunks are as large as the prompt path's host buffers (512 MiB for Qwen3.8 at 8,192 tokens). Qwen3.8's prompt path borrows its device buffers (`Prefill::relayout`, as the old engine borrows cache slots) and its host buffers (`Prefill::set_host_borrowed`: the PLE uploads, a split's hand-offs and the stager's ring). On the Strix Halo (Windows HIP, IQ3_S, the owner's run config, `--gpu-mib 49660 --ram-mib 5322`, MTP on): with `--prefill 2048` on both, a 5,019-token prompt, its continuation, and a 5,020-token prompt read while a story decodes give HEAD's tokens (5 answers, virtual-memory build). The memory report: "Prompt processing" 1,026.8 MiB of VRAM and 365.3 MiB of page-locked RAM before; now 0.03 and 1.1 MiB (small upload tables); during a 32K prompt "Prompt buffers" 2,750.9 MiB of the GPU's pool and 485.0 MiB of the RAM's, gone after it. A lend takes 4-100 ms. Speed, 3 alternating rounds, medians, HEAD (2,048-token chunks) against this build (up to 8,192): full pool, prompts of 4K / 8K / 32K 856 / 864 / 877 tok/s against 934 / 1,150 / 1,197; a 6 GiB GPU pool 117 / 125 / 122 against 196 / 341 / 347. Decode (128 tokens) once the evicted experts are back: 45.1 against 46.5 tok/s (full), 7.2 against 7.2 (6 GiB); right after the prompts, while they come back: 44.9 against 38.8 (full) and 5.7 against 5.3 (6 GiB). After the last prompt they were all back within 3.0-3.1 s (full) and 16-17 s (6 GiB). An 8 GB card (`--gpu-mib 2355`: a 2.24 GiB pool, drafter on) starts, and its prompts read in chunks of 4,096, 5,632 and 4,352 tokens (4K, 8K, 32K): 175-214 / 201-237 / 224-251 tok/s in two runs; decode 4.4 tok/s right after the prompts and 5.7 once settled; the experts were back 15.6 s after the last prompt. The fans were turned up; HEAD's decode right after the prompts fell from 47.8 to 39.5 over its rounds. Done for strata-ds41 (2026-11-10). Its chunked prompt path's buffers are laid out by one function, `Impl::lay_prompt` in `ds4_model.cu` (`lay_chunk`'s and the ring the streamed experts land in on the GPU; the ring's staging and the embedding and Engram rows of two chunks in page-locked memory): strata-dsv41's `alloc_chunks` allocates them once as before, strata-ds41 lays them out in the lent memory (`Model::lay_out_prompt`, the three hooks; `open_ram_pool` gets their host bytes). Both engines' memory reports call them "Prompt processing". Measured at start (owner's 2.00 bpw run config, 512K context, DSpark off): 756 / 982 / 1,435 / 2,340 MiB of VRAM and 106 / 173 / 308 / 576 MiB of page-locked RAM at chunks of 1,024 / 2,048 / 4,096 / 8,192; with DSpark on 3,472 and 1,531 MiB at 8,192 (its start sizes a 1.1 GiB ring). strata-ds41's `--prefill-chunk N` is the ceiling; without it, the old engine's rule (8,192 on the Strix Halo), so the default chunk is unchanged. A lend below the ceiling holds up to 16 rows more, for the prompt's last token, which the read takes along (else a read of a multiple of 256 tokens, plus that one, would take two chunks). strata-ds41 now skips the start's BLAS warm-up (no buffers until a prompt); the first 4K prompt was not slower on the Strix Halo. Memory report: "Prompt processing" 0.1 MiB; during a 32K prompt "Prompt buffers" 2,340 MiB of the GPU's pool and 397 MiB of the RAM's, gone after it. Tokens (`--expert-cache 4500 --no-expert-cache`, the placement pinned and the lends taking free GPU pages only): with DSpark off, HEAD + this change gave HEAD's tokens for a 12-token prompt, a 5,013-token prompt (one chunk), a 10,012-token prompt (two chunks), a continuation and a story; a 5,015-token prompt read while the story decodes differed, as it does between two runs of HEAD. With the 10K prompt read first, two runs of HEAD differed on it, and the second gave this change's tokens for all six. With DSpark on, two runs of HEAD differ from each other on the 5K and 10K prompts, so it was not a test. Speed, 3 alternating rounds, medians, HEAD against this change, prompts 4K / 8K / 32K tok/s and decode (128 tokens) right after them and later: with pools that fit the GPU's dedicated memory and the same GPU memory in both (HEAD `--gpu-mib 47104`, this change 49408; DSpark on 44544 and 48016), DSpark off 168 / 257 / 336 against 160 / 244 / 332, decode 9.2 / 9.2 against 8.6 / 10.1; DSpark on 151 / 228 / 375 against 172 / 314 / 424, decode 8.9 / 10.1 against 7.9 / 9.9. With the realistic run's `--gpu-mib 61440` the prompts were slower (DSpark off, medians of 3: 169 / 247 / 355 against 134 / 195 / 269): Windows kept this process' dedicated GPU memory at 55.1 GiB, the pool's top went to shared memory, and the lent buffers at the top sit there (an own allocation made then landed there too and was as slow). `get_share_room` counts only this process' use of the share; the pools are sized past it. Not measured: a layer split (strata-ds41's hand-offs between stages stay allocated at start); not built for CUDA |

M0 is useful on its own and touches no engine logic. M2 to M6 are the new engine; the old Qwen3.8 path stays until
the new one is at least as fast.

## DeepSeek V4.1-Flash on the memory manager

The owner's order: DeepSeek next, then Qwen3.6. DeepSeek gets the same pools, step, tiers, pausing, live settings,
panel and layer-split planner as Qwen3.8. Its own code is only what is truly DeepSeek's: its attention and the
state it keeps, its kernels, its Engram and its DSpark drafter. The old engine (`strata-dsv41`) stays as it is until
the new one is as fast; its answers must not change.

### What DeepSeek keeps, and where it keeps it today

Numbers from the IST-DAS 2.00 bpw pack's `ds4.txt` and `tensors.txt` (step D1's test computes them):

- **A conversation is small.** Each of the 40 layers and 3 DSpark blocks keeps a ring of the last 128 tokens (528
  bytes a token), and 3 of the 4 compressors keep a partial group (8 KiB each): 2.73 MB for the layers, 0.2 MB for
  the DSpark blocks. What grows is the compressed K/V and index keys of the 4 KV sources (layers 2, 8, 14, 20): a row
  of 288 and of 80 bytes per group of 2 tokens (1 for layer 20), 920 bytes a token in all. A 32K conversation takes
  33.7 MB; one of a million tokens, about 0.9 GB. The old engine reserves the caches for the whole context at load and keeps one
  conversation; the others are slots in RAM and in files (`slot_save`, synchronous copies, a `.meta` per file).
- **The experts are large.** 40 layers x 384 experts in 8 sizes, 7.7 to 10.2 MB, 123.0 GiB in all. Their greatest
  common divisor is 92,160 bytes, so the pages are 92,160 bytes and an expert takes 84 to 111 of them. The DSpark
  blocks have 3 x 128 experts of 18.8 MB (204 pages each, 7.2 GB).
- **Experts today.** Each layer has its own VRAM arena (one allocation, a spare slot for swaps); the kernels find a
  decode token's experts through a device table of pointers per layer (`vram_tab`), but the prompt's chunks address
  them as arena plus slot times stride. The CPU lane reads the others from the model's files in place (a mapping) or
  from its own RAM store (`ds4_store.hpp`: unbuffered reads, its own slots and LRU/LFU). Experts move into VRAM by
  their recent use (`cache_tick_swaps`).
- **No step of several conversations.** One position, one set of rings and caches; a window of up to 16 rows is
  one conversation's tokens (a prompt piece, or a token and its DSpark drafts).

### How it maps onto the shared code

| DeepSeek today | On the memory manager |
|---|---|
| Caches reserved for the whole context at load | `describe_conversation` (D1): per card a fixed region for its layers' rings and partial groups, one for the DSpark rings, and per KV source it reads a `per_block` region of compressed rows and one of index keys (one page holds 320 rows or 1,152 keys, nothing wasted) |
| Kernels read a source's rows at base + group x row bytes | at the page that holds the group (a table per region, as Qwen3.8's `units`, M2b) |
| A VRAM arena per layer and a spare slot | the card's `ExpertPages`, opened without a cache (the pointer table takes any page's address) |
| The prompt's expert sets at arena + slot x stride | a pointer per expert of a set |
| The CPU lane's mapping and RAM store | `RamExperts` over an `ExpertSource` of DeepSeek's files (unbuffered reads); the CPU lane asks it for each expert |
| `cache_tick_swaps` (heat) | the shared adaptive tier (`choose_expert_swaps`, `ExpertPages::replace`) |
| Slots in RAM and files, `.meta` prefix matching | pausing to the RAM's pool and to the SSD (`pool_copy`), `take_conversation`'s longest start |
| Checkpoints (host copies of the rings, up to 4) | stay host copies, kept with the conversation's other state beside its pages and paused with it |
| Its own layer split by free memory, `--tp` | the shared planner (`plan_layer_split`) for a layer split; `--tp` stays in the old engine (see "Several GPUs") |
| DSpark draft and verify | `StepModel::propose` (the drafter's guesses, its confidence floor) and `run` / `keep` (verify, roll back) |
| The Engram's reader and row cache | unchanged; the row cache counted in RAM, the table on the SSD |
| `describe_memory` | the shared report, the same panel |

### One shared paged engine

strata-q38's `PagedEngine` (2,442 lines in `paged_engine.cpp`) is mostly model-neutral: the conversations and their
places, pausing to RAM and to files, priorities and preempting, the batch protocol, the step, `CHANGE_SETTINGS`, the
pools' resizes and refills, the adaptive tier, the report, the cut's planning and moves. About 240 of its lines name
Qwen3.8's parts (the GDN and QSA state, the sessions, the verify windows and prompt path, the PLE history, the
stages, the hop to a card's lent experts), and its start (`run_paged_engine`, `init_card_paths`) is Qwen3.8's.
Step D3 moves the engine to `src/common/serve/paged_engine.cpp` behind a model interface
(`include/strata/common/serve/paged_model.hpp`), and each model's side to its folder:

- per card: the state layout for its layers (`describe_conversation`), the experts' sizes and ranking, loading an
  expert into its pages and publishing where experts are (Qwen3.8: the residency table of slots; DeepSeek: the
  pointer tables), the usage counts of the adaptive tier, a cut move;
- per conversation on a card: pointing the model at its pages, reserving pages as it grows, after a shrink moved
  them, and giving them back (Qwen3.8's `ConversationPages`), and the state beside the pages, as bytes (Qwen3.8: the
  PLE history and the indexer's tail; DeepSeek: the tokens, the Engram's history and the checkpoints);
- reading a prompt piece into a conversation; the step (`StepModel`: run, keep, propose); the drafter on and off;
  the sizes of each layer for the planner; the rest of the report (weights, scratch, the Engram).

`strata-q38` and the new `strata-ds41` stay two thin `main()`s in `src/program/`: DeepSeek's targets build with
their own options (`STRATA_BUILD_DSV41`, a forced HIP header), so one program with both families would tie the two
builds together for no gain.

### Steps

| Step | Work | Checked by |
|---|---|---|
| D1 | DeepSeek's conversation as page-pool regions: `describe_conversation` and `find_kv_sources` (`src/models/deepseekv41/conversation_layout.cpp`); the geometry (`Geometry`, was `Geo`) and the cache rows' sizes (`kv_rows.hpp`) move out of the old engine's sources so both read them; `choose_expert_page_bytes` moves to the page pool's library, which needs no GPU. Done (2026-11-01) | the unit test (`ds4_conversation_layout_test`, 81 checks): the pack's 4 KV sources, one card's regions (2.73 MB of rings and partial groups, 0.2 MB of DSpark rings, 920 bytes a token: the old engine's caches less their spare row), the regions of a cut at 20 and at 10 (a source read across the cut is on both cards), pages of 92,160 bytes (84 to 111 an expert), a block of one page for each growing region, and a pool holding a 32K conversation in 366 pages. The old engine builds; its device code is byte-identical (`.hip_fat`); its host code is the same source moved (the HIP build is not bit-reproducible: an untouched file rebuilt gives another object). Not run: no code path changed |
| D2 | The kernels read and write the compressed rows and index keys through a page table per region (the four readers: `index_score`, `unpack_sel`, `sparse_attn_packed`, the split-K `attn_part`; the writer `finish_kernel`'s packed rows, which may cross a page), and the rings and partial groups at any address; the host's copies of rows (hand-offs, `replicate`, slots) go page by page. With no table the arithmetic is today's. Done (2026-11-02): `KvRows` (`kv_rows.hpp`) is a source's rows on a card, back to back or in blocks of the pool through a device table (and its host copy, for the host's copies); `Finish`, `index_score` and `sparse_attn` take it. The rings and partial groups were already single pointers. The old engine's spare row (`max_context / ratio + 1`) is never written or read: a window ends at `max_context` at most and reads rows below `(pos + 1) / ratio`, so a conversation needs `tokens / ratio` rows, what the pool keeps | the parity test (`ds4_kv_rows_parity`, and with `DS4_ATTN_SPLIT=1` for `attn_part`): 2,500 rows in shuffled pages of 92,160 bytes against one array, written in 212 pieces that cross pages; the same bytes, the host's copies too, and the same bits from `index_score` (window, chunk, candidates) and `sparse_attn` (1 to 16 queries every way, a chunk's slab). The old engine against a build of the commit before: the same tokens in the smoke's main, none and resume scenarios, slots taken back across the two builds. Only the DSpark draft counts differ, and they differ between two runs of the same build too. HIP built and run; CUDA (sm_75/86/120) built. Not run: the hand-offs over several cards with tables (D9) |
| D3 | The paged engine becomes model-neutral (above): `PagedEngine` to `src/common/serve/`, Qwen3.8's side to `src/models/qwen38/paged_model.cpp` | strata-q38's tokens unchanged (M9's checks: 5 prompts, pauses, resizes, a cut moved), its unit tests, `tests/architecture`. Done (2026-11-02): the engine is `src/common/serve/paged_engine.cpp` (library `strata_paged`, run by `serve_paged_model`); it includes no model header and calls the GPU only through `src/base/gpu/`. A model implements `PagedModel` (`include/strata/common/serve/paged_model.hpp`: a class with virtual functions, as `DraftSource` is, since the engine also keeps objects the model makes; its constants are `PagedModelFacts`) and makes a conversation's state on each card (`PagedPart`). A card is a `PagedCard` (its layers, pool, link and lent experts); the model keeps its own side of each card by index. A paused conversation's state beside its pages is one byte string (`PagedModel::save_state`). strata-q38's start and Qwen3.8's side are `src/models/qwen38/paged_model.cpp` (`PagedQwen`, `ModelCard`, `Part`). On the Strix Halo, against the build before D3 (HIP, same tree), every answer's tokens are the same, and the protocol and stderr are the same apart from timings, free memory and how far the RAM pool's refill had got (two runs of the old build differ in the same way): one card (6 chat prompts, a conversation paused in RAM and back); two stages on its GPU with each pool shrunk by 1 GiB and grown back before each prompt; the MTP drafter (lookahead 3) on one card and on two stages, switched off and on; M9's run (the cut moved K=24 -> 11 while decoding and back, the conversations to the SSD and back from files of the other cut); two batch slots with drafting (two conversations decoding together, a priority 5 request preempting one, which came back from RAM in the other slot, `concurrent`, `rows` and `lookahead` changed, a pause to the SSD and back); `--remote-experts on` on two stages (pools of 3 and 12 GiB: 25,962 experts computed in hops, as in M8 part 3). One of two runs of the old build with `--remote-experts on` stalled in the hops (95 s waited, against 4 s) and failed at its 6th prompt with an unspecified launch failure; its first 306 tokens were the same (not caused by D3; not explained). Unit tests (15 of ctest), `tests/architecture` and the server's memory tests pass; strata-q38 and strata build for CUDA (sm_75/86/120, Docker) and strata for SYCL. Not run: two real cards (the NVIDIA PC) |
| D4 | DeepSeek's loader split as Qwen3.8's was (M2c): `Model::load` without placing experts or reserving caches; one conversation's prompt read and decode step over its pages (window and chunk paths); `strata-ds41`, a thin `main()`, with one conversation and drafting off; experts as in the old engine for now | token-identical to `strata-dsv41 --dspark none` at temperature 0 (the pack copy and the profile reset make it deterministic); the conversation's pages grow with it. Done (2026-11-02): `Model::load` loads the dense weights, the embedding table and the Engram's reader; `reserve_caches` gives the old engine its caches for the whole context, `place_experts` the prompt path's buffers, the experts and their threads (`strata-dsv41` calls all three, its options read by the shared `read_option`). `strata-ds41` (`src/models/deepseekv41/paged_model.cpp`, `PagedDeepSeek`; `main()` in `src/program/ds41_main.cpp`) opens the GPU's page pool before `place_experts` and points the model at its conversation's pages (`use_state_memory`): the rings and partial groups in the fixed region, the rows and index keys through D2's tables, which grow to pos + T before each window. The prompt's last token rides along with the prompt as the old engine reads it (`feed_mark`). One conversation, one GPU, one row a step; pausing refuses (D6), so a new conversation forgets the last. What every model's start does moved to `src/common/serve/paged_start.cpp` (the engine's options, the runtime's environment, the links, the pools' sizes and GPU memory); the `DONE` line is the model's (`PagedModel::print_done`), and so are a request's sampling defaults (`PagedModelFacts::sampling`); the residency table stays first pages, which each model publishes its own way. On the Strix Halo (the pack copy, the profile reset, `--max-context 8192`, `DS4_COMMIT_CAP_GB=30`, pools of 256 and 64 MiB): 6 greedy requests (21 tokens, a continuation, 429 and 1,283 tokens on the chunked path, a continuation after one, a new short one) give `strata-dsv41 --dspark none`'s tokens in two pairs of runs with `--no-expert-cache`; with the expert cache the old engine itself differs between two runs from the 4th request on (its moves run on a thread of their own), and strata-ds41 matched one of those runs through the 5th. The conversation's pages: 2.98 MB at 68 and 108 tokens, 3.25 MB at 436 and 475, 4.03 MB at 1,298, 2.98 MB again for a new one of 38 (pages of 20,992 bytes for this pack). strata-dsv41 against the build before: the same tokens in the smoke's main, none and resume scenarios (the DSpark draft counts and the slot folder's order differ, as between two runs). Free commit bottomed at 21.5 GiB with strata-ds41, 19.4 GiB with strata-dsv41 (DSpark on); strata-q38 gives D3's tokens. Also: strata-q38 refuses `--remote-experts on` when two cards share a GPU (`get_hop_refusal`), as such hops reset the GPU driver once (2026-11-01) |
| D5 | Experts in the pools: each card's `ExpertPages` (pages' addresses in the pointer tables), the prompt's expert sets through pointers, the RAM's `RamExperts` over an `ExpertSource` of the GGUF files (unbuffered), the CPU lane asking it, the adaptive tier, copy-back by measured speeds; the old store, mapping and swaps are not used | the same tokens with the tier off; with a small RAM setting the same tokens as with a large one and RAM use at the setting; speed against the old engine's store at the same budgets | Done (2026-11-02): `Options::expert_pages` makes the model leave its routed experts to the caller: no VRAM slots, RAM store, files' mapping or moves of its own. strata-ds41's GPU pool takes the profile's experts rank by rank over the layers (each layer's hottest, then each layer's second, ..., so a pool holds the same share of every layer, as the old engine's VRAM; `--expert-cache N` caps them), filled from the files before READY; `Model::set_expert_addresses` writes each expert's page into the decode's `vram_tab` and, as slots in the profile's order, into a per-layer `slot_tab` the prompt's expert sets read (`SetArgs::wtab`; without a table the kernels' arithmetic is the old one). The RAM's pool is a `RamExperts` over `ExpertFiles` (`expert_files.cpp`: each matrix read without the file cache, sector-aligned, into staging and moved into the blob), opened with the experts the GPU's pool lacks first (`open_ram_pool`, now shared with strata-q38); the CPU lane asks it for 8 experts at a time (`kCpuLaneExperts`, its staging ring), the prompt path DMAs held experts from its page-locked pages and reads the others from the files unbuffered. The shared adaptive tier counts each window's routings (`--adapt-every 4 --adapt-swaps 8 --adapt-min-gain 1.5 --adapt-decay 0.7`), copy-back is strata-q38's `--copy-back auto`, `GET_MEMORY_USE` reports the weights, scratch and experts by tier, and the expert profile is written back after each request (`PagedModel::end_request`) and at the end. On the Strix Halo (the IQ2_XXS pack copy, the profile reset, `--max-context 8192`, `--dspark none`): with the same 1,200 experts on the GPU on both sides (30 per layer: strata-dsv41 `--no-expert-cache --ssd-cache 0` with `DS4_COMMIT_CAP_GB=18.912`, strata-ds41 `--expert-cache 1200 --gpu-mib 10240 --adapt-swaps 0`) D4's 8 requests (short, continued, 429 and 1,283 tokens on the chunked path, a STOP while a prompt is read) give the same tokens; a 1 GiB RAM setting gives the 8 GiB setting's tokens, private bytes 5.03 against 12.05 GiB (the 7 GiB of the settings' difference). Speed at the same budgets (1,200 GPU experts, 8 GB of RAM: strata-dsv41 with its expert cache and `--ssd-cache 8`; strata-ds41 `--ram-mib 7630`, tier on), 3 alternating rounds, medians: decode story / code / count 3.71 / 3.44 / 5.41 tok/s against strata-ds41's 3.96 / 3.31 / 5.44; a 1,283-token prompt 50.5 against 65.0 tok/s; private bytes 11.9 against 11.5 GiB (the old engine also kept 18.7 GB of the files in the OS cache). Left: the layer-major prompt's layer cache (prompts of several chunks) is RAM outside the pool; the drafter's experts (D7); EXL3 packs and several GPUs refused; the Engram's table and row cache are not in the report |
| D6 | Pausing, priorities, `LIST_CONVERSATIONS`, the checkpoints beside the pages (a re-rendered history goes back to its anchor), `GET_MEMORY_USE` and `CHANGE_SETTINGS` (most of it D3's) | a conversation paused to RAM and to the SSD continues with the same tokens; a chat whose earlier reasoning is dropped restores as the old engine's slots do | Done (2026-11-02): `strata-ds41 --batch N` holds the conversations of N slots and "main" and decodes one a step (rows and conversations at once stay 1 until D7 and D8). The model computes one conversation at a time: the host side of each one (its tokens and up to 4 checkpoints, `ConversationState`, `conversation_state.hpp`) is in the model while it is computed and in its part otherwise (`Model::swap_conversation`); its rings, partial groups and rows are its pages. A paused conversation's pages go through `pool_copy` as strata-q38's do; its host side is the state beside the pages (`encode_state`'s bytes, `PagedModel::save_state`), so the checkpoints pause with it. The read keeps the old engine's checkpoints (before the prompt's last token, at its end, and the anchor when the conversation started over or went back), also when a prompt's last token is all that is left to read. A prompt that leaves a conversation before its end continues it from the latest point at or before where they part (`get_resume_tokens`, `include/strata/common/state/resume_point.hpp`, which strata-dsv41's slot choice now calls too; the points are `PagedModel::get_rewind_points`, none for Qwen3.8): it comes back there (`PagedModel::rewind`) and stays paused whole as well, as the old engine's slots keep it. The old engine's slot settings map onto the engine's: `STRATA_DS4_SLOT_KEEP` is always on, `STRATA_DS4_SLOT_DIR` is `--pause-dir`, `STRATA_DS4_SLOT_CAP` (conversations kept in RAM) is `--tokens-max` (the paused conversations' tokens in RAM), `STRATA_DS4_SLOT_GB` is `--ssd-pause-mib`; `STRATA_DS4_SLOT_MIN`, `STRATA_DS4_SLOT_DAYS` and keeping conversations across runs have none (pause files belong to the process). On the Strix Halo (the pack copy, the profile reset, `--max-context 8192`, `--dspark none`, 1,200 GPU experts on both sides: strata-dsv41 `--no-expert-cache --ssd-cache 0`, `DS4_COMMIT_CAP_GB=18.912` with its slots on, strata-ds41 `--expert-cache 1200 --no-expert-cache --gpu-mib 10240`): three interleaved conversations, 8 greedy requests (one of 429 tokens, a history rendered again without its first answer's start, which went back to its anchor at token 20 of 92, and a prompt repeated, back to token 143 of 167) give strata-dsv41's tokens with the pauses in RAM (`--ram-mib 8192`) and on the SSD (`--ram-mib 1024 --tokens-max 0`: 7 pauses to files), and the tokens of the same conversation alone. A paused conversation of 21 to 444 tokens took 3.5 MB of pages (pages of 92,160 bytes): paused in RAM in 6-15 ms (once 85), on the SSD in 11-83 ms; back in 3-5 ms from RAM, 11-17 ms from the SSD. `--batch 2`: two slots decoding by turns, a priority 5 request preempting one after its first token, the preempted conversation back in the other slot, the paused ones moved to the SSD by `tokens_max=16` and back, the RAM's pool shrunk to 4 GiB and grown, `concurrent=2` and `rows=2` refused: 8 requests give the tokens of the same requests one after another; the GPU's pool shrunk by 0.5 GiB and grown back refilled its experts before the next request. strata-dsv41 against the build before: the smoke's main and none scenarios the same apart from timings, resume the same tokens with the slot folder's order. strata-q38 against the build before (one card, two stages with resizes, M9's cut move, the batch with `CHANGE_SETTINGS`): the same tokens, the protocol the same apart from timings and the RAM pool's refill. Not done: a GEN's `pin=N` (checkpoint 3) is not read; a STOP still waits for the prompt's read; the checkpoints of a conversation paused to the SSD stay in RAM and are not in `GET_MEMORY_USE` (strata-dsv41's slots, which carry them, took 10.9 to 11.3 MB for the conversations whose pages took 3.5 to 3.6 MB) |
| D7 | DSpark through the step: `propose` drafts up to the conversation's share while the confidence holds, `run` verifies the rows, `keep` rolls the rings and compressors back; the drafter's experts in the pool as 3 more layers; `drafter on/off` live | the old engine's DSpark tokens (greedy); acceptance and speed against it | Done (2026-11-02). `strata-ds41` drafts as the old engine's rounds do: `propose` runs the drafter's block after the step's token and keeps the drafts while their confidences hold (`Model::count_confident`, now shared with strata-dsv41: `--dspark-conf`, `dspark_cum`), none for the prompt's last token and none after a proposal that offered none; `run_rows` verifies them in one window (`Model::verify_rows`) and `keep_rows` rolls the rings and compressors of the rejected rows back and seeds the drafter with the kept ones (`Model::keep_verified`; `Model::verify` is the two in a row). The drafter's rings are D1's `drafter` region of the conversation's pages, so they pause and come back with it; its 384 experts (18.8 MB) are in the GPU's pool before the trunk's, as layers after them (pages of the trunk's size, rounded up), and the RAM's pool holds none: it drafts only while the GPU holds all of them. `drafter off` gives those 7.22 GB to the trunk's experts and `on` takes them back in place of the coldest. Follow-up 2026-11-04: dense tensors also unload/reload; recent trunk taps in conversation pages and checkpoints let the rings be rebuilt. No new settings: `--dspark none` loads no drafter (as `--lookahead 0`), `--dspark off` is `--drafter off`, `--dspark-block N` is the default `--lookahead`, else the drafter's block (5); `--expert-cache N` counts the drafter's experts; a request's `spec=`, `dspark_conf=`, `dspark_cum=` / `spec_cum=` and `pin=N` (D6's open issue: the read stops at N for the checkpoint other requests go back to) are read as the old engine reads them (`PagedModel::start_request`, was `set_sampling`). One conversation a step (`PagedModelFacts::max_concurrent`) until D8. On the Strix Halo (the pack copy, the profile reset, `--max-context 8192`, `--dspark-conf 0.65`, 1,200 trunk experts and the drafter's 384 on the GPU on both sides: strata-dsv41 `--no-expert-cache --ssd-cache 0` with `DS4_COMMIT_CAP_GB=26.10` and its slots on, strata-ds41 `--expert-cache 1584 --no-expert-cache --gpu-mib 18432 --ram-mib 8192`), 7 greedy requests (48 tokens of code, a continuation, 429 and 1,283 tokens on the chunked path, a continuation after one, a 444-token prompt with `pin=300` and one that goes back to its token 300) give strata-dsv41's DSpark tokens in the first 5 and differ from token 8 and 12 of the last two: the drafter's inputs (`DS4_DSPARK_DUMP`) differ between the engines from the prompt's second row on (by up to 10% at layers 37 to 39; the first row is the same) while the trunk's tokens agree, so confidences near the floor give other draft counts and windows of other sizes (the placement difference is explained below; with `--dspark none` both engines give the same tokens in all 7). Each engine's DSpark tokens differ from its own `--dspark none` in 2 (strata-ds41) and 3 (strata-dsv41) of the 7. strata-ds41 gives the same tokens and draft counts in two runs, and with `drafter off` and `on` again between two requests (the GPU's experts 17.54 -> 10.32 GB and back within 20 s). Speed at these budgets, 2 alternating rounds, 64 tokens each, story / code / count: drafts kept of offered 8/10, 52/52, 52/52 (strata-dsv41) against 6/9, 51/54, 52/52; decode 3.62 / 4.06 / 7.92 tok/s against 3.80 / 4.79 / 9.41. strata-dsv41 against the build before: the smoke's main scenario gives the same tokens (one request's draft counts differ, as between two runs). Lowest free commit 51.0 GiB. Follow-up (2026-11-03): the input difference comes from CPU/GPU expert placement, not DSpark state: D7's equal totals put 69 trunk experts from layer 0 and 29 from each other layer on the old engine's GPU, against 30 per layer on the new one; matching the actual experts in a temporary diagnostic run makes all 21 dump records (taps, rings, hidden states and confidences) bit-identical for a 21-token code prompt and 48 generated tokens, with 38/38 drafts kept on both, so no activation-path fix is needed. Open: a stop token among accepted drafts now leaves the state at the stop (the old engine keeps the drafts past it, so a continuation may resume one token later) |
| D8 | Several conversations share the trunk pass and one batched DSpark network forward, with each row bound to its own rings, compressors, pages and absolute position. Done (2026-11-05): concurrent drafting is back. The shared step divides `--rows`; the existing drafter on/off and `--batch` / live Conversations at once controls apply. There is no new setting or automatic mode. An engine opened with DSpark support keeps synchronous steps because rollback scratch is shared across stage groups. | Windows HIP, Ryzen AI MAX+ 395 / Radeon 8060S, IST-DAS 2.00 bpw, 256-token greedy seasons/repetition/story answers: 49.99 GiB GPU plus 24.74 / 35.74 GiB page-locked RAM pools (80.24 / 92.05 decimal GB). The highest pooled on-mode grid rates were 7.83 tok/s at three conversations, `--rows 5`, and 10.54 tok/s at three conversations, `--rows 4`. Two conversations at `--rows 4` gave 5.97 / 8.24 tok/s; the small-window two-conversation hypothesis is not established. A reversed-order larger-pool seasons/repetition check favors two conversations at `--rows 5` (16.84 tok/s against 14.97 without guesses), without establishing the best width or full-prompt mix. Each prompt has at least two measured answers per cell. In a separate 32 GiB requested-RAM suite, seven fixed-placement concurrent/live answers of 256 tokens, four tiled-batch answers of 16 tokens and an SSD-restored 256-token continuation match standalone references. Adaptive bitwise comparisons differ, including repeated no-guess answers. The earlier 48/6 GiB joint-window experiment was reverted; its sequential drafter timings are excluded. Full grid, per-slot kept/offered counts, SSD expert reads, fresh controls, batched drafter scaling and passed/failed/not-run checks: [5 November report](../bench/results/2026-11-05-ds41-drafting/README.md). |
| D9 | The layer split (shared planner, cut moves) on DeepSeek's per-layer sizes | Implemented; separate-card validation pending. The stage adapter, expert tables, cut-independent state transfer and last-stage DSpark placement are present. Dense tensors now reload on a moved layer's new GPU; DeepSeek computes lent experts with its own kernels under the common placement and safety rules. See the D9 follow-up below for the local validation and remaining hardware checks. |
| D10 | The server and `strata launch` start `strata-ds41` when asked (opt-in); MODELS.md and DETAILS.md get the measured numbers | Implemented (2026-11-03), opt-in through the existing `exe` path: the server config, `strata launch --set exe=PATH`, and the app's Engine program field. Shared Python argument translation preserves context, GPU selection and starting concurrency; unsupported options are logged and omitted, including the picture encoder. Original executables remain the default. The packager includes strata-q38 and strata-ds41 when staged beside strata; release recipes still need to build and stage them. MODELS.md and DETAILS.md carry the D7 and M12 measurements with Windows hardware and budget details. Both engines passed chat, streaming, two simultaneous requests and GET/POST /v1/memory through isolated servers on port 8091. DeepSeek also passed an Anthropic request, the existing smoke answer check and a forced get_weather tool call. The existing DeepSeek server tests with the real tokenizer and architecture checks passed (66 tests). PID-scoped watchdogs checked free commit and physical RAM; no trips. No browser, owner server, original model files, commit or push. Focused launch and engine-choice checks: 209 passed, 1 skipped after initializing two engine-choice mocks with real launch settings. All selected server tests passed. Broad run: 2,168 passed, 28 failed, 53 skipped and one cleanup error; the two mock failures were corrected and verified above, leaving 26 other Terminal failures unverified against HEAD (one TUI marker failure was reproduced with HEAD launch modules). Host serial run: 1,035 passed, 101 skipped, only the known timeskip-real.test.mjs:123 EPERM cleanup failure; current native settings tests: 5 passed. Live concurrency changed from 2 to 3 on both engines. Follow-up: isolated the discovery fixture from checkout run configs and classified virtualenv site-packages as dependencies in the import-graph check; forbidden installer modules remain checked across all imports. The four reported IDs plus focused launch checks passed: 213 passed, 1 skipped (explicit paths only; no lifecycle matrix or engines). Detailed logs and scripts: `%TEMP%/d10-launch`. CUDA, SYCL, Metal and multi-GPU inference were not run. The first broad test command inadvertently included the fake-engine lifecycle matrix; it was stopped and the final run explicitly excludes that file. |

D1 and D2 change no behavior. D3 is the largest refactor and is checked on Qwen3.8 alone, so it could go first; it is
third because DeepSeek's side (D2's tables, D4's loader) shows what the interface must hold. Running DeepSeek on the
test PC needs care: its commit limit is 127.6 GiB, and DeepSeek has reached it before. Every engine start reads the
commit charge first, caps each pool so that the pools and a 16 GiB margin fit, and runs with a watchdog that stops
the engine within 16 GiB of the limit.

## Qwen benchmark correction (2026-11-03)

The suffix (`4c047a0f`), picture (`dce39bcf`) and batch-group (`0b8b86ab`) timings below are
restricted-memory checks. They do not establish speed at the owner's normal IQ3_S settings. The saved
`strata-iq3_s.json` uses `--expert-cache auto --resident-experts --prefill auto --spec 4 --spec-min-p 0.5`,
MTP from `Strata-data/mtp/rt`, `--max-context 204800 --kv int8 --vision --vram-reserve-mib 700`, and
server `parallel: 2`. Its historical log records 24,576 GPU expert slots (46.84 GiB) and 2.69 GiB of resident
RAM experts. The suffix check instead requested 6,000 GPU experts with 14 GiB GPU / 2 GiB RAM pools in the
new engine and a 2 GiB resident budget in the old engine; actual GPU residency differed. The picture check
used 4 GiB GPU / 2 GiB RAM pools and 400 GPU experts; groups used two 2 GiB stage pools and 2 GiB RAM,
requesting 400 experts. Their short answers and fixed warm-up counts do not meet a stable-residency,
two-round, 256-token comparison at realistic budgets.

The realistic rerun completed on 3 November 2026, Windows HIP, Strix Halo / Radeon 8060S, native IQ3_S,
204,800 context, int8 K/V and vision enabled. The old engine used `--expert-cache auto --resident-experts
--resident-budget-gib 4 --vram-reserve-mib 700`; the new one requested `--gpu-mib 61440 --ram-mib 512
--expert-cache 24576 --vram-reserve-mib 4300`, with `STRATA_COMMIT_RESERVE_MIB=3072`.
Actual new GPU pools were 52.50-53.75 GiB. Both engines held all 24,576 experts (46.84 GiB) on the GPU,
with no measured decode expert file loads. Idle residency checks and two three-prompt warm-up rounds
preceded two measured 256-token rounds. No-draft old/new seasons rates were 25.20/23.98 tok/s;
MTP-only rates were 43.21/37.08. Repetition with MTP plus suffix gave 60.29/67.37.
New three-conversation aggregate rates for neither, suffix, MTP and both were
48.38, 49.47, 54.48 and 56.05 tok/s, excluding prefill and tokens before the last admission.

In these runs, shared startup filled the permitted GPU expert pools before `READY`; Qwen can select its resident
graph when that fill completes. The adapter owns prompt scratch, initializes pages on use, and enables the existing
one-row state-commit path. Shared expert touches reuse index nodes; verify-width sizing is shared with the
original Qwen engine. Full-residency startup was old 39.41-40.28 s and new 39.29-45.65 s.
The old saved mode maps files, uploads through one-expert host staging, then trims file-backed working-set
pages and pins the RAM complement. The new engine also uses host staging, into pool pages. Neither streams
experts during these warmed decode rounds. Startup order is not evidence for the remaining steady-state gap.

Since then the paged engines say `READY` right after the dense weights load, and fill their expert pools while they
serve. The GPUs' pools come first, in the profile's order. Then the RAM's pool takes the experts no GPU holds. There
is no setting for it. The memory report says how far the fill is (`EXPERT_FILL`, and `"expert_fill"` in `GET
/v1/memory`), and the speed tools wait for it before they time anything ([DETAILS.md](DETAILS.md#the-memory-panel)).
This start-up fill is the only loading in the profile's order; it ends for good when the pools are full. After it,
only the adaptive tier moves experts, by the conversations' use (prompts included on Qwen). It uses free pages while
there are some and trades experts only in a full pool. It runs every decode step while a pool has more than 1 GiB free
beyond `--tokens-min`'s room. The profile only seeds the use counts, below the tier's threshold. On the Strix Halo
Qwen3.8 said `READY` after 9-10 s instead of 38 s and DeepSeek after 14-18 s instead of 58 s; the measurements are in
DETAILS.md.

The owner's task-only watchdog floors were 3 GiB free commit and 4 GiB free physical RAM. All main matrix
runs passed: minimums 3.538 and 18.356 GiB. The commit limit stayed at 127.295 GiB; observed system charge
increases were old 56.657-62.471 GiB and new 63.204-64.854 GiB, including background changes.
A separate saved-mode mapped startup hit the physical floor before its final trim. Exploratory controls
also hit the commit watchdog; all are kept apart from the completed matrix in the report.

Matching every old-engine speed and universal greedy identity remain unmet. Old/new normal and story
answers differ, as do new concurrent/standalone answers; repetition matches. The old engine's own MTP-on/off
answers differ too. HIP builds, `step_test`, 1,121,445 page-pool checks and 16 architecture/launcher tests
plus two subtests passed. CUDA/SYCL/Metal, DeepSeek inference and real multi-card groups were not rerun.
[DETAILS.md](DETAILS.md#qwen-page-pool-benchmark-budgets) and the
[full report](../bench/results/2026-11-03-q38-realistic/README.md) retain the historical restricted results,
all realistic rates, exact commands, memory logs, source locations and remaining failures.
The initial pass stopped when free commit 64.479 GiB was below the revised 68 GiB
preflight requirement, with 23.879 GiB physical RAM free. Separate failed attempts reached minimum free
commit 0.004 GiB and minimum physical RAM 3.960 GiB; test PIDs were killed. The system-managed page file
stayed at 65,174 MiB. No engine remains running; the main completed matrix's minimums are unchanged.

### Full-residency follow-up: short prompt reads

The old server reads a short prompt through verify windows (`--short-read 64`) and splits it at the
final chat-turn boundary. The paged adapter always used batched prefill. Those paths use different
projection input precisions. This explains the story's token-2 difference; changing owned versus borrowed
prompt scratch could not remove it. The adapter now uses the old short-read width and boundary and feeds
the drafter from those verified rows. All 41 captured story rows matched the old engine exactly across
47 completed-layer HC means and all 248,320 logits per row. The original token-2 logits and per-layer
errors are retained in the [follow-up report](../bench/results/2026-11-03-q38-realistic/README.md).

Single-conversation batch windows reuse the ordinary QSA attention, grouped K/V and indexer appends,
and combined GDN convolution/gate kernel. Per-slot GDN recurrence writes quantized output directly.
Both verifier commit paths now
share the existing all-layer QSA indexer helper, including the conversation's rotary table. Existing
overrides and per-layer fallbacks remain available. These changes do not establish universal identity
across different concurrent row layouts.

The fully resident paged path also keeps shared projections on the main stream for the measured
single-gfx1151 GGUF IQ3_S tier. Two warm and two measured 256-token seasons controls improved to
32.62 tok/s without MTP and 52.04 with it; answers stayed identical. The preceding default-path pair
was 24.83 and 44.34. The model metadata selects the tier. Explicit `STRATA_SH_STREAM` settings win;
other devices, tiers, partial-residency graphs, split runs and legacy callers retain their policy.
These controls and the final comparison are retained in the linked report.

Single-GPU MTP also shares the original engine's asynchronous commit behavior. The shared engine's state
fence runs before commands, actual page growth, interleaved prompt chunks, request completion and cleanup.
Reservations within already published physical units only advance host bookkeeping. Verifier row
coordinates have their own buffer, so the drafter keeps its full cell table between steps. Short prompt windows
preserve pending drafter residuals before reusing verifier scratch. Snapshot and coordinate copies run
on the draft stream, with a wait before prompt admission reuses their source. The guarded text/image
control keeps both 256-token outputs identical after live memory changes and three cancellations. Split and legacy batch callers retain synchronous commits; `STRATA_COMMIT_SYNC=1` remains available.

The final 4 November 2026 two-round Windows HIP comparison on Strix Halo, all 24,576 experts
resident, measured seasons at 26.50/31.85 tok/s old/new without drafts and
44.34/51.14 with MTP. Story was 27.04/31.15 without drafts and
36.46/42.37 with MTP. New was faster in all 12 standalone prompt/mode comparisons, by 11.3% to 40.2%.
New three-conversation aggregate rates were 60.09 without drafts and
65.65 with MTP. The full suffix matrix, exact answers, memory floors and
passed/failed/not-run checks are in the [paired report](../bench/results/2026-11-03-q38-realistic/README.md).
Suffix-history and concurrent-layout differences remain; these measurements do not promise universal identity.

## Pictures in the replacement engines

`strata-q38` shares the old engine's embedding-file reader and rotary-coordinate builder; `strata-ds41` shares
DeepSeek's RGB reader, span cache and built-in encoder. The server keeps the selected picture reader.
Picture identity travels with paused pages and saved sessions and prevents reuse across different pictures.
Qwen keeps its coordinates beside the pages and stages each verify row's own coordinates. Picture and text
conversations share a step; pooled indexer keys use stable per-conversation device tables. Changing the picture set
rereads the whole prompt. CUDA and SYCL still need validation on their own hardware.

The 3 November 2026 Windows HIP check on Strix Halo, IQ3_S, 8K context, 4,096/2,048 MiB GPU/RAM pools
keeps all three 64-token mixed answers identical to standalone and 4c047a0f. Sixteen greedy tokens each match
`strata` on the picture, Python and story prompts. Picture RAM pause/resume and saved restore in a fresh process
keep the next eight tokens. With MTP and suffix drafting, three mixed 32-token answers match standalone; the
picture answer also matches with drafting disabled, and its concurrent suffix guesses keep seven of seven.
The throughput setup and before/after numbers are in [DETAILS.md](DETAILS.md#pictures-in-the-page-pool-engines).
The architecture tests (5), `step_test` and `prompt_images_test` pass. PID watchdogs enforce 16 GiB free commit
and 4 GiB free physical RAM; no trips. Logs, scripts and token lists are in `%TEMP%/pictures/batch-*`.
No CUDA, SYCL, Metal or real multi-GPU validation was run.

Realistic-budget follow-up (3 November 2026): Windows HIP, Strix Halo, native IQ3_S, 204,800 context,
all 24,576 experts on the GPU, the 60 GiB requested new pool and bounded old resident setup above,
no drafting, two measured 256-token rounds after warm-up: picture/counting **25.06 old / 24.76 new tok/s**;
new picture/seasons/repetition together **49.91 aggregate tok/s**, excluding prefill and earlier admissions.
Existing 16-row image embeddings exclude encoder time. The picture matches old/new and standalone in both
mixed batches; seasons differs at token 56, repetition matches. Old/new minimum free commit was 8.068/5.785 GiB,
minimum physical RAM 23.723/26.262 GiB. This supersedes the small-pool picture check as a speed reference
for this machine, without changing that check's historical provenance.

An old-engine full-residency `--batch 3` control at 8K, no drafting, found story differs from standalone
at token 186 in both 256-token rounds; seasons and repetition match. This does not establish the cause
of the new engine's earlier divergence or rerun the exact live-group failure sequence. The old 204,800-context
three-slot startup hit the commit watchdog; the completed main solo matrix retains that context.

## Later

- Qwen3.8's paged verifier applies request penalties to each row's conversation and earlier drafts. On the Strix Halo,
  one greedy penalty test differed with drafting from token 26: its history matched, but its logits differed. The old
  engine's drafted and non-drafted penalty outputs also differed on that prompt. DeepSeek's sampling omits penalties.
- **Tree drafting**: a conversation's share of rows spent as a tree of guesses (several likely continuations
  checked in one pass). The parent table of the step allows it from the start; it needs the drafter to propose trees,
  the verify kernels to take tree-shaped parents, and for GDN layers a running state per branch.
- Qwen3.6 on the new step and pools, after DeepSeek ([above](#deepseek-v41-flash-on-the-memory-manager)).
- More than 8 rows per step, and so more than 8 conversations at once: the verify kernels widened beyond 8 rows.
  Not several passes per step, which would read the weights more than once. Kernel work for a later pull request,
  not this one.

## Open questions

- The desktop app's Windows and Linux installers download the engine from upstream's signed feed. App users get the
  new engine only if upstream takes it, or if the fork publishes a feed of its own. The Terminal and source installs
  can use the fork's build.

- Page size (decided in M2f): one page size, the greatest common divisor of the expert sizes; conversation blocks
  take several consecutive pages where one page would waste much of a page. Open: a conversation's hole wastes the
  pages past its last whole block (a 59-page hole holds 11 blocks of 5 pages, 4 pages unused while the hole is
  partly used); filling a whole hole with K/V units would need a unit table per hole instead of per block.
- Both paged engines use one checksummed file per paused conversation: model and settings identity, tokens,
  checkpoints, model state, the layer cut and each card's pages. `SAVE <path>` persists the most recently used
  conversation; `RESTORE <path>` puts it in the paused set for token-prefix reuse, including after a restart.
  Named sessions and automatic SSD pauses share this format. Qwen's running-state metadata now stores one PLE
  history, independent of stage count; one-stage and split saves can be redistributed through the RAM pool.
  Its layout fingerprint refuses the earlier stage-bound files. DeepSeek's metadata was already cut-independent.
  Startup adopts compatible files of dead processes by renaming them under its prefix, after checking checksums,
  fingerprints, context and cut. The SSD pause budget evicts recovered files by the usual rule (priority zero,
  filename order); consumed files and clean shutdown delete adopted files. Live-process files, incompatible or
  damaged files, links and incomplete temporary files stay untouched for manual cleanup. Adoption failures never
  fail startup. RAM pauses and active conversations need an explicit SAVE to survive a process exit.
  Draft-enabled persistence and cross-stage regression runner: `tools/test_paged_persistence.py`. Windows HIP on
  the Strix Halo passed Qwen IQ3_S at one stage, two same-GPU stages and cuts 20/30 with MTP and suffix drafting;
  DeepSeek IQ2_XXS passed cold restore and orphan adoption with DSpark and suffix drafting. Continued tokens matched
  uninterrupted requests; foreign and damaged files stayed untouched. File checks: 44 passed; architecture: 5.
  CUDA, SYCL, Metal and separate-card inference remain untested. Details and pool budgets: DETAILS.md, Session files.
- What the panel shows for an old engine's settings it cannot change live: greyed out, or "applies after a restart"
  with a restart button?


### Suffix drafting in the page-pool engines (2026-11-03)

Both engines reuse `common/spec/WindowPlanner`: source choice, measured costs, MTP keep/stop and lookup chaining.
`Step` shares rows, applies individual and cumulative confidence cuts, schedules cooldown, and suppresses guesses
when the model's declared conversation capacity is exceeded (DeepSeek without DSpark: one; with it: the row budget).
Model adapters run their drafter networks and maintain rings/K/V; they return raw tokens and confidences.
`ConfidenceCut` also backs D7's
`Model::count_confident` wrapper, so the old DeepSeek round and both new engines share the same cuts.
The existing suffix, chain, confidence and window options are reused; there are no new settings.

`--suffix-draft 3` opts in for greedy decoding; `0` leaves lookup off.
`--lookup-chain K` / `--lookup-chain-min M` can extend model
proposals. `--mtp-max-t M` caps model rows, within `--lookahead` and `--rows`.
`STRATA_SPEC_COST=1` uses the same cost stopping callback with lookup on and off. Confidence floors remain
active in the new engines' cost mode. Costs follow logical conversations through RAM/SSD pauses, rather than
leaking between unrelated prompts sharing a GPU slot. Token affinity retains calibration even when GPU state
cannot rewind. Costs are not serialized; a new process rebuilds them.
DeepSeek's suffix-only verification allocates rollback scratch without DSpark weights.

Windows HIP, Strix Halo: commands, full two-round old/new Qwen speed tables (with and without MTP), draft counts,
token arrays, concurrent answers, pause/restore checks, budgets and limitations are recorded in
`%TEMP%/suffix-drafter/planner/warm/REPORT.md`. CUDA was not compiled; SYCL, Metal, separate GPUs and the excluded
GPU/lifecycle tests were not run.

## Item 6: retain quantizations while replacing the engines

The owner's requirement is format parity with strata.exe and strata-dsv41. The inventory and per-format
verification limits are in [DETAILS.md](DETAILS.md#quantization-coverage-of-the-page-pool-engines-4-november-2026).
Canonical Q2_0 uses the existing verify kernels; it no longer fails the adapter's native-pack check.
DeepSeek EXL3 uses nine-piece file reads into the pool's expert layout and the existing CPU/GPU kernels.
Ready-pack dense and embedding paths survive server/launcher translation. No old pack is retired or rewritten.
IQ3_S and DeepSeek IST-DAS 2.00 bpw have local old/new greedy and concurrent identity evidence. Absent formats remain inference-unverified;
this item is not complete until their end-to-end checks and the remaining backend builds pass.

### DSpark dense memory and panel accounting (2026-11-04; item 8 / D7)

The existing live Drafter switch now releases DeepSeek's dense tensors as well as its expert pages.
The shared engine trades the released dense allocation against whole GPU pool chunks through the normal
resize path. It refuses to reload when the resulting pool cannot protect its running conversations.
Startup with `--drafter off` releases the tensors before opening the pools. `--dspark none` has no drafter
and cannot turn one on live. Per-request `spec=0` suppresses guesses without changing the engine setting.

The model keeps recent trunk taps with each conversation's rings, including checkpoint and pause copies.
It rebuilds DSpark's rings on reload and on checkpoint restore. Memory reports and the shared web/app panel
separate `drafter_experts` from `experts`; `drafter` holds the reloadable dense tensors.

Windows HIP, Strix Halo, IQ2_XXS copy, context 8192, GPU/RAM settings 10240/512 MiB,
480 GPU experts, adaptive swaps off, `DS4_COMMIT_CAP_GB=18`: dense memory released/reclaimed
752,205,056 bytes, pool 10,734,796,800 -> 11,271,536,640 -> 10,734,796,800 bytes, in two cycles.
The pool receives 536,739,840 bytes after chunk rounding. Drafter experts occupy 7,219,445,760 bytes
when refilled and none when off. A 16-token greedy continuation matched uninterrupted output after
off/on, a different conversation, pause/resume and a checkpoint rewind. Minimum free commit/physical RAM:
50.37/26.72 GiB. Startup off and a continuation checkpoint created while off also passed, including
pause/resume (minimum free commit/physical RAM 50.92/26.62 GiB). Switches queued during a single GEN
were applied after DONE; its 32-token answer matched uninterrupted output. No in-window toggle was measured.
See DETAILS.md for commands and limits; artifacts: `%TEMP%/dspark-memory`.

Passed: HIP builds of strata-ds41, strata-dsv41 and strata-q38; conversation layout (785 checks),
conversation state (19), memory report (11), shared step, architecture/server memory tests
(39 tests and 8 subtests), Memory panel (18 tests).
Full `npm test`: 1035 passed, 101 skipped, three failures in unchanged paths: native-control's two
engine-path validations and Timeskip's Windows cleanup (`EPERM`). CUDA, real multi-GPU hardware,
UI rendering, the excluded GPU tests and the lifecycle matrix were not run. No test used port 8080.


### D9 follow-up: dense migration and expert execution

DeepSeek now reloads a moved layer's dense allocations on its destination GPU, including verify scratch
and expert address tables; same-GPU stages share addresses and move allocation accounting. The common
paged engine still owns the cut planner, state migration, expert lending and hop safety decision.
Qwen retains its existing dense-range restriction. DeepSeek's private hop stream computes the lent
experts with DeepSeek kernels and returns the CPU lane's missing rows in their original slots.
Auto compares measured link costs with a startup CPU expert calibration. The chunked prompt path
still streams experts rather than hopping. Same-GPU forced hops remain unsafe and refused.
The NVIDIA kit requires migration log evidence, exercises off/auto/on against one card, and retains
its forced-hop queue comparison. Cross-card identity, dense migration and speed await that hardware.

Local validation on the Radeon 8060S (2026-11-04, Windows HIP): the single-stage, fixed split
and cut-10-to-20 runs matched all 96 output tokens. The quarter-pool stress run moved 20 -> 11 -> 20
but diverged at token 71. Settings, limits and the observed GPU expert shortfall are recorded in
[DETAILS.md](DETAILS.md#deepseek-paged-stages-dense-weights-and-expert-hops).
Separate-card migration and hop execution remain for the NVIDIA kit.


### DeepSeek realistic-pool measurement follow-up, 4 November 2026

The 60 GiB GPU / 20 GiB RAM requested-budget study now covers DSpark off/on, one to four conversations
for strata-ds41 and one for strata-dsv41, using the owner's IST-DAS 2.00 bpw pack and configured context.
Available GPU pools fill before READY; final GPU pools are 51.49 / 49.99 GiB (off/on), with old cache
caps matched to their actual expert bytes. Two measured rounds follow residency convergence. Expert misses,
file payload reads, per-conversation rates and greedy comparisons are retained. DSpark does no actual
drafting in the measured concurrent rounds, while retaining its GPU expert allocation.

The owner reports 96–99 °C under load with a quiet fan; per-round clocks were not logged. Sequential
run order, thermal variation and a restarted new-off segment limit performance attribution. Universal
greedy identity remains unmet; some normal/story answers also change in legacy standalone repeats.
No engine source, default or model file was changed. Host diagnostic relinks, shared step and six
architecture tests passed; other backends, multi-card hardware and excluded GPU/lifecycle tests were not run.
See the [report](../bench/results/2026-11-04-ds41-realistic/README.md) for the matrix, exact commands,
watchdog pilot, passed/failed/not-run checks and limits. This does not close cross-card validation.


### Restore points in both paged engines (2026-11-10)

The server asks for the positions that matter in a chat: the start of the last user message (`marks=`), a point
512 tokens before a long last message ends (`tail=`) and the end of the system prompt and its tools (`shared=`, or
`pin=N`). It now sends them on RUN to strata-q38 and strata-ds41 too, found from the rendered prompt by each
template's turn openers (`get_restore_points`). One policy decides which of them a read keeps and which gives way:
upstream's `ds4_marks.hpp`, now `include/strata/common/state/restore_points.hpp`. The paged engine plans the read
with it and stops there; the model keeps the state (`PagedModel::keep_restore_point`).

- strata-ds41: the shared start of `pin=N`, its fourth fixed checkpoint, is now a restore point given up last
  (`Model::mark`, as strata-dsv41). The three others stay. Its points pause and resume with the conversation.
- strata-q38: a point is 117,683,200 bytes, 40 times DeepSeek's, so a read keeps at most
  `kRestorePointBytes` (384 MiB) of them: 3. They join its checkpoint chain, which still holds at most
  `--prompt-cache` (6) with the shared start pinned. `--prompt-cache-every` stays beside them, for edits the
  server cannot foresee.

No new settings. On the Strix Halo (`strata-q38` IQ3_S, `--gpu-mib 53248` so lending evicts no expert,
`--drafter off --adapt-swaps 0`; greedy, 40 tokens), a chat of 3,715 tokens whose last message was edited read
24 tokens instead of 3,718, and a second chat behind the same system prompt read 1,418 of 3,099. The answers
equal those of a fresh read that stops at the same points, in other runs. A read in one piece (points off)
gives other answers, since its chunks differ. GET_MEMORY_USE shows 354 MB of host checkpoints per conversation.
strata-ds41 (IST-DAS 2.00 bpw, `--gpu-mib 61440 --ram-mib 12288 --expert-cache 5000 --dspark none
--adapt-swaps 0`): the edit read 16 tokens instead of 3,410, the second chat 1,224 of 2,804; 13.7 MB of host
checkpoints. Its greedy answers are not stable from run to run here (a fresh read gave three answers in three
runs, and each lend still evicted 12 to 20 experts). The edit's answer from its restore point matched one of
those fresh reads; the second chat's answer was the same in the two runs that went back to the shared start.
