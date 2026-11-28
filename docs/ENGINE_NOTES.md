# Engine and server notes

Settings and measurements written for this fork: the engine options, the API request monitor, the memory panel
and the page-pool engines (`strata-q38`, `strata-ds41`, `strata-q36`). How the code is laid out is in
[ARCHITECTURE.md](ARCHITECTURE.md); the memory manager's steps are in [MEMORY_MANAGER_PLAN.md](MEMORY_MANAGER_PLAN.md).

## Engine options with an older name

Engines built from `src/` accept both names below wherever they support the setting. Use the first name;
the older name is deprecated. Aliases do not add settings to engines that lack them. Installed older engines
may accept only their original names. Launchers keep those spellings for compatibility.

| Option | Older name, still taken |
| --- | --- |
| `--model GGUF` (the model's file, shard 1) | `--native` (the Qwen engines) |
| `--vram-reserve-mib N` | `--reserve-mib` (the DeepSeek engines) |
| `--eos-ids LIST` | `--eos` (the DeepSeek engines) |
| `--batch N` | `--slots` |
| `--max-new N` | `--n` (strata-q36) |
| `--max-context N` | `--ctx` (strata-q36) |
| `--prefill N` | `--chunk` (strata-q36) |

Some options look alike but are different settings. Qwen's `--gpu` hides the other cards from the engine; DeepSeek's
`--gpus` picks cards among all of them. Qwen's `--pool-workers` counts the threads beside the engine's own; DeepSeek's
`--cpu-threads` counts that one too. Qwen3.8's `--prefill` takes `auto` and `auto:N`; Qwen3.6 takes `auto` or a number.
EOS lists use comma-separated token IDs. DeepSeek's `--prefill-chunk 0` means
windows of 16 tokens.

## API request monitor

Off by default, because it keeps prompts and answers in memory. Turn it on with `"api_monitor": true` or
`strata_server/server.py --api-monitor`; otherwise its endpoints answer 404. `/api-monitor` shows the model state, load and unload
controls, active and queued requests, request bodies, output, separate reasoning and response bodies, with load, queue,
first-token and decode timings. `GET /api/requests` returns summaries and `GET /api/requests?id=<id>` one request. The
monitor keeps the newest 100 requests in memory until restart, 262,144 characters per field, with visible truncation flags.
Headers are not recorded. Treat the history as sensitive when Strata is reachable from a network, and set an API key.

A `/v1/chat/completions` request whose client asks for `logprobs` also keeps the numbers of the entries its answer
carried, for up to 32,768 tokens, then sets `logprobs_truncated`. It keeps ids, and logprobs rounded to steps of
1/2048 (about 4 decimals; -32 and below are kept as -32): 112 bytes a token at 20 alternatives. All requests together
keep at most 32 MiB of them; past that the oldest requests' are dropped first, and their tab says so.
`GET /api/requests?id=<id>` says how many tokens it kept (`logprobs_tokens`); `&logprobs=1` adds them as the API's
entries (`logprobs`: `content` and `reasoning_content`). The request's **Probabilities** tab shows the answer with each
token tinted by the probability the model gave it; hover a token, or focus it and move with the arrow keys, to see
that probability and the likeliest tokens at its place. Other requests have no probabilities: the server does no
extra work for them.

Its **Context in VRAM** panel shows the conversations the engine holds, as the engine lists them when **Refresh** is
pressed. The list next to the views names each one and says where it is (VRAM, or RAM or disk for a parked one), its
size and how its first user message begins. A session's size is the tokens it holds of the most it can hold
(`--max-context`), and the memory those tokens take of what the session is allocated; a parked conversation's is its
tokens and its snapshot in RAM or on disk. Pick one, and its tokens are read from the engine then. The
panel opens on the session a request used last.

The words below — session, slot, parked — are the **per-session engines'**
([ARCHITECTURE.md](ARCHITECTURE.md#the-command-loops)); the paged engines (`strata-q38`, `strata-ds41`)
hold conversations in page pools instead, and list them as paused.

- **Session** (or **Solo session** with `"parallel"`, `--batch`): the engine's own session. A request that is alone
  runs there, and with `--batch` every request is read there before it is copied into its slot.
- **Slot N**: a batch slot's session, idle or decoding. Qwen3.6-35B-A3B's engine has no session of its own: a request
  that is alone runs in one of the slots' sessions.
- **Parked conversation N of M**: a conversation the engine set aside in RAM (DeepSeek V4.1: in RAM or in its folder on
  disk) when another one took its session, so that its next request does not read it again. The first one listed is the
  next to be evicted. The list shows its size.

**Plain text** shows the tokens as text. **Colored tokens** gives each token its own background. The colour comes from
the token's id, so the same token always has the same colour. Hover a token to see its position and id. It shows 8,000
tokens at a time, starting with the newest; **Earlier** and **Later** move between pages. The colours come from a fixed
palette of 4,096 per theme, sampled evenly over every colour that keeps a contrast of at least 7:1 with the text (OKLab
lightness 0.68-0.93 in the light theme, 0.25-0.47 in the dark one), so muted colours, which take up less of that space,
come up less often. **Save .txt** saves the text. Tick **with token ids** next to it, and the ids follow the text as
32-bit little-endian numbers: a line gives their count N, and they are the file's last 4 × N bytes.

The engine answers between requests, and with `--batch` also between the windows of slots that decode. While a request
has the engine to itself (without `--batch`, or a request alone on the solo path) the panel waits up to a few seconds,
then shows the list it read last, and the tokens it read last if they are the same conversation's, with the time they
were read. It needs an engine that lists its conversations (its `READY` line ends in `conversations`); with an older
engine the panel says so.

With `--batch`, Qwen3.8-Flash-Next's original engine keeps its own session next to the slots, so `--batch 2` holds
three sessions, each allocated for the whole context. The engine's log gives 2.88 GiB per slot session at a 204,800-token context with
`--kv int8` (on a Radeon 8060S, Strix Halo); its own session is allocated the same way and is not in that figure.

`GET /api/context` returns `{"available", "loaded", "max_context", "busy", "error", "at", "entries": [{"id", "name",
"kind", "slot", "where", "state", "count", "capacity", "bytes", "used", "preview"}]}`: `id` is the engine's (`main`,
`slot<b>`, `parked<serial>`); `kind` is `session` or `parked`; `where` is `vram`, `ram` or `disk`; `state` is `holding`,
`decoding`, `reading` (a prompt read that gave way to a shorter one and goes on later), `empty` or `parked`; `count` and
`capacity` are the tokens held and the most it can hold; `bytes` is its physical size and `used` what its tokens take of
it (for a session, the size a session sized for just those tokens would have; both 0 where the engine does not know
them, as for DeepSeek V4.1's session); `preview` is the start of the first user message (empty when there is none);
`last_used` is when a request last used a slot (Unix time). `at` is when the list was read; `busy` says a request had the engine's turn, so the list is the
one read last. `available` is `false` with `"reason": "engine"` for an engine without the commands. `GET
/api/context?id=<id>` returns that conversation's `tokens` and `pieces` (each token's text; joined, they give the whole
text), 404 when the engine no longer holds it, and 503 while a request has the engine's turn and they were not read
before. When the monitor is on, the main page's Sampling panel and Monitor tab link to `/api-monitor`.

The engine's commands, one line each on its stdin (include/strata/common/state/token_listing.hpp):

- `LIST_CONVERSATIONS [<marker> <n>]` answers `CONVERSATION <id> <place> <state> <tokens> <capacity> <bytes> <used>
  [<preview>]` for each conversation it holds (place `vram`, `ram` or `disk`), then `END_CONVERSATIONS`. With a
  marker (comma-separated token ids) and a count, the preview is the first n tokens after the first place the
  conversation holds the marker; the server passes the chat template's user-turn header (`<|im_start|>user` and a newline,
  or `<｜User｜>` for DeepSeek V4.1). Without them there are no previews.
- `GET_CONTEXT_WINDOWS <id> [<id> ...]` answers `CONTEXT_WINDOW <id> <n> <id,id,...>` for each id, or `CONTEXT_WINDOW
  <id> gone` when the engine no longer holds it, then `END_CONTEXT_WINDOWS`.

Token ids are left out when there are none. Neither command changes anything.

## The memory panel

The Monitor tab's Memory card shows, for each memory (each GPU, the system RAM, the SSD), what the engine holds there:
weights, experts, conversations, paused conversations, the drafter, and what is free. It reads `GET /v1/memory` every
3 seconds and changes nothing (the first step, M0, of [MEMORY_MANAGER_PLAN.md](MEMORY_MANAGER_PLAN.md)).

On a PC whose graphics memory is the system RAM (Strix Halo), the BIOS divides the memory: a share for the GPU, which
the operating system does not count as RAM, and the rest. Once the share is full, the GPU borrows the operating
system's RAM (Windows' "shared GPU memory", Linux's GTT). The panel draws it that way: the GPU's bar is the BIOS share,
and what the GPU borrows is one striped segment, "GPU (borrowed from RAM)", in the RAM's bar. The engine cannot tell
which of its allocations landed in the share, so its uses in the GPU's bar are drawn scaled down to what it holds
there (the legend keeps the engine's own numbers).
The share and the borrowed part come from DXGI and the `GPU Process Memory` performance counters on Windows (what Task
Manager shows as dedicated and shared GPU memory, for the engine's process), and from the amdgpu driver's
`mem_info_vram_*` and `mem_info_gtt_*` files on Linux (the whole device; Linux has no per-process split). Elsewhere
the GPU keeps the size the runtime reports.

`GET /v1/memory` returns `{"available", "loaded", "machine": {"ram_total", "ram_free", "gpus"}, "busy", "error", "at",
"memories": [{"id", "name", "total", "free", "unified", "uses": {<use>: bytes}, "paused_count", "paused_budget",
"other"}], "settings": {<key>: number}}`; the page-pool engines' GPUs also have `kept_free` (the reserve) and
`dedicated_room` (the space left below the measured dedicated-memory limit after all programs' current use) in bytes,
from their `MEMORY` lines' `kept_free=<B>` and `dedicated_room=<B>` (`null` when the line has none).
`dedicated_room` is absent until a growth finds the limit; zero means no room remains.
On Windows, the `ram` memory also has `commit_room`: what Windows can still promise programs (its commit limit, the RAM
plus the page file, less what all programs have committed). Programs often commit memory they have not used yet, so
this can be far below the free RAM. The RAM pool can grow by the smaller of these two values, less up to 1 GiB kept
free and rounded down to whole pool chunks. The Memory panel explains when commit room is the smaller limit.
To get more, close programs or set a larger page file. A page file that Windows manages grows by itself, but only
while its drive has free space, so freeing disk space sometimes helps too.
`machine` is the server's own reading of the RAM and of the GPUs
(the hardware sampler of `GET /metrics`). The engine's `ram` gets the machine's size and free RAM; `other` is what is in
use in a memory beyond the uses the engine names (other programs, buffers the engine does not count). A value the engine
does not know is `null` or left out of `uses`. `busy` and `available` mean what they mean for `GET /api/context`.
For a GPU whose memory is the system RAM, where the division can be read: `total` and `free` are the BIOS share's,
`runtime_total` and `runtime_free` are what the runtime reported, `in_share` is what the engine holds in the share
(on Linux, estimated as the engine's uses, at most what the share holds), `borrowed` is what is held beyond it in RAM
and `borrow_limit` the most that may be, `borrowed_by` says whose: `engine` (Windows) or `device` (every program,
Linux). The RAM's `gpu_borrowed` is the sum of the GPUs' `borrowed`, and its `other` leaves it out. RAM that the engine
page-locks for the GPU to read (strata-q38's RAM pool, the RAM's `mapped`) is counted by Windows as the GPU's shared use
and by Linux in the GTT; it is drawn once, as the engine's RAM, and taken off `borrowed`. On the Strix Halo (a 63.8 GiB
share, Windows 11) strata-q38 at start held 58.3 GiB in the share and Windows counted 22.1 GiB of shared use, of which
20.9 GiB was its page-locked RAM pool.

How much of the BIOS share a program can really use is not a figure Windows reports. It depends on the size of the
pieces it takes and on the other programs. The owner's test program on the Strix Halo (Windows 11, a 63.83 GiB share,
2026-11-21, two rounds each) mapped pieces into one HIP virtual range and stopped at the first spill. It held
54.50 GiB in dedicated memory in pieces of 256 MiB, in 2.8-2.9 s. It held 58.06-58.08 GiB
(3.56 more) in pieces of 256 MiB and then of 16 MiB once one landed in shared memory, in 3.1 s; in pieces of 16 MiB
only it held the same but took 4.7-9.9 s. Using 256 then 8 MiB pieces held 0.23 GiB more, but 8 MiB pieces as a
program's first allocation looked shared when they were not. So the new engines (strata-q38, strata-ds41) find out
each time a GPU pool grows, at start too: they take its 256 MiB chunks one at a time, each first as one piece. When Windows puts that piece in shared
memory, which is slower (with the pool's top there, strata-ds41 read 4K to 32K prompts 21-24% slower), it goes back
and the chunk is made of 16 MiB pieces instead. The growth stops at the first chunk with a piece in shared memory,
and that chunk goes back. After every growth that stays in dedicated memory, whether it stopped at the end or reached
its size just before it, the pool gives back chunks of that growth until the room the model leaves free
(`--vram-reserve-mib`) is free in dedicated memory. The Memory panel says how much fits in dedicated memory and asks
whether to take the rest as shared memory; yes sends the same size with `shared_memory=on`. That growth first fills
the reserve with temporary 16 MiB placeholders (rounded up, or as much as fits), so that its chunks leave that
room alone, and gives the placeholders back when it ends. A size past what the GPU has room for is not refused: the
pool grows as far as it can, and the change says so ("the GPU has room for 58.45 GiB of pool"). A GPU pool's size
does not weigh Windows' commit charge, though on the Strix Halo GPU allocations are charged to it (strata-q38 on
the Strix Halo, 2026-11-22: free commit fell from 71.9 to 1.0 GiB while the engine took 43.8 GiB of dedicated memory,
4.1 GiB of shared memory and a 4 GiB RAM pool); only the RAM pool is capped by free commit.
A GPU's `largest` estimates the pool size that fits in dedicated memory, using the last measured edge (before that,
the share's size), current use and the reserve. The panel shows a larger size in red but still sends it.
`pool_shared` is the part of the pool in shared memory. Prompt buffers are lent below the first shared chunk.
A shrink frees the top chunks first. If another program frees memory between growths, dedicated chunks can follow
shared ones; the pool does not reorder them. `dedicated_full` says
that the last growth stopped where the dedicated memory ended. Linux and discrete GPUs have no such check: their pools
grow until the runtime has no more.

The new engines leave `--vram-reserve-mib` free on each GPU for other programs. The default is 256 MiB, one pool
chunk, less than the old engines' 700 (strata-dsv41's 600, which it calls `--reserve-mib`; the new engines take both
names). The old engines allocated prompt buffers after the expert cache; the new engines lend them from their pools.
The old engine's Windows paging at 323 MiB free on a 20 GB card (#1376) does not establish a safe reserve for the new
engines, and a discrete card under Windows is not measured yet.

Measured on the Strix Halo (strata-q38, IQ3_S, Windows, the GPU also drawing the desktop, the pool grown to the end of
dedicated memory, three greedy 256-token answers per run, 2026-11-22): with reserves of 700, 512, 256 and 128 MiB
(2-3 alternating runs each), decode was 37.8-38.7 tok/s at every reserve, and none of the engine's memory moved to
shared memory (its shared use stayed at 4.11 GiB). Each 256 MiB less reserve gave the pool one more chunk (56,031 MiB
at 700 and 512, 56,286 at 256, 56,541 at 128). During the first answer the engine's own dedicated use grew by about
95 MiB outside the pools, at every reserve, so at 128 MiB the free dedicated memory fell to 102 MiB. The desktop's
responsiveness at so little free memory was not measured, which is why the default is 256 and not 128.

At start, strata-q38's pools also leave room for what it allocates after them: 256 MiB
a card (352 on a later card of a layer split), what the old engine measured after its expert cache (#4474), and the
drafter's head. The Memory panel changes the reserve while the engine runs ("Reserve free VRAM",
`vram_reserve_mib`). A change counts from the next growth: in the largest size the panel offers, and in what a growth
gives back or holds to keep it free in dedicated memory. Saved as the default, it counts from the next start. It does
not resize a pool by itself. The GPU's bar shows the reserve as "Kept free for other programs", at the start of the free
memory (no more than is free and usable). After a growth found where the dedicated memory ends, the free memory past
that end shows as "Not usable as dedicated memory", at the end of the free memory. The engine finds the end with D3DKMT
statistics and the bar's free memory comes from the performance counters, which count other programs about 0.58 GiB
higher on the Strix Halo (2026-11-22, strata-q38), so the engine says how much dedicated memory is still usable
(`dedicated_room`) and the panel draws the rest of the free memory as unusable: 0.58 GiB there, where 63.23 of the
63.83 GiB share was the most ever in use.

The engine's command (include/strata/common/serve/memory_report.hpp; the engine's `READY` line says `memory`):

- `GET_MEMORY_USE` answers `MEMORY <id> [total=<B>] [free=<B>] [unified=1] [<use>=<B> ...] [paused_count=<n>]
  [paused_budget=<B>] [name=<device name>]` for each memory, then `SETTINGS [context=<tokens>] [concurrent=<n>]
  [lookahead=<n>]` and `END_MEMORY`. `id` is `gpu<N>`, `ram` or `ssd`; sizes are in bytes. The uses are `weights`,
  `experts`, `conversations` (the sessions of the conversations the engine runs, each allocated for its whole context),
  `paused` (parked conversations, from `LIST_CONVERSATIONS`), `drafter`, `scratch` and `engram`.
- `CHANGE_SETTINGS <name>=<value> ...` (strata-q38, whose `READY` says `settings`) changes settings while the engine
  runs and answers as `GET_MEMORY_USE` does, or `ERR <why>` with nothing changed. When a memory had less to give than
  the engine expected, the answer has a `SHORTFALL <what>` line before `END_MEMORY` and shows the sizes reached; the
  server's `POST /v1/memory` returns it as `"shortfall"`, and the panel says it. strata-q38's `ram` line has
  `mapped=<B>`: its page-locked RAM pool, which the GPU reads in place.
- The page-pool engines (strata-q38 and strata-ds41) also send `EXPERT_FILL loaded=<0|1> pending=<n>` before
  `SETTINGS`. The server's `GET /v1/memory` returns it as `"expert_fill": {"loaded", "pending"}`.

How the page-pool engines start: they say `READY` as soon as the dense weights are loaded, so the server takes
requests at once. Their expert pools start empty. A start-up fill loads experts between and during requests, in the
order of the expert profile. The GPUs' pools come first. Then the RAM's pool takes the experts no GPU holds. An
expert that is in no pool yet is read from the SSD. The fill takes only free pages, and it ends for good when the
pools are full or hold every expert. There is no setting for this. `loaded` turns to 1 when the start-up fill ends,
and stays 1. `pending` counts the experts the fill still wants. Strata's speed tools wait for `loaded` before they
time anything (at most 30 minutes); an engine without `EXPERT_FILL` counts as loaded.

After the start-up fill, the profile is only the first guess of how popular each expert is. Only the adaptive tier
moves experts onto a GPU, by how often the conversations route to them. On Qwen, prompts count too: an expert that
every token of a prompt chunk routes to counts like 5 routings while decoding. DeepSeek counts only decoding. The tier
puts an expert into free pages while a pool has some. Only a full pool trades a little-used expert for one used more
(by at least `--adapt-min-gain` routings). The tier runs after every decode step while a card's pool has more than
1 GiB free beyond the room kept for `--tokens-min`, and after every `--adapt-every` steps otherwise. It moves at most
`--adapt-swaps` experts each time. The pages a conversation gives up, a grown pool and a moved layer cut are filled
this way too. The tier moves an expert only onto the card that runs its layer, so after the start-up fill no card
loads experts of another card's layers (`--remote-experts`).

Measured on the Strix Halo (Windows, 2026-11-09), old start against new:

| | Qwen3.8 IQ3_S, 48 GiB GPU pool, 5 GiB RAM pool | DeepSeek V4.1 2.00 bpw, 60 GiB GPU, 20 GiB RAM |
|---|---|---|
| `READY`, old | 38.4 s, everything loaded | 57.7 s, everything loaded |
| `READY`, new | 9.4-10.1 s | 13.9-18.1 s |
| Fully loaded, new, no requests meanwhile | 41.0-43.6 s | 70.1-71.1 s |
| Fully loaded, new, three requests meanwhile | 75.4-75.5 s | 111.0 s |
| Answering while it fills (160 tokens each) | 7.3, 14.6, 17.1 tok/s | 4.3, 5.7, 11.0 tok/s |
| Answering once loaded, old / new | 38-59 / 38-59 tok/s | 9.0-11.6 / 7.6-13.2 tok/s |
| A 14,192-token prompt once loaded, old / new | 845 / 847-849 tok/s | not measured |

Once loaded, Qwen's greedy answers matched the old start's when no request ran during the fill. With requests during
the fill, and on DeepSeek, the answers parted after 15 to 19 tokens. The likely cause, not confirmed, is a slightly
different set of experts on the GPU. With requests during the fill, the experts they read from the SSD stay in the
RAM's pool, so it held copies of GPU experts afterwards: 2,489 of its 2,489 on Qwen, 1,297 of 2,424 on DeepSeek.
Without requests it held none. DeepSeek's fill ends about 12 s later than its old start's `READY`.

What each engine says:

| | Qwen3.8 | Qwen3.6 | DeepSeek V4.1 |
|---|---|---|---|
| GPU | the weights (the pack's arena on the first card, the native projections and head), the expert cache, the sessions, the MTP drafter | the weights, the routed experts, the MTP block's experts (drafter), the sessions | the weights, the experts, the DSpark blocks' experts (drafter), the caches (conversations), the scratch |
| RAM | the expert arena, the K/V streamed to RAM, the parked conversations and their budget | the experts of the hybrid mode | the experts copied into RAM and the RAM store; the CPU lane's experts when they are not read from the files; the parked conversations |
| SSD | (nothing) | (nothing) | the CPU lane's experts read from the model's files, the Engram table, the parked conversations and the slot folder's cap |
| Settings | context, conversations at once (`--batch`), lookahead (the verify window less one: the `spec` of its `INFO` line - 1) | context, conversations at once, lookahead (`--mtp`) | context, lookahead (`--dspark-block`, or 0 without DSpark) |

## Rows per step, lookahead and conversations at once (strata-q38)

strata-q38 decodes each step as one pass of up to 8 rows, 16 in a HIP build ("rows per step"). Each conversation
that decodes gets a row for its next token; the rows left over go to the drafter's guesses, shared evenly, at most
"lookahead" per conversation ([MEMORY_MANAGER_PLAN.md](MEMORY_MANAGER_PLAN.md#how-rows-are-shared-the-owners-rule)).
Both, and the conversations at once, change while the engine runs (the Memory panel, `CHANGE_SETTINGS`).

Measured on the Strix Halo (Radeon 8060S) on Windows 11, Qwen3.8 IQ3_S, 204,800 context, int8 K/V, the MTP drafter
(`--spec-min-p 0.5`), temperature 0, 160 tokens per conversation, 2026-11-01. Three prompts: a story, a Python module
and a travel question ("chat"). One engine start per memory case, the settings changed live between runs, three rounds
with the order reversed every other round, medians (`tools/measure_rows.py`). The memory cases: **GPU**, a 48 GiB GPU
pool, every expert on the GPU; **RAM**, a 16 GiB GPU pool and a 32 GiB RAM pool, two thirds of the experts read by the
GPU from RAM; **SSD**, a 16 GiB GPU pool and a 4 GiB RAM pool, about 26 GiB of experts read from the SSD.

One conversation, tokens/s* for story / code / chat. "1x4" is one conversation and 4 rows, so 3 guesses a step:

| Rows (lookahead) | GPU | RAM | SSD |
|---|---|---|---|
| 1 (no guesses) | 21.2 / 20.3 / 21.2 | 13.4 / 13.0 / 13.0 | 11.7 / 8.7 / 8.8 |
| 3 (2) | 31.3 / 39.3 / 30.6 | 20.5 / 22.5 / 20.1 | 11.7 / 7.7 / 11.5 |
| 4 (3) | 32.4 / 39.3 / 30.9 | 20.2 / 21.9 / 20.6 | 14.1 / 7.2 / 11.6 |
| 8 (7) | 29.3 / 40.0 / 29.1 | 18.8 / 24.3 / 18.8 | 13.6 / 6.5 / 11.6 |

Of the guesses, 70-72% were kept with 2 a step, 62-64% with 3 and 50-51% with 7.

Several conversations at once, tokens/s* of all together (each: story, code, chat):

| Conversations x rows | GPU | RAM | SSD |
|---|---|---|---|
| 2 x 2 (no guesses) | 33.1 (16.6, 16.5) | 20.4 (10.2, 10.1) | 12.7 (6.4, 6.3) |
| 2 x 6 | 48.3 (20.0, 28.3) | 31.9 (13.8, 18.2) | 15.7 (6.7, 9.1) |
| 3 x 3 (no guesses) | 40.8 (13.7, 13.6, 13.6) | 24.9 (8.3, 8.3, 8.3) | 13.5 (4.5, 4.5, 4.5) |
| 3 x 8 | 52.0 (14.9, 21.0, 16.2) | 32.9 (9.5, 13.1, 9.9) | 14.2 (4.1, 5.4, 4.2) |

**Guesses touch fewer experts than other conversations' tokens.** A row of one token routes to 480 experts (48 layers
of 10). A step of three conversations' tokens routed to 462 different experts a row, a step of one token and its
guesses to 353 (2 guesses), 330 (3) and 295 (7): a guess shares about a quarter to two fifths of its experts with the
rows before it, as the same text routes alike. Three conversations with 8 rows (6.3 rows a step, guesses among them)
routed to 337 a row. The counts were the same in the three memory cases (they are the CPU's view of the routing).
Where experts are read from RAM or the SSD, a guess so costs fewer reads than another conversation's token, but it
is also less often kept.

What follows:

- Guesses pay most on one conversation: 1.5 to 1.9 times the speed on the GPU and in RAM with 2 or 3 a step. On
  the SSD they helped the story (+21% with 3) and the chat (+32%) and slowed the code (-17%).
- 3 guesses a step were as fast as 2 and faster than 7 for prose; 7 helped only the code (+2% on the GPU, +11% in
  RAM).
- With several conversations, 8 rows beat one row each: three conversations +27% on the GPU, +32% in RAM, +5% on the
  SSD; two +46%, +56%, +24%. Each conversation is slower than alone (story 14.9 against 32.4 tok/s on the GPU), the
  total higher (52.0 against 32.4).

The defaults chosen from this: rows per step 8; lookahead 3 when the engine has the drafter (`--mtp`; before, it was 0
unless `--lookahead` was given, so no guesses), which gives one conversation the 4 rows that measured best and several
conversations the 8; conversations at once, initially `--batch` (or 1 without it).
For `strata-q38` and `strata-ds41`, `--concurrent` overrides that initial value. The server's `--parallel` sets
the initial `--batch` value too. The live memory panel can raise it without restarting, up to the rows per step
(at most 8) and the model's own limit. These engines report that
capacity as `slots`; the server can admit that many requests, and the live setting decides how many decode in a step.

Two more defaults were checked and kept: the
conversation token minimum (8,192 tokens, about 0.24 GiB of the pool) made no measurable difference to a new 4,000-token
prompt with three idle conversations in a 16 GiB GPU pool (prompt 8.5 s against 8.7 s without it and 8.7 s with 32,768,
four alternating runs, medians); and copying an evicted expert back to RAM (`--copy-back auto`, which copies here: the
copy measured 35-41 GB/s against 2.4-2.5 GB/s for an SSD read) took at most 40 ms in the three
rounds of any configuration above, as the RAM pool seldom had room for a colder expert. The NVIDIA test PC's numbers are
to come: `tools/nvidia_session/README.md` says how to repeat the measurement there.

* on Windows (users have seen Strix Halo much faster on Linux)

---

## Qwen page-pool benchmark budgets

The Qwen timings attached to `4c047a0f` (suffix drafting), `dce39bcf` (pictures) and `0b8b86ab`
(batch groups) used restricted memory. They do not measure performance at the owner's normal settings and do
not establish that the replacement engine matches or beats the original engine at those settings.

The saved source-setup config inspected on 3 November 2026, `strata-iq3_s.json`, selects Windows HIP,
native IQ3_S from `Strata-data`, `--expert-cache auto --resident-experts --prefill auto --spec 4
--spec-min-p 0.5 --mtp ../Strata-data/mtp/rt --max-context 204800 --kv int8 --vision
--vram-reserve-mib 700`, with server `parallel: 2`. Its existing `strata-iq3_s.log` records 24,576 GPU
expert slots using 46.84 GiB, 2.69 GiB of resident RAM experts and no file blob reads in the reported requests.
These are historical configuration and residency records, not a new throughput measurement or proof of stable
residency for a different workload. The page-pool GPU budget must also cover conversation pages; it cannot be
set to the old expert-cache size alone. Unified memory must be counted once.

For comparison, the suffix report at `%TEMP%/suffix-drafter/planner/warm/REPORT.md` used Strix Halo,
Windows HIP, IQ3_S, 8,192 context, greedy decoding and `STRATA_SPEC_COST=1`. The new engine used
`--gpu-mib 14336 --ram-mib 2048 --lookahead 5 --remote-experts off`; the original used
`--mmap-experts --resident-budget-gib 2 --spec 6`. Both requested 6,000 GPU experts, disabled adaptive swaps
and capped MTP with `--mtp-max-t 4`. Actual GPU residency differed. Two fixed warm-up rounds preceded two
measured rounds of 64 tokens per answer; the report does not establish a residency-stability stopping condition.
Rates below pool the two measured decode times and exclude prefill. They are retained only as restricted-memory
results from that report; they were not rerun during this inspection.

| Prompt | MTP | Original suffix off / on, tok/s | Page-pool suffix off / on, tok/s |
|---|---|---:|---:|
| Repetitive list | Off | 3.22 / 4.46 | 4.46 / 4.94 |
| Seasons | Off | 3.93 / 4.03 | 6.91 / 6.68 |
| Repetitive list | On | 4.95 / 4.86 | 5.50 / 5.50 |
| Seasons | On | 5.71 / 5.25 | 8.81 / 9.02 |

A full-residency rerun completed on 3 November 2026, on the same Windows HIP Strix Halo and native IQ3_S,
with the saved 204,800 context, int8 K/V and vision settings. The original engine used
`--expert-cache auto --resident-experts --resident-budget-gib 4 --vram-reserve-mib 700`.
The replacement requested `--gpu-mib 61440 --ram-mib 512 --expert-cache 24576 --vram-reserve-mib 4300`,
with `STRATA_COMMIT_RESERVE_MIB=3072`. Its actual GPU pools were 52.50-53.75 GiB after automatic caps;
**both engines held all 24,576 experts, 46.84 GiB, on the GPU**. The old RAM complement used 2.69 GiB.
The owner explicitly allowed a 3 GiB commit floor for this task; the physical floor stayed at 4 GiB.
Both were checked before every start and every 250 ms by a PID-scoped watchdog. The main matrix had no trip.
No Docker VM, browser rendering or encoding ran with inference. Tests used private engine pipes, with no
listener or use of the owner's server. Exploratory watchdog stops, including an old mapped-startup physical
floor trip, are recorded separately in the report.

GPU residency stayed unchanged through idle checks and two complete warm-up rounds before timing.
Each standalone case pools two measured 256-token answers. Three-conversation rates pool two measured
rounds after a separate batch warm-up, counting only tokens after the last admission and excluding prefill.
Normal is a detailed seasons explanation; repeat is the five-word repeated list from the historical table.
The concurrent workload adds a lighthouse story. These are decode tokens/s:

| MTP | Suffix | Old normal | New normal | Old repeat | New repeat | New three, aggregate |
|---|---|---:|---:|---:|---:|---:|
| Off | Off | 25.20 | 23.98 | 25.05 | 24.28 | 48.38 |
| Off | On | 25.29 | 24.56 | 75.91 | 72.49 | 49.47 |
| On | Off | 43.21 | 37.08 | 60.97 | 58.00 | 54.48 |
| On | On | 39.47 | 37.10 | 60.29 | 67.37 | 56.05 |

These realistic numbers replace the restricted-memory table as evidence of this setup's throughput.
They do not establish a win in every case. Normal MTP accepts 158/239 drafts in the old engine against
144/243 in the new one, requiring 97 against 111 decode windows. Old/new greedy answers differ on normal
and story prompts; the repeated list matches. Concurrent/standalone identity also fails on normal and story.
The old engine's own MTP-on/off answers differ too. These failures are separate from the measured rates;
their cause is not established by the timing data. An owned-prefill control did not remove the old/new mismatch.

In these runs, shared startup filled GPU expert pools before `READY`, instead of relying on later idle refill. (The
engines now fill them after `READY`: [The memory panel](#the-memory-panel).)
Full-residency startup took old 39.41-40.28 s and new 39.29-45.65 s, including 16.55-19.42 s of final pool fill.
The saved old resident mode maps files, stages one expert at a time into the GPU cache, then trims mapped
working-set pages and pins the RAM complement. Its RAM peak is not a full private committed expert arena.
The replacement also stages host data into GPU pages. Once both are fully resident, neither reads expert
files during decode, so startup placement cannot explain the remaining steady-state gap.

The final matrix's minimum free commit was 3.538 GiB; minimum free physical RAM was 18.356 GiB.
Observed system commit increases were 56.657-62.471 GiB for the original and 63.204-64.854 GiB for the
replacement. These include background application changes; they are not an exact per-process attribution.
The commit limit stayed at 127.295 GiB. Do not add GPU use to system charge on unified memory.
[The full report](../bench/results/2026-11-03-q38-realistic/README.md) gives exact flags, round times,
per-case memory floors, load-path source locations, profiling, identity failures and validation limits.
The initial pass stopped at a preflight with 64.479 GiB free commit
against a revised 68 GiB requirement, despite 23.879 GiB free physical RAM. The retained exploratory logs
include a 0.004 GiB commit minimum and a 3.960 GiB physical minimum in separate killed attempts; they are
not successful runs. The main matrix remains separate. No engine is left running. The system-managed
page file was 65,174 MiB and did not grow in the logs. The report includes every retained run's minimums.

## Short prompts in the full-residency follow-up

The story's old/new token-2 difference was caused by different prompt paths. The old server's default
`--short-read 64` reads short segments through verify windows, with a split at the final chat-turn
boundary. The paged adapter always used batched prefill, whose FP16/BF16 projection inputs round
differently from the verifier's Q8_1 inputs. Owned prompt scratch did not change that choice.
At story position 34, the old engine preferred ` storm` over ` wind`; the new engine reversed them.
Their maximum logit difference was 0.8460803. This was a missing prompt-selection policy in the adapter.
No position, expert-layout, K/V-format or HC-weight change was needed.

The adapter now honors short reads with the original window width and final turn boundary, and feeds
the drafter from verified residuals using its own prefill pass. All 41 captured story prompt/decode rows
then matched exactly across 47 completed-layer HC means and all 248,320 logits per row. The verifier's
existing layer-tap exporter also supports batch rows and their actual positions. Raw dumps and their
hashes are retained in the [full report](../bench/results/2026-11-03-q38-realistic/README.md).

Single-conversation batch windows reuse the ordinary QSA attention, grouped K/V and indexer appends,
and combined GDN convolution/gate kernels. Per-slot recurrence writes quantized outputs directly.
Both commit paths share the existing
all-layer QSA indexer helper, including the conversation's rotary table, with the existing per-layer
fallbacks and overrides. These changes do not promise identical answers across concurrent row layouts.

The fully resident paged path also keeps shared projections on the main stream for the measured
single-gfx1151 GGUF IQ3_S tier. Two warm and two measured 256-token seasons controls improved to
32.62 tok/s without MTP and 52.04 with it; answers stayed identical. The preceding default-path pair
was 24.83 and 44.34. The model metadata selects the tier. Explicit `STRATA_SH_STREAM` settings win;
other devices, tiers, partial-residency graphs, split runs and legacy callers retain their policy.
These controls and the final comparison are retained in the linked report.

On one GPU, the next MTP draft can overlap the verified state commit. Commands, actual page growth,
interleaved prompt chunks, request completion and engine cleanup drain pending state first. Advancing
within already published physical units needs no device-table write. Verifier row coordinates have
their own buffer, so the drafter keeps its full cell table between steps. Short prompt reads preserve any
pending drafter residuals from other conversations. Snapshot and coordinate copies run on the draft
stream; prompt admission waits before reusing the source scratch. A guarded text/image control keeps
both 256-token outputs identical after live memory changes and cancellation of three requests.
Legacy batch callers stay synchronous; `STRATA_COMMIT_SYNC=1` keeps the explicit wait available.

The final 4 November 2026 two-round Windows HIP comparison on Strix Halo, all 24,576 experts
resident, measured seasons at 26.50/31.85 tok/s old/new without drafts and
44.34/51.14 with MTP. Story was 27.04/31.15 without drafts and
36.46/42.37 with MTP. New was faster in all 12 standalone prompt/mode comparisons, by 11.3% to 40.2%.
New three-conversation aggregate rates were 60.09 without drafts and
65.65 with MTP. The full suffix matrix, exact answers, memory floors and
passed/failed/not-run checks are in the [paired report](../bench/results/2026-11-03-q38-realistic/README.md).
Suffix-history and concurrent-layout differences remain; these measurements do not promise universal identity.

## Pictures in the page-pool engines

`strata-q38` reads the same `GENI` embedding files as `strata`, using the same prompt row and M-RoPE builder.
The server keeps the configured CPU or GPU encoder. Picture and text conversations decode together in one step.
Each row uses its conversation's rotary coordinates; pooled indexer keys keep a separate table per conversation.
`strata-ds41` uses DeepSeek's existing RGB reader, span cache and built-in vision tower. A model without a tower
refuses pictures. Existing vision settings still apply; there are no new picture settings.

Measured on 3 November 2026, Windows HIP, Strix Halo (gfx1151), Qwen3.8 IQ3_S, 8K context, vision on,
4,096 MiB GPU and 2,048 MiB RAM pools, 400 GPU experts, adaptation off, no drafting, temperature zero:
one conversation gave 2.55 before / 2.65 after tokens/s; three gave 2.01 / 2.15 aggregate tokens/s.
These restricted-memory checks do not describe the owner's normal setup (see the budget correction above).
This is one round of 64-token answers: a picture with counting, then Python and story prompts alongside it.
Timing starts after the last admission and excludes prefill and earlier tokens (one: 63 tokens in each run;
three: 187 before, 183 after). Small pools on this setup do not make three conversations faster than one.
The before binary is 4c047a0f. All three answers match standalone and that binary exactly.

At realistic budgets on the same Windows HIP Strix Halo and native IQ3_S, the 204,800-context,
full-residency setup above gave **25.06 old / 24.76 new tok/s** for a picture followed by counting,
and **49.91 aggregate tok/s** for the new engine with that picture, seasons and repetition together.
Both draft sources were off. Two measured 256-token rounds followed warm-up; the mixed batch also had
its own warm-up and counts 762 tokens after the last admission per measured round. This uses an existing
16-row embedding file, excluding image encoding time. The picture matches old/new and concurrent/standalone
in both rounds; the mixed seasons answer differs from standalone at token 56 and repetition matches.
All 24,576 experts were on the GPU. Minimum free commit/physical RAM was old 8.068/23.723 GiB and
new 5.785/26.262 GiB. These are realistic-budget picture checks; the earlier 4 GiB-pool numbers above
remain historical restricted-memory evidence, not this setup's speed.

Both engines reuse a kept conversation only up to the first picture that differs, in its place or its content;
from there the prompt is read again from the latest checkpoint before that picture, as for changed text. Pauses and
saved sessions keep each picture's place and hash; Qwen also keeps rotary coordinates.
The DeepSeek tower's allocated memory is counted as scratch. An external encoder is included in system use
(the panel's Other), rather than the language engine's weights.

## The page-pool engines

Qwen3.8 runs on `strata-q38` and DeepSeek on `strata-ds41`. The engines they replaced (`strata` and `strata-dsv41`)
are gone from Strataq.

These engines have live memory settings in the Memory panel, several conversations at once, pausing conversations
in RAM or on disk, saved sessions, and layer splitting. DeepSeek's layer split is still under test; separate-card
validation remains open in D9. `--tp` is unavailable; DeepSeek uses the shared group scheduler described below.
The server keeps context, GPU selection and `parallel` (starting conversations at once). It logs options it leaves
out. Use `--set gpu-mib=18432 --set ram-mib=8192` to set the pools; `tokens-min` and `tokens-max` bound the token
pages kept in the pools. These are engine options in the config's `args`, also available through `--set`.

`--batch-groups G` moves groups of conversations through separate GPU stages, so that different cards work on
different groups at once. Qwen3.6 (`strata-q36`) does not implement this option.

`strata-q38` runs pipeline groups through the shared paged step. `--batch-groups 1` keeps its synchronous
path, which remains the default when the option is omitted. `auto` selects the largest divisor of the live
Conversations at once value up to the number of stages. An explicit number must divide that live value, otherwise
it runs one group. `--batch` sets only the initial concurrency. Rows per step and lookahead apply within each group;
MTP, suffix lookup and pictures use the
same acceptance and row-sharing rules as the synchronous path.

Groups have separate hand-off rows, picture coordinates, sampling histories, deferred commit inputs and MTP
residuals. Graphs, sampler space and conversation pages are prepared before any group launches. The shared
scheduler serves each occupied stage without blocking on another stage. Snapshot copies are polled too. It drains
the whole wave before sampling the saved logits, before accepted prefixes are committed and before
it returns to the serve loop. Pauses, saved restores, pool changes and cut moves therefore happen with no groups
in flight. Deferred commit buffers and MTP residuals count as model scratch outside the page pools. Several
stages can work on different groups within one wave; this version drains between waves. Group waves do not
update the drafter's learned single-conversation timing, since their elapsed time covers several groups.

`strata-ds41` uses the same stage launch/poll interface and scheduler. Its independent rows use the same
conversation bindings as the synchronous batch pass: rings, compressors, page tables, Engram history and absolute
positions. Each group retains its embeddings, Engram rows, attention plans, hand-off rows, logits and DSpark taps.
Compressed-row copies keep separate host buffers for each group and row. The CPU expert mailbox assigns sequences
in stage launch order. Layer ownership is read at launch, so a cut move applies to the next drained wave.
The shared step applies the drafting limit to all conversations in a wave, including singleton groups.
With DSpark loaded, eligible conversations share one batched drafter forward and one trunk verification pass.
Each row reads its conversation's own rings and pages. The existing row budget limits the guesses per conversation;
drafting phases out as conversations consume that budget. An engine opened with DSpark support uses synchronous
steps, including while drafting is switched off, because rollback scratch is shared across stage groups.
`--dspark none` keeps the existing group path. There is no new setting or automatic drafting mode.
The omitted option and `1` still use the synchronous pass. Real two-card identity and speed require measurement;
implementation alone does not establish either.

Local DeepSeek validation (4 November 2026, Ryzen AI MAX+ 395 / Radeon 8060S, HIP gfx1151, one physical GPU,
2.00 pack, two stages at cut 20, 1K context, 4 GiB GPU pools per stage and 4 GiB RAM pool, eight output tokens):
all 18 rounds at two, three and four conversations match standalone and groups 1/2/auto. Mean aggregate rates
for auto were 2.255, 1.981 and 1.814 tokens/s respectively; these short timings are noisy. A longer-prompt run
crossed the first compressor boundary and matched standalone through RAM/disk restores, pool resizing,
concurrency 4 -> 2 -> 4 and a cut move 10 -> 20. A separate DSpark test used a 7.5 GiB last-stage pool and
confidence 0 to force draft verification; concurrency 1 -> 2 -> 1 matched standalone in all six rounds.
A one-group comparison with rebuilt original DeepSeek components showed no consistent regression across
opposite-order short runs. Setup, samples and limits are in
[the local report](../bench/results/ds41-groups-20261104/README.md). None of these runs measures two-card overlap.

The NVIDIA session kit adds `memory-groups-q38-SB` / `-BS` and `memory-groups-old-SB` / `-BS`. Each compares
`--batch-groups 1` with `auto` on two actual cards, using four conversations, a cut at layer 24, the fixed
`--expert-cache 400` budget and no drafting. Four 96-token standalone answers must equal two rounds of concurrent answers. Decode
speed counts tokens emitted after the final admission, excluding prompt reads and earlier tokens. The new
engine must also report two stages occupied together. These checks have not been run on real NVIDIA cards here.

Validation limits (3 November 2026): on Windows HIP, Ryzen AI Max+ 395 / Radeon 8060S, native IQ3_S, two stages on
GPU 0 (`--layer-split 24 --split-device 0`), the two 64-token code/count standalone answers and their concurrent
32-token prefixes matched with `1` and `auto`. Standalone answers matched the saved pre-change binary. With 2 GiB
per stage, 2 GiB RAM, 400 requested experts, no adaptation or PCIe expert work, rows 8 and drafting off, one final
round measured 2.216 aggregate tokens/s with `1`, 2.220 with `auto`. Standalone code/count rates were 3.162/2.864
and 3.000/2.468 respectively. These are same-GPU scheduling checks, not evidence of a two-card gain. In two earlier
rounds, omitting the option measured median 3.080/2.509 tokens/s against the saved binary's 3.043/2.532.
The original `strata.exe`, with its own 2 GiB resident budget, measured 1.985/1.879 median standalone tokens/s
(two rounds). It refuses concurrent batching when both split stages use GPU 0, so its group speed comparison
needs real cards. The resolved cache sizes differ, so this is not an equal-cache old/new speed comparison.
These restricted-memory checks do not establish performance at the owner's normal expert residency.

Passed in the final HIP run: two and four concurrent 16-token answers matched standalone with MTP enabled;
the four included a picture, and two RAM-pool changes completed while decoding. A picture's continuation matched
after an SSD pause/resume and after restoring its saved file in a fresh engine. Seeded sampling with repetition,
frequency and presence penalties also matched standalone for two concurrent 16-token requests with `1` and `2`.
Separate suffix-only runs exercised accepted lookup drafts with two groups, including a picture, without a live
concurrency change. All inference starts checked free commit and physical RAM and had a PID-scoped 16/4 GiB
watchdog; no memory watchdog tripped. The user's server was not used.

Build checks passed for `strata-q38`, `strata-ds41`, `strata` and `step_test` on Windows HIP and Linux CUDA 13
(sm_86, compile only). `step_test` passed on both; 57 Python tests and 5 subtests passed for the launcher, NVIDIA
kit and architecture checks. CUDA inference, Metal/SYCL builds and a grouped live cut move were not run here.

Failed checks: exact answer identity is not established for every request sequence. A three-conversation picture
answer after earlier rounds differs from standalone in both the saved pre-change binary and `1`, with identical tokens in those
two builds. A suffix-only picture request with sparse slots and live concurrency 2 -> 1 -> 2 differs with groups:
at output token 38 it repeats a color instead of ending. The corresponding serial control passes, and groups
without that live change pass. The restored logits themselves select the emitted token (19.817 for `white`,
19.693 for the end token); the cause is unresolved. One grouped story run also differed after three tokens while
a repeat passed. The final MTP run's three-conversation picture check also failed. These failures prevent claiming
universal concurrent/standalone identity. Keep the option opt-in while investigating them; the NVIDIA kit records
identity failures separately from the measured speeds.

A separate full-residency old-engine control on 3 November used `--batch 3`, 8K context,
`--resident-budget-gib 4`, no drafting, the seasons/list/story prompts above and two 256-token batch rounds.
The old story differs from standalone at output token 186 in both rounds; seasons and repetition match.
This confirms batch dependence in the original engine too, but does not establish the cause of the new
engine's token-2 story divergence or reproduce the exact sparse-slot/live-group failures above.
At 204,800 context that old three-slot control hit the 3 GiB commit watchdog during startup; the 8K control
completed with all experts on the GPU. The main one-conversation comparison retains 204,800 context.

Both page-pool engines can use greedy prompt lookup with `--suffix-draft 3` (`0`, the default, turns it off).
They use the original Qwen engine's shared `WindowPlanner`, including its measured row costs, lookup acceptance
rates, and `--lookup-chain K` / `--lookup-chain-min M` extension after model drafts. `--mtp-max-t M` caps the
model drafter's window, while `--lookahead` and `--rows` bound all guesses and the step's total work.
`STRATA_SPEC_COST=1` also selects the original planner's probability-based MTP stopping rule; it applies with
suffix drafting both on and off. The model supplies its confidences, and solo steps teach the planner their full
draft, verification and commit time. Joint steps keep acceptance tallies but do not teach solo row costs.
The shared step also applies the configured individual and cumulative confidence floors before the cost choice,
including with cost mode enabled. It reads the existing `dspark_conf`, `dspark_cum` / `spec_cum`, and `spec`
request keys; a negative floor uses the engine's default. Model adapters run their networks, bind state, and
return raw proposals and confidences. DeepSeek's plain step after a rejected model proposal is scheduled here too.
`--lookahead 0` disables all drafting. `--drafter off` switches off only the model drafter.
The Memory panel's Drafter list combines the model drafter, the suffix drafter (`--suffix-draft 3`: a step takes its
guesses instead of the model's where the planner expects them to pay) and lookup after the model's guesses
(`--lookup-chain 1`), six choices in all. It sends only the settings a choice changes, so a `--suffix-draft` or
`--lookup-chain` the engine started with stays unless the choice needs another value.

**Draft acceptance** (`--draft-rule`, the Memory panel's choice of the same name, `CHANGE_SETTINGS draft_rule=`)
decides which guesses a sampled request keeps. A greedy request always compares its guesses with the most likely
token. `exact` keeps a guess only when it is the token the model draws without guesses, so the answers are those of
decoding without drafts. `speculative` keeps a guess with probability min(1, p/q) and otherwise draws from what is
left of p (Leviathan et al. 2023, Chen et al. 2023): each token is still distributed as the model's own draw, so the
answers are as good, though their words can differ from `exact`'s. Three rules keep more guesses and change the
answers: `typical` (Medusa's typical acceptance, ε = 0.09, δ = 0.3) keeps a guess the model finds likely enough,
`top-<N>` (N from 1 to 99; the panel offers 75, 90 and 99) one inside the top N% of the model's probability or at
least as likely as the model's own draw, and `lower-temperature` (top-0) only the latter. A guess they reject is
replaced by the token decoding without guesses writes there. The defaults are each engine's earlier rule: `exact` in
`strata-q38`, `speculative` in `strata-ds41`. Both engines decide in one shared function, `pick_verified_token` in
`include/strata/common/kernels/draft_rule.hpp`. How much each rule changes speed has not been measured yet.

In `strata-ds41`, the Memory panel's Drafter switch (`CHANGE_SETTINGS drafter=off|on`) also releases and
reloads DSpark's dense tensors. `--drafter off` and `--dspark off` start with those tensors released;
`--dspark none` loads no DSpark at all. A request's `spec=0` disables proposals for that request only.
The shared memory manager trades whole pool chunks against the dense allocation. The panel separates
Drafter (dense weights) and Drafter experts from the main model's weights and experts.
Recent trunk taps stay in each conversation's fixed pool region, beside the DSpark rings. They travel with
pauses and checkpoints, so the engine can rebuild the rings after loading the tensors again. This adds
7,864,320 bytes per conversation for the tested pack, plus the checkpoint copies in RAM.

In `strata-q38`, the MTP draft layer's 512 experts (1.32 MiB each, 675 MiB in all) live in the last card's page
pool, ranked before the card's own experts, and the Memory panel lists them as "Drafter / Layer 48". Only the draft
layer's dense weights (about 130 MiB) are an allocation of their own, traded against pool chunks like DeepSeek's.
Switching the drafter on pushes the card's least ranked experts out to make room, and its experts load between
steps; switching it off gives their pages back. A guess whose step routes to a draft expert that is not in VRAM
leaves it out when `--route-tail-skip` would leave out such an expert of the model itself; otherwise the guesses end
before that step. `--route-prior-pct` also makes the draft layer prefer its experts in VRAM. Neither changes the
answers: every guess is still checked. The draft layer's routings count toward its experts' popularity as the
model's own do, so the adaptive tier (`--adapt-every`) swaps its popular experts back in; switching the drafter off
clears those counts. A draft layer that does not load at start is switched off with a warning.
Measured on Strix Halo (Radeon 8060S, Windows, HIP), the IQ3_S pack, one greedy request of 160 tokens, twice each:
123 of 169 to 170 guesses accepted with the experts in the pool, 3.9 to 4.3 s of decode against 6.3 to 6.8 s with
the drafter off; switching it on again loaded its experts in 2.1 s.

Measured on Windows HIP, Strix Halo (Radeon 8060S), the IQ2_XXS pack copy, context 8192,
`--gpu-mib 10240 --ram-mib 512 --expert-cache 480 --no-expert-cache --adapt-swaps 0`,
`--dspark-conf 0.65`, and `DS4_COMMIT_CAP_GB=18`: switching off released 752,205,056 dense bytes;
the GPU pool grew from 10,734,796,800 to 11,271,536,640 bytes. Switching on restored the dense allocation
and the original pool size in two cycles. Chunk rounding transfers 536,739,840 bytes into the pool;
the remainder stays available outside it. The panel reports 7,219,445,760 bytes of drafter experts
separately; after refilling they return to that size. A greedy 16-token continuation after on, off,
a different 8-token conversation, and on again matched uninterrupted tokens, including pause/resume
and rewind to the prompt checkpoint. Lowest free commit/physical RAM: 50.37/26.72 GiB.
Startup with `--drafter off` and a continuation checkpoint created while off also passed; the latter
matched after pause/resume. That run kept at least 50.92 GiB free commit and 26.62 GiB free physical RAM.
Switches sent during a single `GEN` were queued until `DONE`; its 32-token answer matched the
uninterrupted answer. This checks safe deferral, not a switch applied inside a decode window.
These checks used stdin/stdout, not the user's server. Logs and the harness are in
`%TEMP%/dspark-memory` and `%TEMP%/dspark-harness.py`. CUDA, real multi-GPU hardware and browser rendering
were not tested. HIP builds of `strata-ds41`, `strata-dsv41` and `strata-q38` passed.
DeepSeek can offer DSpark guesses when several conversations share a step. The shared step divides `--rows`
among them; `--batch` and live Conversations at once set the concurrency. Each request rebuilds its
own token index from its full prompt, including after pauses and session restores. Learned costs follow the logical
conversation through in-process pauses; costs are rebuilt after a process restart. Stderr reports suffix and
chained guesses kept and offered per request.


Measured on Strix Halo on Windows, HIP, the IQ2_XXS pack, 8,192 context, D7: with DSpark, 1,200 trunk experts
and 384 drafter experts on the GPU, two alternating rounds of 64 tokens per answer, story / code / count decode
was 3.62 / 4.06 / 7.92 tok/s* with `strata-dsv41`, against 3.80 / 4.79 / 9.41 tok/s* with `strata-ds41`.
The new engine used 18,432 MiB GPU and 8,192 MiB RAM pools; the old engine used `DS4_COMMIT_CAP_GB=26.10`,
`--no-expert-cache --ssd-cache 0`. Both used `--dspark-conf 0.65`. Equal expert totals placed different experts
on the GPU, so this compares those placements too.

On the same hardware and pack on Windows, commit `ff7e608e`, 8,192 context, default adaptive placement,
48 GiB GPU and 6 GiB RAM pools, DSpark off, 64-token code / story / count answers: three conversations gave
8.98 / 10.38 aggregate tok/s* in two rounds, against 6.69 / 7.51 tok/s* for one. That is 34–38% more aggregate
speed. Timing starts after the last admission and excludes prefill. With DSpark on, three were slower than one
(7.10 / 7.15 against 8.61 / 7.55 tok/s*); mixed-conversation steps did not draft in that build.
Sources and limits: [D7, D8 and M12](MEMORY_MANAGER_PLAN.md).

Concurrent DSpark drafting is restored on 5 November 2026. On Windows HIP, Ryzen AI MAX+ 395 / Radeon 8060S,
the IST-DAS 2.00 bpw pack and 256-token greedy answers, the allocator granted 49.99 GiB GPU pools and
24.74 / 35.74 GiB page-locked RAM pools (80.24 / 92.05 decimal GB combined). Across the three-prompt grids,
the highest pooled DSpark-on rates were 7.83 tok/s at three conversations, `--rows 5`, and 10.54 tok/s at
three conversations, `--rows 4`, respectively. Two conversations with `--rows 4` gave 5.97 / 8.24 tok/s;
`--rows 5` gave 5.41 / 7.33. These runs do not establish two conversations with one or two guesses each as
the fastest choice. A reversed-order larger-pool check with two concurrent seasons/repetition answers favored
`--rows 5` at 16.84 tok/s against 14.97 without guesses; it does not compare widths or the full prompt mix.
At `--rows 8`, one batched drafter forward takes 35.56 / 40.97 / 53.08 ms for one / two / three eligible
conversations in the larger working pool; the time grows less than the conversation count.
Each grid cell has at least two measured answers per prompt. The long sequential grids include
adaptive placement, accumulated paused histories and any thermal variation; the owner reports 96–99 °C under load,
without per-round clock or temperature logs. In a separate 32 GiB requested-RAM fixed-placement suite,
concurrent and live 1→2→1 answers match standalone
through 256 tokens; SSD restore also matches. Full ranges, per-conversation draft counts, expert reads,
fresh controls, drafter timings and validation limits are in the
[batched drafting report](../bench/results/2026-11-05-ds41-drafting/README.md).

* on Windows (users have seen Strix Halo much faster on Linux)

## Quantization coverage of the page-pool engines (4 November 2026)

The replacement engines must retain the old engines' model formats. They share the model loaders,
quantization readers and kernels. No pack format is retired in this change; no conversion or download is needed
when selecting a page-pool engine. Source files stay read-only. Keep `--native SHARD1` for Qwen's dense weights
and head, including with a canonical Q2_0 pack. Ready packs also retain their explicit `--native-dense-gguf`,
`--embd-gguf` and `--ple-gguf` paths through server/launcher argument translation.

Inventory from `setup.py`, `strata_common/catalog.py`, the shared expert-layout reader and both model loaders:

| Family | Model files / quantization | Pack / reader | Local files and validation |
|---|---|---|---|
| Qwen3.8 original ISTA | Q2_0 | Canonical `strata_pack.py` + `pack_index.py`, or native `iq_pack.py` | Absent; canonical verify kernel and source-reader paths retained; inference unverified |
| Qwen3.8 original ISTA | IQ2_XS, IQ3_XXS, IQ3_S | Native expert manifest | IQ3_S present; old/new and concurrent checks below; others unverified |
| Qwen3.8 Coder | IQ1_M, pruned expert count | Native manifest, experts stored as IQ3_S | Absent; reader fixture checks pruned count; inference unverified |
| Qwen3.8 Swift 1.5 | IQ2_XS, IQ3_XXS, IQ3_S | Native manifest, arbitrary split layout | Absent; split reader fixtures; inference unverified |
| Qwen3.8 Unsloth | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | Native, `--compat-bf16`; mixed types and per-role shards | Absent; pack conversion and split-reader tests; inference unverified |
| Qwen3.8 Gyro | Gyro-S, Gyro-M | Native trellis experts, compatibility dense conversion | Absent; shared reader/kernels; inference unverified |
| Qwen3.8 NVFP4 | Strata GPTQ, NVIDIA, QAD | Ready native pack; local NVIDIA checkpoint preparation also supported | Absent; separate-file launch regression test; inference unverified (NVIDIA only) |
| Qwen3.8 OrcaRouter | Uncensored IQ3_XXS | Explicit native compatibility conversion | Absent; shared native loader; inference unverified |
| Qwen3.8 APR | Hadamard-folded experts | Native manifest v5 plus `hadamard.txt` | Absent; existing shared rotation paths; inference unverified |
| DeepSeek IST-DAS | 1.75 / 2.00 / 2.25 bpw (IQ1_M / IQ2_XXS / IQ2_XS) | `ds4.txt`, `tensors.txt`, Engram hash table, GGUF source spans | 2.00 present; others absent and inference unverified |
| DeepSeek Unsloth | UD-Q8_K_XL (MXFP4 experts, BF16 dense) | Same GGUF pack reader | Absent; shared MXFP4 CPU/GPU paths; inference unverified |
| DeepSeek manual EXL3 | Safetensors, per-matrix trellis widths | Existing EXL3 tensor map and nine-piece expert blobs | Absent; synthetic nine-piece file test passes; inference unverified |

Native Qwen manifests v2/v3, v4 per-role shard names and v5 rotations use the same reader as `strata.exe`.
Experts may be in the model shards or optional `experts.bin`; the canonical Q2_0 reader remains available.
DeepSeek accepts the current six-shard ISTA layout (metadata shard plus four tensor shards and the separately
named Engram), older four-shard ISTA layouts, and first-release `model-2.00bpw` / `model-2.25bpw` names through
setup's pinned layouts and `pack_gguf.py`. These older layouts are absent here and inference is unverified.
DSpark Q8_0 is a separate optional model and uses the same loader in both DeepSeek engines.

At the kernel level, Qwen also shares Q4_0/Q4_1, Q5_0/Q5_1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, Q8_0,
IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, Q2_0, NVFP4 and trellis
144/145/146 implementations (Q6_K depends on the build option). DeepSeek shares MXFP4, the IQ and K-quant
expert kernels, plus experimental TQK6/TQK7 and TQK5X/TQK4X/TQK5T/TQK4T. Kernel coverage does not establish
that an arbitrary model export has the required dense tensors, geometry or metadata.

Local inventory: `Strata-data/models/IQ3_S`, `Strata-data/packs/iq3_s`, and the IST-DAS 2.00 pack under
`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw`. Its original model and DSpark files are in
`../DeepSeek-V4.1-Flash`; tests use copied pack manifests and profiles, never modify those files.
Qwen3.6 UD-Q4_K_XL files are also present; that family is outside this replacement task.

Evidence and limits: Windows HIP on Strix Halo; temporary scripts and logs in `%TEMP%/quant-compat`.
The Qwen IQ3_S test runs two 8-token greedy answers in each engine, then a concurrent pair in strata-q38:
all token IDs match. GPU pool 6144 MiB, RAM pool 1024 MiB, context 8192, 480 GPU experts.
Every launch checks Docker/WSL processes, existing engines, free commit and physical RAM. A PID-scoped watchdog
stops the test below 3 GiB free commit or 4 GiB free physical RAM. No port 8080 or browser work is used.
CUDA, SYCL and Metal builds and absent model inference remain unverified.

DeepSeek IST-DAS 2.00 bpw also passes two 8-token greedy old/new answers and a concurrent pair matching
standalone tokens. Context 8192, DSpark off, paged GPU/RAM pools 4096/1024 MiB, 200 GPU experts;
the old engine uses `DS4_COMMIT_CAP_GB=12` and `--ssd-cache 1`. These are correctness checks, not speed
comparisons. Lowest free commit / physical RAM: Qwen old 57.06/25.26 GiB, new 52.69/25.47;
DeepSeek old 55.51/25.95, new 54.91/26.31. Initial old DeepSeek runs with `--ssd-cache 0` were stopped
by the physical-RAM watchdog; the bounded-store retry passed. The harness stages runtime DLLs beside each
test executable; an initial unstaged launch failed before model load.

File-free checks: 33 `tools/test_iq_pack.py` tests, the native manifest fixtures in `expert_layout_test`,
`file_expert_source_test`, and 343 `ds4_expert_files_test` checks (including nine EXL3 pieces across two files).
Two launch-argument regressions and five architecture checks pass. HIP sources compile and the four selected
targets link; a direct rebuild reports no work. The local Visual Studio environment script prints a missing
`vswhere.exe` diagnostic and returns a nonzero wrapper status despite the successful Ninja build.
EXL3 inference and its chunked prompt path remain unverified; that path retains the old region staging reader.


## DeepSeek paged stages: dense weights and expert hops

`strata-ds41` uses the shared paged engine's cut planner, conversation transfer and expert placement.
A cut may cross the initial dense-weight ranges. On separate GPUs, DeepSeek reloads each moved
layer's dense tensors and verify scratch on its new owner before freeing the previous copy. The
output head and DSpark stay on the last stage. Reloads print
`dense layer L migrated: gpuA -> gpuB (N bytes)`; weights and scratch remain outside the page pools.
A failed layer reload retains that layer's old copy. Extra temporary room for one layer is needed.
Two stages on one GPU reuse the same allocations and transfer their accounting.
`strata-q38` does the same for Qwen: a card loads the dense weights of the layers it gains before it
frees those of the layers it gives away, and prints
`gpuN holds the dense weights of layers A-B (N loaded, N freed)`. A failed load leaves the card as it
was. A move therefore needs free room on the receiving card for the moved layers' dense weights.

The shared expert placement selects at most one holder for each layer's lent experts. DeepSeek's CPU
lane sends that holder the selected activation rows in one batch, runs its existing expert matrix
kernels on a separate stream and private scratch, and returns the rows before publishing the primary's
response. CPU entries retain their original order in the weighted sum. The chunked prompt path still
streams nonlocal experts; hops apply to window decoding. `--remote-experts on` forces eligible hops;
`auto` compares the measured link with a startup calibration of a CPU expert from RAM; `off` disables
lending. Separate GPUs are required. Same-GPU forced hops remain refused because a waiting verify
kernel and hop kernels can stall the GPU and reset its driver. There is no safe same-GPU test override.
Cross-card token identity and speed must be checked by the NVIDIA session; no speed gain is claimed.

Local check (2026-11-04, Radeon 8060S, Windows HIP, two stages on one GPU): with 240 fixed GPU
experts, drafting and adaptive placement off, 2 GiB per GPU pool, 1 GiB requested RAM pool and
`DS4_COMMIT_CAP_GB=24`, a 96-token story at cut 10 moved to cut 20 after token 24 when the first
pool halved. Restoring the pool after token 48 succeeded. All 96 tokens matched the fixed-cut-20
run and a one-stage reference with a 4 GiB GPU pool. A separate cut-20 run with the pool halved but no cut change also matched all 96 tokens.
A stress run shrinking that pool to a quarter moved 20 -> 11 -> 20, but its answer diverged at
output token 71. During the moved cut it held about 69 MB fewer GPU expert bytes; a change of
CPU/GPU arithmetic is a possible cause, not a confirmed diagnosis. Separate-card migration,
hop correctness and speed remain unmeasured here. The kit's identity check starts at cut 10
and halves the pool to exercise a move without that expert shortfall.


## DeepSeek realistic GPU and RAM pools, 4 November 2026

Windows HIP gfx1151, Ryzen AI MAX+ 395 / Radeon 8060S, IST-DAS 2.00 bpw, five experts per token,
524,288 configured context, greedy 256-token answers. Requested new pools were 60 GiB GPU / 20 GiB RAM;
the allocator capped the GPU pool to 51.49 GiB without DSpark and 49.99 GiB with it. Available pools
were filled before READY and stable through warm-up. Old cache budgets match the actual allocations.

| Engine | DSpark | Chats | Total tok/s | Mean each | Kept/offered | GPU experts | RAM experts/slots |
|---|---|---:|---:|---:|---:|---|---|
| Old | Off | 1 | 5.24 | 5.24 | 0/0 | 6266 | 2090 |
| Old | On | 1 | 7.25 | 7.25 | 710/761 | 5259 + 384 draft | 2090 |
| New | Off | 1 | 6.82 | 6.82 | 0/0 | 6308 | 2431 |
| New | Off | 2 | 9.19 | 4.60 | 0/0 | 6308 | 2431 |
| New | Off | 3 | 9.96 | 3.32 | 0/0 | 6308 | 2431 |
| New | Off | 4 | 9.44 | 2.36 | 0/0 | 6308 | 2431 |
| New | On | 1 | 11.16 | 11.16 | 691/746 | 5277 + 384 draft | 2417–2425 |
| New | On | 2 | 7.52 | 3.76 | 0/0 | 5277 + 384 draft | 2395–2414 |
| New | On | 3 | 7.65 | 2.55 | 0/0 | 5277 + 384 draft | 2381–2386 |
| New | On | 4 | 8.96 | 2.24 | 0/0 | 5277 + 384 draft | 2371–2372 |

Rates pool at least two measured rounds; concurrent timing starts after the final admission.
GPU counts exclude the separate drafter count shown; RAM counts use each engine's own cache representation.
The owner reports 96–99 °C under load, quiet fan; clocks were not logged per timed round. Thermal variation
and different greedy answers limit comparisons. Some seasons/story answers also change within each engine.
No production optimization was applied. A restarted new-off segment preserves completed samples.
Exact flags, misses, reads, memory minimums, identity checks and validation are in the
[full report](../bench/results/2026-11-04-ds41-realistic/README.md). Port 8080 and the user's server were untouched.
