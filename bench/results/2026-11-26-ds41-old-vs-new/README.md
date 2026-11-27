# DeepSeek V4.1: old versus new engine, and the new engine's rows, 26 November 2026

Windows 11, Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB unified LPDDR5X (63.6 GiB of it seen as RAM), a 100 GB
page file (commit limit 163.6 GiB), fans on full. DeepSeek V4.1 Flash IST-DAS 2.00 bpw (`ist-das-2.00bpw` pack),
5 experts per token (the engines' default). Old means `strata-dsv41`, new means `strata-ds41`; both built from
`memory-manager` at `5ab1fe692` in the same HIP Release build folder with the same flags (portable CPU code, prefill
MMQ, gfx1151).

- `strata-dsv41.exe`: SHA-256 `c38f1b540db6ba1c19a5b60ac152a5b095e996e1ef5058ff912fd90a91bb325c`.
- `strata-ds41.exe`: SHA-256 `eafa976680ed7d9fe84c2a4d1d587aab98a8753f905b1ec579f96cfc8d182f0a`.

With the drafter off, new was 4.2% slower than old over three prompts (2.8-5.7% per prompt). With DSpark on, new was
2.5% faster overall, but the two engines wrote different answers, so its per-prompt numbers are not a clean comparison.
The gap with the drafter off is smaller than on 5 November (6.1-28.9%), when the machine was throttling.

## Old versus new, one conversation

Same pools as the [5 November study](../2026-11-05-old-local-vs-new/README.md): new `--gpu-mib 51200 --ram-mib 36864
--rows 8 --batch 3 --concurrent 3`; old `--ssd-cache 38.654705664 --vram-experts 1 --dspark-depth 7`, with
`DS4_COMMIT_CAP_GB=62.7` chosen so that its GPU holds as many experts as new's (`scripts/calib_old.py`): 6,238 old
and 6,241 new with DSpark off, 5,400 each with it on. Both got a 38.6 GB RAM pool or store.

Each engine started twice per DSpark mode, old and new alternating with a 3-second gap. Each start used a fresh copy
of the pack and its expert profile, warmed up with two cycles of the three prompts (by then new's GPU experts had
stopped changing; old's GPU slots are fixed at start), then measured three cycles. A cycle is one greedy answer of
256 tokens to each prompt, without thinking:

- **Seasons**: the 5 November explanation prompt (same token IDs).
- **Pac-Man**: "Write a traditional Pac-Man clone in plain HTML with embedded JS (controls: arrow keys and WASD)".
- **Story**: the 5 November lighthouse story prompt (same token IDs).

Rates are tokens divided by the engine's own decode time, pooled over the six answers; ranges are single answers.

| DSpark | Prompt | Old tok/s (range) | New tok/s (range) | New vs old | Guesses kept, old / new |
|---|---|---|---|---:|---|
| off | Seasons | 16.70 (16.38-16.95) | 15.76 (15.34-16.20) | -5.7% | - |
| off | Pac-Man | 14.34 (14.15-14.70) | 13.94 (13.78-14.26) | -2.8% | - |
| off | Story | 16.44 (15.96-17.12) | 15.75 (15.27-16.10) | -4.2% | - |
| off | All three | 15.75 | 15.10 | -4.2% | - |
| on | Seasons | 15.92 (14.79-17.77) | 16.70 (16.19-17.19) | +4.9% | 526/594 / 552/688 |
| on | Pac-Man | 21.29 (20.00-22.44) | 19.19 (18.33-19.81) | -9.9% | 1190/1322 / 1120/1277 |
| on | Story | 14.17 (13.56-15.53) | 15.62 (15.00-16.21) | +10.2% | 294/338 / 305/343 |
| on | All three | 16.63 | 17.04 | +2.5% | 2010/2254 / 1977/2308 |

Greedy answers were not the same between the engines (they parted after 2 to 179 tokens), and they were not the same
between the cycles of one start either: the second and third answers to a prompt parted from the first, at the same
tokens in every start. A likely cause, not confirmed, is that experts move between the GPU, RAM and the CPU lane
between answers, and each place sums in another order.

## The new engine: conversations at once and rows per step

One start per DSpark mode, `--batch 4 --concurrent 4 --vram-conversations 4`, the pools as above, conversations and
rows changed live (`CHANGE_SETTINGS`); `scripts/ds_rows.py`, built on `tools/measure_rows.py`. Three rounds, each
running every configuration once, the order reversed every other round; medians. 256 tokens per conversation. The
prompts, in order: Seasons, Pac-Man, "Propose sprites written in SVG for a Pac-Man clone" (Sprites), and "Write a
recursive-descent parser and evaluator for floating-point math with standard unary/binary operators including
+ - * / % ^, in a modern C++ console app without 3rd-party libraries" (Parser). Two conversations take the first two.

"2x6" is 2 conversations and 6 rows: each conversation gets a row and the others are shared evenly as guesses, so 2
guesses each while both decode. Lookahead was 7 throughout, which only caps a conversation decoding alone. ".7t" is
`temperature=0.7` with the round's number as `seed` and `draft_rule=typical` (lossy: answers change); the rest are
greedy with the default `speculative` rule. Together rates count from when all conversations decode to the first that
ends. SSD MB is the engine process's read bytes (Windows' counter; the engine reads experts without the file cache),
from `scripts/io_sampler.py`, over the rounds the sampler saw whole (it started during the first round).

| DSpark | Config | Seasons | Pac-Man | Sprites | Parser | Together (each) | Kept | Rows/step | Experts/step | SSD MB/token |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 1x1 | 14.91 | 13.31 | 14.10 | 13.35 | | - | 1.00 | 200 | - |
| off | 2x2 | | | | | 18.92 (9.48, 9.44) | - | 1.99 | 395 | - |
| off | 3x3 | | | | | 23.17 (7.74, 7.71, 7.71) | - | 2.98 | 563 | - |
| off | 4x4 | | | | | 24.08 (6.04, 6.02, 6.02, 6.02) | - | 3.95 | 743 | - |
| on | 1x2 | 14.57 | 15.56 | 16.03 | 15.21 | | 94% | 1.72 | 295 | 50.1 |
| on | 1x3 | 14.83 | 17.98 | 18.44 | 17.52 | | 91% | 2.28 | 357 | 49.3 |
| on | 1x4 | 14.33 | 19.21 | 19.08 | 19.68 | | 89% | 2.49 | 374 | 48.2 |
| on | 1x6 | 15.94 | 21.42 | 18.00 | 17.58 | | 85% | 2.86 | 405 | 46.1 |
| on | 1x8.7t | 15.11 | 21.61 | 21.61 | 17.81 | | 94% | 2.94 | 414 | 45.2 |
| on | 2x4 | | | | | 20.51 (8.68, 11.84) | 90% | 3.01 | 512 | 42.9 |
| on | 2x6 | | | | | 25.06 (9.32, 15.74) | 90% | 3.23 | 513 | 13.9 |
| on | 2x6.7t | | | | | 26.96 (8.82, 17.92) | 94% | 3.27 | 517 | 22.7 |
| on | 2x8 | | | | | 26.75 (8.75, 17.50) | 89% | 3.40 | 519 | 25.8 |
| on | 3x6 | | | | | 25.16 (6.88, 9.79, 8.50) | 93% | 4.41 | 707 | 29.6 |
| on | 3x6.7t | | | | | 25.29 (6.72, 9.34, 8.82) | 97% | 4.45 | 716 | 30.2 |
| on | 3x8 | | | | | 26.82 (6.34, 10.96, 9.64) | 92% | 4.53 | 687 | - |

Rounds varied: the single-conversation rates of one configuration differed by up to 20% between rounds. The SSD
column covers one or two rounds per configuration and is a first look.

## What a decode step reads (from the pack's tensor list and the cache layout; not measured)

Per step of one conversation, 5 routed experts per layer: dense weights 5.31 GiB (attention, norms, router and
hyper-connections 3.78; the 40 shared experts, which every token uses, 0.91; head 0.42; Engram's non-table parts
0.20), 7.63 G parameters; routed experts 200 x 8.2 MiB = 1.60 GiB, 7.08 G parameters. By bytes the dense part is 77%,
by parameters 52%, because it is stored at 5.5-6.2 bits a weight and the experts at about 2. The conversation's cache
holds 920 bytes a token (all layers); a step reads about 13.5 MB plus 200 bytes a token of context (10 MB at 4K,
40 MB at 128K, 118 MB at 512K). At the bus's 256 GB/s, reading 7.43 GB a step allows about 34.5 tok/s for one
conversation, and 12.15 GB a step (745 distinct experts) about 84 tok/s for four together. Measured: 44% and 29-32% of
those. Only about 40% of the routed experts were on the GPU; the rest came from the RAM pool or the SSD.

## Files

- `measurements.json`: every old/new start: arguments, environment, startup lines, warm-up and measured answers
  (tokens, DONE lines), GPU experts and the last memory report.
- `rows-off.json`, `rows-on.json`: the rows runs (`tools/measure_rows.py`'s format, with the prompts' token IDs).
- `rows-on-ssd-reads.jsonl`: the read-byte samples of the DSpark-on rows run.
- `scripts/`: the harnesses as they ran (scratch tools, not maintained): `ds_h2h.py` (old/new), `calib_old.py`,
  `ds_rows.py`, `io_sampler.py`, `analyze.py` and `rows_report.py` (the tables above).

**Passed:** both engines built from one commit; GPU experts matched within 0.05%; every measured answer reached 256
tokens; one engine at a time.

**Failed:** one rows start failed at once with `hipErrorLaunchFailure` and `hipMemMap failed`, 3 seconds after the
engine before it was stopped; with 30 seconds between starts the same command ran. Without a commit cap the old engine
filled the GPU with 9,975 experts, left its RAM store empty and stopped at its first request ("layer 0 expert 80: no
store slot and no view of the files").

**Not run:** CUDA, SYCL and Metal; prompts longer than 43 tokens; 4 conversations with the drafter on; suffix and
lookup drafting; `--experts 6`; temperature and power telemetry; SSD reads of the old engine.
