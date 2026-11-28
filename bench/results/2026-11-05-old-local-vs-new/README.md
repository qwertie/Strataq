# Old local versus new engines, 5 November 2026

Windows 11, Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GiB installed unified memory.
Old means `local` at `debc4619c354b242dc2e48af12c2fdbfab2dab04`; new means
`memory-manager` at `d1226394bd3d83b919ac73c70989bb60d43dd287`.

New Qwen was faster in all 12 standalone prompt/mode comparisons, by 4.6–41.9%. New DeepSeek was slower in all six
matched standalone comparisons, by 6.1–28.9%.

Greedy identity: Qwen matches 24/24 standalone old/new answer pairs; DeepSeek matches 4/12.

These are results for this machine and these short prompts. Thermal variation and differing DeepSeek answers
limit causal conclusions. No optimization or default change is part of this task.

[Pooled standalone comparison chart](comparison.html).

## Binaries

- `strata.exe` (old local): SHA-256 `1601ea2234f652c116f21e7e3034627ff920f74585db1dbd359162ab1abfcda5`.
- `strata-dsv41.exe` (old local): SHA-256 `128a1aa271a02924d4efa2b11d04d5b9b6abc406ed9334d30f8f7d2169399ba8`.
- `strata-q38.exe` (new): SHA-256 `0ce3e4428e626b0c688d14d3af47e23f1dc1eb54405004b7c3f1636806765bc1`.
- `strata-ds41.exe` (new): SHA-256 `8c60a1f6246848bc829aae3a7ac454970fc2b1bac4e1d1baa97157833e27081c`.

## Method

The procedure follows the [Qwen full-residency study](../2026-11-03-q38-realistic/README.md),
[DeepSeek realistic study](../2026-11-04-ds41-realistic/README.md), and
[DeepSeek drafting study](../2026-11-05-ds41-drafting/README.md).

Fresh detached old worktree and HIP Release builds used the same ROCm compiler as new,
with portable CPU code, native experts and prefill MMQ enabled. The reused llama.cpp source is at
the same pinned commit, `3cf03257f219afbe7334045ff7c6a06ac68c627d`, in both builds. New targets were rebuilt;
the final build reported no work remaining. No engine source was modified.

The retained mode order was Qwen MTP-only, suffix-only, both; DeepSeek DSpark-on; the Qwen no-draft retry;
then the matched DeepSeek no-DSpark retry. Exact order is also retained in JSON.
Each start uses a private expert profile; DeepSeek also uses private pack metadata. Original model files
and owner settings are read only.
Each mode runs old/new/old/new, with a 20-second engine-free idle gap before every start.
Every process warms through at least two full three-prompt cycles until allocated GPU expert
residency stops changing. Each process then supplies one measured 256-token greedy answer
per prompt, giving two independent measured rounds per engine/mode. New concurrent cells have
a preceding warm batch on each start. Two conversations rotate all three prompt pairs.
The three prompts and no-thinking input token IDs are unchanged from the earlier Qwen and DeepSeek studies.
Max-row values are configured ceilings; a step can use fewer rows, especially with drafting off.
DSpark-on GPU trunk counts exclude the additional 384 drafter experts.

Single-conversation rates use native decode time. Concurrent per-prompt rates count only
tokens after final admission and divide by the common time through the last completion;
aggregate throughput counts all active slots. The total column pools all three prompts; its standalone range
is the rate of each complete three-prompt measured cycle. Concurrent ranges are individual batches. Rates pool counted tokens/time, with individual round ranges.
Draft kept/offered totals cover complete measured answers, including admission.

Temperature, clocks and power were not measured. Background desktop/CPU load was not isolated.
No free-commit or free-physical-RAM floor was imposed, and no requested RAM pool needed reducing. The owner reports
96–99 °C under load.
Equal gaps and alternating order reduce ordering bias but do not isolate throttling.
GPU expert bytes describe allocation, not OS/driver physical residency. Different DeepSeek answers
and expert placements prevent treating its rates as equal-token kernel comparisons.

## Qwen3.8 IQ3_S

All 24,576 experts must remain on GPU. New uses 60 GiB GPU and 512 MiB RAM pool requests.
The granted new GPU page pools were 52.75–53.75 GiB, retaining all experts. Its RAM page pool was about
0.25 GiB, with about 0.20 GiB additional scratch; these differ from the requests.
Old uses the study’s 4 GiB resident RAM budget; every expert is already on GPU.
The old builds lack `GET_MEMORY_USE`; startup INFO and cache logs supply their residency evidence.
Legacy GPU slot allocations are fixed after startup; repeated legacy snapshots retain startup INFO, not live samples.
Old max rows is `--spec 4`, expanded to 6 by suffix drafting. New starts with `--rows 8`.

| Engine | MTP | Suffix | Max rows (configured) | Conversations | Seasons tok/s (range) | Repetition tok/s (range) | Story tok/s (range) | Total tok/s (range) | Drafts kept/offered | GPU trunk experts |
|--------|---|---|---|---|---|---|---|---|---|---|
| old*   | ❌ | ❌ | 4 | 1 | 31.84 (31.82–31.86) | 30.19 (30.13–30.24) | 31.00 (30.11–31.93) | 30.99 (30.68–31.31) | 0/0 | 24576 |
| new    | ❌ | ❌ | 8 | 1 | 33.42 (33.22–33.63) | 33.50 (33.48–33.53) | 33.27 (33.18–33.37) | 33.40 (33.30–33.51) | 0/0 | 24576 |
| new    | ❌ | ❌ | 8 | 3 | 21.16 (21.13–21.19) | 21.24 (21.21–21.27) | 21.33 (21.30–21.35) | 63.72 (63.64–63.81) | 0/0 | 24576 |
| old*   | ❌ | ✅ | 6 | 1 | 31.87 (31.83–31.92) | 66.87 (66.28–67.48) | 30.02 (29.97–30.07) | 37.67 (37.56–37.78) | 353/409 | 24576 |
| new    | ❌ | ✅ | 8 | 1 | 33.49 (33.22–33.76) | 94.87 (93.94–95.82) | 33.37 (33.29–33.44) | 42.63 (42.38–42.88) | 420/472 | 24576 |
| new    | ❌ | ✅ | 8 | 3 | 21.32 (21.23–21.40) | 21.40 (21.32–21.49) | 21.49 (21.40–21.57) | 64.20 (63.95–64.46) | 314/332 | 24576 |
| old*   | ✅ | ❌ | 4 | 1 | 50.81 (50.69–50.92) | 69.19 (69.05–69.34) | 42.14 (42.14–42.14) | 51.84 (51.78–51.91) | 950/1338 | 24576 |
| new    | ✅ | ❌ | 8 | 1 | 53.77 (53.76–53.78) | 74.56 (74.50–74.62) | 44.09 (44.08–44.10) | 54.86 (54.84–54.87) | 946/1332 | 24576 |
| new    | ✅ | ❌ | 8 | 3 | 22.77 (22.52–23.03) | 22.68 (22.43–22.94) | 22.95 (22.69–23.22) | 68.41 (67.64–69.19) | 814/1058 | 24576 |
| old*   | ✅ | ✅ | 6 | 1 | 45.92 (45.82–46.02) | 70.06 (64.62–76.51) | 38.95 (37.57–40.44) | 48.61 (48.44–48.78) | 910/1360 | 24576 |
| new    | ✅ | ✅ | 8 | 1 | 49.39 (49.35–49.44) | 85.78 (85.74–85.81) | 42.20 (42.14–42.26) | 53.96 (53.94–53.98) | 942/1378 | 24576 |
| new    | ✅ | ✅ | 8 | 3 | 22.02 (22.00–22.03) | 21.93 (21.92–21.95) | 22.19 (22.18–22.21) | 66.14 (66.10–66.18) | 792/1102 | 24576 |

\* old = `local` branch, a mininally edited version of the original Strata from Nov 1, 2026 or thereabouts

Greedy identity, old versus new (1-based first differing token; `match` means all 256):

- Draft=0, suffix=0: round 1 ['match', 'match', 'match']; round 2 ['match', 'match', 'match']
  (seasons/repetition/story).
- Draft=0, suffix=3: round 1 ['match', 'match', 'match']; round 2 ['match', 'match', 'match']
  (seasons/repetition/story).
- Draft=1, suffix=0: round 1 ['match', 'match', 'match']; round 2 ['match', 'match', 'match']
  (seasons/repetition/story).
- Draft=1, suffix=3: round 1 ['match', 'match', 'match']; round 2 ['match', 'match', 'match']
  (seasons/repetition/story).

## DeepSeek IST-DAS 2.00 bpw

New requests 50 GiB GPU plus 36 GiB page-locked RAM. Old requests the nearest equivalent
36 GiB RAM store via decimal-GB `--ssd-cache`, with explicit `DS4_COMMIT_CAP_GB` and
`DS4_UMA_GPU_RESERVE_GB=4`. New granted 49.99 GiB GPU and 35.74 GiB page-locked RAM pools. GPU trunk counts differ by less than 1%; this is not an exact
expert-key placement match. DSpark-off means `none` (unloaded). Old has no shared `--rows`;
`--dspark-depth 7` caps proposals at seven, hence at most eight verify rows including the current token.
Rows 5 for two conversations come from the drafting study’s close reversed-order pair check;
rows 4 for three come from its balanced 36 GiB grid. Off controls keep the same row budgets.

| Engine | DSpark | Max rows (configured) | Conversations | Seasons tok/s (range) | Repetition tok/s (range) | Story tok/s (range) | Total tok/s (range) | Drafts kept/offered | GPU trunk experts |
|---|---|---|---|---|---|---|---|---|---|
| old* | ❌ | 8 | 1 | 15.49 (14.98–16.03) | 18.50 (18.25–18.76) | 16.24 (15.61–16.92) | 16.65 (16.16–17.17) | 0/0 | 6120 |
| new | ❌ | 8 | 1 | 12.39 (12.23–12.55) | 13.53 (13.51–13.55) | 12.45 (12.40–12.51) | 12.77 (12.69–12.85) | 0/0 | 6119 |
| new | ❌ | 5 | 2 | 9.50 (8.91–9.81) | 9.50 (8.94–10.11) | 9.71 (9.35–10.15) | 19.13 (17.85–20.25) | 0/0 | 6119 |
| new | ❌ | 4 | 3 | 7.00 (6.98–7.01) | 7.02 (7.01–7.04) | 7.05 (7.03–7.07) | 21.07 (21.02–21.12) | 0/0 | 6119 |
| old* | ✅ | 8 | 1 | 14.55 (14.27–14.85) | 35.30 (34.69–35.93) | 13.80 (13.71–13.90) | 17.70 (17.45–17.96) | 717/758 | 5259 |
| new | ✅ | 8 | 1 | 13.67 (13.38–13.97) | 25.11 (25.10–25.12) | 12.01 (11.81–12.22) | 15.29 (15.27–15.31) | 723/789 | 5277 |
| new | ✅ | 5 | 2 | 8.50 (8.01–9.02) | 8.74 (8.47–9.05) | 8.31 (7.98–8.59) | 17.02 (15.99–18.07) | 1095/1201 | 5277 |
| new | ✅ | 4 | 3 | 6.03 (5.99–6.07) | 6.03 (5.99–6.07) | 6.08 (6.04–6.12) | 18.14 (18.02–18.27) | 238/247 | 5277 |

\* old = `local` branch, a mininally edited version of the original Strata from Nov 1, 2026 or thereabouts

Greedy identity, old versus new (1-based first differing token; `match` means all 256):

- Draft=0, suffix=0: round 1 [35, 'match', 2]; round 2 [1, 'match', 9] (seasons/repetition/story).
- Draft=1, suffix=0: round 1 [1, 'match', 54]; round 2 [1, 'match', 30] (seasons/repetition/story).

## Exact commands and validation

[measurements.json](measurements.json) retains exact executable arguments and environment per start,
input IDs, every measured answer, native DONE lines, warm-up times, memory snapshots and round order.
Run commands use private stdin/stdout (`GEN 256 temperature=0 <comma-separated IDs>` or
`BGEN <slot> 256 temperature=0 <IDs>`), followed by `QUIT`. No HTTP listener was opened.
[commands.txt](commands.txt) also lists every exact Windows invocation with its environment.
The scratch build recipe is reproduced in [build-command.txt](build-command.txt). Initial Qwen no-draft
rounds separated by the failed DLL start are excluded and retained as controls under the scratch folder;
the main Qwen no-draft set was rerun as an uninterrupted A/B/A/B sequence. The initial DeepSeek
no-DSpark set had 6,266 old versus 6,119 new GPU experts (2.4% mismatch). Its timings are
excluded; a full A/B/A/B retry lowers the old commit cap to 61.68 decimal GB for matched counts.

[excluded-controls.json](excluded-controls.json) preserves the excluded timing summaries and commands.

[validation.json](validation.json) records the independent consistency checks.

**Passed:** both old HIP release builds, rebuilding new HIP targets, all requested cells with at least
two measured rounds per prompt, 256-token count checks, one engine at a time, cleanup and diff whitespace check.

**Failed:** universal old/new greedy identity where differences are listed above. Startup harness pilots
exposed missing adjacent runtime DLLs and the unsupported old memory-report command; no pilot timings
are included. The interrupted initial nonportable Qwen startup is excluded.

**Not run:** CUDA, SYCL, Metal, long prompts, sampled-distribution checks, GPU paging or thermal telemetry,
the lifecycle matrix, vmm_test, hip_prefill_gemm and hip_prefill_mmq_parity.

No branch switch, stash, commit, push, user-server operation, model edit or fan/power change occurred.
No browser preview, rendering or encoding ran during inference. Temporary worktree removed with git worktree remove; all harnesses and
engines stopped. Raw logs, scratch harnesses and builds remain under `%TEMP%/strata-old-local-h2h`.
