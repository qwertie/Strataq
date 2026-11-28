# The NVIDIA test session

The code-sharing refactor ([docs/REFACTOR_PLAN.md](../../docs/REFACTOR_PLAN.md)) changed engine code that was only
tested on an AMD card. `run.ps1` checks it on an NVIDIA PC in one go: it builds the engines as they were before the
refactor (commit `87d20a5c`) and as they are now, runs both the same way, and writes a table of what matches. It was
written for a PC with an RTX 3060 (12 GB), an RTX 3080 (20 GB) and 48 GB of RAM on Windows 11. Nobody has to watch it
after it starts.

## Before the day

Install, in this order:

1. The newest NVIDIA driver (580 or newer: CUDA 13 needs it).
2. [CUDA Toolkit 13](https://developer.nvidia.com/cuda-downloads): 13.0 is what the release engines and the plan's
   Docker check are built with; the first session built with 13.4. It sets `CUDA_PATH`.
3. Visual Studio 2022 Build Tools with the workload "Desktop development with C++". CUDA 13 does not take
   Visual Studio 2026.
4. Git for Windows, and Python 3.12 or newer (64-bit, from python.org; the first session used 3.14).
5. This fork, with its `local` branch, and Python's packages for it (`cmake` 4.4.3 and `ninja` 1.13.2 come from
   `requirements-build.txt`):

   ```powershell
   git clone <the fork> Strata; cd Strata; git checkout local
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt -r requirements-build.txt
   ```

Bring the model files on a fast (NVMe) drive, in the same folders as in your `Strata-data` (about 340 GB):

| Folder in `-DataDir` | Size | Used by |
|---|---:|---|
| `models\IQ3_S\` (both `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000?-of-00002.gguf`) | 78 GB | Qwen3.8 checks |
| `packs\iq3_s\` | 1.5 GB | Qwen3.8 checks |
| `mtp\rt\` | 0.8 GB | the drafter; without it Qwen3.8 runs without drafting |
| `models\qwen36-UD-Q4_K_XL\` | 22 GB | Qwen3.6 checks |
| `packs\deepseek-v4.1-flash-ist-das-2.00bpw\` | 9 MB | DeepSeek checks |
| `-DeepSeekDir`: the DeepSeek folder (5 IQ2_XXS shards, the Engram, the DSpark drafter) | 240 GB | DeepSeek checks |

A check whose files are missing is skipped, not failed.

**What fits.** Qwen3.8 IQ3_S's experts are 50.3 GB. By setup's own rule (`low_ram_needed` / `low_ram_fits` /
`low_ram_resident` in `setup.py`) 48 GB of RAM puts it in the low-RAM mode: with the 20 GB card about 14 GB of experts
sit in graphics memory and the other ~36 GB are copied into RAM; with the 12 GB card alone it does not fit. So the
one-card checks run on the card with the most memory, and the multi-GPU checks on both. Qwen3.6 UD-Q4_K_XL (22 GB)
keeps the experts that do not fit the card on the CPU. DeepSeek's guide asks for 40 GB of RAM; it runs on the bigger
card.

## Running it

From the checkout, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash
```

- `-Quick`: one run a build, the main variants only, sm_75 only for the emulated builds: about half the time.
- `-Only multi,speed`: only these phases. `-Results <folder>`: continue in an earlier results folder; runs that
  finished are kept, so a `-Quick` session can be filled in later by a full one into the same folder.
- `-DryRun`: prints every command and runs none.
- Builds go to `C:\sn` (`-Work`): two git worktrees and up to six build folders (a CUDA build of every target
  took 2.2 GB on Linux). Before the first build, `run.ps1` checks each worktree out again at its commit, applies
  `portability.patch` (test sources Visual Studio does not compile at the older commit) and copies llama.cpp's
  `gguf-py` from this checkout into it. Ninja tries all targets even if one fails. A failed build gets a FAIL row and
  blocks its dependent checks; the build log names the failed targets.
- Every kit Git command trusts only the exact checkout or worktree path through `-c safe.directory=<path>`.
  The kit does not change global Git settings. A checkout, preparation or revision-query failure blocks that side's
  build and dependent checks with FAIL. The old HEAD comparison could compare two empty results when both Git
  commands failed; the kit now checks both exit codes and requires two nonempty matching full SHAs.
- Preflight records the kit checkout's full commit. Build rows record the verified source HEAD; the matching
  `build-<side>-<tag>.json` records each executable's SHA256. Dependent checks require that revision and unchanged
  executable. Existing builds without this record need a build phase. A failed build blocks dependent checks.
  Use fresh results when changing revisions: `-Results` intentionally preserves completed runs.

| Phase | What | Default | `-Quick` |
|---|---|---:|---:|
| preflight | Windows, cards, driver, PCIe links, peer-to-peer table, tools, RAM, page file, disks | 1 min | 1 min |
| build | both versions, every target, `86-real`, `STRATA_BUILD_DSV41`, `STRATA_NATIVE_EXPERTS`, tests | 1-3 h | 1-3 h |
| unit | `ctest` in both builds, test by test | 20-40 min | 20-40 min |
| smoke | the engines' protocol: Qwen3.8 (commands, batch slots, SAVE / RESTORE across the builds), Qwen3.6 | 45 min | 25 min |
| multi | the B3 list: layer split (auto, 12) in both card orders, peer tier, helper caches, `--vram-elastic`, `--kv-grow`, page-locked RAM copy | 3-4 h | 1 h |
| deepseek | the B2 list: greedy and seeded answers without the expert cache, DSpark on, off and sampled; decode speed | 2 h | 45 min |
| speed | the server on port 8091 with Qwen3.8: answers and prompt / decode tokens/s, 3 runs a build | 30 min | 20 min |
| emulate | builds of PTX for sm_75 and sm_80 run with `STRATA_EMULATE_CC`: the GPU tests and the protocol checks | 3-5 h | 2 h |
| memory | not in the default list: [the memory manager's checks](#the-memory-managers-checks) | 2-3 h | 2-3 h |

The times are estimates (not measured: they depend on the CPU and the drive); the summary records the real ones.
The emulated builds run last because they are the longest and the least likely to find something. Under
`STRATA_EMULATE_CC=80` the host asks for the A100's 163 KB of shared memory, which these cards do not have, so a test
may fail there in both builds; `decode_cluster_parity` passes without comparing anything below sm_90.

## The follow-up session

The first session (30 October) left a few checks open: answers that changed in one run of the build after, and
decode speeds measured once. `-Followup` runs only those, 4 times a build (`-Runs N` for another count): both builds
again (now with every test), the unit tests, `multi-helper` and `multi-peer` in both card orders,
`multi-split-auto-res` and `multi-split-12-res`, the server speed check (5 runs a build), and `smoke-q38state` on the sm_75 build.

```powershell
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -Followup -DataDir G:\AI-stuff\Strata-data -Work "Z:\C++\strata-builds"
```

| Part | Estimate |
|---|---:|
| builds at sm_86 (the first session's worktrees and build folders are reused) | 1-2 h |
| unit tests, both builds | 20-40 min |
| `multi-helper` and `multi-peer` (BS, SB), `multi-split-auto-res`, `multi-split-12-res` (BS): 48 runs of about 3.7 min | 3 h |
| server speed: 10 runs of about 3.4 min | 35 min |
| sm_75 builds, then `smoke-q38state-75v`: 8 runs of about 2.9 min | 1-1.5 h |

About 6-7 hours in all. The run times are the first session's; the build times are guesses. Send back the zip it
names at the end.

The follow-up's multi-GPU and speed runs, in both builds, also print where a decode's time went (`STRATA_TRACE=1`,
`STRATA_SPLIT_TIMING=1`, `STRATA_DECODE_TIMING=1`: a few lines a request, none a token). Under each of those checks'
row, a line per build gives the medians of its runs: the ms of a decode window and how much of it waited for the
GPUs, each card's wait for its GPU and its CPU pool time per window, the shares of the routed experts that CUDA0's
cache, another GPU or PCIe, and the CPU computed, and how many experts each card holds. A slower build then shows
where it lost the time. The lines stay in the raw logs and are left out of the comparison.

## The memory manager's checks

On the `memory-manager` branch, the `memory` phase checks the new Qwen3.8 engine, `strata-q38`
([docs/MEMORY_MANAGER_PLAN.md](../../docs/MEMORY_MANAGER_PLAN.md), steps M7 to M9). It compares against the build of
M7 (`4fa1d1cf`), so it takes `-Before 4fa1d1cf`, and needs both builds:

```powershell
git checkout memory-manager
powershell -ExecutionPolicy Bypass -File tools\nvidia_session\run.ps1 -Only preflight,build,memory -Before 4fa1d1cf -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash -Work C:\sn-mm-fresh
```

For a new memory session, update the `memory-manager` checkout first. Use a fresh local `-Work` folder and include
`preflight`. Enter the actual DeepSeek folder when prompted:

```powershell
$ds = Read-Host 'Full DeepSeek GGUF and Engram folder path'
$work = "C:\sn-mm-$(Get-Date -Format yyyyMMdd-HHmmss)"
& .\tools\nvidia_session\run.ps1 -Only preflight,build,memory -Before 4fa1d1cf -DataDir 'G:\AI-stuff\Strata-data' -DeepSeekDir $ds -Work $work -Runs 4
```

The groups checks use 8,192 MiB of RAM and 4,096 MiB per GPU for the new engine, and an 8 GiB resident
budget with `--expert-cache 400` for the old engine. They require 40 GiB free commit and 16 GiB free physical RAM
before each start. A PID-scoped watchdog stops that check's engine below 16 GiB free commit or 4 GiB free RAM.
`DS4_COMMIT_CAP_GB=30` is explicit. No server is started. All other memory checks below use the existing budgets.

`strata-q38` gets `--ram-mib 20480`: its RAM pool is page-locked and by default takes what RAM has free, which on 48 GB
leaves Windows' commit limit too little. Each check runs 2 times a build (`-Runs N` for another count).

| Check | What it compares | It confirms |
|---|---|---|
| `memory-groups-q38-SB`, `-BS` | This build's `strata-q38`, two cards, cut 24, four conversations: `--batch-groups 1` against `auto`, drafting off, fixed --expert-cache 400 | Four 96-token standalone answers equal two rounds of concurrent answers; aggregate decode speed after the final admission; stderr proves two stages occupied together |
| `memory-groups-old-SB`, `-BS` | This build's original `strata.exe` with the same split, prompts, groups and expert budget | The same identity checks and speed scope, for comparison with the new engine; no old/new token identity is assumed |
| `memory-q38-1card` | `strata-q38` on the bigger card, M7's build against this one | its answers and speed did not change (it now measures the card's link at start and paces the experts it puts back by it) |
| `multi-split-auto-SB`, `-BS` | the old engine's `--layer-split auto`, both card orders, M7's build against this one | the old engine is unchanged now that its split planner is shared code (`src/common/experts/layer_split.cpp`): the same answers, and the same `layer split auto: K=...` line (the startup lines are compared too) |
| `memory-speed-split-q38` | the old engine's `--layer-split auto` on both cards ("before") against `strata-q38` on the bigger card ("after") | today's split's speed against the new engine's; DIFF here only means `strata-q38` decoded more than 5% slower |
| `memory-q38-2cards-SB`, `-BS` | `strata-q38` on the bigger card against `strata-q38 --layer-split auto` on both cards, both with `--expert-cache 400 --pcie-frac 0 --lookahead 0` and `STRATA_PREFILL_CPU_SHARE=0` (the same experts on the GPUs: the profile's first 400, each on the card of its layer, few enough for the 3060's pool of under 1 GiB as the second card; no drafting, which on a split changes tokens at near-ties; no CPU share of a prompt) | the same tokens on two cards as on one (step M8): each card its own page pool, the verify windows and the prompt path handing their rows from card to card, a conversation paused and resumed across both cards |
| `memory-hop-SB`, `-BS` | `strata-q38` on the bigger card against `strata-q38 --layer-split 24 --gpu-mib 512,8192 --remote-experts on` on both cards, both with the settings of `memory-q38-2cards` | the second card holds the experts of the first card's layers that its 512 MiB pool has no room for and computes them in one hop (step M8): the same tokens as one card, or a DIFF from rounding (see below) |
| `memory-cut-move-SB`, `-BS` | `strata-q38` on both cards (`--expert-cache 3000 --pcie-frac 0`, pools of 12 GiB on the bigger card and 5 GiB on the smaller), a 96-token story in a batch slot, against the same with the first card's memory halved after 24 tokens and set back after 48 (`CHANGE_SETTINGS`; the answers to the changes go to `*.out.settings`) | the layer cut moves while the story decodes (step M9) and the tokens stay the same: the planner runs again on the cards' new pool sizes, the conversations' state for the moved layers goes to the other card, and each card takes its new layers' experts before the next step |
| `memory-rows-card`, `-ram`, `-ssd` | `tools\measure_rows.py` on the bigger card: as many experts on the card as fit, most of them in RAM, about half on the SSD, with the kit's own run config (`rows-config.json`: `drive.py config`, the model's files from `-DataDir`) | M7's table of conversations at once and rows per step on NVIDIA (docs/DETAILS.md has the Strix Halo's numbers). PASS: measured; FAIL: `steps\memory-rows-<case>.log` says why. `memory-rows-report` records aggregation status; incomplete or invalid files fail their case and complete files still contribute to `rows-report.txt` |

`strata-q38` runs on several cards since M8's second part (only tested on one GPU with two stages so far: the
Strix Halo has one). What the session must confirm, beyond the table's PASS:

- `memory-q38-2cards-*`: PASS (the same tokens), in both card orders. The stderr of a two-card run says
  `the page pool of gpu<N> (layers A-B)` once per card, and `--layer-split auto`'s `K=` line; the cut it chose and
  each card's pool size go in the report. Its speed against `memory-q38-1card` goes in it too (no target yet).
- `memory-cut-move-*`: PASS (the same tokens), its `*.out.settings` answers have no `ERR` and no `SHORTFALL` (the
  first session's `ERR gpu0 has room for 1.00 GiB of pool, not 2.25 GiB` came from the engine counting the prompt
  path's buffers as free memory to keep after they were allocated), and the
  `--move-cut` run's stderr has `strata-q38: the layer cut moved: K=A -> K=B (...) in N ms` once or twice (after 24
  tokens, and after 48 when the planner gives the restored sizes another cut than the start's, which `--layer-split
  auto` chose by the old engine's search). Note the lines and their times in the report. No such line: send the
  stderr's `layer cut:` lines and the `GET_MEMORY_USE` answers; a DIFF with the lines there: the stderr, as it may be
  an expert that did not fit on a card after the move.
- `memory-hop-*`: the stderr of an after run says `experts of other cards' layers, computed where they are in one
  hop: gpu<N> <count>` once, and `so far gpu<N> computed <experts> ... in one hop each layer, <ms> ms waited for` after
  each answer, with more experts each time; no run hangs (a hang: send the stderr; the engine asks for 32 CUDA
  connections, `CUDA_DEVICE_MAX_CONNECTIONS`, as a hop waits for a free one). PASS, or a DIFF whose answers read as
  sane text and part from one card's late in an answer: the hop computes those experts with the grouped kernel from
  the CPU pool's rows, and on the Strix Halo that changed one token in four answers against the same experts on the
  first card. Note the decode speed against `memory-q38-2cards-*` and the `waited for` milliseconds. Then run one
  after engine by hand with `--remote-experts auto` instead of `on` and note whether the `one hop` line appears:
  auto lends only where a hop over the card's link costs less than 0.035 ms (the Strix Halo's CPU per expert).
- A check that SKIPs says `is not there yet` in its stderr: drafting on a split is refused for now.

**Measured on this PC by hand.** Under the table, `memory-q38-1card`'s stderr (`*.out.stderr`) has a line
`strata-q38: gpu<N>'s link to RAM: X GB/s to the GPU, Y GB/s back, Z ms a round trip`, and the old engine's split
runs print `layer split auto: CUDA<n> link X GB/s`. Note both for each card (and with `nvidia-smi -q -d PCIE` whether
the 3060 sits on x4): the planner and the rules for experts on another card are priced from these numbers. The two
should now agree within about 10% (the first session's link lines read 3.4-6.8 GB/s for the 3080 on x16, against
13.4 from the old engine's probe: short copies on a link still idling at a lower PCIe generation).

## What the table says

The `memory` phase also prepares D9's DeepSeek checks. They use the build after on both sides, reset
the copied pack's profile before every start, disable drafting and adaptive placement, and cap the GPU and RAM pools.
`DS4_COMMIT_CAP_GB=30` is explicit. They never use port 8080.
Pass `-DeepSeekDir` for these D9 checks, even when running only `memory`. It names the folder containing the
DeepSeek GGUF shards and Engram, separate from the small pack under `-DataDir`. Without it, the summary includes
`memory-ds41 SKIP` and says which argument is missing. Missing model files also produce a visible SKIP.

| Check | What it compares | Required evidence |
|---|---|---|
| `memory-ds41-2cards-SB`, `-BS` | One card's 4 GiB pool against two 2 GiB pools, cut at 20, the same 240 experts | The same greedy tokens for a short prompt, its continuation and a chunked prompt; two pools in stderr |
| `memory-ds41-cut-move-SB`, `-BS` | The split against itself, shrinking the first pool by half after 24 story tokens and restoring it after 48 | The same tokens, successful settings answers and a changed cut in stderr; a missing move fails the run |
| `memory-ds41-hop-default-queues-SB`, `-BS` | Separate cards with hops on and pools of 256 and 4096 MiB, with 16 HIP queues / 32 CUDA connections against explicit runtime defaults of 4 / 8 | Both finish; stderr must show experts actually computed in hops before drawing conclusions about the queues |
| `memory-hop-default-queues-SB`, `-BS` | Qwen3.8's existing hop check with the same two queue configurations | Both finish and compute experts in hops; compare tokens, speed and wait time |

The queue checks are measurements to collect on real NVIDIA cards. No result is claimed here. A run without
any experts computed in hops does not establish whether the extra queues are needed. DeepSeek supports fixed
stages on distinct GPUs, but its expert hops and moving dense weights to another GPU are unfinished: its hop
checks refuse, and its cut-move check on distinct GPUs reports no move. Those checks remain pending. The
DeepSeek split check has not been run on real NVIDIA cards yet. Pool sizes may need adjustment to exercise a cut;
keep the same fixed expert set fitting on both sides, and record any changes with the results.

Each run's output is compared request by request (`report.py`), with times and memory-dependent numbers masked,
two batch slots' lines compared slot by slot, and a continued conversation compared only between runs that sent it
the same prompt. **PASS**: every run printed the same. **VARIES**: some runs printed something else, but a run of
the other build printed it too, or both builds vary: chance, not a change. **DIFF**: a build printed something no run
of the other build printed, while at least one build repeated itself; or every run after decoded more than 5% slower
than every run before (at least two runs a build). The note names each run that differs, from which answer token,
and "(new)" where no run of the other build printed it. **FAIL**: a run did not finish, an engine was not built, or a
test failed or was not built. **SKIP**: files or a second card missing. Hop checks require a nonzero
`computed N experts of other cards' layers` counter in every hop run (both sides of a queue comparison).
Missing proof is FAIL with a `no hops` reason. Cut-move checks require completed shrink and restore replies,
no `ERR` or `SHORTFALL`, and at least one changed cut; saved runs are checked again when reporting.

Under the table, a line per run says what the cards and the PC did while it ran (`drive.py` samples `nvidia-smi`, the
CPU, RAM, page file and disk once a second): SM clock, temperature, load, PCIe link, throttling. A DeepSeek check
that decodes below 3 tokens/s also says how many experts a token its CPU lane read from the model's files.

## Afterwards

Send back `tools\nvidia_session\results\<time>.zip` (the summary is `summary.txt` inside; each run's samples are its
`.gpu.csv` and `.sys.csv` files). Protocol runs also keep `.out.memory`: full `GET_MEMORY_USE` replies after
READY and after the scenario, plus `nvidia-smi` per-process GPU UUID, PID and memory at those points. The engine PID
is recorded. Cut-move `.out.settings` keeps the initial memory report and every resize reply, including per-card
fields. Windows drivers may report process memory as N/A; unavailable queries and their errors stay in the file.
These are allocation snapshots, not a complete allocator ledger; the kit cannot attribute an unreported remainder. Under the table it lists the commits that changed engine code, to bisect a DIFF by
building them one at a time.

To clean up, from the checkout: `git -c safe.directory=<checkout> worktree remove --force C:\sn\before`,
then the equivalent command for `after`. Replace `<checkout>` with the checkout's full path, and use your `-Work`
folder's paths.


### DeepSeek dense migration and remote experts

From the commit containing these changes, use fresh results:

```powershell
.\tools\nvidia_session\run.ps1 -Only preflight,build,memory `
  -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash `
  -Results E:\Strata-results\ds41-dense-hops
```

`memory-ds41-cut-move-SB` and `-BS` now require both successful resizes, a changed cut, and a nonzero
`dense layer L migrated: gpuA -> gpuB (N bytes)` line naming different GPUs. Token identity and speed
are compared with the fixed-cut run. `memory-ds41-remote-off/auto/on-SB` and `-BS` each compare with
one card using the same fixed expert set. Forced `on` must report computed remote experts. Auto may
choose none; inspect its CPU calibration and hop counters alongside the speed report. The existing
`memory-ds41-hop-default-queues-*` compares forced hops with extra versus explicit default queues.
DeepSeek runs check free commit and physical RAM before starting and stop only their engine PID below
3 GiB free commit or 4 GiB free physical RAM. Docker Desktop/backend or vmmem processes block a start.
The original pack is copied before its model path and expert profile are changed. No test uses port 8080.
A same-GPU split cannot validate dense transfers or hops; forced same-GPU hops remain refused.

Local validation on the Radeon 8060S (2026-11-04, Windows HIP): the single-stage, fixed split
and cut-10-to-20 runs matched all 96 output tokens. The quarter-pool stress run moved 20 -> 11 -> 20
but diverged at token 71. Settings, limits and the observed GPU expert shortfall are recorded in
[ENGINE_NOTES.md](../../docs/ENGINE_NOTES.md#deepseek-paged-stages-dense-weights-and-expert-hops).
Separate-card migration and hop execution remain for the NVIDIA kit.

DeepSeek pipeline groups use `dsgroups --conversations 1|2|3|4`. The memory phase runs
`memory-groups-ds41-<N>c-dspark-<on|off>-<SB|BS>`: groups `1` versus `auto` on two real cards,
with explicit GPU pools of 4 GiB on the first stage and 7.5 GiB on the last stage, an 8 GiB RAM pool,
`DS4_COMMIT_CAP_GB=30`, fixed expert placement (2,000 experts with DSpark on, 240 with it off) and remote experts
off. Each side compares standalone answers with two concurrent rounds and reports aggregate decode tokens/s
separately from identity. DeepSeek prompts include a repeated prefix to exercise compressed state.
One and three conversations use one group on two stages.
The PID-scoped watchdog stops only its engine below 4 GiB free physical RAM; free commit has no floor.

Run from PowerShell on the friend's NVIDIA machine (with the current checkout committed by the owner):

```powershell
.\tools\nvidia_session\run.ps1 -DataDir E:\Strata-data -DeepSeekDir E:\DeepSeek-V4.1-Flash `
    -Only preflight,build,memory -Before 30e1e7df
```
