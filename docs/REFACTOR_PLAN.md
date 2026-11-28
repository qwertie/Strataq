# The code-sharing refactor

This fork reorganizes Strata so that code is written once and shared: between the three model engines
(Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, DeepSeek V4.1-Flash), between the GPU backends (CUDA and HIP)
and between the Python programs. The SYCL (Intel) and Metal (Mac) backends have since been deleted. The goal is
less code to read, change and test. Success is measured in lines of code (below).

- The target layout and its rules: [ARCHITECTURE.md](ARCHITECTURE.md).
- Which old path goes where, and how to carry an upstream change over: [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md).

## Measuring

`python tools/loc_report.py` counts the non-blank lines of Strata's own code at a git revision. It leaves out
copied third-party code (`third_party/`, `*/vendor/`, Intel's `sycl/include/dpct/`), community results
(`bench/results/`) and the desktop app (`apps/`, which this work does not touch).

**Baseline:** `06fcbe98`, the last commit before this work: **493,148 lines** (C++ 249,138, Python 155,555,
CUDA 72,947, other 15,508).

After each step: `python tools/loc_report.py --compare 06fcbe98 HEAD` (add `--detail` for one row per folder).
Record the change in the [results](#results) table.

## Order

Steps are ordered by **risk first, lines saved second**. Moving a file without changing it is safe at any time,
because a broken include fails the compile checks. Changes to what NVIDIA-only code does wait for one batched test
session on an NVIDIA machine. Changes nobody can test come last.

"Est." is a rough estimate of lines saved, from file sizes and duplicate counts. It is not a measurement.

### Phase A: fully testable on the owner's PC (AMD Strix Halo, Windows)

| Step | Work | Est. | Status |
|---|---|---:|---|
| A0 | `tools/loc_report.py`, the baseline, this plan | — | done |
| A1 | Skeleton: the new folders with a README each, [ARCHITECTURE.md](ARCHITECTURE.md), [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md), the layering test | — | done |
| A2 | Python: `strata_base/`, `strata_common/`; `serve/` becomes `strata_server/`; remove the import cycles between `serve/`, `strata_control/` and `strata_cli/`; one copy of the helpers repeated in `setup.py` and `tools/` | 2–5K | |
| A3 | Move every whole file whose new place is already clear, without changing it: the folders `src/platform`, `src/artifact`, `src/spec`, `src/dsv41`, `src/qwen35`, `src/ngram`, `src/plan`; from `src/core/`, the runtime files to `src/base/gpu`, the conversation files to `src/common/state` and the Qwen3.8-only files to `src/models/qwen38`; the matching files in `sycl/`. Files that must be split wait for their step. | 0 | |
| A4 | DeepSeek's own copies of host code: `ds4_os.hpp`, `ds4_pool.hpp`, `include/strata/models/deepseekv41/sampler.hpp` | ~1K | `ds4_os.hpp` done; the pool and the sampler wait (open issues) |
| A5 | The `src/base/gpu` runtime interface (the shared code must not call the CUDA runtime directly), then `src/common/state`, `src/common/serve`, `src/common/spec` and the one-GPU parts of `src/common/experts`. These are also the base for the planned paged-KV engine. | 3–7K | serve: all three engines; state: descriptor done, Qwen3.6 and Qwen3.8's running state on it, Qwen3.8's K/V left; spec: window planner; experts: one-GPU helpers (ExpertCache waits for the runtime interface) |
| A6 | SYCL: the 7 files that are 95% or more the same as their `src/` original become shared | ~2.4K | done: 8 files; moot since `sycl/` was deleted |

### Phase B: changes NVIDIA-only code; one batched test session on an NVIDIA machine

| Step | Work | Est. | Status |
|---|---|---:|---|
| B1 | Split `src/kernels/` and `src/prefill/` by what they depend on; remove Qwen's `geometry.hpp` from the 12 shared kernels | small | done: moves only; the 11 GPU kernels that include `geometry.hpp` are Qwen3.8's (or the Qwen families') and moved to `src/models/`, the CPU expert path's check moved to Qwen3.8's session |
| B2 | DeepSeek's own quant kernels (`ds4_iq`, `ds4_gguf`, `ds4_tq`) and its sampler (`vendor/sampler.cu`) replaced by the shared ones | 3–5K | sampler's plain draw shared; quant kernels kept (open issues); unbuilt CPU rows removed |
| B3 | The multi-GPU parts of `src/common/experts` (`peer_experts`, `remote_experts`) | ~0.5K | kept in `src/core` (they use Qwen3.8's kernels and the CUDA runtime directly); duplicates merged, the SYCL copy of `peer_experts.hpp` dropped |
| B✓ | Test session on an RTX 30-series machine with several cards: parity tests on sm_86, sm_75/80 through `STRATA_EMULATE_CC`, multi-GPU, a DeepSeek speed comparison. Ada (sm_89) and Blackwell (sm_120, NVFP4) stay untested unless such a card is found. | | |

### Phase C: removed

C1 (share the other 77 SYCL copies of `src/` files) and C2 (generate the SYCL kernels from the `.cu` sources) are moot:
`sycl/` was deleted.

### Phase D: DeepSeek V4.1 shaped like Qwen3.8

The goal is that `strata-ds41` and `strata-q38` share names, interfaces and file layout, and the shape of the decode
step, before more speed work. The math stays the same: the same kernels, the same order of sums, the same tokens.
Each step is checked by greedy answers of both builds compared token for token (one start with DSpark off: 1, 2 and 4
conversations, route prior 0 and 1000; one with DSpark drafting), and by decode speed in alternating starts.

| Step | Work | Status |
|---|---|---|
| D-1 | Delete what `strata-dsv41` alone used from `ds4::Model`: its API, `--parallel`, the slots, its RAM store, the PCIe lane, the VRAM swaps, the lookahead, tensor-parallel sums | done: `ds4_model.cu` 16,139 → about 9,200 lines; the kernels only it called went too |
| D-2 | Split `ds4_model.cu` by job (loading, prompt path, CPU lane, DSpark, window decode, paged entry points) behind a private header | done: `ds4_model_impl.hpp` and `ds4_{model,load,prefill,verify,cpu_lane,dspark,engram,vision_tower,checkpoints}.cu`; function bodies unchanged |
| D-3 | One place for DeepSeek's `DS4_*` settings; CUDA calls through `src/base/gpu` | not done: Qwen3.8 breaks the same two rules as much (about 420 CUDA runtime calls in `verify.cpp` and `model.cpp`, 160 environment reads), and `src/base/gpu` offers only memory, copies and stream waits, so doing it for DeepSeek alone would make the two less alike. It needs one settings reader and a wider runtime interface for both families first |
| D-4 | Share per-card execution through `CardStage` (`launch`, `poll`, `wait_writes`) and `CardPipelineModel`, which both paged models derive from | done: `include/strata/models/common/card_stage.hpp`; preparing, finishing and keeping a group stay in each family, which do them for the whole group |
| D-5 | A shared CPU-expert service | not planned: see the open issue below |

## Checks for every step

| Check | Where | How |
|---|---|---|
| Python tests | this PC | `.venv/Scripts/python -m pytest -n 16 tests/terminal tests/architecture strata_server --ignore=tests/terminal/test_lifecycle_matrix.py` (that file leaves Monitor windows open when its flaky tests fail; run it on its own), and the tests in `tools/`, `tests/desktop`, `tests/time_skip` and `tests/tuning` one file at a time (`for f in $(git ls-files 'tools/test_*.py' 'tools/**/test_*.py' 'tests/desktop/test_*.py' 'tests/time_skip/test_*.py' 'tests/tuning/test_*.py'); do .venv/Scripts/python -m pytest -q "$f"; done`: pytest-xdist cannot collect `tools/` as a whole). pytest and pytest-xdist are installed in the repo's `.venv`. At the baseline, about 75 tests in `tests/terminal` and 12 in `tools/` already failed on this PC (process handling on Windows, a missing llama.cpp `gguf-py`), and a few time-sensitive ones fail now and then under `-n 16`, so compare the failures by name with a run of the previous commit |
| HIP build and GPU tests | this PC | `build-hip-win` (gfx1151, tests on): build the targets the step touches, run their `*_parity` and `*_test` programs |
| CUDA compile | Docker on this PC | the command below; about 3 minutes after the first build |
| Speed | this PC | decode and prompt speed within noise of a same-day run of the previous commit, for any step that touches engine code |

The CUDA image is made once with an inline Dockerfile (`nvidia/cuda:13.0.0-devel-ubuntu24.04` plus
build-essential, git, python3-venv, cmake 4.4.3 and ninja 1.13.2), tagged `strata-cuda-dev`. It builds into a Docker
volume, so nothing is written to the checkout. From Git Bash:

```sh
# CUDA: every target, both NVIDIA code families (sm_75, sm_120) and the test machine's sm_86, the DeepSeek engine
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/work/Strata" -v strata-cuda-build:/build strata-cuda-dev bash -c '
  cd /work/Strata && ( [ -f /build/build.ninja ] || cmake -G Ninja -S . -B /build -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
    -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON "-DCMAKE_CUDA_ARCHITECTURES=75-real;86-real;120-real;120-virtual" \
    -DSTRATA_GGML_DIR=/work/Strata/third_party/llama.cpp ) && cmake --build /build -j 32 && cmake --build /build --target strata_kq'
```

A move that changes CMake paths can leave a stale build in a volume: delete it with `docker volume rm
strata-cuda-build` and build again.

## Rules for each step

- One subsystem per commit, so that a reviewer can read the work commit by commit.
- A commit that fills a new folder also updates that folder's README and the path map in
  [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md).
- The commit message says what the step changed in `loc_report.py --compare`.
- Upstream's [CONTRIBUTING.md](../CONTRIBUTING.md) rules about byte-identical default paths and agreeing a
  refactor in an issue first do not apply to this fork; the scope will be raised in an upstream issue.

## Open issues

Found during the work; each needs a decision or a later step.

- **Two rules for the model folder.** `setup.py` (started by `START-HERE.bat`) picks the data folder with
  `data_folder` (default `Strata-data` beside the checkout's parent) and ignores the app's saved choice and
  `Documents/Strata`; `strata_control/discovery.py` (`data_dir`) reads both. A source install and the app can therefore
  use different model folders. `strata launch` is not affected (it always passes `--data-dir`). The owner wants one
  rule; not yet scheduled.
- **Ready-made engines in a source checkout** (deferred to phase B, needs NVIDIA testing and an owner decision):
  `setup.py` still downloads GitHub zips into `engine/` and keeps `engine/.previous`, while the Terminal and the app use
  the signed feed and the per-user components folder. Unifying them would remove about 600–850 lines, but drops
  `setup.py --prebuilt <mirror or dist folder>` and may affect the CUDA 12 engine.
- **A test that downloads for real.** `tools/test_setup_dsv41.py::*::test_no_engine_stops_before_the_download` (4
  tests) reaches the live signed feed; since a DeepSeek engine 0.1.52 package for Windows AMD appeared there, each run
  downloads it (137 MB) and the tests fail, at the baseline commit too. The test needs a local fake feed.
- **A third AMD reader.** `strata_common/telemetry.py` has its own Windows display-adapter reader, which works
  differently from `engine_component`'s (it keeps cards with 2 GB or more and reads the vendor from the provider name).
- **Two CPU worker pools, not merged.** They share an idea (one 64-bit claim word tagged with the batch, so a
  late worker cannot take a job of the next batch; spin, then sleep), but differ where the speed is decided:

  | | DeepSeek `ds4::Pool` (`ds4_pool.hpp`) | Qwen `ExpertPool` (`src/common/kernels/cpu/pool.cpp`) |
  |---|---|---|
  | Jobs | any `fn(i)` through `std::function`; `start`/`help`/`wait` run two pools (two NUMA nodes) at once | expert jobs in 7 modes (whole experts, row parts, native rows), one scratch per worker |
  | Claim | `generation << 32 \| index`, the job count in its own atomic | `epoch \| njobs \| index`; per NUMA slice with `STRATA_NUMA_SPLIT` |
  | End of a batch | the caller waits for `done` | the host waits for `done` and for every worker to park |
  | Idle workers | spin 20 ms (the CPU lane) or 50 µs (the prompt path's readers), then sleep | spin 20 ms (`STRATA_POOL_SPIN_US`), then sleep |
  | Placement | the caller's CPU list, hard affinity, Windows processor group 0 | its own: topology, hybrid cores, `--pool-cores`, `--host-core`, processor groups |
  | Stalls | none | 60 s without progress stops the engine through the watchdog hooks of `include/strata/common/state/progress.hpp` |

  What the two have in common is only the waiting: the threads, the spin, the sleep and the wake-up. A
  `WorkerPool` with that and nothing else was written on 2026-10-29 (not compiled or run): every other row of the
  table stays with its owner, so the owner hands it three hooks (a thread's start, its turn at a batch, its sleep).
  It made the code longer: the pool header had 159 non-blank lines, `ExpertPool` and `ds4::Pool` shrank by 151,
  so +8 lines in all (`tools/loc_report.py`). Sharing more (the claim word, the park barrier, the stall wait) means
  one claim layout for both, which would change `ExpertPool`'s issue #29 protocol or `ds4::Pool`'s 32-bit job count,
  and the NUMA paths of both cannot be measured on a one-node PC. Not worth doing for the line count alone.
- **The CPU lanes of Qwen3.8 and DeepSeek do the same job differently** (from reading the code, 2026-11-28). Qwen's
  graph rings a doorbell, the host decides which experts the GPU computes and which the CPU does (`GpuPlanSink`),
  the pool's jobs are compiled for Qwen3.8's shape (`src/common/kernels/cpu/expert.hpp`), and the GPU adds the
  experts' rows up. DeepSeek's GPU decides (`plan_ptr`), a thread of its own computes the CPU's experts on `ds4::Pool`
  and adds them up on the host, weighted per token, before the GPU waits for them (`wait_seq`). One shared service
  would have to choose one place for the sum (the order of float adds, so the tokens), one plan, one thread model and
  one mailbox. They already share the CPU row kernels. A row-job entry point that takes the model's shape at run time
  (in `src/common/kernels/cpu`) is the first piece worth sharing, when one of the two changes anyway.
- **DeepSeek's draft sampling is its own.** Since B2, DeepSeek's plain draw is the shared `sample_tokens` (the same
  tokens: 10,800 rows of random logits, and 136 sampled draws of the engine replayed from its logits, gave the same
  token from DeepSeek's old kernel and from each of the shared sampler's three paths, 2026-10-29, Strix Halo).
  `src/models/deepseekv41/vendor/sampler.cu` keeps the speculative sampling of the NextN / DSpark drafts (`SpecList`,
  `sample_draft`, `sample_verify`, `DS4_DSPARK_SAMPLE=1`). The shared sampler's probabilistic acceptance
  (`spec_draft_sample`, `sample_tokens_spec`, `STRATA_SPEC_PROB=1`) does the same job differently: one draft row at a
  time through a ring of mapped memory, and other Philox streams (the counter, not the seed, is tagged), so moving
  DeepSeek onto it would change its sampled tokens and add the ring set-up to the engine. Not done.
- **DeepSeek's quant kernels are not copies of the shared ones.** `ds4_iq.cu`, `ds4_gguf.cu` and `ds4_tq.cu` round
  the activations to int8 with an F32 scale per 32 values and sum each 32-value block exactly in integers; the
  shared kernels (`src/common/kernels/cuda/iq_kernels.cu`, `src/common/kernels/prefill/`) take llama.cpp's `q8_1`
  activations (an F16 scale) and its `vec_dot` arithmetic. Their arguments also differ: DeepSeek's take an expert's gate, up and down in
  one blob, groups of window entries and its MXFP4, trellis and prompt-set layouts. Using the shared kernels would
  change DeepSeek's numbers, so B2 kept them. The CPU lane already uses the shared CPU kernels (`ds4_gg_cpu.cpp`:
  `iq256x_rows` / `iq512x_rows`; `ds4_tq_cpu.cpp`: the trellis rows of `tq_cpu.hpp`). Left for the owner: the gfx12
  WMMA trellis form `ds4_tqw.cuh` (138 lines) is used only by `tests/dsv41/tqw_hip_test.cpp`, which no build
  compiles.
- **Qwen3.6 drafts nothing at temperature 0.** The server adds the model card's `presence_penalty: 1.5` for
  `qwen35moe` (`MODEL_CARD_SAMPLING` in `strata_server/server.py`); any penalty sends the Qwen3.6 engine down its
  sampling path, which skips MTP drafting (`src/models/qwen36/q36_engine.cu`). Measured 2026-10-28 on the Strix Halo,
  UD-Q4_K_XL, 2,000-token prompt: 47 tok/s decode with the penalty, 65 tok/s with `presence_penalty: 0`. Older than the
  refactor; may be intended.
- **A Qwen3.6 prompt read that never ended.** On 2026-10-28, in one of four server runs of the build before
  `9f3ac8a3` (UD-Q4_K_XL, Strix Halo), a greedy 2,031-token prompt read gave no output for 341 s, after the same
  request had taken 2.5 s in the run before; the server ended the engine (`engine_silence_s`).
  Not seen again; cause unknown.
- **DeepSeek is not deterministic on this PC.** Two runs of the same build give different long answers at
  temperature 0 through the server (2026-10-29, IQ2_XXS, Strix Halo): the experts kept in graphics memory are counted
  from free VRAM at start, and each run rewrites the pack's `expert-profile.bin`. Engine changes to DeepSeek can
  therefore be compared token for token only on short answers or with `--expert-profile none --no-expert-cache`.
  Even then, sampled answers (fixed seed) and some continued conversations differed between runs of one build
  (2026-10-29, three runs a build: e.g. `temperature=0.8 top_p=0.9 top_k=30 min_p=0.05 seed=7` on a story prompt, as
  the first request of the engine); greedy answers to fresh prompts did not.

## B✓ test list

What the NVIDIA session has to run for the phase B steps done so far. It was written before the original Qwen3.8
engine (`strata`, `src/program/generate.cpp`) and the old DeepSeek engine (`strata-dsv41`) were deleted; the runs
through the engine's `--serve` below meant those engines. Build as in [the checks](#checks-for-every-step)
(CUDA, `-DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON`); compare each against the same run of the commit before
the step. [tools/nvidia_session](../tools/nvidia_session/README.md) runs the whole list in one session, against the
build of `87d20a5c` (before any C++ moved), and names the commits to bisect where they differ.

The machine (a friend's, as told on 2026-10-29): Windows 11, an RTX 3060 12 GB and an RTX 3080 said to have 20 GB (not
a standard model; check with `nvidia-smi`), 48 GB of RAM. Both cards are sm_86.

- Build there natively (the CUDA 13 toolkit and the Visual Studio 2022 Build Tools); WSL2 with the Linux build is the
  fallback. Add `86-real` to `CMAKE_CUDA_ARCHITECTURES`; the Docker compile check here should include it too.
- sm_75 and sm_80 through `STRATA_EMULATE_CC`. Paths for sm_90 and later (the cluster argmax of B2, NVFP4 on sm_120)
  do not run on these cards and stay untested.
- Two different cards test the layer split and the multi-GPU expert paths (B3) with an uneven split. GeForce cards on
  Windows usually have no peer-to-peer access, so the peer-expert code runs its fallback path; the direct path needs
  Linux or workstation cards.
- DeepSeek: the IST-DAS 2.00 bpw set, the one measured here (setup's guide: 40 GB of RAM; an RTX 3060 with 46 GB of RAM
  was measured at about 8-10 tokens/s). About 257 GB on an NVMe drive; the owner's files can be carried over and given
  with `--gguf-dir`.

- **B3, the multi-GPU expert paths and the page-locked RAM copy.** What changed, each meant to make the same CUDA calls
  with the same arguments in the same order:
  - `src/core/peer_experts.cpp` (`--peer-device`): its own device switch `On` became the shared `OnDevice`. The
    `cudaSetDevice` calls are the same; the redundant `cudaGetDevice` queries before each switch back are gone.
  - `src/program/generate.cpp`: the helper caches' context start-up (`RemoteExperts::preflight`, two loops: with and
    without `STRATA_EARLY_REMOTE_CONTEXTS=1`), a later stage's `VerifyHits` (two places: the stage's verifier and its
    `--pipeline-windows` twin) and the `STRATA_SPLIT_OWN_BUFFERS` switch (two places) are each written once.
  - `src/core/expert_cache.cpp` (`--vram-elastic`, NVIDIA only): the driver functions are looked up through
    `find_driver_entry` of `src/base/gpu/vmm.cpp` (the same query `--kv-grow` uses). Never run on AMD: HIP builds
    have no segmented cache.
  - `src/core/expert_source.cpp` (`pin_cache_complement`): the two ways of page-locking a prefix of the resident RAM
    copy (the stepped one of `STRATA_PIN_GUARD`, Linux only; `STRATA_PARTIAL_PIN=1` with `--resident-budget-gib`)
    share `cut_at_blob` and `register_prefix`.

  On the two-card machine (call the RTX 3060 12 GB CUDA0 and the 3080 CUDA1 as `nvidia-smi` numbers them; swap them
  once for the other order), with the Qwen3.8 IQ3_S pack and its expert profile, through the engine's `--serve` and
  the same requests for both builds (greedy, a 2,000-token prompt, then a 128-token answer, and a second turn that
  reuses the prompt). For each run compare the tokens and the startup lines (the split plan, the slot and GiB counts
  of every card, "keeps its own prompt buffers", "peer GPU ...", "CUDA1 context ready") with the build of `7e3a3248`,
  and decode tok/s within noise:
  1. Layer split, `"gpu": [0, 1]`: `--layer-split auto`, then the uneven `--layer-split 12` (the first card 12 of 48
     layers); each once more with `STRATA_SPLIT_OWN_BUFFERS=0` (every stage borrows), with `--pipeline-windows 1`,
     and with `--resident-experts` (the stages' lend regions in the RAM copy).
  2. Peer tier: `--peer-device 1` (no `--layer-split`). GeForce cards on Windows are expected to report no peer
     access, so the prompt path's host route runs (activations and rows through mapped host memory); check the log for
     the peer's expert count and that a long prompt works. Then `STRATA_PEER_DIRECT=0` (the result rows copied back
     instead of written straight into host memory) and `STRATA_PEER_NO_P2P=1` (same route forced; on these cards it
     should change nothing). The direct path and the test copy of `p2p_copies_data` need P2P and stay untested.
  3. Helper cache: `--expert-cache-device1 auto`, then with `--remote-expert-opt`, then with
     `STRATA_EARLY_REMOTE_CONTEXTS=1`, then with `STRATA_PREFILL_HELPER_SOURCE=1`.
  4. One card, NVIDIA only: `ctest -R "vmm_test|expert_cache_segmented_test"`, and the engine with `--vram-elastic` and
     with `--kv-grow` (both start; the same tokens as without them).
  5. The page-locked RAM copy on Windows: `--resident-experts --resident-budget-gib 30` with `STRATA_PARTIAL_PIN=1`;
     the log's "GiB of it registered for the GPU (the hottest)" must give the same size as the build before. The
     stepped path (`STRATA_PIN_GUARD=2`) runs only on Linux (WSL2 will do).
- **B2, DeepSeek's sampler** (`include/strata/models/deepseekv41/sampler.hpp`): the plain draw went from DeepSeek's
  one-block kernel to the shared `sample_tokens`: greedy rows take the cluster argmax on sm_90 and later
  (`sampler_greedy_cluster_kernel`, never run on AMD), sampled rows the split top_k. Run `sampler_parity --selftest`
  and `decode_cluster_parity`; then the DeepSeek engine with `--expert-profile none --no-expert-cache`: greedy
  requests, sampled requests with a fixed `seed` (for example `temperature=0.8 top_p=0.9 top_k=30 min_p=0.05
  seed=7`), each with `--dspark auto` and `--dspark none`, and once with `DS4_DSPARK_SAMPLE=1`. The tokens must be
  the same as before the step wherever two runs of the build before agree with each other (on the Strix Halo some
  sampled answers did not, see the open issue on DeepSeek). Speed: the decode tok/s of a 128-token greedy answer,
  three runs a build, alternating.
- **B2, `ds4_iq_cpu.cpp`**: no NVIDIA-only code changed (the removed rows were never compiled).
- **B1, the split of `src/kernels/` and `src/prefill/`**: files moved, no kernel code changed, so no sm-specific path
  changed. The one code change is a compile-time check: the CPU expert path's widths are checked against
  `core/geometry.hpp` in `src/models/qwen38/session.cpp` instead of in `common/kernels/cpu/expert.hpp`. Run the whole
  `ctest` once to see that every moved test is still built and registered; the moved ones that only a CUDA build has
  were not run on the AMD PC: `decode_cluster_parity`, `dequant_wide_parity`, `gdn_rec_parity`, `gemm_bf16_parity`,
  `gr_upmix_parity`, `mmvq_il_parity`, `mmvq_multi_parity`, `pdl_parity`, `pf_wmma_parity`, `verify_parity`, and the
  QSA tests built with `STRATA_ENABLE_CUDA` (`qsa_prompt_attn_parity`, `qsa_topk_active_parity`, `qsa_slots_parity`,
  `qsa_merge_daily_parity`, `qsa_scores_parity`, `qsa_select_bench`, `gdn_chunk_parity`).

## Running engines and GPU tests on this PC

Windows loads the driver's `System32mdhip64_7.dll` before anything on `PATH`, so an engine or GPU test run from
`build-hip-win` gets the wrong HIP runtime (the engine exits with 0xC0000139; tests hang or crash). Copy the program
into a folder together with `engine\*.dll` (as `engine\strata.exe` is installed) and run it there. End-to-end checks
use a copy of the run config with `"exe"` pointing at the engine under test and the server on another port (8091), with
alternating runs of the two builds. Models on this PC: Qwen3.8 IQ3_S (`strata-iq3_s.json`) and Qwen3.6 UD-Q4_K_XL
(`../Strata-data/models/qwen36-UD-Q4_K_XL`); DeepSeek V4.1 IST-DAS 2.00 bpw (`../DeepSeek-V4.1-Flash`, run config `strata-deepseek-v4.1-flash-ist-das-2.00bpw.json`; build `strata-ds41` with `STRATA_BUILD_DSV41=ON`).

## Results

| Step | Commit | Lines after | Change from baseline | Notes |
|---|---|---:|---:|---|
| baseline | `06fcbe98` | 493,148 | — | |
| A0 | `e56c4aed` | 493,264 | +116 | the line counter itself |
| A1 | `d757595f` | 493,367 | +219 | the layering test |
| A2 | `f743b698` | 493,434 | +286 | moves only: imports split where some names moved |
| A2 | `6333ec2a` | 493,526 | +378 | the engine client moved out of `serve/server.py` (8,901 → about 6,100 lines) |
| A2 | `27d2e2eb` | 493,565 | +417 | last upward Python imports removed (launch plan, bench, status client, gguf_reader) |
| A2 | `5d272302` | 493,565 | +417 | `serve/` renamed `strata_server/` |
| A2 | `f63a0488` | 493,170 | +22 | one copy of each duplicated Python helper (−395) |
| A2 | `4e991378` | 493,009 | −139 | setup.py uses the shared settings, file, RAM and DeepSeek helpers |
| A2 | `906b00b2` | 492,982 | −166 | model catalog moved to `strata_common/catalog.py` |
| A2 | `87d20a5c` | 492,885 | −263 | setup.py's AMD detection built on `engine_component` |
| A3 | `d1b2c555` | 492,894 | −254 | whole C++ folders moved (moves only) |
| A3 | `0bb91482` | 492,909 | −239 | 98 files moved out of `src/core` (moves only) |
| A4 | `874ed84c` | 492,906 | −242 | DeepSeek's OS helpers moved to `base/platform`; pools not merged (see open issues) |
| A6 | `89a96beb` | 491,154 | −1,994 | 8 SYCL copies replaced by the shared sources (−1,752) |
| A5 | `b17c98fd` | 491,394 | −1,754 | the shared command loop (`src/common/serve`), Qwen3.8 on it |
| A5 | `57a2e1e0` | 491,316 | −1,832 | Qwen3.6 on the shared command loop |
| A5 | `9f3ac8a3` | 491,505 | −1,643 | the state descriptor (`state_layout.hpp`), Qwen3.6 on it |
| A5 | `27083334` | 491,519 | −1,629 | Qwen3.8's running state on the descriptor (its K/V not yet) |
| A5 | `dc7d7c9f` | 491,494 | −1,654 | one window planner for both Qwen3.8 decode loops (`src/common/spec`) |
| A5 | `268c2fd2` | 490,818 | −2,330 | one-GPU expert-store helpers shared (`src/common/experts`), SYCL copies dropped |
| A5 | `62bb0aac` | 490,952 | −2,196 | DeepSeek on the shared command loop (after the DeepSeek 6-shard support, +146, and two DeepSeek fixes) |
| A2b | `043ccf2e` | 490,670 | −2,478 | a second Python sweep: shared atomic writers, hashing, GPU probes, test fixtures |
| B2 | `b1002bd0` | 489,879 | −3,269 | DeepSeek's plain draw on the shared sampler; ~710 never-compiled lines removed |
| B1 | `35311604` | 486,846 | −6,302 | `src/kernels` and `src/prefill` split into the layered folders; 5 never-compiled SYCL copies removed |
| B3 | `46a56add` | 486,724 | −6,424 | multi-GPU expert code's duplicates merged (it stays in `src/core`) |
| D-1 | `7f4789df6` | 493,274 | +126 | the old DeepSeek engine's code left in `ds4::Model` deleted (−8,409; the merge of `main` before it had added Strataq's later work) |
| D-1 | `4349f4c4f` | 493,302 | +154 | the kernels only the old DeepSeek engine called deleted (−384); counted after D-2 (+296, `82d2bb514`) and D-4 (+77, `69accdf9f`) |
