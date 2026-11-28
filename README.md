# Strataq

**Run large mixture-of-experts models on your own computer, with several conversations at once.**

Strataq is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221. It runs Qwen3.8-Flash-Next,
Qwen3.6-35B-A3B and DeepSeek V4.1-Flash behind an OpenAI- and Anthropic-compatible API, with a chat and monitor web
page. Its engines for Qwen3.8 and DeepSeek keep conversations and experts in shared memory pools, so they serve
several conversations at once, start answering sooner, and change most settings while they run. (Qwen3.6 still runs
on its own engine, as in upstream.)

## What is new in Strataq

- **A shared engine core for Qwen3.8 and DeepSeek.** The code was reorganized into layers
  ([ARCHITECTURE.md](docs/ARCHITECTURE.md)), so Qwen3.8 and DeepSeek V4.1 run on one shared engine (the "paged
  engine"), and NVIDIA (CUDA) and AMD (HIP) builds compile the same sources. A fix in the shared part reaches every
  model.
- **Conversations take memory as they grow.** Upstream reserves memory for an entire full context window for each
  conversation. In Strataq you only pay for what you use, and conversation pages share the same memory pool as experts
  and prompt processing, allowing old ones to be evicted. It's designed for running multiple conversations at once
  to increase total throughput. On an AMD Ryzen AI Max+ 395 (Strix Halo, Windows, no drafting), one Qwen3.8 IQ3_S
  conversation decoded at 33.4 tokens/s, and four at once with drafting at 85 tokens/s in total
  ([measurements](bench/results/2026-11-05-old-local-vs-new/README.md)).
- **Conversations are paused, not dropped.** A conversation that is not decoding moves down to RAM and then the SSD,
  and comes back when it is needed. Paused conversations survive an engine restart. Higher-priority requests decode
  first.
- **Start chatting faster.** The engine says it is ready as soon as the dense weights are in memory, so you can
  see the dashboard and start a conversation as experts load.
- **One memory mode instead of several "low-RAM" modes.** The same GPU, RAM and SSD pools scale smoothly from a PC
  with little RAM to one that holds every expert on the GPU, no specialized modes or code paths. Experts move
  between the GPU and RAM by how often they are used.
- **Live settings.** Most settings (conversations at once, token budgets, the VRAM reserve, pool sizes, drafting,
  prompt chunk size and more) change while the model runs, from the Memory panel of the web page. Each setting has a
  tooltip that says what it is for and how to use it.
- **Raw context windows.** The API monitor lists every conversation the engine holds (in VRAM, RAM or on the SSD) with
  its size, and shows the exact tokens of the one you pick, with token ids, ready to save as text.
- **Several GPUs on a layer split.** Each card holds its own layers and its own page pool, loads only its own
  layers' weights, and the split can be moved while the engine runs.
- **Better drafting.** Qwen3.8 can combine MTP, suffix and lookup drafting, with a live acceptance rule. DeepSeek
  drafts with DSpark for several conversations in one pass.
- **More to watch.** Log probabilities on the API and a Probabilities tab, per-conversation activity, live RAM↔GPU
  transfers, and Generation and Prefill cards with step and expert statistics on the monitor.
- **Faster tool loops.** Qwen keeps rewind checkpoints in RAM and reuses them when an agent continues after a tool
  call.

Measured on the Strix Halo above: the new Qwen3.8 engine was faster than upstream's in all 12 single-conversation
comparisons (by 4.6 to 41.9%) and gave the same greedy answers. The new DeepSeek engine was still 6 to 29% slower
than upstream's for a single conversation ([details](bench/results/2026-11-05-old-local-vs-new/README.md)).

Strataq dropped parts of upstream it does not maintain: the desktop app, the Mac (Metal) and Intel Arc (SYCL)
engines, the all-cards split engine, the DFlash2 drafter, and the Docker, Nix and DGX Spark builds. Use
[Strata](https://github.com/Niko1221/Strata) for those.

## What it runs on

NVIDIA GeForce RTX 20 to 50 series (CUDA) and AMD Radeon cards and Ryzen AI Max (HIP), on Windows and Linux, with an
x86-64 CPU. Most testing is on an AMD Ryzen AI Max+ 395 (Strix Halo) under Windows.

## Models

The sizes below are the model's weights in RAM plus graphics memory. Leave room for the system and other programs;
on a PC whose graphics card shares the RAM (Strix Halo), count the memory once. If RAM is short, the engine reads
experts from the SSD instead, which works but is much slower.

**Qwen3.8-Flash-Next** (`--family qwen`, ISTA-DASLab packs; the default recommendation):

| `--model` | Memory for the model | Notes |
| --- | ---: | --- |
| `Q2_0` | 37.6 GB | fastest |
| `IQ2_XS` | 39.2 GB | |
| `IQ3_XXS` | 47.0 GB | |
| `IQ3_S` | 54.8 GB | best quality of the four |

Other Qwen3.8 packs: `--family unsloth` (UD-IQ4_XS to Q8_0, about 72 to 136 GB for all experts in RAM),
`--family gyro` (Gyro-S and Gyro-M: the least memory, slower), `--family coder` (IQ1_M, for coding, around 32 GB of
RAM), `--family swift` and `--family nvfp4` (NVIDIA cards). When several fit, prefer ISTA-DASLab, then Unsloth, then
Gyro.

**Qwen3.6-35B-A3B** (`--family qwen36`, Unsloth files): `UD-Q4_K_XL` (22.9 GB file, setup plans for 32 GB of RAM),
`UD-Q6_K_XL` (32.6 GB, 44 GB) and `Q8_0` (37.8 GB, 52 GB).

**DeepSeek V4.1-Flash** (`--family deepseek41`, ISTA-DAS sets): `IST-DAS-1.75bpw` (239 GB download),
`IST-DAS-2.00bpw` (257 GB), `IST-DAS-2.25bpw` (274 GB) and `ISTA-DAS-Coder` (191 GB). It needs a graphics card with
12 GB or more, at least 46 GB of RAM (64 GB is comfortable) and an NVMe SSD.

The models can read pictures (`--vision`), with a reader on the GPU or the CPU.

## Get started: build from source

You need Git and Python 3, plus a compiler for your GPU:

- **Windows:** the Visual Studio 2022 Build Tools with the C++ workload, and the
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (13.x) for NVIDIA or the
  [HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) for AMD. If setup does not find the
  build tools, set `STRATA_VCVARS` to the path of their `vcvars64.bat`.
- **Linux:** a C++20 compiler, CMake, and the CUDA toolkit or ROCm/HIP. On Debian and Ubuntu, setup offers to install
  missing packages.

Then clone the repository and let setup build the engine for your card and the model you pick, download the model
and start the server:

```sh
git clone https://github.com/qwertie/Strataq.git
cd Strataq
./setup.sh --build              # Linux
START-HERE.bat --build          # Windows (in PowerShell: .\START-HERE.bat --build)
```

Setup asks which model to use. Useful options (all of them: `--help`):

- `--family qwen --model IQ3_S` picks the model up front; `--yes` accepts the recommended answers.
- `--no-start` builds and prepares without serving; `--setup` adds another model or changes settings.
- `--data-dir DIR` says where the model files go (default: `Strata-data` beside the checkout).
- `--context 65536`, `--vision no|gpu|cpu`, `--gpus 0,1` (several cards of one make), `--backend cuda|hip`.

Setup writes a run config (`strata-<model>.json`), a start script (`run-<model>.bat` or `.sh`) and logs in the
checkout. To update, stop the server and run `UPDATE.bat` or `./update.sh`, which pulls the source, updates the
Python packages and refreshes the run configs; rebuild the engine with `--build` after the C++ changed.

## Using it

When the server is up, open the address it prints. The web page has the chat, the monitor and the Memory panel with
the live settings. Any OpenAI- or Anthropic-compatible client can use the API.

From a source checkout, use `python strata.py` in place of `strata` in the commands below
(Windows: `.\.venv\Scripts\python.exe strata.py`; Linux: `.venv/bin/python strata.py`).

- `strata connect` lists the apps it can set up (Claude Code, Codex CLI, OpenCode and others);
  `strata connect <app> --check` sends one prompt and one tool call the way that app does.
- `strata launch` sets up, serves and monitors in guided screens. With flags it does the same without asking, for
  example `strata launch --family qwen --model IQ3_S --yes --no-tui`. `--plan` prints what it would do and changes
  nothing; `--status`, `--stop` and `--show-key` do what they say.
- `strata monitor` shows the server's activity in the terminal, and `strata --help` lists the other commands.

The server listens on `127.0.0.1` only. `--access lan` (or `cloudflare`, `tailscale`) makes it reachable from other
computers and creates an API key.

## Ready-made packages

A new release of Strataq is being prepared. The [previous release](https://github.com/qwertie/Strataq/releases/latest)
(Terminal 0.2.50) installs the `strata` command and ready-made engines for Windows AMD and Linux NVIDIA:

```sh
curl -fsSL https://github.com/qwertie/Strataq/releases/latest/download/install.sh -o install.sh && sh install.sh   # Linux
irm https://github.com/qwertie/Strataq/releases/latest/download/install.ps1 | iex                                   # Windows
```

Then run `strata launch`.

## Working on the code

How the code is organized and where new code goes: [ARCHITECTURE.md](docs/ARCHITECTURE.md). The rules for agents and
contributors, and how to run the tests: [AGENTS.md](AGENTS.md). Engine settings and measurements:
[ENGINE_NOTES.md](docs/ENGINE_NOTES.md). Making a release: [RELEASING.md](docs/RELEASING.md).

## License and credits

[MIT license](LICENSE). Strataq builds on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its
contributors; if Strata is useful to you, consider [sponsoring its author](https://github.com/sponsors/Niko1221).
Third-party code and model licenses: [CREDITS.md](CREDITS.md).
