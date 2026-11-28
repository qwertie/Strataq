# Strataq

**Run large mixture-of-experts models on your own computer, with several conversations at once.**

Strataq is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221. It runs the same models
(Qwen3.8-Flash-Next, Qwen3.6-35B-A3B and DeepSeek V4.1-Flash) with the same OpenAI- and Anthropic-compatible API, but
its engines for Qwen3.8 and DeepSeek keep conversations and experts in shared memory pools. That lets them serve
several conversations at once, start answering sooner, and change most settings while they run. (Qwen3.6 still runs
on its own engine, as in upstream.)

## What is new in Strataq

- **A shared engine core for Qwen3.8 and DeepSeek.** The code was reorganized into layers
  ([ARCHITECTURE.md](docs/ARCHITECTURE.md)), so Qwen3.8 and DeepSeek V4.1 run on one shared engine (the "paged
  engine"), and NVIDIA (CUDA) and AMD (HIP) builds compile the same sources. A fix in the shared part reaches every model.
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
  prompt chunk size and more) change while the model runs, from the Memory panel of the web page or the desktop app.
  Each setting has a tooltip that says what it is for and how to use it.
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

Measured on the Strix Halo above: the new Qwen3.8 engine was faster than the old one in all 12 single-conversation
comparisons (by 4.6 to 41.9%) and gave the same greedy answers. The new DeepSeek engine was still 6 to 29% slower than
the old one for a single conversation ([details](bench/results/2026-11-05-old-local-vs-new/README.md)).

## What it runs on

NVIDIA GeForce RTX 20 to 50 series (CUDA) and AMD Radeon cards and Ryzen AI Max (HIP), on Windows and Linux.
Most testing is on an AMD Ryzen AI Max+ 395 (Strix Halo) under Windows. Mac (Metal) and Intel Arc (SYCL) support
from upstream was removed; use [Strata](https://github.com/Niko1221/Strata) for those.

Which model fits your computer: [MODELS.md](docs/MODELS.md).

## Get started: build from source

You need Git and Python 3, plus a compiler for your GPU:

- **Windows:** the Visual Studio 2022 Build Tools with the C++ workload, and the
  [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (13.x) for NVIDIA or the
  [HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) for AMD.
- **Linux:** a C++20 compiler, CMake, and the CUDA toolkit or ROCm/HIP. On Debian and Ubuntu, setup offers to install
  missing packages.

Then clone the repository and let setup build the engine for your card and the model you pick, download the model
and start the server:

```sh
git clone https://github.com/qwertie/Strataq.git
cd Strataq
./setup.sh --build              # Linux
START-HERE.bat --build          # Windows
```

Setup asks which model to use; `--family qwen --model IQ3_S` (for example) picks one up front, and `--no-start` builds
and prepares without serving. When the server is up, open the address it prints for the chat and monitor page, or
point your tools at it: `strata connect` lists Claude Code, Codex CLI, OpenCode and others.
More options: [SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md) and [AMD_HIP.md](docs/AMD_HIP.md).

### Ready-made packages

A new release of Strataq is being prepared. The [previous release](https://github.com/qwertie/Strataq/releases/latest)
(Terminal 0.2.50) installs the `strata` command and ready-made engines for Windows AMD and Linux NVIDIA:

```sh
curl -fsSL https://github.com/qwertie/Strataq/releases/latest/download/install.sh -o install.sh && sh install.sh   # Linux
irm https://github.com/qwertie/Strataq/releases/latest/download/install.ps1 | iex                                   # Windows
```

Then run `strata launch` ([Terminal guide](docs/TERMINAL_LAUNCH.md)). It has no desktop-app installer.

## More

[All guides](docs) · [How the code is organized](docs/ARCHITECTURE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) ·
[How it works](docs/HOW_IT_WORKS.md) · [Contributing](CONTRIBUTING.md)

For AI tools: [technical reference](AGENTS.md) · [setup instructions](docs/AI_SETUP.md).

[MIT license](LICENSE). Strataq builds on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its
contributors; if Strata is useful to you, consider [sponsoring its author](https://github.com/sponsors/Niko1221).
[Credits and model licenses](docs/HOW_IT_WORKS.md#credits).
