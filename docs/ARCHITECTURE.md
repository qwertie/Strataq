# How the code is organized

Strata's code is arranged in layers, from all-purpose code at the bottom to programs at the top. Code may use
the layers below it, never the ones above it. The layout is being introduced step by step
([REFACTOR_PLAN.md](REFACTOR_PLAN.md)); until it is complete, some code is still in the old folders listed
[at the end](#old-folders).

Folder names are chosen so that, in alphabetical order, shared code comes before the code that uses it.

## Two stacks that meet at run time

- **The engine** is C++ (with CUDA/HIP, SYCL and Metal for the GPU). Sources are in `src/`, headers in
  `include/strata/`, and the two mirror each other: the headers of `src/common/state/` are in
  `include/strata/common/state/`.
- **The programs around it** are Python packages at the top of the repository: the server, the `strata` command
  (the Terminal) and setup. They are at the top because they are import roots, and the release packages ship them
  under these names.
- Neither stack imports the other. The server starts an engine program (built from `src/program/`) and talks to
  it over the engine's standard input and output. What both sides must agree on is in `contracts/`.
- The desktop app (`apps/`) sits on top of both and is not part of these layers.

## The C++ layers

| Layer | Folder | What belongs here | May use |
|---|---|---|---|
| 0 | `src/base/` | All-purpose code with no inference knowledge | `third_party/` only |
| 0 | `src/base/gpu/` | The device runtime interface (memory, copies, streams, events, graphs), one implementation per backend; GPU hardware traits; the HIP compat headers | `src/base/platform/` |
| 0 | `src/base/platform/` | Operating-system services: files read at an offset, mapping, NUMA, thread pinning, free RAM, worker pools | nothing |
| 1 | `src/common/` | Inference code that does not depend on a model | layer 0 |
| 1 | `src/common/artifact/` | GGUF reading, quantization block formats, CPU reference dequantizers | |
| 1 | `src/common/experts/` | The expert store: RAM tier, VRAM slots, prefetch, eviction | |
| 1 | `src/common/kernels/` | Device kernels that work for any model | |
| 1 | `src/common/sample/` | The sampler chain, logit bias | |
| 1 | `src/common/serve/` | The engine protocol, request queue, scheduler, streaming, cancel; and the shared paged engine | |
| 1 | `src/common/spec/` | Speculative decoding: drafting, accept/reject, rollback | |
| 1 | `src/common/state/` | Per-conversation state: the state descriptor, page pool, snapshots, swap, conversation files | |
| 2 | `src/models/common/` | Code shared by several model families: model geometry, tensor layout tables, GDN / gated attention / MoE-routing layers and their kernels | layers 0–1 |
| 3 | `src/models/<family>/` | One model family's engine: `deepseekv41`, `qwen36`, `qwen38` | layers 0–2; never another family |
| 4 | `src/program/` | One thin `main()` per program | layers 0–3 |

Inside `src/common/`, the subfolders may use each other as long as they do not form a cycle.

### Two command loops in the tree

An engine program is a command loop plus a model. There are two command loops today, and both speak the server's line
protocol (`contracts/`), so the server does not have to know which one it started.

- **The paged engine** — `src/common/serve/paged_engine.cpp`, with its start in `paged_start.cpp`. Conversations live
  in page pools (GPU, RAM, SSD): several decode in one step, and one that is not decoding is paused down a tier
  instead of dropped. `strata-q38` (`src/program/q38_main.cpp`) runs it for Qwen3.8 and `strata-ds41`
  (`src/program/ds41_main.cpp`) for DeepSeek V4.1, each through its own `PagedModel`. **New engine work goes here.**
- **The older engine** — the `main()` of `src/program/generate.cpp` (`strata`), and the same shape inside Qwen3.6's
  `Serve` (`src/models/qwen36/q36_engine.cu`) and DeepSeek V4.1's `serve` (`src/models/deepseekv41/ds4_main.cpp`).
  Each conversation holds a session allocated for the whole context, so conversations at once counts sessions
  (`--batch`); the paged engine counts rows instead, and its `slots` is that capacity, not an allocation. It is still
  what the installers ship, and what most of [DETAILS.md](DETAILS.md) describes.

Both expose the same [request-ID protocol](../contracts/requests.md). The paged engine owns placement and creates
conversation state when it is needed. Older engines retain their execution structures behind a shared adapter.
The Python client does not choose a solo path or an execution slot for an updated engine.

`src/common/serve/README.md` lists which parts each one uses. When a doc says "session", "slot" or "`--batch`", check
which loop it means.

### Other engine backends

- `metal/` is the Mac engine. Its host code compiles the shared `src/` sources unchanged through
  `metal/include/metal_compat`; its kernels are `.metal` files.
- `sycl/` is the Intel engine. It is mostly translated copies of `src/` files. It takes the same layout as
  `src/`, and its copies become shared sources only where that can be tested ([REFACTOR_PLAN.md](REFACTOR_PLAN.md),
  steps A6 and C1). Like Metal, it compiles a shared file unchanged where it can: `sycl/include/sycl_compat` goes
  first on its include path and gives the CUDA names a shared header uses their SYCL types.

## The Python layers

| Layer | Package | What belongs here | May import |
|---|---|---|---|
| 0 | `strata_base/` | All-purpose code: processes, files, HTTP, hashing, downloads | nothing of Strata's |
| 1 | `strata_common/` | Strata-wide code: GPU detection, installing and finding engines, the model family catalog (`catalog.py`: the families, sizes, contexts and file pins), the GGUF reader, the engine protocol client, the client of the server's status API | `strata_base` |
| 2 | `strata_server/` | The API server (OpenAI and Anthropic compatible) and its web page | layers 0–1 |
| 2 | `strata_control/` | The Terminal's library: launch, status, connect, benchmarks | layers 0–1 |
| 3 | `strata_cli/` | The `strata` command's interface | layers 0–2 |
| 4 | `setup.py`, `tools/` | Setup from a source checkout, developer scripts | any |

Packages in the same layer do not import each other: what they share moves down a layer.

## Language-neutral

`contracts/` holds what the C++ and Python sides (and the app) must agree on: the engine protocol and the model
family catalog. It depends on nothing.

## Where does new code go?

1. Would a program that is not an inference engine want it? Then `src/base/` or `strata_base/`.
2. Would every model use it? Then `src/common/` or `strata_common/`.
3. Do two or more model families use it? Then `src/models/common/`.
4. Otherwise, the family's folder in `src/models/`.

## Rules for new code

These replace practices that the old code has, and that new code must not copy:

- **Do not copy a file to make a variant.** Not into `sycl/`, not into another engine's folder. Share it, and
  put the difference behind an interface or a compat header (as `metal/include/metal_compat` does).
- **Code in `src/common/` and `src/models/` does not call the CUDA runtime (`cudaMalloc`, `cudaMemcpyAsync`, …)
  directly.** It uses `src/base/gpu/`, so that it can be built for every backend.
- **No settings in file-wide global variables** (as `g_kv_int8` and the other `qsa_set_*` switches in
  `src/models/qwen38/layer.cpp`). Pass settings in a struct.
- **One settings reader.** Engines do not each parse their own environment variables.
- **Model-specific code does not go in a folder called `core` or `common`.**
- **`main()` parses its arguments and calls a library.** It is not the engine itself (as the 9,500-line
  `main()` in `src/program/generate.cpp` still is; its option parser, model loader and one-conversation step are
  `strata::qwen38::parse_options`, `Model` and `Conversation`, in `src/models/qwen38/`, which `strata-q38`'s
  `main()` calls too). That `main()` is the older command loop: see
  [Two command loops in the tree](#two-command-loops-in-the-tree).
- **No import or include cycles,** and no reaching up a layer, also not at run time (loading `setup.py` or a
  `tools/` script through `importlib` or `sys.path`). `tests/architecture/test_layering.py` checks the new folders;
  its `KNOWN_UPWARD_LOADS` lists the files that still do it.

## Old folders

These are emptied by the refactor and then deleted. Until then, code in the new folders may still use them. Where
each file goes is listed in [UPSTREAM_PORTING.md](UPSTREAM_PORTING.md#path-map).

| Old folder | Goes to |
|---|---|
| `src/core/` | split. Done (A3): `src/base/gpu/`, `src/common/state/`, `src/common/sample/`, `src/models/common/`, `src/models/qwen38/`, `src/program/`. Done (A5): the expert store's host-only parts, to `src/common/experts/`. Left: the expert store's parts that call the CUDA runtime directly (`expert_cache`, `expert_source`), for `src/common/experts/` once `src/base/gpu/` offers those calls; its multi-GPU parts (`peer_experts`, `remote_experts`, `remote_expert_opt`), which use the shared S2 expert backend in `src/models/common/kernels/` (B3 kept them); and the files that must be split first (A5) |
| `src/kernels/` | split. Done (B1): `src/common/kernels/`, `src/models/common/kernels/`, `src/models/qwen38/kernels/`; the folder is gone |
| `src/prefill/` | split. Done (B1): `src/common/kernels/prefill/`, `src/models/qwen38/prefill/`; the folder is gone |
| `serve/` | `strata_server/`; its all-purpose modules to `strata_base/` |
| parts of `strata_control/`, `setup.py`, `tools/` | `strata_base/`, `strata_common/` |
