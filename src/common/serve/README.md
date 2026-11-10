# src/common/serve/

The engine side of the [request-ID protocol](../../../contracts/requests.md) that the Python server speaks
(`strata_common/engine_client.py`), and the paged
engine. Headers: `include/strata/common/serve/`. Libraries: `strata_serve` (no GPU) and `strata_paged` (the paged
engine).

- `loop.hpp`: `Input` reads the server's lines on a thread of its own (STOP raises a flag that a running request
  checks; the other lines wait in a queue). `Loop` gives the engine the commands that are its own and meanwhile runs
  the engine's busy work (batch slots decoding) and answers BSTOP, BYIELD, LIST_CONVERSATIONS,
  GET_CONTEXT_WINDOWS and GET_MEMORY_USE. An engine says what it does through `EngineHooks`.
- `memory_report.hpp`: GET_MEMORY_USE's answer (what the engine holds in each memory, for the server's
  `GET /v1/memory`): each engine fills a `MemoryReport` (`EngineHooks::describe_memory`), and the loop adds the parked
  conversations and writes the lines.
- `request.hpp`: a request line (`RUN`, `RUNI`, plus legacy handler forms) split into its parts, its token ids, and the
  sampling keys every engine reads.
- `paged_engine.hpp`: the paged engine (docs/MEMORY_MANAGER_PLAN.md, "One shared paged engine"): conversations in
  page pools, several decoding at once, pausing to RAM and the SSD, priorities, `CHANGE_SETTINGS`, the pools' resizes
  and start-up fill, the adaptive tier, the layer cut's planning and moves. It runs a model through `paged_model.hpp`
  (`PagedModel`: per card the state layout, experts and a cut move; per conversation its state on each card,
  `PagedPart`; the prompt, the step, the drafter and the `INFO` and `DONE` lines, each in its old engine's fields).
  strata-q38 runs it for Qwen3.8 (`src/models/qwen38/paged_model.cpp`), strata-ds41 for DeepSeek
  (`src/models/deepseekv41/paged_model.cpp`). What each model's start does before it, `paged_start.cpp`: the engine's
  own options, the runtime's environment, the cards' links, the pools' sizes and their GPU memory, and the RAM's pool
  and the copy-back rule (`open_ram_pool`). The pools open empty: the engine says `READY` and then fills them while
  it serves, the GPUs' first, and the RAM's with the experts no GPU holds.
- `decode_selection.hpp`: selects the highest-priority requests for a paged decode step. Equal priorities keep
  request order until completion or preemption; requests waiting for a turn do not rotate into the selected set.
  Cancellations drain first without generating rows. `concurrent` limits this selected set, not admitted requests
  or conversations whose KV cache remains in VRAM.
  RUN admission does not generate a first token or hold the command channel through decoding.
  Higher-priority pending requests may preempt active work; Python resumes the interrupted identity.
- `token_budget.hpp`: accounts for each GPU-resident conversation once, independently of its physical page layout
  or the number of GPUs. Reservations cover prompt reads and decode windows; pausing or forgetting returns them.
  `tokens_max` caps their combined logical tokens. The separate live `context` limit cannot exceed that budget or
  the model's loaded `context_capacity`; inconsistent edits are rejected before changing settings.
  A separate `vram_conversations` count limits fixed state as well as idle caches, independently of token counts.
  Its default adds one cached place to decode concurrency, plus one more for Qwen on cards with at least 10 GiB.
  Extra conversations wait or move to RAM/SSD, releasing their fixed pool regions too. Its live value cannot fall
  below `concurrent`. `conversation_idle_seconds` defaults to 66: old idle state may give way to
  conversation allocations, between steps, while active decode and prompt reads remain protected.
- `state_bytes.hpp` in common/state: bounded host-state fields shared by Qwen, DeepSeek and the pause-file wrapper.
  Qwen's recurrent rewind snapshots and DeepSeek's ring/compressor checkpoints live in ordinary RAM. Both travel
  in the shared pause file's host header. Disk-paused headers retain only lookup metadata in RAM; checkpoint
  payloads are read back from the file on resume.

Every updated engine accepts RUN/RUNI and targeted CANCEL. The paged engine queues and places requests itself;
its command loop remains available between decode steps even with one request. Older handlers use the shared
`request_output` adapter: their execution details stay internal, and all emit request-tagged T/PP/DONE events.
ASK commands have separately tagged replies, including multiline memory/context reports. Python retains its
legacy client only for previously released binaries without the `requests` capability.

## Which engine uses it

**Qwen3.8** (`src/program/generate.cpp`), **Qwen3.6** (`Serve` in `src/models/qwen36/q36_engine.cu`) and
**DeepSeek V4.1** (`serve` in `src/models/deepseekv41/ds4_main.cpp`) use it. Their own habits:

- Qwen3.6 skips empty lines, and answers QUIT with `ERR expected: GEN ...` and ends when stdin does (the server sends
  `STOP`, `QUIT` and then closes stdin). Its `Engine::spec_step` takes the stop check as a function.
- DeepSeek has no batch slots: its `Loop` has `list_conversations` only. Its `Input` is made with
  `quit_ends_request`, so QUIT and the end of stdin also end a running request, and `STOP <anything>` counts as
  STOP. With `--parallel` it runs several requests at once and reads its lines with `Input::next` on a thread of its
  own (no `Loop`): `STOP <rid>` ends request `rid` alone (`is_stop_requested(rid)`). It reads its lines with `parse_request_line(..., keys_after_file)` (keys may follow GENI's picture file),
  refuses a line without max_new (`has_max_new`), and sets `top_k = 64` before it reads the sampling keys. Its
  model's `set_cancel` asks `is_stop_requested`.

Not shared yet: the `DONE` line has 15 fields and named ones (`chunk=`, ...) on Qwen3.8, 15 on Qwen3.6 and 10 on
DeepSeek; a shared writer would have to keep each engine's fields (the paged engine leaves it to the model:
`PagedModel::print_done`).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
