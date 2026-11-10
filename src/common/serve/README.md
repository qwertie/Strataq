# src/common/serve/

The engine side of the line protocol that the Python server speaks (`strata_common/engine_client.py`). Headers:
`include/strata/common/serve/`. Library: `strata_serve` (no GPU).

- `loop.hpp`: `Input` reads the server's lines on a thread of its own (STOP raises a flag that a running request
  checks; the other lines wait in a queue). `Loop` gives the engine the commands that are its own and meanwhile runs
  the engine's busy work (batch slots decoding) and answers BSTOP, BYIELD, LIST_CONVERSATIONS and
  GET_CONTEXT_WINDOWS. An engine says what it does through `EngineHooks`.
- `request.hpp`: a request line (`GEN`, `GENI`, `BGEN`, `BGENI`) split into its parts, and the sampling keys every
  engine reads.

Each engine still handles its own requests, SAVE / RESTORE, VRAM and QUIT, and writes its own `T`, `PP`, `DONE`, `BT`
and `BDONE` lines.

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
DeepSeek; a shared writer would have to keep each engine's fields.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
