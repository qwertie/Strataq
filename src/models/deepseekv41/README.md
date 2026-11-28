# src/models/deepseekv41/

The DeepSeek V4.1-Flash engine. Headers: `include/strata/models/deepseekv41/`.

**A5** and **B2** replace its own copies of shared code; A4 moved its OS services to
`include/strata/base/platform/os.hpp`. Its plain sampling is the shared `sample_tokens`; its quant kernels
(`ds4_iq`, `ds4_gguf`, `ds4_tq`), its draft sampling (`vendor/sampler.cu`) and its worker pool (`ds4_pool.hpp`)
stay its own (see the open issues in REFACTOR_PLAN.md).

`conversation_layout.cpp` describes a conversation's state as page-pool regions (step D1 of
[docs/MEMORY_MANAGER_PLAN.md](../../../docs/MEMORY_MANAGER_PLAN.md#deepseek-v41-flash-on-the-memory-manager)); its
test is `conversation_layout_test.cpp`. The model also uses the geometry (`geometry.hpp`) and the cache rows' sizes
(`kv_rows.hpp`).

The model loads in two calls (`ds4_model.hpp`): `Model::load` (the dense weights, the embedding table, the
Engram's reader) and `place_experts` (the prompt path's buffers, the layers' expert tables, the CPU lane).
`strata-ds41` (`paged_model.cpp`, step D4, `main()` in `src/program/ds41_main.cpp`) points the model at a
conversation's pages (`use_state_memory`). Its routed experts are in the memory manager's page pools (step D5,
`Options::expert_pages`): the GPU computes those `set_expert_addresses` names (a page's address each, in the decode's
`vram_tab` and the prompt's `slot_tab`), and the CPU lane and the prompt path read the others from the RAM's pool,
which reads the ones it lacks from the files without the file cache (`expert_files.cpp`, tested by
`expert_files_test.cpp`). It reads the model's options with `read_option` (`ds4_options.cpp`).

The model (`ds4::Model`) keeps everything in `Model::Impl`, which `ds4_model_impl.hpp` declares for this folder
only (step D-2). Its methods are in files by job, named as Qwen3.8's are:

| File | What it holds |
|---|---|
| `ds4_model_impl.hpp` | `Model::Impl`'s members and declarations, the structs it is made of (`Pack`, `Dev`, `Layer`, `Chunk`, ...) and the small helpers that run for every layer or token |
| `ds4_model.cu` | `Model`'s calls into `Impl`, `Impl`'s tear-down, the sampling, the profile and the debug dumps |
| `ds4_load.cu` | loading: the pack, the dense weights, the experts' place in the files and their blobs, the experts' places for the paged engine, a layer's move to another card, the CPU lane's worker pools |
| `ds4_prefill.cu` | the prompt path: chunks of a prompt through the layers (`forward_chunks`), its buffers, the flow that streams the experts, `Model::feed` and `feed_mark` |
| `ds4_verify.cu` | a window of tokens through the layers (`forward`, `run_attention`, `run_moe`), the head, the check of a window of drafts, and the paged engine's calls for a step (`step_conversations`, `prepare_group` ... `finish_group`) |
| `ds4_cpu_lane.cu` | the CPU lane: the experts the host computes, and its service thread during the decode |
| `ds4_dspark.cu` | the DSpark drafter |
| `ds4_engram.cu` | Engram's rows |
| `ds4_vision_tower.cu` | the vision tower's host side (its kernels are in `ds4_vision.cu`) |
| `ds4_checkpoints.cu` | the state's checkpoints and restore points, and the swap of a conversation's state |

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
