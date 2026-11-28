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

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
