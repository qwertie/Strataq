# src/base/gpu/

The device runtime interface (memory, copies, streams, events, graphs) with one implementation per backend, the GPU
hardware traits, and the HIP compat headers. Headers: `include/strata/base/gpu/`.

The HIP compat headers are two different sets, each put first on the include path of its own build:
`include/strata/base/gpu/hip_compat/` for the wave32 HIP backend (`STRATA_ENABLE_HIP`) and
`include/strata/base/gpu/hip_compat_gfx906/` for the wave64 gfx906 build (`STRATA_HIP_GFX906`).

What is here so far came unchanged from `src/core/` (and `gfx_arch.hpp` from `include/strata/kernels/`): the device
arena and device facts (`device`), the pinned host arena and its budget (`pinned`, `pin_budget`, `pinned_ring`),
graphs (`graph`, `host_graph_launch`), virtual memory (`vmm`), batched copies (`dma_batch`), `on_device`, and the
hardware traits (`emulate`, `sm75_series`, `sm120_series`, `tc_sm75`, `gfx12_platform_defaults`, `gfx_arch`).

`runtime.hpp` is the runtime interface that shared code calls instead of the CUDA runtime. So far it has what
`src/common/state` needs: device memory, 2D copies and memsets; **A5** adds the rest (streams, events) as shared code
needs them. `links` measures a card's links at start (to the host and to another card, each way, and a small copy's
round trip), for the rules in `src/common/experts/card_rules.cpp` (docs/MEMORY_MANAGER_PLAN.md, step M8).

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
