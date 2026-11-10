# Carrying upstream changes over to this fork

This fork is reorganized ([REFACTOR_PLAN.md](REFACTOR_PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md)), so a change
accepted upstream ([Niko1221/Strata](https://github.com/Niko1221/Strata)) often touches files that have moved,
been merged with a copy, or been split here. This page says where each upstream path lives now, and how to carry
a change over.

## Where we are

- The `upstream` remote is `https://github.com/Niko1221/Strata.git`.
- **Last upstream commit carried over: `504b1433`** (2026-11-09, "test: the old-launcher test reads the Terminal
  version from strata_cli …", version 0.2.13.1). Update this line whenever you finish a batch.

## Steps

1. `git fetch upstream`, then list what is new: `git log --oneline --reverse 4b9e05a2..upstream/main` (use the
   commit from the line above). Work through it one upstream pull request at a time.
2. Decide: **carry over**, **skip** (with a reason), or **already done** (the fork has its own version). Record the
   decision in the [log](#log).
3. Try git first: `git cherry-pick -x <commit>` (or `-m 1` for a merge commit). Git follows a file that was moved
   whole, because its content is unchanged. It cannot follow a file that was split, merged with a copy or moved
   into a renamed module; those hunks conflict or land in a deleted file.
4. For each hunk git could not place, look up its old path in the [path map](#path-map) and apply it by hand
   where the code is now. The [special cases](#special-cases) cover changes that touch copies.
5. Run the checks in [REFACTOR_PLAN.md](REFACTOR_PLAN.md#checks-for-every-step) that the change touches.
6. Commit with the upstream reference in the message (`-x` adds it), and update the log and the "last commit"
   line above.

## Special cases

- **A change to a file and its SYCL copy** (`src/X` and `sycl/src/X`): where the two are now one shared file,
  apply it once. Where the SYCL copy is still separate, apply it to both, as upstream did. The files shared since A6
  are listed in `sycl/tools/normalize.sh` (`shared`): apply upstream's change to `src/X` or `include/strata/X` only,
  drop the hunks for the `sycl/` copy, and build SYCL. If the change uses a CUDA name the SYCL build lacks, add the
  name to `sycl/include/sycl_compat/cuda_runtime.h`; do not bring the copy back.
- **A change to only the SYCL copy** of a file that is now shared: decide whether it is a SYCL-only difference.
  If it is, it goes behind the SYCL side of the shared interface; otherwise apply it to the shared file.
- **A change to code that this fork merged** (for example DeepSeek's own sampler header, replaced by the shared
  one): check whether the change also applies to the shared code; often the shared code already has it.
- **A change to the 14,800-line `main()` in `src/program/generate.cpp`**: the code has moved into libraries
  under `src/common/` and `src/models/qwen38/`. Find it by a function or log-message name from the hunk. The part
  from `set_coupled_draft` to "session is up" (the model loading) is `Model::load` in `src/models/qwen38/model.cpp`,
  as it was: its locals that the rest of `main()` uses are `Model`'s members, which `main()` binds to their old names.
  The command line (`usage`, the option loop, the per-device defaults, the process set-up before the first GPU call)
  is in `src/models/qwen38/options.cpp`; the residency tables and the token graph are `Model::build_hit_tables`; the
  CLI run's prompt path and verify windows are `Conversation` (`src/models/qwen38/conversation.cpp`).
- **Python imports** that changed with a module's move (`from serve import …`, `from strata_control import tls`)
  must be rewritten to the new package.

## Path map

Old paths are upstream's. Status is `planned` until the step is done, then `done`. Each refactor commit that moves
code updates its rows here. Rows are grouped by step; `{a,b}` lists several files of one folder.

| Upstream path | Path here | Step | Status |
|---|---|---|---|
| `serve/` (everything not listed below) | `strata_server/` | A2 | done |
| `serve/winjob.py`, `strata_control/{atomic,longpath,loopback,posix_owner,private_files,tls,transfer,winproc}.py` | `strata_base/` | A2 | done |
| `serve/{launch_identity,lifeline,ready,telemetry}.py` | `strata_common/` | A2 | done |
| `strata_control/{acquisition,draft_layer,drafting,engine_component,gpu_fit,gpu_roles,hipblaslt_tuning,offline,paths,presets,procs,profiles,signing,targets,tune_history,tune_knobs,tune_machine,user_args,vision_reader}.py`, `strata_control/update_keys.json` | `strata_common/` | A2 | done |
| the engine client in `serve/server.py` (`StrataEngine`, `MockEngine`, `engine_args`, `child_env`, the API-key helpers, the `Engine*` errors and the module values they read) | `strata_common/engine_client.py` | A2 | done |
| `strata_control/replica_profile.py` (added upstream after 4b9e05a2) | `strata_common/replica_profile.py`: the server reads it and may not import `strata_control` | merge of c62d5e3a | done |
| `serve/{loader_diag,logit_bias}.py` | `strata_common/` | A2 | done |
| `strata_cli/bench.py` | `strata_control/bench.py` | A2 | done |
| `_source`, `_redact_argv`, `_effective`, `_launch_plan` in `strata_cli/cli.py` | `strata_control/launch_plan.py`: `read_source`, `redact_argv`, `get_effective`, `make_launch_plan` | A2 | done |
| `strata_cli/client.py` | `strata_common/client.py` (the monitor bundle's schema 7 lists it there) | A2 | done |
| `tools/gguf_reader.py` | `strata_common/gguf_reader.py` (scripts import `strata_common.gguf_reader` and put the repository on `sys.path`) | A2 | done |
| helpers repeated in `setup.py` and `tools/*.py` (`sha256`, `http_json`, `get_json`, GPU detection, engine download) | `strata_base/`, `strata_common/` | A2 | planned |
| the settings file's path: `settings_path` in `setup.py` and `tools/strata_mcp.py`; `app_settings_file` in `setup.py` | `strata_control/discovery.py`: `own_settings_file`, `in_app_mode` | A2 | done |
| `same_drive`, `same_folder` in `setup.py`; `same_drive`, `same_place` in `strata_cli/launch/library.py` | `strata_base/path_compare.py` | A2 | done |
| `repoint_config`, `library_size_gb` in `setup.py`; `library_gb` in `strata_cli/launch/library.py` | `strata_control/models_move.py`: `repoint_config`, `library_gb` | A2 | done |
| `model_config_files` in `setup.py`, `config_files` in `strata_control/models_move.py` | `strata_control/discovery.py`: `config_files` | A2 | done |
| the RAM probes: `_memory_status`, `ram_gb` in `setup.py`; `memory` in `strata_control/tune_machine.py`; `_CpuRamFallback.ram` in `serve/telemetry.py`; `ram_bytes` in `strata_cli/bench_facts.py`; `own_ram_gb` in `tools/strata_mcp.py`; `total_ram_gib` in `tools/strata_bench.py` | `strata_base/memory.py` (`read_ram`, `windows_memory_status`); the old names that remain call it | A2 | done |
| `_feed_source`, `default_feed`, `DEFAULT_FEED`, `ENV_FEED` in `strata_cli/update_cmd.py` | `strata_control/updater.py`: `feed_source`, `default_feed`, `DEFAULT_FEED`, `ENV_FEED` | A2 | done |
| the DeepSeek engine upgrade: `upgrade_dsv41_engine` in `setup.py` and the body of `Runner.upgrade_dsv41_engine` in `strata_cli/launch/runner.py` | gone: upstream 0.2.13 dropped the upgrade at launch (`Runner.upgrade_dsv41_engine` only adds a notice), so `try_upgrade` was deleted | A2 | done |
| the address test in `local_feed` in `setup.py` | `strata_base/transfer.py`: `is_local` | A2 | done |
| the model catalog in `setup.py` (`MODELS`, `FAMILIES`, `CONTEXTS`, `OWN_PATH_FAMILIES`, `MULTI_FILE_FAMILIES`, the `*_FILES` and `*_SHARDS` pins, `NVFP4_REPOS`, `REPO_PINS`, `HF_REVISIONS`, `hf`, `hf_endpoint`); its copies: `FALLBACK_*` in `tools/strata_mcp.py` and `strata_cli/launch/settings.py`, the sizes in `PINNED_PROJECTORS` in `strata_control/models.py` and the DeepSeek file sizes in `strata_control/dsv41_fit.py` | `strata_common/catalog.py` (`setup.py` imports the names it had) | A2 | done |
| `SPLIT_MIN_VRAM_GB`, `SPLIT_PROMPT_VRAM_GB` in `setup.py` | `strata_common/gpu_fit.py` | A2 | done |
| `strata_control/sysinfo.py` (added upstream after c62d5e3a) | `strata_base/sysinfo.py`: standard library only, and `strata_common` reads it | merge of upstream 0.2.13.1 | done |
| `strata_control/{vram_fit,intel_spec,dsv41_paths,cfg_paths,dsv41_flags}.py` (added upstream after c62d5e3a) | `strata_common/`: the engine client, `drafting`, `profiles` and `engine_component` read them | merge of upstream 0.2.13.1 | done |
| the state-file writers that write a temporary file and rename it: `_atomic` in `strata_control/{profiles,presets}.py`, `_atomic_write` in `strata_control/tune_apply.py`, `_private_write` in `strata_cli/launch/store.py`, `_write` in `strata_control/preset_marks.py`, the bodies of `_replace_bytes` in `strata_control/calibration.py`, `write_private` in `strata_cli/launch/runner.py` and `UpdateState.save` in `strata_control/updater.py` | `strata_base/atomic.py`: `write_bytes`; the old names that remain call it | A2 | done |
| the same for text: `save` in `strata_control/{time_skip,tuning}.py`, `_save_state` in `strata_control/components.py`, `save` in `serve/runconfig.py`, `write_config` in `setup.py`, `_write_status` in `strata_cli/tune_commands.py`, the pointer in `strata_control/updater.py` | `strata_base/atomic.py`: `write_text` | A2 | done |
| `_digest` in `strata_control/calibration.py`; the hashing in `ComponentFile.verify_file` (`strata_control/components.py`) and `verify_engine_archive` (`setup.py`) | `strata_base/hashing.py`: `sha256_file` | A2 | done |
| the nvidia-smi card list in `gpus` (`setup.py`) and `own_nvidia_gpus` (`tools/strata_mcp.py`) | `strata_common/gpu_fit.py`: `NVIDIA_SMI_CARDS`, `parse_nvidia_cards`; the old names remain and call it | A2 | done |
| `_gpu_numbers` in `serve/server.py`, `_int_list` in `strata_control/gpu_roles.py` | `strata_common/gpu_roles.py`: `card_numbers` | A2 | done |
| the CUDA driver probe in `gpu_speed_scores` (`serve/server.py`) | `strata_common/gpu_roles.py`: `nvidia_speeds`, which `gpu_speed_scores` (`strata_common/engine_client.py`) calls | A2 | done |
| `rope` in `tools/dsv41/ref.py` | `tools/dsv41/dspark_ref.py`: `rope` | A2 | done |
| the arm parsing and per-arm server start in `tools/agent_turns_bench.py` | `tools/ab_engine.py`: `parse_arms`, `arm_server` | A2 | done |
| the GGUF opening and sign vectors in `tools/trellis/{gyro_column_profile,gyro_error_profile,gyro_vs_original,proxy_diag_error,proxy_ldl}.py` | `tools/trellis/build_swap.py`: `open_rotated`, `hadamard_signs` | A2 | done |
| `src/platform/`, `include/strata/platform/` (except `hip_compat/`) | `src/base/platform/`, `include/strata/base/platform/` | A3 | done |
| `include/strata/hip_compat/`, `include/strata/platform/hip_compat/` | `include/strata/base/gpu/hip_compat/` (the wave32 HIP backend's) and `include/strata/base/gpu/hip_compat_gfx906/` (gfx906's): two different sets | A3 | done |
| `src/core/{device.cu,pinned.cu,graph.cpp}` and their headers | `src/base/gpu/` | A3 | done |
| `src/core/{vmm.cpp,pin_budget_test.cpp,pinned_shared_test.cpp}`, `include/strata/core/{vmm,pin_budget,pinned_ring,host_graph_launch,on_device,dma_batch}.hpp` | `src/base/gpu/`, `include/strata/base/gpu/` | A3 | done |
| `include/strata/core/{sm75_series,sm120_series,tc_sm75,emulate,gfx12_platform_defaults}.hpp`, `include/strata/kernels/gfx_arch.hpp` | `include/strata/base/gpu/` | A3 | done |
| `src/artifact/`, `include/strata/artifact/` | `src/common/artifact/`, `include/strata/common/artifact/` | A3 | done |
| `src/spec/`, `include/strata/spec/` | `src/common/spec/`, `include/strata/common/spec/` | A3 | done |
| `src/core/conversation_*` and `include/strata/core/{conversation_*,token_listing,progress}.hpp`, except the snapshot files below | `src/common/state/`, `include/strata/common/state/`; the snapshot code's helpers `src/core/{conversation_copy,conversation_checked}.hpp` and its tests `src/core/{conversation_snapshot,conversation_validation}_test.cpp` go with it to `src/models/qwen38/` | A3 | done |
| `include/strata/core/session_save_reclaim.hpp`, `src/core/{session_save_reclaim,token_listing}_test.cpp` | `include/strata/common/state/`, `src/common/state/` | A3 | done |
| `src/core/{conversation_snapshot,conversation_state}.cpp`, `include/strata/core/conversation_snapshot.hpp` (they use Qwen3.8's session) | `src/models/qwen38/` until A5 makes them model-neutral | A3 | done |
| `include/strata/core/{geometry,family_layout}.hpp` | `include/strata/models/common/`; `geometry.hpp` only once `include/strata/common/artifact/gguf_reader.hpp` no longer uses it (the kernels that include it are in `src/models/` since B1) | A3, B1 | `family_layout.hpp` done |
| `src/dsv41/`, `include/strata/dsv41/` | `src/models/deepseekv41/`, `include/strata/models/deepseekv41/` | A3 | done |
| `src/qwen35/`, `include/strata/qwen35/` | `src/models/qwen36/`, `include/strata/models/qwen36/` | A3 | done |
| `src/ngram/`, `src/plan/` and their headers | `src/models/qwen38/{ngram,plan}/`, `include/strata/models/qwen38/{ngram,plan}/` | A3 | done |
| the Qwen3.8-only files of `src/core/` (one row per file once A3 decides) | `src/models/qwen38/`, `include/strata/models/qwen38/`: `{layer,session,mtp,verify,native_dense,native_head,dflash2}` (source and header), `dflash2_{parity,attn_bench,gemm_bench,small_bench}.cpp`, `dflash2_{bw,launch}_bench.cu`, the headers `{adapt_async_default,batch_rows,batch_telemetry,draft_cap}.hpp` and their tests | A3 | done |
| `src/core/{device_main,load_main,concurrent_main,overlap_main}.cpp` | `src/program/` | A3 | done |
| `include/strata/core/logit_bias.hpp`, `src/core/logit_bias_test.cpp` | `include/strata/common/sample/`, `src/common/sample/` | A3 | done |
| `src/core/{arch_defaults,nv_defaults,weights,layout}.cpp`, `include/strata/core/{coupled_draft,spec_prob,uma_cache,foresight_swap,exchange_storage,hit_hook,remote_expert_opt}.hpp` and the other files left in `src/core/` | `uma_cache.hpp` (and `src/core/uma_cache_test.cpp`), `exchange_storage.hpp` and `hit_hook.hpp`: `include/strata/common/experts/` (`src/common/experts/`). `coupled_draft.hpp` and `spec_prob.hpp` (and `src/core/{coupled_draft,spec_prob}_test.cpp`), the sampler's speculative modes: `include/strata/common/kernels/` (`src/common/kernels/`). The rest is decided in A5 or B3 (each is shared by several engines, or mixes model-neutral and Qwen3.8 code) | A5, B1, B3 | the three expert headers and the two sampler headers done |
| `sycl/src/{platform,ngram}/` | `sycl/src/base/platform/`, `sycl/src/models/qwen38/ngram/` | A3 | done |
| the SYCL copies of the `src/core/` files above, `sycl/src/core/copy_queue.cpp` | the same new place under `sycl/`; `copy_queue.cpp` to `sycl/src/base/gpu/` | A3 | done |
| `sycl/src/X`, `sycl/include/strata/X` | the same new place as `src/X`, under `sycl/` | A3 | planned |
| `src/dsv41/ds4_os.hpp` | `include/strata/base/platform/os.hpp`, namespace `strata::platform` (`os::x` is `platform::x`); its `wide` is `widen_utf8`, which `direct_file.cpp`, `pinned.cu` and `expert_source.cpp` now use too | A4 | done |
| `src/dsv41/ds4_pool.hpp`, `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp` | kept apart: a shared pool made the code longer ([open issue](REFACTOR_PLAN.md#open-issues)) | A4 | not done |
| `include/strata/dsv41/sampler.hpp`, `src/dsv41/vendor/sampler.cu` | `include/strata/models/deepseekv41/sampler.hpp`: its `SamplerParams` and `sample_tokens` are the shared ones (`include/strata/common/kernels/sampler.hpp`); `penalty_rows` and the plain-draw kernels are gone. `src/models/deepseekv41/vendor/sampler.cu` keeps only the speculative draft and verify modes (`SpecList`, `sample_draft`, `sample_verify`; [open issue](REFACTOR_PLAN.md#open-issues)) | B2 | plain draw done |
| the protocol code in `src/program/generate.cpp`, `src/qwen35/q36_engine.cu`, `src/dsv41/ds4_main.cpp` | `src/common/serve/`. Done for `generate.cpp`: its stdin reader thread (`StdinShared`, `next_line`, `try_next_line`, `stop_req`) is `strata::serve::Input`; the `--serve` loop's handling of BSTOP, BYIELD, LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS is `Loop`, which gets Qwen3.8's part through `EngineHooks` set up just before the loop (`hooks.before_line` is the pipeline drain before a line); the scans of the queue for BSTOP and BYIELD during a prompt read are `Input::take_slot_controls`; the splitting of GEN / GENI / BGEN lines and of their sampling and `img_keys` keys is `parse_request_line`, `read_sampling_key` and `parse_hex_list`. Done for `q36_engine.cu` (`Serve`): its reader thread, queue and STOP flag are `Input` (`poll_control` calls `take_slot_controls`), `run()` is a `Loop` (`is_busy`, `step` = `window()`, `stop_slot`, `list_conversations`), `request()` reads its line with `parse_request_line`, its `Sampling` struct is `serve::Sampling` (its penalty fields are named `penalty_*`; `Sampling::needs_logits()` is the free function `needs_logits`) and its `img_keys` loop is `parse_hex_list`; `Engine::spec_step` takes the stop check as a `std::function` instead of an `std::atomic<bool>*`; `serve(Engine&)` is gone (`Serve(E).run()`). Done for `ds4_main.cpp`: its `Lines` reader is `Input` (constructed with `quit_ends_request`: QUIT, stdin's end and `STOP <anything>` raise the stop flag), its loop is a `Loop` with `list_conversations` only, and `parse_gen` reads its line with `parse_request_line` (`keys_after_file`; `has_max_new`) and its sampling keys with `read_sampling_key` into `Req::smp` | A5 | done |
| the session state of `src/qwen35/q36_engine.cu` (`alloc_session`, `copy_session`, the zeroing in `reset`, `sess_fixed_bytes` / `sess_cell_bytes`) | described as a `StateLayout` (`include/strata/common/state/state_layout.hpp`; `Engine::describe_session`) and handled by its `alloc_state`, `copy_state`, `clear_fixed_state` and `bytes_for` | A5 | done |
| the running-state copies of `src/core/conversation_state.cpp` (`conversation_checkpoint_save`, `_restore`, `_copy`) | described as a `StateLayout` of fixed regions (`describe_running_state` in `src/models/qwen38/conversation_state.cpp`) and copied with `copy_state`; the K/V parts in `conversation_snapshot.cpp` follow with the paged K/V pool | A5 | running state done |
| the window choice of the two decode loops in `src/program/generate.cpp` (`--serve`, `--spec`): the MTP window by `--spec-min-p` or `STRATA_SPEC_COST`, the suffix lookup window, the `--lookup-chain` tokens (`chain_tail`, `extra_sources_reset`, `extra_sources_append`), their counts and the `DraftPolicy` updates | `strata::spec::WindowPlanner` (`include/strata/common/spec/window_planner.hpp`) | A5 | done |
| DeepSeek's conversation slots (`src/dsv41/ds4_main.cpp`, `ds4_slotmeta.hpp`) | `src/common/state/`: the slot as a `StateLayout` per device (`src/common/state/README.md` says how) | A5 | planned |
| the generic parts of `src/core/{expert_cache,expert_source}.cpp`, `src/dsv41/ds4_store.hpp` | `src/common/experts/`. Done: the profile file (`read_expert_profile`, `rank_learned_profile`, `write_expert_profile`) is `expert_profile.{hpp,cpp}`; the RAM tier's arithmetic and host-memory readings (the `detail::` functions of `expert_source`: `host_available_memory`, the cgroup, NUMA and complement-plan helpers) are `ram_tier.{hpp,cpp}`; the SYCL build compiles them and `exchange_storage.hpp` instead of its copies (in `sycl/src/core/{expert_cache,expert_source}.cpp`, `sycl/include/strata/core/{expert_source,exchange_storage}.hpp`). `ExpertCache::fill_slot_queued` is `fill_slot` on the legacy stream. Left in `src/core/`: `ExpertCache` and the expert sources, which call the CUDA runtime directly; `ds4_store.hpp` is DeepSeek's own design | A5 | profile and RAM tier done |
| 7 SYCL copies that are 95%+ the same as `src/` (listed in the A6 commit) | 8 copies deleted: `sycl/include/strata/{base/gpu/graph.hpp,kernels/f16_bits.hpp,kernels/bf16_bits.hpp,kernels/rope.hpp,kernels/rope_scaling.hpp,models/qwen38/session.hpp}`, `sycl/src/{base/platform/direct_file.cpp,models/qwen38/ngram/ple_reader.cpp}`; the SYCL build compiles the shared `include/strata/X` and `src/X`. The CUDA types the two headers name come from `sycl/include/sycl_compat/cuda_runtime.h`; the SYCL build's 64 reading threads in `direct_file.cpp` are `STRATA_DIRECT_FILE_PREAD_THREADS=64` in `sycl/CMakeLists.txt` | A6 | done |
| `src/kernels/`, `include/strata/kernels/` | split. Every file not in the rows below: `src/common/kernels/`, `include/strata/common/kernels/`, in the same subfolder (`cpu/`, `cuda/`) | B1 | done |
| `src/kernels/cuda/{fused_gdn,native_moe,native_router}.cu`, `include/strata/kernels/{fused_gdn,native_moe,native_router}.hpp`, `src/kernels/{native_multi_parity.cpp,route_resident_parity.cu}` | `src/models/common/kernels/`, `include/strata/models/common/kernels/` (Qwen3.6 and Qwen3.8 use them) | B1 | done |
| `src/kernels/cuda/{cvec,dflash2,fused_gr,gdn,gr,kv_q4,kv_q8,kv_stream,native_flash_attn,native_gdn,native_gdn_preprocess,native_gr_norm,native_gr_postops,native_ple_postops,native_qsa,native_qsa_indexer,native_qsa_score,native_rope,ple,qsa,qsa_decode_attn,qsa_prompt_attn,qsa_scores_gfx1100_test,qsa_select,route_prior,s2_expert_grouped,shared_expert,verify_kernels}.cu`, the headers of the same names and `{fuse_decode,ngram}.hpp` in `include/strata/kernels/`, `src/kernels/{ngram.cpp,ple_oracle_vectors.inc,qsa_oracle_vectors.inc}` and the tests and benches of these kernels in `src/kernels/` (`batch_rows`, `cvec`, `decode_cluster`, `gdn`, `gdn_chunk`, `gr`, `gr_multi`, `hc_persist`, `kv_*`, `native_expert`, `pdl`, `ple*`, `qsa`, `qsa_*`, `resident_plan`, `rope`, `s2_expert_grouped`, `shared_expert`, `verify`, `verify_batch` `_parity.cpp`; `fused_gr_bench.cpp`, `qsa_rows_bench.cpp`, `qsa_select_bench.cpp`, `router_chain_bench.cpp`) | `src/models/qwen38/kernels/`, `include/strata/models/qwen38/kernels/`: Qwen3.8's own layers (QSA, gated residual, GDN, PLE, the verify window) or compiled for its shapes | B1 | done |
| `src/kernels/cuda/s26_tsum.cuh`, `src/prefill/wmma_gemm.h` | `include/strata/common/kernels/s26_tsum.cuh`, `include/strata/common/kernels/prefill/wmma_gemm.h` (included from several folders now) | B1 | done |
| `src/prefill/`, `include/strata/prefill/` | split. Every file not in the row below: `src/common/kernels/prefill/`, `include/strata/common/kernels/prefill/` | B1 | done |
| `src/prefill/{prefill.cpp,kernels.cu,gdn_chunk.cu,mmq_resident_sort.hpp,gdn_rec_parity.cu,gr_upmix_parity.cu,warm_policy_test.cpp}`, `include/strata/prefill/{prefill,kernels,gdn_chunk,share_rules,warm_policy}.hpp` | `src/models/qwen38/prefill/`, `include/strata/models/qwen38/prefill/`: Qwen3.8's prompt path | B1 | done |
| the SYCL copies in `sycl/src/{kernels,prefill}/`, `sycl/include/strata/{kernels,prefill}/` and the SYCL-only files there | the same new place as their original, under `sycl/`; the SYCL-only files by the same rules. Deleted, never built: `sycl/src/prefill/{gemm_bf16_parity,ggml_cuda_host,moe_fused,moe_fused_iq,moe_mmq}.dp.cpp` (the SYCL build compiles `moe_fused_stub.cpp` and no MMQ) | B1 | done |
| `src/dsv41/{ds4_iq,ds4_gguf,ds4_tq}*` | kept in `src/models/deepseekv41/`: not copies of the shared quant kernels ([open issue](REFACTOR_PLAN.md#open-issues)). `ds4_iq_cpu.cpp` lost its unbuilt ggml-cpu rows (`iq_cpu_own_rows` is gone) and now only chooses between `ds4_tq_cpu.cpp` and `ds4_gg_cpu.cpp` | B2 | kept |
| `src/core/{peer_experts,remote_experts}.cpp`, `src/core/remote_expert_opt.cu` | kept in `src/core/`: they use the shared S2 expert backend in `src/models/common/kernels/`, which `src/common/` must not include, and call the CUDA runtime directly (peer access, contexts, streams, events, mapped host memory). `PeerExperts` switches devices with `OnDevice` (`include/strata/base/gpu/on_device.hpp`); the SYCL build compiles the shared `peer_experts.hpp` instead of its copy (`sycl/include/strata/core/peer_experts.hpp`). `remote_experts.hpp` keeps its SYCL copy: the SYCL `ExpertSource` lacks the functions `RemotePrefillSource` overrides | B3 | kept |
| the model loading in `main()` of `src/program/generate.cpp` (from `set_coupled_draft` to "session is up"), its `Options` struct, `take_resident_experts`, and the helpers that only that part used or that it shares with the rest of `main()`: `under_wddm`, `vram_free_note`, `get_auto_lend_pct`, `Drive`, `drive_pool`, `drive_pool_multi`, `SplitDrive`, `drive_pool_split`, `GpuStage`, `on_rotational_disk`, `mem_mark`, `load_control_vectors`, `pcie_auto_split_ok`, `decode_defaults_arch_ok`, `mtp_first_default_on`, `pcie_auto_fork_arch_ok`, `probe_pcie_h2d_gbps`, `pcie_frac_for_gbps`, `measure_pool_affinity`, `g_warm_thr`, `ple_direct_rows_per_s` | `strata::qwen38::Model::load` in `src/models/qwen38/model.cpp`, with the helpers; the locals the rest of `main()` uses are `Model`'s members (`include/strata/models/qwen38/model.hpp`; the CPU pool is `cpu_pool`, an `std::optional`), the lambdas `stage_of` and `gdn_snapshot_bytes` its methods, and `g_last_h2d_gbps` is read through `get_last_h2d_gbps()`. `Options` and `take_resident_experts`: `include/strata/models/qwen38/options.hpp`. The `--tokens` check stayed in `main()` | M2c | done |
| `include/strata/program/{split_rules,batch_groups}.hpp`, `tests/program/split_rules_test.cpp`, `src/program/batch_groups_test.cpp` | `include/strata/models/qwen38/`, `src/models/qwen38/` (the loader uses them; namespaces unchanged) | M2c | done |
| `include/strata/models/qwen38/split_rules.hpp`, `src/models/qwen38/split_rules_test.cpp` | `include/strata/common/experts/`, `src/common/experts/` (the shared layer-split planner uses them; namespace unchanged) | M8 | done |
| in `main()` of `src/program/generate.cpp`: the option loop, `usage`, `parse_i64_list`, the per-device defaults (gfx1151, sm_86, gfx12, the gfx906 check), the set-up before the first GPU call (`--gpu`, `CUDA_MODULE_LOADING`), `launch_blocking`, `prime_hw_queues` | `src/models/qwen38/options.cpp`: `parse_options`, `print_usage`, `parse_i64_list`, `apply_device_defaults`, `prepare_process`, `is_launch_blocking`, `prime_gpu_queues` (`include/strata/models/qwen38/options.hpp`) | M2c2 | done |
| in `main()`: the residency tables and the token graph (`loop_scratch`, `tgraph`, `host_res`, `res_put`, `d_res`, `thits`), `rss_probe`, the verify windows' `VerifyHits`, `share_default_ok` | `Model::build_hit_tables`, `Model::put_residency`, `strata::qwen38::rss_probe`, `Model::get_verify_hits`, `Model::is_cpu_share_default_ok` and `Model::arm_prompt_cpu_share` | M2c2 | done |
| in `main()`: the CLI run's `Prefill prefill` and its init, the spec loop's `Verifier ver`, its init and settings, and the window's expert-pool check; the serve path's request sampling | `Conversation` (`init_prompt`, `init_decode`, `read_prompt`, `run_window`) and `get_sampler_params` in `src/models/qwen38/conversation.cpp` | M2c2 | done |
| `include/strata/program/logits_selection.hpp` (namespace `strata::program::logits_selection`) | `include/strata/models/qwen38/logits_selection.hpp` (namespace `strata::qwen38::logits_selection`; the option loop uses it) | M2c2 | done |
| the other SYCL copies of `src/` files | the shared `src/` file | C1 | planned |
| files upstream added after 4b9e05a2, Qwen3.8's own: `include/strata/core/{ple_residency,window_evict}.hpp`, `include/strata/kernels/{qsa_pattn,window_start}.hpp`, `include/strata/prefill/{copy_events,gdn_probe,kva,kva_floor,moe_ws}.hpp`, `src/kernels/cuda/qsa_pattn.cu`, `src/prefill/{gdn_harness.cu,gdn_tc.cuh,gdn_wmma12.inc,moe_ws*.cu,moe_ws.cuh}`, and their tests | `include/strata/models/qwen38/` (`kernels/`, `prefill/`), `src/models/qwen38/` (`kernels/`, `prefill/`) | merge of c62d5e3a | done |
| files upstream added after 4b9e05a2, shared: `include/strata/core/{auto_cache,gated_proposal,spec_proposal,expert_nibble}.hpp`, `include/strata/kernels/window_limits.hpp`, `include/strata/platform/cpu_relax.hpp`, `src/kernels/cuda/nib_gu_proto.cuh`, `src/kernels/cpu/{portable.cpp,…}` | `auto_cache`: `common/experts/`; `gated_proposal` (and `spec_admission_test`): `common/spec/`; `spec_proposal`, `expert_nibble` (included by `iq_kernels.hpp`), `window_limits`, `nib_gu_proto.cuh`, the CPU files: `common/kernels/`; `cpu_relax`: `base/platform/`. `include/strata/core/defaults_026.hpp` stays beside `arch_defaults`; the new `src/core/window_limits_test.cpp` is in `src/program/` (it reads `program/pipe_probe.hpp`) | merge of c62d5e3a | done |
| `src/dsv41/ds4_os_apple.h`; `include/strata/qwen35/sampling.hpp`; `sycl/include/strata/{kernels/qsa_prompt_attn_esimd,pipeline_auto}.hpp` | `include/strata/base/platform/os_apple.hpp` (`os.hpp` includes it); `include/strata/models/qwen36/sampling.hpp` (its `apply_penalties` takes `serve::Sampling`); `sycl/include/strata/models/qwen38/{kernels/qsa_prompt_attn_esimd,pipeline_auto}.hpp` | merge of c62d5e3a | done |
| `include/strata/program/{stage_reserve,adapt_decay}.hpp`, `src/program/adapt_decay_test.cpp` | `include/strata/models/qwen38/`, `src/models/qwen38/` (memory-manager: `models/qwen38` may not include `program/`) | M | done |
| `include/strata/program/gpu_report.hpp` | `include/strata/base/gpu/gpu_report.hpp` (memory-manager) | M | done |
| upstream's `include/strata/kernels/qsa_two_pass{,_status}.hpp`, `src/kernels/{gr_v3_screen,mmvf_unr_hash,qsa_two_pass_parity}.cpp`, `src/prefill/qsa_nr_parity.cu` | `include/strata/models/qwen38/kernels/`, `src/models/qwen38/kernels/`, `src/models/qwen38/prefill/` | B1 | done |
| upstream's `include/strata/prefill/{mmq_ids_margin,moe_iq4_w4a8,pf_fused_override}.hpp` | `include/strata/common/kernels/prefill/` (the common `moe_mmq`/`moe_fused` include them) | B1 | done |
| `sycl/{include,src}/strata/core/{dflash2,short_cache}*`, `sycl/src/core/{dflash2_ring_test,short_cache_test,verify_refusal_test}.cpp` | `sycl/{include/strata,src}/models/qwen38/` | A3 | done |
| `sycl/src/kernels/{dq_dense_parity,slot_history_test}.cpp` | `sycl/src/common/kernels/` | A3 | done |
| `include/strata/dsv41/sampler.hpp`'s penalty window (216252f0) | `include/strata/common/kernels/sampler_penalties.cuh`, shared by the common sampler and DeepSeek's verify | B2 | done |
| upstream's CMake split by backend (`cmake/backends/*.cmake`, `cmake/strata_gpu_engine.cmake`, 8b166770) | the same files, with the fork's source paths | - | done |
| `include/strata/program/split_walk.hpp` | `include/strata/common/experts/split_walk.hpp` (memory-manager) | M | done |
| upstream's `src/core/dflash2_{q4_parity,ring_test}.cpp` | `src/models/qwen38/` | A3 | done |
| the expert arena's RAM rules in upstream's `include/strata/core/ple_residency.hpp` (`arena_ram_verdict`, `--ram-keep-gib`, from #4846 on) | `include/strata/common/experts/arena_ram.hpp`, which `models/qwen38/ple_residency.hpp` includes | M | done |

## Log

| Upstream | Decision | Here | Notes |
|---|---|---|---|
| `4b9e05a2..c62d5e3a` (0.2.7 to 0.2.9: 733 commits, 146 first-parent units) | carry over, all of it | one merge commit on `local` (not cherry-picks: too many) | details per area below |
| f4612b1a (#3416) gated proposals | carry over | `strata::spec::WindowPlanner` | the planner got a `source` policy and `choose_gated_root()`; `pick_lookup` takes upstream's rule; `spec_cost` is `wplan.cost_aware` |
| 6fb110e5, 1fb3e9a3 coalesced prompt reads (BCO) | carry over | `generate.cpp` | BCO is parsed at the top of the `serve::Loop` loop (an engine command); `pc.drain` is in `hooks.before_line` |
| the concurrent prompt read (`STRATA_BATCH_CONCURRENT_READ`) | carry over | `generate.cpp` | uses `Input::wake`, `wait_for_line`, `take_slot_controls` instead of the old stdin internals |
| 690172c4 `prefill=1`, 9ea04496 `logprobs` / `allowed` | carry over | `generate.cpp` | read from `parse_request_line`'s values |
| 0f7224d1, ca378670 checkpoint opt-ins (`STRATA_CKPT_ASYNC`, `VRAM_SLOTS`, `GATHER`) | carry over | `src/models/qwen38/conversation_state.cpp` | the sync path stays one `copy_running_state`; the async and gather paths are upstream's |
| 07d5ecfa (#4051) `wait_for_ram` | carry over | `src/common/experts/ram_tier.{hpp,cpp}` | |
| `src/dsv41/vendor/sampler.cu` changes (Metal host sampler) | carry over | `src/models/deepseekv41/vendor/sampler.cu` | Metal's `dsv41k::sample_tokens` falls back to the host chain: the CUDA greedy kernel it used is gone (B2) |
| b2dbd942 DeepSeek idle cache settle | carry over | `ds4_main.cpp` around `serve::Loop::next()` | after a command the shared loop answers itself (LIST_CONVERSATIONS), the settle waits for the next engine command; upstream restarted it at once |
| 6c01f85a (#3911) Qwen3.6 penalties | carry over | `include/strata/models/qwen36/sampling.hpp` | over `serve::Sampling` |
| ebe3383a (#3602) PLE reader failed read, SYCL copy | already done | `src/models/qwen38/ngram/ple_reader.cpp` | the SYCL copy stays deleted |
| SYCL `prefill/{moe_fused,moe_fused_iq,moe_mmq}.dp.cpp` | skip | | still never built upstream; upstream only added includes |
| the `serve/` additions (split serving, DeepSeek pictures, web chat page, KaTeX, tests) | carry over | `strata_server/` | imports rewritten as in A2 |
| the engine-client hunks of `serve/server.py` (leftover-engine wait #4051, `PAR_COUNTERS`, coalesced prefill, the parallel and GDN env defaults, logprobs, `prefill_only`) | carry over | `strata_common/engine_client.py` | |
| ISTA's six-shard DeepSeek set (`shards6`, `DSV41_ISTA6_*`, `DSV41_LEGACY_*`, `dsv41_parts`, `dsv41_pack_layout`) | already done | `strata_common/catalog.py` `"layouts"`, `pick_dsv41_layout` | the fork's layouts cover it; upstream's Unsloth UD-IQ1 sets, pictures by default, the re-published shard 1 and the rebuild of a pack from another layout's files went into them |
| 80073010 (#3619, #3326) the redactor's order | carry over | `strata_cli/bench_facts.redactor` | |

**What `memory-manager` must port by hand** when it merges `local` (it moved Qwen3.8's loader and most of `main()` from
`generate.cpp` into `src/models/qwen38/{model,options}.cpp`, and gave the K/V kernels the page-pool `units` argument):

- Into `Model::load()` (`model.cpp`): 7dacaf8e and ad1660aa (PLE table in RAM: its "table + expert arena + 8 GiB" rule
  should count strata-q38's RAM pool), 9c8c9039 (`pcie_share_gfx12_default`, the KVA gate), a3ab37fc
  (`pcie_start_frac`), 2526f2ff (`nv_serial_faster`), 77287f13 (#4474 expert-cache reserve, #4518 PLE lock), c0e245c8
  (`set_batch_slots`), 32f0c44f (the 16-row clamps), d96873fc (2048 fallback), and about 35 hunks of 6ca60e9a (4-bit
  expert slots, later-card VRAM reserve, `STRATA_PIPED_MTP`, ARM64 guards, `drive_pool*_impl`).
- Into `options.{hpp,cpp}`: d96873fc (`prefill_own_keep_chunk`, the gfx1151 16K chunk), 6ca60e9a (`--adapt-decay` check).
- The `units` argument for upstream's new K/V readers: `qsa_block_scores_long` (also in `Verifier::record_window`),
  `qsa_pattn`, the layer-major prefill's `kv_append*` and indexer calls in `prefill.cpp`, and 86bed1c9's new SYCL
  kernels (which take it and refuse it).
- 32f0c44f makes verify windows up to 16 rows (HIP default): memory-manager's 8-row limit (`runconfig._LIVE_RANGE`,
  the window sizing, `share_guess_rows`) can follow `kVerifyMaxT`.
- `src/core/expert_cache.cpp` (`device_slot`, `fill_slot`, `fill_slot_blocking`: `span_`, `get_reach()`); the batch-slot
  code in `strata_common/engine_client.py` (logprobs, parking before an admission, BCO reservations) next to the
  priority and preemption code; `strata_server/web/{app.js,index.html}` (the memory panel's lines); and the
  `set_env` / `unset_env` rename in upstream's new lines of `src/core/{arch,nv}_defaults*.cpp` and the DeepSeek tests.
- Old engine only, keep off in strata-q38: split serving (ccf18ea1), coalesced prompt reads (1fb3e9a3), replicas with
  a shared expert arena (16e08b50), `STRATA_EXPERT_NIBBLE` (slot sizes against the pool's page size), the old
  conversation cache's park and checkpoint opt-ins, `STRATA_RAM_ADAPT`, `STRATA_VRAM_FREE=device`.
- Worth adding to strata-q38 later: `wait_for_ram` before sizing its RAM pool (b82d1d45), `LP` logprob lines in
  `paged_engine.cpp`, and a measurement under the server's Linux CUDA parallel-stack env defaults (dd471d31, 2a9ccfe3,
  b60fbb54), or leaving its engine out of them.

**`memory-manager` merged `local` (debc4619) on 2026-11-09.** The hand ports above are done: the loader hunks are in
`Model::load()` and `options.cpp`; the 4-bit slot code is shared in `include/strata/common/experts/nibble_cache.hpp`;
`Model::is_paged()` keeps 4-bit slots, KVA, the gfx12 0.15 PCIe share and the gfx1151 16K own-buffer chunk off in
strata-q38 until they are measured there; the new K/V readers take `units` (the sm_6x fast `qsa_block_scores_long`,
`qsa_pattn_batch` and the new SYCL entry points refuse it and fall back); strata-q38's rows follow `kVerifyMaxT`
(16 on HIP), `runconfig._LIVE_RANGE` allows 1-16; `child_env` gives paged engines none of the parallel-stack or GDN
env defaults; `split.py` refuses split serving with a paged engine. Still open: `LP` logprob lines over the
request-ID protocol (`generate_request` and `paged_engine.cpp`; until then the server refuses logprobs with a paged
engine), and the Metal DeepSeek build's `FinishIn` layout against memory-manager's `KvRows`.

**0.2.9 to 0.2.13.1, merged 2026-11-09:**

| Upstream | Decision | Here | Notes |
|---|---|---|---|
| `c62d5e3a..504b1433` (0.2.9 to 0.2.13.1: 1231 commits) | carry over, all of it | one merge commit on `local` | details below |
| 8b166770 CMake split by backend | carry over | `cmake/**` | the fork's paths inside the macros |
| 03627326, cc703b5e `--lookup-max-t`; 1ab4c2ce, 1db96cda, 4c5a5549 `--draft auto`; 6de7c8e3 window rule; c2b9218c deferred DFlash2 draft | carry over | `generate.cpp`, `WindowPlanner` (`set_mtp_rows`, `propose_lookup`, `take_lookup`, `begin(pos, WindowRuleIn)`) | `policy_idle` starts as a copy of `wplan.policy` |
| 243fdb21 serve stop bound; d486fbe7 STRATA_FOLLOW; 89f4fa18 run manifest; 24c6c2f6 STRATA_ADAPT_FETCH; ecf7424b `g_lend_keep`; b537586e `--ram-keep-gib`; 62e861c9 `pen_from` | carry over | `generate.cpp` | the manifest flush runs in `hooks.before_line` |
| af7759b6 compiled geometry (`STRATA_GEOMETRY`) | carry over | `common/kernels/cpu/expert.hpp`, `native_expert.hpp`, `moe_fused.hpp` | listed in `KNOWN_UPWARD_INCLUDES` (common sized from `core/geometry.hpp`) |
| 31c3a16f (#4590 Qwen3.6 penalties) | carry over | `include/strata/models/qwen36/sampling.hpp` | |
| 216252f0 penalized DeepSeek verify; 29c2d574, 69896982, 169d5d6a DeepSeek `--parallel` | carry over | `vendor/sampler.cu`, `ds4_main.cpp`, `serve::Input` (`is_stop_requested`, `forget_stop` for `STOP <rid>`) | |
| a280f5bb `ds4_iq_cpu.cpp` (IQ1_S CPU rows) | skip | | not compiled upstream either; `DS4_IQ1S_ROWS` dropped |
| SYCL `prefill/moe_fused{,_iq}.dp.cpp` | skip (deleted) | | still never built upstream |
| `src/split/` (the all-cards split engine), `third_party/strata-ggml/` | carry over | where upstream put them | |
| the `serve/server.py` engine-client hunks (fatal GPU restart e1b27d54, `died_note`, restore points, `--parallel` protocol, split-engine refusals, engine-owned defaults d1121b89) | carry over | `strata_common/engine_client.py` | `request_stats`, `prefill_leg_stats` moved there too |
| catalog additions (Unsloth v3.1 9ed3c778, ISTA-DAS Coder 85dd0852, 128K default bd7e44c9) | carry over | `strata_common/catalog.py` | `dsv41_engine.try_upgrade` deleted (no callers left) |
| upstream's new `strata_control` modules used by the shared layer | carry over | `strata_base/sysinfo.py`; `strata_common/{vram_fit,intel_spec,dsv41_paths,cfg_paths,dsv41_flags}.py` | `shape_engine`, `discovery`, `runtime` stay; known upward loads |

**What `memory-manager` must port by hand** when it merges this: upstream's new Qwen3.8 loader code in `generate.cpp`
(memory rules: `--ram-keep-gib`, `g_lend_keep`, spill gate 2e7db84b, small-card `pf_fused_override` fd59c9c3) goes to
`Model::load()` / `options.cpp`; the `units` argument for new K/V readers (two-pass QSA, layer-split DFlash2 taps);
`paged_engine.cpp` and strata-q38 lack `pen_from`, the stop bound, `--lookup-max-t` and DFlash2. The new engines end
on a failed step with a plain `ERR` line, not upstream's `ERR fatal GPU error: ...` and exit code 3, by which the server
counts restarts (at most 3 in 10 minutes); whether a lost GPU context fails their step is not tested. strata-ds41
needs no `--parallel` protocol (it runs several conversations a step over the request protocol, and the shared
`paged_engine.cpp` already reads prompts in turns between other conversations' steps); it lacks restore points and
the penalized verify.

**`memory-manager` merged `local` at 0.2.13.1 (4281d7a6) on 2026-11-09.** The loader hand ports are in `Model::load()`
and `options.cpp` (`--ram-keep-gib` reaches strata-q38's RAM pool; the spill walk, `g_lend_keep`, the compiled
geometry and DFlash2 loading are shared, DFlash2 and `--lookup-max-t` off in strata-q38; `apply_device_defaults(o,
paged)` skips the engine-owned Linux defaults for it). Two-pass QSA refuses `units` and the prompt path falls back.
One conversation's window in `paged_engine.cpp` stays at `kVerifyNarrowT` (8), since `DraftPolicy::kMaxT` is now 16.
The old DeepSeek engine's `--parallel` runs on `KvRows` (`swap_seq_state`); restore points sit beside
memory-manager's four checkpoints (slot roles 4 and 5). strata-ds41 keeps `--batch`; PCIe lane, one-GPU decode,
recent slots and prompt-path slot borrowing stay off for it. Still open for the new engines: upstream's
`ERR fatal GPU error` line with exit code 3 (they end on a failed step with a plain `ERR`), the penalty window
over the answer alone (strata-q38 counts the prompt too) and `pen_from`, the stop bound, DFlash2, `--lookup-max-t`,
and for strata-ds41 penalties in its joint verify and restore points.
