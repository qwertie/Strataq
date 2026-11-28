# Benchmark results

Raw results and short reports from measurement runs, one folder per run, named by date. They stay here as they were written: each states
the Strata version, hardware and method of its own day, and none is a statement about a later release. Current numbers are in the
[README](../../README.md) and [ENGINE_NOTES.md](../../docs/ENGINE_NOTES.md).

This page is built by `python tools/bench_results_index.py` (`--check` fails when a folder is missing from it). 5 folders and 0 loose files, 82 files in all.


## 2026-11

| Folder | What | Files |
| --- | --- | ---: |
| [2026-11-03-q38-realistic](2026-11-03-q38-realistic/) | Qwen3.8 IQ3_S with all experts on the GPU | 13 |
| [2026-11-04-ds41-realistic](2026-11-04-ds41-realistic/) | DeepSeek realistic pools, 4 November 2026 | 4 |
| [2026-11-05-ds41-drafting](2026-11-05-ds41-drafting/) | DeepSeek batched DSpark drafting, 5 November 2026 | 4 |
| [2026-11-05-old-local-vs-new](2026-11-05-old-local-vs-new/) | Old local versus new engines, 5 November 2026 | 7 |

## undated

| Folder | What | Files |
| --- | --- | ---: |
| [ds41-groups-20261104](ds41-groups-20261104/) | DeepSeek shared pipeline groups, 4 November 2026 | 54 |
