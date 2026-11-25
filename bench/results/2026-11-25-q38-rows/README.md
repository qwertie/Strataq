# strata-q38: conversations at once and rows per step, 25 November 2026

`tools/measure_rows.py` on the Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB unified memory, Windows 11),
Qwen3.8-Flash-Next IQ3_S, engine 0.1.59.1, every expert on the GPU. The run config was `strata-iq3_s.json` (int8 K/V,
the MTP drafter, `--spec-min-p 0.5`, `--draft-rule typical`) with `--ram-mib 2048 --save-on-shutdown off` added.
The GPU pool was 50.74 GiB for the greedy run and 48.75 GiB (`--gpu-mib 50000`) for the sampled run, which ran after a
restart, when 51,000 MiB no longer started (see below). Both pools held all 47.5 GiB of experts.
After its other allocations, the engine reported a 1.03 GiB RAM pool in both runs.

Each conversation decoded 1,024 tokens. The prompts were a five-chapter story, "Print out a JS-based game of
Tetris in HTML" and a ten-day trip plan; no run ended before 1,024 tokens. Three rounds, the order reversed every
other round.

- `greedy.json`: no sampling keys (temperature 0). A guess is kept only when it is the most likely token.
- `sampled.json`: `temperature=0.7 seed=1`, so `--draft-rule typical` applies.

The tables and what follows from them are in [docs/DETAILS.md](../../../docs/DETAILS.md#rows-per-step-lookahead-and-conversations-at-once-strata-q38).
`python tools/measure_rows.py --report greedy.json sampled.json` prints the medians.

Limits: one computer and one model pack. In the greedy run's third round most configurations were 5-35% slower
than the mean of the first two rounds. Windows Setup was running at the time. The medians keep that round's values
out except where they were in the middle. The sampled run was 6-10% faster than the greedy run even without
guesses (1 x 1), so the two
runs differ by more than the sampling. The experts-per-step column counts only what the CPU sees, which is almost
nothing with every expert on the GPU.

Before these runs, the engine crashed at start with a C++ exception, and once hung, with the config's
`--ram-mib 14472` when Windows' commit limit was nearly used up. After the restart, `--gpu-mib 52210` exited without
a message once the GPU had room for only 202 of 204 pool chunks. These crashes are being investigated separately.
