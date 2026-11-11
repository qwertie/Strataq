// include/strata/models/qwen38/paged_model.hpp - strata-q38: Qwen3.8 on the shared paged engine, whose
// conversations live in a page pool (strata/common/serve/paged_engine.hpp; docs/MEMORY_MANAGER_PLAN.md, steps M2d to
// M9 and D3).
//
// Each conversation's GDN state and K/V (and the MTP drafter's K/V) take page-pool pages as it grows; a request
// decodes through the verify windows, one row per conversation and guess.  On a layer split
// (`--layer-split K[,K2..]|auto`, docs/MEMORY_MANAGER_PLAN.md step M8) the verify windows and the prompt path hand
// their rows from card to card.  `--split-device 0,..` puts several cards on one GPU, each with its own weights and
// pool: the test of a split on a PC with one GPU.  The MTP drafter runs on the last card.
#pragma once

namespace strata::qwen38 {

/// Runs strata-q38 with `strata`'s command line, plus `--tokens-min N`: the conversation tokens whose pages the
/// experts leave free (default 8192); `--rows N`: the most rows one step computes (default 8, at least the
/// conversations at once); `--lookahead N`: the most guesses a conversation adds to a step (default 3 with --mtp,
/// whose drafter stops guessing below --spec-min-p, and 0, no drafting, without it); `--drafter on|off`: with a
/// lookahead, whether the drafter starts loaded (off: loaded for the conversations' layout, then given back, until
/// CHANGE_SETTINGS turns it on).  `--batch N` is the batch slots, and `--concurrent N` how many of them decode at once
/// (default all).  `--expert-cache N` is the most experts the GPU holds, as many as N slots of the largest expert
/// hold; `auto`, as many as fit; on a layer split, the profile's first N, each on the card of its layer.  `--gpu-mib
/// N[,N2..]`: the GPU's pool, one per card of a layer split (one value: each card's) (default: what is free less
/// --vram-reserve-mib and the prompt path's buffers, but no more than those experts and a conversation at
/// --max-context take).  The RAM:
/// `--ram-mib N`, the RAM's pool (default: what the RAM has free less the resident headroom, but no more than every
/// expert takes, and like the GPU's default leaving a tenth of Windows' commit limit, at least 8 GiB, uncommitted;
/// `--resident-budget-gib N` stands for it);
/// `--tokens-max N`, the tokens the conversations paused in RAM may hold (as many pages as one conversation of N
/// takes; default --max-context); `--ssd-pause-mib N`, the SSD space paused conversations may take (default 8192);
/// `--pause-dir DIR`, where their files go (default the temporary folder's strata-q38); `--copy-back auto|on|off`,
/// whether an expert the GPU evicts is copied back to RAM (auto: when that measures at least 1.2 times as fast as
/// reading the SSD, or as fast for an expert a prompt's buffers evict); `--remote-experts auto|on|off`, whether a
/// card's spare pages hold other cards' experts.
/// `--mmap-experts`, `--resident-experts` and `--resident-cpu-experts` are taken and do nothing: the RAM's pool
/// replaces them.  Returns the exit status.
int run_paged_engine(int argc, char** argv);

}  // namespace strata::qwen38
