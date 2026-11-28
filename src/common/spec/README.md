# src/common/spec/

Speculative decoding: drafting, accepting and rejecting drafts, rolling back. Headers: `include/strata/common/spec/`.

Only the Qwen3.8 engine uses it. `window_planner.hpp` chooses what each verify window of `strata-q38`'s decode step
holds: the MTP's drafts, a lookup window or a lookup chain. Qwen3.6's MTP steps and
DeepSeek's NextN drafts have no lookup and keep their own short loops.

`step.hpp` is the decode step of [docs/MEMORY_MANAGER_PLAN.md](../../../docs/MEMORY_MANAGER_PLAN.md) (steps M3
and M4): several conversations' rows in one pass through the model, the rows left over shared among their guesses
(`share_guess_rows`, the plan's rule: the extra rows go to the highest priority and rotate between equals), and each
conversation's accepted guesses kept. The model gives it running rows with a parent table, keeping the accepted
ones, and proposing guesses (in `strata-q38`, Qwen3.8's MTP drafter). `strata-q38` decodes through it.

See [docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the layers and
[docs/REFACTOR_PLAN.md](../../../docs/REFACTOR_PLAN.md) for the steps.
