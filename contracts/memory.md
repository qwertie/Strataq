# Labelled memory reports

The shared protocol is defined in
[`memory_report.hpp`](../include/strata/common/serve/memory_report.hpp).
`GET_MEMORY_USE` and `CHANGE_SETTINGS` return the same report.

Each `MEMORY` line identifies a GPU, RAM or SSD and its sizes and settings.
For an adjustable pool, `pool_offset` gives the bytes before it on the bar,
so category names do not determine the divider position.
It is followed by any number of lines of this form:

```text
MEMORY_BLOCK gpu0 1048576 other Device%20allocations Drafter
```

The fields are memory ID, bytes, category, pool name and block label.
The last three fields are percent-encoded UTF-8; `~` represents an empty string.
Categories are engine-provided strings. There is no category allowlist in the
shared engine, Python parser or web panel. Older engines may still send category
totals as numeric fields on `MEMORY`, without block details.

Blocks are disjoint totals, grouped by matching labels. Backing allocations for a
page pool are excluded when its blocks are reported. Page padding belongs to the
block, and unused pool space is reported separately. Experts retain their layer
and size class; conversation blocks retain their conversation and state labels.
When a pool starts inside its backing allocation for alignment, prefix and suffix
bytes are reported separately as `Alignment padding`.

The GPU and pinned-host allocators retain allocation lifetimes through
`base/gpu/managed_memory.hpp` and the shared `MemoryManager`. Page pools use the
same label type. Engines name significant allocations at their allocation sites.
The shared paged engine combines these records with each page pool's block report.
Adding a category or sublabel requires no Python or UI changes.

Python exposes `uses` (category totals) and `blocks` (category, pool, label, bytes).
The web panel and desktop app share the same renderer. Segment and legend tooltips
show all matching block labels and sizes. `other` is Strata's own miscellaneous
allocations. `other_apps` is memory outside the engine's accounted use, and is
drawn after free memory at the right of the bar. Where Windows provides the
engine's dedicated usage, unclassified runtime bytes remain under Strata's Other.

Pool growth is limited by actual allocation capacity and configured pool limits.
There is no automatic Windows commit reserve. The operating system and allocator
can still refuse an allocation, which is reported through the existing refusal
or shortfall response.

Blocks may append an identity group and comma-separated unsigned member IDs:
`MEMORY_BLOCK <memory> <bytes> <category> <pool> <label> <identity-group> <ids>`.
The group is percent-encoded like the labels. Members within one block have
identical allocated sizes (`bytes / member count`). A group and member ID
identify the same logical content across memories, independently of display
labels. Page pools use expert category and layer as the group, and the expert
index as the member ID. Readers that do not need residency can ignore these
optional fields. The Python view preserves them as `identity_group` and `ids`.

The diagram shades RAM experts also held by any GPU in darker teal. Those
copies still count as allocated RAM. SSD experts mean experts held in neither
VRAM nor RAM, rather than the size of the complete model files on disk.

`CONVERSATION` lines can append `last_tokens_at=<Unix seconds>` before the
optional token preview. This is the last time prompt or generated tokens
reached that logical conversation. Pausing, copying and restoring preserve
it. Missing times are unknown, including files written before this field
was recorded. `PAUSE_DIRECTORY <path>` gives the engine's folder for its
automatic SSD pause files, with spaces retained to the end of the line.
The memory view passes it through as `pause_directory`.

Paged engines report three distinct token settings. `tokens_max` is the combined
logical token allowance for conversations resident on the GPUs; a conversation
split across GPUs counts once. GPU state that moves to RAM or SSD returns its
allowance. Physical KV padding and speculative claims do not increase the
displayed conversation counts. Admission and decode growth both enforce the
allowance, pausing idle caches first and then the lowest-priority running request
when needed. Equal priorities keep older requests running first.

`context` is the per-conversation token limit, and `context_capacity` is the
capacity loaded into the model. The latter also advertises that `context` may
change live. Both `context` and `tokens_max` may change in `CHANGE_SETTINGS`,
but a proposed `context` above either `tokens_max` or `context_capacity` is
rejected before settings change. Saved defaults use `--conversation-tokens`
for the per-conversation limit; `--max-context` still sets loaded capacity.
The memory table shows the shared GPU budget beside GPU counts, each individual
limit in a tooltip, and the combined GPU count in its footer.

`vram_conversations` limits the number of GPU-resident conversations, including
idle caches and their fixed regions. Its default leaves one cached place beyond decode concurrency. Qwen gets
one further place when every participating card has at least 10 GiB VRAM.
With decode concurrency one, these defaults are two and three. DeepSeek keeps
the smaller default to leave more space for experts. Explicit settings are preserved. The residency limit must be at least
`concurrent` and at most the engine's `slots`; conflicting edits are rejected.
`--vram-conversations` saves its default. `INFO vram_conversations` advertises
support and lets Python wait before admitting work that cannot fit. The engine
enforces the limit before allocating or restoring fixed state, reclaiming idle
caches first. Lowering the limit can pause the lowest-priority running requests.
Layer-split state counts as one conversation across all cards. Empty fixed
regions count too, and are released when their place is needed.

The existing `conversation_bytes` reports fixed pool state per conversation,
before the growing KV cache. The UI displays this beside the residency control.
For Qwen this includes only live recurrent state. Rewind checkpoints are
allocated individually in RAM as they are captured; evicting a checkpoint frees
its payload. DeepSeek already keeps its ring and compressor checkpoints in RAM.
Both families report host checkpoint storage under the RAM category `Checkpoints`,
labelled by conversation. Active checkpoint heap storage is outside the adjustable
RAM page pool. Paused host payloads count against the RAM pause allowance.

`conversation_idle_seconds` (`--conversation-idle-seconds`) defaults to 66.
Every completed prompt-processing chunk and decode step updates `last_tokens_at`.
When an allocation needs pages, idle conversations at least this old may move to
RAM or SSD before experts are displaced. Blocked expert refill can reclaim those
same idle conversations. Age alone does not evict anything. The prompt reader
and decoding requests are excluded, and queued GPU state writes drain before
copying or releasing pages. A zero age makes all idle caches eligible immediately.
The hard count limit may reclaim newer conversations when necessary.

RAM checkpoints are serialized inside the shared pause-file host header, so named
SAVE/RESTORE and automatic SSD pauses retain them. Disk-paused payloads are released
from RAM after writing and read back on resume. Qwen uses layout identity 4; files
from its former GPU-checkpoint layout are rejected without changing them. DeepSeek's
existing host-state file layout is unchanged.

The report may include `MEMORY_DETAIL <key> <percent-encoded-value>` lines.
Python exposes these as the `details` dictionary. The paged engines supply
`drafter_label`, `drafter_details`, `suffix_label` and `kv_quantization` for
settings help. Drafter cost includes reloadable dense weights and prediction
experts; per-conversation state is accounted separately. Older engines omit
these fields, so readers must allow missing discovery and cost information.

Paged engines report `has_drafter`, `suffix_draft` and `lookup_chain` alongside
`drafter`. The latter two settings accept the corresponding numeric startup
values in `CHANGE_SETTINGS`. The UI selects suffix drafting with a minimum
three-token match and no lookup chain. It requires greedy sampling. Planner
changes apply to the next request; an existing request retains its planner.
Unused protocol addresses are omitted from conversation reports. The `main`
address remains for compatibility with solo requests; it is not a separate
conversation category in the UI. Empty scheduling records allocate no KV pages.

The Memory form stages edits until Apply settings, or Apply and save as default.
Memory dividers still apply when released. Context fill counts GPU-resident tokens
across conversations against `tokens_max`, including idle contexts still held.
The API monitor's Context in VRAM table selects the context to inspect.

The server serializes settings commands with engine protocol operations. A solo
request holds the command channel until generation ends, so Apply can wait for
that boundary. Batch commands may run between engine steps when the control
channel is available. HTTP itself does not require waiting for a generation.
