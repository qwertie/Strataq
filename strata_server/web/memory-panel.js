// strata_server/web/memory-panel.js - the Memory panel (docs/MEMORY_MANAGER_PLAN.md, steps M0, M0b and M6): what the
// engine holds in each memory (each GPU, the RAM, the SSD), the conversations it holds and its settings, drawn from
// GET /v1/memory (strata_server/server.py memory_view).  An engine that changes its settings while it runs
// (`can_change`) gets dividers to drag on the bars and controls for its other settings; the caller sends the changes
// (POST /v1/memory).  No framework: the web page (app.js) imports this file as it is, and the desktop app
// (apps/host/src/components/MemoryPanel.tsx) wraps it.  Its styles are memory-panel.css beside it.
//
// A GPU whose memory is system RAM (Strix Halo) is drawn as the BIOS divides it: its bar is the BIOS share, with the
// engine's uses drawn scaled to what the engine holds in the share (`in_share`; the engine cannot know which
// allocation landed where; the legend keeps the engine's numbers), and what it borrows beyond the share is one segment
// of the RAM's bar.  A bar is as wide as its memory's share of FULL_BAR (one of unknown size fills it), so a card is
// seen beside the RAM without becoming a sliver.

/** Names shown for the uses a memory's bar is divided into (`uses` keys and the segments the panel adds). */
const USES = {weights: "Weights", experts: "Experts", redundant_experts: "Experts also in VRAM",
              conversations: "Conversations", pool_free: "Free in the pool",
              paused: "Paused conversations", paused_unusable: "Unusable pause files", drafter: "Drafter",
              drafter_experts: "Drafter experts", prompt_buffers: "Prompt buffers",
              scratch: "Scratch", engram: "Engram",
              borrowed: "GPU (borrowed from RAM)", other: "Other", other_apps: "Other apps", free: "Free",
              kept_free: "Kept free for other programs", beyond_dedicated: "Not usable as dedicated memory"};
/** What each use of a bar is, in plain words: the tooltip of its segment and its legend entry. */
const USE_HELP = {
  weights: "The parts of the model that must always be loaded here (attention, the shared layers, the output head), " +
    "unlike experts, which are loaded as they are needed",
  experts: "Experts give the model most of its knowledge and skill. Fitting more in VRAM makes the model go faster. " +
    "Conversation memory (KV cache) can take space away from experts.",
  redundant_experts: "RAM copies of experts also held in VRAM. These avoid reading the model files again " +
    "or copying weights back from VRAM when an expert is evicted.",
  conversations: "What the model remembers of the conversations it is working on (their context)",
  pool_free: "Room the engine has set aside here that nothing uses yet; experts or conversations take it as needed",
  prompt_buffers: "Working space lent from the pool while a prompt is read; experts and conversations get it back " +
    "when the prompt has been read",
  paused: "Conversations put aside for now, kept here until they continue",
  paused_unusable: "Paused conversations this model cannot continue: saved by another model or with other " +
    "settings, or damaged. They count toward the SSD allowance and are deleted first, the oldest first, when " +
    "paused conversations need the room",
  drafter_experts: "Expert weights used by the model that guesses words ahead",
  drafter: "The small model that guesses the next words ahead, so that answers come faster",
  scratch: "Working space the engine needs while it computes",
  engram: "DeepSeek's table of word patterns, read as it answers",
  borrowed: "Memory the graphics chip borrows from the system RAM beyond its own share",
  other: "Strata allocations outside the expert and conversation pools",
  other_apps: "Memory used outside Strata, including other programs and Windows",
  free: "Not in use by anything",
  kept_free: "Memory Strata leaves free so that other programs, and Strata's own buffers outside its pools, have " +
    "room. The setting \"Reserve free VRAM\" sizes it",
  beyond_dedicated: "Free in the graphics chip's own memory, yet unusable: when Strata's pool last grew, Windows " +
    "put the next piece in slower shared memory although this much was still free here. So the pool cannot grow " +
    "further in dedicated memory",
};
/** Pool categories used by engines that do not report an explicit adjustable byte interval. */
const POOL_USES = ["experts", "redundant_experts", "conversations", "paused", "paused_unusable", "prompt_buffers",
                   "pool_free"];
/** The SSD's uses that its paused budget (`ssd_pause_mib`) limits. */
const SSD_BUDGET_USES = ["paused", "paused_unusable"];
/** Size in bytes (32 GiB) at which a memory's bar fills the width. */
const FULL_BAR = 32 * 1073741824;
/** Bytes in a MiB, the unit the memory settings are sent in. */
const MIB = 1048576;
/** How far one arrow key moves a divider, in bytes (with Shift held: four times as far). */
const KEY_STEP = 256 * MIB;
/** How long a change's Undo (and its message) stays, in milliseconds. */
const UNDO_MS = 8000;
/** localStorage key set once the reader has changed a size; the hint under the panel then stays away. */
const CHANGED_KEY = "strata.memoryPanel.changed";
/** Names shown for the engine's settings (keys of the view's `settings`); others show as the engine names them.
 * @type {Record<string, string>} */
const SETTINGS = {context: "Per-conversation token limit", concurrent: "Conversations at once", rows: "Rows per step",
                  vram_conversations: "Max conversations in VRAM",
                  conversation_idle_seconds: "Idle conversation eviction age (sec)",
                  decode_share_percent: "Decoding while a prompt is read (%)",
                  prompt_turn_ms: "Prompt reading turn (ms)",
                  prefill_chunk: "Max prompt chunk (tokens)",
                  lookahead: "Max lookahead", tokens_min: "Min conversation tokens",
                  tokens_max: "Max conversation tokens", ssd_pause_mib: "SSD for paused conversations (MiB)",
                  drafter: "Drafter", save_on_shutdown: "Save conversations on shutdown",
                  step_profile: "Log each step's timing",
                  vram_reserve_mib: "Reserve free VRAM (MiB)",
                  slots: "Maximum conversations at once", batch_groups: "Batch groups (0: auto)",
                  route_tail_skip: "⚠ Minimum experts per layer",
                  route_prior_pct: "⚠ Bias toward loaded experts (%)",
                  draft_confidence_pct: "Min draft confidence per token (%)",
                  draft_cumulative_pct: "Minimum draft confidence (%)",
                  expert_half_life: "Expert popularity half-life steps",
                  copy_back: "Experts the GPU evicts"};
/** Engine flags whose names are not the setting's name with dashes; the drafter list sets three. */
const FLAGS = {context: "--conversation-tokens", drafter: "--drafter, --suffix-draft, --lookup-chain"};
/** The number settings whose 0 (off) is shown as an empty field. */
const BLANK_ZERO = ["route_tail_skip"];
/** The settings that are switches: the engine reports them as 1 or 0 and takes "on" or "off". */
const SWITCHES = ["drafter", "save_on_shutdown", "step_profile"];
/** Defines the Memory panel's prediction modes. Suffix guesses may replace model guesses; chained guesses follow
 * them. A null label uses the mode's own words instead of extending the model drafter's label.
 * @type {[choice: string, use: {model: boolean, suffix: boolean, chain: boolean}, label: string | null][]} */
const DRAFTER_CHOICES = [["off", {model: false, suffix: false, chain: false}, null],
                         ["on", {model: true, suffix: false, chain: false}, ""],
                         ["suffix", {model: false, suffix: true, chain: false}, null],
                         ["either", {model: true, suffix: true, chain: false}, ", or suffix drafter"],
                         ["chain", {model: true, suffix: false, chain: true}, " + lookup"],
                         ["all", {model: true, suffix: true, chain: true}, " + lookup, or suffix drafter"]];
/** The settings that take one of a few words: the engine reports the word's index and takes the word.
 * @type {Record<string, string[]>} */
const WORDS = {copy_back: ["off", "on", "auto"]};
/** What each word of WORDS' settings shows in its list. @type {Record<string, string[]>} */
const WORD_LABELS = {copy_back: ["Dropped (read from the SSD again)", "Copied back to RAM",
  "Copied back to RAM if faster"]};
/** The draft acceptance rules offered (the engine's `draft_rule` detail and setting) and their labels; an asterisk
 * marks a rule that trades the answers' distribution for speed (DRAFT_RULE_HELP).
 * @type {[rule: string, label: string][]} */
const DRAFT_RULES = [["exact", "Exact"], ["speculative", "Speculative sampling"],
  ["typical", "Typical acceptance (Medusa) *"], ["lower-temperature", "Lower temperature *"],
  ...[75, 90, 99].map((n) => /** @type {[string, string]} */ ([`top-${n}`, `Top ${n}% *`]))];
/** The draft acceptance choice's tooltip: what every rule in DRAFT_RULES does. */
const DRAFT_RULE_HELP = [
  "Which of the drafter's guessed words an answer keeps. At temperature 0 all options give the same answers.",
  "Exact: keeps a guess only when it is the word the model would pick anyway, so answers are word for word " +
    "the same as without guessing.",
  "Speculative sampling: keeps more guesses. The words can differ from Exact's, but each word is exactly as " +
    "likely as without guessing, so answers are just as good.",
  "Typical acceptance (Medusa): also keeps a guess the model finds likely enough; the less sure the model is, " +
    "the more guesses pass.",
  "Lower temperature: also keeps a guess the model finds at least as likely as the word it picked.",
  "Top N%: as Lower temperature, and also keeps a guess among the model's likeliest words that together hold " +
    "N% of its probability.",
  "* this option can change the output distribution"].join("\n");
/** Guidance shared by the per-conversation limit's tooltip and capacity error. */
const CONTEXT_CAPACITY_HELP = "To raise this limit, increase --max-context in the model's engine arguments, " +
  "then reload the model.";
/** Settings the engine reports that the panel shows elsewhere (the bars) or not at all. */
const UNLISTED = new RegExp("^memory_mib\\.|^(token_bytes|conversation_bytes|context_capacity|" +
  "has_drafter|suffix_draft|lookup_chain|slots|ssd_pause_mib|batch_groups_now|pipeline_stages|" +
  "prefill_chunk_capacity|route_tail_skip_max)$");
/**
 * The number settings the panel's controls change, with the range each accepts given the reading's settings `s`
 * (the engine checks them again).
 * @type {[key: string, smallest: (s: Record<string, number>) => number,
 *   largest: (s: Record<string, number>) => number][]}
 */
const CONTROLS = [["concurrent", () => 1, (s) => Math.min(s.slots ?? 8, s.rows ?? 8, s.vram_conversations ?? Infinity)],
                  ["vram_conversations", () => 1, (s) => s.slots ?? 8],
                  ["conversation_idle_seconds", () => 0, () => Infinity],
                  ["decode_share_percent", () => 0, () => Infinity],
                  ["prompt_turn_ms", () => 0, () => Infinity],
                  ["prefill_chunk", (s) => Math.min(256, s.prefill_chunk_capacity ?? 256), () => Infinity],
                  ["rows", (s) => s.concurrent ?? 1, () => 8],
                  ["lookahead", () => 0, (s) => (s.rows ?? 1) - 1],
                  ["batch_groups", () => 0, () => 2147483647],
                  ["tokens_min", () => 0, (s) => s.tokens_max ?? Infinity],
                  ["tokens_max", (s) => s.tokens_min ?? 0, () => Infinity],
                  ["context", () => 1, (s) => s.context_capacity ?? s.context ?? Infinity],
                  ["vram_reserve_mib", () => 0, () => Infinity],
                  ["route_tail_skip", () => 0, (s) => s.route_tail_skip_max ?? 9],
                  ["route_prior_pct", () => 0, () => 10000],
                  ["draft_confidence_pct", () => 0, () => 100],
                  ["draft_cumulative_pct", () => 0, () => 100],
                  ["expert_half_life", () => 0, () => 2147483647]];
/** What biasing the choice toward experts on the GPU does, and why it is risky: the route_prior_pct tooltip and the
 *  question asked before it is switched on. */
const ROUTE_PRIOR_WARNING = "When the model picks the experts for each word, it prefers those already in the GPU's " +
  "memory, so fewer are read from RAM or the SSD. This may speed up a card that holds few experts " +
  "(RTX 3060 12 GB, IQ3_XXS, 348%: story +15%, code +37% decode), but answers can get worse. " +
  "Long agent sessions repeated themselves and solved fewer tasks in our checks.";
/** HTML entities for the characters esc replaces. @type {Record<string, string>} */
const ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
/** Names shown for where a conversation is held (its `place`). */
const PLACES = {vram: "GPU", ram: "RAM", ssd: "SSD"};
/** What each element the panel is drawn into shows besides the reading. @type {WeakMap<HTMLElement, Panel>} */
const panels = new WeakMap();

/**
 * Draws a GET /v1/memory answer whose `available` is true into `element`: a bar per memory
 * with its legend and the engine's settings with a line that says
 * whether the reading is an old one. Existing elements and focused edits survive each reading. While a divider is
 * being dragged or a change is being applied, the new reading is drawn afterwards.
 * @param {HTMLElement} element
 * @param {MemoryView} view
 * @param {ChangeSettings} [onChange] sends a change to the server; without it the settings are only shown.
 */
export function drawMemoryPanel(element, view, onChange) {
  let panel = panels.get(element);
  if (!panel) {
    panel = {view, dividers: {}, drag: null, busy: false, notice: null, pending: {}};
    panels.set(element, panel);
    listen(element, panel);
  }
  Object.assign(panel, {view, onChange});
  const active = element.ownerDocument.activeElement, inside = !!active && element.contains(active);
  if (panel.drag?.x0 === null && !inside)
    panel.drag = null;    // a keyboard drag the reader left by moving the focus away; it would stop all redraws
  if (!panel.drag && !panel.busy)
    render(element, panel);
}

/**
 * The GET /v1/memory answer as the panel reads it (the fields memory_view and StrataEngine.memory describe).
 * `can_change`: the engine changes its settings while it runs; `config`: a run config exists to save defaults to;
 * `shortfall` (POST /v1/memory's answer): what the change got less of than asked, in the engine's words.
 * @typedef {{available: boolean, reason?: string, busy?: boolean, error?: string | null, can_change?: boolean,
 *   config?: boolean, details?: Record<string, string>, memories: MemoryUse[], settings: Record<string, number>,
 *   conversations?: HeldConversation[], shortfall?: string, pause_directory?: string}} MemoryView
 * `setting`: the size of the engine's page pool there now, which `memory_mib.<id>` changes; `smallest`, `largest`:
 * the sizes it accepts now (a GPU's `largest`: what its dedicated memory holds, as last measured).  `pool_shared`: the
 * bytes of the pool in a GPU's shared memory; `dedicated_full`: its last growth stopped where the dedicated memory
 * ended.  `kept_free`: what the engine's pools leave free there for other programs; `dedicated_room`: bytes below
 * the measured dedicated-memory limit minus all programs' current use, from Windows' D3DKMT statistics. Absent
 * until a growth finds the limit; zero means full. The bar's free-memory counter can differ, so room is capped
 * at `free` and the rest of `free` is drawn as unusable.  `commit_room` (the RAM, on Windows): what Windows can still
 * promise programs (its commit limit, the RAM plus the page file, less all programs' commit).  Bytes.
 * @typedef {{id: string, name: string | null, total: number | null, free: number | null, unified: boolean,
 *   uses: Record<string, number>, paused_count: number | null, paused_budget: number | null,
 *   other?: number, other_apps?: number,
 *   blocks?: MemoryBlock[],
 *   in_share?: number, borrowed?: number | null, gpu_borrowed?: number, allocated_pct?: number | null,
 *   fragmented_pct?: number | null, setting?: number | null, pool_offset?: number | null, smallest?: number | null,
 *   largest?: number | null, pool_shared?: number | null, dedicated_full?: boolean, kept_free?: number | null,
 *   dedicated_room?: number | null, commit_room?: number | null}} MemoryUse
 * @typedef {{category: string, pool: string, label: string, bytes: number,
 *   identity_group?: string, ids?: number[]}} MemoryBlock
 * @typedef {{id: string, place: string, state: string, count: number, capacity: number, bytes: number,
 *   used: number, last_tokens_at?: number, name?: string, preview?: string}} HeldConversation
 * @typedef {Record<string, number | boolean | string>} SettingsChange  POST /v1/memory's "set": MiB for the sizes.
 * @callback ChangeSettings  Sends a change to the engine's server (POST /v1/memory) and says how it went: `view` is
 *   the reading after it, `error` the server's reason for not changing anything (a refusal, 409).
 * @param {SettingsChange} set
 * @param {"session" | "default"} save  "default" also writes the change into the run config.
 * @returns {Promise<{ok: boolean, view?: MemoryView, error?: string}>}
 */

/**
 * What a drawn panel shows besides its reading.  `dividers`: by memory id, those drawn last; `busy`: a change is
 * on its way; `pending`: edits retained until Apply succeeds.
 * @typedef {{view: MemoryView, onChange?: ChangeSettings, dividers: Record<string, Divider>, drag: Drag | null,
 *   busy: boolean, notice: Notice | null, pending: SettingsChange}} Panel
 * A boundary of a bar that can be dragged.  `key`: the setting it changes; `from`: the size now; `start`: where the
 * segment it ends begins on the bar; `scale`: drawn bytes per byte of the engine's; `total`: the drawn bytes the
 * bar's width stands for; `fixed`: Strata allocations outside the adjustable pool; `shared`: the pool's bytes in
 * shared memory; `commit`: Windows' remaining allocation capacity when below the known free RAM (else null).
 * Bytes.
 * @typedef {{key: string, from: number, smallest: number, largest: number, start: number, scale: number,
 *   total: number, fixed: number, shared: number, commit: number | null}} Divider
 * A divider being moved.  `value`: the size it would set (bytes); `x0`: the pointer's x where the drag started (null
 * for the keyboard); `perPx`: bytes per pixel of the bar.
 * @typedef {{id: string, value: number, x0: number | null, perPx: number}} Drag
 * A line under the panel.  `until`: the time (ms) it goes away; `undo`: the change that undoes the last one.
 * @typedef {{text: string, error: boolean, until: number,
 *   undo?: {set: SettingsChange, save: "session" | "default"}}} Notice
 */

/** Draws the panel's reading and its state into `element`.
 * @param {HTMLElement} element
 * @param {Panel} panel */
function render(element, panel) {
  const {view, drag} = panel, editable = !!(view.can_change && panel.onChange);
  const memories = markRedundantExperts(view.memories);
  panel.dividers = editable ? Object.fromEntries(memories.map((x) => [x.id, findDivider(x)]).filter(([, d]) => d))
    : {};
  const listed = Object.entries(view.settings).filter(([k]) => !UNLISTED.test(k) &&
    !(editable && (SWITCHES.includes(k) || k in WORDS ||
      getEditableControls(view.settings).some(([key]) => key === k))));
  const note = [listed.map(([k, v]) => `${SETTINGS[k] || k}: ${SWITCHES.includes(k) ?
    (v ? "on" : "off") : WORDS[k]?.[v] ?? (BLANK_ZERO.includes(k) && v === 0 ? "all" : fmt(v))}` +
      (k === "context" ? " tokens" : "")).join(" · "),
    view.can_change === false ? "these change when the model starts again" : "",
    view.busy ? "a request has the engine: the last reading" : view.error || ""].filter(Boolean).join(" · ");
  const notice = panel.notice && panel.notice.until > Date.now() ? panel.notice : null;
  const hint = Object.keys(panel.dividers).length && !hasChanged() ?
    `<p class="mem-hint">Drag the dividers to change sizes</p>` : "";
  const html = `<div class="mem-bars">${memories.map((x) =>
      drawBar(x, panel.dividers[x.id], drag?.id === x.id ? drag : null, view)).join("")}</div>` +
    (editable ? drawControls(panel) : "") + hint +
    (notice ? `<p class="mem-notice${notice.error ? " mem-error" : ""}" role="status">${esc(notice.text)}` +
      (notice.undo && !panel.busy ? ` <button type="button" class="mem-undo" data-undo>Undo</button>` : "") + "</p>"
      : "") +
    (note ? `<p class="mem-note">${esc(note)}</p>` : "");
  const template = element.ownerDocument.createElement("template");
  template.innerHTML = html;
  updateChildren(element, template.content);
  if (drag)
    /** @type {HTMLElement | null} */ (element.querySelector(`[data-divider="${drag.id}"]`))?.focus();
}

/** Makes `element` answer the reader: dragging a divider (pointer or keys), the controls and Undo.
 * @param {HTMLElement} element
 * @param {Panel} panel */
function listen(element, panel) {
  const redraw = () => render(element, panel);
  element.addEventListener("pointerdown", (e) => {
    const hit = findHit(e);
    if (!hit || e.button !== 0 || panel.busy)
      return;
    e.preventDefault();
    element.setPointerCapture(e.pointerId);
    const {id, d, target} = hit, bar = /** @type {HTMLElement} */ (target.parentElement);
    panel.drag = {id, value: d.from, x0: e.clientX, perPx: d.total / d.scale / bar.getBoundingClientRect().width};
    redraw();
  });
  element.addEventListener("pointermove", (e) => {
    const drag = panel.drag;
    if (drag?.x0 == null)
      return;
    const value = panel.dividers[drag.id].from + (e.clientX - drag.x0) * drag.perPx;
    drag.value = Math.max(0, mib(value) * MIB);
    redraw();
  });
  element.addEventListener("pointerup", () => {
    if (panel.drag?.x0 != null)
      finishDrag(element, panel);
  });
  element.addEventListener("pointercancel", () => {
    panel.drag = null;
    redraw();
  });
  element.addEventListener("keydown", (e) => {
    const hit = findHit(e);
    if (!hit || panel.busy)
      return;
    const step = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1}[e.key];
    if (step) {
      panel.drag ??= {id: hit.id, value: hit.d.from, x0: null, perPx: 0};
      panel.drag.value = Math.max(0, panel.drag.value + step * KEY_STEP * (e.shiftKey ? 4 : 1));
    } else if (e.key === "Escape" && panel.drag)
      panel.drag = null;
    else if (e.key === "Enter" && panel.drag)
      return void finishDrag(element, panel);
    else
      return;
    e.preventDefault();
    redraw();
  });
  element.addEventListener("change", (e) => {
    const target = /** @type {HTMLInputElement} */ (e.target);
    if (target.matches("[data-setting]"))
      changeControl(element, panel, target);
  });
  element.addEventListener("click", (e) => {
    const apply = /** @type {HTMLElement} */ (e.target).closest("[data-apply]");
    if (apply && !panel.busy)
      void applyControls(element, panel, apply.dataset.apply === "default" ? "default" : "session");
    const undo = panel.notice?.undo;
    if (undo && /** @type {HTMLElement} */ (e.target).closest("[data-undo]"))
      void change(element, panel, undo.set, null, undo.save);
  });

  /** Gets the divider an event happened on (its element, its memory's id and its Divider), or null. */
  function findHit(/** @type {Event} */ e) {
    const target = /** @type {HTMLElement | null} */ (/** @type {HTMLElement} */ (e.target).closest("[data-divider]"));
    const id = target?.dataset.divider ?? "", d = panel.dividers[id];
    return target && d ? {target, id, d} : null;
  }
}

/** Ends the drag that has the divider at its new size: sends the change unless it is where it started.  The engine
 *  alone refuses a size (findWarning only foretells it).
 * @param {HTMLElement} element
 * @param {Panel} panel */
function finishDrag(element, panel) {
  const {id, value} = /** @type {Drag} */ (panel.drag), d = panel.dividers[id];
  panel.drag = null;
  if (mib(value) === mib(d.from))
    render(element, panel);
  else
    void change(element, panel, {[d.key]: mib(value)}, {[d.key]: mib(d.from)}, "session", explainLimit(d, value));
}

/** Retains a control's edit until the reader applies the form.
 * @param {HTMLElement} element
 * @param {Panel} panel
 * @param {HTMLInputElement} input */
function changeControl(element, panel, input) {
  const key = input.dataset.setting ?? "";
  if (key === "draft_rule")
    panel.pending.draft_rule = input.value;
  else if (key === "drafter") {
    const s = panel.view.settings;
    const {model, suffix, chain} = (DRAFTER_CHOICES.find(([c]) => c === input.value) ?? DRAFTER_CHOICES[0])[1];
    /** @type {Record<string, number | boolean>} */
    const chosen = {drafter: model};
    delete panel.pending.lookup_chain;
    if (s.suffix_draft != null) {
      chosen.suffix_draft = suffix ? s.suffix_draft || 3 : 0;
      if (model)
        chosen.lookup_chain = chain ? s.lookup_chain || 1 : 0;
    }
    for (const [k, v] of Object.entries(chosen))
      if (Number(v) === Number(s[k]))
        delete panel.pending[k];
      else
        panel.pending[k] = v;
  } else if (input.type === "checkbox")
    panel.pending[key] = input.checked;
  else
    panel.pending[key] = input.value !== "" ? Number(input.value) : BLANK_ZERO.includes(key) ? 0 : NaN;
}

/** Validates and submits the staged form as one settings change. */
async function applyControls(element, panel, save) {
  const s = {...panel.view.settings, ...panel.pending};
  if (s.vram_conversations != null && s.vram_conversations < s.concurrent) {
    panel.notice = {text: "Conversations held in VRAM cannot be fewer than conversations at once.",
      error: true, until: Infinity};
    return render(element, panel);
  }
  for (const [key, smallest, largest] of getEditableControls(s)) {
    const n = s[key], lo = smallest(s), hi = largest(s);
    if (!Number.isInteger(n) || n < lo || n > hi) {
      panel.notice = {text: `${SETTINGS[key]}: choose a whole number ` +
        (hi === Infinity ? `of at least ${fmt(lo)}.` : `from ${fmt(lo)} to ${fmt(hi)}.`) +
        (key === "context" && n > hi ? ` ${CONTEXT_CAPACITY_HELP}` : ""),
        error: true, until: Infinity};
      return render(element, panel);
    }
  }
  if (s.context_capacity != null && s.context > s.tokens_max) {
    panel.notice = {text: "The per-conversation limit cannot exceed the combined GPU token budget.",
      error: true, until: Infinity};
    return render(element, panel);
  }
  if (s.route_prior_pct > 0 && (save === "default" || !panel.view.settings.route_prior_pct) &&
      !confirm(`${ROUTE_PRIOR_WARNING} ${save === "default" ? "Save it as the default?" : "Turn it on?"}`))
    return;
  const editable = new Set(getEditableControls(s).map(([key]) => key));
  const rule = panel.view.details?.draft_rule;
  const set = save === "default" ? Object.fromEntries(Object.entries(s).filter(([key]) =>
    editable.has(key) || key.startsWith("memory_mib.") ||
    ["ssd_pause_mib", "suffix_draft", "lookup_chain", ...SWITCHES, ...Object.keys(WORDS)].includes(key))) :
    {...panel.pending};
  if (save === "default" && rule != null)
    set.draft_rule = s.draft_rule ?? rule;
  if (!Object.keys(set).length || (save === "default" && !panel.view.config))
    return;
  for (const key of SWITCHES)
    if (key in set)
      set[key] = Boolean(set[key]);
  for (const [key, words] of Object.entries(WORDS))
    if (key in set)
      set[key] = words[Number(set[key])];
  await change(element, panel, set, null, save);
}

/** Sends a change through the caller's onChange and shows the answer: the new reading with an Undo for a while, or
 *  the server's reason for changing nothing.
 * @param {HTMLElement} element
 * @param {Panel} panel
 * @param {SettingsChange} set
 * @param {SettingsChange | null} undo  the change back, offered as Undo; null for an Undo itself.
 * @param {"session" | "default"} save
 * @param {string} warning  the panel's size prediction, separate from the engine's answer */
async function change(element, panel, set, undo, save = "session", warning = "") {
  panel.busy = true;
  const predicted = warning ? ` Requested size: ${warning}` : "";
  panel.notice = {text: `Changing…${predicted}`, error: !!warning, until: Infinity};
  render(element, panel);
  const answer = await /** @type {ChangeSettings} */ (panel.onChange)(set, save)
    .catch((/** @type {Error} */ e) => ({ok: false, error: e.message, view: undefined}));
  panel.busy = false;
  if (answer.ok) {
    if (answer.view)
      panel.view = answer.view;
    const full = set.shared_memory ? undefined :
      answer.view?.memories.find((m) => m.dedicated_full &&
        Number(set[`memory_mib.${m.id}`]) * MIB > (m.setting ?? 0));
    if (full && isSharedMemoryWanted(full, Number(set[`memory_mib.${full.id}`]) * MIB))
      return change(element, panel, {...set, shared_memory: true}, undo, save);
    for (const key of Object.keys(set))
      delete panel.pending[key];
    try {
      localStorage.setItem(CHANGED_KEY, "1");
    } catch { /* no storage (a private window): the hint stays */ }
    const done = save === "default" ? "Changed, and saved as the default." : "Changed for this session.";
    const shortfall = answer.view?.shortfall;
    panel.notice = {text: shortfall ? `${done} But it got less than asked: ${shortfall}.` : done, error: !!shortfall,
                    until: shortfall ? Infinity : Date.now() + UNDO_MS, ...(undo ? {undo: {set: undo, save}} : {})};
  } else {
    const said = answer.error || "The engine changed nothing.";
    panel.notice = {text: predicted ? `${said.replace(/\.?$/, ".")}${predicted}` : said, error: true,
                    until: Infinity};
  }
  render(element, panel);
}

/** Asks the reader whether GPU memory `x`, whose pool stopped growing where its dedicated memory ended, may take the
 *  rest of the `asked` bytes (its pool's) from shared memory.
 * @param {MemoryUse} x
 * @param {number} asked */
function isSharedMemoryWanted(x, asked) {
  const fixed = findDivider(x)?.fixed ?? 0;
  const dedicated = fixed + (x.setting ?? 0) - (x.pool_shared ?? 0);
  return confirm(`${(dedicated / 1073741824).toFixed(1)} of ${((fixed + asked) / 1073741824).toFixed(1)} GiB is ` +
    "available as dedicated memory. If you wish, you can allocate the remainder as shared memory, which may be slower");
}

/** Gets whether the reader has changed a size before (in this browser), which retires the hint. */
function hasChanged() {
  try {
    return !!localStorage.getItem(CHANGED_KEY);
  } catch {
    return false;
  }
}

/** Gets the staged settings form and its Apply buttons.
 * @param {Panel} panel */
function drawControls(panel) {
  const s = {...panel.view.settings, ...panel.pending}, off = panel.busy ? " disabled" : "";
  const fields = getEditableControls(s).map(([key, smallest, largest]) => {
    const help = esc(getControlHelp(key, s, panel.view.details));
    const value = BLANK_ZERO.includes(key) && s[key] === 0 ? "" : s[key];
    return `<label data-control="${key}" title="${help}">${esc(SETTINGS[key])} ` +
      `<input class="st-input mem-input" type="number" step="1" data-setting="${key}" ` +
      `value="${value}" min="${smallest(s)}"${largest(s) < Infinity ? ` max="${largest(s)}"` : ""}` +
      ` title="${help}"${off}></label>`;
  });
  if (s.drafter != null)
    fields.push(drawDrafterControl(s, panel.view.details, off));
  for (const [key, words] of Object.entries(WORDS))
    if (s[key] != null) {
      const help = esc(getControlHelp(key, s));
      fields.push(`<label data-control="${key}" title="${help}">${esc(SETTINGS[key])} ` +
        `<select class="st-input mem-input" data-setting="${key}"${off}>` + words.map((_, i) =>
          `<option value="${i}"${Number(s[key]) === i ? " selected" : ""}>${esc(WORD_LABELS[key][i])}</option>`)
          .join("") + "</select></label>");
    }
  const rule = /** @type {string | undefined} */ (panel.pending.draft_rule) ?? panel.view.details?.draft_rule;
  if (rule != null)
    fields.push(drawDraftRuleControl(rule, off));
  for (const key of SWITCHES.filter((k) => k !== "drafter" && s[k] != null)) {
    const help = esc(getControlHelp(key, s));
    fields.push(`<label class="mem-switch" data-control="${key}" title="${help}"><input type="checkbox" ` +
      `data-setting="${key}"${s[key] ? " checked" : ""}${off}> ${esc(SETTINGS[key])}</label>`);
  }
  return `<div class="mem-controls">${fields.join("")}</div><div class="mem-actions">` +
    `<button type="button" data-apply="session"${off}>Apply settings</button> ` +
    `<button type="button" data-apply="default"${off}${panel.view.config ? "" : " disabled"}>` +
    `Apply and save as default</button></div>`;
}

/** Gets the engine's supported prediction modes for the Memory settings form.
 * @param {Record<string, number | boolean>} settings
 * @param {Record<string, string> | undefined} details
 * @param {string} disabled */
function drawDrafterControl(settings, details = {}, disabled = "") {
  const now = {model: !!settings.drafter, suffix: !!settings.suffix_draft,
    chain: !!settings.drafter && !!settings.lookup_chain};
  const help = getControlHelp("drafter", settings) + " " +
    (details.drafter_details || "Drafter discovery details are not reported by this engine.") +
    (settings.suffix_draft == null ? "" :
      " Suffix drafting and lookup use greedy sampling only. \"Or suffix drafter\": a step takes the suffix " +
      "drafter's guesses instead of MTP's where they start alike and have paid off so far. \"+ lookup\": a step " +
      "may append guesses found in the conversation's text after MTP's. Mode changes take effect on the next request.");
  const model = details.drafter_label || (settings.has_drafter === 0 ? "No model drafter loaded" :
    "Use model drafter (VRAM cost not reported)");
  const own = {off: "No MTP", suffix: details.suffix_label || "Use suffix drafter (0 GiB)"};
  const options = DRAFTER_CHOICES.filter(([, use]) => settings.suffix_draft != null || !(use.suffix || use.chain))
    .map(([choice, use, words]) => `<option value="${choice}"` +
      `${use.model === now.model && use.suffix === now.suffix && use.chain === now.chain ? " selected" : ""}` +
      `${use.model && settings.has_drafter === 0 ? " disabled" : ""}>` +
      `${esc(words == null ? own[choice] : model + words)}</option>`);
  return `<label data-control="drafter" title="${esc(help)}">Drafter ` +
    `<select class="st-input mem-input" data-setting="drafter"${disabled}>${options.join("")}</select></label>`;
}

/** Gets the choice of which guesses a sampled answer keeps, with `rule` (the engine's draft_rule text) selected; a
 * rule the panel does not offer (set by --draft-rule, so a top-N one) is listed under its own name.
 * @param {string} rule
 * @param {string} disabled */
function drawDraftRuleControl(rule, disabled = "") {
  const rules = DRAFT_RULES.some(([r]) => r === rule) ? DRAFT_RULES : [...DRAFT_RULES, [rule, `${rule} *`]];
  return `<label data-control="draft_rule" title="${esc(DRAFT_RULE_HELP)}">Draft acceptance ` +
    `<select class="st-input mem-input" data-setting="draft_rule"${disabled}>` +
    rules.map(([r, label]) => `<option value="${esc(r)}"${r === rule ? " selected" : ""}>${esc(label)}</option>`)
      .join("") + "</select></label>";
}

/** Gets controls supported by this engine; context_capacity advertises a live per-conversation limit.
 * @param {Record<string, number>} settings */
function getEditableControls(settings) {
  return CONTROLS.filter(([key]) => settings[key] != null && (key !== "context" || settings.context_capacity != null));
}

/** Gets a control's tooltip: the engine flags it sets, then what it means.
 * @param {string} key
 * @param {Record<string, number | boolean>} settings
 * @param {Record<string, string>} details */
function getControlHelp(key, settings, details = {}) {
  return `${FLAGS[key] ?? `--${key.replaceAll("_", "-")}`}: ${describeControl(key, settings, details)}`;
}

/** Gets the meaning and memory cost of controls with limits that are easy to confuse.
 * @param {string} key
 * @param {Record<string, number | boolean>} settings
 * @param {Record<string, string>} details */
function describeControl(key, settings, details) {
  if (key === "rows")
    return "Maximum number of tokens to compute during each pass through the model. More rows can increase " +
      "throughput but also increases demand for experts, so the more VRAM and RAM you have, " +
      "the more rows you can afford.";
  if (key === "lookahead")
    return 'Maximum number of tokens to predict for each conversation in the drafter. The drafter is also limited ' +
      'by max rows; for example if "Max lookahead" is 3 and "Max rows" is 6, then 3 lookahead tokens are predicted ' +
      'when one conversation is running, but only 2 are predicted when two conversations are running ' +
      "(that's 2 rows for the two conversations plus 2×2 rows for lookahead).";
  if (key === "tokens_min")
    return "Minimum VRAM to allocate for KV cache (shared by all conversations), expressed in tokens.";
  if (key === "tokens_max")
    return "This dictates the shared GPU token budget. The per-conversation limit also bounds each context window. " +
      (settings.token_bytes ? `Every token requires approximately ${fmt(settings.token_bytes)} bytes of memory, ` +
        `so ${fmt(settings.tokens_max)} tokens requires ` +
        `${gb(settings.tokens_max * settings.token_bytes, 2)} GiB. ` : "") +
      `KV cache quantization: ${details.kv_quantization || "not reported by this engine"}. ` +
      "Multiple conversations can share this space. " +
      "If the largest two conversations don't fit, only one conversation can run at a time.";
  if (key === "concurrent")
    return "Maximum conversations decoding in one step. Each active conversation uses at least one row. " +
      (settings.slots != null ? `The engine supports up to ${fmt(settings.slots)} conversations at once. ` : "") +
      "This setting is also limited by Rows per step and Max conversations in VRAM.";
  if (key === "drafter")
    return "This option controls the multi-token prediction (MTP) mode that can speed up generation using rows " +
      "that aren't in use by active conversations.";
  if (key === "context")
    return `Loaded capacity: ${fmt(settings.context_capacity)} tokens. ${CONTEXT_CAPACITY_HELP}`;
  if (key === "conversation_idle_seconds")
    return "Idle GPU conversations this old may move to RAM or SSD when experts or another conversation need space. " +
      "Prompt-processing and decoding requests are protected; age alone does not evict anything.";
  if (key === "step_profile")
    return "Writes a line to the model's log after each decode step, with host elapsed times for drafting guesses, " +
      "running the model, keeping the accepted tokens and moving experts. These are not GPU kernel timings. " +
      "It adds no waits for the GPU, but a log line at every step (many a second) can slow Strata and " +
      "makes the log grow quickly.";
  if (key === "save_on_shutdown")
    return "When the model stops, conversations in VRAM and RAM are written to the SSD, the most recent first, " +
      `up to the SSD allowance (${fmt(settings.ssd_pause_mib ?? 0)} MiB), and the next start picks them up. ` +
      "Stopping takes a few seconds longer.";
  if (key === "decode_share_percent")
    return "While a prompt is read, the conversations that are answering take turns with it: after each turn of " +
      "reading they generate for this percentage of the time the turn took. 0 makes them wait for the whole prompt; " +
      "more lets them answer faster and the prompt read slower. On a graphics card of 12 GB or less, try 0: there " +
      "the prompt and the answers each need most experts read in again, so taking turns slows both.";
  if (key === "prompt_turn_ms")
    return "The least time a prompt is read before the answering conversations get their turn. 0 means one chunk " +
      "of the prompt. Longer turns switch less often between reading and answering.";
  if (key === "prefill_chunk")
    return "The most tokens of a prompt read in one pass through the model. Each pass reads every expert the GPU " +
      "lacks, so larger chunks read a long prompt with fewer expert reads, but borrow more of the pools' memory " +
      "(experts may leave the GPU while the prompt is read). Strata takes the largest chunk up to this that the " +
      "pools can lend at the time. A change applies from the next prompt. The prompt reader is sized for up to " +
      `${fmt(settings.prefill_chunk_capacity)} tokens now; a larger value makes Strata enlarge it, which takes a ` +
      "little memory outside the pools. If that fails, the size stays and Strata says why.";
  if (key === "batch_groups")
    return "With the model split over several GPUs, the conversations decoding together can split into groups, so " +
      "that one GPU works on one group while the next GPU works on another. 1 turns this off; 0 picks the most " +
      "groups that divide the conversations at once evenly, up to one per GPU. A number that does not divide them " +
      "runs one group. " + (settings.pipeline_stages === 0 ?
      "This model runs one group when a drafter is loaded, even if it is switched off. " : "") +
      `Groups now: ${fmt(settings.batch_groups_now ?? 1)}.`;
  if (key === "vram_reserve_mib")
    return "Graphics memory, in MiB, that Strata leaves free on each GPU for other programs and for its own buffers " +
      "outside the pools. Qwen3.8-Flash-Next's on-demand GPU picture reader needs about 1,536 MiB of this space " +
      "on its GPU (1,342 MiB measured on an RTX A6000 and an R9700), so keep 2,048 MiB or more for it. " +
      "With too little free room, a configured CPU reader takes over, more slowly. " +
      "This setting applies to every GPU. A change applies when a pool next grows, and at the next start when it " +
      "is saved as the default; it does not resize a pool by itself.";
  if (key === "route_tail_skip")
    return "In each layer the model picks several experts for each token, ranked by weight. With this set to N, " +
      "the top N are always used. A lower-ranked expert that is not in the GPU is left out (neither read nor " +
      "computed) when it is outside the top N for every token of the step; the others keep their original weights. " +
      "This speeds up decoding when experts are missing from the GPU, and changes the answers slightly. Smaller " +
      "numbers leave out more. If you use it with Qwen3.8-Flash-Next, we recommend at least 7. At 7, the most " +
      "likely next token matched a run without skipping 97.5-98.2% of the time (top-1 agreement; perplexity " +
      "x1.005-1.011). Measured with engine 0.1.42 on 28 chat prompts: RTX 3060 12 GB, IQ3_XXS, and RTX A4000 " +
      "16 GB, Q2_0. It does nothing while every expert is in the GPU. " +
      "Empty or 0 uses all the experts the model picks.";
  if (key === "route_prior_pct")
    return ROUTE_PRIOR_WARNING + " When the experts are picked, one already on the GPU counts as this many " +
      "percent more likely than the model says; the picked experts keep the model's weights. 348 is the value " +
      "that was tested; 0 turns it off. The current value corresponds to STRATA_ROUTE_PRIOR=" +
      `${Math.log1p(Number(settings.route_prior_pct) / 100).toFixed(2)}.`;
  if (key === "draft_confidence_pct" || key === "draft_cumulative_pct")
    return (key === "draft_confidence_pct" ?
      "The drafter's guesses stop before the first one it is less sure of than this. " :
      "The drafter's guesses stop before the one that brings its confidence in all of them together (their " +
      "chances multiplied) below this. ") + "That guess is dropped, not checked: the model works that token out " +
      "itself unless suffix drafting or lookup supplies another guess. 0 disables this confidence limit. " +
      "Higher values check fewer guesses, which wastes less work when guesses are often wrong; lower values " +
      "check more. It applies from the next request; a request can send its own.";
  if (key === "expert_half_life")
    return "How quickly the GPU forgets which experts were used: after this many decode steps a use counts half. " +
      "Shorter makes the GPU's experts follow a change of topic sooner, but swaps more of them. " +
      "0 never forgets.";
  if (key === "copy_back")
    return "What becomes of an expert that the GPU's memory gives up to make room. Copied back to RAM, it comes " +
      "back quickly the next time; dropped, it is read from the SSD again. The automatic choice copies it back " +
      "when Strata measured that faster than reading the SSD.";
  if (key === "vram_conversations")
    return "Limits active conversations and idle GPU caches together. " +
      (settings.conversation_bytes ? `Each takes ${size(settings.conversation_bytes)} of fixed state ` +
        "before its KV cache. Rewind checkpoints are allocated separately in RAM as needed. " : "") +
      "Extra conversations wait, or are paused to RAM or SSD; higher-priority requests run first.";
  return "";
}

/** Lays a memory's bar out: its segments (`drawn`: their bytes as drawn, when scaled) and the drawn bytes the bar's
 *  width stands for.  `scale`: drawn bytes per byte the engine named.
 * @param {MemoryUse} x */
function layOut(x) {
  const named = Object.values(x.uses).reduce((a, b) => a + b, 0);
  const scale = x.in_share == null || !named ? 1 : Math.min(1, x.in_share / named);
  const parts = splitPoolParts(x, scale);
  // Keep the adjustable pool contiguous, with its unused pages nearest the free memory.
  parts.sort((a, b) => getOrder(a) - getOrder(b) || a.use.localeCompare(b.use));
  if (x.gpu_borrowed)
    parts.push({use: "borrowed", bytes: x.gpu_borrowed});
  if (x.free != null) {
    // (the reserve at the start of the free memory, the dedicated memory found unusable at its end)
    const room = Math.min(x.dedicated_room ?? x.free, x.free), kept = Math.min(x.kept_free ?? 0, room);
    parts.push(...[{use: "kept_free", bytes: kept}, {use: "free", bytes: room - kept},
                   {use: "beyond_dedicated", bytes: x.free - room}].filter((p) => p.use === "free" || p.bytes > 0));
  }
  const outside = x.other_apps ?? x.other;
  if (outside)
    parts.push({use: "other_apps", bytes: outside});
  // a memory of unknown size (the SSD) is drawn against what it holds plus the room left in its paused budget
  const used = parts.reduce((a, p) => a + (p.drawn ?? p.bytes), 0);
  const budgeted = SSD_BUDGET_USES.reduce((a, use) => a + (x.uses[use] || 0), 0);
  const total = x.total || Math.max(used, used - budgeted + (x.paused_budget || 0));
  return {parts, scale, total};

  /** Ranks fixed allocations before the pool and its unused pages. */
  function getOrder(/** @type {{use: string, pool?: boolean}} */ part) {
    return part.pool ? part.use === "pool_free" ? 2 : 1 : 0;
  }
}

/** Gets the divider of a memory's bar: the end of the engine's page pool on a GPU or in RAM (`memory_mib.<id>`), the
 *  end of the room for paused conversations on the SSD (`ssd_pause_mib`); null for a memory without one.
 * @param {MemoryUse} x
 * @returns {Divider | null} */
function findDivider(x) {
  const {parts, scale, total} = layOut(x), ssd = x.id === "ssd";
  const from = ssd ? x.paused_budget : x.setting;
  const first = parts.findIndex((p) => ssd ? SSD_BUDGET_USES.includes(p.use) : p.pool);
  if (from == null || !total || (first < 0 && !ssd && x.pool_offset == null))
    return null;
  const start = !ssd && first < 0 && x.pool_offset != null ? x.pool_offset * scale :
    (first < 0 ? parts : parts.slice(0, first)).reduce((a, p) => a + (p.drawn ?? p.bytes), 0);
  const fixed = ssd ? 0 : Math.max(0, Object.values(x.uses).reduce((a, b) => a + b, 0) - from);
  const commit = x.id === "ram" && x.commit_room != null && x.free != null && x.commit_room < x.free ?
    x.commit_room : null;
  return {key: ssd ? "ssd_pause_mib" : `memory_mib.${x.id}`, from, smallest: x.smallest ?? 0,
          largest: x.largest ?? Infinity, start, scale, total, fixed, shared: x.pool_shared ?? 0, commit};
}

/** Gets why the engine may not give a divider's size: it refuses less than it needs and more than a RAM pool can
 *  take; a GPU pool past its dedicated memory gets shared memory only when the reader agrees.  "" when it may.
 * @param {Divider} d
 * @param {number} value  bytes */
function findWarning(d, value) {
  return value < d.smallest ? `at least ${size(d.fixed + d.smallest)} is needed here now (fixed allocations, ` +
    `the biggest active conversation and the minimum conversation tokens)` : value <= d.largest ? "" :
    d.key.startsWith("memory_mib.gpu") ? `about ${size(d.fixed + d.largest)} fits in dedicated memory ` +
    `(${size(d.fixed)} fixed); the rest would be shared memory, which may be slower` :
    `at most ${size(d.fixed + d.largest)} in all is available to Strata here: ${size(d.fixed)} fixed and up to ` +
    `${size(d.largest)} of resizable pool`;
}

/** Explains a predicted size limit for the Free tooltip and release notice, including Windows' commit room.
 *  The engine may still accept the request if conditions have changed.
 * @param {Divider} d
 * @param {number} value  bytes */
function explainLimit(d, value) {
  const warning = findWarning(d, value);
  return warning && `${warning[0].toUpperCase()}${warning.slice(1)}.` + (d.commit == null || value <= d.largest ?
    "" : ` Windows can promise programs only ${size(d.commit)} more (commit room), even with free RAM. Programs ` +
    "can claim memory before using it; Windows backs those claims with RAM and its page file on disk. Strata " +
    "leaves up to 1 GiB of this room free. To get more, close programs or set a larger page file. A page file " +
    "that Windows manages grows by itself, but only while its drive has free space, so freeing disk space " +
    "sometimes helps too.");
}

/** Gets the markup of one memory's row: its name and size, its bar (with its divider, moved by `drag`) and the
 *  bar's legend.
 * @param {MemoryUse} x
 * @param {Divider | undefined} divider
 * @param {Drag | null} drag
 * @param {MemoryView} view */
function drawBar(x, divider, drag, view) {
  const {parts, total} = layOut(x);
  const limit = divider ?? findDivider(x);
  /** @type {Map<string, {use: string, bytes: number}>} */
  const legend = new Map();
  for (const part of parts)
    legend.set(part.use, {use: part.use, bytes: (legend.get(part.use)?.bytes || 0) + part.bytes});
  const name = x.id === "ram" ? "System RAM" : x.id === "ssd" ? "SSD" : x.name || x.id.toUpperCase();
  const label = "borrowed" in x ? `${name} — BIOS share ${gb(x.total)} GiB` :
    x.gpu_borrowed ? `${name} ${gb(x.total)} GiB (GPU borrows ${gb(x.gpu_borrowed)} GiB)` : name;
  const facts = [x.total && label === name ? `${gb(x.total)} GiB` : "", x.paused_count == null ? "" :
    `${fmt(x.paused_count)} paused` + (x.paused_budget ? ` (budget ${gb(x.paused_budget, 2)} GiB)` : ""),
    x.allocated_pct == null ? "" :
    `pool ${fmt(x.allocated_pct, 1)}% allocated, ${fmt(x.fragmented_pct, 2)}% fragmented`]
    .filter(Boolean).join(" · ");
  const title = (/** @type {{use: string}} */ p) => esc(x.id === "ssd" && p.use === "experts" ?
    "Experts not cached" : Object.hasOwn(USES, p.use) ? USES[p.use] : p.use);
  const blocks = [...(x.blocks || [])].sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}));
  const pauseFolder = view.pause_directory ? `SSD pause folder: ${view.pause_directory}` : "";
  const paused = (view.conversations || []).filter((c) => c.state === "parked" &&
    c.place === (x.id === "ssd" ? "disk" : x.id))
    .sort((a, b) => a.id.localeCompare(b.id, undefined, {numeric: true}));
  const tip = (/** @type {{use: string, bytes: number}} */ p) =>
    `${title(p)}: ${size(p.bytes)}${x.id === "ssd" && p.use === "experts" ?
      ". Experts held in neither VRAM nor RAM. The complete model files remain on disk." :
      Object.hasOwn(USE_HELP, p.use) ? `. ${esc(USE_HELP[p.use])}` : ""}` +
    (p.use === "free" && limit?.commit != null ? `&#10;${esc(explainLimit(limit, Infinity))}` : "") +
    (p.use === "paused" ? [pauseFolder,
      ...paused.map((c) => `${c.id}: last tokens ${getActivityAge(c.last_tokens_at)}`)]
      .filter(Boolean).map((line) => `&#10;${esc(line)}`).join("") : "") +
    blocks.filter((b) => POOL_USES.includes(p.use) ? POOL_USES.includes(b.category) : b.category === p.use)
      .map((b) => `&#10;${POOL_USES.includes(b.category) || b.pool === "Expert and conversation pool" ?
        "" : `${esc(b.pool)} / `}` +
        `${esc(b.label)}: ${size(b.bytes)}`).join("");
  const seg = (/** @type {{use: string, bytes: number, drawn?: number}} */ p) =>
    `<span class="mem-seg" data-use="${esc(p.use)}" title="${tip(p)}" ` +
    `style="width:${total ? (100 * (p.drawn ?? p.bytes)) / total : 0}%"></span>`;
  const key = (/** @type {{use: string, bytes: number}} */ p) =>
    `<span title="${tip(p)}"><i class="mem-key" data-use="${esc(p.use)}"></i>${title(p)} ${size(p.bytes)}</span>`;
  return `<div class="mem-row" data-memory="${esc(x.id)}"><div class="mem-head" ` +
    `title="${x.id === "ssd" ? esc(pauseFolder) : ""}">` +
    `<span>${esc(label)}</span><span class="mem-facts">${esc(facts)}` +
    `</span></div><div class="mem-bar" style="width:${total ? 100 * Math.min(total / FULL_BAR, 1) :
      x.id === "ssd" ? 0 : 100}%">` +
    `<div class="mem-segs">${parts.map(seg).join("")}</div>${divider ? drawDivider(x.id, divider, drag) : ""}</div>` +
    `<div class="mem-legend">${[...legend.values()].map(key).join("")}</div>` +
    `${divider && drag ? drawPreview(divider, drag) : ""}</div>`;
}

/** Gets the markup of a bar's divider, where `drag` has moved it (with the stretch it moved over shaded).
 * @param {string} id  the memory's
 * @param {Divider} d
 * @param {Drag | null} drag */
function drawDivider(id, d, drag) {
  const value = drag ? drag.value : d.from, warned = drag && findWarning(d, value) ? " mem-warned" : "";
  const at = (/** @type {number} */ b) => Math.min(100, Math.max(0, (100 * (d.start + b * d.scale)) / d.total));
  const what = id === "ssd" ? "the SSD space paused conversations may take" :
    `Strata's total memory ${id === "ram" ? "in RAM" : "on this GPU"}`;
  const allocation = `${size(d.fixed + value)}${id === "ssd" ? "" :
    ` total (${size(d.fixed)} fixed, ${size(value)} resizable pool)`}`;
  const shared = d.shared ? ` The pool has ${size(d.shared)} in shared memory, which is slower; prompt buffers ` +
    "are lent below the lowest shared chunk." : "";
  const moved = drag && value !== d.from ? `<span class="mem-change${warned}" style="left:` +
    `${Math.min(at(d.from), at(value))}%;width:${Math.abs(at(value) - at(d.from))}%"></span>` : "";
  return `${moved}<span class="mem-divider${warned}" data-divider="${esc(id)}" tabindex="0" role="slider" ` +
    `aria-label="${esc(what)}" aria-valuetext="${allocation}" title="Drag to change ${esc(what)}. ` +
    `${allocation}.${esc(shared)} Arrow keys move it too; Enter applies, Escape cancels." ` +
    `style="left:${at(value)}%"></span>`;
}

/** Gets the markup of the line under a bar whose divider is being moved: the size before and after, and in red why
 *  the engine may not give it (findWarning).
 * @param {Divider} d
 * @param {Drag} drag */
function drawPreview(d, drag) {
  const warning = findWarning(d, drag.value);
  return `<p class="mem-preview${warning ? " mem-warned" : ""}">` +
    `${size(d.fixed + d.from)} → ${size(d.fixed + drag.value)}` +
    (d.key === "ssd_pause_mib" ? "" : ` Strata total (${size(d.fixed)} fixed; ${size(drag.value)} resizable pool)`) +
    (warning ? `: ${esc(warning)}; ` : ": ") + "release to apply, Escape cancels</p>";
}

/** Gets the API monitor's selectable contexts, including engine-reported activity ages when available.
 * @param {HeldConversation[]} list
 * @param {string} [pauseDirectory]
 * @param {number} [gpuBudget] the allowance shared by all GPU rows, independent of each request's context limit */
export function drawConversations(list, pauseDirectory, gpuBudget) {
  list = list.filter((c) => c.count > 0 || c.state === "decoding" || c.state === "reading");
  if (!list.length)
    return "";
  const activity = list.some((c) => c.last_tokens_at || c.state === "parked");
  const gpu = list.filter((c) => c.place === "vram");
  const total = gpu.reduce((n, c) => n + c.count, 0);
  const footer = gpuBudget == null ? "" : `<tfoot><tr><th colspan="2">GPU total (shared budget)</th>` +
    `<td class="mem-num${total > gpuBudget ? " mem-over-budget" : ""}">${fmt(total)} / ${fmt(gpuBudget)}</td>` +
    `<td class="mem-num">${size(gpu.reduce((n, c) => n + c.bytes, 0))}</td>` +
    (activity ? "<td></td>" : "") + "</tr></tfoot>";
  const row = (/** @type {HeldConversation} */ c, /** @type {number} */ i) =>
    `<tr><td${c.state === "parked" && pauseDirectory ?
    ` title="${esc(`SSD pause folder: ${pauseDirectory}`)}"` : ""}>` +
    `<button type="button" data-context="${esc(c.id)}">${esc(c.name || `Conversation ${i + 1}`)}</button>` +
    (typeof c.preview === "string" && c.preview ?
      `<span class="mem-context-preview">${esc(c.preview)}</span>` : "") + "</td>" +
    `<td>${esc(c.state)} (${esc(PLACES[c.place] || c.place)})</td>` +
    `<td class="mem-num"${c.place === "vram" && gpuBudget != null ?
      ` title="Shared GPU budget: ${fmt(gpuBudget)} tokens; ` +
        `per-conversation context limit: ${fmt(c.capacity)} tokens"` :
      ""}>${fmt(c.count)} / ${fmt(c.capacity)}</td>` +
    `<td class="mem-num">${size(c.bytes)}</td>` +
    (activity ? `<td class="mem-num" title="${c.last_tokens_at ?
      esc(new Date(c.last_tokens_at * 1000).toLocaleString()) : "Unknown"}">` +
      `${esc(getActivityAge(c.last_tokens_at))}</td>` : "") + "</tr>";
  return `<table class="mem-conversations"><thead><tr><th>Conversation</th><th>State</th>` +
    `<th class="mem-num">Tokens</th><th class="mem-num">Memory</th>` +
    (activity ? `<th class="mem-num">Last tokens</th>` : "") + "</tr></thead>" +
    `<tbody>${list.map(row).join("")}</tbody>${footer}</table>`;
}

/** Gets time since the engine last processed tokens; missing timestamps remain unknown.
 * @param {number | undefined} timestamp */
function getActivityAge(timestamp) {
  if (!timestamp)
    return "Unknown";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  const [unit, divisor] = seconds >= 86400 ? ["d", 86400] : seconds >= 3600 ? ["h", 3600] :
    seconds >= 60 ? ["min", 60] : ["s", 1];
  return `${Math.floor(seconds / divisor)} ${unit} ago`;
}

/** Splits RAM expert totals by logical residency without changing the server's reading.
 * @param {MemoryUse[]} memories
 * @returns {MemoryUse[]} */
function markRedundantExperts(memories) {
  /** @type {Map<string, Set<number>>} */
  const resident = new Map();
  for (const memory of memories.filter((m) => m.id.startsWith("gpu")))
    for (const block of memory.blocks || []) {
      if (!block.identity_group || !block.ids?.length)
        continue;
      const ids = resident.get(block.identity_group) || new Set();
      for (const id of block.ids)
        ids.add(id);
      resident.set(block.identity_group, ids);
    }
  return memories.map((memory) => {
    if (memory.id !== "ram")
      return memory;
    let redundant = 0;
    const blocks = (memory.blocks || []).flatMap((block) => {
      const ids = block.identity_group && resident.get(block.identity_group);
      if (block.category !== "experts" || !block.ids?.length || !ids)
        return [block];
      const copies = block.ids.filter((id) => ids.has(id));
      const bytes = block.bytes * copies.length / block.ids.length;
      redundant += bytes;
      return [{...block, bytes: block.bytes - bytes},
        {...block, category: "redundant_experts", label: `${block.label} (also in VRAM)`, bytes}]
        .filter((b) => b.bytes > 0);
    });
    if (!redundant)
      return memory;
    const uses = Object.fromEntries(Object.entries(memory.uses).flatMap(([use, bytes]) => use === "experts" ?
      [[use, bytes - redundant], ["redundant_experts", redundant]] : [[use, bytes]]));
    return {...memory, blocks, uses};
  });
}

/** Preserves the engine's adjustable byte interval when categories are reordered for display.
 * @param {MemoryUse} memory
 * @param {number} scale
 * @returns {{use: string, bytes: number, drawn?: number, pool?: boolean}[]} */
function splitPoolParts(memory, scale) {
  const start = memory.pool_offset, end = (start ?? 0) + (memory.setting ?? 0);
  let at = 0;
  return Object.entries(memory.uses).flatMap(([use, bytes]) => {
    const before = at;
    at += bytes;
    if (bytes <= 0)
      return [];
    if (memory.id === "ssd" || start == null)
      return [{use, bytes, drawn: bytes * scale, pool: POOL_USES.includes(use)}];
    const boundaries = [before, at, start, end].filter((value) => value >= before && value <= at)
      .sort((a, b) => a - b);
    return boundaries.slice(1).flatMap((boundary, i) => {
      const from = boundaries[i], size = boundary - from;
      return size ? [{use, bytes: size, drawn: size * scale, pool: from >= start && boundary <= end}] : [];
    });
  });
}

/** Updates the panel's markup while retaining matching rows, controls and slider elements.
 * @param {Node} parent
 * @param {Node} reading */
function updateChildren(parent, reading) {
  let cursor = parent.firstChild;
  for (const next of Array.from(reading.childNodes)) {
    let node = cursor;
    while (node && getNodeKey(node) !== getNodeKey(next))
      node = node.nextSibling;
    if (!node) {
      node = next.cloneNode(true);
      parent.insertBefore(node, cursor);
    } else {
      if (node !== cursor)
        parent.insertBefore(node, cursor);
      if (node.nodeType === 1) {
        const current = /** @type {HTMLElement} */ (node), desired = /** @type {HTMLElement} */ (next);
        for (const attribute of Array.from(current.attributes))
          if (!desired.hasAttribute(attribute.name))
            current.removeAttribute(attribute.name);
        for (const attribute of Array.from(desired.attributes))
          if (current.getAttribute(attribute.name) !== attribute.value)
            current.setAttribute(attribute.name, attribute.value);
        updateChildren(current, desired);
        if (/^(INPUT|SELECT)$/.test(current.tagName) && current !== current.ownerDocument.activeElement) {
          const input = /** @type {HTMLInputElement | HTMLSelectElement} */ (current);
          const value = /** @type {HTMLInputElement | HTMLSelectElement} */ (desired).value;
          if (input.value !== value)
            input.value = value;
        }
      } else if (node.nodeValue !== next.nodeValue)
        node.nodeValue = next.nodeValue;
    }
    cursor = node.nextSibling;
  }
  while (cursor) {
    const next = cursor.nextSibling;
    parent.removeChild(cursor);
    cursor = next;
  }
}

/** Gets a sibling's stable identity within the Memory panel's markup.
 * @param {Node} node */
function getNodeKey(node) {
  const element = node.nodeType === 1 ? /** @type {HTMLElement} */ (node) : null;
  const key = element && ["data-memory", "data-control", "data-divider", "data-setting", "data-apply", "data-use"]
    .map((name) => element.getAttribute(name)).find((value) => value !== null);
  return `${node.nodeName}:${key ?? element?.classList[0] ?? ""}`;
}

/** Escapes text for HTML. @param {unknown} s */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ESCAPES[c] ?? c);
}

/** Formats a number with `d` decimals in the reader's locale; "–" for none. @param {number | null | undefined} n */
function fmt(n, d = 0) {
  return n == null || Number.isNaN(n) ? "–" :
    Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
}

/** Formats bytes as binary GB (GiB, as Windows shows them) without the unit. @param {number | null} b */
function gb(b, d = 1) {
  return b == null ? "–" : fmt(b / 1073741824, d);
}

/** Converts bytes to whole MiB. @param {number} b */
function mib(b) {
  return Math.round(b / MIB);
}

/** Formats bytes with their unit: MiB below 1 GiB (a conversation, the scratch), else GiB. @param {number} b */
function size(b) {
  return b < 1073741824 ? `${fmt(b / 1048576)} MiB` : `${gb(b, 2)} GiB`;
}
