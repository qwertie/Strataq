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
              borrowed: "GPU (borrowed from RAM)", other: "Other", other_apps: "Other apps", free: "Free"};
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
                  lookahead: "Max lookahead", tokens_min: "Min conversation tokens",
                  tokens_max: "Max conversation tokens", ssd_pause_mib: "SSD for paused conversations (MiB)",
                  drafter: "Drafter", save_on_shutdown: "Save conversations on shutdown",
                  slots: "Maximum conversations at once"};
/** The settings that are switches: the engine reports them as 1 or 0 and takes "on" or "off". */
const SWITCHES = ["drafter", "save_on_shutdown"];
/** Guidance shared by the per-conversation limit's tooltip and capacity error. */
const CONTEXT_CAPACITY_HELP = "To raise this limit, increase --max-context in the model's engine arguments, " +
  "then reload the model.";
/** Settings the engine reports that the panel shows elsewhere (the bars) or not at all. */
const UNLISTED = new RegExp("^memory_mib\\.|^(token_bytes|conversation_bytes|context_capacity|" +
  "has_drafter|suffix_draft|lookup_chain|slots|ssd_pause_mib)$");
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
                  ["rows", (s) => s.concurrent ?? 1, () => 8],
                  ["lookahead", () => 0, (s) => (s.rows ?? 1) - 1],
                  ["tokens_min", () => 0, (s) => s.tokens_max ?? Infinity],
                  ["tokens_max", (s) => s.tokens_min ?? 0, () => Infinity],
                  ["context", () => 1, (s) => s.context_capacity ?? s.context ?? Infinity]];
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
 * the sizes it accepts now.  Bytes.
 * @typedef {{id: string, name: string | null, total: number | null, free: number | null, unified: boolean,
 *   uses: Record<string, number>, paused_count: number | null, paused_budget: number | null,
 *   other?: number, other_apps?: number,
 *   blocks?: MemoryBlock[],
 *   in_share?: number, borrowed?: number | null, gpu_borrowed?: number, allocated_pct?: number | null,
 *   fragmented_pct?: number | null, setting?: number | null, pool_offset?: number | null, smallest?: number | null,
 *   largest?: number | null}} MemoryUse
 * @typedef {{category: string, pool: string, label: string, bytes: number,
 *   identity_group?: string, ids?: number[]}} MemoryBlock
 * @typedef {{id: string, place: string, state: string, count: number, capacity: number, bytes: number,
 *   used: number, last_tokens_at?: number, name?: string, preview?: string}} HeldConversation
 * @typedef {Record<string, number | boolean>} SettingsChange  POST /v1/memory's "set": MiB for the sizes.
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
 * bar's width stands for; `fixed`: Strata allocations outside the adjustable pool. Bytes.
 * @typedef {{key: string, from: number, smallest: number, largest: number, start: number, scale: number,
 *   total: number, fixed: number}} Divider
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
    !(editable && (SWITCHES.includes(k) || getEditableControls(view.settings).some(([key]) => key === k))));
  const note = [listed.map(([k, v]) => `${SETTINGS[k] || k}: ${SWITCHES.includes(k) ?
    (v ? "on" : "off") : fmt(v)}` +
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

/** Ends the drag that has the divider at its new size: sends the change, unless it is outside the range the engine
 *  accepts (said instead) or where it started.
 * @param {HTMLElement} element
 * @param {Panel} panel */
function finishDrag(element, panel) {
  const {id, value} = /** @type {Drag} */ (panel.drag), d = panel.dividers[id], refusal = findRefusal(d, value);
  panel.drag = null;
  if (refusal)
    panel.notice = {text: `Nothing changed: ${refusal}.`, error: true, until: Infinity};
  if (refusal || mib(value) === mib(d.from))
    render(element, panel);
  else
    void change(element, panel, {[d.key]: mib(value)}, {[d.key]: mib(d.from)});
}

/** Retains a control's edit until the reader applies the form.
 * @param {HTMLElement} element
 * @param {Panel} panel
 * @param {HTMLInputElement} input */
function changeControl(element, panel, input) {
  const key = input.dataset.setting ?? "";
  if (key === "drafter") {
    panel.pending.drafter = input.value === "on";
    if (panel.view.settings.suffix_draft != null) {
      panel.pending.suffix_draft = input.value === "suffix" ? 3 : 0;
      panel.pending.lookup_chain = 0;
    }
  } else if (input.type === "checkbox")
    panel.pending[key] = input.checked;
  else
    panel.pending[key] = input.value === "" ? NaN : Number(input.value);
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
  const editable = new Set(getEditableControls(s).map(([key]) => key));
  const set = save === "default" ? Object.fromEntries(Object.entries(s).filter(([key]) =>
    editable.has(key) || key.startsWith("memory_mib.") ||
    ["ssd_pause_mib", "suffix_draft", "lookup_chain", ...SWITCHES].includes(key))) : {...panel.pending};
  if (!Object.keys(set).length || (save === "default" && !panel.view.config))
    return;
  for (const key of SWITCHES)
    if (key in set)
      set[key] = Boolean(set[key]);
  await change(element, panel, set, null, save);
}

/** Sends a change through the caller's onChange and shows the answer: the new reading with an Undo for a while, or
 *  the server's reason for changing nothing.
 * @param {HTMLElement} element
 * @param {Panel} panel
 * @param {SettingsChange} set
 * @param {SettingsChange | null} undo  the change back, offered as Undo; null for an Undo itself.
 * @param {"session" | "default"} save */
async function change(element, panel, set, undo, save = "session") {
  panel.busy = true;
  panel.notice = {text: "Changing…", error: false, until: Infinity};
  render(element, panel);
  const answer = await /** @type {ChangeSettings} */ (panel.onChange)(set, save)
    .catch((/** @type {Error} */ e) => ({ok: false, error: e.message, view: undefined}));
  panel.busy = false;
  if (answer.ok) {
    for (const key of Object.keys(set))
      delete panel.pending[key];
    if (answer.view)
      panel.view = answer.view;
    try {
      localStorage.setItem(CHANGED_KEY, "1");
    } catch { /* no storage (a private window): the hint stays */ }
    const done = save === "default" ? "Changed, and saved as the default." : "Changed for this session.";
    const shortfall = answer.view?.shortfall;
    panel.notice = {text: shortfall ? `${done} But it got less than asked: ${shortfall}.` : done, error: !!shortfall,
                    until: shortfall ? Infinity : Date.now() + UNDO_MS, ...(undo ? {undo: {set: undo, save}} : {})};
  } else
    panel.notice = {text: answer.error || "The engine changed nothing.", error: true, until: Infinity};
  render(element, panel);
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
    return `<label data-control="${key}" title="${help}">${esc(SETTINGS[key])} ` +
      `<input class="st-input mem-input" type="number" step="1" data-setting="${key}" ` +
      `value="${s[key]}" min="${smallest(s)}"${largest(s) < Infinity ? ` max="${largest(s)}"` : ""}` +
      ` title="${help}"${off}></label>`;
  });
  if (s.drafter != null)
    fields.push(drawDrafterControl(s, panel.view.details, off));
  if (s.save_on_shutdown != null) {
    const help = esc(getControlHelp("save_on_shutdown", s));
    fields.push(`<label class="mem-switch" data-control="save_on_shutdown" title="${help}"><input type="checkbox" ` +
      `data-setting="save_on_shutdown"${s.save_on_shutdown ? " checked" : ""}${off}> ` +
      `${esc(SETTINGS.save_on_shutdown)}</label>`);
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
  const suffix = !settings.drafter && !!(settings.suffix_draft || settings.lookup_chain);
  const help = getControlHelp("drafter", settings) + " " +
    (details.drafter_details || "Drafter discovery details are not reported by this engine.") +
    (settings.suffix_draft == null ? "" :
      " Suffix drafting uses greedy sampling only. Mode changes take effect on the next request.");
  const model = settings.has_drafter === 0 ? "" :
    `<option value="on"${settings.drafter ? " selected" : ""}>` +
    `${esc(details.drafter_label || "Use model drafter (VRAM cost not reported)")}</option>`;
  const lookup = settings.suffix_draft == null ? "" :
    `<option value="suffix"${suffix ? " selected" : ""}>` +
    `${esc(details.suffix_label || "Use suffix drafter (0 GiB)")}</option>`;
  return `<label data-control="drafter" title="${esc(help)}">Drafter ` +
    `<select class="st-input mem-input" data-setting="drafter"${disabled}>` +
    `<option value="off"${!settings.drafter && !suffix ? " selected" : ""}>No MTP</option>` +
    `${model}${lookup}</select></label>`;
}

/** Gets controls supported by this engine; context_capacity advertises a live per-conversation limit.
 * @param {Record<string, number>} settings */
function getEditableControls(settings) {
  return CONTROLS.filter(([key]) => settings[key] != null && (key !== "context" || settings.context_capacity != null));
}

/** Gets the meaning and memory cost of controls with limits that are easy to confuse.
 * @param {string} key
 * @param {Record<string, number | boolean>} settings
 * @param {Record<string, string>} details */
function getControlHelp(key, settings, details = {}) {
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
  if (x.free != null)
    parts.push({use: "free", bytes: x.free});
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
  return {key: ssd ? "ssd_pause_mib" : `memory_mib.${x.id}`, from, smallest: x.smallest ?? 0,
          largest: x.largest ?? Infinity, start, scale, total, fixed};
}

/** Gets why the engine would refuse a divider's size, or "" when it accepts it.
 * @param {Divider} d
 * @param {number} value  bytes */
function findRefusal(d, value) {
  return value < d.smallest ? `at least ${size(d.fixed + d.smallest)} is needed here now (fixed allocations, ` +
    `the biggest active conversation and the minimum conversation tokens)` : value > d.largest ?
    `at most ${size(d.fixed + d.largest)} is available to Strata here (${size(d.fixed)} fixed)` : "";
}

/** Gets the markup of one memory's row: its name and size, its bar (with its divider, moved by `drag`) and the
 *  bar's legend.
 * @param {MemoryUse} x
 * @param {Divider | undefined} divider
 * @param {Drag | null} drag
 * @param {MemoryView} view */
function drawBar(x, divider, drag, view) {
  const {parts, total} = layOut(x);
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
  const value = drag ? drag.value : d.from, refused = drag && findRefusal(d, value) ? " mem-refused" : "";
  const at = (/** @type {number} */ b) => Math.min(100, Math.max(0, (100 * (d.start + b * d.scale)) / d.total));
  const what = id === "ssd" ? "the SSD space paused conversations may take" :
    `Strata's total memory ${id === "ram" ? "in RAM" : "on this GPU"}`;
  const allocation = `${size(d.fixed + value)}${id === "ssd" ? "" :
    ` total (${size(d.fixed)} fixed, ${size(value)} resizable pool)`}`;
  const moved = drag && value !== d.from ? `<span class="mem-change${refused}" style="left:` +
    `${Math.min(at(d.from), at(value))}%;width:${Math.abs(at(value) - at(d.from))}%"></span>` : "";
  return `${moved}<span class="mem-divider${refused}" data-divider="${esc(id)}" tabindex="0" role="slider" ` +
    `aria-label="${esc(what)}" aria-valuetext="${allocation}" title="Drag to change ${esc(what)}. ` +
    `${allocation}. Arrow keys move it too; Enter applies, Escape cancels." style="left:${at(value)}%"></span>`;
}

/** Gets the markup of the line under a bar whose divider is being moved: the size before and after, or in red why
 *  the engine would refuse it.
 * @param {Divider} d
 * @param {Drag} drag */
function drawPreview(d, drag) {
  const refusal = findRefusal(d, drag.value);
  return `<p class="mem-preview${refusal ? " mem-refused" : ""}">` +
    `${size(d.fixed + d.from)} → ${size(d.fixed + drag.value)}` +
    (d.key === "ssd_pause_mib" ? "" : ` Strata total (${size(d.fixed)} fixed; ${size(drag.value)} resizable pool)`) +
    (refusal ? `: ${esc(refusal)}; releasing here changes nothing` : ": release to apply, Escape cancels") + "</p>";
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
