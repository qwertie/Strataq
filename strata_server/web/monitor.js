"use strict";
const $ = id => document.getElementById(id);
let selected = decodeURIComponent(location.hash.slice(1)) || null;   // the main page links to a request as #id
let detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const active = r => !["completed", "error", "disconnected"].includes(r.state);
// The run's average decode speed: every token produced over the run divided by the run's decode time, from each
// finished request's engine timings (predicted_n over predicted_ms). Token-weighted, so a short request does not
// outweigh a long one - the same figure the engine's DONE lines and the server's totals report. Not the last request.
function averageDecodeSpeed(rows) {
  let tokens = 0, ms = 0;
  for (const r of rows) {
    const t = r.timings;
    if (!t) continue;
    const n = t.predicted_n, d = t.predicted_ms;
    if (typeof n === "number" && typeof d === "number" && n >= 2 && d > 0) { tokens += n; ms += d; }
  }
  return ms > 0 ? tokens / (ms / 1000) : null;
}
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
async function api(path, body) {
  const response = await fetch(path, {cache:"no-store", headers:{...(apiKey ? {Authorization:`Bearer ${apiKey}`} : {}), ...(body !== undefined ? {"Content-Type":"application/json"} : {})}, ...(body !== undefined ? {method:"POST", body:JSON.stringify(body)} : {})});
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(query) && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div"); empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `Request ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; lpPage = 0; list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll:true});
  }
}
function content() {
  if (!detail) return;
  const why = noProbabilities(detail), tab = document.querySelector('[data-pane="probabilities"]');
  tab.setAttribute("aria-disabled", String(Boolean(why))); tab.title = why;
  $("lp-bar").hidden = pane !== "probabilities" || Boolean(why);
  if (pane === "probabilities") { showProbabilities(why); return; }
  lpStamp = ""; hideTip();
  // the Probabilities spans go first: text() keeps them when their text is the same
  if ($("content").childElementCount) $("content").replaceChildren();
  let value = detail[pane] || (pane === "response" && detail.stream ? "Streaming response: see Output and the request error, if any." : "No content.");
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? "Raw model output retained for diagnosis. The API returned an error." : pane === "input" ? "Original request body." : pane === "output" ? "Model answer. JSON is formatted here for readability." : pane === "reasoning" ? "Separate reasoning content, when enabled." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated at 256K characters; the API response was not shortened." : ""));
}
async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s)); text("detail-queue", seconds(value.queue_s)); text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    state = status; records = history.requests;
    status.loaded = history.loaded; status.auto_load = history.auto_load;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model); text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on the next request" : "No active requests");
    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(averageDecodeSpeed(records.filter(r => !active(r)))));
    $("load").disabled = operating || Boolean(inflight) || status.loaded; $("unload").disabled = operating || Boolean(inflight) || !status.loaded;
    list(); if (selected) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("load").disabled = $("unload").disabled = true; }
}
async function control(load) {
  actionError = "";
  operating = true; $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try { await api(load ? "/load" : "/unload", {}); notice(""); }
  catch (error) { actionError = error.message; notice(actionError); }
  finally { operating = false; await refresh(); }
}
$("load").onclick = () => control(true); $("unload").onclick = () => control(false);
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); sessionStorage.setItem("strata.monitor.key", apiKey); $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("strata.theme", theme); };
$("search").oninput = list; $("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => { button.onclick = () => { pane = button.dataset.pane; document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button))); content(); }; });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", "Copied"); setTimeout(() => text("copy", "Copy"), 1500); } catch (_) { notice("Clipboard unavailable; select the text to copy it."); } };
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
// Context in VRAM: what the engine holds - its sessions in VRAM and the conversations it parked - read on request (a
// list, then the tokens of the entry shown: they can be 100K+, so nothing is polled)
let ctx = null, entry = null, ctxView = "plain", ctxPage = -1, ctxPicked = false;   // ctxPage -1: the last page
const STATES = {holding: "idle", decoding: "decoding, still growing", reading: "its prompt is part read and goes on later",
  empty: "empty", parked: "parked"};
const NOTES = {session: "The tokens this session holds in VRAM, as the engine reports them: the prompt and the answer it has fed so far.",
  parked: "A conversation the engine set aside when another one took its session. Its next request takes it back instead of reading it again. The first one listed is the next to be evicted."};
// The colour comes from the id, so the same token always has the same colour: a hash of the id picks one of a fixed
// palette per theme.  The palette samples every allowed colour uniformly by volume in OKLab (roughly perceptual), so
// muted and very light or dark colours, which are a smaller part of that space, get fewer entries.  Allowed: inside
// sRGB, a WCAG contrast of 7:1 or more against the theme's text, and clear of the page background (the lightness
// range).  A seeded generator makes the palette the same on every load.
const PALETTE_SIZE = 4096;
const THEMES = {light: {text: [0x11, 0x11, 0x11], L: [0.66, 0.93]}, dark: {text: [0xee, 0xf1, 0xf3], L: [0.25, 0.50]}};
const toLinear = c => (c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
const toSrgb = c => Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
function oklabToLinear(L, a, b) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3,
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
}
function makePalette({text, L: [lo, hi]}) {
  const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b, ink = lum(text.map(toLinear));
  let seed = 1;
  const random = () => {                              // mulberry32
    let t = seed = (seed + 0x6D2B79F5) | 0;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  const palette = [];
  while (palette.length < PALETTE_SIZE) {
    const rgb = oklabToLinear(lo + (hi - lo) * random(), 0.5 * random() - 0.25, 0.5 * random() - 0.25);
    if (rgb.some(c => c < 0 || c > 1)) continue;
    const srgb = rgb.map(toSrgb), y = lum(srgb.map(toLinear));   // the contrast of the colour as it is drawn
    if ((Math.max(y, ink) + 0.05) / (Math.min(y, ink) + 0.05) < 7) continue;
    palette.push(`rgb(${srgb.join(" ")})`);
  }
  return palette;
}
let palettes = null;                                  // made when the colored view is first shown
const tokenColour = id => (Math.imul(id, 2654435761) >>> 0) >>> 20;   // the hash's top 12 bits: an entry of 4096
const PAGE = 8000;                                    // tokens per page of the colored view: the page stays light
function ctxEntry() { return ctx?.entries?.find(e => e.id === $("ctx-slot").value) || null; }
const gb = bytes => bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(2)} GB` : `${Math.round(bytes / 1048576)} MB`;
const PLACES = {vram: "VRAM", ram: "RAM", disk: "disk"};
// a session holds `count` of `capacity` tokens in `bytes` allocated, its tokens taking `used`; a parked conversation
// is just its tokens.  Bytes are 0 where the engine does not know them.
function sizeText(e) {
  if (e.kind === "parked") return `${e.count.toLocaleString()} tokens, ${gb(e.bytes)}`;
  const bytes = e.bytes ? `, ${e.used ? `${gb(e.used)} of ` : ""}${gb(e.bytes)}` : "";
  return `${e.count.toLocaleString()} of ${e.capacity.toLocaleString()} tokens${bytes}`;
}
// where it is first: its name and VRAM / RAM / disk, then its size, then the start of its first user message
function entryLabel(e) {
  const state = e.kind === "session" && e.state !== "holding" ? ` · ${STATES[e.state] || e.state}` : "";
  return `${e.name} · ${PLACES[e.where] || e.where} · ${sizeText(e)}${state}` + (e.preview ? ` · “${e.preview.slice(0, 60)}”` : "");
}
function showContext() {
  const view = $("ctx-view"), e = ctxEntry(), got = entry && e && entry.id === e.id ? entry : null;
  $("ctx-save").disabled = !got?.tokens?.length; $("ctx-pager").hidden = true;
  if (!ctx) return;
  view.replaceChildren(); text("ctx-note", "");
  if (!ctx.available) {
    text("ctx-sum", ctx.reason === "engine" ? "This engine cannot list what it holds: it predates the LIST_CONVERSATIONS command. Update the engine to see it." : "The model is not loaded.");
    return;
  }
  const when = ctx.at ? new Date(ctx.at * 1000).toLocaleTimeString() : "";
  if (!ctx.at) { text("ctx-sum", ctx.error || "A request has the engine's turn: press Refresh when it has finished."); return; }
  if (!e) { text("ctx-sum", "Nothing held."); return; }
  const stale = ctx.busy ? ` · a request has the engine's turn: as read at ${when}` : ctx.error ? ` · ${ctx.error}: as read at ${when}` : "";
  const n = got?.tokens ? got.tokens.length : e.count;
  const size = got?.tokens && got.tokens.length !== e.count ? {...e, count: n} : e;   // (it may have grown since the list)
  text("ctx-sum", `${e.name} · ${e.kind === "parked" ? `parked in ${PLACES[e.where] || e.where}` : `VRAM, ${STATES[e.state] || e.state}`} · ${sizeText(size)}${stale}`);
  if (!got) { if (n) text("ctx-note", entry?.error || "Reading…"); return; }
  text("ctx-note", (got.busy ? `A request has the engine's turn: these are the tokens read at ${new Date(got.at * 1000).toLocaleTimeString()}. ` : "") +
    (ctxView === "colored" ? "Each token's colour comes from its id, so the same token always has the same colour. Hover a token for its position and id. " : "") + NOTES[e.kind]);
  if (got.error) { text("ctx-note", got.error); return; }
  if (ctxView === "plain") { view.textContent = got.pieces.join(""); return; }
  palettes ??= {light: makePalette(THEMES.light), dark: makePalette(THEMES.dark)};
  ctxPage = drawTokenPage(view, "ctx", got.tokens.length, ctxPage, (span, i) => {
    const id = got.tokens[i], k = tokenColour(id);
    span.style.setProperty("--bl", palettes.light[k]); span.style.setProperty("--bd", palettes.dark[k]);   // both themes
    span.textContent = got.pieces[i]; span.title = `token #${i} · id ${id}`;
  });
}
// Appends one page of `count` tokens to `view` as `.tok` spans, each filled in by `paint(span, index)`, and sets up
// the pager whose element ids start with `prefix` ("ctx": Context in VRAM, "lp": Probabilities). `page` -1 is the
// last page. Returns the page drawn.
function drawTokenPage(view, prefix, count, page, paint) {
  const pages = Math.max(1, Math.ceil(count / PAGE));
  page = Math.min(Math.max(page < 0 ? pages - 1 : page, 0), pages - 1);
  const start = page * PAGE, end = Math.min(count, start + PAGE);
  $(`${prefix}-pager`).hidden = pages < 2;
  $(`${prefix}-prev`).disabled = page === 0; $(`${prefix}-next`).disabled = page === pages - 1;
  text(`${prefix}-range`,
    `Tokens ${start.toLocaleString()}–${Math.max(start, end - 1).toLocaleString()} (page ${page + 1} of ${pages})`);
  const frag = document.createDocumentFragment();     // one append: one layout
  for (let i = start; i < end; i++) {
    const span = document.createElement("span"); span.className = "tok";
    paint(span, i);
    frag.append(span);
  }
  view.append(frag);
  return page;
}
// the tokens of the conversation picked, read from the engine now (they may have grown since the list)
async function loadEntry() {
  const e = ctxEntry();
  if (!e) { entry = null; showContext(); return; }
  text("ctx-note", `Reading tokens for ${e.name}; waiting for the current model operation if needed…`);
  let answer;
  try { answer = e.count ? await api(`/api/context?id=${encodeURIComponent(e.id)}`) :
    {id: e.id, tokens: [], pieces: []}; }
  catch (error) { answer = {id: e.id, error: error.message}; }
  if (ctxEntry() === e) {
    if (answer.error && entry?.id === e.id && entry.tokens) {
      text("ctx-sum", `${e.name}: refresh failed`);
      text("ctx-note", `${answer.error}. Showing the previous context snapshot` +
        (entry.at ? ` from ${new Date(entry.at * 1000).toLocaleTimeString()}.` : "."));
      return;
    }
    entry = answer; ctxPage = -1;
    showContext();
  }
}
async function loadContext() {
  $("ctx-refresh").disabled = true; text("ctx-sum", "Reading…");
  try {
    const keep = ctxEntry()?.id;
    const [context, memory, panel] = await Promise.all([
      api("/api/context"), api("/v1/memory").catch(() => null), import("./memory-panel.js"),
    ]);
    ctx = context;
    const entries = ctx.entries || [], select = $("ctx-slot");
    select.replaceChildren(...entries.map(e => { const o = document.createElement("option"); o.value = e.id; o.textContent = entryLabel(e); return o; }));
    const latest = entries.reduce((best, e) => (e.last_used || 0) > (best.last_used || 0) ? e : best, entries[0]);
    const list = $("ctx-conversations");
    const conversations = entries.map(e => ({...(memory?.conversations || []).find(c => c.id === e.id),
      ...e, place: e.where}));
    list.innerHTML = panel.drawConversations(conversations, memory?.pause_directory, memory?.settings?.tokens_max);
    list.querySelectorAll("[data-context]").forEach(button => {
      button.onclick = () => { select.value = button.dataset.context; ctxPicked = true; loadEntry(); };
    });
    select.value = ctxPicked && entries.some(e => e.id === keep) ? keep : latest?.id ?? "";
    select.hidden = true;
    await loadEntry();
  } catch (error) { text("ctx-sum", error.message); }
  finally { $("ctx-refresh").disabled = false; }
}
function saveContext() {
  const e = ctxEntry(); if (!entry?.tokens || !e) return;
  const binary = $("ctx-binary").checked, parts = [entry.pieces.join("")];
  if (binary) {                                       // the ids are the file's last 4 x N bytes
    const ids = new DataView(new ArrayBuffer(entry.tokens.length * 4));
    entry.tokens.forEach((id, i) => ids.setUint32(i * 4, id, true));
    parts.push(`\n\n----- ${entry.tokens.length} token ids, uint32 little-endian, to the end of the file -----\n`, ids.buffer);
  }
  const a = document.createElement("a"), which = e.id;
  a.href = URL.createObjectURL(new Blob(parts, {type: binary ? "application/octet-stream" : "text/plain;charset=utf-8"}));
  a.download = `strata-context-${which}-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("ctx-refresh").onclick = loadContext; $("ctx-save").onclick = saveContext;
$("ctx-slot").onchange = () => { ctxPicked = true; loadEntry(); };
$("ctx-prev").onclick = () => { ctxPage--; showContext(); }; $("ctx-next").onclick = () => { ctxPage++; showContext(); };
document.querySelectorAll("[data-view]").forEach(button => { button.onclick = () => { ctxView = button.dataset.view; document.querySelectorAll("[data-view]").forEach(b => b.setAttribute("aria-selected", String(b === button))); showContext(); }; });

loadContext();
// Probabilities: the selected request's answer as tokens tinted by the probability the model gave each one, from the
// logprobs entries the server kept for it (OpenAI's: token, logprob, top_logprobs), which only a client that asked has
const LP_STEPS = [0.9, 0.5, 0.2, 0.05, 0];            // the lowest probability of each tint, class lp0 (sure) to lp4
const LP_FLOOR = -65535 / 2048;                      // the lowest logprob the server keeps (MonitorLogprobs)
let lpPage = 0, lpStamp = "", lpShown = [];           // lpShown: the entries drawn, by span index (data-i)
let lpData = null, lpLoading = null;                  // the last entries read (loadProbabilities), the read under way
$("lp-legend").replaceChildren(...LP_STEPS.map((_, i) => {
  const swatch = document.createElement("span");
  swatch.className = `tok lp lp${i}`; swatch.textContent = probabilityRange(i);
  return swatch;
}));
const lpView = $("content");
for (const kind of ["mouseover", "focusin"])
  lpView.addEventListener(kind, event => showTip(event.target));
for (const kind of ["mouseout", "focusout", "scroll"])
  lpView.addEventListener(kind, hideTip);
lpView.addEventListener("keydown", moveTokenFocus);
$("lp-prev").onclick = () => { lpPage--; content(); }; $("lp-next").onclick = () => { lpPage++; content(); };
// Gets why request record `d` has no token probabilities to show, or "" when it has some.
function noProbabilities(d) {
  if (d.logprobs_tokens == null) {
    return "No token probabilities: only a /v1/chat/completions request that asks for logprobs has them.";
  }
  if (d.logprobs_dropped)
    return "The monitor dropped this request's token probabilities to make room for newer requests' (32 MiB in all).";
  return d.logprobs_tokens ? "" : "The client asked for token probabilities, but the answer came with none.";
}
// Gets the entries of record `d` read with its logprobs, in the order the model wrote them: the reasoning's, then
// the answer's.
function entriesInOrder(d) { return [...(d.logprobs?.reasoning_content || []), ...(d.logprobs?.content || [])]; }
// Draws the selected request's entries as tinted token spans, a page at a time, or only says `why` there are none
// (noProbabilities's reason). The entries are read apart from the record, and again only when more were kept.
function showProbabilities(why) {
  const got = !why && lpData?.id === detail.id ? lpData : null, thought = got?.thought || 0;
  text("content-note", why || "Each token is tinted by the probability the model gave it. Hover a token, or " +
    "focus one and move with the arrow keys, to see its probability and the likeliest tokens at that place." +
    (thought ? " The reasoning comes first, in italics, then the answer." : "") +
    " The monitor keeps logprobs rounded to about 4 decimals." +
    (detail.logprobs_truncated ?
      " Monitor capture stopped at 32,768 tokens; the API response was not shortened." : "") +
    (got ? "" : " Reading…"));
  if (!why && got?.tokens !== detail.logprobs_tokens)
    loadProbabilities(detail.id, detail.logprobs_tokens);
  const stamp = `${detail.id} ${got?.tokens} ${lpPage} ${why}`;   // redrawn on a change only: a hover outlives polls
  if (stamp === lpStamp)
    return;
  lpStamp = stamp; hideTip(); lpView.replaceChildren(); lpShown = got?.entries || [];
  if (!got)
    return;
  lpPage = drawTokenPage(lpView, "lp", lpShown.length, lpPage, (span, i) => {
    const e = lpShown[i];
    const part = i < thought ? " lp-thought" : thought && i === thought ? " lp-answer" : "";
    span.className += ` lp lp${probabilityStep(e.logprob)}${part}`;
    span.textContent = e.token; span.dataset.i = i; span.tabIndex = -1;
  });
  if (lpView.firstElementChild)
    lpView.firstElementChild.tabIndex = 0;            // one tab stop: the arrow keys move within the page
}
// Reads the entries of request `id`, of which `tokens` were kept when asked, into lpData, then shows them.
async function loadProbabilities(id, tokens) {
  const key = `${id} ${tokens}`;
  if (lpLoading === key)
    return;
  lpLoading = key;
  try {
    const value = await api(`/api/requests?id=${encodeURIComponent(id)}&logprobs=1`);
    lpData = {id, tokens, entries: entriesInOrder(value), thought: value.logprobs?.reasoning_content?.length || 0};
  } catch (error) { notice(error.message); return; }   // (the next poll tries again)
  finally { lpLoading = null; }
  if (pane === "probabilities" && detail?.id === id)
    content();
}
// Gets the index in LP_STEPS of the tint for a token of log-probability `logprob`.
function probabilityStep(logprob) { return LP_STEPS.findIndex(min => Math.exp(logprob) >= min); }
// Gets the legend's words for tint `i` of LP_STEPS, such as "50–90%".
function probabilityRange(i) {
  const [low, high] = [LP_STEPS[i], LP_STEPS[i - 1]].map(p => Math.round(p * 100));
  return i === 0 ? `${low}% or more` : low ? `${low}–${high}%` : `under ${high}%`;
}
// Gets a probability, given as its natural log, as a percentage for people.
function percent(logprob) {
  const p = 100 * Math.exp(logprob);
  return p < 0.01 ? "under 0.01%" : `${p.toFixed(p < 10 ? 2 : 1)}%`;
}
// Shows, beside token span `span` of the Probabilities view, its probability and the likeliest tokens at its place.
function showTip(span) {
  const e = pane === "probabilities" ? lpShown[span.dataset?.i] : undefined;
  if (!e)
    return;
  const tip = $("lp-tip");
  const line = (token, logprob) =>
    `${JSON.stringify(token)}  ${percent(logprob)} · logprob ${logprob <= LP_FLOOR ? "< -32" : logprob.toFixed(4)}`;
  const head = document.createElement("b"); head.textContent = line(e.token, e.logprob);
  const note = document.createElement("div"); note.className = "small";
  const alternatives = document.createElement("ol");
  for (const a of e.top_logprobs || []) {
    const item = document.createElement("li");
    item.textContent = line(a.token, a.logprob) + (a.token === e.token ? "  (chosen)" : "");
    alternatives.append(item);
  }
  note.textContent = alternatives.childElementCount ? "Likeliest tokens at this place:" :
    "No alternatives: the client asked for none (top_logprobs 0).";
  tip.replaceChildren(head, note, alternatives); tip.hidden = false;
  span.setAttribute("aria-describedby", "lp-tip");
  const r = span.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(r.left, innerWidth - tip.offsetWidth - 8))}px`;
  tip.style.top = `${r.bottom + 6 + tip.offsetHeight > innerHeight ? r.top - tip.offsetHeight - 6 : r.bottom + 6}px`;
}
function hideTip() { $("lp-tip").hidden = true; }
// Moves the focus between the Probabilities view's tokens: left and right arrows, Home and End; Escape hides the tip.
function moveTokenFocus(event) {
  const span = event.target;
  if (span.dataset?.i === undefined)
    return;
  if (event.key === "Escape") { hideTip(); return; }
  const to = {ArrowRight: span.nextElementSibling, ArrowLeft: span.previousElementSibling,
    Home: lpView.firstElementChild, End: lpView.lastElementChild}[event.key];
  if (!to)
    return;
  event.preventDefault(); span.tabIndex = -1; to.tabIndex = 0; to.focus();
}
