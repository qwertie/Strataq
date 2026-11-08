// The Memory panel (strata_server/web/memory-panel.js): one framework-free module that the server's web page imports
// and the app's Monitor screen wraps (src/components/MemoryPanel.tsx), so both draw the same bars from GET /v1/memory.
// The fixture is strata-q38's real GET_MEMORY_USE answer with three conversations on a Strix Halo (2026-10-31),
// passed through the server's StrataEngine.memory and memory_view with a 64 GiB BIOS share.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseHTML } from 'linkedom';
import { drawMemoryPanel, drawConversations } from '../../../strata_server/web/memory-panel.js';

const read = path => readFile(new URL(path, import.meta.url), 'utf8');
const fixture = JSON.parse(await read('./fixtures/memory-strix-halo.json'));
/** Mounts the panel in a DOM; `fire` invokes its listeners with synthetic pointer and control events. */
const mount = (view, onChange) => {
  const {document} = parseHTML('<html><body><div id="panel"></div></body></html>');
  const listeners = {}, element = document.getElementById('panel');
  element.addEventListener = (type, listener) => { listeners[type] = listener; };
  element.setPointerCapture = () => {};
  drawMemoryPanel(element, view, onChange);
  const fire = (type, event) => listeners[type]({ button: 0, pointerId: 1, preventDefault() {}, ...event });
  return { element, fire };
};
/** Gets an event target inside the divider of memory `id`, on a bar 1000 pixels wide. */
const onDivider = id => ({ closest: selector => selector === '[data-divider]'
  ? { dataset: { divider: id }, parentElement: { getBoundingClientRect: () => ({ width: 1000 }) } } : null });
/** Gets an event target that is one of the panel's controls, matching `selector`. */
const control = (selector, fields) => ({ matches: s => s === selector, ...fields });
const draw = view => mount(view).element.innerHTML;
const drawHeld = view => drawConversations(view.conversations || [], view.pause_directory, view.settings?.tokens_max);
const applyButton = save => ({closest: selector => selector === '[data-apply]' ? {dataset: {apply: save}} : null});
const apply = (fire, save = 'session') => fire('click', {target: applyButton(save)});
const rows = html => html.split(/<div class="mem-row"[^>]*>/).slice(1);
const widths = row => [...row.matchAll(/class="mem-seg"[^>]*style="width:([\d.e-]+)%/g)].map(m => Number(m[1]));
const barWidth = row => Number(/class="mem-bar" style="width:([\d.]+)%/.exec(row)[1]);

test('polling updates memory facts without replacing controls, rows or dividers', () => {
  const {element, fire} = mount(live, recorder().onChange);
  const selectors = ['.mem-controls', '[data-memory="gpu0"]', '[data-divider="gpu0"]', '[data-setting="rows"]'];
  const nodes = selectors.map(selector => element.querySelector(selector));
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'rows'}, value: '6'})});
  drawMemoryPanel(element, {...live, memories: live.memories.map(m => ({...m, allocated_pct: 75}))},
    recorder().onChange);
  selectors.forEach((selector, i) => assert.equal(element.querySelector(selector), nodes[i]));
  assert.equal(nodes[3].value, '6');
  assert.match(element.querySelector('.mem-facts').textContent, /75[.,]0% allocated/);
});

test('the right edge counts fixed allocations in the total while sending only the pool size', async () => {
  const view = {available: true, can_change: true, settings: {}, memories: [{id: 'gpu0', total: 64 * GIB,
    free: 33 * GIB, setting: 24 * GIB, pool_offset: 7 * GIB, largest: 57 * GIB,
    uses: {weights: 7 * GIB, experts: 24 * GIB}}]};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  fire('pointerdown', {target: onDivider('gpu0'), clientX: 1000 * 31 / 64});
  fire('pointermove', {clientX: 1000});
  assert.match(element.querySelector('.mem-preview').textContent, /64.00 GiB Strata total/);
  assert.match(element.querySelector('.mem-preview').textContent, /7.00 GiB fixed; 57.00 GiB resizable pool/);
  fire('pointerup', {});
  await settle();
  assert.deepEqual(calls, [[{'memory_mib.gpu0': 57 * 1024}, 'session']]);
});

test('context snippets are escaped and rows show individual limits above the shared total', () => {
  const html = drawConversations([{id: 'conversation0', count: 100, capacity: 204800, place: 'vram',
    state: 'holding', bytes: GIB, preview: '<user> show my files'}], undefined, 401111);
  assert.match(html, /mem-context-preview">&lt;user&gt; show my files/);
  assert.match(html, />100 \/ 204.800<\/td>/);
  assert.match(html, /GPU total \(shared budget\)[^]*100 \/ 401.111/);
});

test('a GPU whose memory is system RAM is drawn as its BIOS share, and what it borrows is a segment of the RAM', () => {
  const [gpu, ram] = rows(draw(fixture));
  assert.match(gpu, /AMD Radeon\(TM\) 8060S Graphics — BIOS share 64.0 GiB/);
  assert.match(ram, /System RAM 63.5 GiB \(GPU borrows 0.3 GiB\)/);
  assert.equal(barWidth(gpu), 100);                       // above 32 GiB: the full width
  // the engine's uses are scaled to what it holds in the share, so the segments fill the bar exactly
  assert.ok(Math.abs(widths(gpu).reduce((a, b) => a + b, 0) - 100) < 1e-9);
  assert.match(gpu, /Experts 15.54 GiB/);                 // the legend keeps the engine's own numbers
  assert.match(gpu, /data-use="pool_free"[^]*Free in the pool 53 MiB/);
  assert.match(ram, /data-use="borrowed"/);
});

test('a memory below 32 GiB is as wide as its share of 32 GiB; names are escaped', () => {
  const card = { id: 'gpu1', name: 'RTX <3060>', total: 16 * 2 ** 30, free: 2 ** 30, unified: false, paused_count: null,
    paused_budget: null, uses: { experts: 15 * 2 ** 30 } };
  const [row] = rows(draw({ available: true, memories: [card], settings: {} }));
  assert.equal(barWidth(row), 50);
  assert.match(row, /RTX &lt;3060&gt;/);
  assert.deepEqual(widths(row), [93.75, 6.25]);
});

test('a page pool says how much of it is allocated and how much is free in runs no expert fits', () => {
  const card = { id: 'gpu0', name: 'GPU', total: 2 ** 30, free: 0, unified: false, paused_count: null,
    paused_budget: null, uses: { experts: 2 ** 29 }, allocated_pct: 97.5, fragmented_pct: 0.38 };
  assert.match(rows(draw({ available: true, memories: [card], settings: {} }))[0],
    /mem-facts">1[.,]0 GiB · pool 97[.,]5% allocated, 0[.,]38% fragmented</);
  // engines without a pool (and memories other than the pool's) say nothing of it
  assert.doesNotMatch(draw({ available: true, memories: [{ ...card, allocated_pct: null }], settings: {} }),
    /allocated/);
});

test('the conversations the engine holds are listed with their state, tokens and memory', () => {
  const html = drawHeld(fixture), table = /<table class="mem-conversations">(.*)<\/table>/.exec(html)?.[1] ?? '';
  assert.equal(table.match(/<tr>/g).length, 1 + fixture.conversations.filter(c => c.count).length);
  assert.match(table, /<td><button type="button" data-context="slot2">Conversation 3<\/button><\/td><td>holding \(GPU\)<\/td><td class="mem-num">8.401 \/ 32.768<\/td>/);
  assert.match(table, /8.401 \/ 32.768<\/td><td class="mem-num">230 MiB<\/td>/);
  // engines before strata-q38 list no conversations
  assert.doesNotMatch(draw({ ...fixture, conversations: undefined }), /mem-conversations/);
});

test('the settings line names the settings and says when the reading is an old one', () => {
  assert.match(draw(fixture), new RegExp('<p class="mem-note">Per-conversation token limit: 32.768 tokens · ' +
    'Conversations at once: 3 · ' +
    'Rows per step: 8 · Max lookahead: 0 · Min conversation tokens: 8.192</p>'));
  assert.match(draw({ ...fixture, busy: true }), /mem-note">[^<]* · a request has the engine: the last reading</);
});

test('the web page and the app load the one module instead of copies', async () => {
  assert.match(await read('../../../strata_server/web/app.js'), /import\("\.\/memory-panel\.js"\)/);
  assert.match(await read('../../../strata_server/web/index.html'), /href="web\/memory-panel\.css"/);
  const wrapper = await read('../src/components/MemoryPanel.tsx');
  assert.match(wrapper, /from '\.\.\/\.\.\/\.\.\/\.\.\/strata_server\/web\/memory-panel\.js'/);
  assert.match(wrapper, /import '\.\.\/\.\.\/\.\.\/\.\.\/strata_server\/web\/memory-panel\.css'/);
  assert.match(await read('../src/screens/Monitor.tsx'), /<MemoryPanel instanceId=\{selectedInstance\.id\}\/>/);
});

// M6: an engine that changes its settings while it runs (strata-q38's READY says `settings`).  The GPU's pool is what
// the fixture's engine holds in it (experts, conversations, free pages); it accepts 4 to 30 GiB.
const GIB = 2 ** 30, MIB = 2 ** 20;
const [gpu0, ram] = fixture.memories, pool = gpu0.uses.experts + gpu0.uses.conversations + gpu0.uses.pool_free;
const live = { ...fixture, can_change: true, config: true,
  memories: [{ ...gpu0, setting: pool, smallest: 4 * GIB, largest: 30 * GIB }, ram],
  settings: { ...fixture.settings, slots: 4, drafter: 1, tokens_max: 32768, 'memory_mib.gpu0': pool / MIB } };
/** Bytes one pixel of the 1000-pixel GPU bar stands for: the BIOS share, drawn scaled to what the engine holds in
 *  it. */
const perPx = gpu0.total / (gpu0.in_share / Object.values(gpu0.uses).reduce((a, b) => a + b)) / 1000;
/** Gets a stand-in for the caller's onChange that records its calls and answers `answer`. */
const recorder = (answer = { ok: true }) => {
  const calls = [];
  return { calls, onChange: async (set, save) => { calls.push([set, save]); return answer; } };
};
const settle = () => new Promise(resolve => setImmediate(resolve));
const undoButton = { closest: selector => selector === '[data-undo]' ? {} : null };

test('dragging a divider previews the new size and changes nothing until it is released', async () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  assert.match(element.innerHTML, /class="mem-divider" data-divider="gpu0" tabindex="0" role="slider"/);
  assert.match(element.innerHTML, /title="Drag to change Strata(?:&#39;|')s total memory on this GPU/);
  assert.match(element.innerHTML, /<p class="mem-hint">Drag the dividers to change sizes<\/p>/);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 600 });
  assert.equal(calls.length, 0);
  assert.match(element.innerHTML, /<p class="mem-preview">19[.,]83 GiB → 26[.,]\d\d GiB Strata total/);
  assert.match(element.innerHTML, /3[.,]75 GiB fixed; 22[.,]\d\d GiB resizable pool/);
  assert.match(element.innerHTML, /class="mem-change"/);
  fire('pointerup', {});
  await settle();
  assert.deepEqual(calls, [[{ 'memory_mib.gpu0': Math.round((pool + 100 * perPx) / MIB) }, 'session']]);
  assert.match(element.querySelector('.mem-notice').textContent, /Changed for this session\. Undo/);
});

test('a change that got less than asked says what it lacked and still offers Undo', async () => {
  const { onChange } = recorder({ ok: true, view: { ...live, shortfall: "the RAM's pool: out of memory" } });
  const { element, fire } = mount(live, onChange);
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowRight' });
  fire('keydown', { target: onDivider('gpu0'), key: 'Enter' });
  await settle();
  assert.match(element.querySelector('.mem-notice.mem-error').textContent,
    /Changed for this session\. But it got less than asked: the RAM's pool: out of memory\. Undo/);
});

test('Apply and save as default persists current controls and live pool sizes', async () => {
  const view = {available: true, can_change: true, config: true, memories: [],
    settings: {rows: 4, concurrent: 1, 'memory_mib.gpu0': 8192}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  assert.doesNotMatch(element.innerHTML, /data-save|Changes apply/);
  apply(fire, 'default');
  await settle();
  assert.deepEqual(calls, [[{rows: 4, concurrent: 1, 'memory_mib.gpu0': 8192}, 'default']]);
  assert.match(mount({...live, config: false}, onChange).element.innerHTML,
    /data-apply="default" disabled>Apply and save as default/);
});

test('Undo sends a live size change back for the session', async () => {
  const {calls, onChange} = recorder(), {fire} = mount(live, onChange);
  fire('keydown', {target: onDivider('gpu0'), key: 'ArrowRight', shiftKey: true});
  fire('keydown', {target: onDivider('gpu0'), key: 'Enter'});
  await settle();
  fire('click', {target: undoButton});
  await settle();
  assert.deepEqual(calls, [[{'memory_mib.gpu0': Math.round(pool / MIB) + 1024}, 'session'],
    [{'memory_mib.gpu0': Math.round(pool / MIB)}, 'session']]);
});

test('a size outside what the engine accepts shows red with the reason, and releasing there changes nothing', () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 250 });                 // about 16.9 GiB less: below the 4 GiB the engine needs
  assert.match(element.innerHTML, /class="mem-divider mem-refused"/);
  assert.match(element.innerHTML,
    /<p class="mem-preview mem-refused">19[.,]83 GiB → [^:]*: at least 7[.,]75 GiB is needed here now/);
  assert.match(element.innerHTML, /releasing here changes nothing/);
  fire('pointerup', {});
  assert.equal(calls.length, 0);
  assert.match(element.querySelector('.mem-notice.mem-error').textContent, /Nothing changed: at least 7[.,]75 GiB/);
});

test('Escape cancels a drag, and dropping a divider where it started changes nothing', () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 650 });
  fire('keydown', { target: onDivider('gpu0'), key: 'Escape' });
  assert.doesNotMatch(element.innerHTML, /mem-preview/);
  fire('pointerup', {});
  fire('pointerdown', { target: onDivider('gpu0'), clientX: 500 });
  fire('pointermove', { clientX: 650 });
  fire('pointermove', { clientX: 500 });
  fire('pointerup', {});
  assert.equal(calls.length, 0);
  assert.doesNotMatch(element.innerHTML, /mem-notice/);
  // a keyboard move the reader leaves by moving the focus away is dropped at the next reading
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowRight' });
  drawMemoryPanel(element, live, onChange);
  assert.doesNotMatch(element.innerHTML, /mem-preview/);
});

test('the server\'s refusal is shown as it said it', async () => {
  const why = 'the biggest active conversation needs 3.10 GiB; the GPU&#39;s pool would keep 2.00 GiB';
  const { onChange } = recorder({ ok: false, error: why.replace('&#39;', "'") });
  const { element, fire } = mount(live, onChange);
  fire('keydown', { target: onDivider('gpu0'), key: 'ArrowLeft' });
  fire('keydown', { target: onDivider('gpu0'), key: 'Enter' });
  assert.match(element.querySelector('.mem-notice').textContent, /^Changing/);
  await settle();
  assert.equal(element.querySelector('.mem-notice.mem-error').textContent, why.replace('&#39;', "'"));
  assert.doesNotMatch(element.innerHTML, /data-undo/);
});

test('the controls change the other settings within their ranges', async () => {
  const { calls, onChange } = recorder(), { element, fire } = mount(live, onChange);
  assert.match(element.innerHTML, /data-setting="concurrent" value="3" min="1" max="4"/);
  assert.match(element.innerHTML, /data-setting="rows" value="8" min="3" max="8"/);
  assert.match(element.innerHTML, /data-setting="lookahead" value="0" min="0" max="7"/);
  assert.match(element.innerHTML, /<select[^>]*data-setting="drafter"/);
  const setting = (key, fields) => control('[data-setting]', { dataset: { setting: key }, ...fields });
  fire('change', { target: setting('concurrent', { value: '5' }) });
  apply(fire);
  assert.equal(calls.length, 0);
  assert.match(element.querySelector('.mem-notice.mem-error').textContent,
    /Conversations at once: choose a whole number from 1 to 4\./);
  fire('change', { target: setting('concurrent', { value: '2' }) });
  fire('change', { target: setting('drafter', { value: 'off' }) });
  assert.deepEqual(calls, []);
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{ concurrent: 2, drafter: false }, 'session']]);
  // the controls' settings are not listed again in the line under the panel
  assert.match(element.innerHTML,
    /<p class="mem-note">Per-conversation token limit:.*Maximum conversations at once: 4<\/p>/);
});

test('every segment of a bar says in its tooltip what it is', () => {
  const card = { id: 'gpu0', name: 'card', total: 16 * GIB, free: GIB, unified: false, paused_count: null,
    paused_budget: null, other: GIB, uses: { weights: 4 * GIB, experts: 8 * GIB, conversations: GIB, pool_free: GIB } };
  const [row] = rows(draw({ available: true, memories: [card], settings: {} }));
  const tips = [...row.matchAll(/class="mem-seg"[^>]*title="([^"]*)"/g)].map(m => m[1]);
  assert.equal(tips.length, 6);
  assert.match(tips[0], /^Weights: 4.00 GiB\. The parts of the model that must always be loaded here \(attention/);
  assert.match(tips[5], /^Other apps: 1.00 GiB\. Memory used outside Strata/);
  for (const t of tips)
    assert.match(t, /: [\d.,]+ (GiB|MiB)\. \S/);
});

test('an engine that cannot change its settings while it runs shows them read-only', () => {
  const { calls, onChange } = recorder(), fixed = mount({ ...live, can_change: false }, onChange).element.innerHTML;
  for (const html of [fixed, draw(live)]) {             // draw(live): a caller that sends no changes
    assert.doesNotMatch(html, /data-divider|<input|<select|mem-hint/);
    assert.match(html, /Conversations at once: 3/);
  }
  assert.match(fixed, /Drafter: on · .* · these change when the model starts again<\/p>/);
  assert.equal(calls.length, 0);
});

test('the SSD\'s divider is the end of the room for paused conversations', async () => {
  const ssd = { id: 'ssd', name: null, total: null, free: null, unified: false, paused_count: 0,
    paused_budget: 8 * GIB, uses: { experts: 8 * GIB } };
  const { calls, onChange } = recorder(), { element, fire } = mount({ ...live, memories: [ssd] }, onChange);
  assert.match(element.innerHTML, /data-divider="ssd"[^>]*style="left:100%"/);
  fire('pointerdown', { target: onDivider('ssd'), clientX: 1000 });
  fire('pointermove', { clientX: 750 });                 // a quarter of the bar's 16 GiB less
  fire('pointerup', {});
  await settle();
  assert.deepEqual(calls, [[{ ssd_pause_mib: 4096 }, 'session']]);
});


test('live concurrency uses the step capacity and available rows', () => {
  const view = { ...live, settings: { ...live.settings, concurrent: 1, slots: 8, rows: 3 } };
  const html = mount(view, recorder().onChange).element.innerHTML;
  assert.match(html, /data-setting="concurrent" value="1" min="1" max="3"/);
});


test('draft experts have their own segment and legend without changing the pool total', () => {
  const card = { id: 'gpu0', name: 'GPU', total: 16 * 2 ** 30, free: 4 * 2 ** 30,
    uses: { experts: 4 * 2 ** 30, drafter_experts: 7 * 2 ** 30, weights: 2 ** 30 } };
  const [row] = rows(draw({ available: true, memories: [card], settings: {} }));
  assert.match(row, /data-use="drafter_experts"/);
  assert.match(row, /Drafter experts 7.00 GiB/);
  assert.match(row, /Experts 4.00 GiB/);
  assert.ok(Math.abs(widths(row).reduce((sum, width) => sum + width, 0) - 100) < 1e-9);
});


test('arbitrary categories retain sublabels in both tooltips; other apps follow free space', () => {
  const memory = {id: 'gpu0', total: 8 * GIB, free: GIB, other_apps: GIB,
    uses: {experts: 3 * GIB, other: 2 * GIB, 'Custom <category>': GIB},
    blocks: [{category: 'experts', pool: 'Pool', label: 'Layer 4 (2.00 MiB experts)', bytes: 3 * GIB},
      {category: 'other', pool: 'Drafter', label: 'Expert weights', bytes: GIB},
      {category: 'other', pool: 'Prompt', label: 'Working buffers', bytes: GIB},
      {category: 'Custom <category>', pool: 'New pool', label: '<arbitrary> & labels', bytes: GIB}]};
  const html = draw({available: true, settings: {}, memories: [memory]});
  assert.match(html, /Layer 4 \(2.00 MiB experts\): 3.00 GiB/);
  assert.match(html, /Drafter \/ Expert weights: 1.00 GiB/);
  assert.match(html, /Prompt \/ Working buffers: 1.00 GiB/);
  assert.match(html, /<arbitrary> & labels: 1.00 GiB/);
  const uses = [...html.matchAll(/class="mem-seg" data-use="([^"]*)"/g)].map(m => m[1]);
  assert.deepEqual(uses.slice(-2), ['free', 'other_apps']);
  assert.equal((html.match(/Drafter \/ Expert weights/g) || []).length, 2);
});


test('an adjustable pool does not require any built-in category names', () => {
  const view = {available: true, can_change: true, settings: {}, memories: [{
    id: 'gpu0', total: 4 * GIB, free: GIB, setting: 2 * GIB, pool_offset: GIB,
    smallest: GIB, largest: 3 * GIB, uses: {'Custom category': 3 * GIB}}]};
  const {element} = mount(view, async () => ({ok: true}));
  assert.match(element.innerHTML, /data-divider="gpu0"/);
  assert.match(element.innerHTML, /style="left:75%"/);
});


test('category names that match Object properties are rendered as labels', () => {
  const html = draw({available: true, settings: {}, memories: [{id: 'gpu0', total: GIB,
    uses: {constructor: GIB},
    blocks: [{category: 'constructor', pool: 'Custom pool', label: 'Custom block', bytes: GIB}]}]});
  assert.match(html, /constructor: 1.00 GiB/);
  assert.match(html, /Custom pool \/ Custom block: 1.00 GiB/);
  assert.doesNotMatch(html, /function Object/);
});


test('fixed allocations precede the pool and conversation sublabels precede layers', () => {
  const memory = {id: 'gpu0', total: 8 * GIB, free: GIB, uses: {
    experts: 3 * GIB, pool_free: GIB, conversations: GIB, weights: GIB, other: GIB},
    setting: 5 * GIB, pool_offset: 0, blocks: [
      {category: 'experts', pool: 'Expert and conversation pool', label: 'Layer 2', bytes: 3 * GIB},
      {category: 'conversations', pool: 'Expert and conversation pool', label: 'Conversation Alice', bytes: GIB},
      {category: 'other', pool: 'Expert and conversation pool', label: 'Alignment padding', bytes: 1024}]};
  const {element} = mount({available: true, can_change: true, settings: {}, memories: [memory]}, async () => ({}));
  const html = element.innerHTML;
  assert.deepEqual([...html.matchAll(/class="mem-seg" data-use="([^"]*)"/g)].map(m => m[1]),
    ['other', 'weights', 'conversations', 'experts', 'pool_free', 'free']);
  assert.match(html, /style="left:87.5%"/);
  const tooltip = /data-use="experts" title="([^"]*)"/.exec(html)[1];
  assert.ok(tooltip.indexOf('Conversation Alice') < tooltip.indexOf('Layer 2'));
  assert.doesNotMatch(tooltip, /Expert and conversation pool/);
  assert.match(html, /\nAlignment padding: 0 MiB/);
  assert.doesNotMatch(html, /Expert and conversation pool \/ Alignment padding/);
  assert.match(tooltip, /Experts give the model most of its knowledge and skill/);
});

test('RAM expert copies in any GPU are shaded without changing totals or the reading', () => {
  const block = (group, ids, bytes) => ({category: 'experts', pool: 'Pool', label: group,
    identity_group: group, ids, bytes});
  const memories = [
    {id: 'gpu0', total: 8 * GIB, uses: {experts: GIB}, blocks: [block('layer/1', [0], GIB)]},
    {id: 'gpu1', total: 8 * GIB, uses: {experts: GIB}, blocks: [block('layer/1', [2], GIB)]},
    {id: 'ram', total: 8 * GIB, setting: 4 * GIB, pool_offset: 0,
      uses: {experts: 4 * GIB, weights: GIB}, blocks: [
      block('layer/1', [0, 1, 2], 3 * GIB), block('layer/2', [0], GIB)]}];
  const before = structuredClone(memories), ram = rows(draw({available: true, settings: {}, memories}))[2];
  assert.match(ram, /Experts also in VRAM 2.00 GiB/);
  assert.match(ram, /Experts 2.00 GiB/);
  assert.deepEqual(widths(ram), [12.5, 25, 25]);
  assert.deepEqual(memories, before);
});

test('SSD scale includes its uncached experts and paused budget, not the full model file', () => {
  const [ssd] = rows(draw({available: true, settings: {}, memories: [
    {id: 'ssd', uses: {experts: GIB, paused: GIB}, paused_budget: 7 * GIB}]}));
  assert.equal(barWidth(ssd), 25);
  assert.match(ssd, /Experts not cached 1.00 GiB/);
  assert.match(ssd, /The complete model files remain on disk/);
  const [empty] = rows(draw({available: true, settings: {}, memories: [{id: 'ssd', uses: {}}]}));
  assert.equal(barWidth(empty), 0);
});


test('custom categories preserve explicit pool boundaries when display order changes', () => {
  const {element} = mount({available: true, can_change: true, settings: {}, memories: [{
    id: 'gpu0', total: 8 * GIB, setting: 3 * GIB, pool_offset: GIB,
    uses: {'Z fixed': GIB, 'Custom pool': 3 * GIB, experts: 2 * GIB, weights: 2 * GIB}}]}, async () => ({}));
  const html = element.innerHTML;
  assert.match(html, /style="left:100%"/);
  const categories = [...html.matchAll(/class="mem-seg" data-use="([^"]*)"/g)].map(m => m[1]);
  assert.equal(categories.at(-1), 'Custom pool');
});

test('a pool interval crossing a category keeps the divider and legend totals correct', () => {
  const {element} = mount({available: true, can_change: true, settings: {}, memories: [{
    id: 'ram', total: 8 * GIB, setting: 5 * GIB, pool_offset: 0,
    uses: {experts: 4 * GIB, other: 2 * GIB}, free: 2 * GIB}]}, async () => ({}));
  const html = element.innerHTML;
  assert.match(html, /style="left:75%"/);
  assert.deepEqual(widths(html), [12.5, 50, 12.5, 25]);
  const legend = html.split('<div class="mem-legend">')[1].split('</div>')[0];
  assert.equal((legend.match(/Other 2.00 GiB/g) || []).length, 1);
});


test('paused conversations show last-token ages and the SSD pause folder', () => {
  const now = Math.floor(Date.now() / 1000);
  const html = draw({available: true, settings: {}, pause_directory: 'C:/pause <files>',
    memories: [{id: 'ssd', uses: {paused: GIB}, paused_budget: 8 * GIB}],
    conversations: [{id: 'parked1', place: 'disk', state: 'parked', count: 5, capacity: 5,
      bytes: GIB, used: GIB, last_tokens_at: now - 7200},
      {id: 'parked2', place: 'disk', state: 'parked', count: 5, capacity: 5, bytes: 1, used: 1}]});
  assert.match(html, /parked1: last tokens 2 h ago/);
  assert.match(html, /parked2: last tokens Unknown/);
  assert.match(html, /SSD pause folder: C:\/pause <files>/);
  assert.doesNotMatch(html, /mem-conversations/);
  assert.match(drawHeld({settings: {}, conversations: [{id: "parked1", count: 5, capacity: 5, place: "disk",
    state: "parked", bytes: GIB}]}), /Last tokens<\/th>/);
});

test('RAM uses the same 32 GiB bar scale as GPU and SSD memory', () => {
  const [ram] = rows(draw({available: true, settings: {}, memories: [
    {id: 'ram', total: 8 * GIB, free: 6 * GIB, uses: {experts: 2 * GIB}}]}));
  assert.equal(barWidth(ram), 25);
  assert.deepEqual(widths(ram), [25, 75]);
});

test('the GPU table shows the combined allowance, its total and each conversation limit', () => {
  const conversation = (id, place, count) => ({id, place, count, capacity: 204800, state: 'holding', bytes: MIB});
  const view = {...live, settings: {...live.settings, context: 204800, context_capacity: 204800, tokens_max: 401000},
    conversations: [conversation('slot0', 'vram', 180000), conversation('slot1', 'vram', 190000),
      conversation('parked0', 'ram', 100000)]};
  const html = drawHeld(view);
  assert.match(html, /Shared GPU budget: 401.000 tokens; per-conversation context limit: 204.800 tokens/);
  assert.match(html, />180.000 \/ 204.800<\/td>/);
  assert.match(html, />100.000 \/ 204.800<\/td>/);
  assert.match(html, /GPU total \(shared budget\)[^]*>370.000 \/ 401.000<\/td>/);
  assert.doesNotMatch(html, /class="mem-num mem-over-budget"/);
  view.conversations[1].count = 230000;
  assert.match(drawHeld(view), /class="mem-num mem-over-budget">410.000 \/ 401.000/);
});

test('token settings are staged together and validated against the complete draft', async () => {
  const view = {...live, settings: {...live.settings, context: 204800, context_capacity: 401000, tokens_max: 401000}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  const setting = (key, value) => control('[data-setting]', {dataset: {setting: key}, value});
  fire('change', {target: setting('tokens_max', '200000')});
  assert.deepEqual(calls, []);
  apply(fire);
  assert.match(element.innerHTML, /cannot exceed the combined GPU token budget/);
  assert.deepEqual(calls, []);
  fire('change', {target: setting('context', '180000')});
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{tokens_max: 200000, context: 180000}, 'session']]);
  assert.doesNotMatch(mount(live, onChange).element.innerHTML, /data-setting="context"/);
});

test('the per-conversation widget respects the capacity loaded into the engine', () => {
  const view = {...live, settings: {...live.settings, context: 204800, context_capacity: 204800, tokens_max: 401000}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'context'}, value: '300000'})});
  apply(fire);
  assert.match(element.innerHTML, /Per-conversation token limit: choose a whole number from 1 to 204.800/);
  assert.deepEqual(calls, []);
});

test('the VRAM conversation cap explains fixed overhead and rejects a count below decoding concurrency', async () => {
  const view = {...live, settings: {...live.settings, concurrent: 2, vram_conversations: 2,
    conversation_bytes: 117683200}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  assert.match(element.innerHTML, /Max conversations in VRAM <input/);
  assert.match(element.innerHTML, /data-setting="vram_conversations" value="2" min="1" max="4"/);
  assert.match(element.innerHTML, /Limits active conversations and idle GPU caches together\. Each takes 112 MiB/);
  assert.match(element.innerHTML, /Rewind checkpoints are allocated separately in RAM as needed/);
  assert.match(element.innerHTML, /Each conversation held in VRAM uses 112 MiB of fixed state before its KV cache/);
  const setting = (key, value) => control('[data-setting]', {dataset: {setting: key}, value});
  for (const [key, value] of [['vram_conversations', '1'], ['concurrent', '3']]) {
    fire('change', {target: setting(key, value)});
    apply(fire);
    assert.match(element.innerHTML, /cannot be fewer than conversations at once/);
  }
  assert.deepEqual(calls, []);
  fire('change', {target: setting('vram_conversations', '3')});
  fire('change', {target: setting('concurrent', '2')});
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{vram_conversations: 3, concurrent: 2}, 'session']]);
  assert.doesNotMatch(mount(live, onChange).element.innerHTML, /data-setting="vram_conversations"/);
});

test('idle eviction age is configurable without implying that age alone evicts a conversation', async () => {
  const view = {...live, settings: {...live.settings, conversation_idle_seconds: 66}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  assert.match(element.innerHTML, /data-setting="conversation_idle_seconds" value="66" min="0"/);
  assert.match(element.innerHTML, /when experts or another conversation need space/);
  assert.match(element.innerHTML, /age alone does not evict anything/);
  const target = value => control('[data-setting]', {dataset: {setting: 'conversation_idle_seconds'}, value});
  fire('change', {target: target('-1')});
  apply(fire);
  assert.deepEqual(calls, []);
  fire('change', {target: target('90')});
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{conversation_idle_seconds: 90}, 'session']]);
});


test('decoding while a prompt is read has a share and a turn length to change', async () => {
  const view = {...live, settings: {...live.settings, decode_share_percent: 50, prompt_turn_ms: 0}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  assert.match(element.innerHTML, /data-setting="decode_share_percent" value="50" min="0"/);
  assert.match(element.innerHTML, /data-setting="prompt_turn_ms" value="0" min="0"/);
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'decode_share_percent'}, value: '0'})});
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{decode_share_percent: 0}, 'session']]);
});

test('pending edits survive polling and a live memory resize', async () => {
  const {calls, onChange} = recorder({ok: true, view: live});
  const {element, fire} = mount(live, onChange);
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'rows'}, value: '6'})});
  drawMemoryPanel(element, live, onChange);
  assert.match(element.innerHTML, /data-setting="rows" value="6"/);
  fire('keydown', {target: onDivider('gpu0'), key: 'ArrowRight'});
  fire('keydown', {target: onDivider('gpu0'), key: 'Enter'});
  await settle();
  assert.match(element.innerHTML, /data-setting="rows" value="6"/);
  apply(fire);
  await settle();
  assert.deepEqual(calls[1], [{rows: 6}, 'session']);
});

test('disabling MTP retains the configured maximum lookahead', async () => {
  const {calls, onChange} = recorder(), {fire} = mount({...live,
    settings: {...live.settings, lookahead: 3}}, onChange);
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'drafter'}, value: 'off'})});
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{drafter: false}, 'session']]);
});

test('suffix mode uses engine labels and is submitted as one mutually exclusive mode', async () => {
  const view = {...live, settings: {...live.settings, suffix_draft: 0, lookup_chain: 0, has_drafter: 1},
    details: {drafter_label: 'Use DSpark drafter (1.5 GiB)', drafter_details: 'Discovered drafter: <DSpark>',
      suffix_label: 'Use suffix drafter (0 GiB)', kv_quantization: 'FP8'}};
  const {calls, onChange} = recorder(), {element, fire} = mount(view, onChange);
  assert.match(element.innerHTML, /Use DSpark drafter \(1.5 GiB\)/);
  assert.match(element.innerHTML, /Discovered drafter: <DSpark>/);
  assert.match(element.innerHTML, /KV cache quantization: FP8/);
  fire('change', {target: control('[data-setting]', {dataset: {setting: 'drafter'}, value: 'suffix'})});
  assert.deepEqual(calls, []);
  apply(fire);
  await settle();
  assert.deepEqual(calls, [[{drafter: false, suffix_draft: 3, lookup_chain: 0}, 'session']]);
});

test('allocation tooltips sort every category by label', () => {
  const html = draw({available: true, settings: {}, memories: [{id: 'gpu0', total: GIB,
    uses: {other: GIB}, blocks: [
      {category: 'other', pool: 'Buffers', label: 'Zebra', bytes: GIB / 2},
      {category: 'other', pool: 'Buffers', label: 'Alpha', bytes: GIB / 2}]}]});
  const tooltip = /data-use="other" title="([^"]*)"/.exec(html)[1];
  assert.ok(tooltip.indexOf('Alpha') < tooltip.indexOf('Zebra'));
});


test('context buttons escape engine names and protocol addresses', () => {
  const html = drawConversations([{id: 'slot" onclick="evil()', name: '<script>evil()</script>',
    place: 'vram', state: 'holding', count: 1, capacity: 8, bytes: 1, used: 1}]);
  assert.doesNotMatch(html, /<script>| onclick="evil/);
  assert.match(html, /&lt;script&gt;evil\(\)&lt;\/script&gt;/);
  assert.match(html, /data-context="slot&quot; onclick=&quot;evil\(\)"/);
});
