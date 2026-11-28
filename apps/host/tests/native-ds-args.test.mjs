/**
 * The options the app hands to the DeepSeek engine (strata-ds41):
 *  - #4358 #4359 #4505 #4507: a model with pictures off had its `--vision off` pair cut in half (`--eos 1 off --dspark auto`);
 *  - the resource preference writes no Qwen-engine options into a DeepSeek model's file.
 * Nothing here starts an engine: Control is a fake that records the run config the host saved for the start (a scratch profile,
 * never the real one).
 */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { withoutVisionFlag } from '../server/native-vision-flag.mjs';

const DS_ARGS = ['--pack', '/p/ds', '--max-context', '32768', '--eos', '1', '--vision', 'off', '--dspark', 'auto', '--dspark-conf', '0.65', '--prefill-chunk', '1024'];

function fixture(config = {}, id = 'deepseek-v4.1-flash-ist-das-2.25bpw') {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-args-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, `strata-${id}.json`);
  writeFileSync(path, JSON.stringify({
    exe: 'C:\\Users\\x\\AppData\\Local\\Strata\\engine-dsv41\\0.1.52-windows-x86_64-cuda-cuda13\\strata-dsv41.exe', arch: 'deepseek41',
    tokenizer: join(install, 'tok'), args: DS_ARGS, model_name: 'DeepSeek V4.1', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.7',
    async installedModels() { return { install_dir: install, models: [{ id, name: 'DeepSeek V4.1', source_config: path, component_status: 'present', context_tokens: 32768 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.52', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 24 }], directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, id, path, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
/** Every word of the argument list that is not an option name or the value right after one: a stray word means a flag lost its name. */
function strayWords(args, valued) {
  const stray = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('-')) { if (valued.has(args[i]) && i + 1 < args.length && !args[i + 1].startsWith('--')) i++; continue; }
    stray.push(args[i]);
  }
  return stray;
}

test('withoutVisionFlag takes the Qwen engine\'s stand-alone --vision out, and a picture word an old file left after it', () => {
  assert.deepEqual(withoutVisionFlag(['--max-context', '4096', '--vision', '--kv', 'int8']), ['--max-context', '4096', '--kv', 'int8']);
  assert.deepEqual(withoutVisionFlag(['--a', '1', '--vision']), ['--a', '1']);
  assert.deepEqual(withoutVisionFlag(['--vision', 'off', '--eos', '1']), ['--eos', '1']);
  assert.deepEqual(withoutVisionFlag(['--vision', 'gpu', '--x']), ['--x']);
});

test('#4358 #4359 #4505 #4507: a DeepSeek load with pictures off leaves the --vision pair whole and no stray word', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    // what the Models screen sends for a model it does not list pictures for: vision "no"
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const args = f.saved[0].args;
    assert.equal(args[args.indexOf('--vision') + 1], 'off', args.join(' '));
    assert.equal(args[args.indexOf('--eos') + 1], '1', 'the word after --eos is its value, not a left-over of --vision');
    assert.equal(args[args.indexOf('--eos') + 2], '--vision');
    assert.deepEqual(strayWords(args, new Set(['--pack', '--max-context', '--eos', '--vision', '--dspark', '--dspark-conf', '--prefill-chunk'])), []);
    assert.deepEqual(args, DS_ARGS, 'nothing was added or removed');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a DeepSeek model with pictures on: Read pictures off in the app turns the pair to `--vision off` (the app lists the switch since 0.2.9)', async () => {
  const f = fixture({ args: DS_ARGS.map(a => (a === 'off' ? 'on' : a)) });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    const args = f.saved[0].args;
    assert.equal(args[args.indexOf('--vision') + 1], 'off');
    assert.equal(args[args.indexOf('--eos') + 1], '1');
    assert.deepEqual(strayWords(args, new Set(['--pack', '--max-context', '--eos', '--vision', '--dspark', '--dspark-conf', '--prefill-chunk'])), []);
  } finally { await f.native.close(); f.cleanup(); }
});

/** A pack folder as pack_gguf.py leaves it: ds4.txt names the tower's geometry and the picture token when the picture file was there. */
function packWith(root, tower) {
  const pack = join(root, tower ? 'pack-pictures' : 'pack-text');
  mkdirSync(pack, { recursive: true });
  writeFileSync(join(pack, 'ds4.txt'), ['n_layers 43', ...(tower ? ['image_token_id 128815', 'vision.num_hidden_layers 32'] : []), ''].join('\n'));
  return pack;
}

test('a DeepSeek model whose pack carries the picture tower is prepared for pictures; a text-only pack is not, and says so', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-pics-'));
  const seen = [];
  try {
    for (const [tower, vision] of [[true, 'on'], [false, 'off']]) {
      const f = fixture({ args: ['--pack', packWith(root, tower), '--vision', vision, '--eos', '1'] });      // (one at a time: the fixture sets STRATA_INSTALL_DIR)
      try {
        await f.native.refresh(true);
        const m = f.native.state().installed.models.find(x => x.model === f.id);
        seen.push([m.visionPrepared, m.images]);
      } finally { await f.native.close(); f.cleanup(); }
    }
    assert.deepEqual(seen, [[true, 'yes'], [false, 'off']], 'a text-only pack: the Models screen offers the missing shared file, nothing is fetched on its own');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a DeepSeek pack with the tower and `--vision off` in its arguments: Read pictures on turns it on, and only where the tower is', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-pics-'));
  const got = [];
  try {
    for (const tower of [true, false]) {
      const f = fixture({ args: ['--pack', packWith(root, tower), '--vision', 'off', '--eos', '1'] });
      try {
        await f.native.refresh(true);
        const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'yes' }).id);
        assert.equal(job.state, 'done', job.error);
        got.push(f.saved[0].args[f.saved[0].args.indexOf('--vision') + 1]);
      } finally { await f.native.close(); f.cleanup(); }
    }
    assert.deepEqual(got, ['on', 'off'], 'no tower in the pack: the engine is not told to read pictures');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('the resource preference no longer writes Qwen-engine options into a DeepSeek model\'s file when it is tuned', async () => {
  const f = fixture({ args: DS_ARGS });
  try {
    await f.native.refresh(true);
    const before = JSON.parse(readFileSync(f.path, 'utf8'));
    const job = await finish(f.native, f.native.operation('tune', { model: f.id, profile: 'work', calibrate: false }).id);
    void job;                                                          // the load and benchmark after it need a real engine; the file is what matters
    const after = JSON.parse(readFileSync(f.path, 'utf8'));
    assert.ok(!(after.args ?? []).includes('--pool-workers'), (after.args ?? []).join(' '));
    assert.ok(!(after.args ?? []).includes('--vram-reserve-mib'), (after.args ?? []).join(' '));
    assert.deepEqual(after.args.slice(0, before.args.length), before.args);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a Qwen model with pictures off still loses its stand-alone --vision (nothing else changed)', async () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-ds-args-q-'));
  const install = join(root, 'install');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--pack', '/p', '--max-context', '4096', '--vision', '--kv', 'int8'], model_name: 'Qwen', port: 8080 }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.7',
    async installedModels() { return { install_dir: install, models: [{ id: 'q2_0', name: 'Qwen', source_config: path, component_status: 'present', context_tokens: 4096 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.2.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 12 }], directory: join(root, 'data'), spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  try {
    await native.refresh(true);
    const job = await finish(native, native.operation('load', { model: 'q2_0', vision: 'no' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(saved[0].args, ['--pack', '/p', '--max-context', '4096', '--kv', 'int8']);
  } finally { await native.close(); rmSync(root, { recursive: true, force: true }); }
});

test('a DeepSeek load with Read pictures on does not ask for a separate reader (#4772)', async () => {
  const f = fixture({ args: DS_ARGS.map(a => (a === 'off' ? 'on' : a)) });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: f.id, vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.saved[0].args[f.saved[0].args.indexOf('--vision') + 1], 'on');
  } finally { await f.native.close(); f.cleanup(); }
});
