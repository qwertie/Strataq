import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {chromium} from 'playwright-core';

const executablePath = [process.env.CHROME,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/usr/bin/google-chrome']
  .find(path => path && existsSync(path));
const read = name => readFile(new URL(`../../../strata_server/web/${name}`, import.meta.url), 'utf8');

test('context selection retains scroll position and shows snippets and individual limits',
  {skip: !executablePath}, async () => {
    const browser = await chromium.launch({executablePath, headless: true});
    let release = () => {};
    try {
      const page = await browser.newPage({viewport: {width: 1280, height: 720}});
      let failContext = false;

      const waiting = new Promise(resolve => { release = resolve; });
      const entries = [1, 2].map(n => ({id: `conversation${n}`, name: `Conversation ${n}`, state: 'holding',
        where: 'vram', count: 100, capacity: 204800, bytes: 1024, used: 1024,
        preview: `User prompt ${n}`, kind: 'session'}));
      await page.route('http://strata.test/**', async route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api-monitor')
          return route.fulfill({contentType: 'text/html', body: await read('monitor.html')});
        if (url.pathname.startsWith('/web/')) {
          const name = url.pathname.slice(5);
          return route.fulfill({contentType: name.endsWith('.js') ? 'text/javascript' : 'text/css',
            body: await read(name)});
        }
        let body = {};
        if (url.pathname === '/api/requests') body = {requests: []};
        if (url.pathname === '/api/context') {
          const id = url.searchParams.get('id');
          if (id && failContext)
            return route.fulfill({status: 503,
              json: {error: {message: 'the engine did not answer GET_CONTEXT_WINDOWS'}}});
          if (id === 'conversation2') await waiting;
          body = id ? {id, tokens: [1], pieces: [`${id}\n` + 'context text\n'.repeat(1000)]} :
            {available: true, at: 1, entries};
        }
        if (url.pathname === '/v1/memory') body = {settings: {tokens_max: 401111}, conversations: []};
        return route.fulfill({json: body});
      });
      await page.goto('http://strata.test/api-monitor');
      await page.waitForFunction(() => document.querySelector('#ctx-view').textContent.startsWith('conversation1'));
      assert.equal(await page.locator('.mem-context-preview').first().textContent(), 'User prompt 1');
      assert.match(await page.locator('tbody .mem-num').first().textContent(), /100 \/ 204.800/);
      await page.locator('[data-context="conversation2"]').scrollIntoViewIfNeeded();
      const before = await page.evaluate(() => window.scrollY);
      await page.locator('[data-context="conversation2"]').click();
      assert.equal(await page.evaluate(() => window.scrollY), before);
      assert.match(await page.locator('#ctx-view').textContent(), /^conversation1/);
      release();
      await page.waitForFunction(() => document.querySelector('#ctx-view').textContent.startsWith('conversation2'));
      assert.equal(await page.evaluate(() => window.scrollY), before);
      failContext = true;
      await page.locator('#ctx-refresh').click();
      await page.waitForFunction(() => !document.querySelector('#ctx-refresh').disabled);
      assert.match(await page.locator('#ctx-view').textContent(), /^conversation2/);
      assert.match(await page.locator('#ctx-sum').textContent(), /refresh failed/);
      assert.match(await page.locator('#ctx-note').textContent(), /GET_CONTEXT_WINDOWS.*previous context snapshot/);
      await page.setContent('<div id="panel" style="width:1210px"></div>');
      await page.addStyleTag({content: await read('tokens.css') + await read('memory-panel.css')});
      await page.evaluate(async () => {
        const {drawMemoryPanel} = await import('/web/memory-panel.js');
        const view = {available: true, can_change: true, settings: {rows: 8, concurrent: 1, lookahead: 3,
          tokens_min: 0, tokens_max: 1000, conversation_idle_seconds: 60}, memories: [
          {id: 'gpu0', total: 8192, setting: 4096, uses: {experts: 4096}, pool_offset: 0}]};
        const panel = document.getElementById('panel');
        drawMemoryPanel(panel, view, async () => ({ok: true}));
        const input = panel.querySelector('[data-setting="rows"]');
        input.focus();
        input.value = '6';
        const controls = panel.querySelector('.mem-controls'), divider = panel.querySelector('[data-divider]');
        drawMemoryPanel(panel, {...view, memories: view.memories.map(m => ({...m, allocated_pct: 75}))},
          async () => ({ok: true}));
        if (document.activeElement !== input || input.value !== '6' ||
            controls !== panel.querySelector('.mem-controls') || divider !== panel.querySelector('[data-divider]'))
          throw new Error('Polling replaced a control, slider or focused edit');
      });
      const labels = await page.locator('.mem-controls label').evaluateAll(elements =>
        elements.map(element => ({width: element.getBoundingClientRect().width,
          top: element.getBoundingClientRect().top})));
      assert.ok(labels.every(label => label.width === 229));
      assert.ok(labels.slice(0, 5).every(label => label.top === labels[0].top));
    } finally {
      release();
      await browser.close();
    }
  });
