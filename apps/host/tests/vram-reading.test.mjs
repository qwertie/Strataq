import test from 'node:test';
import assert from 'node:assert/strict';
import {getVramReading} from '../../../strata_server/web/vram-reading.js';

const GIB = 2 ** 30;

test('hardware readings take priority, including zero usage', () => {
  const reading = getVramReading({gpu_mem_used: 0, gpu_mem_total: 8 * GIB},
    [{id: 'gpu0', total: 64 * GIB, free: GIB}]);
  assert.equal(reading.used, 0);
  assert.equal(reading.total, 8 * GIB);
  assert.equal(reading.source, 'telemetry');
});

test('Windows AMD falls back to total minus free from the Memory report', () => {
  const reading = getVramReading({}, [{id: 'gpu0', total: 64 * GIB, free: 24 * GIB,
    in_share: 32 * GIB, borrowed: GIB, uses: {weights: 4 * GIB, experts: 29 * GIB}},
    {id: 'ram', total: 64 * GIB, free: 16 * GIB}]);
  assert.equal(reading.used, 40 * GIB);
  assert.equal(reading.total, 64 * GIB);
  assert.equal(reading.source, 'memory');
});

test('engine-only readings count Strata allocations and keep an unknown capacity unknown', () => {
  const reading = getVramReading({}, [{id: 'gpu0', uses: {weights: GIB, experts: 3 * GIB}}]);
  assert.equal(reading.used, 4 * GIB);
  assert.equal(reading.total, null);
  assert.equal(reading.source, 'engine');
});

test('partitioned cards do not multiply physical GPU capacity or whole-device use', () => {
  const reading = getVramReading({}, [{id: 'gpu0', total: 8 * GIB, free: 2 * GIB},
    {id: 'gpu0.1', total: 8 * GIB, free: 2 * GIB}, {id: 'gpu1', total: 16 * GIB, free: 8 * GIB}]);
  assert.equal(reading.used, 14 * GIB);
  assert.equal(reading.total, 24 * GIB);
  assert.equal(reading.gpus.length, 2);
});

test('engine-only partitions add allocations while sharing capacity', () => {
  const reading = getVramReading({}, [{id: 'gpu0', total: 8 * GIB, uses: {experts: GIB}},
    {id: 'gpu0.1', total: 8 * GIB, uses: {experts: 2 * GIB}}]);
  assert.equal(reading.used, 3 * GIB);
  assert.equal(reading.total, 8 * GIB);
});

test('missing readings remain unavailable instead of claiming zero use', () => {
  assert.equal(getVramReading({}, []).used, null);
  assert.equal(getVramReading({}, [{id: 'gpu0', total: 8 * GIB}]).used, null);
});
