import test from 'node:test';
import assert from 'node:assert/strict';
import { request, ApiError } from '../src/api.js';
import { chartGeometry, closestIndex, visiblePrices } from '../src/chart-utils.js';

test('API failures expose a useful error rather than returning fabricated data', async () => {
  await assert.rejects(request('/health', {
    fetcher: async () => new Response(JSON.stringify({ detail: 'Snapshot integrity check failed.' }), { status: 503 }),
  }), error => error instanceof ApiError && error.message === 'Snapshot integrity check failed.');
  await assert.rejects(request('/health', { fetcher: async () => { throw new TypeError('Network down'); } }),
    /Unable to reach the local data service/);
});
test('cancelled requests remain cancellations and do not become user-facing API failures', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(request('/health', {
    signal: controller.signal,
    fetcher: async (url, options) => { options.signal.throwIfAborted(); return new Response('{}'); },
  }), error => error.name === 'AbortError');
});
test('slow requests time out with a retryable message', async () => {
  await assert.rejects(request('/health', { timeout: 10,
    fetcher: (url, { signal }) => new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))),
  }), /took too long/);
});
test('chart ranges, constant prices, and pointer boundaries stay finite and select real sessions', () => {
  const prices = Array.from({ length: 300 }, (_, index) => ({ date: String(index), close: 100, volume: index }));
  assert.equal(visiblePrices(prices, '1M').length, 22);
  assert.equal(visiblePrices(prices, '6M').length, 126);
  assert.equal(visiblePrices(prices, 'ALL').length, 300);
  const geometry = chartGeometry(prices, 390);
  assert.ok(!geometry.line.includes('NaN') && !geometry.line.includes('Infinity'));
  assert.equal(closestIndex(-100, geometry, prices.length), 0);
  assert.equal(closestIndex(1000, geometry, prices.length), 299);
  assert.equal(closestIndex(geometry.x(150), geometry, prices.length), 150);
});
