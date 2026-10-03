import test from 'node:test';
import assert from 'node:assert/strict';
import { createModelCache, cachedFetch } from './model-cache.js';

test('small downloads remain reusable when browser storage is unavailable', async () => {
  const events = [];
  const cache = createModelCache({ name: 'unit-memory-fallback', onEvent: event => events.push(event) });
  const url = 'https://example.invalid/resolve/revision/model.onnx';
  await cache.put(url, new Response('complete model'));
  const loaded = await cache.match(url);
  assert.equal(await loaded.text(), 'complete model');
  assert.equal(loaded.headers.get('x-pocket-cache'), 'memory');
  assert.ok(events.some(event => event.type === 'unavailable'));
  await cache.delete(url);
  assert.equal(await cache.match(url), undefined);
});

test('cancellation interrupts a stalled storage lookup before any download', async () => {
  const controller = new AbortController();
  const cache = { match: () => new Promise(() => {}), put: async () => {} };
  const request = cachedFetch('https://example.invalid/model.onnx', { signal: controller.signal }, { cache });
  controller.abort();
  await assert.rejects(request, error => error.name === 'AbortError');
});

test('unsuccessful responses never become reusable models', async () => {
  const cache = createModelCache({ name: 'unit-invalid-response' });
  const url = 'https://example.invalid/missing.onnx';
  await cache.put(url, new Response('not a model', { status: 404 }));
  assert.equal(await cache.match(url), undefined);
});
