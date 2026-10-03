import test from 'node:test';
import assert from 'node:assert/strict';
import { createModelCache, cachedFetch, attachCachePort, createRemoteModelCache, runtimePaths } from './model-cache.js';

function cacheFixture({ maximum = 1024, failWrite = 0 } = {}) {
  const records = new Map();
  const cacheKey = key => typeof key === 'string' ? key : key.url;
  let writes = 0, chunkReads = 0;
  const native = {
    async put(key, response) {
      const body = await response.arrayBuffer();
      if (body.byteLength > maximum || ++writes === failWrite) throw new DOMException('Test quota exceeded', 'QuotaExceededError');
      records.set(cacheKey(key), { body, headers: [...response.headers], status: response.status });
    },
    async match(key) {
      if (cacheKey(key).includes('/chunk/')) chunkReads++;
      const record = records.get(cacheKey(key));
      return record ? new Response(record.body.slice(0), record) : undefined;
    },
    async keys() { return [...records.keys()].map(key => new Request(key)); },
    async delete(key) { return records.delete(cacheKey(key)); },
  };
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', { configurable: true, value: { open: async () => native, keys: async () => [] } });
  return { records, native, get chunkReads() { return chunkReads; }, resetReads() { chunkReads = 0; },
    restore() { if (previous) Object.defineProperty(globalThis, 'caches', previous); else delete globalThis.caches; } };
}

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

test('Cache API fallback splits files that exceed a single-entry limit and validates lazy reads', async () => {
  const fixture = cacheFixture();
  const bytes = Uint8Array.from({ length: 5 * 1024 + 73 }, (_, index) => index % 251);
  const url = 'https://example.invalid/resolve/pinned/large-model.onnx';
  const events = [];
  try {
    const cache = createModelCache({ name: 'unit-chunks', chunkBytes: 1024, onEvent: event => events.push(event) });
    const stored = await cache.put(url, new Response(bytes));
    assert.equal(stored.stored, true);
    assert.ok(events.some(event => event.type === 'stored' && event.verified));
    assert.equal(fixture.records.size, 7); // Six chunks plus one committed manifest.
    assert.ok([...fixture.records.values()].every(record => record.body.byteLength <= 1024));
    fixture.resetReads();
    const fresh = createModelCache({ name: 'unit-chunks', chunkBytes: 1024 });
    const found = await fresh.match(url);
    assert.equal(found.headers.get('content-length'), String(bytes.byteLength));
    assert.equal(fixture.chunkReads, 0); // Header-only probes do not read model data.
    assert.deepEqual(new Uint8Array(await found.arrayBuffer()), bytes);
    assert.equal(fixture.chunkReads, 6);
    await fresh.delete(url);
    assert.equal(fixture.records.size, 0);
  } finally { fixture.restore(); }
});

test('failed chunk writes never publish a manifest or a stored event', async () => {
  const fixture = cacheFixture({ failWrite: 2 });
  const events = [], url = 'https://example.invalid/failed-large-model';
  try {
    const cache = createModelCache({ name: 'unit-quota', chunkBytes: 1024, onEvent: event => events.push(event) });
    const result = await cache.put(url, new Response(new Uint8Array(7000)));
    assert.equal(result.stored, false);
    assert.equal(events.some(event => event.type === 'stored'), false);
    assert.ok(events.some(event => event.type === 'unavailable' && event.reason === 'quota'));
    assert.equal(fixture.records.size, 0);
    assert.equal(await createModelCache({ name: 'unit-quota' }).match(url), undefined);
  } finally { fixture.restore(); }
});

test('an interrupted chunked writer leaves no readable model and abort clears its staged chunks', async () => {
  const fixture = cacheFixture(), url = 'https://example.invalid/partial-large-model';
  try {
    const cache = createModelCache({ name: 'unit-partial', chunkBytes: 1024 });
    const writer = await cache.openWriter(url, { status: 200, headers: [] });
    await writer.write(new Uint8Array(3000));
    assert.ok(fixture.records.size > 0);
    assert.equal(await cache.match(url), undefined);
    await writer.abort();
    assert.equal(fixture.records.size, 0);
  } finally { fixture.restore(); }
});

test('missing committed chunks are a miss and do not report a disk hit', async () => {
  const fixture = cacheFixture(), url = 'https://example.invalid/missing-chunk';
  try {
    await createModelCache({ name: 'unit-missing', chunkBytes: 1024 }).put(url, new Response(new Uint8Array(7000)));
    fixture.records.delete([...fixture.records.keys()].find(key => key.includes('/chunk/')));
    const events = [];
    const found = await createModelCache({ name: 'unit-missing', onEvent: event => events.push(event) }).match(url);
    assert.equal(found, undefined);
    assert.equal(events.some(event => event.type === 'hit'), false);
    assert.ok(events.some(event => event.reason === 'verification'));
    assert.equal(fixture.records.size, 0);
  } finally { fixture.restore(); }
});

test('replacing a committed file reclaims only its previous chunk generation', async () => {
  const fixture = cacheFixture(), url = 'https://example.invalid/replaced-file';
  try {
    const cache = createModelCache({ name: 'unit-replace', chunkBytes: 1024 });
    await cache.put(url, new Response(new Uint8Array(7000)));
    await cache.put(url, new Response(new Uint8Array(1500)));
    assert.equal(fixture.records.size, 3);
    assert.equal((await (await cache.match(url)).arrayBuffer()).byteLength, 1500);
  } finally { fixture.restore(); }
});

test('page-owned storage bridge streams a complete model with verified saved status', async () => {
  const fixture = cacheFixture({ maximum: 5 * 1024 * 1024 });
  const channel = new MessageChannel(), events = [];
  const disconnect = attachCachePort(channel.port1, { cache: createModelCache({ name: 'unit-bridge', forceLocal: true, chunkBytes: 1024 }) });
  try {
    const remote = createRemoteModelCache(channel.port2, { onEvent: event => events.push(event) });
    const bytes = Uint8Array.from({ length: 5000 }, (_, index) => index % 253), url = 'https://example.invalid/bridge-model';
    assert.equal((await remote.put(url, new Response(bytes))).stored, true);
    const response = await remote.match(url);
    assert.deepEqual(new Uint8Array(await response.arrayBuffer()), bytes);
    assert.ok(events.some(event => event.type === 'stored' && event.verified));
    await remote.delete(url);
    assert.equal(await remote.match(url), undefined);
  } finally { disconnect(); channel.port2.close(); fixture.restore(); }
});

test('stale abandoned staging is reclaimed while recent and published chunks survive', async () => {
  const fixture = cacheFixture();
  try {
    const cache = createModelCache({ name: 'unit-stale', chunkBytes: 1024 });
    const abandoned = await cache.openWriter('https://example.invalid/abandoned', { status: 200, headers: [] });
    await abandoned.write(new Uint8Array(1024));
    const abandonedStageKey = [...fixture.records.keys()].find(key => key.includes('/stage/'));
    const stageRecord = fixture.records.get(abandonedStageKey);
    const stage = JSON.parse(new TextDecoder().decode(stageRecord.body));
    stage.updated = Date.now() - 2 * 24 * 60 * 60 * 1000;
    stageRecord.body = new TextEncoder().encode(JSON.stringify(stage)).buffer;
    const abandonedChunk = [...fixture.records.keys()].find(key => key.includes('/chunk/'));
    const active = await cache.openWriter('https://example.invalid/active', { status: 200, headers: [] });
    await active.write(new Uint8Array(1024));
    const activeChunk = [...fixture.records.keys()].filter(key => key.includes('/chunk/')).find(key => key !== abandonedChunk);
    await cache.put('https://example.invalid/published', new Response(new Uint8Array(2000)));
    const fresh = createModelCache({ name: 'unit-stale', chunkBytes: 1024 });
    const next = await fresh.openWriter('https://example.invalid/next', { status: 200, headers: [] });
    assert.equal(fixture.records.has(abandonedChunk), false);
    assert.equal(fixture.records.has(abandonedStageKey), false);
    assert.equal(fixture.records.has(activeChunk), true);
    assert.equal((await (await fresh.match('https://example.invalid/published')).arrayBuffer()).byteLength, 2000);
    await active.abort(); await next.abort(); await abandoned.abort();
  } finally { fixture.restore(); }
});

test('a storage timeout never announces a large model as saved', async () => {
  const fixture = cacheFixture({ maximum: 5 * 1024 * 1024 });
  const nativePut = fixture.native.put;
  fixture.native.put = (key, response) => String(key).includes('/chunk/') ? new Promise(() => {}) : nativePut(key, response);
  const events = [];
  try {
    const cache = createModelCache({ name: 'unit-timeout', storageTimeoutMs: 5, onEvent: event => events.push(event) });
    const result = await cache.put('https://example.invalid/timed-out-model', new Response(new Uint8Array(9 * 1024 * 1024)));
    assert.equal(result.stored, false);
    assert.equal(result.reason, 'timeout');
    assert.equal(events.some(event => event.type === 'stored'), false);
    assert.ok(events.some(event => event.type === 'unavailable' && event.reason === 'timeout'));
  } finally { fixture.restore(); }
});

test('disconnect while acquiring a writer aborts the acquired writer instead of orphaning it', async () => {
  const channel = new MessageChannel();
  let resolveWriter, started;
  const began = new Promise(resolve => { started = resolve; });
  let aborted = 0;
  const disconnect = attachCachePort(channel.port1, { cache: { openWriter() { started(); return new Promise(resolve => { resolveWriter = resolve; }); } } });
  try {
    channel.port2.postMessage({ pocketCache: true, id: 1, op: 'write-start', url: 'https://example.invalid/model', meta: {} });
    await began; disconnect();
    resolveWriter({ abort: async () => { aborted++; } });
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(aborted, 1);
  } finally { disconnect(); channel.port2.close(); }
});

test('disconnect while acquiring a cached response cancels its unconsumed body', async () => {
  const channel = new MessageChannel();
  let resolveMatch, started, cancelled = 0;
  const began = new Promise(resolve => { started = resolve; });
  const disconnect = attachCachePort(channel.port1, { cache: { match() { started(); return new Promise(resolve => { resolveMatch = resolve; }); } } });
  try {
    channel.port2.postMessage({ pocketCache: true, id: 1, op: 'match', url: 'https://example.invalid/model' });
    await began; disconnect();
    resolveMatch(new Response(new ReadableStream({ cancel() { cancelled++; } }, { highWaterMark: 0 })));
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(cancelled, 1);
  } finally { disconnect(); channel.port2.close(); }
});

test('older Safari and explicitly identified older iOS shells use a matching plain CPU runtime pair', () => {
  const base = 'https://example.invalid/project/runtime/ort/';
  const cases = [
    { vendor: 'Apple Computer, Inc.', userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 Version/18.6 Mobile/15E148 Safari/604.1' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.5 Safari/605.1.15' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 CriOS/140.0 Mobile/15E148 Safari/604.1' },
    { vendor: '', userAgent: 'Mozilla/5.0 (iPad; CPU OS 17_6 like Mac OS X) AppleWebKit/605.1.15 FxiOS/140.0 Mobile/15E148 Safari/605.1.15' },
  ];
  for (const browser of cases) {
    const paths = runtimePaths(base, browser);
    assert.equal(new URL(paths.wasm).pathname, '/project/runtime/ort/ort-wasm-simd-threaded.wasm');
    assert.equal(new URL(paths.mjs).pathname, '/project/runtime/ort/ort-wasm-simd-threaded.mjs');
    assert.equal(new URL(paths.wasm).searchParams.get('v'), new URL(paths.mjs).searchParams.get('v'));
  }
});

test('modern, GPU-enabled, and unknown browsers retain asyncify without relying on JSPI support', () => {
  const base = 'https://example.invalid/project/runtime/transformers/';
  const cases = [
    { vendor: 'Apple Computer, Inc.', userAgent: 'AppleWebKit/605.1.15 Version/26.0 Safari/605.1.15' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'AppleWebKit/605.1.15 Version/18.6 Safari/605.1.15', gpu: {} },
    { vendor: 'Google Inc.', userAgent: 'Mozilla/5.0 AppleWebKit/537.36 Chrome/140.0 Safari/537.36' },
    { vendor: '', userAgent: 'Mozilla/5.0 Firefox/140.0' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 CriOS/140.0 Safari/604.1' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 CriOS/140.0 Safari/604.1' },
    { vendor: 'Apple Computer, Inc.', userAgent: 'Unknown Safari version' },
    undefined,
  ];
  for (const browser of cases) for (const [extension, url] of Object.entries(runtimePaths(base, browser))) {
    assert.equal(new URL(url).pathname, `/project/runtime/transformers/ort-wasm-simd-threaded.asyncify.${extension}`);
  }
});

test('a denied IndexedDB read still reuses the complete Cache API chunk copy', async () => {
  const fixture = cacheFixture(), originalIDB = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
  const url = 'https://example.invalid/read-denied-model';
  const bytes = Uint8Array.from({ length: 5000 }, (_, index) => index % 251);
  try {
    const stored = await createModelCache({ name: 'unit-read-denied', chunkBytes: 1024 }).put(url, new Response(bytes));
    assert.equal(stored.stored, true);
    Object.defineProperty(globalThis, 'indexedDB', { configurable: true, value: {
      open() {
        const request = {};
        queueMicrotask(() => {
          request.result = { close() {}, transaction() { throw new DOMException('Storage denied', 'SecurityError'); } };
          request.onsuccess();
        });
        return request;
      },
    } });
    const found = await createModelCache({ name: 'unit-read-denied', chunkBytes: 1024 }).match(url);
    assert.ok(found);
    assert.equal(found.headers.get('x-pocket-cache'), 'disk');
    assert.deepEqual(new Uint8Array(await found.arrayBuffer()), bytes);
  } finally {
    if (originalIDB) Object.defineProperty(globalThis, 'indexedDB', originalIDB); else delete globalThis.indexedDB;
    fixture.restore();
  }
});
