import { chromium, expect, firefox, test, webkit } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Test the production cache module in real dedicated workers. The fixture is
// deliberately not HTTP-cacheable, and redirects across origins like the Hub's
// signed asset URLs. No real model downloads are needed to check persistence.
const bytes = Buffer.from(Array.from({ length: 256 * 1024 }, (_, index) => (index * 31 + 17) % 256));
const expectedDigest = createHash('sha256').update(bytes).digest('hex');
const counts = new Map();
let pageServer;
let assetServer;
let fixtureOrigin;
let assetOrigin;

const workerCode = `
const fault = new URL(location.href).searchParams.get('fault');
if (fault === 'cache-unavailable') {
  Object.defineProperty(globalThis, 'caches', { value: undefined, configurable: true });
}
if (fault === 'storage-full') {
  Object.defineProperty(globalThis, 'caches', {
    value: { open: async () => { throw new DOMException('Test quota reached', 'QuotaExceededError'); } },
    configurable: true,
  });
  Object.defineProperty(globalThis, 'indexedDB', { value: undefined, configurable: true });
}
const modulePromise = import('/src/model-cache.js');
self.onmessage = async ({ data }) => {
  const controller = new AbortController();
  let timer;
  if (data.abortAfter) timer = setTimeout(() => controller.abort(), data.abortAfter);
  try {
    const { cachedFetch, createModelCache } = await modulePromise;
    const events = [];
    const onEvent = event => events.push(event);
    const cache = createModelCache({ onEvent });
    const response = await cachedFetch(data.url, { signal: controller.signal }, { cache, onEvent });
    const buffer = await response.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))]
      .map(value => value.toString(16).padStart(2, '0')).join('');
    self.postMessage({ ok: response.ok, status: response.status, length: buffer.byteLength, digest, cacheSource: response.headers.get('x-pocket-cache'), events });
  } catch (error) {
    self.postMessage({ error: error.message, name: error.name });
  } finally { clearTimeout(timer); }
};
`;

function increment(key, endpoint) {
  const counter = counts.get(key) || { redirect: 0, binary: 0 };
  counter[endpoint] += 1;
  counts.set(key, counter);
}

function listen(server) {
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(`http://127.0.0.1:${server.address().port}`)));
}

test.beforeAll(async () => {
  assetServer = createServer(async (request, response) => {
    const url = new URL(request.url, 'http://fixture.invalid');
    if (url.pathname !== '/binary') { response.writeHead(404); response.end(); return; }
    increment(url.searchParams.get('key'), 'binary');
    response.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': bytes.length,
      'cache-control': 'no-store',
      'access-control-allow-origin': '*',
    });
    if (!url.searchParams.has('slow')) { response.end(bytes); return; }
    response.write(bytes.subarray(0, 8192));
    for (let offset = 8192; offset < bytes.length && !response.destroyed; offset += 8192) {
      await new Promise(resolve => setTimeout(resolve, 20));
      if (!response.destroyed) response.write(bytes.subarray(offset, offset + 8192));
    }
    response.end();
  });
  assetOrigin = await listen(assetServer);
  pageServer = createServer(async (request, response) => {
    const url = new URL(request.url, 'http://fixture.invalid');
    if (url.pathname === '/') {
      response.writeHead(200, { 'content-type': 'text/html', 'cache-control': 'no-store' });
      response.end(`<!doctype html><title>Persistent model cache fixture</title>
        <script type="module">window.cacheReady = import('/src/model-cache.js').then(module => module.primeBrowserCache());</script>`);
      return;
    }
    if (url.pathname === '/cache-worker.js') {
      response.writeHead(200, { 'content-type': 'text/javascript', 'cache-control': 'no-store' });
      response.end(workerCode);
      return;
    }
    if (url.pathname.startsWith('/src/') && /^\/src\/[a-z-]+\.js$/.test(url.pathname)) {
      try {
        const body = await readFile(new URL(`..${url.pathname}`, import.meta.url));
        response.writeHead(200, { 'content-type': 'text/javascript', 'cache-control': 'no-store' });
        response.end(body);
      } catch { response.writeHead(404); response.end(); }
      return;
    }
    if (url.pathname === '/model.onnx') {
      increment(url.searchParams.get('key'), 'redirect');
      response.writeHead(302, {
        location: `${assetOrigin}/binary${url.search}`,
        'cache-control': 'no-store',
        'access-control-allow-origin': '*',
      });
      response.end();
      return;
    }
    response.writeHead(404); response.end();
  });
  fixtureOrigin = await listen(pageServer);
});

test.afterAll(async () => {
  await Promise.all([pageServer, assetServer].filter(Boolean).map(server => new Promise(resolve => server.close(resolve))));
});

async function fetchInFreshWorker(page, { key, fault = '', abortAfter = 0, slow = false }) {
  return page.evaluate(async options => {
    await window.cacheReady;
    const worker = new Worker(`/cache-worker.js?fault=${options.fault}`, { type: 'module' });
    try {
      return await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Fixture worker timed out')), 15000);
        worker.onmessage = ({ data }) => { clearTimeout(timer); resolve(data); };
        worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
        const url = new URL('/model.onnx', location.origin);
        url.searchParams.set('key', options.key);
        if (options.slow) url.searchParams.set('slow', '1');
        worker.postMessage({ url: url.href, abortAfter: options.abortAfter });
      });
    } finally { worker.terminate(); }
  }, { key, fault, abortAfter, slow });
}

function expectFullModel(result) {
  expect(result).toMatchObject({ ok: true, status: 200, length: bytes.length, digest: expectedDigest });
}

for (const browserName of ['chromium', 'firefox', 'webkit']) {
  // A normal browser profile exercises disk persistence. WebKit's disposable
  // automation contexts use ephemeral worker caches unlike an ordinary Safari
  // session, so this also avoids mistaking their storage policy for an app bug.
  const browserType = { chromium, firefox, webkit }[browserName];
  const engineTest = test.extend({
    cacheSession: async ({}, use) => {
      const profile = await mkdtemp(join(tmpdir(), `pocket-cache-${browserName}-`));
      const launch = () => browserType.launchPersistentContext(profile, {
        headless: true,
        viewport: { width: 390, height: 844 },
        ...(browserName === 'webkit' ? { isMobile: true, hasTouch: true } : {}),
      });
      let context = await launch();
      try {
        await use({
          page: context.pages()[0] || await context.newPage(),
          async restart() {
            await context.close();
            context = await launch();
            return context.pages()[0] || await context.newPage();
          },
        });
      } finally {
        await context.close();
        await rm(profile, { recursive: true, force: true });
      }
    },
    cachePage: async ({ cacheSession }, use) => use(cacheSession.page),
  });
  engineTest.describe(`${browserName} persistent model cache`, () => {

    engineTest('reuses redirected model bytes after workers, reload, and browser restart', async ({ cacheSession }, testInfo) => {
      let page = cacheSession.page;
      const key = `${browserName}-persistent`;
      await page.goto(fixtureOrigin);
      const first = await fetchInFreshWorker(page, { key });
      expectFullModel(first);
      const samePage = await fetchInFreshWorker(page, { key });
      expectFullModel(samePage);
      await page.reload();
      const second = await fetchInFreshWorker(page, { key });
      expectFullModel(second);
      page = await cacheSession.restart();
      await page.goto(fixtureOrigin);
      const restarted = await fetchInFreshWorker(page, { key });
      expectFullModel(restarted);
      await testInfo.attach('cache-events.json', { body: JSON.stringify({ first, samePage, second, restarted }), contentType: 'application/json' });
      expect(counts.get(key)).toEqual({ redirect: 1, binary: 1 });
      await testInfo.attach('network-counts.json', { body: JSON.stringify(counts.get(key)), contentType: 'application/json' });
    });

    engineTest('persists in IndexedDB when Cache Storage is unavailable', async ({ cachePage: page }) => {
      const key = `${browserName}-indexeddb`;
      await page.goto(fixtureOrigin);
      expectFullModel(await fetchInFreshWorker(page, { key, fault: 'cache-unavailable' }));
      await page.reload();
      expectFullModel(await fetchInFreshWorker(page, { key, fault: 'cache-unavailable' }));
      expect(counts.get(key)).toEqual({ redirect: 1, binary: 1 });
    });

    engineTest('reuses existing Transformers model files without downloading or duplicating them', async ({ cachePage: page }, testInfo) => {
      const key = `${browserName}-legacy-transformers`;
      await page.goto(fixtureOrigin);
      await page.evaluate(async ({ key, length }) => {
        await window.cacheReady;
        const url = new URL('/model.onnx', location.origin);
        url.searchParams.set('key', key);
        const contents = new Uint8Array(length);
        for (let index = 0; index < length; index += 1) contents[index] = (index * 31 + 17) % 256;
        const legacyCache = await caches.open('transformers-cache');
        await legacyCache.put(url.href, new Response(contents, { headers: { 'content-type': 'application/octet-stream' } }));
      }, { key, length: bytes.length });
      const first = await fetchInFreshWorker(page, { key });
      expectFullModel(first);
      expect(first.cacheSource).toBe('disk');
      await page.reload();
      const second = await fetchInFreshWorker(page, { key });
      expectFullModel(second);
      expect(second.cacheSource).toBe('disk');
      expect(counts.has(key)).toBe(false);
      const duplicated = await page.evaluate(async key => {
        const url = new URL('/model.onnx', location.origin);
        url.searchParams.set('key', key);
        return Boolean(await (await caches.open('pocket-composer-models-v1')).match(url.href));
      }, key);
      expect(duplicated).toBe(false);
      await testInfo.attach('legacy-cache.json', {
        body: JSON.stringify({ first, second, networkRequests: 0, duplicated }),
        contentType: 'application/json',
      });
    });

    engineTest('keeps downloads usable when persistent storage is full or blocked', async ({ cachePage: page }) => {
      const key = `${browserName}-storage-full`;
      await page.goto(fixtureOrigin);
      expectFullModel(await fetchInFreshWorker(page, { key, fault: 'storage-full' }));
      await page.reload();
      expectFullModel(await fetchInFreshWorker(page, { key, fault: 'storage-full' }));
      expect(counts.get(key)).toEqual({ redirect: 2, binary: 2 });
    });

    engineTest('does not reuse an aborted partial model download', async ({ cachePage: page }) => {
      const key = `${browserName}-aborted`;
      await page.goto(fixtureOrigin);
      const aborted = await fetchInFreshWorker(page, { key, slow: true, abortAfter: 100 });
      expect(aborted.name).toBe('AbortError');
      expectFullModel(await fetchInFreshWorker(page, { key, slow: true }));
      await page.reload();
      expectFullModel(await fetchInFreshWorker(page, { key, slow: true }));
      expect(counts.get(key)).toEqual({ redirect: 2, binary: 2 });
    });

    if (browserName === 'webkit') {
      engineTest('keeps a primed page cache available in an ephemeral WebKit session', async ({}, testInfo) => {
        const browser = await webkit.launch({ headless: true });
        try {
          const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
          const page = await context.newPage();
          const key = 'webkit-ephemeral-primed';
          await page.goto(fixtureOrigin);
          expectFullModel(await fetchInFreshWorker(page, { key }));
          expectFullModel(await fetchInFreshWorker(page, { key }));
          const beforeReload = { ...counts.get(key) };
          expect(beforeReload).toEqual({ redirect: 1, binary: 1 });
          await page.reload();
          expectFullModel(await fetchInFreshWorker(page, { key }));
          // Ephemeral sessions may discard their origin's storage when a page
          // exits. A reload must still work even when the browser evicts bytes.
          const afterReload = counts.get(key);
          expect(afterReload.redirect).toBeGreaterThanOrEqual(1);
          expect(afterReload.redirect).toBeLessThanOrEqual(2);
          expect(afterReload.binary).toBe(afterReload.redirect);
          await testInfo.attach('network-counts.json', { body: JSON.stringify({ beforeReload, afterReload }), contentType: 'application/json' });
        } finally { await browser.close(); }
      });
    }
  });
}
