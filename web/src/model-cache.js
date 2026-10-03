// Persistent storage uses the original, versioned URL, never a temporary CDN
// redirect. Both browser storage APIs are available in modern Safari workers.
export const MODEL_CACHE_NAME = 'pocket-composer-models-v1';
const MEMORY_LIMIT = 96 * 1024 * 1024;
const MEMORY_FILE_LIMIT = 32 * 1024 * 1024;

function storageDeadline(operation, milliseconds = 5000) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Browser storage did not respond.')), milliseconds);
  });
  return Promise.race([operation, timeout]).finally(() => clearTimeout(timer));
}

function abortable(operation, signal) {
  if (!signal) return operation;
  signal.throwIfAborted();
  let cancel;
  const aborted = new Promise((_, reject) => { cancel = () => reject(signal.reason || new DOMException('Cancelled', 'AbortError')); signal.addEventListener('abort', cancel, { once: true }); });
  return Promise.race([operation, aborted]).finally(() => signal.removeEventListener('abort', cancel));
}

function keyFor(request) {
  const value = typeof request === 'string' || request instanceof URL ? String(request) : request.url;
  const url = new URL(value, globalThis.location?.href || 'https://pocket.invalid/');
  if (!['https:', 'http:'].includes(url.protocol)) throw new Error('Unsupported model cache URL.');
  url.hash = '';
  return url.href;
}

function marked(response, source) {
  const headers = new Headers(response.headers);
  headers.set('x-pocket-cache', source);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function transactionResult(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = resolve;
    transaction.onabort = transaction.onerror = () => reject(transaction.error || new Error('Browser storage write failed.'));
  });
}

export function createModelCache({ name = MODEL_CACHE_NAME, onEvent = () => {} } = {}) {
  let nativePromise, databasePromise, legacyPromise;
  let warned = false, memoryBytes = 0;
  const memory = new Map();
  const notify = event => { try { onEvent(event); } catch { /* Status must not prevent inference. */ } };
  const warning = () => {
    if (!warned) {
      warned = true;
      notify({ type: 'unavailable', message: 'Browser storage is unavailable or full. Models can be reused in this tab, but may download again after reloading.' });
    }
  };
  async function nativeCache() {
    nativePromise ||= storageDeadline(Promise.resolve().then(() => globalThis.caches?.open(name))).catch(() => undefined);
    return nativePromise;
  }
  async function legacyCaches() {
    // Earlier app versions used Transformers.js' default cache. Reuse those
    // pinned files in place, without another download or duplicating big LLMs.
    legacyPromise ||= storageDeadline(Promise.resolve().then(async () => {
      const names = await globalThis.caches?.keys() || [];
      return names.includes('transformers-cache') && name !== 'transformers-cache'
        ? [await caches.open('transformers-cache')] : [];
    })).catch(() => []);
    return legacyPromise;
  }
  async function database() {
    databasePromise ||= new Promise(resolve => {
      if (!globalThis.indexedDB) { resolve(undefined); return; }
      let request;
      try { request = indexedDB.open(name, 1); } catch { resolve(undefined); return; }
      let expired = false;
      const timeout = setTimeout(() => { expired = true; resolve(undefined); }, 5000);
      request.onupgradeneeded = () => { if (!request.result.objectStoreNames.contains('files')) request.result.createObjectStore('files'); };
      request.onsuccess = () => {
        clearTimeout(timeout);
        const db = request.result;
        if (expired) { db.close(); return; }
        db.onversionchange = () => db.close();
        resolve(db);
      };
      request.onerror = request.onblocked = () => { expired = true; clearTimeout(timeout); resolve(undefined); };
    });
    return databasePromise;
  }
  async function readDatabase(key) {
    const db = await database();
    if (!db) return undefined;
    return storageDeadline(new Promise((resolve, reject) => {
      const request = db.transaction('files').objectStore('files').get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    }));
  }
  function fromRecord(record, source) {
    const response = new Response(record.blob, { status: record.status, statusText: record.statusText, headers: record.headers });
    return marked(response, source);
  }
  function remember(key, record) {
    if (record.blob.size > MEMORY_FILE_LIMIT) return; // Never retain another full title model.
    if (memory.has(key)) memoryBytes -= memory.get(key).blob.size;
    memory.delete(key);
    while (memoryBytes + record.blob.size > MEMORY_LIMIT && memory.size) {
      const oldest = memory.keys().next().value;
      memoryBytes -= memory.get(oldest).blob.size; memory.delete(oldest);
    }
    memory.set(key, record); memoryBytes += record.blob.size;
  }
  return {
    async match(request) {
      let key;
      try { key = keyFor(request); } catch { return undefined; }
      try {
        const response = await storageDeadline(Promise.resolve((await nativeCache())?.match(key)));
        if (response) { notify({ type: 'hit', url: key, source: 'disk' }); return marked(response, 'disk'); }
      } catch { /* Some private browsing modes deny CacheStorage operations. */ }
      for (const cache of await legacyCaches()) {
        try {
          const response = await storageDeadline(cache.match(key));
          if (response) { notify({ type: 'hit', url: key, source: 'disk' }); return marked(response, 'disk'); }
        } catch { /* Legacy storage is optional. */ }
      }
      try {
        const record = await readDatabase(key);
        if (record) { notify({ type: 'hit', url: key, source: 'disk' }); return fromRecord(record, 'disk'); }
      } catch { /* Try the bounded session fallback. */ }
      if (memory.has(key)) { notify({ type: 'hit', url: key, source: 'memory' }); return fromRecord(memory.get(key), 'memory'); }
      return undefined;
    },
    async put(request, response) {
      if (response.status !== 200 || response.type === 'opaque') return;
      const key = keyFor(request);
      // A Blob can be reused by both APIs without an extra giant JS ArrayBuffer.
      // Only completed bodies reach either store; aborted downloads are not hits.
      let blob;
      try { blob = await response.blob(); }
      catch { warning(); return; } // Storage limits can also prevent Blob creation.
      const headers = new Headers(response.headers);
      headers.delete('x-pocket-cache'); headers.delete('content-encoding');
      headers.set('content-length', String(blob.size));
      const record = { blob, headers: [...headers], status: response.status, statusText: response.statusText };
      try {
        const cache = await nativeCache();
        if (cache) {
          await storageDeadline(cache.put(key, new Response(blob, { status: 200, headers })), 120000);
          notify({ type: 'stored', url: key, bytes: blob.size, source: 'disk' }); return;
        }
      } catch { /* Safari or storage policy can reject a Cache API write. */ }
      try {
        const db = await database();
        if (db) {
          const tx = db.transaction('files', 'readwrite');
          const done = transactionResult(tx);
          tx.objectStore('files').put(record, key);
          await storageDeadline(done, 120000);
          notify({ type: 'stored', url: key, bytes: blob.size, source: 'disk' }); return;
        }
      } catch { /* Insufficient quota must not discard the downloaded model. */ }
      remember(key, record); warning();
    },
    async delete(request) {
      const key = keyFor(request);
      if (memory.has(key)) { memoryBytes -= memory.get(key).blob.size; memory.delete(key); }
      let deleted = false;
      try { deleted = Boolean(await storageDeadline(Promise.resolve((await nativeCache())?.delete(key)))); } catch { /* Continue. */ }
      for (const cache of await legacyCaches()) {
        try { deleted = Boolean(await storageDeadline(cache.delete(key))) || deleted; } catch { /* Continue. */ }
      }
      try {
        const db = await database();
        if (db) {
          const tx = db.transaction('files', 'readwrite'); const done = transactionResult(tx);
          tx.objectStore('files').delete(key); await storageDeadline(done); deleted = true;
        }
      } catch { /* Cache recovery remains best effort. */ }
      return deleted;
    },
  };
}

const defaultCache = createModelCache();

export async function primeBrowserCache() {
  // Open from the page before dedicated workers. This also avoids a WebKit
  // ephemeral-context issue where an unowned worker cache vanishes on exit.
  await defaultCache.match(new URL('__cache_probe__', globalThis.location?.href || 'https://pocket.invalid/'));
}

export async function cachedFetch(url, init = {}, { cache = defaultCache, onEvent = () => {} } = {}) {
  init.signal?.throwIfAborted();
  const cached = await abortable(cache.match(url), init.signal);
  init.signal?.throwIfAborted();
  if (cached) { onEvent({ type: 'hit', url: String(url), source: cached.headers.get('x-pocket-cache') }); return cached; }
  const response = await fetch(url, { credentials: 'omit', cache: 'force-cache', ...init });
  if (response.status !== 200) return response;
  const headers = new Headers(response.headers);
  const total = Number(headers.get('content-length')) || 0;
  onEvent({ type: 'download', url: String(url), loaded: 0, total });
  // Blob parts keep large files out of a second concatenated JS byte buffer.
  const parts = []; let loaded = 0, lastProgress = 0;
  if (response.body) {
    const reader = response.body.getReader();
    try {
      while (true) {
        init.signal?.throwIfAborted();
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(new Blob([value])); loaded += value.byteLength;
        if (performance.now() - lastProgress > 150) {
          onEvent({ type: 'download', url: String(url), loaded, total }); lastProgress = performance.now();
        }
      }
    } finally { reader.releaseLock(); }
  } else { const blob = await response.blob(); parts.push(blob); loaded = blob.size; }
  init.signal?.throwIfAborted();
  const blob = new Blob(parts, { type: headers.get('content-type') || 'application/octet-stream' });
  headers.delete('content-encoding'); headers.set('content-length', String(blob.size));
  await abortable(cache.put(url, new Response(blob, { headers })), init.signal);
  init.signal?.throwIfAborted();
  onEvent({ type: 'download', url: String(url), loaded, total: total || loaded, complete: true });
  return marked(new Response(blob, { headers }), 'network');
}

// Mutable runtime filenames get a key derived from the locked build inputs.
export function runtimePaths(base) {
  const revision = typeof __RUNTIME_REVISION__ === 'undefined' ? 'development' : __RUNTIME_REVISION__;
  return Object.fromEntries(['wasm', 'mjs'].map(extension => {
    const url = new URL(`ort-wasm-simd-threaded.asyncify.${extension}`, base);
    url.searchParams.set('v', revision); return [extension, url.href];
  }));
}

export async function loadCachedRuntime(base, { cache = defaultCache, signal, onEvent } = {}) {
  const paths = runtimePaths(base);
  const responses = await Promise.all(Object.values(paths).map(url => cachedFetch(url, { signal }, { cache, onEvent })));
  for (const response of responses) if (!response.ok) throw new Error(`Browser runtime download failed (HTTP ${response.status}).`);
  const [wasm, mjs] = await Promise.all([responses[0].arrayBuffer(), responses[1].blob()]);
  return { wasmBinary: new Uint8Array(wasm), wasmPaths: { wasm: paths.wasm, mjs: URL.createObjectURL(mjs) } };
}
