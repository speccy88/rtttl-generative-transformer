// Cache keys are pinned source URLs, never temporary CDN redirect URLs. Large
// files are committed as small ArrayBuffers: no single giant Cache/IDB Blob.
export const MODEL_CACHE_NAME = 'pocket-composer-models-v1';
export const MODEL_CHUNK_BYTES = 4 * 1024 * 1024;
const MEMORY_LIMIT = 96 * 1024 * 1024;
const MEMORY_FILE_LIMIT = 32 * 1024 * 1024;
const SMALL_FILE_LIMIT = 8 * 1024 * 1024;
const STORAGE_TIMEOUT = 15000;
let pageCache;
const workerEventListeners = new Set();

function storageDeadline(operation, milliseconds = STORAGE_TIMEOUT, onTimeout = () => {}) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => { onTimeout(); reject(new Error('Browser storage did not respond in time.')); }, milliseconds);
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

function metadata(response) {
  const headers = new Headers(response.headers);
  headers.delete('x-pocket-cache'); headers.delete('content-encoding');
  return { headers: [...headers], status: response.status, statusText: response.statusText };
}

function storageReason(error) {
  if (error?.name === 'QuotaExceededError') return { reason: 'quota', message: 'Browser storage is full. The model was not saved and may need to download again.' };
  if (/respond|time.*out/i.test(error?.message || '')) return { reason: 'timeout', message: 'Browser storage did not respond. The model was not confirmed saved and may need to download again.' };
  if (/verif|missing|incomplete|corrupt/i.test(error?.message || '')) return { reason: 'verification', message: 'Browser storage could not verify the saved file. The model may need to download again.' };
  return { reason: 'blocked', message: 'Browser storage is unavailable or blocked. The model was not saved and may need to download again.' };
}

function transactionDone(tx, timeout) {
  return storageDeadline(new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onabort = tx.onerror = () => reject(tx.error || new Error('Browser storage transaction failed.'));
  }), timeout, () => { try { tx.abort(); } catch { /* Already completed. */ } });
}

export function createModelCache({ name = MODEL_CACHE_NAME, onEvent = () => {}, forceLocal = false,
  chunkBytes = MODEL_CHUNK_BYTES, storageTimeoutMs = STORAGE_TIMEOUT } = {}) {
  if (!Number.isInteger(chunkBytes) || chunkBytes < 1024 || chunkBytes > MODEL_CHUNK_BYTES) throw new Error('Invalid model cache chunk size.');
  let nativePromise, databasePromise, legacyPromise;
  const sweptStores = new Set();
  let warned = false, memoryBytes = 0;
  const memory = new Map();
  const notify = event => { try { onEvent(event); } catch { /* Status must not prevent inference. */ } };
  if (!forceLocal && typeof document === 'undefined') workerEventListeners.add(notify);
  const warning = error => {
    if (!warned) { warned = true; notify({ type: 'unavailable', ...storageReason(error) }); }
  };
  const timeout = operation => storageDeadline(operation, storageTimeoutMs);
  async function nativeCache() {
    nativePromise ||= timeout(Promise.resolve().then(() => globalThis.caches?.open(name))).catch(() => undefined);
    return nativePromise;
  }
  async function legacyCaches() {
    legacyPromise ||= timeout(Promise.resolve().then(async () => {
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
      try { request = indexedDB.open(name, 2); } catch { resolve(undefined); return; }
      let expired = false;
      const timer = setTimeout(() => { expired = true; resolve(undefined); }, storageTimeoutMs);
      request.onupgradeneeded = () => {
        for (const store of ['files', 'chunks', 'manifests']) if (!request.result.objectStoreNames.contains(store)) request.result.createObjectStore(store);
      };
      request.onsuccess = () => {
        clearTimeout(timer);
        const db = request.result;
        if (expired) { db.close(); return; }
        db.onversionchange = () => { db.close(); databasePromise = undefined; };
        resolve(db);
      };
      request.onerror = request.onblocked = () => { expired = true; clearTimeout(timer); resolve(undefined); };
    });
    return databasePromise;
  }
  async function readRecord(db, store, key) {
    return timeout(new Promise((resolve, reject) => {
      const request = db.transaction(store).objectStore(store).get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    }));
  }
  async function allRecords(db, store) {
    return timeout(new Promise((resolve, reject) => {
      const request = db.transaction(store).objectStore(store).getAll();
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    }));
  }
  async function mutate(db, store, action) {
    const tx = db.transaction(store, 'readwrite');
    const done = transactionDone(tx, storageTimeoutMs);
    try { action(tx.objectStore(store)); } catch (error) { try { tx.abort(); } catch { /* Already aborted. */ } await done.catch(() => {}); throw error; }
    await done;
  }
  const chunkKey = (id, index) => `${id}:${String(index).padStart(8, '0')}`;
  async function chunkKeys(db, id) {
    return timeout(new Promise((resolve, reject) => {
      const range = IDBKeyRange.bound(`${id}:`, `${id}:\uffff`);
      const request = db.transaction('chunks').objectStore('chunks').getAllKeys(range);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    }));
  }
  async function removeChunks(db, id) {
    await mutate(db, 'chunks', store => store.delete(IDBKeyRange.bound(`${id}:`, `${id}:\uffff`)));
  }
  const cacheKey = (kind, key) => new URL(`/__pocket_model_cache_v2__/${encodeURIComponent(name)}/${kind}/${encodeURIComponent(key)}`, globalThis.location?.href || 'https://pocket.invalid/').href;
  function chunkStore(db, native) {
    if (db) return {
      kind: 'indexeddb-chunks',
      read: key => readRecord(db, 'chunks', key),
      write: (key, value) => mutate(db, 'chunks', store => store.put(value, key)),
      keys: id => chunkKeys(db, id), remove: id => removeChunks(db, id),
      manifest: key => readRecord(db, 'manifests', key),
      publish: (key, value) => mutate(db, 'manifests', store => store.put(value, key)),
      unpublish: key => mutate(db, 'manifests', store => store.delete(key)),
      records: () => allRecords(db, 'manifests'),
      touch: id => mutate(db, 'manifests', store => store.put({ kind: 'staging', id, updated: Date.now() }, `__stage__:${id}`)),
      stage: id => readRecord(db, 'manifests', `__stage__:${id}`),
      unstage: id => mutate(db, 'manifests', store => store.delete(`__stage__:${id}`)),
    };
    if (!native) return null;
    const keys = async id => (await timeout(native.keys())).map(request => request.url)
      .filter(url => url.startsWith(cacheKey('chunk', `${id}:`)))
      .map(url => decodeURIComponent(url.split('/').at(-1))).sort();
    return {
      kind: 'cache-api-chunks',
      async read(key) { const response = await timeout(native.match(cacheKey('chunk', key))); return response ? timeout(response.arrayBuffer()) : undefined; },
      write: (key, value) => timeout(native.put(cacheKey('chunk', key), new Response(value, { headers: { 'content-length': String(value.byteLength) } }))),
      keys,
      async remove(id) { for (const key of await keys(id)) await timeout(native.delete(cacheKey('chunk', key))); },
      async manifest(key) { const response = await timeout(native.match(cacheKey('manifest', key))); return response ? timeout(response.json()) : undefined; },
      publish: (key, value) => timeout(native.put(cacheKey('manifest', key), new Response(JSON.stringify(value), { headers: { 'content-type': 'application/json' } }))),
      unpublish: key => timeout(native.delete(cacheKey('manifest', key))),
      async records() {
        const records = [];
        for (const request of await timeout(native.keys())) {
          if (!request.url.startsWith(cacheKey('manifest', '')) && !request.url.startsWith(cacheKey('stage', ''))) continue;
          const response = await timeout(native.match(request));
          if (response) records.push(await timeout(response.json()));
        }
        return records;
      },
      touch: id => timeout(native.put(cacheKey('stage', id), new Response(JSON.stringify({ kind: 'staging', id, updated: Date.now() })))),
      async stage(id) { const response = await timeout(native.match(cacheKey('stage', id))); return response ? timeout(response.json()) : undefined; },
      unstage: id => timeout(native.delete(cacheKey('stage', id))),
    };
  }
  async function sweepStaleWrites(store) {
    if (!store || sweptStores.has(store.kind)) return;
    sweptStores.add(store.kind);
    const records = await store.records();
    const published = new Set(records.filter(record => record.version === 2).map(record => record.id));
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    for (const record of records) {
      if (record.kind !== 'staging' || record.updated >= cutoff) continue;
      // Refresh the lease immediately before deletion. Recent writes in other
      // tabs and every published generation remain untouched.
      const current = await store.stage(record.id);
      if (!current || current.updated >= cutoff) continue;
      if (!published.has(record.id)) await store.remove(record.id);
      await store.unstage(record.id);
    }
  }
  async function validManifest(store, record) {
    if (!record || record.version !== 2 || typeof record.id !== 'string' || !Array.isArray(record.sizes)
      || !record.sizes.every(size => Number.isInteger(size) && size > 0 && size <= MODEL_CHUNK_BYTES)
      || record.sizes.reduce((sum, size) => sum + size, 0) !== record.size) return false;
    const keys = await store.keys(record.id);
    return keys.length === record.sizes.length && keys.every((key, index) => key === chunkKey(record.id, index));
  }
  function chunkResponse(store, record) {
    let index = 0;
    const stream = new ReadableStream({
      async pull(controller) {
        if (index === record.sizes.length) { controller.close(); return; }
        try {
          const data = await store.read(chunkKey(record.id, index));
          if (!(data instanceof ArrayBuffer) || data.byteLength !== record.sizes[index]) throw new Error('A cached model chunk is missing or incomplete.');
          index++; controller.enqueue(new Uint8Array(data));
        } catch (error) { warning(error); controller.error(error); }
      },
    }, { highWaterMark: 0 });
    return marked(new Response(stream, record), 'disk');
  }
  function fromRecord(record, source) { return marked(new Response(record.blob, record), source); }
  function remember(key, record) {
    if (record.blob.size > MEMORY_FILE_LIMIT) return;
    if (memory.has(key)) memoryBytes -= memory.get(key).blob.size;
    memory.delete(key);
    while (memoryBytes + record.blob.size > MEMORY_LIMIT && memory.size) {
      const oldest = memory.keys().next().value;
      memoryBytes -= memory.get(oldest).blob.size; memory.delete(oldest);
    }
    memory.set(key, record); memoryBytes += record.blob.size;
  }
  async function openWriter(request, meta) {
    const key = keyFor(request);
    const db = await database();
    let store = chunkStore(db, db ? null : await nativeCache());
    const id = `${Date.now()}-${globalThis.crypto?.randomUUID?.() || Math.random()}`;
    let buffer = new Uint8Array(chunkBytes), used = 0, size = 0, closed = false, aborted = false;
    const sizes = [], memoryParts = [];
    let failure = store ? null : new Error('Persistent browser storage is unavailable.');
    if (store) {
      await sweepStaleWrites(store).catch(() => {});
      try { await store.touch(id); }
      catch (error) {
        const fallback = store.kind === 'indexeddb-chunks' ? chunkStore(null, await nativeCache()) : null;
        if (fallback) { store = fallback; try { await store.touch(id); } catch (fallbackError) { failure = fallbackError; } }
        else failure = error;
      }
    }
    const checkCancelled = () => { if (aborted) throw new DOMException('Model cache write cancelled.', 'AbortError'); };
    async function persist(data) {
      checkCancelled();
      if (size <= MEMORY_FILE_LIMIT) memoryParts.push(data.slice()); else memoryParts.length = 0;
      if (failure) return;
      const index = sizes.length;
      try {
        await store.touch(id);
        const value = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
        try { await store.write(chunkKey(id, index), value); }
        catch (error) {
          // Some private modes allow opening IndexedDB but reject its writes.
          // Before any chunks have committed, safely switch the entire writer.
          if (index || store.kind !== 'indexeddb-chunks') throw error;
          const fallback = chunkStore(null, await nativeCache());
          if (!fallback) throw error;
          await store.remove(id).catch(() => {});
          store = fallback; await store.write(chunkKey(id, index), value);
        }
        checkCancelled();
        const checked = await store.read(chunkKey(id, index));
        checkCancelled();
        if (!(checked instanceof ArrayBuffer) || checked.byteLength !== data.byteLength) throw new Error('Saved model chunk verification failed.');
        sizes.push(data.byteLength);
      } catch (error) { failure = error; await store.remove(id).catch(() => {}); await store.unstage(id).catch(() => {}); }
    }
    return {
      async write(value) {
        if (closed) throw new Error('Model cache writer is closed.');
        const data = value instanceof Uint8Array ? value : new Uint8Array(value);
        let offset = 0;
        while (offset < data.byteLength) {
          checkCancelled();
          const count = Math.min(data.byteLength - offset, buffer.byteLength - used);
          buffer.set(data.subarray(offset, offset + count), used);
          used += count; size += count; offset += count;
          if (used === buffer.byteLength) { await persist(buffer); buffer = new Uint8Array(chunkBytes); used = 0; }
        }
      },
      async close() {
        if (closed) throw new Error('Model cache writer is closed.');
        closed = true;
        if (used) await persist(buffer.subarray(0, used));
        checkCancelled();
        buffer = null;
        const headers = new Headers(meta.headers); headers.set('content-length', String(size));
        const record = { ...meta, headers: [...headers], version: 2, id, sizes, size };
        if (!failure) {
          try {
            const previous = await store.manifest(key);
            // Only this last transaction publishes the file; interrupted writes
            // have no readable manifest and are never reported as stored.
            await store.publish(key, record);
            checkCancelled();
            const committed = await store.manifest(key);
            checkCancelled();
            if (committed?.id !== id || !await validManifest(store, committed)) throw new Error('Saved model manifest verification failed.');
            if (previous?.id && previous.id !== id) await store.remove(previous.id).catch(() => {});
            await store.unstage(id).catch(() => {});
            checkCancelled();
            memoryParts.length = 0;
            notify({ type: 'stored', url: key, bytes: size, source: 'disk', storage: store.kind, verified: true });
            return { stored: true, bytes: size, source: 'disk', verified: true };
          } catch (error) {
            failure = error;
            // Delete only this attempted publication, never legacy cache data.
            const current = await store.manifest(key).catch(() => undefined);
            if (current?.id === id) await store.unpublish(key).catch(() => {});
            await store.remove(id).catch(() => {});
            await store.unstage(id).catch(() => {});
          }
        }
        checkCancelled();
        if (size <= MEMORY_FILE_LIMIT) {
          const blob = new Blob(memoryParts, { type: headers.get('content-type') || '' });
          const fallback = { ...meta, headers: [...headers], blob };
          if (size <= SMALL_FILE_LIMIT) {
            try {
              const native = await nativeCache();
              if (native) {
                await timeout(native.put(key, new Response(blob, fallback)));
                const verify = await timeout(native.match(key));
                if (!verify || (await timeout(verify.arrayBuffer())).byteLength !== size) throw new Error('Saved browser cache verification failed.');
                checkCancelled();
                notify({ type: 'stored', url: key, bytes: size, source: 'disk', storage: 'cache-api', verified: true });
                return { stored: true, bytes: size, source: 'disk', verified: true };
              }
            } catch (error) { failure = error; }
          }
          checkCancelled();
          remember(key, fallback);
        }
        warning(failure);
        return { stored: false, bytes: size, source: size <= MEMORY_FILE_LIMIT ? 'memory' : null, reason: storageReason(failure).reason };
      },
      async abort() {
        aborted = true; closed = true; buffer = null; memoryParts.length = 0;
        if (store) { await store.remove(id).catch(() => {}); await store.unstage(id).catch(() => {}); }
      },
    };
  }
  const cache = {
    async prime() { await Promise.all([database(), nativeCache()]); },
    openWriter,
    async match(request) {
      if (!forceLocal && pageCache) {
        try { return await pageCache.match(request); } catch (error) { warning(error); return undefined; }
      }
      let key;
      try { key = keyFor(request); } catch { return undefined; }
      try {
        const db = await database();
        const stores = [chunkStore(db, null), chunkStore(null, await nativeCache())].filter(Boolean);
        for (const store of stores) {
          try {
            const record = await store.manifest(key);
            if (record && await validManifest(store, record)) { notify({ type: 'hit', url: key, source: 'disk', storage: store.kind, verified: true }); return chunkResponse(store, record); }
            if (record) { warning(new Error('Saved model is incomplete.')); await store.unpublish(key); await store.remove(record.id); }
          } catch (error) {
            // A readable Cache API copy remains useful when an existing IDB
            // connection becomes denied, stale, or temporarily unavailable.
            warning(error);
          }
        }
      } catch (error) { warning(error); }
      for (const candidate of [await nativeCache(), ...await legacyCaches()]) {
        try {
          const response = candidate && await timeout(candidate.match(key));
          if (response) { notify({ type: 'hit', url: key, source: 'disk' }); return marked(response, 'disk'); }
        } catch { /* Legacy storage is optional. */ }
      }
      try {
        const db = await database();
        const record = db && await readRecord(db, 'files', key);
        if (record) { notify({ type: 'hit', url: key, source: 'disk' }); return fromRecord(record, 'disk'); }
      } catch { /* Try the bounded session fallback. */ }
      if (memory.has(key)) { notify({ type: 'hit', url: key, source: 'memory' }); return fromRecord(memory.get(key), 'memory'); }
      return undefined;
    },
    async put(request, response, progress) {
      if (!forceLocal && pageCache) {
        try { return await pageCache.put(request, response, progress); }
        catch (error) { warning(error); return { stored: false, reason: storageReason(error).reason }; }
      }
      if (response.status !== 200 || response.type === 'opaque') return { stored: false };
      const writer = await openWriter(request, metadata(response));
      const reader = response.body?.getReader();
      let loaded = 0; const total = Number(response.headers.get('content-length')) || 0;
      try {
        if (reader) {
          while (true) {
            const { done, value } = await reader.read(); if (done) break;
            await writer.write(value); loaded += value.byteLength;
            progress?.({ loaded, total, progress: total ? loaded / total * 100 : 0 });
          }
        } else await writer.write(new Uint8Array(await response.arrayBuffer()));
        return await writer.close();
      } catch (error) { await writer.abort(); warning(error); throw error; }
      finally { reader?.releaseLock(); }
    },
    async delete(request) {
      if (!forceLocal && pageCache) return pageCache.delete(request);
      const key = keyFor(request);
      if (memory.has(key)) { memoryBytes -= memory.get(key).blob.size; memory.delete(key); }
      let deleted = false;
      for (const candidate of [await nativeCache(), ...await legacyCaches()]) {
        try { deleted = Boolean(candidate && await timeout(candidate.delete(key))) || deleted; } catch { /* Continue. */ }
      }
      try {
        const db = await database();
        const nativeStore = chunkStore(null, await nativeCache());
        if (nativeStore) {
          const record = await nativeStore.manifest(key); await nativeStore.unpublish(key);
          if (record) { await nativeStore.remove(record.id); deleted = true; }
        }
        if (db) {
          const record = await readRecord(db, 'manifests', key);
          await mutate(db, 'manifests', store => store.delete(key));
          if (record) await removeChunks(db, record.id);
          await mutate(db, 'files', store => store.delete(key)); deleted = true;
        }
      } catch { /* Cache recovery remains best effort. */ }
      return deleted;
    },
  };
  return cache;
}

// MessagePort RPC keeps the storage connection owned by the foreground page.
// Transfers and acknowledgements are bounded to one chunk at a time.
export function attachCachePort(port, { cache = createModelCache({ forceLocal: true }), onEvent = () => {} } = {}) {
  const readers = new Map(), writers = new Map();
  let serial = 0, disconnected = false;
  const send = (message, transfer = []) => { if (!disconnected) port.postMessage(message, transfer); };
  const local = cache;
  const closeReader = async handle => {
    const state = readers.get(handle);
    if (!state) return;
    readers.delete(handle); clearTimeout(state.timer);
    if (state.reader) { await state.reader.cancel().catch(() => {}); state.reader.releaseLock(); }
    else await state.response.body?.cancel().catch(() => {});
  };
  const touchReader = (handle, state) => {
    clearTimeout(state.timer);
    // Transformers performs header-only cache probes. They must neither read
    // chunks eagerly nor keep unused readers alive for the life of the page.
    state.timer = setTimeout(() => { closeReader(handle); }, state.reader ? 120000 : 30000);
  };
  const closeWriter = async handle => {
    const state = writers.get(handle); if (!state) return;
    writers.delete(handle); clearTimeout(state.timer); await state.writer.abort().catch(() => {});
  };
  const touchWriter = (handle, state) => {
    clearTimeout(state.timer); state.timer = setTimeout(() => { closeWriter(handle); }, 120000);
  };
  const listener = async ({ data }) => {
    if (!data?.pocketCache) return;
    const { id, op } = data;
    try {
      let result;
      if (op === 'match') {
        const response = await local.match(data.url);
        if (disconnected) { await response?.body?.cancel().catch(() => {}); return; }
        if (response) {
          const handle = ++serial, state = { response, reader: null, remainder: null };
          readers.set(handle, state); touchReader(handle, state);
          result = { ...metadata(response), headers: [...response.headers], handle };
        }
        else result = null;
      } else if (op === 'read') {
        const state = readers.get(data.handle); if (!state) throw new Error('Cached model stream is closed.');
        state.reader ||= state.response.body.getReader(); touchReader(data.handle, state);
        const part = state.remainder ? { value: state.remainder, done: false } : await state.reader.read();
        if (part.done) { readers.delete(data.handle); clearTimeout(state.timer); state.reader.releaseLock(); result = { done: true }; }
        else {
          const bytes = part.value, count = Math.min(bytes.byteLength, MODEL_CHUNK_BYTES);
          state.remainder = count < bytes.byteLength ? bytes.subarray(count) : null;
          const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count);
          send({ pocketCache: true, id, result: { buffer, done: false } }, [buffer]); return;
        }
      } else if (op === 'read-cancel') {
        await closeReader(data.handle); result = true;
      } else if (op === 'write-start') {
        const writer = await local.openWriter(data.url, data.meta);
        if (disconnected) { await writer.abort(); return; }
        const handle = ++serial, state = { writer };
        writers.set(handle, state); touchWriter(handle, state); result = { handle };
      } else if (op === 'write') {
        const state = writers.get(data.handle); if (!state) throw new Error('Model storage writer is closed.');
        touchWriter(data.handle, state); await state.writer.write(new Uint8Array(data.buffer)); result = true;
      } else if (op === 'write-end') {
        const state = writers.get(data.handle); if (!state) throw new Error('Model storage writer is closed.');
        touchWriter(data.handle, state); result = await state.writer.close();
        writers.delete(data.handle); clearTimeout(state.timer);
        const event = result.stored ? { type: 'stored', source: result.source, bytes: result.bytes, url: data.url, verified: result.verified }
          : { type: 'unavailable', source: result.source, reason: result.reason, message: storageReason({ name: result.reason === 'quota' ? 'QuotaExceededError' : '', message: result.reason === 'timeout' ? 'storage timeout' : result.reason }).message };
        send({ pocketCache: true, event }); onEvent(event);
      } else if (op === 'write-abort') {
        await closeWriter(data.handle); result = true;
      } else if (op === 'delete') result = await local.delete(data.url);
      else throw new Error('Unsupported cache operation.');
      send({ pocketCache: true, id, result });
    } catch (error) { send({ pocketCache: true, id, error: { name: error.name, message: error.message } }); }
  };
  port.addEventListener('message', listener); port.start();
  return () => {
    disconnected = true; port.removeEventListener('message', listener); port.close();
    for (const handle of readers.keys()) closeReader(handle);
    for (const handle of writers.keys()) closeWriter(handle);
    readers.clear(); writers.clear();
  };
}

export function createRemoteModelCache(port, { onEvent = () => {} } = {}) {
  let serial = 0;
  const pending = new Map();
  const listener = ({ data }) => {
    if (!data?.pocketCache) return;
    if (data.event) { onEvent(data.event); return; }
    const operation = pending.get(data.id); if (!operation) return;
    pending.delete(data.id); clearTimeout(operation.timer);
    if (data.error) { const error = new Error(data.error.message); error.name = data.error.name; operation.reject(error); }
    else operation.resolve(data.result);
  };
  port.addEventListener('message', listener); port.start();
  const rpc = (op, details = {}, transfer = []) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('Page-owned browser storage did not respond in time.')); }, 45000);
    pending.set(id, { resolve, reject, timer });
    try { port.postMessage({ pocketCache: true, id, op, ...details }, transfer); }
    catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
  });
  return {
    async match(request) {
      const url = keyFor(request);
      const record = await rpc('match', { url });
      if (!record) return undefined;
      const source = new Headers(record.headers).get('x-pocket-cache');
      onEvent({ type: 'hit', url, source });
      const stream = new ReadableStream({
        async pull(controller) { try { const part = await rpc('read', { handle: record.handle }); if (part.done) controller.close(); else controller.enqueue(new Uint8Array(part.buffer)); } catch (error) { controller.error(error); } },
        cancel() { return rpc('read-cancel', { handle: record.handle }).catch(() => {}); },
      }, { highWaterMark: 0 });
      return new Response(stream, record);
    },
    async put(request, response, progress) {
      if (response.status !== 200 || response.type === 'opaque') return { stored: false };
      const url = keyFor(request);
      const { handle } = await rpc('write-start', { url, meta: metadata(response) });
      const reader = response.body?.getReader();
      let loaded = 0; const total = Number(response.headers.get('content-length')) || 0;
      try {
        while (reader) {
          const { done, value } = await reader.read(); if (done) break;
          for (let offset = 0; offset < value.byteLength; offset += MODEL_CHUNK_BYTES) {
            const buffer = value.buffer.slice(value.byteOffset + offset, value.byteOffset + Math.min(value.byteLength, offset + MODEL_CHUNK_BYTES));
            await rpc('write', { handle, buffer }, [buffer]);
          }
          loaded += value.byteLength; progress?.({ loaded, total, progress: total ? loaded / total * 100 : 0 });
        }
        return await rpc('write-end', { handle, url });
      } catch (error) { await rpc('write-abort', { handle }).catch(() => {}); throw error; }
      finally { reader?.releaseLock(); }
    },
    delete(request) { return rpc('delete', { url: keyFor(request) }); },
  };
}

export function connectWorkerCache(worker, options = {}) {
  const channel = new MessageChannel();
  const disconnect = attachCachePort(channel.port1, options);
  worker.postMessage({ type: 'pocket-cache-port', port: channel.port2 }, [channel.port2]);
  return disconnect;
}

if (typeof document === 'undefined' && typeof globalThis.addEventListener === 'function') {
  globalThis.addEventListener('message', ({ data }) => {
    if (data?.type === 'pocket-cache-port' && data.port) pageCache = createRemoteModelCache(data.port, {
      onEvent: event => { for (const notify of workerEventListeners) notify(event); },
    });
  });
}

const defaultCache = createModelCache();
export async function primeBrowserCache() { await defaultCache.prime(); }

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
  const parts = []; let loaded = 0, lastProgress = 0;
  if (response.body) {
    const reader = response.body.getReader();
    try {
      while (true) {
        init.signal?.throwIfAborted();
        const { done, value } = await reader.read(); if (done) break;
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

export function runtimePaths(base, browser = globalThis.navigator) {
  const revision = typeof __RUNTIME_REVISION__ === 'undefined' ? 'development' : __RUNTIME_REVISION__;
  const userAgent = browser?.userAgent || '';
  const hasWebGPU = Boolean(browser && 'gpu' in browser);
  const safariVersion = userAgent.match(/Version\/(\d+)/);
  const safari = (browser?.vendor || '').includes('Apple')
    && !/CriOS|FxiOS|EdgiOS|OPiOS|mercury|brave|Chrome|Android/i.test(userAgent);
  const oldSafari = safari && safariVersion && Number(safariVersion[1]) < 26;
  // iOS browser shells share WebKit. Use an explicit OS token for shells that
  // omit Safari's Version token; missing versions keep the modern default.
  const iosVersion = /\b(?:iPhone|iPad|iPod)\b/.test(userAgent) && userAgent.match(/\bOS (\d+)[_.]\d+/);
  const oldIosShell = /CriOS|FxiOS|EdgiOS|OPiOS/i.test(userAgent)
    && iosVersion && Number(iosVersion[1]) < 26;
  // Mirrors Transformers.js 4.3's Safari workaround. Asyncify is not JSPI;
  // checking WebAssembly.Suspending would choose the wrong runtime here.
  const suffix = !hasWebGPU && (oldSafari || oldIosShell) ? '' : '.asyncify';
  return Object.fromEntries(['wasm', 'mjs'].map(extension => {
    const url = new URL(`ort-wasm-simd-threaded${suffix}.${extension}`, base);
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
