import * as ort from 'onnxruntime-web/webgpu';
import { EventTokenizer, GenerationCancelled, generateSong, validateOptions } from './sampler.js';
import { createModelCache, cachedFetch, loadCachedRuntime } from './model-cache.js';

const SITE_BASE = new URL(import.meta.env.BASE_URL, self.location.origin);
const MODEL_TIMEOUT_MS = 300_000;
const SESSION_TIMEOUT_MS = 120_000;
const STEP_TIMEOUT_MS = 45_000;
let active = null;
let loaded = null;
let cached = null;
let runtimeReady = false;
const modelCache = createModelCache({ onEvent: event => {
  if (active && ['stored', 'hit', 'unavailable'].includes(event.type)) send('cache', active.id, { cacheState: event.type, message: event.message, source: event.source });
} });

const send = (type, id, detail = {}) => self.postMessage({ type, id, ...detail });
const status = (id, message, detail = {}) => send('status', id, { message, ...detail });
const errorText = error => error instanceof Error ? error.message : String(error);

function deadline(promise, milliseconds, message, onLate = null) {
  let expired = false;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => { expired = true; reject(new Error(message)); }, milliseconds);
  });
  // Dispose a session if creation eventually finishes after its deadline.
  const operation = promise.then(value => {
    if (expired && onLate) Promise.resolve(onLate(value)).catch(() => {});
    return value;
  });
  return Promise.race([operation, timeout]).finally(() => clearTimeout(timer));
}

async function releaseSession(session) {
  // A lost GPU must not prevent a timeout or cancellation from reaching the UI.
  try { await deadline(Promise.resolve(session.release()), 2_000, 'Session cleanup timed out.'); }
  catch { /* The worker can continue with a fresh CPU session. */ }
}

async function fetchFile(url, job, label, asJson = false, expectedHash = null) {
  const abort = new AbortController();
  job.aborts.add(abort);
  const timer = setTimeout(() => abort.abort(new Error(`${label} download timed out. Please retry.`)), MODEL_TIMEOUT_MS);
  try {
    if (job.cancelled) throw new GenerationCancelled();
    const response = await cachedFetch(url, { signal: abort.signal }, { cache: modelCache, onEvent: event => {
      if (event.type === 'hit') status(job.id, `Loading ${label.toLowerCase()} from browser storage…`, { phase: 'cache' });
      if (event.type === 'download') status(job.id, `Downloading ${label.toLowerCase()}…`, {
        phase: 'download', downloadProgress: event.total ? event.loaded / event.total : null,
      });
    } });
    if (!response.ok) throw new Error(`${label} download failed (HTTP ${response.status}).`);
    const fromStorage = response.headers.get('x-pocket-cache') !== 'network';
    if (asJson && !expectedHash) return await response.json();
    const total = Number(response.headers.get('content-length')) || 0;
    const chunks = [];
    let received = 0; let lastProgress = 0;
    if (response.body) {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); received += value.byteLength;
        if (performance.now() - lastProgress > 200) {
          status(job.id, `Loading ${label.toLowerCase()}${fromStorage ? ' from browser storage' : ''}…`, {
            phase: fromStorage ? 'cache' : 'download', receivedBytes: received, totalBytes: total,
            downloadProgress: total ? received / total : null,
          });
          lastProgress = performance.now();
        }
      }
    } else {
      const bytes = new Uint8Array(await response.arrayBuffer());
      chunks.push(bytes); received = bytes.byteLength;
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    if (expectedHash) {
      if (typeof expectedHash !== 'string' || !/^[a-f0-9]{64}$/i.test(expectedHash)) throw new Error(`${label} checksum in manifest is invalid.`);
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const actual = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
      if (actual !== expectedHash.toLowerCase()) {
        await modelCache.delete(url);
        throw new Error(`${label} checksum did not match the published release. The invalid cached file was removed; please retry.`);
      }
    }
    status(job.id, `${label} ${fromStorage ? 'loaded from browser storage' : 'ready'}.`, { phase: fromStorage ? 'cache' : 'download', receivedBytes: received, totalBytes: total || received, downloadProgress: 1 });
    return asJson ? JSON.parse(new TextDecoder().decode(bytes)) : bytes;
  } catch (error) {
    if (job.cancelled) throw new GenerationCancelled();
    if (abort.signal.aborted) throw abort.signal.reason;
    throw error;
  } finally { clearTimeout(timer); job.aborts.delete(abort); }
}

async function resolveManifest(value, job) {
  const manifest = value && typeof value === 'object' ? value
    : await fetchFile(new URL(typeof value === 'string' ? value : 'model-manifest.json', SITE_BASE), job, 'Model manifest', true);
  if (!manifest.modelUrl || !manifest.tokenizerUrl) throw new Error('The melody model manifest is missing its model or vocabulary URL.');
  const contextLength = manifest.contextLength ?? 256;
  if (!Number.isInteger(contextLength) || contextLength < 4 || contextLength > 4096) throw new Error('Invalid model context length in manifest.');
  if (manifest.vocabSize !== undefined && manifest.vocabSize !== 940) throw new Error('Unsupported melody model vocabulary size.');
  return {
    ...manifest, contextLength,
    modelUrl: new URL(manifest.modelUrl, SITE_BASE).href,
    tokenizerUrl: new URL(manifest.tokenizerUrl, SITE_BASE).href,
    runtimeBaseUrl: new URL(manifest.runtimeBaseUrl ?? 'runtime/ort/', SITE_BASE).href,
  };
}

async function runSession(session, context) {
  const tensor = new ort.Tensor('int64', BigInt64Array.from(context, BigInt), [1, context.length]);
  let result;
  try {
    result = await deadline(session.run({ input_ids: tensor }), STEP_TIMEOUT_MS,
      'Melody inference took too long. Try the CPU backend or a shorter melody.');
    const output = result.logits ?? result[session.outputNames[0]];
    if (!output || output.dims.at(-1) !== 940) throw new Error('Unexpected melody model output shape.');
    // Supports either final-token [1,940] or full-sequence [1,time,940] exports.
    const values = output.data;
    return Float32Array.from(values.subarray(values.length - 940));
  } finally {
    tensor.dispose();
    if (result) for (const output of Object.values(result)) output.dispose();
  }
}

async function createSession(bytes, backend, job) {
  if (backend === 'webgpu' && !self.navigator.gpu) throw new Error('This browser does not expose WebGPU.');
  status(job.id, `Preparing ${backend === 'webgpu' ? 'WebGPU' : 'CPU'} inference…`, { phase: 'initialize', backend });
  const session = await deadline(ort.InferenceSession.create(bytes, {
    executionProviders: [backend], graphOptimizationLevel: 'all',
  }), SESSION_TIMEOUT_MS, `${backend === 'webgpu' ? 'WebGPU' : 'CPU'} initialization timed out.`, releaseSession);
  try {
    // Unsupported GPU operators sometimes fail only on the first real run.
    await runSession(session, [1, 98]);
    if (job.cancelled) throw new GenerationCancelled();
    return session;
  } catch (error) { await releaseSession(session); throw error; }
}

async function initialize(manifestInput, backend, job) {
  if (!['auto', 'webgpu', 'wasm'].includes(backend)) throw new Error('Backend must be auto, webgpu, or wasm.');
  const manifest = await resolveManifest(manifestInput, job);
  const key = JSON.stringify([manifest.modelUrl, manifest.tokenizerUrl, manifest.runtimeBaseUrl,
    manifest.modelSha256, manifest.tokenizerSha256, manifest.contextLength]);
  if (loaded?.key === key && (backend === 'auto' || loaded.backend === backend)) {
    status(job.id, `Ready on ${loaded.backend === 'webgpu' ? 'WebGPU' : 'CPU'}.`, { phase: 'ready', backend: loaded.backend, ready: true });
    return loaded;
  }
  if (loaded) { await releaseSession(loaded.session); loaded = null; }
  if (cached?.key !== key) {
    status(job.id, 'Loading the melody model…', { phase: 'download' });
    // Downloads are independent; a failure aborts the other request as well.
    try {
      const [bytes, vocabulary] = await Promise.all([
        fetchFile(manifest.modelUrl, job, 'Melody model', false, manifest.modelSha256),
        fetchFile(manifest.tokenizerUrl, job, 'Melody vocabulary', true, manifest.tokenizerSha256),
      ]);
      cached = { key, bytes, tokenizer: new EventTokenizer(vocabulary) };
    } catch (error) { for (const abort of job.aborts) abort.abort(); throw error; }
  }
  if (job.cancelled) throw new GenerationCancelled();
  // GitHub Pages is not cross-origin isolated. One CPU thread needs no special
  // headers and still runs entirely inside this dedicated worker.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  if (!runtimeReady) {
    const abort = new AbortController();
    job.aborts.add(abort);
    const timer = setTimeout(() => abort.abort(new Error('Browser runtime download timed out. Please retry.')), MODEL_TIMEOUT_MS);
    try {
      const runtime = await loadCachedRuntime(manifest.runtimeBaseUrl, { cache: modelCache, signal: abort.signal,
        onEvent: event => status(job.id, event.type === 'hit' ? 'Loading browser runtime from storage…' : 'Downloading browser runtime…',
          { phase: event.type === 'hit' ? 'cache' : 'download', downloadProgress: event.total ? event.loaded / event.total : null }),
      });
      Object.assign(ort.env.wasm, runtime); runtimeReady = true;
    } finally { clearTimeout(timer); job.aborts.delete(abort); }
  }
  let actualBackend = backend === 'auto' ? 'webgpu' : backend;
  let session;
  try { session = await createSession(cached.bytes, actualBackend, job); }
  catch (error) {
    if (job.cancelled || backend !== 'auto') throw error;
    const fallbackReason = errorText(error);
    status(job.id, 'WebGPU is unavailable here. Preparing local CPU inference…', { phase: 'fallback', backend: 'wasm', fallbackReason });
    actualBackend = 'wasm';
    session = await createSession(cached.bytes, 'wasm', job);
  }
  loaded = { key, session, backend: actualBackend, tokenizer: cached.tokenizer, manifest };
  status(job.id, `Ready on ${actualBackend === 'webgpu' ? 'WebGPU' : 'CPU'}.`, { phase: 'ready', backend: actualBackend, ready: true });
  return loaded;
}

async function perform(message, job) {
  const backend = message.options?.backend ?? message.backend ?? 'auto';
  // Validate before any network request, model allocation, or GPU work.
  const validated = message.type === 'generate' ? validateOptions(message.options) : null;
  let runtime = await initialize(message.modelManifest, backend, job);
  if (message.type === 'init') { send('done', job.id, { initialized: true, backend: runtime.backend, songs: [] }); return; }
  const { options, plan } = validated;
  const songs = [];
  job.songs = songs;
  const infer = async context => {
    try { return await runSession(runtime.session, context); }
    catch (error) {
      if (job.cancelled || backend !== 'auto' || runtime.backend !== 'webgpu') throw error;
      status(job.id, 'The GPU stopped responding. Continuing locally on CPU…', {
        phase: 'fallback', backend: 'wasm', fallbackReason: errorText(error),
      });
      await releaseSession(runtime.session);
      loaded = null;
      const session = await createSession(cached.bytes, 'wasm', job);
      runtime = { ...runtime, session, backend: 'wasm' };
      loaded = runtime;
      return runSession(runtime.session, context);
    }
  };
  for (let index = 0; index < plan.length; index++) {
    if (job.cancelled) throw new GenerationCancelled();
    status(job.id, `Composing melody ${index + 1} of ${plan.length}…`, { phase: 'generate', backend: runtime.backend });
    const song = await generateSong({
      infer, tokenizer: runtime.tokenizer, options, settings: plan[index],
      seed: options.seed + index, contextLength: runtime.manifest.contextLength,
      name: `Melody${String(index + 1).padStart(2, '0')}`,
      isCancelled: () => job.cancelled,
      onProgress: progress => send('progress', job.id, { ...progress, songIndex: index, numSongs: plan.length, backend: runtime.backend }),
    });
    song.id = `${job.id}-${index}`;
    song.backend = runtime.backend;
    songs.push(song);
    send('song', job.id, { song, songIndex: index, numSongs: plan.length, backend: runtime.backend });
  }
  send('done', job.id, { songs, backend: runtime.backend, cancelled: false });
}

self.onmessage = async ({ data: message }) => {
  if (!message || typeof message !== 'object') return;
  if (message.type === 'cancel') {
    if (active && (message.id === undefined || message.id === active.id)) {
      active.cancelled = true;
      for (const abort of active.aborts) abort.abort();
      status(active.id, 'Stopping after the current inference step…', { phase: 'cancelling' });
    }
    return;
  }
  if (!['generate', 'init'].includes(message.type)) return;
  if (active) { send('error', message.id, { message: 'A melody job is already running. Wait for it to finish or cancel it first.' }); return; }
  const job = { id: message.id, cancelled: false, aborts: new Set(), songs: [] };
  active = job;
  try { await perform(message, job); }
  catch (error) {
    if (job.cancelled || error instanceof GenerationCancelled) send('done', job.id, { songs: job.songs, backend: loaded?.backend ?? null, cancelled: true });
    else send('error', job.id, { message: errorText(error), backend: loaded?.backend ?? null });
  } finally { active = null; }
};
