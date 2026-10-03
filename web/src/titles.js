import { connectWorkerCache } from './model-cache.js';

let worker = null;
let disconnectWorkerCache = null;
let busy = false;
let sequence = 0;
let cancelActive = null;

export function disposeTitleWorker() {
  if (cancelActive) { cancelActive(); return; }
  disconnectWorkerCache?.(); disconnectWorkerCache = null;
  worker?.terminate(); worker = null;
}

/** Optional title inference is loaded only after the user enables it. */
export function titleSongs(songs, { onProgress = () => {}, signal, usedTitles = [], usedNames = [] } = {}) {
  if (busy) return Promise.reject(new Error('Title generation is already running.'));
  if (signal?.aborted) return Promise.reject(new DOMException('Cancelled', 'AbortError'));
  try {
    if (!worker) {
      worker = new Worker(new URL('./title-worker.js', import.meta.url), { type: 'module' });
      disconnectWorkerCache = connectWorkerCache(worker);
    }
  } catch (error) { return Promise.reject(error); }
  busy = true;
  const activeWorker = worker;
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    function cleanup() {
      busy = false;
      cancelActive = null;
      activeWorker.removeEventListener('message', receive);
      activeWorker.removeEventListener('error', failed);
      signal?.removeEventListener('abort', cancel);
    }
    function failed(event) {
      cleanup();
      disconnectWorkerCache?.(); disconnectWorkerCache = null;
      if (!event.disposed) activeWorker.terminate();
      if (worker === activeWorker) worker = null;
      reject(new Error(event.message || 'The title model could not run. Your melodies are ready to play.'));
    }
    function cancel() {
      cleanup();
      disconnectWorkerCache?.(); disconnectWorkerCache = null;
      activeWorker.terminate();
      if (worker === activeWorker) worker = null;
      reject(new DOMException('Title generation cancelled', 'AbortError'));
    }
    function receive({ data }) {
      if (data.id !== id) return;
      if (data.type === 'progress') onProgress(data);
      if (data.type === 'cache') onProgress({ ...data, cacheEvent: true });
      if (data.type === 'done') {
        cleanup();
        // Keep the downloaded files on disk, and release the larger LLM/WASM
        // allocation between batches so mobile tabs can reload safely.
        if (data.disposed) {
          // The worker released its device and will close its own event loop.
          // Do not interrupt native GPU cleanup with an external termination.
          disconnectWorkerCache?.(); disconnectWorkerCache = null;
          if (worker === activeWorker) worker = null;
        } else disposeTitleWorker();
        resolve(data.songs);
      }
      if (data.type === 'error') failed(data);
    }
    activeWorker.addEventListener('message', receive);
    activeWorker.addEventListener('error', failed);
    cancelActive = cancel;
    signal?.addEventListener('abort', cancel, { once: true });
    try {
      activeWorker.postMessage({ type: 'title', id, songs, usedTitles, usedNames,
        runtimeBaseUrl: new URL(`${import.meta.env.BASE_URL}runtime/transformers/`, location.origin).href });
    } catch (error) { failed(error); }
  });
}
