let worker = null;
let busy = false;
let sequence = 0;

/** Optional title inference is loaded only after the user enables it. */
export function titleSongs(songs, { onProgress = () => {}, signal } = {}) {
  if (busy) return Promise.reject(new Error('Title generation is already running.'));
  if (signal?.aborted) return Promise.reject(new DOMException('Cancelled', 'AbortError'));
  try {
    worker ||= new Worker(new URL('./title-worker.js', import.meta.url), { type: 'module' });
  } catch (error) { return Promise.reject(error); }
  busy = true;
  const activeWorker = worker;
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    function cleanup() {
      busy = false;
      activeWorker.removeEventListener('message', receive);
      activeWorker.removeEventListener('error', failed);
      signal?.removeEventListener('abort', cancel);
    }
    function failed(event) {
      cleanup();
      activeWorker.terminate();
      if (worker === activeWorker) worker = null;
      reject(new Error(event.message || 'The title model could not run. Your melodies are ready to play.'));
    }
    function cancel() {
      cleanup();
      activeWorker.terminate();
      if (worker === activeWorker) worker = null;
      reject(new DOMException('Title generation cancelled', 'AbortError'));
    }
    function receive({ data }) {
      if (data.id !== id) return;
      if (data.type === 'progress') onProgress(data);
      if (data.type === 'cache') onProgress({ ...data, cacheEvent: true });
      if (data.type === 'done') { cleanup(); resolve(data.songs); }
      if (data.type === 'error') failed(data);
    }
    activeWorker.addEventListener('message', receive);
    activeWorker.addEventListener('error', failed);
    signal?.addEventListener('abort', cancel, { once: true });
    try {
      activeWorker.postMessage({ type: 'title', id, songs,
        runtimeBaseUrl: new URL(`${import.meta.env.BASE_URL}runtime/transformers/`, location.origin).href });
    } catch (error) { failed(error); }
  });
}
