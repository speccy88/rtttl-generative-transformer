import * as ort from 'onnxruntime-web/wasm';
import { createModelCache, cachedFetch, loadCachedRuntime } from './model-cache.js';
import { MODEL_BYTES, MODEL_SHA256, WINDOW_SAMPLES, runBasicPitchInference, renderTranscription } from './transcription-core.js';

const SITE_BASE = new URL(import.meta.env.BASE_URL, self.location.origin);
let busy = false;
const send = (type, detail = {}) => self.postMessage({ type, ...detail });

self.onmessage = async ({ data }) => {
  if (data?.type !== 'transcribe' || busy) return;
  busy = true;
  let session;
  const abort = new AbortController();
  const timeout = setTimeout(() => abort.abort(new Error('Transcription assets took too long to load. Please retry.')), 120_000);
  try {
    const progress = message => send('progress', { message });
    const cache = createModelCache();
    const modelUrl = new URL('transcription/basic-pitch.onnx', SITE_BASE).href;
    progress('Loading the local transcription model…');
    const response = await cachedFetch(modelUrl, { signal: abort.signal }, { cache });
    if (!response.ok) throw new Error(`Transcription model unavailable (HTTP ${response.status}).`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== MODEL_BYTES) {
      await cache.delete(modelUrl).catch(() => {});
      throw new Error('The transcription model is incomplete. Its cached copy was removed; retry to download it again.');
    }
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const actualHash = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
    if (actualHash !== MODEL_SHA256) {
      await cache.delete(modelUrl).catch(() => {});
      throw new Error('The transcription model checksum does not match this release. Its cached copy was removed; retry to download it again.');
    }
    progress('Preparing browser CPU inference…');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const runtime = await loadCachedRuntime(new URL('runtime/ort/', SITE_BASE).href, { cache, signal: abort.signal,
      onEvent: event => { if (event.type === 'download') progress('Loading browser inference runtime…'); },
    });
    Object.assign(ort.env.wasm, runtime);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    clearTimeout(timeout);
    const evidence = await runBasicPitchInference(data.samples, async window => {
      const input = new ort.Tensor('float32', window, [1, WINDOW_SAMPLES, 1]);
      let outputs;
      try {
        outputs = await session.run({ 'serving_default_input_2:0': input });
        // Official Basic Pitch ONNX names: :1=notes, :2=onsets, :0=contours.
        return { note: Float32Array.from(outputs['StatefulPartitionedCall:1'].data), onset: Float32Array.from(outputs['StatefulPartitionedCall:2'].data) };
      } finally {
        input.dispose();
        if (outputs) for (const output of Object.values(outputs)) output.dispose();
      }
    }, progress);
    progress('Following lead voices and preserving rhythm…');
    const result = renderTranscription(evidence, data.samples, data.options, data.name);
    send('result', { result });
  } catch (error) {
    send('error', { message: error instanceof Error ? error.message : String(error) });
  } finally {
    clearTimeout(timeout);
    if (session) { try { await session.release(); } catch { /* Worker termination releases remaining resources. */ } }
    busy = false;
  }
};
