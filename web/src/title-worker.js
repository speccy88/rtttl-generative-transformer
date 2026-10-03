import { TITLE_MODEL, TITLE_REVISION, TITLE_DTYPE, createTitlePrompt, EXAMPLE_TITLES, extractGeneratedTitle, isRepeatedTitle, applyTitle, fallbackTitle } from './title-utils.js';
import { createModelCache, runtimePaths } from './model-cache.js';
import { isWebKitEngine } from './browser-policy.js';

let generator = null;
let backend = null;
let running = false;
let recentTitles = [], recentNames = [];
let titleEndTokens;
let gpuDevice = null;

async function releaseTitleModel() {
  // Finish native work while the worker's execution context is still alive.
  // Release the session and explicitly lose its device before announcing
  // completion. WebKit still needs the CPU policy: normal disposal alone did
  // not prevent its native GPU teardown crash in the full inference test.
  const activeGenerator = generator, activeDevice = gpuDevice;
  generator = null;
  gpuDevice = null;
  try {
    if (activeDevice) await activeDevice.queue.onSubmittedWorkDone().catch(() => {});
    if (activeGenerator) await activeGenerator.dispose();
  } finally {
    if (activeDevice) {
      activeDevice.destroy();
      await activeDevice.lost;
    }
  }
}

self.addEventListener('message', async ({ data }) => {
  if (data.type !== 'title') return;
  const { id, songs, runtimeBaseUrl } = data;
  const progress = details => self.postMessage({ type: 'progress', id, ...details });
  if (running) { self.postMessage({ type: 'error', id, message: 'Title model is busy.' }); return; }
  running = true;
  try {
    if (!generator) {
      progress({ message: 'Preparing the optional title model…' });
      const { pipeline, env } = await import('@huggingface/transformers');
      const cachedFiles = new Set();
      const titleCache = createModelCache({ onEvent: event => {
        if (event.type === 'hit') cachedFiles.add(new URL(event.url).pathname.split('/').at(-1));
        if (['stored', 'hit', 'unavailable'].includes(event.type)) self.postMessage({ type: 'cache', id, cacheState: event.type, message: event.message, source: event.source });
      } });
      env.allowLocalModels = false;
      // Transformers.js 4.3's pipeline file-discovery calls omit revision.
      // This dedicated worker uses one pinned model, so pin the URL template
      // too: discovery and actual loading now share the same cache keys.
      env.remotePathTemplate = `{model}/resolve/${TITLE_REVISION}/`;
      env.useCustomCache = true;
      env.customCache = titleCache;
      env.useBrowserCache = false;
      env.useWasmCache = true;
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
      env.backends.onnx.wasm.wasmPaths = runtimePaths(runtimeBaseUrl);
      const adapter = isWebKitEngine() ? null : await navigator.gpu?.requestAdapter().catch(() => null);
      backend = adapter ? 'webgpu' : 'wasm';
      const load = () => pipeline('text-generation', TITLE_MODEL, {
        revision: TITLE_REVISION, dtype: TITLE_DTYPE, device: backend,
        progress_callback: event => {
          if (event.status === 'progress') progress({ message: cachedFiles.has(event.file?.split('/').at(-1))
            ? 'Loading the optional title model from browser storage…' : 'Downloading the optional title model…',
            downloadProgress: event.progress, file: event.file });
          if (event.status === 'ready') progress({ message: `Title model ready · ${backend === 'webgpu' ? 'WebGPU' : 'CPU'}` });
        },
      });
      try { generator = await load(); }
      catch (error) {
        if (backend !== 'webgpu') throw error;
        backend = 'wasm';
        progress({ message: 'Using CPU for titles because this GPU could not load the model.' });
        generator = await load();
      }
      if (backend === 'webgpu') gpuDevice = await env.backends.onnx.webgpu.device;
      const configuredEnd = generator.model.generation_config?.eos_token_id ?? generator.model.config.eos_token_id ?? generator.tokenizer.eos_token_id;
      const lineBreak = generator.tokenizer.encode('\n', { add_special_tokens: false });
      titleEndTokens = [...new Set([...(Array.isArray(configuredEnd) ? configuredEnd : [configuredEnd]),
        ...(lineBreak.length === 1 ? lineBreak : [])].filter(Number.isInteger))];
    }
    const usedTitles = new Set([...recentTitles, ...(data.usedTitles || [])].slice(-64).map(title => String(title).toLowerCase()));
    const usedNames = new Set([...recentNames, ...(data.usedNames || [])].slice(-64).map(name => String(name).toLowerCase()));
    const named = [];
    for (let index = 0; index < songs.length; index++) {
      const song = songs[index];
      progress({ message: `Naming melody ${index+1} of ${songs.length}…`, songIndex: index, numSongs: songs.length });
      let result = null;
      for (let attempt = 0; attempt < 2 && !result; attempt++) {
        const outputs = await generator(createTitlePrompt(song, usedTitles, attempt), { max_new_tokens: 18, return_full_text: false, do_sample: true,
          temperature: attempt ? 0.95 : 0.8, top_p: 0.92, repetition_penalty: 1.15, eos_token_id: titleEndTokens });
        const text = outputs[0].generated_text;
        // Completion models may continue the next Sound:/Title: record. Only the
        // first answer line belongs to this melody; it still passes validation.
        try {
          const raw = extractGeneratedTitle(Array.isArray(text) ? text.at(-1).content : text);
          if (isRepeatedTitle(raw, EXAMPLE_TITLES)) continue;
          result = applyTitle(song, raw, usedTitles, usedNames);
        } catch { /* One bounded retry. */ }
      }
      result ||= fallbackTitle(song, usedTitles, usedNames);
      named.push(result);
      recentTitles = Array.from(usedTitles).slice(-64);
      recentNames = Array.from(usedNames).slice(-64);
      progress({ message: result.title ? `Named “${result.title}”` : 'Kept the melody’s existing name.',
        songIndex: index, numSongs: songs.length, title: result.title, songId: song.id, song: result, backend });
    }
    progress({ message: 'Finishing your titles…', phase: 'release' });
    await releaseTitleModel();
    self.postMessage({ type: 'done', id, songs: named, backend, disposed: true });
  } catch (error) {
    try { await releaseTitleModel(); } catch { /* Preserve the original failure. */ }
    self.postMessage({ type: 'error', id, message: `Title generation unavailable: ${error.message}. Your melodies are preserved.`, disposed: true });
  } finally {
    running = false;
    // Closing ourselves lets the current callback unwind. The page must drop its
    // reference without force-terminating this already-disposed worker.
    self.close();
  }
});
