import { TITLE_MODEL, TITLE_REVISION, describeMelody, applyTitle } from './title-utils.js';
import { createModelCache, runtimePaths } from './model-cache.js';

let generator = null;
let backend = null;
let running = false;

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
      const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
      backend = adapter ? 'webgpu' : 'wasm';
      const load = () => pipeline('text-generation', TITLE_MODEL, {
        revision: TITLE_REVISION, dtype: 'q4', device: backend,
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
    }
    const usedTitles = new Set(), usedNames = new Set(), named = [];
    for (let index = 0; index < songs.length; index++) {
      const song = songs[index];
      progress({ message: `Naming melody ${index+1} of ${songs.length}…`, songIndex: index, numSongs: songs.length });
      let result = null;
      for (let attempt = 0; attempt < 2 && !result; attempt++) {
        const avoid = Array.from(usedTitles).slice(-5).join(', ');
        const messages = [
          { role: 'system', content: 'Give each instrumental tune an evocative title matching its character. Reply with the title only, in two to four words. Avoid generic titles containing Melody or Song.' },
          { role: 'user', content: 'Fast, high notes with playful jumps.' },
          { role: 'assistant', content: 'Pixel Fireflies' },
          { role: 'user', content: 'Slow, low notes with a gentle falling movement.' },
          { role: 'assistant', content: 'Velvet Moon' },
          { role: 'user', content: 'Moderate, bright rhythmic notes, rising and falling.' },
          { role: 'assistant', content: 'Sunlit Steps' },
          { role: 'user', content: `${describeMelody(song)}.${avoid ? ` Use different words from: ${avoid}.` : ''}${attempt ? ' Give a different, shorter title.' : ''}` },
        ];
        const outputs = await generator(messages, { max_new_tokens: 20, do_sample: true,
          temperature: 0.65, top_p: 0.9, repetition_penalty: 1.15 });
        const text = outputs[0].generated_text;
        const raw = Array.isArray(text) ? text.at(-1).content : text;
        try { result = applyTitle(song, raw, usedTitles, usedNames); } catch { /* One bounded retry. */ }
      }
      result ||= { ...song, naming: { status:'fallback', reason:'The model did not suggest a usable title.' } };
      named.push(result);
      progress({ message: result.naming.status === 'named' ? `Named “${result.title}”` : 'Kept the melody’s existing name.',
        songIndex: index, numSongs: songs.length, title: result.title, songId: song.id, song: result, backend });
    }
    self.postMessage({ type: 'done', id, songs: named, backend });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: `Title generation unavailable: ${error.message}. Your melodies are preserved.` });
  } finally { running = false; }
});
