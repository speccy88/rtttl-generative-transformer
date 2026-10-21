import { MAX_FILE_BYTES, MAX_SOURCE_SECONDS, SAMPLE_RATE, validateTranscriptionOptions } from './transcription-core.js';
import { connectWorkerCache } from './model-cache.js';

export { MAX_FILE_BYTES, MAX_SOURCE_SECONDS } from './transcription-core.js';
const MAX_DECODED_SAMPLES = SAMPLE_RATE * MAX_SOURCE_SECONDS * 2;
const SUPPORTED_SUFFIXES = /\.(mp3|wav|flac|m4a|ogg|aac|opus|webm)$/i;
let cachedFile = null;
let cachedDecode = null;
let analyzing = false;

const abortError = () => new DOMException('Audio analysis cancelled.', 'AbortError');
function checkAbort(signal) { if (signal?.aborted) throw abortError(); }

function abortable(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const abort = () => reject(abortError());
    signal.addEventListener('abort', abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

export function validateAudioFile(file) {
  if (!file || typeof file.arrayBuffer !== 'function' || !Number.isFinite(file.size) || file.size <= 0) throw new Error('Choose a non-empty audio file.');
  if (file.size > MAX_FILE_BYTES) throw new Error('Choose an audio file smaller than 32 MiB.');
  if (!SUPPORTED_SUFFIXES.test(file.name ?? '') && !String(file.type ?? '').startsWith('audio/')) throw new Error('Choose an audio file supported by this browser, such as WAV or MP3.');
}

function sourceMetadata(file, signal) {
  return new Promise((resolve, reject) => {
    checkAbort(signal);
    const audio = document.createElement('audio');
    const url = URL.createObjectURL(file);
    const cleanup = () => {
      clearTimeout(timer); signal?.removeEventListener('abort', abort);
      audio.onloadedmetadata = null; audio.onerror = null;
      audio.removeAttribute('src'); audio.load(); URL.revokeObjectURL(url);
    };
    const abort = () => { cleanup(); reject(abortError()); };
    const fail = message => { cleanup(); reject(new Error(message)); };
    const timer = setTimeout(() => fail('Could not read this audio file within 15 seconds. Try exporting it as WAV or MP3.'), 15_000);
    signal?.addEventListener('abort', abort, { once: true });
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => {
      const duration = audio.duration;
      if (!Number.isFinite(duration) || duration <= 0) { fail('This file has no readable audio duration. Export it as WAV or MP3.'); return; }
      if (duration > MAX_SOURCE_SECONDS) { fail('Choose a source shorter than 10 minutes, or trim the song before opening it.'); return; }
      cleanup(); resolve(duration);
    };
    audio.onerror = () => fail('This browser cannot decode the audio codec. Export the recording as WAV or MP3 and retry.');
    audio.src = url;
  });
}

async function decodeFile(file, onProgress, signal) {
  validateAudioFile(file);
  onProgress('Checking audio duration…');
  await sourceMetadata(file, signal);
  checkAbort(signal);
  const Context = globalThis.OfflineAudioContext ?? globalThis.webkitOfflineAudioContext;
  if (!Context) throw new Error('This browser does not support Web Audio decoding. Try a current Safari, Firefox, Chrome or Edge browser.');
  onProgress('Decoding audio on this device…');
  const context = new Context(1, 1, SAMPLE_RATE);
  let buffer; let timer;
  try {
    const encoded = await file.arrayBuffer();
    checkAbort(signal);
    const deadline = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Audio decoding exceeded 60 seconds. Try a shorter WAV or MP3 recording.')), 60_000); });
    buffer = await abortable(Promise.race([context.decodeAudioData(encoded), deadline]), signal);
  } catch (error) {
    if (error.name === 'AbortError' || error.message.includes('60 seconds')) throw error;
    throw new Error('The audio codec is not supported by this browser, or the file is damaged. Export it as WAV or MP3 and retry.');
  } finally { clearTimeout(timer); }
  checkAbort(signal);
  if (!buffer.length || buffer.duration > MAX_SOURCE_SECONDS || buffer.length * buffer.numberOfChannels > MAX_DECODED_SAMPLES) throw new Error('Decoded audio exceeds the memory limit. Use a shorter mono or stereo source (up to 10 minutes).');
  return buffer;
}

async function getDecoded(file, { onProgress = () => {}, signal } = {}) {
  checkAbort(signal);
  validateAudioFile(file);
  if (file !== cachedFile || !cachedDecode) {
    cachedFile = file;
    cachedDecode = decodeFile(file, message => { if (!signal?.aborted) onProgress(message); }, signal);
    const pending = cachedDecode;
    pending.catch(() => { if (cachedDecode === pending) { cachedFile = null; cachedDecode = null; } });
  }
  // Native decodeAudioData cannot be stopped midway; cancelled results are
  // discarded. Model inference is in a worker that can be terminated promptly.
  return abortable(cachedDecode, signal);
}

export async function inspectAudio(file, { onProgress = () => {}, signal } = {}) {
  const buffer = await getDecoded(file, { onProgress, signal });
  checkAbort(signal);
  const bins = Math.min(800, buffer.length);
  const peaks = new Array(bins).fill(0);
  for (let bin = 0; bin < bins; bin++) {
    const first = Math.floor(bin / bins * buffer.length);
    const last = Math.floor((bin + 1) / bins * buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const samples = buffer.getChannelData(channel);
      for (let index = first; index < last; index++) peaks[bin] = Math.max(peaks[bin], Math.abs(samples[index]));
    }
  }
  return { duration: buffer.duration, peaks, sampleRate: buffer.sampleRate, channels: buffer.numberOfChannels };
}

export async function decodeAudioForTranscription(file, options = {}, { onProgress = () => {}, signal } = {}) {
  const buffer = await getDecoded(file, { onProgress, signal });
  checkAbort(signal);
  const normalized = validateTranscriptionOptions(options, buffer.duration);
  onProgress('Preparing the selected excerpt…');
  const length = Math.round(normalized.duration * SAMPLE_RATE);
  let samples;
  if (buffer.sampleRate === SAMPLE_RATE) {
    const first = Math.round(normalized.start * buffer.sampleRate);
    samples = new Float32Array(Math.min(length, buffer.length - first));
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const values = buffer.getChannelData(channel);
      for (let i = 0; i < samples.length; i++) samples[i] += values[first + i] / buffer.numberOfChannels;
    }
  } else {
    const Context = globalThis.OfflineAudioContext ?? globalThis.webkitOfflineAudioContext;
    const render = new Context(1, length, SAMPLE_RATE);
    const source = render.createBufferSource(); source.buffer = buffer;
    source.connect(render.destination); source.start(0, normalized.start, normalized.duration);
    const resampled = await abortable(render.startRendering(), signal);
    samples = Float32Array.from(resampled.getChannelData(0));
  }
  checkAbort(signal);
  return { samples, sampleRate: SAMPLE_RATE, options: normalized, sourceDuration: buffer.duration };
}

export async function transcribeInBrowser(file, options = {}, { onProgress = () => {}, signal } = {}) {
  checkAbort(signal);
  if (analyzing) throw new Error('An audio analysis is already running. Cancel it before starting another.');
  analyzing = true;
  let worker; let disconnectCache;
  try {
    const prepared = await decodeAudioForTranscription(file, options, { onProgress, signal });
    checkAbort(signal);
    worker = new Worker(new URL('./transcription-worker.js', import.meta.url), { type: 'module' });
    disconnectCache = connectWorkerCache(worker);
    return await new Promise((resolve, reject) => {
      const abort = () => { worker.terminate(); finish(abortError()); };
      const timer = setTimeout(() => { worker.terminate(); finish(new Error('Audio analysis exceeded five minutes. Try a shorter excerpt.')); }, 300_000);
      const finish = (error, value) => { clearTimeout(timer); signal?.removeEventListener('abort', abort); error ? reject(error) : resolve(value); };
      signal?.addEventListener('abort', abort, { once: true });
      worker.onerror = event => finish(new Error(event.message || 'The browser transcription worker failed. Reload and retry.'));
      worker.onmessage = ({ data }) => {
        if (data.type === 'progress') onProgress(data.message);
        else if (data.type === 'error') finish(new Error(data.message));
        else if (data.type === 'result') finish(null, data.result);
      };
      worker.postMessage({ type: 'transcribe', samples: prepared.samples, options: prepared.options, name: file.name.replace(/\.[^.]+$/, '') }, [prepared.samples.buffer]);
    });
  } finally { disconnectCache?.(); worker?.terminate(); analyzing = false; }
}
