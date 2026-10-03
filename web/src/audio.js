/*
 * Web Audio note playback adapted from Adam Rahwane's rtttl-play (MIT).
 * Original: https://github.com/adamonsoon/rtttl-play/tree/3611762347eb62eadeecbe2b55838f436a9c834a
 * Changes: audio-clock scheduling, envelopes, progress, cancellation and exports.
 * See ../THIRD_PARTY_NOTICES.md for the preserved attribution and license.
 */
import { parseRtttl, encodeRtttl, MAX_AUDIO_SECONDS } from './rtttl.js';

export const SAMPLE_RATE = 44100;
const LEVEL = 0.22;
const WAVEFORMS = new Set(['sine', 'triangle', 'square']);

function validate(song, waveform) {
  if (!WAVEFORMS.has(waveform)) throw new Error('Choose sine, triangle or square sound.');
  // Recompute timing from the musical events; do not trust imported timing fields.
  return typeof song === 'string' ? parseRtttl(song) : parseRtttl(encodeRtttl(song));
}

function volumeValue(volume) {
  if (!Number.isFinite(volume)) throw new Error('Volume must be a number.');
  return Math.max(0, Math.min(1, volume));
}

// Shared by live playback and OfflineAudioContext exports. Adapted from
// rtttl-play's _playMelody: one oscillator per pitched event, frequency and
// duration read from the parsed score. Rests produce no oscillator.
function scheduleMelody(song, audioCtx, destination, start, waveform) {
  let offset = 0;
  const nodes = [];
  for (const note of song.events) {
    if (note.pitch !== null) {
      const osc = audioCtx.createOscillator();
      osc.type = waveform;
      osc.frequency.value = 440 * 2 ** ((note.pitch - 69) / 12);
      const gain = audioCtx.createGain();
      const from = start + offset;
      const active = note.seconds * 0.92;
      const attack = Math.min(0.008, active / 3);
      const release = Math.min(0.016, active / 3);
      gain.gain.setValueAtTime(0, from);
      gain.gain.linearRampToValueAtTime(LEVEL, from + attack);
      gain.gain.setValueAtTime(LEVEL, from + active - release);
      gain.gain.linearRampToValueAtTime(0, from + active);
      osc.connect(gain);
      gain.connect(destination);
      osc.start(from);
      osc.stop(from + active);
      nodes.push({ osc, gain });
    }
    offset += note.seconds;
  }
  return nodes;
}

export function createPlayer({ onProgress = () => {}, onEnded = () => {} } = {}) {
  let context;
  let output;
  let nodes = [];
  let timer;
  let playing = false;
  let revision = 0;
  let currentVolume = 0.65;

  function stop() {
    revision += 1;
    playing = false;
    clearTimeout(timer);
    for (const { osc, gain } of nodes) {
      try { osc.stop(); } catch { /* A completed oscillator is already stopped. */ }
      osc.disconnect();
      gain.disconnect();
    }
    nodes = [];
    if (output) { output.disconnect(); output = null; }
  }

  async function play(input, { waveform = 'triangle', volume = currentVolume } = {}) {
    const song = validate(input, waveform);
    const nextVolume = volumeValue(volume);
    stop();
    const thisRevision = revision;
    const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AudioContextClass) throw new Error('This browser does not support Web Audio playback.');
    context ??= new AudioContextClass();
    await context.resume();
    if (thisRevision !== revision) return;
    currentVolume = nextVolume;
    output = context.createGain();
    output.gain.value = currentVolume;
    output.connect(context.destination);
    const start = context.currentTime + 0.025;
    nodes = scheduleMelody(song, context, output, start, waveform);
    playing = true;
    const ends = [];
    song.events.reduce((sum, note) => { ends.push(sum + note.seconds); return sum + note.seconds; }, 0);
    function tick() {
      if (thisRevision !== revision) return;
      const elapsed = Math.max(0, Math.min(song.durationSeconds, context.currentTime - start));
      const found = ends.findIndex(end => end > elapsed);
      onProgress({ elapsed, duration: song.durationSeconds, progress: elapsed / song.durationSeconds, eventIndex: found });
      if (elapsed >= song.durationSeconds) {
        stop();
        onEnded();
      } else timer = setTimeout(tick, 40);
    }
    tick();
  }

  function setVolume(volume) {
    currentVolume = volumeValue(volume);
    if (output) output.gain.setTargetAtTime(currentVolume, context.currentTime, 0.015);
  }

  return {
    play, stop, setVolume,
    get isPlaying() { return playing; },
    async dispose() { stop(); if (context) { await context.close(); context = null; } },
  };
}

export async function renderPcm(input, { waveform = 'triangle', volume = 0.8 } = {}) {
  const song = validate(input, waveform);
  const level = volumeValue(volume);
  if (song.durationSeconds > MAX_AUDIO_SECONDS) throw new Error('This song is too long to export.');
  const OfflineContext = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  if (!OfflineContext) throw new Error('This browser does not support offline audio export.');
  const context = new OfflineContext(1, Math.ceil(song.durationSeconds * SAMPLE_RATE), SAMPLE_RATE);
  const output = context.createGain();
  output.gain.value = level;
  output.connect(context.destination);
  scheduleMelody(song, context, output, 0, waveform);
  const rendered = await context.startRendering();
  const floatSamples = rendered.getChannelData(0);
  const samples = new Int16Array(floatSamples.length);
  for (let i = 0; i < samples.length; i += 1) samples[i] = Math.round(Math.max(-1, Math.min(1, floatSamples[i])) * 32767);
  return { samples, sampleRate: SAMPLE_RATE };
}

export function encodeWavPcm(samples, sampleRate = SAMPLE_RATE) {
  if (!(samples instanceof Int16Array)) throw new Error('WAV input must be 16-bit PCM.');
  if (!Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 96000) throw new Error('Invalid sample rate.');
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (position, value) => [...value].forEach((character, i) => view.setUint8(position + i, character.charCodeAt(0)));
  writeText(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) view.setInt16(44 + i * 2, samples[i], true);
  return new Blob([buffer], { type: 'audio/wav' });
}

export async function renderWav(song, options = {}) {
  const { samples, sampleRate } = await renderPcm(song, options);
  return encodeWavPcm(samples, sampleRate);
}

export async function renderMp3(song, { onProgress = () => {}, ...options } = {}) {
  onProgress(0);
  const { samples, sampleRate } = await renderPcm(song, options);
  onProgress(0.05);
  // Encoder is a separate, replaceable LGPL module, downloaded only on export.
  if (typeof Worker === 'undefined') {
    const { encodeMp3Pcm } = await import('./mp3-worker.js');
    return encodeMp3Pcm(samples, sampleRate, progress => onProgress(0.05 + progress * 0.95));
  }
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./mp3-worker.js', import.meta.url), { type: 'module' });
    const finish = (error, blob) => { worker.terminate(); error ? reject(error) : resolve(blob); };
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') onProgress(0.05 + data.progress * 0.95);
      else if (data.type === 'done') { onProgress(1); finish(null, data.blob); }
      else if (data.type === 'error') finish(new Error(data.message));
    };
    worker.onerror = () => finish(new Error('The MP3 encoder could not load. Try WAV export or reload the page.'));
    worker.postMessage({ samples, sampleRate }, [samples.buffer]);
  });
}
