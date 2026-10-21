/*
 * Basic Pitch windowing and note-event extraction adapted from Spotify Basic
 * Pitch 0.4.0 (Apache-2.0). See public/transcription/NOTICE.md and APACHE-2.0.txt.
 * The source-aware monophonic decoder is adapted from speccy88/mp3_to_rtttl.
 */
import { encodeRtttl, parseRtttl } from './rtttl.js';

export const SAMPLE_RATE = 22050;
export const WINDOW_SAMPLES = 43844;
export const WINDOW_FRAMES = 172;
export const OVERLAP_FRAMES = 30;
export const WINDOW_HOP = WINDOW_SAMPLES - OVERLAP_FRAMES * 256;
export const PITCH_BINS = 88;
export const MAX_FILE_BYTES = 32 * 1024 * 1024;
export const MAX_SOURCE_SECONDS = 600;
export const MAX_EXCERPT_SECONDS = 30;
export const MODEL_SHA256 = '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec';
export const MODEL_BYTES = 230444;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function validateTranscriptionOptions(options = {}, sourceDuration = Infinity) {
  const number = (key, fallback, low, high) => {
    const value = Number(options[key] ?? fallback);
    if (!Number.isFinite(value) || value < low || value > high) throw new Error(`${key} must be between ${low} and ${high}.`);
    return value;
  };
  const start = number('start', 0, 0, MAX_SOURCE_SECONDS);
  const requestedDuration = number('duration', 12, 0.25, MAX_EXCERPT_SECONDS);
  const duration = Math.min(requestedDuration, sourceDuration - start);
  if (duration < 0.25) throw new Error('Select an excerpt with at least 0.25 seconds of audio remaining.');
  const bpm = options.bpm === null || options.bpm === undefined || options.bpm === '' ? null : number('bpm', 120, 25, 900);
  const preset = options.preset ?? 'faithful';
  if (!['faithful', 'balanced', 'smooth'].includes(preset)) throw new Error('Choose a faithful, balanced or smooth preset.');
  const source_mode = options.source_mode ?? 'auto';
  if (!['auto', 'full'].includes(source_mode)) throw new Error('Browser conversion analyzes the full mix. Select Full mix or use an isolated lead/vocal recording.');
  const leadPreference = options.leadPreference ?? 'balanced';
  if (!['balanced', 'upper', 'lower'].includes(leadPreference)) throw new Error('Choose a balanced, upper or lower lead preference.');
  const grid = Number(options.grid ?? 16);
  if (![16, 32].includes(grid)) throw new Error('Choose a 16th-note or 32nd-note grid.');
  return { start, duration, bpm, preset, source_mode, leadPreference, grid };
}

/** Same 30-frame overlap/15-frame trim contract as Basic Pitch's Python loader. */
export async function runBasicPitchInference(samples, predict, onProgress = () => {}) {
  if (!(samples instanceof Float32Array) || !samples.length || samples.length > SAMPLE_RATE * MAX_EXCERPT_SECONDS) throw new Error('Provide at most 30 seconds of mono 22050 Hz samples.');
  const padding = OVERLAP_FRAMES * 256 / 2;
  const frameCount = Math.max(1, Math.floor(samples.length * 86 / SAMPLE_RATE));
  const note = new Float32Array(frameCount * PITCH_BINS);
  const onset = new Float32Array(note.length);
  const retainedFrames = WINDOW_FRAMES - OVERLAP_FRAMES;
  // Some last padded windows contain only frames discarded by the final crop.
  // Skip them: even TypedArray.set(empty, offset) rejects an out-of-range offset.
  const totalWindows = Math.ceil(frameCount / retainedFrames);
  for (let windowIndex = 0; windowIndex < totalWindows; windowIndex++) {
    const window = new Float32Array(WINDOW_SAMPLES);
    const sourceStart = windowIndex * WINDOW_HOP - padding;
    const first = Math.max(0, sourceStart);
    const last = Math.min(samples.length, sourceStart + WINDOW_SAMPLES);
    if (last > first) window.set(samples.subarray(first, last), first - sourceStart);
    const output = await predict(window);
    if (output.note?.length !== WINDOW_FRAMES * PITCH_BINS || output.onset?.length !== WINDOW_FRAMES * PITCH_BINS) throw new Error('The transcription model returned an unexpected output shape.');
    const targetFrame = windowIndex * retainedFrames;
    const count = Math.max(0, Math.min(retainedFrames, frameCount - targetFrame));
    const offset = OVERLAP_FRAMES / 2 * PITCH_BINS;
    note.set(output.note.subarray(offset, offset + count * PITCH_BINS), targetFrame * PITCH_BINS);
    onset.set(output.onset.subarray(offset, offset + count * PITCH_BINS), targetFrame * PITCH_BINS);
    onProgress(`Analyzing audio locally… ${windowIndex + 1}/${totalWindows}`);
  }
  return { note, onset, frameCount, duration: samples.length / SAMPLE_RATE };
}

export function modelFrameTime(frame) {
  const offset = (256 / SAMPLE_RATE) * (WINDOW_FRAMES - WINDOW_SAMPLES / 256) + 0.0018;
  return frame * 256 / SAMPLE_RATE - offset * Math.floor(frame / WINDOW_FRAMES);
}

/** Onset-led polyphonic events, followed by a remaining-salience fallback. */
export function extractNoteEvents(evidence, { minimumNoteMs = 100, onsetThreshold = 0.45, frameThreshold = 0.25, minConfidence = 0.32 } = {}) {
  const { note, onset, frameCount } = evidence;
  const energy = Float32Array.from(note);
  const attacks = Float32Array.from(onset);
  const differences = new Float32Array(note.length);
  let maxDifference = 0; let maxOnset = 0;
  for (let t = 0; t < frameCount; t++) for (let p = 0; p < PITCH_BINS; p++) {
    const index = t * PITCH_BINS + p;
    if (p + 21 < 36 || p + 21 > 107) { energy[index] = 0; attacks[index] = 0; continue; }
    maxOnset = Math.max(maxOnset, attacks[index]);
    if (t >= 2) {
      const difference = Math.max(0, Math.min(note[index] - note[index - PITCH_BINS], note[index] - note[index - 2 * PITCH_BINS]));
      differences[index] = difference; maxDifference = Math.max(maxDifference, difference);
    }
  }
  const peaks = [];
  const scale = maxDifference > 0 ? maxOnset / maxDifference : 0;
  for (let i = 0; i < attacks.length; i++) attacks[i] = Math.max(attacks[i], differences[i] * scale);
  for (let t = 1; t < frameCount - 1; t++) for (let p = 15; p < PITCH_BINS - 1; p++) {
    const i = t * PITCH_BINS + p;
    if (attacks[i] >= onsetThreshold && attacks[i] > attacks[i - PITCH_BINS] && attacks[i] > attacks[i + PITCH_BINS]) peaks.push(i);
  }
  const minimumFrames = Math.max(1, Math.round(minimumNoteMs / 1000 * SAMPLE_RATE / 256));
  const events = [];
  const clear = (t, p) => {
    energy[t * PITCH_BINS + p] = 0;
    if (p > 0) energy[t * PITCH_BINS + p - 1] = 0;
    if (p + 1 < PITCH_BINS) energy[t * PITCH_BINS + p + 1] = 0;
  };
  const add = (startFrame, endFrame, p) => {
    if (endFrame - startFrame <= minimumFrames) return;
    let strength = 0;
    for (let t = startFrame; t < endFrame; t++) strength += note[t * PITCH_BINS + p];
    strength /= endFrame - startFrame;
    const start = Math.max(0, modelFrameTime(startFrame));
    const end = Math.min(evidence.duration, modelFrameTime(endFrame));
    if (strength >= minConfidence && (end - start) * 1000 >= minimumNoteMs) events.push({ start, end, midi: p + 21, confidence: strength });
  };
  for (let peak = peaks.length - 1; peak >= 0; peak--) {
    const start = Math.floor(peaks[peak] / PITCH_BINS); const p = peaks[peak] % PITCH_BINS;
    let end = start + 1; let quiet = 0;
    while (end < frameCount - 1 && quiet < 11) { quiet = energy[end * PITCH_BINS + p] < frameThreshold ? quiet + 1 : 0; end++; }
    end -= quiet;
    if (end - start <= minimumFrames) continue;
    for (let t = start; t < end; t++) clear(t, p);
    add(start, end, p);
  }
  // Sorting once is equivalent to repeatedly selecting the strongest remaining
  // entry: this stage only clears evidence, never increases it.
  const remaining = [];
  for (let i = 0; i < energy.length; i++) if (energy[i] > frameThreshold) remaining.push(i);
  remaining.sort((a, b) => energy[b] - energy[a] || a - b);
  for (const index of remaining) {
    if (energy[index] <= frameThreshold) continue;
    const middle = Math.floor(index / PITCH_BINS); const p = index % PITCH_BINS;
    energy[index] = 0;
    let cursor = middle + 1; let quiet = 0;
    while (cursor < frameCount - 1 && quiet < 11) { quiet = energy[cursor * PITCH_BINS + p] < frameThreshold ? quiet + 1 : 0; clear(cursor, p); cursor++; }
    const end = cursor - 1 - quiet;
    cursor = middle - 1; quiet = 0;
    while (cursor > 0 && quiet < 11) { quiet = energy[cursor * PITCH_BINS + p] < frameThreshold ? quiet + 1 : 0; clear(cursor, p); cursor--; }
    add(cursor + 1 + quiet, end, p);
  }
  return events.sort((a, b) => a.start - b.start || a.midi - b.midi);
}

export function estimateTempo(evidence) {
  const flux = new Float64Array(evidence.frameCount);
  for (let t = 0; t < flux.length; t++) {
    let maximum = 0;
    for (let p = 15; p < PITCH_BINS - 1; p++) maximum = Math.max(maximum, evidence.onset[t * PITCH_BINS + p]);
    flux[t] = Math.max(0, maximum - 0.12);
  }
  const mean = flux.reduce((a, b) => a + b, 0) / Math.max(1, flux.length);
  if (mean < 0.001 || flux.length < 40) return 120;
  for (let i = 0; i < flux.length; i++) flux[i] -= mean;
  const fps = evidence.frameCount / evidence.duration;
  let best = -Infinity; let tempo = 120;
  for (let lag = Math.max(1, Math.floor(60 * fps / 190)); lag <= Math.min(flux.length - 1, Math.ceil(60 * fps / 60)); lag++) {
    let correlation = 0;
    for (let i = lag; i < flux.length; i++) correlation += flux[i] * flux[i - lag];
    const bpm = 60 * fps / lag;
    const prior = Math.exp(-0.5 * (Math.log2(bpm / 110) / 0.65) ** 2);
    const score = correlation / (flux.length - lag) * prior;
    if (score > best) { best = score; tempo = bpm; }
  }
  return tempo;
}

export function buildMelodyGrid(events, duration, bpm, grid) {
  const stepSeconds = 240 / bpm / grid;
  const count = Math.max(1, Math.round(duration / stepSeconds));
  const slots = Array.from({ length: count }, () => new Map());
  const attacks = Array.from({ length: count }, () => new Map());
  for (const event of events) {
    const first = clamp(Math.round(event.start / stepSeconds), 0, count - 1);
    const last = Math.min(count, Math.max(first + 1, Math.round(event.end / stepSeconds)));
    attacks[first].set(event.midi, Math.max(attacks[first].get(event.midi) ?? 0, event.confidence));
    for (let t = first; t < last; t++) {
      const overlap = Math.max(0, Math.min(event.end, (t + 1) * stepSeconds) - Math.max(event.start, t * stepSeconds));
      const weight = overlap > 0 ? overlap / stepSeconds : Math.min(1, (event.end - event.start) / stepSeconds);
      slots[t].set(event.midi, Math.max(slots[t].get(event.midi) ?? 0, event.confidence * weight));
    }
  }
  return { slots, attacks, stepSeconds };
}

/** Dynamic programming considers the whole excerpt rather than a frame-wise max. */
export function selectLead(slots, attacks, { voiceBias = 0.10, changePenalty = 0.14, jumpPenalty = 0.055, sustainBonus = 0.42 } = {}) {
  let previous = new Map([[null, 0]]);
  const backs = [];
  for (let t = 0; t < slots.length; t++) {
    const states = slots[t].size ? [...slots[t].keys()] : [null];
    const scores = new Map(); const back = new Map(); const onsets = attacks[t];
    for (const pitch of states) {
      const emission = pitch === null ? 0 : 2.5 * slots[t].get(pitch) + voiceBias * (pitch - 60) + (onsets.has(pitch) ? 0.7 * (0.5 + onsets.get(pitch)) : 0);
      let best = -Infinity; let predecessor = null;
      for (const [before, score] of previous) {
        let transition;
        if (pitch === null && before === null) transition = 0.15;
        else if (pitch === null || before === null) transition = -0.2;
        else if (pitch === before) transition = onsets.size ? 0.15 : sustainBonus;
        else transition = -changePenalty - jumpPenalty * Math.abs(pitch - before) + (onsets.has(pitch) ? 0.24 : 0);
        if (score + transition + emission > best) { best = score + transition + emission; predecessor = before; }
      }
      scores.set(pitch, best); back.set(pitch, predecessor);
    }
    previous = scores; backs.push(back);
  }
  let pitch = [...previous].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  const path = new Array(slots.length);
  for (let t = slots.length - 1; t >= 0; t--) { path[t] = pitch; pitch = backs[t].get(pitch); }
  return path;
}

export function selectUpperLead(slots, attacks) {
  let previous = null;
  return slots.map((slot, step) => {
    const credible = [...slot].filter(([pitch, strength]) => !(slot.has(pitch - 12) && slot.get(pitch - 12) >= strength * 1.2 && previous !== pitch - 12));
    const strongest = Math.max(0, ...credible.map(([, strength]) => strength));
    let selected = credible.filter(([, strength]) => strength >= strongest * 0.4).reduce((best, [pitch]) => best === null ? pitch : Math.max(best, pitch), null);
    if (selected === previous && previous !== null && attacks[step].size) {
      const peak = Math.max(...attacks[step].values());
      const fresh = [...attacks[step]].filter(([pitch, confidence]) => pitch !== previous && Math.abs(pitch - previous) <= 8 && confidence >= peak * 0.75);
      if (fresh.length) selected = Math.max(...fresh.map(([pitch]) => pitch));
    }
    previous = selected;
    return selected;
  });
}

function cleanPath(path, attacks, minimumSteps) {
  const result = [...path];
  for (let t = 1; t + 1 < result.length; t++) if (result[t] !== null && !attacks[t].has(result[t]) && result[t - 1] === result[t + 1] && result[t - 1] !== result[t]) result[t] = result[t - 1];
  if (minimumSteps <= 1) return result;
  for (let pass = 0; pass < result.length; pass++) {
    let changed = false;
    for (let start = 0; start < result.length;) {
      let end = start + 1;
      while (end < result.length && result[end] === result[start]) end++;
      const pitch = result[start];
      if (pitch !== null && end - start < minimumSteps && !attacks.slice(start, end).some(slot => slot.has(pitch))) {
        const neighbors = [start ? result[start - 1] : null, end < result.length ? result[end] : null].filter(p => p !== null);
        if (neighbors.length) { const replacement = neighbors.sort((a, b) => Math.abs(a - pitch) - Math.abs(b - pitch))[0]; result.fill(replacement, start, end); changed = true; break; }
      }
      start = end;
    }
    if (!changed) break;
  }
  return result;
}

function silenceMask(samples, stepSeconds, count) {
  const hop = Math.round(SAMPLE_RATE * 0.01);
  const rms = new Float32Array(Math.ceil(samples.length / hop)); let peak = 0;
  for (let frame = 0; frame < rms.length; frame++) {
    let square = 0; const end = Math.min(samples.length, (frame + 1) * hop);
    for (let i = frame * hop; i < end; i++) square += samples[i] ** 2;
    rms[frame] = Math.sqrt(square / hop); peak = Math.max(peak, rms[frame]);
  }
  return Array.from({ length: count }, (_, step) => {
    const first = Math.floor(step * stepSeconds * SAMPLE_RATE / hop);
    const last = Math.min(rms.length, Math.ceil((step + 1) * stepSeconds * SAMPLE_RATE / hop));
    let silent = 0;
    for (let i = first; i < last; i++) if (rms[i] <= peak * 0.005) silent++;
    return last > first && silent / (last - first) >= 0.75;
  });
}

function fitOctaves(path, compact = false) {
  const pitches = path.filter(p => p !== null);
  let bestShift = 0; let bestInside = -1;
  for (let shift = -48; shift <= 48; shift += 12) {
    const inside = pitches.filter(p => p + shift >= 60 && p + shift <= 107).length;
    if (inside > bestInside || (inside === bestInside && Math.abs(shift) < Math.abs(bestShift))) { bestInside = inside; bestShift = shift; }
  }
  let previous = null;
  return path.map(pitch => {
    if (pitch === null) { previous = null; return null; }
    let value = pitch + bestShift;
    while (value < 60) value += 12;
    while (value > 107) value -= 12;
    if (compact && previous !== null && Math.abs(value - previous) > 12) {
      const choices = [];
      for (let p = 60 + pitch % 12; p <= 107; p += 12) choices.push(p);
      value = choices.sort((a, b) => Math.abs(a - previous) + 0.5 * Math.abs(a - value) - Math.abs(b - previous) - 0.5 * Math.abs(b - value))[0];
    }
    previous = value; return value;
  });
}

function durationParts(steps, grid) {
  const options = [];
  for (const duration of [1, 2, 4, 8, 16, 32]) for (const dotted of [false, true]) {
    const units = grid / duration * (dotted ? 1.5 : 1);
    if (Number.isInteger(units) && units >= 1) options.push({ units, duration, dotted });
  }
  options.sort((a, b) => b.units - a.units);
  const best = new Array(steps + 1).fill(null); best[0] = [];
  for (let n = 1; n <= steps; n++) for (const option of options) if (option.units <= n && best[n - option.units] !== null) {
    const candidate = [...best[n - option.units], option];
    if (best[n] === null || candidate.length < best[n].length) best[n] = candidate;
  }
  return best[steps];
}

function pathToSong(path, attacks, bpm, grid, name) {
  const runs = []; let start = 0;
  for (let t = 1; t <= path.length; t++) {
    const reattack = t < path.length && path[t] !== null && path[t] === path[start] && t - start >= 2 && [...attacks[t].keys()].some(p => p % 12 === path[t] % 12);
    if (t === path.length || path[t] !== path[start] || reattack) { runs.push({ pitch: path[start], steps: t - start }); start = t; }
  }
  return { name, bpm, defaultDuration: 4, defaultOctave: 5, events: runs.flatMap(run => durationParts(run.steps, grid).map(({ duration, dotted }) => ({ pitch: run.pitch, duration, dotted }))) };
}

function candidateEvidence(song, evidence) {
  let best = 0; const total = song.events.reduce((sum, event) => sum + event.seconds, 0);
  for (let shift = -48; shift <= 48; shift += 12) {
    let cursor = 0; let support = 0; let pitchedTime = 0; let attacks = 0; let attackWeight = 0;
    for (const event of song.events) {
      const first = Math.floor(cursor / evidence.duration * evidence.frameCount);
      cursor += event.seconds;
      const last = Math.min(evidence.frameCount, Math.ceil(cursor / evidence.duration * evidence.frameCount));
      if (event.pitch === null) continue;
      pitchedTime += event.seconds;
      const p = event.pitch + shift - 21;
      if (p < 0 || p >= PITCH_BINS || last <= first) continue;
      let sum = 0; let maximum = 0;
      for (let t = first; t < last; t++) { const value = evidence.note[t * PITCH_BINS + p]; sum += value; maximum = Math.max(maximum, value); }
      support += (0.6 * sum / (last - first) + 0.4 * maximum) * event.seconds;
      let onset = 0;
      for (let t = Math.max(0, first - 3); t < Math.min(evidence.frameCount, first + 4); t++) onset = Math.max(onset, evidence.onset[t * PITCH_BINS + p]);
      attacks += onset; attackWeight++;
    }
    if (pitchedTime) best = Math.max(best, 0.8 * support / pitchedTime + 0.2 * attacks / Math.max(1, attackWeight));
  }
  return clamp(best * Math.min(total, evidence.duration) / Math.max(total, evidence.duration), 0, 1);
}

export function renderTranscription(evidence, samples, inputOptions = {}, name = 'Converted') {
  const options = validateTranscriptionOptions(inputOptions);
  const estimatedBpm = options.bpm ?? estimateTempo(evidence);
  const bpm = Math.round(clamp(estimatedBpm, 25, 900));
  const events = extractNoteEvents(evidence);
  if (!events.length) throw new Error('No clear notes were detected. Try a shorter passage with a clear lead or an isolated vocal/instrument recording.');
  const { slots, attacks, stepSeconds } = buildMelodyGrid(events, evidence.duration, bpm, options.grid);
  const silent = silenceMask(samples, stepSeconds, slots.length);
  const voiceBias = options.leadPreference === 'lower' ? -0.10 : options.leadPreference === 'upper' ? 0.14 : 0.10;
  const profiles = [
    { name: options.leadPreference === 'lower' ? 'Lower lead' : 'Faithful melody', path: selectLead(slots, attacks, { voiceBias }), compact: options.preset !== 'faithful' },
    { name: 'Upper voice', path: selectUpperLead(slots, attacks), compact: false },
    { name: 'Smooth melody', path: selectLead(slots, attacks, { voiceBias, changePenalty: 0.26, jumpPenalty: 0.08, sustainBonus: 0.55 }), compact: true },
  ];
  const candidates = []; const signatures = new Set();
  for (const profile of profiles) {
    let path = cleanPath(profile.path, attacks, options.preset === 'smooth' ? 3 : options.preset === 'balanced' ? 2 : 1);
    path = path.map((pitch, i) => silent[i] ? null : pitch);
    if (!path.some(pitch => pitch !== null)) continue;
    path = fitOctaves(path, profile.compact);
    const encoded = encodeRtttl(pathToSong(path, attacks, bpm, options.grid, name));
    const signature = encoded.slice(encoded.indexOf(':'));
    if (signatures.has(signature)) continue;
    signatures.add(signature);
    const song = parseRtttl(encoded); let cursor = 0;
    const notes = song.events.map(event => {
      const start = cursor; cursor += event.seconds;
      const gridStart = Math.max(0, Math.floor(start / stepSeconds));
      const gridEnd = Math.min(slots.length, Math.ceil(cursor / stepSeconds));
      const strengths = new Map();
      for (let step = gridStart; step < gridEnd; step++) for (const [midi, strength] of slots[step]) strengths.set(midi, Math.max(strengths.get(midi) ?? 0, strength));
      const alternatives = [...strengths].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([midi, confidence]) => ({ midi, confidence }));
      const matching = alternatives.filter(item => event.pitch !== null && item.midi % 12 === event.pitch % 12);
      return { midi: event.pitch, start, duration: event.seconds, confidence: matching[0]?.confidence ?? 0, alternatives };
    });
    candidates.push({ name: profile.name, rtttl: encoded, score: candidateEvidence(song, evidence), notes,
      settings: { ...options, minimum_note_ms: 100, repeated_note_min_steps: 2, octave_smoothing: profile.compact },
      diagnostics: { estimated_bpm: estimatedBpm, rtttl_bpm: bpm, detected_events: events.length, rtttl_tokens: song.events.length, source_offset_seconds: 0, duration_seconds: song.durationSeconds },
    });
  }
  if (!candidates.length) throw new Error('No melody remained outside near-silent regions. Try a clearer passage.');
  candidates.sort((a, b) => b.score - a.score);
  const warnings = ['Source evidence measures agreement with detected notes, not recognition accuracy. Compare candidates with the original.', 'Full mixes may follow accompaniment. A clear lead or isolated vocal/instrument recording usually works better.'];
  if (options.bpm === null) warnings.push('Tempo is estimated from detected attacks and can be half or double time. Enter the known BPM when possible.');
  if (options.grid === 32) warnings.push('A fine 32nd-note grid can expose false re-attacks. Use 16th notes if playback stutters.');
  return { candidates, analysis: { estimated_bpm: estimatedBpm, source_mode: 'full', start: options.start, duration: evidence.duration, model: 'Spotify Basic Pitch 0.4.0', backend: 'wasm' }, warnings,
    evidence: { events: events.slice(0, 2048), frameCount: evidence.frameCount, sampleRate: SAMPLE_RATE },
  };
}
