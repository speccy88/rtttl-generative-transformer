#!/usr/bin/env node
// Real ONNX Web/WASM inference. Synthetic tones are generated in memory; no
// user recording, mock model, server or Python runtime is needed.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as ort from 'onnxruntime-web/wasm';
import { MODEL_BYTES, MODEL_SHA256, SAMPLE_RATE, WINDOW_SAMPLES, runBasicPitchInference, renderTranscription } from '../src/transcription-core.js';
import { parseRtttl } from '../src/rtttl.js';

const fixtures = [
  { name: 'repeated_intervals', bpm: 120, accompaniment: true, notes: [[60, 1], [60, 1], [67, 1], [67, 1], [69, 1], [69, 1], [67, 2], [65, 1], [65, 1], [64, 1], [64, 1], [62, 1], [62, 1], [60, 2]] },
  { name: 'scale', bpm: 100, notes: [[60, .5], [62, .5], [64, .5], [65, .5], [67, .5], [69, .5], [71, .5], [72, 1], [71, .5], [69, .5], [67, .5], [65, .5], [64, .5], [62, .5], [60, 1]] },
  { name: 'leaps', bpm: 132, accompaniment: true, notes: [[64, 1], [76, 1], [74, .5], [72, .5], [71, 1], [67, 1], [69, .5], [71, .5], [72, 2], [72, .5], [72, .5], [74, 1], [76, 2]] },
  { name: 'short_notes_and_rests', bpm: 90, notes: [[null, .5], [60, 1], [62, .25], [64, .25], [65, .5], [null, .25], [65, .5], [67, .25], [69, .5], [67, 1], [null, .5]] },
  { name: 'short_passing_notes', bpm: 90, notes: [[60, 1], [62, .25], [60, 1], [64, .25], [60, .5], [65, .25], [64, .5], [62, .25], [60, 1]] },
  { name: 'gated_repeated_notes', bpm: 90, notes: [[60, 1], [null, .25], [60, 1], [null, .25], [62, .5], [null, .25], [62, .5], [null, .25], [64, 1]] },
];

function synthesize(fixture) {
  const lengths = fixture.notes.map(([, beats]) => Math.max(1, Math.floor(beats * 60 / fixture.bpm * SAMPLE_RATE)));
  const samples = new Float32Array(lengths.reduce((a, b) => a + b, 0));
  let cursor = 0; let maximum = 0;
  for (let event = 0; event < fixture.notes.length; event++) {
    const [midi] = fixture.notes[event]; const length = lengths[event];
    if (midi !== null) {
      const hz = pitch => 440 * 2 ** ((pitch - 69) / 12);
      const frequency = hz(midi); const bass = hz(midi - (event % 2 === 0 ? 24 : 19));
      const attack = Math.min(length, Math.floor(.015 * SAMPLE_RATE)); const release = Math.min(length, Math.floor(.055 * SAMPLE_RATE));
      for (let i = 0; i < length; i++) {
        const t = i / SAMPLE_RATE;
        const amplitude = .52 * Math.sin(2 * Math.PI * frequency * t) + .18 * Math.sin(4 * Math.PI * frequency * t) + .07 * Math.sin(6 * Math.PI * frequency * t)
          + (fixture.accompaniment ? .11 * Math.sin(2 * Math.PI * bass * t) : 0);
        const envelope = Math.min(1, i / Math.max(1, attack - 1), (length - 1 - i) / Math.max(1, release - 1));
        samples[cursor + i] = amplitude * envelope; maximum = Math.max(maximum, Math.abs(samples[cursor + i]));
      }
    }
    cursor += length;
  }
  // Match the converter's PCM16 WAV fixture quantization.
  return samples.map(sample => Math.floor(sample / (maximum || 1) * .8 * 32768) / 32768);
}

function expand(notes, grid = 16) {
  const frames = []; const onsets = new Set();
  for (const [pitch, beats] of notes) {
    if (frames.length) onsets.add(frames.length);
    frames.push(...new Array(Math.max(1, Math.round(beats * grid / 4))).fill(pitch));
  }
  return { frames, onsets };
}

function metrics(fixture, rtttl) {
  const expected = expand(fixture.notes);
  const song = parseRtttl(rtttl);
  const actual = expand(song.events.map(event => [event.pitch, 4 / event.duration * (event.dotted ? 1.5 : 1)]));
  const length = Math.max(expected.frames.length, actual.frames.length);
  let pitchClass = 0; let octaveAdjusted = 0;
  for (let shift = -48; shift <= 48; shift += 12) {
    let exact = 0;
    for (let i = 0; i < length; i++) {
      if (i >= expected.frames.length || i >= actual.frames.length) continue;
      const a = expected.frames[i]; const b = actual.frames[i];
      if (a === null ? b === null : b !== null && a === b + shift) exact++;
      if (shift === 0 && (a === null ? b === null : b !== null && a % 12 === b % 12)) pitchClass++;
    }
    octaveAdjusted = Math.max(octaveAdjusted, exact / length);
  }
  const unused = new Set(actual.onsets); let matched = 0;
  for (const target of expected.onsets) {
    const closest = [...unused].filter(value => Math.abs(value - target) <= 1).sort((a, b) => Math.abs(a - target) - Math.abs(b - target))[0];
    if (closest !== undefined) { matched++; unused.delete(closest); }
  }
  const onsetF1 = expected.onsets.size + actual.onsets.size ? 2 * matched / (expected.onsets.size + actual.onsets.size) : 1;
  return { pitch_class_accuracy: pitchClass / length, octave_adjusted_midi_accuracy: octaveAdjusted, onset_f1: onsetF1,
    combined: .35 * pitchClass / length + .35 * octaveAdjusted + .30 * onsetF1,
    expected_tokens: fixture.notes.length, actual_tokens: song.events.length,
    rtttl_sha256: createHash('sha256').update(rtttl).digest('hex') };
}

const args = process.argv.slice(2);
const output = args.includes('--output') ? args[args.indexOf('--output') + 1] : null;
const runtime = args.includes('--asyncify') ? 'asyncify' : 'plain';
const root = new URL('../', import.meta.url);
const model = await readFile(new URL('public/transcription/basic-pitch.onnx', root));
if (model.byteLength !== MODEL_BYTES || createHash('sha256').update(model).digest('hex') !== MODEL_SHA256) throw new Error('Pinned model integrity failed.');
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
if (runtime === 'asyncify') {
  const base = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url);
  ort.env.wasm.wasmPaths = { wasm: new URL('ort-wasm-simd-threaded.asyncify.wasm', base).href, mjs: new URL('ort-wasm-simd-threaded.asyncify.mjs', base).href };
  ort.env.wasm.wasmBinary = await readFile(new URL('ort-wasm-simd-threaded.asyncify.wasm', base));
}
const started = performance.now();
const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
const results = [];
try {
  for (const fixture of fixtures) {
    const samples = synthesize(fixture);
    const evidence = await runBasicPitchInference(samples, async window => {
      const input = new ort.Tensor('float32', window, [1, WINDOW_SAMPLES, 1]);
      let outputs;
      try {
        outputs = await session.run({ 'serving_default_input_2:0': input });
        return { note: Float32Array.from(outputs['StatefulPartitionedCall:1'].data), onset: Float32Array.from(outputs['StatefulPartitionedCall:2'].data) };
      } finally { input.dispose(); if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose(); }
    });
    const rendered = renderTranscription(evidence, samples, { bpm: fixture.bpm, duration: samples.length / SAMPLE_RATE, preset: 'faithful' }, fixture.name);
    const candidates = rendered.candidates.map(candidate => ({ name: candidate.name, evidence_score: candidate.score, ...metrics(fixture, candidate.rtttl) }));
    results.push({ fixture: fixture.name, duration_seconds: samples.length / SAMPLE_RATE, detected_events: rendered.evidence.events.length, candidates });
    console.log(`${fixture.name}: selected=${candidates[0].combined.toFixed(4)}, tokens=${candidates[0].actual_tokens}, candidates=${candidates.length}`);
  }
} finally { await session.release(); }
const report = {
  verified_at: new Date().toISOString(), execution: 'Real onnxruntime-web CPU/WASM in Node; no browser audio decode or UI exercised.', runtime,
  node: process.version, ort: JSON.parse(await readFile(new URL('node_modules/onnxruntime-web/package.json', root), 'utf8')).version,
  model: { path: 'web/public/transcription/basic-pitch.onnx', bytes: MODEL_BYTES, sha256: MODEL_SHA256, upstream: 'Spotify basic-pitch 0.4.0 saved_models/icassp_2022/nmp.onnx' },
  source_sha256: Object.fromEntries(await Promise.all(['src/transcription-core.js', 'scripts/verify-browser-transcription.mjs'].map(async path => [path, createHash('sha256').update(await readFile(new URL(path, root))).digest('hex')]))),
  duration_ms: Math.round(performance.now() - started),
  metric: { formula: '0.35 * pitch-class accuracy + 0.35 * octave-adjusted MIDI accuracy + 0.30 * onset F1', grid: 16, onset_tolerance_steps: 1, rests_counted: true, octave_adjustment: 'single global octave shift; no timing stretching', selection: 'first candidate, sorted by model evidence only' },
  results, selected_mean: results.reduce((sum, result) => sum + result.candidates[0].combined, 0) / results.length,
  limits: ['Known BPM is supplied. These deterministic harmonic tones do not measure recognition accuracy on arbitrary full-mix songs.', 'Source decoding, MP3 codec behavior, caching, worker cancellation and browser compatibility require the browser integration test.', 'Browser lead selection is an independently tested JavaScript implementation; no claim of byte-identical Python output.'],
};
if (output) await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(`Selected candidate mean: ${report.selected_mean.toFixed(4)} (${report.duration_ms} ms, ${runtime} WASM)`);
if (report.selected_mean < .95) process.exitCode = 1;
