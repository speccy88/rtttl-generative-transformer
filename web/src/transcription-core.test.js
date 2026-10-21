import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_FRAMES, PITCH_BINS, MODEL_BYTES, MODEL_SHA256,
  validateTranscriptionOptions, runBasicPitchInference, modelFrameTime, extractNoteEvents, buildMelodyGrid, selectLead, renderTranscription } from './transcription-core.js';
import { parseRtttl } from './rtttl.js';

function fakeEvidence(events, duration) {
  const frameCount = Math.floor(duration * 86);
  const note = new Float32Array(frameCount * PITCH_BINS); const onset = new Float32Array(note.length);
  for (const { start, end, midi, confidence = .8 } of events) {
    let first = null;
    for (let t = 0; t < frameCount; t++) if (modelFrameTime(t) >= start && modelFrameTime(t) < end) {
      note[t * PITCH_BINS + midi - 21] = confidence; first ??= t;
    }
    if (first !== null) onset[first * PITCH_BINS + midi - 21] = .9;
  }
  return { note, onset, frameCount, duration };
}

test('excerpt and parameter limits are enforced before model work', () => {
  assert.equal(validateTranscriptionOptions({ duration: 10, start: 9 }, 10).duration, 1);
  assert.equal(validateTranscriptionOptions({}).grid, 16);
  assert.throws(() => validateTranscriptionOptions({ duration: 31 }), /duration/);
  assert.throws(() => validateTranscriptionOptions({ start: 10 }, 10), /remaining/);
  assert.throws(() => validateTranscriptionOptions({ bpm: NaN }), /bpm/);
  assert.throws(() => validateTranscriptionOptions({ grid: 7 }), /grid/);
  assert.throws(() => validateTranscriptionOptions({ source_mode: 'harmonic' }), /full mix/);
});

test('window padding and overlap crop preserve the first and last model frames', async () => {
  const samples = new Float32Array(SAMPLE_RATE * 2).fill(.5); let calls = 0;
  const result = await runBasicPitchInference(samples, async window => {
    assert.equal(window.length, WINDOW_SAMPLES);
    if (!calls) { assert.equal(window[3839], 0); assert.equal(window[3840], .5); }
    const values = Float32Array.from({ length: WINDOW_FRAMES * PITCH_BINS }, (_, i) => calls * 1000 + Math.floor(i / PITCH_BINS));
    calls++; return { note: values, onset: values };
  });
  assert.equal(calls, 2); assert.equal(result.frameCount, 172);
  assert.equal(result.note[0], 15);
  assert.equal(result.note[141 * PITCH_BINS], 156);
  assert.equal(result.note[142 * PITCH_BINS], 1015);
  assert.equal(result.note[171 * PITCH_BINS], 1044);
});

test('clips near a window boundary never write beyond their cropped output', async () => {
  let calls = 0;
  const result = await runBasicPitchInference(new Float32Array(Math.round(1.55 * SAMPLE_RATE)), async () => {
    calls++; return { note: new Float32Array(WINDOW_FRAMES * PITCH_BINS), onset: new Float32Array(WINDOW_FRAMES * PITCH_BINS) };
  });
  assert.equal(calls, 1); assert.equal(result.frameCount, 133);
  await assert.rejects(runBasicPitchInference(new Float32Array(SAMPLE_RATE * 31), () => {}), /30 seconds/);
});

test('lead decoder follows a consistent phrase across a louder transient competitor', () => {
  const slots = [.8, .7, .8].map((strength, i) => new Map([[60, strength], [79, i === 1 ? .9 : .1]]));
  const attacks = slots.map(() => new Map());
  assert.deepEqual(selectLead(slots, attacks, { voiceBias: 0 }), [60, 60, 60]);
});

test('nearest-grid boundaries retain short notes and actual silence', () => {
  const events = [{ start: .35, end: .99, midi: 60, confidence: .8 }, { start: 1.02, end: 1.16, midi: 62, confidence: .8 }];
  const { slots, attacks } = buildMelodyGrid(events, 1.5, 90, 16);
  assert.equal(slots.length, 9); assert.equal(slots[0].size, 0); assert.equal(slots[1].size, 0);
  assert.ok(attacks[6].has(62)); assert.ok(slots[6].has(62)); assert.equal(slots[7].size, 0);
});

test('onset-backed short passing notes survive cleanup and RTTTL encoding', () => {
  const events = [{ start: 0, end: 2 / 3, midi: 60 }, { start: 2 / 3, end: 5 / 6, midi: 62 }, { start: 5 / 6, end: 1.5, midi: 60 }];
  const evidence = fakeEvidence(events, 1.5);
  const extracted = extractNoteEvents(evidence);
  assert.ok(extracted.some(event => event.midi === 62 && event.end - event.start < .2));
  const result = renderTranscription(evidence, new Float32Array(SAMPLE_RATE * 1.5).fill(.2), { bpm: 90, duration: 1.5 });
  const song = parseRtttl(result.candidates[0].rtttl);
  assert.deepEqual(song.events.map(event => event.pitch), [60, 62, 60]);
  assert.equal(song.events[1].duration, 16); assert.equal(song.durationSeconds, 1.5);
});

test('source silence veto preserves leading, internal and trailing rests even under model sustain', () => {
  const duration = 2; const samples = new Float32Array(SAMPLE_RATE * duration);
  samples.fill(.2, SAMPLE_RATE / 4, SAMPLE_RATE); samples.fill(.2, SAMPLE_RATE * 1.25, SAMPLE_RATE * 1.75);
  const evidence = fakeEvidence([{ start: 0, end: duration, midi: 60 }], duration);
  const result = renderTranscription(evidence, samples, { bpm: 120, duration });
  const song = parseRtttl(result.candidates[0].rtttl);
  assert.deepEqual(song.events.map(event => event.pitch), [null, 60, null, 60, null]);
  assert.deepEqual(song.events.map(event => event.seconds), [.25, .75, .25, .5, .25]);
  assert.equal(song.durationSeconds, duration); assert.equal(result.candidates[0].diagnostics.source_offset_seconds, 0);
});

test('bundled model bytes and provenance match the pinned upstream asset', async () => {
  const bytes = await readFile(new URL('../public/transcription/basic-pitch.onnx', import.meta.url));
  assert.equal(bytes.length, MODEL_BYTES); assert.equal(createHash('sha256').update(bytes).digest('hex'), MODEL_SHA256);
  const notice = await readFile(new URL('../public/transcription/NOTICE.md', import.meta.url), 'utf8');
  assert.match(notice, /spotify\/basic-pitch/i); assert.ok(notice.includes(MODEL_SHA256));
  const license = await readFile(new URL('../public/transcription/APACHE-2.0.txt', import.meta.url), 'utf8');
  assert.match(license, /Apache License/);
});
