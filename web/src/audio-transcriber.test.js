import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_FILE_BYTES, validateAudioFile, inspectAudio, decodeAudioForTranscription, transcribeInBrowser } from './audio-transcriber.js';

const file = (name, type = '', size = 1024) => ({ name, type, size, arrayBuffer: async () => new ArrayBuffer(size) });

test('file guard accepts audio formats while native decoding decides codec support', () => {
  for (const name of ['song.mp3', 'song.WAV', 'song.flac', 'song.m4a', 'song.ogg', 'song.aac', 'song.opus', 'song.webm']) assert.doesNotThrow(() => validateAudioFile(file(name)));
  assert.doesNotThrow(() => validateAudioFile(file('recording', 'audio/mp4')));
  assert.throws(() => validateAudioFile(file('data.zip', 'application/zip')), /audio file/);
  assert.throws(() => validateAudioFile(file('song.mp3', 'audio/mpeg', MAX_FILE_BYTES + 1)), /32 MiB/);
  assert.throws(() => validateAudioFile(file('song.mp3', 'audio/mpeg', 0)), /non-empty/);
});

test('already-cancelled analysis never touches audio or starts a worker', async () => {
  const abort = new AbortController(); abort.abort();
  await assert.rejects(transcribeInBrowser(file('song.mp3'), {}, { signal: abort.signal }), { name: 'AbortError' });
});

test('metadata is checked before decode and waveform preparation reuses the decoded buffer', async t => {
  const originalDocument = globalThis.document; const originalContext = globalThis.OfflineAudioContext;
  let decoded = 0; let cleaned = 0; let duration = 1;
  const samples = new Float32Array(22050).fill(.25);
  globalThis.document = { createElement: () => ({
    duration, removeAttribute() {},
    load() { assert.equal(this.onloadedmetadata, null); assert.equal(this.onerror, null); cleaned++; },
    set src(value) { queueMicrotask(() => this.onloadedmetadata?.()); },
  }) };
  globalThis.OfflineAudioContext = class {
    async decodeAudioData() {
      decoded++;
      return { length: samples.length, duration: 1, sampleRate: 22050, numberOfChannels: 1, getChannelData: () => samples };
    }
  };
  t.mock.method(URL, 'createObjectURL', () => 'blob:fixture'); t.mock.method(URL, 'revokeObjectURL', () => {});
  try {
    const source = file('fixture.wav');
    const inspected = await inspectAudio(source);
    assert.equal(inspected.duration, 1); assert.equal(inspected.peaks.length, 800); assert.equal(inspected.peaks[0], .25);
    const prepared = await decodeAudioForTranscription(source, { start: .5, duration: .5 });
    assert.equal(prepared.samples.length, 11025); assert.equal(prepared.samples[0], .25);
    assert.equal(decoded, 1); assert.equal(cleaned, 1);
    duration = 601;
    await assert.rejects(inspectAudio(file('too-long.wav')), /10 minutes/);
    assert.equal(decoded, 1); assert.equal(cleaned, 2);
  } finally {
    if (originalDocument === undefined) delete globalThis.document; else globalThis.document = originalDocument;
    if (originalContext === undefined) delete globalThis.OfflineAudioContext; else globalThis.OfflineAudioContext = originalContext;
  }
});
