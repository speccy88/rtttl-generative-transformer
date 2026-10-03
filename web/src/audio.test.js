import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRtttl, encodeRtttl } from './rtttl.js';
import { createPlayer, renderPcm, encodeWavPcm } from './audio.js';
import { encodeMp3Pcm } from './mp3-worker.js';

test('RTTTL pitches, variable BPM and dotted/rest timing follow the score', () => {
  const song = parseRtttl('Echo:d=8,o=5,b=117:c,4d#.,p,16a.6,16b6.');
  assert.deepEqual(song.events.map(event => event.pitch), [72, 75, null, 93, 95]);
  assert.deepEqual(song.events.map(event => event.duration), [8, 4, 8, 16, 16]);
  assert.deepEqual(song.events.map(event => event.dotted), [false, true, false, true, true]);
  assert.equal(song.events[1].seconds, 60 / 117 * 1.5);
  assert.equal(song.durationSeconds, song.events.reduce((sum, event) => sum + event.seconds, 0));
});

test('parser accepts upstream defaults and H spelling, with exact export roundtrip', () => {
  const song = parseRtttl('Rain::c,h,p');
  assert.equal(song.bpm, 63);
  assert.deepEqual(song.events.map(event => event.pitch), [84, 95, null]);
  assert.deepEqual(parseRtttl(encodeRtttl(song)).events, song.events);
  assert.equal(parseRtttl('UPPER : D=4, O=4, B=120 : C#, P, E').events[0].pitch, 61);
});

test('parser rejects malformed notes, settings and allocation-sized scores', () => {
  for (const text of [
    'no sections', 'x:d=8,o=6,b=0:c', 'x:b=901:c', 'x:d=3:c', 'x:o=9:c',
    'x:b=120,b=120:c', 'x:q=4:c', 'x:b=NaN:c', 'x::cfoo', 'x::c,,d',
    'x::p#', 'x::c..', 'x::c6..', 'x::b#7', 'x::',
    `x:d=1,b=25:${Array(33).fill('c').join(',')}`,
    `x:b=900:${Array(2049).fill('64c').join(',')}`,
  ]) assert.throws(() => parseRtttl(text), undefined, text);
});

test('encoder uses compact safe names and validates manually constructed events', () => {
  const song = parseRtttl('Song:d=8,o=6,b=100:c,c#,p');
  const encoded = encodeRtttl({ ...song, name: '<Summer: Écho>' });
  assert.match(encoded, /^Summer Echo:/);
  assert.deepEqual(parseRtttl(encoded).events, song.events);
  assert.throws(() => encodeRtttl({ ...song, events: [{ pitch: Infinity, duration: 8, dotted: false }] }));
});

class FakeParam {
  constructor() { this.value = 0; this.events = []; }
  setValueAtTime(value, time) { this.events.push(['set', value, time]); }
  linearRampToValueAtTime(value, time) { this.events.push(['ramp', value, time]); }
  setTargetAtTime(value, time, rate) { this.events.push(['target', value, time, rate]); }
}
class FakeContext {
  static instances = [];
  constructor(channels, length, sampleRate) {
    this.length = length; this.sampleRate = sampleRate; this.currentTime = 0;
    this.destination = {}; this.oscillators = []; this.gains = [];
    FakeContext.instances.push(this);
  }
  async resume() {}
  async close() {}
  createOscillator() {
    const osc = { frequency: new FakeParam(), starts: [], stops: [], connect() {}, disconnect() {},
      start(time) { this.starts.push(time); }, stop(time) { this.stops.push(time); } };
    this.oscillators.push(osc); return osc;
  }
  createGain() {
    const gain = { gain: new FakeParam(), connect() {}, disconnect() {} };
    this.gains.push(gain); return gain;
  }
  async startRendering() { return { getChannelData: () => new Float32Array(this.length) }; }
}

test('live and exported audio schedule the same score, rests and note envelopes', async () => {
  const originalLive = globalThis.AudioContext;
  const originalOffline = globalThis.OfflineAudioContext;
  globalThis.AudioContext = FakeContext;
  globalThis.OfflineAudioContext = FakeContext;
  const progress = [];
  const player = createPlayer({ onProgress: value => progress.push(value) });
  try {
    const song = parseRtttl('Pulse:d=4,o=4,b=120:a,p,c5');
    await player.play(song, { waveform: 'square', volume: 0.4 });
    const live = FakeContext.instances.at(-1);
    assert.equal(player.isPlaying, true);
    assert.equal(live.oscillators.length, 2);
    assert.equal(live.oscillators[0].frequency.value, 440);
    assert.deepEqual(live.oscillators.map(osc => osc.starts[0]), [0.025, 1.025]);
    assert.equal(live.oscillators[0].type, 'square');
    assert.equal(live.gains[0].gain.value, 0.4);
    assert.deepEqual(progress[0], { elapsed: 0, duration: 1.5, progress: 0, eventIndex: 0 });
    player.setVolume(0.2);
    assert.deepEqual(live.gains[0].gain.events.at(-1), ['target', 0.2, 0, 0.015]);
    const { samples, sampleRate } = await renderPcm(song, { waveform: 'square' });
    const offline = FakeContext.instances.at(-1);
    assert.equal(samples.length, 66150);
    assert.equal(sampleRate, 44100);
    assert.deepEqual(offline.oscillators.map(osc => osc.starts[0]), [0, 1]);
    assert.deepEqual(offline.oscillators.map(osc => osc.frequency.value), live.oscillators.map(osc => osc.frequency.value));
    assert.equal(offline.gains[1].gain.events.at(-1)[1], 0);
    player.stop();
    assert.equal(player.isPlaying, false);
    assert.equal(live.oscillators[0].stops.length, 2);
  } finally {
    await player.dispose();
    globalThis.AudioContext = originalLive;
    globalThis.OfflineAudioContext = originalOffline;
  }
});

test('stopping while the audio permission resumes prevents stale playback', async () => {
  const original = globalThis.AudioContext;
  let resume;
  class DelayedContext extends FakeContext { resume() { return new Promise(resolve => { resume = resolve; }); } }
  globalThis.AudioContext = DelayedContext;
  const player = createPlayer();
  try {
    const pending = player.play('x:b=120:c');
    player.stop();
    resume();
    await pending;
    assert.equal(player.isPlaying, false);
    assert.equal(FakeContext.instances.at(-1).oscillators.length, 0);
  } finally { await player.dispose(); globalThis.AudioContext = original; }
});

test('WAV bytes contain correct mono PCM header and little-endian samples', async () => {
  const blob = encodeWavPcm(new Int16Array([-32768, 0, 32767]), 44100);
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  assert.equal(blob.type, 'audio/wav');
  assert.equal(buffer.byteLength, 50);
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF');
  assert.equal(new TextDecoder().decode(bytes.slice(8, 12)), 'WAVE');
  assert.equal(view.getUint32(24, true), 44100);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 6);
  assert.deepEqual([0, 1, 2].map(index => view.getInt16(44 + index * 2, true)), [-32768, 0, 32767]);
});

test('MP3 export encodes real MPEG audio frames, with completed progress', async () => {
  const samples = Int16Array.from({ length: 44100 }, (_, i) => Math.round(5000 * Math.sin(2 * Math.PI * 440 * i / 44100)));
  const progress = [];
  const blob = await encodeMp3Pcm(samples, 44100, value => progress.push(value));
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.equal(blob.type, 'audio/mpeg');
  assert.ok(bytes.length > 10000 && bytes.length < 25000);
  assert.equal(bytes[0], 0xff);
  assert.equal(bytes[1] & 0xe0, 0xe0); // MPEG sync
  assert.equal((bytes[1] >> 1) & 3, 1); // Layer III
  assert.equal((bytes[2] >> 2) & 3, 0); // 44.1 kHz
  assert.equal(progress.at(-1), 1);
});
