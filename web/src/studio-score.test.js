import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRtttl } from './rtttl.js';
import { appendStudioSongs, changeScoreTempo, editScoreSelection, transposeScore } from './studio-score.js';

const source = 'Hook:d=8,o=5,b=120:c,16p,e.,4g';
const musicalEvents = text => parseRtttl(text).events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted }));

test('transpose preserves rhythm and rests and rejects range overflow', () => {
  const transposed = musicalEvents(transposeScore(source, 1));
  assert.deepEqual(transposed, musicalEvents(source).map(event => ({ ...event, pitch: event.pitch === null ? null : event.pitch + 1 })));
  assert.throws(() => transposeScore('Low:d=4,o=4,b=120:c', -1), /outside RTTTL/);
  assert.throws(() => transposeScore('High:d=4,o=7,b=120:b', 1), /outside RTTTL/);
});

test('tempo edits change timing without changing any musical event', () => {
  const edited = changeScoreTempo(source, 60);
  assert.deepEqual(musicalEvents(edited), musicalEvents(source));
  assert.equal(parseRtttl(edited).durationSeconds, parseRtttl(source).durationSeconds * 2);
  assert.throws(() => changeScoreTempo(source, 0), /Tempo/);
});

test('new generation and edits retain previous converted and imported scores', () => {
  const original = { id: 'converted', rtttl: source, source: 'converted' };
  const imported = { id: 'imported', rtttl: source, source: 'import' };
  const generated = { id: 'generated', rtttl: source, source: 'generated' };
  const existing = [original, imported];
  assert.deepEqual(appendStudioSongs(existing, [generated]), [original, imported, generated]);
  assert.deepEqual(existing, [original, imported]);
  assert.deepEqual(appendStudioSongs(existing, [generated], 2), [imported, generated]);
  assert.throws(() => appendStudioSongs(existing, [original]), /unique identifier/);
});

test('selected pitch edits keep unselected events, rhythm and rests fixed', () => {
  const before = musicalEvents(source);
  const after = musicalEvents(editScoreSelection(source, { start: 1, end: 3 }, { semitones: 1 }));
  assert.deepEqual(after, before.map((event, index) => index === 2 ? { ...event, pitch: event.pitch + 1 } : event));
  assert.equal(parseRtttl(editScoreSelection(source, { start: 1, end: 3 }, { semitones: 1 })).bpm, 120);
  assert.throws(() => editScoreSelection(source, { start: 3, end: 3 }, { semitones: 1 }), /Select a note/);
  assert.throws(() => editScoreSelection(source, { start: -1, end: 2 }, { rest: true }), /Select a note/);
  assert.throws(() => editScoreSelection('Low:d=4,o=4,b=120:c,d', { start: 0, end: 1 }, { semitones: -1 }), /outside RTTTL/);
});

test('selected rest and duration edits leave the rest of a tune untouched', () => {
  const before = musicalEvents(source);
  const rested = musicalEvents(editScoreSelection(source, { start: 2, end: 3 }, { rest: true }));
  assert.deepEqual(rested, before.map((event, index) => index === 2 ? { ...event, pitch: null } : event));
  const restored = musicalEvents(editScoreSelection(source, { start: 0, end: 3 }, { restoreRests: true }));
  assert.deepEqual(restored, before.map(event => event.pitch === null ? { ...event, pitch: 72 } : event));
  const timed = musicalEvents(editScoreSelection(source, { start: 1, end: 3 }, { duration: 32, dotted: true }));
  assert.deepEqual(timed, before.map((event, index) => index >= 1 && index < 3 ? { ...event, duration: 32, dotted: true } : event));
  assert.throws(() => editScoreSelection(source, { start: 0, end: 1 }, { duration: 3 }), /duration/);
});
