import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRtttl } from './rtttl.js';
import { appendStudioSongs, changeScoreTempo, transposeScore } from './studio-score.js';

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
