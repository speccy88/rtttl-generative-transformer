import { encodeRtttl, parseRtttl } from './rtttl.js';

export function transposeScore(rtttl, semitones) {
  if (!Number.isInteger(semitones)) throw new Error('Transpose by a whole number of semitones.');
  const song = parseRtttl(rtttl);
  const events = song.events.map(event => ({ ...event, pitch: event.pitch === null ? null : event.pitch + semitones }));
  if (events.some(event => event.pitch !== null && (event.pitch < 60 || event.pitch > 107))) throw new Error('This move would place notes outside RTTTL’s C4–B7 range.');
  return encodeRtttl({ ...song, events });
}

export function changeScoreTempo(rtttl, bpm) {
  if (!Number.isInteger(bpm) || bpm < 25 || bpm > 900) throw new Error('Tempo must be a whole number from 25 to 900 BPM.');
  return encodeRtttl({ ...parseRtttl(rtttl), bpm });
}

export function appendStudioSongs(library, additions, maximum = 64) {
  const ids = new Set(library.map(song => String(song.id)));
  const incoming = additions.map(song => {
    parseRtttl(song.rtttl);
    if (!song.id || ids.has(String(song.id))) throw new Error('Each melody version needs a unique identifier.');
    ids.add(String(song.id));
    return song;
  });
  return [...library, ...incoming].slice(-maximum);
}
