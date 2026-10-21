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

export function validScoreSelection(selection, count) {
  return Boolean(selection && Number.isInteger(selection.start) && Number.isInteger(selection.end)
    && selection.start >= 0 && selection.start < selection.end && selection.end <= count);
}

export function editScoreSelection(rtttl, selection, change) {
  const song = parseRtttl(rtttl);
  if (!validScoreSelection(selection, song.events.length)) throw new Error('Select a note or a contiguous phrase first.');
  if (!change || typeof change !== 'object') throw new Error('Choose a note edit.');
  if (change.semitones !== undefined && !Number.isInteger(change.semitones)) throw new Error('Transpose by a whole number of semitones.');
  const events = song.events.map((event, index) => {
    if (index < selection.start || index >= selection.end) return event;
    const edited = { ...event };
    if (change.semitones !== undefined && edited.pitch !== null) edited.pitch += change.semitones;
    if (change.rest === true) edited.pitch = null;
    if (change.restoreRests === true && edited.pitch === null) edited.pitch = 72;
    if (change.duration !== undefined) edited.duration = change.duration;
    if (change.dotted !== undefined) edited.dotted = change.dotted;
    return edited;
  });
  if (events.some(event => event.pitch !== null && (event.pitch < 60 || event.pitch > 107))) throw new Error('This move would place notes outside RTTTL’s C4–B7 range.');
  return encodeRtttl({ ...song, events });
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
