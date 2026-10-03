/*
 * RTTTL parsing adapted from Adam Rahwane's MIT-licensed rtttl-parse,
 * used by adamonsoon/rtttl-play. See ../THIRD_PARTY_NOTICES.md.
 * Adds strict whole-token validation, MIDI pitches, explicit seconds, and limits.
 */
export const MAX_AUDIO_SECONDS = 300;
export const MAX_EVENTS = 2048;
const DURATIONS = new Set([1, 2, 4, 8, 16, 32, 64]);
const NATURAL = { c: 0, d: 2, e: 4, f: 5, g: 7, a: 9, b: 11, h: 11 };
const PITCH_NAMES = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];

// Adapted from rtttl-parse's _calculateDuration: RTTTL BPM counts quarter notes.
export function eventSeconds(bpm, duration, dotted = false) {
  const seconds = (60 / bpm * 4) / duration;
  return seconds + (dotted ? seconds / 2 : 0);
}

export function parseRtttl(text) {
  if (typeof text !== 'string') throw new TypeError('RTTTL must be text.');
  if (text.length > 65536) throw new Error('This RTTTL file is too large (maximum 64 KB).');
  const sections = text.replace(/^\uFEFF/, '').trim().split(':');
  if (sections.length < 3) throw new Error('Use the RTTTL format name:d=4,o=6,b=120:c,d,e.');
  const melody = sections.pop();
  const defaultsText = sections.pop();
  const name = sections.join(':').trim() || 'Untitled';
  if (/[dob]\s*=/i.test(name) && sections.length > 1) throw new Error('Paste one RTTTL song at a time.');
  if (name.length > 256) throw new Error('Song names must be at most 256 characters.');
  const defaults = { d: 4, o: 6, b: 63 };
  const seen = new Set();
  if (defaultsText.trim()) {
    for (const setting of defaultsText.split(',')) {
      const match = /^\s*([dob])\s*=\s*(\d+)\s*$/i.exec(setting);
      if (!match) throw new Error(`Invalid RTTTL setting: ${setting.trim()}.`);
      const key = match[1].toLowerCase();
      if (seen.has(key)) throw new Error(`Duplicate RTTTL setting: ${key}.`);
      seen.add(key);
      defaults[key] = Number(match[2]);
    }
  }
  if (!DURATIONS.has(defaults.d)) throw new Error('Duration must be 1, 2, 4, 8, 16, 32 or 64.');
  if (!Number.isInteger(defaults.o) || defaults.o < 4 || defaults.o > 7) throw new Error('Octave must be between 4 and 7.');
  if (!Number.isInteger(defaults.b) || defaults.b < 25 || defaults.b > 900) throw new Error('Tempo must be between 25 and 900 BPM.');
  const tokens = melody.replace(/\s+/g, '').replace(/,+$/, '').toLowerCase().split(',');
  if (!tokens[0]) throw new Error('Add at least one note.');
  if (tokens.length > MAX_EVENTS) throw new Error(`Songs may contain at most ${MAX_EVENTS} events.`);
  const events = tokens.map((token, index) => {
    const match = /^(\d*)([a-ghp])(#?)(\.?)([4-7]?)(\.?)$/.exec(token);
    if (!match || (match[4] && match[6])) throw new Error(`Invalid note ${index + 1}: ${token || '(empty)'}.`);
    const duration = match[1] ? Number(match[1]) : defaults.d;
    if (!DURATIONS.has(duration)) throw new Error(`Invalid duration in note ${index + 1}.`);
    const dotted = Boolean(match[4] || match[6]);
    const octave = match[5] ? Number(match[5]) : defaults.o;
    if (match[2] === 'p' && match[3]) throw new Error(`A rest cannot be sharp (note ${index + 1}).`);
    const pitch = match[2] === 'p' ? null : 12 * (octave + 1) + NATURAL[match[2]] + Number(Boolean(match[3]));
    if (pitch !== null && pitch > 107) throw new Error(`Note ${index + 1} is above B7.`);
    return { pitch, duration, dotted, seconds: eventSeconds(defaults.b, duration, dotted) };
  });
  const durationSeconds = events.reduce((sum, event) => sum + event.seconds, 0);
  if (durationSeconds > MAX_AUDIO_SECONDS) throw new Error(`Songs must be shorter than ${MAX_AUDIO_SECONDS / 60} minutes.`);
  return { name, bpm: defaults.b, defaultDuration: defaults.d, defaultOctave: defaults.o, events, durationSeconds };
}

export function encodeRtttl(song) {
  if (!song || !Array.isArray(song.events) || !song.events.length) throw new Error('A song needs notes.');
  const defaultDuration = song.defaultDuration ?? 8;
  const defaultOctave = song.defaultOctave ?? 6;
  const name = String(song.name || 'Generated').normalize('NFKD').replace(/[^A-Za-z0-9 _-]/g, '').trim().slice(0, 11) || 'Generated';
  const tokens = song.events.map(event => {
    if (!DURATIONS.has(event.duration) || typeof event.dotted !== 'boolean') throw new Error('Invalid note duration.');
    let token = event.duration === defaultDuration ? '' : String(event.duration);
    if (event.pitch === null) token += 'p';
    else {
      if (!Number.isInteger(event.pitch) || event.pitch < 60 || event.pitch > 107) throw new Error('Pitches must be MIDI 60–107.');
      token += PITCH_NAMES[event.pitch % 12];
      const octave = Math.floor(event.pitch / 12) - 1;
      if (octave !== defaultOctave) token += octave;
    }
    return token + (event.dotted ? '.' : '');
  });
  const text = `${name}:d=${defaultDuration},o=${defaultOctave},b=${song.bpm}:${tokens.join(',')}`;
  parseRtttl(text);
  return text;
}
