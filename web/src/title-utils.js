export const TITLE_MODEL = 'onnx-community/SmolLM2-135M-Instruct-ONNX';
export const TITLE_REVISION = 'b8a5c0f183b78c55955a5364f610c36668b5e681';
export const TITLE_DOWNLOAD_MB = 140;
export const TITLE_DTYPE = 'q8';

function melodyFeatures(song) {
  const events = song.events || [];
  const pitches = events.map(event => event.pitch).filter(pitch => pitch !== null && Number.isFinite(pitch));
  const durations = events.map(event => 4 / event.duration * (event.dotted ? 1.5 : 1)).filter(Number.isFinite);
  const beats = durations.reduce((total, duration) => total + duration, 0);
  const steps = pitches.slice(1).map((pitch, index) => pitch - pitches[index]);
  const moving = steps.filter(value => value !== 0);
  return { events, pitches, beats, moving,
    averagePitch: pitches.length ? pitches.reduce((a,b) => a+b,0)/pitches.length : 72,
    averageStep: moving.length ? moving.reduce((total, value) => total + Math.abs(value), 0)/moving.length : 0,
    span: pitches.length ? Math.max(...pitches) - Math.min(...pitches) : 0,
    restFraction: events.length ? events.filter(event => event.pitch === null).length/events.length : 0,
    longNotes: durations.length ? beats/durations.length >= 1 : false,
  };
}

export function describeMelody(song) {
  const { events, pitches, beats, moving, averagePitch, averageStep, span, restFraction, longNotes } = melodyFeatures(song);
  const contour = !moving.length ? 'level' : moving.every(value => value > 0) ? 'rising' : moving.every(value => value < 0) ? 'falling' : 'rising and falling';
  const pace = song.bpm < 90 ? 'slow' : song.bpm >= 140 ? 'quick' : 'moderate';
  const register = averagePitch >= 84 ? 'high notes' : averagePitch < 65 ? 'low notes' : 'middle notes';
  const motion = averageStep >= 5 ? 'wide melodic jumps' : averageStep >= 2 ? 'a mix of steps and jumps' : 'small gentle steps';
  const rhythm = longNotes ? 'sustained notes' : 'short rhythmic notes';
  const phrases = restFraction >= .15 ? 'frequent breathing spaces' : restFraction > 0 ? 'occasional pauses' : 'continuous motion';
  const settings = song.settings || {};
  // These are requested sampling guides, not a genre classifier or a claim that
  // every generated note follows the requested key.
  const guide = ({ 'pop-hook': 'pop hook', chiptune: 'playful game music', dance: 'dance', cinematic: 'cinematic', lullaby: 'lullaby' })[settings.profile];
  const details = [
    `${pace} melody at ${song.bpm} BPM`, `${Math.round(beats * 60 / song.bpm)} seconds`,
    rhythm, register, `${contour} movement`, motion,
    `${span >= 12 ? 'wide' : 'compact'} pitch range`, phrases,
  ];
  if (events.some(event => event.dotted)) details.push('some dotted rhythms');
  if (pitches.length) details.push(`${new Set(pitches).size} different pitches`);
  if (guide) details.push(`${guide} inspiration`);
  if (settings.mode === 'natural-minor') details.push('a minor-key guide');
  return details.join(', ');
}

const titleWords = title => String(title).toLowerCase().match(/\p{L}+/gu)?.filter(word => !['a','an','the','of','and','in','on','at','by','to','for'].includes(word)) || [];
const titleExamples = [
  ['Soft flowing notes', 'Gentle River'], ['Bright quick high notes', 'Racing Stars'],
  ['Slow dramatic notes', 'Distant Thunder'], ['Warm swaying notes', 'Amber Tides'],
  ['High delicate notes with pauses', 'Floating Lanterns'], ['Lively winding notes', 'Copper Carousel'],
  ['Deep notes rising in long steps', 'Mountain Dawn'], ['Short playful notes', 'Paper Rockets'],
];
export const EXAMPLE_TITLES = new Set(titleExamples.map(([, title]) => title.toLowerCase()));
export function isRepeatedTitle(title, usedTitles) {
  const words = titleWords(title), key = words.join(' ');
  return Array.from(usedTitles).some(used => {
    const previous = titleWords(used);
    return previous.join(' ') === key || (words.length >= 2 && previous.length >= 2
      && new Set(words.filter(word => previous.includes(word))).size >= 2);
  });
}

function melodyHash(song) {
  let hash = 2166136261;
  for (const character of song.rtttl || JSON.stringify(song.events || [])) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619) >>> 0;
  return hash;
}

export function createTitlePrompt(song, usedTitles, attempt = 0) {
  const avoid = Array.from(usedTitles).slice(-8).join('; ');
  const start = (melodyHash(song) + attempt*3) % titleExamples.length;
  const examples = Array.from({ length: 3 }, (_, index) => titleExamples[(start+index)%titleExamples.length]);
  // A short completion template works more reliably than a chat instruction for
  // the compact model. Rotate its examples and reject any copied example below.
  return `Invent a new short music title for each sound. Use two to four words, no explanation.${avoid ? ` Avoid these recent titles: ${avoid}.` : ''}\n`
    + examples.map(([sound, title]) => `Sound: ${sound}.\nTitle: ${title}`).join('\n')
    + `\nSound: ${describeMelody(song)}.\nTitle:`;
}

export function sanitizeTitle(raw) {
  let text = String(raw).trim().replace(/^(?:title|song title|name)\s*:\s*/i, '').trim();
  text = text.replace(/[.!]+$/, '').replace(/^["'“”]+|["'“”]+$/g, '').trim();
  if (!text || text.length > 60 || text.split(/\s+/).length > 6 || !/[a-zA-Z]/.test(text)
      || text.split(/[\s-]+/).some(word => word.length > 20)
      || /[\n\r<>:{}\[\]`]/.test(text) || !/^[\p{L}\p{N} '\-’&]+$/u.test(text)
      || /\b(?:here is|i suggest|based on|this melody|this song|instructions?|assistant|song|melody)\b/i.test(text)) {
    throw new Error('The title model returned an unusable title.');
  }
  return text;
}

export function extractGeneratedTitle(raw) {
  // The model may append a new example or a parenthetical performance note.
  // Neither belongs to the name. All remaining text still passes the strict
  // title validator before it is placed in the library or an RTTTL header.
  const line = String(raw).trim().split(/\r?\n/, 1)[0];
  return sanitizeTitle(line.replace(/\s*\([^()\r\n]{0,60}\)\s*$/, '').trim());
}

export function applyTitle(song, raw, usedTitles, usedNames) {
  const title = sanitizeTitle(raw);
  if (isRepeatedTitle(title, usedTitles)) throw new Error('The title model repeated an earlier title.');
  const colon = song.rtttl.indexOf(':');
  if (colon < 0) throw new Error('Cannot name an invalid RTTTL song.');
  const base = title.normalize('NFKD').replace(/[^A-Za-z0-9]/g, '').slice(0,11) || 'Untitled';
  let name = base, number = 2;
  while (usedNames.has(name.toLowerCase())) {
    const suffix = String(number++);
    name = base.slice(0,11-suffix.length) + suffix;
  }
  usedTitles.add(title.toLowerCase());
  usedNames.add(name.toLowerCase());
  return { ...song, title, name, rtttl: name + song.rtttl.slice(colon),
    naming: { model: TITLE_MODEL, revision: TITLE_REVISION, status: 'named' } };
}

// A small model can ignore its instructions. Keep every tune, and derive a
// distinct name from measured character if two attempts are unusable.
export function fallbackTitle(song, usedTitles, usedNames) {
  const features = melodyFeatures(song);
  const adjectives = song.bpm < 90
    ? ['Quiet', 'Drifting', 'Velvet', 'Distant', 'Gentle', 'Silver', 'Hushed', 'Tender']
    : song.bpm >= 140
      ? ['Electric', 'Dancing', 'Restless', 'Swift', 'Spiraling', 'Flying', 'Flashing', 'Bright']
      : ['Amber', 'Wandering', 'Golden', 'Rolling', 'Shimmering', 'Glowing', 'Winding', 'Open'];
  const nouns = features.averagePitch >= 84
    ? ['Sparks', 'Stars', 'Lanterns', 'Fireflies', 'Kites', 'Comets', 'Bells', 'Feathers']
    : features.averageStep >= 5
      ? ['Flight', 'Horizons', 'Summits', 'Bridges', 'Arches', 'Canyons', 'Wings', 'Heights']
      : ['Tides', 'Footsteps', 'Paths', 'Ripples', 'Rivers', 'Gardens', 'Waves', 'Trails'];
  const hash = melodyHash(song);
  const modifiers = ['', 'Morning ', 'Evening ', 'Midnight '];
  for (let offset = 0; offset < adjectives.length*nouns.length*modifiers.length; offset++) {
    const index = (hash + offset) % (adjectives.length*nouns.length*modifiers.length);
    const title = `${adjectives[index % adjectives.length]} ${modifiers[Math.floor(index/64)]}${nouns[Math.floor(index/8) % nouns.length]}`;
    if (isRepeatedTitle(title, usedTitles)) continue;
    const named = applyTitle(song, title, usedTitles, usedNames);
    return { ...named, naming: { status: 'fallback', reason: 'The title model repeated a name or returned an unusable title.', strategy: 'musical-character' } };
  }
  return { ...song, naming: { status: 'fallback', reason: 'No distinct title was available.' } };
}
