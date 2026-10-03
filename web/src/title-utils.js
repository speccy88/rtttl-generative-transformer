export const TITLE_MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';
export const TITLE_REVISION = 'cc5cc01a65cc3ff17bdb73a7de33d879f62599b0';
export const TITLE_DOWNLOAD_MB = 800;

export function describeMelody(song) {
  const events = song.events || [];
  const pitches = events.map(event => event.pitch).filter(pitch => pitch !== null && Number.isFinite(pitch));
  const beats = events.reduce((total, event) => total + 4 / event.duration * (event.dotted ? 1.5 : 1), 0);
  const seconds = beats * 60 / song.bpm;
  const steps = pitches.slice(1).map((pitch, index) => pitch - pitches[index]);
  const moving = steps.filter(value => value !== 0);
  const contour = !moving.length ? 'level' : moving.every(value => value > 0) ? 'rising' : moving.every(value => value < 0) ? 'falling' : 'rising and falling';
  const pace = song.bpm < 90 ? 'slow' : song.bpm >= 140 ? 'quick' : 'moderate';
  const rhythm = events.length && beats / events.length > 1 ? 'sustained notes' : 'short rhythmic notes';
  const register = pitches.length && pitches.reduce((a,b) => a+b,0)/pitches.length >= 84 ? 'high notes' : 'middle and lower notes';
  return `${pace} melody at ${song.bpm} BPM, ${Math.round(seconds)} seconds, ${rhythm}, ${register}, ${contour} movement`;
}

export function sanitizeTitle(raw) {
  let text = String(raw).trim().replace(/^(?:title|song title|name)\s*:\s*/i, '').trim();
  text = text.replace(/[.!]+$/, '').replace(/^["'“”]+|["'“”]+$/g, '').trim();
  if (!text || text.length > 60 || text.split(/\s+/).length > 6 || !/[a-zA-Z]/.test(text)
      || text.split(/[\s-]+/).some(word => word.length > 20)
      || /[\n\r<>:{}\[\]`]/.test(text) || !/^[\p{L}\p{N} '\-’&]+$/u.test(text)
      || /\b(?:here is|i suggest|based on|this melody|this song|instructions?|assistant)\b/i.test(text)) {
    throw new Error('The title model returned an unusable title.');
  }
  return text;
}

export function applyTitle(song, raw, usedTitles, usedNames) {
  let title = sanitizeTitle(raw);
  let number = 2;
  const baseTitle = title;
  while (usedTitles.has(title.toLowerCase())) title = `${baseTitle.slice(0,55)} ${number++}`;
  usedTitles.add(title.toLowerCase());
  const base = title.normalize('NFKD').replace(/[^A-Za-z0-9]/g, '').slice(0,11) || 'Untitled';
  let name = base;
  number = 2;
  while (usedNames.has(name.toLowerCase())) {
    const suffix = String(number++);
    name = base.slice(0,11-suffix.length) + suffix;
  }
  usedNames.add(name.toLowerCase());
  const colon = song.rtttl.indexOf(':');
  if (colon < 0) throw new Error('Cannot name an invalid RTTTL song.');
  return { ...song, title, name, rtttl: name + song.rtttl.slice(colon),
    naming: { model: TITLE_MODEL, revision: TITLE_REVISION, status: 'named' } };
}
