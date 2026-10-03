// Handcrafted melodic preferences, shared with the Python release. These are
// sampling guides rather than genre-trained models or instrument settings.
export const PITCH_CLASSES = Object.freeze(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']);
export const SCALES = Object.freeze({ major: [0, 2, 4, 5, 7, 9, 11], 'natural-minor': [0, 2, 3, 5, 7, 8, 10] });
const base = {
  scaleBias: .35, registerBonus: .20, registerDistancePenalty: .04, registerPenaltyCap: .40,
  leapStart: 5, leapPenalty: .06, leapPenaltyCap: .60, restBias: -.15,
  consecutiveRestPenalty: .15, consecutiveRestPenaltyCap: .45,
  restDurationPenalty: .15, restDurationPenaltyCap: .60,
};
export const PROFILES = Object.freeze({
  'pop-hook': Object.freeze({ ...base, label: 'Pop hook', bpmRange: [88, 132], defaultMode: 'major', register: [72, 91],
    durations: [[1, -.15, -.15], [2, -.15, -.15], [4, .30, 0], [8, .30, 0], [16, .10, 0], [32, 0, 0]],
    intervals: [[1, 2, .20], [3, 4, .10]] }),
  chiptune: Object.freeze({ ...base, label: 'Chiptune', bpmRange: [120, 180], defaultMode: 'major', register: [79, 100],
    durations: [[1, -.40, -.40], [2, -.30, -.30], [4, 0, -.10], [8, .25, 0], [16, .40, .05], [32, .15, -.05]],
    intervals: [[1, 2, .05], [3, 4, .10], [5, 12, .30]], registerBonus: .25,
    leapStart: 12, leapPenalty: .04, restBias: -.18, restDurationPenalty: .20 }),
  cinematic: Object.freeze({ ...base, label: 'Cinematic', bpmRange: [60, 104], defaultMode: 'natural-minor', register: [60, 79],
    durations: [[1, .15, .10], [2, .40, .35], [4, .15, .20], [8, -.15, -.10], [16, -.40, -.35], [32, -.45, -.40]],
    intervals: [[1, 2, .10], [3, 4, .15], [5, 7, .08]], leapStart: 7,
    restBias: .12, consecutiveRestPenalty: .12, restDurationPenalty: .04 }),
  dance: Object.freeze({ ...base, label: 'Dance', bpmRange: [118, 150], defaultMode: 'major', register: [69, 88],
    durations: [[1, -.35, -.40], [2, -.25, -.30], [4, .30, -.15], [8, .35, -.15], [16, .15, -.15], [32, -.10, -.20]],
    intervals: [[1, 2, .18], [3, 4, .10]], leapStart: 7, restBias: -.35, restDurationPenalty: .25 }),
  lullaby: Object.freeze({ ...base, label: 'Lullaby', bpmRange: [60, 84], defaultMode: 'major', register: [60, 79],
    durations: [[1, .05, 0], [2, .35, .20], [4, .30, .25], [8, 0, .05], [16, -.30, -.25], [32, -.40, -.35]],
    intervals: [[1, 2, .35], [3, 4, .10]], registerBonus: .25, leapStart: 4,
    leapPenalty: .10, restBias: .02, restDurationPenalty: .08 }),
});
export const PROFILE_NAMES = Object.freeze(Object.keys(PROFILES));

export function normalizeTonic(value) {
  if (typeof value !== 'string') throw new Error('Tonic must be a note name such as C, F#, or Bb.');
  const match = value.trim().replaceAll('♯', '#').replaceAll('♭', 'b').match(/^([a-gA-G])([#b]?)$/);
  if (!match) throw new Error('Tonic must be a note name such as C, F#, or Bb.');
  const note = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[match[1].toUpperCase()];
  return PITCH_CLASSES[(note + ({ '': 0, '#': 1, b: -1 }[match[2]]) + 12) % 12];
}

// A deterministic, local RNG. Browser seeds deliberately do not promise exact
// equivalence to PyTorch's random stream or across different GPU arithmetic.
export function createRng(seed) {
  const text = String(seed);
  let state = 2166136261;
  for (let i = 0; i < text.length; i++) state = Math.imul(state ^ text.charCodeAt(i), 16777619);
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function* shuffledCycles(values, rng) {
  while (true) {
    const cycle = [...values];
    for (let i = cycle.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [cycle[i], cycle[j]] = [cycle[j], cycle[i]];
    }
    yield* cycle;
  }
}
const validBpm = value => Number.isInteger(value) && value >= 25 && value <= 900;

export function planBatch(options = {}) {
  const { numSongs = 4, seed = 42, bpm = null, bpmRange = null, tonic = null, mode = null } = options;
  const profile = options.profile === 'none' ? null : (options.profile === undefined ? 'mixed' : options.profile);
  if (!Number.isInteger(numSongs) || numSongs < 1 || numSongs > 24) throw new Error('Choose between 1 and 24 melodies per batch.');
  if (!Number.isSafeInteger(seed)) throw new Error('Seed must be a safe integer.');
  if (profile !== null && profile !== 'mixed' && !Object.hasOwn(PROFILES, profile)) throw new Error('Unknown melody guide.');
  if (bpm !== null && bpmRange !== null) throw new Error('Choose either a fixed BPM or a tempo range.');
  if (bpm !== null && !validBpm(bpm)) throw new Error('BPM must be an integer from 25 to 900.');
  if (bpmRange !== null && (!Array.isArray(bpmRange) || bpmRange.length !== 2 || !bpmRange.every(validBpm) || bpmRange[0] > bpmRange[1])) {
    throw new Error('Tempo range must contain two ordered BPM values from 25 to 900.');
  }
  const canonicalTonic = tonic === null ? null : normalizeTonic(tonic);
  if (mode !== null && !Object.hasOwn(SCALES, mode)) throw new Error('Mode must be major or natural-minor.');
  const profiles = shuffledCycles(PROFILE_NAMES, createRng(`${seed}:profiles`));
  const keys = shuffledCycles(PITCH_CLASSES, createRng(`${seed}:keys`));
  const tempoRng = createRng(`${seed}:tempos`);
  return Array.from({ length: numSongs }, () => {
    const selectedProfile = profile === 'mixed' ? profiles.next().value : profile;
    const spec = selectedProfile === null ? null : PROFILES[selectedProfile];
    const range = bpmRange ?? spec?.bpmRange;
    return {
      profile: selectedProfile,
      bpm: bpm ?? (range ? range[0] + Math.floor(tempoRng() * (range[1] - range[0] + 1)) : null),
      tonic: canonicalTonic ?? (profile === 'mixed' ? keys.next().value : 'C'),
      mode: mode ?? spec?.defaultMode ?? 'major',
      tempoSource: bpm !== null ? 'explicit' : bpmRange !== null ? 'range' : spec ? 'profile_range' : 'model',
      profileSource: profile === 'mixed' ? 'mixed' : profile === null ? 'none' : 'explicit',
    };
  });
}

export function applyGuidance(logits, allowed, history, tokenizer, settings) {
  if (!settings.profile) return logits;
  const spec = PROFILES[settings.profile];
  if (!spec) throw new Error('Unknown melody guide.');
  const scores = Float32Array.from(logits);
  const root = PITCH_CLASSES.indexOf(normalizeTonic(settings.tonic));
  const scale = new Set(SCALES[settings.mode].map(interval => (interval + root) % 12));
  let previous = null;
  for (let i = history.length - 1; i >= 0; i--) {
    if (tokenizer.pitchById.has(history[i])) { previous = tokenizer.pitchById.get(history[i]); break; }
  }
  let trailingRests = 0;
  for (let i = history.length - 1; i >= 0 && history[i] === tokenizer.restId; i--) trailingRests++;
  const currentRest = history.at(-1) === tokenizer.restId;
  for (const id of allowed) {
    if (tokenizer.pitchById.has(id)) {
      const pitch = tokenizer.pitchById.get(id);
      let bias = scale.has(pitch % 12) ? spec.scaleBias : -spec.scaleBias;
      const [low, high] = spec.register;
      bias += pitch >= low && pitch <= high ? spec.registerBonus
        : -Math.min(spec.registerPenaltyCap, spec.registerDistancePenalty * (pitch < low ? low - pitch : pitch - high));
      if (previous !== null) {
        const distance = Math.abs(previous - pitch);
        const interval = spec.intervals.find(([min, max]) => distance >= min && distance <= max);
        if (interval) bias += interval[2];
        if (distance > spec.leapStart) bias -= Math.min(spec.leapPenaltyCap, spec.leapPenalty * (distance - spec.leapStart));
      }
      scores[id] += bias;
    } else if (id === tokenizer.restId) {
      scores[id] += spec.restBias - Math.min(spec.consecutiveRestPenaltyCap, spec.consecutiveRestPenalty * trailingRests);
    } else if (tokenizer.durationById.has(id)) {
      const { duration, dotted } = tokenizer.durationById.get(id);
      let bias = spec.durations.find(row => row[0] === duration)[dotted ? 2 : 1];
      if (currentRest) bias -= Math.min(spec.restDurationPenaltyCap, spec.restDurationPenalty * 4 / duration * (dotted ? 1.5 : 1));
      scores[id] += bias;
    }
  }
  return scores;
}
