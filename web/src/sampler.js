import { applyGuidance, createRng, planBatch } from './profiles.js';

export const DEFAULT_OPTIONS = Object.freeze({
  numSongs: 4, profile: 'mixed', minEvents: 16, maxEvents: 48, temperature: .8,
  topK: 20, topP: .95, seed: 42, repetitionPenalty: 1.2, repetitionWindow: 16,
  maxPitchRun: 4, maxMotifRepeats: 3, maxMotifLength: 16,
});

export function validateOptions(options = {}) {
  const value = { ...DEFAULT_OPTIONS, ...options };
  const integer = (name, minimum, maximum = Number.MAX_SAFE_INTEGER) => {
    if (!Number.isSafeInteger(value[name]) || value[name] < minimum || value[name] > maximum) {
      throw new Error(`${name} must be an integer from ${minimum} to ${maximum}.`);
    }
  };
  integer('minEvents', 1, 256); integer('maxEvents', value.minEvents, 256);
  integer('topK', 0, 940); integer('repetitionWindow', 1, 512);
  integer('maxPitchRun', 0, 256); integer('maxMotifRepeats', 0, 256); integer('maxMotifLength', 2, 256);
  if (value.maxMotifRepeats === 1) throw new Error('Motif repeat limit must be 0 (off), or at least 2.');
  if (!Number.isFinite(value.temperature) || value.temperature <= 0) throw new Error('Temperature must be finite and positive.');
  if (!Number.isFinite(value.topP) || value.topP <= 0 || value.topP > 1) throw new Error('topP must be greater than 0 and at most 1.');
  if (!Number.isFinite(value.repetitionPenalty) || value.repetitionPenalty < 1) throw new Error('Repetition penalty must be finite and at least 1.');
  const plan = planBatch(value);
  return { options: value, plan };
}

export class EventTokenizer {
  constructor(vocabulary = null) {
    const tokens = ['<PAD>', '<BOS>', '<EOS>'];
    for (let bpm = 25; bpm <= 900; bpm++) tokens.push(`<BPM_${bpm}>`);
    for (let pitch = 60; pitch <= 107; pitch++) tokens.push(`<PITCH_${pitch}>`);
    tokens.push('<REST>');
    for (const duration of [1, 2, 4, 8, 16, 32]) for (const suffix of ['plain', 'dot']) tokens.push(`<DUR_${duration}_${suffix}>`);
    if (vocabulary && (vocabulary.version !== 1 || !Array.isArray(vocabulary.tokens)
      || tokens.length !== vocabulary.tokens.length || tokens.some((token, i) => token !== vocabulary.tokens[i]))) {
      throw new Error('The downloaded melody vocabulary is incompatible with this player.');
    }
    this.tokens = tokens;
    this.tokenToId = new Map(tokens.map((token, index) => [token, index]));
    this.bosId = 1; this.eosId = 2; this.vocabSize = tokens.length;
    this.restId = this.tokenToId.get('<REST>');
    this.bpmById = new Map(); this.pitchById = new Map(); this.durationById = new Map();
    for (let bpm = 25; bpm <= 900; bpm++) this.bpmById.set(this.tokenToId.get(`<BPM_${bpm}>`), bpm);
    for (let pitch = 60; pitch <= 107; pitch++) this.pitchById.set(this.tokenToId.get(`<PITCH_${pitch}>`), pitch);
    for (const duration of [1, 2, 4, 8, 16, 32]) for (const dotted of [false, true]) {
      this.durationById.set(this.tokenToId.get(`<DUR_${duration}_${dotted ? 'dot' : 'plain'}>`), { duration, dotted });
    }
    this.bpmIds = [...this.bpmById.keys()];
    this.eventStartIds = [...this.pitchById.keys(), this.restId];
    this.durationIds = [...this.durationById.keys()];
    this.eventStartSet = new Set(this.eventStartIds);
  }

  allowedNext(ids, minEvents = 1, maxEvents = 256) {
    if (!Number.isInteger(minEvents) || !Number.isInteger(maxEvents) || minEvents < 1 || maxEvents < minEvents) throw new Error('Invalid event budget.');
    if (!ids.length) return [this.bosId];
    if (ids[0] !== this.bosId) throw new Error('Melody must start with BOS.');
    if (ids.at(-1) === this.eosId) return [];
    if (ids.length === 1) return [...this.bpmIds];
    if (!this.bpmById.has(ids[1])) throw new Error('BPM must follow BOS.');
    if ((ids.length - 2) % 2) {
      if (!this.eventStartSet.has(ids.at(-1))) throw new Error('Invalid event-start token.');
      return [...this.durationIds];
    }
    if (ids.length > 2 && !this.durationById.has(ids.at(-1))) throw new Error('An event must end with duration.');
    const count = (ids.length - 2) / 2;
    if (count >= maxEvents) return [this.eosId];
    return [...this.eventStartIds, ...(count >= minEvents ? [this.eosId] : [])];
  }

  decode(ids, name = 'Melody') {
    const tokens = [...ids];
    while (tokens.at(-1) === 0) tokens.pop();
    if (tokens.length < 5 || tokens[0] !== this.bosId || tokens.at(-1) !== this.eosId
      || !this.bpmById.has(tokens[1]) || (tokens.length - 3) % 2) throw new Error('Incomplete or invalid melody sequence.');
    const events = [];
    for (let i = 2; i < tokens.length - 1; i += 2) {
      if (!this.eventStartSet.has(tokens[i]) || !this.durationById.has(tokens[i + 1])) throw new Error('Invalid pitch or duration in melody.');
      events.push({ pitch: this.pitchById.get(tokens[i]) ?? null, ...this.durationById.get(tokens[i + 1]) });
    }
    const bpm = this.bpmById.get(tokens[1]);
    return { name, bpm, events, rtttl: toRtttl({ name, bpm, events }) };
  }
}

export function toRtttl({ name, bpm, events }) {
  const compactName = String(name).normalize('NFKD').replace(/[^a-zA-Z0-9 _-]/g, '').trim().slice(0, 11) || 'Melody';
  const pitchNames = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
  const notes = events.map(({ pitch, duration, dotted }) => {
    const note = pitch === null ? 'p' : `${pitchNames[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
    return `${duration}${note}${dotted ? '.' : ''}`;
  });
  return `${compactName}:d=4,o=6,b=${bpm}:${notes.join(',')}`;
}

export function neuralContext(ids, contextLength) {
  if (!Number.isInteger(contextLength) || contextLength < 4) throw new Error('Model context must contain at least four tokens.');
  if (ids.length <= contextLength) return ids;
  const body = ids.slice(2);
  let start = Math.max(0, body.length - (contextLength - 2));
  start += start % 2;
  return [...ids.slice(0, 2), ...body.slice(start)];
}

export function repetitionFilter(logits, allowed, history, options) {
  const counts = { penalty_steps: 0, pitch_run_blocks: 0, motif_blocks: 0 };
  if (!history.length) return { logits, allowed, counts };
  const { repetitionPenalty, repetitionWindow, maxPitchRun, maxMotifRepeats, maxMotifLength } = options;
  const blocked = new Set();
  if (maxPitchRun && history.length >= maxPitchRun && history.slice(-maxPitchRun).every(id => id === history.at(-1))) {
    blocked.add(history.at(-1)); counts.pitch_run_blocks = 1;
  }
  if (maxMotifRepeats) {
    const end = Math.min(maxMotifLength, Math.floor(history.length / maxMotifRepeats));
    for (let period = 2; period <= end; period++) {
      const motif = history.slice(-period);
      if (new Set(motif).size > 1 && history.slice(-period * maxMotifRepeats).every((id, i) => id === motif[i % period])) {
        blocked.add(motif[0]); counts.motif_blocks = 1;
      }
    }
  }
  const kept = allowed.filter(id => !blocked.has(id));
  let scores = logits;
  if (repetitionPenalty !== 1) {
    scores = Float32Array.from(logits);
    const legal = new Set(kept);
    for (const id of new Set(history.slice(-repetitionWindow))) if (legal.has(id)) {
      scores[id] = scores[id] < 0 ? scores[id] * repetitionPenalty : scores[id] / repetitionPenalty;
      counts.penalty_steps = 1;
    }
  }
  return { logits: scores, allowed: kept, counts };
}

export function sampleToken(logits, allowed, rng, { temperature = .8, topK = 20, topP = .95 } = {}) {
  if (!allowed.length) throw new Error('No legal next melody token.');
  if (!Number.isFinite(temperature) || temperature <= 0 || !Number.isInteger(topK) || topK < 0
    || !Number.isFinite(topP) || topP <= 0 || topP > 1) throw new Error('Invalid sampling settings.');
  let entries = allowed.map(id => ({ id, score: logits[id] }));
  if (entries.some(entry => !Number.isFinite(entry.score))) throw new Error('The model produced non-finite legal-token logits.');
  entries.sort((a, b) => b.score - a.score || a.id - b.id);
  if (topK && topK < entries.length) entries = entries.slice(0, topK);
  // Subtract before scaling so even very small positive temperatures are safe.
  const maximum = entries[0].score;
  let sum = 0;
  for (const entry of entries) { entry.probability = Math.exp((entry.score - maximum) / temperature); sum += entry.probability; }
  for (const entry of entries) entry.probability /= sum;
  if (topP < 1) {
    let cumulative = 0; let keep = 0;
    do { cumulative += entries[keep++].probability; } while (keep < entries.length && cumulative <= topP);
    entries = entries.slice(0, keep);
    sum = entries.reduce((total, entry) => total + entry.probability, 0);
    for (const entry of entries) entry.probability /= sum;
  }
  const random = rng();
  if (!Number.isFinite(random) || random < 0 || random >= 1) throw new Error('Random generator must return a value from 0 to 1, excluding 1.');
  let target = random;
  for (const entry of entries) {
    if (target < entry.probability) return entry.id;
    target -= entry.probability;
  }
  return entries.findLast(entry => entry.probability > 0).id;
}

export class GenerationCancelled extends Error {
  constructor() { super('Generation cancelled.'); this.name = 'GenerationCancelled'; }
}

export async function generateSong({ infer, tokenizer, options, settings, seed, contextLength = 256,
  name = 'Melody', isCancelled = () => false, onProgress = () => {} }) {
  const ids = [tokenizer.bosId];
  const history = [];
  const interventions = { penalty_steps: 0, pitch_run_blocks: 0, motif_blocks: 0 };
  const rng = createRng(`${seed}:melody`);
  let forcedEos = false;
  if (settings.bpm !== null) ids.push(tokenizer.tokenToId.get(`<BPM_${settings.bpm}>`));
  while (ids.at(-1) !== tokenizer.eosId) {
    if (isCancelled()) throw new GenerationCancelled();
    let allowed = tokenizer.allowedNext(ids, options.minEvents, options.maxEvents);
    if (allowed.length === 1 && allowed[0] === tokenizer.eosId) { ids.push(tokenizer.eosId); forcedEos = true; break; }
    let logits = await infer(neuralContext(ids, contextLength));
    if (isCancelled()) throw new GenerationCancelled();
    if (logits.length !== tokenizer.vocabSize) throw new Error('The model output does not match its vocabulary.');
    logits = applyGuidance(logits, allowed, history, tokenizer, settings);
    if (ids.length >= 2 && (ids.length - 2) % 2 === 0) {
      const filtered = repetitionFilter(logits, allowed, history, options);
      logits = filtered.logits; allowed = filtered.allowed;
      for (const key of Object.keys(interventions)) interventions[key] += filtered.counts[key];
    }
    const next = sampleToken(logits, allowed, rng, options);
    ids.push(next);
    if (tokenizer.eventStartSet.has(next)) history.push(next);
    onProgress({ tokenCount: ids.length, eventCount: Math.floor((ids.length - 2) / 2), maxEvents: options.maxEvents });
    // WASM can resolve inference in the same task; give cancel messages time to
    // reach the worker even when no GPU await yields to the event loop.
    if (ids.length % 8 === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
  const song = tokenizer.decode(ids, name);
  return { ...song, settings: { ...settings, bpm: song.bpm }, seed, forcedEos, interventions };
}
