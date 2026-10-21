import { applyGuidance, createRng, planBatch } from './profiles.js';
import { eventSeconds, MAX_AUDIO_SECONDS, MAX_EVENTS } from './rtttl.js';

export const DEFAULT_OPTIONS = Object.freeze({
  numSongs: 4, profile: 'mixed', minEvents: 16, maxEvents: 48, temperature: .8,
  topK: 20, topP: .95, seed: 42, repetitionPenalty: 1.2, repetitionWindow: 16,
  maxPitchRun: 4, maxMotifRepeats: 3, maxMotifLength: 16,
});

const MODEL_DURATIONS = new Set([1, 2, 4, 8, 16, 32]);

// A prompt is a parsed score, not raw tokens or an audio recording. Only its
// musical fields enter the model; caller-owned events and cached seconds are
// never mutated or trusted. Unsupported notes are rejected, never rounded.
export function validatePromptSong(song) {
  if (!song || typeof song !== 'object' || Array.isArray(song)) throw new Error('A continuation needs a parsed promptSong.');
  if (!Number.isInteger(song.bpm) || song.bpm < 25 || song.bpm > 900) throw new Error('The source tempo must be an integer from 25 to 900 BPM.');
  if (!Array.isArray(song.events) || !song.events.length || song.events.length >= MAX_EVENTS) {
    throw new Error(`A continuation source must contain between 1 and ${MAX_EVENTS - 1} events.`);
  }
  const events = Array.from(song.events, (event, index) => {
    if (!event || typeof event !== 'object' || Array.isArray(event)) throw new Error(`Invalid source event ${index + 1}.`);
    if (event.pitch !== null && (!Number.isInteger(event.pitch) || event.pitch < 60 || event.pitch > 107)) {
      throw new Error(`Source note ${index + 1} must be a rest or a MIDI pitch from 60 to 107 (C4–B7).`);
    }
    if (!MODEL_DURATIONS.has(event.duration)) {
      throw new Error(`Source note ${index + 1}: the melody model supports durations 1, 2, 4, 8, 16 and 32; 64th notes cannot be continued without editing them first.`);
    }
    if (typeof event.dotted !== 'boolean') throw new Error(`Source note ${index + 1} must specify dotted as true or false.`);
    return { pitch: event.pitch, duration: event.duration, dotted: event.dotted };
  });
  const durationSeconds = events.reduce((total, event) => total + eventSeconds(song.bpm, event.duration, event.dotted), 0);
  if (durationSeconds >= MAX_AUDIO_SECONDS) throw new Error('The source already reaches the five-minute playback limit; select a shorter excerpt to continue.');
  return { name: typeof song.name === 'string' ? song.name.slice(0, 256) : 'Source', bpm: song.bpm, events, durationSeconds };
}

function validateContinuationBudget(song, options) {
  if (!Number.isInteger(options.minEvents) || !Number.isInteger(options.maxEvents)
      || options.minEvents < 1 || options.maxEvents < options.minEvents || options.maxEvents > 256) {
    throw new Error('A continuation must request between 1 and 256 additional events.');
  }
  if (song.events.length + options.maxEvents > MAX_EVENTS) {
    throw new Error(`Source plus additional events must not exceed ${MAX_EVENTS}; select a shorter excerpt or fewer additional events.`);
  }
  if (song.durationSeconds + options.minEvents * eventSeconds(song.bpm, 32) > MAX_AUDIO_SECONDS) {
    throw new Error('There is not enough time for the requested continuation within the five-minute playback limit.');
  }
}

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
  // Explicit continuation mode: min/maxEvents count NEW events. The source
  // tempo is authoritative; controls intended for a new song cannot retime it.
  if (value.promptSong !== undefined && value.promptSong !== null) {
    value.promptSong = validatePromptSong(value.promptSong);
    validateContinuationBudget(value.promptSong, value);
    value.bpm = value.promptSong.bpm;
    value.bpmRange = null;
  }
  const plan = planBatch(value);
  if (value.promptSong) for (const settings of plan) {
    settings.tempoSource = 'prompt';
    settings.continuation = true;
  }
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

  encodePrefix(song) {
    const source = validatePromptSong(song);
    const ids = [this.bosId, this.tokenToId.get(`<BPM_${source.bpm}>`)];
    for (const event of source.events) {
      ids.push(event.pitch === null ? this.restId : this.tokenToId.get(`<PITCH_${event.pitch}>`),
        this.tokenToId.get(`<DUR_${event.duration}_${event.dotted ? 'dot' : 'plain'}>`));
    }
    // No EOS: the next model decision extends the complete final source event.
    return ids;
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
  const prompt = options.promptSong === undefined || options.promptSong === null ? null : validatePromptSong(options.promptSong);
  if (prompt) validateContinuationBudget(prompt, options);
  const prefixEventCount = prompt?.events.length ?? 0;
  const ids = prompt ? tokenizer.encodePrefix(prompt) : [tokenizer.bosId];
  const history = prompt ? ids.slice(2).filter((_, index) => index % 2 === 0) : [];
  const minimum = prefixEventCount + options.minEvents;
  const maximum = prefixEventCount + options.maxEvents;
  const effectiveSettings = prompt ? { ...settings, bpm: prompt.bpm, tempoSource: 'prompt', continuation: true } : settings;
  const interventions = { penalty_steps: 0, pitch_run_blocks: 0, motif_blocks: 0 };
  const rng = createRng(`${seed}:melody`);
  let forcedEos = false;
  let durationSeconds = prompt?.durationSeconds ?? 0;
  if (!prompt && settings.bpm !== null) ids.push(tokenizer.tokenToId.get(`<BPM_${settings.bpm}>`));
  while (ids.at(-1) !== tokenizer.eosId) {
    if (isCancelled()) throw new GenerationCancelled();
    let allowed = tokenizer.allowedNext(ids, minimum, maximum);
    if (prompt) {
      const completeEvents = Math.floor((ids.length - 2) / 2);
      const shortest = eventSeconds(prompt.bpm, 32);
      if ((ids.length - 2) % 2) {
        // Leave enough duration for the minimum NEW event count, and never
        // produce a continued score that the shared player cannot import.
        const reserved = Math.max(0, minimum - completeEvents - 1) * shortest;
        allowed = allowed.filter(id => {
          const duration = tokenizer.durationById.get(id);
          return durationSeconds + eventSeconds(prompt.bpm, duration.duration, duration.dotted) + reserved <= MAX_AUDIO_SECONDS;
        });
        if (!allowed.length) throw new Error('The continuation exceeded the five-minute playback limit.');
      } else if (durationSeconds + shortest > MAX_AUDIO_SECONDS) {
        if (completeEvents < minimum) throw new Error('The continuation cannot fit the requested minimum within the five-minute playback limit.');
        allowed = [tokenizer.eosId];
      }
    }
    if (allowed.length === 1 && allowed[0] === tokenizer.eosId) { ids.push(tokenizer.eosId); forcedEos = true; break; }
    let logits = await infer(neuralContext(ids, contextLength));
    if (isCancelled()) throw new GenerationCancelled();
    if (logits.length !== tokenizer.vocabSize) throw new Error('The model output does not match its vocabulary.');
    logits = applyGuidance(logits, allowed, history, tokenizer, effectiveSettings);
    if (ids.length >= 2 && (ids.length - 2) % 2 === 0) {
      const filtered = repetitionFilter(logits, allowed, history, options);
      logits = filtered.logits; allowed = filtered.allowed;
      for (const key of Object.keys(interventions)) interventions[key] += filtered.counts[key];
    }
    const next = sampleToken(logits, allowed, rng, options);
    ids.push(next);
    if (tokenizer.eventStartSet.has(next)) history.push(next);
    if (prompt && tokenizer.durationById.has(next)) {
      const event = tokenizer.durationById.get(next);
      durationSeconds += eventSeconds(prompt.bpm, event.duration, event.dotted);
    }
    onProgress({ tokenCount: ids.length, eventCount: Math.max(0, Math.floor((ids.length - 2) / 2) - prefixEventCount),
      maxEvents: options.maxEvents, prefixEventCount });
    // WASM can resolve inference in the same task; give cancel messages time to
    // reach the worker even when no GPU await yields to the event loop.
    if (ids.length % 8 === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
  const song = tokenizer.decode(ids, name);
  return { ...song, settings: { ...effectiveSettings, bpm: song.bpm }, seed, forcedEos, interventions,
    continuation: Boolean(prompt), prefixEventCount, generatedEventCount: song.events.length - prefixEventCount };
}
