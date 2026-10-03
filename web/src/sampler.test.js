import test from 'node:test';
import assert from 'node:assert/strict';
import { PROFILES, PROFILE_NAMES, applyGuidance, createRng, normalizeTonic, planBatch } from './profiles.js';
import { DEFAULT_OPTIONS, EventTokenizer, GenerationCancelled, generateSong, neuralContext,
  repetitionFilter, sampleToken, toRtttl, validateOptions } from './sampler.js';

const tokenizer = new EventTokenizer();
const token = text => tokenizer.tokenToId.get(text);
const pitch = note => token(`<PITCH_${note}>`);
const duration = token('<DUR_8_plain>');
const flat = () => new Float32Array(940);
const options = overrides => validateOptions({ ...DEFAULT_OPTIONS, minEvents: 8, maxEvents: 24, ...overrides }).options;

test('tokenizer uses the exact 940-token release vocabulary and rejects incompatible downloads', () => {
  assert.equal(tokenizer.vocabSize, 940);
  assert.equal(tokenizer.restId, 927);
  assert.doesNotThrow(() => new EventTokenizer({ version: 1, tokens: tokenizer.tokens }));
  assert.throws(() => new EventTokenizer({ version: 1, tokens: [...tokenizer.tokens].reverse() }), /incompatible/);
  assert.throws(() => new EventTokenizer({ version: 2, tokens: tokenizer.tokens }), /incompatible/);
});

test('grammar strictly alternates event starts and durations, allowing EOS only at complete events', () => {
  const ids = [1, token('<BPM_120>')];
  assert.deepEqual(tokenizer.allowedNext([1]), tokenizer.bpmIds);
  assert.equal(tokenizer.allowedNext(ids, 2).includes(2), false);
  ids.push(pitch(72));
  assert.deepEqual(tokenizer.allowedNext(ids, 2), tokenizer.durationIds);
  ids.push(duration, tokenizer.restId, duration);
  assert.equal(tokenizer.allowedNext(ids, 2).includes(2), true);
  assert.deepEqual(tokenizer.allowedNext(ids, 2, 2), [2]);
  assert.throws(() => tokenizer.decode([...ids, pitch(72), 2]), /Incomplete/);
  const song = tokenizer.decode([...ids, 2], 'A melody');
  assert.deepEqual(song.events, [{ pitch: 72, duration: 8, dotted: false }, { pitch: null, duration: 8, dotted: false }]);
  assert.equal(song.rtttl, 'A melody:d=4,o=6,b=120:8c5,8p');
});

test('sliding context preserves BOS/BPM and event alignment, including a trailing pitch', () => {
  const ids = [1, token('<BPM_100>'), pitch(60), duration, pitch(61), duration, pitch(62), duration, pitch(63)];
  assert.deepEqual(neuralContext(ids, 6), [1, token('<BPM_100>'), pitch(62), duration, pitch(63)]);
  assert.deepEqual(neuralContext(ids.slice(0, -1), 6), [1, token('<BPM_100>'), pitch(61), duration, pitch(62), duration]);
  assert.throws(() => neuralContext(ids, 3), /four/);
});

test('mixed planning balances profiles, varies keys and tempos, and keeps prefixes stable', () => {
  const plan = planBatch({ numSongs: 24, seed: 1407 });
  assert.deepEqual(plan.slice(0, 5), planBatch({ numSongs: 5, seed: 1407 }));
  assert.equal(new Set(plan.slice(0, 5).map(row => row.profile)).size, 5);
  assert.equal(new Set(plan.slice(0, 12).map(row => row.tonic)).size, 12);
  assert.ok(new Set(plan.map(row => row.bpm)).size > 10);
  for (const row of plan) {
    const spec = PROFILES[row.profile];
    assert.ok(row.bpm >= spec.bpmRange[0] && row.bpm <= spec.bpmRange[1]);
    assert.equal(row.mode, spec.defaultMode);
    assert.equal(row.tempoSource, 'profile_range');
  }
  const fixed = planBatch({ numSongs: 24, seed: 1407, bpm: 113 });
  assert.ok(fixed.every(row => row.bpm === 113));
  assert.deepEqual(fixed.map(({ profile, tonic }) => ({ profile, tonic })), plan.map(({ profile, tonic }) => ({ profile, tonic })));
});

test('custom tempo, tonic and mode override profiles; unguided leaves tempo to the model', () => {
  assert.equal(normalizeTonic('Bb'), 'A#');
  assert.equal(normalizeTonic('f♯'), 'F#');
  assert.throws(() => normalizeTonic('A;bad'), /Tonic/);
  const rows = planBatch({ profile: 'cinematic', numSongs: 8, bpmRange: [130, 131], tonic: 'Bb', mode: 'major' });
  for (const row of rows) {
    assert.ok(row.bpm === 130 || row.bpm === 131);
    assert.equal(row.tonic, 'A#'); assert.equal(row.mode, 'major'); assert.equal(row.tempoSource, 'range');
  }
  assert.equal(planBatch({ profile: 'none' })[0].bpm, null);
  assert.equal(planBatch({ profile: 'none' })[0].profile, null);
});

test('invalid settings fail before inference', async () => {
  const invalid = [
    { numSongs: 0 }, { numSongs: 25 }, { numSongs: true }, { seed: NaN }, { seed: 1.5 },
    { bpm: 0 }, { bpm: true }, { bpm: 120, bpmRange: [90, 100] }, { bpmRange: [100, 90] },
    { bpmRange: [90] }, { profile: 'unknown' }, { mode: 'minor' }, { tonic: 'H' },
    { minEvents: 0 }, { minEvents: 9, maxEvents: 8 }, { maxEvents: 257 },
    { temperature: NaN }, { temperature: 0 }, { topK: -1 }, { topP: 0 },
    { repetitionPenalty: .5 }, { maxMotifRepeats: 1 }, { maxPitchRun: false },
  ];
  for (const settings of invalid) assert.throws(() => validateOptions(settings), JSON.stringify(settings));
});

test('sampler honors legal support, exact top-k ties, nucleus crossing token, and finite logits', () => {
  const scores = flat();
  scores[0] = 1000; // illegal dominant token must not change distribution
  assert.equal(sampleToken(scores, [5, 6, 7], () => .99, { topK: 1 }), 5);
  assert.equal(sampleToken(scores, [5, 6, 7], () => .99, { topK: 0, topP: .5 }), 6);
  assert.equal(sampleToken(scores, [5, 6, 7], () => .99, { topK: 0, topP: .01 }), 5);
  scores[5] = 10; scores[6] = 8;
  assert.equal(sampleToken(scores, [5, 6], () => .99, { temperature: Number.MIN_VALUE }), 5);
  scores[5] = Infinity;
  assert.throws(() => sampleToken(scores, [5, 6], () => .5), /non-finite/);
  assert.throws(() => sampleToken(scores, [], () => .5), /No legal/);
  assert.throws(() => sampleToken(flat(), [5], () => 1), /Random/);
});

test('repetition controls block both collapsed pitch runs and repeating motifs without mutating logits', () => {
  const scores = flat(); scores[pitch(60)] = 12;
  const config = options({ maxPitchRun: 4, maxMotifRepeats: 3 });
  const run = repetitionFilter(scores, tokenizer.eventStartIds, Array(4).fill(pitch(60)), config);
  assert.equal(run.allowed.includes(pitch(60)), false);
  assert.equal(run.counts.pitch_run_blocks, 1);
  assert.equal(scores[pitch(60)], 12);
  const motif = repetitionFilter(scores, [...tokenizer.eventStartIds, 2], [pitch(60), pitch(62), pitch(64), pitch(60), pitch(62), pitch(64), pitch(60), pitch(62), pitch(64)], config);
  assert.equal(motif.allowed.includes(pitch(60)), false);
  assert.equal(motif.allowed.includes(2), true);
  assert.equal(motif.counts.motif_blocks, 1);
  const negative = flat(); negative[pitch(62)] = -2;
  const penalty = repetitionFilter(negative, tokenizer.eventStartIds, [pitch(62)], config);
  assert.ok(penalty.logits[pitch(62)] < -2);
  assert.equal(negative[pitch(62)], -2);
});

test('all soft guides keep EOS unchanged and contrast sustained versus quick notes', () => {
  const scores = flat(); scores[2] = .7;
  for (const profile of PROFILE_NAMES) {
    const settings = planBatch({ profile, numSongs: 1 })[0];
    const guided = applyGuidance(scores, [...tokenizer.eventStartIds, 2], [pitch(72)], tokenizer, settings);
    assert.equal(guided[2], scores[2]); assert.equal(scores[pitch(72)], 0);
    assert.ok([...guided].every(Number.isFinite));
  }
  const cinematic = applyGuidance(flat(), tokenizer.durationIds, [pitch(72)], tokenizer, planBatch({ profile: 'cinematic', numSongs: 1 })[0]);
  const chiptune = applyGuidance(flat(), tokenizer.durationIds, [pitch(72)], tokenizer, planBatch({ profile: 'chiptune', numSongs: 1 })[0]);
  assert.ok(cinematic[token('<DUR_2_plain>')] > chiptune[token('<DUR_2_plain>')]);
  assert.ok(chiptune[token('<DUR_16_plain>')] > cinematic[token('<DUR_16_plain>')]);
});

test('synthetic collapsed model generates valid songs with run controls and reproducible metadata', async () => {
  const config = options({ minEvents: 20, maxEvents: 20, topK: 1 });
  const settings = planBatch({ profile: 'chiptune', numSongs: 1, seed: 12 })[0];
  const infer = async () => { const logits = flat(); logits[pitch(75)] = 100; logits[pitch(77)] = 99; logits[duration] = 90; return logits; };
  const generated = await generateSong({ infer, tokenizer, options: config, settings, seed: 12 });
  const again = await generateSong({ infer, tokenizer, options: config, settings, seed: 12 });
  assert.deepEqual(generated, again);
  assert.equal(generated.events.length, 20);
  assert.equal(generated.bpm, settings.bpm);
  assert.equal(generated.settings.profile, 'chiptune');
  assert.equal(generated.forcedEos, true);
  assert.ok(generated.interventions.pitch_run_blocks > 0);
  let longest = 0, run = 0, last;
  for (const event of generated.events) { run = event.pitch === last ? run + 1 : 1; last = event.pitch; longest = Math.max(longest, run); }
  assert.ok(longest <= 4);
  assert.equal(toRtttl(generated), generated.rtttl);
});

test('learned BPM and natural EOS remain model decisions for unguided generation', async () => {
  const config = options({ minEvents: 2, maxEvents: 8, topK: 1 });
  const settings = planBatch({ profile: 'none', numSongs: 1 })[0];
  const generated = await generateSong({ tokenizer, options: config, settings, seed: 5, infer: async () => {
    const scores = flat(); scores[token('<BPM_135>')] = 100; scores[2] = 90;
    scores[pitch(60)] = 80; scores[duration] = 70; return scores;
  } });
  assert.equal(generated.bpm, 135); assert.equal(generated.settings.bpm, 135);
  assert.equal(generated.events.length, 2); assert.equal(generated.forcedEos, false);
});

test('cancellation rejects an incomplete melody and stops additional model calls', async () => {
  let cancelled = false, calls = 0;
  await assert.rejects(generateSong({ tokenizer, options: options(), settings: planBatch({ numSongs: 1 })[0], seed: 1,
    infer: async () => { calls++; cancelled = true; return flat(); }, isCancelled: () => cancelled,
  }), GenerationCancelled);
  assert.equal(calls, 1);
});

test('local random streams reproduce without affecting each other', () => {
  const one = createRng(10), two = createRng(10), other = createRng(11);
  const a = Array.from({ length: 20 }, one);
  Array.from({ length: 40 }, other);
  assert.deepEqual(Array.from({ length: 20 }, two), a);
  assert.ok(a.every(value => value >= 0 && value < 1));
});
