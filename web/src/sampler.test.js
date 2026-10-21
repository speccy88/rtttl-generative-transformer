import test from 'node:test';
import assert from 'node:assert/strict';
import { PROFILES, PROFILE_NAMES, applyGuidance, createRng, normalizeTonic, planBatch } from './profiles.js';
import { DEFAULT_OPTIONS, EventTokenizer, GenerationCancelled, generateSong, neuralContext,
  repetitionFilter, sampleToken, toRtttl, validateOptions, validatePromptSong } from './sampler.js';
import { parseRtttl } from './rtttl.js';

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

test('continuation tokenization preserves source notes, rests, dots and tempo without EOS', () => {
  const source = parseRtttl('Source:d=8,o=5,b=137:c#,p.,4f#,2a6.');
  const original = structuredClone(source);
  const encoded = tokenizer.encodePrefix(source);
  assert.deepEqual(encoded, [1, token('<BPM_137>'), pitch(73), duration,
    tokenizer.restId, token('<DUR_8_dot>'), pitch(78), token('<DUR_4_plain>'), pitch(93), token('<DUR_2_dot>')]);
  assert.notEqual(encoded.at(-1), tokenizer.eosId);
  assert.deepEqual(source, original);
  const decoded = tokenizer.decode([...encoded, tokenizer.eosId]);
  assert.deepEqual(decoded.events, source.events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted })));
});

test('continuation overrides new-song tempo controls and does not mutate the supplied options', () => {
  const promptSong = parseRtttl('Source:d=8,o=5,b=113:c,d,e');
  const supplied = { promptSong, numSongs: 3, bpm: 172, bpmRange: [90, 100], minEvents: 2, maxEvents: 8 };
  const before = structuredClone(supplied);
  const validated = validateOptions(supplied);
  assert.deepEqual(supplied, before);
  assert.notEqual(validated.options.promptSong, promptSong);
  assert.notEqual(validated.options.promptSong.events, promptSong.events);
  assert.equal(validated.options.minEvents, 2);
  assert.equal(validated.options.maxEvents, 8);
  assert.equal(validated.options.bpm, 113);
  assert.equal(validated.options.bpmRange, null);
  assert.ok(validated.plan.every(row => row.bpm === 113 && row.tempoSource === 'prompt' && row.continuation));
});

test('actual source token prefix reaches inference and only model-predicted events are appended', async () => {
  const promptSong = parseRtttl('Input:d=8,o=5,b=137:c#,p.,4f#,2a6.');
  const original = structuredClone(promptSong);
  const config = options({ promptSong, minEvents: 2, maxEvents: 2, topK: 1, profile: 'cinematic' });
  const contexts = [], progress = [];
  const result = await generateSong({ tokenizer, options: config,
    settings: { ...planBatch({ numSongs: 1, profile: 'cinematic' })[0], bpm: 60 }, seed: 17,
    onProgress: event => progress.push(event), infer: async context => {
      contexts.push([...context]);
      const scores = flat(); scores[pitch(95)] = 20; scores[token('<DUR_16_dot>')] = 20; return scores;
    },
  });
  assert.deepEqual(contexts[0], tokenizer.encodePrefix(promptSong));
  assert.equal(contexts.length, 4); // Two genuine pitch decisions and two duration decisions.
  assert.equal(result.bpm, 137);
  assert.equal(result.events.length, promptSong.events.length + 2);
  assert.deepEqual(result.events.slice(0, 4), promptSong.events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted })));
  assert.deepEqual(result.events.slice(4), Array.from({ length: 2 }, () => ({ pitch: 95, duration: 16, dotted: true })));
  assert.deepEqual(promptSong, original);
  assert.equal(result.continuation, true);
  assert.equal(result.prefixEventCount, 4);
  assert.equal(result.generatedEventCount, 2);
  assert.equal(result.settings.tempoSource, 'prompt');
  assert.equal(result.forcedEos, true);
  assert.deepEqual(progress.map(row => row.eventCount), [0, 1, 1, 2]);
  assert.ok(progress.every(row => row.maxEvents === 2 && row.prefixEventCount === 4));
  assert.deepEqual(parseRtttl(result.rtttl).events.slice(0, 4), promptSong.events);
});

test('continuation cannot immediately stop by emitting EOS or return only a copy', async () => {
  const promptSong = parseRtttl('Input:d=8,o=5,b=120:c,d');
  const config = options({ promptSong, minEvents: 3, maxEvents: 8, topK: 1, profile: 'none' });
  let calls = 0;
  const result = await generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1, profile: 'none' })[0], seed: 1,
    infer: async () => { calls++; const scores = flat(); scores[2] = 100; scores[pitch(67)] = 90; scores[duration] = 90; return scores; },
  });
  assert.equal(result.generatedEventCount, 3);
  assert.equal(calls, 7); // Three complete notes, followed by the first legal EOS decision.
  assert.equal(result.forcedEos, false);
});

test('long continuation prefixes are retained in output while model context slides at event boundaries', async () => {
  const promptSong = { name: 'Long', bpm: 120,
    events: Array.from({ length: 140 }, (_, index) => ({ pitch: 60 + index % 12, duration: 32, dotted: false })) };
  const config = options({ promptSong, minEvents: 2, maxEvents: 2, profile: 'none' });
  const contexts = [];
  const result = await generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1, profile: 'none' })[0], seed: 4,
    contextLength: 16, infer: async context => { contexts.push([...context]); return flat(); },
  });
  assert.deepEqual(contexts[0], neuralContext(tokenizer.encodePrefix(promptSong), 16));
  assert.ok(contexts.every(context => context.length <= 16 && context[0] === 1 && context[1] === token('<BPM_120>')));
  for (const context of contexts) {
    for (let i = 2; i < context.length; i++) {
      assert.equal(i % 2 === 0 ? tokenizer.eventStartSet.has(context[i]) : tokenizer.durationById.has(context[i]), true);
    }
  }
  assert.deepEqual(result.events.slice(0, 140), promptSong.events);
  assert.equal(result.events.length, 142);
  assert.equal(contexts.length, 4);
});

test('continuation repetition controls include source history but never rewrite it', async () => {
  const promptSong = parseRtttl('Repeated:d=8,o=5,b=120:c,c,c,c');
  const config = options({ promptSong, minEvents: 1, maxEvents: 1, topK: 1, profile: 'none', maxPitchRun: 4 });
  const result = await generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1, profile: 'none' })[0], seed: 3,
    infer: async () => { const scores = flat(); scores[pitch(72)] = 100; scores[pitch(74)] = 90; scores[duration] = 80; return scores; },
  });
  assert.deepEqual(result.events.slice(0, 4).map(event => event.pitch), [72, 72, 72, 72]);
  assert.equal(result.events[4].pitch, 74);
  assert.equal(result.interventions.pitch_run_blocks, 1);
});

test('continuation remains deterministic for a seed and varies sampled additions for another seed', async () => {
  const config = options({ promptSong: parseRtttl('Seed:d=8,o=5,b=120:c,e,g'), minEvents: 4, maxEvents: 4,
    profile: 'none', topK: 0, topP: 1 });
  const run = seed => generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1, profile: 'none' })[0],
    seed, infer: async () => flat() });
  const first = await run(5), repeated = await run(5), changed = await run(6);
  assert.deepEqual(repeated, first);
  assert.notDeepEqual(changed.events.slice(3), first.events.slice(3));
  assert.deepEqual(changed.events.slice(0, 3), first.events.slice(0, 3));
});

test('unsupported continuation input is rejected before inference, without coercion', async () => {
  const base = { name: 'Source', bpm: 120, events: [{ pitch: 72, duration: 8, dotted: false }] };
  const invalid = [false, {}, { ...base, events: [] }, { ...base, events: Array(2) },
    { ...base, bpm: 0 }, { ...base, bpm: 120.5 },
    ...[59, 108, 72.5, undefined, '72'].map(pitch => ({ ...base, events: [{ ...base.events[0], pitch }] })),
    ...[64, 3, '8'].map(duration => ({ ...base, events: [{ ...base.events[0], duration }] })),
    { ...base, events: [{ ...base.events[0], dotted: 0 }] },
  ];
  let calls = 0;
  for (const promptSong of invalid) {
    assert.throws(() => validateOptions({ promptSong }), JSON.stringify(promptSong));
    await assert.rejects(generateSong({ tokenizer, options: { ...options(), promptSong }, settings: planBatch({ numSongs: 1 })[0],
      seed: 1, infer: async () => { calls++; return flat(); } }));
  }
  await assert.rejects(generateSong({ tokenizer, options: { ...options(), minEvents: 0, promptSong: base },
    settings: planBatch({ numSongs: 1 })[0], seed: 1, infer: async () => { calls++; return flat(); } }), /additional events/);
  assert.equal(calls, 0);
  assert.throws(() => validatePromptSong({ ...base, events: [{ ...base.events[0], duration: 64 }] }), /64th notes/);
});

test('continuation enforces combined event and playable duration budgets', async () => {
  const short = { pitch: 72, duration: 32, dotted: false };
  assert.throws(() => validateOptions({ promptSong: { bpm: 900, events: Array(2040).fill(short) }, minEvents: 1, maxEvents: 16 }), /2048/);
  const almostFull = { name: 'Long', bpm: 120, events: [
    ...Array.from({ length: 149 }, () => ({ pitch: 72, duration: 1, dotted: false })),
    { pitch: 74, duration: 2, dotted: true }, { pitch: 76, duration: 8, dotted: true },
  ] }; // 299.875 seconds: only two 32nd notes fit.
  assert.throws(() => validateOptions({ promptSong: almostFull, minEvents: 3, maxEvents: 4 }), /not enough time/);
  const config = options({ promptSong: almostFull, minEvents: 2, maxEvents: 6, profile: 'none', topK: 1 });
  const result = await generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1, profile: 'none' })[0], seed: 7,
    infer: async () => { const scores = flat(); scores[pitch(79)] = 100; scores[token('<DUR_1_plain>')] = 100; return scores; },
  });
  assert.equal(result.generatedEventCount, 2);
  assert.deepEqual(result.events.slice(-2).map(event => event.duration), [32, 32]);
  assert.ok(parseRtttl(result.rtttl).durationSeconds <= 300);
  assert.equal(result.forcedEos, true);
});

test('cancellation during continuation retains the caller source and stops inference', async () => {
  const promptSong = parseRtttl('Cancel:d=8,o=5,b=120:c,p,d.');
  const original = structuredClone(promptSong);
  const config = options({ promptSong });
  let cancelled = false, calls = 0;
  await assert.rejects(generateSong({ tokenizer, options: config, settings: planBatch({ numSongs: 1 })[0], seed: 1,
    infer: async () => { calls++; cancelled = true; return flat(); }, isCancelled: () => cancelled,
  }), GenerationCancelled);
  assert.equal(calls, 1);
  assert.deepEqual(promptSong, original);
});
