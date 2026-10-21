// Real-checkpoint continuation/variation/repair CPU verification, not a
// browser/WebGPU/WASM lifecycle test. No generated melodies are published.
// Run from web/: node scripts/verify-continuation.mjs MODEL_DIRECTORY [RESULT_JSON]
// MODEL_DIRECTORY must contain model.onnx and tokenizer.json matching
// public/model-manifest.json. This script does not download model files.
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { EventTokenizer, generateSong, validateOptions } from '../src/sampler.js';
import { parseRtttl } from '../src/rtttl.js';

const [modelDirectory, resultPath, ...extra] = process.argv.slice(2);
if (!modelDirectory || extra.length) {
  throw new Error('Usage: node scripts/verify-continuation.mjs MODEL_DIRECTORY [RESULT_JSON]');
}
const manifest = JSON.parse(readFileSync(new URL('../public/model-manifest.json', import.meta.url)));
const bytes = readFileSync(join(resolve(modelDirectory), 'model.onnx'));
const vocabulary = readFileSync(join(resolve(modelDirectory), 'tokenizer.json'));
const sha256 = value => createHash('sha256').update(value).digest('hex');
assert.equal(sha256(bytes), manifest.modelSha256, 'Model must match the pinned release.');
assert.equal(sha256(vocabulary), manifest.tokenizerSha256, 'Tokenizer must match the pinned release.');
const tokenizer = new EventTokenizer(JSON.parse(vocabulary));
const sourceRtttl = 'Input:d=8,o=5,b=120:c,c,g,g,a,a,4g,p';
const promptSong = parseRtttl(sourceRtttl);
const original = structuredClone(promptSong);
async function verify(ort) {
  const started = performance.now();
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: 2, interOpNumThreads: 1,
  });
  const initialized = performance.now();
  const runs = [];
  const editRuns = [];
  try {
    for (const seed of [42, 42, 43]) {
      const { options, plan } = validateOptions({ promptSong, minEvents: 8, maxEvents: 8,
        profile: 'none', numSongs: 1, seed, temperature: 0.8, topK: 20, topP: 0.95 });
      let inferenceCalls = 0;
      let firstContext = null;
      let logitRange = null;
      const runStart = performance.now();
      const result = await generateSong({ tokenizer, options, settings: plan[0], seed, contextLength: manifest.contextLength,
        infer: async context => {
          inferenceCalls++;
          firstContext ??= [...context];
          const tensor = new ort.Tensor('int64', BigInt64Array.from(context, BigInt), [1, context.length]);
          let prediction;
          try {
            prediction = await session.run({ input_ids: tensor });
            const output = prediction.logits;
            assert.equal(output.dims.at(-1), 940);
            const values = Float32Array.from(output.data.subarray(output.data.length - 940));
            assert.ok([...values].every(Number.isFinite));
            logitRange ??= [Math.min(...values), Math.max(...values)];
            return values;
          } finally {
            tensor.dispose();
            if (prediction) Object.values(prediction).forEach(value => value.dispose());
          }
        },
      });
      assert.deepEqual(firstContext, tokenizer.encodePrefix(promptSong));
      assert.deepEqual(result.events.slice(0, promptSong.events.length), original.events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted })));
      assert.equal(result.bpm, promptSong.bpm);
      assert.equal(result.generatedEventCount, 8);
      assert.equal(inferenceCalls, 16);
      assert.ok(logitRange[1] > logitRange[0] + 1);
      assert.deepEqual(parseRtttl(result.rtttl).events.slice(0, original.events.length), original.events);
      runs.push({ seed, inferenceCalls, elapsedMs: Math.round(performance.now() - runStart), firstContext,
        logitRange, prefixEventCount: result.prefixEventCount, generatedEventCount: result.generatedEventCount,
        rtttlSha256: sha256(result.rtttl) });
    }
    assert.equal(runs[0].rtttlSha256, runs[1].rtttlSha256);
    assert.deepEqual(promptSong, original);
    for (const editMode of ['variation', 'repair']) for (const seed of [42, 42, 43]) {
      const selection = { start: 2, end: 6 };
      const { options, plan } = validateOptions({ promptSong, editMode, selection, editStrength: 'medium',
        profile: 'none', numSongs: 1, seed, temperature: .8, topK: 20, topP: .95 });
      let inferenceCalls = 0;
      let firstContext = null;
      let logitRange = null;
      const runStart = performance.now();
      const result = await generateSong({ tokenizer, options, settings: plan[0], seed, contextLength: manifest.contextLength,
        infer: async context => {
          inferenceCalls++;
          firstContext ??= [...context];
          const tensor = new ort.Tensor('int64', BigInt64Array.from(context, BigInt), [1, context.length]);
          let prediction;
          try {
            prediction = await session.run({ input_ids: tensor });
            const output = prediction.logits;
            assert.equal(output.dims.at(-1), 940);
            const values = Float32Array.from(output.data.subarray(output.data.length - 940));
            assert.ok([...values].every(Number.isFinite));
            logitRange ??= [Math.min(...values), Math.max(...values)];
            return values;
          } finally {
            tensor.dispose();
            if (prediction) Object.values(prediction).forEach(value => value.dispose());
          }
        },
      });
      assert.deepEqual(firstContext, tokenizer.encodePrefix(promptSong).slice(0, firstContext.length));
      assert.equal(result.bpm, original.bpm);
      assert.equal(result.events.length, original.events.length);
      let changedEventCount = 0;
      for (let index = 0; index < original.events.length; index++) {
        const source = original.events[index], edited = result.events[index];
        assert.equal(edited.duration, source.duration);
        assert.equal(edited.dotted, source.dotted);
        if (index < selection.start || index >= selection.end || source.pitch === null) assert.equal(edited.pitch, source.pitch);
        else assert.ok(Math.abs(edited.pitch - source.pitch) <= result.maxPitchDistance);
        changedEventCount += Number(edited.pitch !== source.pitch);
      }
      assert.equal(result.changedEventCount, changedEventCount);
      assert.ok(changedEventCount <= result.maxChangedEvents);
      if (editMode === 'variation') assert.ok(changedEventCount > 0);
      else {
        assert.equal(result.repairMethod, 'forward-likelihood-reranking');
        assert.equal(result.scoreDetails.suffixEventCount, 2);
        assert.equal(result.scoreDetails.candidates[0].original, true);
        assert.ok(result.scoreDetails.candidateCount >= 2);
        assert.ok(result.scoreDetails.scoreDelta >= 0);
        assert.equal(result.originalChosen, changedEventCount === 0);
      }
      assert.equal(inferenceCalls, result.inferenceCalls);
      assert.ok(inferenceCalls > 0 && inferenceCalls <= 320);
      assert.ok(logitRange[1] > logitRange[0] + 1);
      assert.deepEqual(promptSong, original);
      assert.equal(parseRtttl(result.rtttl).events.length, original.events.length);
      editRuns.push({ editMode, seed, selection, editStrength: result.editStrength,
        inferenceCalls, elapsedMs: Math.round(performance.now() - runStart), logitRange,
        firstContextMatchesSourcePrefix: true, unchangedOutsideSelection: true, timingAndRestsPreserved: true,
        changedEventCount, originalChosen: result.originalChosen, repairMethod: result.repairMethod,
        scoreDetails: result.scoreDetails, rtttlSha256: sha256(result.rtttl) });
    }
    for (const mode of ['variation', 'repair']) {
      const matching = editRuns.filter(run => run.editMode === mode);
      assert.equal(matching[0].rtttlSha256, matching[1].rtttlSha256);
    }
  } finally { await session.release(); }
  const evidence = { status: 'PASS', verification: 'Real published ONNX weights, native ONNX Runtime CPU inference; no mocked logits.',
    scope: 'Verifies checkpoint compatibility, genuine source conditioning, exact continuation prefixes, constrained selected-pitch variation, baseline-and-suffix repair scoring, unchanged rhythm/rests/unselected events, and seeded sampling on native CPU. Does not validate browser worker behavior, WebGPU, WASM, mobile devices, musical quality or audio transcription accuracy. Repair is autoregressive likelihood reranking, not trained masked infilling or acoustic correction.',
    modelSha256: sha256(bytes), vocabularySha256: sha256(vocabulary), modelUrl: manifest.modelUrl,
    runtime: ort.env.versions, executionProvider: 'cpu',
    initializationMs: Math.round(initialized - started), sourceRtttl,
    sameSeedIdentical: runs[0].rtttlSha256 === runs[1].rtttlSha256,
    // Different seeds can legitimately sample identical sequences; report this
    // observation rather than making diversity a correctness requirement.
    changedSeedDifferent: runs[0].rtttlSha256 !== runs[2].rtttlSha256,
    editSameSeedIdentical: ['variation', 'repair'].every(mode => {
      const matching = editRuns.filter(run => run.editMode === mode);
      return matching[0].rtttlSha256 === matching[1].rtttlSha256;
    }),
    outputPolicy: 'Only output hashes and aggregate scores are retained; generated model melodies are excluded from the public evidence.', runs, editRuns };
  return evidence;
}

// Native ONNX may write a small telemetry/session file in its working directory.
// Resolve caller paths before moving, then isolate even module initialization.
const outputPath = resultPath ? resolve(resultPath) : null;
const originalDirectory = process.cwd();
const runtimeDirectory = mkdtempSync(join(tmpdir(), 'rtttl-native-verifier-'));
let evidence;
try {
  process.chdir(runtimeDirectory);
  evidence = await verify(await import('onnxruntime-node'));
} finally {
  process.chdir(originalDirectory);
  rmSync(runtimeDirectory, { recursive: true, force: true });
}
const report = JSON.stringify(evidence, null, 2) + '\n';
if (outputPath) writeFileSync(outputPath, report);
process.stdout.write(report);
