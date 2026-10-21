// Real-checkpoint CPU verification, not a browser/WebGPU/WASM lifecycle test.
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
  } finally { await session.release(); }
  const evidence = { status: 'PASS', verification: 'Real published ONNX weights, native ONNX Runtime CPU inference; no mocked logits.',
    scope: 'Verifies checkpoint compatibility, genuine source conditioning, exact prefix preservation and seeded sampling on native CPU. Does not validate browser worker behavior, WebGPU, WASM, mobile devices, musical quality or audio transcription accuracy.',
    modelSha256: sha256(bytes), vocabularySha256: sha256(vocabulary), modelUrl: manifest.modelUrl,
    runtime: ort.env.versions, executionProvider: 'cpu',
    initializationMs: Math.round(initialized - started), sourceRtttl,
    sameSeedIdentical: runs[0].rtttlSha256 === runs[1].rtttlSha256,
    // Different seeds can legitimately sample identical sequences; report this
    // observation rather than making diversity a correctness requirement.
    changedSeedDifferent: runs[0].rtttlSha256 !== runs[2].rtttlSha256,
    outputPolicy: 'Only output hashes are retained; generated model melodies are excluded from the public evidence.', runs };
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
