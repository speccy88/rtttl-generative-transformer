// Real static-site inference checks. No worker, logits, transcription or sampling
// mocks. The generator download is fulfilled with hash-verified published bytes
// to make browser CORS/network timing independent of inference correctness.
import { chromium, expect, firefox, test, webkit } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { parseRtttl } from '../src/rtttl.js';
import { encodeMp3Pcm } from '../src/mp3-worker.js';

const manifest = JSON.parse(await readFile(new URL('../public/model-manifest.json', import.meta.url)));
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
let publishedFiles;
async function generatorFiles() {
  publishedFiles ??= Promise.all([
    [manifest.modelUrl, manifest.modelSha256], [manifest.tokenizerUrl, manifest.tokenizerSha256],
  ].map(async ([url, hash]) => {
    const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
    if (!response.ok) throw new Error(`Published fixture download failed: HTTP ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    expect(sha256(bytes), 'Published bytes must match the application manifest').toBe(hash);
    return [url, bytes];
  }));
  return publishedFiles;
}

function fixturePcm() {
  const sampleRate = 44100, duration = 2.5;
  const samples = new Int16Array(sampleRate * duration);
  // Original synthetic C5/E5/G5/E5 phrase, with a quarter-second leading rest.
  for (const [index, midi] of [72, 76, 79, 76].entries()) {
    const frequency = 440 * 2 ** ((midi - 69) / 12);
    const start = Math.round((0.25 + index * 0.5) * sampleRate);
    const frames = Math.round(0.45 * sampleRate);
    for (let frame = 0; frame < frames; frame++) {
      const time = frame / sampleRate;
      const envelope = Math.min(1, time / 0.015, (frames - frame) / (0.04 * sampleRate));
      const phase = 2 * Math.PI * frequency * time;
      samples[start + frame] = Math.round(24000 * envelope * (0.65 * Math.sin(phase) + 0.24 * Math.sin(2 * phase) + 0.1 * Math.sin(3 * phase)));
    }
  }
  return { sampleRate, samples };
}
function fixtureWav({ sampleRate, samples }) {
  const bytes = Buffer.alloc(44 + samples.length * 2);
  bytes.write('RIFF'); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write('WAVE', 8);
  bytes.write('fmt ', 12); bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22); bytes.writeUInt32LE(sampleRate, 24); bytes.writeUInt32LE(sampleRate * 2, 28);
  bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34); bytes.write('data', 36); bytes.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i++) bytes.writeInt16LE(samples[i], 44 + i * 2);
  return bytes;
}
function installWorkerObserver() {
  window.__actualWorkerEvents = [];
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', ({ data }) => window.__actualWorkerEvents.push(data));
    }
  };
}

function expectPitchOnlyEdit(before, after, selection) {
  expect(after.bpm).toBe(before.bpm);
  expect(after.events.length).toBe(before.events.length);
  let changed = 0;
  for (let index = 0; index < before.events.length; index++) {
    const original = before.events[index], edited = after.events[index];
    expect(edited.duration).toBe(original.duration);
    expect(edited.dotted).toBe(original.dotted);
    expect(edited.pitch === null).toBe(original.pitch === null);
    if (index < selection.start || index >= selection.end || original.pitch === null) expect(edited).toEqual(original);
    else changed += Number(edited.pitch !== original.pitch);
  }
  return changed;
}

async function selectedTranscription(page) {
  const { library, transcription } = await page.evaluate(() => ({
    library: JSON.parse(localStorage.getItem('pocket-composer-library-v1')),
    transcription: window.__actualWorkerEvents.findLast(event => event.type === 'result' && event.result?.candidates)?.result,
  }));
  expect(transcription?.analysis.model).toBe('Spotify Basic Pitch 0.4.0');
  expect(transcription.analysis.backend).toBe('wasm');
  expect(transcription.evidence.frameCount).toBeGreaterThan(0);
  expect(transcription.evidence.events.length).toBeGreaterThan(0);
  const selected = library.songs.find(song => song.id === library.selectedId);
  expect(selected?.source).toBe('converted');
  expect(selected.rtttl).toBe(transcription.candidates[0].rtttl);
  await expect(page.locator('#score-input')).toHaveValue(selected.rtttl);
  return { library, selected, song: parseRtttl(selected.rtttl) };
}

for (const browserName of ['chromium', 'firefox', 'webkit']) {
  test(`real Basic Pitch transcribes a local WAV on static ${browserName} without an API`, async ({ baseURL }, testInfo) => {
    test.setTimeout(180000);
    const browser = await { chromium, firefox, webkit }[browserName].launch();
    const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
    const requests = [];
    const origin = new URL(baseURL).origin;
    context.on('request', request => requests.push({ method: request.method(), url: request.url() }));
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      return url.origin === origin || ['data:', 'blob:'].includes(url.protocol) ? route.continue() : route.abort();
    });
    try {
      const page = await context.newPage();
      await page.addInitScript(installWorkerObserver);
      await page.goto(baseURL);
      await page.getByRole('tab', { name: 'Convert audio' }).click();
      await page.locator('#audio-file').setInputFiles({ name: 'original-test-phrase.wav', mimeType: 'audio/wav', buffer: fixtureWav(fixturePcm()) });
      await page.locator('#conversion-bpm').fill('120');
      await expect(page.locator('#convert-button')).toBeEnabled();
      await page.locator('#convert-button').click();
      await expect(page.locator('#conversion-status')).toContainText('interpretation', { timeout: 150000 });
      await expect(page.locator('#error-message')).not.toBeVisible();
      const { library, song } = await selectedTranscription(page);
      expect(library.songs.length).toBeGreaterThan(0);
      const pitches = song.events.filter(event => event.pitch !== null).map(event => event.pitch % 12);
      expect(pitches.length).toBeGreaterThanOrEqual(3);
      expect(pitches.filter(pitch => [0, 4, 7].includes(pitch)).length / pitches.length).toBeGreaterThanOrEqual(0.75);
      expect(requests.some(request => /\/transcription\/basic-pitch\.onnx/.test(request.url))).toBe(true);
      expect(requests.some(request => /\.wasm/.test(request.url))).toBe(true);
      expect(requests.filter(request => request.method !== 'GET' || /\/api\//.test(request.url))).toEqual([]);
      await testInfo.attach('browser-transcription.json', { body: JSON.stringify({ browserName, format: 'wav', candidates: library.songs.length, pitches, audioUploaded: false, model: 'Basic Pitch ONNX', runtime: 'browser WASM' }, null, 2), contentType: 'application/json' });
      await testInfo.attach('studio.png', { body: await page.screenshot({ fullPage: true }), contentType: 'image/png' });
    } finally { await context.close(); await browser.close(); }
  });
}

test('real MP3 conversion, selected variation and repair all run in the browser', async ({ page }, testInfo) => {
  test.setTimeout(300000);
  const files = await generatorFiles();
  for (const [url, bytes] of files) await page.route(url, route => route.fulfill({ body: bytes, contentType: url.includes('tokenizer') ? 'application/json' : 'application/octet-stream' }));
  const requests = [];
  page.on('request', request => requests.push({ method: request.method(), url: request.url() }));
  await page.addInitScript(installWorkerObserver);
  await page.goto('./');
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  const { samples, sampleRate } = fixturePcm();
  const mp3 = await encodeMp3Pcm(samples, sampleRate);
  await page.locator('#audio-file').setInputFiles({ name: 'original-test-phrase.mp3', mimeType: 'audio/mpeg', buffer: Buffer.from(await mp3.arrayBuffer()) });
  await page.locator('#conversion-bpm').fill('120');
  await expect(page.locator('#convert-button')).toBeEnabled();
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('interpretation', { timeout: 150000 });
  const { selected: converted, song: sourceSong } = await selectedTranscription(page);
  expect(sourceSong.events.filter(event => event.pitch !== null).length).toBeGreaterThanOrEqual(3);
  // Pick a short pitched span from actual transcription, not a fabricated score.
  // Keep at least one known event after it so repair must score a real suffix.
  const firstPitch = sourceSong.events.findIndex(event => event.pitch !== null);
  expect(firstPitch).toBeGreaterThanOrEqual(0);
  const end = Math.min(firstPitch + 3, sourceSong.events.length - 1);
  expect(end).toBeGreaterThan(firstPitch);
  const selection = { start: firstPitch, end };
  await page.locator('#selection-start').fill(String(firstPitch + 1));
  await page.locator('#selection-end').fill(String(end));
  await page.locator('#edit-count').selectOption('1');
  await page.locator('#edit-strength').selectOption('medium');
  await page.getByRole('tab', { name: 'Compose', exact: true }).click();
  await page.locator('#composer-form .advanced-settings summary').click();
  await page.locator('#backend').selectOption('wasm');
  await page.locator('#vary-selection').click();
  await expect.poll(() => page.evaluate(() => window.__actualWorkerEvents.filter(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'variation')).length), { timeout: 120000 }).toBe(1);
  const varied = await page.evaluate(() => window.__actualWorkerEvents.findLast(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'variation')).songs[0]);
  const result = parseRtttl(varied.rtttl);
  expect(varied.selection).toEqual(selection);
  expect(varied.backend).toBe('wasm');
  expect(varied.inferenceCalls).toBeGreaterThan(0);
  const variationChanges = expectPitchOnlyEdit(sourceSong, result, selection);
  expect(variationChanges).toBeGreaterThan(0);
  expect(varied.changedEventCount).toBe(variationChanges);
  await expect(page.locator('#score-input')).toHaveValue(varied.rtttl);
  // Repair the same range of the newly selected version using actual suffix scoring.
  await page.locator('#selection-start').fill(String(firstPitch + 1));
  await page.locator('#selection-end').fill(String(end));
  await page.locator('#repair-selection').click();
  await expect.poll(() => page.evaluate(() => window.__actualWorkerEvents.filter(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'repair')).length), { timeout: 120000 }).toBe(1);
  const repaired = await page.evaluate(() => window.__actualWorkerEvents.findLast(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'repair')).songs[0]);
  expect(repaired.repairMethod).toBe('forward-likelihood-reranking');
  expect(repaired.selection).toEqual(selection);
  expect(repaired.backend).toBe('wasm');
  expect(repaired.inferenceCalls).toBeGreaterThan(0);
  const repairSong = parseRtttl(repaired.rtttl);
  const repairChanges = expectPitchOnlyEdit(result, repairSong, selection);
  expect(repaired.changedEventCount).toBe(repairChanges);
  expect(repaired.originalChosen).toBe(repairChanges === 0);
  const scores = repaired.scoreDetails;
  expect(scores.method).toBe('forward-log-likelihood');
  expect(scores.scoreUnit).toBe('natural-log');
  expect(scores.candidateCount).toBeGreaterThanOrEqual(2);
  expect(scores.candidates.length).toBe(scores.candidateCount);
  expect(scores.suffixEventCount).toBe(Math.min(8, result.events.length - end));
  expect(scores.suffixEventCount).toBeGreaterThan(0);
  expect(scores.scoredTokenCount).toBe(2 * (end - firstPitch + scores.suffixEventCount));
  const baseline = scores.candidates.filter(candidate => candidate.original);
  expect(baseline.length).toBe(1);
  expect(baseline[0].changedEventCount).toBe(0);
  expect(scores.originalLogLikelihood).toBeCloseTo(baseline[0].totalLogLikelihood, 6);
  for (const candidate of scores.candidates) {
    expect(Number.isFinite(candidate.spanLogLikelihood)).toBe(true);
    expect(Number.isFinite(candidate.suffixLogLikelihood)).toBe(true);
    expect(Number.isFinite(candidate.totalLogLikelihood)).toBe(true);
    expect(candidate.totalLogLikelihood).toBeCloseTo(candidate.spanLogLikelihood + candidate.suffixLogLikelihood, 6);
    if (!candidate.original) expect(candidate.changedEventCount).toBeGreaterThan(0);
  }
  expect(scores.winningLogLikelihood).toBeCloseTo(Math.max(...scores.candidates.map(candidate => candidate.totalLogLikelihood)), 6);
  expect(scores.scoreDelta).toBeGreaterThanOrEqual(-1e-6);
  expect(scores.scoreDelta).toBeCloseTo(scores.winningLogLikelihood - scores.originalLogLikelihood, 6);
  if (repaired.originalChosen) expect(scores.scoreDelta).toBeCloseTo(0, 6);
  await expect(page.locator('#score-input')).toHaveValue(repaired.rtttl);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')));
  expect(saved.songs.find(song => song.id === converted.id)?.rtttl).toBe(converted.rtttl);
  expect(saved.songs.find(song => song.id === varied.id)?.rtttl).toBe(varied.rtttl);
  await expect(page.locator('#error-message')).not.toBeVisible();
  expect(requests.filter(request => request.method !== 'GET' || /\/api\//.test(request.url))).toEqual([]);
  await testInfo.attach('real-studio-inference.json', { body: JSON.stringify({ format: 'mp3', originalEvents: sourceSong.events.length, selected: { start: firstPitch, end }, changedEvents: varied.changedEventCount, repairMethod: repaired.repairMethod, originalChosen: repaired.originalChosen, variationHash: sha256(varied.rtttl), repairHash: sha256(repaired.rtttl), audioUploaded: false, modelSha256: manifest.modelSha256 }, null, 2), contentType: 'application/json' });
});
