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
      const library = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')));
      expect(library.songs.length).toBeGreaterThan(0);
      const pitches = parseRtttl(library.songs[0].rtttl).events.filter(event => event.pitch !== null).map(event => event.pitch % 12);
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
  const source = await page.locator('#score-input').inputValue();
  const sourceSong = parseRtttl(source);
  // Pick a short pitched span from actual transcription, not a fabricated score.
  const firstPitch = sourceSong.events.findIndex(event => event.pitch !== null);
  const end = Math.min(firstPitch + 3, sourceSong.events.length);
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
  expect(varied.changedEventCount).toBeGreaterThan(0);
  expect(result.bpm).toBe(sourceSong.bpm);
  expect(result.events.length).toBe(sourceSong.events.length);
  for (let index = 0; index < result.events.length; index++) {
    expect(result.events[index].duration).toBe(sourceSong.events[index].duration);
    expect(result.events[index].dotted).toBe(sourceSong.events[index].dotted);
    if (index < firstPitch || index >= end || sourceSong.events[index].pitch === null) expect(result.events[index]).toEqual(sourceSong.events[index]);
  }
  // Repair the same range of the newly selected version using actual suffix scoring.
  await page.locator('#selection-start').fill(String(firstPitch + 1));
  await page.locator('#selection-end').fill(String(end));
  await page.locator('#repair-selection').click();
  await expect.poll(() => page.evaluate(() => window.__actualWorkerEvents.filter(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'repair')).length), { timeout: 120000 }).toBe(1);
  const repaired = await page.evaluate(() => window.__actualWorkerEvents.findLast(event => event.type === 'done' && event.songs?.some(song => song.editMode === 'repair')).songs[0]);
  expect(repaired.repairMethod).toBeTruthy();
  const repairSong = parseRtttl(repaired.rtttl);
  expect(repairSong.bpm).toBe(result.bpm);
  for (let index = 0; index < repairSong.events.length; index++) {
    if (index < firstPitch || index >= end) expect(repairSong.events[index]).toEqual(result.events[index]);
  }
  await expect(page.locator('#error-message')).not.toBeVisible();
  expect(requests.filter(request => request.method !== 'GET' || /\/api\//.test(request.url))).toEqual([]);
  await testInfo.attach('real-studio-inference.json', { body: JSON.stringify({ format: 'mp3', originalEvents: sourceSong.events.length, selected: { start: firstPitch, end }, changedEvents: varied.changedEventCount, repairMethod: repaired.repairMethod, originalChosen: repaired.originalChosen, variationHash: sha256(varied.rtttl), repairHash: sha256(repaired.rtttl), audioUploaded: false, modelSha256: manifest.modelSha256 }, null, 2), contentType: 'application/json' });
});
