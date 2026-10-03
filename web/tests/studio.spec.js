import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const EXAMPLE = 'TinyTune:d=8,o=5,b=96:c,e,g,4c6';

// Ordinary UI tests exercise the static player without model or third-party
// downloads. The explicit opt-in test below covers actual model inference.
test.beforeEach(async ({ page, baseURL }, testInfo) => {
  if (testInfo.title.startsWith('optional real')) return;
  const origin = new URL(baseURL).origin;
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin === origin || ['data:', 'blob:'].includes(url.protocol)) return route.continue();
    return route.abort('blockedbyclient');
  });
});

async function openImport(page) {
  const panel = page.locator('.import-panel');
  if (!(await panel.evaluate(element => element.open))) await panel.locator('summary').click();
}

async function importText(page, text = EXAMPLE) {
  await openImport(page);
  await page.locator('#rtttl-input').fill(text);
  await page.getByRole('button', { name: 'Load into player' }).click();
}

test('demo playback moves the note timeline and stops on request', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./');
  await expect(page.getByRole('heading', { name: 'Hello, daydream.' })).toBeVisible();
  await expect(page.locator('#name-songs')).not.toBeChecked();
  await expect(page.locator('.roll-note')).toHaveCount(14);
  await page.getByRole('button', { name: 'Play melody', exact: true }).click();
  await expect(page.locator('#player-card')).toHaveClass(/is-playing/);
  await expect.poll(() => page.locator('#playhead').evaluate(element => parseFloat(element.style.left))).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Stop melody', exact: true }).click();
  await expect(page.locator('#player-card')).not.toHaveClass(/is-playing/);
  await expect(page.locator('#elapsed-time')).toHaveText('0:00');
  expect(errors).toEqual([]);
});

test('imports text and files safely and keeps the last song after invalid input', async ({ page }) => {
  await page.goto('./');
  await importText(page, 'Summer <duet>:d=8,o=5,b=96:c,e,g,4c6');
  await expect(page.locator('#now-playing-title')).toHaveText('Summer <duet>');
  await expect(page.locator('#now-playing-title duet')).toHaveCount(0);
  await expect(page.locator('#now-playing-meta')).toContainText('96 BPM');
  await expect(page.locator('.roll-note')).toHaveCount(4);
  await page.locator('#rtttl-file').setInputFiles({ name: 'tiny.rtttl', mimeType: 'text/plain', buffer: Buffer.from(EXAMPLE) });
  await expect(page.locator('#now-playing-title')).toHaveText('TinyTune');
  await expect(page.locator('.song-row')).toHaveCount(2);
  await importText(page, 'This is not an RTTTL tune');
  await expect(page.getByRole('alert')).toContainText('Use the RTTTL format');
  await expect(page.locator('#now-playing-title')).toHaveText('TinyTune');
});

test('exports real RTTTL text and decodable WAV and MP3 audio', async ({ page }) => {
  await page.goto('./');
  await importText(page);
  for (const format of ['txt', 'wav', 'mp3']) {
    const downloadPromise = page.waitForEvent('download');
    await page.locator(`[data-download="${format}"]`).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe(`TinyTune.${format}`);
    const bytes = await readFile(await download.path());
    if (format === 'txt') {
      expect(bytes.toString()).toBe(`${EXAMPLE}\n`);
      continue;
    }
    expect(bytes.length).toBeGreaterThan(1000);
    if (format === 'wav') {
      expect(bytes.subarray(0, 4).toString()).toBe('RIFF');
      expect(bytes.subarray(8, 12).toString()).toBe('WAVE');
    }
    const decoded = await page.evaluate(async values => {
      const context = new AudioContext();
      try {
        const audio = await context.decodeAudioData(new Uint8Array(values).buffer);
        return { duration: audio.duration, audible: audio.getChannelData(0).some(value => Math.abs(value) > 0.001) };
      } finally { await context.close(); }
    }, [...bytes]);
    // Three eighth notes and one quarter note total 2.5 quarter-note beats.
    const expectedSeconds = 60 / 96 * 2.5;
    expect(decoded.duration).toBeGreaterThanOrEqual(expectedSeconds - 0.002);
    expect(decoded.duration).toBeLessThan(expectedSeconds + 0.15); // MP3 frame padding.
    expect(decoded.audible).toBe(true);
  }
});

test('mobile studio remains inside the viewport with usable controls', async ({ page }) => {
  for (const width of [320, 390, 640, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('./');
    const dimensions = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, document: document.documentElement.scrollWidth }));
    expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport);
    await page.getByRole('button', { name: 'Cinematic', exact: true }).click();
    await expect(page.locator('#profile-description')).toContainText('minor-key');
    await page.locator('#tempo-mode').selectOption('range');
    await expect(page.locator('#bpm-min')).toBeVisible();
    await expect(page.locator('#bpm-max')).toBeVisible();
    await expect(page.locator('#generate-button')).toBeEnabled();
  }
});

test('cancels an in-flight manifest request without losing the playable demo', async ({ page }) => {
  let requested = false;
  await page.route('**/model-manifest.json', async route => {
    requested = true;
    // Keep this request pending until browser cancellation; do not fetch a model.
    await new Promise(resolve => setTimeout(resolve, 2000));
    await route.abort('aborted').catch(() => {});
  });
  await page.goto('./');
  await page.getByRole('button', { name: 'Make some music' }).click();
  await expect.poll(() => requested).toBe(true);
  await page.getByRole('button', { name: 'Stop generation', exact: true }).click();
  await expect(page.locator('#generate-button')).toBeEnabled();
  await expect(page.locator('#generation-message')).toContainText('Stopped.');
  await expect(page.getByRole('alert')).not.toBeVisible();
  await expect(page.locator('#now-playing-title')).toHaveText('Hello, daydream.');
});

test('naming stays unloaded when a streamed melody completes without opt-in', async ({ page }) => {
  const requests = [];
  page.on('request', request => requests.push(request.url()));
  await page.addInitScript(() => {
    // Protocol simulation only: actual model quality and device inference are
    // covered by the optional real-model test and saved hardware validation.
    class MelodyProtocolStub extends EventTarget {
      postMessage(message) {
        if (message.type !== 'generate') return;
        window.__requestedMelodyOptions = message.options;
        const song = { id: `${message.id}-0`, name: 'Melody01', rtttl: 'Melody01:d=8,o=5,b=123:c,e,g,4c6', settings: { profile: 'dance', tonic: 'C', mode: 'major' } };
        setTimeout(() => {
          this.dispatchEvent(new MessageEvent('message', { data: { type: 'song', id: message.id, song } }));
          this.dispatchEvent(new MessageEvent('message', { data: { type: 'done', id: message.id, songs: [song], backend: 'wasm', cancelled: false } }));
        }, 10);
      }
      terminate() {}
    }
    window.Worker = MelodyProtocolStub;
  });
  await page.route('**/model-manifest.json', route => route.fulfill({ json: { modelUrl: 'unused.onnx', tokenizerUrl: 'unused.json' } }));
  await page.goto('./');
  await expect(page.locator('#name-songs')).not.toBeChecked();
  await page.getByRole('button', { name: 'Make some music' }).click();
  await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.');
  await expect(page.locator('#now-playing-title')).toHaveText('Melody 01');
  await expect(page.locator('#now-playing-meta')).toContainText('123 BPM');
  await expect(page.locator('.song-row')).toHaveCount(1);
  const requestedOptions = await page.evaluate(() => window.__requestedMelodyOptions);
  expect(requestedOptions.profile).toBe('mixed');
  expect(requestedOptions.maxPitchRun).toBe(4);
  expect(requests.filter(url => /huggingface|hf\.co|title-worker|\/titles-|transformers|onnx-community|SmolLM/i.test(url))).toEqual([]);
});

test('optional real WebGPU generation uses a hardware adapter', async ({ page }) => {
  test.skip(process.env.PLAYWRIGHT_REAL_MODEL !== '1', 'Set PLAYWRIGHT_REAL_MODEL=1 to allow the melody model download and real local GPU inference.');
  test.setTimeout(180000);
  await page.goto('./');
  const adapter = await page.evaluate(async () => {
    const device = await navigator.gpu?.requestAdapter();
    return device ? { vendor: device.info.vendor, architecture: device.info.architecture, fallback: device.info.isFallbackAdapter } : null;
  });
  expect(adapter, 'A hardware WebGPU adapter is required for this explicit test').not.toBeNull();
  expect(adapter.fallback).toBe(false);
  await page.locator('#song-count').selectOption('1');
  await page.locator('#length').selectOption('24');
  await page.locator('.advanced-settings summary').click();
  await page.locator('#backend').selectOption('webgpu');
  await page.getByRole('button', { name: 'Make some music' }).click();
  await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.', { timeout: 160000 });
  await expect(page.locator('.song-row')).toHaveCount(1);
  await expect(page.locator('#hardware-status')).toContainText('WebGPU');
  await expect(page.getByRole('alert')).not.toBeVisible();
  await test.info().attach('adapter.json', { body: JSON.stringify(adapter, null, 2), contentType: 'application/json' });
});
