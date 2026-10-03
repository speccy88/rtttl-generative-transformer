import { chromium, expect, firefox, test, webkit } from '@playwright/test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseRtttl } from '../src/rtttl.js';

const enabled = process.env.PLAYWRIGHT_REAL_PORTABLE === '1';
const modelRequest = url => /\/resolve\//.test(new URL(url).pathname);
const runtimeRequest = url => /\/runtime\//.test(new URL(url).pathname);

async function generateShortMelody(page) {
  await expect(page.locator('#name-songs')).not.toBeChecked();
  await page.locator('#song-count').selectOption('1');
  await page.locator('#length').selectOption('24');
  await page.locator('.advanced-settings summary').click();
  await page.locator('#backend').selectOption('wasm');
  await page.locator('#seed').fill('42');
  await page.locator('#keep-seed').check();
  await page.getByRole('button', { name: 'Make some music' }).click();
  await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.', { timeout: 160000 });
  await expect(page.locator('#hardware-status')).toContainText('CPU');
  await expect(page.locator('.song-row')).toHaveCount(1);
  await expect(page.getByRole('alert')).not.toBeVisible();
  const result = await page.evaluate(() => window.__melodyWorkerEvents.findLast(event => event.type === 'done'));
  expect(result.backend).toBe('wasm');
  expect(result.cancelled).toBe(false);
  expect(result.songs).toHaveLength(1);
  expect(result.songs[0].name).toBe('Melody01');
  const song = parseRtttl(result.songs[0].rtttl);
  expect(song.events.length).toBeGreaterThanOrEqual(16);
  expect(song.events.length).toBeLessThanOrEqual(24);
  expect(song.events.some(event => event.pitch !== null)).toBe(true);
  return { backend: result.backend, name: result.songs[0].name, rtttl: result.songs[0].rtttl, events: song.events.length };
}

for (const browserName of ['chromium', 'firefox', 'webkit']) {
  test(`optional portable ${browserName} CPU inference reuses model and runtime after reload`, async ({ baseURL }, testInfo) => {
    test.skip(!enabled, 'Set PLAYWRIGHT_REAL_PORTABLE=1 for real model downloads and local CPU inference in three browsers.');
    test.setTimeout(360000);
    const profile = await mkdtemp(join(tmpdir(), `pocket-portable-${browserName}-`));
    const context = await { chromium, firefox, webkit }[browserName].launchPersistentContext(profile, {
      headless: true,
      viewport: { width: 390, height: 844 },
      ...(browserName === 'webkit' ? { isMobile: true, hasTouch: true } : {}),
    });
    const requests = { cold: [], warm: [] };
    let phase = 'cold';
    const blocked = [];
    const errors = [];
    context.on('request', request => requests[phase].push(request.url()));
    await context.addInitScript(() => {
      window.__melodyWorkerEvents = [];
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', ({ data }) => window.__melodyWorkerEvents.push(data));
        }
      };
    });
    try {
      const page = context.pages()[0] || await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(baseURL);
      const first = await generateShortMelody(page);
      expect(requests.cold.filter(modelRequest).length).toBeGreaterThanOrEqual(2);
      expect(requests.cold.filter(runtimeRequest).some(url => /\.wasm(?:\?|$)/.test(url))).toBe(true);
      expect(requests.cold.filter(runtimeRequest).some(url => /\.mjs(?:\?|$)/.test(url))).toBe(true);

      // Block the original Hub URLs and every runtime asset. A new page and
      // worker must initialize ONNX and compose entirely from persistent bytes.
      phase = 'warm';
      await context.route('**/*', async route => {
        const url = route.request().url();
        if (modelRequest(url) || runtimeRequest(url)) {
          blocked.push(url);
          await route.abort('blockedbyclient');
        } else await route.continue();
      });
      await page.reload();
      const second = await generateShortMelody(page);
      expect(second).toEqual(first);
      expect(blocked).toEqual([]);
      expect(requests.warm.filter(modelRequest)).toEqual([]);
      expect(requests.warm.filter(runtimeRequest)).toEqual([]);
      await expect(page.locator('#cache-notice')).toContainText('Reusing model files saved in this browser.');
      expect(errors).toEqual([]);
      await testInfo.attach('portable-model-cache.json', {
        body: JSON.stringify({
          browserName, first, second,
          cold: { modelRequests: requests.cold.filter(modelRequest), runtimeRequests: requests.cold.filter(runtimeRequest) },
          warm: { modelRequests: requests.warm.filter(modelRequest), runtimeRequests: requests.warm.filter(runtimeRequest), blocked },
          errors,
        }, null, 2),
        contentType: 'application/json',
      });
    } finally {
      await context.close();
      await rm(profile, { recursive: true, force: true });
    }
  });
}
