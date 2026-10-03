import { chromium, expect, firefox, test, webkit } from '@playwright/test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const enabled = process.env.PLAYWRIGHT_REAL_MOBILE_TITLES === '1';
const modelResource = url => /\/resolve\/|\/runtime\//.test(new URL(url).pathname);
const musicalBytes = song => song.rtttl.slice(song.rtttl.indexOf(':'));

for (const browserName of ['chromium', 'firefox', 'webkit']) {
  test(`optional mobile ${browserName} retains four named melodies and reuses both models after reload`, async ({ baseURL }, testInfo) => {
    test.skip(!enabled, 'Enable real local title inference and model downloads with PLAYWRIGHT_REAL_MOBILE_TITLES=1.');
    test.setTimeout(480000);
    const profile = await mkdtemp(join(tmpdir(), `pocket-naming-${browserName}-`));
    const context = await { chromium, firefox, webkit }[browserName].launchPersistentContext(profile, {
      headless: true, viewport: { width: 390, height: 844 },
      ...(browserName === 'webkit' ? { isMobile: true, hasTouch: true } : {}),
    });
    const requests = { cold: [], warm: [] }, blocked = [], errors = [], titleEvents = [];
    let phase = 'cold';
    context.on('request', request => { if (modelResource(request.url())) requests[phase].push(request.url()); });
    await context.exposeBinding('reportTitleEvent', (_source, event) => titleEvents.push(event));
    await context.addInitScript(() => {
      window.__namingTestMessages = [];
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(url, options) {
          super(url, options);
          this.addEventListener('message', ({ data }) => {
            window.__namingTestMessages.push({ url: String(url), data });
            if (String(url).includes('title-worker') && (data.backend || data.type === 'done' || data.type === 'error')) {
              window.reportTitleEvent({ type: data.type, backend: data.backend, message: data.message, disposed: data.disposed });
            }
          });
        }
      };
    });
    try {
      const page = context.pages()[0] || await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('crash', () => errors.push('Page crashed'));
      const compose = async seed => {
        await page.locator('#song-count').selectOption('4');
        await page.locator('#length').selectOption('24');
        if (!(await page.locator('#name-songs').isChecked())) await page.locator('.naming-option').click();
        await expect(page.locator('#name-songs')).toBeChecked();
        const advanced = page.locator('.advanced-settings');
        if (!(await advanced.evaluate(element => element.open))) await advanced.locator('summary').click();
        await page.locator('#backend').selectOption(browserName === 'webkit' ? 'auto' : 'wasm');
        if (browserName === 'webkit') await expect(page.locator('#backend option[value="webgpu"]')).toHaveJSProperty('disabled', true);
        await page.locator('#seed').fill(String(seed));
        await page.locator('#keep-seed').check();
        await page.getByRole('button', { name: 'Make some music' }).click();
        await expect(page.locator('#generation-message')).toContainText('4 melodies, made on your device.', { timeout: 240000 });
        await expect(page.locator('.song-row')).toHaveCount(4);
        await expect(page.getByRole('alert')).not.toBeVisible();
        const batches = await page.evaluate(() => ({
          melodies: window.__namingTestMessages.findLast(event => event.url.includes('melody-worker') && event.data.type === 'done')?.data.songs,
          named: window.__namingTestMessages.findLast(event => event.url.includes('title-worker') && event.data.type === 'done')?.data.songs,
          titleBackend: window.__namingTestMessages.findLast(event => event.url.includes('title-worker') && event.data.type === 'done')?.data.backend,
        }));
        expect(batches.melodies).toHaveLength(4);
        expect(batches.named).toHaveLength(4);
        expect(batches.named.filter(song => song.naming?.status === 'named').length).toBeGreaterThan(0);
        expect(batches.named.map(song => song.id)).toEqual(batches.melodies.map(song => song.id));
        expect(batches.named.map(musicalBytes)).toEqual(batches.melodies.map(musicalBytes));
        const titles = await page.locator('.song-title').allTextContents();
        expect(new Set(titles.map(title => title.toLowerCase())).size).toBe(4);
        expect(titles).not.toContain('Hello, daydream.');
        expect(titles).toEqual(batches.named.map(song => song.title));
        return { titles, named: batches.named, melodies: batches.melodies, titleBackend: batches.titleBackend };
      };
      await page.goto(baseURL);
      const first = await compose(452);
      expect(requests.cold.some(url => /\/onnx\/.*\.onnx/.test(url))).toBe(true);
      phase = 'warm';
      await context.route('**/*', route => {
        if (modelResource(route.request().url())) {
          blocked.push(route.request().url());
          return route.abort('internetdisconnected');
        }
        return route.continue();
      });
      await page.reload();
      await expect(page.locator('.song-row')).toHaveCount(4);
      expect(await page.locator('.song-title').allTextContents()).toEqual(first.titles);
      const second = await compose(897);
      expect(new Set([...first.titles, ...second.titles].map(title => title.toLowerCase())).size).toBe(8);
      expect(requests.warm).toEqual([]);
      expect(blocked).toEqual([]);
      expect(errors).toEqual([]);
      await testInfo.attach('mobile-title-recovery.json', { body: JSON.stringify({ browserName, first, second, requests, blocked, errors }, null, 2), contentType: 'application/json' });
    } finally {
      await testInfo.attach('mobile-title-diagnostics.json', { body: JSON.stringify({ browserName, requests, blocked, errors, titleEvents }, null, 2), contentType: 'application/json' });
      await context.close();
      await rm(profile, { recursive: true, force: true });
    }
  });
}
