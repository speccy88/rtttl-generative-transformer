import { chromium, expect, test } from '@playwright/test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Use normal website-data storage; private contexts can impose extra limits
// which do not represent a normal Safari visit.
const normalProfileTest = test.extend({
  context: async ({ launchOptions, baseURL }, use) => {
    const profile = await mkdtemp(join(tmpdir(), 'pocket-title-cache-'));
    const context = await chromium.launchPersistentContext(profile, { ...launchOptions, baseURL });
    try { await use(context); }
    finally { await context.close(); await rm(profile, { recursive: true, force: true }); }
  },
});

// Explicit opt-in: downloads the real melody model/runtime on the test device.
test('optional real model files are reused after repeated generation and reload', async ({ page }) => {
  test.skip(process.env.PLAYWRIGHT_REAL_MODEL !== '1', 'Enable real local inference and the first model download explicitly.');
  test.setTimeout(180000);
  const downloads = [];
  page.on('request', request => {
    if (/\/resolve\/.*(?:model\.onnx|tokenizer\.json)|\/runtime\/.*asyncify\.(?:wasm|mjs)/.test(request.url())) downloads.push(request.url());
  });
  async function compose() {
    await page.selectOption('#song-count', '1');
    await page.selectOption('#length', '24');
    const panel = page.locator('.advanced-settings');
    if (!(await panel.evaluate(element => element.open))) await panel.locator('summary').click();
    await page.selectOption('#backend', 'wasm');
    await page.getByRole('button', { name: 'Make some music' }).click();
    await expect(page.locator('#generate-button')).toBeEnabled({ timeout: 60000 });
    await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.');
    await expect(page.getByRole('alert')).not.toBeVisible();
  }
  await page.goto('./');
  await compose();
  expect(downloads.filter(url => url.includes('model.onnx'))).toHaveLength(1);
  const firstCount = downloads.length;
  await compose();
  expect(downloads).toHaveLength(firstCount);
  await page.reload();
  // The app must work even if all model/runtime network requests now fail.
  await page.route('**/*', route => /\/resolve\/|\/runtime\//.test(route.request().url()) ? route.abort('internetdisconnected') : route.continue());
  await compose();
  expect(downloads).toHaveLength(firstCount);
  await expect(page.locator('#cache-notice')).toContainText('Reusing model files saved in this browser.');
  await test.info().attach('model-network-requests.json', { body: JSON.stringify(downloads, null, 2), contentType: 'application/json' });
});

normalProfileTest('optional real title model is reused after a page reload without model networking', async ({ page }) => {
  test.skip(process.env.PLAYWRIGHT_REAL_TITLES !== '1', 'Enable the optional title model download explicitly.');
  test.setTimeout(360000);
  const downloads = [], messages = [];
  page.on('request', request => { if (/\/resolve\/|\/runtime\//.test(request.url())) downloads.push(request.url()); });
  await page.exposeFunction('cacheTestMessage', message => messages.push(message));
  await page.addInitScript(() => {
    new MutationObserver(() => {
      const status = document.querySelector('#generation-message');
      if (status) window.cacheTestMessage(status.textContent);
    }).observe(document, { childList: true, subtree: true, characterData: true });
  });
  async function composeNamed() {
    await page.selectOption('#song-count', '1'); await page.selectOption('#length', '24');
    if (!(await page.locator('#name-songs').isChecked())) await page.locator('.naming-option').click();
    await page.getByRole('button', { name: 'Make some music' }).click();
    await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.', { timeout: 300000 });
    await expect(page.locator('.song-row')).toHaveCount(1);
    await expect(page.getByRole('alert')).not.toBeVisible();
    await expect(page.locator('#now-playing-title')).not.toHaveText('Melody 01');
  }
  await page.goto('./'); await composeNamed();
  const storage = await page.evaluate(async () => ({
    estimate: await navigator.storage?.estimate(),
    notice: document.querySelector('#cache-notice')?.textContent,
    caches: await Promise.all((await caches.keys()).map(async name => {
      const cache = await caches.open(name);
      return { name, files: await Promise.all((await cache.keys()).map(async key => ({ url: key.url, bytes: (await cache.match(key)).headers.get('content-length') }))) };
    })),
    manifests: await new Promise((resolve, reject) => {
      const request = indexedDB.open('pocket-composer-models-v1', 2);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const records = db.transaction('manifests').objectStore('manifests').getAll();
        records.onsuccess = () => { const files = records.result.map(record => ({ size: record.size, chunks: record.sizes.length })); db.close(); resolve(files); };
        records.onerror = () => { db.close(); reject(records.error); };
      };
    }),
  }));
  await test.info().attach('title-storage.json', { body: JSON.stringify(storage, null, 2), contentType: 'application/json' });
  expect(storage.manifests.some(file => file.size > 100000000 && file.chunks > 1)).toBe(true);
  expect(downloads.some(url => /\/onnx\/.*\.onnx/.test(url))).toBe(true);
  const firstRequests = [...downloads]; const firstCount = downloads.length;
  await page.reload(); messages.length = 0;
  await page.route('**/*', route => /\/resolve\/|\/runtime\//.test(route.request().url()) ? route.abort('internetdisconnected') : route.continue());
  await composeNamed();
  // Uncached optional metadata may be probed, but successful model/runtime files
  // must all come from storage; any blocked weight request would fail naming.
  expect(downloads.slice(firstCount).filter(url => /\.(?:onnx|wasm|mjs|json)(?:\?|$)/.test(url))).toEqual([]);
  expect(messages.some(message => message.includes('title model from browser storage'))).toBe(true);
  expect(messages.some(message => message.includes('Downloading the optional title model'))).toBe(false);
  await test.info().attach('title-cache-requests.json', { body: JSON.stringify({ firstRequests, afterReload: downloads.slice(firstCount), messages }, null, 2), contentType: 'application/json' });
});
