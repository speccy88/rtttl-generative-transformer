import { expect, test } from '@playwright/test';
import { parseRtttl } from '../src/rtttl.js';

const HOOK = 'Hook:d=8,o=5,b=120:c,16p,e.,4g';
const IMPORT = 'KeepMe:d=4,o=5,b=96:c,d,e';

function silentWav(seconds = 14) {
  const rate = 8000, frames = rate * seconds;
  const bytes = Buffer.alloc(44 + frames * 2);
  bytes.write('RIFF', 0); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write('WAVE', 8);
  bytes.write('fmt ', 12); bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22); bytes.writeUInt32LE(rate, 24); bytes.writeUInt32LE(rate * 2, 28);
  bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34); bytes.write('data', 36); bytes.writeUInt32LE(frames * 2, 40);
  return bytes;
}

test.beforeEach(async ({ page, baseURL }) => {
  const origin = new URL(baseURL).origin;
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    return url.origin === origin || ['data:', 'blob:'].includes(url.protocol) ? route.continue() : route.abort('blockedbyclient');
  });
  await page.route('**/api/health', route => route.fulfill({ json: { service: 'mp3-to-rtttl', version: 1, ready: true } }));
});

async function importTune(page, text = IMPORT) {
  await page.locator('.import-panel summary').click();
  await page.locator('#rtttl-input').fill(text);
  await page.locator('#load-rtttl').click();
}

async function chooseAudio(page) {
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  await expect(page.locator('#converter-health')).toContainText('connected');
  await page.locator('#audio-file').setInputFiles({ name: 'original-hook.wav', mimeType: 'audio/wav', buffer: silentWav() });
  await expect(page.locator('#convert-button')).toBeEnabled();
}

test('local conversion candidates join the same editable workspace without discarding imports', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  let request;
  await page.route('**/api/transcriptions?*', route => {
    request = { url: route.request().url(), contentType: route.request().headers()['content-type'], length: route.request().postDataBuffer().length };
    return route.fulfill({ status: 202, json: { id: 'conversion-1', status: 'queued' } });
  });
  let polls = 0;
  await page.route('**/api/transcriptions/conversion-1', route => route.fulfill({ json: ++polls === 1
    ? { id: 'conversion-1', status: 'running', progress: 'Following note evidence' }
    : { id: 'conversion-1', status: 'complete', result: {
      candidates: [{ name: 'Detailed hook', rtttl: HOOK, score: 0.86 }, { name: 'Smoother hook', rtttl: 'Smooth:d=8,o=5,b=120:c,e,4g', score: 0.81 }],
      analysis: { estimated_bpm: 120, source_mode: 'full', start: 2, duration: 8 }, warnings: ['Compare the lead by ear.'],
    } } }));
  await page.goto('./');
  await importTune(page);
  await chooseAudio(page);
  await page.locator('#excerpt-start').fill('2');
  await page.locator('#excerpt-duration').fill('8');
  await page.locator('#source-mode').selectOption('full');
  await page.locator('#conversion-bpm').fill('120');
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('2 interpretations added');
  expect(new URL(request.url).searchParams.get('start')).toBe('2');
  expect(new URL(request.url).searchParams.get('duration')).toBe('8');
  expect(new URL(request.url).searchParams.get('source_mode')).toBe('full');
  expect(request.contentType).toBe('application/octet-stream');
  expect(request.length).toBe(silentWav().length);
  await expect(page.locator('.song-row')).toHaveCount(3);
  await expect(page.locator('.song-title').first()).toHaveText('KeepMe');
  await expect(page.locator('#now-playing-title')).toHaveText('Detailed hook');
  await expect(page.locator('#conversion-results')).toContainText('source evidence 86%');
  await expect(page.locator('#conversion-results')).toContainText('not transcription accuracy');
  await page.locator('#score-editor summary').click();
  await expect(page.locator('#score-input')).toHaveValue(HOOK);
  await page.locator('#score-bpm').fill('60');
  await page.locator('#apply-score-tempo').click();
  await expect(page.locator('.song-row')).toHaveCount(4);
  await expect(page.locator('#now-playing-meta')).toContainText('60 BPM');
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')));
  expect(stored.songs[1].rtttl).toBe(HOOK);
  expect(stored.songs[3].parentId).toBe(stored.songs[1].id);
  await page.reload();
  await expect(page.locator('.song-row')).toHaveCount(4);
  await expect(page.locator('#now-playing-meta')).toContainText('60 BPM');
  expect(errors).toEqual([]);
});

test('fractional source duration and excerpt start reach the conversion API', async ({ page }) => {
  let submitted;
  await page.route('**/api/transcriptions?*', route => {
    submitted = new URL(route.request().url()).searchParams;
    return route.fulfill({ status: 202, json: { id: 'fractional-job', status: 'queued' } });
  });
  await page.route('**/api/transcriptions/fractional-job', route => route.fulfill({ json: {
    status: 'complete', result: { candidates: [{ name: 'Fractional excerpt', rtttl: HOOK }], analysis: {}, warnings: [] },
  } }));
  await page.goto('./');
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  const duration = 10 + 107 / 8000;
  await page.locator('#audio-file').setInputFiles({ name: 'short-fractional.wav', mimeType: 'audio/wav', buffer: silentWav(duration) });
  // Metadata fills the exact duration of short recordings. It need not land on
  // the UI's former 0.05-second step; native form validation must allow it.
  await expect.poll(async () => Number(await page.locator('#excerpt-duration').inputValue())).toBeCloseTo(duration, 5);
  await page.locator('#excerpt-start').fill('0.00313');
  await expect(page.locator('#convert-button')).toBeEnabled();
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('1 interpretation added');
  expect(Number(submitted.get('start'))).toBe(0.00313);
  expect(Number(submitted.get('duration'))).toBeCloseTo(duration - 0.00313, 5);
});

test('continuation sends selected notes and retains imported and generated versions', async ({ page }) => {
  // Worker protocol simulation checks UI wiring; sampler tests cover conditioning.
  await page.addInitScript(() => {
    window.__continuationOptions = [];
    window.Worker = class extends EventTarget {
      postMessage(message) {
        if (message.type !== 'generate') return;
        window.__continuationOptions.push(message.options);
        const song = { id: `${message.id}-0`, name: 'NextPhrase', rtttl: 'NextPhrase:d=8,o=5,b=120:c,16p,e.,4g,a,4c6', continuation: true, prefixEventCount: 4 };
        setTimeout(() => {
          this.dispatchEvent(new MessageEvent('message', { data: { type: 'song', id: message.id, song } }));
          this.dispatchEvent(new MessageEvent('message', { data: { type: 'done', id: message.id, songs: [song], backend: 'wasm' } }));
        }, 5);
      }
      terminate() {}
    };
  });
  await page.route('**/model-manifest.json', route => route.fulfill({ json: { modelUrl: 'unused.onnx', tokenizerUrl: 'unused.json' } }));
  await page.goto('./');
  await importTune(page, HOOK);
  await page.locator('#song-count').selectOption('1');
  // Unrelated new-song tempo inputs must not block source-tempo continuation.
  await page.locator('#tempo-mode').selectOption('range');
  await page.locator('#bpm-min').fill('140');
  await page.locator('#bpm-max').fill('90');
  await page.locator('#continue-song').click();
  await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.');
  await expect(page.locator('.song-row')).toHaveCount(2);
  await expect(page.locator('#now-playing-meta')).toContainText('AI continuation');
  const [options] = await page.evaluate(() => window.__continuationOptions);
  expect(options.promptSong.bpm).toBe(120);
  expect(options.promptSong.events).toEqual(parseRtttl(HOOK).events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted })));
  expect(options).not.toHaveProperty('bpm');
  const songs = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')).songs);
  expect(songs[0].rtttl).toBe(HOOK);
  expect(songs[1].parentId).toBe(songs[0].id);
  expect(songs[1].prefixEventCount).toBe(4);
});

test('editing the bundled demo saves its original exactly once', async ({ page }) => {
  await page.goto('./');
  await page.locator('#score-editor summary').click();
  const original = await page.locator('#score-input').inputValue();
  await page.locator('#transpose-up').click();
  await expect(page.locator('.song-row')).toHaveCount(2);
  let songs = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')).songs);
  expect(songs[0].id).toBe('studio-demo');
  expect(songs[0].rtttl).toBe(original);
  expect(songs[1].parentId).toBe('studio-demo');
  await page.locator('[data-song-select="studio-demo"]').click();
  await page.locator('#score-bpm').fill('100');
  await page.locator('#apply-score-tempo').click();
  await expect(page.locator('.song-row')).toHaveCount(3);
  songs = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')).songs);
  expect(songs.filter(song => song.id === 'studio-demo')).toHaveLength(1);
  expect(songs[2].parentId).toBe('studio-demo');
});

test('continuing the bundled demo keeps it as the source in the workspace', async ({ page }) => {
  await page.addInitScript(() => {
    window.Worker = class extends EventTarget {
      postMessage(message) {
        if (message.type !== 'generate') return;
        const original = document.querySelector('#score-input').value;
        const song = { id: `${message.id}-0`, name: 'DemoNext', rtttl: `${original},4c6`, prefixEventCount: message.options.promptSong.events.length };
        setTimeout(() => this.dispatchEvent(new MessageEvent('message', { data: { type: 'done', id: message.id, songs: [song], backend: 'wasm' } })), 5);
      }
      terminate() {}
    };
  });
  await page.route('**/model-manifest.json', route => route.fulfill({ json: { modelUrl: 'unused.onnx', tokenizerUrl: 'unused.json' } }));
  await page.goto('./');
  const original = await page.locator('#score-input').inputValue();
  await page.locator('#continue-song').click();
  await expect(page.locator('.song-row')).toHaveCount(2);
  const songs = await page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')).songs);
  expect(songs[0].id).toBe('studio-demo');
  expect(songs[0].rtttl).toBe(original);
  expect(songs[1].parentId).toBe('studio-demo');
  expect(songs[1].rtttl).toBe(`${original},4c6`);
});

test('unsupported continuation rhythm fails before loading a model', async ({ page }) => {
  let modelRequested = false;
  await page.route('**/model-manifest.json', route => { modelRequested = true; return route.abort(); });
  await page.goto('./');
  await importTune(page, 'Fast:d=64,o=5,b=120:c,d,e');
  await page.locator('#continue-song').click();
  await expect(page.locator('#error-message')).toBeVisible();
  await expect(page.locator('.song-row')).toHaveCount(1);
  expect(modelRequested).toBe(false);
});

test('failed converter inference leaves the shared workspace intact', async ({ page }) => {
  await page.route('**/api/transcriptions?*', route => route.fulfill({ status: 202, json: { id: 'failed-job' } }));
  await page.route('**/api/transcriptions/failed-job', route => route.fulfill({ json: { status: 'error', error: 'Could not decode this recording' } }));
  await page.goto('./'); await importTune(page); await chooseAudio(page);
  await page.locator('#convert-button').click();
  await expect(page.locator('#error-message')).toContainText('Could not decode this recording');
  await expect(page.locator('#now-playing-title')).toHaveText('KeepMe');
  await expect(page.locator('.song-row')).toHaveCount(1);
  await expect(page.locator('#convert-button')).toBeEnabled();
});

test('static hosted studio explains the local converter and keeps compose available', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ status: 404, body: 'Not found' }));
  await page.goto('./');
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  await expect(page.locator('#converter-health')).toContainText('local studio');
  await page.locator('#converter-help summary').click();
  await expect(page.locator('#converter-help')).toContainText('python rtttl_studio.py');
  await expect(page.locator('#convert-button')).toBeDisabled();
  await page.getByRole('tab', { name: 'Compose', exact: true }).click();
  await expect(page.locator('#generate-button')).toBeEnabled();
});

test('conversion and editing controls stay inside a narrow phone viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto('./'); await chooseAudio(page);
  await page.locator('#score-editor summary').click();
  const dimensions = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, width: document.documentElement.scrollWidth }));
  expect(dimensions.width).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.locator('#convert-button')).toBeEnabled();
  await expect(page.locator('#save-score')).toBeVisible();
});
