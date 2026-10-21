import { expect, test } from '@playwright/test';
import { parseRtttl } from '../src/rtttl.js';

const HOOK = 'Hook:d=8,o=5,b=120:c,16p,e.,4g';
const IMPORT = 'KeepMe:d=4,o=5,b=96:c,d,e';
const VARIATION = 'Variation:d=8,o=5,b=120:c,16p,f.,4g';
const events = text => parseRtttl(text).events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted }));

function silentWav(seconds = 14) {
  const rate = 8000, frames = Math.round(rate * seconds);
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
});

// Protocol fixtures isolate UI behavior. browser-inference.spec.js separately
// verifies real decoding, Basic Pitch inference and generative model sampling.
async function mockWorkers(page, { error = null, delay = 5, generationDelay = 5 } = {}) {
  await page.addInitScript(({ hook, variation, error, delay, generationDelay }) => {
    window.__transcriptionRequests = [];
    window.__generationOptions = [];
    window.__terminatedTranscribers = 0;
    window.Worker = class extends EventTarget {
      constructor(url) { super(); this.url = String(url); this.stopped = false; }
      emit(data) {
        if (this.stopped) return;
        const message = new MessageEvent('message', { data });
        this.onmessage?.(message); this.dispatchEvent(message);
      }
      postMessage(message) {
        if (message.type === 'cancel') {
          clearTimeout(this.generationTimer);
          this.emit({ type: 'done', id: message.id, songs: [], cancelled: true, backend: 'wasm' });
          return;
        }
        if (message.type === 'transcribe') {
          this.transcriber = true;
          window.__transcriptionRequests.push({ options: message.options, samples: message.samples.length, name: message.name });
          this.emit({ type: 'progress', message: 'Following notes in this browser…' });
          setTimeout(() => this.emit(error ? { type: 'error', message: error } : { type: 'result', result: {
            candidates: [{ name: 'Detailed hook', rtttl: hook, score: 0.86 }, { name: 'Smoother hook', rtttl: 'Smooth:d=8,o=5,b=120:c,e,4g', score: 0.81 }],
            analysis: { estimated_bpm: 120, source_mode: 'full', start: message.options.start, duration: message.options.duration }, warnings: ['Compare the lead by ear.'],
          } }), delay);
          return;
        }
        if (message.type !== 'generate') return;
        window.__generationOptions.push(message.options);
        const mode = message.options.editMode;
        const source = document.querySelector('#score-input').value;
        const songs = Array.from({ length: message.options.numSongs || 1 }, (_, index) => ({
          id: `${message.id}-${index}`, name: mode === 'variation' ? 'Variation' : mode === 'repair' ? 'Repair' : 'NextPhrase',
          rtttl: mode === 'variation' ? variation : mode === 'repair' ? source : `${source},4c6`,
          editMode: mode, selection: message.options.selection,
          continuation: !mode, prefixEventCount: mode ? 0 : message.options.promptSong?.events.length,
          changedEventCount: mode === 'variation' ? 1 : 0, originalChosen: mode === 'repair',
          repairMethod: mode === 'repair' ? 'forward-likelihood-reranking' : null,
        }));
        this.generationTimer = setTimeout(() => {
          for (const song of songs) this.emit({ type: 'song', id: message.id, song });
          this.emit({ type: 'done', id: message.id, songs, backend: 'wasm' });
        }, generationDelay);
      }
      terminate() { this.stopped = true; if (this.transcriber) window.__terminatedTranscribers++; }
    };
  }, { hook: HOOK, variation: VARIATION, error, delay, generationDelay });
  await page.route('**/model-manifest.json', route => route.fulfill({ json: { modelUrl: 'unused.onnx', tokenizerUrl: 'unused.json' } }));
}

async function importTune(page, text = IMPORT) {
  if ((await page.locator('.import-panel').getAttribute('open')) === null) await page.locator('.import-panel summary').click();
  await page.locator('#rtttl-input').fill(text);
  await page.locator('#load-rtttl').click();
}

async function chooseAudio(page, seconds = 14) {
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  await page.locator('#audio-file').setInputFiles({ name: 'original-hook.wav', mimeType: 'audio/wav', buffer: silentWav(seconds) });
  await expect(page.locator('#convert-button')).toBeEnabled();
  await expect(page.locator('#source-waveform')).toBeVisible();
}

const savedSongs = page => page.evaluate(() => JSON.parse(localStorage.getItem('pocket-composer-library-v1')).songs);

test('browser conversion candidates join the editable workspace without an API or audio upload', async ({ page }) => {
  await mockWorkers(page);
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push({ url: request.url(), method: request.method() }));
  await page.goto('./');
  await importTune(page);
  await chooseAudio(page);
  await page.locator('#excerpt-start').fill('2');
  await page.locator('#excerpt-duration').fill('8');
  await page.locator('#source-mode').selectOption('full');
  await page.locator('#conversion-bpm').fill('120');
  await page.locator('.conversion-advanced summary').click();
  await page.locator('#lead-preference').selectOption('upper');
  await page.locator('#conversion-grid').selectOption('32');
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('2 interpretations added');
  const [request] = await page.evaluate(() => window.__transcriptionRequests);
  expect(request.options).toMatchObject({ start: 2, duration: 8, source_mode: 'full', bpm: 120, leadPreference: 'upper', grid: 32 });
  expect(request.samples).toBeGreaterThan(0);
  expect(requests.filter(request => request.method !== 'GET' || /\/api\//.test(request.url))).toEqual([]);
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
  const songs = await savedSongs(page);
  expect(songs[1].rtttl).toBe(HOOK);
  expect(songs[3].parentId).toBe(songs[1].id);
  await page.reload();
  await expect(page.locator('.song-row')).toHaveCount(4);
  await expect(page.locator('#now-playing-meta')).toContainText('60 BPM');
  expect(errors).toEqual([]);
});

test('fractional source duration and excerpt start reach the browser transcription worker', async ({ page }) => {
  await mockWorkers(page);
  await page.goto('./');
  const duration = 10 + 107 / 8000;
  await chooseAudio(page, duration);
  const available = Number(await page.locator('#excerpt-duration').inputValue());
  expect(available).toBeCloseTo(duration, 4);
  await page.locator('#excerpt-start').fill('0.00313');
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('interpretations added');
  const [{ options }] = await page.evaluate(() => window.__transcriptionRequests);
  expect(options.start).toBe(0.00313);
  expect(options.duration).toBeCloseTo(available - 0.00313, 4);
  await page.locator('#excerpt-start-handle').focus();
  await page.keyboard.press('End');
  expect(Number(await page.locator('#excerpt-duration').inputValue())).toBeGreaterThanOrEqual(0.25);
  await page.locator('#convert-button').click();
  await expect.poll(() => page.evaluate(() => window.__transcriptionRequests.length)).toBe(2);
  expect(await page.evaluate(() => window.__transcriptionRequests[1].options.duration)).toBeGreaterThanOrEqual(0.25);
});

test('waveform drag, edge keys and numeric fields share one excerpt', async ({ page }) => {
  await page.goto('./');
  await chooseAudio(page, 60);
  const bounds = await page.locator('#source-waveform').boundingBox();
  await page.mouse.move(bounds.x + bounds.width * 0.5, bounds.y + 20);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.7, bounds.y + 20, { steps: 5 });
  await page.mouse.up();
  expect(Number(await page.locator('#excerpt-start').inputValue())).toBeCloseTo(30, 0);
  expect(Number(await page.locator('#excerpt-duration').inputValue())).toBeCloseTo(12, 0);
  const before = Number(await page.locator('#excerpt-duration').inputValue());
  await page.locator('#excerpt-end-handle').focus();
  await page.keyboard.press('ArrowLeft');
  expect(Number(await page.locator('#excerpt-duration').inputValue())).toBeCloseTo(before - 0.1, 3);
  await page.locator('#excerpt-start').fill('5.125');
  await page.locator('#excerpt-duration').fill('6.5');
  await expect(page.locator('#excerpt-start-handle')).toHaveAttribute('aria-valuenow', '5.125');
  await expect(page.locator('#excerpt-end-handle')).toHaveAttribute('aria-valuenow', '11.625');
});

test('piano-roll range selection edits only selected pitches and survives comparison and reload', async ({ page }) => {
  await page.goto('./'); await importTune(page, HOOK);
  await page.locator('#roll-notes [data-note="0"]').click();
  await page.locator('#roll-notes [data-note="2"]').click({ modifiers: ['Shift'] });
  await expect(page.locator('#selection-start')).toHaveValue('1');
  await expect(page.locator('#selection-end')).toHaveValue('3');
  await expect(page.locator('#roll-notes [aria-pressed="true"]')).toHaveCount(3);
  await page.locator('#selected-pitch-up').click();
  await expect(page.locator('.song-row')).toHaveCount(2);
  const songs = await savedSongs(page);
  expect(songs[0].rtttl).toBe(HOOK);
  expect(events(songs[1].rtttl)).toEqual(events(HOOK).map((event, index) => index < 3 && event.pitch !== null ? { ...event, pitch: event.pitch + 1 } : event));
  expect(songs[1].selectedRange).toEqual({ start: 0, end: 3 });
  await page.locator('#compare-parent').click();
  await expect(page.locator('#score-input')).toHaveValue(HOOK);
  await page.locator('#compare-version').click();
  await expect(page.locator('#score-input')).toHaveValue(songs[1].rtttl);
  await page.reload();
  await expect(page.locator('#selection-end')).toHaveValue('3');
  await expect(page.locator('#roll-notes [aria-pressed="true"]')).toHaveCount(3);
  await page.locator('#roll-notes [data-note="0"]').focus();
  await page.keyboard.press('Shift+ArrowRight');
  await expect(page.locator('#selection-start')).toHaveValue('1');
  await expect(page.locator('#selection-end')).toHaveValue('2');
});

test('selected rest and duration controls save copies with outside events intact', async ({ page }) => {
  await page.goto('./'); await importTune(page, HOOK);
  await page.locator('#roll-notes [data-note="2"]').click();
  await page.locator('#selected-rest').click();
  await page.locator('#selected-duration').selectOption('4');
  await page.locator('#selected-dotted').check();
  await page.locator('#apply-selected-duration').click();
  await page.locator('#selected-note').click();
  const songs = await savedSongs(page);
  expect(songs).toHaveLength(4);
  expect(events(songs[1].rtttl)[2].pitch).toBeNull();
  expect(events(songs[3].rtttl)[2]).toEqual({ pitch: 72, duration: 4, dotted: true });
  for (const index of [0, 1, 3]) expect(events(songs[3].rtttl)[index]).toEqual(events(HOOK)[index]);
  expect(songs.map(song => song.parentId).slice(1)).toEqual(songs.slice(0, -1).map(song => song.id));
});

test('variation and repair forward the selected region and retain every source version', async ({ page }) => {
  await mockWorkers(page);
  await page.goto('./'); await importTune(page, HOOK);
  await page.locator('#selection-start').fill('3');
  await page.locator('#selection-end').fill('3');
  await page.locator('#edit-strength').selectOption('low');
  await page.locator('#edit-count').selectOption('1');
  await page.locator('#tempo-mode').selectOption('range');
  await page.locator('#bpm-min').fill('140'); await page.locator('#bpm-max').fill('90');
  await page.locator('#vary-selection').click();
  await expect(page.locator('.song-row')).toHaveCount(2);
  await expect(page.locator('#now-playing-meta')).toContainText('AI variation');
  await expect(page.locator('#vary-selection')).toBeEnabled();
  await expect(page.locator('#selection-start')).toHaveValue('3');
  await page.locator('#repair-selection').click();
  await expect(page.locator('.song-row')).toHaveCount(3);
  await expect(page.locator('#edit-result-note')).toContainText('Original scored best; no change suggested');
  const options = await page.evaluate(() => window.__generationOptions);
  expect(options[0]).toMatchObject({ editMode: 'variation', selection: { start: 2, end: 3 }, editStrength: 'low', numSongs: 1 });
  expect(options[0].promptSong.events).toEqual(events(HOOK));
  expect(options[0]).not.toHaveProperty('bpmRange');
  expect(options[1]).toMatchObject({ editMode: 'repair', selection: { start: 2, end: 3 } });
  expect(options[1].promptSong.events).toEqual(events(VARIATION));
  const songs = await savedSongs(page);
  expect(songs[0].rtttl).toBe(HOOK); expect(songs[1].rtttl).toBe(VARIATION);
  expect(songs[2].parentId).toBe(songs[1].id);
  await expect(page.locator('.repair-explanation')).toContainText('not a BERT-trained repair model');
});

test('phrase generation started in Convert exposes progress and cancellation in Compose', async ({ page }) => {
  await mockWorkers(page, { generationDelay: 60000 });
  await page.goto('./'); await importTune(page, HOOK);
  await page.locator('#roll-notes [data-note="2"]').click();
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  await page.locator('#vary-selection').click();
  await expect(page.locator('#compose-tab')).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#generation-status')).toBeVisible();
  await expect(page.locator('#cancel-button')).toBeVisible();
  await expect(page.locator('#convert-tab')).toBeDisabled();
  await page.locator('#cancel-button').click();
  await expect(page.locator('#generation-message')).toContainText('Stopped.');
  await expect(page.locator('.song-row')).toHaveCount(1);
  await expect(page.locator('#convert-tab')).toBeEnabled();
});

test('long or rest-only selections offer manual editing without starting phrase inference', async ({ page }) => {
  await page.goto('./');
  await importTune(page, `Long:d=8,o=5,b=120:${Array(17).fill('c').join(',')}`);
  await page.locator('#select-all-notes').click();
  await expect(page.locator('#vary-selection')).toBeDisabled();
  await expect(page.locator('#repair-selection')).toBeDisabled();
  await expect(page.locator('#phrase-hint')).toContainText('at most 16');
  await expect(page.locator('#selected-pitch-up')).toBeEnabled();
  await importTune(page, HOOK);
  await page.locator('#roll-notes [data-note="1"]').click();
  await expect(page.locator('#vary-selection')).toBeDisabled();
  await expect(page.locator('#selected-note')).toBeEnabled();
});

test('continuation sends the complete selected tune and retains original versions', async ({ page }) => {
  await mockWorkers(page);
  await page.goto('./'); await importTune(page, HOOK);
  await page.locator('#song-count').selectOption('1');
  await page.locator('#tempo-mode').selectOption('range');
  await page.locator('#bpm-min').fill('140'); await page.locator('#bpm-max').fill('90');
  await page.locator('#continue-song').click();
  await expect(page.locator('#generation-message')).toContainText('1 melody, made on your device.');
  await expect(page.locator('.song-row')).toHaveCount(2);
  await expect(page.locator('#now-playing-meta')).toContainText('AI continuation');
  const [options] = await page.evaluate(() => window.__generationOptions);
  expect(options.promptSong.bpm).toBe(120);
  expect(options.promptSong.events).toEqual(events(HOOK));
  expect(options).not.toHaveProperty('bpm');
  const songs = await savedSongs(page);
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
  let songs = await savedSongs(page);
  expect(songs[0].id).toBe('studio-demo'); expect(songs[0].rtttl).toBe(original);
  expect(songs[1].parentId).toBe('studio-demo');
  await page.locator('[data-song-select="studio-demo"]').click();
  await page.locator('#score-bpm').fill('100');
  await page.locator('#apply-score-tempo').click();
  await expect(page.locator('.song-row')).toHaveCount(3);
  songs = await savedSongs(page);
  expect(songs.filter(song => song.id === 'studio-demo')).toHaveLength(1);
  expect(songs[2].parentId).toBe('studio-demo');
});

test('continuing the bundled demo keeps its source in the workspace', async ({ page }) => {
  await mockWorkers(page);
  await page.goto('./');
  const original = await page.locator('#score-input').inputValue();
  await page.locator('#song-count').selectOption('1');
  await page.locator('#continue-song').click();
  await expect(page.locator('.song-row')).toHaveCount(2);
  const songs = await savedSongs(page);
  expect(songs[0].id).toBe('studio-demo'); expect(songs[0].rtttl).toBe(original);
  expect(songs[1].parentId).toBe('studio-demo'); expect(songs[1].rtttl).toBe(`${original},4c6`);
});

test('unsupported continuation rhythm fails before loading a model', async ({ page }) => {
  let modelRequested = false;
  await page.route('**/model-manifest.json', route => { modelRequested = true; return route.abort(); });
  await page.goto('./'); await importTune(page, 'Fast:d=64,o=5,b=120:c,d,e');
  await page.locator('#continue-song').click();
  await expect(page.locator('#error-message')).toBeVisible();
  await expect(page.locator('.song-row')).toHaveCount(1);
  expect(modelRequested).toBe(false);
});

test('failed browser transcription leaves the shared workspace intact', async ({ page }) => {
  await mockWorkers(page, { error: 'Could not detect notes in this recording' });
  await page.goto('./'); await importTune(page); await chooseAudio(page);
  await page.locator('#convert-button').click();
  await expect(page.locator('#error-message')).toContainText('Could not detect notes in this recording');
  await expect(page.locator('#now-playing-title')).toHaveText('KeepMe');
  await expect(page.locator('.song-row')).toHaveCount(1);
  await expect(page.locator('#convert-button')).toBeEnabled();
});

test('cancelling conversion terminates its worker and keeps existing melodies', async ({ page }) => {
  await mockWorkers(page, { delay: 60000 });
  await page.goto('./'); await importTune(page); await chooseAudio(page);
  await page.locator('#convert-button').click();
  await expect(page.locator('#conversion-status')).toContainText('Following notes');
  await page.locator('#cancel-conversion').click();
  await expect(page.locator('#conversion-status')).toContainText('processing stopped');
  await expect(page.locator('#convert-button')).toBeEnabled();
  await expect(page.locator('.song-row')).toHaveCount(1);
  expect(await page.evaluate(() => window.__terminatedTranscribers)).toBeGreaterThan(0);
});

test('the hosted studio explains browser codec support and has no setup dependency', async ({ page }) => {
  await page.goto('./');
  await page.getByRole('tab', { name: 'Convert audio' }).click();
  await expect(page.locator('.browser-processing')).toContainText('Browser processing');
  await expect(page.locator('#converter-form')).not.toContainText('server');
  await expect(page.locator('#converter-form')).toContainText('try WAV or MP3');
  await expect(page.locator('#convert-button')).toBeDisabled();
  await page.getByRole('tab', { name: 'Compose', exact: true }).click();
  await expect(page.locator('#generate-button')).toBeEnabled();
});

test('conversion and phrase controls stay inside a narrow phone viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto('./'); await chooseAudio(page);
  await page.locator('#score-editor summary').click();
  const dimensions = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, width: document.documentElement.scrollWidth }));
  expect(dimensions.width).toBeLessThanOrEqual(dimensions.viewport);
  await expect(page.locator('#convert-button')).toBeEnabled();
  await expect(page.locator('#save-score')).toBeVisible();
  await expect(page.locator('#selection-start')).toBeVisible();
  await expect(page.locator('#vary-selection')).toBeVisible();
});
