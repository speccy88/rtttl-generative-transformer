import test from 'node:test';
import assert from 'node:assert/strict';
import { checkConverter, MAX_UPLOAD_BYTES, transcriptionParameters, transcribeAudio } from './converter-client.js';

const file = { name: 'hook & lead.mp3', size: 4200 };
const json = (data, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });

test('converter request bounds excerpts and uploads without guessing a remote service', () => {
  const params = transcriptionParameters(file, { start: 10.5, duration: 8, bpm: 125, source_mode: 'full', preset: 'faithful' });
  assert.equal(params.get('filename'), file.name);
  assert.equal(params.get('start'), '10.5');
  assert.equal(params.get('bpm'), '125');
  assert.equal(transcriptionParameters(file).has('bpm'), false);
  for (const options of [{ duration: 31 }, { duration: 0.1 }, { duration: 0 }, { start: -1 }, { start: NaN }, { bpm: 0 }, { source_mode: 'vocals' }]) {
    assert.throws(() => transcriptionParameters(file, options));
  }
  assert.throws(() => transcriptionParameters({ ...file, size: MAX_UPLOAD_BYTES + 1 }), /32 MiB/);
});

test('health validates the local service contract', async () => {
  const health = await checkConverter({ fetchImpl: async path => {
    assert.equal(path, '/api/health');
    return json({ service: 'mp3-to-rtttl', version: 1, ready: true });
  } });
  assert.equal(health.ready, true);
  await assert.rejects(checkConverter({ fetchImpl: async () => json({ service: 'other', ready: true }) }), /not connected/);
});

test('conversion uploads raw audio, polls, and returns the actual candidate result', async () => {
  const messages = [];
  const requests = [];
  const result = { candidates: [{ name: 'Hook', rtttl: 'Hook:d=8,o=5,b=125:c,e,g' }], analysis: { start: 10.5, duration: 8 } };
  const responses = [json({ id: 'job/42', status: 'queued' }, 202), json({ status: 'running', progress: 'Tracking the lead' }), json({ status: 'complete', result })];
  const actual = await transcribeAudio(file, { start: 10.5, duration: 8 }, {
    fetchImpl: async (path, options) => { requests.push({ path, options }); return responses.shift(); },
    delay: async () => {}, onProgress: text => messages.push(text),
  });
  assert.equal(actual, result);
  assert.equal(requests[0].options.body, file);
  assert.equal(requests[0].options.headers['Content-Type'], 'application/octet-stream');
  assert.equal(requests[1].path, '/api/transcriptions/job%2F42');
  assert.ok(messages.includes('Tracking the lead'));
});

test('conversion does not invent a melody when inference fails', async () => {
  let count = 0;
  await assert.rejects(transcribeAudio(file, {}, { fetchImpl: async () => ++count === 1 ? json({ id: 'job' }, 202) : json({ status: 'error', error: 'Missing Basic Pitch' }) }), /Missing Basic Pitch/);
  await assert.rejects(transcribeAudio(file, {}, { fetchImpl: async () => json({ error: 'Another conversion is running' }, 409) }), /Another conversion/);
});

test('polling responds to cancellation without another request', async () => {
  const controller = new AbortController();
  let count = 0;
  await assert.rejects(transcribeAudio(file, {}, {
    signal: controller.signal,
    fetchImpl: async () => ++count === 1 ? json({ id: 'job' }, 202) : json({ status: 'running' }),
    delay: async () => { controller.abort(); },
  }), { name: 'AbortError' });
  assert.equal(count, 2);
});
