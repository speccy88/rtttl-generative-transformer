// The converter is deliberately same-origin. Static hosting can still compose;
// audio conversion is available when the local Python studio serves this UI.
export const MAX_UPLOAD_BYTES = 32 * 1024 * 1024;
export const MAX_EXCERPT_SECONDS = 30;
export const MIN_EXCERPT_SECONDS = 0.25;

async function responseJson(response) {
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The local converter returned an unreadable response. Check that RTTTL Studio is running.'); }
  if (!response.ok) throw new Error(data.error || data.message || `The converter returned HTTP ${response.status}.`);
  return data;
}

export async function checkConverter({ fetchImpl = fetch, signal } = {}) {
  const data = await responseJson(await fetchImpl('/api/health', { signal, credentials: 'same-origin' }));
  if (data.service !== 'mp3-to-rtttl' || data.version !== 1) throw new Error('This page is not connected to the RTTTL Studio converter.');
  return data;
}

export function transcriptionParameters(file, options = {}) {
  if (!file || !Number.isFinite(file.size) || file.size <= 0) throw new Error('Choose an audio file first.');
  if (file.size > MAX_UPLOAD_BYTES) throw new Error('Choose an audio file no larger than 32 MiB. Export a short excerpt for larger songs.');
  const start = Number(options.start ?? 0);
  const duration = Number(options.duration ?? 12);
  if (!Number.isFinite(start) || start < 0) throw new Error('Excerpt start must be zero or greater.');
  if (!Number.isFinite(duration) || duration < MIN_EXCERPT_SECONDS || duration > MAX_EXCERPT_SECONDS) throw new Error('Choose an excerpt from 0.25 to 30 seconds long.');
  const source = options.source_mode ?? 'auto';
  const preset = options.preset ?? 'faithful';
  if (!['auto', 'full', 'harmonic'].includes(source)) throw new Error('Choose a valid analysis source.');
  if (!['faithful', 'balanced', 'smooth'].includes(preset)) throw new Error('Choose a valid melody detail preset.');
  const parameters = new URLSearchParams({ filename: file.name || 'audio.wav', start: String(start), duration: String(duration), source_mode: source, preset });
  if (options.bpm !== undefined && options.bpm !== null && String(options.bpm).trim() !== '') {
    const bpm = Number(options.bpm);
    if (!Number.isFinite(bpm) || bpm < 25 || bpm > 900) throw new Error('Tempo must be between 25 and 900 BPM, or blank for automatic detection.');
    parameters.set('bpm', String(bpm));
  }
  return parameters;
}

function wait(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(signal.reason || new DOMException('Aborted', 'AbortError')); return; }
    const abort = () => { clearTimeout(timer); reject(signal.reason || new DOMException('Aborted', 'AbortError')); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, milliseconds);
    signal?.addEventListener('abort', abort, { once: true });
  });
}

export async function transcribeAudio(file, options, { fetchImpl = fetch, signal, onProgress = () => {}, pollInterval = 700, delay = wait } = {}) {
  const parameters = transcriptionParameters(file, options);
  onProgress('Sending the selected file to your local converter…');
  const job = await responseJson(await fetchImpl(`/api/transcriptions?${parameters}`, {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
    credentials: 'same-origin', body: file, signal,
  }));
  if (typeof job.id !== 'string' || !job.id || job.id.length > 128) throw new Error('The converter did not return a valid job identifier.');
  // The backend owns the job; aborting this request stops waiting, not inference.
  for (;;) {
    if (signal?.aborted) throw signal.reason || new DOMException('Aborted', 'AbortError');
    const status = await responseJson(await fetchImpl(`/api/transcriptions/${encodeURIComponent(job.id)}`, { signal, credentials: 'same-origin' }));
    if (status.status === 'complete') {
      if (!Array.isArray(status.result?.candidates) || !status.result.candidates.length) throw new Error('No melody candidates were found. Try a clearer lead or a different excerpt.');
      return status.result;
    }
    if (status.status === 'error') throw new Error(status.error || 'Conversion failed. Try another excerpt.');
    if (!['queued', 'running'].includes(status.status)) throw new Error('The converter returned an unknown job status.');
    onProgress(typeof status.progress === 'string' ? status.progress : 'Analyzing the selected excerpt…');
    await delay(pollInterval, signal);
  }
}
