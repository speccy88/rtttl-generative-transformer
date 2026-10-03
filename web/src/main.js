import './styles.css';
import { parseRtttl } from './rtttl.js';
import { createPlayer, renderWav, renderMp3 } from './audio.js';
import { primeBrowserCache, connectWorkerCache } from './model-cache.js';
import { isWebKitEngine, WEBKIT_CPU_REASON } from './browser-policy.js';

const browserCacheReady = primeBrowserCache().catch(() => {});

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const PROFILE_LABELS = { mixed: 'Mixed', 'pop-hook': 'Pop hook', chiptune: 'Chiptune', cinematic: 'Cinematic', dance: 'Dance', lullaby: 'Lullaby', none: 'Unguided' };
const PROFILE_DESCRIPTIONS = {
  mixed: 'A varied mix of all five melody guides, tempos, and keys.',
  'pop-hook': 'A bright, singable little hook. Naturally paced at 88–132 BPM.',
  chiptune: 'Quick notes and playful leaps. A tiny adventure at 120–180 BPM.',
  cinematic: 'More room to breathe, with a minor-key feel. 60–104 BPM.',
  dance: 'A steadier pulse and a little more energy. 118–150 BPM.',
  lullaby: 'Gentle steps and longer notes. Take it slow at 60–84 BPM.',
  none: 'Follow the model’s learned patterns, with repetition controls enabled.',
};
const TEMPO_DESCRIPTIONS = { mixed: 'A different pace for every melody', 'pop-hook': '88–132 BPM · bright & easy', chiptune: '120–180 BPM · playful & quick', cinematic: '60–104 BPM · room to breathe', dance: '118–150 BPM · find the pulse', lullaby: '60–84 BPM · soft & unhurried', none: 'Tempo chosen by the melody model' };
const DEMO = {
  id: 'studio-demo', title: 'Hello, daydream.', name: 'Daydream',
  rtttl: 'Daydream:d=8,o=5,b=108:c,e,g,e,4a,g,4e,p,d,e,g,4c6,b,4g,4c6',
  settings: { tonic: 'C', mode: 'major' }, source: 'demo',
};
const state = {
  profile: 'mixed', songs: [], currentSong: DEMO, parsedSong: parseRtttl(DEMO.rtttl),
  worker: null, disconnectWorkerCache: null, manifest: null, busy: false, exporting: false, jobId: null,
  incomingSongs: [], receivedFirstSong: false, titleController: null,
  cancelled: false, namingRequested: false, backend: null, activeNote: null, manifestController: null,
  finishing: false,
  leavingDuringGeneration: false,
  usedTitles: [], usedNames: [],
  disposeTitleWorker: null,
};
const LIBRARY_STORAGE_KEY = 'pocket-composer-library-v1';
const MAX_SAVED_SONGS = 64;

function saveLibrary() {
  // Save completed scores before the optional, larger title model starts. Mobile
  // browsers can discard a tab under memory pressure; the music must survive it.
  const songs = state.songs.slice(-MAX_SAVED_SONGS).map(song => ({
    id: song.id, rtttl: song.rtttl, title: song.title, name: song.name,
    source: song.source, settings: song.settings || song.generation_settings,
    naming: song.naming, namingPending: Boolean(song.namingPending), seed: song.seed,
  }));
  try {
    localStorage.setItem(LIBRARY_STORAGE_KEY, JSON.stringify({ version: 1, songs,
      selectedId: state.currentSong.id, interrupted: state.busy || state.leavingDuringGeneration,
      usedTitles: state.usedTitles, usedNames: state.usedNames }));
  } catch { /* Storage can be unavailable; keep the playable in-memory library. */ }
}

function restoreLibrary() {
  try {
    const raw = localStorage.getItem(LIBRARY_STORAGE_KEY);
    if (!raw || raw.length > 5_000_000) return false;
    const saved = JSON.parse(raw);
    if (saved.version !== 1 || !Array.isArray(saved.songs)) return false;
    for (const key of ['usedTitles', 'usedNames']) {
      state[key] = Array.isArray(saved[key]) ? saved[key].filter(value => typeof value === 'string' && value.length < 128).slice(-64) : [];
    }
    const ids = new Set();
    const restored = saved.songs.slice(-MAX_SAVED_SONGS).flatMap(song => {
      try {
        if (!song || typeof song.id !== 'string' || !song.id || song.id.length > 128
            || ids.has(song.id) || typeof song.rtttl !== 'string' || song.rtttl.length > 65536) return [];
        parseRtttl(song.rtttl);
        ids.add(song.id);
        return [{ ...song, namingPending: false }];
      } catch { return []; }
    });
    if (!restored.length) return false;
    state.songs = restored;
    selectSong(restored.find(song => song.id === saved.selectedId) || restored[0]);
    drawLibrary();
    if (saved.interrupted) setProgress('Your completed melodies were restored. Generation was interrupted; press play or make a new batch.', 100);
    saveLibrary();
    return true;
  } catch { return false; }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}
function formatTime(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}
function displayTitle(song) { return song.title || song.name?.replace(/^Melody(\d+)$/, 'Melody $1') || 'Untitled melody'; }
function showError(message) {
  $('#error-text').textContent = message instanceof Error ? message.message : String(message);
  $('#error-message').hidden = false;
  const bounds = $('#error-message').getBoundingClientRect();
  if (bounds.bottom < 0 || bounds.top > window.innerHeight) $('#error-message').scrollIntoView({ block: 'nearest', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}
function clearError() { $('#error-message').hidden = true; }
function songDescription(song, includeDuration = false) {
  const parsed = song === state.currentSong ? state.parsedSong : parseRtttl(song.rtttl);
  const settings = song.settings || song.generation_settings || {};
  const source = song.source === 'demo' ? 'Studio demo' : song.source === 'import' ? 'Your RTTTL' : PROFILE_LABELS[settings.profile || 'none'] || 'Melody';
  const key = settings.tonic && settings.mode ? `${settings.tonic} ${settings.mode === 'natural-minor' ? 'minor' : settings.mode}` : null;
  return [source, `${parsed.bpm} BPM`, key, includeDuration ? formatTime(parsed.durationSeconds) : null].filter(Boolean).join(' · ');
}

const player = createPlayer({
  onProgress({ elapsed, duration, progress, eventIndex }) {
    $('#elapsed-time').textContent = formatTime(elapsed);
    $('#duration-time').textContent = formatTime(duration);
    $('#playhead').style.left = `${Math.min(100, Math.max(0, progress * 100))}%`;
    if (state.activeNote !== eventIndex) {
      $('#roll-notes .is-active')?.classList.remove('is-active');
      $(`#roll-notes [data-note="${eventIndex}"]`)?.classList.add('is-active');
      state.activeNote = eventIndex;
    }
  },
  onEnded() { setPlaybackUi(false); $('#elapsed-time').textContent = formatTime(state.parsedSong.durationSeconds); },
});

function setPlaybackUi(playing) {
  $('#player-card').classList.toggle('is-playing', playing);
  $('#play-button').setAttribute('aria-label', playing ? 'Stop melody' : 'Play melody');
  $('#play-icon').innerHTML = playing ? '<rect x="6" y="6" width="12" height="12" rx="1" />' : '<path d="m9 5 11 7-11 7z"/>';
  if (!playing) { $('#roll-notes .is-active')?.classList.remove('is-active'); state.activeNote = null; }
  for (const row of $$('.song-row')) {
    const selected = row.dataset.songId === String(state.currentSong.id);
    const symbol = row.querySelector('.song-mini-play svg');
    if (symbol) symbol.innerHTML = selected && playing ? '<rect x="4" y="4" width="12" height="12" rx="1" />' : '<path d="m7 4 10 6-10 6z"/>';
    row.querySelector('.song-select')?.setAttribute('aria-label', `${selected && playing ? 'Stop' : 'Play'} ${row.querySelector('.song-title')?.textContent || 'melody'}`);
  }
}

function stopPlayback() {
  player.stop();
  setPlaybackUi(false);
  $('#elapsed-time').textContent = '0:00';
  $('#playhead').style.left = '0%';
}

async function togglePlayback() {
  if (player.isPlaying) { stopPlayback(); return; }
  try {
    await player.play(state.parsedSong, { waveform: $('#waveform').value, volume: Number($('#volume').value) });
    setPlaybackUi(player.isPlaying);
  } catch (error) { stopPlayback(); showError(error); }
}

function drawNotes(song) {
  const pitches = song.events.filter(event => event.pitch !== null).map(event => event.pitch);
  const lower = pitches.length ? Math.min(...pitches) - 2 : 60;
  const upper = pitches.length ? Math.max(...pitches) + 2 : 84;
  let elapsed = 0;
  const fragment = document.createDocumentFragment();
  song.events.forEach((event, index) => {
    if (event.pitch !== null) {
      const note = document.createElement('span');
      note.className = 'roll-note';
      note.dataset.note = String(index);
      note.style.left = `${elapsed / song.durationSeconds * 100}%`;
      note.style.width = `${Math.max(0.25, event.seconds / song.durationSeconds * 100 * 0.9)}%`;
      note.style.top = `${(upper - event.pitch) / (upper - lower) * 85}%`;
      fragment.appendChild(note);
    }
    elapsed += event.seconds;
  });
  $('#roll-notes').replaceChildren(fragment);
}

function selectSong(song) {
  const parsed = parseRtttl(song.rtttl);
  stopPlayback();
  state.currentSong = song;
  state.parsedSong = parsed;
  $('#now-playing-title').textContent = displayTitle(song);
  $('#now-playing-meta').textContent = songDescription(song);
  const eyebrow = song.source === 'demo' ? 'A LITTLE SOMETHING TO START' : song.source === 'import' ? 'YOUR TUNE, RIGHT AT HOME' : 'FRESH FROM YOUR BROWSER';
  $('#player-eyebrow').innerHTML = `<span class="live-dot"></span>${eyebrow}`;
  $('#duration-time').textContent = formatTime(parsed.durationSeconds);
  $('#piano-roll').setAttribute('aria-label', `${parsed.events.length} events across ${formatTime(parsed.durationSeconds)}. Higher bars represent higher notes.`);
  $('#previous-song').disabled = state.songs.length < 2;
  $('#next-song').disabled = state.songs.length < 2;
  drawNotes(parsed);
  for (const row of $$('.song-row')) {
    const selected = row.dataset.songId === String(song.id);
    row.classList.toggle('is-selected', selected);
    row.querySelector('.song-select').setAttribute('aria-pressed', String(selected));
  }
  if (song.source !== 'demo') saveLibrary();
}

function drawLibrary() {
  $('#library-empty').hidden = state.songs.length > 0;
  $('#library-count').textContent = state.songs.length ? `${String(state.songs.length).padStart(2, '0')} LITTLE ${state.songs.length === 1 ? 'MELODY' : 'MELODIES'}` : 'THE LISTENING ROOM';
  $('#song-list').innerHTML = state.songs.map((song, index) => {
    const parsed = parseRtttl(song.rtttl);
    const selected = String(song.id) === String(state.currentSong.id);
    const preview = parsed.events.slice(0, 19).map(event => `<i style="height:${event.pitch === null ? 2 : 6 + ((event.pitch - 60) % 20)}px"></i>`).join('');
    return `<li class="song-row${selected ? ' is-selected' : ''}" data-song-id="${escapeHtml(song.id)}"><span class="song-number">${String(index + 1).padStart(2, '0')}</span><button class="song-select" type="button" data-song-select="${escapeHtml(song.id)}" aria-pressed="${selected}" aria-label="Play ${escapeHtml(displayTitle(song))}"><span class="song-mini-play" aria-hidden="true"><svg viewBox="0 0 20 20"><path d="m7 4 10 6-10 6z"/></svg></span><span class="song-info"><span class="song-title">${escapeHtml(displayTitle(song))}</span><span class="song-meta">${escapeHtml(songDescription(song, true))}${song.namingPending ? '<span class="song-naming">naming next ✦</span>' : ''}</span></span><span class="song-preview" aria-hidden="true">${preview}</span></button><button type="button" class="song-txt-download" data-song-download="${escapeHtml(song.id)}" aria-label="Download ${escapeHtml(displayTitle(song))} as RTTTL text" title="Download RTTTL text">↓</button></li>`;
  }).join('');
  $('#previous-song').disabled = state.songs.length < 2;
  $('#next-song').disabled = state.songs.length < 2;
  setPlaybackUi(player.isPlaying);
}

function updateCurrentTitle() {
  const current = state.songs.find(song => String(song.id) === String(state.currentSong.id));
  if (current) {
    state.currentSong = current;
    $('#now-playing-title').textContent = displayTitle(current);
    // Notes are unchanged by naming. Updating the parsed name keeps audio exports consistent.
    state.parsedSong = parseRtttl(current.rtttl);
  }
}

function moveSong(direction) {
  if (state.songs.length < 2) return;
  const wasPlaying = player.isPlaying;
  const current = state.songs.findIndex(song => String(song.id) === String(state.currentSong.id));
  const next = (Math.max(0, current) + direction + state.songs.length) % state.songs.length;
  selectSong(state.songs[next]);
  if (wasPlaying) void togglePlayback();
}

function fileStem(song) {
  return displayTitle(song).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 72) || 'pocket-melody';
}
function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.style.display = 'none';
  document.body.appendChild(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
function saveText(song) {
  saveBlob(new Blob([`${song.rtttl}\n`], { type: 'text/plain;charset=utf-8' }), `${fileStem(song)}.txt`);
}
async function downloadCurrent(format) {
  if (format === 'txt') { saveText(state.currentSong); return; }
  if (state.exporting) return;
  state.exporting = true;
  const song = { ...state.currentSong };
  const options = { waveform: $('#waveform').value, volume: Number($('#volume').value) };
  $$('.download-actions button').forEach(button => { button.disabled = button.dataset.download !== 'txt'; });
  const status = $('#export-status');
  status.hidden = false;
  status.textContent = `Rendering ${format.toUpperCase()} on your device…`;
  try {
    const blob = format === 'wav' ? await renderWav(song.rtttl, options) : await renderMp3(song.rtttl, {
      ...options, onProgress: progress => { status.textContent = `Making your MP3… ${Math.round(progress * 100)}%`; },
    });
    saveBlob(blob, `${fileStem(song)}.${format}`);
    status.textContent = `${format.toUpperCase()} is ready. Enjoy your little melody.`;
  } catch (error) { status.hidden = true; showError(error); }
  finally { state.exporting = false; $$('.download-actions button').forEach(button => { button.disabled = false; }); }
}

function updateTempoControls() {
  const mode = $('#tempo-mode').value;
  $('#tempo-fixed').hidden = mode !== 'fixed';
  $('#tempo-range').hidden = mode !== 'range';
  $('#tempo-summary').hidden = mode !== 'auto';
  $('#tempo-summary>span:last-child').textContent = TEMPO_DESCRIPTIONS[state.profile];
  $('#bpm').disabled = mode !== 'fixed';
  $('#bpm-min').disabled = mode !== 'range';
  $('#bpm-max').disabled = mode !== 'range';
}
function chooseProfile(profile) {
  state.profile = profile;
  $$('[data-profile]').forEach(button => {
    const selected = button.dataset.profile === profile;
    button.classList.toggle('is-selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  $('#profile-description').textContent = PROFILE_DESCRIPTIONS[profile];
  $('#tonic option[value="auto"]').textContent = profile === 'mixed' ? 'Surprise me' : 'Style default · C';
  $('#tonic').disabled = profile === 'none';
  $('#mode').disabled = profile === 'none';
  updateTempoControls();
}
function integerInput(selector, label, min, max) {
  const raw = $(selector).value.trim();
  const value = Number(raw);
  if (!raw || !Number.isInteger(value) || value < min || value > max) throw new Error(`${label} must be a whole number from ${min} to ${max}.`);
  return value;
}
function shuffleSeed() {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  $('#seed').value = String(value[0] & 0x7fffffff);
}
function generationOptions() {
  const maximum = integerInput('#length', 'Melody length', 16, 96);
  const options = {
    profile: state.profile === 'none' ? null : state.profile,
    maxEvents: maximum, minEvents: 16,
    numSongs: integerInput('#song-count', 'Number of melodies', 1, 8),
    seed: integerInput('#seed', 'Seed', 0, 2147483647),
    temperature: Number($('#temperature').value),
    topK: integerInput('#top-k', 'Top-k', 0, 200),
    topP: Number($('#top-p').value),
    repetitionPenalty: 1.2, maxPitchRun: 4, maxMotifRepeats: 3,
    backend: $('#backend').value,
  };
  if (!Number.isFinite(options.topP) || options.topP < 0.1 || options.topP > 1) throw new Error('Top-p must be between 0.1 and 1.');
  if ($('#tonic').value !== 'auto' && state.profile !== 'none') options.tonic = $('#tonic').value;
  if ($('#mode').value !== 'auto' && state.profile !== 'none') options.mode = $('#mode').value;
  if ($('#tempo-mode').value === 'fixed') options.bpm = integerInput('#bpm', 'Tempo', 25, 900);
  if ($('#tempo-mode').value === 'range') {
    const lower = integerInput('#bpm-min', 'Minimum tempo', 25, 900);
    const upper = integerInput('#bpm-max', 'Maximum tempo', 25, 900);
    if (lower > upper) throw new Error('The minimum tempo must be no greater than the maximum.');
    options.bpmRange = [lower, upper];
  }
  return options;
}

function setBusy(busy) {
  state.busy = busy;
  $('#composer-controls').disabled = busy;
  $('#generate-button').disabled = busy;
  $('#generate-label').textContent = busy ? 'Making a little magic…' : 'Make some music';
  $('#cancel-button').hidden = !busy;
  $('#cancel-button').disabled = false;
  $('#cancel-button').textContent = 'Stop generation';
  if (!busy) { chooseProfile(state.profile); $('#hardware-status').dataset.state = state.backend === 'wasm' ? 'cpu' : 'ready'; }
  else $('#hardware-status').dataset.state = 'busy';
  if (!busy) saveLibrary();
}
function setProgress(message, percent = null) {
  $('#generation-status').hidden = false;
  $('#generation-message').textContent = message;
  const determinate = typeof percent === 'number' && Number.isFinite(percent);
  $('#generation-percent').textContent = determinate ? `${Math.min(100, Math.max(0, Math.round(percent)))}%` : '';
  $('#generation-bar').parentElement.classList.toggle('is-indeterminate', !determinate);
  $('#generation-bar').style.width = determinate ? `${Math.min(100, Math.max(0, percent))}%` : '35%';
}

function updateCacheNotice(event) {
  const notice = $('#cache-notice');
  if (!notice) return;
  if (event.cacheState === 'unavailable') state.cacheStorageWarning = true;
  else if (state.cacheStorageWarning) return;
  notice.hidden = false;
  notice.textContent = event.cacheState === 'unavailable' ? event.message
    : event.source === 'memory' ? 'Reusing model files already loaded in this tab.'
    : event.cacheState === 'hit' ? 'Reusing model files saved in this browser.'
    : 'Model files saved in this browser for your next visit.';
  notice.dataset.state = event.cacheState === 'unavailable' ? 'temporary' : 'saved';
}
function setBackend(backend, fallbackReason) {
  state.backend = backend;
  const cpu = backend === 'wasm' || backend === 'cpu';
  const text = cpu ? 'CPU · all generation stays local' : 'WebGPU · all generation stays local';
  $('#hardware-status').innerHTML = `<span class="status-dot"></span>${text}`;
  $('#hardware-status').dataset.state = state.busy ? 'busy' : cpu ? 'cpu' : 'ready';
  $('#hardware-status').title = fallbackReason || (cpu ? 'Melody inference uses WebAssembly on your CPU.' : 'Melody inference uses your local GPU through WebGPU.');
}

async function loadManifest() {
  if (state.manifest) return state.manifest;
  const url = new URL(`${import.meta.env.BASE_URL}model-manifest.json`, window.location.origin);
  const controller = new AbortController();
  state.manifestController = controller;
  const timer = setTimeout(() => controller.abort(new Error('The melody model information took too long to load. Check your connection and try again.')), 30000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error('The melody model information could not load. Check your connection and try again.');
    state.manifest = await response.json();
    return state.manifest;
  } finally { clearTimeout(timer); state.manifestController = null; }
}

function receiveSong(song) {
  // Validate a completed melody before showing play/download controls.
  parseRtttl(song.rtttl);
  if (!state.receivedFirstSong) {
    state.receivedFirstSong = true;
    stopPlayback();
    state.songs = [];
  }
  const record = { ...song, id: song.id ?? `${state.jobId}-${state.incomingSongs.length}`, source: 'generated', namingPending: state.namingRequested };
  state.incomingSongs.push(record);
  state.songs.push(record);
  if (state.incomingSongs.length === 1) selectSong(record);
  drawLibrary();
  saveLibrary();
}

function mergeSongTitle(original, named) {
  if (!named || String(named.id) !== String(original.id)) return original;
  // Naming can change only the display title and RTTTL name. Never let a title
  // response replace the score, identity, metadata, or the rest of the batch.
  const before = original.rtttl.indexOf(':');
  const after = typeof named.rtttl === 'string' ? named.rtttl.indexOf(':') : -1;
  if (before < 0 || after < 1 || original.rtttl.slice(before) !== named.rtttl.slice(after)) return original;
  const parsed = parseRtttl(named.rtttl);
  return { ...original, title: typeof named.title === 'string' ? named.title : original.title,
    name: parsed.name, rtttl: named.rtttl, naming: named.naming, namingPending: false };
}

function applyNamedSongs(named) {
  const byId = new Map(named.filter(song => song && song.id !== undefined).map(song => [String(song.id), song]));
  const update = song => {
    try { return mergeSongTitle(song, byId.get(String(song.id))); }
    catch { return song; }
  };
  state.incomingSongs = state.incomingSongs.map(update);
  state.songs = state.songs.map(update);
  for (const song of state.incomingSongs) {
    if (song.naming?.status !== 'named'
        && !(song.naming?.status === 'fallback' && song.naming.strategy === 'musical-character')) continue;
    for (const [key, value] of [['usedTitles', song.title], ['usedNames', song.name]]) {
      if (typeof value !== 'string' || !value) continue;
      state[key] = [...state[key].filter(previous => previous.toLowerCase() !== value.toLowerCase()), value].slice(-64);
    }
  }
  updateCurrentTitle(); drawLibrary(); saveLibrary();
}

async function nameGeneratedSongs() {
  if (!state.namingRequested || !state.incomingSongs.length || state.cancelled) return;
  const jobId = state.jobId;
  const songs = [...state.incomingSongs];
  const controller = new AbortController();
  state.titleController = controller;
  setProgress('Loading the optional title model…');
  try {
    // This separate chunk and its LLM dependencies are loaded only after opt-in.
    const { titleSongs, disposeTitleWorker } = await import('./titles.js');
    state.disposeTitleWorker = disposeTitleWorker;
    if (state.cancelled || state.jobId !== jobId) return;
    const named = await titleSongs(songs, {
      signal: controller.signal,
      usedTitles: state.usedTitles, usedNames: state.usedNames,
      onProgress(progress) {
        if (state.cancelled || state.jobId !== jobId) return;
        if (progress.cacheEvent) { updateCacheNotice(progress); return; }
        setProgress(progress.message || 'Finding a name for your melody…', progress.downloadProgress ?? null);
        if (progress.song && progress.songId !== undefined) applyNamedSongs([progress.song]);
      },
    });
    if (!state.cancelled && state.jobId === jobId && Array.isArray(named)) applyNamedSongs(named);
  } catch (error) {
    if (!state.cancelled && state.jobId === jobId && error?.name !== 'AbortError') showError(`Your melodies are ready, but automatic naming did not finish: ${error.message || error}`);
  } finally { if (state.titleController === controller) state.titleController = null; }
}

async function finishGeneration(data) {
  if (state.finishing) return;
  state.finishing = true;
  const jobId = state.jobId;
  if (data.backend) setBackend(data.backend, data.fallbackReason);
  // Some worker versions deliver only the final array; support both streaming and final results.
  if (!state.incomingSongs.length && Array.isArray(data.songs)) data.songs.forEach(receiveSong);
  state.cancelled ||= Boolean(data.cancelled);
  await nameGeneratedSongs();
  if (state.jobId !== jobId) return;
  state.songs.forEach(song => { song.namingPending = false; });
  drawLibrary();
  const count = state.incomingSongs.length;
  setProgress(state.cancelled ? `Stopped. ${count ? `${count} completed ${count === 1 ? 'melody is' : 'melodies are'} ready to play.` : 'Ready whenever you are.'}` : `${count} ${count === 1 ? 'melody' : 'melodies'}, made on your device. Press play.`, 100);
  state.jobId = null;
  setBusy(false);
  state.finishing = false;
  saveLibrary();
}

function getWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(new URL('./melody-worker.js', import.meta.url), { type: 'module' });
  state.disconnectWorkerCache = connectWorkerCache(worker);
  worker.addEventListener('message', ({ data }) => {
    if (!state.busy || (data.id !== undefined && String(data.id) !== String(state.jobId))) return;
    try {
      if (data.type === 'cache') updateCacheNotice(data);
      else if (data.type === 'status') {
        if (data.backend) setBackend(data.backend, data.fallbackReason);
        if (!state.cancelled) setProgress(data.message || 'Preparing the melody model…', typeof data.downloadProgress === 'number' ? data.downloadProgress * 100 : null);
      } else if (data.type === 'progress') {
        if (!state.cancelled) {
          const songIndex = Math.min((data.songIndex ?? 0) + 1, data.numSongs || 1);
          const progress = ((data.songIndex ?? 0) + Math.min(1, (data.eventCount ?? 0) / (data.maxEvents || 48))) / (data.numSongs || 1) * 100;
          setProgress(`Composing melody ${songIndex} of ${data.numSongs}…`, progress);
        }
      } else if (data.type === 'song' && !state.finishing) receiveSong(data.song);
      else if (data.type === 'done') void finishGeneration(data).catch(error => { showError(error); setBusy(false); });
      else if (data.type === 'error') {
        showError(data.message || 'The melody model could not finish. Try again or select CPU in the advanced settings.');
        state.songs.forEach(song => { song.namingPending = false; }); drawLibrary();
        setProgress('Generation stopped. You can adjust the settings and try again.', 0);
        state.jobId = null; setBusy(false);
      }
    } catch (error) { showError(error); state.worker?.postMessage({ type: 'cancel', id: state.jobId }); state.jobId = null; setBusy(false); }
  });
  worker.addEventListener('error', event => {
    if (state.worker !== worker) return;
    if (state.busy) {
      showError(event.message || 'The local generation worker could not start. Reload the page or try another browser.');
      setProgress('The model could not start. Your player is still available.', 0);
      state.jobId = null; setBusy(false);
    }
    state.disconnectWorkerCache?.(); state.disconnectWorkerCache = null;
    worker.terminate(); state.worker = null;
  });
  state.worker = worker;
  return worker;
}

async function generate(event) {
  event.preventDefault();
  if (state.busy) return;
  clearError();
  if (!$('#keep-seed').checked) shuffleSeed();
  let options;
  try { options = generationOptions(); } catch (error) { showError(error); return; }
  state.jobId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  const jobId = state.jobId;
  state.incomingSongs = []; state.receivedFirstSong = false; state.cancelled = false;
  state.finishing = false;
  state.leavingDuringGeneration = false;
  state.namingRequested = $('#name-songs').checked;
  setBusy(true); setProgress('Getting the local melody model ready…');
  try {
    const modelManifest = await loadManifest();
    await browserCacheReady;
    if (state.jobId !== jobId) return;
    if (state.cancelled) { await finishGeneration({ cancelled: true, songs: [] }); return; }
    getWorker().postMessage({ type: 'generate', id: state.jobId, options, modelManifest });
  } catch (error) {
    if (state.jobId !== jobId) return;
    if (state.cancelled) { await finishGeneration({ cancelled: true, songs: [] }); return; }
    showError(error); setProgress('The model could not load. Check your connection and try again.', 0);
    state.jobId = null; setBusy(false);
  }
}

function cancelGeneration() {
  if (!state.busy) return;
  state.cancelled = true;
  state.manifestController?.abort();
  state.titleController?.abort();
  state.worker?.postMessage({ type: 'cancel', id: state.jobId });
  $('#cancel-button').disabled = true;
  $('#cancel-button').textContent = 'Stopping…';
  setProgress('Stopping after the current step. Completed melodies are kept.');
}

function importTune(text) {
  try {
    const parsed = parseRtttl(text);
    const record = { id: `import-${Date.now()}`, title: parsed.name, name: parsed.name, rtttl: text.replace(/^\uFEFF/, '').trim(), source: 'import' };
    clearError(); state.songs.push(record); selectSong(record); drawLibrary();
    saveLibrary();
    $('#player-card').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
  } catch (error) { showError(error); }
}

$$('[data-profile]').forEach(button => button.addEventListener('click', () => chooseProfile(button.dataset.profile)));
$('#composer-form').addEventListener('submit', generate);
$('#cancel-button').addEventListener('click', cancelGeneration);
$('#dismiss-error').addEventListener('click', clearError);
$('#tempo-mode').addEventListener('change', updateTempoControls);
$('#bpm-slider').addEventListener('input', event => { $('#bpm').value = event.target.value; });
$('#bpm').addEventListener('input', event => { $('#bpm-slider').value = event.target.value; });
$('#temperature').addEventListener('input', event => { $('#temperature-value').value = Number(event.target.value).toFixed(2); });
$('#shuffle-seed').addEventListener('click', shuffleSeed);
$('#seed').addEventListener('input', () => { $('#keep-seed').checked = true; });
$('#play-button').addEventListener('click', togglePlayback);
$('#previous-song').addEventListener('click', () => moveSong(-1));
$('#next-song').addEventListener('click', () => moveSong(1));
$('#volume').addEventListener('input', event => player.setVolume(Number(event.target.value)));
$('#waveform').addEventListener('change', () => { if (player.isPlaying) { stopPlayback(); void togglePlayback(); } });
$$('[data-download]').forEach(button => button.addEventListener('click', () => downloadCurrent(button.dataset.download)));
$('#song-list').addEventListener('click', event => {
  const downloadButton = event.target.closest('[data-song-download]');
  if (downloadButton) { const song = state.songs.find(item => String(item.id) === downloadButton.dataset.songDownload); if (song) saveText(song); return; }
  const selectButton = event.target.closest('[data-song-select]');
  if (selectButton) {
    const song = state.songs.find(item => String(item.id) === selectButton.dataset.songSelect);
    if (song) { if (String(state.currentSong.id) !== String(song.id)) selectSong(song); void togglePlayback(); }
  }
});
$('#load-rtttl').addEventListener('click', () => importTune($('#rtttl-input').value));
$('#rtttl-file').addEventListener('change', async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (file.size > 65536) throw new Error('Choose an RTTTL text file smaller than 64 KB.');
    const text = await file.text(); $('#rtttl-input').value = text; importTune(text);
  } catch (error) { showError(error); }
  finally { event.target.value = ''; }
});
document.addEventListener('keydown', event => {
  if (event.code === 'Space' && !event.repeat && !event.target.closest('button, input, textarea, select, a, summary, [contenteditable="true"]')) { event.preventDefault(); void togglePlayback(); }
});
window.addEventListener('pagehide', () => {
  state.leavingDuringGeneration = state.busy;
  saveLibrary(); stopPlayback();
  if (state.busy) {
    state.cancelled = true; state.jobId = null; state.finishing = false;
  }
  state.manifestController?.abort(); state.titleController?.abort();
  state.disposeTitleWorker?.();
  state.disconnectWorkerCache?.(); state.disconnectWorkerCache = null;
  state.worker?.terminate(); state.worker = null;
  if (state.leavingDuringGeneration) {
    state.songs.forEach(song => { song.namingPending = false; });
    drawLibrary(); setBusy(false);
  }
});
window.addEventListener('pageshow', event => {
  if (!event.persisted || !state.leavingDuringGeneration) return;
  setProgress('Your completed melodies were restored. Generation was interrupted; press play or make a new batch.', 100);
  state.leavingDuringGeneration = false;
  saveLibrary();
});

if (!restoreLibrary()) selectSong(DEMO);
chooseProfile('mixed');
async function detectHardware() {
  if (isWebKitEngine()) {
    $('#backend option[value="webgpu"]').disabled = true;
    $('#backend option[value="webgpu"]').textContent = 'WebGPU unavailable in this browser';
    $('#hardware-status').innerHTML = '<span class="status-dot"></span>CPU mode · ready to create';
    $('#hardware-status').dataset.state = 'cpu';
    $('#hardware-status').title = WEBKIT_CPU_REASON;
    return;
  }
  try {
    if (navigator.gpu && await navigator.gpu.requestAdapter()) {
      $('#hardware-status').innerHTML = '<span class="status-dot"></span>WebGPU available · ready to create';
      $('#hardware-status').dataset.state = 'ready';
      return;
    }
  } catch { /* Device availability will also be checked by the model worker. */ }
  $('#hardware-status').innerHTML = '<span class="status-dot"></span>CPU mode · ready to create';
  $('#hardware-status').dataset.state = 'cpu';
}
void detectHardware();
