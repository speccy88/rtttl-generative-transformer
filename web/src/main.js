import './styles.css';
import { parseRtttl } from './rtttl.js';
import { createPlayer, renderWav, renderMp3 } from './audio.js';
import { primeBrowserCache, connectWorkerCache } from './model-cache.js';
import { isWebKitEngine, WEBKIT_CPU_REASON } from './browser-policy.js';
import { inspectAudio, transcribeInBrowser } from './audio-transcriber.js';
import { appendStudioSongs, changeScoreTempo, editScoreSelection, transposeScore, validScoreSelection } from './studio-score.js';
import { validateOptions } from './sampler.js';

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
  audioReady: false, inspecting: false, converting: false, conversionController: null, inspectController: null, conversionSource: null,
  sourceUrl: null, sourceDuration: null, sourcePeaks: [], excerptPlaying: false, waveformDrag: null,
  continuationSource: null, generationMode: null, generationSelection: null,
  selection: null, selectionAnchor: null, focusedEvent: 0, comparison: null,
};
const LIBRARY_STORAGE_KEY = 'pocket-composer-library-v1';
const MAX_SAVED_SONGS = 64;
const MAX_AUDIO_FILE_BYTES = 32 * 1024 * 1024;

function saveLibrary() {
  // Save completed scores before the optional, larger title model starts. Mobile
  // browsers can discard a tab under memory pressure; the music must survive it.
  const songs = state.songs.slice(-MAX_SAVED_SONGS).map(song => ({
    id: song.id, rtttl: song.rtttl, title: song.title, name: song.name,
    source: song.source, settings: song.settings || song.generation_settings,
    naming: song.naming, namingPending: Boolean(song.namingPending), seed: song.seed,
    parentId: song.parentId, prefixEventCount: song.prefixEventCount,
    selectedRange: song.selectedRange, selection: song.selection, editMode: song.editMode,
    changedEventCount: song.changedEventCount, originalChosen: song.originalChosen,
    editStrength: song.editStrength, repairMethod: song.repairMethod, scoreDetails: song.scoreDetails,
    inferenceCalls: song.inferenceCalls, maxChangedEvents: song.maxChangedEvents, maxPitchDistance: song.maxPitchDistance,
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
  const source = ({ demo: 'Studio demo', import: 'Your RTTTL', converted: 'Audio conversion', edited: 'Edited copy', continuation: 'AI continuation', variation: 'AI variation', repair: 'Repair suggestion' })[song.source] || PROFILE_LABELS[settings.profile || 'none'] || 'Melody';
  const key = settings.tonic && settings.mode ? `${settings.tonic} ${settings.mode === 'natural-minor' ? 'minor' : settings.mode}` : null;
  const original = song.source === 'converted' && settings.conversion?.filename ? settings.conversion.filename : null;
  return [source, `${parsed.bpm} BPM`, key, original, includeDuration ? formatTime(parsed.durationSeconds) : null].filter(Boolean).join(' · ');
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
    $('#source-audio').pause(); state.excerptPlaying = false;
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
    const note = document.createElement('button');
    note.type = 'button';
    note.className = `roll-event ${event.pitch === null ? 'roll-rest' : 'roll-note'}`;
    note.dataset.note = String(index);
    note.style.left = `${elapsed / song.durationSeconds * 100}%`;
    note.style.width = `${Math.max(0.25, event.seconds / song.durationSeconds * 100 * 0.9)}%`;
    note.style.top = event.pitch === null ? '88%' : `${(upper - event.pitch) / (upper - lower) * 72}%`;
    const pitch = event.pitch === null ? 'rest' : `${['C', 'C sharp', 'D', 'D sharp', 'E', 'F', 'F sharp', 'G', 'G sharp', 'A', 'A sharp', 'B'][event.pitch % 12]}${Math.floor(event.pitch / 12) - 1}`;
    note.setAttribute('aria-label', `Event ${index + 1}: ${pitch}, ${event.dotted ? 'dotted ' : ''}1/${event.duration} note`);
    note.title = note.getAttribute('aria-label');
    note.tabIndex = index === state.focusedEvent ? 0 : -1;
    fragment.appendChild(note);
    elapsed += event.seconds;
  });
  $('#roll-notes').replaceChildren(fragment);
}

function updateNoteSelection() {
  const selection = validScoreSelection(state.selection, state.parsedSong.events.length) ? state.selection : null;
  const selectedEvents = selection ? state.parsedSong.events.slice(selection.start, selection.end) : [];
  const pitches = selectedEvents.filter(event => event.pitch !== null).length;
  for (const button of $$('#roll-notes [data-note]')) {
    const index = Number(button.dataset.note);
    const selected = Boolean(selection && index >= selection.start && index < selection.end);
    button.setAttribute('aria-pressed', String(selected));
    button.classList.toggle('is-selected', selected);
    button.classList.toggle('is-outside-selection', Boolean(selection && !selected));
    button.tabIndex = index === state.focusedEvent ? 0 : -1;
  }
  $('#selection-start').max = $('#selection-end').max = String(state.parsedSong.events.length);
  $('#selection-start').value = selection ? String(selection.start + 1) : '';
  $('#selection-end').value = selection ? String(selection.end) : '';
  $('#selection-summary').textContent = selection
    ? `${selection.start === selection.end - 1 ? `Event ${selection.end}` : `Events ${selection.start + 1}–${selection.end}`} · ${pitches} ${pitches === 1 ? 'note' : 'notes'}${selectedEvents.length - pitches ? `, ${selectedEvents.length - pitches} rest${selectedEvents.length - pitches === 1 ? '' : 's'}` : ''}`
    : 'No notes selected';
  if (selectedEvents.length) {
    $('#selected-duration').value = String(selectedEvents[0].duration);
    $('#selected-dotted').checked = selectedEvents[0].dotted;
  }
  $('#phrase-hint').textContent = selectedEvents.length > 16
    ? `${selectedEvents.length} events selected. Choose a phrase of at most 16 events for variation or repair; manual edits can use the full range.`
    : selection && pitches === 0
      ? 'This range contains only rests. Include at least one pitched note for variation or repair.'
      : 'Select 1–16 events, including at least one note. Variations keep the rhythm, rests, and tempo; everything outside the selection is locked.';
  updateStudioButtons();
}

function selectNoteRange(anchor, focus, { save = true } = {}) {
  const last = state.parsedSong.events.length - 1;
  anchor = Math.max(0, Math.min(last, anchor));
  focus = Math.max(0, Math.min(last, focus));
  state.selectionAnchor = anchor;
  state.focusedEvent = focus;
  state.selection = { start: Math.min(anchor, focus), end: Math.max(anchor, focus) + 1 };
  state.currentSong.selectedRange = { ...state.selection };
  updateNoteSelection();
  if (save) saveLibrary();
}

function clearNoteSelection() {
  state.selection = null; state.selectionAnchor = null;
  state.currentSong.selectedRange = null;
  updateNoteSelection(); saveLibrary();
}

function updateVersionComparison() {
  const comparison = state.comparison;
  const parent = comparison && state.songs.find(song => String(song.id) === String(comparison.parentId));
  const version = comparison && state.songs.find(song => String(song.id) === String(comparison.versionId));
  $('#version-comparison').hidden = !parent || !version;
  if (parent && version) {
    $('#version-label').textContent = `Compare with ${displayTitle(parent)}`;
    $('#compare-parent').setAttribute('aria-pressed', String(state.currentSong.id === parent.id));
    $('#compare-version').setAttribute('aria-pressed', String(state.currentSong.id === version.id));
  }
  const song = state.currentSong;
  $('#edit-result-note').hidden = !['repair', 'variation'].includes(song.source);
  $('#edit-result-note').textContent = song.source === 'repair' && song.originalChosen
    ? 'Original scored best; no change suggested. Compare it with your recording before making an edit.'
    : song.source === 'repair'
      ? `${song.changedEventCount ?? 'Selected'} pitch${song.changedEventCount === 1 ? '' : 'es'} changed in this repair suggestion. The score reflects melody plausibility; listen and compare.`
      : `${song.changedEventCount ?? 'Selected'} pitch${song.changedEventCount === 1 ? '' : 'es'} varied. Rhythm, rests, tempo, and unselected events are kept.`;
}

function selectSong(song, { preserveComparison = false } = {}) {
  const parsed = parseRtttl(song.rtttl);
  stopPlayback();
  state.currentSong = song;
  state.parsedSong = parsed;
  const savedRange = song.selectedRange === null ? null : song.selectedRange || song.selection;
  state.selection = validScoreSelection(savedRange, parsed.events.length) ? { ...savedRange } : null;
  state.selectionAnchor = state.selection?.start ?? null; state.focusedEvent = state.selection?.start ?? 0;
  if (!preserveComparison) state.comparison = song.parentId ? { parentId: song.parentId, versionId: song.id, selection: state.selection } : null;
  $('#now-playing-title').textContent = displayTitle(song);
  $('#now-playing-meta').textContent = songDescription(song);
  const eyebrow = ({ demo: 'A LITTLE SOMETHING TO START', import: 'YOUR TUNE, RIGHT AT HOME', converted: 'FROM YOUR AUDIO EXCERPT', edited: 'YOUR EDITED VERSION', continuation: 'YOUR MELODY, CONTINUED', variation: 'A VARIATION ON YOUR PHRASE', repair: 'A REPAIR SUGGESTION TO COMPARE' })[song.source] || 'FRESH FROM YOUR BROWSER';
  $('#player-eyebrow').innerHTML = `<span class="live-dot"></span>${eyebrow}`;
  $('#duration-time').textContent = formatTime(parsed.durationSeconds);
  $('#piano-roll').setAttribute('aria-label', `${parsed.events.length} events across ${formatTime(parsed.durationSeconds)}. Higher bars represent higher notes.`);
  $('#previous-song').disabled = state.songs.length < 2;
  $('#next-song').disabled = state.songs.length < 2;
  drawNotes(parsed);
  updateNoteSelection();
  updateVersionComparison();
  $('#score-input').value = song.rtttl;
  $('#score-bpm').value = String(parsed.bpm);
  for (const row of $$('.song-row')) {
    const selected = row.dataset.songId === String(song.id);
    row.classList.toggle('is-selected', selected);
    row.querySelector('.song-select').setAttribute('aria-pressed', String(selected));
  }
  if (song.source !== 'demo') saveLibrary();
}

function drawLibrary() {
  $('#library-empty').hidden = state.songs.length > 0;
  $('#library-count').textContent = state.songs.length ? `${String(state.songs.length).padStart(2, '0')} LITTLE ${state.songs.length === 1 ? 'MELODY' : 'MELODIES'}` : 'YOUR WORKSPACE';
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
function generationOptions({ continuation = false } = {}) {
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
  if (!continuation && $('#tempo-mode').value === 'fixed') options.bpm = integerInput('#bpm', 'Tempo', 25, 900);
  if (!continuation && $('#tempo-mode').value === 'range') {
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
  updateStudioButtons();
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
  }
  if (state.continuationSource) retainSourceVersion(state.continuationSource);
  const record = { ...song, id: song.id ?? `${state.jobId}-${state.incomingSongs.length}`,
    source: state.continuationSource ? state.generationMode : 'generated', parentId: state.continuationSource?.id,
    selectedRange: state.generationSelection ? { ...state.generationSelection } : undefined,
    namingPending: state.namingRequested };
  state.incomingSongs.push(record);
  state.songs = appendStudioSongs(state.songs, [record], MAX_SAVED_SONGS);
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
          const action = state.generationMode === 'repair' ? 'Comparing repair suggestion' : state.generationMode === 'variation' ? 'Varying phrase' : 'Composing melody';
          const detail = state.generationMode === 'repair' && Number.isInteger(data.candidateIndex)
            ? ` · ${data.editPhase === 'score' ? 'scoring' : 'sampling'} candidate ${data.candidateIndex + 1} of ${data.candidateCount || 5}` : '';
          setProgress(`${action} ${songIndex} of ${data.numSongs}${detail}…`, state.generationMode === 'repair' ? null : progress);
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

async function generate(event, action = {}) {
  event?.preventDefault();
  if (state.busy || state.converting || state.inspecting) return;
  const { promptSong = null, editMode = null, selection = null } = action;
  clearError();
  if (!$('#keep-seed').checked) shuffleSeed();
  let options;
  try {
    options = generationOptions({ continuation: Boolean(promptSong) });
    if (promptSong) {
      options.promptSong = promptSong;
      // The source tempo and every source event are preserved by the sampler.
      delete options.bpm; delete options.bpmRange;
    }
    if (editMode) {
      options.editMode = editMode;
      options.selection = selection;
      options.editStrength = $('#edit-strength').value;
      options.numSongs = integerInput('#edit-count', 'Number of versions', 1, 4);
    }
    validateOptions(options);
  } catch (error) { showError(error); return; }
  // Every entry point keeps generation progress and its cancel action visible.
  setStudioMode('compose');
  state.continuationSource = promptSong ? { ...state.currentSong } : null;
  state.generationMode = editMode || (promptSong ? 'continuation' : 'generated');
  state.generationSelection = selection ? { ...selection } : null;
  state.jobId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  const jobId = state.jobId;
  state.incomingSongs = []; state.receivedFirstSong = false; state.cancelled = false;
  state.finishing = false;
  state.leavingDuringGeneration = false;
  state.namingRequested = $('#name-songs').checked;
  setBusy(true); setProgress(editMode === 'repair' ? 'Keeping the surrounding notes fixed and preparing repair suggestions…'
    : editMode === 'variation' ? 'Keeping the surrounding notes fixed and varying your selection…'
    : promptSong ? 'Keeping the selected melody and preparing a new phrase…' : 'Getting the browser melody model ready…');
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
    const record = { id: newSongId('import'), title: parsed.name, name: parsed.name, rtttl: text.replace(/^\uFEFF/, '').trim(), source: 'import' };
    clearError(); state.songs = appendStudioSongs(state.songs, [record], MAX_SAVED_SONGS); selectSong(record); drawLibrary();
    saveLibrary();
    $('#player-card').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
  } catch (error) { showError(error); }
}

function newSongId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${crypto.getRandomValues(new Uint32Array(1))[0].toString(36)}`;
}

function setStudioMode(mode, focus = false) {
  if ((mode === 'convert' && state.busy) || (mode === 'compose' && (state.converting || state.inspecting))) return;
  const converting = mode === 'convert';
  $('#converter-form').hidden = !converting;
  $('#composer-form').hidden = converting;
  $('#composer-heading').textContent = converting ? 'Find the melody.' : 'Set the mood.';
  for (const [id, selected] of [['convert-tab', converting], ['compose-tab', !converting]]) {
    const tab = $(`#${id}`);
    tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  }
}

function updateStudioButtons() {
  const audioWorking = state.converting || state.inspecting;
  const busy = state.busy || audioWorking;
  const selected = validScoreSelection(state.selection, state.parsedSong.events.length);
  const events = selected ? state.parsedSong.events.slice(state.selection.start, state.selection.end) : [];
  const pitched = events.some(event => event.pitch !== null);
  $('#convert-button').disabled = !state.audioReady || !state.conversionSource || busy;
  $('#generate-button').disabled = busy;
  $('#composer-controls').disabled = busy;
  $('#convert-tab').disabled = state.busy;
  $('#compose-tab').disabled = audioWorking;
  $('#continue-song').disabled = busy;
  $('#converter-controls').disabled = state.converting || state.busy;
  $('#convert-label').textContent = state.inspecting ? 'Reading your recording…' : state.converting ? 'Following the melody…' : 'Convert excerpt';
  $('#cancel-conversion').hidden = !audioWorking;
  $('#cancel-conversion').disabled = Boolean(state.conversionController?.signal.aborted || state.inspectController?.signal.aborted);
  $('#play-excerpt').disabled = !state.audioReady || audioWorking;
  $('#source-waveform').setAttribute('aria-disabled', String(!state.audioReady || busy));
  $('#note-edit-controls').disabled = busy || !selected;
  $('#selected-pitch-down').disabled = $('#selected-pitch-up').disabled = !pitched;
  $('#selected-rest').disabled = !pitched;
  $('#selected-note').disabled = !events.some(event => event.pitch === null);
  $('#vary-selection').disabled = $('#repair-selection').disabled = busy || !selected || !pitched || events.length > 16;
  for (const control of $$('#selection-start, #selection-end, #select-all-notes, #clear-note-selection, #edit-strength, #edit-count, #roll-notes button')) control.disabled = busy;
}

function sourceMetadata(duration) {
  if (!Number.isFinite(duration) || duration <= 0 || !state.conversionSource) return;
  state.sourceDuration = duration;
  const start = Number($('#excerpt-start').value);
  const requested = Number($('#excerpt-duration').value);
  const available = duration - Math.max(0, start);
  // Decoding can complete after a person has already adjusted the numeric range.
  if (Number.isFinite(requested) && requested > available && available > 0) $('#excerpt-duration').value = String(available);
  $('#audio-file-info').textContent = `${state.conversionSource.name} · ${(state.conversionSource.size / 1024 / 1024).toFixed(1)} MiB · ${formatTime(duration)}`;
  updateWaveformSelection();
}

function drawSourceWaveform(peaks) {
  const samples = Array.isArray(peaks) || ArrayBuffer.isView(peaks) ? Array.from(peaks) : [];
  const peak = Math.max(0.01, ...samples.filter(Number.isFinite));
  const commands = samples.map((value, index) => {
    const x = (index + 0.5) / samples.length * 1000;
    const amplitude = Number.isFinite(value) ? Math.max(1, Math.min(46, Math.abs(value) / peak * 46)) : 1;
    return `M${x.toFixed(2)},${(50 - amplitude).toFixed(2)}v${(amplitude * 2).toFixed(2)}`;
  });
  $('#waveform-peaks').setAttribute('d', commands.join(' '));
  $('#source-waveform').hidden = false;
  updateWaveformSelection();
}

function updateWaveformSelection() {
  const total = state.sourceDuration;
  if (!total) return;
  const start = Math.max(0, Math.min(total, Number($('#excerpt-start').value) || 0));
  const end = Math.max(start, Math.min(total, start + (Number($('#excerpt-duration').value) || 0)));
  $('#excerpt-selection').style.left = `${start / total * 100}%`;
  $('#excerpt-selection').style.width = `${(end - start) / total * 100}%`;
  for (const [id, value] of [['excerpt-start-handle', start], ['excerpt-end-handle', end]]) {
    const handle = $(`#${id}`);
    handle.setAttribute('aria-valuemax', String(total));
    handle.setAttribute('aria-valuenow', String(value));
    handle.setAttribute('aria-valuetext', `${value.toFixed(2)} seconds`);
  }
}

function setExcerptRange(start, end) {
  const total = state.sourceDuration;
  if (!total || total < 0.25) return;
  start = Math.floor(Math.max(0, Math.min(total - 0.25, start)) * 10000) / 10000;
  end = Math.max(start + 0.25, Math.min(total, start + 30, end));
  $('#excerpt-start').value = String(start);
  // Rounding must never leave a fractional file ending shorter than 0.25s.
  $('#excerpt-duration').value = String(Math.min(30, total - start, Math.max(0.25, Number((end - start).toFixed(4)))));
  updateWaveformSelection();
}

function waveformTime(clientX) {
  const bounds = $('#source-waveform').getBoundingClientRect();
  return Math.max(0, Math.min(state.sourceDuration, (clientX - bounds.left) / bounds.width * state.sourceDuration));
}

function beginExcerptDrag(event) {
  if (!state.audioReady || state.busy || state.converting || state.inspecting || event.button > 0) return;
  const start = Number($('#excerpt-start').value) || 0;
  const end = Math.min(state.sourceDuration, start + (Number($('#excerpt-duration').value) || 12));
  const time = waveformTime(event.clientX);
  const target = event.target.closest('button');
  const mode = target?.id === 'excerpt-start-handle' ? 'start'
    : target?.id === 'excerpt-end-handle' ? 'end'
    : event.shiftKey || event.target.closest('#excerpt-selection') ? 'move' : 'new';
  state.waveformDrag = { pointerId: event.pointerId, mode, time, start, end };
  $('#source-waveform').setPointerCapture(event.pointerId);
  event.preventDefault();
  if (mode === 'new') setExcerptRange(time, time + 0.25);
}

function moveExcerptDrag(event) {
  const drag = state.waveformDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const time = waveformTime(event.clientX);
  if (drag.mode === 'start') setExcerptRange(Math.max(drag.end - 30, Math.min(drag.end - 0.25, time)), drag.end);
  else if (drag.mode === 'end') setExcerptRange(drag.start, time);
  else if (drag.mode === 'move') {
    const length = Math.min(30, Math.max(0.25, drag.end - drag.start));
    const start = Math.max(0, Math.min(state.sourceDuration - length, drag.start + time - drag.time));
    setExcerptRange(start, start + length);
  } else {
    const start = Math.max(0, Math.min(drag.time, time));
    setExcerptRange(start, Math.max(drag.time, time));
  }
}

async function selectAudioFile(file) {
  state.inspectController?.abort();
  const source = $('#source-audio');
  source.pause(); state.excerptPlaying = false;
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = null; state.conversionSource = null; state.sourceDuration = null; state.sourcePeaks = [];
  state.audioReady = false; state.inspecting = false; state.inspectController = null; state.waveformDrag = null;
  source.removeAttribute('src'); source.hidden = true;
  $('#source-waveform').hidden = true;
  $('#play-excerpt').disabled = true;
  $('#conversion-results').hidden = true;
  $('#conversion-status').hidden = true;
  if (!file) { updateStudioButtons(); return; }
  if (file.size > MAX_AUDIO_FILE_BYTES || file.size === 0) {
    $('#audio-file').value = ''; updateStudioButtons();
    showError('Choose a nonempty audio file no larger than 32 MiB. Export a short excerpt for larger songs.'); return;
  }
  state.conversionSource = file;
  state.sourceUrl = URL.createObjectURL(file);
  source.src = state.sourceUrl; source.hidden = false;
  $('#audio-file-info').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB`;
  $('#excerpt-start').value = '0'; $('#excerpt-duration').value = '12';
  const controller = new AbortController();
  state.inspectController = controller; state.inspecting = true;
  const status = $('#conversion-status');
  status.hidden = false; status.textContent = 'Reading audio and drawing the waveform…';
  clearError(); updateStudioButtons();
  try {
    const info = await inspectAudio(file, { signal: controller.signal, onProgress: message => {
      if (state.inspectController === controller && !controller.signal.aborted) status.textContent = message;
    } });
    if (state.inspectController !== controller || controller.signal.aborted) return;
    sourceMetadata(info.duration); state.sourcePeaks = info.peaks;
    drawSourceWaveform(info.peaks); state.audioReady = true;
    status.textContent = 'Choose the phrase on the waveform, then convert. No audio leaves this browser.';
  } catch (error) {
    if (state.inspectController !== controller) return;
    if (controller.signal.aborted || error?.name === 'AbortError') {
      $('#audio-file').value = '';
      status.textContent = 'Audio processing stopped. Choose the file again when you are ready.';
    }
    else { status.textContent = 'This recording could not be read. Try a short WAV or MP3 clip.'; showError(error); }
  } finally {
    if (state.inspectController === controller) { state.inspectController = null; state.inspecting = false; updateStudioButtons(); }
  }
}

function excerptOptions() {
  const options = { start: Number($('#excerpt-start').value), duration: Number($('#excerpt-duration').value),
    bpm: $('#conversion-bpm').value.trim() ? Number($('#conversion-bpm').value) : undefined,
    source_mode: $('#source-mode').value, preset: $('#conversion-preset').value,
    leadPreference: $('#lead-preference').value, grid: Number($('#conversion-grid').value) };
  if (!state.conversionSource) throw new Error('Choose an audio file first.');
  if (!$('#excerpt-start').value.trim() || !Number.isFinite(options.start) || options.start < 0) throw new Error('Choose a start time of zero seconds or later.');
  if (!Number.isFinite(options.duration) || options.duration < 0.25 || options.duration > 30) throw new Error('Choose an excerpt between 0.25 and 30 seconds.');
  if (options.bpm !== undefined && (!Number.isInteger(options.bpm) || options.bpm < 25 || options.bpm > 900)) throw new Error('Tempo must be a whole number from 25 to 900 BPM.');
  if (state.sourceDuration !== null) {
    if (options.start >= state.sourceDuration) throw new Error('Start the excerpt before the end of the audio file.');
    options.duration = Math.min(options.duration, state.sourceDuration - options.start);
  }
  if (options.duration < 0.25) throw new Error('Leave at least 0.25 seconds between the excerpt start and the end of the recording.');
  return options;
}

async function playExcerpt() {
  try {
    const options = excerptOptions();
    $('#excerpt-duration').value = String(options.duration); updateWaveformSelection();
    stopPlayback();
    const source = $('#source-audio');
    source.currentTime = options.start;
    state.excerptPlaying = true;
    await source.play();
  } catch (error) { state.excerptPlaying = false; showError(error); }
}

async function convertExcerpt(event) {
  event.preventDefault();
  if (state.converting || state.busy || state.inspecting || !state.audioReady) return;
  let options;
  try { options = excerptOptions(); } catch (error) { showError(error); return; }
  $('#excerpt-duration').value = String(options.duration); updateWaveformSelection();
  const file = state.conversionSource;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(new Error('Audio processing took longer than ten minutes. Try a shorter excerpt.')), 600000);
  state.conversionController = controller; state.converting = true;
  clearError(); updateStudioButtons();
  const status = $('#conversion-status');
  status.hidden = false; status.textContent = 'Preparing browser transcription…';
  $('#conversion-results').hidden = true;
  try {
    const result = await transcribeInBrowser(file, options, { signal: controller.signal, onProgress: message => { if (!controller.signal.aborted) status.textContent = message; } });
    if (controller.signal.aborted) return;
    if (!Array.isArray(result.candidates) || !result.candidates.length) throw new Error('No melody could be detected. Try a clearer lead or a different excerpt.');
    const records = result.candidates.map((candidate, index) => {
      const parsed = parseRtttl(candidate.rtttl);
      return { id: newSongId('converted'), title: candidate.name || `Interpretation ${index + 1}`, name: parsed.name,
        rtttl: candidate.rtttl, source: 'converted', settings: { ...candidate.settings,
          conversion: { filename: file.name, ...result.analysis, score: candidate.score } } };
    });
    state.songs = appendStudioSongs(state.songs, records, MAX_SAVED_SONGS);
    selectSong(records[0]); drawLibrary(); saveLibrary();
    status.textContent = `${records.length} ${records.length === 1 ? 'interpretation added' : 'interpretations added'} to your workspace. Compare each with the original excerpt.`;
    const summary = $('#conversion-results');
    summary.replaceChildren(); summary.hidden = false;
    records.forEach((record, index) => {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'conversion-candidate';
      const score = result.candidates[index].score;
      button.textContent = `${record.title}${typeof score === 'number' && Number.isFinite(score) ? ` · source evidence ${Math.round(score * 100)}%` : ''}`;
      button.addEventListener('click', () => { selectSong(record); void togglePlayback(); });
      summary.appendChild(button);
    });
    const help = document.createElement('p'); help.className = 'field-description';
    help.textContent = 'Source evidence measures agreement with detected notes; it is not transcription accuracy.';
    summary.appendChild(help);
    for (const warning of (Array.isArray(result.warnings) ? result.warnings : [])) {
      if (typeof warning !== 'string') continue;
      const message = document.createElement('p'); message.className = 'field-description'; message.textContent = warning; summary.appendChild(message);
    }
  } catch (error) {
    if (controller.signal.aborted || error?.name === 'AbortError') {
      status.textContent = 'Audio processing stopped. Your existing melodies are kept.';
      if (controller.signal.reason instanceof Error && controller.signal.reason.name !== 'AbortError') showError(controller.signal.reason);
    } else { status.textContent = 'Conversion did not finish. Your existing melodies are kept.'; showError(error); }
  } finally {
    clearTimeout(timeout); state.conversionController = null; state.converting = false; updateStudioButtons();
  }
}

function retainSourceVersion(song) {
  // The bundled demo is playable before it is a library entry. It becomes a
  // saved source when the user derives a new version, just like an import.
  if (!state.songs.some(existing => String(existing.id) === String(song.id))) {
    state.songs = appendStudioSongs(state.songs, [{ ...song }], MAX_SAVED_SONGS);
  }
}

function saveScoreVersion(rtttl, label) {
  try {
    const parsed = parseRtttl(rtttl);
    const record = { id: newSongId('edited'), title: `${parsed.name} · ${label}`, name: parsed.name,
      rtttl: rtttl.trim(), source: 'edited', parentId: state.currentSong.id,
      selectedRange: validScoreSelection(state.selection, parsed.events.length) ? { ...state.selection } : null };
    retainSourceVersion(state.currentSong);
    state.songs = appendStudioSongs(state.songs, [record], MAX_SAVED_SONGS);
    clearError(); selectSong(record); drawLibrary(); saveLibrary();
  } catch (error) { showError(error); }
}

function editSelectedEvents(change, label) {
  if (state.busy || state.converting || state.inspecting) return;
  try { saveScoreVersion(editScoreSelection(state.currentSong.rtttl, state.selection, change), label); }
  catch (error) { showError(error); }
}

function selectedPromptSong() {
  return { name: state.parsedSong.name, bpm: state.parsedSong.bpm,
    events: state.parsedSong.events.map(({ pitch, duration, dotted }) => ({ pitch, duration, dotted })) };
}

function readSelectionInputs() {
  const rawStart = $('#selection-start').value, rawEnd = $('#selection-end').value;
  const start = Number(rawStart), end = Number(rawEnd || rawStart);
  const selection = { start: start - 1, end };
  if (validScoreSelection(selection, state.parsedSong.events.length)) selectNoteRange(selection.start, selection.end - 1);
  else {
    clearNoteSelection();
    // Preserve a partly entered range while the person moves to the second field.
    $('#selection-start').value = rawStart; $('#selection-end').value = rawEnd;
    $('#selection-summary').textContent = `Enter event numbers from 1 to ${state.parsedSong.events.length}, with the start first.`;
  }
}

$('#compose-tab').addEventListener('click', () => setStudioMode('compose'));
$('#convert-tab').addEventListener('click', () => setStudioMode('convert'));
$('.studio-modes').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const mode = event.key === 'Home' ? 'compose' : event.key === 'End' ? 'convert' : $('#convert-tab').getAttribute('aria-selected') === 'true' ? 'compose' : 'convert';
  setStudioMode(mode, true);
});
$('#audio-file').addEventListener('change', event => { void selectAudioFile(event.target.files?.[0]); });
$('#source-audio').addEventListener('loadedmetadata', () => {
  const source = $('#source-audio');
  if (!state.conversionSource || source.currentSrc !== state.sourceUrl) return;
  sourceMetadata(source.duration);
});
$('#source-audio').addEventListener('play', stopPlayback);
$('#source-audio').addEventListener('timeupdate', () => {
  if (!state.excerptPlaying) return;
  const end = Number($('#excerpt-start').value) + Number($('#excerpt-duration').value);
  if ($('#source-audio').currentTime >= end) { $('#source-audio').pause(); state.excerptPlaying = false; }
});
$('#source-audio').addEventListener('error', () => {
  if (state.conversionSource) $('#audio-file-info').textContent = `${state.conversionSource.name} · This browser cannot preview this codec. If decoding also fails, try a WAV or MP3 version.`;
});
$('#source-waveform').addEventListener('pointerdown', beginExcerptDrag);
$('#source-waveform').addEventListener('pointermove', moveExcerptDrag);
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) $('#source-waveform').addEventListener(type, () => { state.waveformDrag = null; });
for (const id of ['excerpt-start', 'excerpt-duration']) $(`#${id}`).addEventListener('input', updateWaveformSelection);
for (const id of ['excerpt-start-handle', 'excerpt-end-handle']) {
  $(`#${id}`).addEventListener('keydown', event => {
    if (!state.audioReady || state.busy || state.converting || state.inspecting || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const start = Number($('#excerpt-start').value), end = start + Number($('#excerpt-duration').value);
    const step = (event.shiftKey ? 1 : 0.1) * (event.key === 'ArrowLeft' ? -1 : 1);
    if (id === 'excerpt-start-handle') {
      const next = event.key === 'Home' ? Math.max(0, end - 30) : event.key === 'End' ? end - 0.25 : start + step;
      setExcerptRange(Math.max(end - 30, Math.min(end - 0.25, next)), end);
    } else {
      const next = event.key === 'Home' ? start + 0.25 : event.key === 'End' ? Math.min(state.sourceDuration, start + 30) : end + step;
      setExcerptRange(start, next);
    }
  });
}
$('#play-excerpt').addEventListener('click', playExcerpt);
$('#converter-form').addEventListener('submit', convertExcerpt);
$('#cancel-conversion').addEventListener('click', () => {
  state.inspectController?.abort(); state.conversionController?.abort();
  $('#conversion-status').textContent = 'Stopping audio processing… A browser decode already in progress may need to finish.';
  updateStudioButtons();
});
$('#roll-notes').addEventListener('click', event => {
  const button = event.target.closest('[data-note]');
  if (!button || button.disabled) return;
  const index = Number(button.dataset.note);
  selectNoteRange(event.shiftKey && state.selectionAnchor !== null ? state.selectionAnchor : index, index);
});
$('#roll-notes').addEventListener('keydown', event => {
  const button = event.target.closest('[data-note]');
  if (!button || button.disabled) return;
  if (event.key === 'Escape') { event.preventDefault(); clearNoteSelection(); return; }
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const index = Number(button.dataset.note), last = state.parsedSong.events.length - 1;
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? last : Math.max(0, Math.min(last, index + (event.key === 'ArrowLeft' ? -1 : 1)));
  selectNoteRange(event.shiftKey ? (state.selectionAnchor ?? index) : next, next);
  $(`#roll-notes [data-note="${next}"]`).focus();
});
for (const id of ['selection-start', 'selection-end']) $(`#${id}`).addEventListener('input', readSelectionInputs);
$('#select-all-notes').addEventListener('click', () => selectNoteRange(0, state.parsedSong.events.length - 1));
$('#clear-note-selection').addEventListener('click', clearNoteSelection);
$('#selected-pitch-down').addEventListener('click', () => editSelectedEvents({ semitones: -1 }, 'selected pitches −1'));
$('#selected-pitch-up').addEventListener('click', () => editSelectedEvents({ semitones: 1 }, 'selected pitches +1'));
$('#selected-rest').addEventListener('click', () => editSelectedEvents({ rest: true }, 'selected rests'));
$('#selected-note').addEventListener('click', () => editSelectedEvents({ restoreRests: true }, 'rests to C5'));
$('#apply-selected-duration').addEventListener('click', () => editSelectedEvents({ duration: Number($('#selected-duration').value), dotted: $('#selected-dotted').checked }, 'selected rhythm'));
for (const mode of ['variation', 'repair']) $(`#${mode === 'variation' ? 'vary' : 'repair'}-selection`).addEventListener('click', event => {
  void generate(event, { promptSong: selectedPromptSong(), editMode: mode, selection: state.selection && { ...state.selection } });
});
for (const [id, key] of [['compare-parent', 'parentId'], ['compare-version', 'versionId']]) $(`#${id}`).addEventListener('click', () => {
  const comparison = state.comparison;
  const song = comparison && state.songs.find(item => String(item.id) === String(comparison[key]));
  if (!song) return;
  const range = comparison.selection;
  selectSong(song, { preserveComparison: true });
  if (validScoreSelection(range, state.parsedSong.events.length)) selectNoteRange(range.start, range.end - 1);
  void togglePlayback();
});
$('#save-score').addEventListener('click', () => saveScoreVersion($('#score-input').value, 'edited'));
for (const [id, shift] of [['transpose-down', -1], ['transpose-up', 1]]) {
  $(`#${id}`).addEventListener('click', () => {
    try { saveScoreVersion(transposeScore($('#score-input').value, shift), `${shift > 0 ? '+' : ''}${shift} semitone`); }
    catch (error) { showError(error); }
  });
}
$('#apply-score-tempo').addEventListener('click', () => {
  try { saveScoreVersion(changeScoreTempo($('#score-input').value, Number($('#score-bpm').value)), `${$('#score-bpm').value} BPM`); }
  catch (error) { showError(error); }
});
$('#continue-song').addEventListener('click', event => {
  void generate(event, { promptSong: selectedPromptSong() });
});

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
  $('#source-audio').pause();
  state.conversionController?.abort();
  state.inspectController?.abort();
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
