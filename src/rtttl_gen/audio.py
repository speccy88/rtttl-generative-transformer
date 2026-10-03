"""Optional monophonic listening utility: NumPy plus the standard library."""
from __future__ import annotations

import html
import json
from pathlib import Path
import wave
from typing import Sequence

import numpy as np

from .rtttl import Song, parse_rtttl


def render_wav(song: Song, path: str | Path, sample_rate: int = 22050, max_seconds: float = 300) -> dict:
    """Write a gentle sine ringtone, mono PCM16, at RTTTL's quarter-note BPM.

    Each pitched event uses 92% of its duration followed by a brief articulation
    gap. Attack/release ramps avoid clicks. This is an audition synthesizer, not
    a fidelity reference for a particular phone. Existing files are protected.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if sample_rate < 8000 or max_seconds <= 0:
        raise ValueError("sample_rate must be >=8000 and max_seconds positive")
    durations = [240 / song.bpm / e.duration * (1.5 if e.dotted else 1) for e in song.events]
    total_seconds = sum(durations)
    if total_seconds > max_seconds:
        raise ValueError(f"Song duration {total_seconds:.1f}s exceeds --max-seconds {max_seconds}; increase explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for event, seconds in zip(song.events, durations):
            size = max(1, round(seconds * sample_rate))
            signal = np.zeros(size, dtype=np.float64)
            if event.pitch is not None:
                active = max(1, round(size * 0.92))
                times = np.arange(active, dtype=np.float64) / sample_rate
                frequency = 440 * 2 ** ((event.pitch - 69) / 12)
                signal[:active] = 0.18 * np.sin(2 * np.pi * frequency * times)
                ramp = min(round(sample_rate * 0.008), active // 2)
                if ramp:
                    signal[:ramp] *= np.linspace(0, 1, ramp)
                    signal[active - ramp:active] *= np.linspace(1, 0, ramp)
            wav.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
            samples_written += size
    return {"path": path.name, "seconds": samples_written / sample_rate, "sample_rate": sample_rate, "channels": 1, "format": "PCM16"}


def create_listening_batch(records: Sequence[dict], output_dir: Path | str, sample_rate: int = 22050, max_seconds: float = 300) -> list[dict]:
    """Render valid RTTTL records and create a self-contained local HTML player."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, cards = [], []
    for i, record in enumerate(records, 1):
        if not record.get("valid") or not record.get("rtttl"):
            continue
        song = parse_rtttl(record["rtttl"])
        filename = f"sample_{i:04d}.wav"
        try:
            metadata = render_wav(song, output_dir / filename, sample_rate, max_seconds)
            metadata["sample_id"] = record.get("id", i)
            metadata["name"] = song.name
            manifest.append(metadata)
            audio = f'<audio controls preload="none" src="{filename}"></audio>'
        except ValueError as exc:
            metadata = {"sample_id": record.get("id", i), "name": song.name, "error": str(exc)}
            manifest.append(metadata)
            audio = f'<p class="warning">Audio skipped: {html.escape(str(exc))}</p>'
        similarity = record.get("similarity", {})
        label = similarity.get("label", "not evaluated")
        score = similarity.get("score")
        closest = similarity.get("closest_name", "—")
        flags = ", ".join(record.get("degeneracy_flags", [])) or "none"
        cards.append(f'<section><h2>{i}. {html.escape(song.name)}</h2>{audio}<p>{len(song.events)} events · {song.bpm} BPM · {html.escape(str(label))} (score {score})</p><p>Nearest training song: {html.escape(str(closest))}. Review flags: {html.escape(flags)}.</p><details><summary>RTTTL</summary><pre>{html.escape(record["rtttl"])}</pre></details></section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RTTTL listening batch</title><style>body{font:16px system-ui;max-width:960px;margin:40px auto;padding:0 20px;background:#f5f7fa;color:#1b2638}section{background:white;padding:20px;margin:16px 0;border:1px solid #dde3ec;border-radius:10px}h2{font-size:19px}audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{color:#8a3700}</style><h1>RTTTL listening batch</h1><p>Local WAV previews. Similarity scores are heuristic and do not establish originality. Neural smoke checkpoints demonstrate the pipeline, not musical quality.</p>' + "".join(cards) + '</html>'
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "audio_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
