"""Pitch transposition; rhythm, tempo, and rest positions are unchanged."""
from __future__ import annotations

from dataclasses import replace
import random

from .rtttl import Song


def legal_transpositions(song: Song, low: int = -5, high: int = 6,
                         min_pitch: int = 60, max_pitch: int = 107,
                         *, max_semitones: int | None = None) -> list[int]:
    if max_semitones is not None:
        if not isinstance(max_semitones, int) or max_semitones < 0:
            raise ValueError("max_semitones must be a non-negative integer")
        low, high = -max_semitones, max_semitones
    if low > high:
        raise ValueError("Transposition lower bound exceeds upper bound")
    pitches = [event.pitch for event in song.events if event.pitch is not None]
    if not pitches:
        return [0]  # shifting silence is pointless
    return [shift for shift in range(low, high + 1)
            if min(pitches) + shift >= min_pitch and max(pitches) + shift <= max_pitch]


def transpose(song: Song, semitones: int, min_pitch: int = 60, max_pitch: int = 107) -> Song:
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise ValueError("semitones must be an integer")
    if any(e.pitch is not None and not min_pitch <= e.pitch + semitones <= max_pitch for e in song.events):
        raise ValueError("Transposition would leave the legal pitch range")
    events = tuple(replace(e, pitch=None if e.pitch is None else e.pitch + semitones)
                   for e in song.events)
    return replace(song, events=events)


def augment_song(song: Song, rng: random.Random, split: str = "train",
                 max_semitones: int = 5) -> Song:
    """Seeded augmentation guarded against accidental held-out use."""
    if split != "train":
        raise ValueError("Augmentation is restricted to the training split")
    shifts = legal_transpositions(song, max_semitones=max_semitones)
    return transpose(song, rng.choice(shifts))
