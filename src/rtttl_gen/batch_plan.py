"""Reproducible per-song musical settings, separate from note sampling.

Profiles describe handcrafted sampling preferences, not learned genre labels.
The mixed plan visits each profile once per shuffled cycle and varies keys.
Its prefix is stable when a caller asks for a larger batch with the same seed.
"""
from __future__ import annotations

from collections.abc import Iterator
import random

from .guidance import (PITCH_CLASS_NAMES, PROFILE_NAMES, PROFILE_SPECS,
                       SCALE_INTERVALS, normalize_tonic)


def _shuffled_cycles(values: tuple[str, ...], rng: random.Random) -> Iterator[str]:
    while True:
        cycle = list(values)
        rng.shuffle(cycle)
        yield from cycle


def _validate_bpm(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 25 <= value <= 900:
        raise ValueError("BPM must be an integer from 25 to 900")


def plan_batch(num_songs: int, *, seed: int = 42, profile: str | None = None,
               bpm: int | None = None, bpm_range: tuple[int, int] | None = None,
               tonic: str | None = None, mode: str | None = None) -> list[dict]:
    """Resolve independent settings for each song before running inference.

    An explicit BPM overrides a profile's natural tempo range. A custom range
    does the same, but cannot be supplied together with an explicit BPM. With
    no profile or tempo option, BPM remains ``None`` for learned sampling.
    Explicit keys and modes are always honored. Mixed batches balance profiles
    and keys; a single named profile defaults to C and its preferred mode.

    Local, separate random generators keep planning out of both the global
    random state and note sampling. Changing a tempo option does not change
    the mixed profile/key sequence. Callers retain ``seed + song_index`` for
    note generation.
    """
    if isinstance(num_songs, bool) or not isinstance(num_songs, int) or num_songs < 1:
        raise ValueError("num_songs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if profile is not None and (not isinstance(profile, str)
                                or profile not in (*PROFILE_NAMES, "mixed")):
        raise ValueError("profile must be None, 'mixed', or one of: "
                         + ", ".join(PROFILE_NAMES))
    if bpm is not None and bpm_range is not None:
        raise ValueError("bpm and bpm_range cannot be supplied together")
    if bpm is not None:
        _validate_bpm(bpm)
    if bpm_range is not None:
        if not isinstance(bpm_range, (tuple, list)) or len(bpm_range) != 2:
            raise ValueError("bpm_range must contain a minimum and maximum BPM")
        for boundary in bpm_range:
            _validate_bpm(boundary)
        if bpm_range[0] > bpm_range[1]:
            raise ValueError("bpm_range minimum must not exceed its maximum")
    canonical_tonic = normalize_tonic(tonic) if tonic is not None else None
    if mode is not None and (not isinstance(mode, str) or mode not in SCALE_INTERVALS):
        raise ValueError("mode must be 'major' or 'natural-minor'")

    profiles = _shuffled_cycles(PROFILE_NAMES, random.Random(f"{seed}:profiles"))
    keys = _shuffled_cycles(PITCH_CLASS_NAMES, random.Random(f"{seed}:keys"))
    tempo_rng = random.Random(f"{seed}:tempos")
    plan = []
    for _ in range(num_songs):
        selected_profile = next(profiles) if profile == "mixed" else profile
        spec = PROFILE_SPECS[selected_profile] if selected_profile is not None else None
        selected_tonic = canonical_tonic or (next(keys) if profile == "mixed" else "C")
        selected_mode = mode or (spec.default_mode if spec is not None else "major")
        if bpm is not None:
            selected_bpm, tempo_source = bpm, "explicit"
        elif bpm_range is not None:
            selected_bpm, tempo_source = tempo_rng.randint(*bpm_range), "range"
        elif spec is not None:
            selected_bpm, tempo_source = tempo_rng.randint(*spec.bpm_range), "profile_range"
        else:
            selected_bpm, tempo_source = None, "model"
        plan.append({
            "profile": selected_profile,
            "bpm": selected_bpm,
            "tonic": selected_tonic,
            "mode": selected_mode,
            "tempo_source": tempo_source,
            "profile_source": "mixed" if profile == "mixed" else
                              "explicit" if profile is not None else "none",
        })
    return plan
