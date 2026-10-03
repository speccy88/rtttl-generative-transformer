"""Optional, handcrafted musical preferences applied to sampling logits.

These profiles guide monophonic melodies; they are not learned genres,
instrument sounds, or full arrangements. All preferences are soft: chromatic
notes, wide leaps, dotted rhythm, and rests remain legal. Biases are in raw
logit units, before the caller's temperature and sampling filters.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch

from .tokenizer import EventTokenizer


PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural-minor": (0, 2, 3, 5, 7, 8, 10),
}


@dataclass(frozen=True)
class ProfileSpec:
    """Immutable, handcrafted settings, including an inclusive tempo range.

    ``duration_biases`` contains (denominator, plain bias, dotted bias).
    ``interval_biases`` contains (minimum semitones, maximum semitones, bias).
    Unisons have no interval bonus in any preset; repetition controls remain
    independent. The caller selects a tempo before generation.
    """

    label: str
    description: str
    bpm_range: tuple[int, int]
    default_mode: str
    register: tuple[int, int]
    duration_biases: tuple[tuple[int, float, float], ...]
    interval_biases: tuple[tuple[int, int, float], ...]
    scale_bias: float = 0.35
    register_bonus: float = 0.20
    register_distance_penalty: float = 0.04
    register_penalty_cap: float = 0.40
    leap_start: int = 5
    leap_penalty: float = 0.06
    leap_penalty_cap: float = 0.60
    rest_bias: float = -0.15
    consecutive_rest_penalty: float = 0.15
    consecutive_rest_penalty_cap: float = 0.45
    rest_duration_penalty: float = 0.15
    rest_duration_penalty_cap: float = 0.60


PROFILE_NAMES = ("pop-hook", "chiptune", "cinematic", "dance", "lullaby")
PROFILE_SPECS = MappingProxyType({
    "pop-hook": ProfileSpec(
        label="Pop hook",
        description="Compact melodic hooks with mostly stepwise motion and a steady rhythm.",
        bpm_range=(88, 132), default_mode="major", register=(72, 91),
        duration_biases=((1, -0.15, -0.15), (2, -0.15, -0.15), (4, 0.30, 0.0),
                         (8, 0.30, 0.0), (16, 0.10, 0.0), (32, 0.0, 0.0)),
        interval_biases=((1, 2, 0.20), (3, 4, 0.10)),
    ),
    "chiptune": ProfileSpec(
        label="Chiptune",
        description="Bright, quick notes and wider jumps inspired by video game melodies.",
        bpm_range=(120, 180), default_mode="major", register=(79, 100),
        duration_biases=((1, -0.40, -0.40), (2, -0.30, -0.30), (4, 0.0, -0.10),
                         (8, 0.25, 0.0), (16, 0.40, 0.05), (32, 0.15, -0.05)),
        interval_biases=((1, 2, 0.05), (3, 4, 0.10), (5, 12, 0.30)),
        register_bonus=0.25, leap_start=12, leap_penalty=0.04,
        rest_bias=-0.18, rest_duration_penalty=0.20,
    ),
    "cinematic": ProfileSpec(
        label="Cinematic",
        description="Lower, sustained phrases with more breathing room and a minor-key default.",
        bpm_range=(60, 104), default_mode="natural-minor", register=(60, 79),
        duration_biases=((1, 0.15, 0.10), (2, 0.40, 0.35), (4, 0.15, 0.20),
                         (8, -0.15, -0.10), (16, -0.40, -0.35), (32, -0.45, -0.40)),
        interval_biases=((1, 2, 0.10), (3, 4, 0.15), (5, 7, 0.08)),
        leap_start=7, rest_bias=0.12, consecutive_rest_penalty=0.12,
        rest_duration_penalty=0.04,
    ),
    "dance": ProfileSpec(
        label="Dance",
        description="Brisk, even rhythmic motion with shorter gaps between notes.",
        bpm_range=(118, 150), default_mode="major", register=(69, 88),
        duration_biases=((1, -0.35, -0.40), (2, -0.25, -0.30), (4, 0.30, -0.15),
                         (8, 0.35, -0.15), (16, 0.15, -0.15), (32, -0.10, -0.20)),
        interval_biases=((1, 2, 0.18), (3, 4, 0.10)),
        leap_start=7, rest_bias=-0.35, rest_duration_penalty=0.25,
    ),
    "lullaby": ProfileSpec(
        label="Lullaby",
        description="Gentle, slower phrases with small melodic steps and longer notes.",
        bpm_range=(60, 84), default_mode="major", register=(60, 79),
        duration_biases=((1, 0.05, 0.0), (2, 0.35, 0.20), (4, 0.30, 0.25),
                         (8, 0.0, 0.05), (16, -0.30, -0.25), (32, -0.40, -0.35)),
        interval_biases=((1, 2, 0.35), (3, 4, 0.10)),
        register_bonus=0.25, leap_start=4, leap_penalty=0.10,
        rest_bias=0.02, rest_duration_penalty=0.08,
    ),
})


def normalize_tonic(tonic: str) -> str:
    """Return a canonical sharp spelling, accepting single sharps or flats."""
    if not isinstance(tonic, str):
        raise ValueError("tonic must be a note name such as C, F#, or Bb")
    name = tonic.strip().replace("♯", "#").replace("♭", "b")
    natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if not name or name[0].upper() not in natural or name[1:] not in ("", "#", "b"):
        raise ValueError("tonic must be a note name such as C, F#, or Bb")
    accidental = {"": 0, "#": 1, "b": -1}[name[1:]]
    return PITCH_CLASS_NAMES[(natural[name[0].upper()] + accidental) % 12]


class MelodyGuidance:
    """Bounded musical preferences that leave structural tokens unchanged."""

    def __init__(self, tokenizer: EventTokenizer, tonic: str, mode: str,
                 spec: ProfileSpec) -> None:
        self.tonic = normalize_tonic(tonic)
        if not isinstance(mode, str) or mode not in SCALE_INTERVALS:
            raise ValueError("mode must be 'major' or 'natural-minor'")
        self.mode = mode
        self.spec = spec
        self.tempo_range = spec.bpm_range
        self.tokenizer = tokenizer
        root = PITCH_CLASS_NAMES.index(self.tonic)
        self.scale = frozenset((root + interval) % 12 for interval in SCALE_INTERVALS[mode])
        self.pitch_by_id = dict(zip(tokenizer.pitch_ids,
                                    range(tokenizer.MIN_PITCH, tokenizer.MAX_PITCH + 1)))
        self.duration_by_id = {
            tokenizer.token_to_id[f"<DUR_{duration}_{suffix}>"]: (duration, suffix == "dot")
            for duration in tokenizer.DURATIONS for suffix in ("plain", "dot")
        }
        self.duration_biases = {
            (duration, dotted): bias
            for duration, plain, dot in spec.duration_biases
            for dotted, bias in ((False, plain), (True, dot))
        }

    def apply(self, logits: torch.Tensor, allowed_ids: list[int],
              pitches: list[int]) -> torch.Tensor:
        """Adjust only legal pitch/rest/duration scores without mutating input.

        ``pitches`` is the full pitch/rest history. At duration decisions it
        includes the current event start, allowing long-rest discouragement.
        Structural scores, including EOS, are copied without adjustment.
        """
        scores = logits.detach().float().cpu().clone()
        spec = self.spec
        previous = next((self.pitch_by_id[token] for token in reversed(pitches)
                         if token in self.pitch_by_id), None)
        trailing_rests = 0
        for token in reversed(pitches):
            if token != self.tokenizer.rest_id:
                break
            trailing_rests += 1
        current_rest = bool(pitches and pitches[-1] == self.tokenizer.rest_id)
        for token in allowed_ids:
            if token in self.pitch_by_id:
                pitch = self.pitch_by_id[token]
                bias = spec.scale_bias if pitch % 12 in self.scale else -spec.scale_bias
                low, high = spec.register
                if low <= pitch <= high:
                    bias += spec.register_bonus
                else:
                    distance = low - pitch if pitch < low else pitch - high
                    bias -= min(spec.register_penalty_cap, spec.register_distance_penalty * distance)
                if previous is not None:
                    distance = abs(pitch - previous)
                    for minimum, maximum, interval_bias in spec.interval_biases:
                        if minimum <= distance <= maximum:
                            bias += interval_bias
                            break
                    if distance > spec.leap_start:
                        bias -= min(spec.leap_penalty_cap,
                                    spec.leap_penalty * (distance - spec.leap_start))
                scores[token] += bias
            elif token == self.tokenizer.rest_id:
                # Keep this subtraction order for exact replay of the original
                # pop-hook preferences, including float rounding.
                scores[token] -= -spec.rest_bias + min(spec.consecutive_rest_penalty_cap,
                                                      spec.consecutive_rest_penalty * trailing_rests)
            elif token in self.duration_by_id:
                duration, dotted = self.duration_by_id[token]
                bias = self.duration_biases[(duration, dotted)]
                if current_rest:
                    beats = 4 / duration * (1.5 if dotted else 1)
                    bias -= min(spec.rest_duration_penalty_cap, spec.rest_duration_penalty * beats)
                scores[token] += bias
        return scores


class PopHookGuidance(MelodyGuidance):
    """Compatible entry point for the original pop-hook bias settings.

    ``default_bpm`` is retained for older callers. New callers should select
    a seeded tempo from ``tempo_range`` or honor the user's explicit BPM.
    """

    default_bpm = 112

    def __init__(self, tokenizer: EventTokenizer, tonic: str, mode: str) -> None:
        super().__init__(tokenizer, tonic, mode, PROFILE_SPECS["pop-hook"])


def make_guidance(tokenizer: EventTokenizer, profile: str | None = None,
                  tonic: str = "C", mode: str | None = None) -> MelodyGuidance | None:
    """Validate options and resolve an omitted mode from the selected profile.

    A ``None`` profile leaves the learned distribution unchanged. An omitted
    mode defaults to major when no profile is selected.
    """
    if profile is not None and (not isinstance(profile, str) or profile not in PROFILE_SPECS):
        raise ValueError(f"profile must be None or one of {', '.join(PROFILE_NAMES)}")
    canonical_tonic = normalize_tonic(tonic)
    if mode is None:
        mode = PROFILE_SPECS[profile].default_mode if profile is not None else "major"
    if not isinstance(mode, str) or mode not in SCALE_INTERVALS:
        raise ValueError("mode must be 'major' or 'natural-minor'")
    if profile is None:
        return None
    if profile == "pop-hook":
        return PopHookGuidance(tokenizer, canonical_tonic, mode)
    return MelodyGuidance(tokenizer, canonical_tonic, mode, PROFILE_SPECS[profile])
