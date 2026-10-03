"""RTTTL parsing with explicit, validated musical events.

The project uses scientific pitch notation (MIDI 60 = C4), octaves 4--7,
whole-note denominators 1, 2, 4, 8, 16, 32, and at most one dot.  Missing
defaults resolve to d=4,o=6,b=63.  Historical players disagree on a missing
octave; the chosen o=6 is recorded as a warning and always emitted explicitly.
No invalid note is skipped and no unknown setting is silently ignored.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

DURATIONS = (1, 2, 4, 8, 16, 32)
MIN_PITCH = 60
MAX_PITCH = 107
MIN_BPM = 25
MAX_BPM = 900
PITCH_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")
_NATURAL = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


class RTTTLParseError(ValueError):
    """A stable error code plus human-readable evidence for the audit manifest."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class Event:
    pitch: int | None
    duration: int
    dotted: bool = False

    def __post_init__(self) -> None:
        if self.pitch is not None and (type(self.pitch) is not int or not MIN_PITCH <= self.pitch <= MAX_PITCH):
            raise ValueError(f"pitch must be None or MIDI {MIN_PITCH}..{MAX_PITCH}")
        if type(self.duration) is not int or self.duration not in DURATIONS:
            raise ValueError(f"duration must be one of {DURATIONS}")
        if type(self.dotted) is not bool:
            raise ValueError("dotted must be bool")

    @property
    def beats(self) -> float:
        """Length in quarter-note beats, before tempo is applied."""
        return 4.0 / self.duration * (1.5 if self.dotted else 1.0)


@dataclass(frozen=True)
class Song:
    name: str
    bpm: int
    events: tuple[Event, ...]
    default_duration: int = 4
    default_octave: int = 6
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.bpm) is not int or not MIN_BPM <= self.bpm <= MAX_BPM:
            raise ValueError(f"bpm must be an integer {MIN_BPM}..{MAX_BPM}")
        if self.default_duration not in DURATIONS:
            raise ValueError("invalid default duration")
        if type(self.default_octave) is not int or not 4 <= self.default_octave <= 7:
            raise ValueError("invalid default octave")
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("a song needs a nonempty tuple of events")
        if not all(isinstance(event, Event) for event in self.events):
            raise ValueError("events must contain Event objects")


def parse_rtttl(text: str, *, underscore_is_sharp: bool = False) -> Song:
    """Parse one entire record; safe nonstandard spellings generate warnings.

    ``underscore_is_sharp`` is deliberately opt-in: the supplied rtttl3 folder
    contains this convention, confirmed by matching copies with literal '#'.
    The generic parser will otherwise reject '_' as an ambiguous accidental.
    """
    if not isinstance(text, str):
        raise TypeError("RTTTL input must be text")
    warnings: list[str] = []

    def warn(code: str) -> None:
        if code not in warnings:
            warnings.append(code)

    text = text.strip().lstrip("\ufeff")
    if not text:
        raise RTTTLParseError("empty_record", "record contains no RTTTL")
    # The last two separators isolate defaults/melody, allowing ':' in a name.
    if text.count(":") < 2:
        raise RTTTLParseError("missing_sections", "expected name:defaults:melody")
    name, defaults, melody = text.rsplit(":", 2)
    name = name.strip()
    if ":" in name:
        # Another RTTTL header inside the name is not a name-only colon.
        if re.search(r"[dob]\s*=", name, re.I):
            raise RTTTLParseError("concatenated_records", "multiple RTTTL headers in one record")
        warn("colon_in_name")
    if not name:
        name = "Untitled"
        warn("empty_name_replaced")
    if len(name) > 11:
        warn("long_name_preserved_in_metadata")
    if not name.isascii():
        warn("unicode_name_preserved_in_metadata")
    values = {"d": 4, "o": 6, "b": 63}
    found: set[str] = set()
    if defaults.strip():
        for setting in defaults.split(","):
            m = re.fullmatch(r"\s*([a-zA-Z0-9]+)\s*=\s*(.*?)\s*", setting)
            if not m:
                raise RTTTLParseError("malformed_default", repr(setting))
            key, value = m.group(1).lower(), m.group(2)
            if key not in values:
                raise RTTTLParseError("unsupported_default", f"{key}={value}")
            if key in found:
                raise RTTTLParseError("duplicate_default", key)
            found.add(key)
            if key == "b" and re.fullmatch(r"\d+\s*bpm", value, re.I):
                value = re.sub(r"\s*bpm$", "", value, flags=re.I)
                warn("bpm_suffix_removed")
            if not re.fullmatch(r"\d+", value):
                raise RTTTLParseError("nonnumeric_default", f"{key}={value}")
            values[key] = int(value)
    for key in ("d", "o", "b"):
        if key not in found:
            warn(f"missing_default_{key}_resolved_{values[key]}")
    if values["d"] not in DURATIONS:
        raise RTTTLParseError("invalid_default_duration", str(values["d"]))
    if not 4 <= values["o"] <= 7:
        raise RTTTLParseError("invalid_default_octave", str(values["o"]))
    if not MIN_BPM <= values["b"] <= MAX_BPM:
        raise RTTTLParseError("invalid_bpm", str(values["b"]))
    if any(c.isspace() for c in melody.strip()):
        warn("melody_whitespace_removed")
    melody = re.sub(r"\s+", "", melody).lower()
    if melody.endswith(","):
        melody = melody.rstrip(",")
        warn("trailing_comma_removed")
    if not melody:
        raise RTTTLParseError("empty_melody", "no events")
    events: list[Event] = []
    for index, original in enumerate(melody.split(","), 1):
        token = original
        if not token:
            raise RTTTLParseError("empty_event", f"event {index}")
        if "_" in token:
            if not underscore_is_sharp:
                raise RTTTLParseError("ambiguous_underscore", f"event {index}: {original}")
            token = token.replace("_", "#")
            warn("underscore_interpreted_as_sharp")
        if token.count(".") > 1:
            raise RTTTLParseError("multiple_dots", f"event {index}: {original}")
        dotted = "." in token
        if dotted and not token.endswith("."):
            warn("dot_position_normalized")
        token = token.replace(".", "")
        # Prefix accidentals and accidentals after octave have one clear meaning.
        if re.fullmatch(r"\d*#[a-g]\d*", token):
            token = re.sub(r"^(\d*)#([a-g])", r"\1\2#", token)
            warn("prefix_sharp_normalized")
        if re.fullmatch(r"\d*[a-g]\d+#", token):
            token = re.sub(r"([a-g])(\d+)#$", r"\1#\2", token)
            warn("post_octave_sharp_normalized")
        m = re.fullmatch(r"(\d*)([a-gp])(#?)(\d*)", token)
        if not m:
            raise RTTTLParseError("invalid_note_syntax", f"event {index}: {original}")
        duration_s, note, sharp, octave_s = m.groups()
        duration = int(duration_s) if duration_s else values["d"]
        if duration not in DURATIONS:
            raise RTTTLParseError("invalid_duration", f"event {index}: {original}")
        if note == "p":
            if sharp:
                raise RTTTLParseError("sharp_rest", f"event {index}: {original}")
            if octave_s:
                if not 4 <= int(octave_s) <= 7:
                    raise RTTTLParseError("invalid_octave", f"event {index}: {original}")
                warn("rest_octave_ignored")
            pitch = None
        else:
            octave = int(octave_s) if octave_s else values["o"]
            if not 4 <= octave <= 7:
                raise RTTTLParseError("invalid_octave", f"event {index}: {original}")
            pitch = 12 * (octave + 1) + _NATURAL[note] + bool(sharp)
            if pitch > MAX_PITCH:
                raise RTTTLParseError("pitch_out_of_range", f"event {index}: {original}")
            if sharp and note in ("b", "e"):
                warn("enharmonic_sharp_normalized")
        events.append(Event(pitch, duration, dotted))
    return Song(name, values["b"], tuple(events), values["d"], values["o"], tuple(warnings))


def encode_rtttl(song: Song, *, optimize_defaults: bool = True) -> str:
    """Produce ordinary RTTTL with explicit defaults and a <=11-character name."""
    # Revalidate potentially reconstructed objects before claiming validity.
    Song(song.name, song.bpm, song.events, song.default_duration, song.default_octave)
    if optimize_defaults:
        d = Counter(event.duration for event in song.events).most_common(1)[0][0]
        octaves = Counter(event.pitch // 12 - 1 for event in song.events if event.pitch is not None)
        o = octaves.most_common(1)[0][0] if octaves else 6
    else:
        d, o = song.default_duration, song.default_octave
    name = unicodedata.normalize("NFKD", song.name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()[:11] or "Generated"
    notes: list[str] = []
    for event in song.events:
        token = str(event.duration) if event.duration != d else ""
        if event.pitch is None:
            token += "p"
        else:
            octave, pitch_class = divmod(event.pitch, 12)
            token += PITCH_NAMES[pitch_class]
            if octave - 1 != o:
                token += str(octave - 1)
        if event.dotted:
            token += "."
        notes.append(token)
    return f"{name}:d={d},o={o},b={song.bpm}:" + ",".join(notes)


def event_to_dict(event: Event) -> dict[str, Any]:
    return {"pitch": event.pitch, "duration": event.duration, "dotted": event.dotted}


def song_to_dict(song: Song) -> dict[str, Any]:
    return {"name": song.name, "bpm": song.bpm,
            "events": [event_to_dict(event) for event in song.events],
            "default_duration": song.default_duration, "default_octave": song.default_octave,
            "warnings": list(song.warnings)}


def song_from_dict(data: dict[str, Any]) -> Song:
    return Song(name=data["name"], bpm=data["bpm"],
                events=tuple(Event(**event) for event in data["events"]),
                default_duration=data.get("default_duration", 4),
                default_octave=data.get("default_octave", 6),
                warnings=tuple(data.get("warnings", ())))
