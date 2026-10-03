"""Independent edge-case and external-format-contract checks.

The strict decoder below is deliberately independent of the project's parser:
it accepts only the canonical RTTTL grammar emitted by the encoder.
"""
import re
from fractions import Fraction

import pytest

from rtttl_gen.rtttl import Event, Song, encode_rtttl, parse_rtttl
from rtttl_gen.augment import transpose, legal_transpositions


def _strict_decode(encoded):
    title, header, melody = encoded.split(":")
    assert 1 <= len(title) <= 11 and title.isascii()
    match = re.fullmatch(r"d=(1|2|4|8|16|32),o=([4-7]),b=(\d+)", header)
    assert match, header
    default_duration, default_octave, bpm = map(int, match.groups())
    assert 25 <= bpm <= 900
    pitch_offsets = {"c": 0, "c#": 1, "d": 2, "d#": 3, "e": 4, "f": 5,
                     "f#": 6, "g": 7, "g#": 8, "a": 9, "a#": 10, "b": 11}
    events = []
    for note in melody.split(","):
        match = re.fullmatch(r"(1|2|4|8|16|32)?(c#?|d#?|e|f#?|g#?|a#?|b|p)([4-7])?(\.)?", note)
        assert match, note
        duration, pitch, octave, dot = match.groups()
        denominator = int(duration) if duration else default_duration
        if pitch == "p":
            assert octave is None
            midi = None
        else:
            midi = 12 * (1 + (int(octave) if octave else default_octave)) + pitch_offsets[pitch]
        events.append((midi, denominator, bool(dot)))
    return bpm, events


@pytest.mark.parametrize("compact", [False, True])
def test_all_588_legal_events_emit_independently_decodable_rtttl(compact):
    events = tuple(Event(pitch, duration, dot)
                   for pitch in [None, *range(60, 108)]
                   for duration in (1, 2, 4, 8, 16, 32)
                   for dot in (False, True))
    original = Song("Invalid: title ♥ 中文 longer than ten", 127, events)
    encoded = encode_rtttl(original, optimize_defaults=compact)
    bpm, decoded = _strict_decode(encoded)
    assert bpm == 127
    assert decoded == [(event.pitch, event.duration, event.dotted) for event in events]


def test_corpus_underscore_sharp_convention_is_explicit_regression():
    # Evidence of the corpus-specific convention: evidence/independent_corpus_review.json.
    # Other RTTTL dialects sometimes use '_' for flats; these sources use sharps.
    song = parse_rtttl("dialect:d=8,o=5,b=125:c_,d_,f_,g_,a_", underscore_is_sharp=True)
    assert [event.pitch for event in song.events] == [73, 75, 78, 80, 82]
    assert song.events == parse_rtttl("canonical:d=8,o=5,b=125:c#,d#,f#,g#,a#").events


def test_augmentation_at_both_pitch_limits_cannot_escape_domain():
    song = Song("edges", 120, (Event(60, 8), Event(None, 16, True), Event(107, 2)))
    assert legal_transpositions(song, -12, 12) == [0]
    assert transpose(song, 0) == song
    for shift in [-1, 1]:
        with pytest.raises(ValueError):
            transpose(song, shift)


def test_dotted_timing_is_exactly_one_and_half():
    for denominator in (1, 2, 4, 8, 16, 32):
        normal, dotted = Event(69, denominator), Event(69, denominator, True)
        assert Fraction(dotted.beats) / Fraction(normal.beats) == Fraction(3, 2)
    # A quarter note at 120 BPM is 0.5 s, dotted quarter is 0.75 s.
    assert Event(69, 4, True).beats * 60 / 120 == 0.75
