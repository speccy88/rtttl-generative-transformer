from pathlib import Path
import zipfile
import pytest

from rtttl_gen.rtttl import (
    Event, Song, RTTTLParseError, parse_rtttl, encode_rtttl,
    song_to_dict, song_from_dict, DURATIONS,
)
from rtttl_gen.corpus import iter_records, source_inventory, inspect_corpus


def test_defaults_sharps_rests_explicit_octave_and_dots():
    song = parse_rtttl("Demo:d=8,o=5,b=120:c,c#,p,4c6,16d.6,e6.")
    assert song.events == (Event(72, 8), Event(73, 8), Event(None, 8),
                           Event(84, 4), Event(86, 16, True), Event(88, 8, True))
    assert song.bpm == 120
    assert song.default_duration == 8 and song.default_octave == 5
    assert "dot_position_normalized" in song.warnings


def test_missing_defaults_are_explicit():
    song = parse_rtttl("Demo::c")
    assert song.events == (Event(84, 4),)
    assert song.bpm == 63
    assert "missing_default_o_resolved_6" in song.warnings


def test_whitespace_wrapped_notes_and_trailing_comma():
    song = parse_rtttl("Demo: d = 4, o=5, b=120 : 8c,8\nf#,p, ")
    assert song.events == (Event(72, 8), Event(78, 8), Event(None, 4))
    assert set(song.warnings) == {"melody_whitespace_removed", "trailing_comma_removed"}


def test_safe_variant_normalization():
    song = parse_rtttl("Long song name: :d=4,o=5,b=120BPM:8.#f5,8d5#,b#5,e#5,p5")
    assert song.events == (Event(78, 8, True), Event(75, 8), Event(84, 4), Event(77, 4), Event(None, 4))
    assert "prefix_sharp_normalized" in song.warnings
    assert "rest_octave_ignored" in song.warnings


def test_underscore_requires_explicit_convention():
    with pytest.raises(RTTTLParseError, match="ambiguous_underscore"):
        parse_rtttl("Song:d=4,o=5,b=120:c_")
    song = parse_rtttl("Song:d=4,o=5,b=120:c_", underscore_is_sharp=True)
    assert song.events == (Event(73, 4),)
    assert "underscore_interpreted_as_sharp" in song.warnings


@pytest.mark.parametrize("raw,code", [
    ("", "empty_record"), ("Song:cdef", "missing_sections"),
    ("Song:d=5,o=5,b=120:c", "invalid_default_duration"),
    ("Song:d=4,o=3,b=120:c", "invalid_default_octave"),
    ("Song:d=4,o=5,b=0:c", "invalid_bpm"),
    ("Song:d=4,o=5,b=Slow:c", "nonnumeric_default"),
    ("Song:d=4,o=5,b=120,l=15:c", "unsupported_default"),
    ("Song:d=4,d=8,o=5,b=120:c", "duplicate_default"),
    ("Song:d=4,o=5,b=120:", "empty_melody"),
    ("Song:d=4,o=5,b=120:c,,d", "empty_event"),
    ("Song:d=4,o=5,b=120:6c", "invalid_duration"),
    ("Song:d=4,o=5,b=120:c3", "invalid_octave"),
    ("Song:d=4,o=5,b=120:c..", "multiple_dots"),
    ("Song:d=4,o=5,b=120:h", "invalid_note_syntax"),
    ("Song:d=4,o=5,b=120:p#", "sharp_rest"),
    ("Song:d=4,o=5,b=120:b#7", "pitch_out_of_range"),
    ("Song:d=4,o=5,b=120:cOther:d=4,o=5,b=120:c", "concatenated_records"),
])
def test_failures_are_not_silently_discarded(raw, code):
    with pytest.raises(RTTTLParseError) as error:
        parse_rtttl(raw)
    assert error.value.code == code


def test_roundtrip_every_legal_event():
    events = tuple(Event(pitch, duration, dotted)
                   for pitch in [None] + list(range(60, 108))
                   for duration in DURATIONS for dotted in (False, True))
    song = Song("A long accented mélodie", 137, events)
    encoded = encode_rtttl(song)
    assert len(encoded.split(":")[0]) <= 11
    assert encoded.isascii()
    decoded = parse_rtttl(encoded)
    assert decoded.events == song.events and decoded.bpm == song.bpm
    assert encode_rtttl(decoded) == encoded
    assert song_from_dict(song_to_dict(song)) == song


def test_equivalent_spellings_canonicalize_identically():
    first = parse_rtttl("A:d=4,o=5,b=120:8c,8d.6,4p,4e#")
    second = parse_rtttl("B:d=8,o=6,b=120:c5,d6.,4p6,4f5")
    assert first.events == second.events


def test_archive_inventory_compilation_wrapping_and_rejections(tmp_path):
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("rtttl3/", "")
        handle.writestr("rtttl3/two.txt", "A:d=4,o=5,b=120:c_,\n8d\nB:d=4,o=5,b=120:c\n")
        handle.writestr("bad.txt", "Bad:d=4,o=5,b=120:6c")
        handle.writestr("empty", "")
        handle.writestr("alternate.bas", "tune 0, 4,($E0)")
        handle.writestr("__MACOSX/._song.txt", b"metadata")
    before = archive.read_bytes()
    inventory = source_inventory(archive)
    assert len(inventory) == 6
    records = list(iter_records(archive))
    assert len(records) == 5
    good, audit, stats = inspect_corpus(archive)
    assert len(good) == 2
    assert stats["valid_rtttl_records"] == 2 and stats["invalid_rtttl_records"] == 1
    assert stats["rejected_non_rtttl_files"] == 2
    assert stats["inventory"]["metadata_files"] == 1
    assert stats["musical_events"] == 3
    assert any("underscore_interpreted_as_sharp" in row["warnings"] for row in audit)
    assert any(row.get("error") == "invalid_duration" and "raw" in row for row in audit)
    assert archive.read_bytes() == before
