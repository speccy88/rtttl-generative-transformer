from dataclasses import replace
import random

import pytest

from rtttl_gen.augment import augment_song, legal_transpositions, transpose
from rtttl_gen.deduplicate import family_fingerprint
from rtttl_gen.rtttl import Event, Song


def test_transposition_preserves_rhythm_rests_tempo_family():
    song = Song("source", 140, (Event(64, 8), Event(None, 4, True), Event(72, 16)))
    changed = transpose(song, 3)
    assert [e.pitch for e in changed.events] == [67, None, 75]
    assert [(e.duration, e.dotted) for e in changed.events] == [(8, False), (4, True), (16, False)]
    assert changed.bpm == song.bpm
    assert family_fingerprint(changed) == family_fingerprint(song)
    assert song.events[0].pitch == 64  # Input objects are not mutated.


def test_boundary_notes_allow_only_legal_shifts():
    song = Song("limits", 120, (Event(60, 8), Event(107, 8)))
    assert legal_transpositions(song) == [0]
    with pytest.raises(ValueError, match="pitch range"):
        transpose(song, 1)
    with pytest.raises(ValueError, match="pitch range"):
        transpose(song, -1)


def test_train_only_seeded_augmentation():
    song = Song("source", 120, (Event(64, 8), Event(72, 8)))
    assert augment_song(song, random.Random(7)) == augment_song(song, random.Random(7))
    with pytest.raises(ValueError, match="training split"):
        augment_song(song, random.Random(7), split="validation")
    assert legal_transpositions(replace(song, events=(Event(None, 4),))) == [0]
