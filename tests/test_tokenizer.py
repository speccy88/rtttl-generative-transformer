import pytest

from rtttl_gen.augment import legal_transpositions, transpose
from rtttl_gen.rtttl import Event, Song, parse_rtttl
from rtttl_gen.tokenizer import EventTokenizer


def test_explicit_factorized_round_trip():
    song = parse_rtttl("Example:d=8,o=5,b=120:c,e,g,4c6,16p.,f#5")
    tok = EventTokenizer()
    ids = tok.encode(song)
    restored = tok.decode(ids)
    assert restored.events == song.events
    assert restored.bpm == song.bpm
    assert len(ids) == 2 * len(song.events) + 3
    assert tok.token_strings(ids)[1] == "<BPM_120>"


def test_fixed_vocabulary_is_independent_of_training_data():
    tok = EventTokenizer()
    assert tok.vocab_size == 940
    assert EventTokenizer.from_dict(tok.to_dict()).tokens == tok.tokens
    song = Song(name="Bounds", bpm=900,
                events=(Event(pitch=60, duration=1, dotted=True), Event(pitch=107, duration=32, dotted=False)))
    assert tok.decode(tok.encode(song)).events == song.events


def test_grammar_rejects_corrupt_sequence():
    tok = EventTokenizer()
    ids = tok.encode(parse_rtttl("X:d=4,o=5,b=120:c"))
    with pytest.raises(ValueError):
        tok.decode(ids[:-1])
    broken = list(ids)
    broken[2], broken[3] = broken[3], broken[2]
    with pytest.raises(ValueError):
        tok.decode(broken)
    with pytest.raises(ValueError):
        tok.decode([tok.bos_id, ids[1], tok.eos_id])
    assert tok.decode(ids + [tok.pad_id] * 3).events == tok.decode(ids).events


def test_grammar_eos_minimum_and_maximum():
    tok = EventTokenizer()
    ids = tok.encode(parse_rtttl("X:d=4,o=5,b=120:c"))[:-1]
    assert tok.eos_id not in tok.allowed_next(ids, min_events=2)
    assert tok.eos_id in tok.allowed_next(ids, min_events=1)
    assert tok.allowed_next(ids, min_events=1, max_events=1) == [tok.eos_id]
    assert tok.allowed_next(ids[:-1]) == tok.duration_ids


def test_transpose_preserves_rhythm_tempo_and_rests():
    song = parse_rtttl("X:d=8,o=5,b=120:c,e.,p,4g")
    shifted = transpose(song, 2)
    assert shifted.bpm == song.bpm
    for before, after in zip(song.events, shifted.events):
        assert (after.duration, after.dotted) == (before.duration, before.dotted)
        assert after.pitch == (before.pitch + 2 if before.pitch is not None else None)
    assert transpose(shifted, -2).events == song.events


def test_transposition_bounds():
    song = Song(name="Bounds", bpm=120, events=(Event(pitch=60, duration=4, dotted=False),))
    assert legal_transpositions(song, max_semitones=3) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        transpose(song, -1)
    with pytest.raises(ValueError):
        transpose(song, 0.5)
