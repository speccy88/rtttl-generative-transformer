from dataclasses import replace

import pytest

from rtttl_gen.rtttl import Event, Song, song_to_dict
from rtttl_gen.deduplicate import (
    assert_no_leakage, family_fingerprint, group_families, split_families, deduplicate_songs,
)
from rtttl_gen.similarity import SimilarityIndex
from rtttl_gen.deduplicate import exact_fingerprint


def melody(name="one", shift=0, bpm=120, durations=(8, 8, 4, 8, 2)):
    pitches = (60, 64, None, 67, 72)
    return Song(name=name, bpm=bpm, events=tuple(
        Event(None if pitch is None else pitch + shift, duration, False)
        for pitch, duration in zip(pitches, durations)))


def test_renamed_and_transposed_family():
    a, renamed, transposed = melody(), melody("renamed"), melody("transposed", shift=3)
    songs = [a, renamed, transposed]
    ids, duplicate_of, report = group_families(songs)
    assert len(set(ids)) == 1
    assert duplicate_of == [None, 0, None]
    assert report["unique_exact_songs"] == 2
    unique, retained, duplicates = deduplicate_songs(songs)
    assert duplicates == [0, 0, 2]
    assert len(unique) == 2
    assert retained == [0, 2]


def test_uniform_rhythm_and_tempo_change():
    a = melody()
    b = melody(shift=2, bpm=180, durations=(16, 16, 8, 16, 4))
    assert family_fingerprint(a) == family_fingerprint(b)


def test_tempo_variants_retained_but_same_family():
    a, b = melody(bpm=120), melody(bpm=180)
    assert exact_fingerprint(a) != exact_fingerprint(b)
    assert exact_fingerprint(a, include_tempo=False) == exact_fingerprint(b, include_tempo=False)
    assert len(deduplicate_songs([a, b])[0]) == 2
    families, _, _ = group_families([a, b])
    assert len(set(families)) == 1


def test_rests_are_not_pitches():
    a = melody()
    b = replace(a, events=tuple(Event(60 if e.pitch is None else e.pitch, e.duration, e.dotted)
                                for e in a.events))
    assert family_fingerprint(a) != family_fingerprint(b)


def test_family_split_is_reproducible_and_no_leakage():
    songs = []
    for i in range(60):
        pitches = [60, 61 + i % 10, 62 + i // 10, None, 70]
        events = tuple(Event(p, 8, bool(i % 2)) for p in pitches)
        song = Song(name=str(i), bpm=120, events=events)
        songs.extend([song, replace(song, name=f"copy-{i}")])
    families, _, _ = group_families(songs)
    splits = split_families(families, seed=42)
    assert splits == split_families(families, seed=42)
    assert set(splits) == {"train", "validation", "test"}
    assert_no_leakage(songs, splits, families)
    assert abs(splits.count("train") / len(songs) - 0.8) < 0.06


def test_leakage_checker_independently_catches_transposition():
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_leakage([melody(), melody(shift=5)], ["train", "test"], ["a", "b"])


def test_small_number_of_families_preserves_nonempty_splits():
    assignments = split_families(["a"] * 20 + ["b", "c"])
    assert set(assignments) == {"train", "validation", "test"}
    assert assignments[0] == "train"


def test_split_is_independent_of_record_order():
    families = [f"f{i:03}" for i in range(80) for _ in range(1 + i % 4)]
    expected = dict(zip(families, split_families(families, seed=27)))
    reversed_families = list(reversed(families))
    actual = dict(zip(reversed_families, split_families(reversed_families, seed=27)))
    assert actual == expected


def test_multivariant_families_are_not_all_assigned_to_training():
    from collections import Counter
    # The previous size-sorted deficit allocator put every large family into
    # train and filled both held-out splits entirely with singleton families.
    families = [f"single-{i:03}" for i in range(800)]
    families += [f"variant-{i:03}" for i in range(200) for _ in range(3)]
    splits = split_families(families, seed=42)
    sizes = Counter(families)
    for partition in ("train", "validation", "test"):
        retained = {family for family, split in zip(families, splits) if split == partition}
        assert any(sizes[family] > 1 for family in retained)
        assert any(sizes[family] == 1 for family in retained)
    assert abs(splits.count("train") - .8 * len(families)) <= 3
    assert abs(splits.count("validation") - .1 * len(families)) <= 3
    assert abs(splits.count("test") - .1 * len(families)) <= 3


def test_similar_edited_long_melodies_group():
    events = tuple(Event(60 + ((i * 7 + i // 7) % 21), (4, 8, 16)[i % 3], False)
                   for i in range(65))
    a = Song(name="long", bpm=120, events=events)
    changed = list(events)
    changed[31] = Event(changed[31].pitch + 1, changed[31].duration, False)
    b = replace(a, name="edited", events=tuple(changed))
    families, _, report = group_families([a, b])
    assert families[0] == families[1]
    assert report["approximate_pair_count"] == 1


def test_truncated_long_alias_groups_with_original():
    events = tuple(Event(60 + ((i * 7 + i // 7) % 21), (4, 8, 16)[i % 3], False)
                   for i in range(65))
    full = Song(name="full", bpm=120, events=events)
    excerpt = replace(full, name="excerpt", events=events[:48])
    families, _, report = group_families([full, excerpt])
    assert families[0] == families[1]
    assert report["approximate_pairs"][0]["containment"] == 1.0


def test_dotted_rhythms_normalize_without_float_error():
    full = Song(name="dotted", bpm=120,
                events=(Event(60, 4, True), Event(64, 8), Event(None, 16, True)))
    faster = Song(name="scaled", bpm=90,
                  events=(Event(62, 8, True), Event(66, 16), Event(None, 32, True)))
    assert family_fingerprint(full) == family_fingerprint(faster)


def test_similarity_distinguishes_exact_and_transposition():
    index = SimilarityIndex([melody()], ids=["source-1"])
    exact = index.nearest(melody("another name", bpm=160))
    assert exact["label"] == "exact training match"
    assert exact["exact_event_match"] is True
    assert exact["same_bpm"] is False
    assert exact["closest_id"] == "source-1"
    transposed = index.nearest(melody(shift=4))
    assert transposed["label"] == "near match"
    assert transposed["transposition_match"] is True
    assert transposed["exact_event_match"] is False
    assert transposed["score"] == 1.0


def test_similarity_empty_index_is_explicit():
    result = SimilarityIndex([]).nearest(melody())
    assert result["score"] is None
    assert result["closest_index"] is None


def test_similarity_reads_processed_record_dict():
    song = melody()
    index = SimilarityIndex([{"id": "original-7", "song": song_to_dict(song)}])
    result = index.nearest(song)
    assert result["closest_id"] == "original-7"
    assert result["exact_event_match"]


def test_similarity_result_is_global_maximum_of_reported_metric():
    from rtttl_gen.similarity import _features
    from rtttl_gen.deduplicate import jaccard
    songs = [Song(str(i), 120, tuple(Event(60 + (j * (i + 1) + i) % 32, (4, 8, 16)[j % 3])
                                   for j in range(15 + i))) for i in range(15)]
    query = Song("query", 120, tuple(Event(60 + (j * 4 + 3) % 32, (4, 8, 16)[j % 3])
                                   for j in range(19)))
    result = SimilarityIndex(songs).nearest(query)
    query_features = _features(query)
    expected = max(sum(weight * jaccard(a, b) for weight, a, b in
                       zip((.55, .30, .15), query_features, _features(song))) for song in songs)
    assert result["score"] == pytest.approx(expected, abs=1e-6)
