import json
import math
import wave

import pytest
import torch

from rtttl_gen.audio import create_listening_batch, render_wav
from rtttl_gen.evaluation import compare_distributions, degeneracy_flags, distribution_data, entropy, generated_metrics, predictive_metrics, repetition_statistics, song_key, validate_song
from rtttl_gen.rtttl import Event, Song, parse_rtttl
from rtttl_gen.tokenizer import EventTokenizer


def sample():
    return parse_rtttl("Demo:d=8,o=5,b=120:c,e,g,4c6,p,8d.")


def test_entropy_and_distribution():
    assert entropy([]) == 0
    assert entropy([1, 1]) == 0
    assert entropy([1, 2]) == 1
    stats = distribution_data([sample()])
    assert stats["interval"] == [4, 3, 5, -10]
    assert stats["duration_beats"][-1] == 0.75
    assert stats["rest_fraction"] == [1 / 6]
    assert stats["note_range"] == [12]


def test_validity_round_trip_and_flags():
    song = sample()
    valid, rtttl, error = validate_song(song)
    assert valid and error is None
    assert song_key(parse_rtttl(rtttl)) == song_key(song)
    rests = parse_rtttl("Rest:d=8,o=5,b=120:" + ",".join(["p"] * 16))
    assert "all_rests" in degeneracy_flags(rests)
    assert "short_repeated_motif" in degeneracy_flags(rests)


def test_pitch_run_detected_after_intro_despite_changing_durations():
    song = parse_rtttl("Loop:d=8,o=6,b=120:c,e,g," + ",".join(["d#", "4d#."] * 4))
    stats = repetition_statistics(song)
    assert stats["longest_pitch_run"] == 8
    assert stats["longest_event_run"] == 1
    assert stats["unique_pitches"] == 4
    assert stats["max_pitch_motif_repeats"] == 0
    assert "local_pitch_run_at_least_8" in degeneracy_flags(song)
    assert "single_pitch" not in degeneracy_flags(song)
    assert "repeated_pitch_motif_at_least_4" not in degeneracy_flags(song)


@pytest.mark.parametrize("period", [2, 5, 16])
def test_pitch_motifs_detected_after_varied_intro_despite_rhythm_variation(period):
    pitches = [84, 86, 88] + list(range(60, 60 + period)) * 4
    song = Song("Motif", 120, tuple(Event(pitch, (4, 8, 16)[i % 3], i % 5 == 0) for i, pitch in enumerate(pitches)))
    assert repetition_statistics(song)["max_pitch_motif_repeats"] == 4
    assert "repeated_pitch_motif_at_least_4" in degeneracy_flags(song)
    assert "short_repeated_motif" not in degeneracy_flags(song)


@pytest.mark.parametrize("copies", [2, 3])
def test_musical_motif_can_repeat_without_new_degeneracy_flags(copies):
    song = parse_rtttl("Phrase:d=8,o=5,b=120:a,b," + ",".join(["c", "e", "g", "p", "d"] * copies))
    assert repetition_statistics(song)["max_pitch_motif_repeats"] == copies
    assert "repeated_pitch_motif_at_least_4" not in degeneracy_flags(song)
    assert "local_pitch_run_at_least_8" not in degeneracy_flags(song)


def test_rests_are_runs_and_motif_members_but_not_unique_pitches():
    rests = parse_rtttl("Rest:d=8,o=5,b=120:" + ",".join(["p", "4p."] * 4))
    stats = repetition_statistics(rests)
    assert stats["longest_pitch_run"] == 8
    assert stats["longest_event_run"] == 1
    assert stats["unique_pitches"] == 0
    assert stats["max_pitch_motif_repeats"] == 0
    assert stats["repeated_pitch_ngram_fraction"] == pytest.approx(4 / 5)
    assert "local_pitch_run_at_least_8" in degeneracy_flags(rests)
    assert "repeated_pitch_motif_at_least_4" not in degeneracy_flags(rests)
    motif = parse_rtttl("RestMotif:d=8,o=5,b=120:" + ",".join(["c", "p"] * 4))
    assert repetition_statistics(motif)["max_pitch_motif_repeats"] == 4


def test_repetition_statistics_short_song_and_overlapping_ngram_denominator():
    assert repetition_statistics(parse_rtttl("One:d=8,o=5,b=120:c")) == {
        "longest_pitch_run": 1,
        "longest_event_run": 1,
        "unique_pitches": 1,
        "max_pitch_motif_repeats": 0,
        "repeated_pitch_ngram_fraction": 0,
    }
    song = parse_rtttl("Three:d=8,o=5,b=120:c,e,c,e,c,e")
    assert repetition_statistics(song)["repeated_pitch_ngram_fraction"] == pytest.approx(1 / 3)
    song = parse_rtttl("Run:d=8,o=5,b=120:c,c,c,c,4c,4c,4c,4c")
    assert repetition_statistics(song)["longest_event_run"] == 4


def test_generation_metric_denominators():
    song = sample()
    records = [dict(valid=True, similarity={"score": 1., "exact_event_match": True, "label": "exact training match"}, forced_eos=False), dict(valid=True, similarity={"score": 0., "exact_event_match": False, "label": "low similarity"}, forced_eos=True), dict(valid=False)]
    metrics = generated_metrics(records, [song, song])
    assert metrics["rtttl_validity_pct"] == pytest.approx(200 / 3)
    assert metrics["unique_generations_pct"] == 50
    assert metrics["duplicate_generation_pct"] == 50
    assert metrics["exact_training_match_pct"] == 50
    assert metrics["nearest_similarity"]["mean"] == 0.5


def test_generation_repetition_summaries_include_valid_songs_and_recorded_attempts():
    loop = parse_rtttl("Loop:d=8,o=5,b=120:" + ",".join(["c", "e"] * 4))
    records = [
        dict(valid=True, repetition_interventions=dict(penalty_steps=4, pitch_run_blocks=2, motif_blocks=0)),
        dict(valid=True),
        dict(valid=False, repetition_interventions=dict(penalty_steps=0, pitch_run_blocks=0, motif_blocks=1)),
    ]
    metrics = generated_metrics(records, [sample(), loop])
    assert metrics["repetition"]["max_pitch_motif_repeats"] == dict(count=2, min=0, mean=2, median=2, max=4)
    assert metrics["repetition_interventions"]["penalty_steps"]["mean"] == 2
    assert metrics["repetition_interventions"]["records_with_counters"] == 2
    assert metrics["repetition_interventions"]["songs_with_hard_blocks"] == 2
    empty = generated_metrics([], [])
    assert empty["repetition"]["longest_pitch_run"] == dict(count=0, min=None, mean=None, median=None, max=None)
    assert "repetition_interventions" not in empty


class UniformNeural(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = vocabulary

    def forward(self, x):
        return torch.zeros((*x.shape, self.vocabulary), device=x.device)


class UniformNgram:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def next_logits(self, context):
        return torch.zeros(self.vocabulary)


def test_predictive_ce_masks_pad_and_scores_all_original_targets_once():
    tokenizer = EventTokenizer()
    songs = [sample(), parse_rtttl("Long:d=8,o=5,b=120:" + ",".join(["c", "e", "g"] * 7))]
    expected_tokens = sum(len(tokenizer.encode(song)) - 1 for song in songs)
    for model in (UniformNeural(tokenizer.vocab_size), UniformNgram(tokenizer.vocab_size)):
        metrics = predictive_metrics(model, tokenizer, songs, context_length=10, batch_size=2)
        assert metrics["target_tokens"] == expected_tokens
        assert metrics["cross_entropy_nats"] == pytest.approx(math.log(tokenizer.vocab_size), rel=1e-6)
        assert metrics["perplexity"] == pytest.approx(tokenizer.vocab_size, rel=1e-6)
        assert metrics["grammar_masked"] is False


def test_empty_predictive_split_is_not_fake_zero():
    tokenizer = EventTokenizer()
    metrics = predictive_metrics(UniformNeural(tokenizer.vocab_size), tokenizer, [], 16)
    assert metrics["perplexity"] is None
    assert metrics["cross_entropy_nats"] is None
    assert metrics["target_tokens"] == 0


def test_audio_duration_and_existing_output_protection(tmp_path):
    song = parse_rtttl("Audio:d=4,o=5,b=120:c,4p.,8g")
    path = tmp_path / "demo.wav"
    metadata = render_wav(song, path, sample_rate=8000)
    assert metadata["seconds"] == pytest.approx(1.5)
    with wave.open(str(path)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 12000
    with pytest.raises(FileExistsError):
        render_wav(song, path)


def test_listening_html_escapes_names(tmp_path):
    song = parse_rtttl("<img>:d=8,o=5,b=120:c")
    valid, rtttl, _ = validate_song(song)
    batch = tmp_path / "listen"
    manifest = create_listening_batch([dict(valid=valid, rtttl=rtttl)], batch)
    assert len(manifest) == 1
    assert "<img>" not in (batch / "index.html").read_text()
    assert (batch / "sample_0001.wav").exists()


def test_listening_uses_full_title_and_escapes_it_even_for_skipped_audio(tmp_path):
    title = '<img src=x onerror="alert(1)"> & Moon Glow'
    rtttl = "MoonGlow:d=8,o=5,b=120:c"
    records = [dict(valid=True, rtttl=rtttl, title=title)]
    for folder, limit in (("audio", 300), ("skipped", 0.001)):
        batch = tmp_path / folder
        manifest = create_listening_batch(records, batch, sample_rate=8000, max_seconds=limit)
        page = (batch / "index.html").read_text()
        assert manifest[0]["name"] == title
        assert "<img" not in page
        assert '&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; Moon Glow</h2>' in page
        assert rtttl in page
        assert json.loads((batch / "audio_manifest.json").read_text())[0]["name"] == title


def test_listening_shows_and_preserves_song_settings(tmp_path):
    settings = dict(profile="cinematic", tonic="A", mode="natural-minor", bpm=76)
    record = dict(valid=True, rtttl="Night:d=8,o=5,b=76:c", generation_settings=settings)
    batch = tmp_path / "styles"
    manifest = create_listening_batch([record], batch)
    page = (batch / "index.html").read_text()
    assert "Cinematic</strong>" in page
    assert "A natural-minor preference" in page
    assert "76 BPM" in page
    assert manifest[0]["generation_settings"] == settings
    assert manifest[0]["bpm"] == 76
    assert "same gentle sine sound" in page


def test_distribution_plot_and_serializable_evidence(tmp_path):
    result = compare_distributions([sample()], [sample()], tmp_path / "plots")
    assert result["categorical_js_divergence_bits"]["pitch"] == 0
    assert (tmp_path / "plots" / "musical_distributions.png").stat().st_size > 1000
    assert json.loads((tmp_path / "plots" / "distribution_metrics.json").read_text())["real_song_count"] == 1


def test_training_reference_respects_actual_subset_and_fingerprints(tmp_path):
    import importlib.util
    from pathlib import Path
    from rtttl_gen.rtttl import song_to_dict
    from rtttl_gen.training import dataset_fingerprint
    spec = importlib.util.spec_from_file_location("project_generate", Path(__file__).resolve().parents[1] / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {"id": "one", "song": song_to_dict(sample())}
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n" + json.dumps({**record, "id": "two"}) + "\n")
    for name in ("val.jsonl", "test.jsonl"):
        (tmp_path / name).write_text(json.dumps(record) + "\n")
    (tmp_path / "tokenizer.json").write_text(json.dumps(EventTokenizer().to_dict()))
    checkpoint = {"dataset_sha256": dataset_fingerprint(tmp_path), "config": {"data": {"max_train_songs": 1}}}
    songs, ids = module.training_reference(tmp_path, checkpoint)
    assert ids == ["one"] and len(songs) == 1
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="fingerprints"):
        module.training_reference(tmp_path, checkpoint)
