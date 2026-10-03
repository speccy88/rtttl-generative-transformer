import json
import math
import wave

import pytest
import torch

from rtttl_gen.audio import create_listening_batch, render_wav
from rtttl_gen.evaluation import compare_distributions, degeneracy_flags, distribution_data, entropy, generated_metrics, predictive_metrics, song_key, validate_song
from rtttl_gen.rtttl import parse_rtttl
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


def test_generation_metric_denominators():
    song = sample()
    records = [dict(valid=True, similarity={"score": 1., "exact_event_match": True, "label": "exact training match"}, forced_eos=False), dict(valid=True, similarity={"score": 0., "exact_event_match": False, "label": "low similarity"}, forced_eos=True), dict(valid=False)]
    metrics = generated_metrics(records, [song, song])
    assert metrics["rtttl_validity_pct"] == pytest.approx(200 / 3)
    assert metrics["unique_generations_pct"] == 50
    assert metrics["duplicate_generation_pct"] == 50
    assert metrics["exact_training_match_pct"] == 50
    assert metrics["nearest_similarity"]["mean"] == 0.5


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
