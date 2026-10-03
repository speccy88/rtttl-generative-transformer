"""End-to-end sampling checks using small synthetic models, with no downloads."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import random

import pytest
import torch

from rtttl_gen.generation import generate_tokens
from rtttl_gen.guidance import PROFILE_NAMES, PROFILE_SPECS
from rtttl_gen.rtttl import Event, Song, parse_rtttl
from rtttl_gen.tokenizer import EventTokenizer


@pytest.fixture(scope="module")
def generation_cli():
    path = Path(__file__).resolve().parents[1] / "generate.py"
    spec = importlib.util.spec_from_file_location("mixed_generation_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticMelody:
    """Nonconstant logits depend on history without using a global RNG."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def next_logits(self, ids):
        values = torch.sin(
            torch.arange(self.tokenizer.vocab_size, dtype=torch.float32) * 0.137
            + sum(ids[-3:]) * 0.011
        )
        values[self.tokenizer.eos_id] = -5
        return values


@pytest.fixture
def batch_inputs():
    tokenizer = EventTokenizer()
    # These fixtures exercise the actual similarity index, with no corpus access.
    training = [
        Song("Up", 120, tuple(Event(pitch, 8) for pitch in (60, 62, 64, 65, 67, 69))),
        Song("Down", 90, tuple(Event(pitch, 4) for pitch in (79, 77, 76, 74, 72, 71))),
    ]
    return SyntheticMelody(tokenizer), tokenizer, training, ["train-up", "train-down"]


def run_batch(generation_cli, batch_inputs, **kwargs):
    settings = dict(num_songs=10, min_events=8, max_events=8, seed=137,
                    profile="mixed", temperature=0.9, top_k=7, top_p=0.83)
    settings.update(kwargs)
    return generation_cli.generate_batch(*batch_inputs, **settings)


def test_mixed_batch_covers_each_style_with_varied_actual_tempo(generation_cli, batch_inputs):
    records, songs = run_batch(generation_cli, batch_inputs, num_songs=25)
    tokenizer = batch_inputs[1]
    counts = Counter(record["generation_settings"]["profile"] for record in records)
    assert counts == Counter({profile: 5 for profile in PROFILE_NAMES})
    assert len(songs) == len(records) == 25
    assert len({song.bpm for song in songs}) > 5
    assert len({record["generation_settings"]["tonic"] for record in records}) > 1
    assert {record["generation_settings"]["mode"] for record in records} == {
        "major", "natural-minor",
    }
    for record, song in zip(records, songs):
        settings = record["generation_settings"]
        low, high = PROFILE_SPECS[settings["profile"]].bpm_range
        assert low <= song.bpm <= high
        assert settings["bpm"] == record["bpm"] == song.bpm
        assert settings["tempo_source"] == "profile_range"
        assert settings["profile_source"] == "mixed"
        assert tokenizer.decode(record["token_ids"]).bpm == song.bpm
        assert parse_rtttl(record["rtttl"]).bpm == song.bpm
        assert record["valid"]
        assert record["similarity"]["training_reference_count"] == 2
        assert record["similarity"]["closest_training_id"] in {"train-up", "train-down"}


def test_mixed_batch_is_reproducible_without_consuming_global_rng(generation_cli, batch_inputs):
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    records, songs = run_batch(generation_cli, batch_inputs)
    repeated_records, repeated_songs = run_batch(generation_cli, batch_inputs)
    assert records == repeated_records
    assert songs == repeated_songs
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_larger_mixed_batch_preserves_existing_prefix(generation_cli, batch_inputs):
    short_records, short_songs = run_batch(generation_cli, batch_inputs, num_songs=7)
    long_records, long_songs = run_batch(generation_cli, batch_inputs, num_songs=19)
    assert long_records[:7] == short_records
    assert long_songs[:7] == short_songs


def test_explicit_tempo_and_key_override_each_mixed_song(generation_cli, batch_inputs):
    records, songs = run_batch(generation_cli, batch_inputs, bpm=137, tonic="Bb",
                              mode="natural-minor")
    assert {song.bpm for song in songs} == {137}
    for record in records:
        settings = record["generation_settings"]
        assert settings["bpm"] == 137
        assert settings["tonic"] == "A#"
        assert settings["mode"] == "natural-minor"
        assert settings["tempo_source"] == "explicit"
        assert settings["profile_source"] == "mixed"
        assert parse_rtttl(record["rtttl"]).bpm == 137


@pytest.mark.parametrize("profile", [None, "mixed", "pop-hook"])
def test_requested_tempo_range_reaches_tokens_rtttl_and_metadata(
        generation_cli, batch_inputs, profile):
    records, songs = run_batch(generation_cli, batch_inputs, profile=profile,
                              bpm_range=(71, 77), num_songs=14)
    tokenizer = batch_inputs[1]
    assert len({song.bpm for song in songs}) > 1
    for record, song in zip(records, songs):
        assert 71 <= song.bpm <= 77
        assert record["bpm"] == song.bpm
        assert record["generation_settings"]["bpm"] == song.bpm
        assert record["generation_settings"]["tempo_source"] == "range"
        assert tokenizer.decode(record["token_ids"]).bpm == song.bpm
        assert parse_rtttl(record["rtttl"]).bpm == song.bpm


def test_tempo_and_range_conflict_fails_before_inference(generation_cli, batch_inputs):
    class UnusedModel:
        def next_logits(self, ids):
            pytest.fail("Conflicting tempo options must be rejected before inference")

    _, tokenizer, training, training_ids = batch_inputs
    with pytest.raises(ValueError):
        generation_cli.generate_batch(UnusedModel(), tokenizer, training, training_ids,
                                      profile="mixed", bpm=112, bpm_range=(90, 130))


def test_cli_forwards_musical_options_without_naming_arguments(generation_cli):
    parser = argparse.ArgumentParser()
    generation_cli.add_sampling_arguments(parser)
    generation_cli.add_naming_arguments(parser)
    args = parser.parse_args([
        "--profile", "mixed", "--bpm-range", "80", "155", "--tonic", "F#",
        "--mode", "natural-minor", "--name-songs", "--name-offline",
    ])
    options = generation_cli.sampling_kwargs(args)
    assert options["profile"] == "mixed"
    assert tuple(options["bpm_range"]) == (80, 155)
    assert options["bpm"] is None
    assert options["tonic"] == "F#"
    assert options["mode"] == "natural-minor"
    assert not any(key.startswith("name_") for key in options)
    defaults = generation_cli.sampling_kwargs(parser.parse_args([]))
    assert defaults["profile"] is None
    assert defaults["bpm"] is None
    assert defaults["bpm_range"] is None
    assert defaults["tonic"] is None
    assert defaults["mode"] is None


def test_no_profile_preserves_model_tempo_and_legacy_sampling(generation_cli, batch_inputs):
    model, tokenizer, _, _ = batch_inputs
    records, songs = run_batch(generation_cli, batch_inputs, profile=None, num_songs=1,
                              seed=19, repetition_penalty=1, max_pitch_run=0,
                              max_motif_repeats=0)
    expected = generate_tokens(model, tokenizer, min_events=8, max_events=8,
                               temperature=0.9, top_k=7, top_p=0.83, seed=19,
                               repetition_penalty=1, max_pitch_run=0,
                               max_motif_repeats=0)
    assert records[0]["token_ids"] == expected["token_ids"]
    # Captured before tempo/profile changes; this tempo comes from the model.
    assert songs[0].bpm == 125
    assert records[0]["generation_settings"]["profile"] is None
    assert records[0]["generation_settings"]["tempo_source"] == "model"


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_direct_profile_generation_varies_tempo_reproducibly_within_its_range(profile):
    tokenizer = EventTokenizer()
    model = SyntheticMelody(tokenizer)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    outputs = [generate_tokens(model, tokenizer, min_events=2, max_events=2,
                               profile=profile, seed=seed)
               for seed in range(12)]
    repeated = [generate_tokens(model, tokenizer, min_events=2, max_events=2,
                                profile=profile, seed=seed)
                for seed in range(12)]
    assert outputs == repeated
    bpms = [tokenizer.decode(output["token_ids"]).bpm for output in outputs]
    low, high = PROFILE_SPECS[profile].bpm_range
    assert all(low <= bpm <= high for bpm in bpms)
    assert len(set(bpms)) > 1
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("seed,events", [
    (19, [(105, "32_plain"), (91, "16_plain"), (74, "16_dot"), (72, "32_dot"),
          (76, "8_plain"), (74, "4_plain"), (72, "16_plain"), (76, "8_plain")]),
    (72, [(100, "16_plain"), (95, "16_dot"), (72, "32_plain"), (77, "4_plain"),
          (79, "4_plain"), (77, "16_plain"), (79, "32_plain"), (74, "4_plain")]),
])
def test_explicit_pop_tempo_preserves_captured_note_sequence(seed, events):
    # Captured with the original pop guidance and sampler before this change.
    tokenizer = EventTokenizer()
    result = generate_tokens(SyntheticMelody(tokenizer), tokenizer, min_events=8,
                             max_events=8, temperature=0.9, top_k=7, top_p=0.83,
                             profile="pop-hook", bpm=112, seed=seed)
    expected = ["<BOS>", "<BPM_112>"]
    for pitch, duration in events:
        expected.extend([f"<PITCH_{pitch}>", f"<DUR_{duration}>"])
    expected.append("<EOS>")
    assert tokenizer.token_strings(result["token_ids"]) == expected
