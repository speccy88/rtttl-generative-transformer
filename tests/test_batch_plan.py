"""Batch controls are reproducible without changing note-sampling randomness."""
from collections import Counter
import random

import pytest

from rtttl_gen.batch_plan import PROFILE_NAMES, plan_batch
from rtttl_gen.guidance import PITCH_CLASS_NAMES, PROFILE_SPECS


def test_unprofiled_generation_keeps_learned_tempo():
    assert plan_batch(2) == [{
        "profile": None, "bpm": None, "tonic": "C", "mode": "major",
        "tempo_source": "model", "profile_source": "none",
    }] * 2


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_named_profile_uses_variable_natural_range_and_default_mode(profile):
    spec = PROFILE_SPECS[profile]
    plan = plan_batch(40, profile=profile)
    assert all(row["profile"] == profile and row["tonic"] == "C" for row in plan)
    assert all(row["mode"] == spec.default_mode for row in plan)
    assert all(spec.bpm_range[0] <= row["bpm"] <= spec.bpm_range[1] for row in plan)
    assert all(row["tempo_source"] == "profile_range" for row in plan)
    assert all(row["profile_source"] == "explicit" for row in plan)
    assert len({row["bpm"] for row in plan}) > 1


def test_mixed_cycles_are_balanced_and_prefix_stable_at_every_batch_length():
    whole = plan_batch(53, profile="mixed", seed=819)
    for size in range(1, len(whole) + 1):
        prefix = plan_batch(size, profile="mixed", seed=819)
        assert prefix == whole[:size]
        counts = Counter(row["profile"] for row in prefix)
        all_counts = [counts[name] for name in PROFILE_NAMES]
        assert max(all_counts) - min(all_counts) <= 1
    assert {row["profile"] for row in whole[:5]} == set(PROFILE_NAMES)
    assert {row["tonic"] for row in whole[:12]} == set(PITCH_CLASS_NAMES)
    assert all(row["profile_source"] == "mixed" for row in whole)
    assert all(row["mode"] == PROFILE_SPECS[row["profile"]].default_mode for row in whole)


def test_seed_is_reproducible_and_never_changes_global_random_state():
    before = random.getstate()
    first = plan_batch(19, profile="mixed", seed=71)
    assert random.getstate() == before
    assert plan_batch(19, profile="mixed", seed=71) == first
    assert plan_batch(19, profile="mixed", seed=72) != first


@pytest.mark.parametrize("profile", [None, "pop-hook", "cinematic", "mixed"])
def test_explicit_controls_override_all_profile_defaults(profile):
    plan = plan_batch(12, profile=profile, bpm=211, tonic="Bb", mode="natural-minor")
    assert all(row["bpm"] == 211 and row["tempo_source"] == "explicit" for row in plan)
    assert all(row["tonic"] == "A#" and row["mode"] == "natural-minor" for row in plan)


@pytest.mark.parametrize("profile", [None, "lullaby", "mixed"])
def test_custom_tempo_range_overrides_profile_and_includes_both_endpoints(profile):
    plan = plan_batch(20, profile=profile, bpm_range=(25, 26))
    assert {row["bpm"] for row in plan} == {25, 26}
    assert all(row["tempo_source"] == "range" for row in plan)
    assert plan_batch(1, profile=profile, bpm_range=(900, 900))[0]["bpm"] == 900


def test_tempo_override_keeps_mixed_styles_and_keys_in_same_order():
    default = plan_batch(17, profile="mixed", seed=44)
    for kwargs in ({"bpm": 70}, {"bpm_range": (170, 200)}):
        changed = plan_batch(17, profile="mixed", seed=44, **kwargs)
        assert [(row["profile"], row["tonic"], row["mode"]) for row in changed] == [
            (row["profile"], row["tonic"], row["mode"]) for row in default]


@pytest.mark.parametrize("bpm", [25, 900])
def test_explicit_tempo_accepts_valid_extremes(bpm):
    assert plan_batch(1, bpm=bpm)[0]["bpm"] == bpm


@pytest.mark.parametrize("kwargs", [
    {"num_songs": 0}, {"num_songs": -1}, {"num_songs": 2.0}, {"num_songs": True},
    {"seed": "42"}, {"seed": False},
    {"profile": "unknown"}, {"profile": ""}, {"profile": []},
    {"bpm": 24}, {"bpm": 901}, {"bpm": 112.0}, {"bpm": True}, {"bpm": "112"},
    {"bpm_range": (24, 50)}, {"bpm_range": (50, 901)},
    {"bpm_range": (50, True)}, {"bpm_range": (False, 60)},
    {"bpm_range": (50.0, 60)}, {"bpm_range": (50, "60")},
    {"bpm_range": (150, 100)}, {"bpm_range": (80,)},
    {"bpm_range": (80, 90, 100)}, {"bpm_range": "80:100"}, {"bpm_range": 90},
    {"bpm": 112, "bpm_range": (100, 120)},
    {"tonic": "H"}, {"tonic": ""}, {"tonic": False},
    {"mode": "minor"}, {"mode": ""}, {"mode": []},
])
def test_invalid_arguments_fail_before_inference(kwargs):
    with pytest.raises(ValueError):
        plan_batch(**({"num_songs": 5} | kwargs))
