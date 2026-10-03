"""Structural checks for optional handcrafted guidance, not musicality tests."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from rtttl_gen.generation import generate_tokens
from rtttl_gen.guidance import (PROFILE_NAMES, PROFILE_SPECS, PopHookGuidance,
                                make_guidance, normalize_tonic)
from rtttl_gen.tokenizer import EventTokenizer


def pitch_id(tok, pitch):
    return tok.token_to_id[f"<PITCH_{pitch}>"]


def duration_id(tok, denominator, dotted=False):
    return tok.token_to_id[f"<DUR_{denominator}_{'dot' if dotted else 'plain'}>"]


def pitch_biases(tok, *, tonic="C", mode="major", history=()):
    guide = make_guidance(tok, "pop-hook", tonic, mode)
    return guide.apply(torch.zeros(tok.vocab_size), tok.event_start_ids + [tok.eos_id],
                       list(history))


def test_scale_and_register_are_soft_finite_preferences():
    tok = EventTokenizer()
    scores = pitch_biases(tok)
    assert scores[pitch_id(tok, 72)] > scores[pitch_id(tok, 73)]
    assert scores[pitch_id(tok, 72)] > scores[pitch_id(tok, 60)]
    assert scores[pitch_id(tok, 84)] > scores[pitch_id(tok, 96)]
    assert torch.isfinite(scores).all()
    assert scores.abs().max() <= 1.35


@pytest.mark.parametrize("tonic,mode,inside,outside", [
    ("C", "major", 76, 75),
    ("C", "natural-minor", 75, 76),
    ("D", "major", 73, 72),
    ("Bb", "major", 82, 83),
])
def test_key_and_mode_change_scale_preference(tonic, mode, inside, outside):
    tok = EventTokenizer()
    scores = pitch_biases(tok, tonic=tonic, mode=mode)
    assert scores[pitch_id(tok, inside)] > scores[pitch_id(tok, outside)]


def test_steps_and_thirds_preferred_to_wide_leaps_without_unison_bonus():
    tok = EventTokenizer()
    scores = pitch_biases(tok, history=[pitch_id(tok, 72)])
    assert scores[pitch_id(tok, 74)] > scores[pitch_id(tok, 76)]
    assert scores[pitch_id(tok, 76)] > scores[pitch_id(tok, 72)]
    assert scores[pitch_id(tok, 72)] > scores[pitch_id(tok, 79)]
    assert scores[pitch_id(tok, 79)] > scores[pitch_id(tok, 84)]
    after_rest = pitch_biases(tok, history=[pitch_id(tok, 72), tok.rest_id])
    torch.testing.assert_close(scores[tok.pitch_ids], after_rest[tok.pitch_ids])


def test_rhythm_prefers_plain_eighths_and_quarters_and_shorter_rests():
    tok = EventTokenizer()
    guide = make_guidance(tok, "pop-hook")
    baseline = torch.zeros(tok.vocab_size)
    note = guide.apply(baseline, tok.duration_ids, [pitch_id(tok, 72)])
    rest = guide.apply(baseline, tok.duration_ids, [tok.rest_id])
    assert note[duration_id(tok, 8)] == note[duration_id(tok, 4)]
    assert note[duration_id(tok, 8)] > note[duration_id(tok, 8, True)]
    assert note[duration_id(tok, 4)] > note[duration_id(tok, 1)]
    long_rest_change = rest[duration_id(tok, 1)] - note[duration_id(tok, 1)]
    short_rest_change = rest[duration_id(tok, 8)] - note[duration_id(tok, 8)]
    assert long_rest_change < short_rest_change < 0


def test_consecutive_rest_discouragement_is_bounded():
    tok = EventTokenizer()
    first = pitch_biases(tok)[tok.rest_id]
    second = pitch_biases(tok, history=[tok.rest_id])[tok.rest_id]
    many = pitch_biases(tok, history=[tok.rest_id] * 100)[tok.rest_id]
    assert -0.61 < many < second < first < 0


def test_only_allowed_musical_scores_change_and_input_is_not_mutated():
    tok = EventTokenizer()
    guide = make_guidance(tok, "pop-hook")
    logits = torch.linspace(-4, 4, tok.vocab_size)
    original = logits.clone()
    allowed = [pitch_id(tok, 72), tok.eos_id]
    guided = guide.apply(logits, allowed, [pitch_id(tok, 74)])
    changed = torch.nonzero(guided != original).flatten().tolist()
    assert changed == [pitch_id(tok, 72)]
    assert guided[tok.eos_id] == original[tok.eos_id]
    assert guided[tok.bpm_ids].equal(original[tok.bpm_ids])
    assert guided[tok.duration_ids].equal(original[tok.duration_ids])
    assert logits.equal(original)


class StaticModel:
    def __init__(self, tok, pitches=None):
        self.logits = torch.full((tok.vocab_size,), -100.0)
        self.logits[tok.token_to_id["<BPM_150>"]] = 10
        self.logits[duration_id(tok, 8)] = 10
        for pitch, score in (pitches or {72: 4, 76: 3.2}).items():
            self.logits[pitch_id(tok, pitch)] = score

    def next_logits(self, ids):
        return self.logits


def fixed_generation(tok, model, **kwargs):
    return generate_tokens(model, tok, min_events=2, max_events=2, top_k=1,
                           max_pitch_run=0, max_motif_repeats=0, **kwargs)


@pytest.mark.parametrize("profile,explicit_bpm,expected", [
    ("pop-hook", 112, 112), ("pop-hook", 137, 137), (None, None, 150),
])
def test_explicit_profile_tempo_and_unprofiled_model_tempo(profile, explicit_bpm, expected):
    tok = EventTokenizer()
    result = fixed_generation(tok, StaticModel(tok), profile=profile, bpm=explicit_bpm)
    assert tok.decode(result["token_ids"]).bpm == expected


def test_guidance_is_applied_before_greedy_sampling_and_remains_soft():
    tok = EventTokenizer()
    # The chromatic note initially leads, but a modest scale bias changes top-1.
    result = fixed_generation(tok, StaticModel(tok, {72: 0, 73: 0.3}),
                              profile="pop-hook", repetition_penalty=1)
    assert tok.decode(result["token_ids"]).events[0].pitch == 72
    # A strong learned preference still wins; chromatic/out-of-range notes are legal.
    strong = fixed_generation(tok, StaticModel(tok, {61: 10, 72: 0}),
                              profile="pop-hook", repetition_penalty=1)
    assert [event.pitch for event in tok.decode(strong["token_ids"]).events] == [61, 61]


def test_guidance_precedes_repetition_penalty():
    tok = EventTokenizer()
    # On the second event: C=(4+.55)/1.2 < E=3.2+.65. Reversing
    # guidance and the penalty would select C again: 4/1.2+.55 > E.
    result = fixed_generation(tok, StaticModel(tok), profile="pop-hook",
                              repetition_penalty=1.2, repetition_window=1)
    assert [event.pitch for event in tok.decode(result["token_ids"]).events] == [72, 76]


def test_none_profile_ignores_valid_tonal_preferences_and_preserves_output():
    tok = EventTokenizer()
    model = StaticModel(tok)
    implicit = fixed_generation(tok, model, seed=29)
    explicit = fixed_generation(tok, model, seed=29, profile=None,
                                tonic="Bb", mode="natural-minor")
    assert implicit == explicit
    assert make_guidance(tok, None, "Bb", "natural-minor") is None


@pytest.mark.parametrize("input_tonic,canonical", [
    ("C", "C"), ("F#", "F#"), ("Bb", "A#"), ("Db", "C#"),
    ("eb", "D#"), ("G♭", "F#"), ("E#", "F"), ("Cb", "B"),
])
def test_tonic_normalization(input_tonic, canonical):
    assert normalize_tonic(input_tonic) == canonical


@pytest.mark.parametrize("kwargs", [
    {"profile": "pop"}, {"profile": ""}, {"profile": 1},
    {"tonic": "H"}, {"tonic": "C##"}, {"tonic": ""}, {"tonic": None},
    {"mode": "minor"}, {"mode": "lydian"}, {"mode": []},
])
def test_invalid_guidance_arguments_raise_value_error(kwargs):
    tok = EventTokenizer()
    with pytest.raises(ValueError):
        fixed_generation(tok, StaticModel(tok), **kwargs)


def test_profile_registry_exposes_immutable_settings():
    assert PROFILE_NAMES == ("pop-hook", "chiptune", "cinematic", "dance", "lullaby")
    assert tuple(PROFILE_SPECS) == PROFILE_NAMES
    assert [spec.bpm_range for spec in PROFILE_SPECS.values()] == [
        (88, 132), (120, 180), (60, 104), (118, 150), (60, 84),
    ]
    assert PROFILE_SPECS["cinematic"].default_mode == "natural-minor"
    for name, spec in PROFILE_SPECS.items():
        guide = make_guidance(EventTokenizer(), name)
        assert guide.spec is spec
        assert guide.tempo_range == spec.bpm_range
        assert spec.label and spec.description
        with pytest.raises(FrozenInstanceError):
            spec.bpm_range = (112, 112)
    with pytest.raises(TypeError):
        PROFILE_SPECS["pop-hook"] = PROFILE_SPECS["lullaby"]


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_omitted_mode_resolves_profile_default_and_explicit_major_is_preserved(profile):
    tok = EventTokenizer()
    assert make_guidance(tok, profile).mode == PROFILE_SPECS[profile].default_mode
    assert make_guidance(tok, profile, mode=None).mode == PROFILE_SPECS[profile].default_mode
    assert make_guidance(tok, profile, mode="major").mode == "major"
    assert make_guidance(tok, None, mode=None) is None


def test_direct_cinematic_generation_uses_minor_unless_explicitly_overridden():
    tok = EventTokenizer()
    model = StaticModel(tok, {75: 0, 76: 0})
    default = fixed_generation(tok, model, profile="cinematic", repetition_penalty=1)
    explicit_minor = fixed_generation(tok, model, profile="cinematic",
                                      mode="natural-minor", repetition_penalty=1)
    explicit_major = fixed_generation(tok, model, profile="cinematic",
                                      mode="major", repetition_penalty=1)
    assert default == explicit_minor
    assert tok.decode(default["token_ids"]).events[0].pitch == 75
    assert tok.decode(explicit_major["token_ids"]).events[0].pitch == 76


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_every_profile_is_soft_preserves_structure_and_does_not_mutate_logits(profile):
    tok = EventTokenizer()
    guide = make_guidance(tok, profile)
    logits = torch.linspace(-4, 4, tok.vocab_size)
    original = logits.clone()
    allowed = tok.event_start_ids + tok.duration_ids + tok.bpm_ids + [tok.eos_id]
    guided = guide.apply(logits, allowed, [pitch_id(tok, 72), tok.rest_id])
    assert torch.isfinite(guided).all()
    assert torch.equal(logits, original)
    assert guided[tok.eos_id] == original[tok.eos_id]
    assert torch.equal(guided[tok.bpm_ids], original[tok.bpm_ids])
    assert guided[tok.pad_id] == original[tok.pad_id]
    assert guided[tok.bos_id] == original[tok.bos_id]
    # Guidance can never alter a pitch absent from the legal support.
    limited = guide.apply(logits, [tok.rest_id], [tok.rest_id])
    assert torch.equal(limited[tok.pitch_ids], original[tok.pitch_ids])


def profile_scores(tok, profile, history=()):
    return make_guidance(tok, profile).apply(
        torch.zeros(tok.vocab_size), tok.event_start_ids + tok.duration_ids, list(history))


def test_chiptune_favors_higher_register_wider_jumps_and_quick_notes():
    tok = EventTokenizer()
    initial = profile_scores(tok, "chiptune")
    assert initial[pitch_id(tok, 96)] > initial[pitch_id(tok, 60)]
    jump = profile_scores(tok, "chiptune", [pitch_id(tok, 84)])
    assert jump[pitch_id(tok, 96)] > jump[pitch_id(tok, 86)]
    assert jump[duration_id(tok, 16)] > jump[duration_id(tok, 2)]
    # The same octave jump is discouraged by the original pop guide.
    pop = profile_scores(tok, "pop-hook", [pitch_id(tok, 84)])
    assert pop[pitch_id(tok, 86)] > pop[pitch_id(tok, 96)]


def test_cinematic_favors_lower_sustained_phrases_and_more_breathing_room():
    tok = EventTokenizer()
    cinema = profile_scores(tok, "cinematic")
    dance = profile_scores(tok, "dance")
    assert cinema[pitch_id(tok, 60)] > cinema[pitch_id(tok, 96)]
    assert cinema[duration_id(tok, 2)] > cinema[duration_id(tok, 16)]
    assert cinema[tok.rest_id] > dance[tok.rest_id]
    rest = profile_scores(tok, "cinematic", [tok.rest_id])
    assert rest[duration_id(tok, 2)] > rest[duration_id(tok, 16)]


def test_dance_favors_even_motion_and_lullaby_favors_small_steps():
    tok = EventTokenizer()
    dance = profile_scores(tok, "dance")
    assert dance[duration_id(tok, 8)] > dance[duration_id(tok, 8, True)]
    assert dance[duration_id(tok, 8)] > dance[duration_id(tok, 1)]
    gentle = profile_scores(tok, "lullaby", [pitch_id(tok, 60)])
    assert gentle[pitch_id(tok, 62)] > gentle[pitch_id(tok, 72)]
    assert gentle[duration_id(tok, 2)] > gentle[duration_id(tok, 16)]
    assert gentle[tok.rest_id] > dance[tok.rest_id]


def test_profiles_have_distinct_pitch_and_rhythm_preferences_at_identical_tempo():
    tok = EventTokenizer()
    vectors = {name: profile_scores(tok, name, [pitch_id(tok, 72)])
               for name in PROFILE_NAMES}
    # This compares only note/rhythm scores: tempo differences cannot satisfy it.
    for i, name in enumerate(PROFILE_NAMES):
        for other in PROFILE_NAMES[i + 1:]:
            assert not torch.equal(vectors[name][tok.pitch_ids], vectors[other][tok.pitch_ids])
            assert not torch.equal(vectors[name][tok.duration_ids], vectors[other][tok.duration_ids])


def test_pop_compatible_entry_point_retains_known_original_bias_values():
    tok = EventTokenizer()
    guide = PopHookGuidance(tok, "C", "major")
    assert isinstance(make_guidance(tok, "pop-hook"), PopHookGuidance)
    baseline = torch.zeros(tok.vocab_size)
    after_c = guide.apply(baseline, tok.event_start_ids + tok.duration_ids, [pitch_id(tok, 72)])
    expected = {
        pitch_id(tok, 72): 0.55,  # Scale and original register, no unison bonus.
        pitch_id(tok, 73): 0.05,  # Chromatic note plus step bonus.
        pitch_id(tok, 74): 0.75,
        pitch_id(tok, 76): 0.65,
        pitch_id(tok, 84): 0.13,  # Original octave-leap penalty.
        tok.rest_id: -0.15,
        duration_id(tok, 1): -0.15,
        duration_id(tok, 4): 0.30,
        duration_id(tok, 16): 0.10,
    }
    for token, score in expected.items():
        assert after_c[token].item() == pytest.approx(score, abs=1e-7)
