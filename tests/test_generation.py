"""Regression coverage for musical repetition controls at sampling time."""

from itertools import groupby

import pytest
import torch

from rtttl_gen.generation import generate_tokens
from rtttl_gen.tokenizer import EventTokenizer


class ScriptedMelody:
    """Prefer a requested melody, with a lower-scoring escape note available."""

    def __init__(self, tokenizer, pitches, varying_durations=False):
        self.tokenizer = tokenizer
        self.pitches = pitches
        self.varying_durations = varying_durations

    def logits_for_event(self, index):
        tok = self.tokenizer
        logits = torch.full((tok.vocab_size,), -100.0)
        logits[tok.token_to_id["<BPM_120>"]] = 10.0
        logits[tok.token_to_id["<BPM_121>"]] = 9.0
        logits[tok.token_to_id["<PITCH_107>"]] = 9.0
        pitch = self.pitches[index % len(self.pitches)]
        preferred = tok.rest_id if pitch is None else tok.token_to_id[f"<PITCH_{pitch}>"]
        logits[preferred] = 10.0
        duration = tok.duration_ids[index % len(tok.duration_ids)] if self.varying_durations else tok.token_to_id["<DUR_8_dot>"]
        logits[tok.token_to_id["<DUR_4_plain>"]] = 9.0
        logits[duration] = 10.0
        return logits

    def next_logits(self, ids):
        return self.logits_for_event(max(0, (len(ids) - 2) // 2))


def generate_fixed(model, tok, count, **kwargs):
    return generate_tokens(model, tok, min_events=count, max_events=count,
                           top_k=1, top_p=0.01, bpm=120, **kwargs)


def pitches_from(tok, result):
    return [event.pitch for event in tok.decode(result["token_ids"]).events]


@pytest.mark.parametrize("pitch", [60, None])
def test_single_pitch_and_rest_collapse_breaks_before_greedy_filtering(pitch):
    tok = EventTokenizer()
    result = generate_fixed(ScriptedMelody(tok, [pitch]), tok, 24,
                            repetition_penalty=1, max_pitch_run=4,
                            max_motif_repeats=0)
    pitches = pitches_from(tok, result)
    assert pitches[:4] == [pitch] * 4
    assert pitches[4] != pitch
    assert max(len(list(run)) for _, run in groupby(pitches)) <= 4
    assert result["forced_eos"]
    assert result["repetition_interventions"]["pitch_run_blocks"] > 0
    assert result["repetition_interventions"]["motif_blocks"] == 0


@pytest.mark.parametrize("motif", [[60, 62], [60, None, 64], list(range(60, 65)),
                                   list(range(60, 68)), list(range(60, 76))])
def test_nonconstant_pitch_motif_breaks_after_limit_with_varied_intro_and_rhythm(motif):
    tok = EventTokenizer()
    intro = [91, 94, 89, 96]
    scripted = intro + motif * 4
    count = len(intro) + len(motif) * 3 + 1
    result = generate_fixed(ScriptedMelody(tok, scripted, varying_durations=True), tok, count,
                            repetition_penalty=1, max_pitch_run=0,
                            max_motif_repeats=3, max_motif_length=16)
    pitches = pitches_from(tok, result)
    assert pitches[:-1] == intro + motif * 3
    assert pitches[-1] != motif[0]
    assert result["repetition_interventions"]["motif_blocks"] > 0
    # Musical rhythm remains intact even when the pitch continuation is blocked.
    assert result["token_ids"][3:-1:2] == [tok.duration_ids[i % len(tok.duration_ids)]
                                          for i in range(count)]


def test_motif_length_setting_controls_longer_patterns():
    tok = EventTokenizer()
    motif = [60, 62, 64, 65, 67]
    model = ScriptedMelody(tok, motif)
    common = dict(repetition_penalty=1, max_pitch_run=0, max_motif_repeats=3)
    outside_limit = generate_fixed(model, tok, 16, max_motif_length=4, **common)
    inside_limit = generate_fixed(model, tok, 16, max_motif_length=5, **common)
    assert pitches_from(tok, outside_limit) == motif * 3 + motif[:1]
    assert pitches_from(tok, inside_limit)[:15] == motif * 3
    assert pitches_from(tok, inside_limit)[15] != motif[0]


def test_musical_reprises_are_allowed_without_blanket_ngram_ban():
    tok = EventTokenizer()
    motif = [60, 62, 64]
    melody = motif * 2 + [65, 67] + motif * 2
    result = generate_fixed(ScriptedMelody(tok, melody), tok, len(melody),
                            repetition_penalty=1, max_pitch_run=0,
                            max_motif_repeats=3)
    assert pitches_from(tok, result) == melody
    assert result["repetition_interventions"]["motif_blocks"] == 0


class ConstantScores:
    def __init__(self, tokenizer, first, second, preferred=60):
        self.logits = ScriptedMelody(tokenizer, [preferred]).logits_for_event(0)
        self.logits[tokenizer.token_to_id["<PITCH_107>"]] = -100
        first_id = tokenizer.rest_id if preferred is None else tokenizer.token_to_id[f"<PITCH_{preferred}>"]
        self.logits[first_id] = first
        self.logits[tokenizer.token_to_id["<PITCH_62>"]] = second

    def next_logits(self, ids):
        # Returning the same tensor also checks that generation does not mutate it.
        return self.logits


@pytest.mark.parametrize("first,second", [(10.0, 9.0), (-1.0, -1.1)])
@pytest.mark.parametrize("preferred", [60, None])
def test_recent_pitch_penalty_is_sign_aware_and_preserves_duration(first, second, preferred):
    tok = EventTokenizer()
    model = ConstantScores(tok, first, second, preferred)
    original = model.logits.clone()
    result = generate_fixed(model, tok, 3, repetition_penalty=1.2,
                            repetition_window=1, max_pitch_run=0,
                            max_motif_repeats=0)
    assert pitches_from(tok, result) == [preferred, 62, preferred]
    song = tok.decode(result["token_ids"])
    assert [(event.duration, event.dotted) for event in song.events] == [(8, True)] * 3
    assert song.bpm == 120
    assert result["repetition_interventions"]["penalty_steps"] > 0
    assert result["repetition_interventions"]["pitch_run_blocks"] == 0
    assert result["repetition_interventions"]["motif_blocks"] == 0
    torch.testing.assert_close(model.logits, original)


def test_penalty_does_not_suppress_natural_eos():
    tok = EventTokenizer()
    model = ConstantScores(tok, 10, 9)
    model.logits[tok.eos_id] = 9.5
    result = generate_tokens(model, tok, min_events=1, max_events=5, top_k=1,
                             repetition_penalty=2, max_pitch_run=0,
                             max_motif_repeats=0, bpm=120)
    assert len(tok.decode(result["token_ids"]).events) == 1
    assert not result["forced_eos"]


class LegacyProbe:
    def __init__(self, tok):
        self.tok = tok

    def next_logits(self, ids):
        values = torch.sin(torch.arange(self.tok.vocab_size, dtype=torch.float32)
                           * 0.137 + sum(ids[-3:]) * 0.011)
        values[self.tok.eos_id] = -5
        return values


@pytest.mark.parametrize("seed,bpm,event_tokens", [
    (19, 125, [(99, "16_dot"), (88, "16_plain"), (67, "32_dot"), (72, "32_plain"),
               (72, "32_dot"), (71, "8_dot"), (70, "32_plain"), (70, "16_plain")]),
    (72, 538, [(70, "32_dot"), (103, "16_dot"), (69, "16_plain"), (70, "16_dot"),
               (74, "16_dot"), (73, "8_plain"), (69, "8_dot"), (72, "32_plain")]),
])
def test_disabling_controls_preserves_captured_legacy_sampler(seed, bpm, event_tokens):
    # Captured by running the pre-change sampler; includes stochastic tempo,
    # pitches, durations, top-k and nucleus sampling, and repeated pitch values.
    tok = EventTokenizer()
    result = generate_tokens(LegacyProbe(tok), tok, min_events=8, max_events=8,
                             temperature=0.9, top_k=7, top_p=0.83, seed=seed,
                             repetition_penalty=1, max_pitch_run=0,
                             max_motif_repeats=0)
    expected = ["<BOS>", f"<BPM_{bpm}>"]
    for pitch, duration in event_tokens:
        expected.extend([f"<PITCH_{pitch}>", f"<DUR_{duration}>"])
    expected.append("<EOS>")
    assert tok.token_strings(result["token_ids"]) == expected
    assert result["forced_eos"]
    assert result["repetition_interventions"] == {
        "penalty_steps": 0, "pitch_run_blocks": 0, "motif_blocks": 0,
    }


def test_enabled_controls_are_seed_deterministic_and_do_not_consume_global_rng():
    tok = EventTokenizer()
    model = LegacyProbe(tok)
    state = torch.get_rng_state().clone()
    first = generate_tokens(model, tok, min_events=40, max_events=40, seed=17)
    second = generate_tokens(model, tok, min_events=40, max_events=40, seed=17)
    assert first == second
    assert torch.equal(torch.get_rng_state(), state)


class ShortContextMelody(torch.nn.Module):
    def __init__(self, tok, pitches):
        super().__init__()
        self.context_length = 6
        self.script = ScriptedMelody(tok, pitches)
        self.contexts = []

    def forward(self, ids):
        index = len(self.contexts) // 2  # A supplied BPM leaves pitch/duration pairs.
        self.contexts.append(ids[0].tolist())
        logits = self.script.logits_for_event(index).to(ids.device)
        return logits.expand(ids.shape[0], ids.shape[1], -1)


def test_repetition_history_survives_neural_context_truncation():
    tok = EventTokenizer()
    motif = [60, 62, 64, 65, 67]
    model = ShortContextMelody(tok, motif)
    model.train()
    result = generate_fixed(model, tok, 31, repetition_penalty=1,
                            max_pitch_run=0, max_motif_repeats=3)
    pitches = pitches_from(tok, result)
    assert pitches[:15] == motif * 3
    assert pitches[15] != motif[0]
    assert len(pitches) == 31
    assert all(len(context) <= model.context_length for context in model.contexts)
    assert model.training
    assert result["repetition_interventions"]["motif_blocks"] > 0


@pytest.mark.parametrize("training", [True, False])
def test_model_mode_is_restored_after_generation_error(training):
    tok = EventTokenizer()

    class BrokenModel(ShortContextMelody):
        def forward(self, ids):
            assert not self.training
            raise RuntimeError("model failure")

    model = BrokenModel(tok, [60])
    model.train(training)
    with pytest.raises(RuntimeError, match="model failure"):
        generate_fixed(model, tok, 5)
    assert model.training is training


@pytest.mark.parametrize("kwargs", [
    {"repetition_penalty": 0.99}, {"repetition_penalty": float("nan")},
    {"repetition_penalty": float("inf")},
    {"repetition_window": 0}, {"repetition_window": -1},
    {"repetition_window": 1.5}, {"repetition_window": True},
    {"max_pitch_run": -1}, {"max_pitch_run": 2.5}, {"max_pitch_run": True},
    {"max_motif_repeats": 1}, {"max_motif_repeats": -1},
    {"max_motif_repeats": 2.5}, {"max_motif_repeats": True},
    {"max_motif_length": 1}, {"max_motif_length": 3.5}, {"max_motif_length": True},
])
def test_invalid_repetition_options_fail_clearly(kwargs):
    tok = EventTokenizer()
    with pytest.raises(ValueError):
        generate_fixed(ScriptedMelody(tok, [60]), tok, 4, **kwargs)
