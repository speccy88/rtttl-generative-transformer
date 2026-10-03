"""Actual CPU optimization, reload, exact-resume, and loss-denominator checks."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from rtttl_gen.rtttl import parse_rtttl, song_to_dict, encode_rtttl
from rtttl_gen.tokenizer import EventTokenizer
from rtttl_gen.training import evaluate_loss, load_checkpoint, train
from rtttl_gen.generation import generate_tokens


def write_data(directory: Path):
    directory.mkdir()
    examples = ["a:d=8,o=5,b=120:c,e,g,4c6", "b:d=4,o=5,b=140:d,p,f#,a,2d6", "c:d=8,o=5,b=120:g,e,c", "d:d=4,o=5,b=140:a,g,e,c,p"]
    rows = [{"id": f"s{i}", "family_id": f"f{i}", "source": f"fixture{i}.rtttl",
             "song": song_to_dict(parse_rtttl(s))} for i,s in enumerate(examples)]
    for split, subset in (("train", rows), ("val", rows[:2]), ("test", rows[2:])):
        # Split overlap is intentional for this isolated training mechanics unit test.
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in subset))
    (directory / "tokenizer.json").write_text(json.dumps(EventTokenizer().to_dict()))


def tiny_config(directory: Path, output: Path):
    return {"seed": 17, "deterministic": True, "output_root": str(output),
        "data": {"processed_dir": str(directory), "augment_semitones": 0},
        "model": {"type": "transformer", "context_length": 16, "d_model": 16,
                  "n_layers": 1, "n_heads": 2, "dropout": 0.1},
        "training": {"device": "cpu", "mixed_precision": "off", "batch_size": 2,
                     "gradient_accumulation_steps": 1, "epochs": 2, "learning_rate": 0.001,
                     "weight_decay": 0.01, "eval_every_steps": 1, "warmup_steps": 0,
                     "early_stopping_patience": 50, "cpu_threads": 1,
                     "sample_every_evaluations": 0}, "samples": {"count": 0}}


def test_evaluate_loss_counts_non_pad_targets():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((*x.shape, 5))
    items = [{"input_ids": torch.tensor([1,2,0]), "targets": torch.tensor([2,3,0])},
             {"input_ids": torch.tensor([1,0,0]), "targets": torch.tensor([4,0,0])}]
    metrics = evaluate_loss(Fixed(), DataLoader(items, batch_size=1), "cpu", 0)
    assert metrics["token_count"] == 3
    assert metrics["cross_entropy"] == pytest.approx(math.log(5))
    assert metrics["perplexity"] == pytest.approx(5)


def test_training_checkpoint_generation_and_exact_resume(tmp_path):
    data = tmp_path / "data"; write_data(data)
    config = tiny_config(data, tmp_path / "runs")
    full = train(config)
    interrupted_config = copy.deepcopy(config)
    interrupted_config["training"]["stop_after_steps"] = 1
    interrupted = train(interrupted_config)
    resumed = train(config, resume=interrupted / "checkpoint_last.pt")
    assert full != interrupted != resumed
    full_model, tok, full_cp = load_checkpoint(full / "checkpoint_last.pt")
    resumed_model, _, resumed_cp = load_checkpoint(resumed / "checkpoint_last.pt")
    assert full_cp["global_step"] == resumed_cp["global_step"] == 4
    for key in full_model.state_dict():
        assert torch.equal(full_model.state_dict()[key], resumed_model.state_dict()[key]), key
    generated = generate_tokens(resumed_model, tok, min_events=2, max_events=5, temperature=1.,
                                top_k=10, top_p=1., bpm=120, seed=1, device="cpu")
    song = tok.decode(generated["token_ids"], name="test")
    reparsed = parse_rtttl(encode_rtttl(song))
    assert reparsed.events == song.events
    assert (resumed / "metrics.csv").exists()
    assert (resumed / "checkpoint_best.pt").exists()
    assert len(list((tmp_path / "runs").iterdir())) == 3


def test_resume_rejects_modified_dataset(tmp_path):
    data = tmp_path / "data"; write_data(data)
    config = tiny_config(data, tmp_path / "runs")
    config["training"]["stop_after_steps"] = 1
    run = train(config)
    with (data / "test.jsonl").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        train(config, resume=run / "checkpoint_last.pt")


def test_token_weighted_gradient_accumulation(tmp_path):
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    cfg["model"]["dropout"] = 0.0
    cfg["training"].update({"epochs": 1, "max_steps": 1, "batch_size": 4, "gradient_clip": 10000.0})
    whole = train(cfg)
    cfg["training"].update({"batch_size": 1, "gradient_accumulation_steps": 4})
    accumulated = train(cfg)
    a, _, _ = load_checkpoint(whole / "checkpoint_last.pt")
    b, _, _ = load_checkpoint(accumulated / "checkpoint_last.pt")
    for key in a.state_dict():
        assert torch.allclose(a.state_dict()[key], b.state_dict()[key], atol=2e-6, rtol=2e-5), key


@pytest.mark.parametrize("kind", ["gru", "ngram"])
def test_baseline_training_and_reload(tmp_path, kind):
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    if kind == "gru":
        cfg["model"] = {"type": "gru", "context_length": 16, "d_model": 16, "n_layers": 1, "dropout": 0.0}
    else:
        cfg["model"] = {"type": "ngram", "order": 5, "alpha": 0.1}
        cfg["data"]["context_length"] = 8  # forces masked-BPM later windows
    cfg["training"]["epochs"] = 1
    run = train(cfg)
    model, tok, checkpoint = load_checkpoint(run / "checkpoint_best.pt")
    assert checkpoint["model_type"] == kind
    generated = generate_tokens(model, tok, min_events=2, max_events=5, temperature=0.8,
                                top_k=10, top_p=1.0, bpm=120, seed=5, device="cpu")
    song = tok.decode(generated["token_ids"], name=kind)
    assert parse_rtttl(encode_rtttl(song)).events == song.events
    summary = json.loads((run / "summary.json").read_text())
    assert summary["training_seconds"] > 0
    if kind == "ngram":
        # 4 train songs, BPM + EOS once each, pitch + duration for every event.
        n_events = sum(len(parse_rtttl(s).events) for s in [
            "a:d=8,o=5,b=120:c,e,g,4c6", "b:d=4,o=5,b=140:d,p,f#,a,2d6",
            "c:d=8,o=5,b=120:g,e,c", "d:d=4,o=5,b=140:a,g,e,c,p"])
        assert summary["train"]["token_count"] == 2 * n_events + 8



def test_best_checkpoint_uses_literal_minimum_and_patience_anchor(tmp_path, monkeypatch):
    import rtttl_gen.training as training
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    cfg["training"]["early_stopping_min_delta"] = 0.1
    losses = iter([2.0, 1.96, 1.89, 1.86])
    def controlled_loss(*args):
        value = next(losses)
        return {"cross_entropy": value, "perplexity": math.exp(value), "token_count": 10}
    monkeypatch.setattr(training, "evaluate_loss", controlled_loss)
    run = train(cfg)
    _, _, checkpoint = load_checkpoint(run / "checkpoint_best.pt")
    assert checkpoint["best_val_loss"] == 1.86  # improvement smaller than min_delta still saved
    assert checkpoint["patience_anchor"] == 1.89
    assert checkpoint["stale_evaluations"] == 1
    cfg["training"]["min_lr_ratio"] = 0.2
    with pytest.raises(ValueError, match="min_lr_ratio"):
        train(cfg, resume=run / "checkpoint_last.pt")


def test_skipped_amp_step_does_not_trigger_periodic_validation(tmp_path, monkeypatch):
    import rtttl_gen.training as training
    class SkipFirstScaler:
        def __init__(self, *args, **kwargs):
            self.value = 1.; self.calls = 0
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def get_scale(self): return self.value
        def step(self, optimizer):
            if self.calls: optimizer.step()
        def update(self):
            if not self.calls: self.value = 0.5
            self.calls += 1
        def state_dict(self): return {}
    monkeypatch.setattr(training.torch.amp, "GradScaler", SkipFirstScaler)
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    cfg["training"]["epochs"] = 1
    run = train(cfg)
    import csv
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["global_step"]) for row in rows] == [1]
