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
from rtttl_gen.training import evaluate_loss, load_checkpoint, select_device, train
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


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected


def test_explicit_mps_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        select_device("mps")


def test_mps_rng_is_saved_and_restored_with_legacy_checkpoint_support(monkeypatch):
    import rtttl_gen.training as training
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    expected = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: expected.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: restored.append(state.clone()))
    state = training._rng_state()
    assert torch.equal(state["mps"], expected)
    training._restore_rng(state)
    assert len(restored) == 1 and torch.equal(restored[0], expected)
    del state["mps"]
    training._restore_rng(state)
    assert len(restored) == 1  # Older CPU/CUDA checkpoints have no MPS state.


def test_sampling_preserves_mps_rng_and_training_mode(tmp_path, monkeypatch):
    import rtttl_gen.generation as generation
    import rtttl_gen.training as training
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    mps_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mps_state[0].clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: mps_state.__setitem__(0, state.clone()))
    tok = EventTokenizer()
    def fake_generate(*args, **kwargs):
        mps_state[0] = torch.tensor([7, 8, 9], dtype=torch.uint8)
        return {"token_ids": tok.encode(parse_rtttl("s:d=4,o=5,b=120:c,e")), "forced_eos": False}
    monkeypatch.setattr(generation, "generate_tokens", fake_generate)
    model = torch.nn.Linear(2, 2)
    training._sample(model, tok, {"samples": {"count": 1}}, tmp_path / "samples.txt", 1, "mps")
    assert torch.equal(mps_state[0], torch.tensor([1, 2, 3], dtype=torch.uint8))
    assert model.training


def test_inference_checkpoint_warm_start_copies_only_weights_and_tracks_provenance(tmp_path):
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    cfg["training"]["stop_after_steps"] = 1
    source_run = train(cfg)
    source_path = source_run / "checkpoint_last.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    inference = {key: source[key] for key in ("format_version", "model_type", "model_config", "model_state_dict", "tokenizer", "dataset_sha256")}
    inference["inference_only"] = True
    inference_path = tmp_path / "inference.pt"
    torch.save(inference, inference_path)
    with pytest.raises(ValueError, match="--init-checkpoint"):
        train(cfg, resume=inference_path)
    cfg["seed"] = 99
    cfg["data"]["augment_semitones"] = 1
    cfg["training"].update(epochs=1, max_steps=1, learning_rate=0.0)
    warm_run = train(cfg, init_checkpoint=inference_path)
    # A full checkpoint passed as initialization must also discard optimizer/RNG/history.
    full_warm_run = train(cfg, init_checkpoint=source_path)
    warm = torch.load(warm_run / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    full_warm = torch.load(full_warm_run / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert warm["global_step"] == 1 and warm["schedule_steps"] == 1
    assert warm_run != source_run != full_warm_run
    for key, value in source["model_state_dict"].items():
        assert torch.equal(warm["model_state_dict"][key], value), key  # LR=0 isolates weight initialization.
        assert torch.equal(full_warm["model_state_dict"][key], value), key
    assert all(int(state["step"]) == 1 for state in warm["optimizer_state_dict"]["state"].values())
    assert torch.equal(warm["rng_state"]["torch"], full_warm["rng_state"]["torch"])
    assert warm["initialization_checkpoint"] == str(inference_path.resolve())
    info = json.loads((warm_run / "model_info.json").read_text())
    provenance = json.loads((warm_run / "provenance.json").read_text())
    assert info["initialization_checkpoint"] == provenance["initialization_checkpoint"] == str(inference_path.resolve())
    assert info["parent_checkpoint"] is None and provenance["parent_checkpoint"] is None
    assert json.loads((warm_run / "summary.json").read_text())["stop_reason"] in {"max_steps", "stop_after_steps"}


def test_warm_start_rejects_incompatible_checkpoint_before_creating_run(tmp_path):
    data = tmp_path / "data"; write_data(data)
    cfg = tiny_config(data, tmp_path / "runs")
    cfg["training"]["stop_after_steps"] = 1
    run = train(cfg)
    checkpoint = run / "checkpoint_last.pt"
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(cfg, resume=checkpoint, init_checkpoint=checkpoint)
    changed = copy.deepcopy(cfg)
    changed["model"]["d_model"] = 32
    with pytest.raises(ValueError, match="architecture and vocabulary"):
        train(changed, init_checkpoint=checkpoint)
    with (data / "test.jsonl").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        train(cfg, init_checkpoint=checkpoint)
    assert list((tmp_path / "runs").iterdir()) == [run]
