"""Runtime contracts for optional naming; no model downloads or real LLM inference."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rtttl_gen import local_llm
from rtttl_gen.rtttl import parse_rtttl


SONG = parse_rtttl("Generated00:d=8,o=6,b=112:c,e,g,e,d,c,4g5")


@pytest.fixture
def fake_devices(monkeypatch):
    """Advertise hardware without allocating any accelerator tensors."""
    states = {}
    cache_calls = []

    def configure(*available, cleanup_error=None):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: "cuda" in available)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: "mps" in available)
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: "xpu" in available)
        for name in available:
            backend = getattr(torch, name)
            states[name] = torch.tensor([1234], dtype=torch.int64)

            def get_state(device=name):
                return states[device].clone()

            def set_state(value, device=name):
                states[device] = value.clone()

            def seed(value, device=name):
                states[device] = torch.tensor([value], dtype=torch.int64)

            def empty_cache(device=name):
                cache_calls.append(device)
                if cleanup_error is not None:
                    raise cleanup_error

            monkeypatch.setattr(backend, "get_rng_state", get_state)
            monkeypatch.setattr(backend, "set_rng_state", set_state)
            monkeypatch.setattr(backend, "manual_seed", seed)
            monkeypatch.setattr(backend, "empty_cache", empty_cache)
        return SimpleNamespace(states=states, cache_calls=cache_calls)

    configure()
    return configure


@pytest.fixture
def fake_hf(monkeypatch):
    """Minimal HF objects, with controllable load and generation failures."""
    record = SimpleNamespace(tokenizer_loads=[], model_loads=[], models=[],
                             input_devices=[], messages=[], decoded=[],
                             init_failure=None, inference_failure=None)

    class Batch(dict):
        def to(self, device):
            record.input_devices.append(device)
            return self

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            record.messages.append((messages, kwargs))
            return "test prompt"

        def __call__(self, prompt, **kwargs):
            assert prompt == "test prompt"
            assert kwargs == {"return_tensors": "pt"}
            return Batch(input_ids=torch.tensor([[11, 12, 13]]),
                         attention_mask=torch.ones((1, 3), dtype=torch.long))

        def decode(self, tokens, **kwargs):
            record.decoded.append((tokens.tolist(), kwargs))
            return "  Amber Sunrise  "

    class Model:
        def __init__(self, revision):
            self.config = SimpleNamespace(_commit_hash=revision)
            self.device = None
            self.evaluated = False
            self.generations = []

        def to(self, device):
            self.device = device
            if record.init_failure is not None:
                record.init_failure(device)
            return self

        def eval(self):
            self.evaluated = True
            return self

        def generate(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            self.generations.append(kwargs)
            torch.rand(3)  # Simulate a stochastic sampler consuming CPU RNG.
            if record.inference_failure is not None:
                record.inference_failure(self.device)
            return torch.cat((kwargs["input_ids"], torch.tensor([[21, 22]])), dim=1)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            record.tokenizer_loads.append((model_id, kwargs))
            return Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            torch.rand(3)  # Loading must preserve the caller's RNG as well.
            record.model_loads.append((model_id, kwargs))
            model = Model(kwargs["revision"])
            record.models.append(model)
            return model

    module = ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer
    module.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", module)
    return record


@pytest.mark.parametrize("available,expected", [
    ((), "cpu"), (("mps",), "mps"), (("cuda",), "cuda"),
    (("xpu",), "xpu"), (("cuda", "mps", "xpu"), "cuda"),
])
def test_auto_selects_available_accelerator_then_cpu(fake_devices, fake_hf,
                                                     available, expected):
    fake_devices(*available)
    model = local_llm.LocalTitleModel()
    assert model.device == expected
    assert fake_hf.models[0].device == expected
    assert fake_hf.models[0].evaluated
    assert model.metadata()["fallbacks"] == []
    assert model.metadata()["dtype"] == ("float32" if expected == "cpu" else "float16")


@pytest.mark.parametrize("device", ["cuda", "mps", "xpu"])
def test_explicit_unavailable_device_fails_before_optional_import(fake_devices,
                                                                 monkeypatch, device):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        local_llm.LocalTitleModel(device=device)


def test_explicit_cpu_overrides_available_accelerators(fake_devices, fake_hf):
    backend = fake_devices("cuda", "mps", "xpu")
    model = local_llm.LocalTitleModel(device="cpu")
    assert model.device == "cpu"
    assert fake_hf.model_loads[0][1]["dtype"] is torch.float32
    assert model.metadata()["fallbacks"] == []
    model.close()
    assert backend.cache_calls == []


def test_invalid_device_does_not_import_optional_dependencies(fake_devices, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="Naming device"):
        local_llm.LocalTitleModel(device="vulkan")


def test_missing_dependency_has_actionable_install_message(fake_devices, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="requirements-naming.txt"):
        local_llm.LocalTitleModel(device="cpu")


def test_importing_runtime_does_not_import_transformers():
    # A fresh interpreter avoids a false pass due to pytest's import cache.
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import builtins
import torch
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name == 'transformers' or name.startswith('transformers.'):
        raise AssertionError('optional Transformers imported eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
import rtttl_gen.local_llm
"""
    env = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("offline", [False, True])
def test_model_and_tokenizer_share_pinned_private_loading_options(fake_devices,
                                                                 fake_hf, tmp_path,
                                                                 offline):
    model = local_llm.LocalTitleModel(device="cpu", offline=offline, cache_dir=tmp_path)
    tokenizer_id, tokenizer_kwargs = fake_hf.tokenizer_loads[0]
    model_id, model_kwargs = fake_hf.model_loads[0]
    assert tokenizer_id == model_id == local_llm.DEFAULT_MODEL
    common = {"revision": local_llm.DEFAULT_REVISION,
              "local_files_only": offline, "cache_dir": str(tmp_path),
              "trust_remote_code": False, "token": False}
    assert tokenizer_kwargs == common
    assert all(model_kwargs[key] == value for key, value in common.items())
    assert model_kwargs["use_safetensors"] is True
    assert model_kwargs["dtype"] is torch.float32
    assert "device_map" not in model_kwargs
    assert model.metadata()["resolved_revision"] == local_llm.DEFAULT_REVISION
    assert model.metadata()["offline"] is offline


@pytest.mark.parametrize("model_id,revision,expected_revision", [
    (local_llm.DEFAULT_MODEL, "explicit-revision", "explicit-revision"),
    ("local-test/other-model", "custom-revision", "custom-revision"),
    ("local-test/other-model", None, None),
])
def test_revision_override_does_not_apply_qwen_pin_to_other_models(fake_devices,
                                                                 fake_hf, model_id,
                                                                 revision, expected_revision):
    model = local_llm.LocalTitleModel(model_id=model_id, revision=revision, device="cpu")
    assert model.revision == expected_revision
    assert fake_hf.tokenizer_loads[0][1]["revision"] == expected_revision
    assert fake_hf.model_loads[0][1]["revision"] == expected_revision


@pytest.mark.parametrize("failure", [RuntimeError("driver unavailable"),
                                     NotImplementedError("unsupported operation")])
def test_auto_falls_back_after_accelerator_initialization_failure(fake_devices,
                                                                 fake_hf, failure):
    backend = fake_devices("mps")

    def fail_accelerator(device):
        if device != "cpu":
            raise failure

    fake_hf.init_failure = fail_accelerator
    model = local_llm.LocalTitleModel()
    assert [item[1]["dtype"] for item in fake_hf.model_loads] == [torch.float16, torch.float32]
    assert model.device == "cpu"
    assert len(fake_hf.tokenizer_loads) == 1
    assert backend.cache_calls == ["mps"]
    fallback, = model.metadata()["fallbacks"]
    assert fallback["from"] == "mps" and fallback["to"] == "cpu"
    assert type(failure).__name__ in fallback["reason"]
    assert str(failure) in fallback["reason"]


def test_failed_accelerator_cache_cleanup_does_not_prevent_cpu_fallback(fake_devices,
                                                                      fake_hf):
    fake_devices("cuda", cleanup_error=RuntimeError("CUDA driver unavailable"))

    def fail_accelerator(device):
        if device == "cuda":
            raise RuntimeError("CUDA initialization failed")

    fake_hf.init_failure = fail_accelerator
    model = local_llm.LocalTitleModel()
    assert model.device == "cpu"
    assert model.title(SONG, used_titles=[], seed=3) == "Amber Sunrise"


def test_auto_retries_inference_on_cpu_and_records_fallback(fake_devices, fake_hf):
    fake_devices("mps")

    def fail_accelerator(device):
        if device == "mps":
            raise RuntimeError("MPS out of memory")

    fake_hf.inference_failure = fail_accelerator
    model = local_llm.LocalTitleModel()
    before = torch.get_rng_state().clone()
    assert model.title(SONG, used_titles=["Earlier Title"], seed=71) == "Amber Sunrise"
    assert torch.equal(before, torch.get_rng_state())
    assert model.device == "cpu"
    assert fake_hf.input_devices == ["mps", "cpu"]
    assert len(fake_hf.model_loads) == 2
    assert model.metadata()["fallbacks"][0]["reason"] == "RuntimeError: MPS out of memory"
    assert len(fake_hf.messages) == 2
    assert fake_hf.messages[0] == fake_hf.messages[1]


@pytest.mark.parametrize("stage", ["init", "inference"])
def test_explicit_accelerator_failure_never_silently_switches_to_cpu(fake_devices,
                                                                   fake_hf, stage):
    fake_devices("mps")

    def fail(device):
        raise RuntimeError("explicit MPS failure")

    if stage == "init":
        fake_hf.init_failure = fail
        with pytest.raises(RuntimeError, match="explicit MPS failure"):
            local_llm.LocalTitleModel(device="mps")
    else:
        fake_hf.inference_failure = fail
        model = local_llm.LocalTitleModel(device="mps")
        with pytest.raises(RuntimeError, match="explicit MPS failure"):
            model.title(SONG, used_titles=[], seed=2)
        assert model.metadata()["fallbacks"] == []
    assert len(fake_hf.model_loads) == 1


def test_cpu_failure_is_not_retried_indefinitely(fake_devices, fake_hf):
    def fail(device):
        raise RuntimeError("CPU failed")

    fake_hf.inference_failure = fail
    model = local_llm.LocalTitleModel()
    with pytest.raises(RuntimeError, match="CPU failed"):
        model.title(SONG, used_titles=[], seed=2)
    assert len(fake_hf.model_loads) == 1
    assert len(fake_hf.models[0].generations) == 1


def test_batch_reuses_model_and_decodes_only_generated_tokens(fake_devices, fake_hf):
    model = local_llm.LocalTitleModel(device="cpu")
    for seed in (1, 2):
        assert model.title(SONG, used_titles=["Previous Song"], seed=seed) == "Amber Sunrise"
    assert len(fake_hf.model_loads) == len(fake_hf.tokenizer_loads) == 1
    assert fake_hf.decoded == [([21, 22], {"skip_special_tokens": True})] * 2
    assert len(fake_hf.models[0].generations) == 2
    messages, template_kwargs = fake_hf.messages[0]
    assert "Previous Song" in messages[1]["content"]
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["tokenize"] is False
    model.close()
    model.close()  # Closing an already closed model is safe.
    assert model.model is None
    with pytest.raises(RuntimeError, match="closed"):
        model.title(SONG, used_titles=[], seed=3)


def test_model_loading_and_sampling_preserve_cpu_rng(fake_devices, fake_hf):
    before = torch.get_rng_state().clone()
    model = local_llm.LocalTitleModel(device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    model.title(SONG, used_titles=[], seed=17)
    assert torch.equal(before, torch.get_rng_state())


def test_isolated_cpu_seed_is_reproducible_and_restores_after_exception():
    before = torch.get_rng_state().clone()
    with local_llm.isolated_seed("cpu", 42):
        first = torch.rand(4)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="test interruption"):
        with local_llm.isolated_seed("cpu", 42):
            assert torch.equal(first, torch.rand(4))
            raise ValueError("test interruption")
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Apple MPS")
def test_cpu_naming_does_not_reseed_actual_mps(fake_devices, fake_hf):
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone()
    model = local_llm.LocalTitleModel(device="cpu")
    model.title(SONG, used_titles=[], seed=51)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_mps, torch.mps.get_rng_state())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Apple MPS")
def test_actual_mps_rng_is_restored_without_any_llm_inference():
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone()
    with local_llm.isolated_seed("mps", 31):
        first = torch.rand(4, device="mps").cpu()
        torch.rand(3)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_mps, torch.mps.get_rng_state())
    with pytest.raises(ValueError, match="test interruption"):
        with local_llm.isolated_seed("mps", 31):
            assert torch.equal(first, torch.rand(4, device="mps").cpu())
            raise ValueError("test interruption")
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_mps, torch.mps.get_rng_state())
