"""Optional Hugging Face title model, embedded locally without a server.

Only model/tokenizer files are downloaded. Melody descriptions never leave
this process. No optional dependency is imported until naming is requested.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
from pathlib import Path
from typing import Iterator

import torch

from .rtttl import Song
from . import song_description
from .song_description import make_title_prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TITLE_SAMPLING = dict(max_new_tokens=24, do_sample=True, temperature=0.85,
                      top_p=0.9, repetition_penalty=1.15)


def available_naming_devices() -> list[str]:
    """Accelerators usable by the installed PyTorch build, followed by CPU."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    return devices + ["cpu"]


@contextmanager
def isolated_seed(device: str, seed: int) -> Iterator[None]:
    """Seed naming without changing the melody/training RNG streams."""
    cpu_state = torch.get_rng_state()
    backend = getattr(torch, device) if device != "cpu" else None
    device_state = backend.get_rng_state() if backend else None
    try:
        # torch.manual_seed would also reseed other accelerators. Set only the
        # CPU generator and the selected backend here, restoring both below.
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        if backend:
            backend.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if backend:
            backend.set_rng_state(device_state)


class LocalTitleModel:
    """Load one small safetensors model and reuse it for the whole batch.

    ``auto`` tries an available accelerator, then CPU if that backend fails.
    An explicit device never silently changes to another device. The fallback
    is recorded in metadata. CPU always uses FP32; accelerators use FP16.
    """

    def __init__(self, *, model_id: str = DEFAULT_MODEL, revision: str | None = None,
                 device: str = "auto", offline: bool = False,
                 cache_dir: str | Path | None = None) -> None:
        available = available_naming_devices()
        if device not in ("auto", "cuda", "mps", "xpu", "cpu"):
            raise ValueError("Naming device must be auto, cuda, mps, xpu or cpu")
        if device != "auto" and device not in available:
            raise RuntimeError(f"Naming device {device!r} is unavailable in this PyTorch build")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Optional naming dependencies are missing. Install requirements-naming.txt or pip install '.[naming]'.") from exc
        self.model_id = model_id
        self.revision = revision or (DEFAULT_REVISION if model_id == DEFAULT_MODEL else None)
        self.requested_device = device
        self.device = available[0] if device == "auto" else device
        self.offline = offline
        self.fallbacks: list[dict[str, str]] = []
        self._model_class = AutoModelForCausalLM
        self._load_kwargs = dict(revision=self.revision, local_files_only=offline,
                                 cache_dir=str(cache_dir) if cache_dir else None,
                                 trust_remote_code=False, token=False)
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **self._load_kwargs)
        self._load_with_fallback()

    def _load(self) -> None:
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        with isolated_seed(self.device, 0):
            self.model = self._model_class.from_pretrained(
                self.model_id, **self._load_kwargs, use_safetensors=True,
                dtype=dtype, attn_implementation="eager")
            self.model.to(self.device).eval()

    def _fall_back(self, exc: Exception) -> bool:
        if self.requested_device != "auto" or self.device == "cpu":
            return False
        self.fallbacks.append({"from": self.device, "to": "cpu", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"})
        self.close()
        self.device = "cpu"
        try:
            self._load()
        except Exception:
            self.close()
            raise
        return True

    def _load_with_fallback(self) -> None:
        try:
            self._load()
        except (RuntimeError, NotImplementedError) as exc:
            if not self._fall_back(exc):
                self.close()
                raise

    @torch.inference_mode()
    def _title(self, song: Song, used_titles: list[str], seed: int) -> str:
        messages = [
            {"role": "system", "content": "You name short instrumental melodies. Reply with one evocative English title of 2 to 4 words. No explanation, labels, quotes, or formatting."},
            {"role": "user", "content": make_title_prompt(song, used_titles=used_titles)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with isolated_seed(self.device, seed):
            output = self.model.generate(
                **inputs, **TITLE_SAMPLING,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def title(self, song: Song, *, used_titles: list[str], seed: int) -> str:
        if self.model is None:
            raise RuntimeError("Title model has been closed")
        try:
            return self._title(song, used_titles, seed)
        except (RuntimeError, NotImplementedError) as exc:
            if not self._fall_back(exc):
                raise
            return self._title(song, used_titles, seed)

    def metadata(self) -> dict:
        return {"model": self.model_id, "revision": self.revision,
                "resolved_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
                "runtime": "transformers", "requested_device": self.requested_device,
                "device": self.device, "dtype": "float32" if self.device == "cpu" else "float16",
                "offline": self.offline, "fallbacks": list(self.fallbacks),
                "sampling": dict(TITLE_SAMPLING), "prompt_version": getattr(song_description, "TITLE_PROMPT_VERSION", 1),
                "input": "Measured symbolic melody descriptors; no audio or cloud inference",
                "use_safetensors": True, "trust_remote_code": False}

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.device != "cpu":
            backend = getattr(torch, self.device)
            if hasattr(backend, "empty_cache"):
                try:
                    backend.empty_cache()
                except (RuntimeError, NotImplementedError):
                    # Cleanup must not prevent CPU fallback after a driver or
                    # accelerator initialization failure.
                    pass
