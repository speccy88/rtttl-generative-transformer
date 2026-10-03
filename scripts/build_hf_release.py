#!/usr/bin/env python3
"""Build a dataset-free safetensors release locally; never upload anything.

The base release supplies the existing scoped license and corpus provenance.
Checkpoint input must be trusted. Only weights, architecture, fixed vocabulary,
reviewed source/templates, and aggregate measurements are exported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from rtttl_gen.training import load_checkpoint


def flat_imports(source: str) -> str:
    source = source.replace("from . import song_description", "import song_description")
    return re.sub(r"(?m)^from (?:rtttl_gen\.|\.)(\w+) import", r"from \1 import", source)


def selected_functions(path: Path, names: tuple[str, ...], imports: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in tree.body if isinstance(node, ast.FunctionDef)}
    return imports + "\n\n" + "\n\n\n".join(functions[name] for name in names) + "\n"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_release(checkpoint: Path, base: Path, output: Path, source_commit: str) -> dict:
    from safetensors.torch import save_model
    if output.exists():
        raise FileExistsError(output)
    preserved = (".gitattributes", "LICENSE_CODE", "LICENSE_SCOPE.md", "PROVENANCE.md")
    for name in (*preserved, "training_config.json", "evaluation.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    continuation = json.loads((ROOT / "results/mps_finetune.json").read_text())
    expected = continuation["predictive"]["finetuned"]["checkpoint_sha256"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint does not match the published continuation measurements")
    model, tokenizer, data = load_checkpoint(checkpoint, "cpu")
    if data["model_type"] != "transformer":
        raise ValueError("This release template requires a Transformer checkpoint")
    output.mkdir(parents=True)
    for name in preserved:
        shutil.copy2(base / name, output / name)
    shutil.copy2(base / "training_config.json", output / "original_training_config.json")
    shutil.copy2(base / "evaluation.json", output / "original_evaluation.json")
    modules = ("transformer", "tokenizer", "rtttl", "generation", "guidance", "batch_plan",
               "audio", "song_description", "local_llm", "naming")
    for module in modules:
        (output / f"{module}.py").write_text(flat_imports((ROOT / "src/rtttl_gen" / f"{module}.py").read_text()))
    diagnostics = selected_functions(ROOT / "src/rtttl_gen/evaluation.py",
        ("event_key", "_longest_run", "repetition_statistics", "degeneracy_flags"),
        '"""Local repetition diagnostics; no training-reference comparisons."""\n'
        'from __future__ import annotations\nfrom collections import Counter\n'
        'from typing import Any, Sequence\nfrom rtttl import Song\n')
    (output / "diagnostics.py").write_text(diagnostics)
    options = selected_functions(ROOT / "generate.py",
        ("add_sampling_arguments", "add_naming_arguments", "sampling_kwargs", "apply_optional_naming"),
        '"""Generation options shared with the source project."""\n'
        'from __future__ import annotations\nimport argparse\nimport sys\n'
        'from guidance import PROFILE_NAMES\n')
    (output / "options.py").write_text(flat_imports(options))
    for name in ("inference.py", "test_inference.py", "README.md"):
        shutil.copy2(ROOT / "templates/huggingface" / name, output / name)
    (output / "requirements.txt").write_text('torch==2.6.0\nsafetensors==0.8.0\nnumpy>=1.26,<3\n')
    shutil.copy2(ROOT / "requirements-naming.txt", output / "requirements-naming.txt")
    save_model(model, str(output / "model.safetensors"), metadata={"format": "pt"})
    write_json(output / "config.json", model.config)
    write_json(output / "tokenizer.json", tokenizer.to_dict())
    for name in ("mps_finetune", "repetition_comparison_mps", "naming_validation", "variety_validation"):
        shutil.copy2(ROOT / "results" / f"{name}.json", output / f"{name}.json")
    write_json(output / "evaluation.json", {
        "checkpoint": "model.safetensors",
        "predictive": {"validation": continuation["predictive"]["finetuned"]["metrics"], "test": None},
        "generation_comparison": continuation["generation_comparison"],
        "scope": "Current checkpoint validation and paired generation; historical test results belong to original_evaluation.json."
    })
    write_json(output / "training_config.json", {
        "phase": "Three-epoch Apple M2 MPS warm start from original checkpoint",
        "model": data["model_config"], "training": data["config"]["training"],
        "seed": data["config"].get("seed"),
        "augment_semitones": data["config"].get("data", {}).get("augment_semitones"),
        "dataset_sha256": data["dataset_sha256"],
        "scope": "Weights-only inference export; optimizer, RNG, corpus and absolute paths omitted."
    })
    sha = hashlib.sha256((output / "model.safetensors").read_bytes()).hexdigest()
    readme = (output / "README.md").read_text().replace("{{SOURCE_COMMIT}}", source_commit).replace("{{WEIGHTS_SHA256}}", sha)
    (output / "README.md").write_text(readme)
    with (output / "LICENSE_SCOPE.md").open("a") as f:
        f.write('\n\nOptional title generation uses separately downloaded Qwen2.5-0.5B-Instruct '
                'weights; that repository declares Apache-2.0. Its weights are not bundled '
                'here. Transformers, NumPy and other dependencies retain their own licenses.\n')
    with (output / "PROVENANCE.md").open("a") as f:
        f.write('\n\n## MPS continuation release\n\nThe current weights continue the original experiment on the unchanged private '
                'splits for three FP32 epochs on Apple M2 MPS. See `mps_finetune.json` and '
                '`original_training_config.json`. The corpus remains omitted. No genre labels '
                'were added; melody profiles are handcrafted decoding preferences. The optional '
                'title model is downloaded separately and has its own license.\n')
    result = {"weights_sha256": sha, "weights_bytes": (output / "model.safetensors").stat().st_size,
              "source_commit": source_commit, "parameters": model.parameter_count()}
    write_json(output / "RELEASE.json", result)
    return result


def write_checksums(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
                   and "__pycache__" not in path.parts)
    (output / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}\n"
        for path in files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = build_release(args.checkpoint, args.base_release, args.output, commit)
    write_checksums(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
