"""Check the flat inference export without requiring private checkpoints."""
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_hf_release", ROOT / "scripts/build_hf_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_flat_modules_run_without_source_package(tmp_path):
    for module in ("rtttl", "tokenizer", "guidance", "generation", "batch_plan", "audio",
                   "song_description", "local_llm", "naming", "transformer"):
        source = ROOT / "src/rtttl_gen" / f"{module}.py"
        (tmp_path / source.name).write_text(builder.flat_imports(source.read_text()))
    options = builder.selected_functions(ROOT / "generate.py",
        ("add_sampling_arguments", "add_naming_arguments", "sampling_kwargs", "apply_optional_naming"),
        "from __future__ import annotations\nimport argparse\nimport sys\nfrom guidance import PROFILE_NAMES\n")
    (tmp_path / "options.py").write_text(builder.flat_imports(options))
    code = '''import sys
sys.path.insert(0, sys.argv[1])
import argparse
import audio, generation, local_llm, naming, song_description, transformer
from batch_plan import plan_batch
from options import add_sampling_arguments, add_naming_arguments, sampling_kwargs, apply_optional_naming
p = argparse.ArgumentParser()
add_sampling_arguments(p)
add_naming_arguments(p)
args = p.parse_args(['--profile', 'mixed', '--bpm-range', '70', '90'])
rows = plan_batch(**{k:v for k,v in sampling_kwargs(args).items() if k in ['num_songs','seed','profile','bpm','bpm_range','tonic','mode']})
assert len(rows)==20 and len({r['profile'] for r in rows})==5
assert all(70<=r['bpm']<=90 for r in rows)
assert apply_optional_naming([], [], args)==([],[],{'status':'disabled'})
assert 'rtttl_gen' not in sys.modules
assert 'transformers' not in sys.modules
'''
    subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], check=True, capture_output=True, text=True)


def test_checksums_include_nested_files_and_exclude_cache(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "metadata.json").write_text('{}')
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b'cache')
    builder.write_checksums(tmp_path)
    first = (tmp_path / "SHA256SUMS").read_text()
    builder.write_checksums(tmp_path)
    assert (tmp_path / "SHA256SUMS").read_text() == first
    assert len(first.splitlines()) == 2
    assert "nested/metadata.json" in first
    assert "__pycache__" not in first
