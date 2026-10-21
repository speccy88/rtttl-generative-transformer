"""Check the runnable studio package and its narrowly scoped model exception."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_public_package", ROOT / "scripts/build_package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)

MODEL = "web/public/transcription/basic-pitch.onnx"
MODEL_SHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"
NOTICES = {"web/public/transcription/APACHE-2.0.txt", "web/public/transcription/NOTICE.md",
           "web/public/transcription/SPOTIFY-NOTICE.txt"}


def allowed_paths():
    return {line.strip() for line in (ROOT / "PUBLIC_RELEASE_FILES.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")}


class PublicReleaseTests(unittest.TestCase):
    def patch_attribute(self, obj, name, value):
        patcher = mock.patch.object(obj, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_allowlist_contains_existing_explicit_files_and_only_the_reviewed_model(self):
        allowed = allowed_paths()
        assert {MODEL, *NOTICES, "results/studio_edits_native.json", "results/studio_browser_transcription_native.json",
                "results/studio_continuation_native.json", "web/scripts/verify-continuation.mjs"} <= allowed
        for name in allowed:
            path = ROOT / name
            assert not any(character in name for character in "*?[]"), name
            assert path.is_file() and not path.is_symlink(), name
            assert path.resolve().is_relative_to(ROOT), name
        model_extensions = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
        assert {name for name in allowed if Path(name).suffix in model_extensions} == {MODEL}
        assert not any(set(Path(name).parts) & {"node_modules", "generated", "runs", "data", "test-results", "runtime"}
                       for name in allowed)
        assert (ROOT / MODEL).stat().st_size == 230444
        assert hashlib.sha256((ROOT / MODEL).read_bytes()).hexdigest() == MODEL_SHA256
        assert MODEL_SHA256 in (ROOT / "web/public/transcription/NOTICE.md").read_text()
        assert "Apache License" in (ROOT / "web/public/transcription/APACHE-2.0.txt").read_text()


    def test_studio_entrypoint_imports_and_workers_are_in_the_source_package(self):
        """Catch a ZIP that contains main.js but omits a newly imported local module."""
        allowed = allowed_paths()
        pending = [ROOT / "web/src/main.js"]
        visited = set()
        local_reference = re.compile(r"(?:\bfrom\s+|\bimport\s*\(|\bnew\s+URL\s*\(|\bimport\s+)['\"](\.[^'\"]+)['\"]")
        while pending:
            path = pending.pop().resolve()
            if path in visited:
                continue
            visited.add(path)
            assert path.is_file(), path
            assert path.relative_to(ROOT).as_posix() in allowed, path
            if path.suffix not in {".js", ".mjs"}:
                continue
            for relative in local_reference.findall(path.read_text()):
                dependency = (path.parent / relative).resolve()
                assert dependency.is_relative_to(ROOT / "web"), dependency
                pending.append(dependency)


    def test_native_edit_evidence_publishes_hashes_and_scores_without_model_melodies(self):
        evidence = json.loads((ROOT / "results/studio_edits_native.json").read_text())
        assert evidence["status"] == "PASS"
        assert {run["editMode"] for run in evidence["editRuns"]} == {"variation", "repair"}
        for run in evidence["runs"] + evidence["editRuns"]:
            assert re.fullmatch(r"[a-f0-9]{64}", run["rtttlSha256"])
            assert "rtttl" not in run and "events" not in run and "tokens" not in run
            assert run["inferenceCalls"] > 0
        assert "not trained masked infilling" in evidence["scope"]


    def test_archive_contains_the_licensed_asset_not_nearby_unreviewed_artifacts(self):
        """Exercise the exporter against a clean fixture, without scanning caches."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        root = tmp_path / "source"
        root.mkdir()
        approved = {MODEL, *NOTICES, "results/studio_edits_native.json"}
        for relative in approved:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        unreviewed = {"web/public/transcription/other.onnx", "runs/model.pt", "generated/tune.rtttl",
                      "web/node_modules/private/cache.json", "web/public/runtime/cache.wasm"}
        for relative in unreviewed:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("excluded fixture")
        allowlist = root / "PUBLIC_RELEASE_FILES.txt"
        allowlist.write_text("\n".join(sorted(approved)) + "\n")
        archive_path = tmp_path / "release.zip"
        self.patch_attribute(package, "ROOT", root)
        self.patch_attribute(package, "ALLOWLIST", allowlist)
        self.patch_attribute(sys, "argv", ["build_package.py", "--output", str(archive_path)])
        package.main()
        with zipfile.ZipFile(archive_path) as archive:
            prefix = "RTTTL_Generative_Transformer/"
            files = {name.removeprefix(prefix) for name in archive.namelist()}
            assert files == approved | {"PACKAGE_MANIFEST.json"}
            assert hashlib.sha256(archive.read(prefix + MODEL)).hexdigest() == MODEL_SHA256
            manifest = json.loads(archive.read(prefix + "PACKAGE_MANIFEST.json"))
            assert manifest["files"][MODEL]["sha256"] == MODEL_SHA256
            assert set(manifest["files"]) == approved


if __name__ == "__main__":
    unittest.main()
