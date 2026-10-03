"""Read-only ingestion of text RTTTL files, directories and ZIP archives.

Archives are read directly, never extracted.  Every nonmetadata file produces
a record, including unsupported formats and empty files.  source_inventory
accounts for directories and macOS metadata too.  Text compilations and line-
wrapped records are supported; repairs and encodings are explicit metadata.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import statistics
from typing import Iterator, Any
import zipfile

from .rtttl import RTTTLParseError, parse_rtttl, song_to_dict

TEXT_EXTENSIONS = {".txt", ".rtttl", ".rtl", ".rtx"}


def _is_metadata(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return "__MACOSX" in parts or any(p.startswith("._") or p == ".DS_Store" for p in parts)


def _files(path: str | Path) -> Iterator[tuple[str, bytes, bool]]:
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                yield info.filename, b"" if info.is_dir() else archive.read(info), info.is_dir()
    elif path.is_dir():
        for item in sorted(path.rglob("*")):
            # Do not follow outside symlink targets through the supplied corpus.
            if item.is_symlink():
                yield item.relative_to(path).as_posix(), b"", False
            elif item.is_dir():
                yield item.relative_to(path).as_posix() + "/", b"", True
            else:
                yield item.relative_to(path).as_posix(), item.read_bytes(), False
    elif path.is_file():
        yield path.name, path.read_bytes(), False
    else:
        raise FileNotFoundError(path)


def _kind(source: str, raw: bytes, directory: bool) -> str:
    if directory:
        return "directory"
    if _is_metadata(source):
        return "metadata"
    if not raw:
        return "empty"
    if PurePosixPath(source).suffix.lower() in TEXT_EXTENSIONS:
        return "text"
    return "unsupported"


def source_inventory(path: str | Path) -> list[dict[str, Any]]:
    """File-level accounting including otherwise ignored metadata entries."""
    return [{"source": source, "size_bytes": len(raw), "kind": _kind(source, raw, directory),
             "sha256": None if directory else hashlib.sha256(raw).hexdigest()}
            for source, raw, directory in _files(path)]


def _decode(raw: bytes) -> tuple[str, str, list[str]]:
    try:
        return raw.decode("utf-8-sig"), "utf-8", []
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252"), "cp1252", ["decoded_cp1252"]
        except UnicodeDecodeError:
            return raw.decode("latin1"), "latin1", ["decoded_latin1"]


def _text_records(text: str) -> Iterator[tuple[int, str, list[str]]]:
    """A new colon-bearing line starts a record; other lines wrap its notes."""
    current: list[str] = []
    start = 1
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if ":" in line and current:
            yield start, "\n".join(current), ["wrapped_lines_joined"] if len(current) > 1 else []
            current = []
        if not current:
            start = line_number
        current.append(line)
    if current:
        yield start, "\n".join(current), ["wrapped_lines_joined"] if len(current) > 1 else []
    else:
        yield 1, "", []


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield all candidates and explicit unsupported/empty records.

    ``kind == 'rtttl'`` is a candidate, never a validity assertion.  The
    source/line pair is the stable link back to read-only input.  The explicit
    rtttl3 underscore hint applies only to the observed folder convention.
    """
    for source, raw_bytes, directory in _files(path):
        kind = _kind(source, raw_bytes, directory)
        if kind in {"directory", "metadata"}:
            continue
        if kind in {"empty", "unsupported"}:
            yield {"source": source, "line": 1, "raw": "", "kind": kind,
                   "size_bytes": len(raw_bytes), "warnings": [],
                   "reason": "empty_file" if kind == "empty" else "unsupported_file_format"}
            continue
        text, encoding, decode_warnings = _decode(raw_bytes)
        for line, raw, record_warnings in _text_records(text):
            yield {"source": source, "line": line, "raw": raw, "kind": "rtttl",
                   "encoding": encoding, "warnings": decode_warnings + record_warnings,
                   "underscore_is_sharp": "rtttl3" in PurePosixPath(source).parts}


def _summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {key: None for key in ("min", "median", "mean", "p90", "max")}
    ordered = sorted(values)
    pos = 0.9 * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    p90 = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "mean": statistics.mean(ordered), "p90": p90, "max": ordered[-1]}


def inspect_corpus(path: str | Path, *, underscore_is_sharp: bool = False
                   ) -> tuple[list[dict], list[dict], dict]:
    """Validate every record and return accepted records, audit, and statistics.

    The inspected rtttl3 convention is enabled automatically *only* for that
    folder. Elsewhere callers must explicitly opt in. Every such normalization
    is still visible in its individual audit record. Musical statistics cover
    all accepted records before deduplication. No source file is written.
    """
    inventory = source_inventory(path)
    file_metadata = {item["source"]: item for item in inventory}
    records, manifest = [], []
    errors, warnings, encodings = Counter(), Counter(), Counter()
    pitch, octave, durations, bpm, default_d, default_o = (Counter() for _ in range(6))
    lengths: list[int] = []
    default_duration_notes = default_octave_notes = dotted = rests = total = 0
    exact_event_keys: set[tuple] = set()
    exact_tempo_event_keys: set[tuple] = set()
    counts = Counter()
    for record in iter_records(path):
        source, line = record["source"], record["line"]
        raw_hash = hashlib.sha256(record["raw"].encode("utf-8")).hexdigest()
        record_id = hashlib.sha256(f"{source}\n{line}\n{raw_hash}".encode()).hexdigest()[:20]
        audit = {"id": record_id, "source": source, "line": line, "kind": record["kind"],
                 "source_sha256": file_metadata[source]["sha256"], "record_sha256": raw_hash,
                 "encoding": record.get("encoding"), "warnings": list(record["warnings"])}
        if record["kind"] != "rtttl":
            counts[record["kind"]] += 1
            audit.update(status="rejected", error=record["reason"], detail=record["reason"], raw=record["raw"])
            errors[record["reason"]] += 1
            manifest.append(audit)
            continue
        counts["candidate_rtttl_records"] += 1
        encodings[record["encoding"]] += 1
        try:
            song = parse_rtttl(record["raw"], underscore_is_sharp=(underscore_is_sharp or record["underscore_is_sharp"]))
        except RTTTLParseError as exc:
            counts["invalid_rtttl_records"] += 1
            errors[exc.code] += 1
            audit.update(status="rejected", error=exc.code, detail=exc.detail, raw=record["raw"])
            manifest.append(audit)
            continue
        counts["valid_rtttl_records"] += 1
        audit["warnings"] = list(dict.fromkeys(audit["warnings"] + list(song.warnings)))
        warnings.update(audit["warnings"])
        audit.update(status="accepted_with_normalization" if audit["warnings"] else "accepted",
                     event_count=len(song.events), name=song.name)
        manifest.append(audit)
        records.append({"id": record_id, "source": source, "line": line,
                        "source_sha256": audit["source_sha256"], "song": song_to_dict(song)})
        lengths.append(len(song.events))
        bpm[str(song.bpm)] += 1
        default_d[str(song.default_duration)] += 1
        default_o[str(song.default_octave)] += 1
        key = tuple((event.pitch, event.duration, event.dotted) for event in song.events)
        exact_event_keys.add(key)
        exact_tempo_event_keys.add((song.bpm, key))
        for event in song.events:
            total += 1
            dotted += event.dotted
            durations[str(event.duration)] += 1
            if event.pitch is None:
                rests += 1
            else:
                pitch[str(event.pitch)] += 1
                octave[str(event.pitch // 12 - 1)] += 1
        # Event duration is inherited only when no leading duration digits.
        melody_tokens = re.sub(r"\s+", "", record["raw"].rsplit(":", 1)[1]).strip(",").lower().split(",")
        for token in melody_tokens:
            default_duration_notes += re.match(r"\d", token) is None
            tail = re.sub(r"^\d+", "", token).replace(".", "")
            if "p" not in tail:
                default_octave_notes += re.search(r"\d", tail) is None
    kinds = Counter(item["kind"] for item in inventory)
    stats = {
        "input_path": str(Path(path)),
        "inventory": {"archive_entries": len(inventory), "directories": kinds["directory"],
                      "all_files": len(inventory) - kinds["directory"],
                      "metadata_files": kinds["metadata"], "text_files": kinds["text"],
                      "empty_files": kinds["empty"], "unsupported_files": kinds["unsupported"],
                      "nonmetadata_files": len(inventory) - kinds["directory"] - kinds["metadata"],
                      "all_file_bytes": sum(item["size_bytes"] for item in inventory)},
        "candidate_rtttl_records": counts["candidate_rtttl_records"],
        "valid_rtttl_records": counts["valid_rtttl_records"],
        "invalid_rtttl_records": counts["invalid_rtttl_records"],
        "rejected_non_rtttl_files": counts["empty"] + counts["unsupported"],
        "accepted_without_warnings": sum(item["status"] == "accepted" for item in manifest),
        "accepted_with_normalization": sum(item["status"] == "accepted_with_normalization" for item in manifest),
        "unique_exact_event_sequences": len(exact_event_keys),
        "unique_exact_tempo_event_sequences": len(exact_tempo_event_keys),
        "musical_events": total, "sequence_length_events": _summary(lengths),
        "length_histogram": dict(sorted(Counter(lengths).items())),
        "pitch_midi_histogram": dict(sorted(pitch.items(), key=lambda kv: int(kv[0]))),
        "octave_histogram": dict(sorted(octave.items())),
        "duration_denominator_histogram": dict(sorted(durations.items(), key=lambda kv: int(kv[0]))),
        "bpm_histogram": dict(sorted(bpm.items(), key=lambda kv: int(kv[0]))),
        "bpm_summary": _summary([row["song"]["bpm"] for row in records]),
        "default_duration_histogram": dict(sorted(default_d.items(), key=lambda kv: int(kv[0]))),
        "default_octave_histogram": dict(sorted(default_o.items())),
        "inherited_duration_event_count": default_duration_notes,
        "inherited_octave_pitched_event_count": default_octave_notes,
        "dotted_event_count": dotted, "dotted_event_fraction": dotted / total if total else 0,
        "rest_event_count": rests, "rest_event_fraction": rests / total if total else 0,
        "warning_record_counts": dict(warnings.most_common()), "error_record_counts": dict(errors.most_common()),
        "encoding_record_counts": dict(encodings),
        "normalization_policy": "rtttl3 '_' -> '#' corpus convention; optional elsewhere; safe spelling changes audited; no guessed missing notes/durations/octaves",
        "statistics_scope": "all accepted records before deduplication; pitch histograms exclude rests",
    }
    return records, manifest, stats
