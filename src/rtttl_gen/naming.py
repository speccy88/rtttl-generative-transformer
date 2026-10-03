"""Optional title postprocessing that leaves every musical event unchanged.

The model runtime is deliberately supplied by the caller. Importing this module
does not load a model, download weights, or require Transformers.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
import unicodedata
from typing import Protocol, Sequence

from .rtttl import Song, parse_rtttl


class TitleNamer(Protocol):
    def title(self, song: Song, *, used_titles: list[str], seed: int) -> str:
        """Suggest one plain-text title from the song's musical features."""


def sanitize_title(value: str) -> str:
    """Accept a short plain-text title, rejecting markup and explanatory prose.

    Two to four words are requested by the model prompt; one to six are accepted
    here so an otherwise useful short answer is not discarded. Unicode letters
    are retained in display metadata, while RTTTL names are separately ASCII.
    """
    if not isinstance(value, str):
        raise ValueError("Title output must be text")
    title = unicodedata.normalize("NFKC", value).strip()
    if re.match(r"^title:\s*", title, re.IGNORECASE):
        title = re.sub(r"^title:\s*", "", title, count=1, flags=re.IGNORECASE)
    if len(title) >= 2 and (title[0], title[-1]) in (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")):
        title = title[1:-1].strip()
    if any(unicodedata.category(char).startswith("C") for char in title):
        raise ValueError("Title output contains control characters or multiple lines")
    title = " ".join(title.split())
    if not title or len(title) > 60:
        raise ValueError("Title must contain 1 to 60 characters")
    if not any(char.isalpha() for char in title):
        raise ValueError("Title must contain letters")
    if not all(char.isalnum() or char in " '-’&" for char in title):
        raise ValueError("Title contains markup or unsupported punctuation")
    if len(title.split()) > 6:
        raise ValueError("Title output is too verbose")
    if re.search(
        r"\b(?:ignore (?:previous|all|the)|(?:system|user|assistant) (?:prompt|message)|"
        r"(?:here is|here's|this is|the title is|i suggest|i recommend|i would|"
        r"based on|could be named|call it)|instructions?)\b",
        title, re.IGNORECASE,
    ):
        raise ValueError("Title output contains instructions or explanatory prose")
    return title


def _unique(value: str, used: set[str], limit: int, *, separator: str) -> tuple[str, bool]:
    candidate = value[:limit]
    if candidate.casefold() not in used:
        return candidate, False
    number = 2
    while True:
        suffix = f"{separator}{number}"
        candidate = value[:limit - len(suffix)].rstrip() + suffix
        if candidate.casefold() not in used:
            return candidate, True
        number += 1


def _compact(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]", "", ascii_title)[:11] or "Untitled"


def _original_title(record: dict, song: Song) -> str:
    for value in (record.get("title"), song.name):
        try:
            return sanitize_title(value)
        except ValueError:
            continue
    return _compact(song.name)


def name_records(records: Sequence[dict], namer: TitleNamer, *, seed: int = 42) -> tuple[list[dict], list[Song], dict]:
    """Name valid records without mutating inputs or their musical payloads.

    A duplicate or invalid title receives one retry with a different deterministic
    seed. If both attempts repeat an existing title, a numeric suffix makes the
    full title unique and is recorded as a disambiguation. If no valid title was
    obtained, or inference raises, the original name is retained where possible.
    Failures are explicit and never presented as successful model suggestions.

    ``songs`` contains each successfully parsed valid song, in record order.
    Invalid records are copied unchanged. Only title metadata and the RTTTL name
    section are edited: original defaults and melody text remain byte-for-byte.
    ``named`` counts model suggestions (including suffixes); ``failed`` counts
    naming/parse failures; ``disambiguated`` can overlap either count.
    """
    result = deepcopy(list(records))
    songs: list[Song] = []
    used_titles: list[str] = []
    used_title_keys: set[str] = set()
    used_names: set[str] = set()
    report = {"attempted": 0, "named": 0, "failed": 0, "disambiguated": 0, "skipped": 0}
    for index, record in enumerate(result):
        if not record.get("valid"):
            report["skipped"] += 1
            continue
        report["attempted"] += 1
        try:
            song = parse_rtttl(record.get("rtttl"))
        except (ValueError, TypeError) as exc:
            record["naming"] = {"source": "original", "status": "failed", "attempts": 0,
                                "error": f"Cannot parse RTTTL: {exc}"}
            report["failed"] += 1
            continue

        title: str | None = None
        errors: list[str] = []
        attempts = 0
        inference_failed = False
        for attempt in range(2):
            attempts += 1
            try:
                # Retry seeds are separated from first-attempt seeds in the batch.
                output = namer.title(song, used_titles=list(used_titles), seed=seed + index + attempt * len(result))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                inference_failed = True
                break
            try:
                candidate = sanitize_title(output)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            title = candidate
            if candidate.casefold() not in used_title_keys:
                break
            errors.append("Model suggested a duplicate title")

        from_model = title is not None and not inference_failed
        if not from_model:
            title = _original_title(record, song)
        title, title_changed = _unique(title, used_title_keys, 60, separator=" ")
        # A normal fallback keeps the original valid short header, including its
        # spaces/hyphens. Other headers use a portable <=11-character ASCII name.
        if not from_model and re.fullmatch(r"[A-Za-z0-9 _-]{1,11}", song.name):
            name = song.name
        else:
            name = _compact(title)
        name, name_changed = _unique(name, used_names, 11, separator="")
        disambiguated = title_changed or name_changed
        naming = {
            "source": "local_llm" if from_model else "original",
            "status": ("disambiguated" if disambiguated else "named") if from_model else "failed",
            "attempts": attempts,
            "title_disambiguated": title_changed,
            "rtttl_name_disambiguated": name_changed,
        }
        if errors:
            naming["retry_errors"] = errors
        if not from_model:
            naming["error"] = errors[-1] if errors else "No usable title returned"
        _, defaults, melody = record["rtttl"].rsplit(":", 2)
        record.update(title=title, rtttl_name=name, rtttl=f"{name}:{defaults}:{melody}", naming=naming)
        songs.append(replace(song, name=name))
        used_titles.append(title)
        used_title_keys.add(title.casefold())
        used_names.add(name.casefold())
        report["named" if from_model else "failed"] += 1
        report["disambiguated"] += int(disambiguated)
    report["status"] = "partial" if report["named"] and report["failed"] else "failed" if report["failed"] else "complete" if report["named"] else "no_valid_songs"
    return result, songs, report
