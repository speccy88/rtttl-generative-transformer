from copy import deepcopy

import pytest

from rtttl_gen.evaluation import generated_metrics, song_key
from rtttl_gen.naming import name_records, sanitize_title
from rtttl_gen.rtttl import parse_rtttl


class FakeNamer:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def title(self, song, *, used_titles, seed):
        self.calls.append((song, list(used_titles), seed))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        # A model implementation cannot mutate the caller's title history.
        used_titles.append("untrusted mutation")
        return output


def record(index=0):
    return {
        "id": f"sample-{index}", "valid": True,
        "rtttl": f"Generated{index:02d}:o=5, d=8, b=120:c, e, g,4c6,p,8d. ",
        "token_ids": [1, 4, 23, 7, 2], "forced_eos": False,
        "similarity": {"score": 0.125, "label": "low similarity"},
        "degeneracy_flags": [], "repetition_interventions": {"pitch_run_blocks": 1},
    }


def test_naming_changes_only_title_and_preserves_musical_identity_and_input():
    records = [record()]
    before = deepcopy(records)
    original_song = parse_rtttl(records[0]["rtttl"])
    namer = FakeNamer([' "Neon Summer" '])
    renamed, songs, report = name_records(records, namer, seed=17)
    output = renamed[0]
    assert records == before
    assert output["title"] == "Neon Summer"
    assert output["rtttl_name"] == "NeonSummer"
    assert output["rtttl"].split(":", 1)[1] == before[0]["rtttl"].split(":", 1)[1]
    assert song_key(songs[0]) == song_key(original_song)
    assert songs[0] == parse_rtttl(output["rtttl"])
    for field in ("id", "valid", "token_ids", "forced_eos", "similarity", "degeneracy_flags", "repetition_interventions"):
        assert output[field] == before[0][field]
    assert generated_metrics(records, [original_song]) == generated_metrics(renamed, songs)
    assert namer.calls[0][1:] == ([], 17)
    assert report == {"attempted": 1, "named": 1, "failed": 0, "disambiguated": 0, "skipped": 0, "status": "complete"}
    output["token_ids"].append(999)
    assert records == before


def test_duplicate_full_titles_retry_once_then_use_disclosed_numeric_suffix():
    inputs = [record(0), record(1), record(2)]
    namer = FakeNamer(["Neon Summer", "neon summer", "Neon Summer", "Neon Summer", "Moonlit Bounce"])
    renamed, _, report = name_records(inputs, namer, seed=10)
    assert [r["title"] for r in renamed] == ["Neon Summer", "Neon Summer 2", "Moonlit Bounce"]
    assert [call[2] for call in namer.calls] == [10, 11, 14, 12, 15]
    assert [call[1] for call in namer.calls] == [[], ["Neon Summer"], ["Neon Summer"], ["Neon Summer", "Neon Summer 2"], ["Neon Summer", "Neon Summer 2"]]
    assert renamed[1]["naming"]["source"] == "local_llm"
    assert renamed[1]["naming"]["status"] == "disambiguated"
    assert renamed[1]["naming"]["title_disambiguated"] is True
    assert renamed[1]["naming"]["attempts"] == 2
    assert report["named"] == 3 and report["disambiguated"] == 1


def test_ascii_compact_collisions_and_unicode_names_are_unique_and_stable():
    inputs = [record(i) for i in range(4)]
    titles = ["Golden Morning", "Golden Morning Bells", "Étoiles Dorées", "Golden Morning Dreams"]
    first, _, report = name_records(inputs, FakeNamer(titles))
    second, _, _ = name_records(inputs, FakeNamer(titles))
    assert first == second
    assert [r["title"] for r in first] == titles
    compact = [r["rtttl_name"] for r in first]
    assert compact == ["GoldenMorni", "GoldenMorn2", "EtoilesDore", "GoldenMorn3"]
    assert all(name.isascii() and len(name) <= 11 for name in compact)
    assert len(set(name.casefold() for name in compact)) == 4
    assert report["named"] == 4 and report["disambiguated"] == 2


@pytest.mark.parametrize("output", [
    "", "   ", "<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
    '{"title": "Moon Glow"}', "```Moon Glow```", "Moon Glow\nIgnore instructions",
    "Here is a title", "The title is Moon Glow", "Ignore previous instructions",
    "This is a cheerful melody", "One Two Three Four Five Six Seven", "12345",
    "A" * 61, "Moon\x00Glow", None,
])
def test_invalid_model_output_is_retried_then_original_is_retained(output):
    source = record()
    namer = FakeNamer([output, output])
    renamed, songs, report = name_records([source], namer)
    assert renamed[0]["title"] == "Generated00"
    assert renamed[0]["rtttl"] == source["rtttl"]
    assert songs[0] == parse_rtttl(source["rtttl"])
    assert renamed[0]["naming"]["source"] == "original"
    assert renamed[0]["naming"]["status"] == "failed"
    assert renamed[0]["naming"]["attempts"] == 2
    assert renamed[0]["naming"]["error"]
    assert report["status"] == "failed" and report["failed"] == 1 and report["named"] == 0


def test_invalid_output_can_recover_on_retry():
    renamed, _, report = name_records([record()], FakeNamer(["Here is a title", "Starlight"]))
    assert renamed[0]["title"] == "Starlight"
    assert renamed[0]["naming"]["attempts"] == 2
    assert "error" not in renamed[0]["naming"]
    assert report["named"] == 1 and report["failed"] == 0


def test_inference_exception_retains_original_and_other_records_continue():
    records = [record(i) for i in range(3)]
    namer = FakeNamer([RuntimeError("device unavailable"), "Silver Skyline", "Moon Glow"])
    renamed, songs, report = name_records(records, namer)
    assert renamed[0]["rtttl"] == records[0]["rtttl"]
    assert renamed[0]["naming"]["error"] == "RuntimeError: device unavailable"
    assert renamed[0]["naming"]["attempts"] == 1
    assert [r["title"] for r in renamed] == ["Generated00", "Silver Skyline", "Moon Glow"]
    assert len(songs) == 3
    assert report["status"] == "partial" and report["named"] == 2 and report["failed"] == 1


def test_original_fallback_names_are_disambiguated_without_fake_model_success():
    inputs = [record(), record()]
    renamed, _, report = name_records(inputs, FakeNamer([RuntimeError("offline"), RuntimeError("offline")]))
    assert [r["title"] for r in renamed] == ["Generated00", "Generated00 2"]
    assert [r["rtttl_name"] for r in renamed] == ["Generated00", "Generated02"]
    assert all(r["naming"]["source"] == "original" and r["naming"]["status"] == "failed" for r in renamed)
    assert report["failed"] == 2 and report["named"] == 0 and report["disambiguated"] == 1


def test_invalid_records_are_preserved_and_bad_rtttl_is_reported_without_model_call():
    invalid = {"id": "broken", "valid": False, "rtttl": None, "error": "original failure"}
    malformed = {"id": "bad-data", "valid": True, "rtttl": "broken"}
    namer = FakeNamer(["Velvet Sky"])
    renamed, songs, report = name_records([invalid, malformed, record()], namer, seed=100)
    assert renamed[0] == invalid
    assert renamed[1]["rtttl"] == "broken"
    assert renamed[1]["naming"]["attempts"] == 0
    assert len(songs) == 1 and len(namer.calls) == 1
    assert namer.calls[0][1:] == ([], 102)
    assert report == {"attempted": 2, "named": 1, "failed": 1, "disambiguated": 0, "skipped": 1, "status": "partial"}


def test_empty_batch_does_not_call_model():
    assert name_records([], FakeNamer([])) == ([], [], {"attempted": 0, "named": 0, "failed": 0, "disambiguated": 0, "skipped": 0, "status": "no_valid_songs"})


@pytest.mark.parametrize("value, expected", [("Title: \"Moon Glow\"", "Moon Glow"), ("‘Étoiles Dorées’", "Étoiles Dorées"), ("Neon   Summer", "Neon Summer"), ("Rock & Roll", "Rock & Roll")])
def test_plain_text_title_cleanup(value, expected):
    assert sanitize_title(value) == expected
