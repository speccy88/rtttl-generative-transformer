"""Deterministic musical deduplication and leakage-aware family splitting.

Exact duplicates retain tempo by default. Families intentionally ignore tempo,
global transposition, and uniform rhythmic scaling. Near-family edges use an
exhaustive inverted-shingle candidate index, not random LSH. This is a conservative
heuristic, not proof that every arrangement of the same composition is detected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import reduce
import hashlib
import json
from math import gcd
import random
from typing import Iterable, Sequence

from .rtttl import Song


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def duration_ticks(event) -> int:
    """Integer duration in 1/128-whole-note units (supports all RTTTL values)."""
    return (128 // event.duration) * (3 if event.dotted else 2) // 2


def exact_fingerprint(song: Song, include_tempo: bool = True) -> str:
    events = [(e.pitch, e.duration, e.dotted) for e in song.events]
    return _digest([song.bpm if include_tempo else None, events])


def event_key(song: Song) -> tuple:
    """Resolved events, independent of name, default headers, and BPM."""
    return tuple((e.pitch, e.duration, e.dotted) for e in song.events)


def normalized_events(song: Song) -> tuple:
    """Pitch relative to first note; durations divided by their common divisor."""
    first = next((e.pitch for e in song.events if e.pitch is not None), 0)
    ticks = [duration_ticks(e) for e in song.events]
    divisor = reduce(gcd, ticks, 0) or 1
    return tuple((None if e.pitch is None else e.pitch - first, t // divisor)
                 for e, t in zip(song.events, ticks))


def family_fingerprint(song: Song) -> str:
    return _digest(normalized_events(song))


def transition_tokens(song: Song) -> tuple:
    """Local interval and rhythmic-ratio tokens, stable after most insertions.

Rests are explicit; a pitched event after a rest still uses the previous pitched
event for its interval. Every duration ratio is reduced exactly using integers.
"""
    result, previous_pitch, previous_ticks = [], None, None
    for event in song.events:
        ticks = duration_ticks(event)
        if event.pitch is None:
            interval = "REST"
        elif previous_pitch is None:
            interval = "START"
        else:
            interval = event.pitch - previous_pitch
        if previous_ticks is None:
            ratio = (1, 1)
        else:
            divisor = gcd(ticks, previous_ticks)
            ratio = (ticks // divisor, previous_ticks // divisor)
        result.append((interval, *ratio))
        if event.pitch is not None:
            previous_pitch = event.pitch
        previous_ticks = ticks
    return tuple(result)


transition_signature = transition_tokens


def shingles(sequence: Sequence, n: int = 4) -> frozenset:
    if len(sequence) < n:
        return frozenset([tuple(sequence)]) if sequence else frozenset()
    return frozenset(tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1))


def jaccard(left: Iterable, right: Iterable) -> float:
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0


def deduplicate_songs(songs: Sequence[Song], include_tempo: bool = True):
    """Return (unique_songs, retained_original_indices, original_representatives).

The final list maps EVERY original record to its retained original index. Thus a
source manifest can preserve membership even when copies are removed for training.
"""
    seen, retained, duplicate_of = {}, [], []
    for index, song in enumerate(songs):
        fingerprint = exact_fingerprint(song, include_tempo=include_tempo)
        if fingerprint not in seen:
            seen[fingerprint] = index
            retained.append(index)
        duplicate_of.append(seen[fingerprint])
    return [songs[i] for i in retained], retained, duplicate_of


def group_families(songs: Sequence[Song], near_threshold: float = 0.94) -> tuple[list[str], list[int | None], dict]:
    """Cluster exact invariant families plus conservative near-melody edges.

Near edges require >=12 events, >=65% length ratio, >=45% four-gram Jaccard,
and >=near_threshold ordered transition similarity, or >=98% containment with
at least 24 events in the shorter song. Low-information patterns
with fewer than four distinct transitions are exact-family-only. Every possible
shared-shingle candidate passing these gates is considered, independent of order.
Connected components keep ALL accepted edges within one split. Transitive family
growth and undetected rearrangements remain limitations of this heuristic.
"""
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    fingerprints = [family_fingerprint(s) for s in songs]
    exact_seen, duplicate_of = {}, []
    for index, song in enumerate(songs):
        key = exact_fingerprint(song)
        duplicate_of.append(exact_seen.get(key))
        exact_seen.setdefault(key, index)
    buckets = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        buckets[fingerprint].append(index)
    keys = sorted(buckets)
    representatives = [songs[buckets[key][0]] for key in keys]
    sequences = [transition_tokens(s) for s in representatives]
    grams = [shingles(s, 4) for s in sequences]
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    inverted = defaultdict(list)
    approximate_pairs = []
    candidate_comparisons = 0
    for i, (sequence, group_grams) in enumerate(zip(sequences, grams)):
        eligible = len(sequence) >= 12 and len(set(sequence)) >= 4
        if not eligible:
            continue
        shared = Counter()
        for gram in group_grams:
            shared.update(inverted[gram])
        for j in sorted(shared):
            other = sequences[j]
            short, long = sorted((len(sequence), len(other)))
            if short / long < .65:
                continue
            overlap = shared[j] / (len(group_grams) + len(grams[j]) - shared[j])
            if overlap < .45:
                continue
            candidate_comparisons += 1
            matcher = SequenceMatcher(None, sequence, other, autojunk=False)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            similarity = 2 * matched / (len(sequence) + len(other))
            containment = matched / short
            if similarity >= near_threshold or (short >= 24 and containment >= .98):
                union(i, j)
                approximate_pairs.append({
                    "a": buckets[keys[j]][0], "b": buckets[keys[i]][0],
                    "reason": "likely_edited_or_truncated_alias",
                    "transition_similarity": round(similarity, 6),
                    "containment": round(containment, 6),
                    "length_ratio": round(short / long, 6),
                    "shingle_jaccard": round(overlap, 6),
                })
        for gram in group_grams:
            inverted[gram].append(i)
    # The smallest SHA-256 fingerprint names each component deterministically.
    mapping = {key: "family_" + keys[find(i)][:20] for i, key in enumerate(keys)}
    families = [mapping[fingerprint] for fingerprint in fingerprints]
    sizes = Counter(families)
    exact_counts = Counter(exact_fingerprint(song) for song in songs)
    identity_pairs = [
        {"a": indices[0], "b": other,
         "reason": "transposition_rhythm_tempo_invariant_identity"}
        for indices in buckets.values() for other in indices[1:]
    ]
    report = {
        "input_songs": len(songs), "unique_exact_songs": len(exact_seen),
        "exact_duplicate_records": sum(x is not None for x in duplicate_of),
        "exact_duplicate_groups": sum(n > 1 for n in exact_counts.values()),
        "invariant_identity_groups": sum(len(v) > 1 for v in buckets.values()),
        "unique_invariant_identities": len(buckets), "melody_families": len(sizes),
        "multi_record_families": sum(n > 1 for n in sizes.values()),
        "largest_family_size": max(sizes.values(), default=0),
        "approximate_pair_count": len(approximate_pairs),
        "approximate_candidate_comparisons": candidate_comparisons,
        "identity_pairs": identity_pairs, "approximate_pairs": approximate_pairs,
        "method": {
            "exact": "resolved pitch/duration/dot events and BPM; names ignored",
            "identity": "relative pitch, rest positions and GCD-normalized integer durations; BPM ignored",
            "near": "exhaustive inverted 4-transition shingles, ordered alignment",
            "near_similarity_threshold": near_threshold,
            "containment_threshold": .98, "containment_min_events": 24,
            "minimum_length_ratio": .65, "minimum_shingle_jaccard": .45,
            "minimum_events": 12, "minimum_distinct_transitions": 4,
        },
        "limitations": [
            "Approximate matching is heuristic; undetected edited covers can cross splits.",
            "Common motifs can cause false-positive grouping; grouping errs toward leakage prevention.",
            "Connected components can include indirectly linked variants.",
            "Short/low-information songs receive only exact invariant family matching.",
            "No candidate cap; inverted-index worst-case time can still be quadratic for repetitive corpora.",
        ],
    }
    return families, duplicate_of, report


def split_families(family_ids: Sequence[str], seed: int = 42,
                   ratios: tuple = (.8, .1, .1)) -> list[str]:
    """Shuffle families independently of size, then cut cumulative record counts.

    A family's midpoint selects its partition at the requested cumulative ratio
    boundaries. Entire families stay together, so counts can differ from their
    targets by a few records. Sorting IDs before seeded shuffling makes the
    assignment independent of input order. Unlike largest-family-first deficit
    allocation, this does not systematically assign variants to training only.

    For tiny corpora, if a split is empty and >=3 families exist, move the family
    that minimizes squared target-count error from a split with >1 family.
    """
    if len(ratios) != 3 or any(f <= 0 for f in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("ratios must be three positive numbers summing to one")
    groups = defaultdict(list)
    for index, family in enumerate(family_ids):
        groups[family].append(index)
    items = sorted(groups.items())
    random.Random(seed).shuffle(items)
    names = ("train", "validation", "test")
    total = len(family_ids)
    targets = [total * ratio for ratio in ratios]
    boundaries = (targets[0], targets[0] + targets[1])
    assignments, counts, family_counts = {}, [0, 0, 0], [0, 0, 0]
    cumulative = 0
    for family, indices in items:
        midpoint = cumulative + len(indices) / 2
        destination = 0 if midpoint < boundaries[0] else 1 if midpoint < boundaries[1] else 2
        assignments[family] = destination
        counts[destination] += len(indices)
        family_counts[destination] += 1
        cumulative += len(indices)
    if len(items) >= 3:
        for empty in [i for i, count in enumerate(family_counts) if count == 0]:
            candidates = []
            for family, indices in items:
                donor = assignments[family]
                if family_counts[donor] <= 1:
                    continue
                proposed = counts.copy()
                proposed[donor] -= len(indices)
                proposed[empty] += len(indices)
                error = sum((count - target) ** 2 for count, target in zip(proposed, targets))
                candidates.append((error, family, donor))
            _, family, donor = min(candidates)
            assignments[family] = empty
            counts[donor] -= len(groups[family])
            counts[empty] += len(groups[family])
            family_counts[donor] -= 1
            family_counts[empty] += 1
    aligned = [""] * len(family_ids)
    for family, indices in groups.items():
        for index in indices:
            aligned[index] = names[assignments[family]]
    return aligned


def assert_no_leakage(songs: Sequence[Song], splits: Sequence[str],
                      family_ids: Sequence[str]) -> None:
    """Check families and independently recomputed tempo-free musical identities."""
    if not (len(songs) == len(splits) == len(family_ids)):
        raise AssertionError("songs, splits, and family_ids have different lengths")
    memberships = [{}, {}, {}]
    for index, split_name in enumerate(splits):
        if split_name not in {"train", "validation", "test"}:
            raise AssertionError(f"Unknown split: {split_name}")
        values = (exact_fingerprint(songs[index], include_tempo=False),
                  family_fingerprint(songs[index]), family_ids[index])
        for mapping, value in zip(memberships, values):
            if value in mapping and mapping[value] != split_name:
                raise AssertionError(f"Leakage: musical family occurs in {mapping[value]} and {split_name}")
            mapping[value] = split_name
