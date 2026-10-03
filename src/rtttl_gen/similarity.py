"""Approximate musical novelty, not an originality or copyright detector.

For every training melody, the score is the weighted Jaccard overlap of unique
3-grams: joint pitch/rest + rhythm transitions (0.55), pitch/rest transitions
(0.30), and rhythm transitions (0.15). The inverted index computes exact overlap
counts for this *approximate musical metric*, without a candidate cap. Thus the
returned closest song maximizes the defined score over all supplied songs.

Exact events and transposition/global-rhythm-scaled identities are checked
separately and take precedence. BPM is reported but does not lower similarity.
Thresholds are heuristic: exact events -> exact training match; invariant match
or score >=0.90 -> near match; >=0.70 -> high similarity; >=0.40 -> moderate
similarity; otherwise low similarity. Low similarity is not proof of originality.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from .deduplicate import event_key, family_fingerprint, shingles, transition_signature

if TYPE_CHECKING:
    from .rtttl import Song


def _features(song: "Song") -> tuple[frozenset, frozenset, frozenset]:
    transitions = transition_signature(song)
    if not transitions:
        # Do not identify all one-event songs simply because all lack intervals.
        events = event_key(song)
        return (frozenset((events,)) if events else frozenset(), frozenset(), frozenset())
    return (shingles(transitions, 3),
            shingles(tuple(t[0] for t in transitions), 3),
            shingles(tuple(t[1:] for t in transitions), 3))


class SimilarityIndex:
    """Reusable training-only index with explicit exact and invariant lookups."""

    def __init__(self, train_songs: list["Song"], ids: list[str] | None = None):
        from .rtttl import song_from_dict
        original_records = list(train_songs)
        self.songs = [song_from_dict(r.get("song", r)) if isinstance(r, dict) else r
                      for r in original_records]
        self.ids = list(ids) if ids is not None else [
            str(r.get("id", r.get("record_id", i))) if isinstance(r, dict) else str(i)
            for i, r in enumerate(original_records)]
        if len(self.ids) != len(self.songs):
            raise ValueError("ids length must match train_songs")
        self.exact: dict[tuple, int] = {}
        self.invariant: dict[tuple, int] = {}
        self.features = []
        self.postings = [defaultdict(list), defaultdict(list), defaultdict(list)]
        for i, song in enumerate(self.songs):
            self.exact.setdefault(event_key(song), i)
            self.invariant.setdefault(family_fingerprint(song), i)
            features = _features(song)
            self.features.append(features)
            for component, feature_set in enumerate(features):
                for feature in feature_set:
                    self.postings[component][feature].append(i)

    def nearest(self, song: "Song") -> dict:
        if not self.songs:
            return {"closest_index": None, "closest_name": None, "closest_id": None,
                    "score": None, "label": "no training reference", "exact_event_match": False,
                    "transposition_match": False, "rhythm_similarity": None,
                    "ngram_overlap": None, "interval_similarity": None,
                    "training_reference_count": 0, "metric": "weighted_transition_3gram_jaccard_v1",
                    "closest_training_id": None, "closest_training_name": None,
                    "invariant_match": False, "pitch_interval_similarity": None,
                    "rhythmic_similarity": None, "bpm_match": None}
        query = _features(song)
        overlaps = [Counter(), Counter(), Counter()]
        candidates: set[int] = set()
        for component, feature_set in enumerate(query):
            for feature in feature_set:
                overlaps[component].update(self.postings[component].get(feature, ()))
            candidates.update(overlaps[component])

        def metrics(i: int) -> tuple[float, float, float]:
            values = []
            for component in range(3):
                intersection = overlaps[component][i]
                union = len(query[component]) + len(self.features[i][component]) - intersection
                values.append(intersection / union if union else 0.0)
            return tuple(values)

        exact_index = self.exact.get(event_key(song))
        invariant_index = self.invariant.get(family_fingerprint(song))
        if exact_index is not None:
            closest, score = exact_index, 1.0
        elif invariant_index is not None:
            closest, score = invariant_index, 1.0
        else:
            closest, score = 0, 0.0
            for i in sorted(candidates):
                joint, pitch, rhythm = metrics(i)
                value = 0.55 * joint + 0.30 * pitch + 0.15 * rhythm
                if value > score:
                    closest, score = i, value
        joint, pitch, rhythm = metrics(closest)
        exact = exact_index is not None
        invariant = invariant_index is not None
        label = ("exact training match" if exact else "near match" if invariant or score >= 0.90
                 else "high similarity" if score >= 0.70 else "moderate similarity"
                 if score >= 0.40 else "low similarity")
        reference = self.songs[closest]
        return {
            "closest_index": closest, "closest_name": reference.name,
            "closest_id": self.ids[closest], "score": round(score, 6), "label": label,
            "exact_event_match": exact, "transposition_match": invariant,
            "rhythm_similarity": round(rhythm, 6), "ngram_overlap": round(joint, 6),
            "interval_similarity": round(pitch, 6),
            "same_bpm": song.bpm == reference.bpm,
            "generated_length": len(song.events), "reference_length": len(reference.events),
            "training_reference_count": len(self.songs),
            "nonzero_overlap_candidates": len(candidates),
            "search_scope": "all_training_songs_for_defined_metric",
            "metric": "weighted_transition_3gram_jaccard_v1",
            # Explicit aliases make JSON reports readable without library knowledge.
            "closest_training_id": self.ids[closest], "closest_training_name": reference.name,
            "invariant_match": invariant, "pitch_interval_similarity": round(pitch, 6),
            "rhythmic_similarity": round(rhythm, 6), "bpm_match": song.bpm == reference.bpm,
        }
