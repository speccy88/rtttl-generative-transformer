"""A fixed, invertible vocabulary with two tokens per musical event.

Grammar: BOS BPM (PITCH|REST DURATION)+ EOS.  BPM is retained exactly.
Pitch is MIDI 60..107 (RTTTL C4..B7), and duration is a denominator of a
whole note.  The vocabulary is specified before looking at any data.
"""
from __future__ import annotations

from typing import Any, Sequence

from .rtttl import Event, Song


class EventTokenizer:
    VERSION = 1
    MIN_PITCH = 60
    MAX_PITCH = 107
    MIN_BPM = 25
    MAX_BPM = 900
    DURATIONS = (1, 2, 4, 8, 16, 32)

    def __init__(self) -> None:
        self.tokens = ["<PAD>", "<BOS>", "<EOS>"]
        self.tokens += [f"<BPM_{bpm}>" for bpm in range(self.MIN_BPM, self.MAX_BPM + 1)]
        self.tokens += [f"<PITCH_{pitch}>" for pitch in range(self.MIN_PITCH, self.MAX_PITCH + 1)]
        self.tokens += ["<REST>"]
        self.tokens += [f"<DUR_{d}_{suffix}>" for d in self.DURATIONS for suffix in ("plain", "dot")]
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.vocab_size = len(self.tokens)
        self.bpm_ids = [self.token_to_id[f"<BPM_{b}>"] for b in range(self.MIN_BPM, self.MAX_BPM + 1)]
        self.pitch_ids = [self.token_to_id[f"<PITCH_{p}>"] for p in range(self.MIN_PITCH, self.MAX_PITCH + 1)]
        self.rest_id = self.token_to_id["<REST>"]
        self.event_start_ids = self.pitch_ids + [self.rest_id]
        self.duration_ids = [self.token_to_id[f"<DUR_{d}_{suffix}>"] for d in self.DURATIONS for suffix in ("plain", "dot")]
        self._bpm_by_id = dict(zip(self.bpm_ids, range(self.MIN_BPM, self.MAX_BPM + 1)))
        self._pitch_by_id = dict(zip(self.pitch_ids, range(self.MIN_PITCH, self.MAX_PITCH + 1)))
        self._duration_by_id = {self.token_to_id[f"<DUR_{d}_{suffix}>"]: (d, suffix == "dot")
                                for d in self.DURATIONS for suffix in ("plain", "dot")}

    def encode(self, song: Song) -> list[int]:
        if isinstance(song.bpm, bool) or int(song.bpm) != song.bpm or not self.MIN_BPM <= song.bpm <= self.MAX_BPM:
            raise ValueError(f"BPM must be an integer in [{self.MIN_BPM}, {self.MAX_BPM}]")
        if not song.events:
            raise ValueError("A song must contain at least one musical event")
        result = [self.bos_id, self.token_to_id[f"<BPM_{song.bpm}>"]]
        for event in song.events:
            pitch = event.pitch
            if pitch is None:
                result.append(self.rest_id)
            elif isinstance(pitch, bool) or int(pitch) != pitch or not self.MIN_PITCH <= pitch <= self.MAX_PITCH:
                raise ValueError(f"Pitch {pitch} is outside MIDI {self.MIN_PITCH}..{self.MAX_PITCH}")
            else:
                result.append(self.token_to_id[f"<PITCH_{pitch}>"])
            if isinstance(event.duration, bool) or event.duration not in self.DURATIONS:
                raise ValueError(f"Unsupported duration denominator: {event.duration}")
            if not isinstance(event.dotted, bool):
                raise ValueError("Dotted must be a boolean")
            suffix = "dot" if event.dotted else "plain"
            result.append(self.token_to_id[f"<DUR_{event.duration}_{suffix}>"])
        return result + [self.eos_id]

    def decode(self, ids: Sequence[int], name: str = "Generated") -> Song:
        ids = list(ids)
        # Right padding is a batching concern, and is safe to strip only at end.
        while ids and ids[-1] == self.pad_id:
            ids.pop()
        if len(ids) < 5 or ids[0] != self.bos_id or ids[-1] != self.eos_id:
            raise ValueError("Expected BOS BPM (PITCH/REST DURATION)+ EOS")
        if ids[1] not in self._bpm_by_id or (len(ids) - 3) % 2:
            raise ValueError("Invalid tempo or incomplete musical event")
        events = []
        for i in range(2, len(ids) - 1, 2):
            pitch_id, duration_id = ids[i:i + 2]
            if pitch_id not in self._pitch_by_id and pitch_id != self.rest_id:
                raise ValueError(f"Expected pitch/rest at token {i}")
            if duration_id not in self._duration_by_id:
                raise ValueError(f"Expected duration at token {i + 1}")
            duration, dotted = self._duration_by_id[duration_id]
            events.append(Event(pitch=self._pitch_by_id.get(pitch_id), duration=duration, dotted=dotted))
        return Song(name=name, bpm=self._bpm_by_id[ids[1]], events=tuple(events))

    def allowed_next(self, ids: Sequence[int], min_events: int = 1,
                     max_events: int | None = None) -> list[int]:
        """Legal next tokens; callers retain the full prefix for grammar state.

        This fast state lookup assumes a previously validated/generated prefix;
        decode() performs full validation on a completed sequence.
        """
        if min_events < 1 or (max_events is not None and max_events < min_events):
            raise ValueError("Require 1 <= min_events <= max_events")
        if not ids:
            return [self.bos_id]
        if ids[0] != self.bos_id:
            raise ValueError("A token sequence must start with BOS")
        if ids[-1] == self.eos_id:
            return []
        if len(ids) == 1:
            return list(self.bpm_ids)
        if ids[1] not in self._bpm_by_id:
            raise ValueError("BPM must follow BOS")
        if (len(ids) - 2) % 2:
            if ids[-1] not in self.event_start_ids:
                raise ValueError("Invalid event-start token")
            return list(self.duration_ids)
        if len(ids) > 2 and ids[-1] not in self._duration_by_id:
            raise ValueError("An event must end with duration")
        event_count = (len(ids) - 2) // 2
        if max_events is not None and event_count >= max_events:
            return [self.eos_id]
        return list(self.event_start_ids) + ([self.eos_id] if event_count >= min_events else [])

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self.tokens[int(i)] for i in ids]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.VERSION, "tokens": self.tokens,
                "representation": "BOS BPM (PITCH/REST DURATION)+ EOS"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventTokenizer":
        tokenizer = cls()
        if data.get("version") != cls.VERSION or data.get("tokens") != tokenizer.tokens:
            raise ValueError("Unsupported tokenizer version or vocabulary mismatch")
        return tokenizer
