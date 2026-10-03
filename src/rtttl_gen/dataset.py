"""Event-aligned, tempo-conditioned next-token windows; original targets counted once."""
from __future__ import annotations
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .rtttl import song_from_dict
from .augment import transpose


def load_records(path: str | Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_songs(path: str | Path):
    return [song_from_dict(row['song']) for row in load_records(path)]


class SequenceDataset(Dataset):
    """Every excerpt starts BOS/BPM. Only first predicts BPM; only last predicts EOS.

    Two tokens/event, no cross-song attention, no tail truncation. Continuity
    across rare long-song excerpt boundaries is sacrificed to bound memory.
    """
    def __init__(self, songs, tokenizer, context_length: int,
                 augment_semitones: int = 0, seed: int = 42):
        if context_length < 4:
            raise ValueError('context_length must be >=4')
        if augment_semitones < 0:
            raise ValueError('augment_semitones must be nonnegative')
        self.songs, self.tokenizer = songs, tokenizer
        self.context_length, self.augment_semitones = context_length, augment_semitones
        self.seed, self.epoch = seed, 0
        self.events_per_chunk = (context_length - 2) // 2
        self.windows = [(i, start) for i, s in enumerate(songs)
                        for start in range(0, len(s.events), self.events_per_chunk)]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        song_idx, start = self.windows[index]
        song = self.songs[song_idx]
        if self.augment_semitones:
            pitches = [e.pitch for e in song.events if e.pitch is not None]
            low, high = -self.augment_semitones, self.augment_semitones
            if pitches:
                low, high = max(low, 60-min(pitches)), min(high, 107-max(pitches))
            rng = random.Random(self.seed + 1000003*self.epoch + 9176*song_idx)
            song = transpose(song, rng.randint(low, high))
        encoded = self.tokenizer.encode(song)
        stop = min(start + self.events_per_chunk, len(song.events))
        sequence = encoded[:2] + encoded[2+2*start:2+2*stop]
        if stop == len(song.events):
            sequence.append(self.tokenizer.eos_id)
        x = torch.full((self.context_length,), self.tokenizer.pad_id, dtype=torch.long)
        y = x.clone()
        x[:len(sequence)-1] = torch.tensor(sequence[:-1], dtype=torch.long)
        y[:len(sequence)-1] = torch.tensor(sequence[1:], dtype=torch.long)
        if start:
            y[0] = self.tokenizer.pad_id
        return {'input_ids': x, 'targets': y}
