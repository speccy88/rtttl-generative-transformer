#!/usr/bin/env python3
"""Create deterministic artificial examples for functional checks, not music research."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rtttl_gen.rtttl import Event, Song, encode_rtttl
from prepare_dataset import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/demo', help='New or empty directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix='rtttl_synthetic_') as directory:
        source = Path(directory) / 'synthetic.rtttl'
        lines = []
        for i in range(128):
            events = tuple(Event(pitch=None if rng.random() < 0.1 else rng.randint(60, 95),
                                 duration=rng.choice([4, 8, 16]), dotted=rng.random() < 0.15)
                           for _ in range(rng.randint(16, 32)))
            song = Song(name=f'Synthetic{i}', bpm=rng.choice([90, 100, 120, 140]), events=events)
            lines.append(encode_rtttl(song))
        source.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        prepare(source, args.output, args.seed)
    print('Artificial demo only: scores are not musical-quality evidence.')


if __name__ == '__main__':
    main()
