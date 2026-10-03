#!/usr/bin/env python3
"""Inspect, normalize, group musical families, and freeze reproducible splits."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rtttl_gen.corpus import inspect_corpus, source_inventory
from rtttl_gen.rtttl import song_from_dict, encode_rtttl, parse_rtttl
from rtttl_gen.deduplicate import group_families, split_families, assert_no_leakage
from rtttl_gen.tokenizer import EventTokenizer


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def dump_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False)+'\n')


def prepare(input_path, output_path, seed=42):
    started = time.perf_counter()
    output = Path(output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a NEW output directory to preserve prior evidence.')
    output.mkdir(parents=True, exist_ok=True)
    records, audit, stats = inspect_corpus(input_path)
    dump_json(output / 'stats_inspection.json', stats)
    dump_jsonl(output / 'source_inventory.jsonl', source_inventory(input_path))
    dump_jsonl(output / 'parse_audit.jsonl', audit)
    if not records:
        raise ValueError('No accepted RTTTL records. See parse_audit.jsonl; source data were not changed.')
    songs = [song_from_dict(row['song']) for row in records]
    tokenizer = EventTokenizer()
    # Validate the whole accepted corpus, not just unit-test examples.
    for song in songs:
        recovered = parse_rtttl(encode_rtttl(song))
        assert (song.bpm, song.events) == (recovered.bpm, recovered.events)
        recovered = tokenizer.decode(tokenizer.encode(song))
        assert (song.bpm, song.events) == (recovered.bpm, recovered.events)
    print(f'Parsed {len(songs)} valid melodies; grouping families...', flush=True)
    family_ids, duplicate_of, dedup_report = group_families(songs)
    retained = [i for i, parent in enumerate(duplicate_of) if parent is None]
    kept_songs = [songs[i] for i in retained]
    kept_families = [family_ids[i] for i in retained]
    if len(set(kept_families)) < 3:
        raise ValueError('Need at least three musical families for train/validation/test. Inspection audit retained.')
    splits = split_families(kept_families, seed=seed)
    assert_no_leakage(kept_songs, splits, kept_families)
    split_by_original = {original: split for original, split in zip(retained, splits)}
    processed = {'train': [], 'val': [], 'test': []}
    mapping = []
    for i, row in enumerate(records):
        representative = i if duplicate_of[i] is None else duplicate_of[i]
        split = split_by_original[representative]
        split = 'val' if split == 'validation' else split
        mapping.append({'id': row['id'], 'source': row['source'], 'line': row.get('line'),
                        'family_id': family_ids[i], 'split': split,
                        'duplicate_of': None if representative == i else records[representative]['id'],
                        'retained_id': records[representative]['id']})
        if i == representative:
            processed[split].append({**row, 'family_id': family_ids[i]})
    for split, rows in processed.items():
        dump_jsonl(output / f'{split}.jsonl', rows)
    dump_jsonl(output / 'manifest.jsonl', mapping)
    dump_json(output / 'tokenizer.json', tokenizer.to_dict())
    dump_json(output / 'deduplication.json', dedup_report)
    summary = {
        'seed': seed, 'inspection': stats, 'deduplication': dedup_report,
        'retained_songs': len(retained), 'exact_copies_removed': len(songs)-len(retained),
        'musical_families': len(set(kept_families)),
        'splits': {k: {'songs': len(v), 'families': len({r['family_id'] for r in v}),
                       'events': sum(len(r['song']['events']) for r in v)} for k,v in processed.items()},
        'vocab_size': tokenizer.vocab_size,
        'roundtrip_songs_checked': len(songs), 'leakage_checks_passed': True,
        'context_recommendation_tokens': 256,
        'preparation_seconds': time.perf_counter()-started,
    }
    dump_json(output / 'stats.json', summary)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    dump_json(output / 'checksums.json', hashes)
    print(json.dumps({k: summary[k] for k in ('retained_songs','exact_copies_removed','musical_families','splits','vocab_size','preparation_seconds')}, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='ZIP, directory, or RTTTL text file (read-only)')
    p.add_argument('--output', default='data/processed', help='Must be new or empty')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    prepare(args.input, args.output, args.seed)

if __name__ == '__main__':
    main()
