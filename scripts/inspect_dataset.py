#!/usr/bin/env python3
"""Audit source records without constructing training splits."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rtttl_gen.corpus import inspect_corpus, source_inventory
from prepare_dataset import dump_json, dump_jsonl

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True, help='New or empty audit directory')
    a = p.parse_args()
    dest = Path(a.output)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Use a new output folder')
    dest.mkdir(parents=True, exist_ok=True)
    records, audit, stats = inspect_corpus(a.input)
    dump_json(dest/'stats.json', stats)
    dump_jsonl(dest/'parse_audit.jsonl', audit)
    dump_jsonl(dest/'source_inventory.jsonl', source_inventory(a.input))
    print(f'{len(records)} accepted records. Audit saved to {dest}')

if __name__ == '__main__':
    main()
