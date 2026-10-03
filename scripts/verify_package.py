#!/usr/bin/env python3
"""Fast portable package/data integrity check (no training or network)."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rtttl_gen.dataset import load_records
from rtttl_gen.rtttl import song_from_dict,encode_rtttl,parse_rtttl
from rtttl_gen.deduplicate import assert_no_leakage
from rtttl_gen.tokenizer import EventTokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/processed')
    a=parser.parse_args();root=Path(a.data)
    checksums=json.loads((root/'checksums.json').read_text())
    for name,expected in checksums.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, f'Checksum mismatch: {name}'
    songs,families,splits=[],[],[]
    tokenizer=EventTokenizer()
    for split in ['train','val','test']:
        for row in load_records(root/f'{split}.jsonl'):
            song=song_from_dict(row['song'])
            decoded=parse_rtttl(encode_rtttl(song))
            detokenized=tokenizer.decode(tokenizer.encode(song))
            assert song.events==decoded.events==detokenized.events and song.bpm==decoded.bpm==detokenized.bpm
            songs.append(song);families.append(row['family_id']);splits.append("validation" if split=="val" else split)
    assert_no_leakage(songs,splits,families)
    print(json.dumps({'status':'PASS','files_hashed':len(checksums),'songs_roundtripped':len(songs),'families':len(set(families)),'leakage_assertions':'passed'},indent=2))

if __name__=='__main__':main()
