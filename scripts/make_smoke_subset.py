#!/usr/bin/env python3
"""Freeze a small family-disjoint subset for functional experiments, not ranking."""
import argparse
from pathlib import Path
import json
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rtttl_gen.dataset import load_records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/processed')
    p.add_argument('--output',default='data/smoke')
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(20261003)
    evidence={'seed':20261003,'source':a.input,'purpose':'functional smoke only','split_ids':{}}
    for split,n in [('train',64),('val',16),('test',16)]:
        rows=load_records(Path(a.input)/f'{split}.jsonl')
        selected=rng.sample(rows,min(n,len(rows)))
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in selected:f.write(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n')
        evidence['split_ids'][split]=[r['id'] for r in selected]
    shutil.copy2(Path(a.input)/'tokenizer.json',out/'tokenizer.json')
    (out/'subset_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({s:len(ids) for s,ids in evidence['split_ids'].items()}))

if __name__=='__main__':main()
