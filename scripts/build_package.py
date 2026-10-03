#!/usr/bin/env python3
"""Build a public source-only ZIP from the checked-in release allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / 'PUBLIC_RELEASE_FILES.txt'


def included(path):
    """Never infer that an arbitrary local artifact is safe to publish."""
    relative = path.relative_to(ROOT).as_posix()
    allowed = {line.strip() for line in ALLOWLIST.read_text().splitlines()
               if line.strip() and not line.startswith('#')}
    return path.is_file() and not path.is_symlink() and relative in allowed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT.parent/'RTTTL_Generative_Transformer_Source.zip'))
    args=parser.parse_args();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    paths=sorted(p for p in ROOT.rglob('*') if included(p))
    manifest={'format_version':1,'root':'RTTTL_Generative_Transformer','files':{
        p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in paths}}
    manifest_path=ROOT/'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    paths.append(manifest_path)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths:archive.write(p,Path('RTTTL_Generative_Transformer')/p.relative_to(ROOT))
    with zipfile.ZipFile(out) as archive:
        error=archive.testzip()
        if error:raise ValueError(f'ZIP CRC failed: {error}')
    print(json.dumps({'zip':str(out),'files':len(paths),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
