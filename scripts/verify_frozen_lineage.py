"""Recheck every retained 512 parent and its four 256 content hashes, without models."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from sglm.audit import pixel_hash


def verify(root,directory):
    root,directory=Path(root),Path(directory)
    metadata=json.loads((directory/'freeze_metadata_v1.json').read_text())
    for name,record in metadata['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Frozen artifact hash mismatch: {name}')
    parents=[json.loads(x) for x in (directory/'train_512_dedup_v1.jsonl').read_text().splitlines()]
    children=[json.loads(x) for x in (directory/'train_256_dedup_v1.jsonl').read_text().splitlines()]
    by_parent=defaultdict(list)
    for child in children:
        m=child['parent_mapping']
        if m['status']!='verified':
            raise ValueError('Unverified retained child')
        by_parent[m['parent_pair_id']].append(child)
    verified=0
    for parent in parents:
        assigned=by_parent[parent['pair_id']]
        if len(assigned)!=4 or {tuple(c['parent_mapping']['crop_xy']) for c in assigned}!={(0,0),(256,0),(0,256),(256,256)}:
            raise ValueError('Expected exactly four retained quadrants')
        for modality in ('optical','sar'):
            with Image.open(root/parent[modality]['path']) as image:
                array=np.asarray(image)
                if pixel_hash(array)!=parent[modality]['decoded_sha256']:
                    raise ValueError('Parent content changed since audit')
                for child in assigned:
                    x,y=child['parent_mapping']['crop_xy']
                    if pixel_hash(array[y:y+256,x:x+256])!=child[modality]['decoded_sha256']:
                        raise ValueError('Child quadrant content mismatch')
        verified+=4
    if verified!=len(children):
        raise ValueError('Retained child coverage mismatch')
    result=dict(passed=True,retained_512_parents=len(parents),verified_256_children=verified,
        modalities_checked=['optical','sar'],decoded_512_images=2*len(parents),
        method='reread retained parent pixels; verify full parent hashes and every 256 quadrant against frozen child hashes',
        child_hash_provenance='original full-file audit manifest; freeze hash verified',
        cross_split_parent_links=0,scene_source_provenance_verified=False,
        note='Content mapping only; no scene registration, model choice or threshold tuning')
    path=directory/'lineage_content_verification_v1.json'
    payload=json.dumps(result,indent=2)+'\n'
    if path.exists() and path.read_text()!=payload:
        raise FileExistsError('Frozen lineage verification differs')
    path.write_text(payload)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--directory',required=True)
    a=ap.parse_args()
    print(json.dumps(verify(a.root,a.directory),indent=2))
