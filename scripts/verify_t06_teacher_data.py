"""Verify complete mirrored source split before controlled teacher updates."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('runs/t06_teacher_data');manifest=json.loads((root/'manifest.json').read_text());count=0
    for split in ['train','val64']:
        for row in manifest[split]:
            for entry in row['files'].values():
                path=root/entry['path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(f'Source member hash mismatch:{path}')
                count+=1
    train={r['pair_id'] for r in manifest['train']};val={r['pair_id'] for r in manifest['val64']}
    old={r['pair_id'] for r in json.loads(Path('runs/loveda_val_t05_v1/manifest.json').read_text())['samples']}
    assert len(train)==2522 and len(val)==64 and not train&val and not train&old and not val&old
    result=dict(passed=True,files_sha256_verified=count,train_images=2522,new_val_images=64,old8_separate=True,
        no_split_overlap=True,complete_official_Train_zip_MD5=manifest['train_zip_md5'],test_downloaded=False,
        source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    Path('reports/t06_setup_v1/teacher_data_acceptance.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
