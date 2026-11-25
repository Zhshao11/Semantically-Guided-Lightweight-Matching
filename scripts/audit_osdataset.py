"""Read-only inventory, strict opt/sar pairing and exact-content split audit."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def audit(root):
    root = Path(root)
    report = {'root': str(root), 'subsets': {}, 'exact_cross_split_duplicates': {}}
    hashes = defaultdict(list)
    manifests = {}
    for folder in sorted((root / 'Patch-level Subset').rglob('*')):
        if not folder.is_dir():
            continue
        opt = {p.stem[3:]: p for p in folder.glob('opt*.png')}
        sar = {p.stem[3:]: p for p in folder.glob('sar*.png')}
        if not opt and not sar:
            continue
        key = str(folder.relative_to(root))
        ids = sorted(opt.keys() & sar.keys(), key=int)
        sizes, modes = Counter(), Counter()
        entries = []
        for id_ in ids:
            pair = {}
            for modality, paths in [('optical', opt), ('sar', sar)]:
                p = paths[id_]
                with Image.open(p) as im:
                    sizes[str(im.size)] += 1
                    modes[im.mode] += 1
                    im.verify()
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                pair[modality] = str(p.relative_to(root))
                pair[modality + '_sha256'] = sha
                if folder.name in ('train', 'val', 'test'):
                    hashes[(folder.parent.name, modality, sha)].append((folder.name, p.name))
            entries.append({'id': id_, **pair})
        report['subsets'][key] = {
            'pairs': len(ids), 'unpaired_optical': sorted(opt.keys() - sar.keys()),
            'unpaired_sar': sorted(sar.keys() - opt.keys()),
            'image_sizes': dict(sizes), 'image_modes': dict(modes),
            'non_image_files': [p.name for p in folder.iterdir() if p.is_file() and p.suffix != '.png'],
        }
        manifests[key] = entries
    duplicates = defaultdict(list)
    for (resolution, modality, sha), locations in hashes.items():
        if len({x[0] for x in locations}) > 1:
            duplicates[resolution + '/' + modality].append({'sha256': sha, 'locations': locations})
    report['exact_cross_split_duplicates'] = dict(duplicates)
    report['gt_candidates'] = [str(p.relative_to(root)) for p in root.rglob('*')
                               if p.is_file() and p.suffix.lower() in ('.txt', '.mat', '.npy', '.npz', '.csv', '.json')]
    gt = root / 'Scene-level Subset/GT.txt'
    rows = [[float(x) for x in line.split()] for line in gt.read_text().splitlines() if line.strip()]
    report['scene_gt'] = {'rows': len(rows), 'columns': sorted({len(x) for x in rows}), 'first_row': rows[0]}
    return report, manifests


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    report, manifests = audit(a.root)
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    for key, entries in manifests.items():
        (out / (key.replace('/', '_').replace(' ', '_') + '.json')).write_text(json.dumps(entries, indent=2))
    print(json.dumps(report, indent=2))
