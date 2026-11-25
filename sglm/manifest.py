"""Content-based pair deduplication summaries without changing official splits."""
import json
from collections import defaultdict
from pathlib import Path


def analyze_manifest(manifest_path):
    pairs = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for pair in pairs:
        if not pair['usable']:
            continue
        groups[(pair['dataset'], pair['resolution'], pair['optical']['decoded_sha256'], pair['sar']['decoded_sha256'])].append(pair)
    duplicates = [dict(dataset=key[0], resolution=key[1], pair_ids=[p['pair_id'] for p in values],
                       splits=[p['split'] for p in values], optical_pixel_sha256=key[2], sar_pixel_sha256=key[3])
                  for key, values in groups.items() if len(values) > 1]
    summary = {}
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            selected = [p for p in pairs if p['dataset'] == 'OSdataset' and p['resolution'] == size and p['split'] == split]
            unique = {(p['optical']['decoded_sha256'], p['sar']['decoded_sha256']) for p in selected if p['usable']}
            group_count = sum(d['dataset'] == 'OSdataset' and d['resolution'] == size and split in d['splits'] for d in duplicates)
            summary[f'{size}/{split}'] = dict(pairs=len(selected), unique_decoded_pairs=len(unique),
                redundant_pair_entries=len(selected) - len(unique), duplicate_groups=group_count)
    ambiguous = [p for p in pairs if len(p['parent_candidates']) > 1]
    duplicate_group_sets = [set(d['pair_ids']) for d in duplicates if d['dataset'] == 'OSdataset' and d['resolution'] == 512]
    explained = sum(any({c['parent_pair_id'] for c in p['parent_candidates']} <= group for group in duplicate_group_sets) for p in ambiguous)
    return dict(method='both modalities must have identical decoded pixel hashes; no split changes or automatic deduplication',
                summary=summary, groups=duplicates, ambiguous_children=len(ambiguous),
                ambiguity_explained_by_identical_512_pair_groups=explained,
                cross_split_duplicate_pair_groups=sum(len(set(d['splits']) - {'unassigned'}) > 1 for d in duplicates))
