"""Freeze derived joint-content training lists and summarize control-point metadata.

No model, geometric transform fit, registration score, or threshold selection is used.
The original manifest and source files are never modified.
"""
import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from .geometry import read_control_points


def joint_content_id(pair):
    """Ordered pair of decoded content hashes; either modality changing changes the key."""
    optical = pair['optical']['decoded_sha256']
    sar = pair['sar']['decoded_sha256']
    return hashlib.sha256(json.dumps([optical, sar], separators=(',', ':')).encode()).hexdigest()


def representative_key(pair):
    return int(pair['pair_id'].rsplit('/', 1)[1]), pair['pair_id']


def parent_mapping(pair, by_id, canonical=None):
    """Verified identity requires one parent ID+crop and recorded joint pixel evidence."""
    raw = pair.get('parent_candidates', [])
    resolved = {}
    for candidate in raw:
        original_id = candidate['parent_pair_id']
        if original_id not in by_id:
            raise ValueError(f'Unknown parent {original_id}')
        parent = by_id[original_id]
        if parent['resolution'] != 512 or candidate['parent_split'] != parent['split'] or parent['split'] != pair['split']:
            raise ValueError(f'Parent resolution/split mismatch for {pair["pair_id"]}')
        crop = candidate.get('crop_xy', [])
        if len(crop) != 2 or any(x not in (0, 256) for x in crop):
            raise ValueError('Expected a builtin 512 quadrant origin')
        parent_id = canonical.get(original_id, original_id) if canonical else original_id
        key = parent_id, tuple(crop)
        evidence = resolved.setdefault(key, dict(parent_pair_id=parent_id, crop_xy=crop,
            original_parent_ids=[], evidence=candidate.get('proof')))
        if original_id not in evidence['original_parent_ids']:
            evidence['original_parent_ids'].append(original_id)
    options = sorted(resolved.values(), key=lambda x: (x['parent_pair_id'], x['crop_xy']))
    for option in options:
        option['original_parent_ids'].sort()
    verified = len(options) == 1 and all(c.get('proof') == 'exact_decoded_pixels_both_modalities' for c in raw)
    status = 'verified' if verified else ('candidate' if options else 'unresolved')
    return dict(status=status, scope='content_mapping_to_distributed_512_patch',
                parent_pair_id=options[0]['parent_pair_id'] if verified else None,
                crop_xy=options[0]['crop_xy'] if verified else None, candidates=options,
                identity_basis='canonical_joint_content_representative' if canonical else 'original_distributed_pair_id',
                reason='unique_parent_and_crop_with_joint_pixel_proof' if verified else
                       ('multiple_or_incomplete_parent_evidence' if options else 'no_parent_mapping_in_package'))


def derive_manifests(pairs, expected=None):
    by_id = {p['pair_id']: p for p in pairs}
    if len(by_id) != len(pairs):
        raise ValueError('Duplicate pair IDs in source manifest')
    training = [p for p in pairs if p['dataset'] == 'OSdataset' and p['split'] == 'train']
    groups = defaultdict(list)
    for pair in training:
        if not pair['usable'] or pair['resolution'] not in (256, 512):
            raise ValueError('Unusable or unsupported training pair')
        groups[(pair['resolution'], joint_content_id(pair))].append(pair)
    canonical, decisions, duplicate_groups = {}, [], []
    for (size, content), members in sorted(groups.items()):
        members.sort(key=representative_key)
        keep = members[0]['pair_id']
        for pair in members:
            canonical[pair['pair_id']] = keep
            decisions.append(dict(resolution=size, original_pair_id=pair['pair_id'], kept_pair_id=keep,
                action='keep' if pair['pair_id'] == keep else 'exclude', joint_content_id=content,
                optical_decoded_sha256=pair['optical']['decoded_sha256'], sar_decoded_sha256=pair['sar']['decoded_sha256']))
        if len(members) > 1:
            duplicate_groups.append(dict(resolution=size, joint_content_id=content, kept_pair_id=keep,
                excluded_pair_ids=[p['pair_id'] for p in members[1:]], group_size=len(members)))
    annotated, derived = [], {256: [], 512: []}
    for pair in pairs:
        record = copy.deepcopy(pair)
        record['parent_mapping'] = parent_mapping(pair, by_id)
        # Geographic provenance is a separate conclusion; content equivalence cannot fill it.
        record['scene_source_mapping'] = dict(status='unresolved', scene_id=pair.get('scene_id'),
            scope='original_satellite_scene_provenance', reason='no_patch_to_original_scene_index_in_package')
        annotated.append(record)
        if pair['pair_id'] in canonical and canonical[pair['pair_id']] == pair['pair_id']:
            kept = copy.deepcopy(record)
            kept['parent_mapping'] = parent_mapping(pair, by_id, canonical)
            kept['deduplication'] = dict(protocol='joint-content-train-dedup-v1',
                joint_content_id=joint_content_id(pair), representative_rule='smallest_numeric_sample_id_within_resolution_train')
            derived[pair['resolution']].append(kept)
    for size in derived:
        derived[size].sort(key=representative_key)
    kept_parent_ids = {p['pair_id'] for p in derived[512]}
    slots = defaultdict(list)
    for child in derived[256]:
        mapping = child['parent_mapping']
        if mapping['status'] != 'verified' or mapping['parent_pair_id'] not in kept_parent_ids:
            raise ValueError(f'Deduplicated child has no unique retained parent: {child["pair_id"]}')
        slots[(mapping['parent_pair_id'], tuple(mapping['crop_xy']))].append(child['pair_id'])
    expected_slots = {(p, (x, y)) for p in kept_parent_ids for y in (0, 256) for x in (0, 256)}
    if set(slots) != expected_slots or any(len(v) != 1 for v in slots.values()):
        raise ValueError('Dedup 256-to-512 relation must be exactly four unique quadrants per retained parent')
    summary = {}
    for size in (256, 512):
        original = sum(p['resolution'] == size for p in training)
        summary[str(size)] = dict(original=original, kept=len(derived[size]), excluded=original-len(derived[size]),
                                 duplicate_groups=sum(g['resolution'] == size for g in duplicate_groups))
        if expected and summary[str(size)] != expected[str(size)]:
            raise ValueError(f'Freeze count mismatch: {summary[str(size)]}')
    return dict(annotated=annotated, derived=derived, decisions=decisions, duplicate_groups=duplicate_groups,
        summary=summary, lineage=dict(retained_children=len(derived[256]), retained_parents=len(derived[512]),
            verified_links=len(slots), children_per_parent=4, complete_quadrant_coverage=True,
            geographic_scene_provenance_verified=False),
        original_parent_status=dict(Counter(p['parent_mapping']['status'] for p in annotated)),
        retained_256_parent_status=dict(Counter(p['parent_mapping']['status'] for p in derived[256])))


def spatial_coverage(points, width, height, near_collinear_ratio=.01):
    """Metadata diagnostic: bbox, convex hull and normalized-coordinate PCA.

    near_collinear_ratio is a declared metadata rule, never a model threshold.
    """
    points = np.asarray(points, dtype=np.float64)
    normalized = points / np.array([width-1, height-1], dtype=float)
    minimum, maximum = points.min(0), points.max(0)
    hull = cv2.convexHull(normalized.astype(np.float32))
    area = float(cv2.contourArea(hull)) if len(points) >= 3 else 0.
    singular = np.linalg.svd(normalized-normalized.mean(0), compute_uv=False)
    ratio = float(singular[1]/singular[0]) if len(singular) > 1 and singular[0] > 0 else 0.
    cells = np.minimum((normalized*4).astype(int), 3).clip(0, 3)
    return dict(bbox_min_xy=minimum.tolist(), bbox_max_xy=maximum.tolist(),
        bbox_span_fraction_xy=((maximum-minimum)/[width-1, height-1]).tolist(),
        convex_hull_area_fraction=area, grid_4x4_occupied=int(len(np.unique(cells, axis=0))),
        pca_minor_major_singular_ratio=ratio, near_collinear=bool(ratio < near_collinear_ratio),
        near_collinear_rule=dict(metric='normalized_xy_minor_major_singular_ratio', less_than=near_collinear_ratio))


def group_control_metadata(control_path, audit):
    optical, sar = read_control_points(control_path)
    scenes = audit['scene']
    group = dict(pair_id='osdataset2/scene/optical-sar', point_count=len(optical),
        images={m: dict(path='Scene-level Subset/'+scenes[m]['path'], width=scenes[m]['width'],
                       height=scenes[m]['height']) for m in ('optical','sar')},
        coordinate_order=['x_optical','y_optical','x_sar','y_sar'], axis_order='xy_column_row',
        coordinate_origin=dict(project_assumption_index_base=0, project_assumption_pixel_reference='center',
            original_annotation_index_base='unverified', original_annotation_pixel_reference='unverified',
            values_modified=False, evidence='package Evaluation.py defaults to offset_mode=none; annotation convention unproven'),
        usage='metadata_only_no_scene_model_selection_or_threshold_tuning',
        spatial_coverage={m: spatial_coverage(points,scenes[m]['width'],scenes[m]['height'])
                          for m,points in [('optical',optical),('sar',sar)]})
    for m,points in [('optical',optical),('sar',sar)]:
        size=scenes[m]
        if ((points < 0) | (points > [size['width']-1,size['height']-1])).any():
            raise ValueError('Control point outside image bounds')
    return dict(image_pair_count=1, total_control_points=len(optical), groups=[group],
                note='72 control-point correspondences belong to one distributed image pair; not 72 image pairs')


def _json(value):
    return json.dumps(value,ensure_ascii=False,indent=2)+'\n'


def _jsonl(records):
    return ''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in records)


def freeze_metadata(manifest, audit_path, control_path, output):
    manifest, audit_path, control_path, output=map(Path,(manifest,audit_path,control_path,output))
    source_bytes=manifest.read_bytes()
    audit=json.loads(audit_path.read_text())
    source_hash=hashlib.sha256(source_bytes).hexdigest()
    if source_hash != audit['root_manifest_sha256']:
        raise ValueError('Source manifest differs from frozen audit')
    expected={'256':dict(original=8044,kept=7912,excluded=132,duplicate_groups=132),
              '512':dict(original=2011,kept=1978,excluded=33,duplicate_groups=33)}
    result=derive_manifests([json.loads(x) for x in source_bytes.splitlines() if x.strip()],expected)
    controls=group_control_metadata(control_path,audit)
    if controls['total_control_points'] != 72:
        raise ValueError('Expected exactly 72 control-point rows')
    content={
        'dataset_manifest_parent_status_v1.jsonl':_jsonl(result['annotated']),
        'train_256_dedup_v1.jsonl':_jsonl(result['derived'][256]),
        'train_512_dedup_v1.jsonl':_jsonl(result['derived'][512]),
        'duplicate_groups_v1.jsonl':_jsonl(result['duplicate_groups']),
        'control_point_groups_v1.json':_json(controls)}
    csv_buffer=io.StringIO(newline='')
    fields=list(result['decisions'][0])
    writer=csv.DictWriter(csv_buffer,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(result['decisions'],key=lambda x:(x['resolution'],int(x['original_pair_id'].rsplit('/',1)[1]))))
    content['train_keep_exclude_mapping_v1.csv']=csv_buffer.getvalue()
    summary={k:result[k] for k in ('summary','lineage','original_parent_status','retained_256_parent_status')}
    summary.update(protocol='joint-content-train-dedup-v1', source_manifest_sha256=source_hash,
        source_control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        original_files_modified=False, original_manifest_modified=False,
        representative_rule='minimum numeric sample ID independently within train/resolution',
        deduplication_key='ordered optical_decoded_sha256 AND sar_decoded_sha256',
        control_image_pairs=controls['image_pair_count'], control_points=controls['total_control_points'],
        files={name:dict(bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest()) for name,text in content.items()})
    content['freeze_metadata_v1.json']=_json(summary)
    # A published freeze is immutable: only byte-identical regeneration is allowed.
    for name,text in content.items():
        path=output/name
        if path.exists() and path.read_bytes()!=text.encode():
            raise FileExistsError(f'Refusing to overwrite a different frozen artifact: {path}')
    output.mkdir(parents=True,exist_ok=True)
    for name,text in content.items():
        (output/name).write_bytes(text.encode())
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError('Source manifest changed during freeze')
    return summary
