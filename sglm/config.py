"""Portable, validated data/geometry configuration; machine paths stay outside Git."""
import copy
import json
import os
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'configs/data_geometry_v1.json'


def load_config(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported data/geometry schema version')
    if config['coordinates'] != {'axis_order': 'xy', 'index_base': 0, 'pixel_reference': 'center',
                                 'align_corners': False, 'transform_direction': 'source_to_destination'}:
        raise ValueError('Unsupported coordinate convention')
    splits = config['dataset']['expected_pairs']
    for size in ('256', '512'):
        if set(splits[size]) != {'train', 'val', 'test'} or any(v <= 0 for v in splits[size].values()):
            raise ValueError('Expected all train/val/test pair counts')
    aug = config['augmentation']
    if aug['scale'][0] <= 0 or aug['scale'][0] > aug['scale'][1]:
        raise ValueError('Invalid scale range')
    for name in ('rotation_deg', 'translation_fraction'):
        if aug[name][0] > aug[name][1]:
            raise ValueError(f'Invalid {name} range')
    if config['geometry']['ransac_threshold_px'] <= 0:
        raise ValueError('RANSAC threshold must be positive')
    return copy.deepcopy(config)


def resolve_data_root(root=None):
    value = root or os.environ.get('SGLM_DATA_ROOT')
    if not value:
        raise ValueError('Set --root or SGLM_DATA_ROOT to the OSDataset2.0 directory')
    path = Path(value).expanduser().resolve()
    if not (path / 'Patch-level Subset/OSdataset').is_dir():
        raise ValueError(f'Expected OSDataset2.0 root, got {path}')
    return path
