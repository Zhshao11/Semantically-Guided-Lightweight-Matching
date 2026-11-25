import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from sglm.config import load_config
from sglm.data import OSPairs


@pytest.fixture
def fake_data(tmp_path):
    root = tmp_path / 'OSDataset2.0'
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            folder = root / f'Patch-level Subset/OSdataset/{size}/{split}'
            folder.mkdir(parents=True)
            y, x = np.mgrid[:size, :size]
            gray = ((x + y) % 256).astype(np.uint8)
            cv2.imwrite(str(folder/'opt1.png'), gray)
            cv2.imwrite(str(folder/'sar1.png'), gray)
    return root


@pytest.mark.parametrize('resolution', [256, 512])
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_load_splits_channels_range_and_identity(fake_data, resolution, split):
    dataset = OSPairs(fake_data, resolution=resolution, split=split, affine=False)
    item = dataset[0]
    assert item['optical'].shape == (3, resolution, resolution)
    assert item['sar'].shape == (1, resolution, resolution)
    assert torch.equal(item['optical'][0], item['sar'][0])
    assert item['optical'].min() == 0 and item['optical'].max() == 1
    assert item['sar_mask'].all()
    assert item['pair_id'] == f'osdataset/{resolution}/{split}/1'


def test_manifest_split_filtering_and_path_safety(fake_data, tmp_path):
    manifest = tmp_path/'manifest.jsonl'
    rows = []
    for split in ('train', 'val', 'test'):
        rows.append(dict(schema='sglm.pair_manifest.v1', dataset='OSdataset', resolution=256,
            split=split, usable=True, pair_id=f'osdataset/256/{split}/1',
            optical={'path': f'Patch-level Subset/OSdataset/256/{split}/opt1.png'},
            sar={'path': f'Patch-level Subset/OSdataset/256/{split}/sar1.png'}))
    manifest.write_text('\n'.join(map(json.dumps, rows)))
    dataset = OSPairs(fake_data, split='val', manifest=manifest)
    assert len(dataset) == 1 and dataset[0]['pair_id'] == 'osdataset/256/val/1'
    rows[1]['optical']['path'] = 'Patch-level Subset/OSdataset/256/test/opt1.png'
    manifest.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='outside selected split'):
        OSPairs(fake_data, split='val', manifest=manifest)


def test_augmentation_determinism_and_train_epoch(fake_data):
    dataset = OSPairs(fake_data, split='val')
    first = dataset[0]
    dataset.epoch = 20
    assert torch.equal(first['transform'], dataset[0]['transform'])
    training = OSPairs(fake_data)
    first = training[0]
    training.epoch = 1
    assert not torch.equal(first['transform'], training[0]['transform'])
    assert not first['sar_mask'].all()


def test_config_validation_and_missing_pairs(fake_data, tmp_path):
    config = load_config()
    config['coordinates']['align_corners'] = True
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='coordinate'):
        load_config(path)
    (fake_data/'Patch-level Subset/OSdataset/256/train/sar1.png').unlink()
    with pytest.raises(ValueError, match='Unpaired'):
        OSPairs(fake_data)


def test_manifest_duplicate_analysis_preserves_ambiguity(tmp_path):
    from sglm.manifest import analyze_manifest
    rows = []
    for identifier in ('1', '2'):
        rows.append(dict(usable=True, dataset='OSdataset', resolution=512, split='train',
            pair_id=f'osdataset/512/train/{identifier}', optical={'decoded_sha256': 'same_opt'},
            sar={'decoded_sha256': 'same_sar'}, parent_candidates=[]))
    rows.append(dict(usable=True, dataset='OSdataset', resolution=256, split='train',
        pair_id='osdataset/256/train/1', optical={'decoded_sha256': 'child_opt'},
        sar={'decoded_sha256': 'child_sar'}, parent_candidates=[
            {'parent_pair_id': 'osdataset/512/train/1'}, {'parent_pair_id': 'osdataset/512/train/2'}]))
    path = tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(map(json.dumps, rows)))
    report = analyze_manifest(path)
    assert report['summary']['512/train']['unique_decoded_pairs'] == 1
    assert report['summary']['512/train']['redundant_pair_entries'] == 1
    assert report['ambiguous_children'] == report['ambiguity_explained_by_identical_512_pair_groups'] == 1
    assert report['cross_split_duplicate_pair_groups'] == 0
    # Report generation leaves all original entries intact.
    assert len(path.read_text().splitlines()) == 3
