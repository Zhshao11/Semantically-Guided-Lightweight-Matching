import io
import numpy as np
import pytest
from PIL import Image
from sglm.audit import inspect_png, pixel_hash, scene_metadata_and_statistics, _archive_index


def test_full_decode_pixel_range_channels_and_damage():
    data = np.zeros((16, 24, 3), np.uint8)
    data[:, :, 0] = 7
    data[:, :, 1] = np.arange(24)
    data[:, :, 2] = 255
    stream = io.BytesIO()
    Image.fromarray(data).save(stream, format='PNG')
    raw = stream.getvalue()
    record, decoded = inspect_png(raw)
    assert record['channels'] == 3 and record['dtype'] == 'uint8'
    assert record['min'] == [7, 0, 255] and record['max'] == [7, 23, 255]
    assert np.array_equal(decoded, data)
    assert record['decoded_sha256'] == pixel_hash(data)
    assert pixel_hash(data[:, :12]) != pixel_hash(data[:, 12:])
    with pytest.raises(Exception):
        inspect_png(raw[:len(raw)//2])


def test_zip_inventory_rejects_wrong_root_and_repeated_paths(tmp_path):
    from zipfile import ZipFile
    path = tmp_path/'wrong.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('other/patch.png', b'abc')
    with pytest.raises(ValueError, match='Unexpected archive root'):
        _archive_index(path, 'OSDataset2.0')


def test_full_tiled_tiff_stats_exclude_padding_and_nodata(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    # Raster dimensions intentionally do not match tile size.
    array = np.arange(19 * 21, dtype=np.float32).reshape(19, 21)
    array[0, 0] = -9999
    array[1, 1] = np.nan
    path = tmp_path/'scene.tif'
    tifffile.imwrite(path, array, tile=(16, 16), metadata=None,
        extratags=[(42113, 's', 0, '-9999', False),
                   (33550, 'd', 3, (2., 3., 0.), False),
                   (33922, 'd', 6, (0., 0., 0., 100., 200., 0.), False)])
    record = scene_metadata_and_statistics(path)
    valid = array[np.isfinite(array) & (array != -9999)]
    assert record['valid_samples_per_channel'] == [len(valid)]
    assert record['nodata_samples_per_channel'] == [1]
    assert record['nonfinite_samples_per_channel'] == [1]
    assert record['min'] == [float(valid.min())] and record['max'] == [float(valid.max())]
    assert record['mean'][0] == pytest.approx(valid.astype(float).mean())
    assert record['gdal_geotransform'] == [100., 2., 0., 200., 0., -3.]


def test_full_audit_manifest_and_archive_closure(tmp_path):
    """Tiny complete package exercises the same archive->manifest->scene pipeline."""
    import json
    from zipfile import ZipFile
    from sglm.audit import audit_dataset
    from sglm.config import load_config
    tifffile = pytest.importorskip('tifffile')
    root = tmp_path/'OSDataset2.0'
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            folder = root/f'Patch-level Subset/OSdataset/{size}/{split}'
            folder.mkdir(parents=True)
            pixels = np.arange(size*size, dtype=np.uint8).reshape(size, size)
            for prefix in ('opt', 'sar'):
                Image.fromarray(pixels).save(folder/f'{prefix}1.png')
    with ZipFile(root/'Patch-level Subset/OSdataset/ReadMe.docx', 'w') as z:
        z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:t>Fixture</w:t></w:p></w:document>')
    scene = root/'Scene-level Subset'
    scene.mkdir()
    tifffile.imwrite(scene/'Optical.tif', np.zeros((16, 16, 3), np.uint16), tile=(16, 16), photometric='rgb')
    tifffile.imwrite(scene/'sar.tif', np.ones((16, 16), np.float32), tile=(16, 16))
    (scene/'GT.txt').write_text('1 2 2 3\n3 4 4 5\n5 1 6 2\n')
    archive = tmp_path/'fixture.zip'
    files = sorted(p for p in root.rglob('*') if p.is_file())
    with ZipFile(archive, 'w') as z:
        for p in files:
            z.write(p, p.relative_to(tmp_path).as_posix())
    config = load_config()
    config['dataset']['expected_pairs'] = {str(s): {x: 1 for x in ('train','val','test')} for s in (256,512)}
    config['dataset']['expected_new_region_pairs'] = 0
    config['dataset']['expected_archive_files'] = len(files)
    config['control_points']['expected_rows'] = 3
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    report = audit_dataset(root, tmp_path/'output', archive, path)
    assert report['errors'] == []
    assert report['archive']['size_crc_matches'] == len(files)
    assert report['totals']['pairs'] == 6
    assert report['control_points']['rows'] == 3
    manifest = [json.loads(x) for x in (tmp_path/'output/dataset_manifest.jsonl').read_text().splitlines()]
    assert len(manifest) == 6 and len({x['pair_id'] for x in manifest}) == 6
