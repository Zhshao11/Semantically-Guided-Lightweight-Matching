"""Exhaustive read-only OSdataset audit. No network or original-data mutations."""
import csv
import hashlib
import io
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .config import load_config, resolve_data_root
from .geometry import (read_control_points, valid_pixel_centers, fit_affine_least_squares,
                       transform_points, invert_affine, estimate_affine)


def pixel_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f'{array.dtype}/{array.shape}/'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def inspect_png(data):
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()  # Full pixel decoding, beyond header/CRC validation.
        array = np.asarray(image).copy()
        mode, info = image.mode, {k: str(v)[:200] for k, v in image.info.items()}
    if array.ndim not in (2, 3):
        raise ValueError(f'Unsupported array shape {array.shape}')
    channels = array.reshape(-1, 1) if array.ndim == 2 else array.reshape(-1, array.shape[-1])
    float_channels = channels.astype(np.float64)
    record = dict(width=array.shape[1], height=array.shape[0], mode=mode,
                  channels=channels.shape[1], dtype=str(array.dtype),
                  min=channels.min(0).tolist(), max=channels.max(0).tolist(),
                  mean=float_channels.mean(0).tolist(), std=float_channels.std(0).tolist(),
                  zero_fraction=(channels == 0).mean(0).tolist(),
                  constant=bool((channels.min(0) == channels.max(0)).all()),
                  decoded_sha256=pixel_hash(array), png_metadata=info)
    return record, array


def _archive_index(archive, root_name):
    if archive is None:
        return {}, dict(enabled=False)
    with ZipFile(archive) as z:
        infos = [x for x in z.infolist() if not x.is_dir()]
    paths = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.parts[0] != root_name or '..' in path.parts:
            raise ValueError(f'Unexpected archive root/path: {info.filename}')
        relative = str(PurePosixPath(*path.parts[1:]))
        if relative in paths:
            raise ValueError(f'Duplicate archive entry: {relative}')
        paths[relative] = info
    return paths, dict(enabled=True, archive_name=Path(archive).name,
                      archive_size_bytes=Path(archive).stat().st_size,
                      files=len(paths), declared_uncompressed_bytes=sum(x.file_size for x in infos),
                      method='extracted file size + streaming CRC32 vs local ZIP central directory; SHA256 recorded',
                      official_checksum_verified=False, compressed_stream_tested=False)


def _file_digest(path):
    sha, crc, size = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return sha.hexdigest(), crc & 0xffffffff, size


def scene_metadata_and_statistics(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        h, w = page.imagelength, page.imagewidth
        channels = page.samplesperpixel
        nodata_tag = page.tags.get(42113)
        nodata_text = None if nodata_tag is None else str(nodata_tag.value)
        nodata = None if nodata_text is None else np.asarray(float(nodata_text), dtype=page.dtype).item()
        record = dict(path=path.name, width=w, height=h, channels=channels,
                      shape=list(page.shape), axes=page.axes, dtype=str(page.dtype), bigtiff=tif.is_bigtiff,
                      tile_wh=[page.tilewidth, page.tilelength], compression=str(page.compression),
                      nodata_tag_text=nodata_text, nodata_in_storage_dtype=nodata,
                      geotiff=tif.geotiff_metadata or {}, statistics_scope='all decoded tiles, cropped to true raster bounds')
        lo, hi = np.full(channels, np.inf), np.full(channels, -np.inf)
        sums, squares = np.zeros(channels), np.zeros(channels)
        counts, nodata_counts, nonfinite_counts, zeros = [np.zeros(channels, dtype=np.int64) for _ in range(4)]
        decoded = 0
        for array, index, _ in page.segments(maxworkers=1):
            if array is None:
                raise ValueError(f'Undecodable TIFF tile in {path.name}: {index}')
            # Tifffile segment shape is depth, rows, cols, samples; planar CONTIG verified.
            if int(page.planarconfig) != 1:
                raise ValueError('Scene audit currently requires contiguous samples')
            y, x = index[2], index[3]
            block = array[0, :min(array.shape[1], h - y), :min(array.shape[2], w - x)]
            values = block.reshape(-1, channels)
            for c in range(channels):
                v = values[:, c]
                finite = np.isfinite(v)
                missing = np.zeros(len(v), bool) if nodata is None else v == nodata
                good = finite & ~missing
                nonfinite_counts[c] += (~finite).sum()
                nodata_counts[c] += missing.sum()
                v = v[good].astype(np.float64)
                if len(v):
                    lo[c], hi[c] = min(lo[c], v.min()), max(hi[c], v.max())
                    sums[c] += v.sum()
                    squares[c] += np.dot(v, v)
                    counts[c] += len(v)
                    zeros[c] += (v == 0).sum()
            decoded += 1
        record['decoded_segments'] = decoded
        means = sums / np.maximum(counts, 1)
        record.update(valid_samples_per_channel=counts.tolist(), nodata_samples_per_channel=nodata_counts.tolist(),
                      nonfinite_samples_per_channel=nonfinite_counts.tolist(), zero_samples_per_channel=zeros.tolist(),
                      min=[float(lo[c]) if counts[c] else None for c in range(channels)],
                      max=[float(hi[c]) if counts[c] else None for c in range(channels)],
                      mean=[float(means[c]) if counts[c] else None for c in range(channels)],
                      std=[float(np.sqrt(max(0, squares[c] / counts[c] - means[c]**2))) if counts[c] else None for c in range(channels)])
        scale = record['geotiff'].get('ModelPixelScale') or (page.tags[33550].value if 33550 in page.tags else None)
        tie = record['geotiff'].get('ModelTiepoint') or (page.tags[33922].value if 33922 in page.tags else None)
        if scale and tie:
            record['gdal_geotransform'] = [tie[3] - tie[0] * scale[0], scale[0], 0.,
                                           tie[4] + tie[1] * scale[1], 0., -scale[1]]
        return record


def control_point_audit(root, scenes, config):
    source, target = read_control_points(root / config['control_points']['path'])
    optical, sar = scenes['optical'], scenes['sar']
    source_inside = valid_pixel_centers(source, optical['width'], optical['height'])
    target_inside = valid_pixel_centers(target, sar['width'], sar['height'])
    affine = fit_affine_least_squares(source, target)
    residual = np.linalg.norm(transform_points(source, affine) - target, axis=1)
    reverse = transform_points(transform_points(source, affine), invert_affine(affine))
    cv_seed = int(config['seed']) % (2**31 - 1)
    import cv2
    cv2.setRNGSeed(cv_seed)
    ransac, inliers = estimate_affine(source, target, np.ones(len(source)), threshold=config['geometry']['ransac_threshold_px'])
    summary = dict(rows=len(source), expected_rows=config['control_points']['expected_rows'],
                   columns=config['control_points']['columns'], coordinates_assumed='zero_based_xy_pixel_centers',
                   annotation_origin_verified=False, duplicate_rows=len(source) - len(np.unique(np.column_stack([source, target]), axis=0)),
                   duplicate_optical=len(source) - len(np.unique(source, axis=0)), duplicate_sar=len(target) - len(np.unique(target, axis=0)),
                   optical_out_of_bounds=np.where(~source_inside)[0].tolist(), sar_out_of_bounds=np.where(~target_inside)[0].tolist(),
                   optical_min_xy=source.min(0).tolist(), optical_max_xy=source.max(0).tolist(),
                   sar_min_xy=target.min(0).tolist(), sar_max_xy=target.max(0).tolist(),
                   diagnostic_fit_not_independent_accuracy=True, least_squares_affine_optical_to_sar=affine.tolist(),
                   fit_rmse_px=float(np.sqrt(np.mean(residual**2))), fit_median_px=float(np.median(residual)),
                   fit_max_px=float(residual.max()), numerical_roundtrip_max_px=float(np.linalg.norm(reverse - source, axis=1).max()),
                   diagnostic_ransac_inliers=int(inliers.sum()), diagnostic_ransac_matrix=None if ransac is None else ransac.tolist())
    rows = [dict(point_id=i + 1, x_optical=float(s[0]), y_optical=float(s[1]), x_sar=float(t[0]), y_sar=float(t[1]),
                 optical_in_bounds=bool(source_inside[i]), sar_in_bounds=bool(target_inside[i]), diagnostic_fit_error_px=float(residual[i]))
            for i, (s, t) in enumerate(zip(source, target))]
    return summary, rows


def audit_dataset(root, output, archive=None, config_path=None):
    config, root, output = load_config(config_path), resolve_data_root(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected, archive_report = _archive_index(archive, root.name)
    report = dict(schema_version=1, protocol=config['protocol'], root='<OSDataset2.0 root>',
                  archive=archive_report, errors=[], subsets={}, parent_lineage={}, scene={}, source_evidence={})
    files = sorted(p for p in root.rglob('*') if p.is_file())
    relative_paths = {p.relative_to(root).as_posix() for p in files}
    report['archive'].update(extracted_files=len(files), missing_files=sorted(set(expected) - relative_paths) if expected else [],
                             extra_files=sorted(relative_paths - set(expected)) if expected else [])
    image_records, arrays_512, quadrants, file_checks = {}, {}, defaultdict(list), []
    # Each PNG is read once. Only two decoded 512 arrays per pair need to be retained.
    for number, path in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() == '.png':
                data = path.read_bytes()
                sha, crc, size = hashlib.sha256(data).hexdigest(), zlib.crc32(data) & 0xffffffff, len(data)
            else:
                sha, crc, size = _file_digest(path)
            match = expected.get(rel)
            status = None if match is None else size == match.file_size and crc == match.CRC
            file_checks.append(dict(path=rel, size_bytes=size, sha256=sha, crc32=f'{crc:08x}', archive_size_crc_match=status))
            if status is False:
                report['errors'].append(dict(path=rel, reason='extracted_size_or_crc_differs_from_archive'))
            if path.suffix.lower() == '.png':
                image, array = inspect_png(data)
                image.update(path=rel, sha256=sha, size_bytes=size, archive_size_crc_match=status)
                image_records[rel] = image
                if image['dtype'] != 'uint8' or image['channels'] not in (1, 3):
                    report['errors'].append(dict(path=rel, reason='unexpected_patch_dtype_or_channels'))
                if '/OSdataset/512/' in rel:
                    # Sorted file order groups optical then SAR, so retain quadrant hashes only.
                    pieces = []
                    for y in (0, 256):
                        for x in (0, 256):
                            pieces.append((x, y, pixel_hash(array[y:y+256, x:x+256])))
                    arrays_512[rel] = pieces
        except Exception as exc:
            report['errors'].append(dict(path=rel, reason=f'{type(exc).__name__}: {exc}'))
        if number % 1000 == 0:
            print(f'audited_files={number}/{len(files)} errors={len(report["errors"])}', flush=True)
    with (output / 'file_inventory.jsonl').open('w') as f:
        for record in file_checks:
            f.write(json.dumps(record) + '\n')
    report['archive']['size_crc_matches'] = sum(x['archive_size_crc_match'] is True for x in file_checks)
    report['archive']['size_crc_mismatches'] = sum(x['archive_size_crc_match'] is False for x in file_checks)
    pairs, file_hashes, pixel_hashes = [], defaultdict(list), defaultdict(list)
    for folder in sorted((root / 'Patch-level Subset').rglob('*')):
        if not folder.is_dir():
            continue
        opt = {p.stem[3:]: p for p in folder.glob('opt*.png')}
        sar = {p.stem[3:]: p for p in folder.glob('sar*.png')}
        if not opt and not sar:
            continue
        subset = folder.relative_to(root).as_posix()
        builtin = folder.parent.name in ('256', '512') and folder.name in ('train', 'val', 'test')
        split = folder.name if builtin else 'unassigned'
        resolution = int(folder.parent.name) if builtin else 512
        region = None if builtin else folder.name
        summary = dict(pairs=0, unpaired_optical=sorted(opt.keys() - sar.keys()), unpaired_sar=sorted(sar.keys() - opt.keys()),
                       decode_failures=[], images={}, expected_pairs=config['dataset']['expected_pairs'][str(resolution)][split] if builtin else None)
        for id_ in sorted(opt.keys() & sar.keys(), key=int):
            op, sp = opt[id_].relative_to(root).as_posix(), sar[id_].relative_to(root).as_posix()
            uid = f'osdataset/{resolution}/{split}/{id_}' if builtin else f'osdataset2/region/{region}/{id_}'
            o, s = image_records.get(op), image_records.get(sp)
            if o is None or s is None:
                summary['decode_failures'].append(uid)
            pair = dict(schema='sglm.pair_manifest.v1', pair_id=uid, dataset='OSdataset' if builtin else 'OSDataset2.0-new-region',
                        resolution=resolution, split=split, region=region, optical=o, sar=s,
                        usable=o is not None and s is not None, geometry_gt='aligned_pair_assumption_no_explicit_patch_gt',
                        semantic_gt=None, parent_candidates=[], scene_id=None, scene_link_status='not_provided_by_package')
            if pair['usable']:
                if o['width'] != resolution or o['height'] != resolution or s['width'] != resolution or s['height'] != resolution:
                    report['errors'].append(dict(pair_id=uid, reason='resolution_mismatch'))
                    pair['usable'] = False
                for modality, image in [('optical', o), ('sar', s)]:
                    file_hashes[(resolution, modality, image['sha256'])].append((split, uid))
                    pixel_hashes[(resolution, modality, image['decoded_sha256'])].append((split, uid))
                if builtin and resolution == 512:
                    for (x, y, oh), (sx, sy, sh) in zip(arrays_512[op], arrays_512[sp]):
                        assert (x, y) == (sx, sy)
                        quadrants[(oh, sh)].append(dict(parent_pair_id=uid, parent_split=split, crop_xy=[x, y],
                                                        proof='exact_decoded_pixels_both_modalities'))
            pairs.append(pair)
            summary['pairs'] += 1
        for modality, paths in [('optical', opt), ('sar', sar)]:
            ims = [image_records[p.relative_to(root).as_posix()] for p in paths.values() if p.relative_to(root).as_posix() in image_records]
            summary['images'][modality] = dict(count=len(ims), modes=dict(Counter(x['mode'] for x in ims)),
                dtypes=dict(Counter(x['dtype'] for x in ims)), channels=dict(Counter(str(x['channels']) for x in ims)),
                observed_min=min((min(x['min']) for x in ims), default=None),
                observed_max=max((max(x['max']) for x in ims), default=None),
                constant_images=sum(x['constant'] for x in ims),
                metadata_images=sum(bool(x['png_metadata']) for x in ims))
        report['subsets'][subset] = summary
        if summary['unpaired_optical'] or summary['unpaired_sar'] or summary['decode_failures'] or (builtin and summary['pairs'] != summary['expected_pairs']):
            report['errors'].append(dict(subset=subset, reason='pairing_decode_or_expected_count_mismatch'))
    children = [x for x in pairs if x['dataset'] == 'OSdataset' and x['resolution'] == 256]
    conflicts, parent_children = [], defaultdict(list)
    for child in children:
        if child['usable']:
            child['parent_candidates'] = quadrants.get((child['optical']['decoded_sha256'], child['sar']['decoded_sha256']), [])
        for candidate in child['parent_candidates']:
            parent_children[candidate['parent_pair_id']].append(child['pair_id'])
            if candidate['parent_split'] != child['split']:
                conflicts.append(dict(child=child['pair_id'], **candidate))
    report['parent_lineage'] = dict(method='exact decoded 256x256 quadrants of each builtin 512 pair; both modalities must match',
        children=len(children), matched=sum(bool(c['parent_candidates']) for c in children),
        unique_match=sum(len(c['parent_candidates']) == 1 for c in children),
        ambiguous=sum(len(c['parent_candidates']) > 1 for c in children),
        unmatched=[c['pair_id'] for c in children if not c['parent_candidates']],
        cross_split_parent_links=conflicts, parents_with_children=len(parent_children),
        children_per_parent_histogram=dict(Counter(map(len, parent_children.values()))),
        scene_link='No patch-to-scene coordinates/IDs in package. Pixel parent proof does not establish geographic scene independence.')
    def cross_split_duplicates(mapping):
        return [dict(resolution=k[0], modality=k[1], hash=k[2], locations=v)
                for k, v in mapping.items() if len({s for s, _ in v} - {'unassigned'}) > 1]
    report['cross_split_file_duplicates'] = cross_split_duplicates(file_hashes)
    report['cross_split_decoded_duplicates'] = cross_split_duplicates(pixel_hashes)
    with (output / 'dataset_manifest.jsonl').open('w') as f:
        for pair in pairs:
            f.write(json.dumps(pair, separators=(',', ':')) + '\n')
    print('patch_checks_complete; decoding full scene tiles', flush=True)
    for name, filename in [('optical', 'Optical.tif'), ('sar', 'sar.tif')]:
        report['scene'][name] = scene_metadata_and_statistics(root / 'Scene-level Subset' / filename)
        print(f'scene_{name}_complete', flush=True)
    controls, control_rows = control_point_audit(root, report['scene'], config)
    report['control_points'] = controls
    with (output / 'control_points.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(control_rows[0]))
        writer.writeheader()
        writer.writerows(control_rows)
    with ZipFile(root / 'Patch-level Subset/OSdataset/ReadMe.docx') as z:
        xml = ET.fromstring(z.read('word/document.xml'))
        text = '\n'.join(''.join(p.itertext()) for p in xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))
    (output / 'package_readme_extracted.txt').write_text(text)
    report['source_evidence'] = dict(patch_readme='Patch-level Subset/OSdataset/ReadMe.docx',
        patch_sar='Gaofen-3 C-band Spotlight', patch_optical='Google Earth; original package builtin patches are grayscale',
        nominal_original_patch_resolution_m=1, scene_crs='EPSG:4326 from GeoTIFF metadata',
        scene_pixel_spacing_unit='degrees, not meters', scene_source_satellite='not identified in package',
        original_large_patch_parent_scenes='not distributed as an ID/coordinate mapping',
        reference_script='Scene-level Subset/code/Evaluation.py',
        metadata_candidate_files=[p.relative_to(root).as_posix() for p in files if p.suffix.lower() in ('.txt', '.csv', '.json', '.xml', '.mat', '.npy', '.npz', '.docx', '.py')])
    report['totals'] = dict(files=len(files), decoded_png=len(image_records), pairs=len(pairs),
        usable_pairs=sum(x['usable'] for x in pairs), new_region_pairs=sum(x['dataset'] != 'OSdataset' for x in pairs))
    if report['totals']['new_region_pairs'] != config['dataset']['expected_new_region_pairs']:
        report['errors'].append(dict(reason='new_region_pair_count_mismatch'))
    if archive is not None and (len(expected) != config['dataset']['expected_archive_files'] or report['archive']['missing_files'] or report['archive']['extra_files']):
        report['errors'].append(dict(reason='archive_inventory_count_or_paths_mismatch'))
    if controls['rows'] != controls['expected_rows'] or controls['optical_out_of_bounds'] or controls['sar_out_of_bounds'] or controls['duplicate_rows']:
        report['errors'].append(dict(reason='control_point_count_bounds_or_duplicates'))
    report['root_manifest_sha256'] = hashlib.sha256((output / 'dataset_manifest.jsonl').read_bytes()).hexdigest()
    (output / 'effective_config.json').write_text(json.dumps(config, indent=2))
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(totals=report['totals'], errors=report['errors'], parent_lineage={k:v for k,v in report['parent_lineage'].items() if k not in ('unmatched','cross_split_parent_links')})), flush=True)
    return report
