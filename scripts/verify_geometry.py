"""Numeric geometry closure on all six real OSdataset splits (no model training)."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sglm.config import load_config, resolve_data_root
from sglm.data import OSPairs
from sglm.geometry import (affine_from_parameters, transform_points, invert_affine,
    pixel_to_feature, feature_to_pixel, pixel_to_normalized, normalized_to_pixel,
    resize_affine, crop_to_parent_affine, compose_affines, convert_affine_frames, warp_image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--config')
    ap.add_argument('--output', required=True)
    ap.add_argument('--samples-per-split', type=int, default=3)
    args = ap.parse_args()
    if args.samples_per_split < 1:
        raise ValueError('samples-per-split must be positive')
    config = load_config(args.config)
    root = resolve_data_root(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config['seed'])
    records, sample_for_image = [], None
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            dataset = OSPairs(root, split=split, resolution=size, manifest=args.manifest, affine=False, config_path=args.config)
            expected = config['dataset']['expected_pairs'][str(size)][split]
            if len(dataset) != expected:
                raise AssertionError(f'Manifest pair count {len(dataset)} != {expected}: {size}/{split}')
            indices = np.linspace(0, len(dataset) - 1, min(args.samples_per_split, len(dataset)), dtype=int)
            for index in indices:
                item = dataset[int(index)]
                sar = (item['sar'][0].numpy() * 255).round().astype(np.uint8)
                optical = (item['optical'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                identity, mask = warp_image(sar, np.eye(2, 3))
                assert np.array_equal(identity, sar) and mask.all()
                matrix = affine_from_parameters(size, size, 21.3, (.83, 1.14), (.16, -.07), (9.25, -13.5))
                points = rng.uniform(0, size - 1, (1000, 2))
                mapped = transform_points(points, matrix)
                roundtrip = float(np.abs(transform_points(mapped, invert_affine(matrix)) - points).max())
                normalized_error = float(np.abs(normalized_to_pixel(pixel_to_normalized(points, size, size), size, size) - points).max())
                feature_error = max(float(np.abs(feature_to_pixel(pixel_to_feature(points, s), s) - points).max()) for s in config['geometry']['feature_strides'])
                source_frame = compose_affines(resize_affine((size, size), (size//2, size//2)), invert_affine(crop_to_parent_affine([17, 23])))
                target_frame = compose_affines(resize_affine((size, size), (size*2, size*2)), invert_affine(crop_to_parent_affine([11, 19])))
                changed = convert_affine_frames(matrix, source_frame, target_frame)
                frame_error = float(np.abs(transform_points(transform_points(points, source_frame), changed) -
                                             transform_points(mapped, target_frame)).max())
                # An analytic coordinate ramp tests image inverse resampling against point forward transforms.
                y, x = np.mgrid[:size, :size]
                ramp = ((x + 2*y)/(3 * size)).astype(np.float32)
                warped_ramp, support = warp_image(ramp, matrix)
                src_positions = transform_points(np.stack([x, y], -1), invert_affine(matrix))
                analytic_ramp = (src_positions[..., 0] + 2*src_positions[..., 1])/(3*size)
                sampling_error = float(np.abs(warped_ramp[support] - analytic_ramp[support]).max())
                passed = max(roundtrip, normalized_error, feature_error, frame_error) < config['geometry']['roundtrip_tolerance_px'] and sampling_error < 1e-3
                assert passed, item['pair_id']
                records.append(dict(pair_id=item['pair_id'], resolution=size, split=split, passed=passed,
                    input_optical_channels=3, input_sar_channels=1, identity_pixel_difference_max=0,
                    point_roundtrip_max_px=roundtrip, normalized_roundtrip_max_px=normalized_error,
                    feature_roundtrip_max_px=feature_error, crop_resize_frame_closure_max_px=frame_error,
                    image_analytic_ramp_max_error=sampling_error, support_fraction=float(support.mean()),
                    source_to_destination_affine=matrix.tolist()))
                if sample_for_image is None:
                    sample_for_image = (optical, sar, matrix)
    report = dict(protocol=config['protocol'], kind='data_and_geometry_only_no_model_accuracy', seed=config['seed'],
        actual_pairs_checked=len(records), splits_checked=6, points_per_pair=1000,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), config=config,
        passed=all(x['passed'] for x in records),
        max_point_roundtrip_px=max(x['point_roundtrip_max_px'] for x in records),
        max_frame_closure_px=max(x['crop_resize_frame_closure_max_px'] for x in records),
        cases=records)
    (output/'geometry_closure.json').write_text(json.dumps(report, indent=2))
    optical, sar, matrix = sample_for_image
    warped, support = warp_image(sar, matrix)
    tiles = [Image.fromarray(optical), Image.fromarray(sar).convert('RGB'),
             Image.fromarray(warped).convert('RGB'), Image.fromarray((support*255).astype(np.uint8)).convert('RGB')]
    size = sar.shape[0]
    canvas = Image.new('RGB', (size*2, (size+26)*2), 'white')
    draw = ImageDraw.Draw(canvas)
    labels = ['Optical input (not a prediction)', 'SAR original', 'SAR with known full affine', 'Valid resampling support']
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i%2)*size, (i//2)*(size+26)
        canvas.paste(tile, (x, y+26))
        draw.text((x+5,y+5), label, fill='black')
    canvas.save(output/'geometry_warp_example.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','cases')}, indent=2))


if __name__ == '__main__':
    main()
