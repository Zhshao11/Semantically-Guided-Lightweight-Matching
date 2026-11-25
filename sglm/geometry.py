"""Affine and coordinate primitives.

All public point arrays end in (x, y); integer pixel centers start at (0, 0).
Matrices map source to destination. compose_affines(A, B) means A after B.
GeoTIFF geotransforms act on pixel corners, so pixel-center conversion adds 0.5.
"""
from pathlib import Path
import cv2
import numpy as np


def _points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError('Expected finite points with shape (..., 2) in xy order')
    return value


def as_homogeneous(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0., 0., 1.]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite 2x3 or 3x3 affine matrix')
    if not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError('Expected affine bottom row [0, 0, 1]')
    if np.linalg.matrix_rank(matrix[:2, :2]) < 2:
        raise ValueError('Singular affine matrix')
    return matrix.copy()


def invert_affine(matrix):
    return np.linalg.inv(as_homogeneous(matrix))[:2]


def compose_affines(*matrices):
    result = np.eye(3)
    for matrix in matrices:
        result = result @ as_homogeneous(matrix)
    return result[:2]


def transform_points(points, matrix):
    points = _points(points)
    matrix = as_homogeneous(matrix)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def affine_from_parameters(width, height, angle_deg=0., scale=(1., 1.), shear=(0., 0.), translation=(0., 0.)):
    """Full affine about image center: rotation @ shear @ axis-scale, then translation.

    Positive angle rotates the +x axis toward +y, visually clockwise in image axes.
    Shear values are linear coefficients, not angles. Reflection is supported.
    """
    values = np.asarray([width, height, angle_deg, *scale, *shear, *translation], dtype=float)
    if not np.isfinite(values).all() or width <= 0 or height <= 0:
        raise ValueError('Invalid affine parameters')
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    linear = rotation @ np.array([[1., shear[0]], [shear[1], 1.]]) @ np.diag(scale)
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    matrix = np.column_stack([linear, center + translation - linear @ center])
    return as_homogeneous(matrix)[:2]


def pixel_to_feature(points, stride):
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError('stride must be positive')
    return (_points(points) + .5) / stride - .5


def feature_to_pixel(points, stride):
    if not np.isfinite(stride) or stride <= 0:
        raise ValueError('stride must be positive')
    return (_points(points) + .5) * stride - .5


def pixel_to_normalized(points, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size')
    return 2 * (_points(points) + .5) / [width, height] - 1


def normalized_to_pixel(points, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size')
    return (_points(points) + 1) * np.array([width, height]) / 2 - .5


def resize_affine(source_wh, destination_wh):
    """Half-pixel resize map; does not silently multiply coordinates by scale."""
    source, destination = np.asarray(source_wh, float), np.asarray(destination_wh, float)
    if source.shape != (2,) or destination.shape != (2,) or not np.isfinite([source, destination]).all() or (source <= 0).any() or (destination <= 0).any():
        raise ValueError('Expected positive (width, height) sizes')
    ratio = destination / source
    return np.column_stack([np.diag(ratio), (ratio - 1) / 2])


def crop_to_parent_affine(origin_xy):
    origin = _points(origin_xy)
    if origin.shape != (2,):
        raise ValueError('Crop origin must be one xy point')
    return np.column_stack([np.eye(2), origin])


def convert_affine_frames(matrix, source_old_to_new, destination_old_to_new):
    """A_new = T_destination @ A_old @ inv(T_source)."""
    return compose_affines(destination_old_to_new, matrix, invert_affine(source_old_to_new))


def _geotransform_matrix(geotransform):
    gt = np.asarray(geotransform, dtype=float)
    if gt.shape != (6,) or not np.isfinite(gt).all():
        raise ValueError('Expected six finite GDAL geotransform values')
    return as_homogeneous([[gt[1], gt[2], gt[0]], [gt[4], gt[5], gt[3]]])


def pixel_to_world(points, geotransform):
    return transform_points(_points(points) + .5, _geotransform_matrix(geotransform))


def world_to_pixel(points, geotransform):
    return transform_points(points, invert_affine(_geotransform_matrix(geotransform))) - .5


def valid_pixel_centers(points, width, height):
    points = _points(points)
    return ((points >= 0) & (points <= [width - 1, height - 1])).all(axis=-1)


def warp_image(image, matrix, output_wh=None, interpolation=cv2.INTER_LINEAR, mask_min_weight=.999):
    """Forward affine image warp and strict mask for interpolation-supported pixels."""
    image = np.asarray(image)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError('Expected nonempty HxW or HxWxC image')
    if not 0 <= mask_min_weight < 1:
        raise ValueError('mask_min_weight must lie in [0,1)')
    matrix = as_homogeneous(matrix)[:2]
    output_wh = tuple(map(int, output_wh or (image.shape[1], image.shape[0])))
    if len(output_wh) != 2 or min(output_wh) <= 0:
        raise ValueError('Invalid output (width, height)')
    warped = cv2.warpAffine(image, matrix, output_wh, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix, output_wh,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, support > mask_min_weight


def read_control_points(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            values = [float(x) for x in line.replace(',', ' ').split()]
        except ValueError as exc:
            raise ValueError(f'Non-numeric control point at line {line_number}') from exc
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError(f'Expected four finite control-point values at line {line_number}')
        rows.append(values)
    if not rows:
        raise ValueError('No control points')
    points = np.asarray(rows, dtype=np.float64)
    return points[:, :2], points[:, 2:]


def fit_affine_least_squares(source, target):
    source, target = _points(source), _points(target)
    if source.ndim != 2 or source.shape != target.shape or len(source) < 3:
        raise ValueError('Expected at least three paired Nx2 points')
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError('Degenerate source control points')
    matrix = np.linalg.lstsq(design, target, rcond=None)[0].T
    return as_homogeneous(matrix)[:2]


def estimate_affine(source, target, reliability, threshold=3., min_reliability=.5, max_iterations=5000, confidence=.999):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    reliability = np.asarray(reliability, dtype=float)
    if source.ndim != 2 or source.shape[-1] != 2 or source.shape != target.shape or reliability.shape != (len(source),):
        raise ValueError('Expected source/target Nx2 and reliability N')
    if threshold <= 0 or max_iterations < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid RANSAC parameters')
    valid = np.isfinite(source).all(1) & np.isfinite(target).all(1) & np.isfinite(reliability) & (reliability >= min_reliability)
    selected = np.zeros(len(source), dtype=bool)
    if valid.sum() < 3 or np.linalg.matrix_rank(source[valid] - source[valid].mean(0)) < 2:
        return None, selected
    matrix, inliers = cv2.estimateAffine2D(source[valid].astype(np.float32), target[valid].astype(np.float32),
                                         method=cv2.RANSAC, ransacReprojThreshold=threshold,
                                         maxIters=max_iterations, confidence=confidence, refineIters=10)
    try:
        if matrix is None or inliers is None:
            return None, selected
        as_homogeneous(matrix)
    except ValueError:
        return None, selected
    selected[np.where(valid)[0]] = inliers.ravel().astype(bool)
    return matrix, selected
