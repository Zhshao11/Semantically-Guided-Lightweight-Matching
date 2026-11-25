import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sglm.geometry import (as_homogeneous, affine_from_parameters, invert_affine, compose_affines,
    transform_points, pixel_to_feature, feature_to_pixel, pixel_to_normalized, normalized_to_pixel,
    resize_affine, crop_to_parent_affine, convert_affine_frames, pixel_to_world, world_to_pixel,
    valid_pixel_centers, warp_image, read_control_points, fit_affine_least_squares, estimate_affine)


@pytest.mark.parametrize('size', [256, 512])
def test_full_affine_inverse_and_frame_conversion(size):
    rng = np.random.default_rng(1)
    points = rng.uniform(0, size - 1, (1000, 2))
    matrix = affine_from_parameters(size, size, 27, (.82, 1.13), (.19, -.08), (13., -7.))
    assert np.linalg.norm(transform_points(transform_points(points, matrix), invert_affine(matrix)) - points, axis=1).max() < 1e-9
    source_frame = compose_affines(resize_affine((size, size), (size // 2, size // 2)), invert_affine(crop_to_parent_affine([11, 23])))
    destination_frame = compose_affines(resize_affine((size, size), (size * 2, size * 2)), invert_affine(crop_to_parent_affine([5, 17])))
    converted = convert_affine_frames(matrix, source_frame, destination_frame)
    assert np.allclose(transform_points(transform_points(points, source_frame), converted),
                       transform_points(transform_points(points, matrix), destination_frame), atol=1e-9)


def test_composition_order_is_not_commutative():
    scale = [[2, 0, 0], [0, 3, 0]]
    translate = [[1, 0, 5], [0, 1, 7]]
    point = [[1, 1]]
    assert np.array_equal(transform_points(point, compose_affines(translate, scale)), [[7, 10]])
    assert np.array_equal(transform_points(point, compose_affines(scale, translate)), [[12, 24]])


@pytest.mark.parametrize('matrix', [np.zeros((2, 3)), [[1, 0, 0], [0, 1, float('nan')]],
    [[1, 0, 0], [0, 1, 0], [0, .1, 1]], np.eye(2)])
def test_invalid_affine_rejected(matrix):
    with pytest.raises(ValueError):
        invert_affine(matrix)


@pytest.mark.parametrize('stride', [2, 8])
def test_feature_and_normalized_pixel_centers(stride):
    points = np.array([[0, 0], [255, 511], [3.5, 3.5], [19.25, 40.75]])
    assert np.allclose(feature_to_pixel(pixel_to_feature(points, stride), stride), points)
    normalized = pixel_to_normalized(points, 256, 512)
    assert np.allclose(normalized_to_pixel(normalized, 256, 512), points)
    assert np.allclose(normalized[0], [-1 + 1/256, -1 + 1/512])
    if stride == 8:
        assert np.array_equal(pixel_to_feature([[3.5, 3.5]], stride), [[0, 0]])


def test_crop_resize_and_geotransform_half_pixel():
    point = [[0, 0]]
    assert np.array_equal(transform_points(point, resize_affine((512, 512), (256, 256))), [[-.25, -.25]])
    assert np.array_equal(transform_points([[1, 2]], crop_to_parent_affine([256, 0])), [[257, 2]])
    gt = [100, 2, .2, 200, -.1, -3]
    assert np.allclose(pixel_to_world(point, gt), [[101.1, 198.45]])
    points = np.array([[0, 0], [1.25, 2.5], [31000.1, 24000.2]])
    assert np.allclose(world_to_pixel(pixel_to_world(points, gt), gt), points, atol=1e-9)


def test_image_axis_rotation_and_reflection():
    # Positive angle points toward +y, not the opposite OpenCV getRotationMatrix2D convention.
    matrix = affine_from_parameters(5, 5, 90)
    assert np.allclose(transform_points([[3, 2]], matrix), [[2, 3]])
    reflection = affine_from_parameters(5, 5, scale=(-1, 1))
    assert np.allclose(transform_points([[0, 1]], reflection), [[4, 1]])


def test_integer_warp_and_support_mask():
    image = np.arange(32*48, dtype=np.float32).reshape(32, 48)
    matrix = [[1, 0, 5], [0, 1, -3]]
    warped, mask = warp_image(image, matrix)
    assert np.array_equal(warped[:29, 5:], image[3:, :43])
    assert mask.sum() == 29 * 43
    assert not mask[:, :5].any() and not mask[29:].any()
    identity, valid = warp_image(image, np.eye(2, 3))
    assert np.array_equal(identity, image) and valid.all()


def test_opencv_forward_equals_torch_inverse_sampling():
    h, w = 64, 80
    y, x = np.mgrid[:h, :w]
    image = ((x + 2*y)/(w + 2*h)).astype(np.float32)
    matrix = affine_from_parameters(w, h, 17, (.9, 1.1), (.13, 0), (2.4, -1.7))
    cv_image, valid = warp_image(image, matrix)
    points = np.stack([x, y], -1)
    inverse = transform_points(points, invert_affine(matrix))
    grid = pixel_to_normalized(inverse, w, h).astype(np.float32)
    sampled = F.grid_sample(torch.from_numpy(image)[None, None], torch.from_numpy(grid)[None], align_corners=False)[0, 0].numpy()
    assert np.max(np.abs(cv_image[valid] - sampled[valid])) < .001


def test_subpixel_forward_centroid_and_mask():
    image = np.zeros((32, 32), dtype=np.float32)
    image[12, 10] = 1
    warped, valid = warp_image(image, [[1, 0, .5], [0, 1, -.25]])
    y, x = np.mgrid[:32, :32]
    centroid = np.array([(warped*x).sum(), (warped*y).sum()])/warped.sum()
    assert np.allclose(centroid, [10.5, 11.75], atol=1e-6)
    assert not valid[:, 0].any() and not valid[-1].any()


def test_least_squares_and_robust_affine():
    rng = np.random.default_rng(3)
    points = rng.uniform(-100, 400, (100, 2))
    matrix = affine_from_parameters(512, 256, 13, (.9, 1.2), (.11, -.03), (23, -17))
    target = transform_points(points, matrix)
    assert np.allclose(fit_affine_least_squares(points, target), matrix, atol=1e-10)
    noisy = target + rng.normal(0, .05, target.shape)
    noisy[:20] += 100
    cv2.setRNGSeed(1)
    robust, inliers = estimate_affine(points, noisy, np.ones(100), threshold=.5)
    assert robust is not None and inliers.sum() == 80
    assert np.max(np.linalg.norm(transform_points(points, robust) - target, axis=1)) < .1
    with pytest.raises(ValueError, match='Degenerate'):
        fit_affine_least_squares(np.zeros((3, 2)), np.zeros((3, 2)))


def test_control_point_parser_exact_columns_and_validation(tmp_path):
    path = tmp_path/'GT.txt'
    path.write_text('# source xy target xy\n1 2 3 4\n5,6,7,8\n')
    source, target = read_control_points(path)
    assert np.array_equal(source, [[1, 2], [5, 6]])
    assert np.array_equal(target, [[3, 4], [7, 8]])
    for invalid in ('1 2 3', '1 2 3 4 5', '1 2 nan 4', ''):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            read_control_points(path)


def test_boundaries_empty_and_bad_parameters():
    assert valid_pixel_centers([[0, 0], [255, 511], [256, 0], [-.01, 0]], 256, 512).tolist() == [True, True, False, False]
    assert transform_points(np.empty((0, 2)), np.eye(2, 3)).shape == (0, 2)
    with pytest.raises(ValueError):
        pixel_to_feature([[1, 2]], 0)
    with pytest.raises(ValueError):
        pixel_to_normalized([[np.nan, 2]], 32, 32)
    with pytest.raises(ValueError):
        resize_affine((0, 32), (32, 32))
    with pytest.raises(ValueError):
        estimate_affine([[1, 2]], [[1, 2]], [1, 2])
