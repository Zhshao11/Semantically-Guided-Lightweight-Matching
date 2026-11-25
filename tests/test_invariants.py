import numpy as np
import pytest
import torch
from sglm.model import Matcher, JointScore, Budget, Fine, select_queries, grid_masks, sample_windows
from sglm.geometry import estimate_affine
from sglm.losses import matching_loss


def test_budget_coverage_capacity_and_invalid_mask():
    torch.manual_seed(3)
    value = torch.rand(32, 32)
    valid = torch.ones(32, 32, dtype=torch.bool)
    valid[:8, :8] = False
    for k in (128, 256, 384):
        q = select_queries(value, valid, k)
        assert len(q) == k and len(q.unique()) == k
        assert valid.flatten()[q].all()
        counts = [int(m[q].sum()) for m in grid_masks(32, 32, q.device)]
        assert counts[0] == 0 and all(2 <= c <= 32 for c in counts[1:])
    q = select_queries(value, torch.zeros_like(valid), 256)
    assert q.numel() == 0


def test_semantic_stop_gradient():
    head = JointScore().eval()
    fo, fs = torch.randn(2, 64, 8, 8), torch.randn(2, 64, 8, 8)
    out = head(fo, fs)
    out['v'].sum().backward()
    assert all(p.grad is None for p in head.semantic.parameters())
    assert any(p.grad is not None for p in head.residual.parameters())
    head.zero_grad()
    (-out['p'].clamp_min(1e-8).log().mean()).backward()
    assert any(p.grad is not None for p in head.semantic.parameters())


def test_pixel_center_sampling_has_no_half_pixel_shift():
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    image = torch.stack([x, y]).float()[None]
    tokens, delta, mask = sample_windows(image, torch.tensor([[7.5, 11.5]]), 1)
    assert torch.allclose(tokens[0, 4], torch.tensor([3.5, 5.5]))
    assert mask.all()
    assert torch.equal(delta[4], torch.zeros(2))


def test_full_affine_ransac_recovers_shear_and_rejects_degenerate():
    rng = np.random.default_rng(4)
    source = rng.uniform(0, 256, (50, 2))
    gt = np.array([[1.1, .17, 12], [-.08, .87, -4]])
    target = source @ gt[:, :2].T + gt[:, 2]
    target[:10] += 100
    matrix, inliers = estimate_affine(source, target, np.ones(50))
    assert np.allclose(matrix, gt, atol=1e-4) and inliers.sum() == 40
    assert estimate_affine(np.ones((4, 2)), np.ones((4, 2)), np.ones(4))[0] is None


def test_forward_backward_empty_target_and_statistics():
    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = Matcher(use_semantics=False, coarse_threshold=0.)
    image = torch.rand(1, 3, 64, 64)
    sar = torch.rand(1, 1, 64, 64)
    output = model(image, sar)
    assert output['statistics'].shape == (1, 178)
    m = output['matches'][0]
    assert m['logits'].shape == (64, 65)  # Tiny image capacity, not a padded 256-row correlation.
    assert len(m['coarse_target'].unique(dim=0)) == len(m['coarse_target'])
    batch = dict(optical=image, sar=sar, transform=torch.eye(2, 3)[None])
    loss, _ = matching_loss(model, output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = model.eval()(image, sar, sar_mask=torch.zeros(1, 64, 64, dtype=torch.bool))
    assert len(empty['matches'][0]['source']) == 0
    assert torch.isfinite(empty['matches'][0]['logits'][:, -1]).all()
    model.fixed_budget = None
    with pytest.raises(RuntimeError, match='normalization'):
        model(image, sar)


def test_affine_gt_direction_matches_opencv_warp():
    import cv2
    from sglm.losses import warp, targets_for_queries
    image = np.zeros((64, 64), np.uint8)
    image[20, 10] = 255
    matrix = np.array([[1., 0., 8.], [0., 1., -4.]], np.float32)
    transformed = cv2.warpAffine(image, matrix, (64, 64))
    assert transformed[16, 18] == 255
    predicted = warp(torch.tensor([[10., 20.]]), torch.from_numpy(matrix))
    assert torch.equal(predicted, torch.tensor([[18., 16.]]))
    query = torch.tensor([9])  # Original cell center (11.5, 11.5).
    labels, target, valid = targets_for_queries(query, 8, 8, torch.from_numpy(matrix), torch.ones(8, 8, dtype=torch.bool))
    assert labels.item() == 10 and valid.item()
    assert torch.equal(target, torch.tensor([[19.5, 7.5]]))


def test_fine_window_masks_synthetic_padding():
    feature = torch.ones(1, 2, 16, 16)
    valid = torch.ones(1, 32, 32, dtype=torch.bool)
    valid[:, :, 12:] = False
    _, _, mask = sample_windows(feature, torch.tensor([[9.5, 9.5]]), 3, valid)
    assert mask.any() and not mask.all()
    # Offset +2 on the fine grid is input-pixel x=13.5, in synthetic padding.
    assert not mask[0, 3 * 7 + 5]
