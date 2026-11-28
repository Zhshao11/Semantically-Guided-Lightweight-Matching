import numpy as np
import pytest
from sglm.fhreg_coordinate_audit import known_cases,roundtrip_case,preprocessing_matrix,transform,native_to_original
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform


@pytest.mark.parametrize('case',known_cases(),ids=lambda c:c['name'])
def test_no_network_fhreg_coordinate_roundtrips(case):
    result,_=roundtrip_case(case)
    assert all(value['max_px']<1e-4 for value in result['errors'].values())
    assert result['xy_swap_negative_mean_px']>10


def test_crop_padding_order_origin_and_zero_origin_pixel_centers():
    P=preprocessing_matrix((256,256),(512,512),(23,11),(17,9))
    p=transform([[0,0],[255,255]],P)
    assert np.array_equal(p,[[-5.5,-1.5],[504.5,508.5]])
    assert np.array_equal(native_to_original(p,(256,256),(512,512),(23,11),(17,9)),[[0,0],[255,255]])


def test_no_network_coordinates_fit_forward_affine_and_reject_wrong_direction():
    case=known_cases()[1];_,(src,dst)=roundtrip_case(case)
    cfg=dict(minimum_match_score=0,ransac_threshold_px=3,ransac_max_iterations=5000,ransac_confidence=.999,ransac_min_inliers=6)
    fit=estimate_from_correspondences(src,dst,np.ones(len(src)),cfg,seed=20261125)
    probes,truth=fixed_synthetic_probes(case['A']);ev=evaluate_transform(fit['matrix'],probes,truth)
    assert ev['rmse_px']<1e-4 and ev['success']['5']
    wrong=estimate_from_correspondences(dst,src,np.ones(len(src)),cfg,seed=20261125)
    assert evaluate_transform(wrong['matrix'],probes,truth)['rmse_px']>10
