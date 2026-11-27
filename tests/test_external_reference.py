import numpy as np
from sglm.external_reference import restore_official_coordinates,valid_correspondences,stable_pair_seed


def test_official_xy_restore_non_square_not_hw_swap():
    points=np.array([[20,10],[200,100]],np.float32)
    restored=restore_official_coordinates(points,(40,100),(200,400))
    assert np.allclose(restored,[[5,2],[50,20]])
    assert np.array_equal(points,np.array([[20,10],[200,100]],np.float32))


def test_common_filter_rejects_nonfinite_out_of_bounds_and_mask_without_gt():
    source=np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[1,1]],np.float32)
    target=np.array([[1,1],[2,2],[3,3],[8,4],[np.nan,5],[1,1]],np.float32)
    mask=np.ones((8,8),bool);mask[3,3]=False;score=np.array([.9,.8,.7,.6,.5,np.nan])
    s,t,c,v=valid_correspondences(source,target,score,np.ones_like(mask),mask)
    assert v.tolist()==[True,True,False,False,False,False]
    assert np.array_equal(s,source[:2]) and c.tolist()==np.array([.9,.8],np.float32).tolist()


def test_empty_correspondences_and_fixed_seed():
    mask=np.ones((16,16),bool)
    a,b,c,v=valid_correspondences(np.empty((0,2)),np.empty((0,2)),np.empty(0),mask,mask)
    assert a.shape==b.shape==(0,2) and c.shape==v.shape==(0,)
    assert stable_pair_seed('train/1',66)==stable_pair_seed('train/1',66)
    assert stable_pair_seed('train/1',66)!=stable_pair_seed('train/2',66)
