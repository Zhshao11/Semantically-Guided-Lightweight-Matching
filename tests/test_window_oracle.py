import numpy as np
import torch
from sglm.window_oracle import inside_convex_support,support_hits,sample_actual_support,mask_valid_at,replace_targets,frozen_fit_mask,coverage
from sglm.geometry import transform_points,warp_image
from sglm.evaluation import estimate_from_correspondences


def test_convex_hull_continuous_triangle_not_bbox_and_numerical_tolerance():
    tri=np.array([[0,0],[2,0],[0,2]])
    assert inside_convex_support([.3,.7],tri) and inside_convex_support([1,1],tri)
    assert not inside_convex_support([1.9,1.9],tri)
    assert inside_convex_support([0,-.5e-6],tri) and not inside_convex_support([0,-1e-4],tri)
    assert not inside_convex_support([2.001,0],tri)


def test_degenerate_support_point_line_and_empty():
    assert not inside_convex_support([0,0],np.empty((0,2)))
    assert inside_convex_support([1,2],[[1,2]]) and not inside_convex_support([1,2.01],[[1,2]])
    assert inside_convex_support([.5,.5],[[0,0],[1,1],[2,2]]) and not inside_convex_support([.5,.6],[[0,0],[2,2]])
    assert not inside_convex_support([3,3],[[0,0],[2,2]])


def test_real_sampler_feature_boundary_masked_tokens_and_continuous_output():
    f=torch.zeros(1,16,128,128);src=torch.tensor([[3.5,3.5],[131.5,131.5],[251.5,251.5]])
    mask=torch.ones(1,256,256,dtype=torch.bool);positions,valid,sv=sample_actual_support(f,f,src,src,mask,mask,3)
    pos=positions.numpy();v=valid.numpy();assert pos.shape==(3,49,2) and v.sum(1).tolist()==[25,49,25]
    assert np.array_equal(pos[0,v[0]].min(0),[1.5,1.5])
    assert np.array_equal(pos[2,v[2]].max(0),[253.5,253.5])
    # Both points are within +/-6 of the boundary center, but outside actual support.
    assert not inside_convex_support([0,3.5],pos[0,v[0]])
    assert not inside_convex_support([255,251.5],pos[2,v[2]])
    assert inside_convex_support([132.125,130.375],pos[1,v[1]])
    altered=mask.clone();altered[:,:,134:]=False
    q,allowed,_=sample_actual_support(f,f,src,src,mask,altered,3)
    assert not inside_convex_support([136,131.5],q[1,allowed[1]].numpy())


def test_invalid_hole_inside_hull_is_not_replaced_and_all_invalid_fallback_excluded():
    f=torch.zeros(1,16,128,128);s=torch.tensor([[131.5,131.5]]);om=torch.ones(1,256,256,dtype=torch.bool);sm=om.clone();sm[:,130:134,130:134]=False
    pos,valid,sv=sample_actual_support(f,f,s,s,om,sm,3);truth=s.numpy()
    assert inside_convex_support(truth[0],pos[0,valid[0]].numpy())
    hit=support_hits(truth,pos.numpy(),valid.numpy(),sv.numpy(),mask_valid_at(truth,sm[0]));assert not hit[0]
    _,invalid,_=sample_actual_support(f,f,s,s,om,torch.zeros_like(sm),3);assert not invalid.any()
    assert not support_hits(truth,pos.numpy(),invalid.numpy(),sv.numpy(),np.ones(1,bool))[0]


def test_actual_source_coordinate_maps_forward_translation_and_affine():
    # Source is intentionally not a coarse-grid center.
    source=np.array([[17.125,23.75],[40.375,70.125]])
    A=np.array([[1.02,.04,8],[-.03,.97,-4]])
    expected=source@A[:,:2].T+A[:,2]
    assert np.array_equal(transform_points(source,A),expected)
    assert not np.allclose(expected,transform_points(np.round((source-3.5)/8)*8+3.5,A))
    image=np.zeros((256,256),np.uint8);image[40,30]=255;T=np.array([[1.,0,8],[0,1,-4]])
    warped,_=warp_image(image,T);assert warped[36,38]==255 and warped[44,22]==0
    assert np.array_equal(transform_points([[30,40]],T),[[38,36]])


def test_frozen_filter_replacement_only_target_and_empty_intervention_geometry():
    src=np.array([[20,20],[80,20],[20,80],[80,80]],np.float32);dst=src+np.array([3,-2],np.float32);score=np.ones(4,np.float32)
    config=dict(minimum_match_score=0,ransac_threshold_px=3,ransac_max_iterations=5000,ransac_confidence=.999,ransac_min_inliers=3)
    fit=estimate_from_correspondences(src,dst,score,config,seed=20261125,include_inlier_indices=True)
    zero=replace_targets(dst,dst+1,np.zeros(4,bool));emptyfit=estimate_from_correspondences(src,zero,score,config,seed=20261125,include_inlier_indices=True)
    assert fit==emptyfit
    hit=np.array([True,False,True,False]);truth=dst.astype(float)+[.25,.75];oracle=replace_targets(dst,truth,hit)
    assert np.array_equal(oracle[~hit],dst[~hit]) and np.max(np.abs(oracle[hit]-truth[hit]))==0
    changed=dst.copy();changed[1]=np.nan;assert frozen_fit_mask(src,changed,score,0).tolist()==[True,False,True,True]
    assert np.array_equal(src,np.array([[20,20],[80,20],[20,80],[80,80]],np.float32))


def test_coverage_less_than_three_and_covariance_eigenvalue_ratio():
    a=coverage([[1,2],[2,3]]);assert a['less_than_three_points'] and a['minor_major_eigenvalue_ratio'] is None
    b=coverage([[1,1],[2,2],[3,3]]);assert not b['less_than_three_points'] and b['minor_major_eigenvalue_ratio']<1e-12
    c=coverage([[20,20],[240,20],[20,240],[240,240]]);assert c['grid_4x4_occupied']==4 and c['minor_major_eigenvalue_ratio']>.999
