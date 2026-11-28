import numpy as np
from sglm.sar_reference_geometry import restore_resized_coordinates,dense_correspondences,filter_correspondences,sample_dense_targets,evaluate_dense_probes
from sglm.geometry import transform_points
from sglm.evaluation import fixed_synthetic_probes


def affine_flow(A256):
    yy,xx=np.meshgrid(np.arange(512),np.arange(512),indexing='ij');native=np.stack([xx,yy],-1).astype(np.float32)
    source=restore_resized_coordinates(native,(256,256),(512,512));target=transform_points(source.reshape(-1,2),A256).reshape(512,512,2)
    return ((target-source)*2).transpose(2,0,1).astype(np.float32)


def test_halfpixel_coordinates_roundtrip_and_displacement_units():
    source=np.array([[0,0],[255,255],[7.5,31.5]],np.float32);native=(source+.5)*2-.5
    assert np.array_equal(restore_resized_coordinates(native,(256,256),(512,512)),source)
    flow=np.zeros((2,512,512),np.float32);flow[0]=16;flow[1]=-8
    a,b,_=dense_correspondences(flow);assert np.allclose(b-a,[8,-4])
    assert np.array_equal(a[0],[-.25,-.25])


def test_dense_affine_direction_and_independent_probe_oracle():
    A=np.array([[1.02,.03,4],[-.01,.97,-3]])
    flow=affine_flow(A);source,target,_=dense_correspondences(flow)
    assert np.allclose(target,transform_points(source,A),atol=3e-5,rtol=0)
    probes,truth=fixed_synthetic_probes(A)
    assert np.allclose(sample_dense_targets(flow,probes),truth,atol=3e-5,rtol=0)
    result=evaluate_dense_probes(flow,probes,truth);assert result['rmse_px']<3e-5 and result['success']['3']


def test_wrong_dense_translation_increases_error_and_nan_is_failure():
    source,truth=fixed_synthetic_probes(np.array([[1,0,0],[0,1,0]]));flow=np.zeros((2,512,512),np.float32);flow[0]=20
    result=evaluate_dense_probes(flow,source,truth);assert abs(result['rmse_px']-10)<1e-6 and not result['success']['5']
    flow[:]=np.nan;result=evaluate_dense_probes(flow,source,truth);assert not result['solved'] and not result['success']['5']


def test_dense_mask_filter_chunks_keep_every_valid_point_and_align_fields():
    from sglm.external_reference import valid_correspondences
    flow=np.zeros((2,512,512),np.float32);s,t,c=dense_correspondences(flow)
    mask=np.ones((256,256),bool);mask[60:90,100:120]=False
    a=filter_correspondences(s,t,c,mask,mask);b=filter_correspondences(s,t,c,mask,mask,chunk_size=1000)
    assert all(np.array_equal(x,y) for x,y in zip(a,b)) and len(a[0])>200000
    old=valid_correspondences(s[:20000],t[:20000],c[:20000],mask,mask)
    new=filter_correspondences(s[:20000],t[:20000],c[:20000],mask,mask)
    assert all(np.array_equal(x,y) for x,y in zip(old,new))
    t[1000]=np.nan;a=filter_correspondences(s,t,c,mask,mask);assert not a[3][1000] and np.isfinite(a[1]).all()


def test_empty_external_correspondences_remain_explicitly_empty():
    mask=np.ones((256,256),bool);result=filter_correspondences(np.empty((0,2)),np.empty((0,2)),np.empty(0),mask,mask)
    assert result[0].shape==(0,2) and result[3].shape==(0,)
