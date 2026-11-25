import copy
import json
import numpy as np
import pytest
from sglm.metadata_finalize import (joint_content_id,derive_manifests,parent_mapping,
                                     spatial_coverage,group_control_metadata)


def record(size,id_,oh,sh,candidates=None):
    return dict(pair_id=f'osdataset/{size}/train/{id_}',dataset='OSdataset',resolution=size,
        split='train',usable=True,optical={'decoded_sha256':oh},sar={'decoded_sha256':sh},
        parent_candidates=candidates or [],scene_id=None)


def fixture_pairs():
    parents=[record(512,10,'po','ps'),record(512,2,'po','ps')]
    children=[]
    for i,crop in enumerate(((0,0),(256,0),(0,256),(256,256))):
        candidates=[dict(parent_pair_id=p['pair_id'],parent_split='train',crop_xy=list(crop),
                         proof='exact_decoded_pixels_both_modalities') for p in parents]
        children.extend([record(256,i+10,'co'+str(i),'cs'+str(i),candidates),
                         record(256,i+1,'co'+str(i),'cs'+str(i),candidates)])
    return parents+children


def test_joint_dedup_and_numeric_representatives_and_original_preserved():
    pairs=fixture_pairs()
    original=copy.deepcopy(pairs)
    result=derive_manifests(pairs)
    assert pairs==original
    assert result['derived'][512][0]['pair_id']=='osdataset/512/train/2'
    assert [p['pair_id'] for p in result['derived'][256]]==[f'osdataset/256/train/{i}' for i in range(1,5)]
    assert result['summary']['256']['excluded']==4 and result['summary']['512']['excluded']==1
    assert len(result['decisions'])==10
    assert all(p['parent_mapping']['status']=='verified' for p in result['derived'][256])
    assert result['original_parent_status']=={'unresolved':2,'candidate':8}
    assert result['lineage']['complete_quadrant_coverage']
    # Input order cannot change the representative or derived ordering.
    assert result['derived']==derive_manifests(list(reversed(pairs)))['derived']


def test_same_one_modality_is_not_a_duplicate_pair():
    first=record(512,1,'same_optical','sar_1')
    second=record(512,2,'same_optical','sar_2')
    third=record(512,3,'different_optical','sar_1')
    assert len({joint_content_id(p) for p in (first,second,third)})==3


def test_parent_status_and_scene_status_are_separate():
    pairs=fixture_pairs()
    by_id={p['pair_id']:p for p in pairs}
    child=copy.deepcopy(pairs[2])
    child['parent_candidates']=child['parent_candidates'][:1]
    assert parent_mapping(child,by_id)['status']=='verified'
    child['parent_candidates'][0]['proof']='filename_only'
    assert parent_mapping(child,by_id)['status']=='candidate'
    child['parent_candidates']=[]
    assert parent_mapping(child,by_id)['status']=='unresolved'
    result=derive_manifests(pairs)
    assert all(p['scene_source_mapping']['status']=='unresolved' for p in result['annotated'])


def test_lineage_slot_conflicts_and_count_mismatch_are_rejected():
    pairs=fixture_pairs()
    # Remove both records for one quadrant; the derived relation must fail.
    bad=[p for p in pairs if p['pair_id'] not in ('osdataset/256/train/4','osdataset/256/train/13')]
    with pytest.raises(ValueError,match='four unique quadrants'):
        derive_manifests(bad)
    with pytest.raises(ValueError,match='Freeze count mismatch'):
        derive_manifests(pairs,{'256':{},'512':{}})


def test_spatial_coverage_and_near_collinearity():
    distributed=np.array([[0,0],[99,0],[0,99],[99,99],[50,50]])
    report=spatial_coverage(distributed,100,100)
    assert report['convex_hull_area_fraction']==pytest.approx(1.)
    assert not report['near_collinear'] and report['grid_4x4_occupied']==5
    diagonal=np.array([[0,0],[10,20],[20,40],[30,60]])
    assert spatial_coverage(diagonal,100,100)['near_collinear']


def test_controls_are_one_image_pair_not_one_pair_per_point(tmp_path):
    path=tmp_path/'GT.txt'
    path.write_text('1 1 2 2\n9 1 10 2\n1 9 2 10\n9 9 10 10\n')
    audit={'scene':{'optical':dict(path='Optical.tif',width=20,height=20),
                    'sar':dict(path='sar.tif',width=30,height=30)}}
    groups=group_control_metadata(path,audit)
    assert groups['image_pair_count']==1 and groups['total_control_points']==4
    assert groups['groups'][0]['coordinate_origin']['original_annotation_index_base']=='unverified'
    assert groups['groups'][0]['coordinate_order']==['x_optical','y_optical','x_sar','y_sar']
