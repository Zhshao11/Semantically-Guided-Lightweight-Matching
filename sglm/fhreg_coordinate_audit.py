"""Explicit pixel-center/crop/pad coordinates for audit; production matching untouched."""
import numpy as np
from .sar_reference_geometry import restore_resized_coordinates


def preprocessing_matrix(original_hw,resized_hw,crop_xy=(0,0),pad_left_top=(0,0)):
    h,w=original_hw;rh,rw=resized_hw
    if min(h,w,rh,rw)<=0:raise ValueError('Positive image dimensions required')
    sx,sy=rw/w,rh/h;cx,cy=crop_xy;px,py=pad_left_top
    # Resize pixel centers first, crop second, pad third; right/bottom do not shift xy.
    return np.array([[sx,0,(sx-1)/2-cx+px],[0,sy,(sy-1)/2-cy+py],[0,0,1]],np.float64)


def transform(points,matrix):
    points=np.asarray(points,np.float64).reshape(-1,2);M=np.asarray(matrix,np.float64)
    if M.shape==(2,3):M=np.vstack([M,[0,0,1]])
    return points@M[:2,:2].T+M[:2,2]


def native_to_original(points,original_hw,resized_hw,crop_xy=(0,0),pad_left_top=(0,0)):
    # Invoke the exact production FHReg restore after undoing syntheticcrop/pad.
    coords=np.asarray(points,np.float32)+np.asarray(crop_xy,np.float32)-np.asarray(pad_left_top,np.float32)
    return restore_resized_coordinates(coords,original_hw,resized_hw)


def known_cases():
    A=np.array([[1.037,.083,11.25],[-.047,.923,-7.75],[0,0,1]],np.float64)
    T=np.array([[1,0,13.25],[0,1,-9.5],[0,0,1]],np.float64)
    return [dict(name='256_to512_to256_translation',original_hw=(256,256),resized_hw=(512,512),A=T),
        dict(name='256_to512_to256_fullaffine',original_hw=(256,256),resized_hw=(512,512),A=A),
        dict(name='512_to256_to512_fullaffine',original_hw=(512,512),resized_hw=(256,256),A=A),
        dict(name='non_square_xy_scale',original_hw=(192,320),resized_hw=(512,640),A=A),
        dict(name='different_source_target_crops',original_hw=(256,256),resized_hw=(512,512),A=A,source_crop=(23,11),target_crop=(7,31)),
        dict(name='asymmetric_padding',original_hw=(256,256),resized_hw=(512,512),A=A,source_pad=(17,9),target_pad=(3,27)),
        dict(name='crop_and_padding_both_sides',original_hw=(256,256),resized_hw=(512,512),A=A,source_crop=(29,13),target_crop=(5,37),source_pad=(11,23),target_pad=(19,7))]


def roundtrip_case(case,seed=20261128):
    rng=np.random.default_rng(seed);h,w=case['original_hw'];rh,rw=case['resized_hw']
    x=np.r_[rng.uniform([0,0],[w-1,h-1],size=(1000,2)),np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1],[17.125,43.75]])]
    y=transform(x,case['A']);Ps=preprocessing_matrix((h,w),(rh,rw),case.get('source_crop',(0,0)),case.get('source_pad',(0,0)))
    Pt=preprocessing_matrix((h,w),(rh,rw),case.get('target_crop',(0,0)),case.get('target_pad',(0,0)))
    native_x=transform(x,Ps);native_y=transform(native_x,Pt@case['A']@np.linalg.inv(Ps))
    assert np.allclose(native_y,transform(y,Pt),atol=1e-10,rtol=0)
    recovered_x=native_to_original(native_x,(h,w),(rh,rw),case.get('source_crop',(0,0)),case.get('source_pad',(0,0)))
    recovered_y=native_to_original(native_y,(h,w),(rh,rw),case.get('target_crop',(0,0)),case.get('target_pad',(0,0)))
    inverse_original=transform(recovered_y,np.linalg.inv(case['A']))
    errors={key:np.linalg.norm(a-b,axis=1) for key,a,b in [('source_roundtrip',recovered_x,x),('target_roundtrip',recovered_y,y),('forward_from_recovered_source',recovered_y,transform(recovered_x,case['A'])),('affine_inverse_roundtrip',inverse_original,x)]}
    negative_xy=native_to_original(native_y[:,::-1],(h,w),(rh,rw),case.get('target_crop',(0,0)),case.get('target_pad',(0,0)))
    return dict(name=case['name'],points=len(x),original_hw=list(case['original_hw']),resized_hw=list(case['resized_hw']),
        source_transform=Ps.tolist(),target_transform=Pt.tolist(),known_affine=case['A'].tolist(),
        errors={key:dict(max_px=float(v.max()),mean_px=float(v.mean())) for key,v in errors.items()},xy_swap_negative_mean_px=float(np.linalg.norm(negative_xy-y,axis=1).mean())),(recovered_x,recovered_y)
