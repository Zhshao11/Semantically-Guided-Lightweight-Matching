import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from sglm.evaluation import (estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations)
from sglm.geometry import affine_from_parameters,transform_points
from sglm.t02.data import load_t02_config,sample_transform,AffinePairs256
from sglm.t02.model import B0Full
from sglm.t02.loss import correspondence_labels,b0_loss


@pytest.fixture
def config():
    return load_t02_config()


def test_evaluator_is_model_and_torch_independent():
    subprocess.run([sys.executable,'-c',"import sys; import sglm.evaluation; assert 'torch' not in sys.modules; assert 'sglm.model' not in sys.modules"],check=True)


def test_true_wrong_matrix_and_failure_denominator(config):
    matrix=affine_from_parameters(256,256,7,(.97,.97),(.04,-.02),(11,-3))
    probes,reference=fixed_synthetic_probes(matrix)
    exact=evaluate_transform(matrix,probes,reference)
    assert exact['rmse_px']<1e-12
    wrong=matrix.copy();wrong[:,2]+=[5,-3]
    incorrect=evaluate_transform(wrong,probes,reference)
    assert incorrect['rmse_px']==pytest.approx(np.sqrt(34))
    empty=estimate_from_correspondences([],[],[],config['evaluation'])
    assert not empty['success'] and empty['status']=='empty_matches'
    failed=evaluate_transform(empty['matrix'],probes,reference,status=empty['status'])
    summary=summarize_evaluations([exact,incorrect,failed])
    assert summary['image_pairs']==3 and summary['failed_pairs']==1
    assert summary['failure_rate']==pytest.approx(1/3)
    assert summary['success_rate@5px']==pytest.approx(1/3)
    assert summary['registration_failure_rate@5px']==pytest.approx(2/3)


def test_degenerate_matches_fail_and_scene_reference_is_external(config):
    source=np.array([[0,0],[1,1],[2,2],[3,3]])
    assert not estimate_from_correspondences(source,source,np.ones(4),config['evaluation'])['success']
    rng=np.random.default_rng(5)
    prediction_points=rng.uniform(0,255,(30,2))
    matrix=affine_from_parameters(256,256,3,translation=(4,-7))
    fitted=estimate_from_correspondences(prediction_points,transform_points(prediction_points,matrix),np.ones(30),config['evaluation'])
    controls=np.array([[12.2,32.7],[234.5,17.4],[120.9,204.8]])
    result=evaluate_transform(fitted['matrix'],controls,transform_points(controls,matrix))
    assert result['rmse_px']<1e-4
    assert not np.any(np.all(prediction_points[:,None]==controls[None],axis=-1))


def test_debug_perturbation_order_and_ranges(config):
    rng=np.random.default_rng(4)
    for _ in range(50):
        matrix,params=sample_transform(rng,config)
        assert -10<=params['rotation_deg']<=10 and .9<=params['uniform_scale']<=1.1
        assert max(np.abs(params['translation_px']))<=16
        assert max(np.abs(params['shear_deg']))<=5
        center=np.array([[127.5,127.5]])
        assert np.allclose(transform_points(center,matrix),center+params['translation_px'])
        assert np.allclose(matrix,affine_from_parameters(256,256,params['rotation_deg'],
            (params['uniform_scale'],)*2,np.tan(np.deg2rad(params['shear_deg'])),params['translation_px']))


def test_fixed_loader_bitwise_repeat_no_log_and_online_mode(tmp_path,config):
    root=tmp_path/'OSDataset2.0'
    folder=root/'Patch-level Subset/OSdataset/256/train'
    folder.mkdir(parents=True)
    image=np.tile(np.arange(256,dtype=np.uint8),(256,1))
    op,sp=folder/'opt1.png',folder/'sar1.png'
    cv2.imwrite(str(op),image);cv2.imwrite(str(sp),image)
    matrix,_=sample_transform(np.random.default_rng(1),config)
    pair=dict(pair_id='osdataset/256/train/1',split='train',optical_path=str(op.relative_to(root)),sar_path=str(sp.relative_to(root)),
        optical_sha256=hashlib.sha256(op.read_bytes()).hexdigest(),sar_sha256=hashlib.sha256(sp.read_bytes()).hexdigest(),A_forward=matrix.tolist())
    plan=dict(config=config,samples=[pair])
    dataset=AffinePairs256(root,plan)
    first,second=dataset[0],dataset[0]
    for key in ('optical','sar_aug','A_forward','optical_valid_mask','sar_aug_valid_mask'):
        assert torch.equal(first[key],second[key])
    assert first['optical'][0,0,128]==pytest.approx(128/255)
    assert first['sar_aug'].min()>=0 and first['sar_aug'].max()<=1
    assert np.allclose(first['A_forward'],matrix)
    online=AffinePairs256(root,plan,mode='online')
    before=online[0]['A_forward'];online.set_epoch(1)
    assert not torch.equal(before,online[0]['A_forward'])


def test_out_of_bounds_and_invalid_pixels_never_positive():
    source_mask=torch.ones(1,256,256,dtype=torch.bool)
    target_mask=source_mask.clone();target_mask[:,:,:64]=False
    identity=torch.eye(2,3)[None]
    labels,_,positive=correspondence_labels(identity,source_mask,target_mask)
    assert (labels.reshape(1,32,32)[:,:,:8]==1024).all()
    assert not positive.reshape(1,32,32)[:,:,:8].any()
    moved=identity.clone();moved[:,:,2]=300
    labels,_,positive=correspondence_labels(moved,source_mask,source_mask)
    assert not positive.any() and (labels==1024).all()
    # Fixed evaluation lattice is not the model's coarse lattice.
    probes,_=fixed_synthetic_probes(np.eye(2,3))
    assert np.all(np.mod(probes[:,0]-3.5,8)!=0)


def test_b0_full_shapes_and_gt_free_prediction(config):
    torch.set_num_threads(2)
    model=B0Full(config).eval()
    optical,sar=torch.rand(1,3,256,256),torch.rand(1,1,256,256)
    mask=torch.ones(1,256,256,dtype=torch.bool)
    with torch.no_grad():
        output=model(optical,sar,mask,mask)
    assert output['optical_coarse'].shape==(1,64,32,32)
    assert output['optical_fine'].shape==(1,16,128,128)
    assert output['optical_descriptors'].shape==(1,1024,32)
    assert output['coarse_correlation'].shape==(1,1024,1024)
    assert output['row_logits'].shape==(1,1024,1025)
    assert list(inspect.signature(model.predict).parameters)==['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']
    names={module.__class__.__name__ for module in model.modules()}
    assert not names&{'JointScore','Budget','Fine','MultiheadAttention'}
    with pytest.raises(TypeError):
        model.predict(optical,sar,mask,mask,A_forward=torch.eye(2,3))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='BF16 acceptance uses actual CUDA')
def test_cuda_bf16_forward_and_backward(config):
    torch.set_num_threads(2)
    model=B0Full(config).cuda().train()
    batch=dict(optical=torch.rand(1,3,256,256,device='cuda'),sar_aug=torch.rand(1,1,256,256,device='cuda'),
        optical_valid_mask=torch.ones(1,256,256,dtype=torch.bool,device='cuda'),
        sar_aug_valid_mask=torch.ones(1,256,256,dtype=torch.bool,device='cuda'),A_forward=torch.eye(2,3,device='cuda')[None])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(batch['optical'],batch['sar_aug'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
    loss,_=b0_loss(model,output,batch,config,0)
    loss.backward()
    assert output['coarse_correlation'].dtype==torch.bfloat16
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
