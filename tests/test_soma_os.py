import torch
from sglm.soma_os import theta_to_flow,resize_flow,refinement_forward,inverse_targets,dense_to_matches
from sglm.soma_os import soma_loss
from sglm.soma_training import EpochMicroSampler


def test_identity_and_translation_flow_pixel_units():
    identity=torch.tensor([[[1.,0,0],[0,1,0]]])
    assert theta_to_flow(identity,8,12).abs().max()==0
    translated=identity.clone();translated[0,0,2]=2*3/12;translated[0,1,2]=-2*2/8
    flow=theta_to_flow(translated,8,12)
    assert torch.allclose(flow[:,0],torch.full((1,8,12),3.),atol=1e-6)
    assert torch.allclose(flow[:,1],torch.full((1,8,12),-2.),atol=1e-6)
    larger=resize_flow(flow,(16,24));assert torch.allclose(larger[:,0],torch.full((1,16,24),6.),atol=1e-5)


def test_zero_flow_refinement_samples_identity_xy_on_rectangular_feature():
    from torch import nn
    class Capture(nn.Module):
        concat_logits=False
        def __init__(self):
            super().__init__();self.disp_emb=nn.Conv2d(2,16,1);self.encoder=nn.Identity();self.out_conv=nn.Identity()
    obj=Capture();x=torch.arange(96).reshape(1,1,8,12).float();seen=[]
    hook=obj.encoder.register_forward_pre_hook(lambda m,a:seen.append(a[0]))
    refinement_forward(obj,x,x,torch.zeros(1,2,8,12));hook.remove()
    assert torch.allclose(seen[0][:,1:2],x,atol=1e-5)


def test_inverse_target_and_optical_to_augmented_SAR_decode():
    # Original256 SAR translated+8x,-4y; inverse displacement is-8x,+4y.
    A=torch.tensor([[[1.,0,8],[0,1,-4]]]);mask=torch.ones(1,256,256,dtype=torch.bool)
    gt,valid=inverse_targets(dict(A_forward=A,sar_aug_valid_mask=mask),512,'cpu')
    assert torch.allclose(gt[0,0,100:300,100:300],torch.full((200,200),-16.),atol=1e-4)
    source,target,score=dense_to_matches(gt,torch.zeros(1,1,512,512))
    expected=source+torch.tensor([8.,-4.])
    assert torch.allclose(target,expected,atol=1e-4) and torch.all(score==.5)
    assert not valid[0,:,0].any()


def test_paired_distributed_stream_has_no_duplicates_and_resume_preserves_tokens():
    batches=[list(EpochMicroSampler(16,9,rank,4,1,4,2)) for rank in range(4)]
    assert len(batches[0])==4
    ids=[batch[0][0] for rank_batches in batches for batch in rank_batches]
    tokens=[batch[0][1] for rank_batches in batches for batch in rank_batches]
    assert sorted(ids)==list(range(16)) and sorted(tokens)==list(range(32,48))
    resumed=list(EpochMicroSampler(16,9,2,4,1,4,2,start_update=2))
    assert resumed==batches[2][2:]
    single=list(EpochMicroSampler(16,9,0,1,1,4,2))
    assert [single[u*4+r][0] for u in range(4) for r in range(4)]==[batches[r][u][0] for u in range(4) for r in range(4)]


def test_masked_compound_loss_has_finite_gradients_at_perfect_flow():
    flow=torch.zeros(1,2,8,8,requires_grad=True);cer=torch.zeros(1,1,8,8,requires_grad=True)
    output=dict(flow_final=flow,certainty=cer)
    for level in ['8','4','2','1']:
        output['flow_'+level]=flow
        output['flow_'+level+'_delta']=flow
        if level in ['8','4']:output['flow_'+level+'_pred']=flow
    loss,metrics=soma_loss(output,torch.zeros_like(flow),torch.ones(1,8,8,dtype=torch.bool));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(flow.grad).all() and torch.isfinite(cer.grad).all()
