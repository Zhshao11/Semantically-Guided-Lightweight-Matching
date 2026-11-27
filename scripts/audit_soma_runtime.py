"""Read-only SOMA feasibility probes. Random features never imply trained accuracy."""
import argparse,gc,hashlib,importlib.util,json,sys,traceback
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='runs/soma_audit_v1/SOMA');ap.add_argument('--output',default='reports/soma_audit_v1');a=ap.parse_args()
    root=Path(a.source).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source_hash={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
    sys.path.insert(0,str(root));torch.set_num_threads(4);torch.manual_seed(20261127)
    from models.matcher import GLAM,theta_to_flow,RefineMatcher
    import models.matcher as original
    from torchvision.ops import DeformConv2d
    assert Path(original.__file__).resolve().is_relative_to(root)
    result=dict(source=str(root),source_sha256=source_hash,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),trained_SOMA_weights_loaded=False,training_started=False,
        tests_are_geometry_and_random_feature_smoke_only=True)
    device='cuda'
    with torch.inference_mode():
        # Check actual torchvision customCUDA operator on RTX5090.
        op=DeformConv2d(4,4,3,padding=1).cuda().eval();x=torch.rand(1,4,16,16,device=device)
        z=op(x,torch.zeros(1,18,16,16,device=device));result['deform_conv2d']=dict(passed=True,shape=list(z.shape),finite=bool(torch.isfinite(z).all()))
        del op,x,z
        # An identity sampling transform must induce zero displacement.
        theta=torch.tensor([[[1.,0,0],[0,1,0]]],device=device)
        result['identity_theta_to_flow']=[]
        for size in [16,256,512]:
            flow=theta_to_flow(theta,size,size)
            result['identity_theta_to_flow'].append(dict(size=size,max_abs_component_px=float(flow.abs().max()),
                max_vector_norm_px=float(flow.norm(dim=1).max()),zero_flow_expected=True,zero_flow_passed=bool(flow.abs().max()<1e-6)))
        # Capture the original refinement's actual warp, without changing its grid.
        ref=RefineMatcher(3,use_certainty=False,concat_logits=False).cuda().eval()
        h,w=8,12;f=torch.arange(h*w,device=device,dtype=torch.float32).reshape(1,1,h,w).repeat(1,3,1,1)
        captured=[];original_sample=original.F.grid_sample
        def track(*args,**kw):
            value=original_sample(*args,**kw);captured.append((args[1].clone(),value.clone()));return value
        original.F.grid_sample=track
        try:ref(f,f,torch.zeros(1,2,h,w,device=device))
        finally:original.F.grid_sample=original_sample
        grid,warped=captured[0]
        yy,xx=torch.meshgrid(torch.linspace(-1+1/h,1-1/h,h,device=device),torch.linspace(-1+1/w,1-1/w,w,device=device),indexing='ij')
        expected=torch.stack([xx,yy],-1)[None]
        result['refinement_zero_flow_warp']=dict(shape=[h,w],max_abs_grid_vs_expected_xy=float((grid-expected).abs().max()),
            max_abs_warp_minus_original=float((warped-f).abs().max()),identity_expected=True,
            identity_passed=bool(torch.allclose(warped,f,atol=1e-5,rtol=0)),
            cause='upstream stacks meshgrid(y,x) without swapping to grid_sample(x,y)')
        del ref,f,grid,warped,captured
        # Original512 path versus requested256 features. No encoder replacement
        # in these GLAM-only shape probes: encoder was not run at all.
        model=GLAM().cuda().eval();result['GLAM_parameters']=sum(p.numel() for p in model.parameters());probes=[]
        for size in [512,256]:
            features=[]
            for _ in range(2):
                features.append({'1':torch.rand(1,3,size,size,device=device),
                    '2':torch.rand(1,64,size//2,size//2,device=device),
                    '4':torch.rand(1,256,size//4,size//4,device=device),
                    '8':torch.rand(1,512,size//8,size//8,device=device),
                    '16':torch.rand(1,1024,32,32,device=device), # exact original DINO448 ->32 tokens
                    '32':torch.rand(1,2048,size//32,size//32,device=device)})
            try:
                output=model(*features);probes.append(dict(size=size,passed=True,
                    shapes={k:list(v.shape) for k,v in output.items()},finite=all(bool(torch.isfinite(v).all()) for v in output.values())))
                del output
            except Exception as e:
                probes.append(dict(size=size,passed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
            del features;gc.collect();torch.cuda.empty_cache()
        result['GLAM_random_feature_forward']=probes
    # Constructor check records the real hardcodedDINO failure, not a stand-in model.
    try:
        from models.encoder import Encoder
        model=Encoder();result['original_Encoder_constructor']=dict(passed=True)
    except Exception as e:
        result['original_Encoder_constructor']=dict(passed=False,error_type=type(e).__name__,error=str(e))
    result['dependency_imports']={m:importlib.util.find_spec(m) is not None for m in ['torch','torchvision','numpy','cv2','PIL','kornia','wandb','torchsummary']}
    for p,digest in source_hash.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==digest
    result['original_source_unchanged']=True
    (out/'runtime_probes.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
