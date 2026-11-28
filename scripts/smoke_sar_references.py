"""Native examples, exact preprocessing, affine/dense coordinate regression, fixed2train smoke."""
import argparse,hashlib,importlib.util,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from sglm.sar_references import SARReference
from sglm.sar_reference_geometry import dense_correspondences,sample_dense_targets,restore_resized_coordinates
from sglm.external_reference import cpu_input
from sglm.t02.data import AffinePairs256
from sglm.evaluation import estimate_from_correspondences
from sglm.diagnostics import bn_snapshot,bn_unchanged


def read_rgb(path):return np.array(Image.open(path).convert('RGB'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/sar_references_v1.json').read_text());out=Path(p['output']);out.mkdir(exist_ok=True,parents=True)
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    adapter=SARReference(a.method,p,cfg);before=bn_snapshot(adapter.model);native={}
    with torch.inference_mode():
        if a.method.startswith('FHReg_GUSO'):
            repo=Path(p['source_root'])/'GUSO';paths=[repo/'assert/guso_testpairs'/f'18_{kind}.png' for kind in ['opt','sar']]
            images=[read_rgb(path) for path in paths]
            # Repository ships sample inputs under assert/, not demo.py's default cwd.
            raw_inputs=[torch.from_numpy(im.copy()).permute(2,0,1)[None].float().cuda() for im in images]
            raw=adapter.model(dict(image_opt=raw_inputs[0],image_sar=raw_inputs[1]))
            assert all(torch.isfinite(raw[k]).all() for k in ['mkpts0_f','mkpts1_f','mconf_f'])
            native=dict(paths=[str(path) for path in paths],sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                native_demo_raw255_forward=True,matches=len(raw['mkpts0_f']),note='Native demo.py uses rawfloat0..255; OSzero_shot uses TF.to_tensor/255. Primary OS protocol follows zero_shot, never chooses normalization by validation results.')
        elif a.method=='GDROS_WHU':
            root=Path('/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR/extracted_e077a/training/whu_opt_sar512')
            opt={q.stem.split('_')[0]:q for q in (root/'opt').iterdir() if q.suffix.lower() in ['.png','.tif','.tiff']}
            sar={q.stem.split('_')[0]:q for q in (root/'sar_warped').iterdir() if q.suffix.lower() in ['.png','.tif','.tiff']}
            pair=min(opt.keys()&sar.keys());path=opt[pair];other=sar[pair]
            images=[read_rgb(q) for q in [path,other]];assert all(im.shape==(512,512,3) for im in images)
            inputs=[torch.from_numpy(im.copy()).permute(2,0,1)[None].float().cuda() for im in images]
            _,flow=adapter.model(*inputs,iters=32,test_mode=True);lsr=adapter.native_lsr(1,512,512,flow,200000)
            assert flow.shape==lsr.shape==(1,2,512,512) and torch.isfinite(flow).all() and torch.isfinite(lsr).all()
            # Test actual unmodified native LSR with known displacement, not GT-fed model inference.
            yy,xx=torch.meshgrid(torch.arange(512,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
            expected=torch.stack([.02*xx+.03*yy+8,-.01*xx-.03*yy-6]).float()[None]
            projected=adapter.native_lsr(1,512,512,expected,200000);lsr_error=float((projected-expected.cpu()).abs().max());assert lsr_error<.005,lsr_error
            native=dict(paths=[str(path),str(other)],sha256=[hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,other]],
                source='cached author WHU training input in official512 format; no test/val selection',native512_32iterations_LSR_passed=True,LSR_affine_oracle_max_error_native_px=lsr_error)
        plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text());ds=AffinePairs256(a.root,dict(config=plan['config'],samples=plan['samples'][:2]),split='train',preload=False);rows=[]
        for item in ds:
            data=cpu_input(item)
            if a.method.startswith('FHReg_GUSO'):
                # Native OS Dataset applies torchvisionTF.to_tensor on CPU, then cuda.
                import torchvision.transforms.functional as TF
                size=p[a.method]['input_resize'];opt=cv2.resize(data['optical'],(size,size),interpolation=cv2.INTER_LINEAR);sar=cv2.resize(data['sar'],(size,size),interpolation=cv2.INTER_LINEAR)
                native_inputs=[TF.to_tensor(opt)[None].cuda(),TF.to_tensor(np.repeat(sar[:,:,None],3,2))[None].cuda()]
                prepared=adapter.prepare(data);assert all(torch.equal(x,y) and x.stride()==y.stride() for x,y in zip(native_inputs,prepared))
                reference=adapter.model(dict(image_opt=native_inputs[0],image_sar=native_inputs[1]))
            start=time.perf_counter();pred=adapter.predict(data,include_native_dense=a.method=='GDROS_WHU')
            fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
            again=adapter.predict(data);repeat={k:np.array_equal(pred[k],again[k]) for k in ['source','target','score']};assert all(repeat.values()),repeat
            if a.method.startswith('FHReg_GUSO'):
                from sglm.sar_reference_geometry import filter_correspondences
                arrays=[reference[k].float().cpu().numpy() for k in ['mkpts0_f','mkpts1_f','mconf_f']]
                arrays[:2]=[restore_resized_coordinates(v,(256,256),(size,size)) for v in arrays[:2]]
                original=filter_correspondences(*arrays,data['optical_mask'],data['sar_mask'])
                assert all(np.array_equal(original[i],pred[k]) for i,k in enumerate(['source','target','score']))
            rows.append(dict(pair_id=item['pair_id'],raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],
                repeat_exact=repeat,fit_status=fit['status'],inliers=fit['inliers'],smoke_wall_seconds=time.perf_counter()-start,formal_timing=False))
        assert bn_unchanged(before,adapter.model)
    record=dict(passed=True,method=a.method,native_sample=native,train_source_smoke=rows,parameters=adapter.parameters,
        compatibility=adapter.compatibility,all_modules_eval=not any(m.training for m in adapter.model.modules()),BN_unchanged=True,
        precision='nativeFP32' if not a.method.startswith('T05') else 'originalBF16',torch=torch.__version__,cuda=torch.version.cuda,
        no_training=True,test_used=False,threshold_tuning=False,official_model_forward_unchanged=True)
    (out/f'{a.method}_smoke.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
