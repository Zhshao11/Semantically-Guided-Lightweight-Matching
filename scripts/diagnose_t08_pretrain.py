"""Train-only T07 gating/oracle-probability diagnostic; no pseudo labels for matching."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sglm.t02.data import AffinePairs256
from sglm.t07_semantic import load_t07_model
from sglm.t06_appearance import sar_global
from sglm.model import coarse_xy
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from sglm.geometry import transform_points,valid_pixel_centers
from sglm.experiment_io import write_csv


class ProbabilityOverride(nn.Module):
    def __init__(self,head):super().__init__();self.head=head;self.probability=None;self.mode='student';self.metrics={}
    def forward(self,optical,sar):
        h=self.head
        with torch.autocast(optical.device.type,enabled=False):
            fo=optical.detach().float();fs=sar.detach().float();student=h.semantic(fo).softmax(1);p=student if self.probability is None else self.probability
            confidence=1+(p*p.clamp_min(1e-8).log()).sum(1,keepdim=True)/math.log(7)
            xo=F.relu(h.optical(fo));zs=F.relu(h.sar(fs))[:,:,None,None].expand_as(xo)
            appearance=h.appearance(torch.cat([xo,zs],1));residual=h.residual(torch.cat([xo,zs,h.embedding(p),confidence,appearance.sigmoid()],1));delta=confidence*residual
            self.metrics=dict(appearance_logits=appearance.flatten().cpu().numpy(),delta_logits=delta.flatten().cpu().numpy(),confidence=confidence.flatten().cpu().numpy())
            return (appearance if self.mode=='appearance_only' else appearance+delta)[:,0]


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    setup=json.loads(Path('reports/t08_setup_v1/setup.json').read_text());p=json.loads(Path('configs/t08_protocol_v1.json').read_text());out=Path('reports/t08_preflight_v1');out.mkdir(exist_ok=True)
    saved=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config'];model=load_t07_model(saved,'S',256);wrap=ProbabilityOverride(model.appearance_head);model.appearance_head=wrap;model.eval()
    root='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0';ds=AffinePairs256(root,'reports/t08_setup_v1/diagnostic_plan.json',split='train',preload=True)
    cache=Path('runs/t07_semantic_cache_v1');prob=np.load(cache/'conditional7.npy',mmap_mode='r');weight=np.load(cache/'weight.npy',mmap_mode='r');ids=json.loads((cache/'pair_ids.json').read_text());index={uid:i for i,uid in enumerate(ids)}
    stats=[];records=[];deltas=[];base=[];confs=[];sets={};coarse_correct=[]
    with torch.inference_mode():
        for number,item in enumerate(ds):
            uid=item['pair_id'];i=index[uid];assert '/train/' in uid
            args=[item[k][None].cuda() for k in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
            with torch.autocast('cuda',dtype=torch.bfloat16):features=model.encode(*args)
            optical=features['optical_coarse'].float();student=wrap.head.semantic(optical).softmax(1)
            teacher=torch.from_numpy(np.array(prob[i],copy=True)).float().cuda()[None];teacher=teacher/teacher.sum(1,keepdim=True)
            reliable=torch.from_numpy(np.array(weight[i]>0,copy=True)).cuda()[None,None];blend=torch.where(reliable,teacher,student)
            for mode,probability in [('student',None),('appearance_only',None),('teacher_probability',teacher),('teacher_reliable_probability',blend)]:
                wrap.mode=mode;wrap.probability=probability
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model.correlate(features);pred=model.refine(output,args)
                qi=output['query_indices'];sets[(uid,mode)]=set(qi.cpu().tolist())
                src=coarse_xy(qi,32).cpu().numpy();chosen=output['row_logits'][0].argmax(-1).cpu().numpy();target=coarse_xy(torch.tensor(chosen.clip(0,1023),device='cuda'),32).cpu().numpy();truth=transform_points(src,item['A_forward'].numpy());correct=(chosen<1024)&valid_pixel_centers(truth,256,256)&(np.linalg.norm(target-truth,axis=1)<5)
                points=[pred[k].float().cpu().numpy() for k in ['source','target','score']];fit=estimate_from_correspondences(*points,cfg['evaluation'],seed=cfg['seed']);probes,ref=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask']);ev=evaluate_transform(fit['matrix'],probes,ref,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
                record=dict(pair_id=uid,mode=mode,evaluation=ev,selected_query_correct5=float(correct.mean()),matches=len(points[0]),inliers=fit['inliers']);records.append(record)
                if mode=='student':
                    d=wrap.metrics['delta_logits'];a=wrap.metrics['appearance_logits'];c=wrap.metrics['confidence'];deltas.extend(d);base.extend(a);confs.extend(c)
                    stats.append(dict(pair_id=uid,delta_abs_mean=float(np.abs(d).mean()),delta_abs_P95=float(np.percentile(np.abs(d),95)),appearance_std=float(a.std()),correction_std=float(d.std()),correction_std_over_appearance_std=float(d.std()/max(a.std(),1e-12)),delta_abs_le_1e_minus3_fraction=float((np.abs(d)<=1e-3).mean()),semantic_confidence_mean=float(c.mean())))
            if (number+1)%32==0:print('routing diagnosed',number+1,flush=True)
    summaries=[]
    for mode in p['preflight']['modes']:
        rows=[r for r in records if r['mode']==mode];jaccard=np.mean([len(sets[(r['pair_id'],mode)]&sets[(r['pair_id'],'student')])/len(sets[(r['pair_id'],mode)]|sets[(r['pair_id'],'student')]) for r in rows])
        summaries.append(dict(mode=mode,**summarize_evaluations([r['evaluation'] for r in rows]),selected_query_correct5=float(np.mean([r['selected_query_correct5'] for r in rows])),mean_query_Jaccard_vs_student=float(jaccard)))
    d=np.asarray(deltas);a=np.asarray(base);c=np.asarray(confs);gate=dict(abs_mean=float(np.abs(d).mean()),abs_median=float(np.median(np.abs(d))),abs_P95=float(np.percentile(np.abs(d),95)),abs_le_1e_minus3_fraction=float((np.abs(d)<=1e-3).mean()),correction_std=float(d.std()),appearance_std=float(a.std()),std_ratio=float(d.std()/a.std()),semantic_confidence_mean=float(c.mean()))
    write_csv(out/'T07_gate_by_pair.csv',stats);write_csv(out/'teacher_probability_routing.csv',summaries)
    result=dict(training_source_pairs=128,K=256,split='train',gate=gate,routing=summaries,pairs=records,teacher_is_not_GT=True,teacher_used_offline_for_diagnosis_only=True,no_validation_teacher_cache=True,annotation_status='OS has no semanticGT; optional template only; user waived nonexistentannotation prerequisite',source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['scripts/diagnose_t08_pretrain.py','sglm/t07_semantic.py']})
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(gate=gate,routing=summaries),indent=2))


if __name__=='__main__':main()
