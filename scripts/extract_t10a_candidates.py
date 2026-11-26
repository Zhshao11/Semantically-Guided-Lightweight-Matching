"""FrozenT05 TRAIN-only candidate extraction. No optimizer, teacheronSAR or rule fitting."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import torch
from sglm.t05 import UniformB0
from sglm.t02.data import AffinePairs256
from sglm.model import coarse_xy
from sglm.geometry import transform_points
from sglm.diagnostics import bn_snapshot,bn_unchanged
from sglm.t10_signal import sar_candidate_to_nominal_optical,sample_nominal_semantics,negative_pool,choose_controls,semantic_cosine
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=Path('runs/t10a_candidates_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Frozen candidate set cannot be overwritten')
    p=json.loads(Path('configs/t10a_protocol_v1.json').read_text());freeze=json.loads(Path('reports/t10_setup_v1/freeze.json').read_text());plan=json.loads(Path(p['samples']['plan']).read_text());assert sha(p['samples']['plan'])==freeze['plan_sha256'];assert sha(p['matcher_checkpoint'])==p['matcher_sha256']
    cache=Path(p['teacher_cache']);cacheids=json.loads((cache/'pair_ids.json').read_text());cache_summary=json.loads((cache/'summary.json').read_text());assert cache_summary['teacher_checkpoint_sha256']==p['teacher_sha256'] and cache_summary['train_only']
    ps=np.load(cache/'conditional7.npy',mmap_mode='r');ws=np.load(cache/'weight.npy',mmap_mode='r');perms=np.load('runs/t10a_permutation_v1/selected_candidate_probability7.npy',mmap_mode='r')
    inputs={str(cache/name):sha(cache/name) for name in ['conditional7.npy','weight.npy','pair_ids.json']};inputs['runs/t10a_permutation_v1/selected_candidate_probability7.npy']=sha('runs/t10a_permutation_v1/selected_candidate_probability7.npy')
    saved=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=UniformB0(saved['config'],1024).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);before=bn_snapshot(model)
    ds=AffinePairs256('/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0',p['samples']['plan'],split='train',preload=True);assert len(ds)==256
    xy=coarse_xy(torch.arange(1024),32).numpy().astype(np.float64);relation_parts={};coverage=[];files=[];cases=[]
    with torch.inference_mode():
        for pair_index,item in enumerate(ds):
            row=plan['samples'][pair_index];uid=item['pair_id'];ci=row['original_cache_index'];assert uid==cacheids[ci] and '/train/' in uid
            args=[item[key][None].cuda() for key in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(*args)
            scores=prediction['coarse_correlation'][0].float().cpu().numpy();A=np.asarray(row['A_forward'],np.float64);truth=transform_points(xy,A);nominal=sar_candidate_to_nominal_optical(xy,A)
            optical_geom=prediction['optical_coarse_mask'][0].cpu().numpy().reshape(-1);sar_geom=prediction['sar_coarse_mask'][0].cpu().numpy().reshape(-1)
            eroded=cv2.erode(item['sar_aug_valid_mask'].numpy().astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            truth_valid=cv2.remap(eroded.astype(np.float32),truth[:,0].astype(np.float32)[:,None],truth[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0]>.999
            query_geom=optical_geom&truth_valid&((truth>=2)&(truth<=253)).all(1)
            pool,distance=negative_pool(truth,xy,sar_geom,16.);pool&=query_geom[:,None]
            ranked=np.argsort(-np.where(pool,scores,-np.inf),axis=1,kind='stable')[:,:5];has=pool[np.arange(1024)[:,None],ranked];hard=np.where(has,ranked,-1).astype(np.int16);hard_scores=np.where(has,scores[np.arange(1024)[:,None],ranked],np.nan)
            original_p=torch.from_numpy(np.array(ps[ci],copy=True)).cuda().float();original_p/=original_p.sum(0,keepdim=True);weight=torch.from_numpy(np.array(ws[ci],copy=True)).cuda()
            candidate_p,candidate_w,candidate_ok=sample_nominal_semantics(original_p,weight,nominal);perm_p,_,_=sample_nominal_semantics(torch.from_numpy(np.array(perms[pair_index],copy=True)).cuda(),weight,nominal)
            candidate_p=candidate_p.cpu().numpy();perm_p=perm_p.cpu().numpy();candidate_w=candidate_w.cpu().numpy();candidate_ok=candidate_ok.cpu().numpy()&sar_geom
            query_p=original_p.flatten(1).T.cpu().numpy();query_w=weight.flatten().cpu().numpy();query_ok=(query_w>0)&optical_geom;query_class=query_p.argmax(1).astype(np.int8);candidate_class=candidate_p.argmax(1).astype(np.int8);permuted_class=perm_p.argmax(1).astype(np.int8)
            valid_hard=(hard>=0)&query_ok[:,None]&candidate_ok[np.maximum(hard,0)];qidx,rank=np.where(valid_hard);hid=hard[qidx,rank].astype(int);N=len(hid);random_ids=np.full((N,3,5),-1,np.int16);rng=np.random.default_rng(p['seed']+ci*1009)
            for n,(q,h) in enumerate(zip(qidx,hid)):
                choices=choose_controls(pool[q]&candidate_ok,hard[q,hard[q]>=0],distance[q],candidate_w,candidate_class,h,rng,5)
                random_ids[n]=np.stack(choices)
            safe=np.maximum(random_ids,0);random_p=candidate_p[safe];random_perm_p=perm_p[safe];random_cos=semantic_cosine(query_p[qidx,None,None,:],random_p).astype(np.float32);random_perm_cos=semantic_cosine(query_p[qidx,None,None,:],random_perm_p).astype(np.float32);random_valid=random_ids>=0;random_cos[~random_valid]=np.nan;random_perm_cos[~random_valid]=np.nan
            part=dict(pair_index=np.full(N,pair_index,np.int16),query_index=qidx.astype(np.int16),rank=rank.astype(np.int8),hard_id=hid.astype(np.int16),query_class=query_class[qidx],hard_class=candidate_class[hid],hard_perm_class=permuted_class[hid],query_weight=query_w[qidx],hard_weight=candidate_w[hid],hard_distance=distance[qidx,hid].astype(np.float32),hard_descriptor=scores[qidx,hid],hard_semantic=semantic_cosine(query_p[qidx],candidate_p[hid]).astype(np.float32),hard_perm_semantic=semantic_cosine(query_p[qidx],perm_p[hid]).astype(np.float32),random_id=random_ids,random_semantic=random_cos,random_perm_semantic=random_perm_cos,random_distance=distance[qidx[:,None,None],safe].astype(np.float32),random_weight=candidate_w[safe],random_class=np.where(random_valid,candidate_class[safe],-1).astype(np.int8),random_perm_class=np.where(random_valid,permuted_class[safe],-1).astype(np.int8),random_descriptor=scores[qidx[:,None,None],safe])
            for name,array in part.items():relation_parts.setdefault(name,[]).append(array)
            packed=np.packbits(pool,axis=1,bitorder='little');filename=f'pair_{pair_index:03d}.npz'
            np.savez_compressed(out/filename,hard_indices=hard,hard_scores=hard_scores,negative_pool_packbits=packed,A_forward=A,query_truth_xy=truth,nominal_candidate_xy=nominal,query_geometry_valid=query_geom,query_semantic_valid=query_ok,candidate_geometry_valid=sar_geom,candidate_semantic_valid=candidate_ok,query_probability7=query_p,candidate_probability7=candidate_p,candidate_permuted_probability7=perm_p,query_weight=query_w,candidate_weight=candidate_w,relation_query=qidx,relation_hard_rank=rank,random_ids=random_ids)
            files.append(dict(pair_index=pair_index,pair_id=uid,verified512_parent=row['verified512_parent'],file=filename,sha256=sha(out/filename)))
            counts=dict(pair_index=pair_index,pair_id=uid,verified512_parent=row['verified512_parent'],total_queries=1024,geometric_queries=int(query_geom.sum()),existing_semantic_queries=int(query_ok.sum()),both_valid_queries=int((query_geom&query_ok).sum()),geometric_candidates=int(sar_geom.sum()),nominal_semantic_candidates=int(candidate_ok.sum()),all_geometric_hard_slots=int(has.sum()),semantic_query_geometric_hard_slots=int((has&query_ok[:,None]).sum()),reliable_hard_relations=N,reliable_queries_with_hard=len(np.unique(qidx)))
            for j,name in enumerate(['uncontrolled','distance_weight','distance_weight_class']):counts[name+'_relations_with_controls']=int(random_valid[:,j].any(1).sum())
            coverage.append(counts)
            if pair_index<16:
                valid_rows=np.flatnonzero(random_valid[:,1].any(1));case=dict(pair_index=pair_index,pair_id=uid,file=filename,eligible=False)
                if len(valid_rows):
                    start=int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4],'little')%1024;choice=min(valid_rows,key=lambda i:((int(qidx[i])-start)%1024,int(rank[i])));q=int(qidx[choice]);h=int(hid[choice]);rids=random_ids[choice,1];case.update(eligible=True,query_index=q,hard_rank=int(rank[choice]),hard_id=h,matched_random_ids=rids[rids>=0].tolist(),query_xy=xy[q].tolist(),GT_augmented_xy=truth[q].tolist(),hard_augmented_xy=xy[h].tolist(),hard_nominal_optical_xy=nominal[h].tolist(),hard_score=float(scores[q,h]),semantic_similarity=float(part['hard_semantic'][choice]),random_semantic_similarity=float(np.nanmean(random_cos[choice,1])),query_probability=query_p[q].tolist(),hard_probability=candidate_p[h].tolist(),random_probability_mean=candidate_p[rids[rids>=0]].mean(0).tolist())
                cases.append(case)
            if (pair_index+1)%32==0:print('extracted',pair_index+1,'pairs',flush=True)
    arrays={name:np.concatenate(parts,axis=0) for name,parts in relation_parts.items()};np.savez_compressed(out/'relations.npz',**arrays)
    assert bn_unchanged(before,model);assert sha(p['matcher_checkpoint'])==p['matcher_sha256']
    summary=dict(protocol=p,pairs=256,parents=128,relations=len(arrays['hard_id']),inputs_sha256=inputs,checkpoint_sha256=p['matcher_sha256'],plan_sha256=freeze['plan_sha256'],candidate_files=files,relations_sha256=sha(out/'relations.npz'),BN_unchanged=True,model_eval=True,model_all_parameters_require_grad_false=True,training_started=False,teacher_on_SAR=False,validation_test_used=False,original_affine_precision='stored float64 for all nominalcoordinates; warp follows same storedplan',negative_exclusion_radius_px=16.,candidate_selection_independent_of_semantic_similarity=True,hard_candidates_selected_before_semantic_mask=True,controls_fixed=True,cases_fixed_order_first16=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'coverage.json').write_text(json.dumps(coverage,indent=2));(out/'cases.json').write_text(json.dumps(cases,indent=2))
    report=Path('reports/t10_results_v1');report.mkdir(exist_ok=True);write_csv(report/'coverage_by_pair.csv',coverage);(report/'extraction.json').write_text(json.dumps(summary,indent=2));print('Complete',summary['relations'],'reliablehardrelations')


if __name__=='__main__':main()
