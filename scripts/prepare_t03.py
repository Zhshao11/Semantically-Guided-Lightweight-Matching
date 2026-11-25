"""Freeze full train identities and all validation affines, with T02 settings preserved."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from sglm.t02.data import sample_transform


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('configs/t02_b0_full_v1.json').read_text())
    t02=json.loads(Path('reports/t02_b0_full_v1/fixed_plan.json').read_text())
    config=copy.deepcopy(base)
    config.update(protocol='T03-B0-Full-online20-v1',initialization='from_scratch_same_seed_no_T02_weights')
    config['sampling']=dict(train_pairs=7912,val_pairs=952,mode='fresh_online_per_presentation',seed_namespace='T03-online-v1')
    config['training'].update(epochs=20,total_steps=39560,num_workers=4)
    config['validation']=dict(epochs=[1,2,4,6,8,10,12,14,16,18,20],pairs=952,batch_size=1,
        best_metric='success_rate@5px',best_direction='maximize',tie_rule='keep_earlier',
        test_used=False,all_affines_saved=True)
    config['comparison']=dict(only_variable='training.learning_rate',main=.001,control=.0002,
        sample_order='torch randperm(seed+epoch)',augmentation_token='epoch*7912+position_in_epoch')
    config['notes']=['Same B0-Full architecture, distribution, losses, batch and evaluator as T02.',
        '20 epochs are an observation budget, not a convergence claim.','No test images or scene controls used for model selection.']
    paths=dict(train='reports/metadata_finalize_v1/train_256_dedup_v1.jsonl',val='reports/data_geometry_v1/dataset_manifest.jsonl')
    plan=dict(protocol=config['protocol'],source_hashes={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in paths.items()},train=[],val=[])
    known={p['pair_id']:p for p in t02['samples'] if p['split']=='val'}
    for split,path in paths.items():
        rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        rows=sorted([p for p in rows if p['dataset']=='OSdataset' and p['resolution']==256 and p['split']==split],key=lambda p:int(p['pair_id'].rsplit('/',1)[1]))
        if len(rows)!=(7912 if split=='train' else 952): raise ValueError('Full split count mismatch')
        for p in rows:
            sample=dict(pair_id=p['pair_id'],optical_path=p['optical']['path'],sar_path=p['sar']['path'],
                optical_sha256=p['optical']['sha256'],sar_sha256=p['sar']['sha256'])
            if split=='val':
                if p['pair_id'] in known:
                    sample.update(A_forward=known[p['pair_id']]['A_forward'],parameters=known[p['pair_id']]['parameters'],transform_origin='T02_existing_fixed_val')
                else:
                    seed=int.from_bytes(hashlib.sha256(f"{base['seed']}/{p['pair_id']}/T03-fixed-val-v1".encode()).digest()[:8],'little')
                    matrix,parameters=sample_transform(np.random.default_rng(seed),base)
                    sample.update(A_forward=matrix.tolist(),parameters=parameters,augmentation_seed=str(seed),transform_origin='T03_fixed_val_v1')
            plan[split].append(sample)
    if len({(p['optical_sha256'],p['sar_sha256']) for p in plan['train']})!=7912: raise ValueError('Training content duplicates')
    content={'full_data_plan.json':json.dumps(plan,indent=2)+'\n','main_config.json':json.dumps(config,indent=2)+'\n'}
    control=copy.deepcopy(config);control['training']['learning_rate']=.0002
    content['lr2e4_config.json']=json.dumps(control,indent=2)+'\n'
    for name,text in content.items():
        path=out/name
        if path.exists() and path.read_text()!=text: raise FileExistsError('Different frozen T03 setup')
        path.write_text(text)
    print(json.dumps(dict(train=7912,val=952,test_images_loaded=0,epochs=20,steps=39560,T02_val_matrices_preserved=16)))


if __name__=='__main__':main()
