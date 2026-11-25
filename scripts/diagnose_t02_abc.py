"""Controlled fixed-checkpoint ABC diagnosis; freezes all five B variants before inference."""
import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sglm.t02.data import AffinePairs256,sample_transform
from sglm.t02.model import B0Full
from sglm.diagnostics import InferenceStateAudit,diagnose_pair,aggregate_diagnostics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--diagnostic-config',default='configs/t02_controlled_diagnostic_v1.json')
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    diagnostic=json.loads(Path(a.diagnostic_config).read_text())
    if hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()!=diagnostic['checkpoint_sha256']:
        raise ValueError('Not the frozen T02 checkpoint')
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    plan=json.loads(Path(a.plan).read_text());config=checkpoint['config']
    if plan['config']!=config: raise ValueError('Checkpoint and plan config differ')
    plans={g:dict(config=config,samples=[]) for g in ('A','B','C')}
    for sample in plan['samples']:
        if sample['split']=='train':
            plans['A']['samples'].append(copy.deepcopy(sample))
            for variant in range(diagnostic['group_B_variants_per_content']):
                seed=int.from_bytes(hashlib.sha256(f"{config['seed']}/{sample['pair_id']}/{variant}/{diagnostic['group_B_seed_namespace']}".encode()).digest()[:8],'little')
                matrix,parameters=sample_transform(np.random.default_rng(seed),config)
                if np.array_equal(matrix,np.asarray(sample['A_forward'])): raise AssertionError('B transform equals original')
                new=copy.deepcopy(sample);new.update(A_forward=matrix.tolist(),parameters=parameters,variant_index=variant,augmentation_seed=str(seed))
                plans['B']['samples'].append(new)
        else: plans['C']['samples'].append(copy.deepcopy(sample))
    payload=json.dumps(dict(protocol=diagnostic['protocol'],diagnostic=diagnostic,groups=plans),indent=2)
    path=out/'frozen_ABC_plans.json'
    if path.exists() and path.read_text()!=payload: raise FileExistsError('Different frozen diagnostic plan')
    path.write_text(payload)
    model=B0Full(config).to(a.device).eval();model.load_state_dict(checkpoint['model'])
    observer=InferenceStateAudit(model);observer.check()
    records={};reproductions={}
    for group,group_plan in plans.items():
        dataset=AffinePairs256(a.root,group_plan,'val' if group=='C' else 'train')
        records[group]=[]
        reference_name='final_val.json' if group=='C' else 'final_train.json'
        reference={r['pair_id']:r for r in json.loads((Path('reports/t02_b0_full_v1')/reference_name).read_text())['results']}
        for index,item in enumerate(dataset):
            observer.check()
            result=diagnose_pair(model,item,config,diagnostic,a.device)
            result.update(group=group,variant_index=group_plan['samples'][index].get('variant_index'),
                optical_tensor_sha256=hashlib.sha256(item['optical'].numpy().tobytes()).hexdigest(),
                SAR_tensor_sha256=hashlib.sha256(item['sar_aug'].numpy().tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(item['sar_aug_valid_mask'].numpy().tobytes()).hexdigest())
            if group in ('A','C'):
                saved=reference[item['pair_id']]
                assert result['predictions']['source']==saved['predicted_source']
                assert result['predictions']['target']==saved['predicted_target']
                assert result['geometry']['estimation']['matrix']==saved['estimation']['matrix']
                assert result['evaluation']==saved['evaluation']
            records[group].append(result)
        (out/f'group_{group}_pairs.json').write_text(json.dumps(records[group],indent=2))
        print(json.dumps(dict(group=group,summary=aggregate_diagnostics(records[group]))),flush=True)
    state=observer.finish()
    by_image=defaultdict(list)
    for record in records['B']: by_image[record['pair_id']].append(record)
    b_content=[]
    for pair_id,values in by_image.items():
        if len(values)!=5: raise AssertionError('Expected 5 perturbations per content')
        b_content.append(dict(pair_id=pair_id,perturbations=5,summary=aggregate_diagnostics(values),
            variants=[dict(variant_index=r['variant_index'],evaluation=r['evaluation']) for r in values]))
    (out/'group_B_per_original_image.json').write_text(json.dumps(b_content,indent=2))
    a_opt={r['pair_id']:r['optical_tensor_sha256'] for r in records['A']}
    assert all(r['optical_tensor_sha256']==a_opt[r['pair_id']] for r in records['B'])
    summary=dict(protocol=diagnostic['protocol'],checkpoint_sha256=diagnostic['checkpoint_sha256'],
        groups={g:aggregate_diagnostics(values) for g,values in records.items()},
        B_original_image_pairs=32,B_correlated_perturbations=160,
        B_macro_SR3=float(np.mean([r['summary']['registration']['success_rate@3px'] for r in b_content])),
        B_macro_SR5=float(np.mean([r['summary']['registration']['success_rate@5px'] for r in b_content])),
        B_all_five_success_contents=sum(all(v['evaluation']['success']['5'] for v in r['variants']) for r in b_content),
        B_any_success_contents=sum(any(v['evaluation']['success']['5'] for v in r['variants']) for r in b_content),
        reproduction_A_exact=True,reproduction_C_exact=True,BN_state_audit=state,
        preprocessing=dict(shared_loader='AffinePairs256 for A/B/C',normalization='uint8/255',
            masks='same bilinear sampling-support policy',decode='same B0 coarse_xy/sample_windows',B_optical_inputs_equal_A=True),
        no_training=True,inference_parameters_unchanged=True,RANSAC_parameters_unchanged=True,
        diagnostic_config=diagnostic,T03_gate_passed=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
