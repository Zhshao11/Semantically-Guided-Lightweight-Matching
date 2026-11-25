"""Freeze candidate, input-only subsets and actual architecture before evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from sglm.t02.model import B0Full


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reports/t05_frozen_v1')
    ap.add_argument('--weights-output',default='runs/t05_frozen_v1/checkpoint.pt');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists():raise FileExistsError('Frozen protocol already exists; do not overwrite')
    protocol=json.loads(Path('configs/t05_protocol_v1.json').read_text());source=Path(protocol['checkpoint_source'])
    saved=torch.load(source,map_location='cpu',weights_only=False);cfg=saved['config']
    assert saved['epoch']==18 and saved['step']==35604 and saved['model_name']=='B0-Full'
    assert cfg==json.loads(Path('runs/t03_main_lr1e3_v1/config.json').read_text())
    env=json.loads(Path('runs/t03_main_lr1e3_v1/environment.json').read_text())
    planpath=Path('reports/t03_setup_v1/full_data_plan.json');plan=json.loads(planpath.read_text())
    assert digest(planpath)==saved['plan_sha256']==env['plan_sha256'] and len(plan['val'])==952
    target=Path(a.weights_output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert digest(source)==digest(target)
    original_revision=env['code_revision'];code_checks={}
    for path in ['sglm/model.py','sglm/t02/model.py','sglm/t02/loss.py','sglm/evaluation.py','sglm/geometry.py']:
        old=subprocess.check_output(['git','show',f'{original_revision}:{path}'])
        current=Path(path).read_bytes();assert old==current,f'Original B0/evaluator code changed: {path}'
        code_checks[path]=dict(sha256=hashlib.sha256(current).hexdigest(),equal_to_training_revision=True)
    manifest={r['pair_id']:r for r in map(json.loads,Path('reports/metadata_finalize_v1/dataset_manifest_parent_status_v1.jsonl').read_text().splitlines())
              if r['resolution']==256 and r['split']=='val'}
    ids=[r['pair_id'] for r in plan['val']];means=np.array([manifest[uid]['optical']['mean'][0] for uid in ids])
    stds=np.array([manifest[uid]['optical']['std'][0] for uid in ids])
    brightness=means>=np.median(means);contrast=np.searchsorted(np.quantile(stds,[.25,.5,.75]),stds,side='right')
    strata=brightness.astype(int)*4+contrast
    rng=np.random.default_rng(protocol['seed']);pools={i:rng.permutation(np.flatnonzero(strata==i)).tolist() for i in range(8)}
    parents={i:manifest[uid]['parent_mapping']['parent_pair_id'] for i,uid in enumerate(ids)}
    def choose(count,initial=(),unique_parents=False):
        picked=list(initial);used={parents[i] for i in picked}
        while len(picked)<count:
            progressed=False
            for s in range(8):
                eligible=[i for i in pools[s] if i not in picked and (not unique_parents or parents[i] not in used)]
                if eligible:
                    i=eligible[0];picked.append(i);used.add(parents[i]);progressed=True
                    if len(picked)==count:break
            if not progressed:raise ValueError('Insufficient content strata/parents')
        return sorted(picked)
    timing=choose(protocol['timing_pairs'],unique_parents=True)
    diagnostics=choose(protocol['diagnostic_pairs'],initial=timing)
    subset=dict(selection_before_predictions=True,uses_GT_or_prior_registration_results=False,
        optical_mean_median=float(np.median(means)),optical_std_quartiles=np.quantile(stds,[.25,.5,.75]).tolist(),
        diagnostic_indices=diagnostics,timing_indices=timing,
        timing_distinct_verified_512_parents=len({parents[i] for i in timing}),
        source_scene_provenance='unresolved; distributed 512 parent and input strata are not original satellite scenes',
        records=[dict(index=i,pair_id=uid,stratum=int(strata[i]),optical_mean=float(means[i]),optical_std=float(stds[i]),
            verified_parent_pair_id=parents[i],diagnostic=i in diagnostics,timing=i in timing) for i,uid in enumerate(ids)])
    (out/'subsets.json').write_text(json.dumps(subset,indent=2))
    fixed=dict(config=cfg,samples=[dict(r,split='val') for r in plan['val']]);(out/'validation_plan.json').write_text(json.dumps(fixed,indent=2))
    (out/'candidate_config.json').write_text(json.dumps(cfg,indent=2))
    # Actual loaded architecture, not a parameter estimate from a diagram.
    torch.set_num_threads(4);model=B0Full(cfg).eval();model.load_state_dict(saved['model'],strict=True)
    modules=dict(model.named_modules());params=dict(model.named_parameters())
    components=['backbone.adapters.0','backbone.adapters.1',*[f'backbone.stages.{i}' for i in range(3)],
        *[f'backbone.down.{i}' for i in range(2)],*[f'backbone.lateral.{i}' for i in range(3)],
        *[f'backbone.up.{i}' for i in range(2)],*[f'backbone.fuse.{i}' for i in range(2)],
        'backbone.descriptor','local.project','local.log_temperature','log_temperature','no_match']
    counts={name:sum(p.numel() for p in modules[name].parameters()) if name in modules else params[name].numel() for name in components}
    total=sum(p.numel() for p in model.parameters());assert sum(counts.values())==total==37219
    leaves=[dict(name=name,type=type(m).__name__,structure=m.extra_repr(),direct_parameters=sum(p.numel() for p in m.parameters(recurse=False)))
        for name,m in model.named_modules() if not list(m.children())]
    args=[torch.zeros(1,3,256,256),torch.zeros(1,1,256,256),torch.ones(1,256,256,dtype=torch.bool),torch.ones(1,256,256,dtype=torch.bool)]
    with torch.inference_mode():output=model(*args)
    shapes={k:list(v.shape) for k,v in output.items() if torch.is_tensor(v)}
    assert shapes['optical_coarse']==[1,64,32,32] and shapes['optical_fine']==[1,16,128,128]
    assert shapes['optical_descriptors']==[1,1024,32] and shapes['coarse_correlation']==[1,1024,1024]
    architecture=dict(total_parameters=total,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        components_nonoverlapping=counts,leaves=leaves,actual_shapes=shapes,
        parameter_shapes={name:list(p.shape) for name,p in model.named_parameters()},
        buffers_numel=sum(b.numel() for b in model.buffers()),
        note='Shared backbone convolution counted once, both modality BN affine parameter sets counted; running stats are buffers, not parameters',
        repr=str(model),B0_disabled=cfg['disabled'])
    (out/'architecture.json').write_text(json.dumps(architecture,indent=2))
    (out/'actual_layers.txt').write_text(str(model)+'\n')
    freeze=dict(protocol=protocol,frozen_unix=time.time(),candidate_epoch=saved['epoch'],candidate_step=saved['step'],
        source_checkpoint=str(source),frozen_checkpoint=str(target),checkpoint_sha256=digest(target),
        checkpoint_best_record=saved['best'],candidate_config_sha256=digest(out/'candidate_config.json'),
        validation_plan_sha256=digest(out/'validation_plan.json'),subsets_sha256=digest(out/'subsets.json'),
        training_code_revision=original_revision,preparation_base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        original_code_verification=code_checks,source_plan_sha256=digest(planpath),
        original_training_environment=env,test_used=False,training_performed=False)
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps({k:v for k,v in freeze.items() if k not in ('protocol','original_training_environment','original_code_verification')},indent=2))


if __name__=='__main__':main()
