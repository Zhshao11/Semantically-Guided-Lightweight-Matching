"""Check full split, fixed val compatibility, per-read online draws and LR-only control."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from sglm.t02.data import AffinePairs256
from sglm.t02.model import B0Full
from sglm.t03.data import FullAffinePairs256,PresentationBatchSampler
from sglm.t03.runner import state_digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--directory',required=True)
    a=ap.parse_args();p=Path(a.directory)
    main=json.loads((p/'main_config.json').read_text());control=json.loads((p/'lr2e4_config.json').read_text())
    comparison=copy.deepcopy(control);comparison['training']['learning_rate']=main['training']['learning_rate']
    assert comparison==main
    base=json.loads(Path('configs/t02_b0_full_v1.json').read_text())
    assert all(main[key]==base[key] for key in ('augmentation','model','evaluation','disabled'))
    plan=json.loads((p/'full_data_plan.json').read_text())
    assert len(plan['train'])==7912 and len(plan['val'])==952
    val=FullAffinePairs256(a.root,plan,'val',main)
    old=AffinePairs256(a.root,'reports/t02_b0_full_v1/fixed_plan.json','val')
    lookup={s['pair_id']:i for i,s in enumerate(val.samples)}
    for item in old:
        new=val[lookup[item['pair_id']]]
        assert all(torch.equal(item[k],new[k]) for k in ('optical','sar_aug','A_forward','optical_valid_mask','sar_aug_valid_mask'))
    training=FullAffinePairs256(a.root,plan,'train',main)
    first=training[(0,0)];again=training[(0,0)];fresh=training[(0,1)]
    assert torch.equal(first['sar_aug'],again['sar_aug']) and not torch.equal(first['A_forward'],fresh['A_forward'])
    for epoch in (0,1,19):
        samplers=[PresentationBatchSampler(7912,4,c['seed']) for c in (main,control)]
        for sampler in samplers: sampler.epoch=epoch
        batches=[list(s) for s in samplers]
        assert batches[0]==batches[1] and len(batches[0])==1978
        assert len({token for batch in batches[0] for _,token in batch})==7912
    torch.manual_seed(main['seed']);m=B0Full(main)
    initial=state_digest(m)
    torch.manual_seed(control['seed']);c=B0Full(control)
    assert initial==state_digest(c)
    t02=torch.load('runs/t02_b0_full_v1/training/initial.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(t02['model']);assert initial==state_digest(m)
    result=dict(passed=True,train_pairs=7912,val_pairs=952,test_images_loaded=0,
        full_plan_sha256=hashlib.sha256((p/'full_data_plan.json').read_bytes()).hexdigest(),
        T02_model=base['model'],T02_evaluation=base['evaluation'],T02_val_inputs_exact=16,
        online_read_new_token_changes_transform=True,same_token_replays_exact=True,
        main_control_only_LR_differs=True,matched_sample_schedule_checked_epochs=[1,2,20],
        from_scratch_initial_state_sha256=initial,main_control_and_T02_initial_weights_equal=True,
        from_T02_trained_weights=False,validation_epochs=main['validation']['epochs'],
        best_metric=main['validation']['best_metric'])
    (p/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
