"""Freeze paired-run identities and verified validationparent membership before results."""
import hashlib,json
from pathlib import Path
from collections import Counter
from sglm.experiment_io import write_csv


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=Path('reports/t09_audit_v1');out.mkdir(exist_ok=True)
    if (out/'run_index.json').exists():raise FileExistsError('Frozen run index exists')
    p=json.loads(Path('configs/t09_protocol_v1.json').read_text());runs=[]
    for seed in p['seeds']:
        for group in p['groups']:
            reused=seed==20261126 and group in ['M','MS'];root=f'runs/t08_{group}_v1' if reused else f'runs/t09_seed{seed}_{group}_v1'
            runs.append(dict(seed=seed,group=group,root=root,reused_T08=reused,primary_checkpoint=root+'/inference_last.pt',validation=root+'/K256/validation_epoch_004.json'))
    manifest={r['pair_id']:r for r in [json.loads(x) for x in Path('reports/metadata_finalize_v1/dataset_manifest_parent_status_v1.jsonl').read_text().splitlines()]};plan=json.loads(Path('reports/t05_frozen_v1/validation_plan.json').read_text())['samples'];rows=[]
    for sample in plan:
        uid=sample['pair_id'];r=manifest[uid];parent=r['parent_mapping'];rows.append(dict(pair_id=uid,parent_status=parent['status'],parent_pair_id=parent.get('parent_pair_id'),crop_x=parent.get('crop_xy',[None,None])[0],crop_y=parent.get('crop_xy',[None,None])[1],evidence='exact_decoded_pixels_both_modalities' if parent['status']=='verified' else 'not_verified',satellite_scene_status=r['scene_source_mapping']['status']))
    assert len(rows)==952 and all(r['parent_status']=='verified' for r in rows)
    counts=Counter(r['parent_pair_id'] for r in rows);write_csv(out/'validation_parent_groups.csv',rows)
    membership=dict(validation_pairs=952,verified512parent_groups=len(counts),parent_child_counts=dict(sorted(counts.items())),unresolved_pairs=0,satellite_scene_provenance='unresolved; knownparentcorrelation only',manifest_sha256=sha('reports/metadata_finalize_v1/dataset_manifest_parent_status_v1.jsonl'),validation_plan_sha256=sha('reports/t05_frozen_v1/validation_plan.json'))
    (out/'validation_parent_groups.json').write_text(json.dumps(membership,indent=2));(out/'run_index.json').write_text(json.dumps(dict(protocol=p,runs=runs,primary_metric='SR@5px',primary_checkpoint='last epoch4',all_K=256),indent=2))
    protect=['runs/t08_shuffle_v1/'+n for n in ['donor_flat_index.npy','conditional7.npy','weight.npy','summary.json']]+[f'runs/t08_{group}_v1/{name}' for group in ['M','MS','MS-shuffle'] for name in ['summary.json','inference_last.pt','last.pt']]+['scripts/train_t08.py','scripts/prepare_t08_shuffle.py']
    (out/'T08_immutable_snapshot.json').write_text(json.dumps({name:sha(name) for name in protect},indent=2));print('Index frozen9pairedslots,7newruns,parents',len(counts))


if __name__=='__main__':main()
