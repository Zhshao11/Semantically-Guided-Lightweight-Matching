"""Freeze T08 data/diagnostic/annotation plans before fine-tuning."""
import hashlib,json
from pathlib import Path
import numpy as np


def main():
    out=Path('reports/t08_setup_v1');out.mkdir(exist_ok=True)
    if (out/'setup.json').exists():raise FileExistsError('Frozen setup exists')
    protocol=json.loads(Path('configs/t08_protocol_v1.json').read_text());plan=json.loads(Path('reports/t06_setup_v1/training_plan.json').read_text());samples=plan['samples'];ids=[r['pair_id'] for r in samples]
    old_review={r['pair_id'] for r in json.loads(Path('reports/t06_setup_v1/OS_train_review.json').read_text())['samples']};old_probe=set(json.loads(Path('reports/t06_setup_v1/setup.json').read_text())['probe_indices'])
    rng=np.random.default_rng(protocol['seed']);eligible=[i for i in range(7912) if ids[i] not in old_review and i not in old_probe]
    annotation=sorted(rng.choice(eligible,16,replace=False).tolist());remaining=[i for i in range(7912) if i not in annotation];diagnostic=sorted(rng.choice([i for i in remaining if ids[i] not in old_review],128,replace=False).tolist())
    reserve=[samples[i] for i in annotation];(out/'annotation_plan.json').write_text(json.dumps(dict(split='train',samples=reserve,labels_exist=False,annotation_gt='pending independent human labels; never substitute teacher/assistant predictions',label_ids='0 unannotated/ignore;1Background2Building3Road4Water5Barren6Forest7Agricultural',old_T07_student_saw_images=True,excluded_from_new_T08_finetuning=False,independent_of_old32_selection=True),indent=2))
    # OS semanticGT does not exist; optional16 review is not a fine-tuning holdout.
    training=dict(config=plan['config'],samples=samples);(out/'training_plan.json').write_text(json.dumps(training,separators=(',',':')))
    (out/'diagnostic_plan.json').write_text(json.dumps(dict(config=plan['config'],samples=[samples[i] for i in diagnostic]),separators=(',',':')))
    setup=dict(protocol=protocol,train_pairs=7912,annotation_reserved=0,train_original_indices=list(range(7912)),annotation_original_indices=annotation,diagnostic_original_indices=diagnostic,all_groups_same=True,preflight_annotation_status='no OS semanticGT; waived as prerequisite by explicit user clarification',plan_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['annotation_plan.json','training_plan.json','diagnostic_plan.json']})
    (out/'setup.json').write_text(json.dumps(setup,indent=2));print('Frozen7912training,16optionalreview,128routing diagnostic')


if __name__=='__main__':main()
