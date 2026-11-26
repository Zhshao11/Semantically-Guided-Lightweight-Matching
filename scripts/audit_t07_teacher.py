"""Technical freeze record, class distributions and explicit private-license boundary."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='runs/t06_teacher_TG');ap.add_argument('--tag',default='TG_community');a=ap.parse_args()
    out=Path('reports/t07_teacher_v1');out.mkdir(exist_ok=True)
    root=Path(a.run);summary=json.loads((root/'summary.json').read_text());step=summary['best']['step'];checkpoint=root/'best.pt'
    validation=json.loads((root/f'validation_{step:04d}.json').read_text());manifest=json.loads(Path('runs/t06_teacher_data/manifest.json').read_text())
    preflight=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text());rows=[]
    for key,m in validation['groups'].items():
        for c in m['per_class']:
            rows.append(dict(teacher=a.tag,input_group=key,**c,predicted_fraction_on_valid_GT=c['predicted_pixels']/m['valid_GT_pixels'],GT_fraction=c['GT_pixels']/m['valid_GT_pixels']))
    write_csv(out/(a.tag+'_class_audit.csv'),rows)
    explicit=a.tag!='TG_community'
    record=dict(candidate=a.tag,checkpoint_path=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),selected_step=step,
        code_base='e208fc66d33272e02fae1d65feb952c0b265f3f2',training_source_sha256=hashlib.sha256(Path('scripts/train_t06_teacher.py' if not explicit else 'scripts/train_t07_teacher.py').read_bytes()).hexdigest(),
        preprocessing=preflight['preprocessing'],native_gray='OS PNG L uint8; explicitly L->RGB; no recoloring; identical ImageNet processor',source_classes=preflight['source_classes'],target_classes=preflight['target_classes'],class_mapping=preflight['class_mapping'],
        ignore=dict(GT0='masked as255 in training; never a land-cover class',student='retain a=1-pIgnore; conditional7; weight=mask*a*(1-H/log7)'),aggregation=preflight['aggregation'],
        selection='best new64 gray pooled seven-class mIoU; earlier on ties; both input versions/old8 reported; OS not used for checkpoint choice',
        validation64=manifest['val64'],validation64_manifest_sha256=hashlib.sha256(json.dumps(manifest['val64'],sort_keys=True).encode()).hexdigest(),
        class_audit=rows,source_class_collapse=any(c['predicted_pixels']==0 for c in validation['groups']['fixed64/gray_RGB']['per_class']),
        near_absent_source_classes=[c['name'] for c in validation['groups']['fixed64/gray_RGB']['per_class'] if c['predicted_pixels']/max(c['GT_pixels'],1)<.05],
        license='explicit research/evaluation only' if explicit else 'unconfirmed research conditions; historical technical candidate only',
        use_for_T07_student_labels=explicit,weight_release=False,deployment_authorization_claimed=False)
    if not explicit:
        review=json.loads(Path('runs/t06_OS_review_v1/summary.json').read_text());r=[x for x in review['records'] if x['version']=='TG'];record['OS32_fixed_train_records']=r
        record['OS_status']='1/32 fixed all-background flag; fine rooftops/other structures omitted; no OS semantic GT; old candidate excluded from new student labels on license grounds'
    (out/(a.tag+'_freeze.json')).write_text(json.dumps(record,indent=2));print(json.dumps(dict(tag=a.tag,step=step,sha=record['checkpoint_sha256'],source_collapse=record['source_class_collapse'])))


if __name__=='__main__':main()
