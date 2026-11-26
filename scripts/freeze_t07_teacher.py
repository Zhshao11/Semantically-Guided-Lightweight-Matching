"""Freeze once after recorded train-only visual audit; no threshold search."""
import hashlib,json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def main():
    folder=Path('reports/t07_teacher_v1');target=folder/'teacher_freeze.json'
    if target.exists():raise FileExistsError('Frozen teacher must not be overwritten')
    candidate=read(folder/'TG_official_freeze.json');review=read('runs/t07_teacher_review_v1/summary.json');visual=read(folder/'OS_visual_decision.json');source=read(folder/'source_mask_audit.json');license=read(folder/'initialization_license.json')
    tr,tg=[read('runs/t07_teacher_'+name+'_v1/summary.json') for name in ['TR','TG']]
    for key in ['steps','initial_state_sha256','presentations_sha256','cropped_GT_stream_sha256','GT_counts']:assert tr[key]==tg[key]
    assert tr['steps']==3000 and not tg['community_weights_used'] and not tr['community_weights_used']
    assert review['checkpoint_sha256']==candidate['checkpoint_sha256'] and visual['checkpoint_sha256']==candidate['checkpoint_sha256']
    assert visual['all_four_pages_inspected'] and not visual['human_semantic_GT']
    assert not candidate['source_class_collapse'] and not candidate['near_absent_source_classes'],'Do not silently freeze an absent class'
    ids={r['pair_id'] for r in review['records']};assert len(ids)==32 and all('/train/' in x for x in ids)
    assert set(visual['quarantine_image_ids'])<=ids
    candidate.update(technical_decision='freeze_for_research_with_explicit_abstention',license_research_use_confirmed=True,license_audit=license,
        checkpoint_path='runs/t07_teacher_TG_v1/best.pt',teacher_code=dict(base_commit='e208fc66d33272e02fae1d65feb952c0b265f3f2',files_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['scripts/train_t07_teacher.py','scripts/review_t07_teacher.py','sglm/semantic_targets.py']}),
        OS32_train_review=review,OS_visual_decision=visual,source_mask_audit=source,abstention=dict(image_ids=visual['quarantine_image_ids'],disabled_dominant_classes=visual['class_abstention']['names'],disabled_class_reason=visual['class_abstention']['reason'],automatic='whole image if predicted Background>=95% AND mean raw maximum probability>=90%, unchanged T06 diagnostic flag',local_rule='full valid coverage; retained mass>=.8; conditional confidence>=.6; no threshold increase',weight='mask * retained mass * (1-H(conditional7)/log7)',guarantee='trusted weak supervision candidates only; no guarantee all high-confidence errors excluded; no OS semantic GT'),
        source='official ImageNet B2 encoder + newly initialized8-channel decoder; full LoveDA Train2522 gray; no community tensors',deployment_authorization_claimed=False,weights_published=False)
    target.write_text(json.dumps(candidate,indent=2));print(json.dumps(dict(frozen=True,step=candidate['selected_step'],sha256=candidate['checkpoint_sha256'],visual_abstention=len(visual['quarantine_image_ids']))))


if __name__=='__main__':main()
