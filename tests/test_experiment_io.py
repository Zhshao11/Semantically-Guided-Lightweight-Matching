import json
import pytest
from sglm.experiment_io import complete_jsonl,flatten_diagnostic,require_exclusive_gpu,fixed_timing_plan


def test_live_log_ignores_only_incomplete_last_record(tmp_path):
    path=tmp_path/'live.jsonl';path.write_text('{"epoch":1}\n{"epoch":')
    assert complete_jsonl(path)==[dict(epoch=1)]
    path.write_text('{broken}\n{"epoch":2}\n')
    with pytest.raises(json.JSONDecodeError):complete_jsonl(path)


def test_formal_timing_rejects_other_process_and_unknown_device():
    uuid='GPU-abc'
    assert require_exclusive_gpu(uuid,'GPU-abc, 10, python\n',10,[uuid])
    assert require_exclusive_gpu(uuid,'GPU-other, 11, train\n',10,[uuid])
    with pytest.raises(RuntimeError):require_exclusive_gpu(uuid,'GPU-abc, 11, train\n',10,[uuid])
    with pytest.raises(RuntimeError):require_exclusive_gpu('cuda:6','',10,[uuid])
    with pytest.raises(RuntimeError):require_exclusive_gpu(uuid,'[Not Supported]',10,[uuid])


def test_layered_export_preserves_failures_and_separates_levels():
    record=dict(pair_id='train/1',group='B',variant_index=3,coarse=dict(predicted_correspondences=0),
        evaluation=dict(status='failed',rmse_px=None),geometry=dict(estimation=dict(status='insufficient_matches',matrix=None)))
    row=flatten_diagnostic(record)
    assert row['evaluation.status']=='failed' and row['evaluation.rmse_px'] is None
    assert row['coarse.predicted_correspondences']==0
    assert row['geometry.estimation.status']=='insufficient_matches'
    assert row['group']=='B' and row['variant_index']==3


def test_timing_uses_saved_validation_transform_without_sampling():
    matrix=[[1,0,2],[0,1,3]]
    plan=dict(train=[dict(pair_id='train/1')],val=[dict(pair_id='val/1',A_forward=matrix)])
    fixed=fixed_timing_plan(plan,{},'val')
    assert fixed['samples'][0]['A_forward']==matrix and fixed['samples'][0]['split']=='val'
    assert 'split' not in plan['val'][0]
    with pytest.raises(ValueError):fixed_timing_plan(plan,{},'train')
