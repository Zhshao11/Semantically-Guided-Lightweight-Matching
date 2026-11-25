"""Repeat-read all fixed inputs and audit positive-label validity without changing them."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sglm.t02.data import AffinePairs256
from sglm.t02.loss import correspondence_labels


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    torch.set_num_threads(2)
    records=[]
    for split in ('train','val'):
        dataset=AffinePairs256(a.root,a.plan,split)
        for index in range(len(dataset)):
            first,second=dataset[index],dataset[index]
            dataset.set_epoch(123)
            third=dataset[index]
            for key in ('optical','sar_aug','A_forward','optical_valid_mask','sar_aug_valid_mask'):
                assert torch.equal(first[key],second[key]) and torch.equal(first[key],third[key])
            labels,target,positive=correspondence_labels(first['A_forward'][None],first['optical_valid_mask'][None],first['sar_aug_valid_mask'][None])
            outside=((target<0)|(target>255)).any(-1)
            assert not (positive&outside).any()
            cells=torch.floor((target+.5)/8).long().clamp(0,31)
            valid_cells=torch.nn.functional.avg_pool2d(first['sar_aug_valid_mask'][None,None].float(),8,8)[0,0]>.999
            target_supported=valid_cells[cells[0,:,1],cells[0,:,0]]
            assert not (positive[0]&~target_supported).any()
            assert first['sar_aug'].min()>=0 and first['sar_aug'].max()<=1
            records.append(dict(pair_id=first['pair_id'],split=split,repeated_reads_and_epoch_change_bitwise_equal=True,
                positive_queries=int(positive.sum()),out_of_bounds_queries=int(outside.sum()),
                out_of_bounds_positive_queries=0,unsupported_target_positive_queries=0,
                sar_aug_tensor_sha256=hashlib.sha256(first['sar_aug'].numpy().tobytes()).hexdigest(),
                A_forward_tensor_sha256=hashlib.sha256(first['A_forward'].numpy().tobytes()).hexdigest()))
    result=dict(passed=True,pairs_checked=len(records),records=records,sar_transform='uint8/255; no logarithm',
        mask_meaning='optical frame all valid; SAR mask is valid inverse-sampling support, not intensity!=0')
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=True,pairs_checked=len(records))))


if __name__=='__main__':
    main()
