import copy
import hashlib
import numpy as np
import torch
import cv2
from sglm.t02.data import load_t02_config,AffinePairs256,sample_transform
from sglm.t03.data import FullAffinePairs256,PresentationBatchSampler,seed_for_presentation


def test_online_presentation_schedule_unique_and_LR_independent():
    first=PresentationBatchSampler(13,4,123);second=PresentationBatchSampler(13,4,123)
    for epoch in (0,1,19):
        first.epoch=second.epoch=epoch
        a,b=list(first),list(second)
        assert a==b and len(a)==4
        assert len({i for batch in a for i,_ in batch})==13
        assert {token for batch in a for _,token in batch}==set(range(epoch*13,(epoch+1)*13))
    assert seed_for_presentation(1,'pair',0)!=seed_for_presentation(1,'pair',1)


def test_new_read_online_and_fixed_validation_match_T02(tmp_path):
    root=tmp_path/'OSDataset2.0';folder=root/'Patch-level Subset/OSdataset/256/train';folder.mkdir(parents=True)
    pixels=np.tile(np.arange(256,dtype=np.uint8),(256,1))
    for name in ('opt1.png','sar1.png'):cv2.imwrite(str(folder/name),pixels)
    config=load_t02_config()
    matrix,_=sample_transform(np.random.default_rng(8),config)
    sample=dict(pair_id='osdataset/256/train/1',optical_path=str((folder/'opt1.png').relative_to(root)),
        sar_path=str((folder/'sar1.png').relative_to(root)),optical_sha256=hashlib.sha256((folder/'opt1.png').read_bytes()).hexdigest(),
        sar_sha256=hashlib.sha256((folder/'sar1.png').read_bytes()).hexdigest(),A_forward=matrix.tolist())
    plan=dict(train=[sample],val=[sample])
    online=FullAffinePairs256(root,plan,'train',config)
    assert not torch.equal(online[0]['A_forward'],online[0]['A_forward'])
    assert torch.equal(online[(0,7)]['sar_aug'],online[(0,7)]['sar_aug'])
    fixed=FullAffinePairs256(root,plan,'val',config)
    old_plan=dict(config=config,samples=[dict(sample,split='val')])
    old=AffinePairs256(root,old_plan,'val')
    for key in ('optical','sar_aug','A_forward','optical_valid_mask','sar_aug_valid_mask'):
        assert torch.equal(fixed[0][key],old[0][key])
