import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip('transformers')
from scripts.train_t06_teacher import Crops


def test_paired_teacher_crops_keep_same_annotation_and_ignore_no_data(tmp_path):
    image=np.zeros((1024,1024,3),np.uint8);image[:]=[200,10,40]
    labels=np.full((1024,1024),3,np.uint8);labels[:128]=0;labels[200:300]=7
    Image.fromarray(image).save(tmp_path/'image.png');Image.fromarray(labels).save(tmp_path/'labels.png')
    manifest=dict(train=[dict(pair_id='Rural/1',files=dict(image=dict(path='image.png'),label=dict(path='labels.png')))])
    plan=[dict(pair_id='Rural/1',crop_xy=[0,0])]
    class Processor:
        def __call__(self,images,return_tensors):
            return dict(pixel_values=torch.from_numpy(np.asarray(images).copy()).permute(2,0,1)[None].float()/255)
    rgb=Crops(tmp_path,manifest,plan,Processor(),False)[0];gray=Crops(tmp_path,manifest,plan,Processor(),True)[0]
    assert torch.equal(rgb[1],gray[1]) and rgb[2]==gray[2] and torch.equal(rgb[3],gray[3])
    assert 255 in rgb[1].unique() and 0 not in rgb[1].unique() and rgb[3][0]>0
    assert not torch.equal(rgb[0],gray[0]) and torch.equal(gray[0][0],gray[0][1])
