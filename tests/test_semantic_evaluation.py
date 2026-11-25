import numpy as np
import pytest
from sglm.semantic_evaluation import loveda_confusion,loveda_metrics


def test_GT_zero_is_excluded_but_predicted_zero_is_penalized():
    gt=np.array([[0,1,2],[3,4,5],[6,7,7]],np.uint8);pred=gt.copy();pred[0,0]=7;pred[0,1]=0
    c=loveda_confusion(gt,pred);m=loveda_metrics(c)
    assert c.sum()==8 and c[0].sum()==0 and c[1,0]==1
    assert m['overall_accuracy']==7/8 and m['per_class'][0]['IoU']==0
    assert m['predicted_Ignore_fraction_on_valid_GT']==1/8
    assert m['mean_IoU']==pytest.approx(6/7)


def test_loveda_rejects_shifted_or_255_labels_before_scoring():
    with pytest.raises(ValueError):loveda_confusion(np.array([[255]],np.uint8),np.array([[0]],np.uint8))
    with pytest.raises(ValueError):loveda_metrics(np.ones((8,8),int))
