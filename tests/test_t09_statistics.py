import numpy as np
import pytest
from sglm.t09_statistics import metric_values,parent_sample_indices,paired_cluster_bootstrap


def test_parent_sampling_keeps_siblings_and_duplicate_draws():
    parents=['a','a','a','b','b','c'];idx=parent_sample_indices(parents,[1,1,0])
    assert idx.tolist()==[3,4,3,4,0,1,2]


def test_failures_stay_in_SR_denominator_and_errors_use_solved():
    result=metric_values(np.array([[0.,2.,8.,np.nan]]))[0]
    assert result[0]==.5 and result[1]==.5 and result[2]==2 and np.isclose(result[3],6.8)


def test_bootstrap_is_paired_identical_methods_have_exact_zero_CI():
    errors=np.array([[1.,4.,6.,8.],[2.,3.,5.,9.]])
    stats=paired_cluster_bootstrap(errors,errors,['a','a','b','b'],replicates=50)
    assert not stats['point'].any() and not stats['low'].any() and not stats['high'].any()


def test_repeating_seed_does_not_shrink_cluster_uncertainty():
    ref=np.array([[1.,6.,8.,9.]]);alt=np.array([[2.,3.,4.,10.]])
    one=paired_cluster_bootstrap(ref,alt,['a','a','b','b'],replicates=100,seed=7)
    three=paired_cluster_bootstrap(np.repeat(ref,3,0),np.repeat(alt,3,0),['a','a','b','b'],replicates=100,seed=7)
    assert np.array_equal(one['samples'][:,-1],three['samples'][:,-1])
