import numpy as np
from sglm.t10_composition import common_support, cell_measures, fixed_standardization, symmetric_decomposition, weighted_distribution


def test_common_support_requires_same_hard_rows_across_views():
    ids=np.ones((3,3,2),int); scores=np.ones_like(ids,float)
    ids[1,2]=-1; scores[2,1]=np.nan
    keep,_=common_support(ids,scores,np.ones(3))
    assert keep.tolist()==[True,False,False]


def test_control_repeats_split_one_relation_weight():
    mass,total=cell_measures(np.array([0,0,1]),np.array([.8,.8,.2]),np.array([.25,.25,.5]),np.zeros(3,int),1,2)
    assert np.allclose(mass,[[.5,.5]]) and np.allclose(total,[[.4,.1]])


def test_fixed_weights_remove_composition_only_signal_and_missing_not_zero():
    hm=np.array([.9,.1]); rm=np.array([.1,.9]); means=np.array([.9,.2])
    dec=symmetric_decomposition(hm,hm*means,rm,rm*means)
    assert abs(dec['within'])<1e-12 and abs(dec['raw']-.56)<1e-12
    assert abs(dec['identity_error'])<1e-12
    assert abs(fixed_standardization(hm,hm*means,rm,rm*means,np.array([.5,.5])))<1e-12
    assert np.isnan(fixed_standardization(hm,hm*means,np.array([1.,0.]),np.array([.9,0]),np.array([.5,.5])))


def test_decomposition_keeps_unilateral_mass_and_negative_residual():
    hm=np.array([.4,.6,0]); rm=np.array([.7,0,.3]); ht=np.array([.2,.54,0]); rt=np.array([.42,0,.03])
    d=symmetric_decomposition(hm,ht,rm,rt)
    assert d['within']<0 and np.isclose(d['raw'],d['within']+d['composition']+d['unilateral'])


def test_quantile_and_saturation_weights_are_not_individual_counts():
    r=weighted_distribution(np.array([.2,1.,1.]),np.array([.8,.1,.1]),[.5,.9],[.99])
    assert r['P50']==.2 and r['P90']==1 and np.isclose(r['fraction_ge_0.99'],.2)
