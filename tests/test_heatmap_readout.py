import numpy as np
import pytest
import torch
from sglm.heatmap_readout import read_frozen_heatmap,describe_frozen_heatmap,grid_delta


def packet(p,valid=None,center=(35.5,51.5)):
    p=torch.tensor(p,dtype=torch.float32).reshape(1,-1);side=int(p.shape[1]**.5);valid=torch.ones_like(p,dtype=torch.bool) if valid is None else torch.tensor(valid,dtype=torch.bool).reshape_as(p)
    centers=torch.tensor([center]);positions=centers[:,None]+2*grid_delta(side,'cpu')[None]
    return p,valid,centers,positions


def test_two_separated_peaks_whole_mean_is_pulled_and_local_readout_uses_fine_grid():
    p=np.zeros((7,7));p[3,1]=.6;p[3,5]=.4;args=packet(p);r=read_frozen_heatmap(*args)
    assert torch.allclose(r['A'],args[2]+torch.tensor([[-.8,0]]),atol=1e-5)
    assert torch.equal(r['B'],args[2]+torch.tensor([[-4.,0]])) and torch.equal(r['B'],r['C'])
    assert r['local_mask'].sum()==9 and abs(r['local_mass'].item()-.6)<1e-6


def test_local_subpixel_mean_can_differ_from_max_without_expanding_window():
    p=np.zeros((7,7));p[3,3]=.6;p[3,4]=.3;p[0,0]=.1;args=packet(p);r=read_frozen_heatmap(*args)
    assert torch.equal(r['B'],args[2]);assert torch.allclose(r['C'],args[2]+torch.tensor([[2/3,0]]),atol=1e-5)
    assert not r['local_mask'][0,0] and torch.equal(args[0],torch.tensor(p,dtype=torch.float32).reshape(1,49))


def test_boundary_three_by_three_clips_and_invalid_tokens_never_enter_expectation():
    p=np.zeros((7,7));p[0,0]=.7;p[0,1]=.2;p[1,0]=.1;args=packet(p);r=read_frozen_heatmap(*args);assert r['local_mask'].sum()==4
    valid=np.ones((7,7),bool);valid[0,1]=False;p[0,0]=.9;p[0,1]=0
    args=packet(p,valid);r=read_frozen_heatmap(*args);assert r['local_mask'].sum()==3 and not r['local_mask'][0,1]
    assert r['C'][0,0]==args[2][0,0]-6


def test_peak_ties_are_first_valid_row_major_and_not_toroidal():
    p=np.zeros((7,7));p[0,6]=.5;p[1,0]=.5;r=read_frozen_heatmap(*packet(p));assert r['peak_index'].item()==6
    assert not r['local_mask'][0,7]


def test_empty_is_preserved_invalid_all_window_is_rejected_not_recovered():
    p=torch.empty(0,49);valid=torch.empty(0,49,dtype=torch.bool);c=torch.empty(0,2);pos=torch.empty(0,49,2)
    assert read_frozen_heatmap(p,valid,c,pos)['C'].shape==(0,2)
    args=packet(np.zeros((7,7)),np.zeros((7,7),bool))
    with pytest.raises(ValueError):read_frozen_heatmap(*args)


def test_posthoc_truth_mass_and_competing_peak_do_not_change_readout():
    p=np.zeros((7,7));p[3,1]=.6;p[3,5]=.4;args=packet(p);before=read_frozen_heatmap(*args)
    truth=args[2].numpy()+[4.,0];d=describe_frozen_heatmap(args[0].numpy(),args[1].numpy(),args[3].numpy(),truth,before['peak_index'].numpy())
    assert abs(d['truth_mass'][0]-.4)<1e-6 and abs(d['secondary_ratio'][0]-2/3)<1e-6 and d['competing'][0]
    assert d['peak_GT_distance'][0]==8 and not d['peak_inside_truth3x3'][0]
    after=read_frozen_heatmap(*args)
    for key in ['A','B','C']:assert torch.equal(before[key],after[key])


def test_flat_probability_is_reported_as_plateau_not_fabricated_secondary_peak():
    args=packet(np.ones((7,7))/49);r=read_frozen_heatmap(*args);d=describe_frozen_heatmap(args[0].numpy(),args[1].numpy(),args[3].numpy(),args[2].numpy(),r['peak_index'].numpy())
    assert d['max_plateau_size'][0]==49 and not d['competing'][0] and d['secondary_index'][0]==-1
