"""Frozen-cohort descriptive standardization; no model or label generation."""
import numpy as np


def common_support(random_ids, random_scores, hard_scores):
    valid = (np.asarray(random_ids) >= 0) & np.isfinite(random_scores)
    rows = valid.any(axis=2).all(axis=1) & np.isfinite(hard_scores)
    return rows, valid


def cell_measures(cells, values, weights, units, n_units, n_cells=49):
    key = np.asarray(units) * n_cells + np.asarray(cells)
    shape = (n_units, n_cells)
    mass = np.bincount(key, weights=weights, minlength=n_units*n_cells).reshape(shape)
    total = np.bincount(key, weights=weights*values, minlength=n_units*n_cells).reshape(shape)
    return mass, total


def safe_mean(total, mass):
    return np.divide(total, mass, out=np.full_like(total, np.nan, dtype=float), where=mass>0)


def fixed_standardization(hmass, htotal, rmass, rtotal, weights):
    hm = safe_mean(htotal, hmass)
    rm = safe_mean(rtotal, rmass)
    selected = np.asarray(weights)>0
    complete = np.all((hmass[..., selected]>0) & (rmass[..., selected]>0), axis=-1)
    value = np.sum((hm[..., selected]-rm[..., selected])*weights[selected], axis=-1)
    return np.where(complete, value, np.nan)


def symmetric_decomposition(hmass, htotal, rmass, rtotal):
    ph = hmass/hmass.sum(); pr = rmass/rmass.sum()
    mh = safe_mean(htotal, hmass); mr = safe_mean(rtotal, rmass)
    both = (hmass>0)&(rmass>0)
    within = np.sum((ph[both]+pr[both])/2*(mh[both]-mr[both]))
    composition = np.sum((mh[both]+mr[both])/2*(ph[both]-pr[both]))
    unilateral = np.sum(htotal[~both])/hmass.sum()-np.sum(rtotal[~both])/rmass.sum()
    raw = htotal.sum()/hmass.sum()-rtotal.sum()/rmass.sum()
    return dict(raw=float(raw), within=float(within), composition=float(composition),
                unilateral=float(unilateral), identity_error=float(raw-within-composition-unilateral))


def weighted_distribution(values, weights, quantiles, thresholds):
    order = np.argsort(values, kind='stable'); v=values[order]; w=weights[order]
    cdf=np.cumsum(w)/w.sum()
    result={f'P{int(q*100)}':float(v[min(np.searchsorted(cdf,q),len(v)-1)]) for q in quantiles}
    result.update({f'fraction_ge_{t}':float(weights[values>=t].sum()/weights.sum()) for t in thresholds})
    result['mean']=float(np.sum(values*weights)/weights.sum())
    return result
