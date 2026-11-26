"""Audit saved T10-A candidates only; freeze common support before comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv
from sglm.t10_signal import hierarchical_relation_weights
from sglm.t10_composition import (common_support, cell_measures, safe_mean,
    fixed_standardization, symmetric_decomposition, weighted_distribution)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    v=np.asarray(values); valid=np.isfinite(v)
    return dict(CI95_low=float(np.quantile(v[valid],.025)) if valid.any() else None,
                CI95_high=float(np.quantile(v[valid],.975)) if valid.any() else None,
                defined_bootstraps=int(valid.sum()), undefined_bootstraps=int((~valid).sum()))


def counts(cells, target, pairs, queries, parents, relations):
    s=cells==target
    return dict(individuals=int(s.sum()), hard_relations=int(len(np.unique(relations[s]))),
                queries=int(len(np.unique(pairs[s]*1024+queries[s]))),
                pairs=int(len(np.unique(pairs[s]))),parents=int(len(np.unique(parents[s]))))


def main():
    config_path=Path('configs/t10a_common_support_v1.json'); p=json.loads(config_path.read_text())
    out=Path(p['output']); out.mkdir(parents=True,exist_ok=True)
    private=Path(p['private_output']); private.mkdir(parents=True,exist_ok=True)
    root=Path(p['candidates']); meta=json.loads((root/'summary.json').read_text())
    original=json.loads(Path(p['source_protocol']).read_text())
    assert sha(root/'relations.npz')==meta['relations_sha256']
    assert sha(p['plan'])==meta['plan_sha256']
    assert sha(original['matcher_checkpoint'])==original['matcher_sha256']==meta['checkpoint_sha256']
    for entry in meta['candidate_files']:
        assert sha(root/entry['file'])==entry['sha256']
    plan=json.loads(Path(p['plan']).read_text())['samples']
    parent_names=sorted({r['verified512_parent'] for r in plan})
    codes=np.array([parent_names.index(r['verified512_parent']) for r in plan])
    names=['Background','Building','Road','Water','Barren','Forest','Agricultural']
    d=dict(np.load(root/'relations.npz')); keep,valid=common_support(d['random_id'],d['random_semantic'],d['hard_semantic'])
    # Freeze IDs first, before computing any class means or similarities.
    rows=np.flatnonzero(keep); pair=d['pair_index'][keep].astype(int); query=d['query_index'][keep].astype(int)
    keys=pair*1024+query; qkeys=np.unique(keys)
    np.savez_compressed(private/'support.npz',relation_rows=rows,query_keys=qkeys)
    freeze=dict(protocol_sha256=sha(config_path),relations_sha256=sha(root/'relations.npz'),
                plan_sha256=sha(p['plan']),checkpoint_sha256=meta['checkpoint_sha256'],
                candidate_summary_sha256=sha(root/'summary.json'),support_sha256=sha(private/'support.npz'),
                query_keys_sha256=hashlib.sha256(qkeys.astype('<i8').tobytes()).hexdigest(),
                relation_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
                common_queries=len(qkeys),common_relations=len(rows),pairs=256,parents=128,
                no_candidate_changes=True,no_training=True,no_teacher_inference=True)
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    weight=hierarchical_relation_weights(pair,query,codes)
    active_parent_count=len(np.unique(codes[pair])); np.testing.assert_allclose(weight.sum(),1)
    qclass=d['query_class'][keep].astype(int); hclass=d['hard_class'][keep].astype(int)
    hard=dict(cell=qclass*7+hclass,value=d['hard_semantic'][keep].astype(float),weight=weight,
              pair=pair,query=query,parent=codes[pair],relation=rows,candidate_class=hclass,query_class=qclass)
    groups={'hard':hard}
    for j,name in enumerate(p['controls']):
        v=valid[keep,j]; ri,ci=np.nonzero(v); n=v.sum(1)
        groups[name]=dict(cell=qclass[ri]*7+d['random_class'][keep,j][v],
            value=d['random_semantic'][keep,j][v].astype(float),weight=weight[ri]/n[ri],
            pair=pair[ri],query=query[ri],parent=codes[pair[ri]],relation=rows[ri],
            candidate_class=d['random_class'][keep,j][v],query_class=qclass[ri])
    for g in groups.values():
        g['parent_mass'],g['parent_total']=cell_measures(g['cell'],g['value'],g['weight']*active_parent_count,g['parent'],128)
        g['pair_mass'],g['pair_total']=cell_measures(g['cell'],g['value'],g['weight'],g['pair'],256)
        g['mass']=g['parent_mass'].sum(0)/active_parent_count
        g['total']=g['parent_total'].sum(0)/active_parent_count
    bilateral=np.logical_and.reduce([g['mass']>0 for g in groups.values()])
    pooled=(hard['mass']+groups[p['primary_control']]['mass'])/2
    class_weights=np.where(bilateral,pooled,0); class_weights/=class_weights.sum()
    weight_rows=[]
    for c in range(49):
        r=dict(query_pseudoclass=names[c//7],candidate_pseudoclass=names[c%7],
               fixed_weight=float(class_weights[c]),pooled_primary_mass=float(pooled[c]),all_views_supported=bool(bilateral[c]))
        for name,g in groups.items():r[name+'_mass']=float(g['mass'][c])
        weight_rows.append(r)
    write_csv(out/'fixed_class_weights.csv',weight_rows)
    rng=np.random.default_rng(p['bootstrap']['seed']); R=p['bootstrap']['replicates']
    draws=rng.integers(0,128,(R,128))
    multiplicity=np.zeros((R,128),float)
    np.add.at(multiplicity,(np.repeat(np.arange(R),128),draws.ravel()),1)
    draw_hash=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    for g in groups.values():
        g['boot_mass']=multiplicity@g['parent_mass'];g['boot_total']=multiplicity@g['parent_total']
    strata=[]; comparisons=[]; pair_rows=[]; parent_rows=[]; composition=[]; query_results=[]; distributions=[]
    hm,ht=hard['mass'],hard['total']; hb=hard['boot_mass']; hbt=hard['boot_total']
    for name in p['controls']:
        g=groups[name];rm,rt=g['mass'],g['total'];rb=g['boot_mass'];rbt=g['boot_total']
        bdelta=hbt.sum(1)/hb.sum(1)-rbt.sum(1)/rb.sum(1)
        standardized=fixed_standardization(hm,ht,rm,rt,class_weights)
        bs=fixed_standardization(hb,hbt,rb,rbt,class_weights)
        dec=symmetric_decomposition(hm,ht,rm,rt)
        assert abs(dec['identity_error'])<1e-12
        comparisons.append(dict(control=name,raw_delta=dec['raw'],**{'raw_'+k:v for k,v in interval(bdelta).items()},
            standardized_delta=float(standardized),**{'standardized_'+k:v for k,v in interval(bs).items()},
            within_component=dec['within'],composition_component=dec['composition'],unilateral_component=dec['unilateral'],
            identity_error=dec['identity_error'],hard_excluded_cell_mass=float(hm[~bilateral].sum()),random_excluded_cell_mass=float(rm[~bilateral].sum())))
        for c in range(49):
            mh=safe_mean(ht,hm)[c]; mr=safe_mean(rt,rm)[c]
            delta=safe_mean(hbt,hb)[:,c]-safe_mean(rbt,rb)[:,c]
            record=dict(control=name,query_pseudoclass=names[c//7],candidate_pseudoclass=names[c%7],
                        hard_mean=float(mh),random_mean=float(mr),delta=float(mh-mr),fixed_weight=float(class_weights[c]),
                        hard_mass=float(hm[c]),random_mass=float(rm[c]),**interval(delta))
            for prefix,x in [('hard',hard),('random',g)]:
                record.update({prefix+'_'+k:v for k,v in counts(x['cell'],c,x['pair'],x['query'],x['parent'],x['relation']).items()})
            strata.append(record)
        for unit,labels,collection in [('pair',plan,pair_rows),('parent',parent_names,parent_rows)]:
            for i,label in enumerate(labels):
                ah=hard[unit+'_mass'][i]; bh=hard[unit+'_total'][i]; ar=g[unit+'_mass'][i]; br=g[unit+'_total'][i]
                hmean=bh.sum()/ah.sum() if ah.sum()>0 else np.nan;rmean=br.sum()/ar.sum() if ar.sum()>0 else np.nan
                available=(ah>0)&(ar>0)&(class_weights>0)
                collection.append(dict(control=name,index=i,
                    id=label['pair_id'] if unit=='pair' else label,
                    verified512_parent=label['verified512_parent'] if unit=='pair' else label,
                    hard_mean=float(hmean),random_mean=float(rmean),raw_delta=float(hmean-rmean),
                    common_queries=int(len(np.unique(query[pair==i]))) if unit=='pair' else int(len(np.unique(keys[codes[pair]==i]))),
                    fixed_weight_mass_available=float(class_weights[available].sum()),
                    fixed_standardized_delta=float(fixed_standardization(ah,bh,ar,br,class_weights)),
                    fixed_weight_partial_contribution=float(np.sum(class_weights[available]*(safe_mean(bh,ah)[available]-safe_mean(br,ar)[available])))))
        for qc in range(7):
            sl=slice(qc*7,(qc+1)*7);a=hm[sl].sum();b=rm[sl].sum()
            x=ht[sl].sum()/a if a else np.nan;y=rt[sl].sum()/b if b else np.nan
            bd=safe_mean(hbt[:,sl].sum(1),hb[:,sl].sum(1))-safe_mean(rbt[:,sl].sum(1),rb[:,sl].sum(1))
            query_results.append(dict(control=name,query_pseudoclass=names[qc],common_queries=int(len(np.unique(keys[qclass==qc]))),
                hierarchical_query_mass=float(a),hard_mean=float(x),random_mean=float(y),delta=float(x-y),
                hard_same_class_rate=float(hm[qc*7+qc]/a) if a else np.nan,
                random_same_class_rate=float(rm[qc*7+qc]/b) if b else np.nan,**interval(bd)))
    for name,g in groups.items():
        for qc in [-1,*range(7)]:
            select=np.ones(len(g['value']),bool) if qc==-1 else g['query_class']==qc
            denom=g['weight'][select].sum()
            for cc in range(7):
                mask=select&(g['candidate_class']==cc)
                composition.append(dict(group=name,query_pseudoclass='ALL' if qc==-1 else names[qc],candidate_pseudoclass=names[cc],
                    weighted_share=float(g['weight'][mask].sum()/denom) if denom else np.nan,
                    individuals=int(mask.sum()),same_class_rate=float(g['weight'][select&(g['query_class']==g['candidate_class'])].sum()/denom) if denom else np.nan))
        distributions.append(dict(group=name,individuals=len(g['value']),**weighted_distribution(g['value'],g['weight'],p['quantiles'],p['near_one_thresholds'])))
    # All1024 queries: masks are diagnostic only, never used to regenerate controls.
    coverage_rows=[];class_coverage=[]; stage_names=['all_queries','geometry_valid','reliable_label','geometry_and_label','reliable_hard','common_support']
    class_counts=np.zeros((6,7),int)
    for i,item in enumerate(plan):
        a=dict(np.load(root/f'pair_{i:03d}.npz')); cls=a['query_probability7'].argmax(1)
        hardq=np.isin(np.arange(1024),d['query_index'][d['pair_index']==i]); commonq=np.isin(np.arange(1024),query[pair==i])
        masks=[np.ones(1024,bool),a['query_geometry_valid'],a['query_semantic_valid'],a['query_geometry_valid']&a['query_semantic_valid'],hardq,commonq]
        record=dict(pair_id=item['pair_id'],verified512_parent=item['verified512_parent'],
                    original_relations=int((d['pair_index']==i).sum()),common_relations=int((pair==i).sum()))
        for k,(stage,mask) in enumerate(zip(stage_names,masks)):
            record[stage]=int(mask.sum());class_counts[k]+=np.bincount(cls[mask],minlength=7)
        record['hard_queries_excluded_for_control_support']=int(hardq.sum()-commonq.sum());coverage_rows.append(record)
    for k,stage in enumerate(stage_names):
        for c in range(7):class_coverage.append(dict(stage=stage,query_pseudoclass=names[c],queries=int(class_counts[k,c]),
            fraction_of_stage=float(class_counts[k,c]/class_counts[k].sum()),
            fraction_of_all_queries_of_this_pseudoclass=float(class_counts[k,c]/max(class_counts[0,c],1)),
            note='unreliable-label argmax is diagnostic only; not an accepted pseudolabel' if k<2 else 'external teacher pseudo-class; not GT'))
    for filename,data in [('comparisons.csv',comparisons),('class_composition.csv',composition),('query_class_results.csv',query_results),
        ('joint_class_results.csv',strata),('image_results.csv',pair_rows),('parent_results.csv',parent_rows),('similarity_distribution.csv',distributions),
        ('coverage_by_pair.csv',coverage_rows),('query_class_coverage.csv',class_coverage)]:
        # Undefined quantities are emptyCSV cells; no nonstandard NaN JSON.
        clean=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()} for row in data]
        write_csv(out/filename,clean)
    result=dict(freeze=freeze,bootstrap_draws_sha256=draw_hash,comparisons=comparisons,distributions=distributions,
        coverage={stage:int(class_counts[k].sum()) for k,stage in enumerate(stage_names)},
        reliable_hard_relations=len(keep),excluded_relations=int((~keep).sum()),
        represented_pairs=int(len(np.unique(pair))),represented_parents=active_parent_count,
        positive_weight_cells=int((class_weights>0).sum()),total_cells=49,
        pooled_primary_unmatched_mass=float(pooled[~bilateral].sum()),
        support_arrays_private=True,weights_fixed_across_views_and_bootstraps=True,
        teacher_pseudoclasses_are_GT=False,parent_groups_are_satellite_scenes=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Verify source bytes again after analysis; no model execution anywhere in this script.
    assert sha(root/'relations.npz')==freeze['relations_sha256']
    assert sha(p['plan'])==freeze['plan_sha256']
    assert sha(original['matcher_checkpoint'])==freeze['checkpoint_sha256']
    (out/'acceptance.json').write_text(json.dumps(dict(passed=True,**freeze,
        decomposition_identity_max_error=max(abs(r['identity_error']) for r in comparisons),
        fixed_weight_sum=float(class_weights.sum()),bootstrap_draws_sha256=draw_hash),indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
