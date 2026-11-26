"""Predeclared hierarchical semantic-enrichment comparisons, trainonly, no tuning."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.t10_signal import paired_parent_summary,hierarchical_relation_weights
from sglm.experiment_io import write_csv


def nanmean_rows(a):
    valid=np.isfinite(a);n=valid.sum(1);result=np.full(len(a),np.nan);np.divide(np.where(valid,a,0).sum(1),n,out=result,where=n>0);return result


def main():
    p=json.loads(Path('configs/t10a_protocol_v1.json').read_text());plan=json.loads(Path(p['samples']['plan']).read_text())['samples'];d=dict(np.load('runs/t10a_candidates_v1/relations.npz'));coverage=json.loads(Path('runs/t10a_candidates_v1/coverage.json').read_text());out=Path('reports/t10_results_v1');out.mkdir(exist_ok=True)
    parent_names=sorted({r['verified512_parent'] for r in plan});lookup={s:i for i,s in enumerate(parent_names)};codes=np.array([lookup[r['verified512_parent']] for r in plan]);pair=d['pair_index'];query=d['query_index'];N=len(pair);hard=d['hard_semantic'];permhard=d['hard_perm_semantic'];comparisons={};random_means=[];random_perm_means=[]
    for j,name in enumerate(['uncontrolled','distance_weight','distance_weight_class']):
        random_means.append(nanmean_rows(d['random_semantic'][:,j]));random_perm_means.append(nanmean_rows(d['random_perm_semantic'][:,j]));comparisons['hard_minus_'+name]=hard-random_means[-1]
    # Nonlinearinterpolation can change argmax even when gridnode classes preserved.
    stable_hard=d['hard_perm_class']==d['hard_class'];stable_random=d['random_perm_class'][:,2]==d['random_class'][:,2];rperm=np.where(stable_random,d['random_perm_semantic'][:,2],np.nan);rtrue=np.where(stable_random,d['random_semantic'][:,2],np.nan);stable_rtrue=nanmean_rows(rtrue);stable_rperm=nanmean_rows(rperm)
    null=permhard-stable_rperm;true=hard-stable_rtrue;eligible=stable_hard&np.isfinite(stable_rtrue)&np.isfinite(stable_rperm)
    comparisons['classmatched_true_minus_permuted_enrichment']=np.where(eligible,true-null,np.nan)
    comparisons['hard_minus_classconditioned_permuted_hard']=np.where(eligible,hard-permhard,np.nan)
    comparisons['classmatched_permuted_hard_minus_permuted_random']=np.where(eligible,null,np.nan)
    rows=[];pair_rows=[];parent_rows=[];class_rows=[];distribution=[];balance=[];class_matrix=[];leaveout=[];perclass_stats={};bootstrap=p['analysis']['bootstrap_replicates']
    # Maincomparisons are equalweight slots/query/image/parent, never assume millionsindependent.
    for name,values in comparisons.items():
        summary,pm,par,qcount=paired_parent_summary(pair,query,values,codes,bootstrap,p['analysis']['bootstrap_seed']);rows.append(dict(comparison=name,**summary,positive_pair_fraction=float((pm[np.isfinite(pm)]>0).mean()) if np.isfinite(pm).any() else None,positive_parent_fraction=float((par[np.isfinite(par)]>0).mean()) if np.isfinite(par).any() else None))
        for i,item in enumerate(plan):pair_rows.append(dict(comparison=name,pair_index=i,pair_id=item['pair_id'],verified512_parent=item['verified512_parent'],mean_delta=float(pm[i]) if np.isfinite(pm[i]) else None,queries=int(qcount[i]),defined=bool(np.isfinite(pm[i]))))
        for i,parent in enumerate(parent_names):parent_rows.append(dict(comparison=name,verified512_parent=parent,mean_delta=float(par[i]) if np.isfinite(par[i]) else None,defined=bool(np.isfinite(par[i]))))
    names=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text())['target_classes']
    for c,class_name in enumerate(names):
        selector=d['query_class']==c
        for name in ['hard_minus_uncontrolled','hard_minus_distance_weight','hard_minus_distance_weight_class','classmatched_true_minus_permuted_enrichment']:
            values=np.where(selector,comparisons[name],np.nan);summary,_,_,_=paired_parent_summary(pair,query,values,codes,bootstrap,p['analysis']['bootstrap_seed']);class_rows.append(dict(query_pseudoclass=class_name,comparison=name,**summary));perclass_stats[(class_name,name)]=summary
        for cc,candidate_name in enumerate(names):
            selected=selector&(d['hard_class']==cc);values=np.where(selected,comparisons['hard_minus_distance_weight_class'],np.nan);summary,_,_,_=paired_parent_summary(pair,query,values,codes,bootstrap,p['analysis']['bootstrap_seed']);class_matrix.append(dict(query_pseudoclass=class_name,candidate_pseudoclass=candidate_name,**summary))
    for c,label in enumerate(names):
        if not (d['query_class']==c).any():continue
        for name in ['hard_minus_distance_weight','hard_minus_distance_weight_class']:
            vals=np.where(d['query_class']!=c,comparisons[name],np.nan);summary,_,_,_=paired_parent_summary(pair,query,vals,codes,bootstrap,p['analysis']['bootstrap_seed']);leaveout.append(dict(excluded_query_pseudoclass=label,comparison=name,**summary))
    # Distributions compare exactlysame relationcohort with availablecontrols at eachlevel.
    for j,control in enumerate(['uncontrolled','distance_weight','distance_weight_class']):
        valid=np.isfinite(random_means[j]);wi=hierarchical_relation_weights(pair[valid],query[valid],codes);h=hard[valid];r=random_means[j][valid];ph=permhard[valid];pr=random_perm_means[j][valid]
        for name,value in [('hard',h),('random_mean5',r),('candidate_permuted_hard',ph),('candidate_permuted_random_mean5',pr)]:
            hist,edges=np.histogram(value,bins=np.linspace(0,1.000001,51),weights=wi);cdf=np.cumsum(wi[np.argsort(value)]);order=np.argsort(value);quantiles=[float(value[order[min(np.searchsorted(cdf,q),len(order)-1)]]) for q in [.1,.5,.9]] if len(value) else [None]*3
            distribution.append(dict(control_cohort=control,relation_type=name,hierarchical_mean=float(np.sum(value*wi)),P10=quantiles[0],P50=quantiles[1],P90=quantiles[2],relations=int(len(value)),hist_probability=hist.tolist(),bin_edges=edges.tolist()))
        for kind,array in [('random_individual',d['random_semantic'][:,j]),('candidate_permuted_random_individual',d['random_perm_semantic'][:,j])]:
            subset=array[valid];isvalid=np.isfinite(subset);ww=np.broadcast_to(wi[:,None],subset.shape)/np.maximum(isvalid.sum(1,keepdims=True),1);value=subset[isvalid];individual_w=ww[isvalid]
            hist,edges=np.histogram(value,bins=np.linspace(0,1.000001,51),weights=individual_w);order=np.argsort(value);cdf=np.cumsum(individual_w[order]);quantiles=[float(value[order[min(np.searchsorted(cdf,q),len(order)-1)]]) for q in [.1,.5,.9]] if len(value) else [None]*3
            distribution.append(dict(control_cohort=control,relation_type=kind,hierarchical_mean=float(np.sum(value*individual_w)),P10=quantiles[0],P50=quantiles[1],P90=quantiles[2],relations=int(len(value)),hist_probability=hist.tolist(),bin_edges=edges.tolist()))
        rv=d['random_id'][:,j]>=0
        rdist=nanmean_rows(np.where(rv,d['random_distance'][:,j],np.nan));rweight=nanmean_rows(np.where(rv,d['random_weight'][:,j],np.nan));rdescriptor=nanmean_rows(np.where(rv,d['random_descriptor'][:,j],np.nan));samerate=nanmean_rows(np.where(rv,d['random_class'][:,j]==d['query_class'][:,None],np.nan))
        balance.append(dict(control=control,relations_with_controls=int(valid.sum()),relation_coverage=float(valid.mean()),hard_distance_mean=float(np.sum(d['hard_distance'][valid]*wi)),random_distance_mean=float(np.sum(rdist[valid]*wi)),mean_abs_distance_gap=float(np.sum(np.abs(d['hard_distance'][valid]-rdist[valid])*wi)),hard_weight_mean=float(np.sum(d['hard_weight'][valid]*wi)),random_weight_mean=float(np.sum(rweight[valid]*wi)),mean_abs_weight_gap=float(np.sum(np.abs(d['hard_weight'][valid]-rweight[valid])*wi)),hard_descriptor_mean=float(np.sum(d['hard_descriptor'][valid]*wi)),random_descriptor_mean=float(np.sum(rdescriptor[valid]*wi)),hard_same_query_class_fraction=float(np.sum((d['hard_class'][valid]==d['query_class'][valid])*wi)),random_same_query_class_fraction=float(np.sum(samerate[valid]*wi))))
    # Every controlindex andweight is saved; no matchfallback to looser strata.
    counts={key:int(sum(r[key] for r in coverage)) for key in coverage[0] if isinstance(coverage[0][key],int) and key!='pair_index'}
    write_csv(out/'comparison_summary.csv',rows);write_csv(out/'paired_image_differences.csv',pair_rows);write_csv(out/'paired_parent_differences.csv',parent_rows);write_csv(out/'per_query_class.csv',class_rows);write_csv(out/'query_candidate_class_matrix.csv',class_matrix);write_csv(out/'leave_one_class_out.csv',leaveout);write_csv(out/'control_balance.csv',balance)
    write_csv(out/'similarity_distribution.csv',[{key:value for key,value in row.items() if key not in ['hist_probability','bin_edges']} for row in distribution]);hist_rows=[]
    for r in distribution:
        for i,mass in enumerate(r['hist_probability']):hist_rows.append(dict(control_cohort=r['control_cohort'],relation_type=r['relation_type'],bin_low=r['bin_edges'][i],bin_high=r['bin_edges'][i+1],probability=mass))
    write_csv(out/'similarity_histograms.csv',hist_rows)
    primary=next(r for r in rows if r['comparison']=='hard_minus_distance_weight');strict=next(r for r in rows if r['comparison']=='hard_minus_distance_weight_class');support=[]
    for class_name in names[1:]:
        a=perclass_stats[(class_name,'hard_minus_distance_weight_class')]
        if a['valid_parents']>=16 and a['CI95_low'] is not None and a['CI95_low']>0:support.append(class_name)
    eligible_signal=primary['CI95_low'] is not None and primary['CI95_low']>0 and strict['CI95_low'] is not None and strict['CI95_low']>0 and len(support)>=2
    decision=dict(predeclared_gate_positive=eligible_signal,nonbackground_classes_with_strict_support=support,primary=primary,strict=strict,full_training_started=False,next_step='no training inT10A; candidatehypothesis only, not validatedobjective' if eligible_signal else 'do not proceed to fulltraining; revisehypothesis; noforcednewloss',teacher_is_not_GT=True)
    result=dict(protocol=p,comparison=rows,classes=class_rows,controls=balance,distribution=distribution,coverage=counts,paired_parent_bootstrap='128verifiedparents, no satellite scene independence; 20000pairedresamples; query/image/parent means',permutation_sampled_class_stable_relations=int(eligible.sum()),permutation_total_reliable_hard_relations=N,decision=decision,relations_sha256=hashlib.sha256(Path('runs/t10a_candidates_v1/relations.npz').read_bytes()).hexdigest());(out/'signal_summary.json').write_text(json.dumps(result,indent=2));(out/'decision.json').write_text(json.dumps(decision,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,control in zip(axes,['uncontrolled','distance_weight','distance_weight_class']):
        for name,color in [('hard','tab:red'),('random_individual','tab:blue'),('candidate_permuted_hard','tab:orange')]:
            r=next(x for x in distribution if x['control_cohort']==control and x['relation_type']==name);axesx=np.asarray(r['bin_edges']);ax.plot((axesx[:-1]+axesx[1:])/2,np.cumsum(r['hist_probability']),label=name,color=color)
        ax.set(xlabel='Soft semantic cosine',ylabel='Hierarchical weighted CDF',title=control);ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'semantic_similarity_CDF.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(comparison=rows,decision=decision),indent=2))


if __name__=='__main__':main()
