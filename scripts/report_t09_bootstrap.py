"""Three requested T09 tables: seedresults, paired transitions, parentcluster bootstrap."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sglm.t09_statistics import paired_cluster_bootstrap,metric_values
from sglm.experiment_io import write_csv


def read(p):return json.loads(Path(p).read_text())


def main():
    index=read('reports/t09_audit_v1/run_index.json');p=index['protocol'];out=Path('reports/t09_delivery_v1');out.mkdir(exist_ok=True)
    membership=list(csv.DictReader(Path('reports/t09_audit_v1/validation_parent_groups.csv').open()));ids=[r['pair_id'] for r in membership];parents=[r['parent_pair_id'] for r in membership];assert len(ids)==952 and all(r['parent_status']=='verified' for r in membership)
    tables=[];errors={};successes={};source_hashes={}
    for slot in index['runs']:
        seed,group=slot['seed'],slot['group'];data=read(slot['validation']);records={r['pair_id']:r for r in data['pairs']};assert set(records)==set(ids)
        err=np.array([records[uid]['evaluation']['rmse_px'] if records[uid]['evaluation']['solved'] else np.nan for uid in ids]);errors[(seed,group)]=err;successes[(seed,group)]=np.array([records[uid]['evaluation']['success']['5'] for uid in ids]);assert np.array_equal(successes[(seed,group)],err<5)
        metrics=metric_values(err[None])[0];tables.append(dict(row_type='seed',seed=seed,group=group,SR5=metrics[0],SR3=metrics[1],RMSE_median_px=metrics[2],RMSE_P90_px=metrics[3],solver_failures=int(np.isnan(err).sum()),threshold5_failures=int((~successes[(seed,group)]).sum()),image_pairs=952,checkpoint='last epoch4',reused_T08=slot['reused_T08']))
        source_hashes[slot['primary_checkpoint']]=hashlib.sha256(Path(slot['primary_checkpoint']).read_bytes()).hexdigest()
    for group in p['groups']:
        rows=[r for r in tables if r['row_type']=='seed' and r['group']==group]
        for label,func in [('mean',lambda a:np.mean(a)),('sample_std_ddof1',lambda a:np.std(a,ddof=1))]:
            tables.append(dict(row_type=label,seed='all3',group=group,**{name:float(func([r[name] for r in rows])) for name in ['SR5','SR3','RMSE_median_px','RMSE_P90_px','solver_failures','threshold5_failures']},image_pairs=952,checkpoint='last epoch4',notes='same952validation images acrossseeds; not2856independent scenes'))
    write_csv(out/'table1_seed_results.csv',tables)
    transitions=[]
    for seed in p['seeds']:
        for ref in ['M','MS-shuffle']:
            a=successes[(seed,ref)];b=successes[(seed,'MS')];transitions.append(dict(seed=seed,reference=ref,variant='MS',failure_to_success=int((~a&b).sum()),success_to_failure=int((a&~b).sum()),both_success=int((a&b).sum()),both_failure=int((~a&~b).sum()),net_successes=int(b.sum()-a.sum()),delta_SR5_pp=100*float(b.mean()-a.mean()),image_pairs=952))
    write_csv(out/'table2_paired_transitions.csv',transitions)
    ci=[];raw=Path('runs/t09_bootstrap_v1');raw.mkdir(exist_ok=True);metric_names=['SR5','SR3','RMSE_median_px','RMSE_P90_px']
    for ref in ['M','MS-shuffle']:
        reference=np.stack([errors[(seed,ref)] for seed in p['seeds']]);variant=np.stack([errors[(seed,'MS')] for seed in p['seeds']]);result=paired_cluster_bootstrap(reference,variant,parents,p['bootstrap']['replicates'],p['bootstrap']['seed']);np.savez_compressed(raw/(ref+'_paired_parent_samples.npz'),samples=result['samples'])
        for i,seed in enumerate([*p['seeds'],'mean_over3_fixed_seeds']):
            for j,name in enumerate(metric_names):
                scale=100 if name.startswith('SR') else 1
                ci.append(dict(seed=seed,reference=ref,variant='MS',metric=name,unit='percentage_points' if scale==100 else 'px',paired_delta=scale*float(result['point'][i,j]),CI95_low=scale*float(result['low'][i,j]),CI95_high=scale*float(result['high'][i,j]),undefined_replicates=int(result['undefined_replicates'][i,j]),parent_clusters=len(set(parents)),validation_pairs=952,bootstrap_replicates=p['bootstrap']['replicates'],aggregation='sameparentdraws acrossall3fixedseedindices',positive_means_better=name.startswith('SR')))
    write_csv(out/'table3_parent_cluster_bootstrap.csv',ci)
    (out/'statistical_summary.json').write_text(json.dumps(dict(protocol=p,results=tables,transitions=transitions,paired_bootstrap=ci,checkpoint_sha256=source_hashes,bootstrap_scope='conditional uncertainty overknown512parents forfixed3trainingseeds; notsatellitesceneindependence or fullseedpopulation',primary_last=True),indent=2));print('Generated3requestedtables; no pooled2856sceneclaim')


if __name__=='__main__':main()
