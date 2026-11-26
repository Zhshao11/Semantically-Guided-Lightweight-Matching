"""T09 audit, paired seed decision and three requested tables without retuning."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import write_csv


def read(p):return json.loads(Path(p).read_text())


def main():
    out=Path('reports/t09_delivery_v1');out.mkdir(exist_ok=True);stats=read(out/'statistical_summary.json');p=stats['protocol'];audit=read('reports/t09_audit_v1/shuffle_parent_audit.json');labels=read('reports/t09_audit_v1/label_fields.json');replay=read('reports/t09_audit_v1/gradient_replay.json');index=read('reports/t09_audit_v1/run_index.json');trajectories=[];probes=[];diagnostics=[];curves={}
    for slot in index['runs']:
        root=Path(slot['root']);steps=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()];run=read(root/'summary.json');grad=run.get('semantic_gradient_probes',[]);curves[(slot['seed'],slot['group'])]=run['validation_history']
        trajectories.append(dict(seed=slot['seed'],group=slot['group'],updates=len(steps),KL_mean=float(np.mean([r['KL'] for r in steps])),KL_median=float(np.median([r['KL'] for r in steps])),KL_P90=float(np.percentile([r['KL'] for r in steps],90)),matching_loss_mean=float(np.mean([r['matching'] for r in steps])),actual_gradient_probes=len(grad),actual_semantic_encoder_L2_probe_mean=float(np.mean([r['semantic_encoder_L2'] for r in grad])) if grad else None,reused_T08=slot['reused_T08']))
        probes.extend(grad)
    oldsteps=[json.loads(x) for x in Path('runs/t08_MS-shuffle_v1/steps.jsonl').read_text().splitlines()];trajectories.append(dict(seed=20261126,group='T08-original-shuffle-HISTORY',updates=len(oldsteps),KL_mean=float(np.mean([r['KL'] for r in oldsteps])),KL_median=float(np.median([r['KL'] for r in oldsteps])),KL_P90=float(np.percentile([r['KL'] for r in oldsteps],90)),matching_loss_mean=float(np.mean([r['matching'] for r in oldsteps])),actual_gradient_probes=0,reused_T08=True))
    for seed in p['seeds']:
        data=read(f'reports/t09_diagnostics_v1/seed{seed}_summary.json')
        for r in data['rows']:
            raw=read(f"reports/t09_diagnostics_v1/seed{seed}_{r['group']}_K256.json")['pairs']
            inside=[x['refinement']['truth_inside_window'] for x in raw if x['refinement']['truth_inside_window']['before']['count']>0];count=sum(x['before']['count'] for x in inside)
            diagnostics.append(dict(seed=seed,group=r['group'],coarse_correct5=r['mean_coarse_correct_fraction_5px'],window_hit=r['mean_truth_in_window_fraction'],inside_images=len(inside),inside_correspondences=count,inside_before_mean_px_macro=float(np.mean([x['before']['mean_px'] for x in inside])),inside_after_mean_px_macro=float(np.mean([x['after']['mean_px'] for x in inside])),inside_before_RMSE_pooled_px=float(np.sqrt(sum(x['before']['rmse_px']**2*x['before']['count'] for x in inside)/count)),inside_after_RMSE_pooled_px=float(np.sqrt(sum(x['after']['rmse_px']**2*x['after']['count'] for x in inside)/count)),refinement_mean_delta_inside_px=r['mean_refine_delta_inside_px'],refinement_mean_delta_outside_px=r['mean_refine_delta_outside_px'],inlier_coverage=r['mean_inlier_source_hull_fraction'],mean_matches=r['mean_predicted_matches']))
    write_csv(out/'trajectory_KL_summary.csv',trajectories);write_csv(out/'actual_training_gradient_probes.csv',probes);write_csv(out/'layered_diagnostics_summary.csv',diagnostics)
    grouped=[]
    for ref in ['M','MS-shuffle']:
        d=[r for r in stats['transitions'] if r['reference']==ref];ci=next(r for r in stats['paired_bootstrap'] if r['reference']==ref and r['seed']=='mean_over3_fixed_seeds' and r['metric']=='SR5');grouped.append(dict(reference=ref,variant='MS',mean_delta_pp=ci['paired_delta'],seed_delta_sample_std_pp=float(np.std([r['delta_SR5_pp'] for r in d],ddof=1)),seeds_MS_better=sum(r['net_successes']>0 for r in d),seeds_tied=sum(r['net_successes']==0 for r in d),seeds_MS_worse=sum(r['net_successes']<0 for r in d),CI95_low=ci['CI95_low'],CI95_high=ci['CI95_high']))
    write_csv(out/'primary_paired_summary.csv',grouped)
    # Display only; primary criterion remains prespecifiedSR5, no bestcheckpoint/seedselection.
    fig,axes=plt.subplots(1,3,figsize=(15,4));x=np.arange(3);colors={'M':'tab:blue','MS':'tab:orange','MS-shuffle':'tab:green'}
    for group in p['groups']:
        rows=[next(r for r in stats['results'] if r['row_type']=='seed' and r['group']==group and r['seed']==seed) for seed in p['seeds']];axes[0].plot(x,[r['SR5']*100 for r in rows],'o-',label=group,color=colors[group]);axes[1].plot(x,[r['SR3']*100 for r in rows],'o-',label=group,color=colors[group])
    for ax,title in zip(axes[:2],['Primary: fixedlast SR@5','Secondary: fixedlast SR@3']):ax.set(xticks=x,xticklabels=[str(s) for s in p['seeds']],ylabel='Success rate (%)',title=title);ax.tick_params(axis='x',rotation=15);ax.grid(alpha=.25);ax.legend()
    for i,r in enumerate(grouped):axes[2].errorbar(r['mean_delta_pp'],i,xerr=[[r['mean_delta_pp']-r['CI95_low']],[r['CI95_high']-r['mean_delta_pp']]],fmt='o',capsize=4)
    axes[2].axvline(0,color='gray',ls='--');axes[2].set(yticks=[0,1],yticklabels=['MS minus M','MS minus shuffle'],xlabel='SR@5 difference (percentage points)',title='Paired512parent95%CI, seedsfixed');axes[2].grid(alpha=.25);fig.tight_layout();fig.savefig(out/'paired_seed_results.png',dpi=160);plt.close(fig)
    report=['# T09：对照审计与三个配对训练顺序种子','',
      '日期：2026-11-26。目的仅判断当前逐格光学语义伪标签KL辅助是否值得保留；主指标预先固定为均匀K256的SR@5，次指标SR@3与RMSE中位/P90。原网络、T05初始化、7912训练对/保存仿射、解冻层、lr2e-4、lambda1及7912次更新保持不变；统一固定last epoch4，不改为best，不增加其他预算。','',
      '## 对照审计与修正版','',
      f"原T08有{audit['old_same_parent_cells']:,}/{audit['active_cells']:,}接受格（{audit['old_same_parent_fraction']:.4%}）的donor来自同一verified512父图的其他子块，涉及{audit['old_recipient_images_with_same_parent']}张接收图。其余mask/weight/七类概率向量/全局类别保持控制有效，无须按‘标签和mask随意同洗牌’的错误假设返工。",'',
      f"T09修正版只作{audit['swaps']:,}次同权重分层内donor交换，改变{audit['changed_recipient_cells']:,}个接收格，保留原分配{audit['unchanged_recipient_fraction']:.4%}；同父图donor降为0。接收mask/weight逐位保留，原32分层、全局软类与熵分布保持，逐类加权质量差另列。原T08文件/模型/数值不覆盖。旧mapping SHA `{audit['old_mapping_sha256']}`；新SHA `{audit['mapping_sha256']}`。",'',
      "历史seed20261126的T08原shuffle SR@5为55.15%，本轮父图排除修正版为53.15%；M/MS仍复用原检查点。旧结果不覆盖，也不把原/修正shuffle当成两个独立种子。小比例目标修复后的非线性训练结果变化不应按修改格比例直接外推。", '', 'T08固定映射由训练循环外生成、dataset只读memmap，源码没有每epoch重新置换；但旧run没有逐epoch哈希日志，不虚构历史追踪。T09在每epoch前后核对映射/概率/权重文件哈希，同一修正版缓存固定用于3个训练种子。T08原seed的M/MS可复用，shuffle补一个修正版；新增两个seed各3组，共7个新运行构成9个配对位置，原T08shuffle结果仅保留历史。','',
      '| 标签版本 | 接受覆盖 | 标签熵均值(nats) | 加权熵均值 | 相邻接受格同主类比例 |','| --- | ---: | ---: | ---: | ---: |']
    for r in labels['rows']:report.append(f"| {r['version']} | {r['coverage']:.2%} | {r['entropy_mean_nats']:.6f} | {r['weighted_entropy_mean_nats']:.6f} | {r['accepted_neighbor_same_class_fraction']:.2%} |")
    report.extend(['','类别占比和软概率/加权质量见label_class_distribution.csv。逐格打乱明显破坏空间连续性，除破坏内容对应外还改变空间统计，不能当作只改变‘语义正确性’的完美对照。OS无人工语义GT；这些均为外部教师软伪标签统计，不是语义准确率。','',
      '## KL与语义梯度量级','',
      '共同初始状态及T08保存的末状态上，使用相同128对train/32个batch回放，报告全局裁剪前粗编码层语义梯度L2（不含投影/辅助头）及匹配梯度比值。使用丢弃副本、matcher BN冻结、辅助头BN train并逐批重置，未改历史检查点。T09-shuffle标签在T08末状态的回放仍使用原T08-shuffle检查点，只用于隔离标签修正，不代表修正版训练末状态。这不是历史逐步梯度日志；7个新run另有实际步骤1/1978/3956/5934/7912的同forward梯度记录。','',
      '| 回放状态 | 标签 | KL均值 | 语义粗层梯度L2均值 | 语义/匹配梯度比均值 |','| --- | --- | ---: | ---: | ---: |'])
    for r in replay['rows']:report.append(f"| {r['checkpoint']} | {r['label_version']} | {r['KL']:.4f} | {r['semantic_encoder_L2']:.4f} | {r['semantic_over_matching']:.4f} |")
    report.extend(['','KL值大并不等于粗层语义梯度大：T08末状态的MS和shuffle梯度量级差异明显。标签权重/直方图控制不能保证实际辅助约束强度等价；本轮只报告、不为接近结果而改lambda。全7912步KL统计见trajectory_KL_summary.csv，实际探针见actual_training_gradient_probes.csv，历史回放单列。','',
      '## 表1：完整种子结果及均值/样本标准差','',
      '| seed/统计 | 版本 | SR@5（主） | SR@3 | RMSE中位/P90(px) | 求解失败 |','| --- | --- | ---: | ---: | ---: | ---: |'])
    for r in stats['results']:
        label=str(r['seed']) if r['row_type']=='seed' else '均值' if r['row_type']=='mean' else '样本SD(ddof1)'
        report.append(f"| {label} | {r['group']} | {r['SR5']:.2%} | {r['SR3']:.2%} | {r['RMSE_median_px']:.3f}/{r['RMSE_P90_px']:.3f} | {r['solver_failures']:.2f} |")
    report.extend(['','每个seed内部三组样本/仿射流SHA一致；三个种子都使用同T05/同辅助头初始权重，seed只改变训练排列，不是随机初始化消融。各seed仍评价相同952影像，不能当2856个独立场景。失败保留SR分母，误差分位数只对可求解样本。','',
      '## 表2：MS的双向成败变化（固定5px）','',
      '| seed | 对照 | 失败→成功 | 成功→失败 | 净增成功 | SR@5差(百分点) |','| --- | --- | ---: | ---: | ---: | ---: |'])
    for r in stats['transitions']:report.append(f"| {r['seed']} | {r['reference']} | {r['failure_to_success']} | {r['success_to_failure']} | {r['net_successes']:+d} | {r['delta_SR5_pp']:+.2f} |")
    report.extend(['','此处失败指独立探针RMSE未达5px（包括求解失败），不是只计RANSAC求解失败。双成功/双失败完整列于table2_paired_transitions.csv。','',
      '## 表3：已验证512父图的配对重采样','',
      '952对对应238个verified512父图，每次有放回抽238个父图，完整保留父图内4个256子块。20000次、seed20261126；每次同一父图抽样同步用于所有方法及三个固定训练seed，分别计算逐seed差和三seed平均差。下列为三固定seed均值的条件区间，逐seed及次指标CI完整列于table3_parent_cluster_bootstrap.csv。','',
      '| MS相对 | 平均SR@5差(百分点) | 三seed差值SD | MS胜/平/负seed数 | 父图95%CI |','| --- | ---: | ---: | ---: | ---: |'])
    for r in grouped:report.append(f"| {r['reference']} | {r['mean_delta_pp']:+.2f} | {r['seed_delta_sample_std_pp']:.2f} | {r['seeds_MS_better']}/{r['seeds_tied']}/{r['seeds_MS_worse']} | [{r['CI95_low']:+.2f}, {r['CI95_high']:+.2f}] |")
    report.extend(['','SR差为MS−对照的百分点，正为更好；RMSE差为像素，负为更好。区间反映已知分发512父图相关性下、固定3个训练seed的验证抽样不确定性，不估计训练seed总体，也不能替代原始卫星scene独立性；原始scene来源仍unresolved。','',
      '## 是否保留当前逐格KL',''])
    for r in grouped:report.append(f"主指标上，MS相对{r['reference']}平均{r['mean_delta_pp']:+.2f}个百分点，三个seed胜/平/负为{r['seeds_MS_better']}/{r['seeds_tied']}/{r['seeds_MS_worse']}；父图配对区间[{r['CI95_low']:+.2f}, {r['CI95_high']:+.2f}]。")
    robust=all(r['seeds_MS_better']==3 and r['CI95_low']>0 for r in grouped)
    decision='当前3seed与已知父图抽样范围内有一致的主指标收益，支持保留当前KL作为候选，仍需外部/scene层验证；不能把伪标签当真实语义GT。' if robust else '当前未形成同时优于匹配微调与修正版shuffle的稳定主指标证据；不把逐格KL保留为已验证的主方法收益来源。保留代码/结果作为弱语义假设的否证或有限证据，不通过加模块、挑seed/最佳轮次或调lambda掩盖。'
    report.extend([decision,'',
      '梯度强度与空间连续性不等价仍限制因果解释，不能单凭shuffle好坏断言全部是普通正则化或全部是语义。窗口内精化误差同时报告非空影像等权均值和按窗口内对应数量汇总的RMSE，计数列明。次指标和固定96对粗正确率/入窗/窗口内精化误差只用于解释，不替代预定SR@5判据。三组推理均无辅助头、无在线教师，未进行正式效率对比，不新增加速宣称。','',
      '源码、配置、非像素三张表/诊断/训练记录同步本地、zzvr和GitHub；权重、伪标签数组、donor映射及bootstrap重复样本保留私有。本轮没有重训教师、没有新增预算、没有OS语义精度。'])
    (out/'T09实验报告.md').write_text('\n'.join(report)+'\n');(out/'decision.json').write_text(json.dumps(dict(primary='SR@5px',paired_summaries=grouped,retain_as_supported_main_mechanism=robust,decision=decision,limitations=['spatialcontinuitynotmatched','gradientstrengthnotmatched','3fixedtrainingorderseeds','satellitescenesunresolved']),ensure_ascii=False,indent=2));print(decision)


if __name__=='__main__':main()
