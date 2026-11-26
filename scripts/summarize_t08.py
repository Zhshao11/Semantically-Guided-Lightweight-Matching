"""Controlled T08 pseudo-semantic auxiliary report, no additional tuning."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import write_csv


def read(p):return json.loads(Path(p).read_text())


def main():
    out=Path('reports/t08_delivery_v1');out.mkdir(exist_ok=True);p=read('configs/t08_protocol_v1.json');pre=read('reports/t08_preflight_v1/diagnosis.json');labels=read('reports/t08_preflight_v1/pseudo_label_audit.json');shuffle=read('runs/t08_shuffle_v1/summary.json');runs={name:read(f'runs/t08_{name}_v1/summary.json') for name in p['groups']};rows=[];transitions=[]
    base=read('reports/t05_results_v1/accuracy_summary.json') if Path('reports/t05_results_v1/accuracy_summary.json').exists() else None
    for k in [256,1024]:
        old=read(f'reports/t05_results_v1/K{k}_validation.json');metrics=old['summary'];error=[r['evaluation']['rmse_px'] for r in old['pairs'] if r['evaluation']['solved']]
        rows.append(dict(group='T05-initial',K=k,SR3=metrics['success_rate@3px'],SR5=metrics['success_rate@5px'],RMSE_median_px=metrics['solved_median_rmse_px'],RMSE_P90_px=float(np.percentile(error,90)),solver_failures=metrics['failed_pairs'],threshold5_failures=952-round(952*metrics['success_rate@5px']),epoch=18,matching_parameters=37219))
        flags={}
        for name,s in runs.items():
            data=read(f'runs/t08_{name}_v1/K{k}/validation_epoch_004.json');m=data['summary'];err=[r['evaluation']['rmse_px'] for r in data['pairs'] if r['evaluation']['solved']]
            rows.append(dict(group=name,K=k,SR3=m['success_rate@3px'],SR5=m['success_rate@5px'],RMSE_median_px=m['solved_median_rmse_px'],RMSE_P90_px=float(np.percentile(err,90)),solver_failures=m['failed_pairs'],threshold5_failures=952-round(952*m['success_rate@5px']),epoch=4,matching_parameters=s['inference_parameters']))
            flags[name]={r['pair_id']:r['evaluation']['success']['5'] for r in data['pairs']}
        for left,right in [('M','MS'),('MS-shuffle','MS'),('M','MS-shuffle')]:
            a,b=flags[left],flags[right];transitions.append(dict(K=k,reference=left,variant=right,both_success=sum(a[i] and b[i] for i in a),lost_success=sum(a[i] and not b[i] for i in a),gained_success=sum(not a[i] and b[i] for i in a),both_failure=sum(not a[i] and not b[i] for i in a)))
    write_csv(out/'registration_comparison.csv',rows);write_csv(out/'paired_transitions.csv',transitions)
    hist=[];architecture=[]
    for name,s in runs.items():
        arch=read(f'runs/t08_{name}_v1/architecture.json');architecture.extend(dict(group=name,module=key,trainable_parameters=value) for key,value in arch['trainable'].items())
        for row in s['history']:hist.append(dict(group=name,**row))
    write_csv(out/'training_history.csv',hist);write_csv(out/'trainable_parameters.csv',architecture)
    diagnostic=read('reports/t08_diagnostics_v1/summary.json');drows=[]
    for row in diagnostic['rows']:
        drows.append(dict(group=row['group'],K=row['K'],coarse_correct5=row['mean_coarse_correct_fraction_5px'],window_hit=row['mean_truth_in_window_fraction'],inlier_coverage=row['mean_inlier_source_hull_fraction'],refinement_delta_inside_px=row['mean_refine_delta_inside_px'],refinement_delta_outside_px=row['mean_refine_delta_outside_px'],mean_matches=row['mean_predicted_matches']))
    write_csv(out/'layered_diagnostic_summary.csv',drows)
    figs,axes=plt.subplots(1,3,figsize=(15,4));colors=dict(M='tab:blue',MS='tab:orange',**{'MS-shuffle':'tab:green'})
    for name,s in runs.items():
        axes[0].plot([r['epoch'] for r in s['history']],[r['matching'] for r in s['history']],'o-',label=name,color=colors[name]);axes[1].plot([r['epoch'] for r in s['history']],[r['KL'] for r in s['history']],'o-',label=name,color=colors[name]);v=[r for r in s['validation_history'] if r['K']==256];axes[2].plot([r['epoch'] for r in v],[100*r['success_rate@5px'] for r in v],'o-',label=name,color=colors[name])
    for ax,title in zip(axes,['Training matching loss','Training pseudo KL','Full952 uniformK256 SR@5']):ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.25);ax.legend()
    figs.tight_layout();figs.savefig(out/'training_and_validation.png',dpi=160);plt.close(figs)
    shutil.copy2('reports/t08_preflight_v1/pseudo_label_audit.json',out/'pseudo_label_audit.json');shutil.copy2('runs/t08_shuffle_v1/summary.json',out/'shuffle_audit.json')
    gate=pre['gate'];report=['# T08：外部教师驱动的弱语义辅助表征实验','',
      '日期：2026-11-26。几何匹配由OS配准影像对和合成仿射扰动监督；语义辅助由外部光学教师生成软伪标签，经可靠性掩码参与训练。OS没有人工语义GT，不报告OS语义精度、不以此选模型，也不把配准变化解释为语义预测更准确。用户已明确取消寻找不存在OS语义GT的前置要求。','',
      '## 开训前诊断','',
      f"7912图共{labels['total_grids']:,}个粗格，{labels['positive_weight_grids']:,}格有正语义权重（{labels['positive_weight_fraction']:.2%}），忽略{labels['ignored_grids']:,}格；{labels['images_without_weight']}张整图无语义权重。七类覆盖与绝对概率质量见pseudo_label_classes.csv；四邻域至少75%同主类的接受格比例{labels['accepted_neighbor_stable_fraction']:.2%}，只作为空间稳定性诊断，没有据此另调训练mask。",'',
      '沿用T07已冻结的官方MiT-B2来源TG软伪标签与弱监督弃权规则，没有重训教师；Barren主类弃权来自教师质量审计，不是把某类硬编码为不适合匹配。原32张OS的教师漏分/掩码穿透保留为质量观察，仍无语义GT。可选16张训练来源人工复核模板没有标注，未当作独立准确率依据，也不排除出训练。', '',
      f"T07门控修正logit绝对值均值{gate['abs_mean']:.4f}、中位{gate['abs_median']:.4f}、P95{gate['abs_P95']:.4f}；标准差/外观logit标准差={gate['std_ratio']:.3f}，绝对值≤1e-3仅{gate['abs_le_1e_minus3_fraction']:.2%}，因此不支持‘修正几乎为零’。",'',
      '| 路由诊断，固定128对train／K256 | SR@3 | SR@5 | 选中查询正确@5 | 与学生查询Jaccard |','| --- | ---: | ---: | ---: | ---: |']
    for r in pre['routing']:report.append(f"| {r['mode']} | {r['success_rate@3px']:.2%} | {r['success_rate@5px']:.2%} | {r['selected_query_correct5']:.2%} | {r['mean_query_Jaccard_vs_student']:.3f} |")
    report.extend(['','教师概率替代仅用于离线诊断学生信息损失，教师不是真值。直接替换全图概率或仅可靠格替换都未提高SR@5，不能把T07未获益简单解释成学生语义太弱；也不能据训练子集推断验证泛化。未在OS验证/测试生成教师语义缓存。','',
      '## 三组固定协议','',
      '同T05第18轮SHA f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90，全部7912对去重训练及T06保存仿射，同seed20261126/顺序；均匀K256，BF16匹配/FP32语义分支，batch4，AdamW lr2e-4/WD1e-4，clip5，无调度器，4epoch/7912更新。4轮是首轮观察预算，不声明收敛或多seed稳定。主要评价固定epoch4的last，不按验证最优换结果；best仅另存诊断。', '',
      '只开放backbone.down.1、stages.2、lateral.2和descriptor；共享粗编码卷积、两模态高层BN affine可由匹配损失适配，所有骨干BN运行统计冻结。适配器、stage0/1、低层下采样、其余FPN和局部精化参数、no-match/匹配温度冻结。冻结FPN参数仍会接收改变后的粗激活，不声称细特征张量恒定。','',
      'M仅匹配损失；MS加lambda=1的可靠格伪标签KL；MS-shuffle同KL但跨影像打乱概率。MS/MS-shuffle辅助头均从T07同一语义分支初始化，挂光学64粗特征，不detach，使KL更新粗编码层；不对SAR施加像素语义KL，但共享卷积改变会间接影响SAR特征。原匹配损失沿用双向正负分开均值CE及0.2细化项，warmstart不重开GT窗口bootstrap；K256反向类别映射到所选查询，未选源位置归入no-match256。K1024极限与原损失数值一致性有测试。', '',
      f"打乱监督：按接受格权重32分位分层，将七类概率完整向量跨影像置换；全部受监督格的donor来自其他影像。接收侧mask/权重逐位保留，全局接受格软类别直方图保持一致；加权类别质量最大相对偏差{100*max(abs(x) for x in shuffle['weighted_class_mass_relative_difference']):.3f}%。不保持每幅影像类别直方图，也不是完美等价对照，映射SHA和质量差异完整记录。",'',
      '验证epoch0/1/2/4全部952对、原固定探针/阈值/RANSAC，epoch0三组逐pair复现原K256；epoch4另评K1024，仅检查密集改善是否能传递到稀疏。语义伪标签不参与拟合变换，求解失败保留分母。训练期有辅助头，导出时删除，推理仅37219参数匹配器；没有任何学习评分/动态K，也没有本轮正式效率测量，不能声称整网加速。','',
      '## 固定last完整验证','',
      '| K | 组别 | SR@3 | SR@5 | RMSE中位/P90(px) | 求解失败 | 5px未成功 |','| ---: | --- | ---: | ---: | ---: | ---: | ---: |'])
    for r in rows:report.append(f"| {r['K']} | {r['group']} | {r['SR3']:.2%} | {r['SR5']:.2%} | {r['RMSE_median_px']:.3f}/{r['RMSE_P90_px']:.3f} | {r['solver_failures']}/952 | {r['threshold5_failures']}/952 |")
    report.extend(['','## 对照判断',''])
    for k in [256,1024]:
        m=next(r for r in rows if r['group']=='M' and r['K']==k);ms=next(r for r in rows if r['group']=='MS' and r['K']==k);sh=next(r for r in rows if r['group']=='MS-shuffle' and r['K']==k)
        report.append(f"K{k}：MS相对M的SR@5变化{100*(ms['SR5']-m['SR5']):+.2f}个百分点，相对MS-shuffle变化{100*(ms['SR5']-sh['SR5']):+.2f}个百分点；配对新增/丢失见paired_transitions.csv。")
    report.extend(['','本轮K256的MS和MS-shuffle都高于M，且MS比打乱组少1对成功；K1024的MS仅比M高0.53个百分点、比打乱组高0.42个百分点。但K256的SR@3为M33.93%、MS36.76%、shuffle35.82%，MS在较严格阈值上较高，两个阈值的判断不完全一致。SR@5的结果与辅助正则化解释相容，尚不能隔离有效语义内容的稳定额外收益，也不足以强称‘语义促进轻量化’。', '', '必须首先比较MS与M及打乱组，不能只比较T05或只选有利阈值。单seed四轮不构成‘稳定优于’的证据；如果MS无优势，本轮伪语义接入机制未获支持，需要结合教师跨域误差、mask残留错误与蒸馏质量解释，不继续叠模块。本轮不报告OS语义准确率、不把教师输出称真值。','',
      '固定96对T05诊断子集上的粗正确率、窗口命中、精化变化与覆盖另列layered_diagnostic_summary.csv；该子集统计不替代完整952对表。所有GT诊断仅事后使用，推理不读取语义教师或标签。','',
      '## 产物与复现','',
      '完整训练每步损失、样本/仿射流SHA、逐层可训练参数、last/best/推理导出、96对分层结果和前置诊断已保留。方法代码、配置、非像素指标及计划同步本地/zzvr/GitHub；权重、原始/预测影像、伪标签数组和置换映射保留私有。教师许可延续T07明确的非商业研究/评价范围，不宣称部署授权、不发布权重。'])
    (out/'T08实验报告.md').write_text('\n'.join(report)+'\n');(out/'experiment_summary.json').write_text(json.dumps(dict(protocol=p,preflight=pre['gate'],routing=pre['routing'],pseudo_labels=labels,comparison=rows,paired_transitions=transitions,runs=runs,shuffle=shuffle,OS_semantic_accuracy=None),indent=2));print('T08 report generated')


if __name__=='__main__':main()
