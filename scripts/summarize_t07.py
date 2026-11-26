"""T07 fixed comparisons, provenance and limitations; no selection or threshold changes."""
import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import write_csv


def read(p):return json.loads(Path(p).read_text())


def main():
    out=Path('reports/t07_delivery_v1');out.mkdir(exist_ok=True)
    p=read('configs/t07_protocol_v1.json');freeze=read('reports/t07_teacher_v1/teacher_freeze.json');cache=read('runs/t07_semantic_cache_v1/summary.json');head=read('runs/t07_semantic_head_v1/summary.json');selection=read('reports/t07_selection_v1/summary.json');source=read('reports/t07_teacher_v1/source_mask_audit.json')
    rows=[]
    for r in selection['rows']:
        rows.append(dict(mode=r['mode'],K=r['K'],SR3=r['success_rate@3px'],SR5=r['success_rate@5px'],RMSE_median_px=r['solved_median_rmse_px'],RMSE_P90_px=r['RMSE_p90_px'],solver_failures=r['failed_pairs'],threshold5_failures=952-round(952*r['success_rate@5px']),selected_correct5=r['selected_query_correct5'],actual_window_hit=r['actual_refinement_window_hit'],inlier_coverage=r['inlier_hull_fraction'],mean_refined=r['mean_matches'],head_parameters=r['head_parameters'],total_matching_parameters=37219+r['head_parameters']))
    rows.sort(key=lambda r:(r['K'],p['modes'].index(r['mode'])));write_csv(out/'fixed_K_selection.csv',rows)
    transitions=[]
    for k in p['budgets']:
        def flags(mode):
            root='t06_selection_v1' if mode in ['U','A','AC'] else 't07_selection_v1'
            return {r['pair_id']:r['evaluation']['success']['5'] for r in read(f'reports/{root}/{mode}_K{k}.json')['pairs']}
        for a,b in [('A','A_cap'),('A_cap','S'),('A','S'),('AC','SC'),('S','SC')]:
            x,y=flags(a),flags(b);transitions.append(dict(K=k,reference=a,variant=b,both_success=sum(x[i] and y[i] for i in x),lost_success=sum(x[i] and not y[i] for i in x),gained_success=sum(not x[i] and y[i] for i in x),both_failure=sum(not x[i] and not y[i] for i in x)))
    write_csv(out/'paired_registration_transitions.csv',transitions)
    coverage_controls=[]
    old_coverage=read('reports/t06_selection_v1/summary.json')['rows']
    for k in p['budgets']:
        control=next(r for r in old_coverage if r['mode']=='AC-cap' and r['K']==k)
        semantic=next(r for r in rows if r['mode']=='SC' and r['K']==k)
        coverage_controls.append(dict(K=k,appearance='T06 AC-cap',appearance_head_parameters=control['head_parameters'],appearance_SR5=control['success_rate@5px'],semantic='SC',semantic_head_parameters=7977,semantic_SR5=semantic['SR5'],delta_percentage_points=100*(semantic['SR5']-control['success_rate@5px'])))
    write_csv(out/'coverage_capacity_control.csv',coverage_controls)
    teachers=[]
    for mode in ['TR','TG']:
        root=Path('runs/t07_teacher_'+mode+'_v1');summary=read(root/'summary.json');step=summary['best']['step'];v=read(root/f'validation_{step:04d}.json')
        for name,m in v['groups'].items():
            teachers.append(dict(teacher='official_'+mode,completed_steps=summary['steps'],selected_step=step,input_group=name,mIoU=m['mean_IoU'],**{r['name']+'_IoU':r['IoU'] for r in m['per_class']},**{r['name']+'_predicted_fraction':r['predicted_pixels']/m['valid_GT_pixels'] for r in m['per_class']}))
    write_csv(out/'replacement_teacher_comparison.csv',teachers)
    sem=dict(train_pairs=cache['pairs'],positive_weight_fraction=cache['positive_weight_fraction'],raw_teacher_class_fractions=dict(zip(freeze['source_classes'],cache['raw_class_fractions'])),pre_abstention_mask_fraction=cache['pre_abstention_mask_fraction'],quarantined_images=cache['quarantined_images'],class_effective_mass=dict(zip(freeze['target_classes'],cache['class_effective_mass'])))
    (out/'semantic_label_statistics.json').write_text(json.dumps(sem,indent=2));shutil.copy2('runs/t06_reliability_v1/summary.json',out/'reliability_label_statistics.json');shutil.copy2('reports/t07_teacher_v1/teacher_freeze.json',out/'teacher_freeze.json')
    x=np.arange(6);fig,axes=plt.subplots(1,3,figsize=(15,4))
    for k in [256,512]:
        group=[next(r for r in rows if r['mode']==m and r['K']==k) for m in p['modes']]
        axes[0].plot(x,[r['SR5']*100 for r in group],'o-',label=f'K{k}');axes[1].plot(x,[r['selected_correct5']*100 for r in group],'o-',label=f'K{k}')
    for ax,y in zip(axes[:2],['SR@5px (%)','Selected query correct@5px (%)']):ax.set(xticks=x,xticklabels=p['modes'],ylabel=y);ax.legend();ax.grid(alpha=.25)
    axes[0].set_title('Full fixed952 validation');axes[1].set_title('Prediction quality vs registration')
    epochs=[0]+[r['epoch'] for r in head['history']];axes[2].plot(epochs,[head['initial_train_probe']['average_precision']]+[r['probe']['average_precision'] for r in head['history']],label='S train AP')
    cap=read('runs/t06_head_capacity_v1/summary.json');axes[2].plot(epochs,[cap['initial_train_probe']['average_precision']]+[r['probe']['average_precision'] for r in cap['history']],label='A_cap train AP');axes[2].set(xlabel='Epoch',ylabel='Average precision',title='TRAIN probe, not validation');axes[2].legend();axes[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'selection_and_training.png',dpi=160);plt.close(fig)
    report=['# T07：教师冻结与不强制覆盖的语义选择','',
      '日期：2026-11-26。本轮主比较为同K的S与A_cap；AC/SC仅次要配对消融。匹配器epoch18、骨干/描述子/精化器、0.01匹配阈值及原RANSAC不变。教师仅离线，推理只运行学生语义头；无骨干解冻、硬选择适配或动态K。','',
      '## 教师冻结与许可来源','',
      '旧TG技术记录已冻结在TG_community_freeze.json：step400、SHA256 4c4b3d84bb23c947283f99188b33a99b347159d6bb474116241ebb5575c82ff3，旧64灰度mIoU53.62%。社区来源仍未明确研究条款，因此没有继续用它生成T07学生标签，也没有发布权重。','',
      '按用户要求替换为官方nvidia/mit-b2初始化（固定revision3bb39e8739149c3777d0325349b2a6c32c6413db）。官方模型卡明确链接SegFormer许可，条款3.3仅限非商业研究/评价；不是自由部署授权。许可文本、固定模型卡和哈希保留，原始/派生权重均未公开。只加载官方ImageNet编码器，八通道分割解码器重新初始化，未复制旧社区权重。','',
      '替代TR/TG均3000更新、有效batch4、AdamW lr6e-5、相同2522训练图/512裁剪/GT流、仅RGB与灰度不同，每500步检查固定64，统一按灰度累计混淆矩阵七类mIoU选best。这个实验不同于旧TG的1000步继续训练，不能直接把性能变化归因于许可证或仅初始化。','',
      f"正式研究教师为{freeze['checkpoint_path']}，step{freeze['selected_step']}，SHA256 `{freeze['checkpoint_sha256']}`。训练源代码SHA、输入处理器、64清单和类别记录见teacher_freeze.json。",'',
      '| 教师 | 选中步 | 新64 RGB mIoU | 新64灰度mIoU |','| --- | ---: | ---: | ---: |']
    for mode in ['TR','TG']:
        a=next(t for t in teachers if t['teacher']=='official_'+mode and t['input_group']=='fixed64/RGB');b=next(t for t in teachers if t['teacher']=='official_'+mode and t['input_group']=='fixed64/gray_RGB')
        report.append(f"| official_{mode} | {a['selected_step']} | {a['mIoU']:.2%} | {b['mIoU']:.2%} |")
    report.extend(['','64为快速选模验证，不称完整官方Val性能。逐类IoU、预测比例及旧8纵向结果完整列于replacement_teacher_comparison.csv。','',
      'OS输入明确uint8 L→RGB；保存的512双线性缩放、1/255与ImageNet归一化；logits双线性回原256（align_corners=False）后softmax8，再8×8有效面积均值。0Ignore不作地物，1—7映射Background/Building/Road/Water/Barren/Forest/Agricultural，保留a=1-pIgnore，条件分布p7和w=mask*a*(1-H/log7)。基础mask仍为完整有效覆盖、a≥.8、max(p7)≥.6，未提高阈值掩盖系统性错误。','',
      f"原32张OS训练来源复查与掩码叠加完成；弃权规则已冻结：基础mask保持原值，Barren主类格不监督；背景占比≥95%且平均最大概率≥90%的图整图弃权；固定32张抽查中的1154、1182、1500、4410、6090、6537、7524另作保守整图弃权（观察依据见OS_visual_decision.json）。按同一规则扫描全部7912训练图，共{cache['quarantined_images']}张语义监督整图置零；不影响可靠性几何标签或匹配训练来源。",'',
      f"源域GT掩码审计：可接受粗格覆盖{source['accepted_fraction']:.2%}有效像素，通过格内错误比例{source['wrong_fraction_among_accepted']:.2%}。这是粗32分类展开后与全部native GT像素比较，包含混合边界，不等于原生像素分割错误率；它表明高置信mask不能保证语义正确。OS无语义GT，图像观察不冒充目标域准确率。",'',
      '源域Barren接受粗格的展开精度仅36.97%，其主类粗格全局弃权；七类条件概率保持完整，不能声称七类均已获得可靠OS监督。其他类别只作为带绝对质量权重的弱监督候选，并不保证掩码内没有错误。Barren在保留格的条件分布中仍可有非主类概率质量，因此有效质量表不必为零；不把七类分布改成六类。', '', '## 学生与公平对照','',
      f"复用T06冻结7912训练对/扰动/几何标签（正19.89%/负52.27%/忽略27.84%）。仅新增离线语义缓存，正权重格比例{cache['positive_weight_fraction']:.2%}；每类有效质量见semantic_label_statistics.json。A_cap为7831参数外观头，S为7977参数（含学生语义分支），差1.83%。A_cap复用T06已完成同预算最后检查点，不按新验证结果重选。",'',
      '两者均12epoch/2976更新、batch32、相同样本顺序和增强、AdamW lr.001/WD.0001、clip5、同几何正类平衡BCE。S另用预先固定系数1的绝对质量加权KL，除以格数而非sum(w)；语义概率对匹配BCE detach，只由KL训练语义分支。骨干/描述子不实例化进训练。student BN只属于头，验证全部eval冻结。评分先于选择/gather及唯一K×1024相关；推理既不加载教师，也不读GT/缓存。','',
      '| K | 版本 | SR@3 | SR@5 | RMSE中位/P90(px) | 求解失败 | 5px未成功 | 选中查询正确@5 | 精化入窗 | 内点覆盖 |','| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in rows:report.append(f"| {r['K']} | {r['mode']} | {r['SR3']:.2%} | {r['SR5']:.2%} | {r['RMSE_median_px']:.3f}/{r['RMSE_P90_px']:.3f} | {r['solver_failures']}/952 | {r['threshold5_failures']}/952 | {r['selected_correct5']:.2%} | {r['actual_window_hit']:.2%} | {r['inlier_coverage']:.2%} |")
    report.extend(['','求解失败始终保留SR分母，误差分位数仅可求解样本。查询正确率以全部K位置为分母，精化入窗按实际对应，覆盖为内点源凸包面积/图像面积。U/A/AC直接复用完全相同冻结模型、输入和规则的T06完整952记录，A_cap/S/SC本轮完整评价。','',
      '## 主要比较与限制',''])
    for k in [256,512]:
        a=next(r for r in rows if r['K']==k and r['mode']=='A_cap');s=next(r for r in rows if r['K']==k and r['mode']=='S')
        report.append(f"K{k}：S相对等容量A_cap的SR@5差{100*(s['SR5']-a['SR5']):+.2f}个百分点，SR@3差{100*(s['SR3']-a['SR3']):+.2f}个百分点；选中查询正确率差{100*(s['selected_correct5']-a['selected_correct5']):+.2f}个百分点。配对从成功转失败/反向转移见paired_registration_transitions.csv，不能用训练AP提高替代配准机制判断。")
    report.extend(['','次要AC/SC比较还含小外观头与联合头的容量差，不能直接归因语义。既有同覆盖容量对照另见coverage_capacity_control.csv；它也不能替代主S/A_cap比较。', '', '这是固定教师/冻结匹配器/单seed预算的机制初筛，不是解冻后完整语义模型或论文最终性能。替代教师源域表现弱于旧TG且Barren主类已弃权，结论受此教师和冻结特征下的学生蒸馏能力限制，不能直接外推到所有语义来源。不根据验证结果调整K、语义权重、匹配阈值、RANSAC或教师选择规则。语义收益必须主要对照A_cap判断，AC/SC只解释覆盖交互。','',
      '源码、配置、非像素指标/清单同步GitHub；标签数组、特征、私有检查点、原始/预测影像保留本地outputs/zzvr runs。冻结与训练/验证证据、许可范围和测试见同目录acceptance.json及复现文档。'])
    (out/'T07实验报告.md').write_text('\n'.join(report)+'\n')
    (out/'experiment_summary.json').write_text(json.dumps(dict(protocol=p,freeze=freeze,selection=rows,teachers=teachers,semantic_statistics=sem,head=head,paired_transitions=transitions,OS_semantic_accuracy_claim=False),indent=2))
    print('T07 report generated',len(rows),'comparisons')


if __name__=='__main__':main()
