"""Curated T06 tables/figures; no training, model selection or threshold changes."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import write_csv


def main():
    out=Path('reports/t06_delivery_v1');out.mkdir(parents=True,exist_ok=True)
    p=json.loads(Path('configs/t06_protocol_v1.json').read_text())
    labels=json.loads(Path('runs/t06_reliability_v1/summary.json').read_text())
    heads={name:json.loads(Path(path,'summary.json').read_text()) for name,path in [('small','runs/t06_head_small_v1'),('A','runs/t06_head_v1'),('capacity','runs/t06_head_capacity_v1')]}
    selection=json.loads(Path('reports/t06_selection_v1/summary.json').read_text());rows=selection['rows']
    table=[]
    for r in rows:
        table.append(dict(mode=r['mode'],K=r['K'],status='completed',SR3=r['success_rate@3px'],SR5=r['success_rate@5px'],
            RMSE_median_solved_px=r['solved_median_rmse_px'],RMSE_p90_solved_px=r['RMSE_p90_px'],solver_failures=r['failed_pairs'],
            successes5=int(round(952*r['success_rate@5px'])),failures5=952-int(round(952*r['success_rate@5px'])),
            selected_query_correct5=r['selected_query_correct5'],actual_refinement_window_hit=r['actual_refinement_window_hit'],
            filtered_match_correct5=r['filtered_match_correct5'],inlier_hull_fraction=r['inlier_hull_fraction'],
            mean_predicted_matches=r['mean_matches'],head_parameters=r['head_parameters']))
    for k in p['budgets']:table.append(dict(mode='SC',K=k,status='waiting_teacher_finalization',SR3=None,SR5=None))
    write_csv(out/'fixed_K_selection.csv',table)
    teachers={};teacher_rows=[];teacher_last=[]
    for version in ['T0','TR','TG']:
        root=Path('runs/t06_teacher_'+version);summary=json.loads((root/'summary.json').read_text());step=summary['best']['step']
        best=json.loads((root/f'validation_{step:04d}.json').read_text());teachers[version]=dict(summary=summary,best=best)
        for group in ['fixed64','old8']:
            for mode in ['RGB','gray_RGB']:
                m=best['groups'][group+'/'+mode]
                teacher_rows.append(dict(teacher=version,selected_step=step,completed_updates=summary['steps'],group=group,input=mode,
                    mIoU=m['mean_IoU'],OA=m['overall_accuracy'],GT_pixels=m['valid_GT_pixels'],predicted_background_fraction=m['predicted_background_fraction_on_valid_GT'],
                    **{r['name']+'_IoU':r['IoU'] for r in m['per_class']}))
        if version!='T0':
            last=json.loads((root/'validation_1000.json').read_text())
            for mode in ['RGB','gray_RGB']:
                m=last['groups']['fixed64/'+mode];teacher_last.append(dict(teacher=version,step=1000,input=mode,mIoU=m['mean_IoU'],OA=m['overall_accuracy']))
    a,b=teachers['TR']['summary'],teachers['TG']['summary']
    assert a['presentations_sha256']==b['presentations_sha256'] and a['cropped_GT_stream_sha256']==b['cropped_GT_stream_sha256'] and a['GT_counts_in_training']==b['GT_counts_in_training']
    write_csv(out/'teacher_comparison.csv',teacher_rows);write_csv(out/'teacher_last_step_control.csv',teacher_last)
    review=json.loads(Path('runs/t06_OS_review_v1/summary.json').read_text());mass=[]
    for version,r in review['summary'].items():
        mass.append(dict(teacher=version,OS_train_images=32,high_confidence_background_flags=r['high_confidence_all_background_flags'],
            semantic_GT=False,valid_class_mass=r['valid_class_mass_mean'],mask_fraction=r['supervision_mask_fraction'],
            mean_supervision_weight=r['supervision_weight_mean'],**r['effective_class_mass']))
    write_csv(out/'semantic_effective_class_mass.csv',mass)
    timing=json.loads(Path('reports/t06_efficiency_v1/timing_summary.json').read_text());fold=json.loads(Path('reports/t06_efficiency_v1/fold_consistency.json').read_text())
    profile=json.loads(Path('reports/t06_efficiency_v1/encoder_profile.json').read_text());eff=[];paired=[]
    for row in timing['summary']:
        eff.append({k:v for k,v in row.items() if k!='per_pair_round'})
        if not row['folded']:continue
        reference=next(r for r in timing['summary'] if r['mode']==row['mode'] and r['K']==row['K'] and r['precision']==row['precision'] and not r['folded'])
        ref={(r['pair_id'],r['round']):r['median_ms'] for r in reference['per_pair_round']};by_image={}
        for r in row['per_pair_round']:by_image.setdefault(r['pair_id'],[]).append(r['median_ms']-ref[(r['pair_id'],r['round'])])
        deltas=np.array([np.median(values) for values in by_image.values()])
        paired.append(dict(mode=row['mode'],K=row['K'],precision=row['precision'],original_image_pairs=len(deltas),rounds=3,
            median_paired_delta_ms=float(np.median(deltas)),paired_p10_ms=float(np.percentile(deltas,10)),paired_p90_ms=float(np.percentile(deltas,90)),
            images_faster=int((deltas<0).sum())))
    write_csv(out/'efficiency_nohook.csv',eff);write_csv(out/'efficiency_paired_by_image.csv',paired)
    comparison=[]
    for precision in ['FP32','BF16']:
        group=[r for r in fold['records'] if r['precision']==precision]
        comparison.append(dict(precision=precision,passed=sum(r['passed'] for r in group),checks=len(group),
            descriptor_max_abs=max(r['descriptor_max_abs'] for r in group),coarse_pair_Jaccard_min=min(r['coarse_Jaccard'] for r in group),
            matrix_probe_RMSE_max_px=max(r['matrix_probe_RMSE_px'] or 0 for r in group),formal_fold_timing_allowed=fold['global_gate'][precision]))
    write_csv(out/'BN_fold_consistency.csv',comparison)
    transitions=[]
    for k in p['budgets']:
        def flags(mode):
            return {r['pair_id']:r['evaluation']['success']['5'] for r in json.loads(Path(f'reports/t06_selection_v1/{mode}_K{k}.json').read_text())['pairs']}
        for left,right in [('U','A'),('A','AC'),('AC','AC-cap')]:
            x,y=flags(left),flags(right);transitions.append(dict(K=k,reference=left,variant=right,both_success=sum(x[i] and y[i] for i in x),
                lost_success=sum(x[i] and not y[i] for i in x),gained_success=sum(not x[i] and y[i] for i in x),both_failure=sum(not x[i] and not y[i] for i in x)))
    write_csv(out/'paired_registration_transitions.csv',transitions)
    # Metric plots only; raster inputs and predictions remain in private runs/local outputs.
    fig,axes=plt.subplots(1,3,figsize=(14,4));names=['U','A','AC','AC-cap'];x=np.arange(4)
    for k in p['budgets']:
        ys=[next(r for r in table if r['mode']==name and r['K']==k)['SR5']*100 for name in names];axes[0].plot(x,ys,'o-',label=f'K{k}')
    axes[0].set(xticks=x,xticklabels=names,ylabel='SR@5px (%)',title='Full fixed952 validation');axes[0].legend()
    for name,key in [('A','A'),('Capacity','capacity')]:axes[1].plot([0]+[r['epoch'] for r in heads[key]['history']],
        [heads[key]['initial_train_probe']['average_precision']]+[r['probe']['average_precision'] for r in heads[key]['history']],label=name)
    axes[1].set(xlabel='Epoch',ylabel='Average precision',title='Fixed128 TRAIN probe, not validation');axes[1].legend()
    for k in p['budgets']:
        axes[2].plot(x,[100*next(r for r in table if r['mode']==name and r['K']==k)['selected_query_correct5'] for name in names],'o-',label=f'K{k}')
    axes[2].set(xticks=x,xticklabels=names,ylabel='Selected raw correct@5px (%)',title='Prediction gain is not registration gain');axes[2].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'appearance_selection.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    classes=[r['name'] for r in teachers['T0']['best']['groups']['fixed64/gray_RGB']['per_class']]
    for j,version in enumerate(['T0','TR','TG']):
        m=teachers[version]['best']['groups']['fixed64/gray_RGB'];axes[0].bar(np.arange(7)+(j-1)*.25,[r['IoU'] for r in m['per_class']],.25,label=version)
    axes[0].set(xticks=np.arange(7),xticklabels=classes,ylabel='IoU',title='Fixed64 grayscale validation');axes[0].tick_params(axis='x',rotation=25);axes[0].legend()
    for version in ['TR','TG']:
        h=teachers[version]['summary']['history'];axes[1].plot([0]+[r['step'] for r in h],[teachers['T0']['best']['groups']['fixed64/gray_RGB']['mean_IoU']]+[r['gray_mIoU'] for r in h],'o-',label=version)
    axes[1].set(xlabel='Update',ylabel='Gray mIoU',title='Same1000 updates, stream and selection rule');axes[1].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'teacher_controls.png',dpi=160);plt.close(fig)
    # Curated provenance and all comparison records, never model tensors.
    artifact=dict(protocol=p,labels=labels,heads=heads,selection=table,teachers={v:dict(summary=t['summary'],selected_validation=t['best']) for v,t in teachers.items()},
        semantic_review=review,efficiency=eff,fold_consistency=comparison,paired_efficiency=paired,SC_status='waiting teacher finalization; target-domain omissions remain and checkpoint license unresolved',
        matched_stream_verified=True,semantic_weights_publicly_released=False,OS_semantic_accuracy_claim=False)
    (out/'experiment_summary.json').write_text(json.dumps(artifact,indent=2))
    (out/'label_schema.json').write_text(json.dumps(dict(shape=[7912,1024],geometric_correctness='int8:1 positive,0 negative,-1 ignored; only this target trains the scorer',
        ambiguity_margin='float32 logit best(correct<=5px)-best(wrong>=8px),NaN if not measurable; diagnostic only',
        predicted_candidate='int16:0..1023 SAR grid,1024 no-match',raw_SAR_candidate='best SAR cell before no-match',
        prediction_error_px='raw_SAR_candidate relative to forwardA; no-match stays a negative label when GT otherwise supervisable',
        optical_valid='source coarse cell valid',is_supervisable='exactly geometric_correctness>=0',
        GT_reliable='geometric eroded valid-region test only; not proof of exact original optical/SAR registration',
        features='FP32 promotion of frozen BF16 coarse64 and masked SAR GAP64; feature cache not a trainable backbone',
        dataset_split='7912 deduplicated OS train; no val/test scoring labels'),indent=2))
    lines=['# T06：纯外观选择、受控教师适配与效率诊断','',
        '日期：2026-11-26（Asia/Shanghai）。本轮匹配器仅训练评分头；独立SegFormer教师按RGB/灰度做继续训练对照。匹配骨干、描述子、温度、无对应参数、局部精化与RANSAC始终冻结，未使用OS测试集。', '',
        '## 实验边界与训练来源','',
        f"匹配检查点为T05第18轮，SHA256 `{p['matcher_sha256']}`。952对固定验证及T05的24对计时清单原样复用；K256为主、K512参照。U的两档矩阵/评价与T05逐pair完全复现。原匹配阈值0.01、RANSAC3px/min6/5000次/0.999置信度保持不变。",'',
        '标签仅来自7912对去重训练图，每对一个提前保存的同分布仿射；不使用val/test产生评分训练标签。geometry correctness与ambiguity margin分开保存，训练只读取前者。标签是原模型逐查询前向argmax（含no-match）的几何正确性，不等价于互选/阈值后生存或下游配准效用。', '',
        f"总查询{labels['coarse_queries']:,}；正{labels['label_counts']['positive']:,}（{labels['label_fractions']['positive']:.2%}）、负{labels['label_counts']['negative']:,}（{labels['label_fractions']['negative']:.2%}）、忽略{labels['label_counts']['ignored']:,}（{labels['label_fractions']['ignored']:.2%}）。",'',
        '正例要求已预测对应且误差≤5px；≥8px或可监督区域内预测no-match为负例；5—8px灰带忽略。GT近有效边界2px、无有效正确粗候选等位置忽略。5—8px是保守标签容差带，不代表原始配准残差已被测为3px；GT_reliable只说明附加仿射下的几何有效性，原始配准残差仍未知。', '',
        '分数间隔为正确候选（≤5px）最佳冻结logit减错误候选（≥8px）最佳logit；两侧无候选时用NaN并另存有效标志。不与几何标签任意加权混合。完整字段见label_schema.json；服务器逐字段npy及本地labels_and_diagnostics.npz保留正确/错误分数、误差、no-match、光学有效和可监督状态。', '',
        '评分头输入冻结光学粗特征64通道及SAR有效粗区域GAP64，仅用外观；无跨图相关输入。训练用缓存，优化器根本不实例化匹配骨干。FP32头（关闭AMP、cuDNN配置与主推理一致），AdamW lr0.001/WD0.0001、batch32、12epoch/2976更新、按训练正负数平衡的masked BCE，固定最后检查点，不按952验证选头。', '',
        f"128对固定训练子集300步：AP {heads['small']['initial_train_probe']['average_precision']:.4f}→{heads['small']['final_train_probe']['average_precision']:.4f}；全量头固定训练探针AP {heads['A']['initial_train_probe']['average_precision']:.4f}→{heads['A']['final_train_probe']['average_precision']:.4f}、AUC {heads['A']['initial_train_probe']['ROC_AUC']:.4f}→{heads['A']['final_train_probe']['ROC_AUC']:.4f}。这些是训练探针指标，不冒充验证泛化。",'',
        '## 固定K选择表','',
        'A为全局top-K；AC与A加载完全同一头，只增加4×4覆盖分配：每格基础2、上限32、Top8均值经softmax分配余量，确定性最大余数补齐，NMS半径0以免额外混入抑制因素。K512时上限使每格恰32。所有查询最终按行优先排序。', '',
        '正常推理严格为编码→评分→选择→gather→K×1024唯一跨图bmm→原精化→原RANSAC。离线训练标签可做全量相关，评分推理不提前偷算全量匹配。A/AC的稀疏bmm与停止梯度均有接口测试。', '',
        '外观头3489参数，总匹配网络40708；AC-cap宽29、7831头参数，总45050，接近当前v0.1 JointScore参考7977（头参数差1.83%）。AC-cap同样训练预算，是后续SC的容量对照准备，不能称作语义收益；若SC架构改变需重新核对参数。', '',
        '| K | 选择 | SR@3 | SR@5 | RMSE中位数/P90(px) | 求解失败 | 5px成功/失败 | 选中查询正确@5 | 实际精化入窗 | 内点凸包覆盖 |','| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        if r['mode']=='SC':lines.append(f"| {r['K']} | SC：等待教师 | — | — | — | — | — | — | — | — |");continue
        lines.append(f"| {r['K']} | {r['mode']} | {r['SR3']:.2%} | {r['SR5']:.2%} | {r['RMSE_median_solved_px']:.3f}/{r['RMSE_p90_solved_px']:.3f} | {r['solver_failures']}/952 | {r['successes5']}/{r['failures5']} | {r['selected_query_correct5']:.2%} | {r['actual_refinement_window_hit']:.2%} | {r['inlier_hull_fraction']:.2%} |")
    lines.extend(['',
        '选中查询正确率以全部K个原始前向预测为分母，预测no-match不计正确；精化入窗率以实际通过原阈值/互选并进入精化的对应为分母；覆盖是RANSAC内点源位置凸包面积/图像面积。统计为逐影像对等权均值，RMSE分位数仅对可求解样本，SR分母仍是952。', '',
        '判断：K256的A把选中查询正确率16.27%提高至28.73%，SR@5只提高2.10个百分点；AC低于A。K512时A/AC均低于U，AC仅较A略回升。当前覆盖规则未提供一致配准收益，评分预测改善不自动证明配准机制充分有效。容量更大的外观头也未带来稳定大幅提升；本轮为单seed固定预算，不作显著性结论。配对成败转移见paired_registration_transitions.csv。', '',
        '## T0／TR／TG教师对照','',
        '完整官方LoveDA Train2522张；原始Train.zip全包MD5通过、镜像5172个Train/新Val文件SHA通过。新Val64按城乡各32固定，排除旧8张；旧8继续独立纵向评价。新64用于共同选模规则，旧8和OS预览不用于选模。没有下载Test。', '',
        'TR/TG均从同一T04八通道B2权重开始，micro2×累积2=有效batch4，1000更新/4000样本呈现；原生512裁剪、512网络输入、AdamW lr2e-5/WD0.01、clip1、无调度器，BF16训练。除TG将相同裁剪转PIL灰度再复制三通道，没有其他增强差异。样本/裁剪计划SHA及裁剪标注流SHA完全相同，GT计数逐类相同。没有RGB/灰度混合第三支线。', '',
        'GT原生0通过损失mask转255忽略，不训练成真实地物；1—7不移位，教师仍保留8通道。每200步及step0验证，统一按新64灰度累计混淆矩阵七类mIoU选best、同分保留更早。两条均完成1000步，均选中400步；末步对照另列teacher_last_step_control.csv，未凭OS图或RGB额外指标换选模规则。评价按T0相同FP32存储/默认cuDNN配置执行，与效率诊断的严格FP32是不同对象。', '',
        '| 教师 | 完成更新 | 选中步 | 新64 RGB mIoU | 新64灰度mIoU | 旧8 RGB mIoU | 旧8灰度mIoU |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for version in ['T0','TR','TG']:
        t=teachers[version];g=t['best']['groups'];lines.append(f"| {version} | {t['summary']['steps']} | {t['summary']['best']['step']} | {g['fixed64/RGB']['mean_IoU']:.2%} | {g['fixed64/gray_RGB']['mean_IoU']:.2%} | {g['old8/RGB']['mean_IoU']:.2%} | {g['old8/gray_RGB']['mean_IoU']:.2%} |")
    lines.extend(['', '| 教师/新64输入 | Background | Building | Road | Water | Barren | Forest | Agricultural |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in teacher_rows:
        if r['group']=='fixed64':lines.append('| '+r['teacher']+'/'+r['input']+' | '+' | '.join(f"{r[c+'_IoU']:.2%}" for c in classes)+' |')
    lines.extend(['',
        'TG新64灰度mIoU比T0高14.41个百分点、比TR高13.98个百分点，说明并非仅额外训练造成。建筑、道路、裸地、森林和农田等类别改善，灰度各类仍有差异，未出现某类接近零的严重坍塌。RGB相对T0下降约3.06个百分点，但TR也下降约2.88，不能把全部RGB退化归因于灰度转换。未启动混合训练支线。64只是快速筛选，不能称官方全验证成绩或充分收敛。', '',
        '## 八通道接口与OS训练来源抽查','',
        '光学32张（保留原8+固定24张train）仅作目标域抽查，无人工语义GT。逐图随机置换P/Q/R版本名称的4页预测和4页置信度图、空白人工盲审表及私有键已交付；助手先隐藏版本查看并记录观察，再读键，不冒充人工GT审核。', '',
        '按固定“预测Background≥95%且平均最大概率≥90%”输出模式标记，T0/TR/TG为5/5/1张（分母32）；这不是语义错误率。TG在106裸地、3027/3567田块被误作大建筑等样例上更合理，但1500细屋顶漏分等仍存在；仅凭无GT视觉图不能确认类精度提升。SC本轮等待教师最终质量核验：源域灰度改善不自动等于目标域正式教师可用。', '',
        '接口保留a=1-pIgnore与七类条件分布；面积聚合后，可靠性r=1-H(p条件)/log7，监督权重w=valid_mask×a×r。valid_mask沿用有效覆盖、a≥0.8、条件最大概率≥0.6；未因本轮结果调阈值。加权KL对教师/权重detach，除以网格单元数而非sum(w)，避免把整体低有效质量归一化抵消；全空监督可反传零。', '',
        '| 教师 | 有效类质量a均值 | 语义监督mask比例 | 平均权重 | Background有效质量 | Building | Road | Water | Barren | Forest | Agricultural |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in mass:lines.append(f"| {r['teacher']} | {r['valid_class_mass']:.6f} | {r['mask_fraction']:.2%} | {r['mean_supervision_weight']:.4f} | "+' | '.join(f"{r[c]:.4f}" for c in classes)+' |')
    lines.extend(['',
        '上述每类质量为mean(w×p条件类)，总和等于平均监督权重；来自32张OS训练图预览，不是7912张正式语义缓存，也不是各类准确率。TG置信/掩码覆盖下降不能单独判断好坏。Ignore实际占比很低仍保留原概率，且高Ignore/高条件置信反例、全均匀不可靠分布及绝对权重KL均有测试。', '',
        '2026-11-26再次查询固定HF revision，仍无模型卡、license标签或权重许可文件；许可未解决，原始及派生权重均未公开。LoveDA CC-BY-4.0不等于社区权重授权。该初始化不作为最终可发布教师，后续需改用许可明确的初始化再确认适配结论；不宣称可自由部署。', '',
        '## 编码与推理效率诊断','',
        f"固定4对encoder分析：每对实际Conv2d调用{profile['Conv2d_calls_per_pair']:.0f}、BN {profile['BatchNorm_calls_per_pair']:.0f}、Python双线性上采样{profile['Python_interpolate_calls_per_pair']:.0f}；ReLU的ATen事件约56。AMP会出现嵌套分派，例如上采样8个ATen事件对应4次Python调用，不能当8个独立层。",'',
        f"该次profile约{profile['GPU_kernel_events_per_pair']:.0f}个CUDA事件/对，GPU kernel活跃时长合计约{profile['GPU_kernel_active_sum_ms_per_pair']:.3f}ms；另有大量CPU分派/launch。原始trace和算子/内核CSV保留。profiler会改变时序，不能用该活跃时长或嵌套CPU总时长代替正式端到端速度，只支持固定小算子/调度成本值得优化的判断。",'',
        'BN折叠按光学BN0/SAR BN1分别生成卷积副本，原共享训练权重不改；部署副本会复制卷积，参数存储量改变，不能仍按训练图共享数宣称。初次FP32存储检查仍允许cuDNN TF32卷积而失败；排查记录保留。随后将FP32检查明确为cuDNN/matmul均禁TF32，未放宽任何预先容差。', '',
        '| 精度 | 一致性通过 | 最大描述子差 | 最大矩阵探针RMSE差(px) | 折叠正式计时 |','| --- | ---: | ---: | ---: | --- |'])
    for r in comparison:lines.append(f"| {r['precision']} | {r['passed']}/{r['checks']} | {r['descriptor_max_abs']:.8g} | {r['matrix_probe_RMSE_max_px']:.8g} | {'允许' if r['formal_fold_timing_allowed'] else '未计时：门禁失败'} |")
    lines.extend(['',
        'FP32的192项=24对×2预算×4自定义基线，全部通过；BF16均未通过。BF16舍入位置/折叠权重量化改变及离散互选/RANSAC会放大差异，不能将数学上的BN等价当作低精度预测等价。FP32折叠只与同精度参照比较；它尚不能替代主表BF16推理，更不能把FP32折叠时延与BF16精度拼接成主方法结果。', '',
        '正式测量另用空闲独占GPU6，无profiler/无hook，沿用24对，3轮、每对每配置warmup5次和10次随机交错重复。每个原图按轮另算中位数，再做配对差值；24×3轮不当作72个独立场景。光学/SAR编码、评分/选择、gather、匹配、精化、CPU坐标及RANSAC全部包括，I/O/warp/H2D排除。GPU7的算子分析与正式测量分开。', '',
        '| K | 版本 | 原BF16中位/P90(ms) | 原严格FP32中位(ms) | FP32折叠中位/P90(ms) | 逐图配对中位差(ms) |','| ---: | --- | ---: | ---: | ---: | ---: |'])
    for k in p['budgets']:
        for mode in ['U','A','AC','AC-cap']:
            find=lambda precision,folded:next(r for r in eff if r['K']==k and r['mode']==mode and r['precision']==precision and r['folded']==folded)
            raw=find('BF16',False);base=find('FP32',False);opt=find('FP32',True);pair=next(r for r in paired if r['K']==k and r['mode']==mode)
            lines.append(f"| {k} | {mode} | {raw['median_ms']:.3f}/{raw['p90_ms']:.3f} | {base['median_ms']:.3f} | {opt['median_ms']:.3f}/{opt['p90_ms']:.3f} | {pair['median_paired_delta_ms']:.3f} |")
    lines.extend(['',
        '一致通过的FP32优化对U/A/AC/容量对照及两个K全部采用，没有只给某个方法加速。原始逐次无hook数据、按原图的配对差值与每轮中位数均交付。主模型的BF16折叠失败被如实保留，不汇报其“更快但不等价”的速度。编码优化结论受当前5090/256输入/软件实现约束。', '',
        '## 交付与状态','',
        '固定K表fixed_K_selection.csv、教师逐类teacher_comparison.csv、末步teacher_last_step_control.csv、编码器encoder_operators.csv/encoder_GPU_kernels.csv、BN折叠一致性BN_fold_consistency.csv、无hook效率efficiency_nohook.csv及原图配对efficiency_paired_by_image.csv齐备；experiment_summary.json保留所有来源与指标。', '',
        '训练标签仅train，语义抽查仅OS train；测试及scene人工控制点均未用于选择/调参。匹配骨干未解冻、无语义评分训练、无RGB/灰度混合第三教师支线、无动态预算或复杂精化。SC等待教师正式冻结，AC-cap容量对照已训练；后续语义辅助但不参与选择的骨干适配仍单列暂缓。', '',
        '代码/配置/非像素指标同步本地、zzvr与GitHub。标签、冻结特征、私有权重、原始/预测影像及profile trace留本地outputs/服务器runs，不随Git公开。逐字段说明、复现命令和跨文件验收另见label_schema.json、docs/T06_REPRODUCTION_zh.md及acceptance.json。'])
    (out/'T06实验报告.md').write_text('\n'.join(lines)+'\n')
    for src,name in [('runs/t06_reliability_v1/summary.json','reliability_label_statistics.json'),('reports/t06_setup_v1/tests.txt','tests.txt'),('reports/t06_setup_v1/tests.xml','tests.xml')]:
        if Path(src).exists():shutil.copyfile(src,out/name)
    print('T06 curated tables/report written')


if __name__=='__main__':main()
