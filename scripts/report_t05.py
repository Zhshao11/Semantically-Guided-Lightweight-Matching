"""Post-hoc tables/figures and success/failure analysis; no model or training imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import write_csv
from sglm.geometry import transform_points


def mean(values):
    values=[v for v in values if v is not None];return float(np.mean(values)) if values else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='reports/t05_results_v1')
    ap.add_argument('--frozen',default='reports/t05_frozen_v1');ap.add_argument('--teacher',required=True)
    ap.add_argument('--output',default='reports/t05_delivery_v1');a=ap.parse_args()
    root=Path(a.results);frozen=Path(a.frozen);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((frozen/'freeze.json').read_text());architecture=json.loads((frozen/'architecture.json').read_text())
    accuracy=json.loads((root/'accuracy_summary.json').read_text());timing=json.loads((root/'timing/timing.json').read_text())
    teacher=json.loads(Path(a.teacher).read_text());reproduction=json.loads((root/'candidate_reproduction.json').read_text())
    assert reproduction['passed'] and timing['checkpoint_sha256']==accuracy['checkpoint_sha256']==freeze['checkpoint_sha256']
    combined=[];diagnostics={};group_rows=[];transition_rows=[]
    validations={k:json.loads((root/f'K{k}_validation.json').read_text())['pairs'] for k in freeze['protocol']['budgets']}
    full={r['pair_id']:r for r in validations[1024]}
    for row in accuracy['summaries']:
        k=row['K'];time=next(t for t in timing['summary'] if t['K']==k)
        combined.append(dict(K=k,SR3=row['success_rate@3px'],SR5=row['success_rate@5px'],
            RMSE_median_solved_px=row['solved_median_rmse_px'],RMSE_p90_solved_px=row['solved_p90_rmse_px'],
            solver_failures=row['failed_pairs'],evaluation_pairs=row['image_pairs'],
            refinements_mean_full_val=row['mean_attempted_refinements'],refinements_mean_timing_subset=time['mean_attempted_refinements'],
            latency_median_ms=time['end_to_end_median_ms'],latency_p90_ms=time['end_to_end_p90_ms'],
            mean_pair_median_ms=time['mean_pair_median_ms'],peak_allocated_MiB=time['peak_allocated_MiB'],
            **time['stages_median_ms']))
        d=json.loads((root/f'K{k}_diagnostics.json').read_text());diagnostics[k]=d
        for record in d['pairs']:
            p=record['predictions'];truth=transform_points(np.array(p['source'],float).reshape(-1,2),record['A_forward'])
            errors=np.linalg.norm(np.array(p['target'],float).reshape(-1,2)-truth,axis=1)
            idx=record['geometry']['estimation']['inlier_indices'];ie=errors[idx]
            record['posthoc_inlier_GT']=dict(correct_fraction5=float((ie<5).mean()) if len(ie) else None,
                median_error_px=float(np.median(ie)) if len(ie) else None)
        for name,flag in [('success@5px',True),('failure@5px',False)]:
            group=[r for r in d['pairs'] if r['evaluation']['success']['5']==flag]
            row2=dict(K=k,group=name,pairs=len(group),
                GT_visible_prediction_fraction=mean([r['coarse']['GT_visible_predictions']/r['coarse']['predicted_correspondences'] for r in group if r['coarse']['predicted_correspondences']]),
                coarse_correct3=mean([r['coarse']['correctness']['3']['fraction_of_predictions'] for r in group]),
                coarse_correct5=mean([r['coarse']['correctness']['5']['fraction_of_predictions'] for r in group]),
                coarse_correct8=mean([r['coarse']['correctness']['8']['fraction_of_predictions'] for r in group]),
                truth_inside_window=mean([r['local_window']['truth_inside_fraction'] for r in group]),
                inliers_median=float(np.median([r['geometry']['estimation']['inliers'] for r in group])) if group else None,
                inlier_hull_median=float(np.median([r['geometry']['inlier_source_coverage']['hull_area_fraction'] for r in group])) if group else None,
                inlier_GT_correct5=mean([r['posthoc_inlier_GT']['correct_fraction5'] for r in group]),
                inlier_GT_median_error=mean([r['posthoc_inlier_GT']['median_error_px'] for r in group]),
                near_collinear_inlier_sets=sum(r['geometry']['inlier_source_coverage']['near_collinear'] for r in group),
                coarse_fit_successes=sum(r['coarse_transform_evaluation']['success']['5'] for r in group))
            for support in ('truth_inside_window','truth_outside_window'):
                for phase in ('before','after'):
                    row2[support+'_'+phase+'_mean_px']=mean([r['refinement'][support][phase]['mean_px'] for r in group])
                row2[support+'_worsened_fraction']=mean([r['refinement'][support]['worsened_fraction'] for r in group])
            group_rows.append(row2)
        old_flags=[full[r['pair_id']]['evaluation']['success']['5'] for r in validations[k]]
        new_flags=[r['evaluation']['success']['5'] for r in validations[k]]
        transition_rows.append(dict(K=k,both_success=sum(x and y for x,y in zip(old_flags,new_flags)),
            full_success_sparse_failure=sum(x and not y for x,y in zip(old_flags,new_flags)),
            full_failure_sparse_success=sum(not x and y for x,y in zip(old_flags,new_flags)),
            both_failure=sum(not x and not y for x,y in zip(old_flags,new_flags))))
    write_csv(out/'budget_comparison.csv',combined);write_csv(out/'success_failure_diagnostics.csv',group_rows)
    write_csv(out/'paired_full_validation_transitions.csv',transition_rows)
    paired_timing=[]
    base={p['pair_id']:p['median_ms'] for p in timing['summary'][0]['per_pair']}
    for row in timing['summary']:
        delta=np.array([p['median_ms']-base[p['pair_id']] for p in row['per_pair']])
        paired_timing.append(dict(K=row['K'],pair_count=len(delta),median_pair_delta_ms=float(np.median(delta)),
            p10_pair_delta_ms=float(np.percentile(delta,10)),p90_pair_delta_ms=float(np.percentile(delta,90)),
            mean_pair_delta_ms=float(delta.mean())))
    write_csv(out/'paired_timing_deltas.csv',paired_timing)
    # Figures contain metrics only, no dataset imagery.
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for r in combined:
        axes[0].scatter(r['latency_median_ms'],100*r['SR5'],s=55,label=f"K={r['K']}")
    axes[0].set(xlabel='No-hook end-to-end median (ms)',ylabel='Full validation SR@5px (%)',title='Frozen-weight budget screening')
    axes[0].set_xlim(0,1.15*max(r['latency_median_ms'] for r in combined));axes[0].set_ylim(0,65)
    axes[0].legend(loc='lower left')
    x=np.arange(4);axes[1].plot(x,[r['peak_allocated_MiB'] for r in combined],'o-');axes[1].set(xticks=x,xticklabels=[r['K'] for r in combined],xlabel='Query budget',ylabel='Peak allocated MiB',title='Batch1 inference memory')
    axes[1].set_ylim(0,65)
    for k in [1024,512,256,128]:
        errors=np.sort([r['evaluation']['rmse_px'] for r in validations[k] if r['evaluation']['solved']]);axes[2].plot(errors,np.arange(1,len(errors)+1)/952,label=str(k))
    axes[2].set(xlim=(0,25),xlabel='Independent-probe RMSE (px)',ylabel='Fraction of all 952 pairs',title='Failures remain in denominator');axes[2].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'budget_screen.png',dpi=160);plt.close(fig)
    d=diagnostics[1024]['pairs'];fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,title,extract in zip(axes,['Coarse correct@5px','GT inside target window','Inliers','Inlier hull fraction'],
        [lambda r:r['coarse']['correctness']['5']['fraction_of_predictions'],lambda r:r['local_window']['truth_inside_fraction'],
         lambda r:r['geometry']['estimation']['inliers'],lambda r:r['geometry']['inlier_source_coverage']['hull_area_fraction']]):
        ax.boxplot([[extract(r) for r in d if r['evaluation']['success']['5']==flag] for flag in [True,False]],tick_labels=['Success (48)','Failure (48)'])
        ax.set_title(title);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Preselected 96 pairs, K=1024; groups defined by final SR@5px');fig.tight_layout();fig.savefig(out/'success_failure_layers.png',dpi=160);plt.close(fig)
    rgb=teacher['summary']['RGB'];gray=teacher['summary']['gray_RGB'];fig,ax=plt.subplots(figsize=(10,4));x=np.arange(7)
    ax.bar(x-.18,[r['IoU'] for r in rgb['per_class']],.36,label='RGB');ax.bar(x+.18,[r['IoU'] for r in gray['per_class']],.36,label='Gray replicated RGB')
    ax.set(xticks=x,xticklabels=[r['name'] for r in rgb['per_class']],ylabel='IoU',title='LoveDA fixed8 validation: same weights and512 preprocessing, GT0 ignored');ax.legend();ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'teacher_source_class_IoU.png',dpi=160);plt.close(fig)
    outcome=dict(checkpoint_sha256=freeze['checkpoint_sha256'],budget_comparison=combined,diagnostic_groups=group_rows,
        paired_transitions=transition_rows,paired_timing_deltas=paired_timing,
        source_teacher_RGB=rgb,source_teacher_gray=gray,source_teacher_mIoU_delta_pp=100*(gray['mean_IoU']-rgb['mean_IoU']),
        posthoc_only=True,no_training=True,test_used=False)
    (out/'analysis.json').write_text(json.dumps(outcome,indent=2))
    lines=['# T05：冻结候选、均匀稀疏初筛与教师源域对照','',
        '日期：2026-11-26（Asia/Shanghai）。本轮无训练；匹配阈值、局部精化策略及RANSAC参数保持冻结配置。只评价952对固定验证，不读取OS测试影像。', '',
        '## T05-A 候选冻结与逐对复核','',
        f"固定主运行epoch18、step35604。检查点SHA256：`{freeze['checkpoint_sha256']}`。独立副本在服务器 `{freeze['frozen_checkpoint']}`，原best保留。",'',
        f"训练代码版本：`{freeze['training_code_revision']}`；本轮准备基线：`{freeze['preparation_base_revision']}`。原B0骨干、B0模型/损失、几何和评价器源码与训练版本逐字节相同；本轮新增入口按运行源码SHA记录。",'',
        f"配置SHA：`{freeze['candidate_config_sha256']}`；固定验证计划SHA：`{freeze['validation_plan_sha256']}`；预选子集SHA：`{freeze['subsets_sha256']}`。冻结证据位于 `../t05_frozen_v1/`。",'',
        '952对全量入口与原B0预测逐项bitwise相同，与历史epoch18逐pair的预测矩阵、RMSE和成功标志完全相同；最大RMSE差为0。所有模块eval，光学BN0/SAR BN1，BN缓冲区未变化。不存在为改善结果切换train的情况。', '',
        f"复核SR@3px={combined[0]['SR3']:.6%}（332/952），SR@5px={combined[0]['SR5']:.6%}（520/952）；可求解RMSE中位数/P90={combined[0]['RMSE_median_solved_px']:.6f}/{combined[0]['RMSE_p90_solved_px']:.6f}px；求解失败0/952，5px精度失败432/952。",'',
        '诊断96对、计时24对均在预测前由光学亮度/对比度8层确定，随机种子20261126；计时样本来自24个不同的已验证512内容父图。已查看输入，覆盖建筑密集区、工业区、道路、农田及混合地物；不把内容父图或亮度层当作已知卫星原始scene。诊断子集SR@5为50%，低于全验证54.62%，因此不以96对代替952对精度结论。', '',
        '### 实际架构与参数','',
        '实际严格加载的B0为37,219参数。卷积权重按共享只计一次，两个模态的BN仿射参数都计入；running mean/var及计数器是buffer，不计入参数。逐叶层、权重尺寸及完整repr见 `../t05_frozen_v1/architecture.json`、`actual_layers.txt`。', '',
        '| 非重叠模块 | 参数量 |','| --- | ---: |']
    lines.extend(f'| `{name}` | {count:,} |' for name,count in architecture['components_nonoverlapping'].items())
    lines.extend(['| **合计** | **37,219** |','',
        '与v0.1骨干规格核对：两独立Conv3×3/s2适配器（3→16、1→16）；DSRes16×2/32×2/64×3；DW3×3/s2+PW下采样；共享卷积、独立模态BN；三层Add-FPN，横向16→16/32→32/64→64，上采样64→32及32→16，bilinear且align_corners=false；共享Conv64→32描述子。DSRes为DW-BN-ReLU-PW-BN-残差-ReLU，符合骨干图。', '',
        '实测粗特征1×64×32×32，细特征1×16×128×128，描述子1×1024×32，全量相关1×1024×1024，均符合T02/B0规格。实际B0精化是共享Linear16→32、光学单点与SAR7×7余弦相关、可学习温度及soft-argmax；没有源3×3聚合、交叉注意力或可靠性头。后者属于完整v0.1模块C，按B0协议明确关闭；语义/联合评分/动态预算也未实例化。因此不能把37,219参数的B0性能称为完整语义方法性能。', '',
        '### 成功与失败分层比较','',
        '96对按最终5px结果分为48成功/48失败；下表正确率与入窗率是逐影像对等权均值，内点/覆盖为组内中位数。正确对应指相对合成A的像素误差，不是人工精确标注；原始光学/SAR配准残差假设仍成立于该监督定义。', '',
        '| K=1024分组 | 粗正确@3/5/8 | 真值入7×7窗口 | 窗口内误差前→后 | 内点中位数 | 内点凸包覆盖中位数 |','| --- | --- | ---: | --- | ---: | ---: |'])
    for r in [r for r in group_rows if r['K']==1024]:
        lines.append(f"| {r['group']}，n={r['pairs']} | {r['coarse_correct3']:.2%}/{r['coarse_correct5']:.2%}/{r['coarse_correct8']:.2%} | {r['truth_inside_window']:.2%} | {r['truth_inside_window_before_mean_px']:.3f}→{r['truth_inside_window_after_mean_px']:.3f}px | {r['inliers_median']:.1f} | {r['inlier_hull_median']:.2%} |")
    lines.extend(['',
        '主要失败仍在粗对应质量/入窗率，且低覆盖、错误但自洽的内点对几何估计有影响；证据不足以归因为“主要已经转移到精化”。失败组约70%预测的真值不在局部支持窗内，窗口外平均误差约43→42.6px，局部±6px读出无法修正大范围错配。成功/失败组GT可见预测比例约99.39%/97.95%，因此入窗差异主要并非越界mask造成。精化对窗口内对应平均有益，但失败组约38.9%的窗口内对应仍变差，不能宣称精化完全正常。成功/失败组内点的GT正确@5比例分别约88.45%/50.37%；两组均未出现近共线内点集，说明本子集的问题不主要是严格秩退化，内点自洽也不保证是真对应。', '',
        '在同一预测对应集合上事后比较粗坐标拟合与精化坐标拟合：27对两者都成功，21对由粗失败转为精化成功，4对由粗成功转为精化失败，44对两者都失败。净提升17/96；4个回退案例需要检查局部读出及RANSAC内点重排。所有GT分析发生在预测和几何拟合之后，不进入查询、细化或RANSAC决策。', '',
        '完整组别的窗口内/外精化误差、错误比例、近共线性、内点GT正确率及所有K结果见 `success_failure_diagnostics.csv`；对照图见 `success_failure_layers.png`。内点数不等于正确对应数，后者单独事后计算。', '',
        '## T05-B 四档均匀查询初筛','',
        '同一检查点，无重训、无语义评分。1024为32×32，512为16×32，256为16×16，128为8×16；各轴选floor((i+0.5)×32/网格尺寸)，row-major固定。K<1024在唯一跨图bmm前执行index_select；不存在隐藏的全量反向相关。K=1024保留原始描述子视图以严格复现原B0。', '',
        '互选仍按同一argmax/no-match规则，反向竞争集合限制为所选查询。候选集合缩小是稀疏化的内在变化，因此512的微小SR提升不能被解释为已训练出更强稀疏能力。所有阈值及完整6自由度仿射RANSAC均不变。', '',
        '| K | SR@3px | SR@5px | RMSE中位数/P90（px） | 实际精化数均值 | 端到端中位数/P90（ms） | 峰值分配显存（MiB） | 求解失败 |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in combined:lines.append(f"| {r['K']} | {r['SR3']:.2%} | {r['SR5']:.2%} | {r['RMSE_median_solved_px']:.3f}/{r['RMSE_p90_solved_px']:.3f} | {r['refinements_mean_full_val']:.2f} | {r['latency_median_ms']:.3f}/{r['latency_p90_ms']:.3f} | {r['peak_allocated_MiB']:.2f} | {r['solver_failures']}/952 |")
    lines.extend(['',
        '精度及精化数来自全部952对；RMSE分位数只在可求解样本上计算（前三档952，128档949），失败仍留在SR分母。计时为预定24对，每对每K先5次warmup，再20次无hook端到端测量；K顺序按种子交错。时间包含编码、gather/粗匹配、精化、坐标传回CPU及RANSAC，输入已在GPU，排除读取、warp、host-to-device和GT评价。不是包含文件I/O的系统总延迟。', '',
        f"硬件为独占物理GPU6 `{timing['gpu_uuid']}`（RTX5090），torch{timing['torch']}、CUDA{timing['cuda']}、BF16、batch1、torch CPU线程4、TF32关闭、cuDNN deterministic。开始/结束及每张图前后检查同卡无其他CUDA进程；无hook确认通过。24对/20次重复是相关测量，不当作480个独立场景。",'',
        '峰值为torch.max_memory_allocated，包含模型、当前单对输入及临时张量。交错运行时reserved缓存峰值均82 MiB，单列保存在原始结果，不用缓存保留量声称内存节省。', '',
        '| K | 编码（ms） | 粗匹配含解码（ms） | 精化（ms） | CPU坐标拷贝（ms） | RANSAC（ms） |','| ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in combined:lines.append(f"| {r['K']} | {r['encode_ms']:.3f} | {r['coarse_ms']:.3f} | {r['refinement_ms']:.3f} | {r['coordinate_copy_ms']:.3f} | {r['ransac_ms']:.3f} |")
    lines.extend(['',
        '这些阶段是独立同步计时，中位数不能相加当作整体效率；整体只采用上表无hook端到端值。逐pair中位数的均值约8.26—8.30ms，四档基本持平；paired_timing_deltas.csv提供逐pair差值，避免只看池化中位数的小差异。', '',
        '判断：相关张量从1024×1024降至128×1024、精化负载减少、峰值分配显存下降约27.4%，但本实现/256输入下未观察到有意义的端到端加速。编码保持约5.65ms，粗匹配/精化阶段也基本持平，符合固定算子/调度成本占比较高的表现；没有CUDA kernel profiler证据前，不把它当作已证明的底层瓶颈归因。', '',
        '值得继续考察的初筛区间是K512—256：512保留当前精度，256以约2.94个百分点SR@5损失换取更多显存下降；当前没有延迟收益。K128损失约12.61个百分点、P90变大且3次求解失败，不作为优先预算。若后续目标是速度，应先核查编码器和小算子的实际成本；若继续语义选点，需证明同预算下精度/覆盖收益。此结论仅为无需重训的均匀初筛，不是稀疏适配后的最终能力。', '',
        '96对同一诊断集合中，K从1024到128，粗正确@5均值28.12%→20.30%，入窗37.79%→28.36%，内点凸包平均56.96%→39.04%；精化在窗口内仍平均改善约0.50px。精度损失主要伴随有效粗对应及空间约束减少，K128还出现求解失败，不能只归因为精化。完整952对的成功/失败转移见paired_full_validation_transitions.csv。', '',
        '## T05-C 教师源域RGB/灰度对照','',
        '官方LoveDA Val.zip按固定seed取Rural4/Urban4，共8张1024影像及对应标签；选样早于教师预测。通过官方HTTP byte-range取成员，逐成员ZIP CRC及SHA验证，未下载Train/Test，未声称验证整包MD5。样本清单和各文件SHA嵌在teacher_source_control.json；数据许可CC-BY-4.0。', '',
        f"与T04完全同一B2权重，SHA `{teacher['weights_sha256']}`。RGB与PIL灰度后复制RGB均使用保存的512双线性/ImageNet预处理，FP32 eval；logits恢复native1024再读出，不缩放GT。OS仍为此前8张训练光学图、native256复制RGB，同样512输入。相同网络输入尺寸不意味着两数据集物理地面分辨率相同。",'',
        '先核验原生GT合法值域0—7，本批实际出现1—7；0=no-data/ignore不计入GT分母，1—7与教师配置Background至Agricultural一一对应，不做reduce_labels。若预测0落到有效GT上，仍计入错误/FN，绝不删掉这些像素改善指标。8×8混淆矩阵的GT0行固定为0，预测0列保留；本批没有真实GT0，忽略边界另由含0号的合成接口测试验证。', '',
        '| LoveDA固定8张 | OA | 七类mIoU | 有效GT上预测Background占比 |','| --- | ---: | ---: | ---: |',
        f"| 原RGB | {rgb['overall_accuracy']:.2%} | {rgb['mean_IoU']:.2%} | {rgb['predicted_background_fraction_on_valid_GT']:.2%} |",
        f"| 同图灰度复制RGB | {gray['overall_accuracy']:.2%} | {gray['mean_IoU']:.2%} | {gray['predicted_background_fraction_on_valid_GT']:.2%} |",'',
        '| 类别 | RGB IoU | 灰度IoU |','| --- | ---: | ---: |'])
    for x,y in zip(rgb['per_class'],gray['per_class']):lines.append(f"| {x['name']} | {x['IoU']:.2%} | {y['IoU']:.2%} |")
    os_bg=mean([r['raw_prediction_fraction']['Background'] for r in teacher['OS_train_diagnostic']])
    lines.extend(['',
        f"配对mIoU下降{100*(rgb['mean_IoU']-gray['mean_IoU']):.2f}个百分点，说明在这批样本上RGB源域有可用能力、丢失颜色会明显损伤分割。OS8张平均预测Background占比约{os_bg:.2%}（全像素、逐图等权），但没有语义GT，仅作输出分布与人工检查，不能用该比例计算OS准确率。源域与目标域背景统计支持域差异检查，不能用背景占比单独确认错误。",'',
        '下一步优先保留同一教师架构，使用LoveDA训练数据的灰度增强做适配，再复核冻结RGB/灰度验证对照；不先换模型。本轮未执行教师训练，也不能凭8张诊断估计完整LoveDA官方验证性能。即使解决颜色依赖，OS分辨率、地域及影像处理差异仍需目标域抽查或有限标注支持。', '',
        '2026-11-26重新核对固定HF revision：仅.gitattributes/config/preprocessor/权重四个文件，无模型卡、license标签或明确权重许可，状态仍未解决。LoveDA的CC-BY-4.0及Transformers框架许可均不自动授权社区派生权重；不称其可自由发布/部署。上游SegFormer许可背景继续单独保留。', '',
        '## 交付文件与限制','',
        '本目录包括预算表、组别诊断表、逐pair成功转移、配对计时差值、analysis.json和三张指标图。../t05_frozen_v1保存冻结依据/架构/清单，../t05_results_v1保存952×4逐pair验证、96×4分层诊断、全量复现与计时原始记录。教师指标及许可核查记录随报告版本化；原始/预测影像、源域数据及所有权重留在本地outputs与服务器runs，不随Git发布。新增8项接口测试及全仓库77项测试通过，详见tests.txt/xml；跨文件验收见acceptance.json。', '',
        '所有结果基于实际37,219参数的B0、当前合成仿射和原始配准残差较小的假设。未使用测试影像/人工scene控制点选模型；未生成最终可靠性标签，未训练评分器，未修改基础骨干/损失。'])
    (out/'T05实验报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparison=combined,groups=group_rows,paired_transitions=transition_rows,paired_timing_deltas=paired_timing),indent=2))


if __name__=='__main__':main()
