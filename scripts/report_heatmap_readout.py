"""Summarize a falsifiable fixed readout hypothesis; no automaticstrategy selection or training."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def rows(path):return list(csv.DictReader(Path(path).open()))


def main():
    p=json.loads(Path('configs/heatmap_readout_v1.json').read_text());out=Path(p['output']);a=json.loads((out/'acceptance.json').read_text());assert a['passed']
    diag=json.loads((out/'heatmap_diagnostics.json').read_text());distributions=json.loads((out/'correspondence_distributions.json').read_text());inside={r['branch']:r for r in distributions if r['subset']=='inside_actual_support'}
    summary={r['branch']:r for r in a['summaries']};labels={'A':'原全窗口期望','B':'离散最大峰','C':'最大峰附近3×3局部期望'}
    table=['| 分支 | SR@3px | SR@5px | 独立探针RMSE中位/P90（px） | 未达5px对数 | 求解失败数 |','|---|---:|---:|---:|---:|---:|']
    for g,r in summary.items():table.append(f"| {g}：{labels[g]} | {100*r['success_rate@3px']:.2f}% | {100*r['success_rate@5px']:.2f}% | {r['solved_median_rmse_px']:.3f} / {r['solved_P90_RMSE_px']:.3f} | {r['registration_failures5']} | {r['failed_pairs']} |")
    changes=['| 对照 | 阈值 | 失败→成功 | 成功→失败 | 失败→失败 | 成功→成功 | 净SR变化 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in a['paired_changes']:changes.append(f"| {r['comparison']} | {r['threshold_px']}px | {r['failure_to_success']} | {r['success_to_failure']} | {r['failure_to_failure']} | {r['success_to_success']} | {100*r['delta_SR']:+.2f}个百分点 |")
    point_table=['| 同45974条窗口命中对应 | 均值（px） | 中位（px） | P90（px） | <3px比例 | <5px比例 |','|---|---:|---:|---:|---:|---:|']
    for g,r in inside.items():point_table.append(f"| {g}：{labels[g]} | {r['mean_px']:.3f} | {r['median_px']:.3f} | {r['p90_px']:.3f} | {100*r['fraction_below']['3']:.2f}% | {100*r['fraction_below']['5']:.2f}% |")
    condition=rows(out/'conditional_diagnostics.csv');condition_by={r['group']:r for r in condition};cond_table=['| 仅事后分组 | 数量 | A均值（px） | B均值（px） | C均值（px） |','|---|---:|---:|---:|---:|']
    names={'peak_within3px':'最大峰误差<3px','peak_outside3px':'最大峰误差≥3px','competitive_secondary_peak':'存在规定的竞争次峰','no_competitive_secondary_peak':'无规定的竞争次峰'}
    for r in condition:cond_table.append(f"| {names[r['group']]} | {r['count']} | {float(r['A_mean_error_px']):.3f} | {float(r['B_mean_error_px']):.3f} | {float(r['C_mean_error_px']):.3f} |")
    diagnostic_table=['| 诊断量（同45974条） | 均值 | 中位 | P90 |','|---|---:|---:|---:|']
    for key,label in [('peak_GT_distance','最大峰距名义GT（px）'),('truth_mass','名义GT附近3×3原概率质量'),('peak_mass','最大峰附近3×3原概率质量'),('peak_probability','主峰原概率'),('secondary_ratio','主峰3×3以外次峰/主峰概率比'),('normalized_entropy','有效窗口归一化熵')]:
        v=diag[key];diagnostic_table.append(f"| {label} | {v['mean']:.4f} | {v['quantiles']['50']:.4f} | {v['quantiles']['90']:.4f} |")
    movers=rows(out/'mover_groups.csv');mover_table=['| C相对A的5px分组 | 对数 | 命中数量中位 | 逐对命中A误差均值的中位 | 对应C值 |','|---|---:|---:|---:|---:|']
    labels_move={'rescued5':'失败→成功','lost5':'成功→失败','stable_failure5':'仍失败','stable_success5':'仍成功'}
    for r in movers:
        if r['comparison']=='C_vs_A':mover_table.append(f"| {labels_move[r['group']]} | {r['pairs']} | {float(r['inside_count_median']):.0f} | {float(r['inside_A_error_median']):.3f}px | {float(r['inside_new_error_median']):.3f}px |")
    freeze=json.loads((out/'freeze.json').read_text());T=freeze['temperature']
    conclusion=dict(decision='retain_A_do_not_adopt_B_or_C',reason='CnetSR5+2pairs but45974inside-localizationworse, probeP90worse and116rescuedvs114lost; Bdegrades. Maximumpeak oftenwrong ratherthanreliablycorrectbutmisread.',
        supports_readout_only_hypothesis=False,primary_seed=a['paired_seed'],validation_only=True,significance_not_established=True,
        no_further_readout_temperature_window_or_threshold_search=True,no_training_started=True,no_online_timing_in_this_run=True,
        next_research_scope='If separately authorized later, investigate localfeaturediscrimination/localizationsupervision; testfuturefrozenconfiguration on previouslyunselectedtestsplit.')
    (out/'decision.json').write_text(json.dumps(conclusion,indent=2))
    text=f'''# 冻结热图，只改变坐标读出：A/B/C受控结果

本轮不建议将原读出A替换为B或C。C的SR@5仅净增2对（116救回、114退化，+0.21个百分点），窗口内定位误差反而增大，独立探针P90也变差；B整体退化。结果更提示最大峰本身常不可靠，尚不支持“正确位置已稳定成为最大峰，只是全窗口期望读错了”这个低成本解释。

结论限于这三种预先固定读出；本轮保留A，不继续扫描温度、阈值或邻域大小，不启动新训练。由于“局部定位与整体SR共同改善”的前提未满足，没有追加在线延迟/显存实验，也不把缓存读出的耗时当部署性能。

## 1. 实际实现、冻结条件和三种读出

实际原代码sglm/t02/model.py的LocalCorrelation为：共享投影、归一化点积、原温度softmax，得到所有有效位置上的热图，然后`target=center+2*(heatmap@delta)`。它确实采用全窗口期望。原radius=3、细采样步长2px；49个采样位置中原无效位置的概率为零。

复用上一轮952对冻结候选缓存，原T05 epoch18权重、均匀K256、122881条最终拟合对应的ID/数量/顺序/实际源坐标/分数/过滤结果完全不变。实际窗口命中集合仍为45974条，有效凸包与图像mask均继承原冻结判定；没有使用旧46090条矩形命中集合。没有重新跑网络、更新权重或读取测试集。

- A：保存的原始预测目标坐标，同时从原热图重放原BF16期望运算，122881条目标逐位一致，952对矩阵/评价精确复现原T05K256。
- B：对原有效位置取热图最大值，精确并列时按原行优先顺序取第一格；返回该格的原图采样坐标。
- C：使用B同一个最大峰，以该格为中心取原窗口内固定3×3细采样格，和原有效mask相交；仅把该集合的原概率重新归一化，然后使用原center+2期望读出。一步细格=2个原图像素，3格跨度为4px；不是原图3×3像素，也不扩出原窗口。边界截断，原无效格不进入局部期望。

三种读出对所有122881条拟合对应统一执行，绝不只处理GT命中位置。代码read_frozen_heatmap的接口没有真值参数。原热图是FP32缓存，温度参数固定为log={T['log_temperature']:.6f}、exp={T['exp']:.6f}、实际clamp后={T['used_clamped']:.6f}；没有重新softmax或修改温度。A/C保留原BF16 autocast矩阵乘的坐标读出精度，未同时改变精度制造对照差异。

A/B/C都独立调用原完整仿射RANSAC，每对/每分支重置为原seed={a['paired_seed']}，配置、迭代数、最少内点、score过滤和失败处理不变；内点各自重新计算。没有恢复候选、重排或修改confidence，没有按新目标坐标重做mask/误差过滤。独立16×16有效探针不加入拟合。

## 2. 全部952对主结果

{chr(10).join(table)}

SR按每对独立探针RMSE严格<阈值计算，全部952对保留分母。三种读出均求解成功952对，求解失败数为0，但未达5px仍分别460/477/458对。RMSE中位/P90统计所有成功求解对，不仅统计SR成功子集。

A的SR@5=492/952；B=475/952；C=494/952。A的SR@3=310/952；B=294/952；C=315/952。C中位误差小降0.065px，但P90从13.831升至14.190px、平均探针RMSE从6.799升至6.917px；不能只挑中位数或净2对宣称机制成立。

{chr(10).join(changes)}

成败同时改变的样本很多，而净变化很小。没有挑选RANSAC种子、重复运行直至得到正差或改用每对最有利读出。主结果只使用原配对随机设置；没有新模型训练种子。

## 3. 同45974条实际有效窗口命中对应的逐点结果

{chr(10).join(point_table)}

这里的GT是原已配准影像加合成A得到的名义几何对应，继承原始配准残差，不是独立人工物理真值。B误差就是最大峰到该名义GT的欧氏距离。固定集合相同，因此定位分布可直接比较，没有前后换样本。

最大峰距GT<3px的只有{diag['peak_within3px_count']}/45974（{100*diag['peak_within3px_count']/45974:.2f}%），其余{45974-diag['peak_within3px_count']}条峰误差≥3px。C相对A在{100*diag['C_vs_A_improved_fraction']:.2f}%的窗口命中对应上改善，在{100*diag['C_vs_A_worsened_fraction']:.2f}%上变差。C相对B在{100*diag['C_vs_B_improved_fraction']:.2f}%上改善，但不足以达到A的整体定位水平。

分位数P0/P10/P25/P50/P75/P90/P95/P99/P100及<1/3/5px比例均保存于correspondence_distributions.json；全部拟合对应的分布也另列，未与45974条条件子集混用。完整逐对应ID、A/B/C坐标、误差、概率质量和峰诊断保留私有runs/heatmap_readout_v1/*.npz及all45974_inside_correspondences.csv。CDF和质量分布见readout_error_and_mass.png。

## 4. 真值附近质量、主峰与竞争峰：仅作解释

定义事前写入configs/heatmap_readout_v1.json，未因结果改指标：名义GT附近质量以距GT最近的原有效采样格为中心取固定3×3原有效格，求原全窗口概率之和；是原概率质量，不是局部重新归一化后的置信度。GT最近格并列时按原行优先顺序。最大峰附近质量是C所用3×3集合在原热图中的质量。

竞争次峰定义为主峰3×3之外的原有效3×3局部最大值，要求不小于所有有效邻居且至少严格大于一个有效邻居；取其中最大者，并列按行优先。明显竞争标志固定为次峰/主峰概率比≥0.5。全平坦最大值plateau另计，不强行捏造一个次峰。该0.5只用于诊断分组，不进入任何推理、筛选或评分。

{chr(10).join(diagnostic_table)}

存在规定竞争次峰的比例为{100*diag['obvious_competing_fraction']:.2f}%；最大峰落在名义GT最近格3×3区域的比例为{100*diag['peak_inside_truth3x3_fraction']:.2f}%。平均主峰原概率仅{100*diag['peak_probability']['mean']:.2f}%，有效窗口归一化熵均值{diag['normalized_entropy']['mean']:.3f}，显示响应较分散；这不是校准过的真实正确概率，也不能直接拿来增加过滤。

{chr(10).join(cond_table)}

峰本来较准时，C平均比B更准（1.621对1.798px），说明局部期望并非普遍有害；但有{diag['peak_within3px_but_C_not']}条B<3px而C移到≥3px，也有{diag['peak_outside3px_but_C_inside']}条B≥3px而C回到<3px。多数最大峰已偏离GT时，C虽然比B收敛一点，仍明显差于A。存在竞争峰的分组中C也没有整体优于A。这些条件分组从未用于选择读出策略。

## 5. 哪些影像对改善、哪些退化

{chr(10).join(mover_table)}

即便最终被救回的116对，内部命中对应平均误差的逐对中位也从A的约3.234升至C的4.423px；退化114对则从约3.322升至4.591px。最终仿射由鲁棒估计选中的子集与覆盖共同决定，不能用全部点均值推出最终成败，也不能仅凭这两组统计归因于某几个“关键点”。每对A/B/C的内点数、源点覆盖、独立探针误差、命中误差及热图诊断均在per_pair.csv和pair_results.json，保留后续核查依据。

C vs A的逐对误差散点图见paired_C_vs_A.png；绿色为5px失败→成功，红色为成功→失败。没有根据这些结果追加过滤或挑选更有利子集。

## 6. 回归、可视化与解释边界

新增7项针对测试覆盖：远处竞争响应拉动全窗期望、局部连续读出、边界截断/无效mask、峰值并列和非循环邻接、空匹配/全无效拒绝、GT诊断不改变输出、平坦plateau不伪装成第二峰。全仓库161项回归测试通过（旧依赖/NumPy共有4条弃用警告）。A原始坐标逐位重构、全952原矩阵/评价复现、候选/热图/分数哈希不变、C只用原有效局部格、全部候选统一读出均通过。

私有热图图例固定取原T05诊断子集前6对的第一条拟合对应，完全按原顺序、不按GT命中或效果选例。三幅图显示同一原热图，标记A/B/C、名义GT和C的3×3边界，图中一格为2原图像素；原无效格红叉显示。GT在窗口外时不把图例裁剪当作有效GT。清单在heatmap_visualization_manifest.json，图片同步本地交付目录，原热图与逐对应数组不发GitHub。

本轮只是借鉴用户所提《Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed》的峰附近局部读出思路，不是其完整两阶段匹配复现，也不从该机制推断Opt-SAR必然受益。本轮没有新增相关计算、特征交互、窗口或监督。

952对验证集已多轮用于开发；本轮正差不能当最终无偏泛化证据。后续若形成新的固定配置，最终效果需由未参与配置选择的测试集检验；本轮没有读取或评价测试集。

建议主线：停止反复调整当前读出，保留A作为对照，将后续研究问题转向局部特征辨别力与定位监督。该建议不等于已启动训练，也不预设新训练一定有效；当前三分支证据只支持“最大峰经常偏离名义GT”，不能把此前92.54%理想定位潜力当作实际网络可实现收益。
'''
    (out/'实验报告.md').write_text(text)
    # Flat export of all published point distributions for direct review.
    export=[]
    for r in distributions:
        f={k:v for k,v in r.items() if k not in ['quantiles','fraction_below']};f.update({'q'+q+'_px':v for q,v in r['quantiles'].items()});f.update({'fraction_below'+t+'px':v for t,v in r['fraction_below'].items()});export.append(f)
    write_csv(out/'correspondence_distributions.csv',export)
    print(chr(10).join(table));print(chr(10).join(point_table));print('Decision:',conclusion['decision'])


if __name__=='__main__':main()
