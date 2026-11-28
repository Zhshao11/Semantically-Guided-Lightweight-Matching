"""Report ideal within-window localization potential, never deployable method performance."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def percent(x):return f'{100*x:.2f}%'


def main():
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);a=json.loads((out/'acceptance.json').read_text());freeze=json.loads((out/'cache_freeze.json').read_text());assert a['passed']
    base,oracle=a['summaries'];changes={r['threshold_px']:r for r in a['paired_changes']};groups={r['group']:r for r in a['rescue_groups']};support=a['support_summary'];data=list(csv.DictReader((out/'per_pair.csv').open()))
    labels={'original_prediction':'原始T05 K256预测','inside_actual_support_nominalGT_target_only':'实际窗口内理想定位'}
    table=['| 分支 | SR@3 | SR@5 | 探针RMSE中位/P90（px） | 未达5px对数 | 求解失败数 |','|---|---:|---:|---:|---:|---:|']
    for r in [base,oracle]:table.append(f"| {labels[r['branch']]} | {percent(r['success_rate@3px'])} | {percent(r['success_rate@5px'])} | {r['solved_median_rmse_px']:.6g} / {r['solved_P90_RMSE_px']:.6g} | {r['registration_failures5']} | {r['failed_pairs']} |")
    table.append(f"| 差值（理想−原始） | {100*(oracle['success_rate@3px']-base['success_rate@3px']):+.2f}个百分点 | {100*(oracle['success_rate@5px']-base['success_rate@5px']):+.2f}个百分点 | {oracle['solved_median_rmse_px']-base['solved_median_rmse_px']:+.6g} / {oracle['solved_P90_RMSE_px']-base['solved_P90_RMSE_px']:+.6g} | {oracle['registration_failures5']-base['registration_failures5']:+d} | {oracle['failed_pairs']-base['failed_pairs']:+d} |")
    paired=['| 原始→理想 | 3px标准 | 5px标准 |','|---|---:|---:|']
    for field,label in [('failure_to_success','失败→成功'),('success_to_failure','成功→失败'),('failure_to_failure','失败→失败'),('success_to_success','成功→成功')]:paired.append(f"| {label} | {changes[3][field]} | {changes[5][field]} |")
    for field,label in [('failure_rescue_rate','原始失败救回率'),('success_degradation_rate','原始成功退化率')]:paired.append(f"| {label} | {percent(changes[3][field])} | {percent(changes[5][field])} |")
    paired.append(f"| 净SR变化 | {100*changes[3]['delta_SR']:+.2f}个百分点 | {100*changes[5]['delta_SR']:+.2f}个百分点 |")
    support_table=['| 阶段 | 总候选 | 实际窗口命中 | 合并命中率 | 逐对命中率P10/中位/P90 | 逐对命中数量P10/中位/P90 |','|---|---:|---:|---:|---:|---:|']
    for r in support:
        q=r['pair_hit_rate_quantiles'];n=r['pair_hit_count_quantiles'];support_table.append(f"| {'精化输入' if r['stage']=='refine_input' else '最终拟合'} | {r['total']} | {r['hits']} | {percent(r['pooled_hit_rate'])} | {percent(q['10'])} / {percent(q['50'])} / {percent(q['90'])} | {n['10']:.0f} / {n['50']:.0f} / {n['90']:.0f} |")
    distribution=[]
    for stage in ['refine','fit']:
        rates=np.array([float(r[stage+'_hit_rate']) for r in data if r[stage+'_hit_rate']!=''])
        bins=np.linspace(0,1,11);counts,_=np.histogram(rates,bins)
        distribution.extend(dict(stage=stage,lower=float(bins[i]),upper=float(bins[i+1]),pairs=int(count),upper_inclusive=i==9) for i,count in enumerate(counts))
    write_csv(out/'pair_hit_rate_distribution.csv',distribution)
    compare=['| 原始5px失败的分组 | 对数 | 命中数中位 | 命中率中位 | 4×4占用格中位 | 特征值比中位 | 源点凸包覆盖中位 |','|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('rescued5','被救回'),('still_failed5','仍失败')]:
        g=groups[key];compare.append(f"| {label} | {g['pairs']} | {g['fit_hits_median']:.0f} | {percent(g['fit_hit_rate_median'])} | {g['hit_source_grid4x4_median']:.0f} | {g['hit_source_eigen_ratio_median']:.3f} | {g['hit_source_hull_median']:.3f} |")
    sensitivity=json.loads((out/'ransac_sensitivity.json').read_text());reg=[]
    for r in data:
        if r['base_SR5']=='True' and r['oracle_SR5']=='False':
            index=int(r['index']);c=np.load(Path(p['cache'])/f'{index:04d}.npz');mask=c['frozen_fit_mask'];hit=c['hit'][mask]
            probe=next(t for t in sensitivity['probe_results'] if t['pair_id']==r['pair_id'] and t['seed']==a['one_paired_RANSAC_seed'])
            branch=probe['branches']['oracle'];ii=np.array(branch['fit']['inlier_indices'],int)
            reg.append(dict(pair_id=r['pair_id'],base_rmse_px=float(r['base_rmse_px']),oracle_rmse_px=float(r['oracle_rmse_px']),fit_candidates=int(r['fit_count']),replacements=int(r['fit_hits']),
                original_inliers=int(r['base_inliers']),oracle_inliers=int(r['oracle_inliers']),oracle_inliers_replaced_nominal=int(hit[ii].sum()),oracle_inliers_unreplaced=int((~hit[ii]).sum()),
                nominal_transform_consensus=branch['nominal_exact_transform_consensus_under_original_threshold']))
    (out/'regressed_case_diagnosis.json').write_text(json.dumps(dict(cases=reg,seed_probe=sensitivity['predeclared_probe_seeds'],probe_numerical_seed_sensitivity=sensitivity['probe_numerical_seed_sensitivity'],not_a_new_filter_or_seed_selection=True),indent=2))
    r=reg[0];readout=freeze['max_finite_BF16_readout_vs_continuous_heatmap_expectation_px'];box=a['continuous_support_vs_legacy_box'];pool=a['inside_before_pooled_errors']
    text=f'''# 原始T05 K256：固定候选的窗口内理想定位诊断

在本次固定候选、有效窗口、原分数/过滤和原几何估计器条件下，名义几何理想定位将SR@5从{percent(base['success_rate@5px'])}提高到{percent(oracle['success_rate@5px'])}，净增{100*changes[5]['delta_SR']:.2f}个百分点。原460个5px失败中救回390个（84.78%），原成功中1个退化（0.20%）；仍有71个未达5px，其中70个原本失败、1个新增退化。

这回答“现有候选中若可输出窗口内定位理想化，能救回多少失败”。它不证明任何低成本网络能实现这些收益，不是可部署模型成绩，也不是扩大窗口、调整候选或更换RANSAC后的上界。真值替换没有部署意义，本轮不做测速/显存排名、不训练新精化器、不增加训练种子。

## 表1：总体效果（原始配对RANSAC种子）

{chr(10).join(table)}

952对全部保留，SR要求每对独立探针RMSE严格小于3/5px。RMSE中位/P90按原评价口径统计全部求解成功影像对；两分支均求解成功952对，不等于每对都正确。理想分支中大量名义仿射精确对应形成一致集，因而其中位RMSE接近数值零，这是Oracle构造的结果，不是网络达到微米级定位的证据。

原始对照来自本轮重新运行的T05 epoch18原始检查点，952对矩阵/评价逐项精确复现历史K256；不是SR@5=52.63%的同预算微调精化器，更不是局部交互精化器。原始检查点SHA256：`{p['checkpoint_sha256']}`。

## 表2：双向变化

{chr(10).join(paired)}

救回率分母分别为原始3px失败642对、5px失败460对；退化率分母分别为原始3px成功310对、5px成功492对。净SR变化为（失败转成功−成功转失败）/952。没有删除求解失败、没有只统计原始和理想两分支都成功的子集。

## 真值方向和实际窗口

1. 实际模型输入是256×256光学RGB复制与扰动SAR单通道，uint8/255；本路径没有额外缩放、裁剪或padding。使用原固定952对A_forward，光学原始注册坐标名义对应到SAR原坐标，再由正向附加A映射到SAR_aug。没有误用图像逆向采样矩阵。
2. 真值使用实际进入拟合的source_xy：`nominal_truth=transform_points(actual_source,A_forward)`。代码检查当前原精化器不更新源坐标，原始返回source与进入精化source逐项一致；即便当前恰好是粗格中心，也没有另造中心替换实际source。已知平移、全仿射、非粗格位置及图像正向warp测试通过。
3. 目标支持来自实际fine张量、实际预测中心和原始`sample_windows`返回的有效token掩码。原采样链为feature=(center+.5)/2−.5+delta，再反算到输入像素：2*(feature+.5)−.5。逐点核对等于原读出center+2*delta。半像素中心、xy顺序、stride、边界与bilinear/align_cornersFalse均保留。
4. 当前radius实际为{freeze['config']['model']['local_radius_fine_cells']}，fine stride实际为2。内部完整窗口通常有49个采样点，但命中判定没有直接使用矩形半径：取原始有效SAR采样点的二维凸包，检查名义目标是否在其中；source采样有效且名义目标在[0,255]、原SAR_aug mask双线性采样>.999，source影像mask也有效。孔洞即使在凸包内仍需目标mask通过。全无效窗口的fallback中心不视为可定位点。
5. 单点/线段支持按同样数值容差处理；polygon按边法向距离判断，容差固定{p['numeric_tolerance_px']:.0e}px，仅为浮点级容差，不随窗口面积放大，不放宽搜索范围。边界center=3.5时有效支持最小坐标为1.5，center=251.5时最大为253.5，不能将填充值区域算作可定位范围。

“可输出范围”在本题中按连续热图期望的闭凸包理想化，不要求某个现有温度/参数组能实际产生指定one-hot热图，也不穷举BF16离散可达集合。记录到的原始BF16读出与高精度连续期望最大坐标差为{readout:.6f}px；未用该差扩大凸包容差。理想替换数组相对名义真值误差为零，估计器内部cast到float32后的最大误差为{a['max_replacement_nominal_error_after_estimator_float32_cast_px']:.3g}px。

这些对应来自已配准影像加合成扰动，继承原始光学/SAR配准残差。因此这里始终称“名义几何真值”，不保证物理地物完全重合。

## 候选冻结和支持统计

{chr(10).join(support_table)}

本数据中原local valid及估计器finite/最低分数检查没有再删除这122881条精化输入候选，因此两阶段数量恰好相同；粗匹配之前的阈值/互检仍全部执行，不代表恢复了粗筛排除位置。两阶段分别记录，未混淆“无候选”和“被过滤”。各阶段零候选影像对均为0。

实际可替换{box['actual_hull_fit_hits']}条；旧±6px检查为{box['old_box_fit_hits']}条，额外包含{box['box_only']}条不在实际有效凸包中的位置，实际凸包独有{box['hull_only']}条。不是预设必须保留历史46090条。原始实际窗口内对应的合并平均定位误差为{pool['mean_px']:.4f}px、中位{pool['median_px']:.4f}px、P90={pool['p90_px']:.4f}px；替换后名义误差为零。逐对定位误差见per_pair.csv，全部误差记录见results.json。

逐影像对命中率完整分位数P0/P10/P25/P50/P75/P90/P100见support_summary.json，10个区间分布见pair_hit_rate_distribution.csv，图见pair_hit_rate_distribution.png。合并命中率按所有候选池化；逐对分布每对等权，不将122881条相关对应视为独立场景。

## 被救回与仍失败的区别

{chr(10).join(compare)}

仍失败组通常命中更少、覆盖更集中、特征值比更低；这只是诊断关联，不是新筛选/过滤规则。仍失败70对中有{groups['still_failed5']['less_than3_hit_points']}对少于3个命中点，单独标记。4×4只用于统计覆盖，不改变均匀K256选择。特征值比为source二维协方差的lambda_min/lambda_max；保留既有coverage中的凸包占比、奇异值比与near_collinear标记，少于3点时特征值比另记null，不做误导性共线判断。RANSAC两分支的内点数量与源/目标覆盖各自独立计算，见results.json与per_pair.csv。

## 1个退化样本与随机性复核

`{r['pair_id']}`：原始RMSE={r['base_rmse_px']:.6f}px，理想后={r['oracle_rmse_px']:.6f}px；共{r['fit_candidates']}条拟合对应、替换{r['replacements']}条，原RANSAC内点{r['original_inliers']}、理想后{r['oracle_inliers']}。名义正确变换在原3px阈值下有{r['nominal_transform_consensus']}条一致对应，而估计器返回了{r['oracle_inliers']}条内点的错误变换，其内点中仅{r['oracle_inliers_replaced_nominal']}条来自替换集合、{r['oracle_inliers_unreplaced']}条来自保留的原对应。保留窗口外错误对应、竞争一致集及同一估计器的响应，可使局部理想化也产生个例退化；没有复用A的内点掩码。

为核查这个异常，事后仅对原5px成功→失败的这一对，在原种子及随后两个固定种子{str(sensitivity['predeclared_probe_seeds'])}上用同一缓存配对重估。理想分支矩阵/误差未见数值变化，未发现明显种子敏感，因此没有追加全952多种子结果、没有换种子挑好成绩。主表始终使用原{a['one_paired_RANSAC_seed']}。记录见ransac_sensitivity.json和regressed_case_diagnosis.json；本轮没有模型训练种子。

## 七项验收与交付

- 原始分支：952对矩阵和独立评价与T05K256历史逐项完全一致。
- 候选：ID、顺序、数量、实际source、confidence、local/evaluator过滤掩码冻结且哈希一致；没有GT筛选、恢复、重排或新去重。
- 窗口外：目标坐标逐项不变，错误对应没有删除。
- 替换：仅finalfit命中位置使用A(actualsource)，误差在float64缓存中精确零，估计器float32量化误差另报。
- 空干预：952对强制全不命中，重新估计矩阵、状态、内点与原始完全一致。
- 边界/连续支持：新增7项针对测试和全部145项回归测试通过。NumPy二维cross以及pytest依赖共有4条弃用警告，无测试失败；它们未改变命中判断。
- 独立评价：探针不追加进拟合，逐对检查探针位置与拟合source格点不重合；除固定扰动影像的预先生成外，GT只在原始网络、过滤与原拟合完成后用于诊断替换和评价。

固定原T05诊断子集前6对可视化另存私有runs与本地交付目录，清单见visualization_manifest.json；显示的10条线按原拟合顺序均匀取样，局部放大始终选第一条拟合候选，不按GT或结果挑例。图像上的可视化采样不改变实验候选。公开support_geometry_tests.png只含合成几何图，不含数据像素。

主机缓存为`{p['cache']}`，逐候选ID/原src/dst/分数/窗口坐标/有效token/过滤mask/真值/探针均保存，逐文件SHA在cache_freeze.json。公开结果含逐影像对支持、覆盖、定位与两分支矩阵/评价；原始影像和逐对应数组留私有runs，不发GitHub。代码与冻结输入SHA见cache_freeze.json、acceptance.json，复现说明见docs/WINDOW_ORACLE_REPRODUCTION_zh.md。

本轮结论：在这套固定候选与几何估计条件下，窗口内名义定位存在很大的理想改善空间；尚不能据此决定新精化器、成本、扩大窗口或更换候选/过滤策略。仍失败和个例退化也说明，这个条件式Oracle并非严格保证单调提升的全局性能上界。
'''
    (out/'实验报告.md').write_text(text)
    # Machine-readable table1 difference row alongside original metrics.
    write_csv(out/'overall_display.csv',[dict(branch=labels[r['branch']],SR3_percent=100*r['success_rate@3px'],SR5_percent=100*r['success_rate@5px'],median_RMSE_px=r['solved_median_rmse_px'],P90_RMSE_px=r['solved_P90_RMSE_px'],failures5=r['registration_failures5'],solver_failures=r['failed_pairs']) for r in [base,oracle]]+
        [dict(branch='oracle_minus_original',SR3_percent=100*(oracle['success_rate@3px']-base['success_rate@3px']),SR5_percent=100*(oracle['success_rate@5px']-base['success_rate@5px']),median_RMSE_px=oracle['solved_median_rmse_px']-base['solved_median_rmse_px'],P90_RMSE_px=oracle['solved_P90_RMSE_px']-base['solved_P90_RMSE_px'],failures5=oracle['registration_failures5']-base['registration_failures5'],solver_failures=oracle['failed_pairs']-base['failed_pairs'])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rates=np.array([float(r['fit_hit_rate']) for r in data]);fig,ax=plt.subplots(figsize=(7,3.5));ax.hist(rates,bins=np.linspace(0,1,21),color='#3b82ba',edgecolor='white');ax.axvline(np.median(rates),color='darkorange',label=f'Pair median={np.median(rates):.4f}');ax.axvline(support[1]['pooled_hit_rate'],color='darkgreen',linestyle='--',label=f"Pooled={support[1]['pooled_hit_rate']:.4f}")
    ax.set(xlabel='Actual valid-support hit fraction per pair',ylabel='Image pairs (952 total)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'pair_hit_rate_distribution.png',dpi=160);plt.close(fig)
    print(chr(10).join(table));print(chr(10).join(paired))


if __name__=='__main__':main()
