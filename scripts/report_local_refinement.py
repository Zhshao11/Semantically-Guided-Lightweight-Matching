"""Turn the paired experiment's completed measurements into a compact Chinese report."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def read_csv(path):return list(csv.DictReader(Path(path).open()))


def main():
    out=Path('reports/local_interaction_v1');p=json.loads(Path('configs/local_interaction_v1.json').read_text())
    acceptance=json.loads((out/'evaluation_acceptance.json').read_text());assert acceptance['passed']
    train=json.loads((out/'training_acceptance.json').read_text())
    timing=json.loads((out/'latency_summary.json').read_text());latency={r['group']:r for r in timing['summaries']}
    memory={g:json.loads((out/f'memory_{g}_summary.json').read_text())['summaries'][0] for g in p['groups']}
    data={r['group']:r for r in acceptance['summaries']};labels={'untrained_T05':'T05原始参考（未微调）','original_refiner':'原精化器（同预算微调）','local_cross_attention':'一层轻量局部交互精化器'}
    rows=[]
    for g in p['groups']:
        r=data[g];rows.append(dict(group=g,label=labels[g],SR3_percent=100*r['success_rate@3px'],SR5_percent=100*r['success_rate@5px'],
            median_RMSE_px=r['solved_median_rmse_px'],P90_RMSE_px=r['solved_p90_rmse_px'],solver_failures=r['failed_pairs'],
            failures_at5=952-round(952*r['success_rate@5px']),median_E2E_ms=latency[g]['median_ms'],peak_allocated_MiB=memory[g]['peak_allocated_MiB'],
            parameters=r['parameters'],refiner_parameters=r['refiner_parameters'],mean_refinements=r['mean_refinements']))
    write_csv(out/'comparison.csv',rows)
    diagnosis=[]
    for g in ['untrained_T05',*p['groups']]:
        r=data[g];q=r['pair_macro'];diagnosis.append(dict(group=g,inside_correspondences=r['before_inside']['count'],
            pooled_before_inside_px=r['before_inside']['mean_px'],pooled_after_inside_px=r['after_inside']['mean_px'],
            pooled_after_outside_px=r['after_outside']['mean_px'],pair_macro_inside_delta_px=q['mean_refine_delta_inside_px'],
            pair_macro_window_hit=q['mean_truth_in_window_fraction'],pair_macro_coarse_correct5=q['mean_coarse_correct_fraction_5px'],
            pair_macro_inlier_hull_fraction=q['mean_inlier_source_hull_fraction']))
    write_csv(out/'localization_summary.csv',diagnosis)
    b,e=rows;delta=e['SR5_percent']-b['SR5_percent'];overhead=e['median_E2E_ms']-b['median_E2E_ms']
    verdict=('SR@5px提高，但仍需结合SR@3px、尾部误差及配对变化判断是否保留。' if delta>0 else '本轮没有得到SR@5px收益，不据此采用新精化器。')
    table=['| 精化器 | SR@3px | SR@5px | RMSE中位/P90（px） | 求解失败 | ≥5px或失败 | 中位端到端（ms） | 峰值显存（MiB） |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:table.append(f"| {r['label']} | {r['SR3_percent']:.2f}% | {r['SR5_percent']:.2f}% | {r['median_RMSE_px']:.3f} / {r['P90_RMSE_px']:.3f} | {r['solver_failures']} | {r['failures_at5']} | {r['median_E2E_ms']:.3f} | {r['peak_allocated_MiB']:.2f} |")
    ref=data['untrained_T05'];dt=['| 版本 | 窗口内对应数 | 精化前均值（px） | 精化后均值（px） | 逐影像对平均变化（px） |','|---|---:|---:|---:|---:|']
    for d in diagnosis:dt.append(f"| {labels[d['group']]} | {d['inside_correspondences']} | {d['pooled_before_inside_px']:.4f} | {d['pooled_after_inside_px']:.4f} | {d['pair_macro_inside_delta_px']:.4f} |")
    changes=acceptance['paired_changes'];ct=['| 阈值 | 原失败→交互成功 | 原成功→交互失败 | 两者成功 | 两者失败 |','|---|---:|---:|---:|---:|']
    for r in changes:ct.append(f"| {r['threshold_px']}px | {r['failure_to_success']} | {r['success_to_failure']} | {r['both_success']} | {r['both_failure']} |")
    paired_time=read_csv(out/'latency_paired.csv'); paired_delta=np.array([float(r['delta_ms']) for r in paired_time])
    coord=json.loads((out/'soma_timing_coordination.json').read_text())
    detail=f'''# 固定K256局部轻量交互精化：受控结果

两组均从同一T05出发、只微调精化器。{verdict} 交互组相对训练后原精化器的SR@5px变化为{delta:+.3f}个百分点，中位端到端延迟变化{overhead:+.3f}ms（{100*overhead/b['median_E2E_ms']:+.2f}%）。本轮为单随机种子、固定4epoch预算的探索性比较，不能据微小差异宣称稳定提升或显著差异。

## 精度—开销表

{chr(10).join(table)}

精度在全部952对固定验证影像上计算。SR以影像对独立探针RMSE<阈值为成功，求解失败仍在分母；RMSE中位/P90统计成功求解的全部影像对，不仅是SR成功子集。≥5px或失败是配准失败，与“求解失败”分开列出。

未微调T05K256独立复核：SR@3={100*ref['success_rate@3px']:.3f}%，SR@5={100*ref['success_rate@5px']:.3f}%，RMSE中位/P90={ref['solved_median_rmse_px']:.4f}/{ref['solved_p90_rmse_px']:.4f}px，求解失败{ref['failed_pairs']}。其952对矩阵与评价逐项完全复现历史T05K256，仅作继续训练收益的参照，不替代训练后的原精化器对照。

## 固定条件与实际结构

- 查询均匀K256，gather先于256×1024相关；原粗阈值、互检、7×7局部窗口、温度读出、完整仿射RANSAC及固定探针不变。
- 没有新增128对截断：此前128精化仅用于K1024消融，并未成为UniformK256默认。本轮沿用通过原粗筛的全部对应，平均每对{b['mean_refinements']:.5f}个；两组952对的source/粗中心/置信度哈希、尝试精化数、最终对应数完全相同。
- 原精化：16→32共享线性投影、归一化点积、温度softmax、期望坐标。交互组在读出前新增32→16的Q/K/V和16→32输出投影，单头1个光学query与49个SAR窗口token交互、残差更新query，再用原读出。没有FFN、多层或dropout。
- 新增2048参数；两组总参数{b['parameters']}/{e['parameters']}，可训练精化参数{b['refiner_parameters']}/{e['refiner_parameters']}。输出投影零初始化，第一真实训练批次两组初始预测逐位相同。
- 编码器、粗投影、粗温度、无对应类别及全部BN缓冲冻结；训练结束非local状态哈希逐项不变。推理不接收A或GT，GT仅在拟合之后用于诊断。

## 对称训练

相同7912对去重训练集、相同在线仿射流、seed={p['seed']}，batch4，每组4epoch/7912次更新。AdamW lr=2e-4、weight_decay=1e-4、无调度、clip=5，BF16。各组固定最后epoch4评价，不按各自最佳轮次选模。

同批冻结特征只计算一次，两个精化器接收相同预测粗对应。监督仅采用真对应名义上位于原7×7窗口且掩码有效的位置，沿用坐标SmoothL1，权重0.2；不使用GT窗口启动。每组零监督批次{train['zero_supervision_batches']}。A仍继承原始配准残差，不能视为物理真实对应完全精确。

训练逐步损失及梯度在training_steps.csv，逐epoch均值与样本/矩阵流SHA在training_epochs.csv；图见training_loss.svg。

## 窗口内定位与配对成败

{chr(10).join(dt)}

上述窗口内集合要求名义GT有效且每轴距预测粗中心≤6px，两组集合不变。均值同时提供池化对应统计和逐影像对平均；窗口外误差、窗口命中率、粗正确率、内点覆盖见localization_summary.csv与layered_diagnostics.csv。局部改善本身不等于配准改善，最终以相同RANSAC后的影像对结果判断。额外交互相对训练后原精化器仅减少约0.0113px的窗口内平均误差；逐对平均入窗率仍为35.99%，不宜将有限的局部改善解释为解决了粗对应错误。

{chr(10).join(ct)}

## 测速边界

单张独占RTX5090，batch1、BF16、4个CPU线程，使用T05预先固定24对；每对5次预热，3轮×10次重复，逐对随机交错两模型。无hook、无特征缓存、无GT或图像IO：CPU256 uint8影像/掩码→归一化与H2D→两模态编码→K256粗匹配→全部原预算精化→坐标D2H→原完整仿射RANSAC。使用CUDA同步的墙钟端到端测量，不能与GPU常驻输入的旧表直接比较。

显存采用每组单独新进程、同24对的PyTorch peak allocated最大值，包含模型、激活及库工作区；不扣工作区、不用两模型共享进程显存作结论。latency_summary.json中的共享进程显存仅作原始记录，正式显存来自memory_*_summary.json。逐影像对配对时间差见latency_paired.csv，重复测量不视为720个独立场景。逐对配对时间差中位数为{float(np.median(paired_delta)):.3f}ms，范围{float(paired_delta.min()):.3f}到{float(paired_delta.max()):.3f}ms。

三轮绝对中位延迟存在共同漂移：原精化器为{[round(x,3) for x in latency['original_refiner']['round_medians_ms']]}ms，交互组为{[round(x,3) for x in latency['local_cross_attention']['round_medians_ms']]}ms。全部轮次均保留，未删除较快/较慢轮次；两组差值方向一致，但绝对延迟受服务器CPU/时钟状态影响，不据本表外推固定部署耗时。

正式测速仅临时释放本项目SOMA任务，不干预其他用户任务；恢复记录见soma_timing_coordination.json，恢复检查见soma_resumed_verified.json。SOMA不参与本轮精度比较。从update{coord["resume_updates"]}稳定检查点恢复，保留暂停前日志尾部并重放{coord["replayed_updates"]}次尚未存入检查点的更新；未将重放计作额外训练进度。

## 可复现记录与限制

T05 SHA256：`{p['checkpoint_sha256']}`。最终两组检查点SHA、评价代码SHA及验证清单SHA见evaluation_acceptance.json；完整配置见configs/local_interaction_v1.json。训练状态/流一致性验收见training_acceptance.json。首轮评价在交互模块加载后因其training标志被审计拦截（尚未运行交互组）；已将替换后的完整模型显式设为eval并增加回归测试。训练入口原本已统一eval，且模块无BN/dropout，该修复不改变训练检查点。原失败日志和修复记录见evaluation_first_attempt.log与evaluation_loader_fix.json；完整评价重跑后通过。全部133项回归测试及新增6项针对性测试通过（pytest依赖有1条弃用警告）。权重、影像及逐对应数组留在私有runs中，不发布。

本轮未使用测试集，未加入语义、动态K、新评分、新阈值或精化预算调整。训练/评价可与本项目SOMA共享GPU，正式测速使用独占卡；训练墙钟不能当作效率结论。仅一个随机种子与4epoch微调预算，若后续保留机制需再做配对种子复核，不能把当前结果解释为精化能力的上限。
'''
    (out/'report.md').write_text(detail)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps=read_csv(out/'training_steps.csv');fig,ax=plt.subplots(figsize=(7,3.4))
    for g,label in [('original_refiner','Original refiner'),('local_cross_attention','Light local interaction')]:
        y=np.array([float(r[g+'_loss']) for r in steps]);window=100
        ax.plot(np.arange(window,len(y)+1),np.convolve(y,np.ones(window)/window,mode='valid'),label=label,linewidth=1.4)
    ax.set(xlabel='Paired update',ylabel='Fine SmoothL1 loss (100-step mean)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_loss.svg');fig.savefig(out/'training_loss.png',dpi=160);plt.close(fig)
    svg=out/'training_loss.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(chr(10).join(table))


if __name__=='__main__':main()
