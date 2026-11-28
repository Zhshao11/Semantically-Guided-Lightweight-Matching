"""Measured joint accuracy/cost points, Pareto set and one coherent development anchor."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def rows(path):return list(csv.DictReader(Path(path).open()))


def main():
    out=Path('reports/sar_references_v1');p=json.loads(Path('configs/sar_references_v1.json').read_text())
    names=[*p['methods'],'XoFTR','MINIMA_RoMa'];all_rows=[];timing=[]
    labels={'T05_Full':'T05全量K1024','T05_Uniform256':'T05均匀K256','FHReg_GUSO':'FHReg-GUSO／512主设置','FHReg_GUSO_256':'FHReg-GUSO／256尺寸对照','GDROS_WHU':'GDROS-WHU／原始流→仿射','XoFTR':'XoFTR公开权重','MINIMA_RoMa':'MINIMA-RoMa公开权重'}
    table=['| 方法 | SR@3px | SR@5px | RMSE中位/P90（px） | 求解失败 | 失败@5px | E2E中位（ms） | 峰值分配（MiB） |','|---|---:|---:|---:|---:|---:|---:|---:|']
    def timing_summary(name):
        runs=[];env=[]
        for rd in range(1,4):
            runs+=rows(out/f'{name}_timing_round{rd}.csv');record=json.loads((out/f'{name}_timing_round{rd}.json').read_text());assert record['fresh_single_model_process'] and record['exclusive_guards']==48;env.append(record)
        assert len(runs)==360
        values=[float(r['latency_ms']) for r in runs]
        return dict(median_E2E_ms=float(np.median(values)),p95_E2E_ms=float(np.percentile(values,95)),peak_allocated_MiB=max(float(r['peak_allocated_MiB']) for r in runs),
            peak_reserved_MiB=max(float(r['peak_reserved_MiB']) for r in runs),round_medians_ms=[e['median_ms'] for e in env],
            process_peak_RSS_MiB=max(e['process_peak_RSS_MiB'] for e in env),repeats_per_pair=15,distinct_timing_pairs=24),runs
    for name in names:
        source=out/f'{name}_accuracy.json' if name in p['methods'] else Path('reports/external_references_v1')/f'{name}_accuracy.json'
        score=json.loads(source.read_text());assert score['image_pairs']==952
        cost,runs=timing_summary(name);timing+=runs
        record=dict(method=name,label=labels[name],SR3_percent=score['success_rate@3px']*100,SR5_percent=score['success_rate@5px']*100,
            median_RMSE_px=score['solved_median_rmse_px'],P90_RMSE_px=score['solved_p90_rmse_px'],solved_mean_RMSE_px=score['solved_mean_rmse_px'],
            solver_failures=score['failed_pairs'],registration_failures5=score['registration_failures5'],mean_valid_matches=score['mean_valid_matches'],
            parameters=score.get('parameters',score.get('actual_parameters_total')),accuracy_reused=name not in p['methods'],**cost)
        all_rows.append(record)
        table.append(f"| {labels[name]} | {record['SR3_percent']:.2f}% | {record['SR5_percent']:.2f}% | {record['median_RMSE_px']:.3f} / {record['P90_RMSE_px']:.3f} | {record['solver_failures']} | {record['registration_failures5']} | {record['median_E2E_ms']:.3f} | {record['peak_allocated_MiB']:.2f} |")
    def dominates(a,b):return a['SR5_percent']>=b['SR5_percent'] and a['median_E2E_ms']<=b['median_E2E_ms'] and a['peak_allocated_MiB']<=b['peak_allocated_MiB'] and (a['SR5_percent']>b['SR5_percent'] or a['median_E2E_ms']<b['median_E2E_ms'] or a['peak_allocated_MiB']<b['peak_allocated_MiB'])
    for r in all_rows:r['dominated_by']=[s['method'] for s in all_rows if s is not r and dominates(s,r)];r['non_dominated']=not bool(r['dominated_by'])
    frontier=[r for r in all_rows if r['non_dominated']]
    candidates=[r for r in frontier if r['method'] in ['FHReg_GUSO','GDROS_WHU']]
    anchor=max(candidates,key=lambda r:r['SR5_percent']) if candidates else next(r for r in all_rows if r['method']=='T05_Full')
    baseline=next(r for r in all_rows if r['method']=='T05_Full')
    budget=dict(version='measured-SAR-reference-development-budget-v1',reference_method=anchor['method'],reference_point=anchor,
        primary_accuracy_floor=dict(SR5_percent_at_least=anchor['SR5_percent'],SR3_percent_at_least=anchor['SR3_percent']),
        latency_goal=dict(median_E2E_ms_strictly_below=anchor['median_E2E_ms'],boundary=p['timing']['boundary']),
        memory_goal=dict(peak_allocated_MiB_strictly_below=anchor['peak_allocated_MiB'],measurement='freshsingleprocess, fullPyTorchpeak, idleRTX5090'),
        secondary=dict(reference_median_RMSE_px=anchor['median_RMSE_px'],reference_P90_RMSE_px=anchor['P90_RMSE_px'],reference_solver_failures=anchor['solver_failures'],report_extreme_failures_without_clipping=True),
        baseline_precision_gap_pp=anchor['SR5_percent']-baseline['SR5_percent'],one_actual_joint_point=True,not_achieved_by_our_model=True,
        no_mixing_best_metrics=True,not_a_promise_of_feasibility=True,no_new_training=True,test_used=False,
        decision='Main reference chosen among actual common-affine non-dominated SAR-specific primary configurations;256resolution sensitivity does not replace preregistered512FHprimary.')
    write_csv(out/'external_comparison.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v) for k,v in r.items()} for r in all_rows]);write_csv(out/'all_latency_runs.csv',timing)
    (out/'non_dominated.json').write_text(json.dumps(dict(objectives=['maximizeSR5','minimizeE2Emedian','minimizeGPUpeak'],points=all_rows,members=[r['method'] for r in frontier],note='Observed frontier, not statistical significance or independent-satellite-scene claims.'),indent=2))
    (out/'development_budget_v1.json').write_text(json.dumps(budget,indent=2))
    native=[]
    for label in ['raw_dense','native32_LSR']:
        r=json.loads((out/f'GDROS_WHU_{label}_accuracy.json').read_text());cost,_=timing_summary('GDROS_WHU_native32_LSR') if label=='native32_LSR' else ({},[])
        native.append(dict(protocol=label,SR3_percent=100*r['success_rate@3px'],SR5_percent=100*r['success_rate@5px'],median_RMSE_px=r['solved_median_rmse_px'],P90_RMSE_px=r['solved_p90_rmse_px'],evaluation_failures=r['failed_pairs'],**cost))
    write_csv(out/'GDROS_native_protocols.csv',[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in native])
    n=next(r for r in native if r['protocol']=='native32_LSR');fh=next(r for r in all_rows if r['method']=='FHReg_GUSO');small=next(r for r in all_rows if r['method']=='FHReg_GUSO_256')
    (out/'外部对比报告.md').write_text(f'''# SAR专用公开权重：统一评价与资源参照

FHReg-GUSO和GDROS-WHU均已通过严格权重加载、原生输入/后处理、坐标回归与固定训练来源样本检查，并完成952对固定验证。本轮不训练、不微调、不调阈值，不使用OS测试集。RIFT2已完成源码/运行条件审计，尚未数值复现；SOMA为独立任务，恢复后因非有限loss停止，不使用早期权重充当最终参照。

## 统一仿射精度—成本表

{chr(10).join(table)}

SR的分母均为952；失败@5px包含求解失败以及独立探针RMSE≥5px。RMSE中位/P90/均值统计所有成功求解影像对，不仅是SR成功子集。FH/GDROS/T05为本轮完整重跑；MINIMA-RoMa/XoFTR的精度复用前轮同952对冻结记录，源/权重/数据哈希均核验、另在2个预定验证输入上逐位复核预测；其资源在本轮重新测量。见*_legacy_reuse.json，不声称重跑了旧方法的全部952对。

T05均为冻结epoch18原始权重：Full为K1024，Uniform256为均匀K256。没有使用上一轮精化器微调权重。T05经过本OS训练集域内训练；FHReg公开权重按作者说明来自GUSO，GDROS按WHU官方文件名/说明使用WHU权重，本轮均未做OS适配。这是外部跨域参照与域内基础方法的对比，不是相同训练数据的架构公平比较，也不宣称原始卫星scene完全无重叠。

## FHReg的输入与尾部问题

主设置预先固定为512：512原生样例、原始3通道模型和完整64×64粗网格；OS256先OpenCV线性缩放，再按官方zero_shot脚本TF.to_tensor等价处理到0..1，返回对应按半像素规则还原。另预先固定256输入作为尺寸敏感性对照，仍使用完整原生粗/细匹配；不强制两种尺寸相同K，不改0.1阈值。

官方wrapper的imsize512字段未被实际resize调用；demo.py输入为rawfloat0..255，而OSzero_shot Dataset使用/255。原生demo样例已检查能前向，主OS协议明确采用zero_shot归一化。本轮未根据验证结果改归一化或在512/256之间重选主配置，也未使用原脚本按GT筛点/误差封顶后的RMSE。256输入SR@5为{small['SR5_percent']:.2f}%，与512的{fh['SR5_percent']:.2f}%差距很大；两行分别报告，不能组合256开销与512精度。

FHReg512的成功求解平均RMSE为{fh['solved_mean_RMSE_px']:.3f}px，远高于其中位数；存在7个对应拟合出的极端错误仿射，最坏独立探针RMSE约67988.656px。该例源点近共线、凸包覆盖仅约4.05e-6，数值rank仍为2，未被原精确rank检查拒绝。未剪裁误差、删除坏例或事后修改退化/RANSAC门限，详见FHReg_GUSO_tail_audit.json。SR提升不等于所有尾部行为都改善。

## GDROS两类协议

主表使用完整原始512稠密流（32次迭代）转成光学→扰动SAR对应，经过公共有效区过滤后，由原T05完整仿射RANSAC拟合。全部512²位置作为原始候选，不截成固定K。像素点和位移同时按512→256尺度还原，已用已知仿射/位移及独立探针验证。另核对作者原生训练datum矩阵与flo，零起点前向位移公式最大差7.63e-6个原生像素；用该矩阵正向扭曲原SAR与发布sar_warped高度一致，反向明显不符，方向证据见GDROS_native_coordinate_regression.json。

另报告原始稠密场和原生32次+LSR在同一组固定有效探针上的直接预测误差，无额外公共RANSAC。原生LSR保持flatten[10000:210000]的200000点、原始未加权最小二乘；它不使用公共有效掩码，也没有加载LSRnet.pth。仓库虽提供LSRnet权重，当前test.py实际调用的LSR函数无需该权重。

原生32+LSR：SR@3={n['SR3_percent']:.2f}%，SR@5={n['SR5_percent']:.2f}%，RMSE中位/P90={n['median_RMSE_px']:.3f}/{n['P90_RMSE_px']:.3f}px，端到端中位{n['median_E2E_ms']:.3f}ms，峰值{n['peak_allocated_MiB']:.2f}MiB。完整两种稠密结果见GDROS_native_protocols.csv。该端到端终点为原生CPU稠密场，主表终点为公共仿射；全稠密转对应、掩码过滤与公共RANSAC会增加额外成本，不能把两协议差额都归因于网络本身。精度/耗时不能跨协议拼接，原生稠密结果不进入公共仿射前沿。没有使用仓库将方差命名为RMSE的统计。

GDROS的OS权重未进入任何主比较：作者训练split与当前952对是否重叠不能确认。WHU权重复用服务器既有official_models_reference缓存，复制前后SHA一致，并严格匹配官方源架构；作者没有提供可独立核对的远端校验和，本轮未重新从网盘下载，也未审计作者原始训练样本。来源/边界见freeze.json。

## 测速与非支配集合

单张独占RTX5090、batch1、torch.set_num_threads(4)，CPU不独占，其余库保留原生线程设置；固定24对（已验证父图24组，scene来源仍未完全确认）、3轮×每对5次计时、每对3次预热。每模型/每轮独立新进程、轮次轮换加载顺序；无hook、无特征缓存、CUDA同步墙钟计时。起点为CPU256uint8图像+掩码，包含缩放/归一化/H2D/原生模型/采样或稠密转换/坐标D2H/公共过滤/原仿射RANSAC；排除IO、合成扰动、GT和加载初始化。保留方法原生内部尺寸与精度：T05 BF16、FH/GDROS/XoFTR FP32、RoMa官方混合精度。

显存为完整PyTorch peak allocated（模型、激活、库工作区），取同24对全部新进程测量最大值；不扣库缓存，不等同GPU驱动上下文或CPU总内存。reserved与含初始化的进程峰值RSS另存JSON。不同轮次时钟/CPU负载漂移见各round JSON，全部保留；重复计时不能当成360个独立场景。RIFT2没有GPU结果，也不以零CUDA显存宣称资源优势。

本轮公共仿射三目标：SR@5最大化、端到端中位最小化、GPU峰值最小化。实测非支配集合：{', '.join(r['label'] for r in frontier)}。这是观察到的点集，不是统计显著或场景独立性保证。完整支配关系见non_dominated.json，图见accuracy_cost.png。

开发预算采用其中单个实际参照{anchor['label']}，不拆取不同方法的最好字段。详见开发预算_v1.md与development_budget_v1.json。

## 运行与来源边界

138项回归测试通过；新增5项覆盖半像素坐标、位移缩放、仿射Oracle、错误矩阵/场、NaN失败及大于32767个稠密对应的分块掩码过滤。Native smoke另检查原始GDROS-LSR仿射Oracle、严格加载、TF等价预处理、重复预测和BN不更新。官方核心模型文件未修改；FH仅通过外部namespace alias适配PL1旧profiler导入到PL2.profilers命名，实际使用PassThroughProfiler。

Mamba-SSM2.3.2.post1以CUDA12.8编译，包含SM120，wheelSHA为ba56ceaed9173039cbe8940991e390edc8d0ed9b1f7b85e4e8f30cbb4b848166。独立venv共享只读基础Torch2.11，新依赖仅安装在该venv，未修改SOMA环境。为运行selective_scan路径安装了PyWavelets与Transformers4.51.3；Mamba其他未调用的语言模型/新kernel额外依赖没有声称全部满足，pip安装日志保留。失败的缺依赖/原生tif扩展名探查日志保留，均发生在完整评价前。

RIFT2官方实现是MATLAB：FAST/matchFeatures需要ComputerVision运行条件，imresize等需要图像工具支持，源码含parfor。当前本机/服务器PATH未找到MATLAB/Octave，也未确认许可/工具箱，故只交付运行审计，不提供伪造精度/延迟、不用Python替代版本冒充原方法。MATLAB1起点xy转公共0起点的回归尚待真实运行时完成。见RIFT2_runtime_audit.json。

正式测速只临时释放本项目SOMA任务并从稳定checkpoint恢复同一4GPU协议；soma_resumed_verified.json记录了恢复后的真实更新。SOMA随后于epoch12/update22050遇到非有限loss自动停止，目前不在训练，最后稳定checkpoint为update22000；故障日志与权重保留，未换协议强行续跑。详情见SOMA_later_failure.json，不能将其早期best当最终模型。其他用户任务未干预。权重、图像、逐对应数组和第三方源码留私有runs，不随Git发布。

FHReg论文DOI：[10.1016/j.isprsjprs.2026.03.010](https://doi.org/10.1016/j.isprsjprs.2026.03.010)，Crossref卷期2026年5月；[官方GUSO仓库](https://github.com/vision-heng/GUSO)。GDROS DOI：[10.1109/TGRS.2025.3627132](https://doi.org/10.1109/TGRS.2025.3627132)，Crossref年份2025；[官方仓库](https://github.com/Zi-Xuan-Sun/GDROS)。元数据见publication_metadata.json。FH代码MIT、GDROS代码Apache2.0；不因此替权重/数据推断未给出的部署授权。

本表为OS256合成仿射patch验证，几何参考继承原始配准残差；不是人工72控制点场景评价，也不是论文原始OS指标复现。没有将教师伪标签当语义GT，没有新语义实验、动态预算或任何新增训练。
''')
    (out/'开发预算_v1.md').write_text(f'''# 开发预算v1：来自同一个实测点

参照：**{anchor['label']}**。该点属于本轮公共仿射协议实测非支配集合。

| 项目 | 预算／验收目标 |
|---|---:|
| SR@5px底线 | ≥{anchor['SR5_percent']:.4f}% |
| SR@3px底线 | ≥{anchor['SR3_percent']:.4f}% |
| 端到端中位延迟 | <{anchor['median_E2E_ms']:.3f}ms |
| 峰值分配显存 | <{anchor['peak_allocated_MiB']:.2f}MiB |
| 误差参考（中位/P90） | {anchor['median_RMSE_px']:.3f}/{anchor['P90_RMSE_px']:.3f}px |
| 求解失败参考 | {anchor['solver_failures']}/952 |

四项主门槛全部来自这一个参照点。延迟和显存要求更低，同时SR@3/5达到其参照水平；误差尾部、求解失败和极端失败必须单独报告，不能删掉坏例达标。没有将FH的精度、T05的延迟与另一个方法的显存拼起来。

当前T05全量SR@5={baseline['SR5_percent']:.4f}%，距目标{anchor['SR5_percent']-baseline['SR5_percent']:.4f}个百分点；其延迟{baseline['median_E2E_ms']:.3f}ms、显存{baseline['peak_allocated_MiB']:.2f}MiB是当前起点，不能假定加模块仍维持这些成本。本预算是后续开发的研究验收目标，不是已达到的结果或可行性承诺。保持本轮固定验证、输入/拟合边界、独占卡和无hook测量；测试集继续不用于选结构或调阈值。

本轮没有启动任何新训练。FH512主配置与256敏感性对照保持事前登记，不用差异倒推更有利的阈值。GDROS原生稠密协议另列，不能用其时间替换公共仿射精度成本。
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5));offsets={'T05_Full':(8,8),'T05_Uniform256':(8,-16),'FHReg_GUSO':(8,8),'FHReg_GUSO_256':(8,12),'GDROS_WHU':(-170,8),'XoFTR':(8,-15),'MINIMA_RoMa':(-105,10)}
    english={'T05_Full':'T05 K1024','T05_Uniform256':'T05 uniform K256','FHReg_GUSO':'FHReg 512','FHReg_GUSO_256':'FHReg 256','GDROS_WHU':'GDROS WHU + affine','XoFTR':'XoFTR','MINIMA_RoMa':'MINIMA-RoMa'}
    for i,r in enumerate(all_rows):
        size=70+50*np.log2(max(r['peak_allocated_MiB'],32)/32)
        ax.scatter(r['median_E2E_ms'],r['SR5_percent'],s=size,c=[plt.cm.tab10(i)],edgecolors='black' if r['non_dominated'] else 'white',linewidths=1.7 if r['non_dominated'] else .6,alpha=.85)
        ax.annotate(english[r['method']],(r['median_E2E_ms'],r['SR5_percent']),xytext=offsets[r['method']],textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set(xlabel='Median end-to-end latency (ms, log scale)',ylabel='SR@5px (%)',ylim=(-6,100),title='Fixed952 validation; marker area encodes log2 GPU peak allocation')
    ax.text(.02,.97,'Black border: non-dominated in SR5 / latency / GPU peak',transform=ax.transAxes,va='top',fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'accuracy_cost.png',dpi=160);fig.savefig(out/'accuracy_cost.svg');plt.close(fig)
    svg=out/'accuracy_cost.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(chr(10).join(table));print('Anchor:',anchor['method'])


if __name__=='__main__':main()
