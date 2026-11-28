"""Audit report: random30 evidence, coordinates, denominators and one unifiedcosttable."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def read_csv(path):return list(csv.DictReader(Path(path).open()))


def main():
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);den=json.loads((out/'denominator_audit.json').read_text());coord=json.loads((out/'coordinate_roundtrip.json').read_text());evidence=json.loads((out/'evidence_manifest.json').read_text());selection=json.loads((out/'random30_selection.json').read_text())
    assert den['passed'] and coord['passed'] and evidence['passed']
    timing=[];all_runs=[];environments={};input_hashes=None
    for name in p['timing']['methods']:
        runs=[];env=[]
        for rd in [1,2,3]:
            r=json.loads((out/f'{name}_timing_round{rd}.json').read_text());assert r['fresh_single_model_process'] and r['exclusive_guards']==48 and r['cuda_sync_before_after'] and r['include_RANSAC'] and r['include_common_filter'] and r['no_hooks'] and not r['GT_used']
            assert r['warmup_per_pair']==3 and r['repeats_per_pair']==5
            if input_hashes is None:input_hashes=r['sameCPUinput_hashes']
            else:assert input_hashes==r['sameCPUinput_hashes'],'Timing inputs differ'
            runs+=read_csv(out/f'{name}_timing_round{rd}.csv');env.append(r)
        assert len(runs)==360;values=np.array([float(r['latency_ms']) for r in runs]);environments[name]=env;all_runs+=runs
        timing.append(dict(method=name,median_E2E_ms=float(np.median(values)),P95_E2E_ms=float(np.percentile(values,95)),mean_E2E_ms=float(values.mean()),
            peak_allocated_MiB=max(float(r['peak_allocated_MiB']) for r in runs),peak_reserved_MiB=max(float(r['peak_reserved_MiB']) for r in runs),
            round_medians_ms=[r['median_ms'] for r in env],pairs=24,repeats_per_pair_total=15,precision=env[0]['precision']))
    write_csv(out/'unified_timing.csv',[{**r,'round_medians_ms':json.dumps(r['round_medians_ms'])} for r in timing]);write_csv(out/'unified_timing_runs.csv',all_runs)
    paired=[]
    for meta in input_hashes:
        index=meta['pair_index'];groups={name:np.array([float(r['latency_ms']) for r in all_runs if r['method']==name and int(r['pair_index'])==index]) for name in p['timing']['methods']}
        paired.append(dict(pair_id=meta['pair_id'],pair_index=index,T05_median_ms=float(np.median(groups['T05_Full'])),FHReg_median_ms=float(np.median(groups['FHReg_GUSO'])),
            paired_delta_ms=float(np.median(groups['FHReg_GUSO'])-np.median(groups['T05_Full'])),ratio=float(np.median(groups['FHReg_GUSO'])/np.median(groups['T05_Full']))))
    write_csv(out/'timing_pair_differences.csv',paired)
    old_protocol=json.loads(Path('configs/external_references_v1.json').read_text());sar_protocol=json.loads(Path('configs/sar_references_v1.json').read_text());old_timing=next(r for r in read_csv('reports/external_references_v1/latency_summary.csv') if r['method']=='T05')
    sar_comparison=next(r for r in read_csv('reports/sar_references_v1/external_comparison.csv') if r['method']=='T05_Full')
    history=[dict(record='older_external_table',median_ms=float(old_timing['latency_median_ms']),round_medians_ms=json.loads(old_timing['round_medians_ms']),
        model='originalT05FullK1024',pairs=24,warmup_per_pair=old_protocol['timing']['warmup_per_pair'],repeats_per_pair_per_round=old_protocol['timing']['repeats_per_pair_per_round'],rounds=3,
        CPUuint8_to_affine=True,includes_preprocessing_H2D_D2H_filter_RANSAC=True,process_policy='one long process cyclingT05/XoFTR/RoMa; load one model at a time; separate freshprocess authorityforpeak',
        filter_implementation='sglm.external_reference.valid_correspondences',precision='BF16',memory_authority='reports/external_references_v1/T05_memory.json'),
        dict(record='SAR_reference_table',median_ms=float(sar_comparison['median_E2E_ms']),round_medians_ms=json.loads(sar_comparison['round_medians_ms']),model='originalT05FullK1024',pairs=24,warmup_per_pair=3,repeats_per_pair_per_round=5,rounds=3,
            CPUuint8_to_affine=True,includes_preprocessing_H2D_D2H_filter_RANSAC=True,process_policy='freshsinglemodelprocess eachround, rotatedorderamongallmethods',filter_implementation='sglm.sar_reference_geometry.filter_correspondences (chunk capable)',precision='BF16',memory_authority='samefreshprocessroundrecords'),
        dict(record='this_controlled_audit',**timing[0],model='originalT05FullK1024',warmup_per_pair=3,repeats_per_pair_per_round=5,rounds=3,CPUuint8_to_affine=True,includes_preprocessing_H2D_D2H_filter_RANSAC=True,
            process_policy='freshsinglemodelprocess eachround, T05/FH alternation',filter_implementation='sameSARReference adapter as priorSARtable',memory_authority='samefreshprocessroundrecords')]
    (out/'timing_history_audit.json').write_text(json.dumps(dict(records=history,common_input_indices_sha256=hashlib.sha256(Path('reports/t05_frozen_v1/subsets.json').read_bytes()).hexdigest(),
        conclusion='BotholdvaluesalreadyincludeRANSAC and CPUinputboundary; processlifecycle/filterimplementation/timingstatesdiffer. No causalattribution or speedup inferredfrom3.5percenthistoricalgap; currentjointtableused consistently.',
        shared_CPU_not_isolated=True,uncontrolled_GPU_clocks_logged=True,no_cherry_picking_rounds=True),indent=2))
    identity=json.loads((out/'identity_null_summary.json').read_text());reviews=json.loads((out/'visual_review_notes.json').read_text());assert reviews['all30_reviewed']
    timing_table=['| 方法 | 内部精度 | 端到端中位 | P95 | 峰值allocated | 峰值reserved |','|---|---|---:|---:|---:|---:|']
    for r in timing:timing_table.append(f"| {'T05全量K1024' if r['method']=='T05_Full' else 'FHReg-GUSO512'} | {r['precision']} | {r['median_E2E_ms']:.3f}ms | {r['P95_E2E_ms']:.3f}ms | {r['peak_allocated_MiB']:.2f}MiB | {r['peak_reserved_MiB']:.2f}MiB |")
    coordinate_table=['| 无网络测试 | 点数 | 全链最大误差（px） | 全链平均误差最大值（px） | 独立探针拟合RMSE（px） |','|---|---:|---:|---:|---:|']
    for r in coord['cases']:coordinate_table.append(f"| {r['name']} | {r['points']} | {max(e['max_px'] for e in r['errors'].values()):.3g} | {max(e['mean_px'] for e in r['errors'].values()):.3g} | {r['known_affine_independent_probe_rmse_px']:.3g} |")
    category_labels={'FHReg_success_T05_failure':'FHReg成功、T05失败','T05_success_FHReg_failure':'T05成功、FHReg失败','both_success':'两者都成功'}
    review_table=['| 分组 | pair_id | FHReg RMSE | 逐对视觉观察（定性） |','|---|---|---:|---|']
    byid={r['pair_id']:r for r in selection['pairs']}
    for r in reviews['pairs']:
        error=byid[r['pair_id']]['FHReg_rmse_px'];review_table.append(f"| {category_labels[r['category']]} | {r['pair_id']} | {'无解' if error is None else f'{error:.3f}'} | {r['observation_zh']} |")
    pools=selection['eligible_pools'];T,F=timing
    text=f'''# FHReg结果线核查：随机30对、坐标、分母与统一测速

本轮坐标回环、SR分母核查及30对原结果复现均通过。没有发现一个统一的坐标方向/尺度错误可解释FHReg的65.44%成绩；多例道路与建筑布局有跨模态结构支持。但重复农田、低纹理区域不能仅凭内点数或探针成功证明每条物理对应正确，也存在明显错配和近恒等失败。原精度表可作为明确协议下的名义几何patch评价解释，不能据此宣称所有匹配都经人工物理真值验证。

本轮不训练、不微调、不改变阈值/RANSAC；仅对固定30对重新运行网络取完整证据，没有重跑全部952对网络精度。952对固定台账从已冻结记录独立重算。T05对照始终为原epoch18全量K1024，与9.787ms/10.14ms两张历史表一致；没有混入K256或微调精化器。

## 一、随机抽样与保存证据

按SR@5px原结果分层，seed={selection['seed']}，每层按pair_id排序后无放回随机抽10对，保留抽签顺序，无人工换例。候选池分别为：FHReg成功/T05失败{pools['FHReg_success_T05_failure']}对、T05成功/FHReg失败{pools['T05_success_FHReg_failure']}对、两者成功{pools['both_success']}对。原始抽签在random30_selection.json/csv。

每对保存光学原图、未经扰动SAR原图、实际送入模型的SAR_aug、两张512模型输入、原始网络匹配（native512和还原256）、公共过滤后的匹配、RANSAC内点、独立探针、估计变换叠加、棋盘格和全量坐标CSV/NPZ。实际配准为光学→SAR_aug；叠加时用其逆矩阵将SAR_aug扭回光学，没有把未扰动SAR直接混到匹配图上。

“原始匹配”指FHReg内部原生粗/细阈值之后、公共有效区过滤之前的网络输出，不伪称未阈值的全部相关矩阵。原始/过滤/内点图显示全部点位置，连线按原顺序均匀取最多80条；精确全点另存CSV。额外固定显示6条均匀取样的内点局部patch，不根据GT误差挑点。展示抽样不参与拟合。

30对均逐项复现之前的原始/过滤数量、估计矩阵、内点数、独立探针误差及成功标记。抽中无匹配的pair726原样保留：无变换时输出“FIT FAILED”占位，不用GT变换生成预测叠加；名义GT叠加单独标注、只作诊断。

逐对观察已覆盖三张完整叠加/棋盘格联系表与三张固定局部对应patch联系表，详见下表、visual_review_notes.csv/json。它们是助手的定性图像审查，不是新增人工对应GT、不是OS语义标签，也不能估计全部623个成功样本的物理准确率。分层抽样按结果选组，不能将30对视作总体无偏物理正确率抽样。

{chr(10).join(review_table)}

代表性发现：698、460、428、693等道路/建筑布局对齐提供结构证据；875、907、669、9、789等重复农田较难逐点确认。失败534/942虽各有3600个RANSAC内点，仍有明显整体偏移且估计接近恒等，说明大量一致内点可来自错误几何。364/170出现严重拉伸错位；726无解。938仅略超5px，未因目测“大致对齐”而改成功标签。

另用原固定独立探针对“固定恒等变换”作只读零假设核查，不运行新网络或重新拟合：恒等SR@5={100*identity['identity_SR5']:.2f}%（5/952），FHReg为65.44%；FHReg成功而恒等失败{identity['FHReg_success_identity_failed']}对，未见1e-6矩阵容差内的精确恒等预测。它排除了“全表只是固定恒等输出”的解释，但不证明每条对应真实，也不等价于对图像内容因果依赖的验证。局部近恒等失败仍如实记录。见identity_null_all952.csv和random30_identity_diagnostic.csv。

证据像素与全量匹配数组留在私有runs/fhreg_result_audit_v1/evidence，并同步本地交付目录；Git只保存清单、指标、哈希和文字，不发布影像或权重。本地证据索引提供每对原图、原始/过滤/内点图、探针及叠加的直达链接。

## 二、无网络坐标回归

{chr(10).join(coordinate_table)}

七种测试各生成1000个随机点+5个角点/亚像素点，已知平移或完整仿射。先用源图预处理矩阵Ps映射，再用Pt*A*inverse(Ps)转换目标坐标，最后调用FHReg生产路径原函数restore_resized_coordinates还原。除了源/目标往返，还检查恢复源点的前向映射、A逆变换回环、独立探针仿射拟合；x/y互换和错误逆向作为负对照必须显著增大误差。

全链最大误差{coord['max_error_px']:.6g}px，所有等点数场景/子链合并平均误差{coord['mean_error_px_mean_across_equal_size_cases']:.6g}px，均低于事前数值验收1e-4px；各子链最大/均值在coordinate_roundtrip.json。像素中心规则为native=(original+.5)*scale−.5，逆向为(original_restore)=(native+.5)/scale−.5；crop在resize后扣除左上原点，padding再加左/上偏移，源/目标两边可不同。独立角点断言防止错误公式互相抵消而“自洽通过”。

实际FHReg主推理只执行256→512，没有crop和padding；后两类是明确分开的合成扩展测试，不声称原生主流程执行了它们。没有因此修改生产转换或重算整张精度表。9项针对测试通过，全仓库154项通过（依赖/旧NumPy共4条弃用警告，无失败）。

## 三、952对固定台账与统计分母

固定表FHReg_all952_fixed_ledger.csv具有pair_id、fit_success、num_matches、num_inliers、probe_error、success_at_3px、success_at_5px、failure_reason，并补记raw数量、公共过滤数与fit_status。num_matches是公共过滤后送入估计器的对应数，raw另列，不能混用。

独立重算：SR@3=415/952={100*den['SR3']:.6f}%；SR@5=623/952={100*den['SR5']:.6f}%。成功标记由fit_success与有限probe_error严格<阈值重新生成，并与旧记录逐对核对，不直接求原标记均值。

21个求解失败包括14个empty_matches、6个insufficient_matches、1个ransac_failed_or_insufficient_inliers；它们全部保留在952分母中。另有308个求解成功但RMSE≥5px，所以未达5px总数=21+308=329。失败原因没有混同“求解失败”和“矩阵错误”。

RMSE中位数/P90使用全部931个有定义误差的求解成功影像对，包含这308个未达5px样本：中位{den['solved_median_RMSE_px']:.6f}px、P90={den['solved_P90_RMSE_px']:.6f}px、均值{den['solved_mean_RMSE_px']:.6f}px。21个无定义误差不以零填充，也不从SR删除。623个SR@5成功子集的中位数另为{den['SR5_success_only_median_RMSE_px']:.6f}px，只作口径对照，未替换主RMSE。既有极端坏例仍保留、误差未封顶。

## 四、T05与FHReg统一端到端测速

{chr(10).join(timing_table)}

同一物理RTX5090，batch1，原24对及每张CPU图像/掩码字节哈希完全相同；每对预热3次、每轮5次测量、3轮总15次/对。每模型每轮独立新进程，顺序T05/FH、FH/T05、T05/FH。每张前后检查独占GPU，共6×48=288次检查。CPU起点为原256uint8影像+掩码，包含全部本方法resize/归一化/H2D、未缓存原生推理、坐标D2H、公共有效区过滤和原完整仿射RANSAC；排除IO、合成扰动、模型初始化、GT与图像绘制。CUDA同步包围计时，无hook、无特征缓存。精度保留各原生设置，不宣称内部FP32/BF16相同。

显存均在新进程、每对预热后reset_peak，取24对和三轮的最大完整PyTorch allocated，包含模型/激活/库工作区，不扣工作区；reserved单独列出，不当作模型总内存。CPU共享，torch.set_num_threads(4)，其余库原生线程数、GPU时钟和硬件快照逐进程留档，不锁时钟、不干预其他任务。重复样本不能当作360个独立场景。

本轮T05三轮中位为{[round(v,3) for v in T['round_medians_ms']]}ms；FHReg为{[round(v,3) for v in F['round_medians_ms']]}ms。全部轮次保留，没有为了接近历史数值反复重跑或删轮。逐影像对配对差值见timing_pair_differences.csv，median差{float(np.median([r['paired_delta_ms'] for r in paired])):.3f}ms。

### 9.787ms与10.14ms的历史差异

两次都是原始T05全量K1024、相同24对、3次预热、3轮×5次测量、CPU256uint8→仿射边界，并且都包含公共过滤和RANSAC。因此差异不能解释成“有一次漏算后处理”或混了K256。

但计时进程生命周期不同：10.138ms来自一个长期进程中轮换加载T05/XoFTR/RoMa；9.787ms来自每模型每轮新进程。后者使用可处理稠密点的分块公共过滤函数，前者是旧过滤实现；对T05输出两者数值一致。旧表权威显存另外使用新进程测量，不能将旧共享计时进程的峰值当最终显存。两次运行时GPU时钟/共享CPU状态也不同，不能从历史记录分离每项影响的因果贡献。

旧10.138ms三轮中位为{[round(v,3) for v in history[0]['round_medians_ms']]}ms；旧9.787ms三轮中位为{[round(v,3) for v in history[1]['round_medians_ms']]}ms。本轮统一比较值为T05={T['median_E2E_ms']:.3f}ms、FHReg={F['median_E2E_ms']:.3f}ms。保留历史值及其协议，不把约3.5%差异宣称为架构加速，也不把本轮FH精度与旧57.452ms随意拼接。timing_history_audit.json逐项列出调用边界与差异；后续同一张方法表应采用同次共同协议的配对资源记录。

## 本轮结论与限制

坐标回环与分母没有发现足以否定FHReg65.44%名义几何评价的错误，30对复现也通过；因此该成绩可以在既定公开GUSO权重、512输入、公共仿射协议下解释。视觉证据支持多例结构对应，但不支持将密集内点全部认定为物理真值；重复农田、弱纹理、建筑散射/视差与近恒等失败仍需明确保留。没有新增独立人工地标GT，也没有因观察结果改任何评价成功标签。

不重跑全952网络、不改变外部权重/骨干/精化器/阈值/RANSAC、不训练、不使用测试集。完整复现与私有产物说明见docs/FHREG_RESULT_AUDIT_REPRODUCTION_zh.md；公开代码/配置与最终验收哈希留在本目录。
'''
    (out/'核查报告.md').write_text(text);print(chr(10).join(timing_table));print('SR5',den['success5'],'/',den['pairs'])


if __name__=='__main__':main()
