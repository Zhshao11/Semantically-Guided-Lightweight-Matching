"""Deliver T10-A distributions, coverage, pair/parent differences and fixedcases."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def read(path):return json.loads(Path(path).read_text())


def main():
    out=Path('reports/t10_results_v1');p=read('configs/t10a_protocol_v1.json');s=read(out/'signal_summary.json');plan=read(p['samples']['plan'])['samples'];data=dict(np.load('runs/t10a_candidates_v1/relations.npz'));names=read('reports/t07_teacher_v1/teacher_freeze.json')['target_classes'];class_counts={name:dict(total_optical_grid=0,reliable_optical_grid=0,geometry_reliable_optical_grid=0,reliable_nominal_SAR_grid=0,hard_relations=0,queries_with_hard=0,queries_with_strict_random=0) for name in names};coverage=read('runs/t10a_candidates_v1/coverage.json');sensitivity=list(csv.DictReader((out/'post_primary_sensitivity.csv').open()));notes=read(out/'post_primary_sensitivity_notes.json')
    for i,item in enumerate(plan):
        z=np.load(f'runs/t10a_candidates_v1/pair_{i:03d}.npz');qc=z['query_probability7'].argmax(1);cc=z['candidate_probability7'].argmax(1);selected=data['pair_index']==i
        for c,name in enumerate(names):
            record=class_counts[name];record['total_optical_grid']+=int((qc==c).sum());record['reliable_optical_grid']+=int(((qc==c)&z['query_semantic_valid']).sum());record['geometry_reliable_optical_grid']+=int(((qc==c)&z['query_semantic_valid']&z['query_geometry_valid']).sum());record['reliable_nominal_SAR_grid']+=int(((cc==c)&z['candidate_semantic_valid']).sum());rows=selected&(data['query_class']==c);record['hard_relations']+=int(rows.sum());record['queries_with_hard']+=len(np.unique(data['query_index'][rows]));record['queries_with_strict_random']+=len(np.unique(data['query_index'][rows&(data['random_id'][:,2]>=0).any(1)]))
    classes=[]
    for name,record in class_counts.items():classes.append(dict(query_pseudoclass=name,**record,reliable_fraction_of_class=record['reliable_optical_grid']/record['total_optical_grid'] if record['total_optical_grid'] else None,hard_query_fraction_of_class=record['queries_with_hard']/record['total_optical_grid'] if record['total_optical_grid'] else None,strict_random_queries_fraction_of_class=record['queries_with_strict_random']/record['total_optical_grid'] if record['total_optical_grid'] else None))
    write_csv(out/'class_coverage.csv',classes)
    total=s['coverage'];primary=next(r for r in s['comparison'] if r['comparison']=='hard_minus_distance_weight');strict=next(r for r in s['comparison'] if r['comparison']=='hard_minus_distance_weight_class');raw=next(r for r in s['comparison'] if r['comparison']=='hard_minus_uncontrolled');tight=next(r for r in sensitivity if r['sensitivity']=='classmatched_tight_2px_weight0.02' and r['query_pseudoclass']=='all');leave=list(csv.DictReader((out/'leave_one_class_out.csv').open()));nonbg=next(r for r in leave if r['excluded_query_pseudoclass']=='Background' and r['comparison']=='hard_minus_distance_weight_class');rel=s['distribution'];means={r['relation_type']:r for r in rel if r['control_cohort']=='distance_weight_class'}
    common=read(out/'common_cohort_notes.json')['rows'];attenuation=1-common[2]['mean']/common[0]['mean'];same=next(r for r in s['controls'] if r['control']=='distance_weight_class');case=read('runs/t10a_cases_v1/case_order.json')
    verdict=dict(offline_signal='positive but small afterconditioning; exists in several pseudoclasses, not Backgroundonly',primary_delta=primary['mean'],classcontrolled_delta=strict['mean'],raw_to_classcontrolled_attenuation=attenuation,tighter_posthoc_delta=float(tight['mean']),predeclared_exploratory_gate=s['decision']['predeclared_gate_positive'],training_started=False,training_recommended_now=False,why_no_fulltraining='classcontrolledcosine difference about0.00102 andposthoc tightdifference0.000321, strong ceiling andremainingdomain/pseudolabel/smoothnessconfounding; enrichment doesnotprove usable loss or suppressionbenefit',not_negative_signal_claim=True,labels_not_SAR_semanticGT=True,validation_test_used=False)
    (out/'interpretation.json').write_text(json.dumps(verdict,indent=2))
    report=['# T10-A：不训练的高分错误候选语义信号诊断','',
      '日期：2026-11-26。本轮没有训练、没有新损失、没有验证/测试规则选择。目的仅检查当前T05描述子高分错误是否具有可利用的名义语义相似性；存在相关性不等于某种语义损失会改善配准。','',
      '## 冻结模型、样本与候选','',
      f"基础模型是T05 epoch18，SHA256 `{p['matcher_sha256']}`，没有继承T08/T09语义微调组。训练7912对中按固定seed先抽128个verified512父图、每父图随机2个256子块，共256对，无语义质量/图像内容筛选。仿射原样复用T06保存矩阵；样本/矩阵/父图/源文件SHA在plan.json，模型eval、requires_gradFalse、BN缓冲区不变。",'',
      '使用全部1024个光学粗查询与1024个SAR候选，冻结BF16描述子相关；这只是离线诊断，不宣称K稀疏在线机制。对几何有效查询，在剩余有效SAR位置中排除距A(query)≤16px的圆形邻域，再按描述子选top5，随后才检查语义可靠性，避免语义相似度参与候选选择。这些是raw描述子潜在错误候选，不按模型no-match/阈值/互选过滤，不能等同最终预测匹配。所有候选池packbits、top5ID/分数、随机对照ID、矩阵和坐标逐pair保存并哈希。','',
      'SAR候选坐标y在扰动帧，先算x=A⁻¹y，再在原光学T07外部教师七类条件概率32网格上双线性采样，align_corners=False、输入256半像素坐标。已有可靠性weight>0且所有双线性支持角点均可靠才接受；不跨过掩码孔洞补标签。映回坐标限定[3.5,251.5]，GT名义位置需在侵蚀2px有效区域。没有对SAR运行光学教师。','',
      '这些只是继承原始配准关系的名义候选语义分布，不是SAR语义真值；原始配准残差未知。16px排除带是保守诊断选择，不是测得残差。现有T07Barren主类弃权、整图弃权照用，不通过新阈值美化标签。','',
      '## 对照与统计单位','',
      '每条可靠top5错误关系固定采样最多5个随机负样本，均排除GT邻域及本查询全部5个高分错误。三层随机对照分别为：全部可靠负样本；同GT距离16px分层+可靠性权重0.1分层；再加候选主伪类完全一致。缺少匹配样本时保留缺失并计覆盖，不退回较松分层。','',
      '额外语义置换只作用于候选侧，查询概率不动：从7912训练缓存中按原主伪类及类内8层权重分位，对七类概率完整向量固定置换，donor排除同verified512父图，接收mask/weight不变。在网格节点严格保持主类，双线性插值后的主类可能变化，因此差中差另限插值后主类也相同的关系。完整donor数组和SHA保留。置换也改变空间连续性/软分布平滑性，不是纯语义正确性干预。','',
      '软相似度为七类余弦。随机重复先在每条关系内均值，然后top5槽在查询内平均、查询在影像内平均、两子块在父图内平均。128父图作20000次整组配对重采样。没有把65万关系或同父图子块当独立样本；置信区间仅对已知分发512父图，不建立原始卫星scene独立性。分类子表为探索性分层，无多重比较校正。','',
      '## 相似度分布与差异','',
      '| 比较（高分错误−随机或null） | 父图均值差 | 父图95%CI | 有效影像/256 | 有效父图/128 |','| --- | ---: | ---: | ---: | ---: |']
    for r in s['comparison']:report.append(f"| {r['comparison']} | {r['mean']:+.6f} | [{r['CI95_low']:+.6f}, {r['CI95_high']:+.6f}] | {r['valid_pairs']} | {r['valid_parents']} |")
    report.extend(['',f"未控制差异{raw['mean']:.6f}，控制距离/可靠性后{primary['mean']:.6f}，进一步控制候选主类后{strict['mean']:.6f}，在共同637136条可比关系上，未控差{common[0]['mean']:.6f}、距离/权重控差{common[1]['mean']:.6f}、同类控差{common[2]['mean']:.6f}，描述性衰减{attenuation:.2%}（不是因果解释方差）。严格同类对照的高分错误平均余弦{means['hard']['hierarchical_mean']:.6f}，随机候选{means['random_individual']['hierarchical_mean']:.6f}，两者中位数均接近1，存在明显相似度饱和。",'',
      f"同类严格对照中，query/candidate同主类比例在两组完全一致（{same['hard_same_query_class_fraction']:.2%}）；平均GT距离分别{same['hard_distance_mean']:.2f}/{same['random_distance_mean']:.2f}px，绝对间距差均值{same['mean_abs_distance_gap']:.2f}px，权重绝对差均值{same['mean_abs_weight_gap']:.4f}。分层不等于连续变量完全匹配，小量残余混杂须保留。",'',
      '原CDF/50bin直方图、P10/中位/P90及层级加权均值齐备。CDF主要画随机候选个体分布，另列每条关系5次随机均值分布，避免把均值收窄的分布冒充个体负样本分布。各对照用其共同可比较关系，不混用覆盖。','',
      '## 各伪类与覆盖','',
      '| 查询伪类 | 原粗格数 | 有可靠标签格比例 | 有错误候选查询比例 | 同类随机差 | 95%CI | 有效父图 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in classes:
        c=next(x for x in s['classes'] if x['query_pseudoclass']==r['query_pseudoclass'] and x['comparison']=='hard_minus_distance_weight_class');delta='—' if c['mean'] is None else f"{c['mean']:+.6f}";ci='—' if c['mean'] is None else f"[{c['CI95_low']:+.6f}, {c['CI95_high']:+.6f}]"
        report.append(f"| {r['query_pseudoclass']} | {r['total_optical_grid']} | {r['reliable_fraction_of_class']:.2%} | {r['hard_query_fraction_of_class']:.2%} | {delta} | {ci} | {c['valid_parents']} |" if r['total_optical_grid'] else f"| {r['query_pseudoclass']} | 0 | — | — | {delta} | {ci} | {c['valid_parents']} |")
    report.extend(['',f"去掉Background查询后，严格同类差为{float(nonbg['mean']):+.6f}，父图CI[{float(nonbg['CI95_low']):+.6f},{float(nonbg['CI95_high']):+.6f}]，因此不能说信号只来自单一背景大类。Building/Road/Forest/Agricultural的同类差为正，但Road/Forest仅21个有效父图，Forest下界很接近零；Water区间跨零，Barren未获得现有可靠主类监督，不能填成零效应或称七类普遍成立。",'',
      f"总262,144查询格，几何有效{total['geometric_queries']:,}、现有可靠标签{total['existing_semantic_queries']:,}、两者交集{total['both_valid_queries']:,}；有可靠错误关系的查询{total['reliable_queries_with_hard']:,}。共653,721条可靠错误关系；距离/权重随机可比644,114条，同类严格可比637,136条。有效影像233/256、父图126/128；其余23图/2父图没有可比较关系，单独报覆盖，不假设它们是零信号。",'',
      '## 主分析之后的敏感性检查','',
      '以下在看到主结果后添加，只过滤原已保存候选/随机集合，不改主分析、候选选择或预定排除半径，也不按结果调参数。','',
      '| 敏感性（同类随机） | 差值 | 95%CI | 关系数 | 有效父图 |','| --- | ---: | ---: | ---: | ---: |'])
    for r in sensitivity:
        if r['query_pseudoclass']=='all':report.append(f"| {r['sensitivity']} | {float(r['mean']):+.6f} | [{float(r['CI95_low']):+.6f},{float(r['CI95_high']):+.6f}] | {r['relations']} | {r['valid_parents']} |")
    report.extend(['',f"把同类随机差进一步限制为距离差≤2px且权重差≤0.02，剩余差仅{float(tight['mean']):.6f}（CI[{float(tight['CI95_low']):.6f},{float(tight['CI95_high']):.6f}]）；仍为正但效应更小、覆盖更少。排除24/32px及只看rank1的补充结果另列，不能冒充预注册独立复验。",'',
      '## 固定顺序案例与结论','',
      '前16个冻结影像对按原plan顺序展示，查询由UID哈希起点和候选rank确定，只要求可比较，不按语义差排序。无可比关系的影像也原位展示，不换成好例。四页同时给光学查询、扰动SAR高分错误/随机点、A⁻¹映回光学位置、教师伪分割及概率柱。case_order.json保留规则和ID。助手已查看四页；不作为人工语义GT评审。','',
      '可见多幅农田/背景样例的query、错误和随机候选均几乎同类、余弦接近1；也有跨建筑/背景的低相似错误。案例强调大块同类与教师漏分都可能影响诊断，不能凭例图反推教师准确率。','',
      '结论：存在跨若干伪类别的弱相似性富集，不是完全无信号，也不是仅背景大类；但差异在距离、可靠性和类别组成控制后大幅缩小，不能把描述性衰减当作因果归因，严格控制后的剩余约0.00102，更紧后约0.000321。它满足本轮预先设定的探索性正信号门槛，却不能证明可用的新损失、错误抑制或配准收益。结合语义饱和、伪标签跨域误差和残余空间混杂，本轮不进入全量训练，也不强行实现新损失；保留该弱信号供后续假设判断。','',
      '候选、随机集合、父图汇总、概率置换、所有输入SHA与无训练验收已交付。方法代码/配置/非像素指标同步本地、zzvr及GitHub；原始影像/案例图、概率与候选数组保留私有。教师许可延续非商业研究/评价范围，不发布权重、不宣称部署授权。'])
    (out/'T10A实验报告.md').write_text('\n'.join(report)+'\n');print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()
