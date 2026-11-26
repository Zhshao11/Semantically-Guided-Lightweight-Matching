# T10-A：固定T05的离线错误候选语义诊断

日期：2026-11-26；起点提交`276b023d037a1ff71b7246dd8f0e98928c956d6d`。只做信号诊断，不训练、不新增损失、不以验证/测试挑规则。完整协议在`configs/t10a_protocol_v1.json`，结果在`reports/t10_results_v1/T10A实验报告.md`。

## 固定样本和模型

使用T05 epoch18检查点`runs/t05_frozen_v1/checkpoint.pt`，SHA256 `f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`。没有加载任何T08/T09微调模型。UniformB0 K1024用于离线提取全部粗相关，eval、requires_gradFalse、BN缓冲区冻结；不以该全量计算声称在线稀疏节省。

7912去重训练对有1978个双模态像素验证的512父图，每图4个子块。固定seed20261126，先抽128父图、每父图抽2子块，共256对；不根据图像内容、伪标签覆盖或匹配结果选择。原T06保存的每图仿射/增强范围原样复用，源影像SHA、矩阵和cache索引随计划保存。全部来自train。

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_t10a.py
PYTHONPATH=. .venv/bin/python scripts/prepare_t10a_permutation.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/extract_t10a_candidates.py
PYTHONPATH=. .venv/bin/python scripts/analyze_t10a_signal.py
PYTHONPATH=. .venv/bin/python scripts/render_t10a_cases.py
# 下面是看到主结果后的敏感性检查，不冒充预注册独立检验。
PYTHONPATH=. .venv/bin/python scripts/check_t10a_sensitivity.py
PYTHONPATH=. .venv/bin/python scripts/check_t10a_common_cohort.py
PYTHONPATH=. .venv/bin/python scripts/summarize_t10a.py
```

准备/提取入口拒绝覆盖已完成计划/缓存；重现实验应另建目录并同步修改加载路径。当前所有实验已完成，复现命令不是要求再次运行。

## 名义SAR候选语义与掩码

不对SAR运行光学教师。扰动SAR32×32候选中心为`y=(8u+3.5,8v+3.5)`，名义原坐标为`x=inverse(A_forward)*y`。在T07已经冻结的光学教师七类条件概率32×32网格上采样，使用`grid=2*(x+.5)/256-1`和`grid_sample(..., align_corners=False)`。这是复用已聚合的8×8教师概率，不是新生成像素级预测或SAR语义GT；保留粗网格的空间分辨率限制。

现有光学query权重必须>0；映射候选的所有非零双线性支撑角点也必须权重>0，不允许通过可靠/不可靠混合插值补标签。x限定粗网格完整支持域[3.5,251.5]，同时SAR候选是原T05完整8×8有效cell。query的A映射需在SAR有效像素侵蚀2px后区域内。原始配准残差未知，不能将名义采样称为真实跨模态语义标注。

对每个几何有效query，排除距`A(query)`≤16px的全部候选，再按描述子相关排序top5。这是潜在高分错误候选，不按模型no-match、概率阈值或互选过滤，并非最终预测匹配。这一步不读取语义概率/相似度，选完才应用语义可靠性条件；因此“高分错误”是保守排除带外的名义几何错误，不保证原始残差为零。每pair保存全部负样本池packbits、top5索引/分数、原矩阵、nominal坐标、query/candidate概率和mask。

## 随机与置换对照

随机对照逐query/每条hard关系固定抽最多5个、无放回；排除GT邻域和该query全部5个hard候选。三种对照为：

1. 全部可靠有效负样本均匀抽样。
2. 同query，GT欧氏距离分层floor(distance/16)、可靠性权重分层floor(weight×10)相同。
3. 在2基础上要求候选主伪类与hard完全相同；query主伪类另外分层报告。

配置的no_match_stratum只指没有合适的随机分层候选，不是模型no-match类别。无合适随机位置则保存-1/NaN，不退回较松匹配；全部缺失计覆盖，不填成零效应。距离与权重只是粗分层，连续变量仍有残余差，`control_balance.csv`报告距离差、权重差、类别比例及描述子分数。

候选语义null使用全7912训练缓存，在每个原主伪类内按权重排序分8层，完整七类概率向量作一一置换；按verified512父图分组循环移位，所有接受格donor来自别的父图。query概率、原位置mask/weight固定。若某类无法8层跨父图置换，预定退到4/2/1层并记录，不能改类别组成或混入val/test。现有Barren主类没有可靠格，保持无数据。

节点处主类严格保持，但双线性混合后的主类可能变化。正式的同类“真实富集减null富集”只在采样后主类仍一致、且相同随机候选可比的关系上计算。null也改变软分布平滑性/空间连续性，只是条件置换对照，不是纯语义正确性干预。

## 统计与敏感性边界

软余弦为`dot/(norm_product+1e-8)`。随机5次先在relation内平均；hard槽在query内平均、query在pair内平均、2个pair在512父图内平均，最后父图等权。128父图整组有放回配对bootstrap20000次，同一关系的hard/random/null保持配对。没有把65万关系当作独立场景，原始卫星scene仍未知。

`similarity_histograms.csv`给50bin层级加权分布；CDF使用随机候选个体分布，`similarity_distribution.csv`也单列每relation随机均值分布，不混淆均值与个体。CI只针对有效可比的父图条件均值，233/256图、126/128父图有主关系；缺失覆盖始终单列。

主分析在提取前固定：距离/可靠性控制差、进一步候选类控制、条件置换、query伪类与去掉单一query类。探索性正门槛预定为主差和类控制差的父图CI下界>0、至少两个非背景query类在≥16父图有同类正差；这不设定可用损失或训练授权，不作多重检验显著性承诺。

主结果后用`check_t10a_common_cohort.py`在相同637136条可比关系上对比控制层级；它是描述性敏感性，不把衰减比例当因果解释方差。另加`check_t10a_sensitivity.py`：仅从已保存候选/随机集合筛距离差≤2px且权重差≤.02的严格子集；或过滤排除半径>24/>32px；或仅rank1。没有重抽、更改主候选集合或主阈值。它们是事后敏感性，不是独立预注册复验，覆盖和时间顺序明确列于notes。

## 固定案例与交付

前16个plan影像对原顺序展示；query由UID哈希产生起点后按循环位置/候选rank确定，要求可比但不根据相似度差选例，无效样例也原位显示。四页显示光学query、SARaug错误/随机位置、A⁻¹映回光学、教师伪分割及概率。SAR绿叉仅为A(query)名义参考，不称人工GT。此类图只作说明，不作为语义正确率评估。

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q \
  --junitxml=reports/t10_results_v1/tests.xml > reports/t10_results_v1/tests.txt
PYTHONPATH=. .venv/bin/python scripts/verify_t10a.py
```

测试涵盖逆矩阵方向、粗网格半像素坐标、严格插值mask、排除带和随机分层、层级权重/父图bootstrap及软余弦。跨文件验收检查256训练ID、矩阵/坐标、候选池/随机关系、top5排名、类别分层、跨父图donor、输入SHA、模型冻结、固定案例和无训练。

代码、配置、非像素指标/计划同步GitHub。全部逐候选数组、语义置换donor、图像案例留本地outputs/zzvr runs；教师权重未重新下载或公开。当前研究教师仍受T07的非商业研究/评价许可范围约束，不产生部署授权声明。
