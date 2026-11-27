# 真实计算路径与选择插入点

本轮沿用T05 epoch18原权重、K1024全量粗查询、BF16 autocast。不是T05的“先选K个查询、再做K×1024相关”；本轮先做原1024×1024相关并按原阈值/互选产生N个候选，之后只选M个进入局部精化。无训练、无语义、无新增评分头。

| 检查项 | 实际实现与含义 |
|---|---|
| 全图特征 | `sglm/model.py:Backbone.forward`，光学模态0、SAR模态1。每端粗特征B×64×32×32、细特征B×16×128×128、描述子B×1024×32。细特征已经在backbone/FPN全图生成，早于相关与选择；不是候选构建。 |
| 粗相关 | `sglm/t05.py:UniformB0.correlate`，固定k=1024，B×1024×1024，FP32温度/no-match与原互选规则不变。 |
| 候选输出 | `UniformB0.coarse_matches(output,index)`，source/target均N×2，score为N，N为该影像对通过原光学有效掩码、no-match、概率≥0.01和互选的数量。 |
| 候选坐标 | 原输入像素中心(x,y)，由原`coarse_xy`按粗网格索引生成；沿用`index*8+3.5`，选择器不再乘除步长。 |
| 原有顺序 | 32×32光学粗网格行序，布尔筛选保持行序；不是置信度排序。prefix优先上部区域，含空间偏置。 |
| 显式候选字段 | `CANDIDATE_FIELDS`集中列出source/target N×2、score N、batch_index N、valid_mask N、original_candidate_idx N。粗索引、其他元数据若未来新增须显式扩展schema；没有依赖“首维恰好N”自动识别。批次索引为影像编号，original_candidate_idx为该对影像内原候选位置，不是batch展平索引。 |
| 选择器 | `sglm/sparse_refinement.py:select_refinement_indices`只返回LongTensor索引；source仅作4×4名额分组，不改坐标/置信度，不读GT。all/None/M≥N直接返回原arange，无排序。coverage按行优先轮询水位名额、格内稳定置信度；global_conf与coverage均将最终集合按原索引升序输出，避免额外RANSAC顺序混杂。 |
| 插入点 | `SparseRefinementB0.refine`先生成packet并统一gather，随后进入`refine_packet`中的`self.local(...)`。这行是首个按候选逐点采窗的入口，M取代N。 |
| 窗口生成 | 原`sglm/t02/model.py:LocalCorrelation.forward`调用原`sample_windows`：光学radius0、SAR radius3。后者直接`grid_sample`采特征和有效掩码；没有整图unfold。光学每候选1 token、SAR每候选49 token。 |
| 坐标采样 | 原`sample_windows`保留`(centers+.5)/2-.5+delta`、`2*(positions+.5)/[W,H]-1`、bilinear与`align_corners=False`。候选边界按原valid判定，不夹紧采窗中心。覆盖网格的边界夹紧只影响归格。 |
| 局部计算 | 两端16→32共享线性投影、L2归一化、单对应内49位置点积/温度/softmax、期望偏移。softmax仅沿49窗口位置，无跨候选归一化/注意力/BN，参数不变。虽然代码调用`.float()`，autocast下Linear和末端矩阵乘仍可为BF16，不能称全FP32局部算子。 |
| 精化后过滤 | 继续沿用`local.valid`（光学有效且SAR窗口有有效点）过滤。选中数和最终有效数分别记录；只用有效精化点拟合，不补粗点、不失败回退。 |
| 几何接口 | 原`estimate_from_correspondences`接收source Nvalid×2、refined target Nvalid×2、粗score Nvalid；无新增置信度或语义。minscore0、完整仿射RANSAC阈值3px/min内点6/max迭代5000/confidence.999及种子均来自原配置。 |

## 真正减少与保持固定的成本

减少：逐对应光学/SAR掩码与特征采样、局部投影、局部相关/softmax/偏移、返回点与CPU复制数；RANSAC输入点与内点分布可能改变，耗时不一定单调减少。

固定：两端全图编码/BN/FPN/细特征、描述子、完整粗相关、no-match/互选/概率筛选。选择器本身增加开销：coverage的候选格容量传CPU用于16格水位分配，会同步设备；该成本包含在正式计时内，不隐藏。

无新增跨候选依赖。BF16全量与子集因矩阵行数改变可能出现形状相关的舍入差，初始严格回归失败及算子定位分别见`initial_regression_failure.txt`、`initial_subset_failure_diagnosis.json`、`bf16_operator_diagnosis.json`。不改算子，在扩大32对回归/完整验证前记录容差修订；FP32诊断用于区分索引错误和BF16数值差，不替代正式BF16评价。

## 计时边界

正式计时不缓存特征，batch1，每次重做完整预测。输入已驻GPU，排除I/O和CPU→GPU搬运，包含选择和精化后过滤。network边界至GPU预测输出；complete边界额外包含坐标GPU→CPU和原仿射RANSAC。无hook、无逐次写盘、无GT和可视化。

分阶段诊断另行运行，仅临时包裹`sample_windows`加入同步计时，退出后恢复原函数。窗口时间与local剩余时间用于定位，含同步/仪器开销，不能相加替代无hook端到端结果。该诊断不修改正式预测的任何算子。
