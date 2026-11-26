# T09：打乱对照审计与三个配对训练顺序种子

日期：2026-11-26。起点为T08提交`4496e4bf2ca1ad26af624d7b7f2d8b0af5943b1b`。目的仅是决定当前逐格光学伪语义KL是否值得保留，不改网络，不扩展预算，不根据结果重调lambda或改用best。主指标在配置中预先固定为K256的SR@5px；次指标SR@3、RMSE中位/P90；诊断粗正确率、入窗率和窗口内精化误差。

## T08审计与最小修正版

T08原打乱完整保留接收侧mask/weight，受监督概率向量跨影像置换、全局软类直方图保持，这些控制继续保留。复核verified512父图后发现：5,478,278个接受格中4,274个donor仍来自同一父图的其他256子块（0.0780%，涉及173张接收图）。T08原实验、模型、映射和历史数值不覆盖。

修正版`runs/t09_shuffle_parent_v1`仅在原32个权重分层内交换违反父图排除条件的donor，要求交换后的两端都来自不同verified512父图。除此以外保留原映射。全部接收mask/weight逐位不变，所有接受格donor仍为原集合的一一置换，七类概率和标签熵的全局分布保持，逐类加权质量微小偏差单列，不当作完全相等。原/新SHA和修复交换记录全部保留。

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_t09_shuffle.py
PYTHONPATH=. .venv/bin/python scripts/prepare_t09_index.py
PYTHONPATH=. .venv/bin/python scripts/audit_t09_label_fields.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/audit_t09_gradients.py
```

`run_index.json`在最终结果前冻结9个配对位置，7个新运行：种子20261126的M/MS复用T08完全相同检查点，只重训父图排除修正版shuffle；20261127、20261128各训练M/MS/MS-shuffle三组。原T08-shuffle数值另作历史，不与修正版混为一个种子结果。`T08_immutable_snapshot.json`记录历史权重/映射/源码哈希，验收再次核对。

T08映射由训练循环外一次生成，dataset构造时只读memmap；源代码没有每轮置换，因此固定是实现事实。T08没有逐epoch写映射SHA，不虚构历史日志。T09每epoch前后核对mapping/probability/weight SHA，同一修正版缓存用于全部三个种子。

逐格置换同时破坏内容对应与空间连续性，不能称为只改变“语义正确性”的完美对照。`label_coverage_entropy.csv`及`label_class_distribution.csv`记录接受覆盖、主类/软类/加权分布、标签熵和相邻同类率；空间相关性变化明确保留。

## 梯度量级审计

T08没有保存逐步语义粗层梯度，故只在同128对train组成的32个batch上回放：共同初始匹配器/语义头，以及各自T08最后状态。两组使用相同批次，matcher BN冻结，辅助头BN为train模式且每批重置到检查点buffers；使用一次forward后的autograd.grad，报告未全局裁剪前的粗编码层L2、匹配梯度L2、比值和余弦。排除描述子投影与辅助头参数。回放使用丢弃副本，不改历史检查点。

新运行另在实际训练步1/1978/3956/5934/7912记录同样的梯度指标，在现有forward图上计算，不额外前向、不修改`.grad`或优化器，不重复更新BN。历史回放与实际训练探针分别列出。KL始终lambda=1，观察到量级不同也不事后缩放。

## 配对训练协议

三个种子均从相同T05匹配器和T07辅助头状态初始化；随机种子影响训练顺序，初始模型不随机更换。复用T08的7912张保存仿射，跨种子几何矩阵也相同，各种子内部三组的样本/仿射流SHA必须完全一致。不能把它描述成三种随机初始化、三套新增强或三组独立验证影像。

相同解冻层、所有matching BN统计冻结、batch4、AdamW lr2e-4/WD1e-4、clip5、无scheduler、KL权重1、4epoch/7912更新；匹配阈值与RANSAC不变。M只有匹配损失；MS匹配+原光学外部教师软伪标签KL；MS-shuffle只替换为父图排除修正版伪标签。T08匹配模型、稀疏损失、独立几何评价器均原样复用。

```bash
# 每个新运行一次；按空闲卡分批运行，不覆盖既有目录。
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t09.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --seed 20261126 --group MS-shuffle
# 20261127与20261128各自运行M、MS、MS-shuffle，其余参数相同。
```

各新运行epoch0/1/2/4在同952固定验证上评价K256，主要结果只用最后epoch4；best即使保存，也不参与三张主表。没有新增K1024或其他预算运行；旧T08密集结果仅属历史，不进入T09主表。所有推理导出仍是无语义头的37219参数匹配器。

## 配对结果与父图重采样

```bash
# 每个种子的三个最终检查点齐备后，在空闲卡执行；T08同检查点诊断可直接复用。
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/diagnose_t09_last.py --seed 20261126
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/diagnose_t09_last.py --seed 20261127
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/diagnose_t09_last.py --seed 20261128
PYTHONPATH=. .venv/bin/python scripts/report_t09_bootstrap.py
PYTHONPATH=. .venv/bin/python scripts/summarize_t09.py
PYTHONPATH=. .venv/bin/python scripts/verify_t09.py
```

表1列9个种子/组别的完整结果及三种子均值、样本标准差（ddof1）。相同952影像在不同训练种子下重复评价，不能把2856当作独立场景数。

表2逐种子分别列MS对M、MS对修正版shuffle的失败→成功/成功→失败、双成功/双失败及净变化，成功固定为独立探针RMSE<5px。失败→成功不等同求解器从失败转成功，几何可解但误差超阈值也记失败。

表3按952对中已验证的238个分发512父图重采样，每次抽238个父图、有放回并完整保留其所有256子块，20000次、固定bootstrap seed20261126。同一次父图抽样同步用于两个方法和全部三个训练种子，报告逐种子及三固定种子平均配对差的percentile95%区间。SR差为百分点（正值有利MS）；RMSE差为像素（负值有利MS），分位误差只对可求解样本，失败始终计入SR分母。

三固定训练种子的平均区间是已知父图不确定性的条件区间，不估计训练种子总体分布；三个种子样本标准差另列。不根据CI改主阈值或选择有利次指标。没有找到原卫星scene来源，父图分组只处理已知局部相关性，不能建立原始卫星scene独立性。

## 测试和私有产物

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q
```

专项统计测试涵盖整父图抽样、重复父图保留所有子块、求解失败留在SR分母、同方法零配对CI，以及重复同一个种子不能人为缩窄父图区间。另验证autograd梯度探针在CPU单步不改优化器更新或BN buffers；不把这一性质夸大为不同GPU环境下全轨迹逐位复现。最终验收核对T08未覆盖、父图donor排除、epoch哈希、同种子流一致、固定last/更新数/lambda，以及三张表的分母/父图数。

原始/派生权重、伪标签数组、置换映射、bootstrap重复样本与逐对应预测只保留本地outputs和服务器runs。Git公开代码、配置、非像素指标、计划与修复映射的数值记录；OS无人工语义GT，不新增OS语义精度。教师未重训，研究许可边界仍按T07，不公开教师权重或声称部署授权。
