# 冻结热图、只改坐标读出的A/B/C复现

本轮没有网络训练或推理，不重新计算相关热图/温度/特征，不改候选、掩码、分数或RANSAC。原T05 K256与同952对固定验证的冻结缓存来自`reports/window_oracle_v1/cache_freeze.json`，数据私有目录`runs/window_oracle_v1/candidates/`。完整协议：`configs/heatmap_readout_v1.json`，结论与主表：`reports/heatmap_readout_v1/实验报告.md`。

## 已核实原实现

`sglm/t02/model.py:LocalCorrelation.forward`在有效窗口上用原点积温度softmax得到FP32概率，再执行`center+2*(heatmap@delta)`。原评估BF16 autocast使矩阵乘的读出包含BF16量化；本实验A/C保持该算术，不同时换成FP64来制造定位差异。

窗口来自实际7×7细采样格、stride2原图像素和逐token有效掩码。A为原全有效窗口期望，B为最大概率的有效格（并列取原行优先第一格），C为同一最大峰周围固定3×3细格与原mask相交后，对原概率重新归一化再期望。3格跨度4原图像素，不是3×3原图像素，不能越过原窗口或恢复原无效格。

## 数据前提

- 原T05 epoch18检查点SHA256：`f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`；只为核对与读取已冻结温度，不构造模型。
- `cache_freeze.json`逐候选文件SHA和原模型/几何代码SHA必须一致。
- 原候选共122881条，实际有效凸包命中45974条，不能改回旧矩形命中46090条。
- 每个npz保存原heatmap、目标/源坐标、预测窗口中心和采样点、逐token有效mask、原过滤mask、候选ID、置信度、名义真值、独立探针。所有分支首先在完整原精化输入packet上读出，再使用原冻结拟合mask，避免改变矩阵乘形状或候选集合。

## 运行

使用zzvr已有Torch2.11.0+cu128环境及可执行BF16读出的5090。该GPU仅用于重放小矩阵读出，不运行编码器或匹配网络，也不把耗时作为方法速度。

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
 .venv/bin/python -m pytest -q tests/test_heatmap_readout.py
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python scripts/evaluate_heatmap_readout.py
```

评价入口拒绝覆盖已完成结果；其读出API没有GT参数。三种读出统一作用于全部拟合候选，GT命中与否不参与分支选择。A必须从原热图逐位重构原坐标，且952对原矩阵/评价精确复现历史T05K256。B/C使用原score和source，在每对每分支重置原seed20261125后独立重估完整仿射；不复用A内点、不追加探针或恢复候选。

没有基于新坐标重新筛掩码、错误距离或置信度；估计器内部原有finite/minscore检查仍存在，三分支数据均有限、相同score，集合不改变。全部952对在SR分母中，RMSE中位/P90按原求解成功范围报告。

## 事后热图诊断

`describe_frozen_heatmap`只在全部三种坐标与拟合完成后调用，结果从不进入预测决策。

- 真值附近质量：名义GT最近的原有效采样格为中心，固定3×3有效邻域内原全局概率质量；并列按原行优先顺序。
- 最大峰距离：B的原图坐标与名义GT的欧氏距离。
- 竞争次峰：主峰3×3以外的有效局部极大值，至少大于一个有效邻居；取最大者。固定比值≥0.5作为“明显竞争”诊断，未作阈值扫描或推理门控。
- 平坦最大值plateau单独统计，不伪造一个次峰。

45974条窗口命中对应保留相同集合，逐点误差及概率分布输出到私有`runs/heatmap_readout_v1/all45974_inside_correspondences.csv`，每对完整所有候选坐标/ID/score/错误/诊断保存为`0000.npz`等。公开目录只放聚合分布、逐影像对结果与几何估计；不发布原热图或逐对应数组。

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
 .venv/bin/python scripts/render_heatmap_readout.py
PYTHONPATH=. .venv/bin/python scripts/report_heatmap_readout.py
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python scripts/verify_heatmap_readout.py
```

可视化固定原T05诊断子集前6对的第一条拟合候选，不按GT或结果挑例；三图显示同一原概率，标记各读出与C邻域。实际GT出窗口时不扩大显示框冒充命中。局部图私有，CDF/配对散点只用聚合/逐对指标可公开。

最终验证重新核对全部缓存SHA、候选ID/顺序/source/score/过滤mask，重放A/B/C并逐对再拟合/评价；B峰和C邻域用独立NumPy行列逻辑复核。原坐标与A重放逐位一致。新增7项测试，全仓库161项通过。

## 结论与后续边界

C对A仅净增2个SR@5成功、P90和窗口内均值均更差，不能据+0.21个百分点宣称机制有效。保留A、不继续扫描读出/温度/邻域、不启动训练。当前结果未满足“局部定位与最终配准共同改善”的后续成本测量前提，因此本轮不测在线延迟/显存，也不能把缓存算子耗时当部署开销。

只借鉴Efficient LoFTR的峰附近局部读出思想，不是完整两阶段实现。验证集已经多次用于开发；未来配置须固定后在未参与选择的测试集检验，当前实验不读取测试集，也不直接把此前窗口理想定位Oracle成绩当成可实现上限。
