# T08：外部教师驱动的弱语义辅助实验复现

日期：2026-11-26；起点提交`5be7e120d4c994217e4327060dee358aec62cb65`。完整参数见`configs/t08_protocol_v1.json`。OS只有已配准光学/SAR影像，没有人工语义GT；几何监督来自配准假设和合成仿射，语义监督仅为外部教师的光学软伪标签。不能报告OS语义精度、用不存在的精度选模或把配准收益直接解释为语义预测更准确。

## 前置诊断与数据流

复用T07已冻结、研究用途许可明确的官方MiT-B2来源TG和7912张训练光学伪标签，未重训教师。原低可靠区域/整图弃权不变；教师质量上的Barren主类弃权不等于把某类定义为不适合匹配。`reports/t08_preflight_v1/pseudo_label_audit.json`给出网格数量、类别占比与软概率质量；四邻域稳定性仅作诊断，没有据此调mask。

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_t08.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/diagnose_t08_pretrain.py
PYTHONPATH=. .venv/bin/python scripts/audit_t08_labels.py
PYTHONPATH=. .venv/bin/python scripts/prepare_t08_shuffle.py
```

setup固定训练7912对及T06保存仿射、诊断128对train。16张可选人工复核清单来自训练、独立于旧32抽查及128探针选择，但没有人工标注；不会作为开训条件，不从新训练中剔除。最早“留16张待标注”的规划由用户明确纠正，存于`superseded_before_GT_clarification/`，从未用于正式训练；当前setup/训练计划才是依据。T07学生已经见过这些训练影像，不把可选清单称为对其完全未见内容的测试。

128对离线诊断固定K256，对比T07学生、关闭修正仅用同头外观、全图教师概率替换、可靠格教师概率替换。替换只改变语义概率输入，保持评分头/匹配器/阈值不变；检查点和train来源均记录。教师不是真值，这个诊断不能当作oracle性能上限。此处不生成OS val/test语义缓存，诊断结果不用于调参。

## 三组控制

M、MS、MS-shuffle均从T05 epoch18/step35604初始化，SHA256：

```text
f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90
```

固定均匀K256，7912对同一训练图/保存仿射、相同seed和每epoch排列。batch4、AdamW lr2e-4/WD1e-4、clip5、无调度器，4epoch/7912更新。4轮只是首轮观察预算、单seed实验，不宣称收敛或多seed稳定；不因曲线/最后结果另改预算。主要结果为固定last epoch4，按验证SR@5保存的best仅作诊断，不换主结果。

只训练粗层：`backbone.down.1.*`、`backbone.stages.2.*`、`backbone.lateral.2.*`、`backbone.descriptor.*`。所有骨干BN running stats冻结；允许粗层BN affine训练。适配器、stage0/1、低层下采样、其余FPN、局部精化参数以及no-match/温度冻结。FPN参数冻结不意味着细特征张量不变，因为它仍接收更新后的粗激活。逐参数数量和hash见各run的architecture.json。

MS和MS-shuffle语义辅助头挂光学64维粗特征，同从T07已训练语义分支初始化；输入不detach，KL反传至光学粗编码层。共享卷积会间接改变SAR特征，但没有SAR像素语义标签/语义KL。M没有辅助损失，MS为匹配损失+lambda1的加权KL，shuffle使用打乱软伪标签；KL沿用绝对可靠性权重，除以网格数而非sum(weight)。

匹配损失保留双向CE（正/无对应分别求均值，无对应权重.25）和.2的细化损失。温启动不重新使用GT bootstrap，细化训练窗口来自预测粗对应。K256前向GT按query_indices收集；反向类别从原1024映射到所选256，未选源位置归入no-match256。原无效源仍ignore，越界不作正例。K1024极限与原始b0_loss数值一致性有专项测试。几何算子和独立探针评价器没有重写。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/train_t08.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --group M
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t08.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --group MS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/train_t08.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --group MS-shuffle
```

三个运行分别独占空闲GPU3/4/5。已有run拒绝覆盖；重现实验请改独立目录及下游路径。当前最小脚本不实现中途恢复，不声明续跑逐位等价；保留每轮last、验证best及最终纯匹配inference_last.pt。

## 打乱伪标签

受监督格按权重排序分成32个等数量层；每层中对完整七类概率向量作固定跨影像置换，禁止donor来自同一影像。接收侧mask和绝对权重逐位保持，全局接受格的软类别直方图为同一组向量，严格保持。权重层内细微差异意味着加权类别质量不完全相同，其最大相对差约0.23%，逐类差异完整报告。

不保持每幅影像的类别直方图，也不声称空间边界/频谱等统计完全等价。它是破坏内容对应关系的辅助诊断，不能当作完美等价正则化。私有donor_flat_index.npy记录每格映射，summary.json记录SHA和跨影像检查。

## 验证、导出与解释

三组epoch0均在全部952固定验证上逐pair复现T05原K256结果。epoch1/2/4同样完整验证K256；epoch4另用相同last做K1024评价，只为检查密集收益能否传至稀疏，不改训练K或以此选模型。保留原probe16×16、SR@1/3/5、RMSE/失败分母及原RANSAC3px/min6/5000/.999。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/evaluate_t08_diagnostics.py
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q \
  --junitxml=reports/t08_delivery_v1/tests.xml > reports/t08_delivery_v1/tests.txt
PYTHONPATH=. .venv/bin/python scripts/summarize_t08.py
PYTHONPATH=. .venv/bin/python scripts/verify_t08.py
```

分层诊断固定复用T05预先定义的96对val，记录粗正确率、入窗、精化前后与覆盖；完整精度仍以952为分母。用户数据没有OS语义GT，因此教师/学生主要错在哪只能沿用无GT抽查观察和外部LoveDA已有GT结果，不能新增OS语义准确率。

导出为原`UniformB0`，state_dict仅37219匹配参数，不含语义头/评分器/教师。同输入的训练模型eval和删除头副本输出一致性有测试。推理只做均匀选择/gather→256×1024相关→原精化→仿射估计。没有本轮正式计时，因此“移除辅助头无该分支在线开销”不等于“整网已经加速”。

必须先比较MS对M与shuffle。若MS和shuffle都改善，更接近辅助正则化解释，不能强称语义内容收益；若MS无M优势，本轮机制未获支持；若只密集改善而稀疏不改善，不能支撑帮助轻量化；单seed的微小差异不能称稳定。

代码/非像素计划、指标、每步损失和流SHA同步GitHub。所有权重、伪标签数组、置换映射和影像留本地outputs/zzvr runs，遵守T07非商业研究/评价许可范围，不发布权重、不宣称部署授权。
