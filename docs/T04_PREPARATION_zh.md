# T04：独立于最终匹配权重的准备

本轮只做语义教师候选核查、训练影像预览、软目标算子、外部基线审计和汇总/计时工具。未生成最终可匹配性或可靠性标签，未训练联合评分器，未修改B0骨干、匹配损失、阈值或RANSAC配置。

结果见 [T04准备核查报告](../reports/t04_preparation_v1/T04准备核查报告.md)。影像预览和概率文件保存在本地 `outputs/T04_准备核查/teacher_preview/` 与服务器 `runs/teacher_preflight_v1/preview/`，不进入Git。第三方权重不进入Git。

## 教师候选与软标签接口

`configs/semantic_teacher_preflight_v1.json` 固定社区权重的revision、SHA256、八类声明、七类映射和预处理。该权重许可未明确、灰度OS影像适用性未验证，`production_teacher_ready=false`。

`aggregate_semantic_targets` 输入 `B×S×H×W` 概率、`B×H×W`有效掩码、`T×S`显式类映射；输出 `B×T×32×32` 条件概率，以及valid fraction、retained mass、excluded class mass、confidence和监督mask。它先在有效像素上做8×8概率均值，再合并类别；不使用argmax投票。Ignore被排除后的概率质量单独保留，不能只靠七类重归一化通过掩码。

预览顺序为：原始灰度uint8复制为RGB → 保存的处理器resize512与ImageNet归一化 → B2推理 → logits双线性还原native256 → softmax8 → 掩码面积均值 → 显式8→7映射。聚合结果为32×32。若未来增加光学几何变换，概率和有效mask必须同步变换；当前仅SAR几何扰动不移动光学语义目标。

`masked_semantic_kl` 对教师detach，只对有效位置计算KL；全mask时返回可反传的零值。本轮没有将它接入B0损失。

服务器独立教师环境：`runs/teacher_preflight_v1/.venv`，transformers4.44.2；没有往B0环境安装transformers。预览使用空闲GPU6，FP32、eval、inference_mode，逐项验证预处理和缓冲区不变。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/preview_semantic_teacher.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --weights runs/teacher_preflight_v1/teacher_candidate \
  --output runs/teacher_preflight_v1/preview --device cuda

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  tests/test_semantic_targets.py tests/test_experiment_io.py -q
```

旧的 `cache_teacher.py` 是要求另有已训练七类教师的历史模板，使用双线性缩小概率，不是本轮面积聚合协议；不能给它传入当前八类候选，更不能将其产物冒充已冻结的正式缓存。本轮运行入口是 `preview_semantic_teacher.py`。

## 汇总、分层与计时

```bash
PYTHONPATH=. .venv/bin/python scripts/summarize_experiments.py \
  --run runs/t03_main_lr1e3_v1 --run runs/t03_control_lr2e4_v1 \
  --diagnostics reports/t02_controlled_ABC_v1/group_A_pairs.json \
  --diagnostics reports/t02_controlled_ABC_v1/group_B_pairs.json \
  --diagnostics reports/t02_controlled_ABC_v1/group_C_pairs.json \
  --output reports/t04_preparation_v1/observations
```

输出epoch训练CSV、验证CSV、逐pair验证CSV、分层诊断CSV、曲线和JSON快照。检查完整验证的pair分母、eval与BN冻结；失败及阈值成功标志保留。运行中JSONL只容忍最后一行尚未写完，不吞掉文件中部损坏。分层导出复用已有事后诊断，不重跑预测；保留group、pair_id和variant_index，B组仍为32对影像各5次相关扰动。

`benchmark_b0_stages.py` 默认dry-run，仅准备协议，不加载最终权重、不测延迟：

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_b0_stages.py \
  --checkpoint runs/t03_main_lr1e3_v1/best.pt \
  --plan reports/t03_setup_v1/full_data_plan.json \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --output runs/timing_protocol_preparation
```

将来正式执行需显式添加 `--execute --gpu-uuid GPU-实际UUID`，先冻结候选检查点及比较协议。本轮没有执行该选项。脚本通过nvidia-smi核对物理UUID和全部计算进程，遇到其他进程或无法查询则拒绝；每轮测量前再次检查。T03固定val计划可直接适配，只有在线训练记录、没有保存矩阵的plan被拒绝。

batch1、BF16、eval、每对至少5次warmup/20次测量；分阶段同步计时，记录光学/SAR编码、粗相关与解码、精化、传回CPU、完整仿射RANSAC。数据读取（含校验/解码/warp）和传输单列；另做无hook的预测+RANSAC端到端计时，排除I/O与传输。记录实际匹配数、内点、求解状态、权重/plan SHA和环境。阶段同步有额外开销，阶段数据不能直接当作部署端到端延迟。当前只有dry-run及接口测试，CUDA计时路径待空闲独占卡验证；该脚本不等于已实现所有外部基线适配器。

## 基线复现环境核查

`configs/external_baselines_preflight_v1.json` 保存6个官方仓库的固定提交、论文日期、许可、requirements及来源文件哈希。近期检索直接访问arXiv，并按首次提交时间排序；官方代码链接另核验。没有使用SourceMux。

`audit_baseline_environment.py` 只读检查已有目录身份、修改状态、包版本和CPU导入/冒烟；不安装依赖、不触碰GPU、不修改第三方目录。UltraMatch快照按36个来源文件哈希验证，不借用其父目录的项目Git身份。

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/audit_baseline_environment.py \
  --repo XFeat=/mnt/sda/zhushihao/图像匹配/accelerated_features \
  --repo JamMa=/mnt/sda/zhushihao/图像匹配/JamMa \
  --repo EfficientLoFTR=/mnt/sda/zhushihao/图像匹配/EfficientLoFTR \
  --output runs/external_baselines_preflight/environment.json
```

官方旧环境多为Python3.8、PyTorch2.0、CUDA11.x，不能直接假设支持5090/sm120。下一步若实际复现，应逐方法建立独立环境，使用支持硬件的CUDA/PyTorch，并记录偏离官方版本的移植改动及数值核验。不能把当前项目环境降级来满足旧requirements。第三方权重的许可及哈希需在确定具体artifact时单独冻结。
