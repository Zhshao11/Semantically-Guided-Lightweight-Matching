# 固定K256局部精化受控比较

本轮只训练原精化器和一层轻量局部交互精化器。完整固定协议在`configs/local_interaction_v1.json`，结果在`reports/local_interaction_v1/report.md`。

## 数据与私有依赖

- 已审计OSDataset2.0原版256影像，训练7912对去重记录，验证952对。
- `reports/t03_setup_v1/full_data_plan.json`和原在线增强实现。
- T05 epoch18私有检查点`runs/t05_frozen_v1/checkpoint.pt`，SHA256为`f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`。
- `reports/t05_frozen_v1/validation_plan.json`保存全部验证矩阵；`subsets.json`保存原24对测速清单。
- 不涉及教师、语义缓存或测试集。

## 训练与评价

服务器项目目录内，使用已有Torch2.11+cu128环境：

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_local_interaction.py
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python scripts/train_local_refinement.py --root /path/to/OSDataset2.0
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python scripts/evaluate_local_refinement.py --root /path/to/OSDataset2.0
```

训练入口会拒绝覆盖已有检查点。两个优化器在同一步内依次更新，使用同一冻结特征、预测粗候选、矩阵和有效监督掩码；只更新`local.*`。每epoch核验冻结权重及BN缓冲哈希。固定4epoch/7912更新，最后检查点评价；不进行中间验证选模。

评价会独立复现原T05K256的952对历史矩阵/评价，然后检查训练后两组的候选哈希及精化数量完全相同。GT仅用于模型预测与RANSAC之后的诊断。逐对应数组留在`runs/local_interaction_v1/*_predictions/`，公开报告仅含汇总与逐影像对指标。

## 无hook端到端计时与独立显存

必须先使配置指定RTX5090上无其他CUDA进程，脚本会拒绝共卡正式计时：

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_local_refinement.py --root /path/to/OSDataset2.0
PYTHONPATH=. .venv/bin/python scripts/benchmark_local_refinement.py --root /path/to/OSDataset2.0 --memory-group original_refiner
PYTHONPATH=. .venv/bin/python scripts/benchmark_local_refinement.py --root /path/to/OSDataset2.0 --memory-group local_cross_attention
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. .venv/bin/python scripts/report_local_refinement.py
```

第一个进程交错测量两组延迟。后两个必须分别启动新进程，正式显存取各自`memory_*_summary.json`中的peak allocated，避免共享进程残留cuBLAS工作区污染比较。CPU uint8输入到仿射结果为共同边界，排除数据读取/合成扰动/GT，包含预处理、H2D、完整双模态编码、粗匹配、全部原预算精化、坐标D2H与RANSAC。

`coordinate_local_refinement_timing.py`是本次服务器专用协调入口：仅识别并临时释放本项目SOMA进程树，记录稳定检查点，串行测速后在finally恢复同一4GPU训练协议。该脚本写死了本次数据目录，不是通用任务管理工具。恢复后还需检查真实训练状态有新进度，不能把Popen成功当成恢复成功。其他用户任务不受干预。

## 解释边界

K256下此前并未固定128对精化上限；本轮保留原阈值/互检后全部候选，平均精化数见报告。新增交互2048参数，没有FFN或多层堆叠。名义仿射监督继承原始配准残差；单种子、4epoch只用于本次机制筛查。最终精度比较必须是“训练后的原精化器”对“训练后的交互精化器”，不能用未微调T05冒充对照。
