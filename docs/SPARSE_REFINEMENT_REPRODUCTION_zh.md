# 固定预算稀疏精化复现

结果：[报告](../reports/sparse_refinement_v1/report.md)、[调用链审计](../reports/sparse_refinement_v1/implementation_audit.md)。原T05检查点、原952固定验证和原24计时样本保留。模型k=1024固定，预算M只减少粗筛选后的局部精化数，不是粗查询K。

新增API位于`sglm/sparse_refinement.py`：`select_refinement_indices`、`gather_candidate_fields`和不增加参数的`SparseRefinementB0`。all/None/M≥N保持原顺序；zero仅在限制预算模式下返回空，all优先遵守用户规定的全量规则。所有逐候选字段显式声明，逐image pair分配M。

zzvr工程目录：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching`。数据根：`/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0`。以下命令按依赖顺序执行，已经完成的冻结无需重复生成；重做相同协议会覆盖本轮派生指标，历史T05不变。

```bash
# 仅首次准备；已有freeze时明确拒绝覆盖。
PYTHONPATH=. .venv/bin/python scripts/prepare_sparse_refinement.py

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv/bin/python \
  scripts/regress_sparse_refinement.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv/bin/python \
  scripts/evaluate_sparse_refinement.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
```

已有freeze包含BF16容差修订，初始freeze/config/失败均归档。`--amend-bf16-tolerance`是本轮一次性迁移入口，要求尚未完成regression且有逐算子诊断，不用于再次修改已完成协议；不要在现在的完成状态运行它。

先确认指定GPU无其他CUDA进程；脚本启动与每对样本前后再次检查，冲突则失败退出，不终止他人任务。物理GPU7 UUID如下，其他服务器/卡需显式替换：

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_sparse_refinement.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --gpu-uuid GPU-e0ef04e8-e8b6-4435-c3e7-c8827ae9e45b

# 与真正未修改原入口作配对校准；不能与上面同卡同时运行。
PYTHONPATH=. .venv/bin/python scripts/calibrate_sparse_refinement_reference.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --gpu-uuid GPU-e0ef04e8-e8b6-4435-c3e7-c8827ae9e45b

# 仅重做预热、交错的阶段诊断，保留初版诊断，不覆盖正式计时。
PYTHONPATH=. .venv/bin/python scripts/diagnose_sparse_refinement_stages.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --gpu-uuid GPU-e0ef04e8-e8b6-4435-c3e7-c8827ae9e45b

PYTHONPATH=. .venv/bin/python scripts/render_sparse_refinement_cases.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0

PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q \
  --junitxml=reports/sparse_refinement_v1/tests.xml \
  > reports/sparse_refinement_v1/tests.txt

PYTHONPATH=. .venv/bin/python scripts/verify_sparse_refinement.py
PYTHONPATH=. .venv/bin/python scripts/report_sparse_refinement.py
```

本轮6配置是all、prefix256/128/64、global_conf128、coverage_conf128。三轮正式计时每边界每配置720次，GPU输入常驻，包含选择；complete还包含坐标GPU→CPU和RANSAC，排除I/O/CPU→GPU。阶段诊断带同步仪器开销，不能相加替代无hook完整延迟。显存用单module六组正式计时；原版配对校准同时常驻两个相同权重module，仅用于延迟比较。

`reports/sparse_refinement_v1/`版本化聚合表和逐样本数值（无像素）；`runs/sparse_refinement_v1/`保存私有逐配置JSON、5712份逐匹配NPZ、案例图。模型权重仍在原T05目录，不复制或公开。文件验收使用freeze/acceptance内SHA；原特征、窗口和几何代码保持原字节。
