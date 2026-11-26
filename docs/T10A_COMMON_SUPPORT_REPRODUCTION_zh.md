# T10-A共同支持与类别组成审计复现

这是一轮独立的冻结候选补充审计，历史T10-A报告保留。配置为`configs/t10a_common_support_v1.json`，完整结论和CSV字段解释见`reports/t10_common_support_v1/审计报告.md`。

只需原有私有候选目录`runs/t10a_candidates_v1`、T05检查点和已版本化的256对计划。不会载入模型对象、调用教师、训练或重新生成候选。检查点仅作SHA核验。

在zzvr工程目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python scripts/audit_t10a_composition.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python scripts/verify_t10a_composition.py
.venv/bin/python scripts/plot_t10a_composition.py
```

`verify`先独立复算共同支持、总体层级差和区间、固定类权重及分解恒等式，再重新执行一次分析，要求所有数值CSV和统计JSON字节完全相同。运行会重写本轮派生结果，不改原候选、仿射、历史结果或模型。

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q \
  --junitxml=reports/t10_common_support_v1/tests.xml \
  > reports/t10_common_support_v1/tests.txt
```

本轮分析仅用CPU；GPU可见性变量仅用于工程完整测试，不是进行效率测量。

`runs/t10a_common_support_v1/support.npz`保留原关系行号和`pair_index*1024+query_index`的有序唯一查询集合；索引均为原计划索引，没有重新抽样。原`relations.npz`按保存的三套`random_id`和有限分数共同判定支持。随机个体每条记录等分权重，避免可用5次与1次的记录权重不同。

49格权重由高分错误/主对照的混合组成只确定一次，详见配置及`fixed_class_weights.csv`。20000次父图配对重采样固定使用同一父图抽样矩阵。若正权重格缺失，固定标准化统计不可计算，禁止用零补均值或重新归一化类别权重。现有仅4473次可计算，其区间是条件区间，不是通常完整bootstrap置信区间。

原逐候选数组、概率图和模型不发布；私有输入缺失时脚本明确失败，不联网下载或替代数据。所有伪类别仅表示外部光学教师的判断，不能当作OS真实土地覆盖。
