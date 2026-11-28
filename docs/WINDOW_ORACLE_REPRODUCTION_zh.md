# 原始T05 K256窗口内理想定位诊断复现

本轮仅做冻结候选的名义几何诊断，无训练、无新精化器、无测速/显存排名、无测试集。协议：`configs/window_oracle_v1.json`；完整报告与两张主表：`reports/window_oracle_v1/实验报告.md`、`overall_display.csv`、`paired_changes.csv`。

## 输入与模型

原始T05 epoch18私有检查点：`runs/t05_frozen_v1/checkpoint.pt`，SHA256 `f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`。模型为`UniformB0(k=256)`，原`LocalCorrelation`，不是同预算微调模型或交互精化器。固定952对及原仿射矩阵在`reports/t05_frozen_v1/validation_plan.json`。输入256×256、uint8/255，原模态BN、阈值/互检、窗口与原完整仿射RANSAC均不变。

## 缓存只生成一次

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python -m pytest -q tests/test_window_oracle.py
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. .venv/bin/python scripts/cache_window_oracle.py \
  --root /path/to/OSDataset2.0
```

缓存入口拒绝覆盖已有`cache_freeze.json`。每对先运行网络、原local过滤、原finite/score过滤和原始几何拟合，逐对核对与原T05历史矩阵/评价完全一致；之后才访问GT做支持诊断。`sample_actual_support`调用原`sample_windows`，从真实fine张量与mask得到原始有效token；采样点按实际像素→feature→像素链还原，再与原target读出位置逐项核对。

`runs/window_oracle_v1/candidates/0000.npz`等私有文件保存：

- 原始候选ID（光学粗格ID×1024＋SAR粗格ID）、uniform query IDs、原顺序；
- 实际source、原target、原confidence、预测窗口中心；
- 每窗口49个实际采样位置与逐token有效mask、source mask、原local过滤与原估计器过滤mask；
- 原热图、名义GT、实际支持命中、旧box对照命中；
- 原A、固定独立probe/source与probe/truth。

支持判断使用有效token坐标的连续闭凸包，不用±6px矩形替代。另要求原source有效、名义GT边界与原SAR_aug mask>.999；全无效窗口的fallback不算有效点。孔洞内部即使在凸包中，也需目标mask检查。数值容差1e-6px，不按面积放宽。

原始T05源点不被精化器移动，但GT仍用实际进入拟合的source计算，不能重造粗中心。附加矩阵A是原SAR→SAR_aug正向映射，借助原始注册关系名义上也是光学→SAR_aug；不是图像逆采样矩阵。原始注册残差仍存在，不宣称物理精确GT。

BF16原读出与高精度期望的数值差单独记录；不因此扩大支持。Oracle使用连续热图闭凸包理想化，不枚举BF16离散可达值，也不假定现有网络参数能生成该理想热图。

## 从缓存做两分支

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_window_oracle.py
```

运行前核验原始输入/模型/几何/支持代码与全部缓存SHA。仅对最终拟合mask内的命中target替换名义GT，其他字段不变；未命中target逐项不变。每对A/B分别将OpenCV设到原seed20261125，调用原`estimate_from_correspondences`。不复用内点，阈值、迭代数、refineIters、置信度、最少内点和失败处理完全原版。原候选finite/score过滤冻结，替换不增加/恢复候选；估计器内部原检查仍保留，在本数据上各字段均有效、不会改变集合。

每对额外执行强制全不命中的验收，RANSAC矩阵/状态/内点必须与A完全相同。该零干预仅是验收，不列为第三主实验。GT目标不追加到拟合中，probe点集始终独立，source格点与probe坐标逐对不重合。

`results.json`含每对两分支矩阵/评价、内点数与源/目标覆盖；逐行表在`per_pair.csv`。候选命中率同时提供池化与逐对分位数，不将对应条数视为独立场景。沿用既有4×4覆盖与凸包等指标，额外记录协方差特征值比lambda_min/lambda_max；少于3个命中点标记，比例记null，不用覆盖诊断再筛选。

## 异常与有限RANSAC复核

```bash
PYTHONPATH=. .venv/bin/python scripts/check_window_oracle_ransac.py
```

主实验完成后，因1个原成功对在37条理想替换后退化，检查所有这类5px退化对（本轮只有一对）。使用原种子与随后两个固定种子，均从原缓存配对重估。若该局部probe结果确实出现种子敏感，脚本才运行这三个种子的全部952对；不得挑最好种子替换主结果。本轮probe结果一致，未执行额外全952多种子。所有记录保留`ransac_sensitivity.json`；本轮没有任何训练种子。

## 可视化、汇总与验证

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
 .venv/bin/python scripts/render_window_oracle.py --root /path/to/OSDataset2.0
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
 .venv/bin/python scripts/report_window_oracle.py
PYTHONPATH=. .venv/bin/python scripts/verify_window_oracle.py
```

图像固定取原T05诊断清单前6对，显示候选按原顺序均匀取10条，放大第一条，不按成败或GT挑例。私有PNG在`runs/window_oracle_v1/visualizations/`，会同步本地交付目录，不发布数据像素到Git。合成支持几何示意和命中率分布图可公开。

145项回归测试通过（新增7项）。4条依赖/NumPy二维cross弃用警告留存，无失败。该诊断不选择新精化器、不推断可部署速度、不允许窗口扩张，也不将理论潜力解释为现实模型可实现增益。
