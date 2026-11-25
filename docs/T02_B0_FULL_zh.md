# T02工程说明与复现

本轮任务：256仿射监督加载器、独立评价器与B0-Full小样本训练。实际验收报告位于 `reports/t02_b0_full_v1/T02验收报告.md`。报告完整保留弱验证集结果，不以训练集拟合代替泛化。

## 实现

| 模块 | 文件 | 关键接口 |
| --- | --- | --- |
| 固定/在线256加载器 | `sglm/t02/data.py` | `AffinePairs256(root, plan, split, mode)` |
| B0-Full | `sglm/t02/model.py` | `B0Full.forward` 和不接收GT的 `predict` |
| 正向仿射监督 | `sglm/t02/loss.py` | `correspondence_labels`、`b0_loss` |
| 独立评价器 | `sglm/evaluation.py` | 预测匹配求矩阵、矩阵与独立评价点求误差 |
| 实际配置 | `configs/t02_b0_full_v1.json` | 本轮固定输入、矩阵顺序、训练和评价参数 |

模型复用已经验收的 `Backbone`，不修改它的规格；采样、正逆矩阵和RANSAC复用既有几何实现。独立评价器不导入模型或Torch；`sglm.__init__` 改为延迟导出Matcher，原公开用法保持兼容。

## 输入与坐标

加载器返回 `pair_id, optical, sar_aug, A_forward, optical_valid_mask, sar_aug_valid_mask`。光学灰度仅复制三通道，归一化为uint8/255；SAR同样只除255，没有对数处理。光学mask指坐标域有效；SAR mask指逆采样完整支持，并不把所有0值当NoData。

正向对应为 `x_sar_aug=A_forward*x_sar`，影像生成是 `I_aug(y)=I_original(inverse(A_forward)*y)`；warpImage封装接受正向矩阵并由OpenCV逆向取样。监督不使用错误的逆矩阵。原图光学坐标与SAR原始坐标近似对齐是假设，原配准残差仍可能存在。

源和目标cell越界、mask不完整时不进入正样本；目标无有效对应监督no-match，源无效cell忽略。标签、矩阵和几何运算在FP32计算，避免BF16坐标量化改变匹配标签。模型粗前向及其反向链使用BF16混合精度，FP32参数与优化器状态保留。

`fixed_plan.json`冻结32对去重train与16对原val、完整正向矩阵和源图SHA。固定模式改变epoch也不改变输入；在线模式以seed、pair_id和epoch采样，供后续全量版本使用，本轮训练入口不切换在线模式。

## 推理与评价隔离

模型predict签名只有影像和mask，不接收A、控制点或评价探针。模型先从全量粗匹配预测局部中心，简单相关精化后交给独立RANSAC；没有oracle回退。

训练前400步可用真值附近窗口学习局部相关；401步后局部训练窗口也来自预测粗对应，并用GT判断窗口是否可监督。验收的任何图像对都不使用GT局部窗口。训练局部监督最多256个窗口/对，仅控制监督计算量；全量粗匹配未裁剪，推理精化全部通过匹配筛选的对应。

合成模式：`fixed_synthetic_probes(A, masks)`生成固定格点参考，再调用`evaluate_transform(A_hat, probe_source, probe_target)`。评价格点独立于拟合匹配和inlier，也不按预测误差筛选。

scene接口：同一个`evaluate_transform(T_hat, external_control_source, external_control_target)`；控制点只作为评价参考，不能传回预测拟合路径。本轮只验收该通用接口，没有运行真实scene评价或据其选模型调阈值。

`failure_rate`描述无法得到可评价变换的比例；`registration_failure_rate@t`为包含全部图像对的误差未达阈值比例。成功样本RMSE条件均值、所有可评价矩阵的RMSE均值及失败数分别保存，避免把RANSAC返回矩阵误当作配准成功。

## 复现命令

从zzvr仓库根目录运行。已有`.venv`和支持BF16的RTX5090；本次只使用空闲GPU4。

```bash
export PYTHONPATH=.
export SGLM_DATA_ROOT='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0'

.venv/bin/python scripts/prepare_t02.py \
  --root "$SGLM_DATA_ROOT" --output reports/t02_b0_full_v1

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest -q tests/test_t02.py

# 初次运行使用新的输出目录；不要覆盖已有训练。
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t02.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t02_b0_full_v1/fixed_plan.json \
  --output runs/t02_reproduction/training --stop-step 400

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t02.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t02_b0_full_v1/fixed_plan.json \
  --output runs/t02_reproduction/training \
  --resume runs/t02_reproduction/training/last.pt --stop-step 1600

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/evaluate_t02.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t02_b0_full_v1/fixed_plan.json \
  --checkpoint runs/t02_reproduction/training/last.pt \
  --split val --output runs/t02_reproduction/val.json

# 完全不给GT、计划文件或控制点的预测入口；sar_aug应是已准备好的输入影像。
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/predict_b0.py \
  --optical /path/to/optical.png --sar-aug /path/to/sar_aug.png \
  --sar-mask /path/to/sar_aug_valid_mask.png \
  --checkpoint runs/t02_b0_full_v1/training/last.pt \
  --output runs/t02_prediction.json
```

checkpoint保存模型、Adam状态、步数、固定计划SHA、配置及所有随机状态。batch序列由step和epoch决定，恢复可以落在epoch中间。本次另从400步快照重放401步，与主训练的401步batch、loss和梯度范数完全一致；没有宣称跨硬件所有步逐位确定。

## 本地与服务器产物

- 本地方法仓库：`implementation/`；正式交付：`outputs/T02_B0_Full/`。
- 服务器模型：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching/runs/t02_b0_full_v1/training/last.pt`。
- 最终权重同时复制到本地正式交付的`checkpoints/`；checkpoint、源图和20组GT图不提交Git。
- 配置、固定计划、代码、测试、逐步日志、逐对评价结果、训练曲线和说明同步GitHub。

固定样本复用200个epoch，train达到较高成功率而val仍弱。下一步应优先检查更丰富的在线扰动、扩大训练样本和稳定粗匹配学习，保留当前B0及固定协议作为排错参照；语义、评分、预算和交叉注意力仍属于后续完整方法阶段。
