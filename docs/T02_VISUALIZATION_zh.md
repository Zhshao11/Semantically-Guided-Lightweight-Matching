# T02同一批样本的预测可视化补充

这批图覆盖原T02固定计划的全部32个train样本、16个val样本；初始化和1600步模型使用同一影像、同一扰动、同一mask、同一匹配阈值与RANSAC配置。没有重训或挑选其他checkpoint。

每个pair_id生成一张四面板图：上排为初始化与训练后预测粗对应，下排为初始化与训练后预测矩阵配准结果（光学/已配准SAR棋盘格）。同样保存最终配准SAR单图以及完整粗对应、精化点、预测矩阵与评价数据JSON。失败明确显示无预测变换，不回退到identity或GT。

显示连线最多64条，按源坐标顺序均匀抽取；JSON保留全部对应。颜色仅在预测结束后由已知A计算粗匹配误差：绿<5px、黄5–10px、红≥10px，不参与筛选或拟合。最终配准仅用预测optical→SAR_aug矩阵的逆矩阵；图中红边是无效采样支持。

每个可视化重新推理的精化对应和预测矩阵都逐项核对既有评价JSON，48例全部一致。样本不按结果选取；联系预览只显示固定计划前4条，完整图册保留所有成功与失败样本。

本地完整交付：`outputs/T02_B0_Full/六项交付核对/`。Git保留代码、统计与完整粗预测JSONL，图像产物保留在本地和zzvr。

复现命令（zzvr仓库根目录）：

```bash
export PYTHONPATH=.
export SGLM_DATA_ROOT='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0'

# 原T02已有初始化train、最终train/val；本次补齐初始化val，同阈值、不重训。
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/evaluate_t02.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t02_b0_full_v1/fixed_plan.json \
  --checkpoint runs/t02_b0_full_v1/training/initial.pt \
  --split val --output reports/t02_b0_full_v1/initial_val.json

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/visualize_t02_predictions.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t02_b0_full_v1/fixed_plan.json \
  --initial-checkpoint runs/t02_b0_full_v1/training/initial.pt \
  --trained-checkpoint runs/t02_b0_full_v1/training/last.pt \
  --evaluation-directory reports/t02_b0_full_v1 \
  --output runs/t02_b0_full_v1/prediction_visuals

.venv/bin/python scripts/report_t02_six_items.py \
  --visual-manifest runs/t02_b0_full_v1/prediction_visuals/visualization_manifest.json \
  --output runs/t02_b0_full_v1/six_item_delivery
```

用户要求的六项完整数值说明见 `reports/t02_six_item_delivery/六项交付核对报告.md`；本地版本包含全部图册链接和示例图。
