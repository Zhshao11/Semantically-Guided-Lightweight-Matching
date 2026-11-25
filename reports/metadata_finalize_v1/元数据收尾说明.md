# 数据元数据收尾与冻结训练清单 v1

日期：2026-11-25。本轮只派生清单、核查父图内容关系和控制点元数据；没有执行scene配准、选择模型或调整匹配阈值。原始文件及 `reports/data_geometry_v1/dataset_manifest.jsonl` 均保留。

## 去重口径和冻结结果

“冗余记录”指每个唯一内容组确定性保留一条之后，组内多出的记录数。重复键是有顺序的 `(光学 decoded_sha256, SAR decoded_sha256)`，不是一侧相同就删整对；decoded哈希包含dtype、shape和像素字节。每个分辨率的train内部独立分组，保留数值样本ID最小的一条，不按字符串字典序保留，不跨split去重。

| 清单 | 原始记录 | 排除的冗余记录 | 冻结保留 |
| --- | ---: | ---: | ---: |
| 256 train | 8044 | 132 | **7912** |
| 512 train | 2011 | 33 | **1978** |

派生文件独立命名，位于 `reports/metadata_finalize_v1/`：

- `train_256_dedup_v1.jsonl` 与 `train_512_dedup_v1.jsonl`：可直接由OSPairs和train.py的 `--manifest` 读取，沿用原schema并附加状态。
- `train_keep_exclude_mapping_v1.csv`：原始10055条train记录逐条给出action（keep/exclude）、保留代表ID、joint_content_id、光学和SAR内容哈希。排除记录只从派生训练视图排除，原始文件不删除。
- `duplicate_groups_v1.jsonl`：165个重复组的保留ID与全部排除ID。
- `dataset_manifest_parent_status_v1.jsonl`：原全体17168条记录加状态的独立派生副本，保留原始split和候选；不是覆盖旧清单。
- `freeze_metadata_v1.json`：口径、预期/实际计数、源清单与源控制点SHA，以及派生文件SHA。

生成器拒绝覆盖字节不同的已冻结文件；允许按同一输入进行字节一致的重建。更换规则应另立v2目录和协议。

## 父图状态与去重后关系

`parent_mapping.status`的含义：

| 状态 | 判定 |
| --- | --- |
| verified | 父图ID和裁剪xy唯一，并有双模态精确裁块内容证据 |
| candidate | 找到候选，但身份/裁剪位置不唯一，或证据尚不足 |
| unresolved | 包内没有可确认的父图映射 |

原始ID空间内：10428条verified、264条candidate、6476条unresolved。后者包括原版512及新增区域512，缺少再上一级父图，不能误记为已验证的卫星scene。

去重训练清单把原重复父图ID规范化到确定性保留的父图代表；该代表按双模态完整内容验证，不是任意挑候选。去重后 **7912个256子图全部verified，每个1978个512父图恰好对应4个不同象限，覆盖(0,0)、(256,0)、(0,256)、(256,256)**。原始264条candidate在带状态的原全量派生清单中保持candidate；不会倒改原始候选身份。

`lineage_content_verification_v1.json`记录再次读取所有保留父图、检查其完整内容哈希及全部象限与冻结子图哈希一致的结果。这项只验证内容关系。256子图哈希来自上一轮全量解码审计并受原清单SHA保护。

`scene_source_mapping`独立保存卫星原始场景来源结论，当前均为unresolved。`parent_mapping.scope=content_mapping_to_distributed_512_patch` 与 `scene_source_mapping.scope=original_satellite_scene_provenance` 是不同结论。内容一致不证明原始场景来源或地理划分独立。

## 控制点按影像对分组

目前是 **1对影像、72个控制点对应**，不是72对影像。分组文件 `control_point_groups_v1.json`：

- pair_id：`osdataset2/scene/optical-sar`。
- optical：`Scene-level Subset/Optical.tif`，宽35783×高35873。
- SAR：`Scene-level Subset/sar.tif`，宽35507×高27298。
- 点数72，列顺序 `x_optical,y_optical,x_sar,y_sar`，x列、y行。
- 项目内部假设零起点像素中心；原始标注的起点与中心/角点约定仍标unverified，保留原坐标不偏移。
- 空间统计按像素中心范围`[0,W-1]×[0,H-1]`归一化；双侧分别输出bbox、xy跨度比例、凸包面积占影像面积比例、4×4网格占用和归一化坐标PCA次/主奇异值比。
- 近共线元数据规则预先声明为上述奇异值比<0.01。此规则只描述控制点分布，不是匹配阈值；没有使用scene性能选择它。

实测凸包覆盖光学约51.03%、SAR约67.57%；4×4网格分别占11/16和13/16；PCA比约0.702和0.881，两侧都不近共线。该轮不计算新的控制点配准误差、不拟合scene变换。

## 复现与使用

在zzvr仓库根目录执行：

```bash
export PYTHONPATH=.

.venv/bin/python scripts/freeze_training_metadata.py \
  --manifest reports/data_geometry_v1/dataset_manifest.jsonl \
  --audit reports/data_geometry_v1/audit.json \
  --control-points '/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0/Scene-level Subset/GT.txt' \
  --output reports/metadata_finalize_v1

.venv/bin/python scripts/verify_frozen_lineage.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --directory reports/metadata_finalize_v1

# 后续训练选择冻结视图；本轮没有启动训练。
.venv/bin/python scripts/train.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --manifest reports/metadata_finalize_v1/train_256_dedup_v1.jsonl \
  --resolution 256 --output runs/visual_dedup256_v1
```

原data_geometry_v1配置的expected_pairs继续描述原包分发数量，不能改写成去重后数量；新冻结视图及其SHA由freeze_metadata_v1.json定义。验证/测试仍使用原split，不从训练派生清单读取验证集。
