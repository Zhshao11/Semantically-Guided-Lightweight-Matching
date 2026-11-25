# Semantically Guided Lightweight Matching

面向光学–SAR 影像配准的轻量稀疏匹配网络，按当前 **v0.1 设计材料**建立的首版 PyTorch 实现。

**当前状态：核心网络、训练/评估入口与数据审计已实现；已完成真实 OSdataset 小规模运行验证。尚未完成七类语义教师训练、完整数据集训练、动态预算实验或论文级精度验证。** 目前的短训练 checkpoint 是关闭语义引导的 visual baseline，不是完整方法的最终模型。


## 当前优先阶段：数据与几何基础

本次阶段交付见 [数据与几何工程说明](docs/DATA_GEOMETRY_STAGE_zh.md)、[控制点格式](docs/CONTROL_POINTS_zh.md)及 `reports/data_geometry_v1/` 的完整审计和测试证据。配置实际读取 `configs/data_geometry_v1.json`；原有模型参考配置和短训练报告保留为历史。

统一清单覆盖原版256/512六个split和2.0新增区域，加载器支持按resolution/split筛选；父图关系通过双模态像素验证。该阶段不报告新的训练精度。

## 冻结去重训练清单

元数据收尾见 [冻结清单与状态说明](docs/METADATA_FINALIZE_V1_zh.md)。`reports/metadata_finalize_v1/train_256_dedup_v1.jsonl` 保留7912对，512版本保留1978对；原始清单和文件保留。使用双模态联合解码内容键、最小数值ID代表，全部keep/exclude映射可追溯。后续训练通过 `--manifest` 显式选择该视图，验证和测试仍使用原split。

父图内容映射独立标为verified/candidate/unresolved，卫星scene来源另设状态。控制点分组为一对大场景影像的72个点对应，本轮仅检查元数据，不做scene模型选择或阈值调整。

## T02：B0-Full最小训练闭环

已完成独立基础版本：全量1024×1024粗匹配、简单局部相关、完整仿射RANSAC；关闭语义、联合评分、预算与局部交叉注意力。固定32对train/16对val完成BF16训练1600步，并验证断点恢复和不接收GT的推理入口。

预测对应的独立探针评价：train的5px成功率29/32（90.625%），val为1/16（6.25%），当前泛化仍弱。详见 [T02验收报告](reports/t02_b0_full_v1/T02验收报告.md) 和 [工程复现](docs/T02_B0_FULL_zh.md)。T02不替代后续完整方法。

## 方法

1. 独立光学/SAR 输入适配器，共享卷积、模态独立 BN；三阶段 DSRes（16/32/64 通道，2/2/3 个残差块）和加法 FPN。
2. 光学轻量七类语义头，输出概率 `p`；置信 `c=1-H(p)/log(7)`。外观可匹配性 `m=sigmoid(a)` 与语义修正组成 `v=sigmoid(a+c*r)`；评分分支对 `p` 停止梯度。
3. 在任何跨图相关矩阵之前，用 178 维统计预测预算；4×4 网格按基础覆盖和 MeanTop8 价值分配配额，单格最多 32 点；局部抑制后补齐。
4. 真正 Gather 光学查询后计算 `K×1024` 描述子相关，SAR 不裁剪；行 Softmax 额外包含 no-match 类，筛选并去除目标冲突。
5. 光学 3×3 / SAR 7×7 局部窗口，中心锚定源聚合，单层双头完整预归一化交叉注意力，独立内容相关读出，soft-argmax 亚像素定位与 36 维可靠性头。
6. 网络外 `cv2.estimateAffine2D` 完整六自由度 RANSAC，返回 2×3 矩阵或显式失败。

网络约 74032 个参数（含语义头与预算头；以实际脚本统计为准）。固定预算默认 256；动态预算可选 128/256/384。输入 256×256 时粗层 stride=8，细层 stride=2；统一 `align_corners=False` 与零起点像素中心 `(x,y)`。

## 数据核对

使用 **OSDataset2.0 中内置的原版 OSdataset/256**：

| split | 图像对 |
| --- | ---: |
| train | 8044 |
| val | 952 |
| test | 1696 |

原版 optical 与 SAR 都是灰度图；optical 在加载时复制为三通道。patch 是已配准图像对，没有逐对矩阵/点 GT，也没有七类语义标签。使用已知 SAR 扰动生成合成几何 GT。独立 scene 子集有1对影像，附72个控制点对应。

TAR 论文与此原版数据来源一致，但论文使用 6297 训练 / 1000 测试，未给出验证集与完整样本列表；**本项目不声称与 TAR 的样本和变换协议完全相同**。详见 [完整核对记录](docs/DATASET_AUDIT_zh.md)。

## 安装与测试

```bash
python -m venv .venv
source .venv/bin/activate
# RTX 5090 请先安装支持 Blackwell 的 PyTorch/CUDA 版本。
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

在服务器上已建立专用 `.venv`，继承原有 `compared` 环境中的 Torch 2.11.0+cu128、OpenCV 等依赖，单独安装 pytest，没有修改共享环境。运行命令也可以使用 `PYTHONPATH=.`，不必 editable install。

数据根目录应是 `OSDataset2.0`，而不是内层 `256` 或旁边的空 `OSdataset` 文件夹。

```bash
export SGLM_DATA_ROOT='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0'
export PYTHONPATH=.

python scripts/audit_osdataset.py --root "$SGLM_DATA_ROOT" --output reports/local_audit

# 快速验证：关闭语义引导，64 对训练样本、100 步。
CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
  --root "$SGLM_DATA_ROOT" --output runs/smoke \
  --epochs 7 --batch-size 4 --limit 64 --max-steps 100

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate.py \
  --root "$SGLM_DATA_ROOT" --checkpoint runs/smoke/last.pt \
  --output runs/smoke/val32.json --split val --limit 32

# 完整固定预算基线。epoch、学习率是首版工程默认，需只在 val 上选择。
CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
  --root "$SGLM_DATA_ROOT" --output runs/visual_fixed256 \
  --epochs 10 --batch-size 4
```

`train.py` 仅加载 train，记录配置、每步 loss、梯度范数并保存 checkpoint。验证与测试显式运行；测试集不参与预算归一化、训练标签或阈值选择。`--resume` 支持加载模型/优化器并从后续 epoch 继续，**不保证中途断点的随机数与批次逐位复现**。

## 语义教师与蒸馏

需要额外的七类带标注遥感数据或训练好的七类 SegFormer-B2；OSdataset 自身无法提供这部分分割 GT。原版光学为灰度，必须验证教师在灰度域上的准确性。

```bash
python -m pip install 'transformers>=4.45'
```

外部语义数据的 train/val manifest 是 JSON 列表，例如：

```json
[{"image": "images/a.png", "label": "labels/a.png"}]
```

标签必须事先明确映射为 `0..6`，无效像素 `255`。`class_map.json` 将键 `"0".."6"` 映射为真实类别名称。不要直接混用不同数据集的类别编号；例如 LoveDA 的原始标注需要显式转换。初始化目录为本地 SegFormer-B2 Hugging Face checkpoint，脚本不会自动下载或虚构权重。

```bash
python scripts/train_teacher.py \
  --data-root /path/to/labeled_data \
  --train-manifest /path/to/train.json --val-manifest /path/to/val.json \
  --initial-model /path/to/local_segformer_b2 \
  --class-map /path/to/class_map.json --output checkpoints/teacher

python scripts/cache_teacher.py --root "$SGLM_DATA_ROOT" \
  --teacher checkpoints/teacher --output semantic_cache

python scripts/train.py --root "$SGLM_DATA_ROOT" \
  --semantic-cache semantic_cache --output runs/semantic_fixed256
```

教师用 masked CE 训练、验证 mIoU 选权重。缓存先算七类 softmax，再双线性缩放到 32×32 并归一化，掩码默认取教师最大概率 ≥0.6。学生以 masked `KL(t || p)` 蒸馏。教师不进入匹配推理网络。

当前加载器只对 SAR 做几何增强，optical 与其语义缓存不变；若后续增强 optical，必须同步变换软标签与 mask。教师训练/缓存脚本已做静态编译检查，但因没有外部标注和 checkpoint，本次未运行真实教师训练。

## 动态预算

先训练可靠的固定预算 matcher，再只在训练集上生成 oracle 并拟合 178 维标准化与预算分类头：

```bash
python scripts/train_budget.py --root "$SGLM_DATA_ROOT" \
  --checkpoint runs/semantic_fixed256/last.pt \
  --output checkpoints/dynamic.pt --limit 1000
```

首版 oracle 是工程补充：选最小预算，使 GT 正确粗对应数达到三种预算最佳值的 90%，至少 6 对且覆盖至少 4 个宏网格；全部失败的样本跳过，不伪造标签。归一化均值/标准差存入 checkpoint；未拟合统计时动态推理会报错。该 oracle、训练超参数与预算分类泛化尚未完成实验验证；未来应加入精度/耗时约束并在 val 上验证。

## 训练目标与首版边界

现有设计材料主要定义网络结构。为使其可训练，首版补充：粗匹配交叉熵（含 no-match）、外观/选点的几何可见性 BCE、GT 局部窗口 bootstrap 的 SmoothL1、可靠性 BCE 和可选语义 KL。

**几何可见性只是初始可匹配性的替代监督，尚未覆盖纹理辨识度和实际下游匹配收益。** 硬选点本身不反向传播，评分依赖直接监督。这些训练目标是工程初版，不是已有论文结论，也不是 TAR 的原始损失。后续需要实际匹配成功率/效用监督、分阶段训练和消融。

默认阈值、loss 权重、预算 oracle、基础学习率尚未调优。没有在测试集上调参。本次 100 步短训练在 32 对验证样本上尚未取得成功仿射配准，不能把“跑通”当作“精度达标”。

## 评估与复现材料

- `scripts/evaluate.py`：deterministic per-pair SAR 扰动；记录每对 GT 变换、匹配数、RANSAC 结果和失败。
- correspondence RMSE/CMR：预测对应相对合成 GT 的误差；CMR 是图像对比例。
- affine-grid RMSE/CMR：在共同可见的固定网格上评估估计仿射，额外衡量配准；与 TAR 表格指标分开。
- CMR 分母保留失败样本；RMSE 均值只对误差有定义的样本计算，并输出有效样本数量。
- `scripts/benchmark_forward.py`：前向时延，含选点与冲突处理，不含 I/O、CPU RANSAC；未训练模型的精化负载不能代表训练后部署。
- `reports/`：实际数据审计、原版 256 的样本 SHA 清单、短训练/验证/测试记录。
- `configs/osdataset_v01.json`：参考配置描述；实际运行参数以各次保存的 config.json 为准。

数据集影像、第三方 PDF、外部权重与训练 checkpoint 不随 Git 发布。
