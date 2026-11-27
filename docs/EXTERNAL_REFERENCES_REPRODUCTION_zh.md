# MINIMA-RoMa、XoFTR与T05统一任务评价复现

本轮只运行公开预训练权重，不微调、不重训、不调匹配阈值。完整来源/训练域/输入差异见`reports/external_references_v1/method_audit.md`，结果表与开发预算由实测生成。

## 私有文件与源码

工程：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching`。

- `runs/external_references_v1/MINIMA`：官方MINIMA快照`796e7721174f9f829b79b3702bf8c2ae9a3d447a`。
- `MINIMA/third_party/RoMa_minima`：父仓库固定子模块`0d3fd22dfaf5b7e463a93c3d7f0cbc26df953663`。
- `runs/external_references_v1/XoFTR`：官方`e0fbea431b30be9742effbf5577c90aa8eb938f9`。
- `runs/external_references_v1/weights/`：官方`minima_roma.pth`、`weights_xoftr_640.ckpt`、`dinov2_vitl14_pretrain.pth`。
- `runs/t05_frozen_v1/checkpoint.pt`：原epoch18检查点。

来源网址、所有权重SHA与大小保存在`freeze.json`；源码132文件SHA保存在`official_source_snapshot.json`。不把服务器上其他未核验fork或权重当作本轮来源。权重及源码原副本不随本工程发布；缺失时明确失败，不自动替换模型。

本轮实际使用`.venv/bin/python`（Torch2.11.0+cu128）与已有依赖路径`runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages`；版本记录在`environment.txt`。没有安装官方旧Torch2.0/CUDA11来冒充RTX5090兼容环境，没有修改上游源码。

## 运行顺序

已完成freeze会拒绝覆盖，现有工程不需重新准备。数据根保持`/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0`。

```bash
# 仅首次冻结
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/prepare_external_references.py

# 首先每个方法跑两个固定训练来源样本；METHOD依次为T05/XoFTR/MINIMA_RoMa
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/smoke_external_references.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --method METHOD

# 相同两训练样本与官方wrapper核对，不能依据验证成绩调整适配器
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/check_external_official_io.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0

# METHOD三个方法依次执行，全部952对固定验证
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/evaluate_external_references.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --method METHOD

# 须等全部评价结束且GPU7空闲；启动和每对前后检查无他人CUDA进程
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/benchmark_external_references.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0

# 权威显存：三个METHOD各自启动一个全新进程，不复用上一方法的库工作区。
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/measure_external_memory.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 --method METHOD

PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest -q \
  --junitxml=reports/external_references_v1/tests.xml \
  > reports/external_references_v1/tests.txt

PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/verify_external_references.py
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/summarize_external_references.py
```

`METHOD`是需要替换的三个明确方法名，不是一个真实模型。脚本会重写本轮派生结果；研究主模型权重与历史实验不变。正式计时不能与同卡评价/检查同时执行。GPU型号或UUID改变时应建立新版本协议，不能混入已有记录。

## 协议要点

同952对源256图、固定仿射与mask；预测后恢复256坐标，用相同完整仿射RANSAC和独立16×16探针评价。成功率分母保留952，空匹配、退化及求解失败均失败。教师、语义或GT不进入外部预测。

保留各方法原生尺寸/精度：T05为256/BF16；XoFTR为640/FP32；MINIMA外层640、内部560/864和原混合精度。因此表格属于相同任务与计时边界的原生参照，不能称同内部像素数、同精度或同训练数据的严格公平架构比较。

统一正式计时从CPU常驻的原256 uint8图与mask开始，含预处理、H2D、推理/原采样、坐标恢复、有效过滤、D2H、原RANSAC；不含I/O、合成仿射生成、初始化和GT评价。每次完整推理，无feature cache、无hook。原24对×3轮×5次每方法360次，3次逐对预热，按轮轮换方法顺序，GPU只常驻一个模型。由于同进程卸载模型后仍保留32–33MiB cuBLAS库工作区，权威显存另由每方法独立新进程测量：同24对、3次预热、逐对1次峰值。完整PyTorch峰值allocated含非注册DINO与本方法库工作区，不含驱动context；reserved另外保存。初始显存检查失败、原三轮延迟与跨方法峰值均保留。

RoMa采样是随机的，本轮每对独立SHA种子固定，准确率与计时使用相同对应；不改变官方10000样本数、0.05 certainty阈值或平衡策略。

## 输出与限制

`reports/external_references_v1/`：对比表、开发预算、逐图数值、计时原始数值、哈希/测试/适配核查。`runs/external_references_v1/predictions/`：私有逐对应NPZ及探针；各方法完整JSON也私有保留，能独立重新拟合和评价。

外部模型未声明OS训练，但完整大规模预训练逐图重叠不可核实；T05是OS训练域内模型，不能把这组结果称泛化SOTA结论。数据本身只有配准与合成几何监督，没有人工语义GT。开发预算是测量后的工程目标，不是已达成结果或统计证明的最优值。
