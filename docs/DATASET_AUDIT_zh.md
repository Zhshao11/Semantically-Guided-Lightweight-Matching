# OSdataset 初步核对记录

更新：本文件保留首版核对历史；最新完整审计、重复项和父图证据请看 `reports/data_geometry_v1/数据审计报告.md` 与 `docs/DATA_GEOMETRY_STAGE_zh.md`。

核对日期：2026-11-25。依据服务器文件实际盘点、图像解码完整性检查、SHA-256 清单、数据集内置 ReadMe.docx、Scene-level Subset/code/Evaluation.py，以及用户提供的 TAR 论文第 6、7、11 页。未使用 SourceMux。

## 服务器数据与划分

实际数据根目录：`/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0`。旁边的 `OSdataset/` 是空目录，不能把它作为训练根目录。

| 子集 | 训练对数 | 验证对数 | 测试对数 | 总计 |
| --- | ---: | ---: | ---: | ---: |
| Patch-level Subset/OSdataset/256 | 8044 | 952 | 1696 | 10692 |
| Patch-level Subset/OSdataset/512 | 2011 | 238 | 424 | 2673 |

两种尺寸均约为 75.23% / 8.90% / 15.86%。目录已经划分好，本项目直接沿用，不重新随机混分。文件名在不同 split 内会重复，应使用 `split + 文件名` 作为标识，不能只按数字编号去重。

每个目录内，`opt<ID>.png` 与 `sar<ID>.png` 按相同 ID 配对。全量检查无缺失配对，PIL 图像 verify 全部通过；每种分辨率、每种模态内，三个 split 之间未发现 SHA-256 完全相同的文件。该检查只排除文件字节完全相同，**不证明没有相邻地理区域、近似重复、跨分辨率重叠或来自同一大图的相关性**。数据包没有原始场景 ID 清单，不能声称是地理场景隔离划分。

原版 OSdataset 的 optical 和 SAR 都是 **8 位灰度 PNG**。加载器把 optical 灰度复制为三通道；没有恢复原始彩色信息。256 与 512 不是两套可以随意混合的数据，必须防范同源裁块造成的泄漏。

OSDataset2.0 另外提供 13 个国家/区域文件夹，共 **3803 对 512×512** patch。这些 optical 为 RGB、SAR 为灰度；目录没有内置 train/val/test。此项目首版不自动把这些数据混入原版实验，以免改变协议。

| 新增区域 | 对数 |
| --- | ---: |
| American | 297 |
| Argentina | 436 |
| Australian | 341 |
| Germany | 409 |
| Japan | 313 |
| Katar | 399 |
| Malaysia | 263 |
| Poland | 337 |
| Russia | 257 |
| Russia2 | 139 |
| Russia3 | 96 |
| Singapore | 183 |
| Turkey | 333 |

整个 patch 目录共 17168 对文件，其中包含原版 256/512 的两种尺寸，**不能将 17168 视为互不重叠的独立场景样本数量**。

## GT 到底有没有

### 原版 patch

内置 ReadMe 明确称这些数据为 **co-registered optical and SAR patch-pairs**，来自 GF-3 Spotlight SAR 与 Google Earth，空间分辨率 1 m。256 有 10692 对、512 有 2673 对。

但 patch 目录中没有单独的对应点文件、逐对 affine/homography 矩阵、稠密 flow、语义标签或变换 JSON/NPZ。初始对应来自“影像对已对齐”的数据前提，工程上可以将原始坐标映射近似视为 identity，但这并不是提供了亚像素人工标注 GT，也不能假设完全不存在残余配准误差。

训练办法：对原始 SAR 施加已知变换 `A`，保留 optical 不变；则 optical 原始坐标 `x` 到扰动后 SAR 的监督目标为 `A x`。必须保存 `A` 的方向、坐标 convention 和有效区域 mask，不能把变换后图像仍按 identity 监督。

本仓库实现此协议：像素坐标 `(x,y)`、零起点像素中心；SAR 原图→扰动后 SAR；旋转 ±35°、尺度 0.7–1.3、平移各方向 ±图像尺寸 10%。当前采样为旋转+等比缩放+平移，是仿射变换的子类；几何求解器支持完整六自由度 affine。这里的范围对照 TAR，但随机种子和变换样本并非 TAR 作者提供。

### Scene 子集

`Scene-level Subset/GT.txt` 有 **72 行、每行 4 个数**。自带 Evaluation.py 确认格式为：

```text
x_optical y_optical x_sar y_sar
```

它是两个大幅 GeoTIFF 之间的参考控制点对应，不能直接套到任意 patch 编号。Evaluation.py 使用这些点作为评估参考，用估计匹配点建立 GCP 变换。无 scene train/val/test 划分；应作为独立场景评估，避免把评估控制点当训练标签。

### 语义监督

OSdataset 没有七类地物分割 GT。七类 SegFormer-B2 教师必须使用外部带标注遥感数据训练，或提供已训练、类定义明确的 checkpoint，然后缓存软标签并蒸馏。尤其原版 optical 为灰度，教师必须验证灰度域上的质量；将灰度复制三通道不代表 RGB 语义能力保持不变。

## 与 TAR 论文是否相同

用户文件：`2026_Cai_TAR_Text_Semantic_Assisted_Cross.pdf`。

- 第 6 页 IV-A：OSdataset 为 10692 对、256×256、1 m；SAR 是 GF-3 Spotlight，optical 来自 Google Earth；覆盖全球 20 个城乡区域。
- 第 11 页参考文献 [46]：Y. Xiang, R. Tao, F. Wang, H. You, B. Han, *Automatic Registration of Optical and SAR Images Via Improved Phase Congruency Model*, IEEE JSTARS, vol.13, pp.5847–5861, 2020。
- 数据包 ReadMe 的数据规模、传感器、来源、分辨率和引用论文与上述信息一致。

**结论：可以确认是同一来源的原版 OSdataset；服务器是包含原版的 2.0 扩展包。但无法确认与 TAR 使用的是逐文件完全相同的副本，更不能声称实验划分一致。**

| 项目 | TAR 论文 | 服务器内置 256 划分 |
| --- | --- | --- |
| 总体 | 10692 对 | 10692 对 |
| 训练 | 6297 对 | 8044 对 |
| 验证 | 文中未说明 | 952 对 |
| 测试 | 1000 对 | 1696 对 |
| 随机种子、样本 ID | 未提供 | 已生成本项目清单 |
| 每对测试变换 | 给出采样范围，未提供全部矩阵 | 本项目确定性采样并记录矩阵 |

不能把 10692−6297−1000=3395 自动叫作 TAR 验证集，论文没有这样写。本项目保持内置划分，并将协议命名为 `builtin-OSdataset-256-synthetic-affine-v1`。在拿到 TAR 作者的样本列表及评估脚本之前，不将结果称为“严格复现 TAR 协议”。

TAR 第 7 页式 (13)(14) 的 CMR@τ 是测试图像对中 **pair RMSE < τ 的比例**，不是单个匹配点正确率。仓库评估分别报告 correspondence CMR 与 affine-grid CMR；后者是额外的几何指标，不能直接冒充论文 CMR。空匹配与几何失败均记录，CMR 分母保留所有被评估图像对；均值 RMSE 另注明只在有定义的样本上计算。

## 可复核材料

- `reports/osdataset_audit.json`：全量目录计数、图片尺寸/通道、缺失配对、GT 候选和重复检查结果。
- `reports/train_manifest.json`、`val_manifest.json`、`test_manifest.json`：原版 256 样本相对路径与每张图的 SHA-256，不包含图像内容。
- `scripts/audit_osdataset.py`：可在新机器重新运行同样检查。
- 数据来源下载记录中的官方 DOI：<https://doi.org/10.57760/sciencedb.radars.00061>。本次没有另外在线核验 DOI 记录，判断依据是服务器包内说明和用户原论文。

本次未上传数据集图像、外部论文 PDF 或第三方权重；数据集内置说明要求仅用于科研，并保留原作者权利。
