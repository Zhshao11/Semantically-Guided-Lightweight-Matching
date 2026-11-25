# 阶段一：数据审计、工程初始化和几何闭环

当前阶段只建立可审计的数据与几何基础。方法结构沿用v0.1；此前训练记录为历史，不在本阶段延长训练或把几何闭环当作新精度结果。

## 工作位置与同步规则

- 本地项目：`/Users/shihao/Documents/ChatGPT/轻量化`。
- 本地方法仓库：该目录的 `implementation/`。
- 本地正式交付：`outputs/data_geometry_v1/`，包含完整报告、统一清单、控制点CSV、配置和原始测试证据。
- zzvr方法仓库：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching`。
- 原始数据：`/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0`，只读使用。
- GitHub：`https://github.com/Zhshao11/Semantically-Guided-Lightweight-Matching`，同步方法代码、配置、测试、审计报告和不含像素的清单。
- 原始图像、控制点CSV、提取的原始ReadMe和诊断样图保存在本地/服务器，不放进Git。`audit.json`包含控制点数量、边界和拟合统计。

## 统一配置

`configs/data_geometry_v1.json` 是本阶段数据与几何的实际配置源，包含256/512全部split预期数量、坐标约定、增强范围、mask策略、几何阈值和控制点格式。

`sglm/config.py` 加载并验证；机器上的数据根通过 `--root` 或 `SGLM_DATA_ROOT` 给出。`configs/osdataset_v01.json`保留模型结构参考，不再承担实际数据/几何配置职责；网络结构参数与数据几何协议的职责不同。

`OSPairs`、审计和几何验证默认读取本阶段配置；train/evaluate/train_budget增加`--config`和`--manifest`。train/evaluate支持`--resolution 256|512`，实际配置存入运行目录/结果。512的数据可加载不代表v0.1模型在512上已完成训练或性能验证。

## 清单schema

`dataset_manifest.jsonl`每行一对图像，schema=`sglm.pair_manifest.v1`：

- `pair_id`：原版`osdataset/<resolution>/<split>/<ID>`；新增区域`osdataset2/region/<region>/<ID>`，防止不同split重名。
- `dataset/resolution/split/region`：原始来源与归属。新增区域`split=unassigned`。
- `optical/sar`：相对路径、宽高、mode、channels、dtype、逐通道min/max/mean/std/zero_fraction、是否常量、文件大小、文件SHA256、解码像素SHA256、ZIPCRC核验。
- `parent_candidates`：只有双模态解码像素完全匹配512四象限才产生记录，保留父图pair_id、split与crop_xy。
- `geometry_gt`：已对齐图像对假设，没有独立patch变换文件；`semantic_gt=null`。
- `scene_id=null`及`scene_link_status`：包内缺少patch到最初scene的映射，不编造来源ID。
- `usable`：是否可以使用。加载器拒绝损坏/不可用记录、重复ID、选中split以外路径。

`file_inventory.jsonl`每行一个文件，记录相对路径、字节数、SHA256、CRC32、与ZIP目录声明是否相同。它不是训练样本清单。

## 复现命令

在仓库根目录运行，使用已有专用`.venv`：

```bash
export SGLM_DATA_ROOT='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0'
export PYTHONPATH=.

# 完整TIFF审计需要tifffile，安装在本项目虚拟环境。
.venv/bin/python -m pip install -r requirements-audit.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_osdataset.py \
  --archive /mnt/sda/zhushihao/data/vis_sar/archives/OSDataset2.0.zip \
  --config configs/data_geometry_v1.json --output reports/data_geometry_v1

.venv/bin/python -m pytest -q \
  --junitxml=reports/data_geometry_v1/geometry_tests.xml \
  > reports/data_geometry_v1/geometry_tests.txt

.venv/bin/python scripts/analyze_manifest.py \
  --manifest reports/data_geometry_v1/dataset_manifest.jsonl \
  --output reports/data_geometry_v1/pair_duplicates.json

.venv/bin/python scripts/verify_geometry.py \
  --manifest reports/data_geometry_v1/dataset_manifest.jsonl \
  --config configs/data_geometry_v1.json --output reports/data_geometry_v1

.venv/bin/python scripts/render_data_geometry_report.py \
  --directory reports/data_geometry_v1
```

审计逐文件读取、完整PNG解码、全部scene tile解码和哈希，耗时与存储速度有关。CRC比较的是本机ZIP中心目录，不等同于官方签名，也不等同于本次重新解压整个ZIP的压缩流验证。

几何验证在六个split各取首/中/末真实样本，每对1000点，并通过manifest核对全split数量；全部像素数值范围和损坏检查由完整审计负责。两者覆盖范围不同，不能把18对的几何抽查描述成17168对全部做过affine闭环。

## 几何接口

`sglm/geometry.py`提供2×3/齐次affine合法性、求逆、组合、完整仿射参数化、点变换、crop/resize框架变换、stride特征和归一化坐标转换、GeoTIFF中心到世界坐标、有效边界、图像warp及support mask、控制点严格解析、最小二乘拟合与完整affine RANSAC。

正常原版训练仍用旋转+等比尺度+平移扰动；“求解器支持完整6DoF”不代表扰动默认包含shear。基础测试额外使用非等比尺度和shear检查完整模型。

## 阶段验收

报告必须包含六个split对数和数值范围、配对/损坏结论、父图证据与scene来源未知项、72点控制格式及边界；统一清单与配置可被加载器实际读取；几何JUnit与真实样本闭环均通过；本地/zzvr/GitHub代码提交一致后交付。
