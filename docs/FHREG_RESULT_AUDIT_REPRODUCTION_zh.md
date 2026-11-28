# FHReg结果线核查复现

本轮仅审计已冻结的FHReg-GUSO512和原始T05全量K1024。协议在`configs/fhreg_result_audit_v1.json`，报告、固定952台账、随机30清单与统一测速在`reports/fhreg_result_audit_v1/`。无训练、无微调、无全952网络重跑、无阈值或RANSAC修改，不使用测试集。

## 已有依赖

沿用`docs/SAR_REFERENCES_REPRODUCTION_zh.md`中已验证的私有源码/公开权重、Torch2.11+cu128、SM120 Mamba环境。FH checkpoint SHA256：`8eda4899bf45c730c67812c5d5efd2d6172c36b821e8ac054c90fd2952838077`。T05 epoch18 SHA256：`f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`。

既有全952结果保存在私有`runs/sar_references_v1/{FHReg_GUSO,T05_Full}_validation.json`，配套公开摘要/逐对CSV可独立对照。审计冻结它们的SHA而不覆盖历史结果。

## 1. 无网络坐标与固定抽样/分母

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
 .venv/bin/python -m pytest -q tests/test_fhreg_coordinate_audit.py
PYTHONPATH=. .venv/bin/python scripts/prepare_fhreg_result_audit.py
```

prepare会拒绝覆盖`audit_freeze.json`。它执行七种无网络坐标场景、写入每种最大/平均误差，检查完整仿射的公共估计器/独立探针回归。生产FH输入只resize256→512，实际crop/pad为零；不同裁剪原点和非对称padding只是合成扩展测试，不改变生产接口。

resize遵循零起点像素中心：`native=(original+.5)*scale-.5`，再扣resize后的crop原点、加左上padding。先构造不同的源/目标预处理矩阵Ps/Pt，将已知A映射为`Pt @ A @ inverse(Ps)`，再调用生产`restore_resized_coordinates`还原。对角点、亚像素点、非正方形尺度、xy互换负对照、仿射逆变换和公共拟合均检查，防止仅对称往返掩盖错误公式。

固定台账按fit_success和有限probe_error严格<3/5重新生成标记，分母始终952。21个无定义误差计为SR失败；RMSE统计范围另外声明为全部931个求解成功对，包含308个未达5px的估计结果。

分类依据原SR@5：FH成功/T05失败、T05成功/FH失败、两者成功。各池按pair_id排序，用seed20261128一次无放回抽10对，抽签顺序保存；不因坏图、弱纹理或无匹配更换样本。

## 2. 30对证据

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. runs/sar_references_v1/.venv/bin/python \
 scripts/capture_fhreg_evidence.py --root /path/to/OSDataset2.0
```

只重新运行选中30对网络，两次推理分别用于原生分解输出和adapter等价检查，匹配数组重复逐项相同。保存原生512输出、还原256输出、公共过滤mask、过滤后对应、原RANSAC独立内点mask与矩阵。逐对核对原raw/valid数量、矩阵、内点数、probe评价完全一致。

GT在模型预测、公共过滤与拟合后才用于独立probe、诊断对应误差和单独的名义GT叠加。原始未扰动SAR与实际SAR_aug分别保存；所有匹配图的目标都为实际SAR_aug。预测叠加使用inverse(光学→SAR_aug估计矩阵)把SAR_aug扭回光学；无变换时只保存失败占位，不使用GT补出预测图。

每对私有文件夹在`runs/fhreg_result_audit_v1/evidence/<category>/<order>_pair<ID>/`，包含：

- `optical_original.png`、`sar_original_unaugmented.png`、`sar_augmented_actual_input.png`；
- `FHReg_optical_native512.png`、`FHReg_SAR_aug_native512.png`；
- `matches_raw.png`、`matches_filtered.png`、`ransac_inliers.png`；
- `independent_probes.png`及全部探针CSV；
- `registered_overlay.png`、`registered_checkerboard.png`、有矩阵时的`SAR_registered_to_optical.png`；
- 单独标注的`nominal_GT_overlay_diagnostic_only.png`；
- `raw_matches_all.csv`、`filtered_matches_all.csv`、`full_correspondence_evidence.npz`；
- `metadata.json`、review卡与后续局部patch证据。

raw指原生网络内部粗/细阈值之后、公共过滤之前。图中全点散点+原顺序均匀80条线只用于显示，拟合和CSV保留全部点。

## 3. 局部观察与只读恒等核查

```bash
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/extend_fhreg_visual_audit.py
```

固定从每对RANSAC内点原顺序均匀取6个位置，显示光学/SAR_aug的41px局部patch，不按GT选择。无内点时保留空占位，不补出对应。原图和局部图的人工/助手观察只作定性证据，不能当独立物理对应GT或新训练标签。30对文字观察在`visual_review_notes.csv/json`，没有改变成功标记或随机样本。

另用上一轮已冻结的同952对独立probe缓存评估恒等矩阵，仅作排除固定identity解释的零假设检查，无网络、无拟合、无新增匹配基线调优；不改原FH表。probe缓存为`runs/window_oracle_v1/candidates`，与原固定输入/矩阵来自同一验证协议。

## 4. 统一资源测量

```bash
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/benchmark_fhreg_t05_audit.py \
 --root /path/to/OSDataset2.0 --method T05_Full --round 1
```

同入口另测FHReg_GUSO。每模型/每轮必须单独新进程，三轮顺序T05/FH、FH/T05、T05/FH。同一GPU7 UUID，脚本拒绝任何同卡其他CUDA进程。保持CPU256uint8输入+mask→原生预处理/H2D/推理/坐标还原D2H/公共过滤/完整仿射RANSAC，包含相同后处理边界。每对预热3次、测5次、三轮15次；无hook、无缓存，CUDA同步包围墙钟计时。每个进程记录24对CPU输入字节SHA，六个进程必须一致。

内存在每对预热后reset_peak_memory_stats，取完整PyTorch peak allocated最大值；reserved另列，不扣cuBLAS等工作区。CPU共享、torch线程4，不改变GPU时钟；硬件/其他库线程元数据保留。T05原BF16、FH原FP32，不强称内部计算精度相同。

旧10.138ms和9.787ms都包含RANSAC，差异不是漏计后处理：前者跨模型长期计时进程，后者每轮新进程、分块过滤实现；还混有时钟/共享CPU运行状态差异，不能仅凭历史记录分离各项原因。保持历史结果不覆盖，今后同表使用本轮共同协议的配对计时值。

## 5. 汇总与最终验收

```bash
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/report_fhreg_result_audit.py
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/verify_fhreg_result_audit.py
```

154项测试通过（新增9项坐标测试）。最终验收检查952台账、21失败、抽样各10、30对复现/全套文件哈希、坐标误差、六个独占测速进程同输入和边界。报告/清单/坐标测试/指标代码可同步GitHub，数据像素、权重和逐对应全量数组保持私有，仅落本地与服务器。
