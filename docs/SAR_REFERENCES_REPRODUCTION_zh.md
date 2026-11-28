# FHReg、GDROS与既有参照的复现说明

实验协议在`configs/sar_references_v1.json`，全部验证/预处理/代码SHA在`reports/sar_references_v1/freeze.json`，结果与开发预算位于同一报告目录。所有新增方法仅运行公开权重，不训练；源代码快照逐文件SHA见`official_source_snapshot.json`。

## 原版来源与权重

- FHReg：[vision-heng/GUSO](https://github.com/vision-heng/GUSO)，固定`f3e5420d59902edd025f06159b9a2b2320ea51d4`。GoogleDrive公开`fhreg_guso.ckpt`，78,646,305字节，SHA256 `8eda4899bf45c730c67812c5d5efd2d6172c36b821e8ac054c90fd2952838077`。下载URL在配置中。没有下载整个GUSO训练集，仅克隆代码/原生样例并下载该权重。
- GDROS：[Zi-Xuan-Sun/GDROS](https://github.com/Zi-Xuan-Sun/GDROS)，固定`ee6b6216fe780bc03bfb4759ae094daa0cfddc60`。使用公开WHU文件`GDROS-whu-opt-sar512.pth`，51,534,966字节，SHA256 `b6ebe175e59429272ae6aecda5f8fe5c5da2d451a283ac95b4892b5732d22a16`。复用服务器`/mnt/sda/zhushihao/VS_Match/GDROS/official_models_reference/`的既有缓存，逐字节复制到本项目私有weights目录；未借用其他项目微调权重。官方百度链接在配置中，HTTP提取尝试返回105，本轮没有重新下载；作者未提供可独立核对的远端checksum。来源身份按官方WHU文件名/README说明，原始训练样本未独立审计。
- GDROS的OS权重未用：不能确认作者的训练划分与本952对不重叠。`LSRnet.pth`虽公开，当前`test.py`实际调用无权重的`LSR`函数，不加载它。
- RIFT2：[官方MATLAB代码](https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation)，固定`0e980ce4124d2abd62727dcf040198db0a18b269`，本轮仅运行条件核查，详见`RIFT2_runtime_audit.json`。

源码均在私有`runs/sar_references_v1/{GUSO,GDROS,RIFT2}`，权重在该目录的`weights/`；不随本仓库发布。GDROS上游跟踪的CPython3.9 bytecode未复制，Python3.10直接加载未修改的源代码。

## 已验证环境

服务器RTX5090，Torch2.11.0+cu128、torchvision0.26.0+cu128、CUDA编译器12.8、timm1.0.30、Kornia0.8.2、NumPy2.2.6、OpenCV5.0.0.93、PL2.6.6。独立venv共享只读基础Torch；新增依赖没有安装到SOMA的venv。

```bash
.venv/bin/python -m venv --system-site-packages runs/sar_references_v1/.venv
runs/sar_references_v1/.venv/bin/python -m pip install PyWavelets==1.8.0 transformers==4.51.3
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH MAX_JOBS=4 \
  runs/sar_references_v1/.venv/bin/python -m pip install --no-deps --no-build-isolation mamba-ssm==2.3.2.post1
```

本次Mamba构建无ninja，走本地编译回退，包含SM120；成功wheel为`mamba_ssm-2.3.2.post1-cp310-cp310-linux_x86_64.whl`，282,168,392字节，SHA256 `ba56ceaed9173039cbe8940991e390edc8d0ed9b1f7b85e4e8f30cbb4b848166`。该wheel未发布。只验证FHReg使用的selective_scan路径，没有安装/声称满足Mamba所有其他模型与新kernel依赖；实际运行路径以native smoke证据为准。构建及补依赖日志留在私有runs。

FHReg上游仍导入PL1的`pytorch_lightning.profiler`。适配器仅将该旧namespace指向PL2的`pytorch_lightning.profilers`，使用官方`PassThroughProfiler`；没有修改模型forward、核心算子、权重结构或匹配阈值。GDROS无需启用alt_cuda_corr，32次迭代直接使用原版PyTorch路径。不能按README旧Torch2.3/CUDA12.1默认宣称5090可运行，本次是实际运行验证的现代环境。

## 准入与完整评价

需要本项目已审计OS原始256数据、T05 epoch18私有检查点和冻结952对矩阵。`--root`指向OSDataset2.0根目录。

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. runs/sar_references_v1/.venv/bin/python \
  scripts/smoke_sar_references.py --root /path/to/OSDataset2.0 --method FHReg_GUSO
```

同一smoke入口另外运行`FHReg_GUSO_256`、`GDROS_WHU`、`T05_Full`、`T05_Uniform256`。该次GDROS native smoke使用服务器已缓存WHU训练输入，路径/图像SHA写入JSON；仅作原生512接口前向和LSR回归，不当作泛化成绩。FH native示例位于仓库`assert/guso_testpairs/`，并非demo的默认相对路径。

首次新实验通过全部smoke后运行`prepare_sar_references.py`建立freeze；已发布freeze会拒绝覆盖，可直接校验并复跑固定配置。完整评价：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. runs/sar_references_v1/.venv/bin/python \
  scripts/evaluate_sar_references.py --root /path/to/OSDataset2.0 --method FHReg_GUSO
```

其余四组同入口。模型只接收影像/允许的输入掩码，拟合只接收预测对应；GT矩阵和独立探针在模型及RANSAC运行之后使用。全部失败留在952分母。仿射RANSAC、阈值、seed、探针与T05不变。T05两预算必须逐对复现历史矩阵和评价，否则停止比较。

FH主配置为256→512 OpenCV线性resize后CPU/255；逐位核验与外层同resize+原生TF.to_tensor一致。256是事前单独登记的尺寸敏感性对照，不根据验证成绩替换主配置。FH原生全量对应之后使用公共RANSAC，未额外使用FSC或KDE100截断，不等同论文原始FSC指标。

GDROS输入为256→512的rawfloat0..255，内部原版模型自行归一化。完整262144个原始流点转换为预测对应，再执行公共有效掩码和RANSAC；不为统一K而截断内部预算。新过滤器按16384块调用remap，避免OpenCV目标维度32767限制，数值行为已与原公共过滤器比较。半像素还原点坐标、同时缩放位移；独立NumPy几何模块没有Torch/模型依赖。

GDROS原始稠密场和原生32+LSR另在同固定探针直接评价，不添加公共RANSAC。LSR固定200000点、flatten起点10000，不读公共mask、不加载LSRnet。原生稠密协议的终点和主表仿射终点不同，结果与时间分别列出。

## 既有精度复用与本轮重新测速

`verify_sar_reference_legacy.py`核验旧MINIMA-RoMa/XoFTR全部权重、代码与数据SHA，并在两个预定验证输入上核对source/target/score逐位相同。主表明确复用前轮完整952对精度，本轮重新测资源，不声称旧方法已重新跑完952对。

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. runs/sar_references_v1/.venv/bin/python \
  scripts/verify_sar_reference_legacy.py --root /path/to/OSDataset2.0 --method MINIMA_RoMa
```

空闲GPU7上的正式计时每模型/每轮使用新进程，固定24对、3轮、每对预热3次/测5次：

```bash
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/benchmark_sar_references.py \
  --root /path/to/OSDataset2.0 --method FHReg_GUSO --round 1
```

支持五个本轮配置、`XoFTR`、`MINIMA_RoMa`和单独的`GDROS_WHU_native32_LSR`。存在同卡其他CUDA进程时直接拒绝。无hook、无特征缓存；显存报告包括全部模型/激活/库工作区，RoMa隐藏在Python列表里的DINO也计入实际CUDA分配。轮次以预定循环顺序轮换，不剔除较慢轮次。

`coordinate_sar_reference_timing.py`是本服务器专用协调入口：只识别当前本项目SOMA监督器及其后代，记录稳定checkpoint后临时释放，串行完成所有正式进程，再用原`.venv/bin/python`恢复同一4GPU任务。不是通用进程管理工具；恢复必须核实真实更新数超过checkpoint，而不只检查Popen。日志/未存盘尾部重放记录保留，其他用户任务不动。

```bash
PYTHONPATH=. runs/sar_references_v1/.venv/bin/python scripts/summarize_sar_references.py
```

汇总产生完整对比CSV、三目标实测非支配关系、开发预算JSON/Markdown与图。预算来自一个实际精度—延迟—显存点，不拼接不同方法的极值；它是后续研究目标，不是当前模型已达成的承诺。
