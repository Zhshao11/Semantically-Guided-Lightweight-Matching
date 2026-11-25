# T05复现与冻结说明

本轮是对T03主运行epoch18检查点的固定评价与无需重训的均匀稀疏初筛。完整结果见 [T05实验报告](../reports/t05_delivery_v1/T05实验报告.md)。

## 冻结身份

- 匹配检查点SHA256：`f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90`，epoch18、step35604。
- 服务器原文件：`runs/t03_main_lr1e3_v1/best.pt`；冻结副本：`runs/t05_frozen_v1/checkpoint.pt`，原文件未改。
- 实际训练代码：`85c7dea0d5be270647d991b285053ff0a4871120`；本轮源码SHA见accuracy_summary.json/timing.json，冻结依据见freeze.json。
- `reports/t05_frozen_v1/`：完整配置、952个固定矩阵、96对诊断/24对计时索引及输入统计、逐模块参数、逐叶层和实际张量形状。
- `configs/t05_protocol_v1.json`：固定K/网格规则、诊断阈值、warmup/重复数及源域教师标签规则。

`prepare_t05.py` 已执行一次；冻结文件存在时拒绝覆盖。若另建新的冻结实验，应指定新的output及weights-output目录，不覆盖本次记录。

## 原B0与均匀查询入口

`sglm/t05.py` 继承B0，严格加载同一state_dict，无新增参数。K1024使用原始描述子视图；K<1024在唯一跨图bmm之前gather成BxKx32，与全部SAR的Bx32x1024相乘。反向相关复用转置；无暗中保留的全量反向匹配。

均匀网格rows×cols分别为32×32、16×32、16×16、8×16；每轴坐标为floor((i+0.5)×32/尺寸)。顺序row-major，不读取语义、GT、纹理评分或置信度。不同K的网格不保证嵌套。互选规则相同，但SAR反向候选是所选K个查询；不声称与全量互选竞争集合相同。

所有预算保持原B0参数、BF16、归一化/掩码、no-match类别、0.01粗阈值、局部半径3（stride2）、RANSAC3px/min6/5000次/0.999置信度和同一随机seed。37,219参数及完整架构核对见报告。没有语义、评分、动态预算或交叉注意力。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/evaluate_t05.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
```

运行952×4固定验证，并为预选96对输出全部K的事后分层诊断。K1024与原B0预测逐项比较，再和历史epoch18逐pair比较；任何差异先停，不继续稀疏比较。952全量SR一直保留求解失败，RMSE分位数另注明可求解数量。诊断不进入推理或RANSAC。

## 正式计时

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_t05.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --gpu-uuid GPU-c32175d3-7e65-83aa-a3f6-fb85fe6a6c17
```

必须先确认此物理GPU当前空闲。脚本开始、每个影像对前后及结束检查同卡无其他CUDA进程，无法查询或占用时拒绝。不得与训练共卡。24对在推理前已冻结并人工查看输入覆盖范围，没有根据最终成败挑计时样本；24个512内容父图不同，原始卫星scene身份仍未知。

每对每K warmup5次、测量20次，K顺序种子交错。主时间为无hook预测+坐标CPU拷贝+完整仿射RANSAC，开始/结束CUDA同步；包含全部在线匹配计算，排除预读/warp/传入GPU和GT评价。分阶段计时是单独的同步调用，不能累加阶段中位数作为端到端结果。实际对应数、内点/求解状态在跨文件验收中和准确性运行逐pair一致。

峰值为PyTorch `max_memory_allocated`，包含模型/当前单对输入/临时张量，单位MiB。cached reserved另存，不能当作实际张量峰值。结果受本次5090、当前PyTorch、256输入和算子实现约束，不用MACs比例外推时间。24对各20次是相关重复，按原pair另导出中位数及相对全量的差值。

## 教师源域对照

```bash
# 在可联网主机，只取官方Val的固定8对成员；拒绝忽略Range的整包下载。
python3 scripts/fetch_loveda_val_subset.py --output /path/to/loveda_val_t05_v1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/diagnose_teacher_source.py \
  --source-root runs/loveda_val_t05_v1 \
  --os-root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --output runs/t05_teacher_source_v1
```

LoveDA官方Zenodo Val.zip，固定seed20261126，Rural/Urban各4张。成员CRC及SHA可查，不声称整包MD5已核验。源码/权重/512处理器与T04固定候选一致。RGB与PIL L→RGB仅改变颜色信息，源GT保持native1024，logits恢复native后算指标。OS8张train图无语义GT，只检查分布/可视图。

GT原生0忽略、1—7地类；预测0在有效GT上计错。mIoU为本批像素聚合的七类IoU均值，非官方全验证成绩。本批真实出现1—7；0号边界由单元测试验证。许可复核仍未发现社区权重许可，不将其描述为可自由分发/部署。下一步可考虑LoveDA训练图灰度增强适配；本轮未做训练。

## 汇总与验收

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/report_t05.py \
  --teacher reports/t05_delivery_v1/teacher_source_control.json
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/visualize_t05.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
.venv/bin/python scripts/verify_t05_artifacts.py
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q
```

新增8项接口测试、总77项通过。跨文件验收核对检查点/配置/清单/运行源码SHA、逐pair复现、分母/失败保留、1920次无hook与1920次阶段计时及对应负载、教师输入和类别边界。例图在完整诊断子集结果上按成功/失败各自RMSE分位选4对，所有K展示同8对；只是事后说明，不改变任何指标/阈值。配准棋盘格使用预测仿射的逆映射，不使用GT拟合。

原始/预测影像及权重留在本地outputs与服务器runs。Git仅含代码、配置和非影像像素的指标/架构证据。
