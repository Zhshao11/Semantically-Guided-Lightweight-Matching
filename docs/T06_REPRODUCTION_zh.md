# T06复现、冻结边界与产物定位

日期：2026-11-26。完整结果见[实验报告](../reports/t06_delivery_v1/T06实验报告.md)，所有实验参数见[固定协议](../configs/t06_protocol_v1.json)。本轮全部实验已完成；下列命令供复现，不要求重跑已有任务。完成的标签、评分头及教师运行拒绝覆盖，应先为新实验设置独立目录，并同步修改相关加载路径。

## 运行身份和数据边界

仓库：`https://github.com/Zhshao11/Semantically-Guided-Lightweight-Matching.git`；服务器工作目录：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching`。T06起点提交`3e3182041c93ad476ca7acece28b7d243f5620ec`；本轮运行源码SHA见`reports/t06_delivery_v1/acceptance.json`。

冻结匹配器为T03主运行epoch18/step35604，训练图参数37,219，文件`runs/t05_frozen_v1/checkpoint.pt`，SHA256：

```text
f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90
```

匹配骨干、粗描述子、温度/no-match、局部精化、匹配阈值0.01及完整仿射RANSAC（3px、最少6对应、5000次、0.999置信度）均保持T05。受保护源码和原检查点SHA均通过验收；模型eval、模态BN分支和缓冲区冻结保持不变。T05固定952对验证与24对计时清单不变。没有用OS测试或scene控制点选模/调参。

7912对去重OS训练图对应提前保存的一组仿射，旋转±10°、尺度0.9—1.1、每轴平移±16px、每轴剪切±5°，围绕中心按原T02顺序组合。矩阵方向仍为`SAR_aug=A_forward(SAR)`，图像使用逆向采样。`reports/t06_setup_v1/training_plan.json`保留矩阵和样本，`setup.json`保留计划SHA及128对训练探针索引。

## 可靠性标签与评分头

```bash
export PYTHONPATH=.
export SGLM_T06_ROOT=/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
.venv/bin/python scripts/prepare_t06.py
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/generate_t06_labels.py --root "$SGLM_T06_ROOT"
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t06_head.py --small-fit --output runs/t06_head_small_v1
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t06_head.py --width 16 --output runs/t06_head_v1
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/train_t06_head.py --width 29 --output runs/t06_head_capacity_v1
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/evaluate_t06_selection.py --root "$SGLM_T06_ROOT"
```

离线全量相关生成逐查询标签：≤5px正确、≥8px错误、5—8px忽略；可监督但预测no-match记负。忽略GT不在侵蚀2px有效区、没有有效正确候选等位置。正负忽略为1,611,266/4,235,185/2,255,437。容差带并非已测得的原始配准残差。正确候选最佳logit减错误候选最佳logit单独保存，只作诊断。字段及方向见`label_schema.json`。

`runs/t06_reliability_v1/`保留逐字段npy、pair_ids、summary和压缩包`labels_and_diagnostics.npz`。压缩包不包含2.1GB特征缓存；重训评分头需要服务器上的`optical_features.npy`与`sar_mean.npy`，或重新生成缓存。标签生成的中间`status.json`最后写到7900，完成依据为summary、7912个ID和全字段shape/计数验收，不把中间状态当作不完整标签。

评分输入为冻结光学粗特征64和SAR有效粗网格GAP64；FP32头、不读相关/语义、不实例化骨干。小样本128对300步；主头及容量头7912对、12epoch/2976更新、batch32、AdamW lr0.001/WD0.0001、clip5、正类平衡masked BCE，固定last，不用952选评分头。

U完全复用T05均匀规则；A全局top-K；AC与A用同一检查点，4×4格基础2/格、上限32/格、MeanTop8效用softmax及确定性最大余数分配、NMS0。K512时各格恰32。AC-cap仅换7831参数外观头，与当前7977参数JointScore参考接近（差1.83%），未来SC改变时需重新核对容量。

正常推理只有一次跨图相关：`encode → score → select → gather → K×1024 bmm → 原精化 → 原RANSAC`。A/AC的bmm形状及梯度边界有测试。事后GT诊断不进入选择、精化或估计器。

## LoveDA教师的受控比较

初始化：`wu-pr-gw/segformer-b2-finetuned-with-LoveDA`，revision`5c74556c08bebb5f45f50b6f78f61a62c5d220c7`，权重SHA256`f1117b6abbc21949a0cdffec4aa2806202c9c534fd7de3171129ab10cc182b3b`。固定512双线性处理器、1/255及ImageNet归一化，原始配置保留在`runs/teacher_preflight_v1/teacher_candidate/`。

LoveDA来源[官方Zenodo](https://zenodo.org/records/5706578)，DOI`10.5281/zenodo.5706578`，数据CC-BY-4.0。完整Train2522图及新Val64的5172个文件SHA通过；Train.zip MD5为`de2b196043ed9b4af1690b3f9a7d558f`。新Val64城乡各32，独立于旧8；不下载Test。样本、来源成员和SHA见`reports/t06_training_v1/teacher_data_manifest.json`。

```bash
# 在能访问Zenodo的主机运行，再把Train/Val64/manifest镜像到服务器runs/t06_teacher_data。
python3 scripts/download_loveda_t06.py --output /private/path/t06_teacher_data \
  --train-md5 de2b196043ed9b4af1690b3f9a7d558f
.venv/bin/python scripts/verify_t06_teacher_data.py

PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/train_t06_teacher.py --mode T0 --output runs/t06_teacher_T0
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/train_t06_teacher.py --mode TR --output runs/t06_teacher_TR
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/train_t06_teacher.py --mode TG --output runs/t06_teacher_TG
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/review_t06_teachers.py --root "$SGLM_T06_ROOT"
```

T0不更新；TR/TG各1000更新、有效batch4（micro2×累积2）、同样本/原生512裁剪、AdamW lr2e-5/WD0.01、clip1、无scheduler、BF16训练；仅TG把相同RGB裁剪转PIL灰度复制三通道。4000次呈现计划SHA`5dd4a78855b303e9961b0892407ab92f88b1135bdaf47ffe0bc9458129fea558`，标注流SHA`6c6692249bc4a5a20b8ea9eac8c13a022abc4734ccecb59da9121e7c78c04eb3`。两者一致。

0为Ignore，1—7依次Background/Building/Road/Water/Barren/Forest/Agricultural。训练GT0→255掩码，八通道输出保留，不将Ignore当地类；有效GT上预测0计错。step0及每200更新用全图native1024 GT累计混淆矩阵，不做逐图mIoU平均。统一按新64灰度七类mIoU选best、同分保留更早，两者均选400但完成1000。末步表单独保留。旧8纵向检查，OS32不选模；64不是官方完整Val。

32张OS训练图保留旧8+新24、逐图置换教师版本后预览；盲审表供人工填写，助手观察不代替人工GT。没有OS分割准确率。八转七接口显式保留`a=1-pIgnore`，`p7=p1:7/a`，`r=1-H(p7)/log7`，`w=mask*a*r`；mask要求完整有效覆盖、a≥0.8、max(p7)≥0.6。KL对教师/权重detach，除以格数以保留绝对弱监督质量。每类有效质量为mean(w*p7)，仅OS32预览统计，不是7912正式缓存。

2026-11-26许可复核仍没有模型卡、license标签或权重许可文件，见`teacher_license_audit.json`。数据许可不授权社区权重；不公开原始或派生权重、不宣称可自由部署。当前初始化不作为最终可发布教师，需要许可明确的初始化来源。SC等待教师质量及最终冻结，未用语义训练评分/骨干，也未启动混合第三教师、动态预算或复杂精化。

## 编码分析与BN折叠门禁

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/diagnose_t06_efficiency.py \
  --phase profile --root "$SGLM_T06_ROOT"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/diagnose_t06_efficiency.py \
  --phase foldcheck --root "$SGLM_T06_ROOT"
PYTHONPATH=. .venv/bin/python scripts/diagnose_t06_efficiency.py --phase timing \
  --root "$SGLM_T06_ROOT" --gpu-uuid GPU-c32175d3-7e65-83aa-a3f6-fb85fe6a6c17
```

profiler只分析固定4对，trace不作速度依据。正式无hook另用独占GPU6，24对×3轮，每配置每对每轮warmup5、10次交错重复；不将相关轮次当独立场景。包含编码到CPU RANSAC，排除I/O、warp、H2D；逐轮/逐图配对数据完整保留。

每个共享卷积按光学BN0、SAR BN1分别复制折叠，原训练检查点不变。FP32明确同时关闭cuDNN和matmul TF32，192/192门禁通过；初次允许cuDNN TF32失败的记录也保留。BF16为0/192通过，不正式计时其折叠副本。预先数值门限未放宽。

FP32折叠的约1.27ms配对节省只与同精度原模型比较；不与主表BF16精度拼接。优化一致用于U/A/AC/AC-cap及两个K。分模态部署复制使存储参数增加，见`inference_copy_parameters.csv`，不能仍将训练图共享参数数当作部署存储数。

## 验收和文件保留

```bash
# 主环境有pytest；独立教师环境提供Transformers4.44.2，不改任何共享环境。
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/summarize_t06.py
PYTHONPATH=. .venv/bin/python scripts/verify_t06.py
```

87项测试通过，一条第三方anyio pytest弃用警告。验收核对冻结SHA、train-only标签、U精确复现、教师共同数据流/掩码、折叠门禁和正式计时分离。`reports/t06_training_v1/`保留头/教师配置、损失、历史、呈现计划及数据清单等非像素证据。

Git仅公开源码、配置、非像素指标、计划和指标曲线；`reports/t06_selection_v1/*_K*.json`的原始预测记录另留本地/服务器。所有权重、标签数组、特征、原始/预测图及profiler trace均不进入Git。本地交付位于`/Users/shihao/Documents/ChatGPT/轻量化/outputs/T06_外观选择与教师适配/`，服务器私有产物在`runs/t06_*`；本地索引列出下载副本与仍只在服务器的文件。
