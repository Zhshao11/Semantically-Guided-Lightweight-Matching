# T07复现协议：教师冻结与全局语义选择

日期：2026-11-26。起点为T06提交`e208fc66d33272e02fae1d65feb952c0b265f3f2`。实验参数先固定于`configs/t07_protocol_v1.json`，完整实际结果见`reports/t07_delivery_v1/T07实验报告.md`，原始文件路径以冻结配置为准。已有运行不能覆盖；复现实验应建立独立目录并相应调整加载路径。

## 教师来源决策

旧TG（T06 step400）的检查点、SHA、训练源码、64张列表和逐类分布完整保留在`reports/t07_teacher_v1/TG_community_freeze.json`。其技术效果改善，但LoveDA社区权重未提供明确研究条款；不继续将该权重用于T07学生标签，不发布原始/派生权重。

替代初始化为官方`nvidia/mit-b2`，固定revision`3bb39e8739149c3777d0325349b2a6c32c6413db`。其模型卡明确把模型许可链接到NVlabs/SegFormer；许可第3.3条仅允许非商业研究或评价。固定官方卡、许可文本、仓库commit和下载SHA见`initialization_license.json`。这里确认的是研究用途范围，不是自由部署或商业授权；本轮仍不发布任何模型权重。

只从官方模型加载SegFormer编码器，按原B2八通道规格随机初始化分割解码器。原社区配置仅作为层规格/类别表模板，没有加载其权重。替代TR/TG同初始完整state SHA、LoveDA完整Train2522图、原生512裁剪、相同呈现/标注流，RGB和PIL L→RGB是唯一支线差异。3000更新，micro2×累积2，有效batch4，AdamW lr6e-5/WD.01、clip1、无调度器、BF16训练，每500步验证。此预算和优化器用于重新建立有许可来源的分割头，不能当作T06继续训练1000步的延续或直接控制对照。

```bash
# 先在能联网主机下载固定revision的config/preprocessor/pytorch_model.bin，核对initialization_license.json。
# 将三文件镜像到服务器runs/t07_official_init，数据复用已核验的runs/t06_teacher_data。
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/train_t07_teacher.py --mode TR --output runs/t07_teacher_TR_v1
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/train_t07_teacher.py --mode TG --output runs/t07_teacher_TG_v1
PYTHONPATH=. .venv/bin/python scripts/audit_t07_teacher.py
PYTHONPATH=. .venv/bin/python scripts/audit_t07_teacher.py --run runs/t07_teacher_TG_v1 --tag TG_official
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/review_t07_teacher.py --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/audit_t07_mask.py
```

选模规则仍为新Val64（城乡各32、排除旧8）的灰度累计混淆矩阵七类mIoU，平手保留更早。RGB、旧8均报告但不另选模型；64只是快速验证预算。GT0→255损失掩码，1—7不移位；有效GT上预测0计错。

输入uint8灰度复制三通道，512双线性缩放、1/255、ImageNet归一化；logits先回原生256、align_corners=False，再softmax8。8×8有效面积均值到32粗网格，显式排除Ignore并保留概率质量a，形成七类条件概率。基础mask沿用完整有效覆盖、a≥.8、条件最大概率≥.6；监督权重为mask×a×(1-H/log7)。不将高Ignore低质量在重新归一化后当作高置信。

OS固定32张全部来自训练光学，检查原图/预测/接受mask/绝对权重。源域通过掩码的粗格中，Barren展开精度仅36.97%，因此Barren为主类的格权重置零，七类概率完整保留，不删类重归一化。按原T06固定全背景旗标（背景≥95%且平均最大概率≥90%）全图弃权；固定32中视觉含歧义/结构误分的影像ID也可保守整图弃权，记录依据在freeze配置。此规则不使用匹配验证成绩，既不提高阈值掩盖大面积错误，也不生成OS准确率。源域另以真实GT统计接受粗格的错误；32粗格展开比较会包含混合边界，不能当作原生像素IoU或OS错误率。弃权前后展示可用`scripts/render_t07_abstention.py --root ...`从原32张私有概率重绘，不能把预览float16量化后的格统计当全量缓存的逐位复现。正式冻结须记录完整决策、SHA、类别分布、弃权清单和检查结果；填写OS_visual_decision.json后执行`PYTHONPATH=. .venv/bin/python scripts/freeze_t07_teacher.py`，已有冻结文件拒绝覆盖。

## 训练集缓存与学生评分

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 runs/teacher_preflight_v1/.venv/bin/python \
  scripts/cache_t07_semantics.py --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t07_head.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/evaluate_t07_selection.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
```

缓存入口要求技术冻结和明确研究许可。7912个ID及顺序必须与T06几何标签缓存完全相同，核验光学源SHA，不读取OS val/test。条件概率以float16存盘，训练恢复FP32后重新归一化；绝对权重FP32。Barren主类粗格、极端全背景及固定助手保守审查ID权重为零，不改可靠性几何标签。概率、权重、保留质量及全量弃权记录保存在私有runs。

主比较A_cap/S均全局top-K，K256为主、K512仅参照。U/A/AC复用T06同一冻结模型/输入/规则的完整952结果；A_cap复用已训练的7831参数纯外观头，新增S为7977参数（包括学生语义分支），差1.83%。S/SC加载同一个最后检查点，只改变全局选择或原覆盖分配；AC/SC是次要比较，不能替代S/A_cap。

两头同7912对/保存仿射、12epoch/2976更新、batch32、lr.001/WD.0001、clip5、同样本顺序、正类平衡masked BCE及固定last规则。S附加预先固定系数1的加权KL，除以格数保留绝对质量。学生语义概率对匹配BCE detach，只由蒸馏更新语义分支。骨干不实例化进训练，不解冻描述子，也不做硬选择适配或动态K。

推理使用冻结B0编码器和训练好的学生头：光学64及masked SAR GAP64→学生语义/联合评分→选择→gather→唯一K×1024相关→原局部精化→原完整仿射RANSAC。推理不导入Transformers、不加载教师/语义缓存/GT。光学、SAR的骨干BN仍eval，学生头BN在验证eval不更新。

952固定验证沿用独立探针、失败分母和原阈值/RANSAC。报告选中查询正确比例、实际精化窗口命中、内点凸包覆盖，以及SR@3/5、RMSE中位/P90、求解失败和5px未成功数。指标提高需首先对照A_cap，并查看逐pair成败转移；训练AP改善不自动证明配准机制有效。

## 验收和同步

```bash
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/summarize_t07.py
PYTHONPATH=. .venv/bin/python scripts/verify_t07.py
```

最终验收核对原匹配器和受保护源码SHA、教师研究许可和checkpoint、7912训练ID/权重、相同头预算与流、同一S/SC检查点及完整952结果。训练权重/原始预测记录私有，Git仅含代码、配置、非像素计划和指标。权重、影像、缓存数组不公开；本地交付索引和同步状态列明本地/zzvr/GitHub提交及私有产物位置。

教师下载的固定文件URL、大小与SHA均已保存；不需要访问社区权重才能重建替代教师。复用服务器上的处理器配置只是512/ImageNet的数值配置，不加载处理器目录中的模型权重。
