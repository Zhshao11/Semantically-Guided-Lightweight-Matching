# SOMA复现可行性核查

核查日期：2026-11-27。官方仓库：[traslauc/SOMA](https://github.com/traslauc/SOMA)。固定提交：`481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3`（提交日期2026-07-07）。论文：[arXiv:2511.13168](https://arxiv.org/abs/2511.13168)，首次提交2025-11-17。

**结论：SOMA比前轮两个通用跨模态模型更直接针对SAR—光学，但当前公开仓库不具备“下载官方权重后直接复现精度”的条件。可以检查部分架构与算子，尚不能列为已完成复现的外部基线。主要阻塞是训练好的SOMA权重未找到，以及公开实现的坐标/分辨率问题需要作者确认。**

## 1. 当前公开材料与缺失项

| 项目 | 核查结果 | 对复现的影响 |
|---|---|---|
| 官方代码 | 有Encoder/FGE、GLAM、损失、dataloader、train.py、test.py，共8个Python源文件；MIT许可 | 能做源码审计和部分前向，不代表权重或论文结果可复现 |
| 训练好的SOMA权重 | 主分支无checkpoint；README无下载入口；官方Releases页面明确显示尚无release；公开refs仅main、无tag | 当前不能进行有意义的预训练模型952对评价 |
| 权重搜索范围 | 核查主分支文件、README、官方release与heads/tags；API列表请求被限流，采用Git refs和GitHub页面核对 | “未找到公开发布”不等于断言作者没有私有权重；不擅自联系作者或发issue |
| 环境配置 | 没有requirements.txt、environment.yml或明确版本锁定/安装文档 | 要按实际源码整理环境，不能宣称原官方环境已复现 |
| Encoder初始化 | 硬编码作者本地`/home/F/whd/proj/somatchv5/facebookresearch_dinov2_main`，原样运行FileNotFoundError | 需要匹配的本地DINO源码与初始化配置 |
| DINO变体 | `dinov2_vitl14_reg`，即带register的ViT-L/14 | 前轮MINIMA使用的无register ViT-L/14权重不是同一artifact，不能直接冒充替换 |
| 数据/划分 | 按`root/dataset/split/{sar,opt}`同文件名配对；无发布训练/验证/测试manifest或固定扰动清单 | 有数据根目录也不足以复原论文样本协议 |
| 依赖现状 | zzvr现有Torch2.11.0+cu128、torchvision/kornia等可导入；缺wandb、torchsummary | 缺包属于可处理的工程项，但不是当前最大阻塞；未为本轮安装或训练 |
| 256输入 | 原GLAM存在硬编码512/256，实测256特征路径尺寸报错 | 不能直接挂入现有256流程，需明确512原生输入/恢复坐标或作者支持的适配方式 |

训练脚本读取`checkpoints/backup.pth`恢复，保存`checkpoints/best_modelss_ex.pth`；测试默认读取`checkpoints/model.pth`，要求文件内含`encoder`与`matcher`两组state_dict。代码中出现这些路径只是本地文件引用，不是公开权重下载。README目前仅有标题与论文链接。

## 2. 已在单张RTX5090实测的内容

使用服务器原环境，只把官方源代码快照放入私有`runs/soma_audit_v1/SOMA`；未修改上游文件。诊断入口：`scripts/audit_soma_runtime.py`。证据：[runtime_probes.json](runtime_probes.json)、[原始日志](runtime_probes.log)。

| 检查 | 结果 | 能证明什么 |
|---|---|---|
| torchvision DeformConv2d CUDA前向 | 通过，输出有限 | 5090关键可变形卷积算子可用，不存在已知的这一项硬件阻塞 |
| GLAM，512特征尺寸，随机初始化/随机特征 | 前向通过，flow_final为1×2×512×512，输出有限 | 仅证明这一匹配器子模块在原生尺寸可执行；没有运行完整Encoder、没有加载训练好的SOMA权重 |
| GLAM，256特征尺寸 | 失败：`128 must match256 at dimension3`，位于matcher.py:401 | 原公开实现不能直接接受现有256特征尺度 |
| 原Encoder构造 | FileNotFoundError，作者DINO本地路径不存在 | 完整端到端尚未跑通 |

GLAM组件实际参数为10,046,932，**不含两端ResNet/FGE和DINO，不能当作SOMA整网参数量与T05比较**。本轮没有给随机模型测“精度”、没有运行952对、没有报告正式端到端延迟或完整模型显存。

## 3. 两处需优先确认的几何问题

### A. 单位仿射没有产生零位移

[theta_to_flow](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/models/matcher.py#L210)使用`affine_grid(..., align_corners=False)`，但减去的base_grid由`linspace(-1,1)`生成。这两套网格的像素中心约定不同。

将单位矩阵直接输入原函数，结果如下：

| 尺寸 | 位移分量最大绝对值(px) | 位移向量最大范数(px) |
|---:|---:|---:|
| 16 | 0.468750 | 0.662913 |
| 256 | 0.498047 | 0.704345 |
| 512 | 0.499023 | 0.705726 |

按位移场定义，单位变换应产生零flow，这一性质未满足。它可能影响像素级精度解释；不是经过SOMA训练权重评价后得到的性能结论。

### B. 零flow精化的采样网格交换了x/y

[RefineMatcher.forward](https://github.com/traslauc/SOMA/blob/481a28cf0b4ef1086a5cb03d605ed7ac2dc158f3/models/matcher.py#L329)把`meshgrid(y,x)`的两个输出直接stack后传给要求末维为`(x,y)`的grid_sample。

在8×12非方形、内容不对称的合成特征上输入全零flow，原始采样结果与输入最大差75.041672（此处是合成特征值差，**不是配准像素误差**）；原采样grid与独立构造的正确xy恒等grid最大差约1.791667（归一化网格单位）。零flow未保持恒等采样。

这两项均独立于训练权重，只检查坐标契约。已训练网络可能对特定实现产生补偿，因此不能擅自修完源码后仍声称原权重/原论文的忠实复现；应先请作者确认实际实验版本与坐标约定。官方源码SHA核对未变化，本轮没有修SOMA模型。

## 4. 论文协议与当前项目不相同

论文[原文实验部分](https://arxiv.org/html/2511.13168v1)使用SOPatch提供的精配准SEN1-2，以及GFGE_SO；声明WHU-SEN-City与OS用于泛化、没有用于训练。公开train.py目前只选择SEN1-2，没有提供GFGE对应训练配置/权重映射和数据划分清单。

论文主协议为最大平移32px、尺度变化0.2、旋转±5°；更强的50px/±20°用于消融。公开dataloader实际在线抽样±20°、每轴±50px，scale固定1.0，而且验证/测试每次读取也重新随机生成变换、未设固定种子。因此不能把现成test.py默认运行当作论文主表复现。

论文称512×512每对约94ms；这不是本轮实测，不能移植到5090速度表。论文使用RTX4090实验平台，其输入尺寸、原生稠密flow和评价均不同于现有T05。

论文报告的OS CMR@5px为92.96%、R_avg为3.37，来自SOPatch提供的OS影像。**尚未核实其具体图像/划分与当前OSDataset2.0的952对相同；原稠密flow误差与现有独立探针仿射误差也不同，不能直接与T05的54.62%比较。**

当前T05是在256输入、固定±10°/0.9–1.1尺度/±16px/剪切±5°下，先预测对应再拟合完整仿射，使用独立探针评价；SOMA原测试直接算稠密场误差，无RANSAC。

另外，SOMA数据加载器GT为`d(y)=A_forward^{-1}y−y`，定义在扰动SAR坐标y上，是反向采样位移；不能直接把它当光学→扰动SAR的A_forward。若以后加入统一仿射参照，应先依据方向构造预测对应`x=y+d_hat(y), target=y`，并验证512→256恢复、掩码及采样规则。该适配会单独标为“统一仿射协议”，不冒充论文原生稠密结果。

## 5. 建议的复现准入条件

先取得并核对：

1. 训练好的SOMA checkpoint（包含encoder/matcher）、来源和SHA，明确SEN1-2/GFGE模型分别对应哪张表。
2. 其实际使用的代码commit、DINOv2带register的版本/初始化方式与环境依赖。
3. 训练/验证/测试清单、输入尺寸、主实验增强分布、固定随机种子/矩阵。
4. 单位仿射与零flow检查的预期、xy顺序/align_corners/flow单位和方向说明，以及256输入是否受支持。

拿到上述材料后，先做官方512完整推理与几何回归，再考虑同一952对与单卡计时。不应因为有train.py就默认启动100epoch重训；这既超出当前核查，也不能保证复现论文或保留零样本OS设置。

**本轮状态：源码与部分运行可核查；官方预训练精度复现暂不具备条件，不能加入“已复现外部方法”表。**

原文摘录/官网入口状态与源码哈希：[source_audit.json](source_audit.json)；综合状态：[disposition.json](disposition.json)。
