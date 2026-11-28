# Semantically Guided Lightweight Matching

面向光学–SAR 影像配准的轻量稀疏匹配网络，按当前 **v0.1 设计材料**建立的首版 PyTorch 实现。

**当前状态：已完成MINIMA-RoMa、XoFTR官方预训练权重的零样本参照，不微调、不重训。** 同952对，SR@5为T05 54.62%、MINIMA-RoMa 17.33%、XoFTR 1.37%；统一CPU原图到仿射边界的5090单卡中位延迟分别10.14/405.00/36.74ms，独立进程峰值allocated显存53.90/7183.18/1237.58MiB。原生内部尺寸、精度、训练域不同，不能宣称同域公平架构对比或泛化SOTA。122项测试通过。见[外部对比表](reports/external_references_v1/外部对比表.md)、[开发预算v1](reports/external_references_v1/开发预算_v1.md)、[复现说明](docs/EXTERNAL_REFERENCES_REPRODUCTION_zh.md)。此前[稀疏精化](reports/sparse_refinement_v1/report.md)与[T10审计](reports/t10_common_support_v1/审计报告.md)保留。

SOMA另完成[复现可行性核查](reports/soma_audit_v1/SOMA复现核查.md)：有官方SAR—光学代码，但未找到公开训练权重；原Encoder路径不可用，512的GLAM随机特征前向可运行、256路径报错，两项几何恒等检查未通过。未训练、未报告模型精度，也未加入已复现外部表。

在新的训练授权下，SOMA已另立[OS适配训练任务](reports/soma_os_v1/训练启动与耗时估计.md)：保持原上游文件，修正几何并使用正确register版DINO，7912训练/952验证选模/1696最终测试，4张空闲5090后台训练，上限100轮。实测估计满100轮约18–26小时；127项测试通过。训练后自动做测试与单卡计时，最终SR和推理成本当前待产生，不替代官方复现核查结论。


固定K256局部精化受控比较已完成：冻结T05编码/粗匹配，两组仅微调精化器、相同7912次更新。952对验证，原精化器/一层2048参数局部交互的SR@5为52.63%/51.89%，窗口内平均定位误差仅额外改善0.011px；单卡无hook中位端到端9.377/9.786ms，独立进程峰值allocated41.11/42.78MiB。该单种子、4epoch预算结果不支持采用新模块；未改变查询、精化数、阈值或RANSAC。133项测试通过。见[精度—开销报告](reports/local_interaction_v1/report.md)及[复现说明](docs/LOCAL_REFINEMENT_REPRODUCTION_zh.md)。SOMA在测速后已从检查点恢复原4卡训练。


新增SAR专用公开权重参照已完成：FHReg-GUSO（512主设置）/GDROS-WHU在同952对公共仿射协议的SR@5为65.44%/31.83%，独占5090三轮新进程中位端到端57.45/261.03ms、峰值allocated1133.86/335.72MiB。GDROS原生32次+LSR另列31.51%、60.99ms，协议不与公共仿射拼接。FH的256尺寸对照仅0.11%；512有21对求解失败及极端近共线坏例，全部保留。T05两预算逐对复现；旧MINIMA/XoFTR精度经哈希与2输入逐位复核后复用，资源本轮重测。实测非支配集合为T05全量、均匀K256和FH512；开发预算取FH512这一个完整点。RIFT2仅核查MATLAB运行条件，未数值复现。138项测试通过，本轮无新训练。见[外部对比报告](reports/sar_references_v1/外部对比报告.md)、[预算v1](reports/sar_references_v1/开发预算_v1.md)、[复现说明](docs/SAR_REFERENCES_REPRODUCTION_zh.md)。

SOMA测速后恢复原4卡并继续更新，但于epoch12/update22050出现非有限loss自动停止；稳定checkpoint保留在update22000，当前未运行、未产生最终测试成绩。本轮未为续跑修改SOMA协议，见[故障记录](reports/sar_references_v1/SOMA_later_failure.json)。


原始T05 K256的[窗口内理想定位诊断](reports/window_oracle_v1/实验报告.md)已完成：同952对、固定全部122881条拟合候选与原RANSAC，按真实有效采样点凸包+影像mask判断45974条可替换对应（比旧box少116条）。SR@5从原始51.68%升至理想92.54%，救回390/460个原失败、1个原成功退化；SR@3为32.56%→91.07%。此为名义GT目标替换的条件式潜力诊断，不是可部署成绩或严格单调性能上界；没有训练、测速、新候选/过滤或新精化器。145项测试通过，952对空干预完全一致；退化个例在3个配对RANSAC种子下不变，未扩大多种子。见[复现说明](docs/WINDOW_ORACLE_REPRODUCTION_zh.md)。


FHReg结果线追加核查已完成：[随机三类各10对、无网络坐标回环、952台账和统一测速](reports/fhreg_result_audit_v1/核查报告.md)。30对逐项复现原结果，原图/原始匹配/过滤/内点/独立探针/叠加及局部patch留私有证据；多例结构支持存在，重复农田的逐点物理身份仍无法仅靠目测确认。坐标七场景最大误差4.43e-5px，SR@5独立重算623/952，21求解失败保留分母，RMSE中位/P90范围为全部931求解对。相同24对/预热3/三轮各5次/CPUuint8到仿射边界/新进程显存的本轮中位为T05全量10.365ms、FH512 62.607ms，峰值allocated53.90/1133.86MiB；旧9.787和10.138ms都包含RANSAC，差异不当架构加速。154项测试通过，无新训练或全952网络重跑。见[复现说明](docs/FHREG_RESULT_AUDIT_REPRODUCTION_zh.md)。


冻结原T05 K256热图的[A/B/C坐标读出实验](reports/heatmap_readout_v1/实验报告.md)已完成：A原全窗期望、B离散最大峰、C峰附近3×3细格局部期望，统一作用于全部122881拟合候选。952对SR@5为51.68%/49.89%/51.89%；C仅净+2对（116救回/114退化），P90和同45974条实际窗口命中对应的定位均值均更差（A3.300px、B4.831px、C4.449px）。最大峰<3px仅34.63%，结果不支持只改读出回收主要缺口；保留A，不再扫温度/阈值/邻域，不启动训练或成本测量。原A坐标逐位重构、952对矩阵/评价复现，161项测试通过。验证集已多轮开发，后续固定配置仍需未参与选择的测试集检验。见[复现说明](docs/HEATMAP_READOUT_REPRODUCTION_zh.md)。


## 当前优先阶段：数据与几何基础

本次阶段交付见 [数据与几何工程说明](docs/DATA_GEOMETRY_STAGE_zh.md)、[控制点格式](docs/CONTROL_POINTS_zh.md)及 `reports/data_geometry_v1/` 的完整审计和测试证据。配置实际读取 `configs/data_geometry_v1.json`；原有模型参考配置和短训练报告保留为历史。

统一清单覆盖原版256/512六个split和2.0新增区域，加载器支持按resolution/split筛选；父图关系通过双模态像素验证。该阶段不报告新的训练精度。

## 冻结去重训练清单

元数据收尾见 [冻结清单与状态说明](docs/METADATA_FINALIZE_V1_zh.md)。`reports/metadata_finalize_v1/train_256_dedup_v1.jsonl` 保留7912对，512版本保留1978对；原始清单和文件保留。使用双模态联合解码内容键、最小数值ID代表，全部keep/exclude映射可追溯。后续训练通过 `--manifest` 显式选择该视图，验证和测试仍使用原split。

父图内容映射独立标为verified/candidate/unresolved，卫星scene来源另设状态。控制点分组为一对大场景影像的72个点对应，本轮仅检查元数据，不做scene模型选择或阈值调整。

## T02：B0-Full最小训练闭环

已完成独立基础版本：全量1024×1024粗匹配、简单局部相关、完整仿射RANSAC；关闭语义、联合评分、预算与局部交叉注意力。固定32对train/16对val完成BF16训练1600步，并验证断点恢复和不接收GT的推理入口。

预测对应的独立探针评价：train的5px成功率29/32（90.625%），val为1/16（6.25%），当前泛化仍弱。详见 [T02验收报告](reports/t02_b0_full_v1/T02验收报告.md) 和 [工程复现](docs/T02_B0_FULL_zh.md)。T02不替代后续完整方法。

## T02受控诊断与T03全量观察

同一T02检查点的A/B/C受控结果：原训练固定扰动SR@5px90.625%，相同32对内容的新扰动宏平均12.5%，原val固定扰动6.25%。B为每对5次相关扰动，按32对原影像汇总。A/C逐项复现，eval/模态BN/统计冻结检查通过。见 [受控分层报告](reports/t02_controlled_ABC_v1/受控诊断与分层报告.md)。

T03从头启动7912对在线训练、952对固定验证，主lr0.001和仅改lr为0.0002的单卡对照，batch4与20epoch观察预算相同；每epoch记录、epoch1及偶数epoch完整验证，best按SR@5px预先选择。两者现已完成20epoch，best SR@5分别54.6218%（epoch18）和52.3109%（epoch20）。这是观察预算完成，不代表收敛。见 [协议](docs/T03_PROTOCOL_zh.md) 和 [最新只读汇总](reports/t04_preparation_v1/T04准备核查报告.md)。旧运行观察保留为历史快照。

## T04：不依赖最终权重的准备

已核查SegFormer-B2 LoveDA社区权重及预处理，发现实际8通道（含Ignore）、权重许可未明确；8张训练光学预览显示部分合理轮廓，也有高置信全Background，不能直接批准正式教师。新增面积聚合/保留质量掩码与KL梯度测试，未接入B0。新增12项、全仓库69项测试通过。

核查6个官方外部基线的提交/环境/许可；汇总22次T03完整验证与208条既有分层诊断。正式计时脚本默认dry-run且拒绝同卡其他CUDA进程，本轮未产生正式效率数字或最终可匹配性标签。详见 [T04说明](docs/T04_PREPARATION_zh.md) 和 [核查报告](reports/t04_preparation_v1/T04准备核查报告.md)。下方旧版七类缓存模板不适用于当前八类候选，本轮使用独立preview入口。

## T05：冻结候选与无需重训的稀疏初筛

已冻结T03主运行epoch18检查点，952对逐pair矩阵/RMSE复现完全一致；实际B0为37,219参数，已输出逐模块和逐叶层结构。K1024/512/256/128完整验证SR@5分别54.62%/55.15%/51.68%/42.02%。独占5090、预选24对的无hook端到端中位延迟约8.8—8.9ms，未见明显加速；峰值分配显存53.91→39.15MiB。96对诊断表明粗对应/入窗率及内点覆盖仍是主要问题，精化平均有益但存在回退案例。

同一B2在固定LoveDA Val8张上RGB mIoU63.63%，同图灰度复制RGB为41.16%；支持先做灰度增强适配诊断，尚未训练教师，许可仍未明确。新增8项、总77项测试通过。本轮不训练、不改阈值或RANSAC，不生成最终可靠性标签。见 [T05报告](reports/t05_delivery_v1/T05实验报告.md) 和 [复现说明](docs/T05_REPRODUCTION_zh.md)。


## T06：纯外观选择、教师适配和效率诊断

同一冻结epoch18、原阈值/RANSAC，7912对train生成可靠性标签（正19.89%/负52.27%/忽略27.84%），只训练外观评分头。952对固定验证K256的SR@5：U51.68%、A53.78%、AC52.52%、容量对照53.26%；K512仍是U最高55.15%。A显著提高选中查询正确比例，但最终配准收益有限，当前覆盖规则没有一致收益；正常推理严格先评分/选择/gather，再做K×1024相关。

LoveDA完整Train2522张，TR/TG同1000步、有效batch4、完全相同样本/裁剪/标注流，均按独立64张城乡分层验证的灰度七类mIoU选中step400。T0/TR/TG灰度39.21%/39.64%/53.62%，RGB61.55%/58.67%/58.49%。旧8另作纵向检查；32张OS训练图匿名抽查仍有漏分，不报告无GT的OS分割精度。社区权重许可未解决，原始/派生权重均不发布，SC等待教师最终冻结。

独占单卡、24对、三轮交错无hook计时：严格FP32分模态BN折叠通过192/192一致性门禁，配对节省约1.27ms；BF16折叠0/192通过，不用于正式速度表。FP32副本速度不能与BF16主表精度拼接。87项测试通过。见 [T06实验报告](reports/t06_delivery_v1/T06实验报告.md)、[复现与私有产物说明](docs/T06_REPRODUCTION_zh.md)。


## T07：教师冻结与全局语义选择

旧TG的checkpoint/哈希/64验证/类别分布已完整留档，但社区权重研究条款仍不明，不用于T07学生标签。替换为官方`nvidia/mit-b2`固定初始化，模型卡明确链接仅限非商业研究/评价的SegFormer许可；不宣称自由部署，不公开权重。替代TR/TG同3000步，分别选中500/2500步，64灰度mIoU36.15%/44.59%；新TG低于旧社区TG53.62%，不混作同初始化继续训练。

教师Barren在接受粗格中的展开精度仅36.97%，其主类粗格弃权；保留七类概率和Ignore质量，不删类重归一化。原32张OS训练图掩码复查后7张保守整图忽略；同一高置信全背景规则扫描7912图，共967张整图弃权。正语义权重格为67.62%，仅作弱监督候选，没有OS语义准确率或无错保证。

主比较改为全局top-K的S/A_cap（头参数7977/7831）。同7912训练扰动、12epoch/2976更新及last规则，仅训练学生语义/评分头，教师离线。952固定验证K256 SR@5：U51.68%、A53.78%、A_cap53.99%、S53.89%；K512 A_cap54.62%、S53.57%，均无求解失败。S本轮没有等容量收益；AC/SC是次要覆盖消融，额外容量对照另列，不能替代主结论。

91项测试和跨文件验收通过。见 [T07报告](reports/t07_delivery_v1/T07实验报告.md)、[教师冻结配置](reports/t07_teacher_v1/teacher_freeze.json)及[复现说明](docs/T07_REPRODUCTION_zh.md)。


## T08：外部教师软伪标签辅助粗表征

三组同T05初始化、7912对训练图及保存仿射、均匀K256、粗编码/描述子开放范围、batch4、AdamW lr2e-4和4epoch/7912更新。M仅匹配损失，MS加可靠格光学伪标签KL，MS-shuffle跨影像置换伪标签。接收侧mask/权重完全相同，全局软类直方图保持，加权类别质量最大偏差0.23%。语义头不detach粗特征，KL能更新粗编码层；不向SAR提供像素语义KL，所有骨干BN统计冻结。

固定last在952验证上的K256 SR@3/5：M33.93%/53.36%、MS36.76%/55.04%、MS-shuffle35.82%/55.15%，均0求解失败；K1024 SR@5为56.41%/56.93%/56.51%。MS的SR@3较高，但SR@5未胜打乱组，单seed四轮尚不能隔离稳定语义内容收益；两辅助组都高于M与正则化解释相容。没有因结果修改预算、lambda、阈值或RANSAC。

前置128对train诊断显示T07门控修正并非接近零，直接替换教师概率没有SR@5收益；教师不是真值。OS无人工语义GT，只使用外部光学教师的软伪标签和掩码，不报告OS语义精度。导出纯匹配器仍37219参数，删除辅助头不等于整网加速，本轮未正式测时。96项测试与验收通过。见[T08报告](reports/t08_delivery_v1/T08实验报告.md)和[复现说明](docs/T08_REPRODUCTION_zh.md)。


## T09：对照审计与配对种子复核

原T08打乱中4274/5478278接受格（0.0780%）donor仍来自同一verified512父图。T09仅在原权重分层内作4273次交换，保留99.8441%原分配，排除此类donor，同时逐位保持接收mask/weight及全局概率/熵分布。旧T08文件与数值不覆盖：原seed的M/MS复用，shuffle另补修正版，再新增两个seed共6组，合计7个新run构成3个配对种子。

三个seed只改变训练顺序，同T05初始化/7912图/保存仿射、同解冻层、lr2e-4/lambda1/7912更新，统一last epoch4和均匀K256，不新增预算。三seed SR@5均值±样本SD：M53.75±0.34%、MS54.83±0.86%、修正版shuffle54.52±1.27%。MS相对M均值+1.09个百分点，238父图配对95%CI[-0.56,+2.73]；相对shuffle+0.32，CI[-1.19,+1.86]。MS对M胜2负1，对shuffle胜1负2，尚无稳定语义内容收益。

梯度审计显示MS/shuffle辅助约束强度不等，保持lambda不变，仅报告共同初始/T08末状态回放和新run预定步数实际梯度。打乱还破坏空间连续性，相邻接受格同类率98.26%→53.11%；不能视为只改变语义正确性的完美对照。父图重采样同步保留同一父图内4个子块与3seed相关性，不建立原卫星scene独立性。101项测试和验收通过。见[T09报告](reports/t09_delivery_v1/T09实验报告.md)及[复现说明](docs/T09_REPRODUCTION_zh.md)。


## T10-A：高分错误候选的离线语义信号

固定T05第18轮、256对train（128verified512父图各2子块）及保存仿射，全部1024粗查询的raw描述子top5错误在GT名义位置16px外选取，随后应用已有伪标签mask。SAR候选仅经A逆矩阵映回光学教师概率，未对SAR运行教师，不称SAR语义GT。候选池、随机对照、概率置换与矩阵均已固定。

653721条可靠错误关系中，高分错误减未匹配随机的余弦差+0.071447；同距离/可靠性分层后+0.013235，进一步同候选伪类后+0.001019（父图95%CI[0.000763,0.001276]）。去掉Background仍+0.003133，并非仅背景类，共同可比关系的描述性衰减约98.59%，不作因果解释。事后更紧匹配（距离差≤2px、权重差≤.02）只余+0.000321；这是敏感性，不改主分析。

233/256图、126/128父图有可比关系，缺失覆盖另报。Building/Road/Forest/Agricultural的同类差有探索性支持，Road/Forest父图数有限，Water不确定、Barren无可靠主类数据。存在弱信号不等于语义损失有用或能提高配准；本轮不进入全量训练、不实现新损失。106项测试及候选/掩码/坐标验收通过，固定16例不按差异挑图。见[T10-A报告](reports/t10_results_v1/T10A实验报告.md)和[复现说明](docs/T10A_REPRODUCTION_zh.md)。

## v0.1完整方法模板（与当前实际实验架构分开）

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

## 早期七类教师模板（非本轮八通道运行入口）

下列命令保留作历史模板；T06实际八通道教师、GT0掩码、质量权重和数据来源以[复现说明](docs/T06_REPRODUCTION_zh.md)为准。

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

当前加载器只对 SAR 做几何增强，optical 与其语义缓存不变；若后续增强 optical，必须同步变换软标签与 mask。下列早期七类模板当时仅做静态编译检查；T06已另用八通道独立入口完成真实教师对照训练，不能将两个入口的标签编号或缓存规则混用。

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
