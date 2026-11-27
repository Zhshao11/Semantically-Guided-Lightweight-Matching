# 外部方法来源、输入与协议审计

核查日期：2026-11-27。直接核查论文原页与作者官方GitHub，不使用SourceMux。只运行MINIMA-RoMa和原XoFTR，不使用MINIMA-XoFTR，也不使用服务器上已有但版本来源未验证的xoftr_sar代码。

| 项目 | MINIMA-RoMa | XoFTR |
|---|---|---|
| 论文 | [MINIMA: Modality Invariant Image Matching](https://arxiv.org/abs/2412.19412)，CVPR2025；arXiv首次2024-12-27 | [XoFTR: Cross-modal Feature Matching Transformer](https://arxiv.org/abs/2404.09692)，CVPR2024 Image Matching Workshop；arXiv首次2024-04-15 |
| 官方仓库 | [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA/tree/796e7721174f9f829b79b3702bf8c2ae9a3d447a) | [OnderT/XoFTR](https://github.com/OnderT/XoFTR/tree/e0fbea431b30be9742effbf5577c90aa8eb938f9) |
| 固定源码 | MINIMA `796e7721174f9f829b79b3702bf8c2ae9a3d447a`；RoMa作者适配子模块`0d3fd22dfaf5b7e463a93c3d7f0cbc26df953663`，按父仓库gitlink固定，没有运行update --remote | `e0fbea431b30be9742effbf5577c90aa8eb938f9` |
| 官方权重入口 | README直链[官方release minima_roma.pth](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth)；另需[Meta DINOv2 ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | README链接[作者Google Drive文件夹](https://drive.google.com/drive/folders/1RAI243OHuyZ4Weo1NiTy280bCE_82s4q)，选择`weights_xoftr_640.ckpt`，文件ID `1oRkEGsLpPIxlulc6a7c2q5H1XTNbfkVj`；未改用840版本 |
| 模型 | large RoMa，原对称匹配、certainty衰减、threshold_balanced随机采样10000对应 | 原640模型，coarse阈值0.3、fine阈值0.1、denser=False |
| 输入 | 官方外层OpenCV长边640，RGB PIL；RoMa内部coarse560×560、upsample864×864；ImageNet归一化由原transform执行。OS灰度图复制RGB，不推测彩色信息 | 灰度，OpenCV长边640、df8，本次640×640；uint8/255，模型内部原逐图均值方差标准化 |
| 精度 | 官方FP16/FP32混合，保留原内部autocast；未装xformers，使用官方attention fallback | 官方FP32，不额外加全局autocast |
| 训练来源声明 | 官方README/[训练说明](https://github.com/LSXI7/MINIMA/blob/796e7721174f9f829b79b3702bf8c2ae9a3d447a/train_orders/README.md)：MegaDepth及生成的MD-syn；包含红外/深度/事件/法线/草图/绘画等生成模态。RoMa发布训练yaml列visible/infrared/depth/normal、初始化Roma outdoor。该checkpoint未附完整逐图训练manifest，不能把所有六个生成模态都断言用于该artifact | [官方训练说明](https://github.com/OnderT/XoFTR/blob/e0fbea431b30be9742effbf5577c90aa8eb938f9/docs/TRAINING.md)：KAIST可见光/LWIR预训练，MegaDepth微调。原论文主要是visible/TIR，不是专门训练的SAR模型 |
| 训练重叠边界 | 未见官方声明使用OS；DINOv2大规模预训练/原RoMa初始化的完整逐图来源无法与本地OS一一核验，不宣称已证明零重叠 | 声明数据集未包含OS，但缺少该权重完整逐图训练名单；不能以“数据集名称不同”证明绝对无重叠 |
| 本轮定位 | OS光学–SAR零样本跨域参照 | OS光学–SAR零样本跨域参照 |
| 许可 | MINIMA仓库Apache2，RoMa子模块MIT；第三方DINOv2另有原许可。官方权重入口存在，但仓库许可不自动构成所有权重/训练数据的完整授权声明 | 仓库Apache2；METU-VisTIR数据另为CC BY-NC-SA4.0（本轮未下载/使用该数据）；权重需区分仓库/数据许可 |

所有权重、第三方原源码快照与像素/逐对应预测保持私有，不上传本工程GitHub；仅发布源网址、版本、SHA、适配器与聚合/逐图数值。权重SHA、文件大小和源码132文件快照见`freeze.json`与`official_source_snapshot.json`。

## 与T05相同和不同的条件

相同：原952个256×256输入影像对与固定仿射、原有效掩码、独立16×16探针、SR严格<3/<5px、完整仿射RANSAC阈值3px/min内点6/5000迭代/.999置信度、原RANSAC种子、RTX5090单卡、batch1、线程4、确定性设置，以及CPU原图到最终仿射的完整计时边界。无测试集选型，所有失败保留在952分母。

不同且必须明示：T05在OS去重7912对训练，外部模型本轮零样本；T05内部256/BF16，XoFTR内部640/FP32，MINIMA内部560/864和混合精度。**这是相同任务与边界下的原生配置参照，不是同训练数据、同内部像素数、同精度的严格架构公平比较，也不复现两篇论文的原姿态/单应性指标。** 没有通过压低外部尺寸、阈值或采样数来获得好看的速度。

统一后过滤只检查预测两端有限性、原256坐标范围与已知有效掩码双线性采样>.999，不读取A或误差。T05也应用此过滤，验证仍逐对精确复现。外部不擅自注入GT派生的语义、正确性或候选筛选。

坐标恢复沿官方公开wrapper：XoFTR输出像素坐标×256/640，RoMa先用原`to_pixel_coordinates`恢复640再×256/640。不凭步长或假设另加半像素偏移；官方接口约定继承并显式记录。独立测试覆盖xy/HW顺序、非方形恢复、边界/无效过滤和空匹配。

## 兼容性与实现边界

官方环境是Python3.8/Torch2.0.x/CUDA11.x，不能直接当作RTX5090复现环境。本轮实际为Python3.10.20、Torch2.11.0+cu128，依赖版本保存在`environment.txt`。使用现有兼容依赖路径解决requests/urllib3等导入，不修改现有匹配训练环境的包，不安装训练框架的旧版本。

上游RoMa的`CNNandDinov2.train`只委派给cnn，未更新容器自身training元数据；原CNN的BN和单独DINO均已eval。本轮显式将该容器training标记设False以通过状态审计；forward没有读取该标记，不改任何算子或权重。初始状态检查失败留档。DINO存于普通列表，参数统计与GPU峰值必须包含它，不能只看`model.parameters()`。

MINIMA原wrapper使用已移除的`np.float`且含GPU→PIL往返。正式适配器直接构造同样的640 RGB PIL，保持官方model.match/sample/to_pixel API；诊断时仅临时提供np.float别名运行原wrapper以核验输出，不用于正式计时。XoFTR直接走原model API和官方等价预处理。训练来源的官方wrapper等价检查见`official_io_equivalence.json`。

XoFTR wrapper坐标与分数逐位一致。MINIMA的官方wrapper用float64尺度，适配器用float32尺度，最终原图坐标最大差0.0000152588px、分数一致；保存最初1e-5px严格检查结果，并按256像素图坐标范围内2个float32间隔（3.1e-5px）作数值等价验收，不更改模型、坐标公式或匹配阈值。适配器移除冗余GPU→CPU→PIL往返，时延是本轮适配器的完整任务时延，不冒充原demo脚本的原样运行时间。

RoMa原`match`内部会在upsample前执行`torch.cuda.empty_cache()`，本轮保留并包含其成本。平衡采样前固定每对seed，保证评价/测速同一组对应；不能把随机重复当成独立场景。细节均在冻结配置，未根据验证成绩修改。
