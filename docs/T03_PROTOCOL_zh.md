# T03：B0-Full全量在线增强首轮观察协议

本轮先固定T02检查点执行受控A/B/C与分层诊断，再从头启动全量训练。T02权重SHA保持`8c0600c8748d1358dac30d564368c42593c8938e86cf7a5ff690d40ddbc6fdc3`。诊断不重新训练，不改变匹配阈值、RANSAC参数或BN评价模式。

## A/B/C诊断

- A：32对原训练影像、原固定矩阵，结果/预测/矩阵与既有T02逐项复现。
- B：相同32对影像，每对5个新采样同分布变换；先保存全部160矩阵再推理。逐变换记录，同时按32对原影像汇总；不称160个独立场景。
- C：原16对val、原固定矩阵，逐项复现。
- 所有阶段使用eval，光学BN分支0、SAR分支1；hook验证实际分支，比较全部BN running mean/var与num_batches_tracked，不允许更新。
- A/B/C使用同一AffinePairs256、归一化、mask、B0坐标解码。B的光学输入张量哈希与A相同；只改变SAR变换。

逐pair分层结果包括有效查询/候选数、粗预测端点误差、正确率阈值3/5/8px、真值在7×7窗口内比例、同一批对应精化前后误差（窗口内/外分别统计）、实际RANSAC输入数和内点索引、源/目标内点凸包/网格/PCA分布和退化描述，以及独立探针误差/SR/失败状态。新增几何API参数只让求解器输出原本的内点索引，默认行为和参数不变。

真值只在预测和几何拟合结束后计算诊断，不改变对应、窗口、inlier或变换。7×7细层stride2窗口范围按B0既有定义为中心每轴±6px，GT须可见才算落入。精化窗口外组包含不可见GT，并另记录GT可见数。

## T03固定协议

- train：去重7912对256样本，全部使用；每次presentation分配新draw_token，在线采样T02相同分布/组合顺序，不增加颜色或散斑增强。
- val：全部952对，每对固定一个保存变换；保留T02原16对val的矩阵，其他936对用固定seed生成。test图像不加载，不用于模型选择。
- 模型、loss、AdamW、weight_decay、梯度裁剪、BF16、有效batch4及fine bootstrap 400步均沿用T02。
- 初始化：从头，同seed；与T02初始随机权重相同，不使用T02训练权重。
- 主运行lr=0.001；独立对照仅改lr=0.0002。均无scheduler，样本顺序/增强种子/预算相同，分别单卡运行，不扩展8卡。
- 首轮20epoch=39560步，是观察预算，不宣称收敛。每epoch记录训练汇总；epoch1及每偶数epoch（2/4/.../20）完整验证952对。
- best指标预先确定为包含失败样本的SR@5px最大值；相同时保留较早epoch。last和best独立保存；每次验证确认eval及BN统计不变。
- 两运行每epoch保存样本顺序+增强种子+实际矩阵的流SHA，便于核对仅学习率不同。

## 运行与恢复

```bash
export PYTHONPATH=.
export SGLM_DATA_ROOT='/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0'

.venv/bin/python scripts/prepare_t03.py --output reports/t03_setup_v1
.venv/bin/python scripts/preflight_t03.py --root "$SGLM_DATA_ROOT" --directory reports/t03_setup_v1

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/train_t03.py \
  --root "$SGLM_DATA_ROOT" --plan reports/t03_setup_v1/full_data_plan.json \
  --config reports/t03_setup_v1/main_config.json \
  --diagnostic-gate reports/t02_controlled_ABC_v1/summary.json \
  --preflight reports/t03_setup_v1/preflight.json --output runs/t03_main_lr1e3_v1

# 对照使用GPU5和lr2e4_config.json；其余参数相同。
```

长任务通过后台进程运行，PID、实时phase/epoch/step保存在各run的status.json及启动清单。原始step日志、逐epoch训练/验证汇总、全部952对逐pair结果和checkpoint留在服务器；本地同步观察快照。可用`--resume <run>/last.pt`从最后完成epoch恢复；未完成epoch需重跑。若checkpoint标记待验证，恢复时先补完该次验证再训练下一epoch，日志越过恢复点的记录另存orphaned。没有通过train模式评价来提高分数。

诊断可用于判断粗匹配/精化/几何环节；B下滑不能单独证明全部机制。20epoch结束后应依据完整验证曲线再决定是否延长，不能直接把轮数完成称为收敛。
