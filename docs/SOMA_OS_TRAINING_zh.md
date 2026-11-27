# SOMA-OS后台训练与恢复

配置：`configs/soma_os_training_v1.json`；启动协议、实测时长、数据角色与结果路径见`reports/soma_os_v1/训练启动与耗时估计.md`。原SOMA源码在`runs/soma_audit_v1/SOMA`，不覆盖；几何适配在`sglm/soma_os.py`。正确的register版DINO源码与权重位于`runs/soma_os_v1/`，哈希见launch_freeze。

本次任务已经启动，不要再次运行launch以免重复占卡。主管进程1422704，原训练使用物理4–7号卡。最后由独立进程在GPU7做测试/测速，训练与计时不会同卡重叠。

```bash
# 查看一次状态（没有在本轮设置周期监控）
cat runs/soma_os_v1/training/status.json
cat runs/soma_os_v1/training/pipeline.json

# 恢复前先确认旧任务已结束，且指定4卡没有他人CUDA进程。
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/run_soma_pipeline.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0 \
  --gpus 4,5,6,7 --resume runs/soma_os_v1/training/last.pt
```

长任务可由独立会话/进程启动；本次已经通过`start_new_session=True`脱离SSH并将stdout/stderr重定向到日志。不是需要当前聊天保持运行的工具会话。

`EpochMicroSampler`用固定epoch置换与显式presentation token；各rank分配无重复，合并流与单卡累计流一致。每250更新保存可恢复位置，epoch与验证结果写入日志。恢复会接着last中的epoch/微批位置继续，日志作为原始追加记录；若中断时有未checkpoint的尾部更新，可能重复记录相同步号，应以checkpoint与恢复段为准，不把重复日志行当额外训练。

从训练后best单独恢复自动测试/测速：

```bash
CUDA_VISIBLE_DEVICES=GPU-e0ef04e8-e8b6-4435-c3e7-c8827ae9e45b \
PYTHONPATH=runs/teacher_preflight_v1/.venv/lib/python3.10/site-packages:. \
  .venv/bin/python scripts/finish_soma_os.py \
  --root /mnt/sda/zhushihao/data/vis_sar/OSDataset2.0
```

必须存在training_finished标记。该入口在同bestSHA已有final_test.json时复用测试结果，仅重做未完成项；不会根据测试成绩选模型。正式测速默认要求GPU7无其他CUDA进程，冲突会停止并记录，不终止他人进程、不静默使用另一张卡。

自动最终CSV包含val/test的SR3、SR5、失败数、RMSE中位/P90，以及统一计时清单得到的完整中位延迟和推理峰值allocated。val/test两行引用同一组24对正式计时，不把它们误称各自在全量split上的平均时延。

所有模型权重、检查点、原始像素和DINO源快照保持本地/服务器私有。GitHub同步实现、协议、数据清单与聚合/启动证据；训练结束后的指标先在服务器生成，之后可拉取到本地交付目录。
