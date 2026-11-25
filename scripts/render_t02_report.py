"""Generate T02 figures and report from persisted run evidence, without new training."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',required=True)
    a=ap.parse_args()
    root=Path(a.directory)
    train=json.loads((root/'final_train.json').read_text())
    val=json.loads((root/'final_val.json').read_text())
    initial=json.loads((root/'initial_train.json').read_text())
    acceptance=json.loads((root/'training_acceptance.json').read_text())
    loader=json.loads((root/'fixed_loader_acceptance.json').read_text())
    evaluator=json.loads((root/'evaluator_interface_acceptance.json').read_text())
    environment=json.loads((root/'environment.json').read_text())
    logs=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    steps=np.array([x['step'] for x in logs])
    for axis,key,title in [(axes[0],'total','Training loss'),(axes[1],'positive_coarse_accuracy','Positive coarse classification accuracy')]:
        values=np.array([x[key] for x in logs])
        axis.plot(steps,values,color='#7196ae',alpha=.3,linewidth=.6,label='Per step')
        mean=np.convolve(values,np.ones(50)/50,mode='valid')
        axis.plot(steps[49:],mean,color='#125977',linewidth=1.5,label='50-step mean')
        axis.axvspan(0,400,color='#ddd5c7',alpha=.35,label='GT local-window bootstrap (training only)')
        axis.axvline(400,color='#966a39',linestyle='--',linewidth=1)
        axis.set(title=title,xlabel='Training step')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[1].set_ylim(bottom=0)
    fig.savefig(root/'training_curves.png',dpi=170)
    fig.savefig(root/'training_curves.svg')
    svg=root/'training_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    lines=['# T02：256仿射监督加载器、独立评价器与B0-Full小样本训练', '',
        '日期：2026-11-25。最小训练闭环已完成，报告的对应与变换全部来自预测粗对应；验证集效果仍弱，暂不据此主张泛化达标。', '',
        '## 本轮范围', '',
        '沿用已定独立适配器、共享卷积/模态独立BN、DSRes16×2/32×2/64×3和Add-FPN，只保留32维粗描述子、全量相关、简单局部相关、完整affine RANSAC。未实例化语义头、联合评分、预算/选点或局部交叉注意力。', '',
        '## A：固定256仿射加载器', '',
        '去重训练清单7912对中固定抽取32对；原val划分固定抽取16对。样本选择、参数、A_forward、源图SHA和来源清单SHA写入fixed_plan.json；训练与val不重选、不在线改变本轮矩阵。', '',
        '每条返回pair_id、optical、sar_aug、A_forward、optical_valid_mask、sar_aug_valid_mask。A_forward为SAR原图→SAR增强图；图像由原有warp_image做逆向采样，传给OpenCV的是其forward-matrix API要求的正矩阵，不能误读为监督的逆矩阵。', '',
        '扰动：旋转±10°、等比尺度0.9–1.1、各轴平移±16px、各轴剪切角±5°，剪切系数取tan；绕(127.5,127.5)组合：', '',
        '```text', 'A = T_translation @ T_center @ R_rotation @ H_shear_xy @ S_uniform @ T_minus_center',
        'H = [[1,tan(shear_x)],[tan(shear_y),1]]', '```', '',
        '原图保持uint8/255，不做对数或第二次辐射变换。光学mask表示原图坐标域有效；SAR mask表示双线性逆采样支持完整，不以灰度是否为0判无效。影像采样沿用计划的float64矩阵，监督返回FP32矩阵张量；两者仅有浮点舍入差异。标签使用FP32坐标和保守粗cell有效mask，越界或目标cell不完整的查询监督为no-match，不作为正匹配。', '',
        f"全部{loader['pairs_checked']}个固定样本重复读取及改变epoch后图像、矩阵、mask逐位一致；逐样本记录out_of_bounds_positive_queries=0和unsupported_target_positive_queries=0。已目视检查两张联系图的20组真值连线与红色无效区。",
        '提供显式mode=online的epoch级在线扰动接口用于后续全量训练，本轮train_t02固定使用mode=fixed。合成A监督仍假设原始光学/SAR配准残差较小，不能把它解释为人工真实对应完全精确。', '',
        '## B：独立评价器', '',
        'sglm/evaluation.py只依赖NumPy、OpenCV和既有geometry，独立子进程测试确认不加载Torch或model。包入口改为lazy export以保证这一点。', '',
        '- estimate_from_correspondences只接收预测点和分数，输出完整2×3 affine或明确失败。',
        '- evaluate_transform只接收预测矩阵和外部评价点；合成patch由固定16×16有效探针构造目标，scene可直接传独立控制点。',
        '- 探针为独立固定格心，区别于32×32粗匹配格心；不根据预测、拟合点或inlier选择。有效性只由GT和输入mask决定。',
        '- 实际T02未读取scene控制点、未拟合scene变换、未用scene选模型或调阈值。',
        '- 所有图像对保留在成功率分母中；另报求解失败率、各阈值配准失败率和成功样本的误差。', '',
        f"接口验收：真值矩阵RMSE={evaluator['true_matrix_rmse_px']:.3e}px；错误矩阵额外平移(5,-3)后RMSE={evaluator['wrong_matrix_rmse_px']:.6f}px；空匹配和共线匹配明确失败。4个测试图像对中2次求解失败，@5px成功率仍为1/4，而非删除失败后的1/2。", '',
        '## C：B0-Full模型与混合精度', '',
        '| 张量 | 实测训练尺寸 |', '| --- | --- |',
        '| 粗特征 | 4×64×32×32 |', '| 细特征 | 4×16×128×128 |',
        '| 粗描述子 | 4×1024×32 |', '| 全量粗相关 | 4×1024×1024 |',
        '| 含无对应类别的行logits | 4×1024×1025 |', '',
        f"模型总参数{environment['parameters']}。粗匹配使用双向分类监督，两个方向均包含no-match；推理以全量行匹配和互为最近邻筛选，固定概率阈值0.01，未做预算或Top-K选点。训练局部监督最多采样256个窗口/对，不裁剪全量粗相关；推理对全部通过筛选的对应精化。简单精化将源细特征中心与SAR 7×7局部候选做共享线性投影和归一化相关，再soft-argmax；没有交叉注意力。",
        'RTX5090上的BF16 autocast前后向均通过：粗特征/相关为BF16，描述子归一化、坐标、loss和局部采样保留FP32以保持数值稳定，主权重和Adam状态为FP32。所有1600步loss和梯度有限。', '',
        '前400步训练可用GT附近局部窗口bootstrap；401–1600步局部训练用预测粗窗口，并只对真实位置落在窗口内的预测施加定位loss。报告中的全部推理从未使用GT窗口、GT对应、GT变换或oracle回退。', '',
        '完整affine通过既有estimateAffine2D RANSAC实现：阈值3px、最少6个inlier、最多5000次、confidence0.999，均在运行前固定。', '',
        '## 保存、恢复与无GT推理', '',
        '保存模型、优化器、步数、固定计划SHA、配置、Python/NumPy/Torch/CUDA随机状态；batch由步数、epoch和固定seed确定，支持epoch中间恢复。本次先训练400步，实际恢复后继续到1600步。',
        f"147组Adam状态在最终checkpoint均为1600步。从400步快照另行重放第401步，batch相同；loss、粗loss、细loss、梯度范数及粗分类准确率与主训练记录的最大差均为{max(acceptance['replay_max_absolute_difference'].values()):.1e}。这里只验证第401步重放，不宣称跨硬件全程逐位确定。", '',
        'predict_b0.py入口只接收光学图、SAR增强图、可选mask和checkpoint，不接收A或评价点。已用实际导出的影像和mask调用该CLI，输出对应与独立批评价路径的首个val样本逐项相同。', '',
        '## 实际小样本结果', '',
        f"1600步，batch4，共200个固定样本epoch。前50步平均loss {acceptance['first_50_mean_loss']:.6f}，后50步平均loss {acceptance['last_50_mean_loss']:.6f}。没有根据val结果修改阈值或选择checkpoint，验收使用预先约定的最后一步。", '',
        '| 数据 | 对数 | 求解失败率 | RMSE<1px | RMSE<3px | RMSE<5px | <5px成功样本平均RMSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,record in [('未训练train',initial),('训练后train',train),('训练后val',val)]:
        s=record['summary'];error=s['successful_mean_rmse@5px']
        lines.append(f"| {label} | {s['image_pairs']} | {s['failure_rate']:.2%} | {s['success_rate@1px']:.2%} | {s['success_rate@3px']:.2%} | {s['success_rate@5px']:.2%} | {'无成功样本' if error is None else f'{error:.6f}px'} |")
    lines.extend(['',
        f"训练后train在5px阈值下配准失败3/32；val失败15/16（{val['summary']['registration_failure_rate@5px']:.2%}）。两组求解器均返回矩阵，但矩阵可能错误；不能把求解失败率0%当作配准成功率100%。",
        f"所有可评价矩阵的RMSE均值：train {train['summary']['solved_mean_rmse_px']:.6f}px，val {val['summary']['solved_mean_rmse_px']:.6f}px。该均值与阈值内成功样本的条件均值分开保存。", '',
        '训练集结果说明基于预测对应的粗匹配→局部精化→完整affine→独立探针评价链条已能工作。val结果显示当前32对固定扰动训练主要完成小样本拟合，泛化仍弱；没有完成全量训练、全测试集评估或完整研究方法验证。', '',
        '## 文件与验收证据', '',
        '- tests.txt / tests.xml：52项测试全部通过，其中本轮8项；包括真实CUDA BF16前后向。',
        '- fixed_plan.json / fixed_loader_acceptance.json / loader_review_20.json：48个固定输入与矩阵、重复读取和正样本有效性。',
        '- gt_review_sheet_01.png / gt_review_sheet_02.png：20组抽查图，留本地/服务器。',
        '- evaluator_interface_acceptance.json：独立评价器接口验收；training_acceptance.json：训练和断点恢复。',
        '- train.jsonl / resume_history.jsonl / environment.json：完整逐步记录与硬件。',
        '- initial_train.json / final_train.json / final_val.json：逐对预测点、预测矩阵、独立探针、失败与误差。',
        '- gt_free_cli_prediction.json：不提供GT的CLI推理结果；visual_review.json：目视检查记录。',
        '- training_curves.png/svg：实际loss及正粗分类准确率曲线，包含原始值和50步均值。',
        '- 服务器checkpoint：runs/t02_b0_full_v1/training/last.pt（1600步），以及step_000400.pt；本地输出保留最终checkpoint。权重和原始图像不放入Git。', '',
        '原有几何算子未重写；原去重清单及数据保持不变。T02是完整方法所依赖的B0底座验证，后续语义、联合评分、预算与交叉注意力实验仍保留原研究方向。'])
    (root/'T02验收报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(root/'T02验收报告.md'),train_success_5px=train['summary']['success_rate@5px'],val_success_5px=val['summary']['success_rate@5px'])))


if __name__=='__main__':
    main()
