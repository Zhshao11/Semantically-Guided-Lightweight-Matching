"""Human-readable ABC/layered results, keeping B's image-pair grouping explicit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args();p=Path(a.directory)
    summary=json.loads((p/'summary.json').read_text())
    b=json.loads((p/'group_B_per_original_image.json').read_text())
    a_rows={r['pair_id']:r for r in json.loads((p/'group_A_pairs.json').read_text())}
    lines=['# T02同检查点受控诊断与分层结果', '',
        '日期：2026-11-25。没有重训、没有改变匹配阈值或RANSAC参数，使用原1600步T02检查点。A/C预测点、矩阵及评价字典逐项复现。', '',
        '## 受控比较', '', '| 组 | 原始影像对 | 扰动评价次数 | SR@3px | SR@5px | 求解失败数 | 可求解RMSE均值 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g,images in [('A',32),('B',32),('C',16)]:
        r=summary['groups'][g]['registration']
        lines.append(f"| {g} | {images} | {r['image_pairs']} | {r['success_rate@3px']:.3%} | {r['success_rate@5px']:.3%} | {r['failed_pairs']} | {r['solved_mean_rmse_px']:.6f}px |")
    lines.extend(['', 'A=32对训练原内容+原固定变换；B=同32对内容、每对5个新采样同分布变换；C=16对原val+原变换。B只包含32对原始影像，其160次是相关重复扰动，不是160个独立场景。',
        f"B逐原图宏平均SR@5px为{summary['B_macro_SR5']:.3%}；{summary['B_any_success_contents']}/32对至少一次新扰动成功，{summary['B_all_five_success_contents']}/32对五次全成功。因每对都是5次，宏平均恰好等于20/160的扰动级比例，但统计单位仍为32对内容。", '',
        '**解释：A好、B明显下降，说明同一影像内容换成新几何后也不稳，固定扰动记忆/适应较强的可能性得到支持。不能仅把C差解释成影像内容泛化问题。** 这不是单独的机制证明，内容和几何泛化仍需后续全量在线实验。', '',
        '## 推理状态核查', '',
        '- 208次评价全部model.eval，所有子模块training=False。',
        '- 实际hook检查光学走adapter/BN模态0，SAR走模态1；每侧adapter被调用208次，每侧ModalBN共5616次。',
        '- 对比全部running_mean、running_var和num_batches_tracked，BN统计逐位不变。',
        '- A/B/C使用同一个加载器、uint8/255归一化、有效mask、粗坐标解码与局部采样；B光学输入哈希与A相同。',
        '- 没有用train模式提高验证结果；T02检查点SHA保持不变。', '',
        '## 分层汇总（逐影像pair的均值）', '',
        '| 组 | 粗对应数 | 粗正确比例@5px | GT落入7×7窗口 | 窗口内精化误差变化 | 窗口外精化误差变化 | 内点源凸包覆盖 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for g,v in summary['groups'].items():
        lines.append(f"| {g} | {v['mean_predicted_matches']:.2f} | {v['mean_coarse_correct_fraction_5px']:.3%} | {v['mean_truth_in_window_fraction']:.3%} | {v['mean_refine_delta_inside_px']:.6f}px | {v['mean_refine_delta_outside_px']:.6f}px | {v['mean_inlier_source_hull_fraction']:.3%} |")
    lines.extend(['', '粗正确比例分母为预测对应数，GT不可见的对应不算正确；另存GT可见对应的误差和正确比例。阈值3/5/8px写入诊断配置，未用于改动推理。窗口定义为GT可见且相对预测粗中心每轴≤6px（7×7细层，stride2）。精化变化=精化后−精化前，负值为改善；窗口内/外始终使用同一批预测对应，不用GT重新选择推理窗口。',
        '分层误差均值只对存在该类对应的pair计算，空类别保留count=0与null，不填0；最终SR的分母仍含所有失败。B/C的大多数真对应未落入局部窗口，粗正确比例也大幅下降；局部精化平均只能小幅改善，不能拯救错误粗窗口。部分对应会精化变差，逐pair保留worsened_fraction。几何内点覆盖也下降，错误或分布不足的对应可产生有矩阵但高误差的结果；求解失败为0不代表全部配准成功。', '',
        '## B按原始影像对汇总', '',
        '| pair_id | A原变换RMSE | B有效扰动数 | B SR@3 | B SR@5 | B求解失败 | B可求解RMSE均值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    csv_rows=[]
    for row in b:
        r=row['summary']['registration'];uid=row['pair_id'];ae=a_rows[uid]['evaluation']['rmse_px']
        lines.append(f"| {uid} | {ae:.4f}px | 5 | {r['success_rate@3px']:.0%} | {r['success_rate@5px']:.0%} | {r['failed_pairs']} | {r['solved_mean_rmse_px']:.4f}px |")
        csv_rows.append(dict(pair_id=uid,A_rmse_px=ae,B_variants=5,B_SR3=r['success_rate@3px'],B_SR5=r['success_rate@5px'],
            B_solver_failures=r['failed_pairs'],B_solved_mean_rmse_px=r['solved_mean_rmse_px']))
    lines.extend(['', '## 原始文件与边界', '',
        '- frozen_ABC_plans.json：B全部160个新变换及原始A/C计划；推理前保存。',
        '- group_A/B/C_pairs.json：每次扰动的全部分层指标、预测点、矩阵和内点索引。',
        '- group_B_per_original_image.json / group_B_per_original_image.csv：32对原影像的分组统计。',
        '- summary.json：状态核查、总体结果及启动T03的门禁。',
        '所有GT诊断只在预测与几何拟合后计算；没有把GT正确对应或GT内点重新送入RANSAC，没有使用scene控制点。'])
    (p/'受控诊断与分层报告.md').write_text('\n'.join(lines)+'\n')
    with (p/'group_B_per_original_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)


if __name__=='__main__':main()
