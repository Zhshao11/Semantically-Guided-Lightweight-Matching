"""Describe a read-only observation snapshot; does not control running training."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args();out=Path(a.directory)
    s=json.loads((out/'live_snapshot.json').read_text())
    when=datetime.fromtimestamp(s['snapshot_unix'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z')
    lines=['# T03全量B0-Full协议与运行观察', '', f'观察时间：{when}。这是运行中快照，20个epoch为首轮预算，尚不能据此声称收敛。', '',
        '## 已完成的门禁', '',
        'T02原检查点A/C的预测、矩阵、误差逐项复现；208次受控评价全程eval，正确使用光学BN0/SAR BN1，BN缓冲区逐位不变。A/B/C和全量验证沿用相同归一化、mask及坐标解码。全仓库57项测试通过。', '',
        '## 固定协议', '',
        '- train：7912对去重256影像；每次presentation在线新采样，显式draw_token与增强seed保证对照、恢复可复核。',
        '- val：全部952对，每对一个提前保存的矩阵；保留原T02的16个val矩阵，预检确认其图像、A和mask逐位相同。',
        '- 模型：T02 B0-Full，37219参数；从头初始化，两运行初始状态SHA相同，并等于T02的初始随机权重，不加载T02训练权重。',
        '- 只比较lr：主运行0.001，对照0.0002；两者AdamW、WD=0.0001、batch4、BF16、clip5、loss、fine bootstrap400步、样本顺序和增强流一致，无scheduler。',
        '- 20epoch/39560步；epoch1与2/4/.../20完整验证；best预先按SR@5px严格改善保存，相同分数保留较早版本；last独立保存。',
        '- 不加载test图像，不使用scene控制点评价选模，不切换train来改善验证成绩；每次验证检查BN不更新。', '',
        '## 当前后台运行', '', '| 运行 | GPU | PID | phase | epoch | step | last/best |', '| --- | --- | ---: | --- | ---: | ---: | --- |']
    for label,run in s['runs'].items():
        st,env=run['status'],run['environment']
        lines.append(f"| {label} | {env['physical_visible_devices']} | {st['pid']} | {st['phase']} | {st['epoch']} | {st['step']} | {run['checkpoint_exists']['last.pt']}/{run['checkpoint_exists']['best.pt']} |")
    lines.extend(['', '## 已完成的952对完整验证', '',
        '| 运行 | epoch | SR@3px | SR@5px | 求解失败 | 可求解RMSE均值 | 5px内成功样本RMSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for label,run in s['runs'].items():
        for row in run['validation_summaries']:
            lines.append(f"| {label} | {row['epoch']} | {row['success_rate@3px']:.3%} | {row['success_rate@5px']:.3%} | {row['failed_pairs']}/952 | {row['solved_mean_rmse_px']:.5f}px | {row['successful_mean_rmse@5px']:.5f}px |")
    lines.extend(['', '第一轮结果只作异常检查，不能据此提前判定某个学习率最终更好；两者按相同预算继续观察。失败一直计入SR分母，可求解均值与成功条件均值分开。', '',
        '## 实际控制一致性', '',
        f"初始化状态一致：{s['initialization_states_equal']}。完成并可比的epoch增强/样本流SHA检查：", ''])
    for row in s['shared_epoch_stream_checks']:
        lines.append(f"- epoch {row['epoch']}：matched={row['matched']}，`{row['sha256']}`。")
    lines.extend(['', '## 服务器位置与后续观察', ''])
    for label,run in s['runs'].items():
        lines.append(f"- {label}：`/mnt/sda/zhushihao/Semantically-Guided-Lightweight-Matching/{run['run_directory']}`。")
    lines.extend(['', '每个run内的status.json为实时状态；epoch_summaries.jsonl为逐epoch训练汇总，validation_summaries.jsonl为完整验证汇总，validation_epoch_*.json为952对逐pair结果，last.pt/best.pt为检查点。首次加载和验证阶段可能耗时，状态文件会记录phase。',
        '恢复使用--resume last.pt，从最后完成epoch继续；若上次保存后尚未完成验证，会先补验证。20epoch预算结束后应检查完整曲线和误差分布，再决定是否延长，不将固定轮数完成当成收敛。', '',
        '本地live_snapshot.json只是上述观察时刻的记录，不会自行更新；服务器训练进程继续按已实现的观察节点验证和保存。'])
    (out/'T03协议与运行观察.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
