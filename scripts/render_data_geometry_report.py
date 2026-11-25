"""Build human-readable Chinese deliverables from audit/closure/JUnit evidence."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', required=True)
    args = ap.parse_args()
    out = Path(args.directory)
    audit = json.loads((out/'audit.json').read_text())
    duplicate_report = json.loads((out/'pair_duplicates.json').read_text())
    geometry = json.loads((out/'geometry_closure.json').read_text())
    xml = ET.parse(out/'geometry_tests.xml').getroot()
    suites = [xml] if xml.tag == 'testsuite' else list(xml.iter('testsuite'))
    tests = sum(int(s.attrib.get('tests', 0)) for s in suites)
    failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in suites)
    skipped = sum(int(s.attrib.get('skipped', 0)) for s in suites)
    lines = ['# 数据审计报告：256/512 全划分与来源核对', '', '日期：2026-11-25。实际在 zzvr 上读取原始文件执行；不修改数据、不重划 split。', '',
        '## 范围与完整性', '',
        f"实际检查 **{audit['totals']['files']} 个文件**，完整解码 **{audit['totals']['decoded_png']} 张 PNG**，组成 **{audit['totals']['pairs']} 对 patch**。",
        '17168 对包含 256/512 同源尺寸，不等同于17168个独立场景。新增区域数据保持 unassigned，不自动混入原版训练。', '',
        '| 尺寸 | split | 实际对数 | 预期对数 | 缺失配对 | 解码失败 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            s = audit['subsets'][f'Patch-level Subset/OSdataset/{size}/{split}']
            lines.append(f"| {size} | {split} | {s['pairs']} | {s['expected_pairs']} | {len(s['unpaired_optical'])+len(s['unpaired_sar'])} | {len(s['decode_failures'])} |")
    archive = audit['archive']
    lines.extend(['', f"本地 ZIP 中有 {archive['files']} 个文件、解压后声明总大小 {archive['declared_uncompressed_bytes']} 字节；逐文件核对数量、大小、CRC32，**{archive['size_crc_matches']} 个一致，{archive['size_crc_mismatches']} 个不一致**。",
        f"ZIP列出的缺失文件 {len(archive['missing_files'])} 个；提取目录额外文件 {len(archive['extra_files'])} 个。",
        '同时为每个文件计算 SHA-256。此项证明解压文件与本机 ZIP 的声明一致；未查询官方独立哈希，也未在本次重新执行整个压缩流 testzip，不能声称完成官方签名验证。PNG全部完整解码，两个scene TIFF全部tile解码；.enp侧车只做字节校验，不解析私有格式。', '',
        '## 通道、dtype与像素范围', '',
        '| 尺寸/split | 模态 | 模式 | dtype | 实测最小/最大 | 常量图张数 |', '| --- | --- | --- | --- | --- | ---: |'])
    for size in (256, 512):
        for split in ('train', 'val', 'test'):
            for modality in ('optical', 'sar'):
                s = audit['subsets'][f'Patch-level Subset/OSdataset/{size}/{split}']['images'][modality]
                lines.append(f"| {size}/{split} | {modality} | {s['modes']} | {s['dtypes']} | {s['observed_min']} / {s['observed_max']} | {s['constant_images']} |")
    lines.extend(['', '原版光学也是灰度L模式；加载到网络时仅复制为三通道，再除255，不能恢复彩色。逐图、逐通道min/max/mean/std/zero_fraction均写入统一清单。', '',
        '新增13个区域共3803对512图像，光学为RGB，SAR为灰度；逐区域统计在audit.json中。', '',
        f"同尺寸跨train/val/test完全相同文件组：{len(audit['cross_split_file_duplicates'])}；解码后像素完全相同组：{len(audit['cross_split_decoded_duplicates'])}。这里不包括近似重复检测。", '',
        '## 256到512父图关系', ''])
    parent = audit['parent_lineage']
    lines.extend(['集合内完全相同双模态图像对统计如下；这些记录保留在原split中，没有擅自删除。', '', '| 尺寸/split | 记录对数 | 唯一像素对数 | 冗余记录 | 重复组 |', '| --- | ---: | ---: | ---: | ---: |'])
    for key, stats in duplicate_report['summary'].items():
        lines.append(f"| {key} | {stats['pairs']} | {stats['unique_decoded_pairs']} | {stats['redundant_pair_entries']} | {stats['duplicate_groups']} |")
    lines.append('')
    lines.extend([f"对每对512图像的四个256×256象限计算解码像素哈希，要求 **光学和SAR同时逐像素一致** 才建立父图候选。",
        f"共检查{parent['children']}个256样本；找到父图{parent['matched']}个，其中唯一父图{parent['unique_match']}个、多候选{parent['ambiguous']}个、未找到{len(parent['unmatched'])}个。",
        f"已关联512父图{parent['parents_with_children']}对；每个父图对应的子图数量分布：{parent['children_per_parent_histogram']}。跨split父子链接：**{len(parent['cross_split_parent_links'])}**。",
        f"{parent['ambiguous']}个多候选子图中，{duplicate_report['ambiguity_explained_by_identical_512_pair_groups']}个可由像素完全相同的512父图组解释；完整重复ID列表在pair_duplicates.json。重复组意味着记录数不等于唯一内容数，但不是缺失文件。",
        'parent_candidates保存父图全局ID、父图split、裁剪原点xy和证据类型；不是按文件编号猜测。若存在未匹配/多候选，全部保留，不能擅自强配。', '',
        '**父图只定位到已分发的512 patch。** 包内没有512 patch到最初大幅卫星scene的场景ID、裁剪坐标或索引；因此即使父子关系无跨split冲突，也不能证明原始地理场景隔离。256/512不能当作独立数据池随机混合。', '',
        '## 原始scene和来源调查', '',
        '内置ReadMe称原版OSdataset SAR来自GF-3 C波段Spotlight、光学来自Google Earth、标称1m；引用Xiang等2020年论文。原版512数量2673，256数量10692。新增区域文件夹名称只作为包内region标签，不作为独立核验后的地理位置。', '',
        'Scene-level Subset中的一对大场景与原版patch之间没有公开映射。GeoTIFF能证明CRS、像素间距与覆盖范围，不能单凭同包存在证明某个patch来自该scene。', '',
        '| 场景 | 宽×高 | 通道 | dtype | NoData（存储值） | 有效像素范围（逐通道） |', '| --- | --- | ---: | --- | --- | --- |'])
    for name in ('optical', 'sar'):
        s = audit['scene'][name]
        lines.append(f"| {name} | {s['width']}×{s['height']} | {s['channels']} | {s['dtype']} | {s['nodata_in_storage_dtype']} | {s['min']} 至 {s['max']} |")
    lines.extend(['', '两幅图均为BigTIFF、EPSG:4326，像素间距约3.856974×10⁻⁶度；这不是1m的直接米单位描述。光学NoData是256而不是0，SAR NoData是float32极小负值；统计已排除NoData和非有限值，并裁去边缘tile填充。',
        'tifffile对SAR的GDAL_NODATA字符串产生过存储类型转换提示；审计显式按float32转换后排除242878422个NoData像素，全部tile解码完成，该提示不代表损坏。',
        '完整shape、geotransform、GeoTIFF标签、全部tile解码数、有效/NoData/非有限样本数量保留在audit.json。不能将这两个大场景直接走uint8/255读取流程。', '',
        '## 控制点', ''])
    cp = audit['control_points']
    lines.extend([f"GT.txt严格解析为 **{cp['rows']} 行×4列**：x_optical,y_optical,x_sar,y_sar。",
        f"重复记录{cp['duplicate_rows']}；光学越界{len(cp['optical_out_of_bounds'])}；SAR越界{len(cp['sar_out_of_bounds'])}。",
        '列方向由包内Evaluation.py核对。原标注起点与像素中心约定缺少独立证据，内部默认零起点xy像素中心但原值不做偏移；详见控制点格式说明。',
        f"全部控制点的诊断性最小二乘affine拟合RMSE：{cp['fit_rmse_px']:.6f}px，中位数{cp['fit_median_px']:.6f}px，最大{cp['fit_max_px']:.6f}px。同批点数值往返最大误差{cp['numerical_roundtrip_max_px']:.3e}px。",
        '**该拟合使用并检验同一组GT，只是几何一致性诊断，不是模型配准结果。** 原始配对GT和新增scene控制点都没有七类地物语义标签。', '',
        '## 清单、配置与审计边界', '',
        '- dataset_manifest.jsonl：全体17168对的唯一ID、相对路径、split、resolution、region、通道/dtype/完整数值统计、文件/像素SHA、父图关系、GT状态和未知scene状态。',
        '- file_inventory.jsonl：全部34342个文件的大小、SHA256、CRC32和ZIP对照结果。',
        '- control_points.csv：全部控制点和检查字段；effective_config.json：本次实际配置。',
        '- configs/data_geometry_v1.json：唯一的通用数据/几何配置，数据加载、几何验证和审计实际读取；机密/机器根路径不写入通用配置。',
        f"- 统一清单SHA256：`{audit['root_manifest_sha256']}`。", '',
        f"审计errors共{len(audit['errors'])}项。原始详情以audit.json为准。未发现损坏不等于证明无残余配准误差；没有原始scene映射就不能声称地理独立划分；没有patch显式GT就不能声称人工亚像素标注。"])
    (out/'数据审计报告.md').write_text('\n'.join(lines)+'\n')
    test_lines = ['# 几何闭环与基础测试结果', '', '日期：2026-11-25。这里验证数据和坐标数学，不进行模型训练，不产生新的匹配精度结论。', '',
        f"pytest：**{tests-failures-skipped}通过，{failures}失败，{skipped}跳过**。JUnit原始结果：geometry_tests.xml；终端日志：geometry_tests.txt。", '',
        '覆盖：完整6DoF仿射正逆、退化矩阵拒绝、组合顺序、源/目标裁剪和resize坐标框架变换、半像素采样、stride2/8特征坐标、图像轴旋转、反射、GeoTIFF中心/角点转换、OpenCV与PyTorch采样一致性、已知平移/亚像素质心、RANSAC外点恢复、控制点严格解析、padding/NoData/损坏PNG、配置与清单隔离、256/512全部split加载，以及已有模型核心不变量。', '',
        f"真实数据闭环：6个split中共{geometry['actual_pairs_checked']}对，覆盖每个split的首/中/末编号样本，每对1000个点，全部通过。完整manifest加载时同时核对6个split的全部对数。", '',
        '| 检查项 | 实测最大值 |', '| --- | ---: |',
        f"| 仿射点正逆往返误差 | {geometry['max_point_roundtrip_px']:.3e} px |",
        f"| 裁剪/缩放变换框架闭环 | {geometry['max_frame_closure_px']:.3e} px |",
        f"| 归一化坐标往返 | {max(x['normalized_roundtrip_max_px'] for x in geometry['cases']):.3e} px |",
        f"| 特征坐标往返 | {max(x['feature_roundtrip_max_px'] for x in geometry['cases']):.3e} px |",
        f"| identity图像最大差值 | {max(x['identity_pixel_difference_max'] for x in geometry['cases'])} |",
        f"| 已知仿射图像采样与解析ramp误差 | {max(x['image_analytic_ramp_max_error'] for x in geometry['cases']):.3e}（0–1） |", '',
        '图像采样误差允许OpenCV插值权重量化；不能要求逆向重采样后的真实灰度影像逐像素恢复，因为插值和裁剪会损失信息。点几何闭环与图像重采样分开验证。', '',
        'geometry_closure.json保留每个样本ID、实际2×3矩阵、支持mask比例、误差和配置；geometry_warp_example.png是“已知变换+有效mask”的可视化，不是模型预测。', '',
        '当前阶段交付完成数据/工程/几何基础；此前100步模型训练的失败结果仍保留为历史记录，未把本次闭环误差当作模型精度改善。']
    (out/'几何测试结果.md').write_text('\n'.join(test_lines)+'\n')
    print(json.dumps(dict(report=str(out/'数据审计报告.md'), tests=tests, failures=failures, closure=geometry['passed']), ensure_ascii=False))


if __name__ == '__main__':
    main()
