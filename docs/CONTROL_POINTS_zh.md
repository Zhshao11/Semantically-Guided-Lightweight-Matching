# Scene 控制点格式和坐标约定

## 原始文件

`OSDataset2.0/Scene-level Subset/GT.txt` 是独立大场景的参考对应点。每个非空行必须恰好有四个有限浮点数：

```text
x_optical  y_optical  x_sar  y_sar
```

列方向由包内 `Scene-level Subset/code/Evaluation.py` 的 `x1_g,y1_g` 和 `x2_g,y2_g` 变量确认：前两列属于光学参考图，后两列属于 SAR 待配准图。`x` 是列，`y` 是行，不是 `(row,col)`，也不是经纬度。读取时不取整，不改写原始文件。

完整审计的 CSV 保留全部 72 点，增加一基 `point_id`（只是记录序号）、两幅影像边界检查和 diagnostic affine 拟合残差。此 point_id 不改变坐标索引。

## 统一内部约定

- 图像原点左上角，x 向右，y 向下。
- 项目内部采用零起点像素中心坐标：第一个像素中心 `(0,0)`，合法中心边界 `[0,W-1]×[0,H-1]`。
- 原脚本说明零起点，但同时允许 offset_mode；包内没有提供标注软件和原始标注索引证据。因此 **GT 的原始零/一起点及中心/角点约定没有被独立证明**。本项目默认沿用零起点，完整保留原值，不能据此声称亚像素绝对精度已核实。
- 原文脚本 `offset_mode=none` 不等于已经证明没有半像素问题。未来若取得原始标注信息，应显式记录转换，不能通过看测试误差来选择偏移。

## 变换方向

几何库所有仿射矩阵均为源→目标。若源是光学、目标是 SAR：

```text
[x_sar]   [a00 a01 tx] [x_optical]
[y_sar] = [a10 a11 ty] [y_optical]
                        [1]
```

文件保存为 2×3，齐次扩展为底行 `[0,0,1]` 的 3×3。反向注册 SAR→光学必须先求逆；不能将 optical→SAR 矩阵直接当反向矩阵使用。

`compose_affines(A,B)` 表示先 B、再 A。裁剪窗口原点 `(ox,oy)` 对应 `crop_to_parent_affine([ox,oy])`，从 patch 坐标加偏移到父图。若源、目标分别改用坐标变换 Ts、Tt，则新变换为：

```text
A_new = Tt @ A_old @ inverse(Ts)
```

## 分辨率与采样

半像素 resize 约定：`x_new=(x_old+0.5)*W_new/W_old-0.5`。256 patch 若是 512 的无缩放裁块，使用 crop 平移，不能误用 resize；只有像素精确匹配才记录该父图关系。

特征 stride s：`feature=(pixel+0.5)/s-0.5`；反变换为 `pixel=(feature+0.5)*s-0.5`。

PyTorch `align_corners=False`：`grid_x=2*(x+0.5)/W-1`；y 同理。OpenCV warpAffine 在未指定 `WARP_INVERSE_MAP` 时接收源→目标矩阵，内部进行逆采样。

正旋转角按图像坐标从+x转向+y，视觉上为顺时针。此方向与 `cv2.getRotationMatrix2D` 的正角约定相反，本项目不混用两者。

## GeoTIFF 世界坐标

大场景 CRS 为 WGS84 / EPSG:4326；GeoTIFF PixelIsArea 的 tiepoint 定义在像素角点。对项目内部像素中心 `(x,y)`，先加 `(0.5,0.5)`，再应用 GDAL 六参数 geotransform。世界坐标在此为 `(longitude,latitude)`，单位为度，不是米。

`pixel_to_world` 和 `world_to_pixel` 实现此明确约定，支持旋转 geotransform。若单独调用原始 GDAL 像素角点接口，不应再偷偷加减第二次0.5。

大场景 optical 为 uint16 三通道，SAR 为 float32，均有 NoData；它们不适用 patch 的 `uint8/255` 规则。包内 Evaluation.py 假定存在 GeoTIFF 地理信息，通用 PIL 可能无法读取 BigTIFF；审计使用 tifffile 分块解码。

## 使用边界

这 72 点只用于 scene 的数据格式、边界和几何诊断，以及未来独立评估。它们没有映射到原版 patch ID，不能直接作为某个 patch 的 GT。用全部点拟合后在同一批点上算残差只是 **拟合诊断**，不是模型预测精度、独立验证或泛化结果。几何正反变换的近零闭环误差也不等于影像配准成功。
