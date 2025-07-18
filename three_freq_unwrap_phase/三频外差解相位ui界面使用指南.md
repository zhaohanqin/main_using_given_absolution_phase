# 三频外差法相位解包裹程序使用指南

## 1. 软件简介

三频外差法相位解包裹程序是一个基于PySide6构建的图形界面工具，专为三频外差法相位解包裹算法设计。该程序支持处理水平和垂直方向的相位解包裹，可以将包裹相位图转换为连续的绝对相位，以便用于三维重建和结构光扫描等应用。

### 主要功能

- 支持读取三个不同频率的相移图像（高频、中频、低频）
- 支持同时处理水平和垂直两个方向的相位解包裹
- 交互式查看解包裹后的相位图，包括2D和3D可视化
- 提供组合视图窗口，显示水平和垂直方向的相位信息
- 实时显示鼠标位置对应的相位值和条纹序数
- 保存多种格式的结果，包括原始相位数据(.tiff)、伪彩色可视化图(.png)和3D可视化效果图

## 2. 图像输入要求

### 图像文件命名规则

程序要求图像文件按照特定的规则命名，以便自动识别相移序列。命名格式应为：

```bash
I<序号>.<扩展名>
```

其中：

- `<序号>` 是从1开始的整数，表示图像在相移序列中的位置
- `<扩展名>` 可以是常见图像格式，如 .png, .jpg, .tif 等

### 图像组织方式

对于每个频率，程序假设：

- 序号1到N的图像用于水平方向的相位计算（N为相移步数，默认为4）
- 序号N+1到2N的图像用于垂直方向的相位计算

例如，使用4步相移时：

- I1.png, I2.png, I3.png, I4.png 为水平方向的相移图像
- I5.png, I6.png, I7.png, I8.png 为垂直方向的相移图像

每个频率（高频、中频、低频）都需要遵循这个规则，因此一个完整的数据集通常包含24张图像（3个频率 × 2个方向 × 4步相移）。

### 图像加载顺序和处理逻辑

程序在处理时会将图像按以下顺序组织:

1. 垂直方向的高频相移图像（4张）
2. 垂直方向的中频相移图像（4张）
3. 垂直方向的低频相移图像（4张）
4. 水平方向的高频相移图像（4张）
5. 水平方向的中频相移图像（4张）
6. 水平方向的低频相移图像（4张）

这与 `get_abs_phase.py` 中的 `multi_phase` 类所需的输入格式一致。程序会自动组织这些图像以正确传递给相位解包裹算法。

## 3. 操作流程

### 步骤1: 设置参数

1. **相移步数**: 设置每个频率和方向使用的相移步数（通常为4）
2. **初始相位偏移**: 设置相移序列的初始相位偏移（通常为0.5）
3. **频率设置**: 分别设置高频、中频和低频的频率值
4. **滤波设置**: 设置用于平滑相位图的高斯滤波器大小

#### 初始相位偏移详解

初始相位偏移(ph0)是相位解包裹中的一个重要参数，它直接影响相位计算的准确性：

- **物理含义**：表示投影或获取相移图像时的起始相位，反映了图案投影系统和图像采集系统之间的相位关系。
  
- **取值范围**：通常介于0.0到1.0之间，在UI中可以通过滑块精确调整。

- **默认值**：程序默认值为0.5，这对应于相移步长为π/2时的常见设置。在N=4步相移中，0.5的初始相位偏移意味着相位从中间位置开始。

- **调整建议**：
  - 对于同一次采集的所有图像（所有频率的水平和垂直图像），应使用相同的初始相位偏移值
  - 如果解包裹结果不理想，可以尝试微调此值（例如0.45-0.55之间）
  - 理想情况下，应通过校准过程确定最佳值，例如使用已知几何形状的平面作为参考

- **影响**：不正确的初始相位偏移会导致：
  - 相位计算偏差
  - 条纹序数错误
  - 解包裹结果中出现系统性波纹或倾斜

在实际使用中，一旦为特定的投影-相机系统确定了最佳的初始相位偏移值，建议记录并始终使用该值，除非硬件设置发生变化。

### 步骤2: 加载图像

为每个频率（高频、中频、低频）选择对应的图像文件夹：

1. 点击"高频(F1)"、"中频(F2)"和"低频(F3)"对应的"选择文件夹"按钮
2. 在弹出的对话框中选择包含相应频率相移图像的文件夹
3. 程序会自动扫描文件夹，根据命名规则识别水平和垂直方向的相移图像
4. 加载成功后，按钮文本会更新为"水平+垂直 (8张)"或相应的图像数量

### 步骤3: 设置解包裹方向

选择需要处理的解包裹方向：

- **仅水平方向**: 只处理水平方向的相位解包裹
- **仅垂直方向**: 只处理垂直方向的相位解包裹
- **两个方向**: 同时处理水平和垂直方向的相位解包裹（默认选项）

### 步骤4: 设置输出目录

点击"选择..."按钮，指定处理结果的保存位置。

### 步骤5: 开始处理

点击"开始处理"按钮，程序会:

1. 加载并组织所有相移图像
2. 应用三频外差法算法进行相位解包裹
3. 保存处理结果到指定目录
4. 在界面中显示解包裹后的相位图
5. 如果处理了两个方向，会自动打开组合相位图窗口

## 4. 结果解释

### 输出文件

处理完成后，程序会在指定的输出目录生成以下文件：

- **unwrapped_phase_horizontal.tiff**: 水平方向的原始解包裹相位数据
- **unwrapped_phase_vertical.tiff**: 垂直方向的原始解包裹相位数据
- **phase_quality_horizontal.tiff**: 水平方向的相位质量图
- **phase_quality_vertical.tiff**: 垂直方向的相位质量图
- **unwrapped_phase_horizontal_2d.png**: 水平方向的伪彩色2D相位图
- **unwrapped_phase_vertical_2d.png**: 垂直方向的伪彩色2D相位图
- **unwrapped_phase_horizontal_3d.png**: 水平方向的3D相位图
- **unwrapped_phase_vertical_3d.png**: 垂直方向的3D相位图
- **combined_phase_map.png**: 水平和垂直相位的组合图（红=水平，绿=垂直）
- **image_info.log**: 处理过程中的图像信息日志

### 交互式查看

程序提供以下交互功能：

1. **主界面相位查看器**:
   - 在2D视图中移动鼠标，会显示对应位置的坐标和相位值
   - 可以切换到3D视图查看相位的三维表示

2. **组合相位查看器窗口**:
   - 显示水平和垂直方向相位的组合图（红=水平，绿=垂直）
   - 鼠标悬停时显示十字准星，并在底部显示详细信息
   - 显示当前位置的水平和垂直相位值以及对应的条纹序数

## 5. 三频外差法原理简介

三频外差法相位解包裹算法利用三个不同频率的相移图像，通过比较不同频率的包裹相位差异来确定条纹序数，从而实现相位的解包裹。

### 算法流程

1. 通过相移图像计算每个频率的包裹相位（范围为[-π, π]）
2. 利用高频与中频、中频与低频之间的相位差来计算条纹序数
3. 使用条纹序数将包裹相位转换为连续的绝对相位

本程序实现的三频外差法基于中频相位的解包裹，即最终的绝对相位是基于中频计算的。算法利用两种不同的解包裹路径（一种通过高频，一种通过低频）来提高解包裹的准确性和鲁棒性。

### 特别说明

程序采用的数据处理流程为：

1. 加载三个频率各自的水平和垂直相移图像（共24张）
2. 按照算法所需的特定顺序组织这些图像
3. 调用 `multi_phase` 类进行相位解包裹处理
4. 返回水平和垂直方向的解包裹相位和相位质量图

## 6. 常见问题

### 无法加载图像

- 确保图像按照正确的命名规则（I1.png, I2.png等）
- 检查图像格式是否支持（建议使用PNG或TIFF格式）
- 验证每个频率文件夹中包含足够数量的图像（通常为8张，4张水平+4张垂直）

### 处理结果有噪声或不准确

- 尝试调整滤波器大小以减少噪声
- 检查相移步数设置是否与实际采集的相移步数一致
- 验证频率值设置是否与实际投影的条纹频率一致
- 确保选择了正确的频率组合（高频>中频>低频）

### 无法显示组合相位图

- 确保同时处理了水平和垂直两个方向
- 检查两个方向的图像尺寸是否一致

## 7. 技术细节

### 依赖库

- PySide6: 用于构建图形界面
- OpenCV: 用于图像处理和保存
- NumPy: 用于数值计算
- Matplotlib: 用于绘制2D和3D可视化图

### 核心文件

- **three_freq_unwrap_phase_ui.py**: 主界面和交互逻辑
- **get_abs_phase.py**: 三频外差法相位解包裹算法实现

### 关键参数解释

- **频率值(F1, F2, F3)**：这些值应与投影图案的实际条纹频率匹配，按从高到低排序。频率值之间的差异会影响解包裹的鲁棒性。

- **初始相位偏移(ph0)**：如前所述，这影响相位计算过程。在`get_abs_phase.py`中，包裹相位通过以下公式归一化并应用初始相位偏移：

  ```python
  result = (result+np.pi)/(2*np.pi)-self.ph0
  ```
  
  这将arctan2计算得到的[-π, π]范围相位归一化到[0, 1]区间，然后减去初始相位偏移。
  
- **相移步数(N)**：定义每个频率每个方向使用的相移图像数量。增加步数可以提高抗噪性能，但会增加采集时间。

- **滤波器尺寸**：控制用于平滑相位图的高斯滤波器大小。较大的值提供更强的平滑效果，但可能丢失细节。
