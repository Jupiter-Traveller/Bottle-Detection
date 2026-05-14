# 瓶体定位实验说明

本项目只保留一个训练入口 `train.py`。每次实验通过 `--name` 指定实验名，训练结果会自动保存到 `weights/<实验名>/`。

## 数据集脚本区别

`split_data.py` 负责第一次切分数据：

- 输入：`bottle_dataset/images/train` 和 `bottle_dataset/labels/train`
- 输出：`dataset/images/train`、`dataset/images/val`、`dataset/labels/train`、`dataset/labels/val`
- 默认按照 `80%/20%` 划分训练集和验证集
- 会按原始类别 `0/1/2/3` 分层切分，尽量保持各类水位比例稳定

`make_position_dataset.py` 负责生成单类瓶体定位数据集：

- 输入：已经切分好的 `dataset`
- 输出：`dataset_position`
- 不重新随机切分，而是沿用 `dataset` 里的 train/val 结构
- 会复制图片，并把每个标签的类别都改成 `0 bottle`
- 坐标不变，只改类别编号

推荐流程：

```bash
python3 split_data.py
python3 make_position_dataset.py
```

之后训练定位模型时，统一使用：

```text
dataset_position/bottle_data.yaml
```

## 数据量建议

题目要求不少于 100 张，但 100 张只能满足最低要求，不足以支撑泛化结论。建议最终定位数据集控制在 `800-1500` 张左右，优先保证质量和覆盖面，不盲目堆到几千张。

数据构成建议：

- 自拍数据：保留 300-500 张，保证贴合真实测试场景。
- 网络数据：补充 500-1000 张，重点增加复杂背景、不同瓶型、不同光照。
- 验证集：必须包含一部分复杂背景图片，不能全是单一背景。
- 剔除样本：多瓶同图、瓶体严重遮挡、非饮料瓶、标注不清、生成式图片。

如果明天加入网络数据，最终模型应重新统一切分再训练；现在已有数据可以先训练一个 baseline，用于检查流程和写“数据扩充前后对比”。

## 对照实验分类

建议把实验分成 5 类，不要把所有变量混在一起改。每次实验只改变一个主要因素，论文里才好解释。

### 1. 数据规模对照

用于回答“加入网络复杂背景数据是否提升泛化能力”。

| 实验名 | 数据集 | 模型 | 输入尺寸 | 增强 | 目的 |
| --- | --- | --- | ---: | --- | --- |
| `yolov8n_real_img640_base` | 仅自拍数据 | `yolov8n.pt` | 640 | base | 当前数据 baseline |
| `yolov8n_mixed_img640_base` | 自拍 + 网络数据 | `yolov8n.pt` | 640 | base | 最终主模型候选 |

### 2. 输入分辨率对照

用于回答“降低分辨率会不会影响定位精度，以及能否提升速度”。

| 实验名 | 数据集 | 模型 | 输入尺寸 | 增强 | 目的 |
| --- | --- | --- | ---: | --- | --- |
| `yolov8n_mixed_img640_base` | 自拍 + 网络数据 | `yolov8n.pt` | 640 | base | 精度基准 |
| `yolov8n_mixed_img512_base` | 自拍 + 网络数据 | `yolov8n.pt` | 512 | base | 平衡精度和速度 |
| `yolov8n_mixed_img416_base` | 自拍 + 网络数据 | `yolov8n.pt` | 416 | base | 更快推理，观察裁剪是否变差 |

### 3. 模型大小对照

用于回答“是否有必要从 YOLOv8n 换成更大的模型”。

| 实验名 | 数据集 | 模型 | 输入尺寸 | 增强 | 目的 |
| --- | --- | --- | ---: | --- | --- |
| `yolov8n_mixed_img640_base` | 自拍 + 网络数据 | `yolov8n.pt` | 640 | base | 轻量模型 |
| `yolov8s_mixed_img640_base` | 自拍 + 网络数据 | `yolov8s.pt` | 640 | base | 较大模型，对比精度提升是否值得 |

### 4. 数据增强对照

用于回答“物理合理的轻量增强是否提升复杂背景下的泛化能力”。

| 实验名 | 数据集 | 模型 | 输入尺寸 | 增强 | 目的 |
| --- | --- | --- | ---: | --- | --- |
| `yolov8n_mixed_img640_base` | 自拍 + 网络数据 | `yolov8n.pt` | 640 | base | 不使用几何增强 |
| `yolov8n_mixed_img640_lightaug` | 自拍 + 网络数据 | `yolov8n.pt` | 640 | light | 轻微旋转、平移、缩放、颜色变化 |

### 5. 置信度和裁剪参数对照

这个不需要重新训练，只用最终候选模型测试不同推理参数。

| 实验项 | 候选值 | 观察指标 |
| --- | --- | --- |
| 置信度阈值 `conf` | `0.15`、`0.25`、`0.35`、`0.50` | 漏检率、误检率、裁剪成功率 |
| 裁剪边距 `padding` | `0`、`0.03`、`0.05`、`0.08` | 是否截断瓶体、背景冗余是否过多 |

推荐最终只选 1 个主模型，例如：

```text
yolov8n_mixed_img640_base
```

如果轻量增强效果更好，再改选：

```text
yolov8n_mixed_img640_lightaug
```

## 训练命令

每次只运行其中一条。`--name` 后面的名字就是保存目录名。

```bash
python3 train.py --name yolov8n_real_img640_base --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16
python3 train.py --name yolov8n_mixed_img640_base --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16
python3 train.py --name yolov8n_mixed_img512_base --model yolov8n.pt --imgsz 512 --epochs 100 --batch 16
python3 train.py --name yolov8n_mixed_img416_base --model yolov8n.pt --imgsz 416 --epochs 100 --batch 16
python3 train.py --name yolov8s_mixed_img640_base --model yolov8s.pt --imgsz 640 --epochs 100 --batch 16
python3 train.py --name yolov8n_mixed_img640_lightaug --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16 --augment-preset light
```

`train.py` 默认不会覆盖已有实验目录。如果确认要继续写入已有目录，再加：

```bash
--exist-ok
```

## 数据增强建议

本项目只做瓶体定位，增强应服务于“真实拍摄下仍能框住瓶子”，不能制造明显不符合实际场景的样本。

当前 `train.py` 里有两个 preset：

- `base`：关闭旋转、缩放、平移、翻转、mosaic、mixup 等几何增强，用作干净基准。
- `light`：只做很轻的增强，包含 `degrees=3`、`translate=0.05`、`scale=0.2` 和轻微 HSV 变化。

不建议使用：

- 大角度旋转，例如 `degrees=10` 以上，除非测试集确实会出现明显歪斜的瓶子。
- 上下翻转 `flipud`，会产生倒置瓶子。
- 左右翻转 `fliplr`，如果瓶身标签、文字或光照方向有明显语义，会引入不真实样本。
- mosaic、mixup、copy-paste，因为题目设定通常是一张图一个瓶子，这些增强会制造不符合任务描述的组合图。

如果明天加入网上复杂背景数据，优先依靠真实复杂背景样本提升泛化，而不是靠强增强硬造背景。

## 结果文件

每个实验目录里会有论文需要的结果文件：

```text
weights/<实验名>/
├── args.yaml
├── results.csv
├── results.png
├── BoxF1_curve.png
├── BoxPR_curve.png
└── weights/
    ├── best.pt
    └── last.pt
```

主要看这些文件：

- `weights/best.pt`：验证集表现最好的模型，后续预测和裁剪优先用它
- `results.csv`：每轮训练指标，适合整理论文表格
- `results.png`：训练曲线图
- `BoxF1_curve.png`：选择置信度阈值的依据
- `BoxPR_curve.png`：精确率和召回率关系图

论文对比表建议从 `results.csv` 读取这些列：

- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

其中 `metrics/mAP50-95(B)` 更能反映定位框是否精细，建议作为主要对比指标。

## 裁剪时指定权重

默认裁剪使用：

```text
weights/yolov8n_img640_base/weights/best.pt
```

如果要换成其他实验的权重，用 `--weights` 指定：

```bash
python3 crop_bottle.py test.jpg --weights weights/yolov8n_mixed_img512_base/weights/best.pt
```

## 论文中本部分建议框架

你负责的是“瓶体区域自动定位与裁剪”，论文可以按下面结构写。

### 1. 问题建模

- 将原始图像中的瓶体定位问题建模为单目标检测问题。
- 输入为 RGB 图像，输出为瓶体边界框 `(x_center, y_center, width, height)`。
- 由于后续只需要瓶体区域，水位类别不参与定位模型训练，将原始多类别标签统一为单类 `bottle`。

### 2. 数据集构建

- 说明自拍数据和网络数据来源。
- 说明剔除规则：多瓶、遮挡严重、非饮料瓶、标注不清样本不进入训练集。
- 说明 `split_data.py` 负责 train/val 切分，`make_position_dataset.py` 负责单类标签转换。
- 给出训练集、验证集数量，以及背景、瓶型、光照、姿态覆盖情况。

### 3. 模型选择

- 说明选择 YOLOv8n 的原因：轻量、推理快、适合作为后续液体识别的前置裁剪模块。
- 简要对比 Faster R-CNN、SSD、YOLO 系列，强调本任务更关注快速稳定定位，不追求复杂两阶段检测器。
- 说明额外训练 YOLOv8s 用于验证“更大模型是否带来明显收益”。

### 4. 训练与参数设置

- 写明输入尺寸、训练轮数、batch size、优化器、预训练权重。
- 写明 `base` 和 `light` 两种增强策略。
- 强调增强不制造不符合物理场景的样本，例如不使用倒置、强旋转、mosaic、mixup。

### 5. 对照实验

- 数据规模对照：仅自拍数据 vs 自拍 + 网络数据。
- 分辨率对照：`640`、`512`、`416`。
- 模型大小对照：YOLOv8n vs YOLOv8s。
- 数据增强对照：base vs light。
- 推理参数对照：不同 `conf` 和 `padding` 对裁剪成功率的影响。

### 6. 评价指标

- 检测指标：Precision、Recall、mAP50、mAP50-95。
- 裁剪指标：漏检率、瓶体截断率、背景冗余情况、单张平均推理时间。
- 主指标建议使用 `mAP50-95` 和裁剪成功率，因为你的任务重点是位置框是否精确可用。

### 7. 结果分析

- 用表格列出各实验指标。
- 说明最终选择哪个模型和参数。
- 分析失败案例，例如透明边缘不清、强反光、复杂背景、瓶身标签干扰。
- 总结该定位模块如何为后续液体识别提供稳定输入。
