# 饮料瓶瓶体定位项目

本项目负责完成“原始饮料瓶图片 -> 瓶体位置框 -> 瓶体裁剪图”的前置定位流程。当前重点不是判断瓶内液体多少，而是稳定找到瓶体区域，为后续液体识别或水位分类提供裁剪后的输入。

## 任务目标

- 输入：一张包含单个饮料瓶的图片。
- 输出：瓶体边界框坐标 `[x1, y1, x2, y2]`，或 YOLO 归一化坐标 `(x_center, y_center, width, height)`。
- 训练方式：先从原始 4 类水位标签生成单类定位标签，再训练单类 `bottle` 检测模型。
- 模型用途：部署时主要读取预测框位置，并用 `crop_bottle.py` 裁剪瓶体区域。

原始标签中的类别含义如下：

```text
0 Empty
1 Liquid_Low
2 Liquid_Medium
3 Liquid_High
```

定位训练时会把这些类别统一转换为：

```text
0 bottle
```

坐标不变，只改变类别编号。

## 当前目录结构

```text
Bottle_Detection/
├── bottle_dataset/                 # 原始完整数据源，切分前
│   ├── images/train/               # 原始图片
│   └── labels/train/               # 原始 YOLO 标签
├── dataset/                        # split_data.py 生成的 4 类切分数据集
│   ├── bottle_data.yaml
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── dataset_position/               # make_position_dataset.py 生成的单类定位数据集
│   ├── bottle_data.yaml
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── cropped/                        # crop_bottle.py 默认裁剪输出目录
├── weights/                        # 训练输出目录
│   ├── bottle_detector/            # 旧的 4 类模型结果
│   └── yolov8n_img640_base/        # 当前已有的瓶体定位基准实验
├── split_data.py                   # 从 bottle_dataset 切分生成 dataset
├── make_position_dataset.py        # 从 dataset 生成 dataset_position，并把类别统一为 0
├── train.py                        # 统一训练入口，通过参数指定实验名、模型和分辨率
├── predict.py                      # 输出预测框坐标
├── crop_bottle.py                  # 根据检测框裁剪瓶体图片
├── EXPERIMENTS.md                  # 对照实验计划、命令和论文建议
└── yolov8n.pt                      # YOLOv8n 预训练权重
```

说明：如果当前目录中没有 `dataset_position/`，先运行 `python3 make_position_dataset.py` 生成。

## 数据处理流程

### 1. 切分原始 4 类数据

原始图片和标签放在：

```text
bottle_dataset/images/train
bottle_dataset/labels/train
```

运行：

```bash
python3 split_data.py
```

输出：

```text
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

`split_data.py` 默认按 `80%/20%` 划分训练集和验证集，并按原始类别 `0/1/2/3` 分层切分。

### 2. 生成单类瓶体定位数据集

运行：

```bash
python3 make_position_dataset.py
```

输出：

```text
dataset_position/images/train
dataset_position/images/val
dataset_position/labels/train
dataset_position/labels/val
dataset_position/bottle_data.yaml
```

这个脚本不会重新随机切分，而是沿用 `dataset/` 里已有的 train/val 结构，只把所有标签类别统一改成 `0 bottle`。

## 模型训练

统一使用 `train.py`，不要复制多个训练脚本。每次实验通过 `--name` 指定保存目录。

基准训练命令：

```bash
python3 train.py --name yolov8n_img640_base --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16
```

训练结果保存到：

```text
weights/yolov8n_img640_base/
├── args.yaml
├── results.csv
├── results.png
└── weights/
    ├── best.pt
    └── last.pt
```

注意：

- `train.py` 默认使用 `dataset_position/bottle_data.yaml`。
- `--name` 是实验名，也是 `weights/` 下的输出目录名。
- 默认不会覆盖已有实验目录；如果确认要继续写入已有目录，才加 `--exist-ok`。
- 对照实验命令和论文实验设计见 `EXPERIMENTS.md`。

## 预测与裁剪

### 输出预测框

```bash
python3 predict.py test.jpg --weights weights/yolov8n_img640_base/weights/best.pt
```

输出格式：

```text
class_id x_center y_center width height
```

其中坐标为归一化坐标。

### 裁剪瓶体区域

使用默认基准权重：

```bash
python3 crop_bottle.py test.jpg
```

指定某次实验的权重：

```bash
python3 crop_bottle.py test.jpg --weights weights/yolov8n_mixed_img512_base/weights/best.pt
```

增加裁剪边距，避免框略小导致截断瓶体：

```bash
python3 crop_bottle.py test.jpg --padding 0.05
```

默认输出到：

```text
cropped/test_crop.jpg
```

## 对照实验

建议按下面维度做对照，不要一次改变多个因素：

- 数据规模：仅自拍数据 vs 自拍 + 网络数据。
- 输入分辨率：`640`、`512`、`416`。
- 模型大小：YOLOv8n vs YOLOv8s。
- 数据增强：`base` vs `light`。
- 推理参数：不同 `conf` 和 `padding` 对裁剪成功率的影响。

详细实验命名、训练命令、指标表和论文框架见：

```text
EXPERIMENTS.md
```

## 主要评价指标

检测指标：

- Precision
- Recall
- mAP50
- mAP50-95

裁剪指标：

- 漏检率
- 瓶体截断率
- 背景冗余情况
- 单张平均推理时间

本任务重点是“框是否足够准，裁剪后能否用于后续识别”，因此论文中建议重点分析 `mAP50-95` 和裁剪成功率。
