### 项目简介

瓶子液体残留检测全流程：检测瓶身位置 → 判断是否有液体残留 → 识别液位高低（low/medium/high）。

---

### 一、运行说明
1. 安装依赖：`pip install -r requirements.txt`
2. 推理测试：`python3 predict_all.py --image "图片途径"`
3. 保存剪裁：`python3 predict_all.py --image "图片路径" --save-crop "保存路径"`

### 二、文件结构

```
support_materials
├── code_appendix
│   ├── A_liquid_presence              # 液体残留检测（有/无液体）
│   ├── B_water_level_classification   # 水位分类
│   │   ├── 3_classification           # 3分类（low/medium/high）
│   │   └── 4_classification           # 4分类（empty/low/medium/high）
│   └── C_bottle_detection             # 瓶身位置识别
├── examples                           # 模型输出样例
├── predict_all.py                     # 总推理脚本
├── src                                # 推理模块
├── weights                            # 模型权重参数
├── dataset.zip                        # 数据集压缩包
└── requirements.txt                   # Python 依赖

```

#### 1.权重参数（最终结果）
```
weights
├── efficientnet_b0_img320_color_best.pt   # 水位3分类模型（EfficientNet-B0）
├── two_class_best_model.joblib            # 液体二分类模型（传统特征+sklearn）
├── two_class_scaler.joblib                # 二分类特征标准化器
└── yolov8n_img512_light_best.pt           # 瓶身检测器权重（YOLOv8-nano）
```

---

### 2.附录（训练过程文件）
```
code_appendix
├── A_liquid_presence
│   ├── config.py                          # 全局配置（路径、随机种子等）
│   ├── dataset.py                         # 数据加载（读CSV、校验标签、bbox裁剪）
│   ├── features_deep.py                   # 深度特征提取（ResNet18/50 embedding）
│   ├── features_traditional.py            # 传统特征提取（HSV/灰度/纹理/边缘/液面线）
│   ├── predict_image.py                   # 单图二分类推理（有/无液体）
│   ├── preprocess.py                      # 图像预处理（读图/裁剪/缩放/保存）
│   ├── run_pretest.py                     # 二分类训练CLI入口
│   ├── train_eval.py                      # 二分类训练与评估流程（LR/SVM/RF）
│   └── visualize.py                       # 可视化（混淆矩阵/PCA/液面线图）
├── B_water_level_classification
│   ├── 3_classification
│   │   ├── predict_level.py               # YOLOv8-cls 推理
│   │   ├── predict_level_torch.py         # PyTorch模型 3分类推理
│   │   ├── prepare_level_dataset.py       # 构建3分类数据集
│   │   ├── split_data.py                  # 数据集切分
│   │   ├── train_level_cls.py             # YOLOv8-cls 训练
│   │   ├── train_level_efficientnet.py    # EfficientNet-B0 训练
│   │   └── train_level_mobilenet.py       # MobileNetV3-Small 训练
│   └── 4_classification
│       ├── config.py                      # 全局配置（路径、输出目录等）
│       ├── features_traditional.py        # 传统特征提取
│       ├── liquid_level_train_eval.py     # 4分类训练与评估（LR/SVM/RF）
│       ├── predict_liquid_level.py        # 单图4分类推理
│       └── preprocess.py                  # 图像预处理
└── C_bottle_detection
    ├── benchmark_detector.py              # 检测推理速度
    ├── crop_bottle.py                     # 裁剪瓶身
    ├── make_position_dataset.py           # 构建瓶身检测数据集
    ├── predict.py                         # 瓶身位置推理
    ├── split_data.py                      # 数据集切分
    └── train.py                           # YOLOv8瓶身检测训练
```