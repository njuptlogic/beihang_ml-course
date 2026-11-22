# 实验三：多尺度视觉表征与人眼注视密度预测

- **姓名 / 学号**：徐基恒 / `SY2631122`
- **硬件与环境**：Python 3.11、PyTorch 2.6.0+cu124、NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），Conda 环境 `ml-course`
- **核心架构图**：[ccf_architecture.svg](figures/ccf_architecture.svg) · [ccf_architecture.png](figures/ccf_architecture.png)
- **多尺度特征演化图**：[四类场景 DINOv2 粗/细尺度特征与注视热图叠加对照](figures/multiscale_stage_progression.png)

## 1. 方法创新与核心结果

在课程注视预测数据集（拟合 1,360 / 验证 240 / 测试 400 张 1,920×1,080 刺激图，20 类场景）上，仅凭训练集平均注视先验即可获得 0.7834 CC。本实验通过 **三尺度独立重编码（224×126 / 448×252 / 672×378 共享冻结 DINOv2-S/14-reg4）**、**DINO/PE-Spatial-T16 双编码器 + 浅层 ResNet 纹理引导 3×3 局部重采样 + 场景自适应三尺度扩散**，以及验证集选定的 **45% 金字塔分支 + 45% 双编码器分支 + 10% 未校准原始 UNISAL** 概率空间融合，显著提升了非中央目标与多峰场景的注视定位精度。

| 模型阶段 / 消融配置（400 张原尺寸测试图） | CC ↑ | KL ↓ | SIM ↑ |
|---|---:|---:|---:|
| 共享位置先验基准（0 参数） | 0.7834 | 0.5152 | 0.6413 |
| 第一阶段：ResNet 多尺度 + 先验 | 0.8609 | 0.3173 | 0.7238 |
| 第二阶段：DINOv2 中尺度 + 浅层纹理 | 0.8910 | 0.2641 | 0.7504 |
| 消融：DINOv2 中尺度单路 | 0.8937 | 0.2596 | 0.7542 |
| 消融：DINOv2 三尺度金字塔 | 0.8989 | 0.2511 | 0.7596 |
| 消融：DINOv2 三尺度 + 3×3 纹理引导局部重采样 | 0.9005 | 0.2474 | 0.7618 |
| 消融：DINO/PE 双编码器 + 局部重采样 + 场景扩散 | 0.9002 | 0.2463 | 0.7624 |
| 纯课程双分支等权融合（50% + 50%） | 0.90206 | **0.24435** | **0.76352** |
| **最终选定配置（45% 金字塔 + 45% 双编码器 + 10% 原始 UNISAL）** | **0.90210** | **0.24551** | **0.76056** |

## 2. 环境准备

以下所有命令均从**项目根目录（`ml-course/`）**使用相对路径运行：

```bash
conda activate ml-course
```

## 3. 一键复现六个课程读出头训练与原图评测

```bash
# 1) 复用冻结训练特征，以种子 17/42/73 重训三尺度金字塔与双编码器共 6 个读出头（每头 18 轮），
#    并在 400 张 1,920×1,080 原始测试图上以双精度复测 CC / KL / SIM（RTX 3070 Ti 约 7 分钟）
python -m course.report_replay 3 --output experiment3/outputs/fresh_gaze_run

# 2) 基于独立复现结果重绘 20 类箱线图与训练曲线
python -m course.report_figures 3 --input-dir experiment3/outputs/fresh_gaze_run

# 3) 重绘典型退步/增益热图对比案例板
python -m course.report_cases 3 --input-dir experiment3/outputs/fresh_gaze_run
```

## 4. 全流程模块化复现与单图推理

```bash
# 1) 下载与预处理数据及基础 ResNet 浅层纹理特征
python -m course.download 3
python -m course.prepare 3
python -m course.features 3

# 2) 提取三尺度 DINOv2（224/448/672）与 PE-Spatial-T16 / UNISAL 冻结表征
python -m experiment3.multiscale_features

# 3) 训练全部消融分支（9 种架构 × 3 种子 = 27 个读出头）并搜索验证集融合权重
python -m experiment3.multiscale_train
python -m experiment3.multiscale_evaluate
python -m experiment3.multiscale_figures

# 4) 单张场景图像注视密度预测（使用相对路径，输出与原图同宽高且像素和为 1 的 .npy 密度图）
python -m experiment3.multiscale_predict --image ./sample_scene.jpg --output ./outputs/gaze_pred.npy
```

## 5. 主要文件索引

- `multiscale_features.py`：三尺度图像独立解码重编码（DINOv2 第 6、12 块 + 拟合集 PCA 投影）
- `multiscale_train.py`：三尺度融合、3×3 纹理引导局部重采样、对数先验与场景自适应三尺度扩散头训练
- `multiscale_predict.py`：多分支概率空间融合、原图尺寸双线性恢复与二次归一化推理入口
- `outputs/report_replay/`：6 读出头独立复现权重、逐图原尺寸指标与分类别统计
- `figures/`：CCF-A 矢量架构图、多尺度特征演化热图与全套消融插图
