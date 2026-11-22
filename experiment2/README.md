# 实验二：基于图宽留组验证与亮区空间读出的眼底图像分类

- **姓名 / 学号**：徐基恒 / `SY2631122`
- **硬件与环境**：Python 3.11、PyTorch 2.6.0+cu124、NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），Conda 环境 `ml-course`
- **核心架构图**：[ccf_architecture.svg](figures/ccf_architecture.svg) · [ccf_architecture.png](figures/ccf_architecture.png)
- **表征去偏可视化**：[类内图宽子空间投影前后 PCA 与误报压缩图](figures/subspace_pca_diagnosis.png)

## 1. 研究问题与核心结果

原始眼底数据存在显著的采集分辨率偏差（仅用原图宽高查训练集患病比例即可达到 0.9664 测试 AUROC，但固定阈值准确率仅 42.8%）。本实验采用 **四折图宽留组验证** 阻断尺寸泄露，结合 **三视图亮区中心/环带空间读出** 与 **类内图宽正交子空间投影（秩上限 18）**，在推理期无需图宽输入，直接将投影吸收进线性权重。

| 模型阶段 / 消融配置 | 250 张测试集 AUROC | 测试准确率 | 患病灵敏度（/100） | 正常特异度（/150） | 正常图误报数 |
|---|---:|---:|---:|---:|---:|
| 第一阶段：冻结 ResNet 全图 | 0.6540 | — | — | — | — |
| 第二阶段：RETFound-Green 三视图 | 0.7519 | 68.40% | — | — | — |
| 第二阶段：DINOv2 全图基线 | 0.8528 | 72.40% | 92.00%（92张） | 59.33% | 61 张 |
| 消融对照：全区域三视图 + 关闭投影（秩=0） | 0.8591 | 76.40% | 95.00%（95张） | 64.00% | 54 张 |
| **最终主模型：DINOv2 全区域 + 类内投影（秩=18）** | **0.9296** | **86.40%** | **92.00%（92张）** | **82.67%** | **26 张** |

## 2. 环境准备

以下所有命令均从**项目根目录（`ml-course/`）**使用相对路径运行：

```bash
conda activate ml-course
# 如首次配置环境（实验 2–4 共享）：
# conda create -n ml-course python=3.11 pip -y && conda activate ml-course
# python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# python -m pip install -r requirements-234.txt
```

## 3. 一键复现最终主结果与重绘图表

```bash
# 1) 从原始 JPEG 重新提取 1,889 张 × 3 视图的冻结 DINOv2-S/14-reg4 特征，
#    并复现秩=0 与秩=18 的四折留组拟合、折外概率校准与 250 张测试集评测（RTX 3070 Ti 约 55 秒）
python -m course.report_replay 2

# 2) 从逐图预测记录重新绘制 ROC / PR / 阈值分布等核心结果图
python -m course.report_figures 2

# 3) 生成极端误报/漏报样本原图与亮区裁图案例板
python -m course.report_cases 2
```

## 4. 模块化独立复现与单图推理

```bash
# 1) 下载与预处理课程眼底数据包（如本地尚未准备）
python -m course.download 2
python -m course.prepare 2

# 2) 提取三视图与亮区中心/环带空间特征库
python -m experiment2.spatial_features

# 3) 执行四折图宽留组网格搜索（330 组配置）与主模型训练
python -m experiment2.spatial

# 4) 执行同读出消融、操作点分析与亮度/Gamma/JPEG 图像质量扰动测试
python -m experiment2.spatial_controls
python -m experiment2.spatial_operating
python -m experiment2.spatial_stress
python -m experiment2.spatial_figures

# 5) 单张眼底图像推理（无需图宽标签，输出患病概率与固定阈值判决）
python -m experiment2.spatial predict --image ./sample_fundus.jpg
```

## 5. 主要文件索引

- `spatial_features.py`：96×96 绿/红通道亮区定位、镜像补边三视图生成与 DINOv2 中心/环带池化
- `spatial.py`：块归一化拼接、类内图宽 SVD 子空间投影、四折留组岭回归闭式解与折外概率校准
- `spatial_controls.py` / `spatial_stress.py` / `spatial_figures.py`：消融对照、鲁棒性扰动与可视化绘图
- `outputs/report_replay/`：独立复现输出（`metrics.csv`、`predictions.csv`、`cases.csv`、`run.json`）
- `figures/`：CCF-A 矢量架构图、特征子空间去偏对比图与历轮实验插图
