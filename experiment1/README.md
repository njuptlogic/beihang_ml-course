# 实验一：手写数字识别（小幅几何增强与位移鲁棒性分析）

- **姓名 / 学号**：徐基恒 / `SY2631122`
- **硬件与环境**：Python 3.11、PyTorch 2.6.0+cu124、NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），Conda 环境 `ml-course`
- **核心架构图**：[ccf_architecture.svg](figures/ccf_architecture.svg) · [ccf_architecture.png](figures/ccf_architecture.png)
- **深度特征可视化**：[Conv1 卷积核与四层特征响应](figures/conv1_and_layers.png) · [128 维全连接特征 t-SNE 与错例落点](figures/tsne_embedding.png)

## 1. 核心实验结果

| 模型方案 | 验证准确率均值 | 测试准确率（三种子均值 ± 标准差） | 种子 42 测试错判 | 3 px 平移验证准确率（种子 42） |
|---|---:|---:|---:|---:|
| MLP 基线（种子 42） | 98.37% | 98.47%（单次） | 153 / 10,000 | — |
| 原 CNN 基线 | 99.439% | 99.520% ± 0.035% | 50 / 10,000 | 92.20% |
| CNN + EMA（0.995） | 99.439% | 99.550% ± 0.000% | 45 / 10,000 | — |
| **CNN + 仿射增强（验证选定，单视图）** | **99.500%** | **99.600% ± 0.052%** | **46 / 10,000** | **98.87%（+6.67 pp）** |
| CNN + 仿射增强 + EMA（对照） | 99.478% | 99.630% ± 0.056% | 43 / 10,000 | — |

## 2. 环境准备与数据下载

以下命令均使用**相对路径**，在 `experiment1/` 目录内运行（若从仓库根目录运行，请先执行 `cd experiment1`）：

```bash
cd experiment1
conda env create -f environment.yml   # 首次配置环境时运行
conda activate ml-course

# 若缺少教师数据集 ZIP（data/raw/1-Digit-TrainSet.zip 与 1-Digit-TestSet.zip），执行下载：
test -f data/raw/1-Digit-TrainSet.zip && test -f data/raw/1-Digit-TestSet.zip || python download_data.py
```

## 3. 快速核验与单图推理

```bash
# 1) 独立复核已选定代表权重（预期输出 "correct": 9954, "errors": 46）
python evaluate.py --checkpoint outputs/ablation/views/selected.pt

# 2) 单张手写数字推理（白底黑字图片可附加 --invert）
python predict.py examples/7_0.bmp --checkpoint outputs/ablation/views/selected.pt

# 3) 运行全套单元测试
python -m pytest -q
```

## 4. 完整复现训练、消融与实时监控

```bash
# 1) 训练 MLP 与原 CNN 基线并生成数据分布图
python run_experiment.py --output outputs/reproduce_original

# 2) 三种子（42/43/44）完整消融训练（原图 vs 仿射增强 × 即时权重 vs EMA），并记录 TensorBoard 日志
python train_ablation.py --output outputs/reproduce_ablation --seeds 42 43 44 --epochs 20 --log-every 50

# 3) 在验证集上搜索推理多视图融合参数（最终选中单视图 view_mix=0）
python compare_views.py --run outputs/reproduce_ablation

# 4) 生成全部消融分析曲线、混淆矩阵、平移鲁棒性图与逐图预测 CSV/NPZ
python analyze_ablation.py --run outputs/reproduce_ablation

# 5) 重新加载选定权重独立核对测试集
python evaluate.py --checkpoint outputs/reproduce_ablation/views/selected.pt --output outputs/reproduce_ablation/verification.json
```

如需查看训练实时曲线与位移鲁棒性演化，可执行：

```bash
tensorboard --logdir outputs/reproduce_ablation/tensorboard
```

## 5. 目录结构说明

- `src/`：数据读取与分层划分（`data.py`）、网络定义（`models.py`）、多视图封装（`views.py`）、评测指标（`evaluation.py`）与监控（`training_monitor.py`）
- `outputs/main/`：MLP 与原 CNN 基线权重及曲线
- `outputs/ablation/`：三种子四方案完整消融权重、配对统计、平移敏感性与测试集预测
- `figures/`：CCF-A 矢量架构图及 Conv1 卷积核 / t-SNE 特征可视化
