# 实验四：CIFAR-10 四步条件直线流图像生成

- **姓名 / 学号**：徐基恒 / `SY2631122`
- **硬件与环境**：Python 3.11、PyTorch 2.6.0+cu124、NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），Conda 环境 `ml-course`
- **核心架构图**：[ccf_architecture.svg](figures/ccf_architecture.svg) · [ccf_architecture.png](figures/ccf_architecture.png)
- **GPU 去噪轨迹与瓶颈特征图**：[四步条件流去噪轨迹与 U-Net 8×8 瓶颈自注意力特征演化](figures/flow_trajectory_and_unet_features.png) · [十类频谱校正前后完整对照](figures/report_replay_10class.png)

## 1. 方法创新与核心结果

本实验构建了轻量级条件直线流 U-Net（总参数量 673,532，单条件分支前向 0.135 GFLOPs，时间+类别 FiLM 调制注入 `Enc/Mid1/Mid2/Dec` 四个残差块），在 Logit-Normal 时间采样基础上依次引入三项核心机制：
1. **EMA 教师条件速度差修正目标**：`v_target = (x - z) + 0.65 * (v_c - v_u)`（无梯度回传，空标签修正项为 0）；
2. **四步历史速度外推采样**：`ṽ_k = v_k + 0.9 * (v_k - m_k)`，零额外网络求值补偿大步长离散误差；
3. **逐步 2D FFT 径向 12 频带振幅校正**：按 `t²P_b + (1-t)²` 参考功率缩放振幅（强度 0.35），严格保留相位与直流分量。

| 生成配置（10,000 生成图 vs 10,000 官方测试图） | 分支求值 / 图 | FID ↓ | IS ↑ | KID × 1,000 ↓ | 探针条件一致率 ↑ |
|---|---:|---:|---:|---:|---:|
| 20 轮直线流基线（12 步全程引导） | 24 | 78.76 | 4.58 ± 0.10 | 63.28 ± 1.89 | — |
| 20 轮基线直接压缩至 4 步 | 8 | 92.33 | 4.34 ± 0.09 | 71.22 ± 1.48 | — |
| 32 轮 Logit-Normal 时间采样 + 强引导 | 24 | 62.95 | 6.06 | 33.01 | 67.28% |
| 44 轮双模块对照（原引导） | 24 | 62.06 | 5.43 | 45.21 | 49.00% |
| 52 轮 EMA 目标修正 + 4 步（关闭历史外推） | 8 | 78.25 | — | — | 78.67% |
| 52 轮 EMA 目标修正 + 4 步外推（无频谱校正） | 8 | 50.55 | 7.154 ± 0.165 | 19.02 ± 0.94 | 80.32% |
| **最终主方案：52 轮 + 4 步外推 + 频谱校正（0.35）** | **8** | **48.40** | **7.193 ± 0.108** | **18.17 ± 0.82** | **80.46%** |
| 独立噪声复测（种子 27619，同权重） | 8 | **48.51** | — | 17.35（CIFAR-10.1） | — |

## 2. 环境准备

以下所有命令均从**项目根目录（`ml-course/`）**使用相对路径运行：

```bash
conda activate ml-course
```

## 3. 一键复现 8 轮续训、四步生成与评测图表

```bash
# 1) 从第 44 轮检查点续训 8 轮（3,128 步，约 102 秒），使用同一起始噪声（种子 109）
#    各生成 10,000 张无校正与频谱强度 0.35 图像并计算 FID / IS / KID / 探针一致率
python -m course.report_replay 4 --output experiment4/outputs/fresh_flow_run

# 2) 重绘十类探针一致率与同噪声生成对照图
python -m course.report_figures 4 --input-dir experiment4/outputs/fresh_flow_run

# 3) 重绘鸟/猫/犬错转对与对转错配对案例图版
python -m course.report_cases 4 --input-dir experiment4/outputs/fresh_flow_run
```

## 4. 全流程模块化复现与单图条件生成

```bash
# 1) 下载 CIFAR-10 数据集与早期 1–20 / 20–32 轮对照训练
python -m experiment4.download
python -m experiment4.train
python -m experiment4.research_train

# 2) 执行第 44–52 轮 EMA 教师差分目标续训与采样预算网格评测
python -m experiment4.guidance_train
python -m experiment4.guidance_eval
python -m experiment4.guidance_probe

# 3) 执行径向频谱强度扫描（0～0.65）、独立噪声复测与 CIFAR-10.1 外部队列 KID 评测
python -m experiment4.spectral_guidance
python -m experiment4.external_cohort

# 4) （独立替代路线对照）冻结第 32 轮速度场的学习型少步求解器实验
python -m experiment4.learned_solver
python -m experiment4.learned_solver_evaluate

# 5) 单张条件图像生成（相对路径示例：类别 0~9 对应飞机~卡车，8 为船）
python -m experiment4.spectral_guidance sample --class-id 8 --seed 123 --output ./outputs/ship.png
```

## 5. 主要文件索引

- `model.py`：条件速度网络 `FlowNet`（PixelUnshuffle/Shuffle、4 个 FiLM 残差块、8×8 四头自注意力瓶颈）
- `guidance_train.py` / `guidance_eval.py`：EMA 教师条件速度差修正训练与历史速度外推采样器
- `spectral_guidance.py`：2D 正交 FFT 径向 12 频带功率统计与保相位/保直流逐步振幅校正
- `learned_solver.py`：冻结第 32 轮速度场的替代路线学习型少步求解器对照实验
- `outputs/report_replay/`：第 52 轮检查点、两组 10,000 张生成图像数组、分布指标与逐图探针记录
- `figures/`：CCF-A 矢量架构图、四步去噪轨迹与 8×8 瓶颈特征图、十类完整对照图及历轮消融插图
