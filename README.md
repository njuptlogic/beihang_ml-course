# 机器学习课程实验（SY2631122 · 徐基恒）

本仓库收录《机器学习》课程全部四个实验的完整代码、消融记录、CCF-A 风格矢量架构图、特征可视化结果与最终实验报告。

- **姓名 / 学号**：徐基恒 / `SY2631122`
- **默认运行环境**：Python 3.11 · PyTorch 2.6.0+cu124 · torchvision 0.21.0+cu124 · Conda 环境 `ml-course`
- **实测硬件平台**：NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB）
- **配套汇总文档**：[方法来源与选型依据](method-origins.md) · [最终轮逐指标总表](frontier-results.csv) · [前两轮逐指标总表](research-results.csv) · [依赖锁定清单](environment-234.lock.txt)

---

## 1. 实验报告直达与核心结果总览

> 点击下表或清单中的报告地址即可直接跳转查看对应 PDF 实验报告。

### 报告地址速览

- **实验一报告（手写数字识别）**：[experiment1/report/SY2631122徐基恒.pdf](experiment1/report/SY2631122徐基恒.pdf)
- **实验二报告（眼底图像分类）**：[experiment2/report/SY2631122徐基恒.pdf](experiment2/report/SY2631122徐基恒.pdf)
- **实验三报告（人眼注视密度预测）**：[experiment3/report/SY2631122徐基恒.pdf](experiment3/report/SY2631122徐基恒.pdf)
- **实验四报告（CIFAR-10 四步条件直线流图像生成）**：[experiment4/report/SY2631122徐基恒.pdf](experiment4/report/SY2631122徐基恒.pdf)

### 四项实验核心成果一览

| 实验 | 实验报告（点击查看） | 详细说明与架构图 | 核心实测结果 |
|---|---|---|---|
| **实验一**：手写数字识别 | [experiment1/report/SY2631122徐基恒.pdf](experiment1/report/SY2631122徐基恒.pdf) | [实验一 README](experiment1/README.md) · [架构图 (SVG)](experiment1/figures/ccf_architecture.svg) | 三种子测试准确率均值由 **99.52%** 提至 **99.60% ± 0.052%**（代表权重错判 **46 / 10,000**）；验证图 3 px 平移准确率由 **92.20%** 提至 **98.87%（+6.67 pp）** |
| **实验二**：眼底图像分类 | [experiment2/report/SY2631122徐基恒.pdf](experiment2/report/SY2631122徐基恒.pdf) | [实验二 README](experiment2/README.md) · [架构图 (SVG)](experiment2/figures/ccf_architecture.svg) | 250 张测试集 AUROC 由 **0.8528** 升至 **0.9296**，准确率由 **72.40%** 升至 **86.40%**；同等 92/100 患病检出下，正常图误报由 **61 张降至 26 张** |
| **实验三**：人眼注视密度预测 | [experiment3/report/SY2631122徐基恒.pdf](experiment3/report/SY2631122徐基恒.pdf) | [实验三 README](experiment3/README.md) · [架构图 (SVG)](experiment3/figures/ccf_architecture.svg) | 400 张 1920×1080 原尺寸测试图：CC **0.8910 → 0.90210**，KL **0.2641 → 0.24551**，SIM **0.7504 → 0.76056**（纯课程双分支 KL **0.24435** / SIM **0.76352**） |
| **实验四**：CIFAR-10 图像生成 | [experiment4/report/SY2631122徐基恒.pdf](experiment4/report/SY2631122徐基恒.pdf) | [实验四 README](experiment4/README.md) · [架构图 (SVG)](experiment4/figures/ccf_architecture.svg) | 10,000 生成图 vs 10,000 测试图：FID **62.06 → 48.40**，IS **7.193 ± 0.108**，KID×10³ **18.17 ± 0.82**，每图分支求值由 **24 次降至 8 次（4 步）**；独立噪声复测 FID **48.51** |

---

## 2. 环境配置与数据说明

以下命令均默认在**仓库根目录**执行。克隆仓库后，请先激活或创建 `ml-course` Conda 环境，再按各实验说明下载数据并运行训练/评测脚本（模型权重与原始数据集体积较大，未纳入 Git 版本控制，结果汇总 CSV 与报告插图随仓库提供）。

### 方式 A：从零创建统一环境（支持全部四个实验）

```bash
conda create -n ml-course python=3.11 pip -y
conda activate ml-course
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-234.txt
```

### 方式 B：基于实验一环境配置并补充实验二至四依赖

```bash
conda env create -f experiment1/environment.yml
conda activate ml-course
python -m pip install -r requirements-234.txt
```

> 若本地旧版环境缺少 TensorBoard，可执行 `python -m pip install tensorboard==2.21.0`。完整包版本见 [environment-234.lock.txt](environment-234.lock.txt)。

---

## 3. 实验一复现操作（手写数字识别）

- **实验报告**：[experiment1/report/SY2631122徐基恒.pdf](experiment1/report/SY2631122徐基恒.pdf)
- **分实验说明**：[experiment1/README.md](experiment1/README.md)
- **新增可视化**：[CCF-A 架构图](experiment1/figures/ccf_architecture.png) · [Conv1 卷积核与四层特征响应](experiment1/figures/conv1_and_layers.png) · [128 维全连接特征 t-SNE](experiment1/figures/tsne_embedding.png)

### 3.1 数据准备与快速核验

```bash
conda activate ml-course

# 若缺少教师数据集 ZIP（experiment1/data/raw/1-Digit-TrainSet.zip 与 1-Digit-TestSet.zip），执行下载：
test -f experiment1/data/raw/1-Digit-TrainSet.zip && test -f experiment1/data/raw/1-Digit-TestSet.zip || python experiment1/download_data.py

# 1) 独立复核已选定代表权重（预期输出 correct: 9954, errors: 46）
python experiment1/evaluate.py --checkpoint experiment1/outputs/ablation/views/selected.pt

# 2) 单张手写数字推理（白底黑字图片可附加 --invert）
python experiment1/predict.py experiment1/examples/7_0.bmp --checkpoint experiment1/outputs/ablation/views/selected.pt

# 3) 运行实验一单元测试
(cd experiment1 && python -m pytest -q)
```

### 3.2 从头完整训练、消融与 TensorBoard 监控

```bash
# 1) 训练 MLP 与原 CNN 基线并生成数据分布图
python experiment1/run_experiment.py --output experiment1/outputs/reproduce_original

# 2) 三种子（42/43/44）完整消融训练（原图 vs 仿射增强 × 即时权重 vs EMA），同步记录 TensorBoard 日志
digit_run="experiment1/outputs/live_$(date +%Y%m%d_%H%M%S)"
python experiment1/train_ablation.py --output "$digit_run" --seeds 42 43 44 --epochs 20 --log-every 50

# 3) 验证集搜索推理多视图融合参数（最终选定单视图 view_mix=0）
python experiment1/compare_views.py --run "$digit_run"

# 4) 生成消融曲线、混淆矩阵、平移鲁棒性图与逐图预测文件
python experiment1/analyze_ablation.py --run "$digit_run"

# 5) 独立核验选定模型在测试集上的表现
python experiment1/evaluate.py --checkpoint "$digit_run/views/selected.pt" --output "$digit_run/verification.json"

# 6) 查看训练曲线与位移鲁棒性演化（在 Scalars 中检索 robustness/shift_3px/accuracy_pct）
tensorboard --logdir "$digit_run/tensorboard"
```

---

## 4. 实验二复现操作（眼底图像分类）

- **实验报告**：[experiment2/report/SY2631122徐基恒.pdf](experiment2/report/SY2631122徐基恒.pdf)
- **分实验说明**：[experiment2/README.md](experiment2/README.md)
- **关键结果表**：[330 组留组网格](experiment2/outputs/spatial/cv_grid.csv) · [同读出消融对照](experiment2/outputs/spatial/matched_controls.csv) · [同图配对置信区间](experiment2/outputs/spatial/paired.csv)
- **新增可视化**：[CCF-A 架构图](experiment2/figures/ccf_architecture.png) · [类内图宽子空间去偏 PCA 诊断图](experiment2/figures/subspace_pca_diagnosis.png)

### 4.1 一键复现最终主结果与重绘图表

```bash
conda activate ml-course

# 1) 从原始 JPEG 重新提取 1,889 张 × 3 视图冻结 DINOv2-S/14-reg4 特征，
#    复现秩=0 与秩=18 四折图宽留组拟合、折外概率校准与 250 张测试集评测（RTX 3070 Ti 约 55 秒）
python -m course.report_replay 2

# 2) 基于逐图预测记录重新绘制 ROC / PR / 阈值分布等结果图
python -m course.report_figures 2

# 3) 生成典型误报/漏报样本原图与亮区裁图案例板
python -m course.report_cases 2
```

### 4.2 全流程模块化复现与单图推理

```bash
# 1) 下载并预处理眼底数据集（若本地尚未准备）
python -m course.download 2
python -m course.prepare 2

# 2) 提取三视图与亮区中心/环带空间特征
python -m experiment2.spatial_features

# 3) 执行四折图宽留组网格搜索（330 组配置）与主模型训练
python -m experiment2.spatial

# 4) 执行同读出消融、操作点分析与亮度/Gamma/JPEG 扰动压力测试
python -m experiment2.spatial_controls
python -m experiment2.spatial_operating
python -m experiment2.spatial_stress
python -m experiment2.spatial_figures

# 5) 单张眼底图像推理（推理期无需图宽输入，直接输出患病概率与固定阈值判决）
python -m experiment2.spatial predict --image ./sample_fundus.jpg
```

---

## 5. 实验三复现操作（人眼注视密度预测）

- **实验报告**：[experiment3/report/SY2631122徐基恒.pdf](experiment3/report/SY2631122徐基恒.pdf)
- **分实验说明**：[experiment3/README.md](experiment3/README.md)
- **关键结果表**：[400 张原尺寸逐图指标](experiment3/outputs/multiscale/native_images.csv) · [验证集选定融合权重](experiment3/outputs/multiscale/selection.json) · [同图配对置信区间](experiment3/outputs/multiscale/paired_intervals.csv)
- **新增可视化**：[CCF-A 架构图](experiment3/figures/ccf_architecture.png) · [四类场景多尺度特征与注视热图对照](experiment3/figures/multiscale_stage_progression.png)

### 5.1 一键复现六个读出头训练与原尺寸评测

```bash
conda activate ml-course

# 1) 复用冻结特征，以种子 17/42/73 重训三尺度金字塔与双编码器共 6 个读出头（每头 18 轮），
#    并在 400 张 1920×1080 原尺寸测试图上双精度复测 CC / KL / SIM（RTX 3070 Ti 约 7 分钟）
python -m course.report_replay 3 --output experiment3/outputs/fresh_gaze_run

# 2) 基于独立复现结果重绘 20 类场景箱线图与训练曲线
python -m course.report_figures 3 --input-dir experiment3/outputs/fresh_gaze_run

# 3) 重绘典型增益/退步案例热图对比板
python -m course.report_cases 3 --input-dir experiment3/outputs/fresh_gaze_run
```

### 5.2 全流程模块化复现与单图推理

```bash
# 1) 下载并预处理注视预测数据集及浅层 ResNet 纹理特征
python -m course.download 3
python -m course.prepare 3
python -m course.features 3

# 2) 提取三尺度 DINOv2（224/448/672）及 PE-Spatial-T16 / UNISAL 冻结表征
python -m experiment3.multiscale_features

# 3) 训练全部 9 种消融架构 × 3 种子（共 27 个读出头）并完成融合权重搜索与评测
python -m experiment3.multiscale_train
python -m experiment3.multiscale_evaluate
python -m experiment3.multiscale_figures

# 4) 单张场景图注视密度预测（输出与原图同宽高且像素和为 1 的 .npy 密度图）
python -m experiment3.multiscale_predict --image ./sample_scene.jpg --output ./outputs/gaze_pred.npy
```

---

## 6. 实验四复现操作（CIFAR-10 四步条件直线流图像生成）

- **实验报告**：[experiment4/report/SY2631122徐基恒.pdf](experiment4/report/SY2631122徐基恒.pdf)
- **分实验说明**：[experiment4/README.md](experiment4/README.md)
- **关键结果表**：[14 组引导与算量对照](experiment4/outputs/guidance/test/metrics.csv) · [频谱校正与独立噪声复测](experiment4/outputs/spectral_guidance/repeat/metrics.csv) · [CIFAR-10.1 配对差值](experiment4/outputs/external_cohort/supplement_contrasts.csv) · [12 组学习型求解器对照](experiment4/outputs/learned_solver/evaluation/metrics.csv)
- **新增可视化**：[CCF-A 架构图](experiment4/figures/ccf_architecture.png) · [四步去噪轨迹与 8×8 瓶颈特征图](experiment4/figures/flow_trajectory_and_unet_features.png) · [十类频谱校正完整对照](experiment4/figures/report_replay_10class.png)

### 6.1 一键复现 8 轮续训、四步生成与指标评测

```bash
conda activate ml-course

# 1) 从第 44 轮检查点续训 8 轮（3,128 步，约 102 秒），以同一起始噪声（种子 109）
#    生成无校正与径向频谱校正（强度 0.35）各 10,000 张图像，并复算 FID / IS / KID / 探针一致率
python -m course.report_replay 4 --output experiment4/outputs/fresh_flow_run

# 2) 重绘十类探针一致率与同噪声生成对比图
python -m course.report_figures 4 --input-dir experiment4/outputs/fresh_flow_run

# 3) 重绘鸟/猫/犬等错转对与对转错配对案例板
python -m course.report_cases 4 --input-dir experiment4/outputs/fresh_flow_run
```

### 6.2 全流程模块化复现与单图条件生成

```bash
# 1) 下载 CIFAR-10 数据集并完成早期 1–20 / 20–32 轮对照训练
python -m experiment4.download
python -m experiment4.train
python -m experiment4.research_train

# 2) 执行第 44–52 轮 EMA 教师速度差目标续训、采样预算网格评测与探针分类核验
python -m experiment4.guidance_train
python -m experiment4.guidance_eval
python -m experiment4.guidance_probe

# 3) 执行 2D FFT 径向频谱强度扫描（0～0.65）、独立噪声复测与 CIFAR-10.1 外部队列 KID 评测
python -m experiment4.spectral_guidance
python -m experiment4.external_cohort

# 4) （替代路线对照）冻结第 32 轮速度场的学习型少步求解器实验
python -m experiment4.learned_solver
python -m experiment4.learned_solver_evaluate

# 5) 单张条件图像生成（类别 0～9 对应 CIFAR-10 顺序，8 为 ship）
python -m experiment4.spectral_guidance sample --class-id 8 --seed 123 --output ./outputs/ship.png
```

---

## 7. 跨实验总表复核与历史基线复现

### 7.1 汇总指标核对（生成总表 CSV）

```bash
# 汇总实验 2–4 最终轮指标并写入 frontier-results.csv
python -m course.frontier_results

# 汇总实验 2–4 前两轮指标并写入 research-results.csv
python -m course.research_results

# 核对第一轮基线权重重载与原图推理
python -m course.verify
```

### 7.2 早期轮次基线复现（可选）

<details>
<summary>展开查看第一轮 / 第二轮基线复现与单图推理命令</summary>

```bash
# 第一轮基线：实验二
python -m course.download 2 && python -m course.prepare 2 && python -m course.features 2
python -m experiment2.train && python -m experiment2.followup && python -m experiment2.visualize && python -m experiment2.evidence

# 第一轮基线：实验三
python -m course.download 3 && python -m course.prepare 3 && python -m course.features 3
python -m experiment3.train && python -m experiment3.native && python -m experiment3.visualize

# 第一轮基线：实验四
python -m experiment4.download && python -m experiment4.train
OPENBLAS_NUM_THREADS=4 python -m experiment4.evaluate && python -m experiment4.visualize
python -m experiment4.train --parameterization preconditioned --output experiment4/outputs/preconditioned
OPENBLAS_NUM_THREADS=4 python -m experiment4.evaluate --output experiment4/outputs/preconditioned --methods flow_main
python -m experiment4.compare_training

# 第一轮基线单图推理
python -m course.predict retina ./sample_fundus.jpg
python -m course.predict gaze ./sample_scene.jpg ./outputs/gaze_v1.png
python -m course.predict generate ./outputs/ship_v1.png --class-id 8 --seed 123

# 第二轮中间模型单图推理
python -m experiment2.foundation predict --image ./sample_fundus.jpg
python -m experiment3.semantic_predict --image ./sample_scene.jpg --output ./outputs/gaze_v2.npy
python -m experiment4.research_verify --scenario combined_strong --class-id 8 --seed 123 --generate ./outputs/ship_v2.png
```

</details>

---

## 8. 仓库目录结构

```text
ml-course/
├── README.md                          # 项目总说明、报告导航与全实验复现指南（本文件）
├── requirements-234.txt               # 实验 2–4 核心依赖清单
├── environment-234.lock.txt           # 完整实测环境版本锁定文件
├── method-origins.md                  # 方法来源、论文时间线与模块选型依据
├── frontier-results.csv               # 实验 2–4 最终轮逐指标汇总表
├── research-results.csv               # 实验 2–4 前两轮逐指标汇总表
├── course/                            # 共享下载、预处理、统一复现与重绘工具包
├── experiment1/                       # 实验一：手写数字识别（CNN + 仿射增强与位移鲁棒性）
│   ├── README.md                      # 实验一复现说明
│   ├── report/SY2631122徐基恒.pdf     # 实验一最终报告（PDF）
│   ├── src/                           # 数据、模型、视图、评测与监控模块
│   ├── figures/                       # CCF-A 矢量架构图与卷积核/t-SNE 可视化
│   └── outputs/                       # 消融统计表与分析图
├── experiment2/                       # 实验二：眼底图像分类（图宽留组 + 亮区读出 + 子空间投影）
│   ├── README.md                      # 实验二复现说明
│   ├── report/SY2631122徐基恒.pdf     # 实验二最终报告（PDF）
│   ├── figures/                       # CCF-A 矢量架构图与子空间去偏 PCA 图
│   └── outputs/                       # 330 组留组网格与配对统计 CSV/JSON
├── experiment3/                       # 实验三：人眼注视密度预测（三尺度 DINOv2 + PE-Spatial + 扩散）
│   ├── README.md                      # 实验三复现说明
│   ├── report/SY2631122徐基恒.pdf     # 实验三最终报告（PDF）
│   ├── figures/                       # CCF-A 矢量架构图与多尺度特征演化热图
│   └── outputs/                       # 400 张原尺寸测试逐图指标与融合配置
└── experiment4/                       # 实验四：CIFAR-10 四步条件直线流图像生成
    ├── README.md                      # 实验四复现说明
    ├── report/SY2631122徐基恒.pdf     # 实验四最终报告（PDF）
    ├── figures/                       # CCF-A 矢量架构图、去噪轨迹与十类对照图
    └── outputs/                       # 分布指标、频谱扫描与外部队列 CSV/JSON
```
