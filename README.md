# 机器学习课程实验

| 实验 | 报告 | 实测结果 |
|---|---|---|
| 1 手写数字识别 | [Markdown 报告](experiment1/SY2631122徐基恒.md) · [复现](experiment1/README.md) | 三种子平均测试准确率由 99.52% 提至 99.60%。代表模型在验证图平移 3 像素时由 92.20% 提至 98.87% |
| 2 眼底图像分类 | [Markdown 报告](experiment2/SY2631122徐基恒.md) · [历轮实验](experiment2/report.md) | 250 张眼底图，像素分类器的 AUROC 从 0.8528 升至 0.9296，准确率从 72.4% 升至 86.4%。同样检出 92 张患病图，正常图误报从 61 张降到 26 张 |
| 3 人眼注视密度预测 | [Markdown 报告](experiment3/SY2631122徐基恒.md) · [历轮实验](experiment3/report.md) | 400 张原尺寸测试图，CC 从 0.8910 升至 0.9021，KL 从 0.2641 降至 0.2455，SIM 从 0.7504 升至 0.7606 |
| 4 CIFAR-10 彩色图像生成 | [Markdown 报告](experiment4/SY2631122徐基恒.md) · [历轮实验](experiment4/report.md) | 在 10,000 张生成图对 10,000 张真实测试图的协议下，FID 从 62.06 降至 48.40，每图分支求值从 24 次降至 8 次。换一组生成噪声复测得到 48.51 |

[教师作业说明](../Machine-Learning-Class-Assignment26/README.md) · [三个实验的模块、论文日期与选择依据](method-origins.md) · [第三轮逐指标数据](frontier-results.csv) · [前两轮逐指标数据](research-results.csv)

三份报告使用同一套课程划分追踪各轮差异。复制 Markdown 时带上同级 `figures` 目录即可保留插图，包括可编辑的同名 SVG。各轮生成模型共享续训起点。第二组采样噪声检验固定生成器对起点的敏感性，训练种子的影响仍待独立重训测量。

克隆仓库后，请按下文各实验的复现步骤下载数据并生成模型权重。报告中的数据、权重和批量预测文件路径对应本地运行产物，结果表和报告插图随仓库提供。

## 实验一复现与实时曲线

以下命令从项目根目录运行。新机器可先用 `conda env create -f experiment1/environment.yml` 建立环境。已有旧版环境如缺少 TensorBoard，可在激活后执行 `python -m pip install tensorboard==2.21.0`。教师的两个 ZIP 放在 `experiment1/data/raw/`，缺失时运行 `python experiment1/download_data.py`。

```bash
cd /home/ubuntu/Documents/ml-course
conda activate ml-course

# 直接复核已交付的代表模型：预期答对 9,954 / 10,000 张
python experiment1/evaluate.py --checkpoint experiment1/outputs/ablation/views/selected.pt

# 从头训练，后续命令在同一终端使用本次输出目录
digit_run="experiment1/outputs/live_$(date +%Y%m%d_%H%M%S)"
python experiment1/train_ablation.py --output "$digit_run" --seeds 42 43 44 --epochs 20 --log-every 50
python experiment1/compare_views.py --run "$digit_run"
python experiment1/analyze_ablation.py --run "$digit_run"
python experiment1/evaluate.py --checkpoint "$digit_run/views/selected.pt" --output "$digit_run/verification.json"
```

**训练时**，终端每轮显示一行训练、验证、EMA 验证的 loss 和准确率，以及学习率和耗时。打开启动时打印的 TensorBoard 地址，勾选右上角齿轮中的 **Reload data**，可同步查看 batch 曲线和逐轮曲线。`--log-every 50` 设置看板采样间隔，每轮首批和末批也会记录。训练结束后可用 `tensorboard --logdir "$digit_run/tensorboard"` 重新查看。

**看仿射增强的效果。** 在 TensorBoard 的 Scalars 中选中 `seed42/plain/val` 和 `seed42/affine/val`，搜索 `robustness/shift_3px/accuracy_pct`。每个点比较同一轮模型在同一批 6,000 张验证图上的表现，取上、下、左、右各平移 3 像素后的平均准确率。搜索 `robustness/shift_3px/drop_pp` 可看相对各自原图掉了多少个百分点，越小表示越耐平移。`rotate_12deg` 和 `grid_scale_10pct` 分别对应旋转和缩放。每个种子先训练 `plain` 再训练 `affine`，两条曲线按轮次对齐。已跑好的种子 42 曲线可以从项目根目录直接打开。

```bash
conda activate ml-course
tensorboard --logdir experiment1/outputs/robustness_seed42/tensorboard
```

**核对结果时**，`$digit_run/analysis/runs.csv` 中三个 `affine` 行的 `test` 列用于计算三种子平均准确率（参考值 **99.60%**）。`verification.json` 记录种子 42 代表模型的独立测试准确率（**99.54%**）。[实验一详细说明](experiment1/README.md)

## 第三轮材料与复现

| 实验 | 研究问题 | 结果与原始记录 |
|---|---|---|
| 2 | 用图宽分组验证选模型，在相近采集条件下分辨患病与正常。比较全图、局部中心和环带，以及移除采集方向前后的差异 | [330 组训练侧比较](experiment2/outputs/spatial/cv_grid.csv) · [锁定其他设置的消融](experiment2/outputs/spatial/matched_controls.csv) · [同图置信区间](experiment2/outputs/spatial/paired.csv) |
| 3 | 让 DINO 以三个真实输入尺度观察同一场景，再加入 PE-Spatial 检查两种表征是否互补。在每张原图上与公开 UNISAL 权重比较 | [逐图指标](experiment3/outputs/multiscale/native_images.csv) · [验证选出的权重](experiment3/outputs/multiscale/selection.json) · [同图置信区间](experiment3/outputs/multiscale/paired_intervals.csv) |
| 4 | 训练时增强条件速度，采样时比较历史速度外推与逐步校正频谱。另在冻结的旧模型上检验短轨迹蒸馏 | [14 组引导与算量对照](experiment4/outputs/guidance/test/metrics.csv) · [频谱校正及第二个种子](experiment4/outputs/spectral_guidance/repeat/metrics.csv) · [CIFAR-10.1 的配对差值](experiment4/outputs/external_cohort/supplement_contrasts.csv) · [12 组求解器](experiment4/outputs/learned_solver/evaluation/metrics.csv) |

在项目根目录运行以下命令即可载入训练好的第三轮模型。最后一条命令从逐图预测和 Inception 特征库复算主指标，写出 `frontier-results.csv`。

```bash
python -m experiment2.spatial predict --image /path/to/fundus.jpg
python -m experiment3.multiscale_predict --image /path/to/scene.jpg --output /tmp/gaze.npy
python -m experiment4.spectral_guidance sample --class-id 8 --seed 123 --output /tmp/ship.png
python -m course.frontier_results
```

## 第二轮材料与模型

| 实验 | 设计 | 按图回查结果 |
|---|---|---|
| 2 | 在拟合集按图像宽度分组做四折验证，比较预训练来源、全图、局部亮区与颜色归一化，再测压暗、压缩和降采样的影响 | [84 组验证记录](experiment2/outputs/foundation/cv_grid.csv) · [每张图的模型分数](experiment2/outputs/foundation/predictions.csv) |
| 3 | 用冻结的自监督 patch 表征预测注视分布，分别加入局部纹理和场景先验。每种配置训练三个种子，再在同类场景之间交换特征和先验 | [逐图原尺寸指标](experiment3/outputs/semantic/native_images.csv) · [高概率区域的面积与覆盖](experiment3/outputs/semantic/region_by_image.csv) |
| 4 | 将训练时间分布与掩码特征对齐做成 2 × 2 消融，在相同权重上复测采样引导。再用独立的 DINO 表征检查生成样本覆盖了哪些真实图像 | [生成配置与指标](experiment4/outputs/research/evaluation/metrics.csv) · [同类最近邻与覆盖判断](experiment4/outputs/research/semantic_evaluation/nearest.csv) |

各报告的复现章节给出了从已有数据到训练、评测和重新出图的命令。要直接使用第二轮的权重，可以从项目根目录运行下面的示例。

```bash
python -m experiment2.foundation predict --image /path/to/fundus.jpg
python -m experiment3.semantic_predict --image /path/to/scene.jpg --output /tmp/gaze.npy
python -m experiment4.research_verify --scenario combined_strong --class-id 8 --seed 123 --generate /tmp/ship.png
python -m course.research_results
```

最后一条命令从眼底预测分数、注视预测的逐图结果和生成图的特征库重新核对测试指标，并写出项目根目录的结果总表。

## 环境与数据

以下命令从本项目根目录运行。本机沿用实验一的 ml-course 环境，Python 3.11、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124，本次报告复跑使用 NVIDIA GeForce RTX 3070 Ti Laptop GPU（8 GB），各实验的运行记录列出驱动和 CUDA 版本。全量依赖版本见[实测环境](environment-234.lock.txt)。

```bash
cd /home/ubuntu/Documents/ml-course
conda activate ml-course
```

在新机器上重建环境可执行下列命令。

```bash
conda create -n ml-course python=3.11 pip -y
conda activate ml-course
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-234.txt
```

实验 2 和 3 使用教师分享的北航网盘 ZIP。实验 4 使用 CIFAR-10 Python 版，数据包 MD5 与 torchvision 登记值一致。项目还使用 ImageNet ResNet-18、torch-fidelity Inception-v3-compat、RETFound-Green 和 DINOv2-S/14-reg。实验三还使用了 PE-Core、PE-Spatial 与 UNISAL 的官方模型，实验四在 CIFAR-10.1 v6 的 2,000 张真实图上复测了 KID。各报告记录数据来源和实际加载的权重。

## 第一轮基线的复现

已有图表和结果可以直接阅读。下列完整流程使用默认目录并重写相应结果。希望保留这次运行时，可先复制项目到另一目录再运行。实验 4 的训练脚本从已完成的一轮继续，重复执行同一配置会跳过已经完成的轮次。要从头训练，给训练和评测传入同一个新的 `--output` 目录。

```bash
# 实验二
python -m course.download 2
python -m course.prepare 2
python -m course.features 2
python -m experiment2.train
python -m experiment2.followup
python -m experiment2.visualize
python -m experiment2.evidence

# 实验三
python -m course.download 3
python -m course.prepare 3
python -m course.features 3
python -m experiment3.train
python -m experiment3.native
python -m experiment3.visualize

# 实验四
python -m experiment4.download
python -m experiment4.train
OPENBLAS_NUM_THREADS=4 python -m experiment4.evaluate
python -m experiment4.visualize
python -m experiment4.train --parameterization preconditioned --output experiment4/outputs/preconditioned
OPENBLAS_NUM_THREADS=4 python -m experiment4.evaluate --output experiment4/outputs/preconditioned --methods flow_main
python -m experiment4.compare_training

# 三项实验的权重重载、原图推理与汇总指标复核
python -m course.verify
```

原始测试集与训练集的身份在输出中保留。实验 2 的各配置先按验证 AUROC 选轮次和阈值，实验 3 按验证 CC 选轮次，实验 4 使用预定的第 20 轮。医学追加比较及生成模型第二种参数化的实验顺序在各自报告中说明。

## 第一轮基线的单图推理

```bash
# 替换为自己的输入图像。眼底输出患病分数、验证阈值与类别。
python -m course.predict retina /path/to/fundus.jpg

# 输出与输入同宽高的灰度图；输出后缀换成 .npy 可取单位和密度。
python -m course.predict gaze /path/to/scene.jpg /tmp/gaze.png

# 从随机噪声生成一张 32 × 32 彩色图。类别 0～9 沿用 CIFAR-10 顺序。
python -m course.predict generate /tmp/ship.png --class-id 8 --seed 123
```

数值图表与架构图按本机 paper 项目的 scientific-visualization 与 embodied-architecture-diagrams 技能制作。原始图像、排序规则和统计口径随报告保留。绘图方法参考 Kassis、Agarwal、He、Patel 和 Brueckner 的 Scientific Agent Skills。[论文](https://arxiv.org/abs/2609.00065)
