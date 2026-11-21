# 实验一：手写数字识别

学生：**SY2631122 徐基恒**。

当前报告为可编辑的 **[实验一 Markdown 报告](SY2631122徐基恒.md)**。报告按课程大纲组织，包含 15 幅图、9 张表、代码导读、逐例分析和其他分类任务的迁移示例。

| 方法 | seed 42 测试 | 三种子测试均值 ± 样本标准差 |
|---|---:|---:|
| MLP | 98.47% | 单次 |
| 原 CNN | 99.50% | 99.520% ± 0.035% |
| **增强 CNN，验证所选** | **99.54%** | **99.600% ± 0.052%** |

固定训练/验证划分，三个独立训练种子为 42、43、44。增强模型在同图配对下分别净少错 4、13、7 张；种子 42 的准确率差 95% 区间为 −0.09 至 +0.17 个百分点。推理保持原网络大小和单视图。权重平均、平移读出的完整消融及其选择依据见[结果表](outputs/ablation/analysis/runs.csv)和[验证网格](outputs/ablation/views/validation_grid.csv)。

## 训练时看曲线

在已激活的 `ml-course` 环境和本目录下运行。每次使用新的输出目录。

~~~bash
python train_ablation.py --output "outputs/live_$(date +%Y%m%d_%H%M%S)" --seeds 42 43 44 --epochs 20
~~~

终端在每轮结束时输出一行，汇总当前种子和分支的训练、验证 loss 与准确率，以及学习率和耗时。实时 batch 曲线在 TensorBoard 中查看，默认每 50 批记录一个点，也记录每轮首批和末批。加 `--log-every 10` 可提高曲线采样频率。

启动时会打印本次 TensorBoard 的本地地址。打开链接，在右上角齿轮的 **Settings → Reload data** 启用浏览器自动刷新，也可点刷新按钮立即读取。不同种子和训练分支在左侧 Runs 中切换。看板使用以下横轴和统计口径：

| 面板 | 横轴 | 指标 |
|---|---|---|
| `batch` | 当前种子、当前分支累计的优化步 | 当前采样批次的交叉熵、准确率（%）、学习率 |
| `epoch` | 轮次 | 整轮训练 loss、准确率（按样本加权）；轮末验证集 loss 和准确率；消融训练还显示 EMA 权重的验证结果。可勾选 train、val、ema_val 在同一张图叠加 |
| `epoch/seconds` | 轮次 | 每轮实测耗时（秒） |
| `robustness/shift_3px` | 轮次 | 同一批验证图依次向四个方向移动 3 像素 |
| `robustness/rotate_12deg` | 轮次 | 绕图像中心分别旋转 −12°、+12° |
| `robustness/grid_scale_10pct` | 轮次 | 采样网格的缩放系数分别为 0.9、1.1，图像缩放倍数为其倒数 |

**叠加两条曲线。** 在 Scalars 的 Runs 中勾选 `seed42/plain/val` 和 `seed42/affine/val`，先看 `robustness/shift_3px/accuracy_pct`。左侧 Runs 也可以用正则 `^seed42/(plain|affine)/val$` 筛选。将 Smoothing 调到 0 能直接读出每轮的实测值。要比较 EMA，就把两个运行名末尾的 `val` 都换成 `ema_val`。训练顺序是先 `plain` 后 `affine`，第二条线在增强训练开始后从第 1 轮叠加。

三个扰动面板都给出 `accuracy_pct`（%）、`loss`（交叉熵）和 `drop_pp`（该轮原图准确率减去扰动准确率，单位为百分点）。每个方向或幅度都在全部 6,000 张验证图上独立预测，然后取均值。两种训练方式用相同的变换、黑色补边和原始标签。平移面板的设置与报告图 8 的 3 像素点一致，旋转和缩放分别只改变一个因素。训练仍按原图验证指标选权重，新版每轮耗时包含额外的验证前向计算。

`plain` 在原图上训练，`affine` 每批都要识别新的变形图，所以两条训练准确率对应的输入难度不同。比较增强效果时，先看两者在原图上的 `epoch/accuracy_pct`，再切到同一种扰动的验证曲线。同一条件下，准确率越高越好，掉点越少说明对该扰动越稳。

已完成的种子 42 对照位于 `outputs/robustness_seed42/tensorboard`。看第 20 轮的当前权重，`plain/val` 与 `affine/val` 在原图上分别为 99.43% 和 99.37%，在平移 3 像素后分别为 92.29% 和 99.03%。报告表 5 的方案选择仍按 42、43、44 三个种子的平均验证成绩计算。以前的事件文件只含当时记录的指标。新训练会把各条件的准确率（0～1）和损失同步写入 `seed*/plain_history.csv` 和 `seed*/affine_history.csv`，例如 `val_shift_3px_accuracy` 对应当前权重的平移验证准确率。

测试准确率在训练结束后通过评测脚本计算。`run_experiment.py` 的 MLP/CNN 训练也使用相同看板。加 `--no-dashboard` 只关闭随训练启动的 Web 服务，终端和事件记录继续工作。训练进程退出后，事件保存在本次输出目录的 `tensorboard/`，可重新打开，也可以直接打开这次种子 42 的对照。

~~~bash
tensorboard --logdir outputs/robustness_seed42/tensorboard
~~~

新环境安装 [requirements.txt](requirements.txt) 即包含 TensorBoard。旧环境如尚未安装，可运行 `python -m pip install tensorboard==2.21.0`。

## 当前版本的复现与代码入口

~~~bash
cd /home/ubuntu/Documents/ml-course/experiment1
conda activate ml-course

# 已交付权重可直接复核，预期 9954/10000
python evaluate.py --checkpoint outputs/ablation/views/selected.pt
python predict.py examples/7_0.bmp --checkpoint outputs/ablation/views/selected.pt

# 从头复跑时选空输出目录。原始 ZIP 缺失时先执行 python download_data.py。
python train_ablation.py --output outputs/reproduce_ablation --seeds 42 43 44 --epochs 20
python compare_views.py --run outputs/reproduce_ablation
python analyze_ablation.py --run outputs/reproduce_ablation
python evaluate.py --checkpoint outputs/reproduce_ablation/views/selected.pt --output outputs/reproduce_ablation/verification.json
python -m pytest -q
~~~

- [训练代码](train_ablation.py)中的 `affine_batch` 在训练图上采样仿射变换，`train_pair` 在每步更新后维护 EMA 对照。划分种子与训练种子分别保存。
- [视图实验](compare_views.py)仅使用验证集选择邻域半径与权重。[推理读取](src/views.py)从保存文件恢复对应前向规则。
- [分析脚本](analyze_ablation.py)从权重重建各划分指标、消融/鲁棒性曲线、特征图、正确与错误变化图，并导出文件级证据。[结果目录](outputs/ablation/analysis/)
- [模型工厂](src/models.py)支持 `input_dim`、`num_classes`。改图像格式、标签和预处理的完整步骤见报告附录 B。
- 不指定 `--checkpoint` 的原有预测和评价命令仍读取 `outputs/main/cnn/best.pt`。复核本次结果请使用上方的明确权重路径。

## 1. 环境

```bash
conda activate ml-course
cd /home/ubuntu/Documents/ml-course/experiment1

# 新机器重建时执行；当前环境已安装则不必重复。
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Python 3.11；当前 GPU 使用 CUDA 12.4 版 PyTorch。CUDA 运行库由 wheel 安装，无需额外配置系统 CUDA Toolkit。CPU 可用，运行时指定 `--device cpu`。完整实测依赖保存在 `requirements-lock.txt`，环境摘要保存在 `environment.yml`。

安装命令来源：[PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v260)。

## 2. 数据

```bash
python download_data.py
```

脚本仅从[教师提供的北航网盘](https://bhpan.buaa.edu.cn/link/AAB14093C4DF1F4085922390C07C87498F)下载实验一：

```text
data/raw/1-Digit-TrainSet.zip    60,000 张，TrainingSet/*.bmp
data/raw/1-Digit-TestSet.zip     10,000 张，TestSet/*.bmp
```

图像均为 `28×28` 灰度图，黑底白字。标签取文件名的第一个字符，例如 `3_24.bmp → 3`。这里使用的是教师原始压缩包，没有用其他 MNIST 数据副本替代。直接读取 ZIP，不必解压 70,000 个文件。首次运行将生成 NumPy 缓存和数据清单。

若网盘失效，可通过教师 README 的其他入口手动下载这两个同名 ZIP，放入 `data/raw/`。下载脚本不会保存临时授权令牌。数据 SHA256 记录在 `data/raw/download_manifest.json` 和各次实验的 `data_manifest.json`。

## 3. 第一版基线与归档位置

本次固定种子实测（模型均按验证集选择）：

| 模型 | 优化子集准确率 | 验证准确率 | 完整训练集准确率 | 测试准确率 | 测试错误数 |
|---|---:|---:|---:|---:|---:|
| MLP（第 12 轮） | 99.87% | 98.37% | 99.72% | 98.47% | 153 / 10,000 |
| CNN（第 16 轮） | 99.97% | 99.45% | 99.92% | **99.50%** | **50 / 10,000** |

CNN 测试宏平均 F1 为 99.49%。以上为单次运行结果，不代表多次运行的平均值。独立重载 CNN 权重后已复核测试结果。

- 中文 PDF：`report/SY2631122徐基恒.pdf`
- 主模型权重：`outputs/main/cnn/best.pt`
- 对照模型权重：`outputs/main/mlp/best.pt`
- 全部结果：`outputs/main/summary.json`
- 实际执行日志：`outputs/training.log`
- 复核结果：`outputs/verification.json`
- 每个模型子目录包含：`history.csv`、`metrics.json`、`learning_curves.png`、`confusion_matrix.png`、`failure_examples.png`、训练和测试集的逐图预测 CSV。

`metrics.json` 的 `accuracy` 等值为 0～1 比例，报告中转为百分数。

## 4. 复现第一版 MLP 与 CNN

```bash
# 默认：MLP 15 轮、CNN 20 轮，seed=42，batch=128。
# 已有的结果目录不会被覆盖，复跑请使用新目录。
python run_experiment.py --output outputs/reproduce

# 用复跑结果生成报告；更换报告路径可保留原报告。
python make_report.py --run outputs/reproduce --output report/reproduce.pdf
```

训练流程：

1. 校验原始文件、图像数量、标签和尺寸；审计完全相同的像素样本。
2. 使用固定随机种子进行按类别分层划分：54,000 张用于参数优化、6,000 张验证；原始 10,000 张测试图保持不变。
3. 像素除以 255，再以 **54,000 张优化子集**的均值和标准差进行标准化。验证、测试、预测使用相同统计量。
4. 两个模型都使用交叉熵、Adam、权重衰减和余弦学习率调度。不使用数据增强或预训练权重。
5. 每轮仅检查验证集。按验证准确率最高选择 checkpoint，同分时选择验证损失较低者。
6. 两个模型都训练完毕后才进行最终测试。测试结果不用于调整超参数或重新选择 checkpoint。

训练曲线中的训练指标为训练过程中启用 dropout 的逐批累计值；最终表格中的训练指标为最佳 checkpoint 在 `eval()` 模式下重新计算的值。因此，训练曲线低于验证曲线并不必然代表数据异常。

`train_full` 是教师原始 60,000 张训练数据（包括验证子集）的最终评估；`fit` 是实际用于梯度优化的 54,000 张；`validation` 是 6,000 张验证图。三者在报告中明确区分。

## 5. 第一版独立评价与单图预测

```bash
# 重载最佳 CNN，重新计算全部 10,000 张测试图的指标。
python evaluate.py --checkpoint outputs/main/cnn/best.pt

# 直接试运行：附带一张来自教师测试集的数字 7 图片。
python predict.py examples/7_0.bmp

# 识别自己的黑底白字数字图片。
python predict.py /absolute/path/to/digit.bmp

# 白底黑字时可以反色；本项目未针对拍照场景进行训练。
python predict.py /absolute/path/to/digit.png --invert

# 生成身份信息明确的报告。
python make_report.py --student-id SY2631122 --name 徐基恒
```

预测脚本会转灰度，并在需要时调整为 28×28，不会自动做纸张检测、数字定位或复杂去背景。建议输入与训练数据相似的居中单数字图片。

## 6. 文件分工

| 文件 | 作用 |
|---|---|
| `download_data.py` | 下载、校验教师原始数据 |
| `src/data.py` | BMP 解析、标签提取、划分、标准化统计和缓存 |
| `src/models.py` | MLP 与 CNN 网络结构 |
| `run_experiment.py` | 训练、验证选择、最终评价 |
| `src/evaluation.py` | 指标、预测导出与图表 |
| `evaluate.py` | 独立加载权重复核测试结果 |
| `predict.py` | 单张图像预测 |
| `make_report.py` | 根据实际结果生成中文 PDF |
| `report/analysis.json` | 对实际失败案例的人工观察说明 |
| `tests/test_pipeline.py` | 数据标签、输入校验、指标与权重往返检查 |

```bash
python -m pytest -q
```

报告生成需要一个包含中英文字形的 TrueType 中文字体。默认使用系统文鼎宋体；可通过 `--font /path/to/font.ttf` 指定其他字体。

## 7. 第一版提交包

教师要求：PDF 报告以“学号＋姓名”命名，并提交模型代码。`submission.zip` 包含报告、源码、依赖、两个模型权重和关键结果，不包含原始数据集。报告已生成到本地，未上传课程网盘。

打包脚本用于重建第一版 PDF 提交包。本轮结果以目录顶部的 Markdown 报告、其相对路径插图及新权重为准。

官方 API 参考：[CrossEntropyLoss](https://docs.pytorch.org/docs/2.6/generated/torch.nn.CrossEntropyLoss.html)。网络输出 logits，直接交给交叉熵；仅在预测和评价时转为 softmax 概率。
