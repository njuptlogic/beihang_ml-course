"""从实测 JSON 和图表生成可提交的中文 PDF，不填造实验结果。"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak)

from src.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/main")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--student-id", default="SY2631122")
    parser.add_argument("--name", default="徐基恒")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config, data, models = summary["config"], summary["data"], summary["models"]
    if not {"mlp", "cnn"}.issubset(models):
        parser.error("完整报告需要同一次运行中的 MLP 和 CNN 结果")
    cnn, mlp = models["cnn"], models["mlp"]
    cm = np.array(cnn["metrics"]["test"]["confusion_matrix"])
    output = args.output or ROOT / "report" / f"{args.student_id}{args.name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("Chinese", str(args.font)))
    navy, blue, gray = colors.HexColor("#17324d"), colors.HexColor("#267ba0"), colors.HexColor("#526475")
    styles = {
        "body": ParagraphStyle("body", fontName="Chinese", fontSize=10.2, leading=16.5,
                               textColor=navy, spaceAfter=7, wordWrap="CJK"),
        "h1": ParagraphStyle("h1", fontName="Chinese", fontSize=18, leading=25,
                             textColor=navy, spaceAfter=13),
        "h2": ParagraphStyle("h2", fontName="Chinese", fontSize=12.5, leading=19,
                             textColor=blue, spaceBefore=8, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="Chinese", fontSize=8.7, leading=13.2,
                                textColor=gray, spaceAfter=6, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName="Chinese", fontSize=25, leading=34,
                                textColor=navy, spaceAfter=15),
        "cell": ParagraphStyle("cell", fontName="Chinese", fontSize=8.5, leading=12,
                               textColor=navy, wordWrap="CJK"),
    }
    story = []
    width = A4[0] - 96
    def p(text, style="body"):
        story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style]))
    def title(text): p(text, "h1")
    def table(rows, widths=None):
        cells = [[Paragraph(escape(str(c)), styles["cell"]) for c in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeef4")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), .7, blue),
        ]))
        story.extend([t, Spacer(1, 9)])
    def fig(path, caption, w=width):
        with PILImage.open(path) as im: h = w * im.height / im.width
        story.append(Image(str(path), width=w, height=h))
        p(caption, "small")
    def pct(value): return f"{value * 100:.2f}%"
    def page(): story.append(PageBreak())

    p("机器学习基础 · 实验一", "h2")
    p("手写数字识别实验报告", "title")
    p(f"学号：{args.student_id}    姓名：{args.name}")
    p(f"实验日期：{config['started_at'][:10]}    主模型：卷积神经网络（CNN）")
    p("1  问题描述与实验目标", "h2")
    p("本实验对 28×28 灰度手写数字图像进行十分类，预测 0～9 的类别。"
      "使用教师指定的训练集进行模型学习，在独立测试集上评价性能。"
      "以 MLP 为对照，研究卷积结构对局部笔画与空间关系建模的效果。")
    test = cnn["metrics"]["test"]
    p(f"实测结果：CNN 在 10,000 张测试图像上正确分类 {test['correct']:,} 张，"
      f"准确率为 {pct(test['accuracy'])}，错误 {test['errors']} 张；"
      f"MLP 测试准确率为 {pct(mlp['metrics']['test']['accuracy'])}。所有结果来自保存的最佳验证集 checkpoint。")
    p("2  数据来源与标签", "h2")
    table([["数据", "图像数", "格式与标签"],
           ["教师训练集", "60,000", "28×28 灰度 BMP；文件名首字符为类别"],
           ["教师测试集", "10,000", "同上；不用于参数更新与模型选择"],
           ["内部划分", "54,000 / 6,000", "按类别分层划分为优化子集 / 验证集"]],
          [100, 100, width-200])
    p("原始压缩包从教师 README 提供的北航网盘下载。实际检查文件数、尺寸和标签，"
      "没有以其他来源的数据副本替代。模型输入为黑底白字图像。")
    fig(args.run / "data_examples.png", "图 1  原始训练集中每一类的两个样本。")
    page()

    title("3  模型原理、结构与参数")
    p("MLP 将图像展平后通过全连接层学习非线性分类边界；它不显式利用像素的邻域结构。"
      "CNN 用共享卷积核提取局部笔画，通过池化减少空间尺寸，并用多层特征组合形成分类表示。")
    p("3.1  网络结构", "h2")
    table([["CNN 层", "输出形状（省略 batch）", "设置"],
           ["输入", "1×28×28", "灰度、标准化"],
           ["Conv + ReLU ×2", "32×28×28", "1→32→32；3×3，padding=1"],
           ["MaxPool", "32×14×14", "2×2，stride=2"],
           ["Conv + ReLU ×2", "64×14×14", "32→64→64；3×3，padding=1"],
           ["MaxPool + Flatten", "3,136", "2×2 池化后展平"],
           ["Linear + ReLU + Dropout", "128", "Dropout=0.5"],
           ["Linear", "10", "输出未归一化的类别分数 logits"]], [145, 125, width-270])
    p(f"CNN 可训练参数量：{cnn['parameters']:,}。MLP 结构为 784→256→128→10，"
      f"两个隐藏层均使用 ReLU 和 Dropout(0.2)，参数量为 {mlp['parameters']:,}。")
    p("3.2  损失与训练设置", "h2")
    p("采用多类别交叉熵：L = -(1/N) Σ log p(y_i|x_i)。训练时直接将 logits 和整数类别标签"
      "交给 CrossEntropyLoss，不在损失前额外使用 softmax。Dropout 在训练时随机屏蔽部分隐藏单元，"
      "评价时关闭；权重衰减用于约束参数规模。")
    table([["配置项", "实际设置"],
           ["优化器 / 初始学习率", f"Adam / {config['lr']}"],
           ["权重衰减 / batch size", f"{config['weight_decay']} / {config['batch_size']}"],
           ["学习率调度", "余弦退火，最低学习率 0.00001"],
           ["训练轮数", f"MLP {mlp['epochs']} 轮；CNN {cnn['epochs']} 轮"],
           ["随机种子 / 数据增强", f"{config['seed']} / 无"],
           ["预训练 / 实验环境", f"无预训练；Python {config['python']}，PyTorch {config['torch']}"],
           ["训练设备", config["gpu"] or config["device"]]], [145, width-145])
    page()

    title("4  实验流程与训练过程")
    p("像素先除以 255，再进行 (x-μ)/σ 标准化。均值和标准差仅从 54,000 张优化子集计算："
      f"μ={data['mean']:.6f}，σ={data['std']:.6f}。同一变换用于验证集和测试集。")
    p("两种模型使用同一分层划分。每轮根据验证准确率选择模型，若准确率相同则选择验证交叉熵较低者。"
      "所有训练完成后统一评价测试集，没有使用测试准确率调整模型或超参数。")
    p(f"最终选择：MLP 第 {mlp['best_epoch']} 轮，CNN 第 {cnn['best_epoch']} 轮。"
      f"训练耗时分别为 {mlp['training_seconds'] / 60:.2f} 和 {cnn['training_seconds'] / 60:.2f} 分钟，"
      "不含安装依赖、下载数据及最终报告生成时间。")
    fig(args.run / "mlp/learning_curves.png", "图 2  MLP 的交叉熵与准确率曲线。")
    fig(args.run / "cnn/learning_curves.png", "图 3  CNN 的交叉熵与准确率曲线。")
    p(f"MLP 验证准确率由首轮的 {pct(mlp['history'][0]['val_accuracy'])} 提升至最佳 "
      f"{pct(mlp['metrics']['validation']['accuracy'])}；CNN 最佳验证准确率为 "
      f"{pct(cnn['metrics']['validation']['accuracy'])}。后期在线训练准确率仍在提高，"
      "验证表现改善较小，说明继续拟合训练样本不一定带来同等的泛化收益。", "small")
    p("训练曲线统计的是启用 Dropout 的逐批在线指标；验证指标是在每轮结束后关闭 Dropout 计算的。"
      "因此二者不是完全相同的评估条件，验证准确率高于在线训练准确率并不矛盾。最终结果均以 eval 模式复算。", "small")
    page()

    title("5  实验结果与对照分析")
    rows = [["模型", "评估数据", "准确率", "交叉熵", "错误数"]]
    for name in ("mlp", "cnn"):
        for key, label in (("fit", "优化子集 54,000"), ("validation", "验证集 6,000"),
                           ("train_full", "完整训练集 60,000"), ("test", "测试集 10,000")):
            m = models[name]["metrics"][key]
            rows.append([name.upper(), label, pct(m["accuracy"]), f"{m['cross_entropy']:.5f}", m["errors"]])
    table(rows, [50, 155, 80, 80, width-365])
    p("完整训练集结果包含内部验证子集，不能视为完全未见数据上的泛化结果；测试集独立于训练与模型选择。", "small")
    delta = (cnn["metrics"]["test"]["accuracy"] - mlp["metrics"]["test"]["accuracy"]) * 100
    p(f"本次运行 CNN 相比 MLP 的测试准确率变化为 {delta:+.2f} 个百分点。"
      "这一对照反映了本次结构与训练配置的整体差异；由于参数量、正则化和轮数也不同，不能把差异完全归因于卷积操作。")
    p("5.1  CNN 各类别测试结果", "h2")
    rows = [["数字", "Precision", "Recall", "F1", "样本数"]]
    for label in range(10):
        m = test["classification_report"][str(label)]
        rows.append([label, pct(m["precision"]), pct(m["recall"]), pct(m["f1-score"]), int(m["support"])])
    rows.append(["宏平均", pct(test["macro_precision"]), pct(test["macro_recall"]), pct(test["macro_f1"]), "10,000"])
    table(rows, [55, 110, 110, 110, width-385])
    p("Precision 衡量预测为该数字的样本中有多少正确；Recall 衡量该类真实样本中有多少被找出；"
      "F1 为两者的调和平均。宏平均对各类等权，能补充总体准确率。", "small")
    page()

    title("5.2  混淆矩阵与常见错误方向")
    fig(args.run / "cnn/confusion_matrix.png", "图 4  CNN 测试混淆矩阵。行是真实数字，列是预测数字，对角线表示预测正确。", 440)
    off_diagonal = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j]]
    top = sorted(off_diagonal, key=lambda t: (-t[0], t[1], t[2]))[:6]
    table([["真实 → 预测", "错误数", "占该类测试样本的比例"]] +
          [[f"{i} → {j}", n, pct(n / cm[i].sum())] for n, i, j in top], [150, 100, width-250])
    p("上述统计按真实类别条件化，可定位召回率下降的主要原因。"
      "笔画闭合程度、倾斜、局部缺失和不同书写习惯可能使不同类别在低分辨率下呈现相似轮廓，"
      "具体原因需结合原始图片观察，不能仅由矩阵确定。")
    page()

    title("5.3  失败案例与观察")
    fig(args.run / "cnn/failure_examples.png", "图 5  CNN 的测试错误，按错误预测的置信度从高到低展示；标题包含原文件名、真值、预测与置信度。")
    analysis_path = ROOT / "report/analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    if analysis.get("run_started_at") == config["started_at"]:
        for item in analysis.get("observations", []):
            p(f"{item['filename']}：{item['analysis']}")
    else:
        p("上图为真实错误样本。可从笔画交叉、闭环缺失、倾斜与类别轮廓相似性等方面逐一观察。"
          "自动生成报告不对未人工检查的样本写入具体视觉结论。")
    p("展示的样本按置信度选择，属于刻意挑出的错误子集，不能代表全部测试图像。"
      "高置信度错误说明 softmax 最大值不能直接作为可靠性的保证；本实验没有执行概率校准。", "small")
    page()

    title("6  总结、局限与复现")
    p(f"本实验完成了课程 BMP 数据的读取、分层划分、MLP/CNN 训练、验证集模型选择及测试评价。"
      f"主模型 CNN 测试准确率为 {pct(test['accuracy'])}，宏平均 F1 为 {pct(test['macro_f1'])}。"
      "逐图预测、模型权重和训练日志均保存，可从结果表追溯到原始样本。")
    p("6.1  局限与可尝试的改进", "h2")
    p("本次为固定种子的单次运行，没有多随机种子方差或显著性检验。训练图像为已裁剪、居中的黑底白字数字，"
      "不能据此保证手机照片、复杂背景或连写数字上的性能。")
    p("后续可在验证集上比较轻微平移、旋转或弹性形变增强，观察对倾斜与不规则笔画的影响；"
      "也可比较不同网络规模与正则化设置，并进行多种子重复实验。上述改进尚未在本次运行中实施。")
    overlap = data["pixel_overlap"]
    p("6.2  数据完整性审计", "h2")
    p(f"按灰度像素计算 SHA256：训练集内部重复像素数为 {overlap['train_duplicate_pixels']}，"
      f"测试集内部为 {overlap['test_duplicate_pixels']}；验证图中与优化子集完全相同的样本数为 "
      f"{overlap['validation_samples_identical_to_fit']}；测试图中与原始训练集完全相同的样本数为 "
      f"{overlap['test_samples_identical_to_train']}。本实验保留教师原始训练/测试边界。该检查仅覆盖完全相同像素，"
      "不排除近似图像或同一书写者关联。")
    p("6.3  运行命令与产物", "h2")
    p("conda activate ml-course\ncd /home/ubuntu/Documents/ml-course/experiment1\n"
      "python run_experiment.py --output outputs/reproduce\n"
      "python evaluate.py --checkpoint outputs/main/cnn/best.pt\n"
      "python predict.py /absolute/path/to/digit.bmp", "small")
    p("报告生成脚本 make_report.py 读取 summary.json 和实际图表；原实验结果目录默认受保护，不会被复跑覆盖。"
      "源代码、依赖锁定文件、两种模型的最佳权重与主要结果均随提交包保存。")
    p("参考与来源", "h2")
    p("[1] 教师课程仓库 Machine-Learning-Class-Assignment26/README.md，实验一任务及报告要求。\n"
      "[2] 教师公开数据：\nhttps://bhpan.buaa.edu.cn/link/AAB14093C4DF1F4085922390C07C87498F\n"
      "[3] PyTorch 官方安装说明：https://pytorch.org/get-started/previous-versions/\n"
      "[4] PyTorch 2.6 CrossEntropyLoss 文档：\nhttps://docs.pytorch.org/docs/2.6/generated/torch.nn.CrossEntropyLoss.html", "small")

    def decorate(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#d8e4eb"))
        canvas.line(48, 39, A4[0]-48, 39)
        canvas.setFont("Chinese", 8); canvas.setFillColor(gray)
        canvas.drawString(48, 25, f"{args.student_id}  {args.name}  |  手写数字识别")
        canvas.drawRightString(A4[0]-48, 25, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                            topMargin=42, bottomMargin=51,
                            title="实验一：手写数字识别", author=f"{args.student_id} {args.name}")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"已生成报告：{output}")


if __name__ == "__main__":
    main()
