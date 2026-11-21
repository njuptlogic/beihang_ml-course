"""TensorBoard 实时曲线与终端逐轮摘要。"""
from contextlib import ExitStack, contextmanager
from pathlib import Path


def add_monitor_args(parser):
    parser.add_argument("--log-every", type=int, default=50, metavar="N",
                        help="每 N 个 batch 向 TensorBoard 记录一次训练指标（首、末批也会记录）")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="仅记录 TensorBoard 日志并输出逐轮摘要")


def start_dashboard(output):
    from tensorboard import program

    logdir = (Path(output) / "tensorboard").resolve()
    logdir.mkdir(exist_ok=True)
    board = program.TensorBoard()
    board.configure(argv=["tensorboard", "--logdir", str(logdir), "--host", "127.0.0.1",
                          "--reload_interval", "1", "--load_fast", "false"])
    try:
        url = board.launch()
    except (OSError, program.TensorBoardServerException) as exc:
        print(f"本地看板启动失败（{exc}）。事件目录 {logdir}", flush=True)
    else:
        print(f"TensorBoard {url}  |  浏览器自动刷新：Settings → Reload data\n"
              f"事件目录 {logdir}", flush=True)


@contextmanager
def monitor_run(output, name, interval, *, ema=False):
    from torch.utils.tensorboard import SummaryWriter

    splits = ("train", "val", "ema_val") if ema else ("train", "val")
    with ExitStack() as stack:
        writers = {}
        for split in splits:
            directory = Path(output) / "tensorboard" / name / split
            writers[split] = stack.enter_context(SummaryWriter(str(directory), flush_secs=1))
        yield TrainingMonitor(name, interval, writers)


class TrainingMonitor:
    def __init__(self, name, interval, writers):
        self.name = name
        self.interval = interval
        self.writers = writers

    def batch(self, epoch, batch, batches, loss, correct, size, lr):
        if batch not in (1, batches) and batch % self.interval:
            return
        loss_value = loss.item()
        accuracy = correct.item() * 100 / size
        step = (epoch - 1) * batches + batch
        writer = self.writers["train"]
        writer.add_scalar("batch/loss", loss_value, step)
        writer.add_scalar("batch/accuracy_pct", accuracy, step)
        writer.add_scalar("batch/learning_rate", lr, step)

    def perturbation(self, epoch, split, condition, loss, accuracy, clean_accuracy):
        writer = self.writers[split]
        prefix = f"robustness/{condition}"
        writer.add_scalar(f"{prefix}/accuracy_pct", 100 * accuracy, epoch)
        writer.add_scalar(f"{prefix}/loss", loss, epoch)
        writer.add_scalar(f"{prefix}/drop_pp", 100 * (clean_accuracy - accuracy), epoch)

    def epoch(self, row):
        epoch = row["epoch"]
        metrics = []
        for split, writer in self.writers.items():
            loss = row[split + "_loss"]
            accuracy = row[split + "_accuracy"]
            writer.add_scalar("epoch/loss", loss, epoch)
            writer.add_scalar("epoch/accuracy_pct", 100 * accuracy, epoch)
            metrics.append(f"{split} loss={loss:.5f} acc={accuracy:.4%}")
        self.writers["train"].add_scalar("epoch/seconds", row["seconds"], epoch)
        for writer in self.writers.values():
            writer.flush()
        print(f"{self.name} epoch {epoch}  " + "  |  ".join(metrics)
              + f"  |  lr={row['lr']:.2e}  {row['seconds']:.1f}s", flush=True)
