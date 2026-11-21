"""打包报告、源码、权重与关键结果；排除原始数据及临时缓存。"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    required = [ROOT / "report/SY2631122徐基恒.pdf", ROOT / "outputs/main/summary.json",
                ROOT / "outputs/main/cnn/best.pt", ROOT / "outputs/main/mlp/best.pt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"尚未完成的必要产物: {path}")
    files = [*ROOT.glob("*.py"), *ROOT.glob("*.txt"), ROOT / "README.md", ROOT / "environment.yml"]
    for folder in ("src", "tests", "report", "examples", "outputs/main"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for name in ("outputs/training.log", "outputs/verification.json", "outputs/verification_checks.json",
                 "data/raw/download_manifest.json"):
        if (ROOT / name).exists(): files.append(ROOT / name)
    archive_path = ROOT / "submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("SY2631122徐基恒实验1") / path.relative_to(ROOT))
    print(f"已打包：{archive_path} ({archive_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
