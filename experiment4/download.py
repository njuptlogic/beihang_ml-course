"""Verify the CIFAR-10 Python distribution before extracting it."""
import hashlib
from pathlib import Path
import requests
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import extract_archive
from course.common import ROOT, dump


def main():
    root = ROOT / "experiment4/data"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / CIFAR10.filename
    source = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"

    def valid(path):
        if not path.is_file():
            return False
        with path.open("rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest() == CIFAR10.tgz_md5

    if not valid(archive):
        temp = archive.with_suffix(".part")
        with requests.get(source, timeout=(20, 120), stream=True) as response:
            response.raise_for_status()
            with temp.open("wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
        if not valid(temp):
            raise ValueError("CIFAR-10 archive checksum mismatch")
        temp.replace(archive)
    try:
        CIFAR10(root, train=True)
        CIFAR10(root, train=False)
    except RuntimeError:
        extract_archive(str(archive), str(root))
        CIFAR10(root, train=True)
        CIFAR10(root, train=False)
    with archive.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    dump(
        root / "download_manifest.json",
        dict(
            dataset="CIFAR-10 Python",
            mirror=source,
            official=CIFAR10.url,
            md5=CIFAR10.tgz_md5,
            sha256=sha,
            bytes=archive.stat().st_size,
            train=50000,
            test=10000,
        ),
    )


if __name__ == "__main__":
    main()
