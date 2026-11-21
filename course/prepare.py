"""Make compact RGB and density arrays; retain original archive paths for native evaluation."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from course.common import ROOT, SPLIT_SEED, dump, table


def resize(blob, mode, size):
    with Image.open(BytesIO(blob)) as im:
        original = im.size
        # Decode a near-native JPEG pyramid level before the final area reduction.
        im.draft(mode, (size[0] * 2, size[1] * 2))
        return (
            np.asarray(im.convert(mode).resize(size, Image.Resampling.LANCZOS)),
            original,
        )


def prepare(number):
    root = ROOT / f"experiment{number}" / "data"
    images, targets, rows = [], [], []
    for part, expected in zip(
        ("Train", "Test"), {2: (1639, 250), 3: (1600, 400)}[number]
    ):
        (archive,) = (root / "raw").glob(f"*{part}Set.zip")
        with ZipFile(archive) as z:
            jpgs = {
                n
                for n in z.namelist()
                if n.lower().endswith(".jpg") and "__MACOSX" not in n
            }
            names = sorted(n for n in jpgs if number == 2 or "/Stimuli/" in n)
            assert len(names) == expected
            if number == 3:
                assert {
                    n.replace("/Stimuli/", "/FIXATIONMAPS/") for n in names
                } == jpgs - set(names)
            for name in names:
                blob = z.read(name)
                image, wh = resize(
                    blob, "RGB", (224, 224) if number == 2 else (320, 192)
                )
                images.append(image)
                category = Path(name).parent.name
                if number == 2:
                    assert category in ("disease", "normal") and Path(
                        name
                    ).name.startswith(category)
                else:
                    gt, gt_wh = resize(
                        z.read(name.replace("/Stimuli/", "/FIXATIONMAPS/")),
                        "L",
                        (160, 90),
                    )
                    assert wh == gt_wh and gt.max() > 0
                    targets.append(gt)
                rows.append(
                    dict(
                        index=len(rows),
                        archive=archive.name,
                        member=name,
                        category=category,
                        split=part.lower(),
                        width=wh[0],
                        height=wh[1],
                        sha256=hashlib.sha256(blob).hexdigest(),
                    )
                )
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    fit, val = train_test_split(
        tr,
        test_size=0.15,
        random_state=SPLIT_SEED,
        stratify=[rows[i]["category"] for i in tr],
    )
    for ids, split in ((fit, "fit"), (val, "validation")):
        for i in ids:
            rows[i]["split"] = split
    np.save(root / "images.npy", np.stack(images))
    if targets:
        np.save(root / "targets.npy", np.stack(targets))
    table(root / "images.csv", rows)
    fit_hash = {r["sha256"] for r in rows if r["split"] == "fit"}
    dump(
        root / "cohort.json",
        dict(
            counts=dict(Counter(r["split"] for r in rows)),
            identical_fit_test=sum(
                r["sha256"] in fit_hash for r in rows if r["split"] == "test"
            ),
            split_seed=SPLIT_SEED,
        ),
    )
    print(number, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=int, choices=[2, 3])
    prepare(p.parse_args().experiment)
