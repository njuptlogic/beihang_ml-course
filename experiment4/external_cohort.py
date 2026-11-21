"""Evaluate locked CIFAR generators against two equally sized real cohorts.

From the project root, run `python -m experiment4.external_cohort ACTION`
with the sequential actions download, extract, score, verify, figure. The
separately selected spectral correction uses freeze-supplement, supplement,
verify-supplement after the main scores and spectral selection are available.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/cifar10_1"
OUT = BASE / "outputs/external_cohort"
REF = BASE / "outputs/evaluation"
GUIDE = BASE / "outputs/guidance"
COMMIT = "d9982abb0bfc4846b8d13a11e66b887d946205d0"
RAW = f"https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/{COMMIT}/datasets/"
FILES = {
    "data": (
        6144128,
        "24349cc5f3fb59648d75cd8015b5a184b1f5c250",
        "2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6",
    ),
    "labels": (
        8128,
        "e713b09cecf2ecd933970001909fcdec7c907ab0",
        "ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448",
    ),
}
N = 2000
R = 1000
SEED = 31091
CLASSES = (
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)
NAMES = {
    "train_real": "Real training images",
    "time32": "32 epochs · time / strong",
    "extended44": "44 epochs · joint / standard",
    "quality8": "Selected · 8 evaluations",
    "quality24": "Selected · 24 evaluations",
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def checked(name):
    path = DATA / f"cifar10.1_v6_{name}.npy"
    size, git_sha1, digest = FILES[name]
    if not path.exists() or path.stat().st_size != size or sha(path) != digest:
        return False
    image = path.read_bytes()
    return (
        hashlib.sha1(b"blob " + str(size).encode() + b"\0" + image).hexdigest()
        == git_sha1
    )


def download():
    import requests

    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (size, git_sha, digest) in FILES.items():
        file = f"cifar10.1_v6_{name}.npy"
        url = RAW + file
        path = DATA / file
        if not checked(name):
            step = 256 * 1024

            def chunk(start):
                end = min(start + step, size) - 1
                for attempt in range(5):
                    try:
                        response = requests.get(
                            url,
                            params={"r": start, "retry": attempt},
                            headers={"Range": f"bytes={start}-{end}"},
                            timeout=(15, 45),
                        )
                        response.raise_for_status()
                        blob = response.content
                        if len(blob) == size:
                            return blob[start : end + 1]
                        if (
                            len(blob) == end - start + 1
                            and response.headers.get("Content-Range")
                            == f"bytes {start}-{end}/{size}"
                        ):
                            return blob
                        raise IOError("Incomplete range")
                    except (requests.RequestException, IOError):
                        if attempt == 4:
                            raise

            with ThreadPoolExecutor(6) as pool:
                blob = b"".join(pool.map(chunk, range(0, size, step)))
            if hashlib.sha256(blob).hexdigest() != digest:
                raise IOError("Checksum mismatch for " + file)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(blob)
            tmp.replace(path)
        assert checked(name)
        manifest[name] = dict(
            file=file, url=url, git_blob=git_sha, sha256=digest, bytes=size
        )
    y = np.load(DATA / "cifar10.1_v6_labels.npy")
    x = np.load(DATA / "cifar10.1_v6_data.npy")
    assert (
        x.shape == (2000, 32, 32, 3)
        and x.dtype == np.uint8
        and np.array_equal(np.bincount(y), np.full(10, 200))
    )
    write_json(
        DATA / "manifest.json",
        dict(
            commit=COMMIT,
            release="v6",
            n=len(x),
            class_counts=np.bincount(y).tolist(),
            files=manifest,
        ),
    )


def originals(split):
    files = (
        ["test_batch"] if split == "test" else [f"data_batch_{i}" for i in range(1, 6)]
    )
    for f in files:
        with open(BASE / "data/cifar-10-batches-py" / f, "rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        yield batch[b"data"].reshape(-1, 3, 32, 32), np.asarray(batch[b"labels"])


def locked_sources():
    lockfile = GUIDE / "selection/locked.json"
    lock = json.loads(lockfile.read_text())
    research = BASE / "outputs/research/evaluation"
    paths = {
        "time32": research / "time_strong",
        "extended44": research / "extended_12",
        "quality8": GUIDE / "test" / lock["nfe8_quality"],
        "quality24": GUIDE / "test" / lock["nfe24_quality"],
    }
    rows = {}
    for name, path in paths.items():
        m = json.loads((path / "metrics.json").read_text())
        assert (
            m["scenario"] == path.name
            and m["samples"] == 10000
            and m["noise_seed"] == 109
        )
        if name.startswith("quality"):
            assert m["budget"] == int(name[7:])
        cases = pd.read_csv(path / "per_sample.csv", usecols=["condition", "sample"])
        assert np.array_equal(cases["sample"], np.arange(10000)) and np.array_equal(
            cases["condition"], np.arange(10000) % 10
        )
        rows[name] = dict(
            scenario=path.name,
            checkpoint_sha256=m["checkpoint_sha256"],
            feature_sha256=sha(path / "features.npy"),
            source=str((path / "features.npy").relative_to(BASE)),
        )
    return {k: p / "features.npy" for k, p in paths.items()}, dict(
        selection_file=str(lockfile.relative_to(BASE)),
        selection_sha256=sha(lockfile),
        models=rows,
    )


def encode_external(device, batch):
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from course.common import configure

    download()
    _, lock = locked_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "locked_models.json", lock)
    configure(42)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = (
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval().to(device)
    )

    @torch.inference_mode()
    def encode(a):
        parts = []
        for i in range(0, len(a), batch):
            parts.append(
                model(
                    torch.as_tensor(np.ascontiguousarray(a[i : i + batch])).to(device)
                )[0]
                .cpu()
                .numpy()
            )
        return np.concatenate(parts)

    ext = np.load(DATA / "cifar10.1_v6_data.npy").transpose(0, 3, 1, 2)
    feats = encode(ext)
    assert (
        feats.shape == (2000, 2048)
        and feats.dtype == np.float32
        and np.isfinite(feats).all()
    )
    np.save(OUT / "v6_inception.npy", feats)

    def fingerprint(im):
        return hashlib.sha256(im.tobytes()).digest()

    hashes = {fingerprint(im) for im in ext}
    overlap = {}
    for split in ("train", "test"):
        intersect = 0
        for pixels, labels in originals(split):
            intersect += sum(fingerprint(im) in hashes for im in pixels)
        overlap[split] = intersect
        if split == "test":
            ids = np.linspace(0, len(pixels) - 1, 24, dtype=int)
            old = np.load(REF / "real_test_features.npy", mmap_mode="r")[ids]
            new = encode(pixels[ids])
            err = np.abs(old - new)
            cos = (old * new).sum(1) / (
                np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
            )
            assert cos.min() > 0.99995
    write_json(
        OUT / "encoder.json",
        dict(
            extractor="torch-fidelity 0.3.0 inception-v3-compat, raw uint8 NCHW, 2048 pooled FP32",
            feature_sha256=sha(OUT / "v6_inception.npy"),
            device=device,
            batch_size=batch,
            feature_shape=list(feats.shape),
            external_unique_pixels=len(hashes),
            exact_rgb_overlap_with_original=overlap,
            original_test_reencoding=dict(
                n=len(ids),
                max_abs_error=float(err.max()),
                mean_abs_error=float(err.mean()),
                min_cosine=float(cos.min()),
            ),
        ),
    )
    print(
        "Encoded 2,000 CIFAR-10.1 images, archive comparison cosine",
        cos.min(),
        flush=True,
    )


def split_class(labels, groups, seed):
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(groups)]
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(labels == c))
        assert len(ids) >= 200 * groups
        for k in range(groups):
            out[k].extend(ids[k * 200 : (k + 1) * 200])
    return np.asarray(out, dtype=np.int64)


def counts(seed):
    # Every observation set is stored in class-major order. Resample within
    # each of its 10 strata so that the class frequencies stay exactly fixed.
    rng = np.random.default_rng(seed)
    weights = np.empty((R, N), dtype=np.float32)
    for j in range(10):
        weights[:, 200 * j : 200 * (j + 1)] = rng.multinomial(
            200, np.full(200, 1 / 200), size=R
        )
    return weights


def kernel(x, y):
    # Subtracting the constant kernel improves float32 accumulation and cancels
    # exactly from MMD. The input to the cubic kernel is the raw Inception vector.
    value = x @ y.T
    value /= x.shape[1]
    value += 1.0
    np.power(value, 3, out=value)
    value -= 1.0
    return value


def within(k):
    return (k.sum(dtype=np.float64) - np.trace(k, dtype=np.float64)) / (
        len(k) * (len(k) - 1)
    )


def bootstrap_term(k, left, right, same):
    n = len(k)
    result = np.empty(len(left), dtype=np.float64)
    diag = np.diag(k).copy() if same else None
    for start in range(0, len(left), 125):
        a = left[start : start + 125]
        b = right[start : start + 125]
        v = np.sum((a @ k) * b, axis=1, dtype=np.float64)
        if same:
            v -= np.sum(a * diag, axis=1, dtype=np.float64)
        result[start : start + len(a)] = v / (n * (n - 1) if same else n * n)
    return result


def centered_interval(theta, draws):
    # The bootstrap expectation of a quadratic distance on the empirical
    # distribution contains a diagonal term. Centre the sampling fluctuations.
    lo, hi = np.quantile(draws - draws.mean(), [0.025, 0.975])
    return float(theta + lo), float(theta + hi)


def label_arrays():
    test = np.concatenate([y for _, y in originals("test")])
    train = np.concatenate([y for _, y in originals("train")])
    ref = np.load(REF / "reference_indices.npz")
    assert np.array_equal(ref["test"], np.arange(10000))
    return test, train[ref["train"]]


def supplement_lock():
    spectral = BASE / "outputs/spectral_guidance"
    selected = json.loads((spectral / "selection.json").read_text())["selected"]
    base = json.loads((spectral / "protocol.json").read_text())["base"]
    assert (
        base
        == json.loads((GUIDE / "selection/locked.json").read_text())["nfe8_quality"]
    )
    return dict(
        relationship="later, separately locked paired comparison; not a member of the four-model primary table",
        selected=selected["name"],
        base=base,
        strength=selected["strength"],
        selection_sha256=sha(spectral / "selection.json"),
        train_protocol_sha256=sha(spectral / "protocol.json"),
        primary_models_sha256=sha(OUT / "locked_models.json"),
        indices_sha256=sha(OUT / "indices.npz"),
        bootstrap_sha256=sha(OUT / "bootstrap.npz"),
        generator_features="outputs/spectral_guidance/test/features.npy",
        contrast="spectral8 minus quality8",
        endpoints=["original", "external"],
        conditioning="same references, generator partitions, paired class-resamples, fixed checkpoints as primary",
    )


def supplement_source():
    """Verify the follow-up freeze before accessing the additional feature matrix."""
    lock = json.loads((OUT / "supplement_lock.json").read_text())
    assert lock == supplement_lock()
    path = BASE / lock["generator_features"]
    meta = json.loads((path.parent / "metrics.json").read_text())
    base_dir = GUIDE / "test" / lock["base"]
    base = json.loads((base_dir / "metrics.json").read_text())
    assert meta["selected"] and meta["name"] == lock["selected"]
    assert meta["base"] == lock["base"] and meta["strength"] == lock["strength"]
    assert (
        meta["n"] == base["samples"] == 10000 and meta["budget"] == base["budget"] == 8
    )
    assert meta["checkpoint_sha256"] == base["checkpoint_sha256"]
    assert meta["noise_seed"] == base["noise_seed"] == 109
    assert meta["base_pixel_exact"] == meta["corrected_pixel_exact"] == 64
    paired = pd.read_csv(path.parent / "per_sample.csv")
    original = pd.read_csv(base_dir / "per_sample.csv")
    for col in ("condition", "sample"):
        assert np.array_equal(paired[col], original[col])
    assert np.array_equal(paired.base_prediction, original.probe_prediction)
    return {"spectral8": path}


def measure(supplement=False):
    prefix = "supplement_" if supplement else ""
    sources, lock = locked_sources()
    assert lock == json.loads((OUT / "locked_models.json").read_text())
    assert (
        sha(OUT / "v6_inception.npy")
        == json.loads((OUT / "encoder.json").read_text())["feature_sha256"]
    )
    old_label, train_label = label_arrays()
    ext_label = np.load(DATA / "cifar10.1_v6_labels.npy")
    src = split_class(np.arange(10000) % 10, 5, SEED)
    train = split_class(train_label, 5, SEED + 1)
    refrows = {
        "original": split_class(old_label, 1, SEED + 2)[0],
        "external": np.argsort(ext_label, kind="stable"),
    }
    assert np.unique(src).size == np.unique(train).size == 10000
    partitions = dict(generated_rows=src, train_reference_rows=train, **refrows)
    if supplement:
        cached = np.load(OUT / "indices.npz")
        assert all(np.array_equal(cached[k], value) for k, value in partitions.items())
        sources = supplement_source()
    else:
        np.savez(OUT / "indices.npz", **partitions)
        sources = {"train_real": REF / "real_train_features.npy", **sources}
    rp = {
        "original": REF / "real_test_features.npy",
        "external": OUT / "v6_inception.npy",
    }
    refs = {}
    base = {}
    bins = [slice(j * 200, (j + 1) * 200) for j in range(10)]
    for j, (name, path) in enumerate(rp.items()):
        x = np.load(path, mmap_mode="r")[refrows[name]]
        refs[name] = x
        w = counts(SEED + 100 + j)
        k = kernel(x, x)
        base[name] = (
            w,
            within(k),
            bootstrap_term(k, w, w, True),
            np.array([within(k[s, s]) for s in bins]),
        )
        del k
    draws = {}
    est = {}
    records = []
    perclass = []
    summary = []
    for model, path in sources.items():
        bank = np.load(path, mmap_mode="r")
        assert bank.dtype == np.float32 and bank.shape == (10000, 2048)
        idx = train if model == "train_real" else src
        vals = {r: [] for r in refs}
        bs = {r: np.zeros(R) for r in refs}
        for b, ids in enumerate(idx):
            f = bank[ids]
            v = counts(SEED + (400 if model == "train_real" else 200) + b)
            kk = kernel(f, f)
            fk = within(kk)
            fv = bootstrap_term(kk, v, v, True)
            fg = np.array([within(kk[s, s]) for s in bins])
            del kk
            for ref, x in refs.items():
                w, rk, rv, rg = base[ref]
                cross = kernel(x, f)
                xy = cross.mean(dtype=np.float64)
                point = rk + fk - 2 * xy
                bc = rv + fv - 2 * bootstrap_term(cross, w, v, False)
                bs[ref] += bc / 5
                vals[ref].append(point)
                records.append(
                    dict(
                        model=model,
                        reference=ref,
                        block=b,
                        n=N,
                        kid=point,
                        within_real=rk,
                        within_generated=fk,
                        cross=xy,
                    )
                )
                cvals = (
                    rg
                    + fg
                    - 2 * np.array([cross[s, s].mean(dtype=np.float64) for s in bins])
                )
                perclass.extend(
                    dict(
                        model=model,
                        reference=ref,
                        block=b,
                        category=CLASSES[c],
                        n=200,
                        kid=score,
                    )
                    for c, score in enumerate(cvals)
                )
                del cross
        for ref in refs:
            point = float(np.mean(vals[ref]))
            ci = centered_interval(point, bs[ref])
            summary.append(
                dict(
                    model=model,
                    reference=ref,
                    n_ref=N,
                    n_generated=10000,
                    n_blocks=5,
                    kid=point,
                    low=ci[0],
                    high=ci[1],
                    block_sd=np.std(vals[ref], ddof=1),
                )
            )
            draws[model, ref] = bs[ref]
            est[model, ref] = point
        print(
            "Measured",
            model,
            dict((r, round(est[model, r] * 1000, 3)) for r in refs),
            flush=True,
        )
    contrasts = []

    def add(desc, model, reference, pair1, pair2):
        point = est[pair1] - est[pair2]
        ci = centered_interval(point, draws[pair1] - draws[pair2])
        contrasts.append(
            dict(
                contrast=desc,
                model=model,
                reference=reference,
                delta_kid=point,
                low=ci[0],
                high=ci[1],
            )
        )

    if supplement:
        old_summary = pd.read_csv(OUT / "summary.csv").set_index(["model", "reference"])
        old_draws = np.load(OUT / "bootstrap.npz")
        for ref in refs:
            est["quality8", ref] = old_summary.loc[("quality8", ref), "kid"]
            draws["quality8", ref] = old_draws["quality8_" + ref]
        comparisons = [("spectral8", "quality8")]
    else:
        for model in sources:
            add(
                "external minus original",
                model,
                "both",
                (model, "external"),
                (model, "original"),
            )
        comparisons = [
            ("quality8", "time32"),
            ("quality24", "time32"),
            ("quality8", "extended44"),
            ("quality24", "extended44"),
            ("quality8", "quality24"),
        ]
    for model, comp in comparisons:
        for r in refs:
            add("minus " + comp, model, r, (model, r), (comp, r))
    for name, rows in [
        ("summary", summary),
        ("blocks", records),
        ("by_class", perclass),
        ("contrasts", contrasts),
    ]:
        pd.DataFrame(rows).to_csv(OUT / (prefix + name + ".csv"), index=False)
    np.savez(
        OUT / (prefix + "bootstrap.npz"),
        **{k + "_" + r: v for (k, r), v in draws.items()},
    )
    if supplement:
        write_json(
            OUT / "supplement_protocol.json",
            dict(
                main_protocol_sha256=sha(OUT / "protocol.json"),
                followup_lock_sha256=sha(OUT / "supplement_lock.json"),
                corrected_feature_sha256=sha(sources["spectral8"]),
                baseline_bootstrap="Quality8 draws reused verbatim from bootstrap.npz; spectral8 uses the matching stratified counts for every draw.",
            ),
        )
        return
    write_json(
        OUT / "protocol.json",
        dict(
            external_commit=COMMIT,
            old_reference_size=N,
            new_reference_size=N,
            generator_partitions=5,
            per_class_per_partition=200,
            seed=SEED,
            bootstrap_replicates=R,
            bootstrap="stratified size-200 multinomial per class for each reference and each generated block; one shared reference draw over all five blocks; same generated indices and weights across locked samplers. Centred percentile 95% intervals. Model weights fixed.",
            kernel="torch-fidelity polynomial MMD degree=3, gamma=1/2048, coef0=1, diagonal-excluded sample self-terms, all cross-pairs; subtract constant 1 from each kernel for numerical precision",
            primary="Mean across five disjoint 2,000-image generated blocks versus the same frozen reference of 2,000 images, conditional labels pooled",
            classwise="Same estimator restricted to the 200 examples of a class per side and block; descriptive only",
            reference_indices_sha256=sha(OUT / "indices.npz"),
        ),
    )


def verify(supplement=False):
    from torch_fidelity.metric_kid import polynomial_mmd

    prefix = "supplement_" if supplement else ""
    ix = np.load(OUT / "indices.npz")
    tab = pd.read_csv(OUT / (prefix + "blocks.csv"))
    summary = pd.read_csv(OUT / (prefix + "summary.csv"))
    checks = []
    sources, lock = locked_sources()
    sources["train_real"] = REF / "real_train_features.npy"
    assert lock == json.loads((OUT / "locked_models.json").read_text()) and all(
        checked(n) for n in FILES
    )
    if supplement:
        sources = supplement_source()
    archived_draws = np.load(OUT / (prefix + "bootstrap.npz"))
    summary_index = summary.set_index(["model", "reference"])
    for (model, ref), row in summary_index.iterrows():
        bs = archived_draws[model + "_" + ref]
        interval = np.percentile(bs, [2.5, 97.5]) - np.mean(bs) + row.kid
        np.testing.assert_allclose(interval, [row.low, row.high], atol=1e-12, rtol=0)
    if supplement:
        summary_index = pd.concat(
            [
                summary_index,
                pd.read_csv(OUT / "summary.csv").set_index(["model", "reference"]),
            ]
        )
    contrasts = pd.read_csv(OUT / (prefix + "contrasts.csv"))
    for row in contrasts.itertuples():
        if row.reference == "both":
            first, second = (row.model, "external"), (row.model, "original")
        else:
            first, second = (row.model, row.reference), (
                row.contrast.removeprefix("minus "),
                row.reference,
            )
        delta = summary_index.at[first, "kid"] - summary_index.at[second, "kid"]
        bs = archived_draws["_".join(first)] - archived_draws["_".join(second)]
        interval = np.percentile(bs, [2.5, 97.5]) - np.mean(bs) + delta
        np.testing.assert_allclose(
            [delta, *interval], [row.delta_kid, row.low, row.high], atol=1e-12, rtol=0
        )
    for model in sources:
        ids = ix["train_reference_rows" if model == "train_real" else "generated_rows"]
        assert np.unique(ids).size == 10000
        fake = np.load(sources[model], mmap_mode="r")[ids[0]]
        for ref, path in [
            ("original", REF / "real_test_features.npy"),
            ("external", OUT / "v6_inception.npy"),
        ]:
            real = np.load(path, mmap_mode="r")[ix[ref]]
            actual = float(polynomial_mmd(real, fake, 3, None, 1.0))
            sel = tab[(tab.model == model) & (tab.reference == ref)]
            e = abs(actual - sel.loc[sel.block == 0, "kid"].iloc[0])
            point = summary[
                (summary.model == model) & (summary.reference == ref)
            ].kid.iloc[0]
            assert e < 7e-6 and abs(sel.kid.mean() - point) < 1e-12
            checks.append(
                dict(
                    model=model,
                    reference=ref,
                    n=2000,
                    block=0,
                    official_kid=actual,
                    absolute_error=e,
                )
            )
    # Validate multiplicity-weighted off-diagonal and cross terms independently
    # on a small resampled feature set (literal explicit duplicate rows).
    rr = np.random.default_rng(91)
    x = rr.standard_normal((25, 12)).astype("float32")
    y = rr.standard_normal((25, 12)).astype("float32")
    a = rr.choice(25, 25)
    b = rr.choice(25, 25)
    aw = np.bincount(a, minlength=25)[None, :].astype("float32")
    bw = np.bincount(b, minlength=25)[None, :].astype("float32")
    v = (
        bootstrap_term(kernel(x, x), aw, aw, True)[0]
        + bootstrap_term(kernel(y, y), bw, bw, True)[0]
        - 2 * bootstrap_term(kernel(x, y), aw, bw, False)[0]
    )
    expected = float(polynomial_mmd(x[a], y[b], 3, None, 1.0))
    assert abs(v - expected) < 2e-6
    write_json(
        OUT / (prefix + "verification.json"),
        dict(
            reference_checks=checks,
            duplicate_resampling_error=abs(v - expected),
            paired_intervals_recomputed=len(contrasts),
            source_counts=10000,
            reference_count=2000,
        ),
    )
    print(
        f"Verified torch-fidelity KID on {len(checks)} model/cohort combinations at block 0 and the explicit resampling test."
    )


def figure():
    from course.figures import setup, page, axes_title, save, TEAL, BLUE, AMBER

    setup()
    q = pd.read_csv(OUT / "summary.csv")
    g = (
        pd.read_csv(OUT / "by_class.csv")
        .query("reference=='external'")
        .groupby(["model", "category"])
        .kid.mean()
        .unstack()
    )
    fig, axes = page(1, 2, size=(12.5, 5.35), width_ratios=[1.12, 1])
    names = list(NAMES)
    xx = np.arange(len(names))
    for j, (ref, col, mark, label) in enumerate(
        [
            ("original", BLUE, "o", "CIFAR-10 original"),
            ("external", TEAL, "D", "CIFAR-10.1 v6"),
        ]
    ):
        t = q[q.reference == ref].set_index("model").loc[names]
        v = 1000 * t.kid
        axes[0].errorbar(
            v,
            xx + (0.115 if j else -0.115),
            xerr=np.stack((v - 1000 * t.low, 1000 * t.high - v)),
            ls="none",
            marker=mark,
            c=col,
            capsize=3,
            ms=5,
            label=label,
        )
    axes[0].set_yticks(xx, [NAMES[n] for n in names])
    axes[0].invert_yaxis()
    axes[0].axvline(0, c="0.7", ls="--", lw=1)
    axes_title(
        axes[0],
        "A  Same samples, two references",
        "Pooled KID × 1,000 (lower is better)",
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for model, col, mark, label in [
        ("quality8", TEAL, "D", "Selected · 8"),
        ("quality24", AMBER, "s", "Selected · 24"),
    ]:
        delta = (g.loc[model] - g.loc["time32"]).loc[list(CLASSES)] * 1000
        axes[1].scatter(delta, np.arange(10), label=label, c=col, marker=mark, s=29)
    axes[1].set_yticks(np.arange(10), CLASSES)
    axes[1].invert_yaxis()
    axes[1].axvline(0, c="0.6", lw=1.1, ls="--")
    axes_title(
        axes[1],
        "B  Changes within each external class",
        "Within-class ΔKID × 1,000 vs. 32 epochs",
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, 4, "cohort_reference_kid")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=(
            "download",
            "extract",
            "score",
            "verify",
            "figure",
            "freeze-supplement",
            "supplement",
            "verify-supplement",
        ),
    )
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.action == "download":
        download()
    elif a.action == "extract":
        encode_external(a.device, a.batch)
    elif a.action == "freeze-supplement":
        lockpath = OUT / "supplement_lock.json"
        lock = supplement_lock()
        if lockpath.exists():
            assert lock == json.loads(lockpath.read_text())
        else:
            write_json(lockpath, lock)
    elif a.action in ("score", "supplement"):
        with threadpool_limits(3):
            measure(a.action == "supplement")
    elif a.action in ("verify", "verify-supplement"):
        with threadpool_limits(2):
            verify(a.action == "verify-supplement")
    elif a.action == "figure":
        figure()
