"""Rebuild the research scorecard from official-test predictions, maps and feature banks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from course.common import ROOT, table


def record(rows, number, method, role, n, path, values):
    for name, value in values.items():
        rows.append(
            dict(
                experiment=number,
                method=method,
                comparison=role,
                split="official_test",
                n=n,
                metric=name,
                value=float(value),
                source=str(path.relative_to(ROOT)),
            )
        )


def retina(rows):
    out = ROOT / "experiment2/outputs/foundation"
    path = out / "metrics.csv"
    summary = pd.read_csv(path).query("split=='test'").set_index("feature")
    cases = (
        pd.read_csv(out / "predictions.csv").query("split=='test'").set_index("member")
    )
    y = cases.category.eq("disease").to_numpy()
    selected = json.loads((out / "selection.json").read_text())["feature"]
    assert cases.index.is_unique and len(cases) == 250
    for name, row in summary.iterrows():
        p = cases[name].to_numpy()
        decision = p >= row.threshold
        vals = dict(
            auc=roc_auc_score(y, p),
            accuracy=np.mean(decision == y),
            sensitivity=decision[y].mean(),
            specificity=1 - decision[~y].mean(),
        )
        np.testing.assert_allclose(
            list(vals.values()), row[list(vals)].to_numpy(dtype=float), atol=1e-8
        )
        role = (
            "selected_by_group_cv"
            if name == selected
            else ("previous" if name == "previous_resnet" else "control")
        )
        record(rows, 2, name, role, len(y), path, vals)


def saliency(rows):
    out = ROOT / "experiment3/outputs/semantic"
    path = out / "native_metrics.csv"
    metrics = pd.read_csv(path).query("split=='test'").set_index("model")
    cases = pd.read_csv(out / "native_images.csv").query("split=='test'")
    reference = cases.loc[cases.model == "previous", "member"]
    assert len(reference) == reference.nunique() == 400
    for name, row in metrics.iterrows():
        data = cases[cases.model == name]
        assert len(data) == 400 and set(data.member) == set(reference)
        assert data.category.value_counts().eq(20).all()
        vals = data[["cc", "kl", "sim"]].mean().to_dict()
        np.testing.assert_allclose(
            list(vals.values()), row[list(vals)].to_numpy(dtype=float), atol=1e-7
        )
        role = {"previous": "previous", "selected": "selected_by_validation"}.get(
            name, "feature_exchange" if name.startswith("swap_") else "control"
        )
        record(rows, 3, name, role, len(data), path, vals)


def covariance_root(images):
    mu = images.mean(0, dtype=np.float64)
    cov = np.cov(images, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    return mu, np.trace(cov), (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T


def audit_fid(features, stats):
    mean, trace, root = stats
    cov = np.cov(features, rowvar=False)
    delta = mean - features.mean(0, dtype=np.float64)
    product = root @ cov @ root
    cross = np.maximum(np.linalg.eigvalsh(0.5 * (product + product.T)), 0)
    return delta @ delta + trace + np.trace(cov) - 2 * np.sqrt(cross).sum()


def generation(rows):
    from experiment4.research_evaluate import SCENARIOS

    base = ROOT / "experiment4/outputs"
    out = base / "research/evaluation"
    previous = base / "evaluation/metrics.csv"
    old = pd.read_csv(previous).set_index("method").loc["flow_main"]
    features = np.load(base / "evaluation/real_test_features.npy", mmap_mode="r")
    assert features.shape == (10000, 2048)
    stats = covariance_root(features)
    old_fd = audit_fid(
        np.load(base / "evaluation/flow_main_features.npy", mmap_mode="r"), stats
    )
    assert abs(old_fd - old.fid_test) < 0.005
    probe = pd.read_csv(base / "evaluation/conditional_diagnostics.csv")
    assert len(probe) == 10000
    assert np.array_equal(probe.condition, np.arange(10000) % 10)
    old_agreement = (probe.condition == probe.probe_prediction).mean()
    record(
        rows,
        4,
        "old_12",
        "previous",
        10000,
        previous,
        dict(
            fid_test=old.fid_test,
            inception_score_mean=old.inception_score_mean,
            agreement=old_agreement,
            kid=old.kernel_inception_distance_mean,
        ),
    )
    planned = ("control_12", "time_12", "align_12", "combined_12", "extended_12")
    paths = sorted(out.glob("*/metrics.json"))
    assert set(SCENARIOS) == {p.parent.name for p in paths}
    for path in paths:
        row = json.loads(path.read_text())
        name = row["scenario"]
        feat = np.load(path.parent / "features.npy", mmap_mode="r")
        case = pd.read_csv(path.parent / "per_sample.csv")
        assert feat.shape == features.shape and len(case) == row["samples"] == 10000
        assert np.array_equal(case["sample"], np.arange(10000))
        assert np.array_equal(case.condition, np.arange(10000) % 10)
        assert (
            abs((case.probe_prediction == case.condition).mean() - row["agreement"])
            < 1e-8
        )
        independent = audit_fid(feat, stats)
        assert abs(independent - row["fid_test"]) < 0.005, (
            name,
            row["fid_test"],
            independent,
        )
        role = (
            "additional_training"
            if name == "extended_12"
            else ("matched_training" if name in planned else "sampler_comparison")
        )
        record(
            rows,
            4,
            name,
            role,
            len(feat),
            path,
            {
                **{
                    k: row[k] for k in ["fid_test", "inception_score_mean", "agreement"]
                },
                "kid": row["kernel_inception_distance_mean"],
            },
        )


def semantic_generation(rows):
    folder = ROOT / "experiment4/outputs/research/semantic_evaluation"
    scores = pd.read_csv(folder / "metrics.csv").query("k == 5").set_index("source")
    groups = pd.read_csv(folder / "by_class.csv").query("k == 5")
    grouped = groups.groupby("source")
    nearest = pd.read_csv(folder / "nearest.csv")
    mapping = {
        "old": "old_12",
        "control": "control_12",
        "time": "time_12",
        "align": "align_12",
        "combined": "combined_12",
        "strong": "combined_strong",
        "control_strong": "control_strong",
        "time_strong": "time_strong",
        "align_strong": "align_strong",
        "extended": "extended_12",
    }
    assert set(mapping) | {"train", "permuted"} == set(scores.index)
    assert grouped.size().eq(10).all()
    for source, method in {**mapping, "permuted": "combined_labels_permuted"}.items():
        row = scores.loc[source]
        data = nearest[nearest.source == source]
        assert len(data) == data.real_reference_row.nunique() == 10000
        observed = data.covered_k5.mean()
        assert abs(observed - grouped.coverage.mean()[source]) < 1e-8
        record(
            rows,
            4,
            method,
            "second_encoder",
            len(data),
            folder / "metrics.csv",
            dict(
                dino_fd=row.fd,
                dino_probe_agreement=row.agreement,
                dino_pooled_coverage_k5=row.coverage,
            ),
        )
        record(
            rows,
            4,
            method,
            "second_encoder",
            len(data),
            folder / "nearest.csv",
            dict(dino_within_class_coverage_k5=observed),
        )


if __name__ == "__main__":
    rows = []
    retina(rows)
    saliency(rows)
    with threadpool_limits(3):
        generation(rows)
    semantic_generation(rows)
    path = ROOT / "research-results.csv"
    table(path, rows)
    print(f"{len(rows)} cross-checked test metrics written to {path}")
