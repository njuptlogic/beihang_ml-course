"""Rebuild and audit the third-round local scorecard from matched official test artifacts."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

from course.common import ROOT, table
from course.research_results import audit_fid, covariance_root, record
from experiment4.learned_solver_evaluate import SCENARIOS


def retina(result):
    out = ROOT / "experiment2/outputs/spatial"
    p = out / "predictions.csv"
    frame = pd.read_csv(p).query("split=='test'").set_index("member")
    lock = json.loads((out / "selection.json").read_text())
    assert frame.index.is_unique and len(frame) == 250
    y = frame.category.eq("disease")
    assert y.sum() == 100
    values = pd.read_csv(out / "metrics.csv").query("split=='test'")
    assert np.array_equal(frame.selected, frame[lock["design"]])
    for row in values.itertuples(index=False):
        prob = frame[row.model]
        decision = prob >= row.threshold
        metrics = dict(
            accuracy=float((decision == y).mean()),
            auc=roc_auc_score(y, prob),
            ap=average_precision_score(y, prob),
            sensitivity=float(decision[y].mean()),
            specificity=float((~decision[~y]).mean()),
        )
        for k, v in metrics.items():
            np.testing.assert_allclose(v, getattr(row, k), atol=1e-8)
        comparison = "group_cv_selected" if row.model == "selected" else "comparison"
        record(result, 2, row.model, comparison, len(y), p, metrics)


def saliency(result):
    out = ROOT / "experiment3/outputs/multiscale"
    p = out / "native_images.csv"
    stats = (
        pd.read_csv(out / "native_metrics.csv")
        .query("split=='test'")
        .set_index("model")
    )
    img = pd.read_csv(p).query("split=='test'")
    ids = set(img.query("model=='selected'")["index"])
    assert len(ids) == 400
    for method, rows in img.groupby("model"):
        assert set(rows["index"]) == ids and len(rows) == 400
        assert (
            len(rows.category.unique()) == 20
            and rows.category.value_counts().eq(20).all()
        )
        metrics = rows[["cc", "kl", "sim"]].mean().to_dict()
        np.testing.assert_allclose(
            list(metrics.values()),
            stats.loc[method, list(metrics)].astype(float),
            atol=1e-7,
        )
        role = {
            "previous": "previous_selected",
            "selected": "validation_selected",
            "unisal_native": "external_frozen",
        }.get(method, "comparison")
        record(result, 3, method, role, len(rows), p, metrics)


def generation(result):
    root = ROOT / "experiment4/outputs"
    field = root / "guidance"
    new = root / "spectral_guidance/test"
    learned = root / "learned_solver/evaluation"
    old = root / "research/evaluation"
    cases = []
    for name in ("extended_12", "time_strong", "combined_strong"):
        cases.append((old / name, "previous", name))
    locked = json.loads((field / "selection/locked.json").read_text())
    roles = defaultdict(list)
    for role, name in locked.items():
        roles[name].append(role)
    dirs = [p.parent for p in sorted((field / "test").glob("*/metrics.json"))]
    assert set(roles) <= {d.name for d in dirs}
    for d in dirs:
        cases.append((d, ",".join(roles.get(d.name, ["compute_control"])), d.name))
    assert set(SCENARIOS) == {p.parent.name for p in learned.glob("*/metrics.json")}
    for name in SCENARIOS:
        cases.append((learned / name, "distillation", name))
    cases.append((new, "spectrum_train_reference_selection", "spectral_correction"))
    ref = np.load(root / "evaluation/real_test_features.npy", mmap_mode="r")
    assert ref.shape == (10000, 2048)
    stats = covariance_root(ref)
    for d, role, method in cases:
        p = d / "metrics.json"
        m = json.loads(p.read_text())
        prob = pd.read_csv(d / "per_sample.csv")
        assert np.array_equal(prob["sample"], np.arange(10000))
        assert np.array_equal(prob.condition, np.arange(10000) % 10)
        feat = np.load(d / "features.npy", mmap_mode="r")
        assert feat.shape == ref.shape
        independent = audit_fid(feat, stats)
        np.testing.assert_allclose(independent, m["fid_test"], atol=0.005)
        agreement = (prob.condition == prob.probe_prediction).mean()
        np.testing.assert_allclose(agreement, m["agreement"], atol=1e-8)
        record(
            result,
            4,
            method,
            role,
            len(prob),
            p,
            dict(
                fid_test=m["fid_test"],
                inception_score=m["inception_score_mean"],
                kid=m["kernel_inception_distance_mean"],
                probe_agreement=agreement,
            ),
        )
    repeat = root / "spectral_guidance/repeat"
    p = repeat / "metrics.csv"
    predictions = pd.read_csv(repeat / "per_sample.csv")
    assert np.array_equal(predictions["sample"], np.arange(len(ref)))
    assert np.array_equal(predictions.condition, np.arange(len(ref)) % 10)
    for row in pd.read_csv(p).itertuples(index=False):
        features = np.load(repeat / row.method / "features.npy", mmap_mode="r")
        assert features.shape == ref.shape and row.n == len(ref)
        agreement = (predictions[row.method] == predictions.condition).mean()
        np.testing.assert_allclose(audit_fid(features, stats), row.fid_test, atol=0.005)
        np.testing.assert_allclose(agreement, row.agreement, atol=1e-8)
        record(
            result,
            4,
            f"spectral_{row.method}_seed{row.noise_seed}",
            "fresh_noise",
            row.n,
            p,
            dict(
                fid_test=row.fid_test,
                inception_score=row.inception_score_mean,
                kid=row.kernel_inception_distance_mean,
                probe_agreement=agreement,
            ),
        )


if __name__ == "__main__":
    records = []
    retina(records)
    saliency(records)
    with threadpool_limits(3):
        generation(records)
    output = ROOT / "frontier-results.csv"
    table(output, records)
    print(f"Verified {len(records)} third-round comparisons/metrics at {output}")
