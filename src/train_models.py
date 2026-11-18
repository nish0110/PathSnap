"""Train and compare the five selected model families.

    python -m src.train_models                     # all five models
    python -m src.train_models --only hgb          # one model

Protocol (Models_and_Training_Guidelines §4):
  * fit + GroupKFold CV on training trips (group = trip, so no trip leaks across folds)
  * compare on the 10 student-testing trips
  * log configs, seed, train/inference time, all metrics -> results/
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS, N_FOLDS, PROCESSED, RESULTS, SEED
from .data_loader import find_trips, load_trip
from .features import FEATURES
from .metrics import (candidate_metrics, candidate_miss_rate, route_continuity,
                      top1_accuracy, top1_from_scores)
from .network import build_network


def get_models():
    """name -> (family, estimator). Every config here goes into the report table."""
    return {
        "logreg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000))),
        "gnb": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
        "rf": ("Ensemble (bagging)", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)),
        "hgb": ("Boosting", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=300, early_stopping=True, random_state=SEED)),
        "mlp": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True,
            max_iter=200, random_state=SEED))),
    }


def load_partition(part):
    C = pd.read_parquet(PROCESSED / f"{part}_cands.parquet")
    P = pd.read_parquet(PROCESSED / f"{part}_points.parquet")
    C = C[C.label >= 0].reset_index(drop=True)
    return C, P


def continuity_for(C, preds, nets):
    vals = []
    for tid, g in preds.groupby("trip_id"):
        vals.append(route_continuity(g.sort_values("obs").pred_arc.tolist(), nets[tid]))
    return float(np.nanmean(vals))


def evaluate(model, C, P, nets=None):
    X = C[FEATURES].to_numpy()
    t0 = time.perf_counter()
    prob = model.predict_proba(X)[:, 1]
    inf_ms = (time.perf_counter() - t0) * 1000 / max(len(P), 1)
    out = candidate_metrics(C.label.to_numpy(), prob)
    Cp = C.assign(prob=prob)
    preds = top1_from_scores(Cp)
    out["top1_acc"] = top1_accuracy(P, preds)
    out["infer_ms_per_point"] = inf_ms
    if nets is not None:
        out["route_continuity"] = continuity_for(C, preds, nets)
    return out, prob


def main(only=None):
    models = get_models()
    if only:
        unknown = sorted(set(only) - models.keys())
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Choose from: "
                             f"{', '.join(models)}")

    Ctr, Ptr = load_partition("train")
    Cst, Pst = load_partition("student_test")
    index = find_trips()
    nets_st = {tid: build_network(load_trip(tid, index, with_route=False))[0]
               for tid in Pst.trip_id.unique()}
    X, y, groups = Ctr[FEATURES].to_numpy(), Ctr.label.to_numpy(), Ctr.trip_id.to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)

    rows, fold_rows = [], []
    (RESULTS / "models").mkdir(exist_ok=True)
    for name, (family, est) in models.items():
        if only and name not in only:
            continue
        print(f"\n=== {name} ({family})")
        # ---- group-aware CV on training trips
        for f, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = clone(est)
            m.fit(X[tr], y[tr])
            r, _ = evaluate(m, Ctr.iloc[va], Ptr[Ptr.trip_id.isin(set(groups[va]))])
            fold_rows.append({"model": name, "fold": f, **r})
            print(f"  fold {f}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f}")
        # ---- refit on all training trips, test on student_test
        m = clone(est)
        t0 = time.perf_counter(); m.fit(X, y); train_s = time.perf_counter() - t0
        r, prob = evaluate(m, Cst, Pst, nets_st)
        fr = pd.DataFrame([x for x in fold_rows if x["model"] == name])
        rows.append({"model": name, "family": family, "config": str(est.get_params(deep=False))[:300],
                     "cv_macro_f1": fr.macro_f1.mean(), "cv_macro_f1_std": fr.macro_f1.std(),
                     "cv_top1": fr.top1_acc.mean(), "cv_log_loss": fr.log_loss.mean(),
                     **{f"st_{k}": v for k, v in r.items()},
                     "st_candidate_miss_rate": candidate_miss_rate(Pst), "train_s": train_s})
        joblib.dump(m, RESULTS / "models" / f"{name}.joblib")
        # confusion matrix
        cm = confusion_matrix(Cst.label, (prob >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["wrong arc", "true arc"]).plot()
        plt.title(f"{name} – student test"); plt.savefig(RESULTS / f"cm_{name}.png", dpi=120); plt.close()
        # epoch-wise history only where it really exists
        mlp = m[-1] if hasattr(m, "steps") else m
        if hasattr(mlp, "loss_curve_"):
            plt.plot(mlp.loss_curve_, label="train loss")
            if getattr(mlp, "validation_scores_", None) is not None:
                plt.plot(mlp.validation_scores_, label="val accuracy")
            plt.xlabel("epoch"); plt.legend(); plt.title(f"{name} training history")
            plt.savefig(RESULTS / f"history_{name}.png", dpi=120); plt.close()

    res = pd.DataFrame(rows)
    old = RESULTS / "model_comparison.csv"
    if only and old.exists():
        res = pd.concat([pd.read_csv(old).query("model not in @only"), res], ignore_index=True)
    res.to_csv(old, index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / f"cv_folds{'_' + '_'.join(only) if only else ''}.csv", index=False)
    json.dump({"seed": SEED, "n_folds": N_FOLDS, "features": FEATURES},
              open(RESULTS / "experiment_meta.json", "w"), indent=2)
    print(res[["model", "cv_macro_f1", "st_macro_f1", "st_top1_acc", "st_log_loss",
               "st_route_continuity", "st_infer_ms_per_point"]].to_string(index=False))
    plot_comparison(res)


def plot_comparison(res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].bar(res.model, res.cv_macro_f1, yerr=res.cv_macro_f1_std); ax[0].set_title("CV Macro F1 (train trips)")
    w = 0.4; xs = np.arange(len(res))
    ax[1].bar(xs - w / 2, res.st_top1_acc, w, label="Top-1 acc")
    ax[1].bar(xs + w / 2, res.st_macro_f1, w, label="Macro F1")
    ax[1].set_xticks(xs, res.model); ax[1].legend(); ax[1].set_title("Student test")
    ax[2].bar(res.model, res.st_log_loss); ax[2].set_title("Student-test log loss")
    for a in ax: a.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(RESULTS / "model_comparison.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    main(ap.parse_args().only)
