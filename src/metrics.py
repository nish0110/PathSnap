"""Metrics shared by training, HMM evaluation and the API."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score)


def candidate_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    out = {"accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}
    try:
        out["log_loss"] = log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])
    except ValueError:
        out["log_loss"] = np.nan
    return out


def top1_from_scores(cdf: pd.DataFrame, score_col="prob"):
    """Per (trip, obs): arc with max score."""
    idx = cdf.groupby(["trip_id", "obs"])[score_col].idxmax()
    return cdf.loc[idx, ["trip_id", "obs", "arc"]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(points: pd.DataFrame, preds: pd.DataFrame):
    """Points with no reference are excluded; points with no prediction count as wrong."""
    m = points[points.ref_arc >= 0].merge(preds, on=["trip_id", "obs"], how="left")
    return float((m.pred_arc == m.ref_arc).mean())


def candidate_miss_rate(points):
    p = points[points.ref_arc >= 0]
    return float(1 - p.ref_in_cands.mean())


def route_continuity(pred_arcs, net):
    """Fraction of consecutive predicted arc pairs that are the same or adjacent."""
    a = [int(x) for x in pred_arcs if x is not None and x >= 0]
    if len(a) < 2:
        return np.nan
    ok = sum(net.adjacent(a[i], a[i + 1]) for i in range(len(a) - 1))
    return ok / (len(a) - 1)


def collapse(seq):
    out = []
    for s in seq:
        if s is None or s < 0:
            continue
        if not out or out[-1] != s:
            out.append(int(s))
    return out


def route_set_f1(pred_arcs, route):
    """Sequence-level: precision/recall/F1 of the set of traversed arcs vs the .route file."""
    P, R = set(collapse(pred_arcs)), set(int(r) for r in route)
    if not P or not R:
        return {"route_precision": np.nan, "route_recall": np.nan, "route_f1": np.nan}
    tp = len(P & R)
    pr, rc = tp / len(P), tp / len(R)
    return {"route_precision": pr, "route_recall": rc,
            "route_f1": 0 if tp == 0 else 2 * pr * rc / (pr + rc)}
