"""Tune HMM parameters and compare classifier-only vs classifier+HMM.

    python -m src.tune_hmm --max-trips 30

* Out-of-fold probabilities on TRAINING trips (GroupKFold) are used to grid-search
  beta / emission weight (in-sample probabilities would be over-confident).
* Final comparison on STUDENT-TEST trips with the model in artifacts/model.joblib.
* Writes artifacts/hmm_params.json and results/hmm_*.csv
"""
import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .config import ARTIFACTS, N_FOLDS, RESULTS
from .data_loader import find_trips, load_trip
from .features import FEATURES
from .hmm import steps_from_table, transition_costs, viterbi_from_costs
from .metrics import route_continuity, route_set_f1, top1_from_scores
from .network import build_network
from .train_models import load_partition

BETAS = [5, 10, 20, 40, 80]
W_EMIT = [0.5, 1.0, 2.0]


_CACHE = {}


def per_trip_eval(C, P, nets, routes, params):
    """Network distances are cached per trip, so each grid point only reruns Viterbi."""
    rows = []
    for tid, Pt in P.groupby("trip_id"):
        Ct = C[C.trip_id == tid]
        Pt = Pt.sort_values("obs")
        clf = top1_from_scores(Ct).set_index("obs").pred_arc
        clf_pred = [int(clf[o]) if o in clf else None for o in Pt.obs]
        steps = steps_from_table(Ct, Pt)
        if tid not in _CACHE:
            _CACHE[tid] = transition_costs(steps, nets[tid])
        hmm_pred, info = viterbi_from_costs(steps, *_CACHE[tid], **params)
        ref = Pt.ref_arc.to_numpy()
        for name, pred in (("classifier", clf_pred), ("classifier+hmm", hmm_pred)):
            pa = np.array([-1 if p is None else p for p in pred])
            rows.append({"trip_id": tid, "method": name,
                         "top1_acc": float((pa == ref)[ref >= 0].mean()),
                         "route_continuity": route_continuity(pred, nets[tid]),
                         **route_set_f1(pred, routes[tid]), "breaks": info["breaks"] if name != "classifier" else 0})
    return pd.DataFrame(rows)


def main(max_trips):
    index = find_trips()
    model = joblib.load(ARTIFACTS / "model.joblib")
    Ctr, Ptr = load_partition("train")
    Cst, Pst = load_partition("student_test")

    # OOF probabilities on train
    X, y, g = Ctr[FEATURES].to_numpy(), Ctr.label.to_numpy(), Ctr.trip_id.to_numpy()
    oof = np.zeros(len(Ctr))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        oof[va] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    Ctr["prob"] = oof
    Cst["prob"] = model.predict_proba(Cst[FEATURES].to_numpy())[:, 1]

    tune_ids = sorted(Ptr.trip_id.unique())[:max_trips]
    print(f'tuning HMM on {len(tune_ids)} training trips (OOF probabilities)')
    trips = {t: load_trip(t, index) for t in list(tune_ids) + list(Pst.trip_id.unique())}
    nets = {t: build_network(tr)[0] for t, tr in trips.items()}
    routes = {t: tr.route for t, tr in trips.items()}

    grid = []
    for beta, w in itertools.product(BETAS, W_EMIT):
        params = {"beta": beta, "w_emit": w}
        r = per_trip_eval(Ctr[Ctr.trip_id.isin(tune_ids)], Ptr[Ptr.trip_id.isin(tune_ids)], nets, routes, params)
        acc = r[r.method == "classifier+hmm"].top1_acc.mean()
        grid.append({**params, "train_oof_hmm_top1": acc})
        print(params, f"{acc:.4f}")
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULTS / "hmm_grid.csv", index=False)
    best = grid.sort_values("train_oof_hmm_top1", ascending=False).iloc[0]
    params = {"beta": float(best.beta), "w_emit": float(best.w_emit)}
    json.dump(params, open(ARTIFACTS / "hmm_params.json", "w"), indent=2)

    st = per_trip_eval(Cst, Pst, nets, routes, params)
    st.to_csv(RESULTS / "hmm_student_test_per_trip.csv", index=False)
    summary = st.groupby("method")[["top1_acc", "route_continuity", "route_f1", "breaks"]].mean()
    summary.to_csv(RESULTS / "hmm_comparison.csv")
    print("best params:", params); print(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trips", type=int, default=30)
    main(ap.parse_args().max_trips)
