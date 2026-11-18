"""Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.

    python -m src.bayes_opt --model hgb --trials 40

Objective: mean GroupKFold Top-1 accuracy on the 80 training trips (change if you justify
another, e.g. log loss for better-calibrated HMM emissions). Student test is used ONLY
afterwards to compare tuned vs untuned. Evaluator-test is never touched.
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS, N_FOLDS, RESULTS, SEED
from .features import FEATURES
from .train_models import evaluate, get_models, load_partition


def space(name, trial):
    """Search spaces – JUSTIFY each range in the report."""
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 800),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10, log=True),
            early_stopping=True, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name == "mlp":
        w = trial.suggest_categorical("width", [32, 64, 128])
        d = trial.suggest_int("depth", 1, 3)
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(w,) * d, alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            early_stopping=True, max_iter=300, random_state=SEED))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
            random_state=SEED, verbose=-1)
    raise ValueError(f"No search space for {name}")


def main(name, n_trials):
    Ctr, Ptr = load_partition("train")
    Cst, Pst = load_partition("student_test")
    X, y, g = Ctr[FEATURES].to_numpy(), Ctr.label.to_numpy(), Ctr.trip_id.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, g))

    def objective(trial):
        model = space(name, trial)
        scores = []
        for tr, va in folds:
            m = clone(model).fit(X[tr], y[tr])
            r, _ = evaluate(m, Ctr.iloc[va], Ptr[Ptr.trip_id.isin(set(g[va]))])
            scores.append(r["top1_acc"])
            trial.set_user_attr("macro_f1", r["macro_f1"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_s = time.time() - t0
    study.trials_dataframe().to_csv(RESULTS / f"bayes_trials_{name}.csv", index=False)

    tuned = space(name, optuna.trial.FixedTrial(study.best_params)).fit(X, y)
    untuned = clone(get_models()[name][1]).fit(X, y)
    rows = []
    for label, m in (("untuned", untuned), ("tuned", tuned)):
        cv = []
        for tr, va in folds:
            mm = clone(m).fit(X[tr], y[tr])
            cv.append(evaluate(mm, Ctr.iloc[va], Ptr[Ptr.trip_id.isin(set(g[va]))])[0])
        r, _ = evaluate(m, Cst, Pst)
        rows.append({"variant": label, "cv_macro_f1": np.mean([c["macro_f1"] for c in cv]),
                     "cv_top1": np.mean([c["top1_acc"] for c in cv]),
                     **{f"st_{k}": v for k, v in r.items()}})
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / f"tuned_vs_untuned_{name}.csv", index=False)
    print(comp.to_string(index=False))

    joblib.dump(tuned, ARTIFACTS / "model.joblib")
    json.dump({"model": name, "best_params": study.best_params, "best_cv_top1": study.best_value,
               "n_trials": n_trials, "optimisation_seconds": opt_s, "features": FEATURES},
              open(ARTIFACTS / "model_meta.json", "w"), indent=2)

    vals = [t.value for t in study.trials]
    plt.plot(vals, "o", alpha=.5, label="trial"); plt.plot(np.maximum.accumulate(vals), label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.legend(); plt.title(f"Bayesian optimisation – {name}")
    plt.savefig(RESULTS / f"bayes_history_{name}.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    main(a.model, a.trials)
