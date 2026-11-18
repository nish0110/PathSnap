"""Build candidate/feature tables for train + student_test.

    python -m src.build_dataset

Writes data/processed/{partition}_cands.parquet and {partition}_points.parquet.
The eval_test partition is deliberately NOT processed here (leakage rule) — only the API touches it.
"""
import time

import pandas as pd

from .config import PROCESSED, REF_MAX_MEDIAN_M, RESULTS
from .data_loader import find_trips, load_splits, load_trip
from .features import build_trip_table


def main():
    index = find_trips()
    splits = load_splits(index)
    summary, quality = [], []
    for part in ("train", "student_test"):
        cands, points = [], []
        t0 = time.time()
        for tid in splits[part]:
            trip = load_trip(tid, index)
            c, p, _ = build_trip_table(trip)
            med = float(p.ref_dist.median())
            excluded = part == "train" and med > REF_MAX_MEDIAN_M
            quality.append({"partition": part, "trip_id": tid, "points": len(p), "median_ref_dist_m": med,
                            "p95_ref_dist_m": float(p.ref_dist.quantile(.95)),
                            "candidate_miss_rate": 1 - p.ref_in_cands.mean(), "excluded_from_training": excluded})
            print(f"[{part}] {tid}: {len(p)} pts, {len(c)} cands, miss={1 - p.ref_in_cands.mean():.3f}, "
                  f"median ref dist={med:.1f} m{'  -> EXCLUDED' if excluded else ''}")
            if excluded:
                continue
            cands.append(c); points.append(p)
        C = pd.concat(cands, ignore_index=True); P = pd.concat(points, ignore_index=True)
        C.to_parquet(PROCESSED / f"{part}_cands.parquet"); P.to_parquet(PROCESSED / f"{part}_points.parquet")
        summary.append({"partition": part, "trips": int(P.trip_id.nunique()), "points": len(P),
                        "candidates": len(C), "pos_rate": (C.label == 1).mean(),
                        "candidate_miss_rate": 1 - P.ref_in_cands.mean(),
                        "points_without_candidates": (P.n_cand == 0).mean(),
                        "build_seconds": round(time.time() - t0, 1)})
    pd.DataFrame(quality).to_csv(RESULTS / "trip_quality.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "dataset_build_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
