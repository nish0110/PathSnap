"""Dataset exploration (Dataset_Exploration.pdf §4 and §8).

    python -m src.eda

Outputs in results/eda/: inventory.csv, partition_summary.csv, network_summary.csv,
anomalies.csv, sampling_intervals.png, points_per_trip.png, trajectories.png,
gps_to_route_distance.png. Run src.visualize for Folium maps.
EVERY table/plot states the partition it came from. eval_test is only counted (sizes), never analysed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from .config import RESULTS
from .data_loader import find_trips, load_splits, load_trip
from .geo import haversine
from .labels import align_route
from .network import build_network

OUT = RESULTS / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    index = find_trips()
    splits = load_splits(index)
    part_of = {t: p for p, ids in splits.items() for t in ids}

    inv, net_rows, anomalies = [], [], []
    all_dt, align_d = {"train": [], "student_test": []}, []
    fig, ax = plt.subplots(figsize=(8, 8))
    for tid, e in index.items():
        part = part_of.get(tid, "unassigned")
        row = {"trip_id": tid, "partition": part,
               **{f"{k}_bytes": (e[k].stat().st_size if e.get(k) else None) for k in ("track", "nodes", "arcs", "route")}}
        if part == "eval_test":     # inventory only
            inv.append(row); continue
        trip = load_trip(tid, index)
        tr = trip.track
        t = tr.t.to_numpy(); dt = np.diff(t)
        step = haversine(tr.lon.values[:-1], tr.lat.values[:-1], tr.lon.values[1:], tr.lat.values[1:])
        row.update({"n_points": len(tr), "n_missing": int(tr.isna().sum().sum()),
                    "duration_s": t.max() - t.min(), "non_monotonic_t": int((dt <= 0).sum()),
                    "dt_median": np.median(dt) if len(dt) else np.nan, "dt_max": dt.max() if len(dt) else np.nan,
                    "frac_dt_1s": float(np.mean(dt == 1)) if len(dt) else np.nan,
                    "length_km": step.sum() / 1000, "max_step_m": step.max() if len(step) else np.nan,
                    "lon_min": tr.lon.min(), "lon_max": tr.lon.max(), "lat_min": tr.lat.min(), "lat_max": tr.lat.max(),
                    "route_len": len(trip.route) if trip.route is not None else 0})
        inv.append(row)
        all_dt.setdefault(part, []).extend(dt.tolist())
        if (dt > 5).any() or (step > 100).any():
            anomalies.append({"trip_id": tid, "partition": part, "gaps_over_5s": int((dt > 5).sum()),
                              "jumps_over_100m": int((step > 100).sum())})
        # network
        n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
        net, proj = build_network(trip)
        ncomp, lab = connected_components(net.graph, directed=True, connection="weak")
        deg = np.bincount(trip.arcs.ravel(), minlength=n_nodes)
        bad_arcs = int(((trip.arcs < 0) | (trip.arcs >= n_nodes)).any(1).sum())
        bad_route = int(((trip.route < 0) | (trip.route >= n_arcs)).sum()) if trip.route is not None else 0
        net_rows.append({"trip_id": tid, "partition": part, "nodes": n_nodes, "arcs": n_arcs,
                         "weak_components": ncomp, "largest_comp_frac": np.bincount(lab).max() / n_nodes,
                         "isolated_nodes": int((deg == 0).sum()), "invalid_arc_refs": bad_arcs,
                         "invalid_route_refs": bad_route, "mean_arc_len_m": net.length.mean()})
        if part == "train":
            ax.plot(tr.lon, tr.lat, lw=.6)
            if trip.route is not None and bad_route == 0:
                x, y = proj.forward(tr.lon.values, tr.lat.values)
                _, d = align_route(x, y, trip.route, net)
                align_d.extend(d.tolist())
    ax.set_title("GPS trajectories – training partition"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.savefig(OUT / "trajectories.png", dpi=130); plt.close()

    inv = pd.DataFrame(inv); inv.to_csv(OUT / "inventory.csv", index=False)
    pd.DataFrame(net_rows).to_csv(OUT / "network_summary.csv", index=False)
    pd.DataFrame(anomalies).to_csv(OUT / "anomalies.csv", index=False)
    summ = inv.groupby("partition").agg(trips=("trip_id", "count"), points=("n_points", "sum"),
                                        mean_points=("n_points", "mean"), mean_len_km=("length_km", "mean"))
    summ.to_csv(OUT / "partition_summary.csv"); print(summ)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p, v in all_dt.items():
        if v: ax[0].hist(np.clip(v, 0, 15), bins=30, alpha=.6, label=p)
    ax[0].set_yscale("log"); ax[0].set_title("Sampling interval (s), clipped at 15"); ax[0].legend()
    for p in ("train", "student_test"):
        ax[1].hist(inv[inv.partition == p].n_points.dropna(), bins=20, alpha=.6, label=p)
    ax[1].set_title("GPS points per trip"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "sampling_and_counts.png", dpi=130); plt.close()

    if align_d:
        d = np.asarray(align_d)
        plt.hist(np.clip(d, 0, 100), bins=50)
        plt.title(f"GPS distance to aligned reference arc (train) – p50={np.median(d):.1f}m "
                  f"p95={np.percentile(d, 95):.1f}m p99={np.percentile(d, 99):.1f}m")
        plt.xlabel("metres"); plt.savefig(OUT / "gps_to_route_distance.png", dpi=130); plt.close()
        print(f"GPS->reference distance p50/p95/p99: {np.median(d):.1f} / {np.percentile(d,95):.1f} / "
              f"{np.percentile(d,99):.1f} m  -> use p99 to choose CAND_RADIUS_M")


if __name__ == "__main__":
    main()
