"""Candidate generation + feature engineering for one trip.

Output: one row per (GPS point, candidate arc). Label = 1 if candidate arc is the
reference arc for that point (only when a route is available).

The ML task is therefore: binary classification "is this candidate the true road?",
and per point we take argmax probability = Top-1 prediction. The classifier probabilities
become the HMM emission scores.

>>> Feature ideas are a STARTING POINT. Add/remove based on your own EDA and ablations. <<<
"""
import numpy as np
import pandas as pd

from .config import CAND_RADIUS_M, CAND_TOP_K
from .geo import angle_diff, bearing_xy
from .labels import align_route
from .network import build_network

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "heading_diff", "speed", "arc_len", "frac", "dt_prev", "dt_next",
    "prev_support", "next_support",
]


def _kinematics(x, y, t):
    """Speed (m/s) and heading (deg) from centred differences."""
    n = len(x)
    xp, yp, tp = np.r_[x[0], x[:-1]], np.r_[y[0], y[:-1]], np.r_[t[0], t[:-1]]
    xn, yn, tn = np.r_[x[1:], x[-1]], np.r_[y[1:], y[-1]], np.r_[t[1:], t[-1]]
    dt = np.maximum(tn - tp, 1e-6)
    speed = np.hypot(xn - xp, yn - yp) / dt
    heading = bearing_xy(xn - xp, yn - yp)
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    return speed, heading, dt_prev, dt_next


def _support(cands, net, shift, radius):
    """For each candidate arc at point i: distance from point i+shift to the nearest
    candidate of that point that is the same arc or adjacent. Captures local continuity
    without using labels."""
    n = len(cands)
    out = []
    for i in range(n):
        arcs_i = cands[i][0]
        j = i + shift
        if j < 0 or j >= n or len(cands[j][0]) == 0:
            out.append(np.full(len(arcs_i), radius))
            continue
        arcs_j, d_j = cands[j][0], cands[j][1]
        vals = []
        for a in arcs_i:
            m = [dj for b, dj in zip(arcs_j, d_j) if net.adjacent(int(a), int(b))]
            vals.append(min(m) if m else radius)
        out.append(np.asarray(vals))
    return out


def build_trip_table(trip, radius=CAND_RADIUS_M, k=CAND_TOP_K):
    """Returns (candidate_df, point_df, net). point_df has per-point reference info."""
    net, proj = build_network(trip)
    tr = trip.track.sort_values("t").reset_index(drop=True)  # timestamp ordering check in EDA!
    tr = tr.dropna()
    x, y = proj.forward(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    t = tr["t"].to_numpy(float)
    speed, heading, dt_prev, dt_next = _kinematics(x, y, t)
    cands = net.candidates(x, y, radius, k)
    prev_sup = _support(cands, net, -1, radius)
    next_sup = _support(cands, net, +1, radius)

    ref, ref_d = np.full(len(x), -1), np.full(len(x), np.nan)
    if trip.route is not None and len(trip.route):
        ref, ref_d = align_route(x, y, trip.route, net)

    rows = []
    for i, (arcs, d, fr) in enumerate(cands):
        if len(arcs) == 0:
            continue
        dmin = d.min()
        hd = angle_diff(heading[i], net.bearing[arcs])
        for r, (a, di, fi) in enumerate(zip(arcs, d, fr)):
            rows.append({
                "trip_id": trip.trip_id, "obs": i, "arc": int(a),
                "dist": di, "dist_rank": r, "dist_minus_min": di - dmin,
                "dist_ratio": di / (dmin + 1.0), "n_cand": len(arcs),
                "heading_diff": hd[r], "speed": speed[i], "arc_len": net.length[a],
                "frac": fi, "dt_prev": dt_prev[i], "dt_next": dt_next[i],
                "prev_support": prev_sup[i][r], "next_support": next_sup[i][r],
                "label": int(a == ref[i]) if ref[i] >= 0 else -1,
            })
    cdf = pd.DataFrame(rows)
    pdf = pd.DataFrame({"trip_id": trip.trip_id, "obs": np.arange(len(x)), "lon": tr["lon"],
                        "lat": tr["lat"], "t": t, "x": x, "y": y, "ref_arc": ref, "ref_dist": ref_d,
                        "n_cand": [len(c[0]) for c in cands]})
    pdf["ref_in_cands"] = [ref[i] in set(cands[i][0].tolist()) for i in range(len(x))]
    return cdf, pdf, net
