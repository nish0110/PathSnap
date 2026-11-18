"""HMM + Viterbi map matching.

Hidden states  : at time i, the candidate arcs of GPS point i (state = (arc_id, frac on arc)).
Emissions      : w_emit * log P(candidate is correct) from the classifier.
Transitions    : log p = -|d_route - d_straight| / beta   (Newson & Krumm 2009 idea)
                 d_route = directed network distance between the two projected positions,
                 d_straight = straight-line distance between the two GPS points.
                 Unreachable (no path within HMM_SP_LIMIT_M) or implied speed > max -> impossible.
GPS gaps       : beta is scaled by dt / gap_s when dt > gap_s (more uncertainty after a gap).
No candidates  : the point is skipped (prediction None); the chain continues from the last valid point.
Dead ends      : if every transition into step i is impossible (e.g. long gap, disconnected
                 network component), the chain is broken and restarted at i (counted in `breaks`).

Distances are computed once per trip (`transition_costs`), so tuning beta / w_emit is cheap.
"""
import numpy as np

from .config import (HMM_BETA_M, HMM_EMISSION_WEIGHT, HMM_GAP_S, HMM_MAX_SPEED_MS)

NEG = -1e18


def transition_costs(steps, net, max_speed=HMM_MAX_SPEED_MS):
    """For consecutive valid points: matrix |d_route - d_straight| (inf = impossible), and dt."""
    valid = [i for i, s in enumerate(steps) if len(s["arcs"])]
    out = []
    for prev_i, i in zip(valid[:-1], valid[1:]):
        sp, sc = steps[prev_i], steps[i]
        dt = max(sc["t"] - sp["t"], 1e-3)
        straight = np.hypot(sc["x"] - sp["x"], sc["y"] - sp["y"])
        cost = np.full((len(sp["arcs"]), len(sc["arcs"])), np.inf)
        cap = max_speed * dt + 50
        for a, (pa, pt) in enumerate(zip(sp["arcs"], sp["fracs"])):
            for c, (ca, ct) in enumerate(zip(sc["arcs"], sc["fracs"])):
                d = net.route_dist(int(pa), float(pt), int(ca), float(ct))
                if d <= cap:
                    cost[a, c] = abs(d - straight)
        out.append(cost)
    return valid, out


def viterbi_from_costs(steps, valid, costs, beta=HMM_BETA_M, w_emit=HMM_EMISSION_WEIGHT,
                       gap_s=HMM_GAP_S):
    n = len(steps)
    pred = [None] * n
    if not valid:
        return pred, {"breaks": 0}

    def emis(s):
        return w_emit * np.log(np.clip(s["probs"], 1e-6, 1.0))

    breaks, segments = 0, []
    cur = [(valid[0], None)]
    score = emis(steps[valid[0]])
    for k, (prev_i, i) in enumerate(zip(valid[:-1], valid[1:])):
        sc = steps[i]
        dt = max(sc["t"] - steps[prev_i]["t"], 1e-3)
        b = beta * max(1.0, dt / gap_s)
        trans = np.where(np.isfinite(costs[k]), -costs[k] / b, NEG)
        tot = score[:, None] + trans
        best_prev, new = tot.argmax(0), tot.max(0)
        if np.all(new <= NEG / 2):          # no feasible transition -> break chain
            breaks += 1
            segments.append((cur, score))
            cur, score = [(i, None)], emis(sc)
            continue
        cur.append((i, best_prev))
        score = new + emis(sc)
    segments.append((cur, score))

    for seg, sc in segments:                 # backtrack each segment
        j = int(np.argmax(sc))
        for i, bp in reversed(seg):
            pred[i] = int(steps[i]["arcs"][j])
            if bp is not None:
                j = int(bp[j])
    return pred, {"breaks": breaks}


def viterbi(steps, net, beta=HMM_BETA_M, w_emit=HMM_EMISSION_WEIGHT, gap_s=HMM_GAP_S,
            max_speed=HMM_MAX_SPEED_MS):
    valid, costs = transition_costs(steps, net, max_speed)
    return viterbi_from_costs(steps, valid, costs, beta, w_emit, gap_s)


def steps_from_table(cdf, pdf, prob_col="prob"):
    """Build Viterbi input for one trip from candidate table (+probs) and point table."""
    pdf = pdf.sort_values("obs")
    groups = {o: g for o, g in cdf.groupby("obs")}
    steps = []
    for r in pdf.itertuples():
        g = groups.get(r.obs)
        if g is None:
            steps.append({"arcs": np.empty(0, int), "fracs": np.empty(0), "probs": np.empty(0),
                          "x": r.x, "y": r.y, "t": r.t})
        else:
            steps.append({"arcs": g.arc.to_numpy(), "fracs": g.frac.to_numpy(),
                          "probs": g[prob_col].to_numpy(), "x": r.x, "y": r.y, "t": r.t})
    return steps
