"""Generate a tiny FAKE dataset in the real file format, only to smoke-test the code
before/without the real data. NEVER report numbers from this.

    python scripts/make_synthetic.py /tmp/synth
    PATHSNAP_DATA=/tmp/synth PATHSNAP_SPLITS=/tmp/synth/splits.json python -m src.build_dataset
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth")
rng = np.random.default_rng(0)
G = 12                       # grid G x G, 100 m spacing
lon0, lat0 = 77.5, 12.9
dlon, dlat = 100 / (111320 * np.cos(np.radians(lat0))), 100 / 110540
nodes = np.array([[lon0 + i * dlon, lat0 + j * dlat] for j in range(G) for i in range(G)])
arcs = []
for j in range(G):
    for i in range(G):
        u = j * G + i
        if i + 1 < G: arcs += [(u, u + 1), (u + 1, u)]
        if j + 1 < G: arcs += [(u, u + G), (u + G, u)]
arcs = np.array(arcs)
arc_of = {tuple(a): k for k, a in enumerate(arcs)}

ids = {"train": [f"tr{k:02d}" for k in range(12)], "student_test": ["st00", "st01", "st02"],
       "eval_test": ["ev00", "ev01"]}
for part, tids in ids.items():
    d = out / part; d.mkdir(parents=True, exist_ok=True)
    for tid in tids:
        u = int(rng.integers(G * G)); route = []
        for _ in range(25):               # random walk on grid
            nbrs = [k for k, (a, b) in enumerate(arcs) if a == u and (not route or arcs[k][1] != arcs[route[-1]][0])]
            k = int(rng.choice(nbrs)); route.append(k); u = arcs[k][1]
        pts, t = [], 0
        for k in route:
            a, b = nodes[arcs[k][0]], nodes[arcs[k][1]]
            for s in np.linspace(0, 1, 9, endpoint=False):
                p = a + s * (b - a) + rng.normal(0, 12, 2) * [dlon / 100, dlat / 100]
                t += 1 if rng.random() > .03 else int(rng.integers(3, 15))
                pts.append((p[0], p[1], t))
        np.savetxt(d / f"{tid}.track", pts, fmt="%.7f %.7f %d")
        np.savetxt(d / f"{tid}.nodes", nodes, fmt="%.7f")
        np.savetxt(d / f"{tid}.arcs", arcs, fmt="%d")
        np.savetxt(d / f"{tid}.route", route, fmt="%d")
json.dump(ids, open(out / "splits.json", "w"), indent=2)
print("synthetic data written to", out)
