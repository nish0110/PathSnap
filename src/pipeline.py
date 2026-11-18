"""End-to-end inference used by the API. Takes ONLY the GPS track + road network.
The reference route is never passed in here."""
import json
import time

import joblib
import numpy as np

from .config import ARTIFACTS
from .features import FEATURES, build_trip_table
from .hmm import steps_from_table, viterbi
from .metrics import top1_from_scores


class PathSnapPipeline:
    def __init__(self, model_path=ARTIFACTS / "model.joblib", hmm_path=ARTIFACTS / "hmm_params.json"):
        self.model = joblib.load(model_path)
        self.hmm = json.loads(hmm_path.read_text()) if hmm_path.exists() else {}

    def predict(self, trip):
        """trip must be loaded with with_route=False."""
        assert trip.route is None, "pipeline must not see the reference route"
        t0 = time.perf_counter()
        cdf, pdf, net = build_trip_table(trip)
        cdf["prob"] = self.model.predict_proba(cdf[FEATURES].to_numpy())[:, 1] if len(cdf) else []
        clf = top1_from_scores(cdf).set_index("obs").pred_arc if len(cdf) else {}
        clf_pred = [int(clf[o]) if o in clf else None for o in pdf.obs]
        hmm_pred, info = viterbi(steps_from_table(cdf, pdf), net, **self.hmm)
        return {"pdf": pdf, "net": net, "clf_pred": clf_pred, "hmm_pred": hmm_pred,
                "breaks": info["breaks"], "seconds": time.perf_counter() - t0}


def route_polyline(pred, net):
    """Collapse per-point arcs into an ordered lat/lon polyline for display."""
    coords, last = [], None
    for a in pred:
        if a is None or a == last:
            continue
        seg = net.arc_latlon(a)
        if coords and coords[-1] == seg[0]:
            coords.append(seg[1])
        else:
            coords.extend(seg)
        last = a
    return [[float(lat), float(lon)] for lat, lon in coords]
