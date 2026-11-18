"""PathSnap API – runs on LAPTOP 1.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

GET  /health
GET  /trips                 -> the 10 evaluator trip ids (no labels)
POST /predict {"trip_id"}   -> predicted route + server-side accuracy

Ground truth (.route) is loaded ONLY inside `_score()` after prediction and is never
returned to the client. Every request is logged to the console (evaluators will check this).
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import find_trips, load_splits, load_trip  # noqa: E402
from src.labels import align_route                              # noqa: E402
from src.metrics import route_continuity, route_set_f1          # noqa: E402
from src.pipeline import PathSnapPipeline, route_polyline       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PathSnap API] %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

INDEX = find_trips()
EVAL_TRIPS = load_splits(INDEX).get("eval_test", [])
PIPE = PathSnapPipeline()
log.info("loaded model, %d evaluation trips available", len(EVAL_TRIPS))


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "n_eval_trips": len(EVAL_TRIPS)}


@app.get("/trips")
def trips():
    return {"trips": [{"trip_id": t} for t in EVAL_TRIPS]}


def _score(trip_id, out):
    """Server-side evaluation. Reference route is read here and nowhere else."""
    ref_trip = load_trip(trip_id, INDEX, with_route=True)
    if ref_trip.route is None:
        return {}
    pdf, net = out["pdf"], out["net"]
    ref, _ = align_route(pdf.x.to_numpy(), pdf.y.to_numpy(), ref_trip.route, net)
    res = {}
    for name, pred in (("classifier", out["clf_pred"]), ("hmm", out["hmm_pred"])):
        pa = np.array([-1 if p is None else p for p in pred])
        res[name] = {"top1_accuracy": float((pa == ref).mean()),
                     "route_continuity": route_continuity(pred, net),
                     **route_set_f1(pred, ref_trip.route)}
    return res


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    if req.trip_id not in EVAL_TRIPS:
        raise HTTPException(404, f"unknown evaluation trip {req.trip_id}")
    log.info("REQUEST /predict trip_id=%s", req.trip_id)
    trip = load_trip(req.trip_id, INDEX, with_route=False)      # model never sees the route
    out = PIPE.predict(trip)
    metrics = _score(req.trip_id, out)
    pdf = out["pdf"]
    resp = {
        "trip_id": req.trip_id,
        "n_points": int(len(pdf)),
        "gps": pdf[["lat", "lon"]].round(7).values.tolist(),
        "predicted_route": route_polyline(out["hmm_pred"], out["net"]),
        "classifier_route": route_polyline(out["clf_pred"], out["net"]),
        "metrics": metrics,
        "hmm_breaks": out["breaks"],
        "inference_seconds": round(out["seconds"], 3),
        "total_seconds": round(time.perf_counter() - t0, 3),
    }
    log.info("DONE trip_id=%s points=%d hmm_top1=%.4f clf_top1=%.4f time=%.2fs", req.trip_id,
             resp["n_points"], metrics.get("hmm", {}).get("top1_accuracy", float("nan")),
             metrics.get("classifier", {}).get("top1_accuracy", float("nan")), resp["total_seconds"])
    return resp
