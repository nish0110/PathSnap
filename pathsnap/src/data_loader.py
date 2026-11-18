"""Reading raw trip files (.track, .nodes, .arcs, .route) and dataset partitions.

File formats (from Dataset_Exploration.pdf), all whitespace separated, no header:
  .track : lon lat timestamp_s
  .nodes : lon lat              (row index = node_id, 0-based)
  .arcs  : from_node to_node    (row index = arc_id, 0-based)
  .route : arc_id               (chronological ground-truth arcs)

VERIFY these on the real data in EDA before trusting them (coordinate order, column count).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_RAW, SPLITS_FILE

PARTITIONS = ("train", "student_test", "eval_test")


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame           # columns lon, lat, t
    nodes: np.ndarray             # (N, 2) lon, lat
    arcs: np.ndarray              # (M, 2) from, to
    route: Optional[np.ndarray]   # (R,) arc ids or None


def _read(path, ncols, dtype=float):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < ncols:
        raise ValueError(f"{path}: expected >= {ncols} columns, got {df.shape[1]}")
    return df.iloc[:, :ncols].to_numpy(dtype=dtype)


def find_trips(root: Path = DATA_RAW) -> dict:
    """Map trip_id -> {track, nodes, arcs, route} paths. Trip id = file stem of the .track file."""
    trips = {}
    for tr in sorted(Path(root).rglob("*.track")):
        d, stem = tr.parent, tr.stem
        entry = {"track": tr, "dir": d}
        for ext in ("nodes", "arcs", "route"):
            p = d / f"{stem}.{ext}"
            if not p.exists():  # fallback: single file of that type in the same folder (shared network)
                found = list(d.glob(f"*.{ext}"))
                p = found[0] if len(found) == 1 else None
            entry[ext] = p
        trips[stem] = entry
    return trips


def load_trip(trip_id: str, index: dict = None, with_route: bool = True) -> Trip:
    index = index or find_trips()
    e = index[trip_id]
    tr = _read(e["track"], 3)
    track = pd.DataFrame(tr, columns=["lon", "lat", "t"])
    nodes = _read(e["nodes"], 2)
    arcs = _read(e["arcs"], 2, dtype=np.int64)
    route = None
    if with_route and e.get("route") is not None and Path(e["route"]).exists():
        route = _read(e["route"], 1, dtype=np.int64).ravel()
    return Trip(trip_id, track, nodes, arcs, route)


def load_splits(index: dict = None) -> dict:
    """Return {'train': [...], 'student_test': [...], 'eval_test': [...]}.

    Uses data/splits.json if present. Otherwise infers from folder names
    (a path containing 'train' / 'student' / 'eval'). Write splits.json once and commit it.
    """
    if SPLITS_FILE.exists():
        return json.loads(SPLITS_FILE.read_text())
    index = index or find_trips()
    splits = {k: [] for k in PARTITIONS}
    for tid, e in index.items():
        p = str(e["track"]).lower()
        if "student" in p:
            splits["student_test"].append(tid)
        elif "eval" in p:
            splits["eval_test"].append(tid)
        elif "train" in p:
            splits["train"].append(tid)
    if not any(splits.values()):
        raise RuntimeError("Could not infer partitions. Create data/splits.json with keys "
                           "train / student_test / eval_test.")
    return splits
