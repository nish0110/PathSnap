"""Central configuration. Change values here, not inside scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data folder. Override with env var PATHSNAP_DATA if your data lives elsewhere.
DATA_RAW = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
SPLITS_FILE = Path(os.environ.get("PATHSNAP_SPLITS", ROOT / "data" / "splits.json"))
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

SEED = 42
N_FOLDS = 5  # GroupKFold over training trips

# ---- Candidate generation ----
CAND_RADIUS_M = 50.0   # EDA: GPS->reference distance p50 ~5.7 m; p95 < 45 m on all well-aligned trips
CAND_TOP_K = 8         # keep the K closest arcs

# ---- HMM / Viterbi (defaults; tuned by src/tune_hmm.py) ----
HMM_BETA_M = 20.0          # transition scale: exp(-|d_route - d_straight| / beta)
HMM_EMISSION_WEIGHT = 1.0  # weight on log P(candidate correct)
HMM_GAP_S = 10.0           # time gap (s) above which transitions are relaxed
HMM_MAX_SPEED_MS = 50.0    # route distances implying faster than this are treated as impossible
HMM_SP_LIMIT_M = 2000.0    # dijkstra search limit (1 Hz data -> consecutive points are metres apart)

# ---- Data quality ----
# Training trips whose median GPS->reference distance exceeds this are excluded from TRAINING
# (EDA: 00000054 ~140 m, 00000033 ~122 m -> reference/GPS clearly misaligned). Logged to results/.
REF_MAX_MEDIAN_M = 50.0

for _p in (PROCESSED, ARTIFACTS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
