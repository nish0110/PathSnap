# PathSnap – Distributed Map-Matching & Route Reconstruction

Matches noisy GPS points to road-network arcs. A candidate classifier scores each road segment near each GPS point, and an HMM with Viterbi decoding turns those scores into a continuous route. The trained model is served by a FastAPI backend (laptop 1) and shown in a Streamlit UI (laptop 2).

## Approach
1. **Candidates**: for every GPS point, keep the `K` nearest arcs within `R` metres (projected coordinates; see `src/geo.py`).
2. **Labels**: the `.route` file is an ordered list of arcs without timestamps. Each GPS point is aligned to a route arc with a monotone DP (`src/labels.py`). Label = 1 for the aligned arc's candidate, else 0.
3. **Classifier**: binary "is this the true arc?" model on per-candidate features (`src/features.py`). Top-1 = highest-probability candidate per point.
4. **HMM/Viterbi**: states = candidates, emission = classifier log-probability, transition = `-|network distance − straight distance| / β` (`src/hmm.py`).
5. **Serving**: `api/main.py` loads `artifacts/model.joblib` + `artifacts/hmm_params.json`; the reference route is used only server-side for scoring.

The five classifiers compared are logistic regression (`logreg`), Gaussian Naive Bayes (`gnb`), random forest (`rf`), histogram gradient boosting (`hgb`), and an MLP neural network (`mlp`).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The repository includes the raw training and student-test trips under `data/raw/`. `data/splits.json` (committed) lists the 80 train + 10 student-test trip ids. Evaluator trips are added on laptop 1 at demo time (put them in `data/raw/eval_test/` and add their ids to `eval_test` in splits.json).

## Data facts found in EDA (results/eda, results/trip_quality.csv)
- 80 train / 10 student-test trips, 1 Hz GPS (94% of steps exactly 1 s), no missing values, timestamps sorted
- One road network per trip: 300 – 155k nodes, ~37 weakly connected components on average, no invalid ids
- GPS→reference distance: median 5.7 m. Trips 00000054 (~140 m) and 00000033 (~122 m) are misaligned → excluded from training (`REF_MAX_MEDIAN_M`)
- Largest time gap 153,064 s (one trip); 49 trips have gaps > 5 s → HMM gap handling
- Candidate radius 50 m, top-8 → candidate miss rate 1.8% train / 0.4% student test

The EDA counts missing GPS-track cells but does not impute values. The feature builder drops any track row with a missing field before generating features; the current 90-trip dataset has no missing GPS-track cells.

## Run order
```bash
python -m src.eda                              # results/eda/*
python -m src.visualize --trip <TRIP_ID>       # results/maps/<TRIP_ID>.html
python -m src.build_dataset                    # data/processed/*.parquet
python -m src.train_models                     # results/model_comparison.csv/png, cm_*.png
python -m src.train_models --only hgb          # train just histogram gradient boosting
python -m src.bayes_opt --model hgb --trials 40   # artifacts/model.joblib, results/bayes_*
python -m src.tune_hmm --max-trips 30          # artifacts/hmm_params.json, results/hmm_*
python -m src.visualize --trip <TRIP_ID> --predict
```

## Deployment (two laptops, same Wi-Fi/hotspot)
**Laptop 1 (model + API)** – needs repo, data, artifacts:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# IP: ipconfig (Windows) / ip a (Linux) / ipconfig getifaddr en0 (mac). Allow port 8000 in the firewall.
```
If Windows Application Control blocks Python DLLs (pandas/scikit-learn), run everything in WSL (Ubuntu): `python3 -m venv ~/psvenv && ~/psvenv/bin/pip install -r requirements.txt`, then run the same commands from `/mnt/d/...`. WSL2 uses NAT, so for laptop 2 to reach it either enable mirrored networking (`networkingMode=mirrored` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`) or forward the port: `netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>` (admin) plus a firewall rule for 8000.

Local testing before the evaluator trips exist: `PATHSNAP_SERVE_SPLIT=student_test uvicorn api.main:app --port 8000` (prints a DEV MODE warning; never use for the demo).

**Laptop 2 (UI)** – needs only `ui/` and `pip install streamlit streamlit-folium folium requests`:
```bash
API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py   # Windows: set API_URL=... then streamlit run ui/app.py
```

## Smoke test without real data
```bash
python scripts/make_synthetic.py /tmp/synth
PATHSNAP_DATA=/tmp/synth PATHSNAP_SPLITS=/tmp/synth/splits.json python -m src.build_dataset
```
Synthetic numbers are meaningless. Never report them.

## Repository layout
| Folder | Contents |
|---|---|
| `src/` | data loading, EDA, features, training, Bayesian optimisation, HMM, maps |
| `api/` | FastAPI backend |
| `ui/` | Streamlit evaluator UI |
| `artifacts/` | final model + HMM parameters |
| `results/` | CSV logs, plots, Folium maps |
| `report/` | report sources |

## Leakage rules we follow
- `eval_test` is never processed by `build_dataset`, training, tuning or EDA (EDA only counts files).
- GroupKFold by trip; scalers sit inside each model pipeline, so they are fit on training folds only.
- The API loads each eval trip with `with_route=False` for prediction; `.route` is read only in `_score()` and never returned to the UI.

## AI assistance disclosure
Starter scaffolding was generated with AI assistance. All experiments, results, analysis and report text are the team's own.
