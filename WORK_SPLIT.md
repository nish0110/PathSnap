# PathSnap – Plan & Work Split (4 people)

Assumed timing: start ~10:30, **Review 1 at 11:30**, lunch ~13:00–13:45, code freeze 17:30, end ~18:30. Shift the times if your slot differs.

## 10:30 – 10:45 · Everyone, together (15 min)
1. P4 creates the **public GitHub repo**, pushes this folder, adds everyone. Commit often from now on (commit history is checked).
2. Everyone: `pip install -r requirements.txt`, put the data in `data/raw/`.
3. Open 2 trips by hand and **check the assumptions in `src/data_loader.py`**: column order (lon, lat, t), whether each trip has its own `.nodes/.arcs` or they're shared, whether route arc ids are valid. Write `data/splits.json` if folder names don't show the split.

## Owners

| | P1 – Data & EDA | P2 – Features & 5+ models | P3 – Bayes opt & HMM | P4 – API, UI, integration |
|---|---|---|---|---|
| Files | `data_loader.py`, `eda.py`, `visualize.py`, `geo.py` | `features.py`, `build_dataset.py`, `train_models.py`, `metrics.py` | `bayes_opt.py`, `hmm.py`, `tune_hmm.py`, `labels.py` | `api/`, `ui/`, `pipeline.py`, README, repo |
| Report sections | 2 (EDA) + 6 (Folium maps) | 3.1–3.3 (model table, per-model Qs, plots) | 4 (Bayesian) + 5 (HMM) | 7 (API/UI/deploy) + assembling the final PDF |

## Phase 1 – until Review 1 (11:30)
Goal: show that you understand the data, have a working pipeline, and have first results.

- **P1:** run `python -m src.eda`. Check `inventory.csv`, `anomalies.csv` and `network_summary.csv`, and note down the gaps, the time ordering and the components. Make 2 Folium maps (`src.visualize --trip X`). Read the p99 GPS-to-route distance off `gps_to_route_distance.png` and **tell P2 the candidate radius** (`CAND_RADIUS_M`).
- **P2:** run `python -m src.build_dataset` and note the **candidate miss rate**. Run `python -m src.train_models --only logreg rf hgb` for first numbers.
- **P3:** study `labels.py` (route→point alignment) and `hmm.py` until you can explain both on a whiteboard. Draw the HMM (states, emission, transition) for the review.
- **P4:** run the API on laptop 1 and the UI on laptop 2 over the same hotspot using the synthetic data or the first model. Fix firewall/IP problems now, not at 5 pm.
- **Review 1 talking points (all):** dataset structure, data-quality issues, problem framing (candidate classification + HMM), the leakage plan, a live demo of the map and the first model table.

## Phase 2 – 11:30 → 13:00
- **P1:** add/justify 2–3 new features based on EDA (e.g. curvature, distance to the previous chosen arc, road-degree). Give them to P2 and run an ablation with and without them.
- **P2:** run all 7–8 models (`python -m src.train_models`). Fill the comparison table and pick the best model **with reasons** (CV stability, log loss for HMM emissions, inference speed).
- **P3:** as soon as P2 picks a model, run `python -m src.bayes_opt --model <best> --trials 40`. Justify the search space. Then run `python -m src.tune_hmm`.
- **P4:** point the API at the real `artifacts/`, test all 10 eval trips through the UI, and time each request. Make sure the API console logs are visible (evaluators check them).

## Phase 3 – 13:45 → 16:00
- **P1:** make final Folium maps (`--predict`) for 3 student-test trips: one where the HMM helps, one where it fails, one with a GPS gap. Take screenshots.
- **P2:** write the per-model report subsections (Q1–Q9 for each model, in your own words, using your own numbers). Make the comparison plots.
- **P3:** HMM analysis: classifier vs HMM per trip (`results/hmm_student_test_per_trip.csv`), cases where the HMM hurt, the effect of β. Optionally improve transitions (e.g. a heading penalty).
- **P4:** full two-laptop rehearsal. Take UI screenshots and API log screenshots. Clean up the repo structure and the README.

## Phase 4 – 16:00 → 17:30
Write the report sections (owners as in the table above). P4 merges them into one PDF. **17:00:** final retrain, then commit artifacts and results. **17:30:** freeze. Every member reads every section so anyone can answer the viva.

## Rules that can lose you marks
- Never use `eval_test` for anything except the final API demo.
- Every number in the report must come from a file in `results/`.
- Do not paste AI text into the report. Write it yourselves (it is checked).
- No hardcoded predictions. The API must show logs on laptop 1.
