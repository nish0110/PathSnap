# Report outline (fill in in your own words; every number from results/)

## 1. Problem & approach overview  — P4
## 2. Dataset understanding & EDA  — P1
- Preprocessing steps (sorting by time, dropping NaNs, projection choice + justification)
- Partition sizes (results/eda/partition_summary.csv) — state partition for every plot
- Missing values, timestamp ordering, sampling intervals, gaps/jumps (inventory.csv, anomalies.csv)
- Network size, components, isolated nodes (network_summary.csv)
- GPS→reference distance distribution → choice of candidate radius
- Features extracted and WHY; route→point alignment assumption and its limitations
- ≥1 Folium map screenshot
## 3. Model experimentation — P2
### 3.1 Comparison table (results/model_comparison.csv)
| Model | Family | Key config | CV Macro F1 | Top-1 | Route continuity | ST Macro F1 | Log loss | Inference time |
### 3.2 One subsection per model: Q1, Q2, Q5, Q6, Q7, Q8, Q9
### 3.3 Plots: model_comparison.png, cm_*.png, history_mlp.png (only real histories)
Primary selection metric + why.
## 4. Bayesian optimisation — P3
Hyperparameters, ranges, objective, #trials, best values, tuned vs untuned table (tuned_vs_untuned_*.csv), bayes_history_*.png, compute cost.
## 5. HMM & Viterbi — P3
Hidden states, emissions, transitions, gaps/missing candidates, disconnected segments (chain breaks), classifier vs HMM (hmm_comparison.csv), where it did not help.
## 6. Folium visualisation — P1
Raw GPS, network, reference, predicted; legend + layer control.
## 7. API, UI, deployment — P4
Architecture diagram, endpoints, UI screenshots, API log screenshots on laptop 1.
## 8. Conclusions & limitations
## AI-assistance disclosure
