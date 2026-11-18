"""PathSnap evaluator UI – runs on LAPTOP 2.

    API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

Only talks to the API over HTTP. Has no access to data, model or ground truth.
Needs: pip install streamlit streamlit-folium folium requests
"""
import json
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap – Map-matching evaluation")
st.caption("Noisy GPS points → candidate classifier (HistGradientBoosting) → HMM/Viterbi route")

api = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")


@st.cache_data(ttl=10, show_spinner=False)
def fetch_trips(url):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return [t["trip_id"] for t in r.json()["trips"]]


try:
    trips = fetch_trips(api)
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}\n\n{e}")
    st.info("Start the API on laptop 1 and check the IP, port 8000 and the firewall.")
    st.stop()
if not trips:
    st.warning("The API reports no evaluation trips. Add them to `eval_test` in data/splits.json on laptop 1.")
    st.stop()
st.sidebar.success(f"Connected – {len(trips)} evaluation trips")

show_gps = st.sidebar.checkbox("Show raw GPS points", value=True)
show_clf = st.sidebar.checkbox("Show classifier-only route", value=False)
tile = st.sidebar.selectbox("Base map", ["OpenStreetMap", "CartoDB positron"])

st.subheader("Evaluation trips")
cols = st.columns(5)
for i, t in enumerate(trips):  # ten trip buttons
    if cols[i % 5].button(f"Trip {t}", key=f"b_{t}", use_container_width=True):
        st.session_state["trip"] = t
trip_id = st.session_state.get("trip", trips[0])
st.write(f"Selected trip: **{trip_id}**")

if st.button("Run map matching", type="primary"):
    with st.spinner(f"Calling API for trip {trip_id}…"):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}"); st.stop()
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}"); st.stop()
    st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    st.markdown(f"### Results for trip {res['trip_id']}")
    m = res.get("metrics", {})
    pct = lambda d, k: f"{m.get(d, {}).get(k, float('nan')):.2%}"  # noqa: E731
    c = st.columns(6)
    c[0].metric("Top-1 accuracy (HMM)", pct("hmm", "top1_accuracy"))
    c[1].metric("Top-1 accuracy (classifier)", pct("classifier", "top1_accuracy"))
    c[2].metric("Route continuity (HMM)", pct("hmm", "route_continuity"))
    c[3].metric("Route F1 (HMM)", pct("hmm", "route_f1"))
    c[4].metric("GPS points", res["n_points"])
    c[5].metric("API time", f"{res['total_seconds']} s")
    st.caption(f"HMM breaks (disconnected jumps / large gaps): {res.get('hmm_breaks', 'n/a')} · "
               f"inference {res.get('inference_seconds', 'n/a')} s")

    gps = res["gps"]
    fm = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles=tile)
    fm.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                   [max(p[0] for p in gps), max(p[1] for p in gps)]])
    if show_gps:
        g = folium.FeatureGroup("Raw GPS")
        for lat, lon in gps:
            folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True).add_to(g)
        g.add_to(fm)
    if show_clf and res["classifier_route"]:
        folium.PolyLine(res["classifier_route"], color="#ff7f0e", weight=3, opacity=.8,
                        tooltip="Classifier only").add_to(fm)
    if res["predicted_route"]:
        folium.PolyLine(res["predicted_route"], color="#d62728", weight=5,
                        tooltip="Predicted route (HMM)").add_to(fm)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fm)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(fm)
    folium.LayerControl().add_to(fm)
    st.markdown("🔵 raw GPS · 🔴 predicted route (HMM/Viterbi) · 🟠 classifier-only route")
    st_folium(fm, height=600, use_container_width=True, returned_objects=[])
    d1, d2 = st.columns(2)
    d1.download_button("Download result JSON", json.dumps(res), f"pathsnap_{res['trip_id']}.json")
    with d2.expander("Raw metrics JSON"):
        st.json(m)
