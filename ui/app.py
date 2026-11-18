"""PathSnap evaluator UI – runs on LAPTOP 2.

    API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

Only talks to the API over HTTP. Has no access to data, model or ground truth.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap – Map-matching evaluation")

api = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000"))
try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
    st.sidebar.success(f"Connected – {len(trips)} evaluation trips")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API: {e}")
    st.stop()

trip_id = st.selectbox("Choose an evaluation trip", [t["trip_id"] for t in trips])
show_clf = st.sidebar.checkbox("Also show classifier-only route", value=False)

if st.button("Run map matching", type="primary"):
    with st.spinner("Calling API…"):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=300)
    if r.status_code != 200:
        st.error(r.text); st.stop()
    st.session_state["res"] = r.json()

res = st.session_state.get("res")
if res:
    m = res["metrics"]
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", f"{m.get('hmm', {}).get('top1_accuracy', float('nan')):.2%}")
    c[1].metric("Top-1 accuracy (classifier)", f"{m.get('classifier', {}).get('top1_accuracy', float('nan')):.2%}")
    c[2].metric("Route continuity (HMM)", f"{m.get('hmm', {}).get('route_continuity', float('nan')):.2%}")
    c[3].metric("Route F1 (HMM)", f"{m.get('hmm', {}).get('route_f1', float('nan')):.2%}")
    c[4].metric("API time", f"{res['total_seconds']} s")

    gps = res["gps"]
    fm = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles="OpenStreetMap")
    fm.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                   [max(p[0] for p in gps), max(p[1] for p in gps)]])
    g = folium.FeatureGroup("Raw GPS")
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True).add_to(g)
    g.add_to(fm)
    if show_clf and res["classifier_route"]:
        folium.PolyLine(res["classifier_route"], color="#ff7f0e", weight=3, tooltip="Classifier only").add_to(fm)
    if res["predicted_route"]:
        folium.PolyLine(res["predicted_route"], color="#d62728", weight=5, tooltip="Predicted route (HMM)").add_to(fm)
    folium.LayerControl().add_to(fm)
    st_folium(fm, height=600, use_container_width=True)
    with st.expander("Raw metrics JSON"):
        st.json(m)
