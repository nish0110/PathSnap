"""Folium maps (mandatory in the report).

    python -m src.visualize --trip <trip_id>             # raw GPS + network + reference route
    python -m src.visualize --trip <trip_id> --predict   # + classifier-only and HMM routes

Saved to results/maps/<trip_id>.html. Open in a browser, screenshot for the report.
"""
import argparse

import folium

from .config import RESULTS
from .data_loader import find_trips, load_trip
from .network import build_network


def base_map(trip):
    tr = trip.track
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    return m


def add_network(m, net, trip, margin_deg=0.003):
    tr = trip.track
    fg = folium.FeatureGroup("Road network", show=True)
    lo, la = net.nodes_lonlat[:, 0], net.nodes_lonlat[:, 1]
    box = ((lo > tr.lon.min() - margin_deg) & (lo < tr.lon.max() + margin_deg) &
           (la > tr.lat.min() - margin_deg) & (la < tr.lat.max() + margin_deg))
    for i, (u, v) in enumerate(net.arcs):
        if box[u] or box[v]:
            folium.PolyLine(net.arc_latlon(i), color="#999999", weight=1.5, opacity=.6).add_to(fg)
    fg.add_to(m)


def add_gps(m, trip, color="#1f77b4"):
    fg = folium.FeatureGroup("Raw GPS")
    for r in trip.track.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color=color, fill=True,
                            tooltip=f"t={r.t:.0f}").add_to(fg)
    fg.add_to(m)


def add_route(m, coords, name, color, weight=5):
    fg = folium.FeatureGroup(name)
    if coords:
        folium.PolyLine(coords, color=color, weight=weight, opacity=.8, tooltip=name).add_to(fg)
    fg.add_to(m)


def legend(m, items):
    html = "".join(f'<div><span style="background:{c};width:14px;height:4px;display:inline-block;'
                   f'margin-right:6px"></span>{n}</div>' for n, c in items)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        f'padding:8px;border:1px solid #ccc;font-size:13px">{html}</div>'))


def trip_map(trip_id, predict=False):
    from .pipeline import route_polyline
    index = find_trips()
    trip = load_trip(trip_id, index)
    net, _ = build_network(trip)
    m = base_map(trip)
    add_network(m, net, trip)
    items = [("Road network", "#999999"), ("Raw GPS", "#1f77b4")]
    if trip.route is not None:
        add_route(m, route_polyline(list(trip.route), net), "Reference route", "#2ca02c", 7)
        items.append(("Reference route", "#2ca02c"))
    if predict:
        from .pipeline import PathSnapPipeline
        out = PathSnapPipeline().predict(load_trip(trip_id, index, with_route=False))
        add_route(m, route_polyline(out["clf_pred"], net), "Classifier only", "#ff7f0e", 3)
        add_route(m, route_polyline(out["hmm_pred"], net), "Classifier + HMM", "#d62728", 4)
        items += [("Classifier only", "#ff7f0e"), ("Classifier + HMM", "#d62728")]
    add_gps(m, trip)
    folium.LayerControl(collapsed=False).add_to(m)
    legend(m, items)
    out_dir = RESULTS / "maps"; out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{trip_id}{'_pred' if predict else ''}.html"
    m.save(str(path))
    print("saved", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    trip_map(a.trip, a.predict)
