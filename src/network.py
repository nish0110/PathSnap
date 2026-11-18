"""Road network for one trip: projected geometry, candidate search and network distances."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .config import HMM_SP_LIMIT_M
from .geo import LocalProjection, bearing_xy, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs.astype(np.int64)
        nx_, ny_ = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.nxy = np.c_[nx_, ny_]
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay = self.nxy[a, 0], self.nxy[a, 1]
        self.bx, self.by = self.nxy[b, 0], self.nxy[b, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)
        mid = np.c_[(self.ax + self.bx) / 2, (self.ay + self.by) / 2]
        self.half_max = float(self.length.max() / 2) if len(self.length) else 0.0
        self.tree = cKDTree(mid)
        n = len(nodes_lonlat)
        self.graph = csr_matrix((np.maximum(self.length, 1e-3), (a, b)), shape=(n, n))
        self._sp_cache = {}
        # arcs touching each node (for adjacency / continuity)
        self.node_arcs = [[] for _ in range(n)]
        for i, (u, v) in enumerate(self.arcs):
            self.node_arcs[u].append(i)
            self.node_arcs[v].append(i)

    # ---------- candidates ----------
    def candidates(self, px, py, radius, k):
        """Return list of (arc_ids, dists, fracs) per point, sorted by distance."""
        out = []
        lists = self.tree.query_ball_point(np.c_[px, py], r=radius + self.half_max)
        for x, y, idx in zip(px, py, lists):
            if not idx:
                out.append((np.empty(0, int), np.empty(0), np.empty(0)))
                continue
            idx = np.asarray(idx)
            d, t = point_segment(x, y, self.ax[idx], self.ay[idx], self.bx[idx], self.by[idx])
            keep = d <= radius
            idx, d, t = idx[keep], d[keep], t[keep]
            order = np.argsort(d)[:k]
            out.append((idx[order], d[order], t[order]))
        return out

    # ---------- network distance ----------
    def node_dist(self, u: int, v: int) -> float:
        # cache only reached nodes (networks have up to ~155k nodes; full arrays would blow memory)
        if u not in self._sp_cache:
            d = dijkstra(self.graph, directed=True, indices=u, limit=HMM_SP_LIMIT_M)
            idx = np.flatnonzero(np.isfinite(d))
            self._sp_cache[u] = dict(zip(idx.tolist(), d[idx].tolist()))
        return self._sp_cache[u].get(v, float("inf"))

    def route_dist(self, a: int, ta: float, b: int, tb: float) -> float:
        """Driving distance from position ta on arc a to position tb on arc b (directed)."""
        if a == b:
            # small backward moves on the same arc are GPS jitter -> allow
            return abs(tb - ta) * self.length[a]
        rest = (1 - ta) * self.length[a]
        mid = self.node_dist(int(self.arcs[a, 1]), int(self.arcs[b, 0]))
        return rest + mid + tb * self.length[b]

    def adjacent(self, a: int, b: int) -> bool:
        """Same arc, or arcs share a node (end of a == start of b, or any shared node)."""
        if a == b:
            return True
        u1, v1 = self.arcs[a]; u2, v2 = self.arcs[b]
        return u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2

    def arc_latlon(self, arc_id: int):
        u, v = self.arcs[arc_id]
        return [(self.nodes_lonlat[u, 1], self.nodes_lonlat[u, 0]),
                (self.nodes_lonlat[v, 1], self.nodes_lonlat[v, 0])]


def build_network(trip):
    lon0, lat0 = trip.track["lon"].mean(), trip.track["lat"].mean()
    proj = LocalProjection(lon0, lat0)
    return RoadNetwork(trip.nodes, trip.arcs, proj), proj
