"""Turn the .route (sequence of arcs) into a per-GPS-point reference arc.

The .route file has no timestamps, so each GPS point is aligned to one route arc by a
monotone dynamic program: point i is assigned route position j(i), with j non-decreasing,
minimising the total point-to-arc distance. This respects travel order (a plain nearest-
arc lookup would mis-assign points where the route passes the same place twice).
Document this as an assumption + limitation in the report.
"""
import numpy as np

from .geo import point_segment


def align_route(px, py, route, net):
    route = np.asarray(route, dtype=int)
    n, m = len(px), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1), np.full(n, np.nan)
    d, _ = point_segment(px[:, None], py[:, None],
                         net.ax[route][None, :], net.ay[route][None, :],
                         net.bx[route][None, :], net.by[route][None, :])  # (n, m)
    d = d.astype(np.float32)  # long trips: 19k points x 2k arcs
    dp = np.empty_like(d)
    back = np.empty((n, m), dtype=np.int32)
    dp[0] = d[0]
    for i in range(1, n):
        prev = dp[i - 1]
        cm = np.minimum.accumulate(prev)                      # best previous cost with pos <= j
        cm_idx = np.maximum.accumulate(np.where(prev == cm, np.arange(m), 0))  # its argmin
        dp[i] = d[i] + cm
        back[i] = cm_idx
    j = int(np.argmin(dp[-1]))
    pos = np.empty(n, dtype=int)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j] if i > 0 else j
    return route[pos], d[np.arange(n), pos]
