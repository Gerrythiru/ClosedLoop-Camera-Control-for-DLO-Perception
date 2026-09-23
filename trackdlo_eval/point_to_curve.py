"""Point-to-curve distance: nearest point on a piecewise-linear curve, not
nearest indexed node (G1 error analysis, Step C).

Handles endpoint-count/ordering mismatch between two polylines without
resampling or trimming assumptions -- for each query point, find the
closest point anywhere on the target curve's segments.
"""

from __future__ import annotations

import numpy as np


def point_to_curve_distances(
    query_points: np.ndarray, curve_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in query_points, return (distance, nearest_point) to the
    piecewise-linear curve defined by consecutive rows of curve_points.

    query_points: Nx3. curve_points: Mx3 (M >= 2), any frame/units -- both
    must already be in the same frame. Curve segment order does not need to
    match any particular direction; a polyline's shape (and hence nearest
    point) is direction-independent.
    """
    query = np.asarray(query_points, dtype=np.float64)
    curve = np.asarray(curve_points, dtype=np.float64)
    best_dist = np.full(len(query), np.inf)
    best_point = np.zeros_like(query)

    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-18:
            continue
        t = np.clip(((query - a) @ ab) / ab_len2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(query - proj, axis=1)
        mask = d < best_dist
        best_dist[mask] = d[mask]
        best_point[mask] = proj[mask]

    return best_dist, best_point
