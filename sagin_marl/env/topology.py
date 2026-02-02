from __future__ import annotations

import numpy as np


def thomas_cluster_process(
    num_points: int,
    map_size: float,
    num_clusters: int = 3,
    cluster_std: float = 80.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    num_clusters = max(1, int(num_clusters))
    # Sample cluster centers uniformly
    centers = rng.uniform(0.1 * map_size, 0.9 * map_size, size=(num_clusters, 2))
    # Allocate points per cluster
    counts = rng.multinomial(num_points, [1 / num_clusters] * num_clusters)
    points = []
    for c, n in zip(centers, counts):
        if n == 0:
            continue
        pts = rng.normal(loc=c, scale=cluster_std, size=(n, 2))
        pts = np.clip(pts, 0.0, map_size)
        points.append(pts)
    if not points:
        return rng.uniform(0.0, map_size, size=(num_points, 2))
    return np.vstack(points)
