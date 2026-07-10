from __future__ import annotations

import numpy as np

from ..models import DeprojResult
from ._calculator import SurfaceDistanceCalculator
from ._graph import SurfaceGraph
from ._helpers import cell_centers_xy_pixels


def surface_straight_cell_distance(
    result: DeprojResult,
    heightmap: np.ndarray | None,
    i: int,
    j: int,
    *,
    prepared: bool = True,
    **kwargs,
) -> float:
    """Return straight-line surface distance between two result cell centers."""
    calc = SurfaceDistanceCalculator.from_result(result, heightmap, prepared=prepared)
    centers = cell_centers_xy_pixels(result)
    return calc.straight_distance(centers[int(i)], centers[int(j)], **kwargs)


def surface_graph_cell_distance(
    result: DeprojResult,
    surface_graph: SurfaceGraph,
    i: int,
    j: int,
    *,
    return_path=False,
):
    """Return graph-geodesic surface distance between two result cell centers."""
    centers = cell_centers_xy_pixels(result)
    return surface_graph.distance(centers[int(i)], centers[int(j)], return_path=return_path)
