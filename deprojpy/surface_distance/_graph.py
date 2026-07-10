from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

from ._calculator import SurfaceDistanceCalculator
from ._helpers import (
    _as_pixel_xy,
    _validate_positive,
    _validate_xy_array,
    _validate_xy_point,
)


def choose_surface_graph_step(
    shape: tuple[int, ...],
    *,
    target_nodes: int = 80_000,
    min_step: int = 1,
    max_step: int = 16,
) -> int:
    """Choose a grid step that keeps a surface graph near ``target_nodes``."""
    if len(shape) != 2:
        raise ValueError("shape must be a (height, width) tuple")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    target_nodes = int(target_nodes)
    if target_nodes <= 0:
        raise ValueError("target_nodes must be positive")
    raw_step = math.sqrt((height * width) / target_nodes)
    step = int(math.ceil(raw_step))
    return int(np.clip(step, int(min_step), int(max_step)))


def _sampled_axis(length: int, step: int) -> np.ndarray:
    values = np.arange(0, length, step, dtype=np.int64)
    if values.size == 0 or values[-1] != length - 1:
        values = np.r_[values, length - 1]
    return values


def _connectivity_offsets(connectivity: str) -> list[tuple[int, int]]:
    offsets = {
        "4": [(1, 0), (0, 1)],
        "8": [(1, 0), (0, 1), (1, 1), (1, -1)],
        "16": [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 1),
            (2, -1),
            (1, 2),
            (-1, 2),
        ],
    }
    if connectivity not in offsets:
        raise ValueError("connectivity must be one of '4', '8', or '16'")
    return offsets[connectivity]


@dataclass
class SurfaceGraph:
    """
    Sparse graph approximation to geodesic distances on a height-map surface.

    ``SurfaceGraph`` represents a sampled version of the height-field surface
    ``r(x, y) = (x, y, h(x, y))`` as a sparse weighted graph. Each graph node is
    an ``(x, y)`` pixel location lifted to the surface, and each edge weight is
    the local 3-D distance between two sampled surface points. Dijkstra shortest
    paths on this graph approximate surface geodesic distances.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse adjacency matrix. Entry ``graph[i, j]`` is the physical 3-D
        distance between sampled surface nodes ``i`` and ``j``. The graph is
        treated as undirected by distance methods.
    xy : np.ndarray
        ``(n_nodes, 2)`` array of node coordinates in geometric ``(x, y)`` pixel
        units. These are the coordinates returned in shortest paths.
    z : np.ndarray
        Height values for each node. Values are interpreted together with
        ``voxel_depth`` when edge weights are constructed.
    shape : tuple[int, int]
        Shape of the original height map in array order ``(rows, columns)``.
    step : int
        Grid sampling step in pixels. ``step=1`` uses every pixel. Larger values
        build smaller, faster, more approximate graphs. If constructed with
        ``step="auto"``, the step is selected by :func:`choose_surface_graph_step`.
    pixel_size : float
        Physical size of one pixel in ``x`` and ``y``. Graph distances are
        returned in these physical units.
    voxel_depth : float
        Physical scale factor for height values used during graph construction.
        For prepared physical height maps this is typically ``1.0``.
    connectivity : {"4", "8", "16"}
        Neighborhood stencil used to add graph edges:

        - ``"4"`` connects horizontal and vertical neighbors. It is fastest but
          strongly grid-biased and gives Manhattan-like distances on flat
          surfaces.
        - ``"8"`` also connects diagonal neighbors. It is a good default for
          many uses.
        - ``"16"`` adds longer oblique moves such as ``(2, 1)`` and ``(1, 2)``.
          It reduces directional bias at higher memory and compute cost.

    Coordinate and unit conventions
    -------------------------------
    Public methods accept ``(x, y)`` pixel coordinates by default, with ``x`` as
    column and ``y`` as row. Pass ``input_units="physical"`` when points are in
    the same physical units as result boundaries and centers. Input points are
    converted to pixel units and snapped to their nearest graph nodes before
    Dijkstra searches. Returned distances are physical lengths using
    ``pixel_size`` for ``x/y`` and ``voxel_depth`` for height differences as
    encoded in the graph edge weights. Returned paths are arrays of ``(x, y)``
    pixel coordinates along graph nodes, not physical coordinates.

    Construction helpers
    --------------------
    :meth:`from_calculator` builds a graph from a
    :class:`SurfaceDistanceCalculator`, reusing its prepared height map and
    scale factors. :meth:`from_heightmap` builds directly from a height map when
    scale factors are supplied manually. The lower-level :func:`build_surface_graph`
    function performs the same construction.

    Notes
    -----
    Graph distances are approximate. They depend on ``step`` and
    ``connectivity`` and should not be described as exact continuous geodesics.
    For one source against many targets, prefer :meth:`distances_from_source`;
    it runs one Dijkstra search and indexes the requested targets.
    """

    graph: sparse.csr_matrix
    xy: np.ndarray
    z: np.ndarray
    shape: tuple[int, ...]
    step: int
    pixel_size: float
    voxel_depth: float
    connectivity: str
    _tree: KDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tree = KDTree(np.asarray(self.xy, dtype=float))

    @classmethod
    def from_calculator(
        cls,
        calculator: SurfaceDistanceCalculator,
        *,
        step: int | str = "auto",
        target_nodes: int = 80_000,
        connectivity: str = "8",
    ) -> "SurfaceGraph":
        """
        Build a sparse surface graph from a distance calculator.

        Parameters
        ----------
        calculator : SurfaceDistanceCalculator
            Calculator whose prepared height map and scale factors define the
            surface. The graph uses ``calculator.heightmap``,
            ``calculator.pixel_size``, and ``calculator.voxel_depth``.
        step : int or "auto", optional
            Grid sampling step in pixels. ``1`` includes every pixel.
            ``"auto"`` chooses a step with :func:`choose_surface_graph_step`
            using ``target_nodes``.
        target_nodes : int, optional
            Approximate maximum node count used when ``step="auto"``.
        connectivity : {"4", "8", "16"}, optional
            Edge stencil. ``"4"`` is most grid-biased, ``"8"`` is the default,
            and ``"16"`` reduces directional bias at higher cost.

        Returns
        -------
        SurfaceGraph
            Sparse graph whose distances are in ``calculator.units``-compatible
            physical units.
        """
        return build_surface_graph(
            calculator.heightmap,
            pixel_size=calculator.pixel_size,
            voxel_depth=calculator.voxel_depth,
            step=step,
            target_nodes=target_nodes,
            connectivity=connectivity,
        )

    @classmethod
    def from_heightmap(
        cls,
        heightmap: np.ndarray,
        *,
        pixel_size: float,
        voxel_depth: float = 1.0,
        step: int | str = "auto",
        target_nodes: int = 80_000,
        connectivity: str = "8",
    ) -> "SurfaceGraph":
        """
        Build a sparse surface graph directly from a height map.

        Parameters
        ----------
        heightmap : np.ndarray
            Two-dimensional surface raster indexed as
            ``heightmap[row=y, column=x]``. Values are multiplied by
            ``voxel_depth`` when edge weights are constructed.
        pixel_size : float
            Physical size of one pixel in the ``x`` and ``y`` directions.
        voxel_depth : float, optional
            Physical scale factor for height values. Use ``1.0`` for height maps
            already in physical z units.
        step : int or "auto", optional
            Grid sampling step in pixels. Larger values create coarser, faster,
            more approximate graphs. ``"auto"`` uses ``target_nodes`` to choose
            a step.
        target_nodes : int, optional
            Approximate maximum node count used when ``step="auto"``.
        connectivity : {"4", "8", "16"}, optional
            Edge stencil used to connect sampled grid nodes.

        Returns
        -------
        SurfaceGraph
            Sparse graph suitable for approximate geodesic distance queries.
        """
        return build_surface_graph(
            heightmap,
            pixel_size=pixel_size,
            voxel_depth=voxel_depth,
            step=step,
            target_nodes=target_nodes,
            connectivity=connectivity,
        )

    def nearest_nodes(
        self,
        points_xy: np.ndarray,
        *,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        """
        Return nearest graph node indices for ``(x, y)`` points.

        Parameters
        ----------
        points_xy : np.ndarray
            Coordinates with shape ``(n, 2)`` in ``(x, y)`` order.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.
        """
        points = _validate_xy_array(
            _as_pixel_xy(points_xy, pixel_size=self.pixel_size, input_units=input_units),
            name="points_xy",
        )
        _, indices = self._tree.query(points)
        return np.asarray(indices, dtype=np.int64)

    def distance(
        self,
        p0_xy,
        p1_xy,
        *,
        return_path: bool = False,
        input_units: Literal["pixel", "physical"] = "pixel",
    ):
        """
        Return approximate graph-geodesic distance between two points.

        Parameters
        ----------
        p0_xy, p1_xy
            Endpoints in ``(x, y)`` order.
        return_path : bool, optional
            If ``True``, also return the shortest graph path. The returned path
            xy coordinates are in pixel units matching graph node coordinates,
            even when ``input_units="physical"``.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.
        """
        endpoints = np.vstack(
            [
                _validate_xy_point(
                    _as_pixel_xy(p0_xy, pixel_size=self.pixel_size, input_units=input_units),
                    name="p0_xy",
                ),
                _validate_xy_point(
                    _as_pixel_xy(p1_xy, pixel_size=self.pixel_size, input_units=input_units),
                    name="p1_xy",
                ),
            ]
        )
        source, target = self.nearest_nodes(endpoints, input_units="pixel")
        if return_path:
            dist, predecessors = dijkstra(
                self.graph,
                directed=False,
                indices=int(source),
                return_predecessors=True,
            )
            distance_value = float(dist[int(target)])
            path = self._path_from_predecessors(predecessors, int(source), int(target))
            return distance_value, path
        dist = dijkstra(self.graph, directed=False, indices=int(source))
        return float(dist[int(target)])

    def distances_from_source(
        self,
        source_xy,
        target_xy: np.ndarray | None = None,
        *,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        """
        Return graph-geodesic distances from one source to targets.

        Parameters
        ----------
        source_xy
            Source coordinate in ``(x, y)`` order.
        target_xy : np.ndarray, optional
            Target coordinates with shape ``(n, 2)`` in ``(x, y)`` order. If
            omitted, distances to every graph node are returned.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.

        Returns
        -------
        np.ndarray
            Graph-geodesic distances in physical units. Input points are snapped
            to their nearest graph nodes.
        """
        source_point = _validate_xy_point(
            _as_pixel_xy(source_xy, pixel_size=self.pixel_size, input_units=input_units),
            name="source_xy",
        )
        source = int(self.nearest_nodes(np.asarray([source_point]), input_units="pixel")[0])
        dist = np.asarray(dijkstra(self.graph, directed=False, indices=source), dtype=float)
        if target_xy is None:
            return dist
        targets = self.nearest_nodes(target_xy, input_units=input_units)
        return dist[targets]

    def _path_from_predecessors(
        self,
        predecessors: np.ndarray,
        source: int,
        target: int,
    ) -> np.ndarray:
        if source == target:
            return self.xy[[source]].copy()
        if predecessors[target] < 0:
            return np.empty((0, 2), dtype=float)
        nodes = [target]
        current = target
        while current != source:
            current = int(predecessors[current])
            if current < 0:
                return np.empty((0, 2), dtype=float)
            nodes.append(current)
        nodes.reverse()
        return self.xy[np.asarray(nodes, dtype=np.int64)].copy()


def build_surface_graph(
    heightmap,
    *,
    pixel_size,
    voxel_depth=1.0,
    step: int | str = "auto",
    target_nodes: int = 80_000,
    connectivity: str = "8",
) -> SurfaceGraph:
    """
    Build a sparse graph approximation of a height-map surface.

    The graph samples the image grid every ``step`` pixels. Edges connect nearby
    sampled grid points and are weighted by 3-D Euclidean distances.
    """
    h = np.asarray(heightmap, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    pixel_size = _validate_positive(pixel_size, "pixel_size")
    voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
    if step == "auto":
        step = choose_surface_graph_step(h.shape, target_nodes=target_nodes)
        if step > 1:
            warnings.warn(
                f"Building an approximate surface graph with step={step}; "
                "increase target_nodes or pass a smaller step for higher accuracy.",
                RuntimeWarning,
                stacklevel=2,
            )
    step = int(step)
    if step <= 0:
        raise ValueError("step must be a positive integer or 'auto'")

    ys = _sampled_axis(h.shape[0], step)
    xs = _sampled_axis(h.shape[1], step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    z = h[yy.ravel(), xx.ravel()].astype(float)
    n_rows, n_cols = len(ys), len(xs)
    n_nodes = n_rows * n_cols
    if n_nodes > 200_000:
        warnings.warn(
            "surface graph has more than 200,000 nodes; construction and "
            "Dijkstra searches may be slow. Increase step or reduce target_nodes.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    offsets = _connectivity_offsets(str(connectivity))

    # Use one direction per stencil edge, then add both orientations to create
    # an undirected sparse graph without relying on implicit symmetry.
    for row in range(n_rows):
        for col in range(n_cols):
            source = row * n_cols + col
            for drow, dcol in offsets:
                rr = row + drow
                cc = col + dcol
                if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
                    continue
                target = rr * n_cols + cc
                dx = (xy[target, 0] - xy[source, 0]) * pixel_size
                dy = (xy[target, 1] - xy[source, 1]) * pixel_size
                dz = (z[target] - z[source]) * voxel_depth
                weight = math.sqrt(dx * dx + dy * dy + dz * dz)
                rows.extend([source, target])
                cols.extend([target, source])
                data.extend([weight, weight])

    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    return SurfaceGraph(
        graph=graph,  # type: ignore[arg-type]
        xy=xy,
        z=z,
        shape=h.shape,
        step=step,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        connectivity=str(connectivity),
    )
