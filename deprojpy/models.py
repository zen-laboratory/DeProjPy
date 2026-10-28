from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SERIALIZATION_FORMAT_VERSION = "1.1"
SUPPORTED_SERIALIZATION_FORMAT_VERSIONS = frozenset({"1.0", SERIALIZATION_FORMAT_VERSION})

DATAFRAME_COLUMNS = [
    "id",
    "source_label",
    "xc",
    "yc",
    "zc",
    "surface_centroid_x",
    "surface_centroid_y",
    "surface_centroid_z",
    "area",
    "perimeter",
    "euler_alpha",
    "euler_beta",
    "euler_gamma",
    "curv_mean",
    "curv_gauss",
    "curv_k1",
    "curv_k2",
    "ellipse_xc",
    "ellipse_yc",
    "ellipse_zc",
    "ellipse_a",
    "ellipse_b",
    "ellipse_theta",
    "eccentricity",
    "proj_direction",
    "uncorrected_area",
    "uncorrected_perimeter",
    "n_neighbors",
    "area_error",
    "perimeter_error",
]


def _final_serialization_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".npz")
    return path


def _require_keys(data: np.lib.npyio.NpzFile, keys: set[str]) -> None:
    missing = sorted(keys.difference(data.files))
    if missing:
        raise ValueError(f"serialized DeprojResult is missing required field(s): {missing}")


def _as_scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar string")
    return str(value.item())


def _as_scalar_float(data: np.lib.npyio.NpzFile, key: str) -> float:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar")
    scalar = float(value.item())
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return scalar


def _as_1d_int(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(data[key], dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{key} must be a one-dimensional integer array")
    return values


def _as_1d_float(data: np.lib.npyio.NpzFile, key: str, count: int) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.shape != (count,):
        raise ValueError(f"{key} must have shape ({count},)")
    return values


def _as_2d_float(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return values


def _validate_offsets(offsets: np.ndarray, points_length: int, count: int, field: str) -> None:
    if offsets.shape != (count + 1,):
        raise ValueError(f"{field}_offsets must have length {count + 1}")
    if len(offsets) == 0 or offsets[0] != 0:
        raise ValueError(f"{field}_offsets must start at 0")
    if offsets[-1] != points_length:
        raise ValueError(f"{field}_offsets must end at the number of packed points")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{field}_offsets must be monotonically nondecreasing")


def _pack_arrays(
    arrays: list[np.ndarray],
    *,
    width: int,
    dtype=float,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    packed_parts = []
    cursor = 0
    for index, array in enumerate(arrays):
        values = np.asarray(array, dtype=dtype)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"ragged array {index} must have shape (n, {width})")
        packed_parts.append(values)
        cursor += len(values)
        offsets[index + 1] = cursor
    packed = (
        np.vstack(packed_parts).astype(dtype, copy=False)
        if packed_parts
        else np.empty((0, width), dtype=dtype)
    )
    return packed, offsets


def _pack_1d_arrays(arrays: list[np.ndarray], *, dtype=np.int64) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    packed_parts = []
    cursor = 0
    for index, array in enumerate(arrays):
        values = np.asarray(array, dtype=dtype)
        if values.ndim != 1:
            raise ValueError(f"ragged vector {index} must be one-dimensional")
        packed_parts.append(values)
        cursor += len(values)
        offsets[index + 1] = cursor
    packed = (
        np.concatenate(packed_parts).astype(dtype, copy=False)
        if packed_parts
        else np.empty((0,), dtype=dtype)
    )
    return packed, offsets


def _validated_surface_centroids(epicells: list["Epicell"]) -> np.ndarray:
    values = np.empty((len(epicells), 3), dtype=float)
    for index, cell in enumerate(epicells):
        centroid = np.asarray(cell.surface_centroid, dtype=float)
        if centroid.shape != (3,):
            raise ValueError(f"Epicell {index} surface_centroid must have shape (3,)")
        if not (np.all(np.isfinite(centroid)) or np.all(np.isnan(centroid))):
            raise ValueError(
                f"Epicell {index} surface_centroid must be entirely finite or entirely NaN"
            )
        values[index] = centroid
    return values


def _serialize_epicells(epicells: list["Epicell"]) -> dict[str, np.ndarray]:
    boundaries, boundary_offsets = _pack_arrays([cell.boundary for cell in epicells], width=3)
    junction_ids, junction_offsets = _pack_1d_arrays(
        [cell.junction_ids for cell in epicells],
        dtype=np.int64,
    )
    source_label_present = np.asarray(
        [cell.source_label is not None for cell in epicells],
        dtype=bool,
    )
    source_labels = np.asarray(
        [int(cell.source_label) if cell.source_label is not None else 0 for cell in epicells],
        dtype=np.int64,
    )
    return {
        "cell_count": np.asarray(len(epicells), dtype=np.int64),
        "cell_ids": np.asarray([cell.id for cell in epicells], dtype=np.int64),
        "cell_source_label_present": source_label_present,
        "cell_source_labels": source_labels,
        "cell_boundary_points": boundaries,
        "cell_boundary_offsets": boundary_offsets,
        "cell_centers": np.asarray([cell.center for cell in epicells], dtype=float).reshape(-1, 3),
        "cell_surface_centroids": _validated_surface_centroids(epicells),
        "cell_junction_ids": junction_ids,
        "cell_junction_offsets": junction_offsets,
        "cell_n_neighbors": np.asarray([cell.n_neighbors for cell in epicells], dtype=np.int64),
        "cell_area": np.asarray([cell.area for cell in epicells], dtype=float),
        "cell_perimeter": np.asarray([cell.perimeter for cell in epicells], dtype=float),
        "cell_euler_angles": np.asarray(
            [cell.euler_angles for cell in epicells],
            dtype=float,
        ).reshape(-1, 3),
        "cell_curvatures": np.asarray(
            [cell.curvatures for cell in epicells],
            dtype=float,
        ).reshape(-1, 4),
        "cell_ellipse_fit": np.asarray(
            [cell.ellipse_fit for cell in epicells],
            dtype=float,
        ).reshape(-1, 6),
        "cell_eccentricity": np.asarray(
            [cell.eccentricity for cell in epicells],
            dtype=float,
        ),
        "cell_proj_direction": np.asarray([cell.proj_direction for cell in epicells], dtype=float),
        "cell_uncorrected_area": np.asarray(
            [cell.uncorrected_area for cell in epicells],
            dtype=float,
        ),
        "cell_uncorrected_perimeter": np.asarray(
            [cell.uncorrected_perimeter for cell in epicells],
            dtype=float,
        ),
        "cell_area_error": np.asarray([cell.area_error for cell in epicells], dtype=float),
        "cell_perimeter_error": np.asarray(
            [cell.perimeter_error for cell in epicells],
            dtype=float,
        ),
    }


def _deserialize_epicells(
    data: np.lib.npyio.NpzFile, *, format_version: str
) -> list["Epicell"]:
    count_array = np.asarray(data["cell_count"])
    if count_array.ndim != 0:
        raise ValueError("cell_count must be a scalar")
    count = int(count_array.item())
    if count < 0:
        raise ValueError("cell_count must be nonnegative")

    cell_ids = _as_1d_int(data, "cell_ids")
    source_label_present = np.asarray(data["cell_source_label_present"], dtype=bool)
    source_labels = _as_1d_int(data, "cell_source_labels")
    n_neighbors = _as_1d_int(data, "cell_n_neighbors")
    if not (
        cell_ids.shape
        == source_label_present.shape
        == source_labels.shape
        == n_neighbors.shape
        == (count,)
    ):
        raise ValueError("per-cell integer fields must all have length cell_count")
    if len(set(map(int, cell_ids))) != count:
        raise ValueError("cell_ids must be unique")

    boundary_points = np.asarray(data["cell_boundary_points"], dtype=float)
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 3:
        raise ValueError("cell_boundary_points must have shape (n, 3)")
    boundary_offsets = _as_1d_int(data, "cell_boundary_offsets")
    _validate_offsets(boundary_offsets, len(boundary_points), count, "cell_boundary")

    junction_ids = _as_1d_int(data, "cell_junction_ids")
    junction_offsets = _as_1d_int(data, "cell_junction_offsets")
    _validate_offsets(junction_offsets, len(junction_ids), count, "cell_junction")
    junction_counts = np.diff(junction_offsets)
    if np.any(n_neighbors < 0) or np.any(n_neighbors != junction_counts):
        raise ValueError("cell_n_neighbors must match each cell's serialized junction id count")

    centers = _as_2d_float(data, "cell_centers", (count, 3))
    if format_version == "1.0":
        surface_centroids = np.full((count, 3), np.nan, dtype=float)
    else:
        surface_centroids = _as_2d_float(data, "cell_surface_centroids", (count, 3))
        finite_rows = np.all(np.isfinite(surface_centroids), axis=1)
        unavailable_rows = np.all(np.isnan(surface_centroids), axis=1)
        if not np.all(finite_rows | unavailable_rows):
            raise ValueError(
                "cell_surface_centroids rows must be entirely finite or entirely NaN"
            )
    euler_angles = _as_2d_float(data, "cell_euler_angles", (count, 3))
    curvatures = _as_2d_float(data, "cell_curvatures", (count, 4))
    ellipse_fit = _as_2d_float(data, "cell_ellipse_fit", (count, 6))

    areas = _as_1d_float(data, "cell_area", count)
    perimeters = _as_1d_float(data, "cell_perimeter", count)
    eccentricity = _as_1d_float(data, "cell_eccentricity", count)
    proj_direction = _as_1d_float(data, "cell_proj_direction", count)
    uncorrected_area = _as_1d_float(data, "cell_uncorrected_area", count)
    uncorrected_perimeter = _as_1d_float(data, "cell_uncorrected_perimeter", count)
    area_error = _as_1d_float(data, "cell_area_error", count)
    perimeter_error = _as_1d_float(data, "cell_perimeter_error", count)

    epicells: list[Epicell] = []
    for index in range(count):
        source_label = int(source_labels[index]) if source_label_present[index] else None
        epicells.append(
            Epicell(
                id=int(cell_ids[index]),
                source_label=source_label,
                boundary=boundary_points[boundary_offsets[index] : boundary_offsets[index + 1]],
                center=centers[index],
                junction_ids=junction_ids[junction_offsets[index] : junction_offsets[index + 1]],
                n_neighbors=int(n_neighbors[index]),
                area=float(areas[index]),
                perimeter=float(perimeters[index]),
                euler_angles=euler_angles[index],
                curvatures=curvatures[index],
                ellipse_fit=ellipse_fit[index],
                eccentricity=float(eccentricity[index]),
                proj_direction=float(proj_direction[index]),
                uncorrected_area=float(uncorrected_area[index]),
                uncorrected_perimeter=float(uncorrected_perimeter[index]),
                area_error=float(area_error[index]),
                perimeter_error=float(perimeter_error[index]),
                surface_centroid=surface_centroids[index],
            )
        )
    return epicells


def _serialize_junction_graph(graph: nx.Graph) -> dict[str, np.ndarray]:
    node_ids = np.asarray(list(graph.nodes), dtype=np.int64)
    node_centroids = []
    node_has_labels = []
    node_labels = []
    node_has_pixel_coords = []
    node_pixel_coords = []
    for node_id in node_ids:
        data = graph.nodes[int(node_id)]
        centroid = np.asarray(data.get("centroid"), dtype=float)
        if centroid.shape != (3,):
            raise ValueError("junction graph node centroid must have shape (3,)")
        node_centroids.append(centroid)
        if "labels" in data:
            node_has_labels.append(True)
            node_labels.append(np.asarray(sorted(map(int, data["labels"])), dtype=np.int64))
        else:
            node_has_labels.append(False)
            node_labels.append(np.empty((0,), dtype=np.int64))
        if "pixel_coords" in data:
            coords = np.asarray(data["pixel_coords"], dtype=np.int64)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("junction graph node pixel_coords must have shape (n, 2)")
            node_has_pixel_coords.append(True)
            node_pixel_coords.append(coords)
        else:
            node_has_pixel_coords.append(False)
            node_pixel_coords.append(np.empty((0, 2), dtype=np.int64))

    label_values, label_offsets = _pack_1d_arrays(node_labels, dtype=np.int64)
    pixel_values, pixel_offsets = _pack_arrays(node_pixel_coords, width=2, dtype=np.int64)
    edges = np.asarray(list(graph.edges), dtype=np.int64).reshape(-1, 2)
    return {
        "junction_node_ids": node_ids,
        "junction_node_centroids": np.asarray(node_centroids, dtype=float).reshape(-1, 3),
        "junction_node_has_labels": np.asarray(node_has_labels, dtype=bool),
        "junction_node_label_values": label_values,
        "junction_node_label_offsets": label_offsets,
        "junction_node_has_pixel_coords": np.asarray(node_has_pixel_coords, dtype=bool),
        "junction_node_pixel_coords": pixel_values,
        "junction_node_pixel_offsets": pixel_offsets,
        "junction_edge_sources": edges[:, 0].astype(np.int64, copy=False),
        "junction_edge_targets": edges[:, 1].astype(np.int64, copy=False),
    }


def _deserialize_junction_graph(data: np.lib.npyio.NpzFile) -> nx.Graph:
    node_ids = _as_1d_int(data, "junction_node_ids")
    count = len(node_ids)
    if len(set(map(int, node_ids))) != count:
        raise ValueError("junction_node_ids must be unique")

    centroids = _as_2d_float(data, "junction_node_centroids", (count, 3))
    has_labels = np.asarray(data["junction_node_has_labels"], dtype=bool)
    if has_labels.shape != (count,):
        raise ValueError("junction_node_has_labels must have length junction node count")
    label_values = _as_1d_int(data, "junction_node_label_values")
    label_offsets = _as_1d_int(data, "junction_node_label_offsets")
    _validate_offsets(label_offsets, len(label_values), count, "junction_node_label")

    has_pixel_coords = np.asarray(data["junction_node_has_pixel_coords"], dtype=bool)
    if has_pixel_coords.shape != (count,):
        raise ValueError("junction_node_has_pixel_coords must have length junction node count")
    pixel_coords = np.asarray(data["junction_node_pixel_coords"], dtype=np.int64)
    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        raise ValueError("junction_node_pixel_coords must have shape (n, 2)")
    pixel_offsets = _as_1d_int(data, "junction_node_pixel_offsets")
    _validate_offsets(pixel_offsets, len(pixel_coords), count, "junction_node_pixel")

    graph = nx.Graph()
    for index, node_id in enumerate(node_ids):
        node = int(node_id)
        graph.add_node(node, id=node, centroid=centroids[index])
        if has_labels[index]:
            labels = label_values[label_offsets[index] : label_offsets[index + 1]]
            graph.nodes[node]["labels"] = frozenset(map(int, labels))
        if has_pixel_coords[index]:
            graph.nodes[node]["pixel_coords"] = pixel_coords[
                pixel_offsets[index] : pixel_offsets[index + 1]
            ]

    sources = _as_1d_int(data, "junction_edge_sources")
    targets = _as_1d_int(data, "junction_edge_targets")
    if sources.shape != targets.shape:
        raise ValueError("junction_edge_sources and junction_edge_targets must have equal length")
    node_set = set(map(int, node_ids))
    for source, target in zip(sources, targets, strict=True):
        source_id = int(source)
        target_id = int(target)
        if source_id not in node_set or target_id not in node_set:
            raise ValueError("junction edge endpoint is not present in junction_node_ids")
        graph.add_edge(source_id, target_id)
    return graph


def _serialize_prepared_heightmap(heightmap: np.ndarray | None) -> dict[str, np.ndarray]:
    if heightmap is None:
        return {
            "has_prepared_heightmap": np.asarray(False),
            "prepared_heightmap": np.empty((0, 0), dtype=float),
        }
    values = np.asarray(heightmap, dtype=float)
    if values.ndim != 2:
        raise ValueError("prepared_heightmap must be a two-dimensional array")
    return {
        "has_prepared_heightmap": np.asarray(True),
        "prepared_heightmap": values,
    }


@dataclass
class Epicell:
    """Measurements and geometry for one deprojected epithelial cell.

    ``boundary`` has shape ``(n, 3)`` and both centers have shape ``(3,)`` in
    geometric ``(x, y, z)`` order. ``center`` averages simplified boundary
    vertices; ``surface_centroid`` weights footprint pixels by reconstructed
    surface area and can lie off the surface. Coordinates, lengths, and areas
    use the parent result's physical units, except angles (radians),
    eccentricity, and neighbor counts.
    """

    id: int
    source_label: int | None
    boundary: np.ndarray
    center: np.ndarray
    junction_ids: np.ndarray
    n_neighbors: int
    area: float
    perimeter: float
    euler_angles: np.ndarray
    curvatures: np.ndarray
    ellipse_fit: np.ndarray
    eccentricity: float
    proj_direction: float
    uncorrected_area: float
    uncorrected_perimeter: float
    area_error: float
    perimeter_error: float
    surface_centroid: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=float)
    )


@dataclass
class DeprojResult:
    """Collection returned by :func:`deprojpy.from_heightmap`.

    The source image shape is stored in array ``(row, column)`` order, while
    all cell boundaries, centers, and junction centroids are geometric
    ``(x, y, z)`` coordinates. ``pixel_size`` and ``voxel_depth`` record the
    conversions used to express those coordinates in ``units``.
    """

    epicells: list[Epicell]
    junction_graph: nx.Graph
    units: str = "pixels"
    pixel_size: float = 1.0
    voxel_depth: float = 1.0
    source_shape: tuple[int, int] = (0, 0)
    prepared_heightmap: np.ndarray | None = field(default=None, repr=False)

    @property
    def cells(self) -> list[Epicell]:
        return self.epicells

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per cell using the stable morphology column schema.

        Position, length, and area columns use this result's physical
        ``units``; Euler/orientation columns are radians and curvature columns
        use inverse physical units. An empty result still returns all columns.
        """
        rows = []
        for cell in self.epicells:
            rows.append(
                {
                    "id": cell.id,
                    "source_label": cell.source_label,
                    "xc": cell.center[0],
                    "yc": cell.center[1],
                    "zc": cell.center[2],
                    "surface_centroid_x": cell.surface_centroid[0],
                    "surface_centroid_y": cell.surface_centroid[1],
                    "surface_centroid_z": cell.surface_centroid[2],
                    "area": cell.area,
                    "perimeter": cell.perimeter,
                    "euler_alpha": cell.euler_angles[0],
                    "euler_beta": cell.euler_angles[1],
                    "euler_gamma": cell.euler_angles[2],
                    "curv_mean": cell.curvatures[0],
                    "curv_gauss": cell.curvatures[1],
                    "curv_k1": cell.curvatures[2],
                    "curv_k2": cell.curvatures[3],
                    "ellipse_xc": cell.ellipse_fit[0],
                    "ellipse_yc": cell.ellipse_fit[1],
                    "ellipse_zc": cell.ellipse_fit[2],
                    "ellipse_a": cell.ellipse_fit[3],
                    "ellipse_b": cell.ellipse_fit[4],
                    "ellipse_theta": cell.ellipse_fit[5],
                    "eccentricity": cell.eccentricity,
                    "proj_direction": cell.proj_direction,
                    "uncorrected_area": cell.uncorrected_area,
                    "uncorrected_perimeter": cell.uncorrected_perimeter,
                    "n_neighbors": cell.n_neighbors,
                    "area_error": cell.area_error,
                    "perimeter_error": cell.perimeter_error,
                }
            )
        return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)

    def to_csv(self, path: str | Path, **kwargs: object) -> None:
        """Write :meth:`to_dataframe` to CSV.

        ``path`` may be a string or :class:`pathlib.Path`. Additional keyword
        arguments are forwarded to :meth:`pandas.DataFrame.to_csv`; ``index``
        defaults to ``False``.
        """
        kwargs.setdefault("index", False)
        self.to_dataframe().to_csv(path, **kwargs) # type: ignore[no-untyped-call]

    def save(
        self,
        filename: str | Path,
        *,
        compressed: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """Serialize this result to a versioned NumPy ``.npz`` archive.

        The archive stores numerical arrays and scalar metadata explicitly; it
        does not pickle ``DeprojResult``, ``Epicell``, NetworkX, interpolator, or
        calculator objects. If ``filename`` has no suffix, ``.npz`` is appended.
        Existing files are protected unless ``overwrite=True``.
        """
        path = _final_serialization_path(filename)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        source_shape = np.asarray(self.source_shape, dtype=np.int64)
        if source_shape.shape != (2,):
            raise ValueError("source_shape must be a two-element (row, column) tuple")
        arrays = {
            "format_version": np.asarray(SERIALIZATION_FORMAT_VERSION),
            "coordinate_order": np.asarray("cell_xyz_graph_xyz_image_yx"),
            "pixel_size": np.asarray(float(self.pixel_size)),
            "voxel_depth": np.asarray(float(self.voxel_depth)),
            "units": np.asarray(str(self.units)),
            "source_shape": source_shape,
            **_serialize_prepared_heightmap(self.prepared_heightmap),
            **_serialize_epicells(self.epicells),
            **_serialize_junction_graph(self.junction_graph),
        }
        save = np.savez_compressed if compressed else np.savez
        save(path, **arrays)
        return path

    @classmethod
    def load(cls, filename: str | Path) -> "DeprojResult":
        """Load a :class:`DeprojResult` saved with :meth:`save`."""
        path = Path(filename)
        required = {
            "format_version",
            "coordinate_order",
            "pixel_size",
            "voxel_depth",
            "units",
            "source_shape",
            "has_prepared_heightmap",
            "prepared_heightmap",
            "cell_count",
            "cell_ids",
            "cell_source_label_present",
            "cell_source_labels",
            "cell_boundary_points",
            "cell_boundary_offsets",
            "cell_centers",
            "cell_junction_ids",
            "cell_junction_offsets",
            "cell_n_neighbors",
            "cell_area",
            "cell_perimeter",
            "cell_euler_angles",
            "cell_curvatures",
            "cell_ellipse_fit",
            "cell_eccentricity",
            "cell_proj_direction",
            "cell_uncorrected_area",
            "cell_uncorrected_perimeter",
            "cell_area_error",
            "cell_perimeter_error",
            "junction_node_ids",
            "junction_node_centroids",
            "junction_node_has_labels",
            "junction_node_label_values",
            "junction_node_label_offsets",
            "junction_node_has_pixel_coords",
            "junction_node_pixel_coords",
            "junction_node_pixel_offsets",
            "junction_edge_sources",
            "junction_edge_targets",
        }
        with np.load(path, allow_pickle=False) as data:
            _require_keys(data, {"format_version"})
            version = _as_scalar_string(data, "format_version")
            if version not in SUPPORTED_SERIALIZATION_FORMAT_VERSIONS:
                supported = ", ".join(sorted(SUPPORTED_SERIALIZATION_FORMAT_VERSIONS))
                raise ValueError(
                    f"unsupported DeprojResult serialization format version {version!r}; "
                    f"supported versions: {supported}"
                )
            if version == "1.1":
                required.add("cell_surface_centroids")
            _require_keys(data, required)
            coordinate_order = _as_scalar_string(data, "coordinate_order")
            if coordinate_order != "cell_xyz_graph_xyz_image_yx":
                raise ValueError("coordinate_order must be 'cell_xyz_graph_xyz_image_yx'")
            pixel_size = _as_scalar_float(data, "pixel_size")
            voxel_depth = _as_scalar_float(data, "voxel_depth")
            units = _as_scalar_string(data, "units")
            source_shape_array = _as_1d_int(data, "source_shape")
            if source_shape_array.shape != (2,) or np.any(source_shape_array < 0):
                raise ValueError("source_shape must be a nonnegative two-element array")
            source_shape = (int(source_shape_array[0]), int(source_shape_array[1]))

            has_heightmap = np.asarray(data["has_prepared_heightmap"])
            if has_heightmap.ndim != 0:
                raise ValueError("has_prepared_heightmap must be a scalar boolean")
            prepared_heightmap = None
            if bool(has_heightmap.item()):
                prepared_heightmap = np.asarray(data["prepared_heightmap"], dtype=float)
                if prepared_heightmap.ndim != 2:
                    raise ValueError("prepared_heightmap must be a two-dimensional array")
                if source_shape != (0, 0) and prepared_heightmap.shape != source_shape:
                    raise ValueError("prepared_heightmap shape must match source_shape")

            epicells = _deserialize_epicells(data, format_version=version)
            graph = _deserialize_junction_graph(data)

        return cls._from_serialized_components(
            epicells=epicells,
            junction_graph=graph,
            units=units,
            pixel_size=pixel_size,
            voxel_depth=voxel_depth,
            source_shape=source_shape,
            prepared_heightmap=prepared_heightmap,
        )

    @classmethod
    def _from_serialized_components(
        cls,
        *,
        epicells: list[Epicell],
        junction_graph: nx.Graph,
        units: str,
        pixel_size: float,
        voxel_depth: float,
        source_shape: tuple[int, int],
        prepared_heightmap: np.ndarray | None,
    ) -> "DeprojResult":
        """Rebuild a result directly from validated serialized components."""
        return cls(
            epicells,
            junction_graph,
            units=units,
            pixel_size=pixel_size,
            voxel_depth=voxel_depth,
            source_shape=source_shape,
            prepared_heightmap=prepared_heightmap,
        )
