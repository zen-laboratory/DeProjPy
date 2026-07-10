from __future__ import annotations

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)

from ._helpers import _validate_positive, _validate_xy_array


def _average_duplicate_xy(points_xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average z-values for exact duplicate xy coordinates."""
    points = _validate_xy_array(points_xy, name="points_xy")
    values = np.asarray(z, dtype=float)
    if values.ndim != 1 or len(values) != len(points):
        raise ValueError("z must be a 1-D array with one value per xy point")
    unique_xy, inverse = np.unique(points, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return unique_xy, sums / counts


def _boundary_interpolator(method: str, points_xy: np.ndarray, z: np.ndarray):
    if method == "linear":
        return LinearNDInterpolator(points_xy, z, fill_value=np.nan)
    if method in {"clough", "clough_tocher"}:
        return CloughTocher2DInterpolator(points_xy, z, fill_value=np.nan)
    if method == "nearest":
        return NearestNDInterpolator(points_xy, z)
    raise ValueError("method must be one of 'linear', 'clough', 'clough_tocher', or 'nearest'")


def _linear_plane_extrapolate(
    points_xy: np.ndarray,
    z: np.ndarray,
    grid_xy: np.ndarray,
) -> np.ndarray:
    """Evaluate the least-squares plane fitted to scattered xyz samples."""
    design = np.column_stack([points_xy[:, 0], points_xy[:, 1], np.ones(len(points_xy))])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    grid_design = np.column_stack([grid_xy[:, 0], grid_xy[:, 1], np.ones(len(grid_xy))])
    return grid_design @ coeffs


def heightmap_from_cell_boundaries(
    result,
    *,
    shape: tuple[int, ...] | None = None,
    method: str = "linear",
    extrapolation: str = "linear",
    fill_value: float | None = None,
) -> np.ndarray:
    """
    Interpolate a raster height map from deprojected cell-boundary points.

    Boundary coordinates are expected in physical ``(x, y, z)`` order. The
    returned raster stores physical z values and is indexed as
    ``surface[row=y, column=x]``.
    """
    if extrapolation not in {"linear", "nearest", "constant", "none"}:
        raise ValueError("extrapolation must be one of 'linear', 'nearest', 'constant', or 'none'")
    pixel_size = _validate_positive(result.pixel_size, "result.pixel_size")
    boundaries = []
    for cell in getattr(result, "epicells", []):
        boundary = np.asarray(cell.boundary, dtype=float)
        if boundary.ndim != 2 or boundary.shape[1] < 3:
            raise ValueError("each cell boundary must have shape (n, >=3)")
        finite = np.all(np.isfinite(boundary[:, :3]), axis=1)
        if np.any(finite):
            boundaries.append(boundary[finite, :3])
    if not boundaries:
        raise ValueError("result contains no finite cell-boundary xyz points")

    points_xyz = np.vstack(boundaries)
    points_xy_px = points_xyz[:, :2] / pixel_size
    points_z = points_xyz[:, 2]

    # Adjacent cells can contribute the same xy boundary point with slightly
    # different z-values. Average exact duplicates before scattered interpolation.
    points_xy_px, points_z = _average_duplicate_xy(points_xy_px, points_z)

    if shape is None:
        prepared = getattr(result, "prepared_heightmap", None)
        if prepared is not None:
            shape = np.asarray(prepared).shape
        else:
            max_x = float(np.nanmax(points_xy_px[:, 0]))
            max_y = float(np.nanmax(points_xy_px[:, 1]))
            shape = (int(np.ceil(max_y)) + 1, int(np.ceil(max_x)) + 1)
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError("shape must be a positive (rows, columns) tuple")
    height, width = int(shape[0]), int(shape[1])

    yy, xx = np.mgrid[:height, :width]
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    interpolator = _boundary_interpolator(method, points_xy_px, points_z)
    surface = np.asarray(interpolator(grid_xy), dtype=float).reshape(height, width)
    missing = ~np.isfinite(surface)
    if not np.any(missing):
        return surface

    if extrapolation == "none":
        return surface
    if extrapolation == "nearest":
        nearest = NearestNDInterpolator(points_xy_px, points_z)
        surface[missing] = nearest(grid_xy[missing.ravel()])
        return surface
    if extrapolation == "constant":
        surface[missing] = np.nan if fill_value is None else float(fill_value)
        return surface
    if extrapolation == "linear":
        plane_values = _linear_plane_extrapolate(points_xy_px, points_z, grid_xy)
        surface[missing] = plane_values.reshape(height, width)[missing]
        return surface
    raise AssertionError("unreachable extrapolation branch")
