from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates


def _validate_xy_array(xy: np.ndarray, *, name: str = "xy") -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2) in (x, y) order")
    return points


def _validate_xy_point(point_xy: np.ndarray, *, name: str = "point_xy") -> np.ndarray:
    point = np.asarray(point_xy, dtype=float)
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,) in (x, y) order")
    return point


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _as_pixel_xy(
    xy: np.ndarray,
    *,
    pixel_size: float,
    input_units: Literal["pixel", "physical"] = "pixel",
) -> np.ndarray:
    """Return xy coordinates in pixel units."""
    points = np.asarray(xy, dtype=float)
    if input_units == "pixel":
        return points
    if input_units == "physical":
        return points / _validate_positive(pixel_size, "pixel_size")
    raise ValueError("input_units must be 'pixel' or 'physical'")


def sample_height_at_xy(
    heightmap: np.ndarray,
    xy: np.ndarray,
    *,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Sample a height map at ``(x, y)`` coordinates.

    ``heightmap`` is indexed as ``heightmap[row=y, column=x]`` while ``xy`` is
    passed in geometric ``(x, y)`` order.
    """
    h = np.asarray(heightmap, dtype=float)
    if h.ndim != 2:
        raise ValueError("heightmap must be a 2-D array")
    points = _validate_xy_array(xy)
    coords_yx = np.vstack([points[:, 1], points[:, 0]])
    return np.asarray(map_coordinates(h, coords_yx, order=order, mode=mode), dtype=float)


def cell_centers_xy_pixels(result) -> np.ndarray:
    """
    Return deprojected cell centers in ``(x, y)`` pixel coordinates.

    Result cell centers are stored in physical units. This helper divides their
    ``x/y`` components by ``result.pixel_size``.
    """
    pixel_size = _validate_positive(result.pixel_size, "result.pixel_size")
    centers = np.asarray([cell.center[:2] for cell in result.epicells], dtype=float)
    return centers.reshape(-1, 2) / pixel_size
