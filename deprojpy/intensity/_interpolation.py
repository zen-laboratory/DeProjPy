"""Specialized and reference trilinear interpolation for supported samples."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage


def can_use_specialized_trilinear(image: np.ndarray) -> bool:
    """Return whether Numba safely supports this 3-D image representation."""
    dtype = image.dtype
    supported_dtype = dtype.kind in {"b", "i", "u"} or (
        dtype.kind == "f" and dtype.itemsize in {4, 8}
    )
    # Singleton axes need fewer than eight neighbors. They are uncommon for
    # tissue stacks, so the exact SciPy path remains simpler than a slow branch
    # inside every sample of the representative 3-D kernel.
    return bool(
        supported_dtype
        and dtype.isnative
        and image.ndim == 3
        and all(size >= 2 for size in image.shape)
    )


@njit(inline="always")
def _weighted_neighbors(
    image: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    fz: float,
    fy: float,
    fx: float,
) -> float:
    """Combine eight neighbors in float64 without unsigned differences."""
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    wz0, wz1 = 1.0 - fz, fz
    wy0, wy1 = 1.0 - fy, fy
    wx0, wx1 = 1.0 - fx, fx

    # SciPy evaluates all eight terms, including zero-weight nonfinite values.
    # Keeping those multiplications preserves its NaN/±infinity propagation.
    value = 0.0
    coefficient = wy0 * wx0 * wz0
    value += coefficient * float(image[z0, y0, x0])
    coefficient = wy0 * wx1 * wz0
    value += coefficient * float(image[z0, y0, x1])
    coefficient = wy1 * wx0 * wz0
    value += coefficient * float(image[z0, y1, x0])
    coefficient = wy1 * wx1 * wz0
    value += coefficient * float(image[z0, y1, x1])
    coefficient = wy0 * wx0 * wz1
    value += coefficient * float(image[z1, y0, x0])
    coefficient = wy0 * wx1 * wz1
    value += coefficient * float(image[z1, y0, x1])
    coefficient = wy1 * wx0 * wz1
    value += coefficient * float(image[z1, y1, x0])
    coefficient = wy1 * wx1 * wz1
    value += coefficient * float(image[z1, y1, x1])
    return value


@njit(cache=True)
def _trilinear_sample(image: np.ndarray, coordinates_zyx: np.ndarray) -> np.ndarray:
    """Sample supported coordinates directly, one serial output block at a time."""
    output = np.empty(len(coordinates_zyx), dtype=np.float64)
    upper_z = image.shape[0] - 1
    upper_y = image.shape[1] - 1
    upper_x = image.shape[2] - 1

    for index in range(len(coordinates_zyx)):
        z, y, x = coordinates_zyx[index]
        # At the last valid index SciPy uses the preceding interval with a
        # fraction of one, so its zero-weight lower neighbor still participates.
        if z == upper_z:
            z0, fz = upper_z - 1, 1.0
        else:
            z0 = int(np.floor(z))
            fz = z - z0
        if y == upper_y:
            y0, fy = upper_y - 1, 1.0
        else:
            y0 = int(np.floor(y))
            fy = y - y0
        if x == upper_x:
            x0, fx = upper_x - 1, 1.0
        else:
            x0 = int(np.floor(x))
            fx = x - x0
        output[index] = _weighted_neighbors(image, z0, y0, x0, fz, fy, fx)

    return output


@njit(cache=True)
def _prepare_trilinear_coordinates(
    coordinates_zyx: np.ndarray,
    image_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Cache lower indices and fractions for reuse by aligned channels."""
    lower_indices = np.empty((len(coordinates_zyx), 3), dtype=np.int64)
    fractions = np.empty((len(coordinates_zyx), 3), dtype=np.float64)

    for sample_index in range(len(coordinates_zyx)):
        for axis in range(3):
            coordinate = coordinates_zyx[sample_index, axis]
            upper = image_shape_zyx[axis] - 1
            if coordinate == upper:
                lower_indices[sample_index, axis] = upper - 1
                fractions[sample_index, axis] = 1.0
            else:
                lower = int(np.floor(coordinate))
                lower_indices[sample_index, axis] = lower
                fractions[sample_index, axis] = coordinate - lower

    return lower_indices, fractions


@njit(cache=True)
def _trilinear_sample_prepared(
    image: np.ndarray,
    lower_indices: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """Sample one aligned channel using bounded per-block coordinate scratch."""
    output = np.empty(len(lower_indices), dtype=np.float64)
    for index in range(len(lower_indices)):
        output[index] = _weighted_neighbors(
            image,
            lower_indices[index, 0],
            lower_indices[index, 1],
            lower_indices[index, 2],
            fractions[index, 0],
            fractions[index, 1],
            fractions[index, 2],
        )
    return output


def prepare_trilinear_coordinates(
    coordinates_zyx: np.ndarray,
    image_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare bounded coordinate scratch for two or more aligned channels."""
    return _prepare_trilinear_coordinates(coordinates_zyx, image_shape_zyx)


def interpolate_values(
    image: np.ndarray,
    coordinates_zyx: np.ndarray,
    *,
    prepared: tuple[np.ndarray, np.ndarray] | None = None,
    use_specialized: bool = True,
) -> np.ndarray:
    """Interpolate with the specialized kernel or the exact SciPy reference."""
    if use_specialized and can_use_specialized_trilinear(image):
        if prepared is None:
            return _trilinear_sample(image, coordinates_zyx)
        lower_indices, fractions = prepared
        return _trilinear_sample_prepared(image, lower_indices, fractions)

    return ndimage.map_coordinates(
        image,
        coordinates_zyx.T,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
        output=float,
    )
