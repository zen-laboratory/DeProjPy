from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import inpaint


def _nan_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(image)
    values = ndi.gaussian_filter(np.where(valid, image, 0.0), sigma, mode="nearest")
    weights = ndi.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    return np.divide(values, weights, out=np.full_like(values, np.nan), where=weights > 1e-12)


def prepare_heightmap(
    heightmap: np.ndarray,
    voxel_depth: float = 1.0,
    smooth_scale: float = -1.0,
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
) -> np.ndarray:
    """Prepare a height map and return Z in physical units.

    Zero regions are biharmonically interpolated when requested. Remaining
    zeros become NaN when pruning is enabled. Inversion happens before unit
    scaling, as in the MATLAB workflow.
    """
    h = np.asarray(heightmap, dtype=float).copy()
    if h.ndim != 2:
        raise ValueError(f"heightmap must be a 2-D array; received shape {h.shape}")
    if h.size == 0:
        raise ValueError("heightmap must not be empty")
    if not np.isfinite(voxel_depth) or voxel_depth <= 0:
        raise ValueError("voxel_depth must be a positive finite number")
    if not np.any(np.isfinite(h)):
        raise ValueError("heightmap must contain at least one finite value")
    zero_mask = h == 0
    if inpaint_zeros and zero_mask.any():
        if zero_mask.all():
            if prune_zeros:
                raise ValueError(
                    "heightmap contains only zeros, so no finite surface remains "
                    "after zero pruning"
                )
        else:
            h = inpaint.inpaint_biharmonic(h, zero_mask, channel_axis=None)
    if prune_zeros:
        h[h == 0] = np.nan
    if smooth_scale > 0:
        h = _nan_gaussian(h, smooth_scale)
    if invert_z:
        finite = h[np.isfinite(h)]
        if finite.size:
            h = finite.max() - h
    h *= float(voxel_depth)
    if not np.any(np.isfinite(h)):
        raise ValueError(
            "heightmap contains no finite values after preprocessing; check zero "
            "handling and the input image"
        )
    return h


def get_z(points_xy: np.ndarray, heightmap: np.ndarray, pixel_size: float) -> np.ndarray:
    """Sample nearest height-map pixels for physical ``(x, y)`` points."""
    points = np.asarray(points_xy, dtype=float)
    # Coordinates in this port are zero-based; np.rint is nearest-pixel sampling.
    pixels = np.rint(points / pixel_size).astype(int)
    x = np.clip(pixels[:, 0], 0, heightmap.shape[1] - 1)
    y = np.clip(pixels[:, 1], 0, heightmap.shape[0] - 1)
    return np.asarray(heightmap[y, x], dtype=float)


def surface_centroids(
    heightmap: np.ndarray,
    footprints_rc: list[np.ndarray],
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Return area-weighted surface centroids for pixel footprints.

    ``heightmap`` must already be prepared and expressed in physical Z units.
    Footprints contain integer ``(row, column)`` pixel indices. Derivatives are
    calculated once in physical XY units with NumPy's central differences in
    the interior and one-sided differences at image boundaries. A footprint is
    unavailable (an all-NaN triple) if any of its heights or derivative samples
    is non-finite; invalid samples are never silently omitted.
    """
    h = np.asarray(heightmap, dtype=float)
    if h.ndim != 2:
        raise ValueError(f"heightmap must be a 2-D array; received shape {h.shape}")
    if min(h.shape) < 2:
        raise ValueError("heightmap must have at least two rows and two columns")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be a positive finite number")

    hy, hx = np.gradient(h, pixel_size, pixel_size, edge_order=1)
    weights = np.sqrt(1.0 + hx * hx + hy * hy)
    result = np.full((len(footprints_rc), 3), np.nan, dtype=float)
    for index, footprint in enumerate(footprints_rc):
        rc = np.asarray(footprint, dtype=np.int64)
        if rc.ndim != 2 or rc.shape[1] != 2 or len(rc) == 0:
            continue
        rows, cols = rc[:, 0], rc[:, 1]
        if (
            np.any(rows < 0)
            or np.any(rows >= h.shape[0])
            or np.any(cols < 0)
            or np.any(cols >= h.shape[1])
        ):
            continue
        z = h[rows, cols]
        w = weights[rows, cols]
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)):
            continue
        total = np.sum(w)
        result[index] = (
            np.array(
                [np.sum(w * cols * pixel_size), np.sum(w * rows * pixel_size), np.sum(w * z)]
            )
            / total
        )
    return result


def compute_curvatures(
    heightmap: np.ndarray,
    object_scale: float | None = None,
    pixel_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, Gaussian, and principal curvatures of z=f(x,y)."""
    h = np.asarray(heightmap, dtype=float)
    if object_scale is not None and np.isfinite(object_scale) and object_scale > 0:
        h = _nan_gaussian(h, 3.0 * object_scale)
    # np.gradient returns derivatives in axis order: y, x.
    hy, hx = np.gradient(h, pixel_size, pixel_size)
    hxy, hxx = np.gradient(hx, pixel_size, pixel_size)
    hyy, _ = np.gradient(hy, pixel_size, pixel_size)
    norm = 1.0 + hx * hx + hy * hy
    gaussian = (hxx * hyy - hxy * hxy) / (norm * norm)
    mean = ((1.0 + hx * hx) * hyy + (1.0 + hy * hy) * hxx - 2.0 * hx * hy * hxy) / (2.0 * norm**1.5)
    discriminant = np.maximum(mean * mean - gaussian, 0.0)
    root = np.sqrt(discriminant)
    return mean, gaussian, mean + root, mean - root
