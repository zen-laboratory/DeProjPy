"""Conversions between deprojection geometry and acquisition-stack indices."""

from __future__ import annotations

import numpy as np

from ._models import PreparedRegions


def geometric_xyz_to_stack_index_zyx(
    geometric_xyz: np.ndarray,
    regions: PreparedRegions,
) -> np.ndarray:
    """Convert geometric XYZ coordinates to floating stack-index ZYX.

    Parameters
    ----------
    geometric_xyz
        Array of shape ``(n, 3)`` in DeProjPy geometric ``(x, y, z)`` order
        and the result's physical length unit.
    regions
        Prepared regions containing acquisition calibration and the saved Z
        reflection transform.

    Returns
    -------
    numpy.ndarray
        Newly allocated ``(n, 3)`` floating coordinates in array ``(z, y, x)``
        index order.

    Notes
    -----
    When the heightmap was inverted, geometric Z satisfies
    ``original_z = reflection_offset - geometric_z``. XY is never reflected.
    """
    points_xyz = np.asarray(geometric_xyz, dtype=float)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("geometric_xyz must have shape (n, 3)")

    # Reorder geometric XYZ into array ZYX before applying the acquisition
    # frame transform. Only Z can be reflected by heightmap preparation.
    original_physical_zyx = points_xyz[:, [2, 1, 0]].copy()
    if regions.z_reflected:
        original_physical_zyx[:, 0] = regions.z_reflection_offset - points_xyz[:, 2]

    # Subtract the physical crop origin, then divide by calibrated spacing to
    # obtain floating indices suitable for trilinear interpolation.
    stack_origin_zyx = np.asarray(regions.calibration.origin_zyx, dtype=float)
    stack_spacing_zyx = np.asarray(regions.calibration.spacing_zyx, dtype=float)
    return (original_physical_zyx - stack_origin_zyx) / stack_spacing_zyx
