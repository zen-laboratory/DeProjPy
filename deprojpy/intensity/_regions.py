"""Prepare fitted cell prisms and their coordinate metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..geometry import fit_plane
from ._models import CellRegion, PreparedRegions, StackCalibration
from ._quadrature import build_lateral_quadrature, signed_polygon_area

if TYPE_CHECKING:
    from ..models import DeprojResult


_UNIT_TO_MICROMETRES = {
    "um": 1.0,
    "µm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "mm": 1000.0,
}


def _unit_conversion_factor(source_unit: str, target_unit: str) -> float:
    source = source_unit.lower()
    target = target_unit.lower()
    if source == target:
        return 1.0
    if source not in _UNIT_TO_MICROMETRES or target not in _UNIT_TO_MICROMETRES:
        raise ValueError(f"cannot convert length units {source_unit!r} to {target_unit!r}")
    return _UNIT_TO_MICROMETRES[source] / _UNIT_TO_MICROMETRES[target]


def _resolve_calibration(
    result: "DeprojResult",
    calibration: StackCalibration | None,
    spacing_zyx: tuple[float, float, float] | np.ndarray | None,
    origin_zyx: tuple[float, float, float] | np.ndarray | None,
) -> StackCalibration:
    """Resolve defaults while rejecting ignored/conflicting calibration input."""
    # A complete calibration owns both spacing and origin; accepting convenience
    # overrides alongside it would make one caller input silently ineffective.
    if calibration is not None:
        if spacing_zyx is not None or origin_zyx is not None:
            raise ValueError(
                "calibration cannot be combined with spacing_zyx or origin_zyx"
            )
        return calibration

    # Unmodified acquisition stacks share the XY/Z calibration used to build
    # the result. Explicit values describe a cropped or resampled stack instead.
    if spacing_zyx is None:
        resolved_spacing = (
            float(result.voxel_depth),
            float(result.pixel_size),
            float(result.pixel_size),
        )
    else:
        spacing_array = np.asarray(spacing_zyx, dtype=float)
        if spacing_array.shape != (3,):
            raise ValueError("spacing_zyx must have exactly three values (dz, dy, dx)")
        resolved_spacing = (
            float(spacing_array[0]),
            float(spacing_array[1]),
            float(spacing_array[2]),
        )

    if origin_zyx is None:
        resolved_origin = (0.0, 0.0, 0.0)
    else:
        origin_array = np.asarray(origin_zyx, dtype=float)
        if origin_array.shape != (3,):
            raise ValueError("origin_zyx must have exactly three values (z0, y0, x0)")
        resolved_origin = (
            float(origin_array[0]),
            float(origin_array[1]),
            float(origin_array[2]),
        )
    return StackCalibration(
        spacing_zyx=resolved_spacing,
        origin_zyx=resolved_origin,
    )


def _resolve_z_mapping(
    result: "DeprojResult",
    z_reflected: bool | None,
    z_reflection_offset: float | None,
) -> tuple[bool, float]:
    """Resolve and validate the saved geometry-to-original-Z transform."""
    # Explicit values are required only for legacy results. Never infer the
    # missing reflection translation from cell geometry or the stack depth.
    reflected_value = result.z_reflected if z_reflected is None else z_reflected
    offset_value = (
        result.z_reflection_offset
        if z_reflection_offset is None
        else z_reflection_offset
    )

    if reflected_value is None:
        raise ValueError(
            "result lacks original-stack Z metadata; pass z_reflected explicitly"
        )
    if not isinstance(reflected_value, (bool, np.bool_)):
        raise TypeError("z_reflected must be a boolean")

    # A reflected frame needs its translation as well as its sign: geometric
    # Z alone cannot recover an original acquisition-stack index.
    if bool(reflected_value):
        if offset_value is None or not np.isfinite(offset_value):
            raise ValueError(
                "a reflected result requires a finite z_reflection_offset"
            )
        return True, float(offset_value)

    if offset_value is not None and not np.isfinite(offset_value):
        raise ValueError("z_reflection_offset must be finite when supplied")
    return False, 0.0 if offset_value is None else float(offset_value)


def _invalid_region(
    cell_id: int,
    source_label: int | None,
    start_depth: float,
    end_depth: float,
    reason: str,
) -> CellRegion:
    """Retain a failed cell with explicit empty geometry and diagnostics."""
    return CellRegion(
        cell_id=cell_id,
        source_label=source_label,
        plane_origin_xyz=np.zeros(3, dtype=float),
        plane_basis_xyz=np.eye(3, dtype=float)[:, :2],
        polygon_uv=np.empty((0, 2), dtype=float),
        normal_xyz=np.asarray([0.0, 0.0, 1.0]),
        start_depth=start_depth,
        end_depth=end_depth,
        footprint_area=0.0,
        lateral_points_uv=np.empty((0, 2), dtype=float),
        area_weights=np.empty(0, dtype=float),
        status="invalid_geometry",
        reason=reason,
    )


def prepare_regions(
    result: "DeprojResult",
    depth_range: tuple[float, float],
    *,
    depth_units: str = "result",
    tissue_side: str,
    sampling_spacing: float | None = None,
    calibration: StackCalibration | None = None,
    spacing_zyx: tuple[float, float, float] | np.ndarray | None = None,
    origin_zyx: tuple[float, float, float] | np.ndarray | None = None,
    z_reflected: bool | None = None,
    z_reflection_offset: float | None = None,
) -> PreparedRegions:
    """Fit and extrude one planar prism per DeProjPy cell.

    Parameters
    ----------
    result
        Deprojection result whose cell boundaries use physical geometric XYZ.
    depth_range
        Finite ``(start, end)`` with ``start < end``. Positive depth points
        toward ``tissue_side``; negative depth points to the opposite side.
    depth_units
        ``"result"`` (default), ``"slices"``, or a supported physical unit
        (``"um"``, ``"µm"``, ``"mm"`` and documented aliases). One slice is
        one calibrated axial-spacing equivalent along the physical normal.
    tissue_side
        Required original-stack direction: ``"increasing_z"`` or
        ``"decreasing_z"``.
    sampling_spacing
        Maximum lateral subtriangle edge and maximum depth-bin width in result
        length units. Defaults to ``result.pixel_size``.
    calibration
        Complete stack calibration. It cannot be combined with the convenience
        ``spacing_zyx`` or ``origin_zyx`` arguments.
    spacing_zyx, origin_zyx
        Convenience calibration values in result length units, both in
        original-stack physical ZYX order.
    z_reflected, z_reflection_offset
        Explicit mapping override for legacy results. Missing legacy metadata
        is rejected rather than inferred from a cell or image depth.

    Returns
    -------
    PreparedRegions
        Reusable compact regions with cached lateral quadrature.

    Notes
    -----
    The plane origin is the boundary mean used by :func:`fit_plane`, not a
    surface centroid. The fitted normal is reoriented only after translating
    its Z direction into the original acquisition frame.
    """
    # Validate the user-facing depth/calibration contract before touching any
    # cells so configuration errors cannot be mistaken for cell failures.
    start_input, end_input = (float(value) for value in depth_range)
    if not np.all(np.isfinite([start_input, end_input])) or start_input >= end_input:
        raise ValueError("depth_range must contain finite start < end")
    if tissue_side not in {"increasing_z", "decreasing_z"}:
        raise ValueError("tissue_side must be 'increasing_z' or 'decreasing_z'")

    resolved_calibration = _resolve_calibration(
        result,
        calibration,
        spacing_zyx,
        origin_zyx,
    )
    reflected, reflection_offset = _resolve_z_mapping(
        result,
        z_reflected,
        z_reflection_offset,
    )

    # Convert once into the result's physical length unit. Slice values are
    # axial-spacing equivalents along the normal, not imaging-Z displacements.
    if depth_units == "result":
        depth_factor = 1.0
    elif depth_units == "slices":
        depth_factor = resolved_calibration.spacing_zyx[0]
    else:
        depth_factor = _unit_conversion_factor(depth_units, result.units)
    start_depth = start_input * depth_factor
    end_depth = end_input * depth_factor

    resolved_sampling_spacing = float(
        result.pixel_size if sampling_spacing is None else sampling_spacing
    )
    if not np.isfinite(resolved_sampling_spacing) or resolved_sampling_spacing <= 0:
        raise ValueError("sampling_spacing must be positive and finite")

    desired_original_z_sign = 1.0 if tissue_side == "increasing_z" else -1.0
    cells: list[CellRegion] = []
    for cell in result.epicells:
        cell_id = int(cell.id)
        try:
            # Fit in physical geometric XYZ. The boundary mean must remain the
            # plane origin because fit_plane centers on that exact point.
            boundary_xyz = np.asarray(cell.boundary, dtype=float)
            if boundary_xyz.ndim != 2 or boundary_xyz.shape[1] != 3:
                raise ValueError("boundary must have shape (n, 3)")
            if len(boundary_xyz) < 3 or not np.all(np.isfinite(boundary_xyz)):
                raise ValueError("boundary must contain at least three finite XYZ vertices")

            plane_origin_xyz = np.mean(boundary_xyz, axis=0)
            plane_rotation, _ = fit_plane(boundary_xyz)
            plane_basis_xyz = plane_rotation[:, :2]
            normal_xyz = plane_rotation[:, 2].copy()

            # SVD chooses an arbitrary normal sign. Orient it only after mapping
            # its Z component back into the original acquisition frame.
            original_z_component = -normal_xyz[2] if reflected else normal_xyz[2]
            if abs(original_z_component) < 1e-10:
                raise ValueError(
                    "plane normal has numerically ambiguous original-stack Z direction"
                )
            if np.sign(original_z_component) != desired_original_z_sign:
                normal_xyz *= -1.0

            # Orthogonal projection preserves the ordered, possibly nonconvex
            # outline. Cleanup/triangulation occurs in this fitted plane.
            projected_polygon_uv = (
                boundary_xyz - plane_origin_xyz
            ) @ plane_basis_xyz
            lateral_points_uv, area_weights, cleaned_polygon_uv = (
                build_lateral_quadrature(
                    projected_polygon_uv,
                    resolved_sampling_spacing,
                )
            )
            footprint_area = abs(signed_polygon_area(cleaned_polygon_uv))

            # Cache channel-independent lateral samples and exact area weights;
            # measurement reuses them without retriangulating each channel.
            cells.append(
                CellRegion(
                    cell_id=cell_id,
                    source_label=cell.source_label,
                    plane_origin_xyz=plane_origin_xyz,
                    plane_basis_xyz=plane_basis_xyz,
                    polygon_uv=cleaned_polygon_uv,
                    normal_xyz=normal_xyz,
                    start_depth=start_depth,
                    end_depth=end_depth,
                    footprint_area=footprint_area,
                    lateral_points_uv=lateral_points_uv,
                    area_weights=area_weights,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            # Geometry failure is cell-local. Retaining an explicit failed
            # region preserves stable identifiers and one output row per cell.
            cells.append(
                _invalid_region(
                    cell_id=cell_id,
                    source_label=cell.source_label,
                    start_depth=start_depth,
                    end_depth=end_depth,
                    reason=str(error),
                )
            )

    return PreparedRegions(
        cells=cells,
        units=result.units,
        depth_range=(start_input, end_input),
        depth_units=depth_units,
        converted_depth_range=(start_depth, end_depth),
        sampling_spacing=resolved_sampling_spacing,
        tissue_side=tissue_side,
        calibration=resolved_calibration,
        z_reflected=reflected,
        z_reflection_offset=reflection_offset,
    )
