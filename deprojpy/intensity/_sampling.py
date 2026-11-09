"""Bounded sample generation and geometric stack-coverage classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numba import njit

from ._coordinates import geometric_xyz_to_stack_index_zyx
from ._models import CellRegion, PreparedRegions
from ._quadrature import depth_midpoint_quadrature


@dataclass(frozen=True)
class RegionSamplingPlan:
    """Channel-independent quadrature selected for one valid region."""

    region_index: int
    lateral_points_uv: np.ndarray
    area_weights: np.ndarray
    depth_points: np.ndarray
    depth_widths: np.ndarray

    @property
    def sample_count(self) -> int:
        """Return the Cartesian-product sample count without materializing it."""
        return len(self.lateral_points_uv) * len(self.depth_points)


@dataclass(frozen=True)
class SampleBlock:
    """Supported stack coordinates, contiguous region spans, and completions."""

    stack_index_zyx: np.ndarray
    volume_weights: np.ndarray
    region_spans: tuple["RegionSpan", ...]
    completed_region_indices: tuple[int, ...]


@dataclass(frozen=True)
class RegionSpan:
    """One region's contiguous supported samples within a block."""

    region_index: int
    start: int
    stop: int
    sampled_volume: float


@dataclass(frozen=True)
class RegionCoverage:
    """Geometry-only relationship between a requested prism and an image box."""

    clipped: bool
    definitely_disjoint: bool


def classify_region_coverage(
    region: CellRegion,
    regions: PreparedRegions,
    image_shape_zyx: tuple[int, int, int],
) -> RegionCoverage:
    """Classify clipping and provable disjointness from full prism vertices.

    Coordinate extrema of a linearly extruded polygon occur at its two sets of
    face vertices. A separating image axis therefore proves disjointness. Box
    overlap on every axis does not prove intersection for a tilted prism, so
    that harder case remains a potential intersection for quadrature to assess.
    """
    footprint_xyz = (
        region.plane_origin_xyz
        + region.polygon_uv @ region.plane_basis_xyz.T
    )
    prism_vertices_xyz = np.vstack(
        [
            footprint_xyz + depth * region.normal_xyz
            for depth in (region.start_depth, region.end_depth)
        ]
    )
    prism_vertices_zyx = geometric_xyz_to_stack_index_zyx(
        prism_vertices_xyz,
        regions,
    )

    lower = np.min(prism_vertices_zyx, axis=0)
    upper = np.max(prism_vertices_zyx, axis=0)
    supported_upper = np.asarray(image_shape_zyx, dtype=float) - 1.0
    clipped = bool(np.any(lower < 0.0) or np.any(upper > supported_upper))
    definitely_disjoint = bool(np.any(upper < 0.0) or np.any(lower > supported_upper))
    return RegionCoverage(clipped=clipped, definitely_disjoint=definitely_disjoint)


def build_sampling_plans(
    regions: PreparedRegions,
    coverages: list[RegionCoverage | None],
) -> list[RegionSamplingPlan]:
    """Reuse base-spacing quadrature for every potentially measurable region.

    Clipping changes only which sampling points are accepted later. It does not
    trigger hidden oversampling: clipped and unclipped cells use the same cached
    lateral rule and requested normal-depth spacing. Users can explicitly lower
    ``sampling_spacing`` when they need a convergence check.
    """

    plans: list[RegionSamplingPlan] = []
    for region_index, region in enumerate(regions.cells):
        coverage = coverages[region_index]
        if region.status != "ok" or coverage is None or coverage.definitely_disjoint:
            continue

        depth_points, depth_widths = depth_midpoint_quadrature(
            region.start_depth,
            region.end_depth,
            regions.sampling_spacing,
        )

        plans.append(
            RegionSamplingPlan(
                region_index=region_index,
                lateral_points_uv=region.lateral_points_uv,
                area_weights=region.area_weights,
                depth_points=depth_points,
                depth_widths=depth_widths,
            )
        )
    return plans


@njit(cache=True)
def _append_supported_samples(
    output_stack_index_zyx: np.ndarray,
    output_volume_weights: np.ndarray,
    output_start: int,
    cursor: int,
    take: int,
    lateral_points_uv: np.ndarray,
    area_weights: np.ndarray,
    depth_points: np.ndarray,
    depth_widths: np.ndarray,
    plane_origin_xyz: np.ndarray,
    plane_basis_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    stack_spacing_zyx: np.ndarray,
    stack_origin_zyx: np.ndarray,
    z_reflected: bool,
    z_reflection_offset: float,
    supported_upper_zyx: np.ndarray,
) -> tuple[int, float]:
    """Generate one plan slice directly in stack ZYX and compact in bounds."""
    depth_count = len(depth_points)
    written = 0
    sampled_volume = 0.0

    for flat_index in range(cursor, cursor + take):
        lateral_index = flat_index // depth_count
        depth_index = flat_index % depth_count
        u = lateral_points_uv[lateral_index, 0]
        v = lateral_points_uv[lateral_index, 1]
        depth = depth_points[depth_index]

        # Preserve the existing plane-UV -> geometric-XYZ arithmetic order,
        # then reorder into acquisition ZYX before applying crop calibration.
        geometric_x = (
            plane_origin_xyz[0]
            + u * plane_basis_xyz[0, 0]
            + v * plane_basis_xyz[0, 1]
            + depth * normal_xyz[0]
        )
        geometric_y = (
            plane_origin_xyz[1]
            + u * plane_basis_xyz[1, 0]
            + v * plane_basis_xyz[1, 1]
            + depth * normal_xyz[1]
        )
        geometric_z = (
            plane_origin_xyz[2]
            + u * plane_basis_xyz[2, 0]
            + v * plane_basis_xyz[2, 1]
            + depth * normal_xyz[2]
        )
        original_z = (
            z_reflection_offset - geometric_z if z_reflected else geometric_z
        )
        stack_z = (original_z - stack_origin_zyx[0]) / stack_spacing_zyx[0]
        stack_y = (geometric_y - stack_origin_zyx[1]) / stack_spacing_zyx[1]
        stack_x = (geometric_x - stack_origin_zyx[2]) / stack_spacing_zyx[2]

        if (
            stack_z < 0.0
            or stack_z > supported_upper_zyx[0]
            or stack_y < 0.0
            or stack_y > supported_upper_zyx[1]
            or stack_x < 0.0
            or stack_x > supported_upper_zyx[2]
        ):
            continue

        output_index = output_start + written
        output_stack_index_zyx[output_index, 0] = stack_z
        output_stack_index_zyx[output_index, 1] = stack_y
        output_stack_index_zyx[output_index, 2] = stack_x
        volume_weight = area_weights[lateral_index] * depth_widths[depth_index]
        output_volume_weights[output_index] = volume_weight
        sampled_volume += volume_weight
        written += 1

    return written, sampled_volume


def iter_sample_blocks(
    regions: PreparedRegions,
    plans: list[RegionSamplingPlan],
    max_samples: int,
    image_shape_zyx: tuple[int, int, int],
) -> Iterator[SampleBlock]:
    """Yield bounded supported-coordinate blocks with contiguous region spans.

    The configured budget counts requested quadrature samples, including those
    later rejected as out of bounds. Stack coordinates and weights are written
    only for supported samples, so no geometric-XYZ or per-sample region-ID
    scratch arrays are needed. Consecutive regions remain contiguous after this
    stable compaction and can share one interpolation call.
    """
    if not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    stack_spacing_zyx = np.asarray(regions.calibration.spacing_zyx, dtype=float)
    stack_origin_zyx = np.asarray(regions.calibration.origin_zyx, dtype=float)
    supported_upper_zyx = np.asarray(image_shape_zyx, dtype=float) - 1.0

    # Each yielded block owns its buffers. Their allocated capacity follows the
    # requested-sample budget even when clipping leaves only a short valid view.
    stack_index_zyx = np.empty((max_samples, 3), dtype=float)
    volume_weights = np.empty(max_samples, dtype=float)
    region_spans: list[RegionSpan] = []
    completed_regions: list[int] = []
    requested_count = 0
    supported_count = 0

    for plan in plans:
        region = regions.cells[plan.region_index]
        cursor = 0

        while cursor < plan.sample_count:
            capacity = max_samples - requested_count
            take = min(capacity, plan.sample_count - cursor)
            written, span_volume = _append_supported_samples(
                stack_index_zyx,
                volume_weights,
                supported_count,
                cursor,
                take,
                plan.lateral_points_uv,
                plan.area_weights,
                plan.depth_points,
                plan.depth_widths,
                region.plane_origin_xyz,
                region.plane_basis_xyz,
                region.normal_xyz,
                stack_spacing_zyx,
                stack_origin_zyx,
                regions.z_reflected,
                regions.z_reflection_offset,
                supported_upper_zyx,
            )
            if written:
                region_spans.append(
                    RegionSpan(
                        region_index=plan.region_index,
                        start=supported_count,
                        stop=supported_count + written,
                        sampled_volume=span_volume,
                    )
                )
                supported_count += written

            requested_count += take
            cursor += take
            if cursor == plan.sample_count:
                # Completion means every coordinate for this region has been
                # generated. Measurement finalizes it only after all channels
                # have consumed the enclosing block.
                completed_regions.append(plan.region_index)

            # Yield exactly at the budget so a single large cell is naturally
            # split and several small cells are naturally combined.
            if requested_count == max_samples:
                yield SampleBlock(
                    stack_index_zyx=stack_index_zyx[:supported_count],
                    volume_weights=volume_weights[:supported_count],
                    region_spans=tuple(region_spans),
                    completed_region_indices=tuple(completed_regions),
                )
                stack_index_zyx = np.empty((max_samples, 3), dtype=float)
                volume_weights = np.empty(max_samples, dtype=float)
                region_spans.clear()
                completed_regions.clear()
                requested_count = 0
                supported_count = 0

    if requested_count:
        yield SampleBlock(
            stack_index_zyx=stack_index_zyx[:supported_count],
            volume_weights=volume_weights[:supported_count],
            region_spans=tuple(region_spans),
            completed_region_indices=tuple(completed_regions),
        )
