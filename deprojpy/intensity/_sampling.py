"""Bounded sample generation and geometric stack-coverage classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

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
    """At most one configured budget of samples plus explicit completions."""

    geometric_xyz: np.ndarray
    volume_weights: np.ndarray
    region_indices: np.ndarray
    completed_region_indices: tuple[int, ...]


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


def iter_sample_blocks(
    regions: PreparedRegions,
    plans: list[RegionSamplingPlan],
    max_samples: int,
) -> Iterator[SampleBlock]:
    """Yield bounded coordinate/weight blocks, combining cells when practical.

    Flat Cartesian-product indices are expanded only for the slice that fits in
    the current block. Thus one large cell cannot allocate coordinates beyond
    ``max_samples``, while consecutive small cells share interpolation calls.
    """
    if not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    # Parts let the tail of one small cell share a block with the next cell.
    # Their combined length, and every derived coordinate array, stays bounded.
    xyz_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    region_parts: list[np.ndarray] = []
    completed_regions: list[int] = []
    block_size = 0

    for plan in plans:
        region = regions.cells[plan.region_index]
        depth_count = len(plan.depth_points)
        cursor = 0

        while cursor < plan.sample_count:
            capacity = max_samples - block_size
            take = min(capacity, plan.sample_count - cursor)
            # Decode only this block's slice of the lateral × depth Cartesian
            # product. No full-cell meshgrid or coordinate array is constructed.
            flat_indices = np.arange(cursor, cursor + take, dtype=np.int64)
            lateral_indices = flat_indices // depth_count
            depth_indices = flat_indices % depth_count

            # Map plane UV plus signed normal depth into physical geometric XYZ.
            footprint_xyz = (
                region.plane_origin_xyz
                + plan.lateral_points_uv[lateral_indices] @ region.plane_basis_xyz.T
            )
            geometric_xyz = (
                footprint_xyz
                + plan.depth_points[depth_indices, None] * region.normal_xyz
            )
            volume_weights = (
                plan.area_weights[lateral_indices]
                * plan.depth_widths[depth_indices]
            )

            xyz_parts.append(geometric_xyz)
            weight_parts.append(volume_weights)
            region_parts.append(
                np.full(take, plan.region_index, dtype=np.int64)
            )
            block_size += take
            cursor += take
            if cursor == plan.sample_count:
                # Completion means every coordinate for this region has been
                # generated. Measurement finalizes it only after all channels
                # have consumed the enclosing block.
                completed_regions.append(plan.region_index)

            # Yield exactly at the budget so a single large cell is naturally
            # split and several small cells are naturally combined.
            if block_size == max_samples:
                yield SampleBlock(
                    geometric_xyz=np.concatenate(xyz_parts, axis=0),
                    volume_weights=np.concatenate(weight_parts),
                    region_indices=np.concatenate(region_parts),
                    completed_region_indices=tuple(completed_regions),
                )
                xyz_parts.clear()
                weight_parts.clear()
                region_parts.clear()
                completed_regions.clear()
                block_size = 0

    if block_size:
        yield SampleBlock(
            geometric_xyz=np.concatenate(xyz_parts, axis=0),
            volume_weights=np.concatenate(weight_parts),
            region_indices=np.concatenate(region_parts),
            completed_region_indices=tuple(completed_regions),
        )
