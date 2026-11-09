"""Bounded interpolation orchestration and selective statistical reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

from . import _statistics as statistics
from ._models import MeasurementResult, PreparedRegions
from ._sampling import (
    RegionCoverage,
    build_sampling_plans,
    classify_region_coverage,
    iter_sample_blocks,
)

DEFAULT_MEASUREMENTS = ("mean", "median", "std", "sampled_volume")
QUALITY_MEASUREMENTS = (
    "requested_volume",
    "sampled_volume",
    "finite_volume",
    "coverage_fraction",
)
ALL_MEASUREMENTS = DEFAULT_MEASUREMENTS + (
    "mode",
    "minimum",
    "maximum",
    "q25",
    "q75",
    "requested_volume",
    "coverage_fraction",
    "finite_volume",
    "integrated_intensity",
    "raw_sample_sum",
)


def _selected_measurements(measurements: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize public metric selection while preserving caller order."""
    if measurements == "all":
        return ALL_MEASUREMENTS
    if isinstance(measurements, str):
        selected = (measurements,)
    else:
        selected = tuple(measurements)
    unknown = sorted(set(selected) - set(ALL_MEASUREMENTS))
    if unknown:
        raise ValueError(
            f"unknown measurements {unknown}; valid names are "
            f"{list(ALL_MEASUREMENTS)} or 'all'"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("measurements must not contain duplicate names")
    return selected


def _validated_channels(
    stack: np.ndarray | Mapping[str, np.ndarray],
    channel: str | int,
) -> list[tuple[object, np.ndarray]]:
    """Validate explicit channels and require a shared aligned acquisition grid."""
    # A mapping is the only multi-channel form: axis guessing would make small
    # Z stacks indistinguishable from channel-first arrays.
    items = list(stack.items()) if isinstance(stack, Mapping) else [(channel, stack)]
    channels: list[tuple[object, np.ndarray]] = []
    expected_shape = None
    for channel_name, image_like in items:
        image = np.asarray(image_like)
        if image.ndim != 3 or image.size == 0:
            raise ValueError(
                f"channel {channel_name!r} must be a nonempty (z, y, x) array"
            )
        is_real_numeric = (
            np.issubdtype(image.dtype, np.number)
            and not np.issubdtype(image.dtype, np.complexfloating)
        ) or np.issubdtype(image.dtype, np.bool_)
        if not is_real_numeric:
            raise TypeError(f"channel {channel_name!r} must contain real numeric data")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError("all channels must share the same aligned (z, y, x) shape")
        channels.append((channel_name, image))
    return channels


def _initial_coverages(
    regions: PreparedRegions,
    image_shape_zyx: tuple[int, int, int],
) -> list[RegionCoverage | None]:
    """Classify valid cells geometrically before quadrature samples are examined."""
    return [
        classify_region_coverage(region, regions, image_shape_zyx)
        if region.status == "ok"
        else None
        for region in regions.cells
    ]


def _empty_statistics() -> dict[str, float]:
    """Initialize intensity outputs as unavailable, never as measured zeros."""
    return {
        name: np.nan
        for name in ALL_MEASUREMENTS
        if name not in QUALITY_MEASUREMENTS
    }


def _finalize_reducer(
    reducer: statistics.StreamingReducer,
    selected: tuple[str, ...],
    mode_bins: str | int | np.ndarray,
) -> tuple[dict[str, float], list[float] | None]:
    """Finalize only selected metrics, sharing buffered work where possible."""
    output = _empty_statistics()
    if reducer.finite_count == 0:
        return output, None

    # Streaming fields are already final scalars. Assign only requested names
    # so metric selection remains computationally and structurally meaningful.
    if "mean" in selected:
        output["mean"] = reducer.mean
    if "std" in selected:
        output["std"] = float(
            np.sqrt(reducer.weighted_m2 / reducer.finite_volume)
        )
    if "integrated_intensity" in selected:
        output["integrated_intensity"] = reducer.weighted_sum
    if "raw_sample_sum" in selected:
        output["raw_sample_sum"] = reducer.raw_sum
    if "minimum" in selected:
        output["minimum"] = reducer.minimum
    if "maximum" in selected:
        output["maximum"] = reducer.maximum

    requested_quantiles = {
        name: quantile
        for name, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))
        if name in selected
    }
    actual_mode_edges = None
    if requested_quantiles or "mode" in selected:
        # Quantiles and mode share one materialized per-cell buffer. A
        # streaming-only selection never enters this allocation path.
        values, weights = reducer.buffered_samples()
        if requested_quantiles:
            quantile_values = statistics.weighted_quantiles(
                values,
                weights,
                requested_quantiles.values(),
            )
            for name, quantile in requested_quantiles.items():
                output[name] = quantile_values[quantile]
        if "mode" in selected:
            mode, edges = statistics.weighted_histogram_mode(
                values,
                weights,
                mode_bins,
            )
            output["mode"] = mode
            actual_mode_edges = edges.tolist()
    return output, actual_mode_edges


def _status_for_result(
    region_status: str,
    region_reason: str,
    coverage: RegionCoverage | None,
    sampled_volume: float,
    nonfinite_omitted: bool,
) -> tuple[str, str]:
    """Keep geometry, sampling coverage, and missing intensities distinguishable."""
    if region_status != "ok":
        return region_status, region_reason
    if coverage is not None and coverage.definitely_disjoint:
        return "outside_stack", "prism is geometrically disjoint from interpolation support"
    if sampled_volume == 0.0:
        return (
            "unresolved_sampling_coverage",
            "potential prism/stack intersection contains no in-bounds quadrature point",
        )
    if nonfinite_omitted:
        return "nonfinite_omitted", "nonfinite interpolates were omitted"
    if coverage is not None and coverage.clipped:
        return "partial_coverage", "statistics use the in-bounds part of a clipped prism"
    return "ok", ""


def measure(
    stack: np.ndarray | Mapping[str, np.ndarray],
    regions: PreparedRegions,
    measurements: str | Sequence[str] = DEFAULT_MEASUREMENTS,
    *,
    channel: str | int = 0,
    max_batch_samples: int = 250_000,
    max_buffered_samples_per_cell: int = 5_000_000,
    mode_bins: str | int | Sequence[float] = "auto",
) -> MeasurementResult:
    """Measure fluorescence within prepared surface-normal cell prisms.

    Parameters
    ----------
    stack
        One nonempty ``(z, y, x)`` array, or a mapping of explicit channel IDs
        to aligned arrays with the same shape. Integer arrays are not converted
        wholesale; interpolation output alone is floating point.
    regions
        Reusable regions returned by :func:`prepare_regions`.
    measurements
        Metric name sequence, one name, or ``"all"``. The default is mean,
        median, population standard deviation, and sampled volume.
    channel
        Identifier used only for a single array input.
    max_batch_samples
        Hard upper bound for generated coordinates, index coordinates, and
        interpolation values in one block. Blocks may combine several cells.
    max_buffered_samples_per_cell
        Separate per-cell/channel bound for exact quantiles and histogram mode.
        Streaming-only selections do not allocate this buffer.
    mode_bins
        Histogram bins used only when ``mode`` is selected: ``"auto"``, a
        positive integer, or finite strictly increasing explicit edges.

    Returns
    -------
    MeasurementResult
        One row per input cell and channel. Quality volume columns are always
        present even when not selected as metrics.

    Notes
    -----
    Mode is the midpoint of the highest volume-density histogram bin; ties use
    the lower bin. Automatic edge selection is unweighted, while histogram
    accumulation is volume weighted. Explicit edges exclude out-of-range data.
    """
    # Resolve the metric contract before allocating reducers. This determines
    # whether sample values can be discarded immediately after each block.
    selected = _selected_measurements(measurements)
    if not isinstance(max_batch_samples, int) or max_batch_samples <= 0:
        raise ValueError("max_batch_samples must be a positive integer")
    if (
        not isinstance(max_buffered_samples_per_cell, int)
        or max_buffered_samples_per_cell <= 0
    ):
        raise ValueError("max_buffered_samples_per_cell must be a positive integer")

    # Mode validation is deliberately mode-specific; unrelated measurements do
    # not gain a general binning/statistics configuration surface.
    normalized_mode_bins = (
        statistics.validate_mode_bins(mode_bins)
        if "mode" in selected
        else "auto"
    )
    channels = _validated_channels(stack, channel)
    requirements = statistics.requirements_for(selected)
    cell_count = len(regions.cells)

    # Classify full prism geometry before sampling. This keeps ``clipped`` and
    # provable disjointness independent of whether quadrature hits the boundary.
    # An empty channel mapping has no rows but retains a stable table schema.
    image_shape_zyx = channels[0][1].shape if channels else (0, 0, 0)
    coverages = (
        _initial_coverages(regions, image_shape_zyx)
        if channels
        else [None] * cell_count
    )
    plans = build_sampling_plans(regions, coverages) if channels else []

    # Sampled volume depends only on geometry and is shared across channels.
    # Finite volume and statistical state are channel-specific because NaNs can
    # differ between otherwise aligned images.
    sampled_volume = np.zeros(cell_count, dtype=float)
    nonfinite_omitted = np.zeros((len(channels), cell_count), dtype=bool)
    # Reducers are created lazily for regions that contribute an in-bounds
    # sample. Finalized scalar results live in fixed output slots, while the
    # potentially large exact-statistic buffers remain only in active reducers.
    active_reducers: list[dict[int, statistics.StreamingReducer]] = [
        {} for _ in channels
    ]
    finalized_metrics = [
        [_empty_statistics() for _ in regions.cells]
        for _ in channels
    ]
    finalized_mode_edges: list[list[list[float] | None]] = [
        [None for _ in regions.cells]
        for _ in channels
    ]
    finite_volume = np.zeros((len(channels), cell_count), dtype=float)

    # Coordinates are already compacted in bounds and grouped into contiguous
    # region spans. Reuse each bounded coordinate block across aligned channels.
    for block in iter_sample_blocks(
        regions,
        plans,
        max_batch_samples,
        image_shape_zyx,
    ):
        for span in block.region_spans:
            sampled_volume[span.region_index] += span.sampled_volume

        if len(block.stack_index_zyx):
            # Interpolate one channel at a time so channel count does not
            # multiply coordinate/value scratch memory. The same span offsets
            # partition every channel's output without per-region mask scans.
            for channel_index, (_, image) in enumerate(channels):
                values = ndimage.map_coordinates(
                    image,
                    block.stack_index_zyx.T,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                    output=float,
                )
                # Nonfinite interpolates contribute to sampled volume but not
                # to statistics or their finite-volume denominator.
                for span in block.region_spans:
                    span_values = values[span.start : span.stop]
                    span_weights = block.volume_weights[span.start : span.stop]
                    finite = np.isfinite(span_values)
                    region_key = span.region_index
                    if not np.all(finite):
                        nonfinite_omitted[channel_index, region_key] = True
                    reducer = active_reducers[channel_index].get(region_key)
                    if reducer is None:
                        reducer = statistics.StreamingReducer(
                            requirements=requirements,
                            max_buffered_samples=max_buffered_samples_per_cell,
                        )
                        active_reducers[channel_index][region_key] = reducer
                    reducer.update(
                        span_values[finite],
                        span_weights[finite],
                    )

        # A completion marker is attached only to the block containing the
        # region's final coordinates. It must be handled even when this block is
        # entirely out of bounds, because earlier blocks may own active reducers.
        # Finalize after every channel has consumed the block, then release exact
        # quantile/mode buffers immediately.
        for region_index in block.completed_region_indices:
            for channel_index in range(len(channels)):
                reducer = active_reducers[channel_index].pop(region_index, None)
                if reducer is None:
                    continue
                metric_values, mode_edges = _finalize_reducer(
                    reducer,
                    selected,
                    normalized_mode_bins,
                )
                finalized_metrics[channel_index][region_index] = metric_values
                finalized_mode_edges[channel_index][region_index] = mode_edges
                finite_volume[channel_index, region_index] = reducer.finite_volume
                reducer.clear_buffered_samples()

    if any(active_reducers_for_channel for active_reducers_for_channel in active_reducers):
        raise RuntimeError("internal sampling error: unfinished region reducers remain")

    # Finalize every cell/channel explicitly so sparse IDs, invalid geometry,
    # and completely uncovered prisms remain represented in the table.
    rows = []
    for channel_index, (channel_name, _) in enumerate(channels):
        for region_index, region in enumerate(regions.cells):
            coverage = coverages[region_index]
            requested_volume = (
                region.footprint_area * (region.end_depth - region.start_depth)
                if region.status == "ok"
                else np.nan
            )
            coverage_fraction = (
                sampled_volume[region_index] / requested_volume
                if np.isfinite(requested_volume) and requested_volume > 0
                else np.nan
            )
            status, reason = _status_for_result(
                region.status,
                region.reason,
                coverage,
                sampled_volume[region_index],
                nonfinite_omitted[channel_index, region_index],
            )
            metric_values = finalized_metrics[channel_index][region_index]
            mode_edges = finalized_mode_edges[channel_index][region_index]

            row = {
                "id": region.cell_id,
                "source_label": region.source_label,
                "channel": channel_name,
                "clipped": bool(coverage.clipped) if coverage is not None else False,
                "status": status,
                "reason": reason,
                "requested_volume": requested_volume,
                "sampled_volume": sampled_volume[region_index],
                "finite_volume": finite_volume[channel_index, region_index],
                "coverage_fraction": coverage_fraction,
            }
            for name in selected:
                if name not in QUALITY_MEASUREMENTS:
                    row[name] = metric_values[name]
            if "mode" in selected:
                row["mode_bin_edges"] = mode_edges
            rows.append(row)

    metric_columns = [name for name in selected if name not in QUALITY_MEASUREMENTS]
    columns = [
        "id",
        "source_label",
        "channel",
        "clipped",
        "status",
        "reason",
        *QUALITY_MEASUREMENTS,
        *metric_columns,
    ]
    if "mode" in selected:
        columns.append("mode_bin_edges")

    # Provenance records numerical conventions, not just user inputs, so saved
    # results remain interpretable after implementation details evolve.
    mode_policy = (
        normalized_mode_bins.tolist()
        if isinstance(normalized_mode_bins, np.ndarray)
        else normalized_mode_bins
    )
    provenance = {
        "schema_version": "1.0",
        "units": regions.units,
        "depth_range": list(regions.depth_range),
        "depth_units": regions.depth_units,
        "converted_depth_range": list(regions.converted_depth_range),
        "sampling_spacing": regions.sampling_spacing,
        "tissue_side": regions.tissue_side,
        "spacing_zyx": list(regions.calibration.spacing_zyx),
        "origin_zyx": list(regions.calibration.origin_zyx),
        "z_reflected": regions.z_reflected,
        "z_reflection_offset": regions.z_reflection_offset,
        "measurements": list(selected),
        "nonfinite_policy": "omit; finite_volume is the contributing denominator",
        "interpolation_support": "closed index box [0, size - 1]",
        "lateral_quadrature": "longest-edge subdivision; triangle centroids",
        "boundary_quadrature": (
            "base sampling_spacing; include an element iff its sampling point "
            "is inside interpolation support"
        ),
        "weighted_quantile": "inverse weighted empirical CDF",
        "standard_deviation": "weighted population standard deviation",
        "mode": {
            "bins": mode_policy,
            "representative": "midpoint of highest volume-density bin; lower tie",
            "automatic_edge_selection": "numpy unweighted auto rule",
        }
        if "mode" in selected
        else None,
        "max_batch_samples": max_batch_samples,
        "max_buffered_samples_per_cell": max_buffered_samples_per_cell,
    }
    return MeasurementResult(
        table=pd.DataFrame(rows, columns=columns),
        provenance=provenance,
    )
