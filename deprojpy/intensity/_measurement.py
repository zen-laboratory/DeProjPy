"""Bounded interpolation orchestration and selective statistical reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from . import _interpolation as interpolation
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


def _normalize_channel_id(value, argument: str) -> str | int:
    """Return one supported channel identifier without bool/int ambiguity."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{argument} channel identifiers must not be booleans")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"{argument} channel identifiers must be strings or integers")


def _ordered_sequence(value) -> list | None:
    """Return a concrete non-string sequence, including one-dimensional arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim == 1 else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return None


def _validate_channel_images(
    items: Sequence[tuple[str | int, object]],
) -> list[tuple[str | int, np.ndarray]]:
    """Require real, nonempty ZYX channels on one aligned acquisition grid."""
    validated: list[tuple[str | int, np.ndarray]] = []
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
        validated.append((channel_name, image))
    return validated


def _array_channel_names(channel_names, channel_count: int) -> list[str | int]:
    """Resolve the full source-axis label sequence before channel selection."""
    if channel_names is None:
        return list(range(channel_count))
    names = _ordered_sequence(channel_names)
    if names is None:
        raise TypeError("channel_names for a CZYX array must be an ordered sequence")
    if len(names) != channel_count:
        raise ValueError(
            f"channel_names must contain one label for each of {channel_count} channels"
        )
    normalized = [_normalize_channel_id(name, "channel_names") for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("channel_names must not contain duplicate labels")
    return normalized


def _array_channel_indices(channels, channel_count: int) -> list[int]:
    """Resolve ordered zero-based source indices for a CZYX array."""
    if channels is None:
        return list(range(channel_count))
    if isinstance(channels, (bool, np.bool_)):
        raise TypeError("channels array indices must not be booleans")
    if isinstance(channels, (int, np.integer)):
        requested = [int(channels)]
    else:
        sequence = _ordered_sequence(channels)
        if sequence is None:
            raise TypeError("channels for a CZYX array must be an integer or sequence")
        if not sequence:
            raise ValueError("channels must not be an empty selection")
        requested = []
        for index in sequence:
            if isinstance(index, (bool, np.bool_)):
                raise TypeError("channels array indices must not be booleans")
            if not isinstance(index, (int, np.integer)):
                raise TypeError("channels for a CZYX array must contain only integers")
            requested.append(int(index))

    if len(set(requested)) != len(requested):
        raise ValueError("channels must not contain duplicate array indices")
    for index in requested:
        if index < 0 or index >= channel_count:
            raise IndexError(
                f"channel index {index} is outside the valid range [0, {channel_count})"
            )
    return requested


def _mapping_channel_keys(channels, available: dict[str | int, np.ndarray]) -> list[str | int]:
    """Resolve ordered mapping keys, treating a string as one key."""
    if channels is None:
        return list(available)
    if isinstance(channels, (str, int, np.integer)) and not isinstance(
        channels, (bool, np.bool_)
    ):
        requested = [_normalize_channel_id(channels, "channels")]
    else:
        sequence = _ordered_sequence(channels)
        if sequence is None:
            # Route bools through the identifier validator for its clearer error.
            _normalize_channel_id(channels, "channels")
            raise AssertionError("unreachable")
        if not sequence:
            raise ValueError("channels must not be an empty selection")
        requested = [_normalize_channel_id(key, "channels") for key in sequence]

    if len(set(requested)) != len(requested):
        raise ValueError("channels must not contain duplicate mapping keys")
    unknown = [key for key in requested if key not in available]
    if unknown:
        raise KeyError(f"unknown channel key(s): {unknown}")
    return requested


def _resolve_channels(
    stack: np.ndarray | Mapping[str | int, np.ndarray],
    channel_names,
    channels,
) -> list[tuple[str | int, np.ndarray]]:
    """Normalize ZYX, CZYX, and mapping inputs to ordered labeled ZYX views."""
    if isinstance(stack, Mapping):
        if channel_names is not None:
            raise ValueError("channel_names must be omitted for mapping input")
        source_items = [
            (_normalize_channel_id(key, "mapping"), image)
            for key, image in stack.items()
        ]
        if len({key for key, _ in source_items}) != len(source_items):
            raise ValueError("mapping channel keys must be distinct")
        validated = _validate_channel_images(source_items)
        available = dict(validated)
        selected_keys = _mapping_channel_keys(channels, available)
        return [(key, available[key]) for key in selected_keys]

    array = np.asarray(stack)
    if array.ndim == 3:
        if channels is not None:
            raise ValueError("channels must be omitted for a single ZYX array")
        if channel_names is None:
            channel_name = 0
        else:
            if _ordered_sequence(channel_names) is not None:
                raise TypeError("channel_names for a ZYX array must be one scalar identifier")
            channel_name = _normalize_channel_id(channel_names, "channel_names")
        return _validate_channel_images([(channel_name, array)])

    if array.ndim == 4:
        if any(size == 0 for size in array.shape):
            raise ValueError("CZYX stack axes must all be nonempty")
        names = _array_channel_names(channel_names, array.shape[0])
        selected_indices = _array_channel_indices(channels, array.shape[0])
        # Integer indexing is deliberately performed one channel at a time so
        # selected CZYX channels remain views, including for strided inputs.
        return _validate_channel_images(
            [(names[index], array[index]) for index in selected_indices]
        )

    raise ValueError("stack must be a ZYX array, a CZYX array, or a channel mapping")


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
                tuple(requested_quantiles.values()),
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
    stack: np.ndarray | Mapping[str | int, np.ndarray],
    regions: PreparedRegions,
    measurements: str | Sequence[str] = DEFAULT_MEASUREMENTS,
    *,
    channel_names=None,
    channels=None,
    max_batch_samples: int = 250_000,
    max_buffered_samples_per_cell: int = 5_000_000,
    mode_bins: str | int | Sequence[float] = "auto",
) -> MeasurementResult:
    """Measure fluorescence within prepared surface-normal cell prisms.

    Parameters
    ----------
    stack
        One nonempty ``(z, y, x)`` array, a channel-first ``(c, z, y, x)``
        array, or a mapping of channel IDs to aligned ZYX arrays. Integer arrays
        are not converted wholesale; interpolation output alone is floating
        point.
    regions
        Reusable regions returned by :func:`prepare_regions`.
    measurements
        Metric name sequence, one name, or ``"all"``. The default is mean,
        median, population standard deviation, and sampled volume.
    channel_names
        A scalar string/integer label for ZYX input, or one ordered, distinct
        label per source channel for CZYX input. Defaults to ``0`` for ZYX and
        source indices for CZYX. Omit for mapping input, whose keys are labels.
    channels
        Ordered channel subset. For CZYX this is one zero-based source index or
        a sequence of indices. For mappings it is one key or a sequence of keys.
        ``None`` selects every channel in source order. Omit for ZYX input.
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
    resolved_channels = _resolve_channels(stack, channel_names, channels)
    requirements = statistics.requirements_for(selected)
    cell_count = len(regions.cells)

    # Classify full prism geometry before sampling. This keeps ``clipped`` and
    # provable disjointness independent of whether quadrature hits the boundary.
    # An empty channel mapping has no rows but retains a stable table schema.
    image_shape_zyx = resolved_channels[0][1].shape if resolved_channels else (0, 0, 0)
    coverages: list[RegionCoverage | None] = (
        _initial_coverages(regions, image_shape_zyx)
        if resolved_channels
        else [None] * cell_count
    )
    plans = build_sampling_plans(regions, coverages) if resolved_channels else []

    # Sampled volume depends only on geometry and is shared across channels.
    # Finite volume and statistical state are channel-specific because NaNs can
    # differ between otherwise aligned images.
    sampled_volume = np.zeros(cell_count, dtype=float)
    nonfinite_omitted = np.zeros((len(resolved_channels), cell_count), dtype=bool)
    # Reducers are created lazily for regions that contribute an in-bounds
    # sample. Finalized scalar results live in fixed output slots, while the
    # potentially large exact-statistic buffers remain only in active reducers.
    active_reducers: list[dict[int, statistics.StreamingReducer]] = [
        {} for _ in resolved_channels
    ]
    finalized_metrics = [
        [_empty_statistics() for _ in regions.cells]
        for _ in resolved_channels
    ]
    finalized_mode_edges: list[list[list[float] | None]] = [
        [None for _ in regions.cells]
        for _ in resolved_channels
    ]
    finite_volume = np.zeros((len(resolved_channels), cell_count), dtype=float)

    # ULP-level arithmetic differences can change a histogram bin at an exact
    # edge. Mode requests therefore retain SciPy's reference interpolation;
    # other metrics use the specialized path for supported image dtypes/shapes.
    specialized_channels = [
        "mode" not in selected and interpolation.can_use_specialized_trilinear(image)
        for _, image in resolved_channels
    ]
    specialized_channel_count = sum(specialized_channels)

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
            prepared_coordinates = (
                interpolation.prepare_trilinear_coordinates(
                    block.stack_index_zyx,
                    image_shape_zyx,
                )
                if specialized_channel_count >= 2
                else None
            )
            # Interpolate one channel at a time so channel count does not
            # multiply coordinate/value scratch memory. The same span offsets
            # partition every channel's output without per-region mask scans.
            for channel_index, (_, image) in enumerate(resolved_channels):
                use_specialized = specialized_channels[channel_index]
                values = interpolation.interpolate_values(
                    image,
                    block.stack_index_zyx,
                    prepared=prepared_coordinates if use_specialized else None,
                    use_specialized=use_specialized,
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
            for channel_index in range(len(resolved_channels)):
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
    for channel_index, (channel_name, _) in enumerate(resolved_channels):
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
        "channels": [channel_name for channel_name, _ in resolved_channels],
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
        table=pd.DataFrame(rows, columns=pd.Index(columns)),
        provenance=provenance,
    )
