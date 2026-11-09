"""Selective streaming and buffered volume-weighted statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numba import njit


@dataclass(frozen=True)
class ReductionRequirements:
    """Computations needed for a selected public metric set."""

    weighted_mean: bool
    weighted_variance: bool
    integrated_intensity: bool
    raw_sample_sum: bool
    minimum: bool
    maximum: bool
    quantiles: bool
    mode: bool

    @property
    def buffers_samples(self) -> bool:
        """Return whether exact finalization needs finite samples retained."""
        return self.quantiles or self.mode


def requirements_for(selected: Sequence[str]) -> ReductionRequirements:
    """Translate public metrics into the smallest required reduction set."""
    # Standard deviation needs a running mean, but a mean-only request does not
    # need the more expensive second central moment.
    names = set(selected)
    return ReductionRequirements(
        weighted_mean=bool(names & {"mean", "std"}),
        weighted_variance="std" in names,
        integrated_intensity="integrated_intensity" in names,
        raw_sample_sum="raw_sample_sum" in names,
        minimum="minimum" in names,
        maximum="maximum" in names,
        quantiles=bool(names & {"median", "q25", "q75"}),
        mode="mode" in names,
    )


@dataclass
class StreamingReducer:
    """Accumulate only the weighted statistics requested by one channel/cell."""

    requirements: ReductionRequirements
    max_buffered_samples: int
    finite_volume: float = 0.0
    finite_count: int = 0
    mean: float = 0.0
    weighted_m2: float = 0.0
    weighted_sum: float = 0.0
    raw_sum: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf
    _value_parts: list[np.ndarray] = field(default_factory=list, repr=False)
    _weight_parts: list[np.ndarray] = field(default_factory=list, repr=False)

    def update(self, values: np.ndarray, weights: np.ndarray) -> None:
        """Consume one finite-value block without retaining it unnecessarily."""
        if len(values) == 0:
            return

        # Finite volume is always accumulated because it is a quality field and
        # the denominator for all finite-data weighted statistics.
        block_volume = float(np.sum(weights))
        previous_volume = self.finite_volume
        self.finite_volume += block_volume
        self.finite_count += len(values)

        block_weighted_sum = None
        if self.requirements.weighted_mean or self.requirements.integrated_intensity:
            block_weighted_sum = float(np.sum(values * weights))

        # Merge block moments with the parallel weighted-Welford formula. This
        # avoids the cancellation of E[x²] - E[x]² across many batches.
        if self.requirements.weighted_mean:
            block_mean = block_weighted_sum / block_volume # type: ignore (block_weighted_sum is not None if weighted_mean is True)
            block_m2 = (
                float(np.sum(weights * (values - block_mean) ** 2))
                if self.requirements.weighted_variance
                else 0.0
            )
            if previous_volume == 0.0:
                self.mean = block_mean
                self.weighted_m2 = block_m2
            else:
                delta = block_mean - self.mean
                combined_volume = previous_volume + block_volume
                self.mean += delta * block_volume / combined_volume
                self.weighted_m2 += (
                    block_m2
                    + delta**2 * previous_volume * block_volume / combined_volume
                )

        if self.requirements.integrated_intensity:
            self.weighted_sum += block_weighted_sum # type: ignore (block_weighted_sum is not None if weighted_mean is True)
        if self.requirements.raw_sample_sum:
            self.raw_sum += float(np.sum(values))
        if self.requirements.minimum:
            self.minimum = min(self.minimum, float(np.min(values)))
        if self.requirements.maximum:
            self.maximum = max(self.maximum, float(np.max(values)))

        # Exact empirical quantiles and automatic histogram mode need the raw
        # finite samples. Their memory budget is independent of interpolation.
        if self.requirements.buffers_samples:
            if self.finite_count > self.max_buffered_samples:
                raise MemoryError(
                    "exact quantile/mode buffering exceeded "
                    f"max_buffered_samples_per_cell={self.max_buffered_samples}"
                )
            self._value_parts.append(np.array(values, dtype=float, copy=True))
            self._weight_parts.append(np.array(weights, dtype=float, copy=True))

    def buffered_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Return buffered finite values/weights, materializing them once."""
        if not self.requirements.buffers_samples or self.finite_count == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        return np.concatenate(self._value_parts), np.concatenate(self._weight_parts)

    @property
    def buffered_sample_count(self) -> int:
        """Return retained sample count without concatenating buffer parts."""
        return sum(len(part) for part in self._value_parts)

    def clear_buffered_samples(self) -> None:
        """Release exact-statistic buffers after this region is finalized.

        The reducer's scalar streaming state remains available, but measurement
        removes the reducer immediately afterward. Clearing explicitly makes
        lifetime independent of garbage-collector timing and is observable in
        focused memory-lifetime tests.
        """
        self._value_parts.clear()
        self._weight_parts.clear()


def _stable_weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> dict[float, float]:
    """Reference inverse weighted empirical CDF using one stable sort."""
    # One stable ordering serves every requested probability and gives
    # deterministic behavior when repeated values straddle a threshold.
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    total_weight = cumulative_weights[-1]

    result = {}
    for quantile in quantiles:
        target = quantile * total_weight
        index = int(np.searchsorted(cumulative_weights, target, side="left"))
        result[float(quantile)] = float(
            ordered_values[min(index, len(ordered_values) - 1)]
        )
    return result


@njit(cache=True)
def _weighted_select_candidates(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Select weighted candidates with deterministic three-way partitions."""
    output = np.empty(len(quantiles), dtype=np.float64)
    work_values = values.copy()
    work_weights = weights.copy()
    total_weight = 0.0
    for weight in work_weights:
        total_weight += weight

    # Quantiles are few but may arrive in caller order. Sorting their targets
    # lets left/equal/right branches share every partition of the sample arrays.
    quantile_order = np.argsort(quantiles)
    targets = quantiles[quantile_order] * total_weight
    stack_capacity = 2 * len(quantiles) + 1
    low_stack = np.empty(stack_capacity, dtype=np.int64)
    high_stack = np.empty(stack_capacity, dtype=np.int64)
    quantile_low_stack = np.empty(stack_capacity, dtype=np.int64)
    quantile_high_stack = np.empty(stack_capacity, dtype=np.int64)
    prefix_weight_stack = np.empty(stack_capacity, dtype=np.float64)
    stack_size = 1
    low_stack[0] = 0
    high_stack[0] = len(work_values)
    quantile_low_stack[0] = 0
    quantile_high_stack[0] = len(quantiles)
    prefix_weight_stack[0] = 0.0

    while stack_size:
        stack_size -= 1
        low = low_stack[stack_size]
        high = high_stack[stack_size]
        quantile_low = quantile_low_stack[stack_size]
        quantile_high = quantile_high_stack[stack_size]
        prefix_weight = prefix_weight_stack[stack_size]

        # Median-of-three avoids the deterministic worst case of choosing an
        # already sorted endpoint while guaranteeing a present pivot.
        first = work_values[low]
        middle = work_values[low + (high - low) // 2]
        last = work_values[high - 1]
        if first < middle:
            pivot = middle if middle < last else (last if first < last else first)
        else:
            pivot = first if first < last else (last if middle < last else middle)

        less = low
        scan = low
        greater = high
        less_weight = 0.0
        equal_weight = 0.0
        while scan < greater:
            value = work_values[scan]
            if value < pivot:
                saved_value = work_values[less]
                work_values[less] = work_values[scan]
                work_values[scan] = saved_value
                saved_weight = work_weights[less]
                work_weights[less] = work_weights[scan]
                work_weights[scan] = saved_weight
                less_weight += work_weights[less]
                less += 1
                scan += 1
            elif value > pivot:
                greater -= 1
                saved_value = work_values[greater]
                work_values[greater] = work_values[scan]
                work_values[scan] = saved_value
                saved_weight = work_weights[greater]
                work_weights[greater] = work_weights[scan]
                work_weights[scan] = saved_weight
            else:
                equal_weight += work_weights[scan]
                scan += 1

        less_boundary = prefix_weight + less_weight
        equal_boundary = less_boundary + equal_weight
        left_end = quantile_low
        while left_end < quantile_high and targets[left_end] <= less_boundary:
            left_end += 1
        equal_end = left_end
        while equal_end < quantile_high and targets[equal_end] <= equal_boundary:
            output[quantile_order[equal_end]] = pivot
            equal_end += 1

        if equal_end < quantile_high:
            low_stack[stack_size] = greater
            high_stack[stack_size] = high
            quantile_low_stack[stack_size] = equal_end
            quantile_high_stack[stack_size] = quantile_high
            prefix_weight_stack[stack_size] = equal_boundary
            stack_size += 1
        if quantile_low < left_end:
            low_stack[stack_size] = low
            high_stack[stack_size] = less
            quantile_low_stack[stack_size] = quantile_low
            quantile_high_stack[stack_size] = left_end
            prefix_weight_stack[stack_size] = prefix_weight
            stack_size += 1

    return output


def _summation_relative_error(count: int, *, includes_product: bool = False) -> float:
    """Bound reference float64 sums relative to a long-double estimate."""
    float_epsilon = np.finfo(float).eps
    long_epsilon = np.finfo(np.longdouble).eps
    float_error = count * float_epsilon / (1.0 - count * float_epsilon)
    long_error = count * long_epsilon / (1.0 - count * long_epsilon)
    combined = float_error + long_error + float_error * long_error
    if includes_product:
        combined += float_epsilon + combined * float_epsilon
    return float(combined)


def _selection_is_certified(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
    candidate: float,
) -> bool:
    """Prove that summation order cannot move the reference threshold."""
    # The stable reference adds positive weights in value order. Long-double
    # sums estimate the exact totals; conservative error bounds cover both that
    # estimate and every float64 cumulative addition in the reference path.
    total = np.sum(weights, dtype=np.longdouble)
    total_error = _summation_relative_error(len(weights), includes_product=True)

    below_mask = values < candidate
    below_count = int(np.count_nonzero(below_mask))
    below = np.sum(weights[below_mask], dtype=np.longdouble)
    del below_mask

    through_mask = values <= candidate
    through_count = int(np.count_nonzero(through_mask))
    through = np.sum(weights[through_mask], dtype=np.longdouble)
    del through_mask

    below_error = _summation_relative_error(below_count)
    through_error = _summation_relative_error(through_count)
    target = np.longdouble(quantile) * total
    target_low = target * (1.0 - total_error)
    target_high = target * (1.0 + total_error)
    below_high = below * (1.0 + below_error)
    through_low = through * (1.0 - through_error)
    return bool(target_low > below_high and target_high <= through_low)


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> dict[float, float]:
    """Calculate exact inverse-weighted-empirical-CDF quantiles."""
    requested = tuple(float(quantile) for quantile in quantiles)
    candidates = _weighted_select_candidates(
        values,
        weights,
        np.asarray(requested, dtype=float),
    )

    result = {}
    for quantile, candidate in zip(requested, candidates, strict=True):
        if not _selection_is_certified(values, weights, quantile, candidate):
            # Near a cumulative-weight boundary, partition regrouping can alter
            # the chosen sample. Stable sorting retains the historical answer.
            return _stable_weighted_quantiles(values, weights, requested)
        result[quantile] = float(candidate)
    return result


def validate_mode_bins(mode_bins) -> str | int | np.ndarray:
    """Normalize the histogram-mode bin specification."""
    if isinstance(mode_bins, str):
        if mode_bins != "auto":
            raise ValueError("mode_bins string must be 'auto'")
        return mode_bins
    if isinstance(mode_bins, (int, np.integer)):
        if int(mode_bins) < 1:
            raise ValueError("integer mode_bins must be at least one")
        return int(mode_bins)

    # Explicit edges are copied into a predictable floating representation;
    # unequal widths are allowed because selection later compares density.
    edges = np.asarray(mode_bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("explicit mode_bins must be a one-dimensional edge sequence")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("explicit mode bin edges must be finite and strictly increasing")
    return edges


def weighted_histogram_mode(
    values: np.ndarray,
    weights: np.ndarray,
    mode_bins: str | int | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Estimate a continuous-data mode from weighted histogram density.

    The representative is the midpoint of the highest-density bin, where
    density is volume weight divided by bin width. Ties select the lowest bin.
    Explicit edges exclude values outside their range. Constant data returns
    the constant exactly and records a degenerate two-edge range around it.
    Automatic edges use NumPy's unweighted ``'auto'`` rule; only histogram
    accumulation is volume weighted.
    """
    if len(values) == 0:
        return np.nan, np.empty(0, dtype=float)

    # Explicit edges define a comparison range. Values outside it contribute
    # neither weight nor a modal candidate; NumPy includes the rightmost edge.
    if isinstance(mode_bins, np.ndarray):
        included = (values >= mode_bins[0]) & (values <= mode_bins[-1])
        values = values[included]
        weights = weights[included]
        if len(values) == 0:
            return np.nan, mode_bins
    # A constant distribution has an unambiguous mode even though an automatic
    # histogram would otherwise invent an arbitrary finite-width neighborhood.
    if np.all(values == values[0]):
        value = float(values[0])
        if isinstance(mode_bins, np.ndarray):
            return value, mode_bins
        scale = max(abs(value), 1.0)
        half_width = np.finfo(float).eps * scale
        return value, np.asarray([value - half_width, value + half_width])

    if isinstance(mode_bins, np.ndarray):
        edges = mode_bins
    else:
        edges = np.histogram_bin_edges(values, bins=mode_bins)
    # Density, rather than total bin weight, prevents a wide unequal bin from
    # winning solely because it covers a larger intensity interval.
    histogram_weights, edges = np.histogram(values, bins=edges, weights=weights)
    density = histogram_weights / np.diff(edges)
    if not np.any(histogram_weights > 0):
        return np.nan, edges
    modal_index = int(np.argmax(density))
    representative = 0.5 * (edges[modal_index] + edges[modal_index + 1])
    return float(representative), edges
