"""Benchmark region preparation, bounded measurement, and clipped accuracy.

The repository has no toy 3-D fluorescence dataset, so this script uses a
deterministic synthetic fixture containing many small cells, one large cell,
one slender cell, and clipped/disjoint cells. Run from the repository root::

    PYTHONPATH=. python benchmarks/benchmark_intensity_end_to_end.py
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import scipy

from deprojpy import intensity
from deprojpy.intensity._sampling import (
    build_sampling_plans,
    classify_region_coverage,
)


@dataclass
class PeakRSS:
    """Sample Linux resident process memory while one callable runs."""

    interval_seconds: float = 0.002

    def _resident_mib(self) -> float:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        raise RuntimeError("VmRSS is unavailable in /proc/self/status")

    def run(self, function):
        baseline_mib = self._resident_mib()
        peak_mib = baseline_mib
        stop = threading.Event()

        def sample_memory():
            nonlocal peak_mib
            while not stop.is_set():
                peak_mib = max(peak_mib, self._resident_mib())
                sleep(self.interval_seconds)

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        started = perf_counter()
        try:
            result = function()
        finally:
            elapsed_seconds = perf_counter() - started
            stop.set()
            monitor.join()
            peak_mib = max(peak_mib, self._resident_mib())
        return result, elapsed_seconds, baseline_mib, peak_mib


def _result(boundaries):
    """Build the result-like geometry contract needed by prepare_regions."""
    cells = [
        SimpleNamespace(
            id=1000 + 3 * index,
            source_label=2000 + index,
            boundary=np.asarray(boundary, dtype=float),
        )
        for index, boundary in enumerate(boundaries)
    ]
    return SimpleNamespace(
        epicells=cells,
        units="um",
        pixel_size=1.0,
        voxel_depth=1.0,
        z_reflected=False,
        z_reflection_offset=0.0,
    )


def _many_small_boundaries():
    """Return 400 separated 2×2 horizontal cells."""
    boundaries = []
    for row in range(20):
        for column in range(20):
            x0 = 2.0 + 3.0 * column
            y0 = 2.0 + 3.0 * row
            z0 = 10.0 + 0.04 * x0 + 0.02 * y0
            boundaries.append(
                [[x0, y0, z0], [x0 + 2, y0, z0],
                 [x0 + 2, y0 + 2, z0], [x0, y0 + 2, z0]]
            )
    return boundaries


def _large_and_slender_boundaries():
    """Return one area-dominant and one extent-dominant footprint."""
    return [
        [[5, 5, 9], [45, 5, 11], [45, 45, 11], [5, 45, 9]],
        [[5, 52, 10], [58, 52, 11], [58, 53, 11], [5, 53, 10]],
    ]


def _clipped_boundaries():
    """Return fully supported, partially clipped, and disjoint cells."""
    return [
        [[5, 5, 10], [15, 5, 10], [15, 15, 10], [5, 15, 10]],
        [[-0.8, 20, 10], [10, 20, 10], [10, 30, 10], [-0.8, 30, 10]],
        [[70, 5, 10], [80, 5, 10], [80, 15, 10], [70, 15, 10]],
    ]


def _channels(shape=(24, 64, 64)):
    """Create one affine and two nonlinear aligned fluorescence channels."""
    z_index, y_index, x_index = np.indices(shape, dtype=np.float32)
    affine = x_index + 0.25 * y_index + 1.5 * z_index
    wave = np.sin(x_index / 7.0) + np.cos(y_index / 9.0) + z_index / 20.0
    spot = np.exp(
        -((x_index - 30) ** 2 + (y_index - 28) ** 2) / 300.0
        - (z_index - 12) ** 2 / 20.0
    )
    return {"affine": affine, "wave": wave, "spot": spot}


def _prepare(result, sampling_spacing):
    return intensity.prepare_regions(
        result,
        depth_range=(-1.0, 4.0),
        tissue_side="increasing_z",
        sampling_spacing=sampling_spacing,
    )


def _sample_count(regions, image_shape):
    coverages = [
        classify_region_coverage(region, regions, image_shape)
        if region.status == "ok"
        else None
        for region in regions.cells
    ]
    plans = build_sampling_plans(regions, coverages)
    return sum(plan.sample_count for plan in plans)


def _run_case(name, boundaries, selected_channels, measurements, batch_sizes):
    """Report preparation once and measurement for several batch budgets."""
    result = _result(boundaries)
    channels = _channels()
    stack = {key: channels[key] for key in selected_channels}
    monitor = PeakRSS()
    regions, preparation_seconds, _, preparation_peak = monitor.run(
        lambda: _prepare(result, sampling_spacing=0.5)
    )
    total_samples = _sample_count(regions, next(iter(stack.values())).shape)

    print(
        f"\n{name}: cells={len(boundaries)}, channels={len(stack)}, "
        f"metrics={measurements}, samples={total_samples:,}\n"
        f"  prepare_regions: {preparation_seconds:.4f} s, "
        f"peak RSS={preparation_peak:.1f} MiB"
    )
    for batch_size in batch_sizes:
        # Bind this iteration's budget explicitly so the timed closure cannot
        # observe a later loop value.
        def run_measurement(current_batch_size=batch_size):
            return intensity.measure(
                stack,
                regions,
                measurements=measurements,
                max_batch_samples=current_batch_size,
            )

        measured, seconds, baseline_mib, peak_mib = monitor.run(
            run_measurement
        )
        # Historical retention buffered every finite value and weight until all
        # cells finished. This lower-bound estimate excludes Python/list overhead.
        historical_buffer_mib = total_samples * len(stack) * 16 / 1024**2
        print(
            f"  batch={batch_size:>7,}: measure={seconds:.4f} s, "
            f"baseline/peak RSS={baseline_mib:.1f}/{peak_mib:.1f} MiB, "
            f"increment={peak_mib - baseline_mib:.1f} MiB, rows={len(measured.table)}, "
            f"historical exact-buffer lower bound={historical_buffer_mib:.1f} MiB"
        )


def _clipped_accuracy_comparison():
    """Compare base boundary quadrature with an explicitly finer reference."""
    result = _result(_clipped_boundaries()[:2])
    wave = _channels()["wave"]
    coarse_regions = _prepare(result, sampling_spacing=1.0)
    fine_regions = _prepare(result, sampling_spacing=0.125)
    coarse = intensity.measure(
        wave,
        coarse_regions,
        measurements=("mean", "sampled_volume"),
        max_batch_samples=4096,
    ).table
    fine = intensity.measure(
        wave,
        fine_regions,
        measurements=("mean", "sampled_volume"),
        max_batch_samples=4096,
    ).table

    clipped_index = 1
    coarse_row = coarse.iloc[clipped_index]
    fine_row = fine.iloc[clipped_index]
    coverage_difference = abs(
        coarse_row.coverage_fraction - fine_row.coverage_fraction
    )
    mean_difference = abs(coarse_row["mean"] - fine_row["mean"])
    print(
        "\nclipped convergence (spacing 1.0 versus explicit 0.125 reference):\n"
        f"  coverage={coarse_row.coverage_fraction:.6f} / "
        f"{fine_row.coverage_fraction:.6f}, "
        f"absolute difference={coverage_difference:.6f}\n"
        f"  mean={coarse_row['mean']:.6f} / {fine_row['mean']:.6f}, "
        f"absolute difference={mean_difference:.6f}"
    )


def main():
    """Run all reproducible end-to-end performance scenarios."""
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, {platform.platform()}\n"
        "Peak process memory: VmRSS sampled every 2 ms from /proc/self/status (MiB)."
    )
    batches = (256, 4096, 65_536)
    _run_case(
        "many-small/single-channel/streaming",
        _many_small_boundaries(),
        ("affine",),
        ("mean", "std", "integrated_intensity"),
        batches,
    )
    _run_case(
        "many-small/multi-channel/default",
        _many_small_boundaries(),
        ("affine", "wave", "spot"),
        ("mean", "median", "std", "sampled_volume"),
        batches,
    )
    _run_case(
        "large-and-slender/multi-channel/default",
        _large_and_slender_boundaries(),
        ("affine", "wave", "spot"),
        ("mean", "median", "std", "sampled_volume"),
        batches,
    )
    _run_case(
        "clipped-unclipped-disjoint/single-channel/default",
        _clipped_boundaries(),
        ("wave",),
        ("mean", "median", "std", "sampled_volume"),
        batches,
    )
    _clipped_accuracy_comparison()


if __name__ == "__main__":
    main()
