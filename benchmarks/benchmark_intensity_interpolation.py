"""Compare reusable linear interpolators on representative intensity samples.

Run from the repository root with::

    python benchmarks/benchmark_intensity_interpolation.py

This is evidence-producing development code, not a wall-clock unit test.
"""

from __future__ import annotations

import platform
from time import perf_counter

import numpy as np
import scipy
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


def _time(callable_, repeats=5):
    durations = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = callable_()
        durations.append(perf_counter() - started)
    return result, min(durations), float(np.median(durations))


def _case(name, shape, sample_count, channel_count, seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=shape).astype(np.float32)
    channels = [base + channel_index for channel_index in range(channel_count)]
    coordinates_zyx = np.column_stack(
        [rng.uniform(0, size - 1, sample_count) for size in shape]
    )
    grid = tuple(np.arange(size, dtype=float) for size in shape)

    map_values, map_min, map_median = _time(
        lambda: [
            ndimage.map_coordinates(
                image,
                coordinates_zyx.T,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
                output=float,
            )
            for image in channels
        ]
    )

    setup_started = perf_counter()
    interpolators = [
        RegularGridInterpolator(
            grid,
            image,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for image in channels
    ]
    setup_seconds = perf_counter() - setup_started
    regular_values, regular_min, regular_median = _time(
        lambda: [interpolator(coordinates_zyx) for interpolator in interpolators]
    )

    maximum_error = max(
        float(np.max(np.abs(first - second)))
        for first, second in zip(map_values, regular_values, strict=True)
    )
    coordinate_megabytes = (
        coordinates_zyx.nbytes + coordinates_zyx.T.copy().nbytes
    ) / 1024**2
    print(
        f"{name}: shape={shape}, samples={sample_count:,}, channels={channel_count}\n"
        f"  map_coordinates min/median: {map_min:.6f}/{map_median:.6f} s\n"
        f"  RegularGridInterpolator setup: {setup_seconds:.6f} s\n"
        f"  RegularGridInterpolator eval min/median: "
        f"{regular_min:.6f}/{regular_median:.6f} s\n"
        f"  max absolute disagreement: {maximum_error:.3e}\n"
        f"  coordinate + transposed-copy memory: {coordinate_megabytes:.2f} MiB"
    )


def main():
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, {platform.platform()}"
    )
    _case("many-small-cells-equivalent", (48, 256, 256), 200_000, 3, seed=1)
    _case("long-thin-or-large-cell-equivalent", (96, 192, 192), 750_000, 2, seed=2)


if __name__ == "__main__":
    main()
