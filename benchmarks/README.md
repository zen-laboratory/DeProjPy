# Intensity interpolation benchmark

Recorded 2026-11-02 with Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, and
Linux 6.8.0-138-generic x86-64. The script was
`benchmarks/benchmark_intensity_interpolation.py`; each timing reports the
minimum and median of five evaluations.

| Case | Shape | Samples | Channels | `map_coordinates` min/median | RGI setup | RGI evaluation min/median | Maximum disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| many small cells equivalent | 48×256×256 | 200,000 | 3 | 0.0854 / 0.0870 s | 0.000112 s | 0.1552 / 0.1613 s | 1.33e-15 |
| long/thin or large cell equivalent | 96×192×192 | 750,000 | 2 | 0.1551 / 0.1631 s | 0.000074 s | 0.4249 / 0.4394 s | 1.33e-15 |

The methods agree to floating-point precision. Reusable
`RegularGridInterpolator` has negligible setup cost, but its evaluation was
about 1.8× and 2.7× slower in these cases. The implementation therefore keeps
batched `scipy.ndimage.map_coordinates(order=1, prefilter=False)` as its single
backend. Coordinate plus transposed-coordinate storage would be approximately
9.16 MiB and 34.33 MiB if these cases were evaluated in one block; the public
pipeline instead bounds this scratch memory with `max_batch_samples`.

These timings characterize this environment rather than promise user-facing
performance. They are deliberately outside unit tests.
