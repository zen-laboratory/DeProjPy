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

## End-to-end sampling and measurement

Recorded 2026-11-02 in the same environment with
`benchmark_intensity_end_to_end.py`. The repository has no toy 3-D fluorescence
stack, so the benchmark uses its documented deterministic fixture. Peak process
memory is Linux `VmRSS` sampled every 2 ms from `/proc/self/status`; values are
MiB. RSS increments can understate reusable allocator memory in later cases, so
both baseline and absolute peak are reported by the script.

Preparation and representative measurement results:

| Case | Cells | Samples | `prepare_regions` | Batch | `measure` | Baseline/peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| many small, 1 channel, streaming | 400 | 256,000 | 0.618 s | 256 / 4,096 / 65,536 | 0.122 / 0.058 / 0.071 s | 218.8 / 220.1–234.3 MiB |
| many small, 3 channels, defaults | 400 | 256,000 | 0.599 s | 256 / 4,096 / 65,536 | 0.236 / 0.143 / 0.172 s | 229.7 / 230.2–238.4 MiB |
| large + slender, 3 channels, defaults | 2 | 372,000 | 0.765 s | 256 / 4,096 / 65,536 | 0.426 / 0.163 / 0.157 s | 233.3–236.0 / 252.1–271.3 MiB |
| clipped + unclipped + disjoint, defaults | 3 | 40,960 | 0.126 s | 256 / 4,096 / 65,536 | 0.017 / 0.007 / 0.007 s | 235.8 / 235.8 MiB |

A batch budget around 4,096 was best for many small cells. A 65,536 budget was
slightly faster for the two large cells but used roughly 19 MiB more peak RSS;
there is no evidence for a separate generic high-memory mode.

The historical implementation refined a clipped cell at one-quarter spacing
and retained completed exact-statistic buffers. On this same fixture/environment
it generated 1,331,200 samples for the clipped case and took 0.895 s at batch
4,096, versus 40,960 samples and 0.007 s after adopting the base-spacing policy.
For 400 cells × 3 channels with default median, historical first-run RSS grew by
11.2 MiB; completed-region release reduced the comparable increment to 0.5 MiB.
The historical exact-buffer lower bound is `samples × channels × 16` bytes and
excludes Python/list and temporary concatenation/sort overhead.

### Clipped-boundary accuracy tradeoff

For the representative clipped nonlinear field, requested spacing 1.0 versus an
explicit spacing-0.125 reference produced:

| Quantity | Spacing 1.0 | Spacing 0.125 | Absolute difference |
|---|---:|---:|---:|
| coverage fraction | 0.937500 | 0.925781 | 0.011719 |
| weighted mean | 0.280963 | 0.288550 | 0.007586 |

This comparison demonstrates why clipped coverage is a quadrature estimate, not
an exact intersection. Users needing a convergence check should explicitly
reduce the single public `sampling_spacing`.
