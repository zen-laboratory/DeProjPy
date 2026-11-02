# Surface-normal intensity measurements

`deprojpy.intensity` measures a calibrated `(z, y, x)` fluorescence stack in
planar prisms fitted to each cell boundary. The footprint preserves the ordered,
possibly non-convex outline; positive depth follows the explicitly selected
tissue-facing side of the **original** stack.

```python
from deprojpy import intensity

regions = intensity.prepare_regions(
    result,
    depth_range=(-1, 5),
    depth_units="slices",
    tissue_side="increasing_z",
)
values = intensity.measure(
    {"membrane": membrane_zyx, "cytosol": cytosol_zyx},
    regions,
    measurements=("mean", "mode", "std", "sampled_volume"),
    mode_bins="auto",
)
frame = values.to_dataframe()
intensity.plot_feature_map(result, values, channel="cytosol", feature="mean")
values.save("intensity-results.npz")
```

The default acquisition spacing is `(result.voxel_depth, result.pixel_size,
result.pixel_size)`. For a cropped or resampled stack, pass
`StackCalibration((dz, dy, dx), (z0, y0, x0))`; all values must use the result's
length unit. The origin is the original physical coordinate represented by the
supplied stack's index zero. Results serialized before schema 1.2 lack the Z
reflection mapping, so `prepare_regions` requires an explicit `z_reflected` and,
when reflected, `z_reflection_offset`.

Slice depths are axial-spacing equivalents along the physical normal. Thus
`(0, 5)` is exactly five axial spacings thick, including for tilted cells.
Physical depths accept result units, `um`/`µm`, or `mm`. A single
`sampling_spacing` controls the maximum subtriangle edge and depth-bin width.
Equal-width midpoint bins partition the requested depth interval exactly.
Triangle-centroid samples use exact subtriangle area weights, retain narrow and
non-convex footprints, and integrate affine fields exactly when fully in bounds.

Linear interpolation uses the closed image-index domain `[0, size - 1]`.
Partially clipped prisms use the supported portion and are flagged. Nonfinite
interpolates are omitted separately: `sampled_volume` remains the in-bounds
quadrature volume and `finite_volume` is the contributing denominator. Mean,
population standard deviation, median and quartiles are volume weighted;
quantiles use the inverse weighted empirical CDF. `raw_sample_sum` depends on
sampling density. Interpolated measurements estimate a physical region—not an
original-voxel histogram—and oversampling does not add optical resolution.
Clipped and unclipped cells use the same requested `sampling_spacing` and
quadrature. A boundary element contributes its full existing volume weight when
its sampling point lies inside interpolation support, and contributes nothing
otherwise. Thus clipped `sampled_volume` and `coverage_fraction` are numerical
estimates, not exact arbitrary prism/box intersections. Reduce the one explicit
`sampling_spacing` to perform a convergence check. If geometry may intersect
the stack but samples miss it, status is `unresolved_sampling_coverage` rather
than the incorrect `outside_stack`.

`mode` is a volume-weighted histogram estimate rather than exact-value
frequency counting. `mode_bins` accepts `"auto"`, a positive bin count, or
explicit increasing edges. The result is the midpoint of the bin with greatest
volume density (weight divided by width); ties choose the lower bin. Automatic
edges use NumPy's unweighted rule, while bin accumulation remains volume
weighted. The measurement table records the actual edges used for each row.

The model assumes that one biological interior Z direction applies across a
sheet representable by a heightmap. It does not follow curvature within a cell,
taper prisms, resolve overlap, or provide a volumetric segmentation. Exact
median/quantiles buffer one cell's interpolated values; interpolation itself is
generated and interpolated in blocks of at most `max_batch_samples`, including
within one large cell. Streaming-only selections do not sort or buffer sample
values. Exact quantiles and mode have a separate
`max_buffered_samples_per_cell` limit. Each cell is finalized after its last
sample block has been consumed by every channel, and its exact-statistic buffers
are then released. Peak buffer memory therefore follows active regions plus the
current batch rather than all previously completed cells. One large cell can
still require substantial exact-statistic storage, and concatenation/sorting
temporarily adds memory. Cached lateral quadrature is a separate preparation
cost and is not bounded by `max_batch_samples`.

`plot_region_preview` draws the original boundary, both requested depth faces,
connecting prism sides, and the positive-depth normal with physical XYZ aspect.
Measurement archives are versioned `.npz` files containing the table and
provenance but never the image stack. Files preserve channel identifiers,
nonfinite values, mode edges, metric selection, units, calibration, coordinate
mapping, and numerical conventions without NumPy pickle.
