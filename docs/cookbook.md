# DeProjPy cookbook

Start with the [README quick start](../README.md#quick-start) for a first run.
The recipes below show how to inspect a result, join measurements, compare
distances, and check intensity coverage.

- [DeProjPy cookbook](#deprojpy-cookbook)
  - [Prepare a sample result](#prepare-a-sample-result)
  - [Inspect geometry and quantify the projection error](#inspect-geometry-and-quantify-the-projection-error)
  - [Join cell features to another table](#join-cell-features-to-another-table)
  - [Compare distances and draw a graph path](#compare-distances-and-draw-a-graph-path)
  - [Preview prisms and measure fluorescence](#preview-prisms-and-measure-fluorescence)

## Prepare a sample result

Run this from the repository root. Later recipes reuse `result` and `cells`.

```python
import deprojpy as dp
from deprojpy import plotting

labels, heightmap = dp.load_label_heightmap_pair(
    "samples/Labels-2.tif", "samples/HeightMap-2.tif"
)
sample_scale = dict(pixel_size=0.183, voxel_depth=1.0, units="µm", invert_z=True)
result = dp.from_labels(labels, heightmap, **sample_scale)
cells = result.to_dataframe()
```

To run the same recipes on the bundled ridge mask, replace the input and
deprojection calls with these:

```python
mask, heightmap = dp.load_tiff_pair(
    "samples/Segmentation-2.tif", "samples/HeightMap-2.tif"
)
result = dp.from_mask(mask, heightmap, **sample_scale)
cells = result.to_dataframe()
```

For your own images, use their calibration and height-map orientation. If
your labels are separated by background, pass `drop_border_cells=False`
and inspect cells cut off by the field of view. See the
[input guide](deprojection_inputs.md) for details on border filtering and
interpreting contact information.

## Inspect geometry and quantify the projection error

Check that cell locations agree with the height map and inspect the lifted
contours before comparing measurements across a curved region.

```python
fig, ax = plotting.plot_heightmap_with_centers(heightmap, result)
fig, ax = plotting.plot_3d_boundaries(result, "area")
```

Map the relative difference between surface-corrected and projected area.
Positive values mark cells where the surface-corrected area is larger.

```python
fig, ax = plotting.plot_relative_error_map(result, "area")
```

![Relative area correction across the sample tissue](images/relative_area_error.jpg)

Change `"area"` to `"perimeter"` for the equivalent perimeter diagnostic.
For a measured feature such as eccentricity, use
`plotting.plot_feature_map(result, "eccentricity")`. The
[plot gallery script](../examples/02_plot_gallery.py) shows histograms,
curvature maps, and side-by-side feature maps.

## Join cell features to another table

For label input, join by `source_label` to keep the association with the
original segmentation, even if some cells were filtered out. Here
`other_features.csv` has one row per original label in this image.

```python
import pandas as pd

other = pd.read_csv("other_features.csv")
combined = cells.merge(other, on="source_label", how="left", validate="one_to_one")
combined.to_csv("combined_cell_features.csv", index=False)
```

Use the left join to retain every deprojected cell. For multiple images,
include an image identifier in the join because label IDs can repeat.
To export geometry alone, use `result.to_csv("cell_geometry.csv")`.

## Compare distances and draw a graph path

Compare projected XY distance with the straight lifted route and a graph
geodesic between the same pair. Here the first and last rows provide sample
endpoints. For your own analysis, select the rows by cell ID.

```python
import numpy as np
from deprojpy import SurfaceDistanceCalculator, SurfaceGraph

start, end = cells.iloc[[0, -1]][["xc", "yc"]].to_numpy()
calculator = SurfaceDistanceCalculator.from_cell_boundaries(result)
graph = SurfaceGraph.from_calculator(calculator, step="auto", connectivity="16")

projected = np.linalg.norm(end - start)
straight = calculator.straight_distance(start, end, input_units="physical")
geodesic, path_px = graph.distance(
    start, end, input_units="physical", return_path=True
)
print(result.units, projected, straight, geodesic)

fig, ax = plotting.plot_3d_boundaries(result, "area")
if len(path_px):
    path_xy = path_px * calculator.pixel_size
    path_z = calculator.sample_height(path_px, input_units="pixel")
    ax.plot(path_xy[:, 0], path_xy[:, 1], path_z, color="black", linewidth=2)
```

The overlay uses a surface reconstructed from the cell boundaries. To measure
on the prepared height map, build the calculator with `from_result(result)`
instead. See the [distance guide](surface_distances.md) for path definitions,
graph resolution, and endpoint snapping.

## Preview prisms and measure fluorescence

Supply an original stack aligned to the segmentation and height map. This
example assumes two channels in CZYX order and uses the default stack spacing
from the deprojection result. Choose the tissue side in the original stack
and inspect a prism before processing the full image.

```python
from deprojpy import intensity

regions = intensity.prepare_regions(
    result,
    depth_range=(0.0, 3.0),
    depth_units="slices",
    tissue_side="increasing_z",
)
cell_id = int(cells.iloc[0]["id"])
fig, ax = intensity.plot_region_preview(result, regions, cell_ids=[cell_id])

measured = intensity.measure(
    stack_czyx,
    regions,
    channel_names=["membrane", "cytosol"],
    channels=1,
    measurements=("mean", "std"),
)
fluorescence = measured.to_dataframe()
fig, ax = intensity.plot_feature_map(
    result, measured, channel="cytosol", feature="mean"
)
```

Inspect flagged rows before comparing intensities. For example, a clipped
cell may still have a finite mean based on the part sampled inside the stack.

```python
needs_review = fluorescence.loc[
    fluorescence["clipped"] | fluorescence["status"].ne("ok"),
    ["id", "channel", "coverage_fraction", "finite_volume", "status", "reason"],
]
print(needs_review.head())
measured.save("intensity-measurements.npz")
```

The [intensity guide](intensity.md) explains calibration for cropped or
resampled stacks, depth units, quality fields, and metric costs. Run the
[synthetic intensity example](../examples/06_surface_normal_intensity.py)
if you want to try the workflow without supplying your own stack.
