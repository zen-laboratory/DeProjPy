# DeProjPy

DeProjPy measures cells on curved tissue surfaces. Start with a 2D
segmentation and a height map that gives the tissue's height at each pixel.
DeProjPy uses them to recover each cell's 3D outline and measure features
such as area and perimeter in physical units.

The method comes from [MATLAB DeProj](https://github.com/Image-Analysis-Hub/DeProj).
The Python package also accepts integer label images, measures distances
along the tissue, and samples fluorescence at a chosen depth from the surface.

## Quick start

Install with `python -m pip install deprojpy`. The example below uses the
sample images in the repository. Run it from the root of a checkout.

```python
import deprojpy as dp
from deprojpy.plotting import plot_feature_map

labels, heightmap = dp.load_label_heightmap_pair(
    "samples/Labels-2.tif", "samples/HeightMap-2.tif"
)
result = dp.from_labels(
    labels,
    heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
    invert_z=True,  # Orientation used for the bundled sample
)

cells = result.to_dataframe()
print(cells[["source_label", "area", "perimeter"]].head())
fig, ax = plot_feature_map(result, "area")
```

![Deprojected cell area mapped across the sample tissue](examples/output/plots/area_map.png)

Use `result.to_csv("cells.csv")` to export the table. The result also holds
the individual cells, their 3D boundaries, and the prepared height map.

## Choose your segmentation input

| What you have | Entry point |
| --- | --- |
| An integer label image with a unique ID per cell and a background value (usually zero) | `dp.from_labels(labels, heightmap, ...)` |
| A DeProj-style mask with nonzero interiors separated by zero-valued ridges | `dp.from_mask(mask, heightmap, ...)` |

Use a height map that lines up pixel for pixel with the segmentation.
`pixel_size` gives the physical size of a pixel in XY. `voxel_depth` gives
the physical length represented by one unit of the height-map values.

If your labels are separated by background, set `drop_border_cells=False`
to keep them. You can still measure their outlines. Gaps break the contact
information used to infer neighbors and junctions.
[Segmentation inputs and calibration](docs/deprojection_inputs.md)

## How deprojection works

Two cells can look the same size from above even if one lies on a slope
and has a larger surface area. DeProj lifts the outlines by assigning each
contour point a height from the map. The resulting 3D geometry accounts
for the slope when measuring cell area and perimeter.
[Projection artifacts in the original DeProj guide](https://github.com/Image-Analysis-Hub/DeProj#projection-artefacts) · [Method paper](https://doi.org/10.1186/s12915-021-01037-w)

The cell table uses physical `(x, y, z)` coordinates. The fields `xc`,
`yc`, and `zc` give the average position of the simplified boundary
vertices. The `surface_centroid_*` fields give the center of the
reconstructed surface, weighted by area. With label input, `source_label`
lets you match a row back to the original segmentation.
[Property definitions in the original DeProj project](https://github.com/Image-Analysis-Hub/DeProj/blob/master/DeProjProperties.md)

## What DeProjPy adds

- **Label-image input.** Measure contours directly from integer segmentations
  and retain the source labels for joining other measurements. [Input guide](docs/deprojection_inputs.md)
- **Surface distances.** Compare how far apart two cells are when you
  follow a straight line in the image or let the path bend across the tissue.
  [Calculator guide](docs/surface_distances.md)
- **Surface-normal fluorescence.** Measure signal beneath each cell by
  following the local surface direction into the image stack. Choose the
  depth range, channels, and statistics to report. [Intensity guide](docs/intensity.md)

## Examples and development

The [cookbook](docs/cookbook.md) collects plotting and analysis recipes. The
[example scripts](examples/) include complete labeled, mask, distance, and
intensity workflows.

For development, install from a checkout with `python -m pip install -e ".[test]"`
and run `python -m pytest`. Validation covers synthetic invariants and the
original sample data. Full numerical parity with MATLAB has yet to be
established.

DeProjPy and the original MATLAB DeProj are distributed under the BSD
3-Clause license. See [LICENSE](LICENSE).
