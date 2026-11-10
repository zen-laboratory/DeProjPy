# Segmentation inputs and calibration

DeProjPy uses the height map to lift each cell outline into 3D. Both input
formats give you the same kind of result. Choose the function that matches
how your segmentation marks cells.

| | Label image | Ridge mask |
| --- | --- | --- |
| Function | `from_labels` | `from_mask` |
| Cell interior | Pixels share a unique integer label | Nonzero connected interior |
| Separation | Distinct IDs, with or without intervening background | Zero-valued ridges separate interiors |
| Cell identity | `source_label` records the original segmentation ID | IDs come from the mask workflow |

## Label images

```python
import deprojpy as dp

labels, heightmap = dp.load_label_heightmap_pair("labels.tif", "heightmap.tif")
result = dp.from_labels(
    labels,
    heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
    drop_border_cells=False,
)
```

If your cells are separated by background, use `drop_border_cells=False`
as shown above. The default follows the original mask workflow's practice
of excluding border cells. For label images, it drops any label touching
the image edge or the configured background (usually zero). Turning it
off keeps separated cells and also keeps cells cut off by the image edge,
so check those outlines before using their measurements.

Area and perimeter come from each cell's own outline, so gaps between
labels are fine for those measurements. Neighbors and junctions are
inferred from which labels meet in the image. If your segmentation leaves
gaps, ignore those fields when assessing cell contacts.

## Ridge masks

```python
mask, heightmap = dp.load_tiff_pair("segmentation.tif", "heightmap.tif")
result = dp.from_mask(
    mask,
    heightmap,
    pixel_size=0.183,
    voxel_depth=1.0,
    units="µm",
)
```

`from_mask` is the preferred name for the original mask workflow.
`from_heightmap` remains available as its legacy alias. Keep the zero-valued lines
between cells continuous so that each interior is detected separately.
[Segmentation masks in the original DeProj guide](https://github.com/Image-Analysis-Hub/DeProj#the-segmentation-mask)

## Height map and coordinates

The height map must have the same shape as the segmentation, and each
height value must line up with the corresponding segmentation pixel. Use the
same length unit for XY and Z scales. If the height map stores slice indices,
`voxel_depth` is the physical slice spacing. If it already stores heights in
the chosen physical unit, use `voxel_depth=1`.

Set `invert_z` according to the desired surface orientation. Zero-height
inpainting and pruning are enabled by default. If zero is a valid height in
your map, review those options before processing. Image arrays use
`(row, column)` indexing. The result's geometric coordinates use physical
`(x, y, z)`, where x follows columns and y follows rows.

See the original [DeProj property definitions](https://github.com/Image-Analysis-Hub/DeProj/blob/master/DeProjProperties.md)
for inherited morphology fields.
