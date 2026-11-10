# DeProjPy

DeProjPy measures cells on curved tissue surfaces. It lifts a 2D cell
segmentation onto an aligned height map to estimate cell area, perimeter, and
other features in physical units. It supports labeled segmentations,
DeProj-style ridge masks, surface distances, and surface-normal fluorescence
measurements in calibrated image stacks.

```bash
python -m pip install deprojpy
```

Start with the [quick start and sample data](https://github.com/zen-laboratory/DeProjPy#quick-start),
then explore the guides for
[segmentation and calibration](https://github.com/zen-laboratory/DeProjPy/blob/main/docs/deprojection_inputs.md),
[surface distances](https://github.com/zen-laboratory/DeProjPy/blob/main/docs/surface_distances.md),
and [fluorescence intensity](https://github.com/zen-laboratory/DeProjPy/blob/main/docs/intensity.md).

DeProjPy builds on the original [MATLAB DeProj](https://github.com/Image-Analysis-Hub/DeProj)
workflow. It is distributed under the BSD 3-Clause license.
