# Distances along the tissue surface

Two cells can look close together in the image even when the tissue rises
steeply between them. DeProjPy measures how far apart they are along the
surface. You can follow a fixed route or let the path find a shorter way.

- **Straight lifted path.** Draw a straight line between the endpoints in
  the XY image. Read the height along that line and add up the lengths of
  the resulting 3D segments. This answers how far you travel if you follow
  that line across the tissue.
- **Graph geodesic.** Sample the surface at a grid of points. Connect nearby
  points and give each connection its physical 3D length. The shortest
  route through these connections can bend in XY to go around a steep
  feature. It approximates the shortest route on the continuous surface.

The graph uses nearby grid points in place of the exact endpoints. Its
resolution and choice of connections can also affect the route length.

## Choose the surface

- `SurfaceDistanceCalculator.from_result(result)` uses the prepared height
  map stored with the result.
- `SurfaceDistanceCalculator.from_cell_boundaries(result)` builds a
  new height map from the 3D cell outlines. Use it to draw paths on the
  surface formed by those outlines. If a path leaves the area covered by
  the outlines, the extrapolation setting determines how heights are filled.
- `SurfaceDistanceCalculator.from_heightmap(...)` starts directly from a
  height map when no deprojection result is needed.

## Measure and inspect a path

```python
import numpy as np
from deprojpy import SurfaceDistanceCalculator, SurfaceGraph

calculator = SurfaceDistanceCalculator.from_result(result)
centers = result.to_dataframe()[["xc", "yc"]].to_numpy()
start, end = centers[:2]

straight = calculator.straight_distance(start, end, input_units="physical")

graph = SurfaceGraph.from_calculator(calculator, step="auto", connectivity="16")
geodesic, path_px = graph.distance(
    start, end, input_units="physical", return_path=True
)

if len(path_px):
    path_z = calculator.sample_height(path_px, input_units="pixel")
    path_xyz = np.column_stack([path_px * calculator.pixel_size, path_z])
```

Inputs use `(x, y)` in pixels by default. Set `input_units="physical"` for
coordinates from the cell table. Distances are in physical units in either
case. Graph path coordinates are always in pixels. The graph snaps endpoints
to nearby nodes, so its distance covers the node-to-node route.

![Example paths on a reconstructed tissue surface](images/3D_distance.png)

## Resolution and repeated queries

With `step=1`, every image pixel can be a graph point. With `step="auto"`,
the graph can use a coarser grid to limit its size. A finer grid can pick
up smaller surface features and put graph points closer to the endpoints.

Connectivity (`"4"`, `"8"`, or `"16"`) controls the directions the route
can take between points. More directions make routes less dependent on
how the image grid is oriented, but add connections to search. If a small
distance difference matters, check whether it holds at a finer resolution.

Reuse the calculator and graph for repeated queries on the same surface.
Use `straight_distances_for_pairs` for selected pairs,
`straight_pairwise_distances` for a full pairwise matrix, and
`graph.distances_from_source` for several graph destinations from one
source. See the [surface-distance example](../examples/05_surface_distances.py).
