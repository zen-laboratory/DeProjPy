"""Compute straight-line and graph-geodesic distances on a height-map surface."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import deprojpy as dp  # noqa: E402
from deprojpy.surface_distance import (  # noqa: E402
    SurfaceDistanceCalculator,
    SurfaceGraph,
    cell_centers_xy_pixels,
)

LABELS_PATH = REPOSITORY / "samples" / "Labels-2.tif"
HEIGHTMAP_PATH = REPOSITORY / "samples" / "HeightMap-2.tif"
OUTPUT_DIR = REPOSITORY / "examples" / "output"

PIXEL_SIZE = 0.183
VOXEL_DEPTH = 1.0
UNITS = "µm"
INVERT_Z = True


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Edit LABELS_PATH and HEIGHTMAP_PATH near the top of "
            "examples/05_surface_distances.py to point at your TIFF files."
        )


def run_surface_distance_example(
    labels_path: Path = LABELS_PATH,
    heightmap_path: Path = HEIGHTMAP_PATH,
    output_dir: Path = OUTPUT_DIR,
    *,
    dpi: int = 150,
):
    """Run the surface-distance example and return computed distances and output paths."""
    _require_file(labels_path)
    _require_file(heightmap_path)
    labels, heightmap = dp.load_label_heightmap_pair(labels_path, heightmap_path)
    result = dp.from_labels(
        labels,
        heightmap,
        pixel_size=PIXEL_SIZE,
        voxel_depth=VOXEL_DEPTH,
        units=UNITS,
        invert_z=INVERT_Z,
        drop_border_cells=False,
    )

    heightmap_calc = SurfaceDistanceCalculator.from_result(
        result,
        heightmap,
        prepared=False,
        invert_z=INVERT_Z,
        inpaint_zeros=True,
        prune_zeros=True,
    )
    boundary_calc = SurfaceDistanceCalculator.from_cell_boundaries(
        result,
        method="linear",
        extrapolation="linear",
    )
    centers_px = cell_centers_xy_pixels(result)
    centers_phys = np.asarray([cell.center[:2] for cell in result.epicells], dtype=float)

    i = 10 if len(centers_px) > 10 else 0
    j = min(len(centers_px) - 1, 200) if len(centers_px) else 0
    p0_px = centers_px[i]
    p1_px = centers_px[j]
    p0_phys = centers_phys[i]
    p1_phys = centers_phys[j]

    d_xy = float(np.linalg.norm(p1_phys - p0_phys))
    d_center = float(np.linalg.norm(result.epicells[j].center - result.epicells[i].center))
    d_heightmap = heightmap_calc.straight_distance(p0_px, p1_px, input_units="pixel")
    d_boundary_px = boundary_calc.straight_distance(p0_px, p1_px, input_units="pixel")
    d_boundary_phys = boundary_calc.straight_distance(
        p0_phys,
        p1_phys,
        input_units="physical",
    )

    graph = SurfaceGraph.from_calculator(
        boundary_calc,
        step="auto",
        target_nodes=80_000,
        connectivity="16",
    )
    d_graph, path_px = graph.distance(
        p0_phys,
        p1_phys,
        input_units="physical",
        return_path=True,
    )  # type: ignore
    path_z = (
        boundary_calc.sample_height(path_px, input_units="pixel")
        if len(path_px)
        else np.array([])
    )
    path_xyz = (
        np.column_stack([path_px * result.pixel_size, path_z])
        if len(path_px)
        else np.empty((0, 3))
    )

    subset = centers_px[: min(100, len(centers_px))]
    pairwise = heightmap_calc.straight_pairwise_distances(subset, input_units="pixel")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(boundary_calc.heightmap, cmap="viridis")
    ax.plot(
        [p0_px[0], p1_px[0]],
        [p0_px[1], p1_px[1]],
        color="white",
        linewidth=1.2,
        label="straight xy",
    )
    if len(path_px):
        ax.plot(path_px[:, 0], path_px[:, 1], color="red", linewidth=1.0, label="graph path")
    ax.scatter([p0_px[0], p1_px[0]], [p0_px[1], p1_px[1]], c=["cyan", "orange"], s=25)
    ax.set_title("Boundary-interpolated surface paths")
    ax.set_xlabel("x / column (pixels)")
    ax.set_ylabel("y / row (pixels)")
    ax.legend(loc="best")
    path_plot = plot_dir / "surface_distance_paths.png"
    figure.savefig(path_plot, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"2D xy distance:                 {d_xy:.3f} {boundary_calc.units}")
    print(f"3D Euclidean center distance:   {d_center:.3f} {boundary_calc.units}")
    print(f"Heightmap straight distance:    {d_heightmap:.3f} {heightmap_calc.units}")
    print(f"Boundary straight distance:     {d_boundary_px:.3f} {boundary_calc.units}")
    print(f"Boundary distance, phys input:  {d_boundary_phys:.3f} {boundary_calc.units}")
    print(f"Boundary graph-geodesic:        {d_graph:.3f} {boundary_calc.units}")
    print(f"Pairwise matrix shape:          {pairwise.shape}")
    print(f"Wrote path plot:                {path_plot}")

    return {
        "result": result,
        "heightmap_calculator": heightmap_calc,
        "boundary_calculator": boundary_calc,
        "surface_graph": graph,
        "pairwise": pairwise,
        "distances": {
            "xy": d_xy,
            "center_3d": d_center,
            "heightmap_straight_surface": d_heightmap,
            "boundary_straight_surface_pixel_input": d_boundary_px,
            "boundary_straight_surface_physical_input": d_boundary_phys,
            "boundary_graph_geodesic": d_graph,
        },
        "graph_path_xy_pixels": path_px,
        "graph_path_xyz_physical": path_xyz,
        "path_plot": path_plot,
    }


if __name__ == "__main__":
    run_surface_distance_example()
