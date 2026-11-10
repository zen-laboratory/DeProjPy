"""Measure and visualize small synthetic surface-normal intensity regions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deprojpy as dp


def _synthetic_result():
    """Build four cells on a tilted heightmap without external data files."""
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[4:20, 4:20] = 11
    labels[4:20, 21:36] = 22
    labels[21:36, 4:20] = 33
    labels[21:36, 21:36] = 44

    rows, columns = np.indices(labels.shape, dtype=float)
    heightmap = 8.0 + 0.05 * columns + 0.03 * rows
    result = dp.from_labels(
        labels,
        heightmap,
        pixel_size=0.5,
        voxel_depth=1.0,
        units="um",
        inpaint_zeros=False,
        prune_zeros=False,
        drop_border_cells=False,
    )
    return result, labels.shape


def _synthetic_channels(image_shape_yx):
    """Return one small channel-first stack with axis order ``(c, z, y, x)``."""
    z_index, y_index, x_index = np.indices((14, *image_shape_yx), dtype=float)
    affine = x_index + 0.5 * y_index + 2.0 * z_index
    spot = np.exp(
        -(
            (x_index - 20.0) ** 2 / 60.0
            + (y_index - 18.0) ** 2 / 80.0
            + (z_index - 10.0) ** 2 / 5.0
        )
    )
    return np.stack([affine, spot])


def main(output_dir: str | Path = "examples/output/intensity"):
    """Run preparation, measurement, persistence, and plotting end to end."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result, image_shape_yx = _synthetic_result()
    stack_czyx = _synthetic_channels(image_shape_yx)
    calibration = dp.intensity.StackCalibration(
        spacing_zyx=(1.0, 0.5, 0.5),
        origin_zyx=(0.0, 0.0, 0.0),
    )

    # Signed slice depths are axial-spacing equivalents along each cell normal.
    # The long positive interval intentionally clips the shallow synthetic stack.
    regions = dp.intensity.prepare_regions(
        result,
        depth_range=(-1.5, 6.0),
        depth_units="slices",
        tissue_side="increasing_z",
        calibration=calibration,
        sampling_spacing=0.5,
    )
    measurements = dp.intensity.measure(
        stack_czyx,
        regions,
        channel_names=["affine", "spot"],
        channels=[0, 1],
        measurements=("mean", "mode", "std", "sampled_volume"),
        mode_bins=np.linspace(0.0, 70.0, 15),
        max_batch_samples=2_000,
    )

    # A physical-unit interval uses the same API; it is prepared here to make
    # the distinction explicit without duplicating the channel measurement.
    physical_regions = dp.intensity.prepare_regions(
        result,
        depth_range=(-0.75, 3.0),
        depth_units="um",
        tissue_side="increasing_z",
        calibration=calibration,
    )
    print("physical depth bounds:", physical_regions.converted_depth_range)
    print(measurements.to_dataframe())

    archive = measurements.save(output_path / "measurements.npz", overwrite=True)
    reloaded = dp.intensity.MeasurementResult.load(archive)

    # Preview both prism faces and the inward normal for one stable cell ID.
    first_cell_id = result.epicells[0].id
    preview_figure, preview_axis = dp.intensity.plot_region_preview(
        result,
        regions,
        cell_ids=[first_cell_id],
    )
    preview_axis.view_init(elev=24, azim=-55)
    preview_figure.savefig(output_path / "prism_preview.png", dpi=150)

    feature_figure, _ = dp.intensity.plot_feature_map(
        result,
        reloaded,
        channel="affine",
        feature="mean",
        title="Mean affine intensity",
    )
    feature_figure.savefig(output_path / "mean_feature_map.png", dpi=150)
    plt.close(preview_figure)
    plt.close(feature_figure)
    return result, regions, reloaded


if __name__ == "__main__":
    main()
