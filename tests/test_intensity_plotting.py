from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize

from deprojpy import intensity
from deprojpy.plotting import plot_values_as_feature_map


def _result_and_regions():
    boundaries = [
        np.asarray([[1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3]], dtype=float),
        np.asarray([[5, 1, 4], [7, 1, 4], [7, 3, 4], [5, 3, 4]], dtype=float),
    ]
    cells = [
        SimpleNamespace(id=cell_id, source_label=cell_id, boundary=boundary)
        for cell_id, boundary in zip((7, 91), boundaries, strict=True)
    ]
    result = SimpleNamespace(
        epicells=cells,
        units="um",
        pixel_size=1.0,
        voxel_depth=1.0,
        z_reflected=False,
        z_reflection_offset=0.0,
    )
    regions = intensity.prepare_regions(
        result,
        (-1, 2),
        tissue_side="increasing_z",
    )
    return result, regions


def _measurements():
    # Deliberately reverse row order to exercise identifier-based alignment.
    table = pd.DataFrame(
        {
            "id": [91, 7],
            "channel": ["green", "green"],
            "mean": [20.0, 10.0],
        }
    )
    return intensity.MeasurementResult(table=table)


def test_feature_map_aligns_shuffled_rows_and_accepts_colorbar_label():
    result, _ = _result_and_regions()
    figure, axis = intensity.plot_feature_map(
        result,
        _measurements(),
        channel="green",
        feature="mean",
        colorbar=False,
        colorbar_label="custom intensity",
    )

    color_map = plt.colormaps.get_cmap("viridis")
    normalizer = Normalize(10, 20)
    np.testing.assert_allclose(axis.patches[0].get_facecolor(), color_map(normalizer(10)))
    np.testing.assert_allclose(axis.patches[1].get_facecolor(), color_map(normalizer(20)))
    plt.close(figure)


def test_feature_map_rejects_channel_id_and_shape_mismatches():
    result, _ = _result_and_regions()
    measurements = _measurements()

    with pytest.raises(ValueError, match="absent"):
        intensity.plot_feature_map(result, measurements, channel="red")

    duplicate = intensity.MeasurementResult(
        table=pd.concat([measurements.table, measurements.table.iloc[[0]]], ignore_index=True)
    )
    with pytest.raises(ValueError, match="duplicate"):
        intensity.plot_feature_map(result, duplicate, channel="green")

    missing = intensity.MeasurementResult(table=measurements.table.iloc[[0]])
    with pytest.raises(ValueError, match="do not match"):
        intensity.plot_feature_map(result, missing, channel="green")

    with pytest.raises(ValueError, match="shape"):
        plot_values_as_feature_map(result, np.asarray([1.0]), colorbar=False)


def test_feature_map_keeps_integer_and_string_channel_ids_distinct():
    result, _ = _result_and_regions()
    measurements = intensity.MeasurementResult(
        table=pd.DataFrame(
            {
                "id": [7, 91, 7, 91],
                "channel": [1, 1, "1", "1"],
                "mean": [10.0, 20.0, 100.0, 200.0],
            }
        )
    )
    figure, axis = intensity.plot_feature_map(
        result,
        measurements,
        channel=1,
        colorbar=False,
    )
    color_map = plt.colormaps.get_cmap("viridis")
    np.testing.assert_allclose(axis.patches[0].get_facecolor(), color_map(0.0))
    np.testing.assert_allclose(axis.patches[1].get_facecolor(), color_map(1.0))
    plt.close(figure)


def test_prism_preview_draws_depth_faces_sides_source_and_normal():
    result, regions = _result_and_regions()
    figure, axis = intensity.plot_region_preview(result, regions, cell_ids=[7])

    # One source boundary, two depth faces, and four connecting sides are lines;
    # the normal is a separate 3-D quiver collection.
    assert len(axis.lines) == 7
    assert len(axis.collections) >= 1
    plotted_z = np.concatenate([np.asarray(line.get_data_3d()[2]) for line in axis.lines])
    assert np.min(plotted_z) == pytest.approx(2.0)
    assert np.max(plotted_z) == pytest.approx(5.0)
    assert "normal points into tissue" in axis.get_title()
    plt.close(figure)


def test_prism_preview_handles_invalid_and_unknown_selection():
    result, regions = _result_and_regions()
    regions.cells[0].status = "invalid_geometry"
    regions.cells[0].reason = "test failure"

    figure, axis = intensity.plot_region_preview(result, regions, cell_ids=[7])
    assert len(axis.texts) == 1
    assert "invalid" in axis.texts[0].get_text()
    plt.close(figure)

    with pytest.raises(ValueError, match="unknown"):
        intensity.plot_region_preview(result, regions, cell_ids=[999])
