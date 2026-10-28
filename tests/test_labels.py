from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

import deprojpy as dp
from deprojpy.labels import labels_to_objects
from deprojpy.plotting import plot_label_objects, save_plots


def quadrant_labels() -> np.ndarray:
    return np.array(
        [
            [5, 5, 5, 10, 10, 10],
            [5, 5, 5, 10, 10, 10],
            [5, 5, 5, 10, 10, 10],
            [20, 20, 20, 30, 30, 30],
            [20, 20, 20, 30, 30, 30],
            [20, 20, 20, 30, 30, 30],
        ],
        dtype=np.int64,
    )


def test_labels_to_objects_preserves_nonconsecutive_source_labels_and_high_valence():
    objects, graph = labels_to_objects(quadrant_labels(), drop_border_cells=False)
    assert {obj.source_label for obj in objects} == {5, 10, 20, 30}
    assert graph.number_of_nodes() >= 1
    assert any(data["labels"] == frozenset({5, 10, 20, 30}) for _, data in graph.nodes(data=True))
    for _, data in graph.nodes(data=True):
        assert {"centroid", "labels", "pixel_coords"} <= set(data)
        assert len(data["centroid"]) == 2
    assert all(obj.boundary.shape[1] == 2 for obj in objects)


def test_labels_to_objects_drop_border_cells_can_remove_all_cells():
    with pytest.raises(ValueError, match="labeled-image parsing found no retained cells"):
        labels_to_objects(quadrant_labels(), drop_border_cells=True)


def test_from_labels_returns_result_with_source_labels():
    labels = quadrant_labels()
    heightmap = np.ones(labels.shape, dtype=float)
    result = dp.from_labels(
        labels,
        heightmap,
        inpaint_zeros=False,
        prune_zeros=False,
        drop_border_cells=False,
    )
    frame = result.to_dataframe()
    assert set(frame["source_label"]) == {5, 10, 20, 30}
    assert frame["id"].tolist() == frame["source_label"].tolist()
    assert np.isfinite(frame["area"]).all()
    assert (frame["area"] > 0).all()
    assert np.isfinite(
        frame[["surface_centroid_x", "surface_centroid_y", "surface_centroid_z"]]
    ).all().all()
    for cell in result.epicells:
        rows, cols = np.where(labels == cell.source_label)
        np.testing.assert_allclose(cell.surface_centroid[:2], [cols.mean(), rows.mean()])


def test_from_labels_rejects_heightmap_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        dp.from_labels(quadrant_labels(), np.ones((5, 6)))


def test_label_pair_loader_and_plotting(tmp_path):
    labels = quadrant_labels()
    heightmap = np.ones(labels.shape, dtype=np.float32)
    labels_path = tmp_path / "labels.tif"
    heightmap_path = tmp_path / "heightmap.tif"
    tifffile.imwrite(labels_path, labels.astype(np.uint16))
    tifffile.imwrite(heightmap_path, heightmap)
    loaded_labels, loaded_heightmap = dp.load_label_heightmap_pair(
        labels_path,
        heightmap_path,
        preprocess_labels=False,
    )
    assert np.array_equal(loaded_labels, labels)
    assert loaded_heightmap.shape == labels.shape

    result = dp.from_labels(
        loaded_labels,
        loaded_heightmap,
        inpaint_zeros=False,
        prune_zeros=False,
        drop_border_cells=False,
    )
    fig, ax = plot_label_objects(loaded_labels, result)
    assert ax.figure is fig
    plt.close(fig)
    paths = save_plots(
        tmp_path / "plots",
        None,
        loaded_heightmap,
        result,
        labels=loaded_labels,
        features=("area",),
    )
    assert "label_objects.png" in {path.name for path in paths}
