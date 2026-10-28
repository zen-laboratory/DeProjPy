from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import deprojpy as dp
from deprojpy.models import DeprojResult, Epicell
from deprojpy.surface_distance import SurfaceDistanceCalculator


def _manual_result() -> DeprojResult:
    graph = nx.Graph()
    graph.add_node(
        11,
        id=11,
        centroid=np.array([0.0, 0.0, 1.0]),
        labels=frozenset({101, 202}),
        pixel_coords=np.array([[0, 0], [0, 1]], dtype=np.int64),
    )
    graph.add_node(
        22,
        id=22,
        centroid=np.array([3.0, 2.0, 1.5]),
        labels=frozenset({202, 303}),
        pixel_coords=np.array([[2, 3]], dtype=np.int64),
    )
    graph.add_edge(11, 22)
    cells = [
        Epicell(
            id=101,
            source_label=101,
            boundary=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.1],
                    [2.0, 1.0, 1.2],
                    [0.0, 1.0, 1.1],
                ]
            ),
            center=np.array([1.0, 0.5, 1.1]),
            junction_ids=np.array([11, 22], dtype=np.int64),
            n_neighbors=2,
            area=2.2,
            perimeter=6.1,
            euler_angles=np.array([0.1, 0.2, 0.3]),
            curvatures=np.array([0.01, 0.02, 0.03, 0.04]),
            ellipse_fit=np.array([1.0, 0.5, 1.1, 2.0, 1.0, 0.2]),
            eccentricity=0.5,
            proj_direction=0.25,
            uncorrected_area=2.0,
            uncorrected_perimeter=6.0,
            area_error=0.2,
            perimeter_error=0.1,
            surface_centroid=np.array([1.1, 0.6, 1.15]),
        ),
        Epicell(
            id=202,
            source_label=None,
            boundary=np.array(
                [
                    [2.0, 0.0, 1.1],
                    [4.0, 0.0, 1.2],
                    [4.0, 2.0, 1.4],
                    [3.0, 3.0, 1.6],
                    [2.0, 2.0, 1.3],
                ]
            ),
            center=np.array([3.0, 1.4, 1.32]),
            junction_ids=np.array([22], dtype=np.int64),
            n_neighbors=1,
            area=4.4,
            perimeter=8.2,
            euler_angles=np.array([0.4, 0.5, 0.6]),
            curvatures=np.array([0.05, 0.06, 0.07, 0.08]),
            ellipse_fit=np.array([3.0, 1.4, 1.32, 2.5, 1.5, 0.4]),
            eccentricity=0.6,
            proj_direction=0.35,
            uncorrected_area=4.0,
            uncorrected_perimeter=8.0,
            area_error=0.4,
            perimeter_error=0.2,
        ),
    ]
    return DeprojResult(
        cells,
        graph,
        units="µm",
        pixel_size=0.5,
        voxel_depth=2.0,
        source_shape=(5, 6),
        prepared_heightmap=np.arange(30, dtype=float).reshape(5, 6) / 10.0,
    )


def _assert_cells_equal(original: Epicell, loaded: Epicell) -> None:
    assert loaded.id == original.id
    assert loaded.source_label == original.source_label
    np.testing.assert_allclose(loaded.boundary, original.boundary)
    np.testing.assert_allclose(loaded.center, original.center)
    np.testing.assert_allclose(loaded.surface_centroid, original.surface_centroid, equal_nan=True)
    np.testing.assert_array_equal(loaded.junction_ids, original.junction_ids)
    assert loaded.n_neighbors == original.n_neighbors
    assert loaded.area == original.area
    assert loaded.perimeter == original.perimeter
    np.testing.assert_allclose(loaded.euler_angles, original.euler_angles)
    np.testing.assert_allclose(loaded.curvatures, original.curvatures)
    np.testing.assert_allclose(loaded.ellipse_fit, original.ellipse_fit)
    assert loaded.eccentricity == original.eccentricity
    assert loaded.proj_direction == original.proj_direction
    assert loaded.uncorrected_area == original.uncorrected_area
    assert loaded.uncorrected_perimeter == original.uncorrected_perimeter
    assert loaded.area_error == original.area_error
    assert loaded.perimeter_error == original.perimeter_error


def _assert_graph_equal(original: nx.Graph, loaded: nx.Graph) -> None:
    assert set(loaded.nodes) == set(original.nodes)
    assert {tuple(sorted(edge)) for edge in loaded.edges} == {
        tuple(sorted(edge)) for edge in original.edges
    }
    for node_id, original_data in original.nodes(data=True):
        loaded_data = loaded.nodes[node_id]
        assert loaded_data["id"] == original_data["id"]
        np.testing.assert_allclose(loaded_data["centroid"], original_data["centroid"])
        assert loaded_data.get("labels") == original_data.get("labels")
        if "pixel_coords" in original_data:
            np.testing.assert_array_equal(
                loaded_data["pixel_coords"],
                original_data["pixel_coords"],
            )
        else:
            assert "pixel_coords" not in loaded_data


def _rewrite_npz(path, **updates) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(updates)
    np.savez(path, **arrays)


def test_result_save_load_round_trip_preserves_global_cells_graph_and_dataframe(tmp_path):
    result = _manual_result()

    path = result.save(tmp_path / "manual")
    loaded = DeprojResult.load(path)

    assert path == tmp_path / "manual.npz"
    assert loaded.units == result.units
    assert loaded.pixel_size == result.pixel_size
    assert loaded.voxel_depth == result.voxel_depth
    assert loaded.source_shape == result.source_shape
    np.testing.assert_allclose(loaded.prepared_heightmap, result.prepared_heightmap)
    assert len(loaded.epicells) == len(result.epicells)
    for original_cell, loaded_cell in zip(result.epicells, loaded.epicells, strict=True):
        _assert_cells_equal(original_cell, loaded_cell)
    _assert_graph_equal(result.junction_graph, loaded.junction_graph)
    pd.testing.assert_frame_equal(result.to_dataframe(), loaded.to_dataframe())


def test_ragged_boundaries_and_junction_ids_round_trip_exactly(tmp_path):
    result = _manual_result()
    loaded = DeprojResult.load(result.save(tmp_path / "ragged.npz"))

    assert [len(cell.boundary) for cell in loaded.epicells] == [4, 5]
    assert [cell.junction_ids.tolist() for cell in loaded.epicells] == [[11, 22], [22]]


def test_surface_distance_calculators_work_after_load(tmp_path):
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[4:11, 4:11] = 255
    y, x = np.indices(mask.shape)
    heightmap = 1.0 + 0.1 * x + 0.05 * y
    result = dp.from_heightmap(
        mask,
        heightmap,
        pixel_size=0.4,
        voxel_depth=1.2,
        units="µm",
        inpaint_zeros=False,
        prune_zeros=False,
    )
    loaded = DeprojResult.load(result.save(tmp_path / "distance.npz"))

    original_height_calc = SurfaceDistanceCalculator.from_result(result)
    loaded_height_calc = SurfaceDistanceCalculator.from_result(loaded)
    points = np.array([[2.0, 2.0], [7.0, 8.0], [10.0, 5.0]])
    np.testing.assert_allclose(
        loaded_height_calc.sample_height(points),
        original_height_calc.sample_height(points),
    )
    assert np.isclose(
        loaded_height_calc.straight_distance(points[0], points[1]),
        original_height_calc.straight_distance(points[0], points[1]),
    )

    original_boundary_calc = SurfaceDistanceCalculator.from_cell_boundaries(result)
    loaded_boundary_calc = SurfaceDistanceCalculator.from_cell_boundaries(loaded)
    np.testing.assert_allclose(
        loaded_boundary_calc.sample_height(points),
        original_boundary_calc.sample_height(points),
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_save_supports_compressed_and_uncompressed(tmp_path, compressed):
    result = _manual_result()
    path = result.save(tmp_path / f"compressed_{compressed}.npz", compressed=compressed)
    assert path.is_file()
    loaded = DeprojResult.load(path)
    pd.testing.assert_frame_equal(result.to_dataframe(), loaded.to_dataframe())


def test_save_overwrite_policy(tmp_path):
    result = _manual_result()
    path = result.save(tmp_path / "result.npz")

    with pytest.raises(FileExistsError):
        result.save(path)

    modified = _manual_result()
    modified.epicells[0].area = 99.0
    modified.save(path, overwrite=True)
    assert DeprojResult.load(path).epicells[0].area == 99.0


def test_load_rejects_unsupported_future_version(tmp_path):
    path = _manual_result().save(tmp_path / "future.npz")
    _rewrite_npz(path, format_version=np.asarray("999.0"))

    with pytest.raises(ValueError, match="999.0.*supported versions"):
        DeprojResult.load(path)


def test_load_rejects_missing_version(tmp_path):
    path = _manual_result().save(tmp_path / "missing_version.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "format_version"}
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="missing required.*format_version"):
        DeprojResult.load(path)


def test_epicell_surface_centroid_defaults_are_independent_nan_arrays():
    result = _manual_result()
    first, second = result.epicells
    assert np.isnan(second.surface_centroid).all()
    second.surface_centroid[0] = 3.0
    assert first.surface_centroid[0] == 1.1


def test_load_legacy_1_0_archive_uses_unavailable_centroids_and_resaves(tmp_path):
    path = _manual_result().save(tmp_path / "legacy.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: data[key]
            for key in data.files
            if key != "cell_surface_centroids"
        }
    arrays["format_version"] = np.asarray("1.0")
    np.savez(path, **arrays)

    loaded = DeprojResult.load(path)
    assert all(np.isnan(cell.surface_centroid).all() for cell in loaded.epicells)
    new_path = loaded.save(tmp_path / "resaved.npz")
    reloaded = DeprojResult.load(new_path)
    assert all(np.isnan(cell.surface_centroid).all() for cell in reloaded.epicells)


def test_load_1_1_requires_and_validates_surface_centroids(tmp_path):
    path = _manual_result().save(tmp_path / "missing_centroids.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "cell_surface_centroids"}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="cell_surface_centroids"):
        DeprojResult.load(path)

    path = _manual_result().save(tmp_path / "partial_centroids.npz")
    centroids = np.array([[1.0, np.nan, 2.0], [np.inf, np.inf, np.inf]])
    _rewrite_npz(path, cell_surface_centroids=centroids)
    with pytest.raises(ValueError, match="entirely finite or entirely NaN"):
        DeprojResult.load(path)


def test_load_rejects_invalid_junction_edge_endpoint(tmp_path):
    path = _manual_result().save(tmp_path / "bad_edge.npz")
    _rewrite_npz(path, junction_edge_targets=np.asarray([999]))

    with pytest.raises(ValueError, match="edge endpoint"):
        DeprojResult.load(path)
