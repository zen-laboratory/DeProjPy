from types import SimpleNamespace

import numpy as np
import pytest

from deprojpy import intensity


def _result(boundary, *, reflected=False, offset=0.0, cell_id=7):
    cell = SimpleNamespace(id=cell_id, source_label=42, boundary=np.asarray(boundary, float))
    return SimpleNamespace(epicells=[cell], units="um", pixel_size=1.0, voxel_depth=1.0,
                           z_reflected=reflected, z_reflection_offset=offset)


def test_constant_stack_volume_and_zero_is_valid():
    result = _result([[2, 2, 3], [4, 2, 3], [4, 4, 3], [2, 4, 3]])
    regions = intensity.prepare_regions(result, (0, 2), tissue_side="increasing_z")
    measured = intensity.measure(np.zeros((8, 8, 8)), regions, measurements="all").table.iloc[0]
    assert measured["status"] == "ok"
    assert measured["mean"] == 0
    assert measured["requested_volume"] == pytest.approx(8)
    assert measured["sampled_volume"] == pytest.approx(8)
    assert measured["coverage_fraction"] == pytest.approx(1)
    assert measured["integrated_intensity"] == 0


def test_affine_stack_and_tilted_normal_are_interpolated():
    z, y, x = np.indices((12, 12, 12), dtype=float)
    stack = 2 * x + 3 * y + 5 * z + 1
    boundary = [[2, 2, 3], [6, 2, 5], [6, 6, 5], [2, 6, 3]]
    result = _result(boundary)
    regions = intensity.prepare_regions(result, (-.5, 1.5), tissue_side="increasing_z",
                                        sampling_spacing=.4)
    row = intensity.measure(stack, regions, measurements=("mean", "integrated_intensity",
                                                           "finite_volume")).table.iloc[0]
    assert row["status"] == "ok"
    assert row["integrated_intensity"] == pytest.approx(row["mean"] * row["finite_volume"])


def test_reflection_orients_normal_in_original_stack_frame():
    result = _result([[1, 1, 7], [3, 1, 7], [3, 3, 7], [1, 3, 7]],
                     reflected=True, offset=10)
    inc = intensity.prepare_regions(result, (0, 1), tissue_side="increasing_z")
    dec = intensity.prepare_regions(result, (0, 1), tissue_side="decreasing_z")
    assert inc.cells[0].normal_xyz[2] < 0
    assert dec.cells[0].normal_xyz[2] > 0


def test_slice_units_fractional_depth_and_negative_interval():
    result = _result([[1, 1, 2], [3, 1, 2], [3, 3, 2], [1, 3, 2]])
    regions = intensity.prepare_regions(result, (-.5, 2.25), depth_units="slices",
                                        tissue_side="increasing_z", spacing_zyx=(2, 1, 1))
    assert regions.converted_depth_range == (-1, 4.5)


def test_clipping_nonfinite_multiple_channels_and_sparse_identifier():
    result = _result([[1, 1, 3], [3, 1, 3], [3, 3, 3], [1, 3, 3]], cell_id=91)
    regions = intensity.prepare_regions(result, (-5, 1), tissue_side="increasing_z")
    a = np.ones((6, 6, 6))
    a[2, 2, 2] = np.nan
    measured = intensity.measure({"a": a, "b": np.full_like(a, 2)}, regions,
                                 measurements=("mean", "sampled_volume", "finite_volume"))
    assert set(measured.table.channel) == {"a", "b"}
    assert set(measured.table.id) == {91}
    assert measured.table.clipped.all()
    assert measured.table.loc[measured.table.channel == "b", "mean"].iloc[0] == 2


def test_legacy_mapping_and_validation_errors():
    result = _result([[0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1]])
    result.z_reflected = None
    with pytest.raises(ValueError, match="metadata"):
        intensity.prepare_regions(result, (0, 1), tissue_side="increasing_z")
    with pytest.raises(ValueError, match="start < end"):
        intensity.prepare_regions(result, (1, 1), tissue_side="increasing_z", z_reflected=False)


def test_measurement_roundtrip(tmp_path):
    result = _result([[1, 1, 2], [3, 1, 2], [3, 3, 2], [1, 3, 2]])
    regions = intensity.prepare_regions(result, (0, 1), tissue_side="increasing_z")
    original = intensity.measure(np.ones((5, 5, 5)), regions)
    path = original.save(tmp_path / "measurements")
    loaded = intensity.MeasurementResult.load(path)
    assert loaded.provenance == original.provenance
    assert loaded.table["mean"].iloc[0] == 1
