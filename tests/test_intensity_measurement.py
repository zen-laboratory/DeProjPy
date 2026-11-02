from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deprojpy import intensity
from deprojpy.intensity import _statistics as statistics
from deprojpy.intensity._sampling import (
    build_sampling_plans,
    classify_region_coverage,
    iter_sample_blocks,
)


def _result(boundaries, *, ids=None):
    if ids is None:
        ids = range(1, len(boundaries) + 1)
    cells = [
        SimpleNamespace(
            id=cell_id,
            source_label=100 + cell_id,
            boundary=np.asarray(boundary, dtype=float),
        )
        for cell_id, boundary in zip(ids, boundaries, strict=True)
    ]
    return SimpleNamespace(
        epicells=cells,
        units="um",
        pixel_size=1.0,
        voxel_depth=1.0,
        z_reflected=False,
        z_reflection_offset=0.0,
    )


def _two_cell_regions(sampling_spacing=0.5):
    result = _result(
        [
            [[2, 2, 3], [4, 2, 3], [4, 4, 3], [2, 4, 3]],
            [[6, 2, 4], [9, 2, 4], [9, 5, 4], [6, 5, 4]],
        ],
        ids=[7, 91],
    )
    return intensity.prepare_regions(
        result,
        (0, 2),
        tissue_side="increasing_z",
        sampling_spacing=sampling_spacing,
    )


def test_sample_blocks_respect_budget_and_combine_cells():
    regions = _two_cell_regions(sampling_spacing=0.25)
    shape = (12, 12, 12)
    coverages = [
        classify_region_coverage(region, regions, shape)
        for region in regions.cells
    ]
    plans = build_sampling_plans(regions, coverages)
    blocks = list(iter_sample_blocks(regions, plans, max_samples=31))

    assert blocks
    assert max(plan.sample_count for plan in plans) > 31
    assert all(len(block.geometric_xyz) <= 31 for block in blocks)
    assert all(len(block.volume_weights) <= 31 for block in blocks)
    assert all(len(block.region_indices) <= 31 for block in blocks)
    assert any(len(np.unique(block.region_indices)) > 1 for block in blocks)
    assert sum(len(block.geometric_xyz) for block in blocks) == sum(
        plan.sample_count for plan in plans
    )


def test_measurement_agrees_across_batch_sizes_and_channels():
    regions = _two_cell_regions(sampling_spacing=0.25)
    z, y, x = np.indices((12, 12, 12), dtype=float)
    first = x + 2 * y + 3 * z
    channels = {"first": first, "second": 2 * first + 5}

    small = intensity.measure(channels, regions, measurements="all", max_batch_samples=17)
    large = intensity.measure(channels, regions, measurements="all", max_batch_samples=100_000)
    columns = [name for name in small.table.columns if name != "mode_bin_edges"]
    pd.testing.assert_frame_equal(
        small.table[columns],
        large.table[columns],
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )
    assert small.table["mode_bin_edges"].tolist() == large.table["mode_bin_edges"].tolist()

    single = intensity.measure(first, regions, measurements="mean", channel="first")
    mapped = intensity.measure({"first": first}, regions, measurements="mean")
    pd.testing.assert_frame_equal(single.table, mapped.table)


def test_streaming_only_selection_does_not_sort_or_buffer(monkeypatch):
    regions = _two_cell_regions(sampling_spacing=0.25)

    def forbidden(*args, **kwargs):
        raise AssertionError("quantile code must not run")

    monkeypatch.setattr(statistics, "weighted_quantiles", forbidden)
    measured = intensity.measure(
        np.ones((12, 12, 12)),
        regions,
        measurements=("mean", "integrated_intensity"),
        max_buffered_samples_per_cell=1,
    )

    assert "median" not in measured.table
    assert "std" not in measured.table
    assert "finite_volume" in measured.table
    np.testing.assert_allclose(measured.table["mean"], 1)


def test_exact_quantile_buffer_has_separate_limit():
    regions = _two_cell_regions(sampling_spacing=0.25)
    with pytest.raises(MemoryError, match="max_buffered_samples_per_cell"):
        intensity.measure(
            np.ones((12, 12, 12)),
            regions,
            measurements="median",
            max_buffered_samples_per_cell=2,
        )


def test_weighted_histogram_mode_uses_density_and_volume_weights():
    mode, edges = statistics.weighted_histogram_mode(
        np.asarray([0.25, 1.25]),
        np.asarray([1.0, 5.0]),
        np.asarray([0.0, 1.0, 2.0]),
    )
    assert mode == 1.5
    np.testing.assert_array_equal(edges, [0, 1, 2])

    unequal_width_mode, _ = statistics.weighted_histogram_mode(
        np.asarray([0.5, 2.0]),
        np.asarray([2.0, 3.0]),
        np.asarray([0.0, 1.0, 4.0]),
    )
    assert unequal_width_mode == 0.5


def test_streaming_weighted_moments_are_stable_across_updates():
    requirements = statistics.requirements_for(
        ("mean", "std", "integrated_intensity", "minimum", "maximum")
    )
    reducer = statistics.StreamingReducer(requirements, max_buffered_samples=1)
    reducer.update(np.asarray([1.0]), np.asarray([1.0]))
    reducer.update(np.asarray([3.0]), np.asarray([3.0]))

    assert reducer.finite_volume == 4
    assert reducer.mean == pytest.approx(2.5)
    assert reducer.weighted_m2 / reducer.finite_volume == pytest.approx(0.75)
    assert reducer.weighted_sum == pytest.approx(10)
    assert reducer.minimum == 1
    assert reducer.maximum == 3


def test_mode_constant_tie_nonfinite_and_validation():
    constant_mode, _ = statistics.weighted_histogram_mode(
        np.zeros(4),
        np.ones(4),
        3,
    )
    assert constant_mode == 0

    tied_mode, _ = statistics.weighted_histogram_mode(
        np.asarray([0.25, 1.25]),
        np.ones(2),
        np.asarray([0.0, 1.0, 2.0]),
    )
    assert tied_mode == 0.5

    for invalid in (0, "fd", [0, 1, 1], [0, np.nan, 2]):
        with pytest.raises(ValueError):
            statistics.validate_mode_bins(invalid)

    regions = _two_cell_regions()
    stack = np.full((12, 12, 12), np.nan)
    measured = intensity.measure(
        stack,
        regions,
        measurements=("mode", "integrated_intensity"),
        mode_bins=[0, 1, 2],
    )
    assert measured.table["mode"].isna().all()
    assert measured.table["integrated_intensity"].isna().all()
    assert (measured.table["finite_volume"] == 0).all()
    assert (measured.table["sampled_volume"] > 0).all()
    assert set(measured.table["status"]) == {"nonfinite_omitted"}


def test_geometric_disjointness_and_missed_sliver_are_distinct():
    result = _result(
        [
            [[-5, 1, 2], [-3, 1, 2], [-3, 3, 2], [-5, 3, 2]],
            [[-0.99, 1, 2], [0.01, 1, 2], [0.01, 3, 2], [-0.99, 3, 2]],
        ],
        ids=[10, 20],
    )
    regions = intensity.prepare_regions(
        result,
        (0, 1),
        tissue_side="increasing_z",
        sampling_spacing=1,
    )
    table = intensity.measure(
        np.ones((6, 6, 6)),
        regions,
        measurements=("mean", "integrated_intensity"),
    ).table.set_index("id")

    assert table.loc[10, "status"] == "outside_stack"
    assert np.isnan(table.loc[10, "integrated_intensity"])
    assert table.loc[20, "status"] != "outside_stack"
    assert bool(table.loc[20, "clipped"])


def test_partial_clipping_reports_numerical_coverage_volume():
    result = _result(
        [[[-0.2, 1, 2], [1.8, 1, 2], [1.8, 3, 2], [-0.2, 3, 2]]],
        ids=[55],
    )
    regions = intensity.prepare_regions(
        result,
        (0, 1),
        tissue_side="increasing_z",
        sampling_spacing=0.5,
    )
    row = intensity.measure(
        np.ones((6, 6, 6)),
        regions,
        measurements=("mean", "integrated_intensity"),
    ).table.iloc[0]

    assert row["status"] == "partial_coverage"
    assert 0 < row["sampled_volume"] < row["requested_volume"]
    assert row["coverage_fraction"] == pytest.approx(
        row["sampled_volume"] / row["requested_volume"]
    )
    assert row["mean"] == 1
    assert row["integrated_intensity"] == pytest.approx(row["finite_volume"])


def test_mode_edges_and_sparse_identifiers_are_retained_per_channel(tmp_path):
    regions = _two_cell_regions()
    measured = intensity.measure(
        {"zero": np.zeros((12, 12, 12)), "one": np.ones((12, 12, 12))},
        regions,
        measurements="mode",
        mode_bins=[-0.5, 0.5, 1.5],
    )

    assert set(measured.table["id"]) == {7, 91}
    assert set(measured.table["channel"]) == {"zero", "one"}
    assert measured.table["mode_bin_edges"].apply(lambda value: value == [-0.5, 0.5, 1.5]).all()
    assert measured.provenance["mode"]["bins"] == [-0.5, 0.5, 1.5]

    loaded = intensity.MeasurementResult.load(
        measured.save(tmp_path / "mode-measurements.npz")
    )
    assert loaded.provenance["mode"] == measured.provenance["mode"]
    assert loaded.table["mode_bin_edges"].apply(
        lambda value: value == [-0.5, 0.5, 1.5]
    ).all()
