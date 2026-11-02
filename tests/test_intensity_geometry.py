from types import SimpleNamespace

import numpy as np
import pytest

from deprojpy import intensity
from deprojpy.heightmap import _prepare_heightmap_with_metadata
from deprojpy.intensity._coordinates import geometric_xyz_to_stack_index_zyx
from deprojpy.intensity._quadrature import (
    build_lateral_quadrature,
    clean_polygon,
    depth_midpoint_quadrature,
    materialize_region_samples,
    signed_polygon_area,
    subdivide_triangles,
    triangulate_polygon,
)


def _result(boundary, *, reflected=False, offset=0.0):
    cell = SimpleNamespace(
        id=31,
        source_label=103,
        boundary=np.asarray(boundary, dtype=float),
    )
    return SimpleNamespace(
        epicells=[cell],
        units="um",
        pixel_size=1.0,
        voxel_depth=2.0,
        z_reflected=reflected,
        z_reflection_offset=offset,
    )


def _polygon_centroid(polygon):
    polygon = np.asarray(polygon, dtype=float)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    area = 0.5 * cross.sum()
    centroid = np.sum((polygon + following) * cross[:, None], axis=0) / (6 * area)
    return centroid


def test_triangle_centroid_rule_is_affine_exact_at_coarse_spacing():
    triangle = np.asarray([[2.0, 2.0], [5.0, 2.0], [2.0, 5.0]])
    points_uv, area_weights, _ = build_lateral_quadrature(triangle, sampling_spacing=10)

    assert area_weights.sum() == pytest.approx(4.5)
    assert np.average(points_uv[:, 0], weights=area_weights) == pytest.approx(3.0)
    assert np.average(points_uv[:, 1], weights=area_weights) == pytest.approx(3.0)


def test_long_thin_polygon_is_subdivided_by_extent_not_area():
    slender = np.asarray([[0.0, 0.0], [100.0, 0.0], [100.0, 0.01], [0.0, 0.01]])
    points_uv, area_weights, _ = build_lateral_quadrature(slender, sampling_spacing=1)

    assert len(points_uv) > 100
    assert area_weights.sum() == pytest.approx(1.0)

    subtriangles = subdivide_triangles(triangulate_polygon(slender), 1)
    edges = np.roll(subtriangles, -1, axis=1) - subtriangles
    assert np.linalg.norm(edges, axis=2).max() <= 1 + 1e-12


def test_nonconvex_polygon_recovers_area_and_first_moment():
    polygon = np.asarray(
        [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]],
        dtype=float,
    )
    points_uv, area_weights, cleaned = build_lateral_quadrature(polygon, 0.7)

    assert area_weights.sum() == pytest.approx(abs(signed_polygon_area(cleaned)))
    np.testing.assert_allclose(
        np.average(points_uv, axis=0, weights=area_weights),
        _polygon_centroid(polygon),
        atol=1e-14,
    )


def test_tiny_polygon_is_not_inflated_or_dropped():
    side = 1e-9
    tiny = np.asarray([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)
    points_uv, area_weights, _ = build_lateral_quadrature(tiny, sampling_spacing=1)

    assert len(points_uv) == 2
    assert area_weights.sum() == pytest.approx(side**2)


def test_polygon_cleanup_handles_closing_duplicates_collinear_and_orientation():
    clockwise_closed = np.asarray(
        [[0, 0], [0, 1], [1, 1], [2, 1], [2, 0], [0, 0], [0, 0]],
        dtype=float,
    )
    cleaned = clean_polygon(clockwise_closed)

    assert cleaned.shape == (4, 2)
    assert signed_polygon_area(cleaned) == pytest.approx(2)
    with pytest.raises(ValueError, match="zero|degenerate"):
        clean_polygon(np.asarray([[0, 0], [1, 0], [2, 0]], dtype=float))
    with pytest.raises(ValueError, match="self-intersecting"):
        clean_polygon(np.asarray([[0, 0], [2, 2], [0, 2], [2, 0]], dtype=float))


def test_depth_midpoints_span_exact_signed_interval_and_integrate_affine_depth():
    points, widths = depth_midpoint_quadrature(-1.25, 4.0, 2.0)

    assert widths.sum() == pytest.approx(5.25)
    assert np.sum(points * widths) == pytest.approx(
        0.5 * (4.0**2 - (-1.25) ** 2)
    )


def test_prism_quadrature_is_affine_exact_for_tilted_cell():
    boundary = [[2, 2, 3], [6, 2, 5], [2, 6, 3]]
    result = _result(boundary)
    regions = intensity.prepare_regions(
        result,
        (-0.5, 1.5),
        tissue_side="increasing_z",
        sampling_spacing=20,
        spacing_zyx=(1, 1, 1),
    )
    region = regions.cells[0]
    points_xyz, volume_weights = materialize_region_samples(region, 20)

    polygon_centroid_uv = _polygon_centroid(region.polygon_uv)
    expected_centroid_xyz = (
        region.plane_origin_xyz
        + polygon_centroid_uv @ region.plane_basis_xyz.T
        + 0.5 * (region.start_depth + region.end_depth) * region.normal_xyz
    )
    affine_values = 2 * points_xyz[:, 0] - points_xyz[:, 1] + 4 * points_xyz[:, 2] + 7
    expected_mean = (
        2 * expected_centroid_xyz[0]
        - expected_centroid_xyz[1]
        + 4 * expected_centroid_xyz[2]
        + 7
    )

    assert volume_weights.sum() == pytest.approx(
        region.footprint_area * (region.end_depth - region.start_depth)
    )
    assert np.average(affine_values, weights=volume_weights) == pytest.approx(expected_mean)

    z_index, y_index, x_index = np.indices((12, 12, 12), dtype=float)
    affine_stack = 2 * x_index - y_index + 4 * z_index + 7
    measured = intensity.measure(
        affine_stack,
        regions,
        measurements=("mean", "integrated_intensity"),
    ).table.iloc[0]
    assert measured["mean"] == pytest.approx(expected_mean)
    assert measured["integrated_intensity"] == pytest.approx(
        expected_mean * volume_weights.sum()
    )


def test_anisotropic_crop_and_reflection_coordinate_conversion():
    result = _result(
        [[1, 1, 7], [3, 1, 7], [3, 3, 7], [1, 3, 7]],
        reflected=True,
        offset=10,
    )
    calibration = intensity.StackCalibration(
        spacing_zyx=(2.0, 0.5, 0.25),
        origin_zyx=(2.0, 1.0, 0.5),
    )
    regions = intensity.prepare_regions(
        result,
        (0, 1),
        tissue_side="increasing_z",
        calibration=calibration,
    )
    converted = geometric_xyz_to_stack_index_zyx(
        np.asarray([[1.0, 2.0, 7.0]]),
        regions,
    )
    np.testing.assert_allclose(converted, [[0.5, 2.0, 2.0]])
    assert regions.cells[0].normal_xyz[2] < 0


def test_ndarray_spacing_and_conflicting_calibration_are_validated():
    result = _result([[0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]])
    regions = intensity.prepare_regions(
        result,
        (0, 1),
        tissue_side="increasing_z",
        spacing_zyx=np.asarray([2.0, 1.0, 1.0]),
    )
    assert regions.calibration.spacing_zyx == (2.0, 1.0, 1.0)

    calibration = intensity.StackCalibration((2.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="cannot be combined"):
        intensity.prepare_regions(
            result,
            (0, 1),
            tissue_side="increasing_z",
            calibration=calibration,
            origin_zyx=(1, 0, 0),
        )


def test_heightmap_reflection_offset_comes_from_same_preparation():
    heightmap = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    prepared, offset = _prepare_heightmap_with_metadata(
        heightmap,
        voxel_depth=2.5,
        smooth_scale=-1,
        invert_z=True,
        inpaint_zeros=False,
        prune_zeros=False,
    )

    assert offset == pytest.approx(12.5)
    np.testing.assert_allclose(prepared, (5.0 - heightmap) * 2.5)
