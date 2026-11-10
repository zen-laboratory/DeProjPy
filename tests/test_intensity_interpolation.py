import numpy as np
import pytest
from scipy import ndimage

from deprojpy.intensity import _interpolation as interpolation


def _scipy_reference(image, coordinates_zyx):
    return ndimage.map_coordinates(
        image,
        coordinates_zyx.T,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
        output=float,
    )


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
def test_specialized_trilinear_matches_scipy_across_dtypes_and_layouts(dtype, layout):
    base = np.arange(5 * 8 * 9, dtype=np.int64).reshape(5, 8, 9) % 173
    if np.issubdtype(dtype, np.signedinteger):
        base = base - 80
    image = base.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image = image / 7.0
    if layout == "strided":
        image = image[:, ::2, ::2]
    elif layout == "reversed":
        image = image[:, ::-1, ::-1]

    upper = np.asarray(image.shape, dtype=float) - 1.0
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            upper,
            upper / 2.0,
            [upper[0], 0.25 * upper[1], 0.75 * upper[2]],
            [0.25 * upper[0], upper[1], 0.5 * upper[2]],
            [1.125, 1.375, 2.625],
        ]
    )

    expected = _scipy_reference(image, coordinates)
    actual = interpolation.interpolate_values(image, coordinates)

    assert interpolation.can_use_specialized_trilinear(image)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=6e-14)


@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf])
def test_specialized_trilinear_preserves_zero_weight_nonfinite_neighbors(special):
    image = np.arange(27, dtype=float).reshape(3, 3, 3)
    image[0, 0, 1] = special
    image[2, 2, 1] = special
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )

    expected = _scipy_reference(image, coordinates)
    actual = interpolation.interpolate_values(image, coordinates)

    np.testing.assert_array_equal(actual, expected)


def test_prepared_multichannel_coordinates_match_direct_sampling():
    first = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    second = first[:, ::-1, :] * 3
    coordinates = np.asarray(
        [[0.25, 0.5, 0.75], [2.5, 3.0, 4.125], [3.0, 4.0, 5.0]]
    )
    prepared = interpolation.prepare_trilinear_coordinates(
        coordinates,
        first.shape,
    )

    for image in (first, second):
        direct = interpolation.interpolate_values(image, coordinates)
        reused = interpolation.interpolate_values(
            image,
            coordinates,
            prepared=prepared,
        )
        np.testing.assert_array_equal(reused, direct)


@pytest.mark.parametrize("shape", [(1, 3, 3), (3, 1, 3), (3, 3, 1), (1, 1, 1)])
def test_singleton_axes_use_exact_scipy_fallback(shape):
    image = np.arange(np.prod(shape), dtype=float).reshape(shape)
    image.flat[0] = np.inf
    coordinates = ((np.asarray(shape, dtype=float) - 1.0) / 2.0)[None, :]

    expected = _scipy_reference(image, coordinates)
    actual = interpolation.interpolate_values(image, coordinates)

    assert not interpolation.can_use_specialized_trilinear(image)
    np.testing.assert_array_equal(actual, expected)
