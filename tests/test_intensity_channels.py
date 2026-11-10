from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deprojpy import intensity
from deprojpy.intensity._measurement import _resolve_channels


def _regions():
    cell = SimpleNamespace(
        id=17,
        source_label=117,
        boundary=np.asarray([[1, 1, 2], [3, 1, 2], [3, 3, 2], [1, 3, 2]], float),
    )
    result = SimpleNamespace(
        epicells=[cell],
        units="um",
        pixel_size=1.0,
        voxel_depth=1.0,
        z_reflected=False,
        z_reflection_offset=0.0,
    )
    return intensity.prepare_regions(
        result,
        (0, 1),
        tissue_side="increasing_z",
        sampling_spacing=1,
    )


def _stack_czyx():
    z, y, x = np.indices((6, 6, 6), dtype=float)
    first = x + y + z
    return np.stack([first, 2 * first + 3, 5 * first - 1])


@pytest.mark.parametrize("label", ["GFP", 4, np.int64(7)])
def test_zyx_array_uses_one_scalar_channel_name(label):
    measured = intensity.measure(
        _stack_czyx()[0],
        _regions(),
        measurements="mean",
        channel_names=label,
    )

    expected = int(label) if isinstance(label, np.integer) else label
    assert measured.table["channel"].tolist() == [expected]
    assert measured.provenance["channels"] == [expected]


@pytest.mark.parametrize("invalid", [["GFP"], ("GFP",), np.asarray(["GFP"])])
def test_zyx_array_rejects_channel_name_sequences(invalid):
    with pytest.raises(TypeError, match="one scalar"):
        intensity.measure(_stack_czyx()[0], _regions(), channel_names=invalid)


def test_czyx_names_selection_order_and_mapping_equivalence():
    stack = _stack_czyx()
    regions = _regions()
    selected = intensity.measure(
        stack,
        regions,
        measurements=("mean", "std"),
        channel_names=["Junctions", "GFP", "Delta"],
        channels=[2, 0],
    )
    mapped = intensity.measure(
        {"Delta": stack[2], "Junctions": stack[0]},
        regions,
        measurements=("mean", "std"),
    )

    assert selected.table["channel"].tolist() == ["Delta", "Junctions"]
    assert selected.provenance["channels"] == ["Delta", "Junctions"]
    pd.testing.assert_frame_equal(selected.table, mapped.table)


def test_czyx_defaults_to_source_indices_and_accepts_numpy_scalar_selection():
    stack = _stack_czyx()
    regions = _regions()

    all_channels = intensity.measure(stack, regions, measurements="mean")
    one_channel = intensity.measure(
        stack,
        regions,
        measurements="mean",
        channel_names=[10, 20, 30],
        channels=np.int64(1),
    )

    assert all_channels.table["channel"].tolist() == [0, 1, 2]
    assert one_channel.table["channel"].tolist() == [20]


def test_single_channel_czyx_still_requires_a_name_sequence():
    stack = _stack_czyx()[:1]
    with pytest.raises(TypeError, match="ordered sequence"):
        intensity.measure(stack, _regions(), channel_names="only")

    measured = intensity.measure(stack, _regions(), channel_names=["only"])
    assert measured.table["channel"].tolist() == ["only"]


def test_selected_czyx_channels_are_views_even_for_strided_input():
    stack = _stack_czyx()[:, :, ::2, ::-1]
    resolved = _resolve_channels(
        stack,
        channel_names=["first", "second", "third"],
        channels=[2, 0],
    )

    assert [name for name, _ in resolved] == ["third", "first"]
    assert np.shares_memory(resolved[0][1], stack)
    assert np.shares_memory(resolved[1][1], stack)
    np.testing.assert_array_equal(resolved[0][1], stack[2])
    np.testing.assert_array_equal(resolved[1][1], stack[0])


def test_mapping_scalar_and_ordered_key_selection():
    stack = _stack_czyx()
    mapping = {"GFP": stack[0], "Delta": stack[1], 8: stack[2]}
    regions = _regions()

    scalar = intensity.measure(mapping, regions, measurements="mean", channels="Delta")
    ordered = intensity.measure(
        mapping,
        regions,
        measurements="mean",
        channels=[np.int64(8), "GFP"],
    )

    assert scalar.table["channel"].tolist() == ["Delta"]
    assert ordered.table["channel"].tolist() == [8, "GFP"]


def test_empty_mapping_keeps_stable_empty_result():
    measured = intensity.measure({}, _regions())

    assert measured.table.empty
    assert measured.table.columns.tolist()[:3] == ["id", "source_label", "channel"]
    assert measured.provenance["channels"] == []


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"channels": []}, ValueError, "empty selection"),
        ({"channels": [0, 0]}, ValueError, "duplicate"),
        ({"channels": -1}, IndexError, "outside"),
        ({"channels": 3}, IndexError, "outside"),
        ({"channels": True}, TypeError, "booleans"),
        ({"channels": [0, False]}, TypeError, "booleans"),
        ({"channels": [0, "1"]}, TypeError, "only integers"),
        ({"channel_names": ["a", "b"]}, ValueError, "each of 3"),
        ({"channel_names": ["a", "a", "b"]}, ValueError, "duplicate"),
        ({"channel_names": "name"}, TypeError, "ordered sequence"),
        ({"channel_names": ["a", True, "b"]}, TypeError, "booleans"),
    ],
)
def test_czyx_rejects_invalid_names_and_selections(kwargs, error, message):
    with pytest.raises(error, match=message):
        intensity.measure(_stack_czyx(), _regions(), **kwargs)


def test_mapping_rejects_ambiguous_arguments_and_unknown_or_duplicate_keys():
    stack = _stack_czyx()
    mapping = {"GFP": stack[0], "Delta": stack[1]}
    regions = _regions()

    with pytest.raises(ValueError, match="omitted for mapping"):
        intensity.measure(mapping, regions, channel_names=["a", "b"])
    with pytest.raises(KeyError, match="unknown"):
        intensity.measure(mapping, regions, channels="missing")
    with pytest.raises(ValueError, match="duplicate"):
        intensity.measure(mapping, regions, channels=["GFP", "GFP"])
    with pytest.raises(ValueError, match="empty selection"):
        intensity.measure(mapping, regions, channels=[])
    with pytest.raises(TypeError, match="booleans"):
        intensity.measure({True: stack[0]}, regions)


def test_mapping_validates_unselected_channels_and_alignment():
    stack = _stack_czyx()
    regions = _regions()

    with pytest.raises(ValueError, match="same aligned"):
        intensity.measure(
            {"valid": stack[0], "misaligned": np.ones((5, 6, 6))},
            regions,
            channels="valid",
        )
    with pytest.raises(TypeError, match="real numeric"):
        intensity.measure(
            {"valid": stack[0], "invalid": np.full((6, 6, 6), "x")},
            regions,
            channels="valid",
        )


@pytest.mark.parametrize(
    "stack",
    [
        np.ones((6, 6)),
        np.ones((1, 2, 3, 4, 5)),
        np.empty((0, 6, 6)),
        np.empty((0, 2, 6, 6)),
        np.empty((2, 0, 6, 6)),
    ],
)
def test_array_input_rejects_invalid_dimensionality_or_empty_axes(stack):
    with pytest.raises(ValueError, match="stack|nonempty"):
        intensity.measure(stack, _regions())


def test_zyx_rejects_selection_and_invalid_scalar_names():
    stack = _stack_czyx()[0]
    regions = _regions()

    with pytest.raises(ValueError, match="omitted"):
        intensity.measure(stack, regions, channels=0)
    with pytest.raises(TypeError, match="booleans"):
        intensity.measure(stack, regions, channel_names=False)
    with pytest.raises(TypeError, match="strings or integers"):
        intensity.measure(stack, regions, channel_names=1.5)
    with pytest.raises(TypeError, match="unexpected keyword argument 'channel'"):
        intensity.measure(stack, regions, channel="removed")
