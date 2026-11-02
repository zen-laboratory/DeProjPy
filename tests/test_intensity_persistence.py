import json

import numpy as np
import pandas as pd
import pytest

from deprojpy.intensity import MeasurementResult


def _measurement_result():
    table = pd.DataFrame(
        {
            "id": pd.Series([7, 91], dtype="int64"),
            "source_label": pd.Series([None, 12], dtype="object"),
            "channel": pd.Series(["α-channel", 3], dtype="object"),
            "clipped": pd.Series([False, True], dtype="bool"),
            "mean": pd.Series([np.nextafter(1.0, 2.0), np.nan], dtype="float64"),
            "mode_bin_edges": pd.Series([[0.0, 1.0], None], dtype="object"),
        }
    )
    provenance = {
        "units": "µm",
        "measurements": ["mean", "mode"],
        "mode": {"bins": [0.0, 1.0]},
        "nonfinite": np.nan,
    }
    return MeasurementResult(table=table, provenance=provenance)


def test_measurement_round_trip_preserves_precision_types_and_unicode(tmp_path):
    original = _measurement_result()
    path = original.save(tmp_path / "résultats")
    loaded = MeasurementResult.load(path)

    assert path == tmp_path / "résultats.npz"
    pd.testing.assert_frame_equal(loaded.table, original.table)
    assert loaded.table.loc[0, "mean"].hex() == original.table.loc[0, "mean"].hex()
    assert loaded.provenance["units"] == "µm"
    assert np.isnan(loaded.provenance["nonfinite"])


def test_measurement_save_extension_and_overwrite_are_exact(tmp_path):
    result = _measurement_result()
    path = result.save(tmp_path / "measurements.npz")

    with pytest.raises(FileExistsError):
        result.save(path)
    assert result.save(path, overwrite=True) == path
    with pytest.raises(ValueError, match=".npz"):
        result.save(tmp_path / "measurements.txt")
    assert not (tmp_path / "measurements.txt.npz").exists()


def test_empty_measurement_table_round_trip_preserves_schema(tmp_path):
    table = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "channel": pd.Series(dtype="object"),
            "mean": pd.Series(dtype="float64"),
            "clipped": pd.Series(dtype="bool"),
        }
    )
    original = MeasurementResult(table=table, provenance={"measurements": []})
    loaded = MeasurementResult.load(original.save(tmp_path / "empty.npz"))

    pd.testing.assert_frame_equal(loaded.table, table)
    assert loaded.provenance == original.provenance


def test_legacy_measurement_archive_remains_loadable(tmp_path):
    original = _measurement_result()
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        format_version=np.asarray("1.0"),
        table_json=np.asarray(original.table.to_json(orient="table", index=False)),
        provenance_json=np.asarray(json.dumps({"units": "µm"})),
    )

    loaded = MeasurementResult.load(path)
    assert loaded.provenance == {"units": "µm"}
    assert list(loaded.table.columns) == list(original.table.columns)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"format_version": np.asarray("999")}, "unsupported"),
        ({"table_json": np.asarray("not-json")}, "malformed"),
        ({"provenance_json": np.asarray("[]")}, "type tag"),
        ({"format_version": np.asarray(["2.0"])}, "scalar string"),
    ],
)
def test_malformed_measurement_archives_fail_actionably(tmp_path, updates, message):
    path = _measurement_result().save(tmp_path / "malformed.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(updates)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match=message):
        MeasurementResult.load(path)


def test_missing_measurement_archive_field_is_rejected(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, format_version=np.asarray("2.0"))
    with pytest.raises(ValueError, match="missing required field"):
        MeasurementResult.load(path)
