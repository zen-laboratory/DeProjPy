"""Versioned, non-pickle persistence for intensity measurement results."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MEASUREMENT_FORMAT_VERSION = "2.0"
SUPPORTED_MEASUREMENT_FORMAT_VERSIONS = frozenset({"1.0", MEASUREMENT_FORMAT_VERSION})


def _measurement_path(filename: str | Path) -> Path:
    """Normalize a measurement filename to the only supported archive suffix."""
    path = Path(filename)
    if path.suffix == "":
        return path.with_suffix(".npz")
    if path.suffix.lower() != ".npz":
        raise ValueError("measurement filename must have a .npz suffix or no suffix")
    return path


def _encode_value(value: Any) -> dict[str, Any]:
    """Encode one scalar/list value without pickle or lossy float formatting."""
    if value is None or value is pd.NA:
        return {"type": "none"}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return {"type": "nan"}
        if np.isposinf(number):
            return {"type": "positive_infinity"}
        if np.isneginf(number):
            return {"type": "negative_infinity"}
        # Python's repr is a shortest round-trip representation for binary64.
        return {"type": "float", "value": repr(number)}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, (list, tuple, np.ndarray)):
        return {
            "type": "list",
            "value": [_encode_value(item) for item in list(value)],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": {str(key): _encode_value(item) for key, item in value.items()},
        }
    raise TypeError(f"cannot serialize measurement value of type {type(value).__name__}")


def _decode_value(encoded: Any) -> Any:
    """Decode and validate one tagged value from an untrusted archive."""
    if not isinstance(encoded, dict) or not isinstance(encoded.get("type"), str):
        raise ValueError("serialized value must be an object with a string type tag")
    value_type = encoded["type"]
    if value_type == "none":
        return None
    if value_type == "nan":
        return np.nan
    if value_type == "positive_infinity":
        return np.inf
    if value_type == "negative_infinity":
        return -np.inf
    if value_type == "bool" and isinstance(encoded.get("value"), bool):
        return encoded["value"]
    if value_type == "int" and isinstance(encoded.get("value"), int):
        return encoded["value"]
    if value_type == "float" and isinstance(encoded.get("value"), str):
        number = float(encoded["value"])
        if not np.isfinite(number):
            raise ValueError("finite float tag contains a nonfinite value")
        return number
    if value_type == "string" and isinstance(encoded.get("value"), str):
        return encoded["value"]
    if value_type == "list" and isinstance(encoded.get("value"), list):
        return [_decode_value(item) for item in encoded["value"]]
    if value_type == "dict" and isinstance(encoded.get("value"), dict):
        return {key: _decode_value(item) for key, item in encoded["value"].items()}
    raise ValueError(f"invalid serialized value for type tag {value_type!r}")


def _encode_table(table: pd.DataFrame) -> str:
    """Encode table values and column dtypes with exact binary64 round trips."""
    columns = [str(column) for column in table.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("measurement table column names must be unique")

    # Row-wise tagged values preserve mixed channel IDs and list-valued mode
    # edges. Dtype strings restore stable empty and nonempty table schemas.
    rows = [
        [_encode_value(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    payload = {
        "columns": columns,
        "dtypes": [str(dtype) for dtype in table.dtypes],
        "rows": rows,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _decode_table(text: str) -> pd.DataFrame:
    """Decode a validated table payload and restore recorded pandas dtypes."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("measurement table JSON is malformed") from error
    if not isinstance(payload, dict):
        raise ValueError("measurement table payload must be an object")

    columns = payload.get("columns")
    dtypes = payload.get("dtypes")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ValueError("measurement table columns must be a string list")
    if len(set(columns)) != len(columns):
        raise ValueError("measurement table columns must be unique")
    if not isinstance(dtypes, list) or len(dtypes) != len(columns):
        raise ValueError("measurement table dtypes must match its columns")
    if not all(isinstance(dtype, str) for dtype in dtypes):
        raise ValueError("measurement table dtypes must be strings")
    if not isinstance(rows, list):
        raise ValueError("measurement table rows must be a list")

    decoded_rows = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError("each measurement row must match the column count")
        decoded_rows.append([_decode_value(value) for value in row])

    # Construct each Series with its target dtype immediately. A whole-table
    # inference pass would turn ``None`` in object columns into float NaN before
    # a later astype call had a chance to preserve the original distinction.
    restored_columns = {}
    for column_index, (column, dtype) in enumerate(
        zip(columns, dtypes, strict=True)
    ):
        values = [row[column_index] for row in decoded_rows]
        try:
            restored_columns[column] = pd.Series(values, dtype=dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"measurement column {column!r} cannot be restored as {dtype!r}"
            ) from error
    return pd.DataFrame(restored_columns, columns=pd.Index(columns))


def _encode_provenance(provenance: dict) -> str:
    """Serialize provenance through the same explicit tagged-value grammar."""
    return json.dumps(
        _encode_value(provenance),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _decode_provenance(text: str) -> dict:
    """Decode provenance and require a mapping at the archive boundary."""
    try:
        encoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("measurement provenance JSON is malformed") from error
    provenance = _decode_value(encoded)
    if not isinstance(provenance, dict):
        raise ValueError("measurement provenance must decode to a dictionary")
    return provenance


def save_measurement_result(
    table: pd.DataFrame,
    provenance: dict,
    filename: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Write a validated measurement archive and return its actual path."""
    path = _measurement_path(filename)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Passing an open handle prevents NumPy from silently appending another
    # suffix; overwrite checks and the returned path therefore match reality.
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            format_version=np.asarray(MEASUREMENT_FORMAT_VERSION),
            table_json=np.asarray(_encode_table(table)),
            provenance_json=np.asarray(_encode_provenance(provenance)),
        )
    return path


def _scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str:
    """Read one required scalar string field from an NPZ archive."""
    if key not in data.files:
        raise ValueError(f"measurement archive is missing required field {key!r}")
    value = np.asarray(data[key])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"measurement archive field {key!r} must be a scalar string")
    return str(value.item())


def load_measurement_result(filename: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load current or legacy measurement data without enabling NumPy pickle."""
    path = _measurement_path(filename)
    with np.load(path, allow_pickle=False) as data:
        version = _scalar_string(data, "format_version")
        if version not in SUPPORTED_MEASUREMENT_FORMAT_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_MEASUREMENT_FORMAT_VERSIONS))
            raise ValueError(
                f"unsupported intensity measurement format {version!r}; "
                f"supported versions: {supported}"
            )
        table_json = _scalar_string(data, "table_json")
        provenance_json = _scalar_string(data, "provenance_json")

    if version == "1.0":
        # StringIO is required by current pandas and removes the deprecated
        # interpretation of literal JSON text as a filesystem path.
        table = pd.read_json(StringIO(table_json), orient="table")
        try:
            provenance = json.loads(provenance_json)
        except json.JSONDecodeError as error:
            raise ValueError("legacy measurement provenance JSON is malformed") from error
        if not isinstance(provenance, dict):
            raise ValueError("legacy measurement provenance must be a dictionary")
        return table, provenance

    return _decode_table(table_json), _decode_provenance(provenance_json)
