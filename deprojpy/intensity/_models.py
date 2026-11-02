"""Data models shared by surface-normal intensity modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StackCalibration:
    """Describe the supplied image stack in original acquisition coordinates.

    Parameters
    ----------
    spacing_zyx
        Physical spacing ``(dz, dy, dx)`` between stack indices. Values use
        the same length unit as the associated :class:`DeprojResult`.
    origin_zyx
        Original-stack physical coordinate ``(z, y, x)`` represented by index
        ``(0, 0, 0)`` in the supplied array. A nonzero value describes a crop
        or resampled grid without changing the saved deprojection geometry.

    Raises
    ------
    ValueError
        If either tuple has the wrong shape, spacing is not positive, or a
        value is nonfinite.
    """

    spacing_zyx: tuple[float, float, float]
    origin_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        spacing_zyx = np.asarray(self.spacing_zyx, dtype=float)
        origin_zyx = np.asarray(self.origin_zyx, dtype=float)

        if spacing_zyx.shape != (3,):
            raise ValueError("spacing_zyx must have exactly three values (dz, dy, dx)")
        if not np.all(np.isfinite(spacing_zyx)) or np.any(spacing_zyx <= 0):
            raise ValueError("spacing_zyx values must be positive and finite")
        if origin_zyx.shape != (3,) or not np.all(np.isfinite(origin_zyx)):
            raise ValueError("origin_zyx must contain three finite values (z, y, x)")


@dataclass
class CellRegion:
    """Compact geometry and cached lateral quadrature for one cell prism.

    All XYZ arrays and depths use the parent result's physical length unit.
    ``plane_basis_xyz`` has two orthonormal columns mapping plane UV to XYZ.
    ``lateral_points_uv`` and ``area_weights`` are cached because they are
    independent of image channel and depth sampling.
    """

    cell_id: int
    source_label: int | None
    plane_origin_xyz: np.ndarray
    plane_basis_xyz: np.ndarray
    polygon_uv: np.ndarray
    normal_xyz: np.ndarray
    start_depth: float
    end_depth: float
    footprint_area: float
    lateral_points_uv: np.ndarray
    area_weights: np.ndarray
    status: str = "ok"
    reason: str = ""

    @property
    def origin_xyz(self) -> np.ndarray:
        """Compatibility alias for the fitted-plane origin."""
        return self.plane_origin_xyz

    @property
    def basis_xyz(self) -> np.ndarray:
        """Compatibility alias for the two fitted-plane basis vectors."""
        return self.plane_basis_xyz

    @property
    def start(self) -> float:
        """Compatibility alias for ``start_depth``."""
        return self.start_depth

    @property
    def end(self) -> float:
        """Compatibility alias for ``end_depth``."""
        return self.end_depth

    @property
    def area(self) -> float:
        """Compatibility alias for ``footprint_area``."""
        return self.footprint_area


@dataclass
class PreparedRegions:
    """Reusable cell regions and the coordinate contract for their stack."""

    cells: list[CellRegion]
    units: str
    depth_range: tuple[float, float]
    depth_units: str
    converted_depth_range: tuple[float, float]
    sampling_spacing: float
    tissue_side: str
    calibration: StackCalibration
    z_reflected: bool
    z_reflection_offset: float


@dataclass
class MeasurementResult:
    """Tabular intensity measurements plus reproducibility metadata."""

    table: pd.DataFrame
    provenance: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return an independent copy of the measurement table."""
        return self.table.copy()

    def save(self, filename: str | Path, *, overwrite: bool = False) -> Path:
        """Save table and provenance to a versioned, non-pickle NPZ archive."""
        from ._persistence import save_measurement_result

        return save_measurement_result(
            self.table,
            self.provenance,
            filename,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filename: str | Path) -> "MeasurementResult":
        """Load a current or backward-compatible legacy measurement archive."""
        from ._persistence import load_measurement_result

        table, provenance = load_measurement_result(filename)
        return cls(table=table, provenance=provenance)
