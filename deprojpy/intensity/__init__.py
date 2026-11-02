"""Surface-normal intensity measurements for curved epithelial tissues.

Image arrays use ``(z, y, x)`` order; DeProjPy geometry uses physical
``(x, y, z)`` coordinates. Regions approximate cells with fitted planar prisms
and are estimates of subapical volume, not volumetric segmentations.
"""

from ._measurement import measure
from ._models import CellRegion, MeasurementResult, PreparedRegions, StackCalibration
from ._plotting import plot_feature_map, plot_region_preview
from ._regions import prepare_regions

__all__ = [
    "CellRegion",
    "MeasurementResult",
    "PreparedRegions",
    "StackCalibration",
    "measure",
    "plot_feature_map",
    "plot_region_preview",
    "prepare_regions",
]
