from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaskObject:
    """Shared 2-D object extracted from a segmentation input.

    Coordinates are geometric ``(x, y)`` pixel coordinates, not array
    ``(row, column)`` indices. ``source_label`` stores the original label value
    for labeled-image inputs and is ``None`` for binary-ridge masks.
    """

    boundary: np.ndarray
    center: np.ndarray
    junction_ids: np.ndarray
    source_label: int | None = None
    footprint_rc: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64)
    )
