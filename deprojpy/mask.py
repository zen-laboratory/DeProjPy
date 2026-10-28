from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from .objects import MaskObject


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and return a two-dimensional black-ridge/white-cell mask."""
    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError(f"mask must be a 2-D array; received shape {image.shape}")
    if image.size == 0:
        raise ValueError("mask must not be empty")
    if not np.all(np.isfinite(image)):
        raise ValueError("mask must contain only finite values")
    values = np.unique(image)
    if len(values) != 2 or 0 not in values:
        raise ValueError(
            "mask must be binary-like with exactly two values: black ridges "
            "(0) and nonzero cell interiors"
        )
    return image


def _ordered_boundary(component: np.ndarray) -> np.ndarray:
    """Return an ordered boundary as integer (x, y) pixel-center coordinates."""
    contours = measure.find_contours(component, 0.5, fully_connected="high")
    if not contours:
        return np.empty((0, 2), dtype=float)
    contour = max(contours, key=len)
    # find_contours follows pixel edges. Snap inward to actual dilated-cell
    # pixels, matching MATLAB bwboundaries' integer row/column coordinates.
    candidates = np.rint(contour).astype(int)
    candidates[:, 0] = np.clip(candidates[:, 0], 0, component.shape[0] - 1)
    candidates[:, 1] = np.clip(candidates[:, 1], 0, component.shape[1] - 1)
    for k, (row, col) in enumerate(candidates):
        if not component[row, col]:
            neighborhood = np.argwhere(
                component[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
            )
            if neighborhood.size:
                candidates[k] = neighborhood[0] + [max(0, row - 1), max(0, col - 1)]
    keep = np.r_[True, np.any(np.diff(candidates, axis=0) != 0, axis=1)]
    rc = candidates[keep]
    if len(rc) > 1 and np.array_equal(rc[0], rc[-1]):
        rc = rc[:-1]
    return rc[:, ::-1].astype(float)  # explicit x, y order


def mask_to_objects(mask: np.ndarray) -> tuple[list[MaskObject], nx.Graph]:
    """Convert white 4-connected cells and the black ridge into objects/graph.

    Input array indices are ``(row, column)``; all returned coordinates are
    geometric ``(x, y)`` pixel coordinates, with the first pixel at (0, 0).
    """
    # Ensure the input has the expected binary structure before processing.
    image = validate_binary_mask(mask)
    # Label each white region as an individual cell using 4-connectivity.
    white = image != 0
    labels, n_components = ndi.label(white, ndi.generate_binary_structure(2, 1)) # type: ignore

    # The black pixels form ridges; local ridge density helps identify junctions.
    ridge = ~white
    ridge_neighbors = ndi.convolve(
        ridge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    # Ridge pixels with enough neighboring ridge pixels are treated as junctions.
    junction_mask = ridge & (ridge_neighbors >= 4)
    junction_labels, n_junctions = ndi.label(junction_mask, ndi.generate_binary_structure(2, 2)) # type: ignore
    centers_rc = ndi.center_of_mass(junction_mask, junction_labels, range(1, n_junctions + 1))

    # Create a graph node for each detected junction.
    graph = nx.Graph()
    for junction_id, (row, col) in enumerate(centers_rc, start=1):
        graph.add_node(junction_id, id=junction_id, centroid=np.array([col, row], float))

    footprint = morphology.diamond(1)
    height, width = image.shape
    objects: list[MaskObject] = []
    for label_id in range(1, n_components + 1):
        rows, cols = np.where(labels == label_id)
        # Skip border-touching cells because their boundaries are incomplete.
        if (
            rows.min() <= 1
            or rows.max() >= height - 2
            or cols.min() <= 1
            or cols.max() >= width - 2
        ):
            continue
        # Slight dilation stabilizes boundary tracing around each cell.
        component = ndi.binary_dilation(labels == label_id, structure=footprint)
        boundary = _ordered_boundary(component)
        if len(boundary) < 3:
            continue

        # Map the boundary back onto the junction labels to find touched nodes.
        xy = np.rint(boundary).astype(int)
        visited = junction_labels[xy[:, 1], xy[:, 0]]
        visited = visited[visited > 0]
        junction_ids = np.unique(visited)

        # Keep traversal order so adjacent junctions can be connected in the graph.
        sequence: list[int] = []
        for jid in visited:
            jid = int(jid)
            if not sequence or sequence[-1] != jid:
                sequence.append(jid)
        if len(sequence) > 1 and sequence[0] == sequence[-1]:
            sequence.pop()
        for first, second in zip(sequence, sequence[1:] + sequence[:1], strict=True):
            if first != second:
                graph.add_edge(first, second)

        # Store the cell boundary and approximate center for later use.
        footprint_rc = np.argwhere(component).astype(np.int64, copy=False)
        objects.append(
            MaskObject(
                boundary,
                boundary.mean(axis=0),
                junction_ids,
                footprint_rc=footprint_rc,
            )
        )
    if not objects:
        raise ValueError(
            "mask parsing found no retained cells; expected nonzero cell interiors "
            "separated by a connected black ridge, with cells away from image borders"
        )
    return objects, graph
