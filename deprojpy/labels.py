from __future__ import annotations

import labelimage_tools as lit
import networkx as nx
import numpy as np

from .objects import MaskObject


def validate_labeled_input(labels: np.ndarray) -> np.ndarray:
    """Validate and return a 2-D integer labeled-cell image."""
    return lit.validate_label_image(labels)


def _border_labels(labels: np.ndarray, background: int) -> set[int]:
    neighbors = lit.adjacency_from_labels(
        labels,
        background=background,
        eight=True,
        allow_background_contacts=True,
    )
    by_background = set(map(int, lit.border_labels(neighbors, background=background)))
    edge_values = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    by_image_edge = {int(value) for value in edge_values if value != background}
    return by_background | by_image_edge


def _ordered_junction_ids_for_label(label: int, contour_yx: np.ndarray, junctions) -> np.ndarray:
    indexed = []
    for junction in junctions:
        if label not in junction.labels:
            continue
        jyx = np.asarray(junction.yx, dtype=float)
        distances2 = np.sum((contour_yx - jyx) ** 2, axis=1)
        indexed.append((int(np.argmin(distances2)), int(junction.id)))
    indexed.sort()
    sequence: list[int] = []
    for _, junction_id in indexed:
        if junction_id not in sequence:
            sequence.append(junction_id)
    return np.asarray(sequence, dtype=np.int64)


def labels_to_objects(
    labels: np.ndarray,
    *,
    background: int = 0,
    drop_border_cells: bool = True,
    min_junction_labels: int = 3,
    junction_background: int | None = None,
    junction_connectivity: int = 2,
    junction_merge_epsilon: float = 0.0,
) -> tuple[list[MaskObject], nx.Graph]:
    """Convert a labeled cell image into shared DeProj objects and junction graph."""
    labels = lit.validate_label_image(labels, background=background)
    contours = lit.ordered_contours_from_labels(labels, background=background)
    _, junctions = lit.junctions_from_labels(
        labels,
        background=junction_background,
        min_labels=min_junction_labels,
        connectivity=junction_connectivity,
        merge_epsilon=junction_merge_epsilon,
    )
    graph = nx.Graph()
    for junction in junctions:
        graph.add_node(
            int(junction.id),
            id=int(junction.id),
            centroid=np.array([junction.yx[1], junction.yx[0]], dtype=float),
            labels=frozenset(int(label) for label in junction.labels),
            pixel_coords=np.asarray(junction.pixel_coords, dtype=np.int64),
        )

    skipped = _border_labels(labels, background) if drop_border_cells else set()
    objects: list[MaskObject] = []
    for label, contour_yx in contours.items():
        label = int(label)
        if label in skipped:
            continue
        if len(contour_yx) < 3:
            continue
        boundary_xy = np.asarray(contour_yx[:, ::-1], dtype=float)
        junction_ids = _ordered_junction_ids_for_label(label, contour_yx, junctions)
        if len(junction_ids) >= 2:
            sequence = [int(value) for value in junction_ids]
            for first, second in zip(sequence, sequence[1:] + sequence[:1], strict=True):
                if first != second:
                    graph.add_edge(first, second)
        objects.append(
            MaskObject(
                boundary=boundary_xy,
                center=boundary_xy.mean(axis=0),
                junction_ids=junction_ids,
                source_label=label,
            )
        )
    if not objects:
        raise ValueError(
            "labeled-image parsing found no retained cells; check background, "
            "border-cell dropping, junction detection, and label connectivity"
        )
    return objects, graph
