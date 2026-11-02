"""Deterministic polygon and depth quadrature for fitted cell prisms."""

from __future__ import annotations

from collections import deque

import numpy as np


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Return the scalar 2-D cross product without NumPy's 3-D promotion."""
    return float(first[0] * second[1] - first[1] * second[0])


def signed_polygon_area(polygon_uv: np.ndarray) -> float:
    """Return the signed shoelace area of an ordered plane-UV polygon."""
    polygon = np.asarray(polygon_uv, dtype=float)
    next_vertices = np.roll(polygon, -1, axis=0)
    twice_area = np.sum(
        polygon[:, 0] * next_vertices[:, 1]
        - polygon[:, 1] * next_vertices[:, 0]
    )
    return 0.5 * float(twice_area)


def _geometry_tolerances(points_uv: np.ndarray) -> tuple[float, float]:
    """Return scale-sensitive length and signed-area tolerances."""
    extent = np.ptp(points_uv, axis=0)
    # Plane UV is centered near the boundary mean, so extent is the relevant
    # scale. Avoiding a unit-sized floor preserves valid tiny physical cells.
    scale = max(float(np.max(extent)), np.finfo(float).eps)
    length_tolerance = 64.0 * np.finfo(float).eps * scale
    area_tolerance = 64.0 * np.finfo(float).eps * scale**2
    return length_tolerance, area_tolerance


def _segments_intersect(
    first_start_uv: np.ndarray,
    first_end_uv: np.ndarray,
    second_start_uv: np.ndarray,
    second_end_uv: np.ndarray,
    area_tolerance: float,
) -> bool:
    """Return whether two closed 2-D segments intersect within tolerance."""
    first_direction = first_end_uv - first_start_uv
    second_direction = second_end_uv - second_start_uv
    orientations = (
        _cross_2d(first_direction, second_start_uv - first_start_uv),
        _cross_2d(first_direction, second_end_uv - first_start_uv),
        _cross_2d(second_direction, first_start_uv - second_start_uv),
        _cross_2d(second_direction, first_end_uv - second_start_uv),
    )

    # Crossing or touching segments must have no same-side endpoint pair and
    # overlapping bounding boxes. This also catches a nonadjacent vertex that
    # lands on an edge, which would make the polygon non-simple.
    separated_by_first_line = orientations[0] * orientations[1] > area_tolerance**2
    separated_by_second_line = orientations[2] * orientations[3] > area_tolerance**2
    if separated_by_first_line or separated_by_second_line:
        return False
    first_min = np.minimum(first_start_uv, first_end_uv)
    first_max = np.maximum(first_start_uv, first_end_uv)
    second_min = np.minimum(second_start_uv, second_end_uv)
    second_max = np.maximum(second_start_uv, second_end_uv)
    coordinate_scale = max(
        float(np.max(np.abs([first_start_uv, first_end_uv, second_start_uv, second_end_uv]))),
        np.finfo(float).eps,
    )
    coordinate_tolerance = 64.0 * np.finfo(float).eps * coordinate_scale
    return bool(
        np.all(first_max + coordinate_tolerance >= second_min)
        and np.all(second_max + coordinate_tolerance >= first_min)
    )


def _validate_simple_polygon(points_uv: np.ndarray, area_tolerance: float) -> None:
    """Reject crossings or contacts between nonadjacent polygon edges."""
    count = len(points_uv)
    for first_index in range(count):
        first_next = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if first_index in {second_index, second_next}:
                continue
            if first_next in {second_index, second_next}:
                continue
            if _segments_intersect(
                points_uv[first_index],
                points_uv[first_next],
                points_uv[second_index],
                points_uv[second_next],
                area_tolerance,
            ):
                raise ValueError("polygon is self-intersecting or touches a nonadjacent edge")


def clean_polygon(polygon_uv: np.ndarray) -> np.ndarray:
    """Normalize an ordered polygon before triangulation.

    Consecutive duplicate vertices, an explicit closing vertex, and collinear
    middle vertices are removed using tolerances scaled to the polygon extent.
    The returned polygon is open and counter-clockwise. The input is not
    modified.

    Parameters
    ----------
    polygon_uv
        Ordered vertices with shape ``(n, 2)`` in fitted-plane coordinates.

    Returns
    -------
    numpy.ndarray
        A newly allocated, counter-clockwise polygon with no repeated closing
        vertex.

    Raises
    ------
    ValueError
        If coordinates are malformed/nonfinite or fewer than three
        non-collinear vertices remain.
    """
    points_uv = np.asarray(polygon_uv, dtype=float)
    if points_uv.ndim != 2 or points_uv.shape[1] != 2:
        raise ValueError("polygon_uv must have shape (n, 2)")
    if len(points_uv) < 3 or not np.all(np.isfinite(points_uv)):
        raise ValueError("polygon must contain at least three finite vertices")

    length_tolerance, area_tolerance = _geometry_tolerances(points_uv)

    # DeProjPy boundaries are normally open, but accepting a repeated closing
    # vertex makes the lower-level routine safe for conventional polygon input.
    retained = [points_uv[0].copy()]
    for point_uv in points_uv[1:]:
        if np.linalg.norm(point_uv - retained[-1]) > length_tolerance:
            retained.append(point_uv.copy())
    if len(retained) > 1 and np.linalg.norm(retained[-1] - retained[0]) <= length_tolerance:
        retained.pop()

    # Removing one collinear vertex can make its neighbor newly collinear, so
    # repeat until a full pass is stable. Only a middle vertex is removed.
    changed = True
    while changed and len(retained) >= 3:
        changed = False
        simplified: list[np.ndarray] = []
        count = len(retained)
        for index, current_uv in enumerate(retained):
            previous_uv = retained[index - 1]
            next_uv = retained[(index + 1) % count]
            incoming = current_uv - previous_uv
            outgoing = next_uv - current_uv
            is_collinear = abs(_cross_2d(incoming, outgoing)) <= area_tolerance
            points_forward = float(np.dot(incoming, outgoing)) >= -length_tolerance**2
            if is_collinear and points_forward:
                changed = True
                continue
            simplified.append(current_uv)
        retained = simplified

    if len(retained) < 3:
        raise ValueError("polygon is degenerate after duplicate/collinear cleanup")

    cleaned = np.asarray(retained, dtype=float)
    _validate_simple_polygon(cleaned, area_tolerance)
    area = signed_polygon_area(cleaned)
    if abs(area) <= area_tolerance:
        raise ValueError("polygon has numerically zero projected area")
    if area < 0:
        cleaned = cleaned[::-1].copy()
    return cleaned


def _point_in_triangle(
    point_uv: np.ndarray,
    triangle_uv: np.ndarray,
    area_tolerance: float,
) -> bool:
    """Return whether a point lies in/on a counter-clockwise triangle."""
    for edge_index in range(3):
        start_uv = triangle_uv[edge_index]
        end_uv = triangle_uv[(edge_index + 1) % 3]
        if _cross_2d(end_uv - start_uv, point_uv - start_uv) < -area_tolerance:
            return False
    return True


def triangulate_polygon(polygon_uv: np.ndarray) -> np.ndarray:
    """Triangulate a simple, potentially nonconvex polygon by ear clipping.

    The custom implementation avoids adding a geometry dependency for this
    narrowly scoped operation. It first normalizes the polygon, then removes a
    convex ear containing no other polygon vertex. Failure to find an ear is
    treated as evidence of self-intersection or numerical invalidity rather
    than fabricating geometry.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n - 2, 3, 2)``. Triangle orientation is
        counter-clockwise and triangle areas sum to polygon area.
    """
    cleaned_uv = clean_polygon(polygon_uv)
    _, area_tolerance = _geometry_tolerances(cleaned_uv)
    remaining = list(range(len(cleaned_uv)))
    triangles: list[np.ndarray] = []

    while len(remaining) > 3:
        ear_found = False
        for position, current_index in enumerate(remaining):
            previous_index = remaining[position - 1]
            next_index = remaining[(position + 1) % len(remaining)]
            triangle_uv = cleaned_uv[[previous_index, current_index, next_index]]

            first_edge = triangle_uv[1] - triangle_uv[0]
            second_edge = triangle_uv[2] - triangle_uv[1]
            if _cross_2d(first_edge, second_edge) <= area_tolerance:
                continue

            contains_vertex = any(
                _point_in_triangle(cleaned_uv[index], triangle_uv, area_tolerance)
                for index in remaining
                if index not in {previous_index, current_index, next_index}
            )
            if contains_vertex:
                continue

            triangles.append(triangle_uv)
            del remaining[position]
            ear_found = True
            break

        if not ear_found:
            raise ValueError(
                "polygon is self-intersecting or numerically unsuitable for ear clipping"
            )

    triangles.append(cleaned_uv[remaining])
    return np.asarray(triangles, dtype=float)


def _triangle_area(triangle_uv: np.ndarray) -> float:
    """Return the positive area of a counter-clockwise triangle."""
    return 0.5 * _cross_2d(
        triangle_uv[1] - triangle_uv[0],
        triangle_uv[2] - triangle_uv[0],
    )


def subdivide_triangles(
    triangles_uv: np.ndarray,
    maximum_edge_length: float,
) -> np.ndarray:
    """Bisect longest edges until every triangle edge respects the spacing.

    Longest-edge bisection makes sampling depend on geometric extent rather
    than area alone. This is important for long, narrow cells. Bisection
    exactly preserves area and terminates because each split halves the chosen
    edge. The returned subtriangles may have hanging nodes across original ear
    boundaries; that is harmless for independent area quadrature.
    """
    if not np.isfinite(maximum_edge_length) or maximum_edge_length <= 0:
        raise ValueError("maximum_edge_length must be positive and finite")

    pending = deque(np.asarray(triangle, dtype=float) for triangle in triangles_uv)
    accepted: list[np.ndarray] = []
    comparison_tolerance = 32.0 * np.finfo(float).eps

    while pending:
        triangle_uv = pending.popleft()
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        edge_lengths = np.asarray(
            [np.linalg.norm(triangle_uv[end] - triangle_uv[start]) for start, end in edge_pairs]
        )
        longest_edge_index = int(np.argmax(edge_lengths))
        if edge_lengths[longest_edge_index] <= maximum_edge_length * (
            1.0 + comparison_tolerance
        ):
            accepted.append(triangle_uv)
            continue

        start_index, end_index = edge_pairs[longest_edge_index]
        opposite_index = 3 - start_index - end_index
        midpoint_uv = 0.5 * (triangle_uv[start_index] + triangle_uv[end_index])

        first_child = np.asarray(
            [triangle_uv[start_index], midpoint_uv, triangle_uv[opposite_index]]
        )
        second_child = np.asarray(
            [midpoint_uv, triangle_uv[end_index], triangle_uv[opposite_index]]
        )
        if _triangle_area(first_child) < 0:
            first_child[[1, 2]] = first_child[[2, 1]]
        if _triangle_area(second_child) < 0:
            second_child[[1, 2]] = second_child[[2, 1]]
        pending.extend((first_child, second_child))

    return np.asarray(accepted, dtype=float)


def build_lateral_quadrature(
    polygon_uv: np.ndarray,
    sampling_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build first-moment-exact footprint quadrature.

    Each extent-controlled subtriangle contributes one centroid with its exact
    area as weight. A triangle centroid integrates constants and affine fields
    exactly. Consequently, the complete rule recovers polygon area and first
    moments up to floating-point roundoff, including when a tiny footprint is
    represented by only one sample.

    Returns
    -------
    points_uv, area_weights, cleaned_polygon_uv
        Centroid samples ``(n, 2)``, positive physical area weights ``(n,)``,
        and the normalized polygon used for triangulation.
    """
    # Normalize and triangulate once per region; the resulting centroid/area
    # rule is compact and can be reused for every aligned image channel.
    cleaned_polygon_uv = clean_polygon(polygon_uv)
    triangles_uv = triangulate_polygon(cleaned_polygon_uv)
    subtriangles_uv = subdivide_triangles(triangles_uv, sampling_spacing)

    # One centroid with the exact subtriangle area is first-moment exact, unlike
    # arbitrary point clouds whose weights merely sum to polygon area.
    points_uv = np.mean(subtriangles_uv, axis=1)
    area_weights = np.asarray(
        [_triangle_area(triangle_uv) for triangle_uv in subtriangles_uv],
        dtype=float,
    )
    return points_uv, area_weights, cleaned_polygon_uv


def depth_midpoint_quadrature(
    start_depth: float,
    end_depth: float,
    sampling_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a signed depth interval exactly using midpoint samples."""
    # Equal bins avoid a special short final bin while still spanning the exact
    # requested interval; every width is at most sampling_spacing.
    depth_span = end_depth - start_depth
    bin_count = max(1, int(np.ceil(depth_span / sampling_spacing)))
    bin_width = depth_span / bin_count
    depth_points = start_depth + (np.arange(bin_count, dtype=float) + 0.5) * bin_width
    depth_widths = np.full(bin_count, bin_width, dtype=float)
    return depth_points, depth_widths
