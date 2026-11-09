"""Deterministic polygon and depth quadrature for fitted cell prisms."""

from __future__ import annotations

import numpy as np
from numba import njit


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


@njit(cache=True)
def _has_nonadjacent_edge_intersection(
    points_uv: np.ndarray,
    area_tolerance: float,
) -> bool:
    """Scan nonadjacent edge pairs without returning to Python per pair."""
    count = len(points_uv)
    squared_area_tolerance = area_tolerance * area_tolerance
    machine_epsilon = np.finfo(np.float64).eps

    for first_index in range(count):
        first_next = (first_index + 1) % count
        first_start_x = points_uv[first_index, 0]
        first_start_y = points_uv[first_index, 1]
        first_end_x = points_uv[first_next, 0]
        first_end_y = points_uv[first_next, 1]
        first_dx = first_end_x - first_start_x
        first_dy = first_end_y - first_start_y

        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            # Adjacent edges legitimately share one endpoint. All other
            # crossing, touching, and collinear-overlap cases are invalid.
            if first_index == second_index or first_index == second_next:
                continue
            if first_next == second_index or first_next == second_next:
                continue

            second_start_x = points_uv[second_index, 0]
            second_start_y = points_uv[second_index, 1]
            second_end_x = points_uv[second_next, 0]
            second_end_y = points_uv[second_next, 1]
            second_dx = second_end_x - second_start_x
            second_dy = second_end_y - second_start_y

            first_start_to_second_start_x = second_start_x - first_start_x
            first_start_to_second_start_y = second_start_y - first_start_y
            first_start_to_second_end_x = second_end_x - first_start_x
            first_start_to_second_end_y = second_end_y - first_start_y
            second_start_to_first_start_x = first_start_x - second_start_x
            second_start_to_first_start_y = first_start_y - second_start_y
            second_start_to_first_end_x = first_end_x - second_start_x
            second_start_to_first_end_y = first_end_y - second_start_y

            first_orientation_start = (
                first_dx * first_start_to_second_start_y
                - first_dy * first_start_to_second_start_x
            )
            first_orientation_end = (
                first_dx * first_start_to_second_end_y
                - first_dy * first_start_to_second_end_x
            )
            second_orientation_start = (
                second_dx * second_start_to_first_start_y
                - second_dy * second_start_to_first_start_x
            )
            second_orientation_end = (
                second_dx * second_start_to_first_end_y
                - second_dy * second_start_to_first_end_x
            )
            if (
                first_orientation_start * first_orientation_end
                > squared_area_tolerance
                or second_orientation_start * second_orientation_end
                > squared_area_tolerance
            ):
                continue

            coordinate_scale = max(
                abs(first_start_x),
                abs(first_start_y),
                abs(first_end_x),
                abs(first_end_y),
                abs(second_start_x),
                abs(second_start_y),
                abs(second_end_x),
                abs(second_end_y),
                machine_epsilon,
            )
            coordinate_tolerance = 64.0 * machine_epsilon * coordinate_scale
            if (
                max(first_start_x, first_end_x) + coordinate_tolerance
                >= min(second_start_x, second_end_x)
                and max(second_start_x, second_end_x) + coordinate_tolerance
                >= min(first_start_x, first_end_x)
                and max(first_start_y, first_end_y) + coordinate_tolerance
                >= min(second_start_y, second_end_y)
                and max(second_start_y, second_end_y) + coordinate_tolerance
                >= min(first_start_y, first_end_y)
            ):
                return True

    return False


def _validate_simple_polygon(points_uv: np.ndarray, area_tolerance: float) -> None:
    """Reject crossings or contacts between nonadjacent polygon edges."""
    if _has_nonadjacent_edge_intersection(points_uv, area_tolerance):
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


@njit(cache=True)
def _triangulate_cleaned_kernel(
    cleaned_uv: np.ndarray,
    area_tolerance: float,
) -> tuple[np.ndarray, bool]:
    """Clip the first valid ear on each pass, preserving deterministic order."""
    vertex_count = len(cleaned_uv)
    remaining = np.arange(vertex_count)
    remaining_count = vertex_count
    triangles = np.empty((vertex_count - 2, 3, 2), dtype=np.float64)
    triangle_count = 0

    while remaining_count > 3:
        ear_position = -1
        for position in range(remaining_count):
            previous_index = remaining[(position - 1) % remaining_count]
            current_index = remaining[position]
            next_index = remaining[(position + 1) % remaining_count]

            first_dx = cleaned_uv[current_index, 0] - cleaned_uv[previous_index, 0]
            first_dy = cleaned_uv[current_index, 1] - cleaned_uv[previous_index, 1]
            second_dx = cleaned_uv[next_index, 0] - cleaned_uv[current_index, 0]
            second_dy = cleaned_uv[next_index, 1] - cleaned_uv[current_index, 1]
            if first_dx * second_dy - first_dy * second_dx <= area_tolerance:
                continue

            contains_vertex = False
            for candidate_position in range(remaining_count):
                candidate_index = remaining[candidate_position]
                if (
                    candidate_index == previous_index
                    or candidate_index == current_index
                    or candidate_index == next_index
                ):
                    continue

                point_x = cleaned_uv[candidate_index, 0]
                point_y = cleaned_uv[candidate_index, 1]
                triangle_indices = (previous_index, current_index, next_index)
                inside = True
                for edge_index in range(3):
                    start_index = triangle_indices[edge_index]
                    end_index = triangle_indices[(edge_index + 1) % 3]
                    edge_dx = cleaned_uv[end_index, 0] - cleaned_uv[start_index, 0]
                    edge_dy = cleaned_uv[end_index, 1] - cleaned_uv[start_index, 1]
                    point_dx = point_x - cleaned_uv[start_index, 0]
                    point_dy = point_y - cleaned_uv[start_index, 1]
                    if edge_dx * point_dy - edge_dy * point_dx < -area_tolerance:
                        inside = False
                        break
                if inside:
                    contains_vertex = True
                    break

            if not contains_vertex:
                ear_position = position
                triangles[triangle_count, 0] = cleaned_uv[previous_index]
                triangles[triangle_count, 1] = cleaned_uv[current_index]
                triangles[triangle_count, 2] = cleaned_uv[next_index]
                triangle_count += 1
                break

        if ear_position < 0:
            return triangles[:triangle_count], False

        for position in range(ear_position, remaining_count - 1):
            remaining[position] = remaining[position + 1]
        remaining_count -= 1

    for position in range(3):
        triangles[triangle_count, position] = cleaned_uv[remaining[position]]
    return triangles, True


def _triangulate_cleaned_polygon(cleaned_uv: np.ndarray) -> np.ndarray:
    """Triangulate a normalized polygon using tolerances from that polygon."""
    _, area_tolerance = _geometry_tolerances(cleaned_uv)
    triangles, succeeded = _triangulate_cleaned_kernel(cleaned_uv, area_tolerance)
    if not succeeded:
        raise ValueError(
            "polygon is self-intersecting or numerically unsuitable for ear clipping"
        )
    return triangles


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
    return _triangulate_cleaned_polygon(cleaned_uv)


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

    triangles = np.asarray(triangles_uv, dtype=float)
    if len(triangles) == 0:
        return np.empty(0, dtype=float)
    comparison_tolerance = 32.0 * np.finfo(float).eps
    edge_limit = maximum_edge_length * (1.0 + comparison_tolerance)
    accepted, _, _ = _subdivide_triangles(triangles, edge_limit, emit_triangles=True)
    return accepted


@njit(cache=True)
def _subdivide_triangles(
    triangles: np.ndarray,
    edge_limit: float,
    emit_triangles: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traverse one breadth-first queue and emit the requested compact rule."""
    capacity = max(len(triangles), 16)
    pending = np.empty((capacity, 3, 2), dtype=np.float64)
    pending[:len(triangles)] = triangles
    accepted = np.empty_like(pending) if emit_triangles else np.empty((0, 3, 2))
    centroids = np.empty((0, 2)) if emit_triangles else np.empty((capacity, 2))
    area_weights = np.empty(0) if emit_triangles else np.empty(capacity)
    read = 0
    write = len(triangles)
    count = 0

    while read < write:
        triangle = pending[read]
        read += 1

        # Scalar edge lengths avoid allocating three tiny arrays and calling
        # general-purpose NumPy reductions for every triangle in the queue.
        longest_edge = 0
        longest_length = -1.0
        for start in range(3):
            end = (start + 1) % 3
            dx = triangle[end, 0] - triangle[start, 0]
            dy = triangle[end, 1] - triangle[start, 1]
            squared_length = dx * dx + dy * dy
            if squared_length > longest_length:
                longest_edge = start
                longest_length = squared_length

        if np.sqrt(longest_length) <= edge_limit:
            if emit_triangles:
                if count == len(accepted):
                    larger = np.empty((2 * count, 3, 2), dtype=np.float64)
                    larger[:count] = accepted
                    accepted = larger
                accepted[count] = triangle
            else:
                if count == len(centroids):
                    larger_centroids = np.empty((2 * count, 2), dtype=np.float64)
                    larger_weights = np.empty(2 * count, dtype=np.float64)
                    larger_centroids[:count] = centroids
                    larger_weights[:count] = area_weights
                    centroids = larger_centroids
                    area_weights = larger_weights

                # The builder needs only the first-moment-exact centroid and
                # signed physical area, so accepted triangles need not survive.
                centroids[count, 0] = (
                    triangle[0, 0] + triangle[1, 0] + triangle[2, 0]
                ) / 3.0
                centroids[count, 1] = (
                    triangle[0, 1] + triangle[1, 1] + triangle[2, 1]
                ) / 3.0
                first_dx = triangle[1, 0] - triangle[0, 0]
                first_dy = triangle[1, 1] - triangle[0, 1]
                second_dx = triangle[2, 0] - triangle[0, 0]
                second_dy = triangle[2, 1] - triangle[0, 1]
                area_weights[count] = 0.5 * (
                    first_dx * second_dy - first_dy * second_dx
                )
            count += 1
            continue

        start = longest_edge
        end = (start + 1) % 3
        opposite = 3 - start - end
        midpoint_x = 0.5 * (triangle[start, 0] + triangle[end, 0])
        midpoint_y = 0.5 * (triangle[start, 1] + triangle[end, 1])

        # Reclaim the consumed queue prefix before growing its capacity.
        # Keep the parent view alive while writing its two children.
        if write + 2 > len(pending):
            remaining = write - read
            capacity = len(pending)
            if remaining + 2 > capacity // 2:
                capacity *= 2
            compacted = np.empty((capacity, 3, 2), dtype=np.float64)
            compacted[:remaining] = pending[read:write]
            pending = compacted
            read = 0
            write = remaining

        pending[write, 0] = triangle[start]
        pending[write, 1, 0] = midpoint_x
        pending[write, 1, 1] = midpoint_y
        pending[write, 2] = triangle[opposite]
        pending[write + 1, 0] = pending[write, 1]
        pending[write + 1, 1] = triangle[end]
        pending[write + 1, 2] = triangle[opposite]

        # Keep the existing counter-clockwise child orientation convention.
        for index in range(write, write + 2):
            child = pending[index]
            dx1 = child[1, 0] - child[0, 0]
            dy1 = child[1, 1] - child[0, 1]
            dx2 = child[2, 0] - child[0, 0]
            dy2 = child[2, 1] - child[0, 1]
            if dx1 * dy2 - dy1 * dx2 < 0:
                x, y = child[1, 0], child[1, 1]
                child[1] = child[2]
                child[2, 0], child[2, 1] = x, y
        write += 2

    return (
        accepted[:count].copy(),
        centroids[:count].copy(),
        area_weights[:count].copy(),
    )


def _subdivide_to_quadrature(
    triangles_uv: np.ndarray,
    maximum_edge_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return accepted centroids and areas without storing accepted triangles."""
    if not np.isfinite(maximum_edge_length) or maximum_edge_length <= 0:
        raise ValueError("maximum_edge_length must be positive and finite")

    comparison_tolerance = 32.0 * np.finfo(float).eps
    edge_limit = maximum_edge_length * (1.0 + comparison_tolerance)
    _, centroids, area_weights = _subdivide_triangles(
        triangles_uv,
        edge_limit,
        emit_triangles=False,
    )
    return centroids, area_weights


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
    triangles_uv = _triangulate_cleaned_polygon(cleaned_polygon_uv)

    # Emit the reusable centroid rule directly; retaining millions of accepted
    # triangles would add a temporary six-float record for every sample.
    points_uv, area_weights = _subdivide_to_quadrature(
        triangles_uv,
        sampling_spacing,
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
