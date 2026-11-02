from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from .geometry import (
    BOUNDARY_SIMPLIFICATION_TOLERANCE_PX,
    fit_ellipse_3d,
    fit_plane,
    polygon_metrics,
    simplify_boundary_pixels,
)
from .heightmap import (
    _prepare_heightmap_with_metadata,
    compute_curvatures,
    get_z,
    surface_centroids,
)
from .labels import labels_to_objects
from .mask import mask_to_objects
from .models import DeprojResult, Epicell
from .objects import MaskObject


def _validate_common_inputs(
    segmentation: np.ndarray,
    heightmap: np.ndarray,
    *,
    segmentation_name: str,
    pixel_size: float,
    voxel_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    segmentation = np.asarray(segmentation)
    heightmap = np.asarray(heightmap)
    if segmentation.ndim != 2 or heightmap.ndim != 2:
        raise ValueError(
            f"{segmentation_name} and heightmap must both be 2-D arrays; "
            f"received shapes {segmentation.shape} and {heightmap.shape}"
        )
    if segmentation.shape != heightmap.shape:
        raise ValueError(
            f"{segmentation_name} and heightmap must have identical shapes; "
            f"got {segmentation.shape} and {heightmap.shape}"
        )
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be a positive finite number")
    if not np.isfinite(voxel_depth) or voxel_depth <= 0:
        raise ValueError("voxel_depth must be a positive finite number")
    return segmentation, heightmap


def _from_objects_and_graph(
    objects: list[MaskObject],
    graph: nx.Graph,
    heightmap: np.ndarray,
    *,
    source_shape: tuple[int, int],
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
    cell_id_policy: Literal["sequential", "source_label"] = "sequential",
) -> DeprojResult:
    """Run shared height-map deprojection from extracted 2-D objects and graph."""
    if not objects:
        raise ValueError("object parsing found no retained cells")
    median_area = np.median(
        [polygon_metrics(np.c_[obj.boundary, np.zeros(len(obj.boundary))])[2] for obj in objects]
    )
    smooth_scale = 2.0 * np.sqrt(median_area) / np.pi
    prepared, z_reflection_offset = _prepare_heightmap_with_metadata(
        heightmap,
        voxel_depth=voxel_depth,
        smooth_scale=smooth_scale,
        invert_z=invert_z,
        inpaint_zeros=inpaint_zeros,
        prune_zeros=prune_zeros,
    )
    if not prune_zeros and not np.all(np.isfinite(prepared)):
        raise ValueError(
            "heightmap contains NaN or infinite values after preprocessing; "
            "enable prune_zeros or provide a finite height map"
        )

    graph = graph.copy()
    bad_junctions: set[int] = set()
    for node_id, data in graph.nodes(data=True):
        xy = np.asarray(data["centroid"], dtype=float) * pixel_size
        z = get_z(xy[None, :], prepared, pixel_size)[0]
        data["centroid"] = np.array([xy[0], xy[1], z])
        if prune_zeros and not np.isfinite(z):
            bad_junctions.add(int(node_id))

    curvatures = compute_curvatures(prepared, smooth_scale, pixel_size)
    centroids = surface_centroids(
        prepared, [obj.footprint_rc for obj in objects], pixel_size
    )
    cells: list[Epicell] = []
    for obj, surface_centroid in zip(objects, centroids, strict=True):
        if bad_junctions.intersection(map(int, obj.junction_ids)):
            continue
        if not np.all(np.isfinite(surface_centroid)):
            if prune_zeros:
                continue
            label = f" source label {obj.source_label}" if obj.source_label is not None else ""
            raise ValueError(
                f"cell{label} footprint lacks finite height or finite-difference "
                "support for its surface centroid"
            )

        # Simplify in pixel coordinates before height-map sampling or physical
        # scaling. This keeps the tolerance independent of pixel_size and avoids
        # carrying redundant contour vertices into the 3-D geometry path.
        boundary_px = simplify_boundary_pixels(
            np.asarray(obj.boundary, dtype=float),
            tolerance_px=BOUNDARY_SIMPLIFICATION_TOLERANCE_PX,
        )
        xy = boundary_px * pixel_size
        z = get_z(xy, prepared, pixel_size)
        if prune_zeros and not np.all(np.isfinite(z)):
            continue
        boundary = np.c_[xy, z]
        area, perimeter, area2d, perimeter2d = polygon_metrics(boundary)
        rotation, euler = fit_plane(boundary)
        ellipse, eccentricity, direction = fit_ellipse_3d(boundary, rotation)
        center = boundary.mean(axis=0)
        px = int(np.clip(np.rint(center[0] / pixel_size), 0, source_shape[1] - 1))
        py = int(np.clip(np.rint(center[1] / pixel_size), 0, source_shape[0] - 1))
        local_curvature = np.array([field[py, px] for field in curvatures])
        source_label = obj.source_label
        cell_id = (
            int(source_label)
            if cell_id_policy == "source_label" and source_label is not None
            else len(cells) + 1
        )
        cells.append(
            Epicell(
                id=cell_id,
                source_label=source_label,
                boundary=boundary,
                center=center,
                junction_ids=obj.junction_ids.copy(),
                n_neighbors=len(obj.junction_ids),
                area=area,
                perimeter=perimeter,
                euler_angles=euler,
                curvatures=local_curvature,
                ellipse_fit=ellipse,
                eccentricity=eccentricity,
                proj_direction=direction,
                uncorrected_area=area2d,
                uncorrected_perimeter=perimeter2d,
                area_error=area - area2d,
                perimeter_error=perimeter - perimeter2d,
                surface_centroid=surface_centroid,
            )
        )
    graph.remove_nodes_from(bad_junctions)
    return DeprojResult(
        cells,
        graph,
        units=units,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        source_shape=source_shape,
        prepared_heightmap=prepared,
        z_reflected=bool(invert_z),
        z_reflection_offset=z_reflection_offset,
    )


def from_heightmap(
    mask: np.ndarray,
    heightmap: np.ndarray,
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
) -> DeprojResult:
    """Deproject cell contours from a binary mask onto a height-map surface.

    This is the original DeProj-style entry point. It expects a two-dimensional
    binary segmentation mask and a matching height map. Cell objects, contours,
    and junctions are extracted from the mask, then projected onto the height map
    to compute deprojected three-dimensional cell measurements.

    The function name is kept for consistency with the original MATLAB DeProj
    API. Conceptually, the input segmentation is a binary mask, so
    :func:`from_mask` is provided as an alias with the same behavior.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional binary or binary-like segmentation mask with the same
        shape as ``heightmap``. This mask should follow the original DeProj input
        convention: cell interiors are foreground and cell borders/ridges are
        background. The mask is parsed into individual cell objects by
        :func:`deprojpy.mask.mask_to_objects`.
    heightmap : np.ndarray
        Two-dimensional height map sampled on the same pixel grid as ``mask``.
        Height values are converted to physical Z units by multiplication with
        ``voxel_depth``. The height map may be optionally inverted, inpainted,
        and pruned according to ``invert_z``, ``inpaint_zeros``, and
        ``prune_zeros``.
    pixel_size : float, default: 1.0
        Physical size of one pixel in the XY plane, expressed in ``units`` per
        pixel. Boundary coordinates and junction coordinates are multiplied by
        this value before three-dimensional measurements are computed.
    voxel_depth : float, default: 1.0
        Physical size of one height-map unit in the Z direction, expressed in
        ``units`` per height-map unit. Height values are multiplied by this value
        during height-map preparation.
    units : str, default: "pixels"
        Name of the physical unit associated with ``pixel_size`` and
        ``voxel_depth``. This string is stored in the returned
        :class:`DeprojResult` and used for reporting; it does not affect the
        numerical calculations.
    invert_z : bool, default: False
        If True, invert the height map during preprocessing. Use this when the
        raw height map has the opposite orientation from the desired surface
        convention.
    inpaint_zeros : bool, default: True
        If True, fill zero-valued regions in the height map during preprocessing
        before geometric measurements are computed.
    prune_zeros : bool, default: True
        If True, discard cells and junctions whose sampled height values are
        zero, NaN, or infinite after preprocessing. If False, non-finite values
        remaining after preprocessing raise an error.

    Returns
    -------
    DeprojResult
        Result object containing the deprojected cells, the junction graph, the
        prepared height map, source image shape, unit metadata, and measurement
        metadata. Cell identifiers are assigned sequentially in this binary-mask
        pathway.

    Raises
    ------
    ValueError
        If ``mask`` and ``heightmap`` are not two-dimensional arrays with
        identical shapes; if ``pixel_size`` or ``voxel_depth`` is not positive
        and finite; if mask parsing finds no retained cells; or if the height map
        contains invalid values that cannot be handled with the requested
        preprocessing options.

    Notes
    -----
    The XY and Z calibration parameters are multiplicative scale factors. For
    example, if microscope metadata gives ``xy_scale`` in pixels per micrometre
    and ``z_scale`` in height-map units per micrometre, then use
    ``pixel_size = 1 / xy_scale`` and ``voxel_depth = 1 / z_scale``.

    This function differs from :func:`from_labels`, which expects a dense
    integer label image where each cell has its own label. In the binary-mask
    pathway, cell identifiers are assigned sequentially after object parsing.
    Surface centroids use the dilated component already used for boundary
    tracing; neighboring cells can therefore share ridge pixels.

    See Also
    --------
    from_mask : Alias of ``from_heightmap`` with a clearer name.
    from_labels : Deproject cells from a dense integer label image.
    deprojpy.mask.mask_to_objects : Parse a binary DeProj-style mask into cell
        objects and a junction graph.

    Examples
    --------
    Run DeProjPy on an original DeProj-style binary segmentation mask and a
    matching height map:

    >>> import tifffile as tiff
    >>> import deprojpy as dp
    >>> mask = tiff.imread("segmentation.tif")
    >>> heightmap = tiff.imread("heightmap.tif")
    >>> result = dp.from_heightmap(
    ...     mask,
    ...     heightmap,
    ...     pixel_size=0.183,
    ...     voxel_depth=0.5,
    ...     units="µm",
    ... )
    >>> table = result.to_dataframe()
    >>> table.head()

    The same call can be written using the clearer alias:

    >>> result = dp.from_mask(
    ...     mask,
    ...     heightmap,
    ...     pixel_size=0.183,
    ...     voxel_depth=0.5,
    ...     units="µm",
    ... )
    """
    mask, heightmap = _validate_common_inputs(
        mask,
        heightmap,
        segmentation_name="mask",
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
    )
    objects, graph = mask_to_objects(mask)
    return _from_objects_and_graph(
        objects,
        graph,
        heightmap,
        source_shape=mask.shape,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        units=units,
        invert_z=invert_z,
        inpaint_zeros=inpaint_zeros,
        prune_zeros=prune_zeros,
        cell_id_policy="sequential",
    )

from_mask = from_heightmap

def from_labels(
    labels: np.ndarray,
    heightmap: np.ndarray,
    pixel_size: float = 1.0,
    voxel_depth: float = 1.0,
    units: str = "pixels",
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
    *,
    background: int = 0,
    drop_border_cells: bool = True,
    min_junction_labels: int = 3,
    junction_background: int | None = None,
    junction_connectivity: int = 2,
    junction_merge_epsilon: float = 0.0,
) -> DeprojResult:
    """Deproject labeled-cell contours onto a height-map surface.

    This is the main entry point for running DeProjPy from an integer labeled
    segmentation image. Each non-background label is interpreted as one cell.
    Cell contours are extracted from the label image, multi-cell junctions are
    detected from local label neighborhoods, and the resulting 2-D cell geometry
    is projected onto the supplied height map to compute deprojected 3-D cell
    measurements.

    In contrast to :func:`from_heightmap`, which expects a binary ridge/interior
    segmentation in the style of the original MATLAB DeProj input, this function
    expects a dense label image where pixels belonging to the same cell share the
    same integer label. By default, cells touching either the explicit background
    or the image border are discarded before deprojection, but this behavior can
    be toggled.

    Parameters
    ----------
    labels : np.ndarray
        Two-dimensional integer label image. Each non-background value identifies
        one segmented cell. The array must have the same shape as ``heightmap``.
        Floating-point arrays may be accepted only if they contain finite,
        integer-like values that can be safely converted to integer labels by
        the label-image validation routine.
    heightmap : np.ndarray
        Two-dimensional height map sampled on the same pixel grid as ``labels``.
        Height values are converted to physical Z units by multiplication with
        ``voxel_depth``. The height map may be optionally inverted, inpainted,
        and pruned according to ``invert_z``, ``inpaint_zeros``, and
        ``prune_zeros``.
    pixel_size : float, default: 1.0
        Physical size of one pixel in the XY plane, expressed in ``units`` per
        pixel. Boundary coordinates and junction coordinates are multiplied by
        this value before 3-D measurements are computed.
    voxel_depth : float, default: 1.0
        Physical size of one height-map unit in the Z direction, expressed in
        ``units`` per height-map unit. Height values are multiplied by this
        value during height-map preparation.
    units : str, default: "pixels"
        Name of the physical unit associated with ``pixel_size`` and
        ``voxel_depth``. This string is stored in the returned
        :class:`DeprojResult` and used for reporting; it does not affect the
        numerical calculations.
    invert_z : bool, default: False
        If True, invert the height map during preprocessing. Use this when the
        raw height map has the opposite orientation from the desired surface
        convention.
    inpaint_zeros : bool, default: True
        If True, fill zero-valued regions in the height map during preprocessing
        before geometric measurements are computed.
    prune_zeros : bool, default: True
        If True, discard cells and junctions whose sampled height values are
        zero, NaN, or infinite after preprocessing. If False, non-finite values
        remaining after preprocessing raise an error.
    background : int, default: 0
        Label value used for background pixels in ``labels``. Background is not
        treated as a cell.
    drop_border_cells : bool, default: True
        If True, discard cells that touch the background or the outer image
        boundary before deprojection.
    min_junction_labels : int, default: 3
        Minimum number of distinct labels that must meet in a local neighborhood
        for a pixel cluster to be considered a junction. The default detects
        ordinary multicellular junctions where at least three regions meet.
    junction_background : int or None, default: None
        Background value to use during junction detection. If None, background
        pixels are included according to the junction-detection routine's
        default behavior. Pass ``background`` to explicitly treat the same value
        as background during junction detection.
    junction_connectivity : int, default: 2
        Connectivity used to cluster junction pixels. This is passed to the
        underlying label-image junction detection routine. A value of ``2``
        corresponds to 8-connectivity in two dimensions.
    junction_merge_epsilon : float, default: 0.0
        Distance threshold, in pixels, for merging nearby detected junctions.
        The default disables post-detection merging.

    Returns
    -------
    DeprojResult
        Result object containing the deprojected cells, the junction graph, the
        prepared height map, source image shape, unit metadata, and measurement
        metadata. Cell identifiers are derived from the original source labels
        when using this entry point.

    Raises
    ------
    ValueError
        If ``labels`` and ``heightmap`` are not two-dimensional arrays with
        identical shapes; if ``pixel_size`` or ``voxel_depth`` is not positive
        and finite; if the label image cannot be validated as an integer label
        image; if no cells remain after label parsing and filtering; or if the
        height map contains invalid values that cannot be handled with the
        requested preprocessing options.

    Notes
    -----
    The XY and Z calibration parameters are multiplicative scale factors. For
    example, if microscope metadata gives ``xy_scale`` in pixels per micrometre
    and ``z_scale`` in height-map units per micrometre, then use
    ``pixel_size = 1 / xy_scale`` and ``voxel_depth = 1 / z_scale``.

    The labeled-image pathway preserves source labels in the returned cells.
    This differs from the binary-mask pathway, where cell identifiers are
    assigned sequentially. Surface centroids use pixels belonging to each
    original source label.

    See Also
    --------
    from_heightmap : Deproject the original binary-ridge DeProj-style input.
    labelimage_tools.validate_label_image : Validate integer-like label images.
    labelimage_tools.ordered_contours_from_labels : Extract ordered contours
        from labeled images.
    labelimage_tools.junctions_from_labels : Detect multicellular junctions
        from labeled images.

    Examples
    --------
    Run DeProjPy on a labeled segmentation and a matching height map with
    physical calibration in micrometres:

    >>> import tifffile as tiff
    >>> import deprojpy as dp
    >>> labels = tiff.imread("labels.tif")
    >>> heightmap = tiff.imread("heightmap.tif")
    >>> result = dp.from_labels(
    ...     labels,
    ...     heightmap,
    ...     pixel_size=0.183,
    ...     voxel_depth=0.5,
    ...     units="µm",
    ...     background=0,
    ... )
    >>> table = result.to_dataframe()
    >>> table.head()
    """
    labels, heightmap = _validate_common_inputs(
        labels,
        heightmap,
        segmentation_name="labels",
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
    )
    objects, graph = labels_to_objects(
        labels,
        background=background,
        drop_border_cells=drop_border_cells,
        min_junction_labels=min_junction_labels,
        junction_background=junction_background,
        junction_connectivity=junction_connectivity,
        junction_merge_epsilon=junction_merge_epsilon,
    )
    return _from_objects_and_graph(
        objects,
        graph,
        heightmap,
        source_shape=labels.shape,
        pixel_size=pixel_size,
        voxel_depth=voxel_depth,
        units=units,
        invert_z=invert_z,
        inpaint_zeros=inpaint_zeros,
        prune_zeros=prune_zeros,
        cell_id_policy="source_label",
    )
