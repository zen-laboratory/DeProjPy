"""Plotting adapters for intensity measurements.

Matplotlib is imported only when this module is requested through a plotting
entry point; numerical intensity workflows do not import pyplot.
"""

from __future__ import annotations

import numpy as np

from ._models import MeasurementResult, PreparedRegions


def plot_feature_map(
    result,
    measurements: MeasurementResult,
    *,
    channel=0,
    feature: str = "mean",
    **kwargs,
):
    """Plot one measured feature after strict channel/identifier alignment.

    Raises
    ------
    ValueError
        If the channel is absent, IDs are duplicated, or measurement IDs do
        not match the result's cells.
    """
    from ..plotting import plot_values_as_feature_map

    table = measurements.table
    # Compare identifiers directly: integer ``1`` and string ``"1"`` are
    # distinct persisted channels and must not be merged for convenience.
    channel_rows = table[table["channel"] == channel]
    if channel_rows.empty:
        available = list(dict.fromkeys(table["channel"].tolist()))
        raise ValueError(
            f"channel {channel!r} is absent from measurements; available: {available}"
        )
    if feature not in channel_rows.columns:
        raise ValueError(f"measurement feature {feature!r} is not present in the table")
    if channel_rows["id"].duplicated().any():
        raise ValueError(
            "measurement table contains duplicate cell identifiers for this channel"
        )

    # Require an exact ID join. Silently painting a missing measurement as a
    # missing color could hide a failed upstream row-retention invariant.
    result_ids = [cell.id for cell in result.epicells]
    measured_ids = set(channel_rows["id"].tolist())
    missing_ids = sorted(set(result_ids) - measured_ids)
    extra_ids = sorted(measured_ids - set(result_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "measurement/result cell identifiers do not match; "
            f"missing={missing_ids}, extra={extra_ids}"
        )
    values_by_id = channel_rows.set_index("id")[feature]
    aligned_values = np.asarray(
        [values_by_id.loc[cell.id] for cell in result.epicells],
        dtype=float,
    )
    kwargs.setdefault("colorbar_label", feature)
    return plot_values_as_feature_map(result, aligned_values, **kwargs)


def plot_region_preview(
    result,
    regions: PreparedRegions,
    *,
    cell_ids=None,
    ax=None,
    normal_scale=None,
):
    """Preview selected fitted prisms and tissue-facing normals in XYZ.

    Both requested depth faces and corresponding side edges are drawn. The
    source cell boundary is overlaid as a dashed reference. Invalid selected
    cells retain that reference and receive a diagnostic text label.

    Parameters
    ----------
    result
        Deprojection result used for original 3-D boundary context.
    regions
        Prepared prisms to preview.
    cell_ids
        Optional iterable of stable cell IDs. Unknown IDs are rejected.
    ax
        Optional Matplotlib 3-D axis.
    normal_scale
        Arrow length in result units. By default it spans the larger of the
        requested depth magnitude and one sampling spacing.

    Returns
    -------
    figure, axis
        Matplotlib parent figure and 3-D axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        figure = plt.figure(layout="constrained")
        axis = figure.add_subplot(projection="3d")
    else:
        axis = ax
        figure = axis.figure

    result_cells = {cell.id: cell for cell in result.epicells}
    region_ids = {region.cell_id for region in regions.cells}
    wanted_ids = region_ids if cell_ids is None else set(cell_ids)
    unknown_ids = sorted(wanted_ids - region_ids)
    if unknown_ids:
        raise ValueError(f"unknown cell_ids for prism preview: {unknown_ids}")

    if normal_scale is None:
        requested_depth_scale = max(
            (
                max(abs(region.start_depth), abs(region.end_depth))
                for region in regions.cells
                if region.status == "ok" and region.cell_id in wanted_ids
            ),
            default=0.0,
        )
        scale = max(requested_depth_scale, regions.sampling_spacing)
    else:
        scale = float(normal_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("normal_scale must be positive and finite")

    plotted_points = []
    for region in regions.cells:
        if wanted_ids is not None and region.cell_id not in wanted_ids:
            continue
        source_cell = result_cells.get(region.cell_id)
        if source_cell is not None:
            source_boundary_xyz = np.asarray(source_cell.boundary, dtype=float)
            closed_source_xyz = np.vstack([source_boundary_xyz, source_boundary_xyz[0]])
            axis.plot(*closed_source_xyz.T, color="0.45", linestyle="--", linewidth=0.8)
            plotted_points.append(source_boundary_xyz)

        if region.status != "ok":
            if source_cell is not None:
                center_xyz = np.mean(source_boundary_xyz, axis=0)
                axis.text(*center_xyz, f"{region.cell_id}: invalid", color="tab:red")
            continue

        # Start/end faces share the fitted polygon and differ only by signed
        # displacement along the biologically oriented unit normal.
        fitted_boundary_xyz = (
            region.plane_origin_xyz
            + region.polygon_uv @ region.plane_basis_xyz.T
        )
        start_face_xyz = fitted_boundary_xyz + region.start_depth * region.normal_xyz
        end_face_xyz = fitted_boundary_xyz + region.end_depth * region.normal_xyz
        for face_xyz, color in (
            (start_face_xyz, "tab:blue"),
            (end_face_xyz, "tab:orange"),
        ):
            closed_face_xyz = np.vstack([face_xyz, face_xyz[0]])
            axis.plot(*closed_face_xyz.T, color=color, linewidth=1.2)
        for start_xyz, end_xyz in zip(start_face_xyz, end_face_xyz, strict=True):
            axis.plot(
                [start_xyz[0], end_xyz[0]],
                [start_xyz[1], end_xyz[1]],
                [start_xyz[2], end_xyz[2]],
                color="0.35",
                linewidth=0.7,
            )

        # The arrow always points toward positive depth/tissue, independent of
        # whether the requested interval itself begins at a negative depth.
        axis.quiver(
            *region.plane_origin_xyz,
            *(region.normal_xyz * scale),
            color="tab:red",
        )
        plotted_points.extend((start_face_xyz, end_face_xyz))

    # Matplotlib's default 3-D box distorts physical angles when XYZ ranges
    # differ. Match box aspect to the plotted physical extents instead.
    if plotted_points:
        all_points_xyz = np.vstack(plotted_points)
        extent_xyz = np.ptp(all_points_xyz, axis=0)
        extent_xyz[extent_xyz == 0] = 1.0
        axis.set_box_aspect(extent_xyz)

    axis.set(
        xlabel=f"x ({regions.units})",
        ylabel=f"y ({regions.units})",
        zlabel=f"z ({regions.units})",
        title="Surface-normal cell prisms (normal points into tissue)",
    )
    return figure, axis
