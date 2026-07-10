from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import labelimage_tools as lit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .models import DeprojResult, Epicell

CMAP = plt.colormaps.get_cmap("viridis")
EPICELL_FEATURES = {field.name for field in fields(Epicell)}
ERROR_METRICS = {
    "area": ("area_error", "uncorrected_area"),
    "perimeter": ("perimeter_error", "uncorrected_perimeter"),
}
FEATURE_COMPONENTS = {
    # EpiCell.curvatures = (mean, gaussian, k1, k2)
    "curvature_mean": ("curvatures", 0),
    "curvature_gaussian": ("curvatures", 1),
    "curvature_k1": ("curvatures", 2),
    "curvature_k2": ("curvatures", 3),
}


def _normalizer(
    values: np.ndarray, vmin: float | None = None, vmax: float | None = None,
    norm: Normalize | None = None ) -> Normalize:
    """Return a matplotlib Normalize object for the given values and limits."""
    finite = values[np.isfinite(values)]

    if norm is not None:
        if vmin is not None or vmax is not None:
            raise ValueError(
                "Pass either norm or vmin/vmax, not both. "
                "If you need custom limits, put them in the norm itself, e.g. "
                "Normalize(vmin=..., vmax=...) or TwoSlopeNorm(vcenter=0, vmin=..., vmax=...)."
            )
        if finite.size:
            norm.autoscale_None(finite)
        return norm

    low = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    high = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)

    if low == high:
        high = low + 1.0

    return Normalize(low, high)

def _feature_values(result: DeprojResult, feature: str) -> np.ndarray:
    """Return one scalar value per Epicell for color normalization."""
    return np.asarray(
        [_cell_feature_value(cell, feature) for cell in result.epicells],
        dtype=float,
    )


def _cell_feature_value(cell, feature: str) -> float:
    """Return one scalar value from an EpiCell, including named array components."""
    if feature in FEATURE_COMPONENTS:
        attr, idx = FEATURE_COMPONENTS[feature]
        return float(np.asarray(getattr(cell, attr))[idx])
    if not hasattr(cell, feature):
        valid = sorted(EPICELL_FEATURES | set(FEATURE_COMPONENTS))
        raise ValueError(
            f"unknown Epicell feature {feature!r}; valid features are:\n  "
            + "\n  ".join(valid)
        )

    value = getattr(cell, feature)
    arr = np.asarray(value)

    if arr.ndim != 0:
        valid = [
            name
            for name, val in vars(cell).items()
            if np.asarray(val).ndim == 0
        ]
        valid += list(FEATURE_COMPONENTS)
        valid = sorted(valid)

        raise ValueError(
            f"Feature {feature!r} is not scalar and cannot be plotted directly.\n"
            f"Valid features are:\n  " + "\n  ".join(valid)
        )

    return float(arr)

def _relative_error_values(result: DeprojResult, metric: str, as_percent: bool) -> np.ndarray:
    if metric not in ERROR_METRICS:
        known = ", ".join(sorted(ERROR_METRICS))
        raise ValueError(f"unknown relative-error metric {metric!r}; choose one of: {known}")
    numerator, denominator = ERROR_METRICS[metric]
    absolute = _feature_values(result, numerator)
    baseline = _feature_values(result, denominator)
    values = np.divide(
        absolute,
        baseline,
        out=np.full_like(absolute, np.nan, dtype=float),
        where=baseline != 0,
    )
    if as_percent:
        values = values * 100.0
    return values

def _parent_figure(ax):
    return ax.figure


def plot_mask_objects(
    mask: np.ndarray,
    result: DeprojResult,
    *,
    ax=None,
    title: str | None = None,
    linewidth: float = 0.5,
    boundary_color=None,
):
    """Plot retained boundaries over the input mask."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    ax.imshow(mask, cmap="gray")
    for cell in result.epicells:
        p = cell.boundary / result.pixel_size
        ax.plot(p[:, 0], p[:, 1], linewidth=linewidth, color=boundary_color)
    ax.set_title(title or f"Parsed mask: {len(result.epicells)} retained cells")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax


def plot_label_objects(
    labels: np.ndarray,
    result: DeprojResult | None = None,
    *,
    ax=None,
    background: int = 0,
    use_graph_coloring: bool = True,
    K: int = 8,
    seed: int | None = None,
    title: str | None = None,
    show_junctions: bool = True,
    show_cell_boundaries: bool = True,
    cmap: str | None = None,
    cyclic_cmap: bool = False,
):
    """Plot a labeled image with optional DeProjPy boundaries and junctions."""
    fig, ax = lit.plot_label_image(
        labels,
        ax=ax,
        background=background,
        use_graph_coloring=use_graph_coloring,
        K=K,
        seed=seed,
        title=title or "Labeled cell image",
        cmap=cmap,
        cyclic_cmap=cyclic_cmap
    )
    if show_cell_boundaries:
        lit.plot_contours(labels, ax=ax, background=background, color="black", linewidth=0.5)
    if result is not None:
        for cell in result.epicells:
            p = cell.boundary[:, :2] / result.pixel_size
            ax.plot(p[:, 0], p[:, 1], color="white", linewidth=0.6, alpha=0.8)
        if show_junctions:
            centroids = []
            for _, data in result.junction_graph.nodes(data=True):
                if "centroid" in data:
                    centroids.append(np.asarray(data["centroid"][:2]) / result.pixel_size)
            if centroids:
                xy = np.asarray(centroids)
                ax.scatter(xy[:, 0], xy[:, 1], s=12, c="red", edgecolors="white", linewidths=0.4)
    return fig, ax


def plot_heightmap_with_centers(
    heightmap: np.ndarray,
    result: DeprojResult,
    *,
    ax=None,
    cmap="viridis",
    title: str | None = None,
    center_size: float = 3,
    center_color="red",
    colorbar: bool = True,
    colorbar_label: str = "height-map value",
):
    """Plot the input height map and retained cell centers."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    image = ax.imshow(heightmap, cmap=cmap)
    centers = np.asarray(
        [c.center[:2] / result.pixel_size for c in result.epicells], dtype=float
    ).reshape(-1, 2)
    if len(centers):
        ax.scatter(centers[:, 0], centers[:, 1], s=center_size, c=center_color)
    if colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title or "Height map and retained cell centers")
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")
    return fig, ax

def plot_curvature_map(
    heightmap: np.ndarray,
    result: DeprojResult | None = None,
    *,
    kind: str = "mean",
    ax=None,
    prepared: bool = False,
    pixel_size: float | None = None,
    voxel_depth: float | None = None,
    units: str | None = None,
    smooth_scale: float = -1.0,
    invert_z: bool = False,
    inpaint_zeros: bool = True,
    prune_zeros: bool = True,
    object_scale: float | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust_percentile: float | None = 99.0,
    symmetric: bool | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    show_centers: bool = False,
    center_size: float = 3.0,
    center_color="black",
):
    """
    Plot a curvature-derived map from a height map.

    Parameters
    ----------
    heightmap
        Raw input height map, unless ``prepared=True``. If raw, the same
        preparation logic as DeProjPy is applied before curvature calculation.

    result
        Optional DeProjResult. If provided, ``pixel_size``, ``voxel_depth``,
        and ``units`` default to values stored in the result.

    kind
        One of:

        - ``"mean"``: mean curvature
        - ``"gaussian"``: Gaussian curvature
        - ``"k1"``: first principal curvature
        - ``"k2"``: second principal curvature
        - ``"slope"``: height-map gradient magnitude

    prepared
        If False, run ``prepare_heightmap`` before computing curvature.
        If True, assume ``heightmap`` is already in physical z units.

    pixel_size
        Physical pixel size in x/y units. Defaults to ``result.pixel_size``
        when result is provided, otherwise 1.

    voxel_depth
        Physical z scaling. Defaults to 1, or to ``result.voxel_depth`` if
        present. Ignored when ``prepared=True``.

    units
        Unit label for axes/colorbar. Defaults to ``result.units`` when
        available.

    smooth_scale, invert_z, inpaint_zeros, prune_zeros
        Passed to ``prepare_heightmap`` when ``prepared=False``.

    object_scale
        Optional smoothing scale passed to ``compute_curvatures``.

    robust_percentile
        If provided and vmin/vmax are not explicitly given, use symmetric or
        direct percentile limits to reduce the effect of outliers.

    symmetric
        If True, use symmetric color limits around zero. By default this is
        True for signed curvature maps and False for slope.

    show_centers
        If True and result is provided, overlay cell centers.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    from .heightmap import compute_curvatures, prepare_heightmap

    valid_kinds = {"mean", "gaussian", "k1", "k2", "slope"}
    if kind not in valid_kinds:
        known = ", ".join(sorted(valid_kinds))
        raise ValueError(f"unknown curvature map kind {kind!r}; choose one of: {known}")

    if result is not None:
        if pixel_size is None:
            pixel_size = result.pixel_size
        if units is None:
            units = result.units
        if voxel_depth is None and hasattr(result, "voxel_depth"):
            voxel_depth = result.voxel_depth

    if pixel_size is None:
        pixel_size = 1.0
    if voxel_depth is None:
        voxel_depth = 1.0
    if units is None:
        units = "a.u."

    if prepared:
        h = np.asarray(heightmap, dtype=float)
    else:
        h = prepare_heightmap(
            heightmap,
            voxel_depth=voxel_depth,
            smooth_scale=smooth_scale,
            invert_z=invert_z,
            inpaint_zeros=inpaint_zeros,
            prune_zeros=prune_zeros,
        )

    if kind == "slope":
        hy, hx = np.gradient(h, pixel_size, pixel_size)
        values = np.sqrt(hx * hx + hy * hy)
    else:
        mean, gaussian, k1, k2 = compute_curvatures(
            h,
            object_scale=object_scale,
            pixel_size=pixel_size,
        )
        values_by_kind = {
            "mean": mean,
            "gaussian": gaussian,
            "k1": k1,
            "k2": k2,
        }
        values = values_by_kind[kind]

    if symmetric is None:
        symmetric = kind != "slope"

    if cmap is None:
        cmap = "viridis" if kind == "slope" else "coolwarm"

    finite = values[np.isfinite(values)]

    if vmin is None or vmax is None:
        if finite.size == 0:
            auto_vmin, auto_vmax = 0.0, 1.0
        elif robust_percentile is not None:
            p = float(robust_percentile)
            if symmetric:
                limit = float(np.nanpercentile(np.abs(finite), p))
                if limit == 0:
                    limit = 1.0
                auto_vmin, auto_vmax = -limit, limit
            else:
                low = float(np.nanpercentile(finite, 100.0 - p))
                high = float(np.nanpercentile(finite, p))
                if low == high:
                    high = low + 1.0
                auto_vmin, auto_vmax = low, high
        else:
            auto_vmin = float(np.nanmin(finite))
            auto_vmax = float(np.nanmax(finite))
            if symmetric:
                limit = max(abs(auto_vmin), abs(auto_vmax))
                if limit == 0:
                    limit = 1.0
                auto_vmin, auto_vmax = -limit, limit
            elif auto_vmin == auto_vmax:
                auto_vmax = auto_vmin + 1.0

        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)

    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    if show_centers:
        if result is None:
            raise ValueError("show_centers=True requires result=...")
        centers = np.asarray(
            [cell.center[:2] / result.pixel_size for cell in result.epicells],
            dtype=float,
        ).reshape(-1, 2)
        if len(centers):
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                s=center_size,
                c=center_color,
            )

    label_by_kind = {
        "mean": "mean curvature",
        "gaussian": "Gaussian curvature",
        "k1": "principal curvature k1",
        "k2": "principal curvature k2",
        "slope": "slope magnitude",
    }

    units_by_kind = {
        "mean": f"1/{units}",
        "gaussian": f"1/{units}²",
        "k1": f"1/{units}",
        "k2": f"1/{units}",
        "slope": "dimensionless",
    }

    ax.set_title(title or label_by_kind[kind].capitalize())
    ax.set_xlabel("column / x (pixels)")
    ax.set_ylabel("row / y (pixels)")
    ax.set_aspect("equal")

    if colorbar:
        default_label = f"{label_by_kind[kind]} ({units_by_kind[kind]})"
        fig.colorbar(image, ax=ax, label=colorbar_label or default_label)

    return fig, ax

def plot_feature_map(
    result: DeprojResult,
    feature: str = "area",
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
):
    """Color cell polygons by a scalar :class:`Epicell` feature."""
    values = _feature_values(result, feature)
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    normalizer = _normalizer(values, vmin, vmax, norm)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell, value in zip(result.epicells, values, strict=True):
        p = cell.boundary
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
        )

    ax.set_title(title or f"Cell feature: {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_aspect("equal")

    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label or feature)
    return fig, ax


def _plot_values_as_feature_map(
    result: DeprojResult,
    values: np.ndarray,
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = _parent_figure(ax)
    normalizer = _normalizer(values, vmin, vmax, norm)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell, value in zip(result.epicells, values, strict=True):
        p = cell.boundary
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.fill(
            p[:, 0],
            p[:, 1],
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_title(title or "Cell feature map")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_aspect("equal")

    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label)
    return fig, ax


def plot_relative_error_map(
    result: DeprojResult,
    metric: str = "area",
    *,
    ax=None,
    cmap="magma",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    edgecolor: str | None = None,
    linewidth: float = 0.0,
    alpha: float = 1,
    missing_color="0.7",
    as_percent: bool = True,
):
    """Color cell polygons by relative area or perimeter deprojection error.

    ``metric`` must be ``"area"`` or ``"perimeter"``. Values are computed as
    ``(corrected - projected) / projected`` from the corresponding absolute
    error and uncorrected measurement fields. By default values are shown as
    percentages.
    """
    values = _relative_error_values(result, metric, as_percent=as_percent)
    unit = "%" if as_percent else "fraction"
    return _plot_values_as_feature_map(
        result,
        values,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title or f"Relative {metric} error",
        colorbar=colorbar,
        colorbar_label=colorbar_label or f"relative {metric} error ({unit})",
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        missing_color=missing_color,
    )


def plot_3d_boundaries(
    result: DeprojResult,
    feature: str = "area",
    *,
    ax=None,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    title: str | None = None,
    linewidth: float = 0.6,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    missing_color="0.7",
):
    """Plot deprojected cell boundaries in three dimensions."""
    values = _feature_values(result, feature)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = _parent_figure(ax)
        if not hasattr(ax, "get_zlim"):
            raise ValueError("plot_3d_boundaries requires a 3D matplotlib axis")
    normalizer = _normalizer(values, vmin, vmax, norm)
    color_map = plt.colormaps.get_cmap(cmap)
    for cell, value in zip(result.epicells, values, strict=True):
        p = np.vstack([cell.boundary, cell.boundary[0]])
        color = color_map(normalizer(value)) if np.isfinite(value) else missing_color
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=linewidth)
    if colorbar:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        fig.colorbar(scalar_map, ax=ax, label=colorbar_label or feature, shrink=0.7)
    ax.set_title(title or f"Deprojected boundaries colored by {feature}")
    ax.set_xlabel(f"x ({result.units})")
    ax.set_ylabel(f"y ({result.units})")
    ax.set_zlabel(f"z ({result.units})")  # type: ignore[assignment]
    if result.epicells:
        ax.set_box_aspect((1, 1, 0.5))  # type: ignore[assignment]
    return fig, ax


def _integer_bins(values: np.ndarray) -> np.ndarray | str:
    finite = values[np.isfinite(values)]
    if not finite.size or not np.allclose(finite, np.round(finite)):
        return "auto"
    low = int(np.floor(finite.min()))
    high = int(np.ceil(finite.max()))
    return np.arange(low - 0.5, high + 1.5, 1.0)


def plot_feature_histograms(
    dataframe,
    *,
    columns=("area", "perimeter", "eccentricity", "n_neighbors"),
    bins="auto",
    infer_integer_columns: bool = True,
    title: str | None = "DeProj morphology distributions",
    figsize=None,
):
    """Plot compact histograms of core morphology measurements."""
    columns = tuple(columns)
    count = max(len(columns), 1)
    ncols = min(3, count)
    nrows = int(np.ceil(count / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        ax.set_title(column.replace("_", " "))
        ax.set_ylabel("cell count")
        if column not in dataframe:
            ax.text(0.5, 0.5, "Missing column", ha="center", va="center")
            continue
        values = dataframe[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
            continue
        column_bins = (
            _integer_bins(finite)
            if infer_integer_columns and bins == "auto"
            else bins
        )
        ax.hist(finite, bins=column_bins)
    for ax in axes.flat[len(columns) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def save_plots(
    directory,
    mask: np.ndarray | None = None,
    heightmap: np.ndarray | None = None,
    result: DeprojResult | None = None,
    dataframe=None,
    *,
    labels: np.ndarray | None = None,
    features=("area",),
    dpi: int = 150,
    figsize=None,
    close: bool = True,
) -> list[Path]:
    """Save the standard plot bundle and return written paths."""
    if heightmap is None or result is None:
        raise ValueError("save_plots requires heightmap=... and result=...")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dataframe is None:
        dataframe = result.to_dataframe()

    plot_specs = [
        ("heightmap_centers.png", lambda ax: plot_heightmap_with_centers(heightmap, result, ax=ax)),
        ("feature_histograms.png", lambda ax: plot_feature_histograms(dataframe)),
        ("boundaries_3d.png", lambda ax: plot_3d_boundaries(result, "area")),
    ]
    if labels is not None:
        plot_specs.insert(
            0,
            ("label_objects.png", lambda ax: plot_label_objects(labels, result, ax=ax)),
        )
    elif mask is not None:
        plot_specs.insert(
            0,
            ("mask_objects.png", lambda ax: plot_mask_objects(mask, result, ax=ax)),
        )
    else:
        raise ValueError("save_plots requires either mask=... or labels=...")
    plot_specs.extend(
        (
            f"{feature}_map.png",
            lambda ax, feature=feature: plot_feature_map(result, feature, ax=ax), # type: ignore (result is not None)
        )
        for feature in features
    )

    written = []
    for filename, make_plot in plot_specs:
        placeholder_figure, ax = plt.subplots(figsize=figsize, layout="constrained")
        figure, _ = make_plot(ax)
        path = directory / filename
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(figure)
            if placeholder_figure is not figure:
                plt.close(placeholder_figure)
        written.append(path)
    return written
