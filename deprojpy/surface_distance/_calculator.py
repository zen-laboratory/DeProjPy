from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..heightmap import prepare_heightmap
from ..models import DeprojResult
from ._boundary_surface import heightmap_from_cell_boundaries
from ._helpers import (
    _as_pixel_xy,
    _validate_positive,
    _validate_xy_array,
    cell_centers_xy_pixels,
    sample_height_at_xy,
)
from ._straight import (
    _pairwise_surface_straight_distances_numba,
    _selected_surface_straight_distances_numba,
    surface_straight_distance,
)


@dataclass
class SurfaceDistanceCalculator:
    """
    Reusable calculator for distances on a height-map surface.

    This class stores the height map and physical scale factors needed to
    evaluate distances on the height-field surface
    ``r(x, y) = (x, y, h(x, y))``. It is intended to be constructed once and
    reused for many distance queries, especially all-pairs distances between
    cell centers.

    Parameters
    ----------
    heightmap : np.ndarray
        Two-dimensional height map. Arrays are indexed in image order as
        ``heightmap[row=y, column=x]``. Public point inputs to calculator
        methods are geometric ``(x, y)`` coordinates.
    pixel_size : float
        Physical size of one pixel in the ``x`` and ``y`` directions. If
        ``units="µm"``, for example, distances returned by the calculator are
        in micrometers.
    voxel_depth : float, optional
        Physical scale factor applied to height-map values. Use ``1.0`` when
        ``heightmap`` is already prepared in physical z units. When using
        :meth:`from_heightmap` or :meth:`from_result` with ``prepared=False``,
        the returned calculator stores a prepared physical-height map and sets
        ``voxel_depth`` to ``1.0`` internally.
    units : str, optional
        Text label for the physical distance units. This does not change
        calculations; it is stored for display and examples.
    prepared : bool, optional
        Whether the stored ``heightmap`` is already the array used directly for
        distance calculations. Direct construction does not call
        :func:`deprojpy.heightmap.prepare_heightmap`; use
        :meth:`from_heightmap` or :meth:`from_result` to prepare raw maps.
    result : object, optional
        Optional DeProjPy result-like object. When present, methods such as
        :meth:`straight_pairwise_distances` can use its cell centers if explicit
        ``points_xy`` are not supplied.

    Coordinate and unit conventions
    -------------------------------
    Public xy inputs use image pixel coordinates by default, with ``x`` as
    column and ``y`` as row. Pass ``input_units="physical"`` when points are in
    the same physical units as result boundaries and centers. Cell centers
    stored in :class:`deprojpy.models.Epicell` are physical ``(x, y, z)``
    coordinates; use :func:`cell_centers_xy_pixels` for pixel-coordinate center
    arrays.

    Methods return physical distances. In the ``x`` and ``y`` directions,
    pixel-coordinate differences are multiplied by ``pixel_size``. In the
    ``z`` direction, sampled height-map differences are multiplied by
    ``voxel_depth`` unless the map was already prepared into physical units.

    Main methods
    ------------
    ``sample_height(xy, input_units="pixel")``
        Bilinearly sample the calculator's height map at ``(x, y)`` coordinates
        and return physical z values.
    ``straight_distance(p0_xy, p1_xy, ...)``
        Follow the straight segment between two ``xy`` points, lift it onto the
        surface, and return the 3-D polyline length. This is useful and fast,
        but it is not a shortest-path geodesic.
    ``straight_pairwise_distances(points_xy=None, result=None, ...)``
        Return a symmetric ``N × N`` matrix of straight-line surface distances.
        If ``points_xy`` is omitted, cell centers are taken from the stored
        result or from the ``result`` argument.
    ``straight_distances_for_pairs(points_xy, pairs, ...)``
        Compute distances only for selected index pairs, where ``pairs`` has
        shape ``(n_pairs, 2)`` and indexes rows of ``points_xy``.

    Coordinate and unit conventions
    -------------------------------
    Public xy inputs use image pixel coordinates by default, with ``x`` as
    column and ``y`` as row. Pass ``input_units="physical"`` when points are in
    the same physical units as result boundaries and centers. Internally, all
    sampling still occurs in pixel coordinates. Distances are always returned in
    physical units using ``pixel_size`` for ``x/y`` and ``voxel_depth`` for
    height differences.

    Construction helpers
    --------------------
    Use :meth:`from_result` when a DeProjPy result is available; it infers
    ``pixel_size``, ``voxel_depth``, and ``units`` from the result while still
    building the surface from a height map. Use :meth:`from_cell_boundaries`
    when distances should follow an interpolated surface reconstructed from the
    deprojected cell-boundary xyz points. Use :meth:`from_heightmap` when
    working from an external height map or exported cell-center coordinates.
    """

    heightmap: np.ndarray
    pixel_size: float
    voxel_depth: float = 1.0
    units: str = "a.u."
    prepared: bool = True
    result: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        h = np.asarray(self.heightmap, dtype=np.float64)
        if h.ndim != 2:
            raise ValueError("heightmap must be a 2-D array")
        self.heightmap = h
        self.pixel_size = _validate_positive(self.pixel_size, "pixel_size")
        self.voxel_depth = _validate_positive(self.voxel_depth, "voxel_depth")

    @classmethod
    def from_heightmap(
        cls,
        heightmap: np.ndarray,
        *,
        pixel_size: float,
        voxel_depth: float =1.0,
        prepared: bool = True,
        units: str = "a.u.",
        smooth_scale: float = -1.0,
        invert_z: bool = False,
        inpaint_zeros: bool = True,
        prune_zeros: bool = True,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a calculator from a raw or already prepared height map.

        Parameters
        ----------
        heightmap : np.ndarray
            Two-dimensional height map indexed as ``heightmap[row=y, column=x]``.
            If ``prepared=True``, values are used directly. If
            ``prepared=False``, the map is passed through
            :func:`deprojpy.heightmap.prepare_heightmap`.
        pixel_size : float
            Physical size of one pixel in the ``x`` and ``y`` directions.
            Public point inputs remain ``(x, y)`` pixel coordinates; returned
            distances are physical lengths.
        voxel_depth : float, optional
            Physical scale factor for raw height-map values. When
            ``prepared=True``, this factor is applied during distance
            calculations. When ``prepared=False``, it is consumed by
            ``prepare_heightmap`` and the returned calculator stores
            ``voxel_depth=1.0`` because its height map is already in physical
            z units.
        prepared : bool, optional
            If ``True`` (default), assume ``heightmap`` is already the surface
            used for distance calculations. If ``False``, run DeProjPy height-map
            preprocessing first.
        units : str, optional
            Label for the physical distance units.
        smooth_scale : float, optional
            Gaussian smoothing scale passed to ``prepare_heightmap`` when
            ``prepared=False``. Values ``<= 0`` disable smoothing.
        invert_z : bool, optional
            Whether to invert height values during preparation.
        inpaint_zeros : bool, optional
            Whether zero-valued regions are inpainted during preparation.
        prune_zeros : bool, optional
            Whether remaining zero-valued pixels become ``NaN`` during
            preparation.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator whose stored ``heightmap`` is the actual surface used for
            distance calculations.
        """
        pixel_size = _validate_positive(pixel_size, "pixel_size")
        voxel_depth = _validate_positive(voxel_depth, "voxel_depth")
        if prepared:
            return cls(
                heightmap=np.asarray(heightmap, dtype=float),
                pixel_size=pixel_size,
                voxel_depth=voxel_depth,
                units=units,
                prepared=True,
            )
        h = prepare_heightmap(
            heightmap,
            voxel_depth=voxel_depth,
            smooth_scale=smooth_scale,
            invert_z=invert_z,
            inpaint_zeros=inpaint_zeros,
            prune_zeros=prune_zeros,
        )
        return cls(h, pixel_size=pixel_size, voxel_depth=1.0, units=units, prepared=True)

    @classmethod
    def from_result(
        cls,
        result: DeprojResult,
        heightmap: np.ndarray | None = None,
        *,
        prepared: bool = False,
        pixel_size: float | None = None,
        voxel_depth: float | None = None,
        units: str | None = None,
        invert_z: bool | None = None,
        smooth_scale: float = -1.0,
        inpaint_zeros: bool = True,
        prune_zeros: bool = True,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a height-map-based calculator using metadata from a result.

        This constructor still builds the surface from a height map. It does
        not use deprojected cell-boundary points; use
        :meth:`from_cell_boundaries` for that surface.

        Parameters
        ----------
        result : DeprojResult
            DeProjPy result-like object. ``pixel_size``, ``voxel_depth``, and
            ``units`` are inferred from this object unless explicitly supplied.
            The result is also stored so pairwise methods can use cell centers
            when ``points_xy`` is omitted.
        heightmap : np.ndarray, optional
            Raw or prepared height map. If omitted, ``result.prepared_heightmap``
            is used and treated as prepared.
        prepared : bool, optional
            If ``True``, use ``heightmap`` directly. If ``False`` (default),
            prepare the provided raw height map with DeProjPy preprocessing.
        pixel_size : float, optional
            Override ``result.pixel_size``.
        voxel_depth : float, optional
            Override ``result.voxel_depth``. Ignored after construction when
            ``prepared=False`` because the stored height map is converted to
            physical z units.
        units : str, optional
            Override ``result.units``.
        invert_z : bool, optional
            Whether to invert height values during preparation. If omitted,
            defaults to ``False`` because current result objects do not record
            the original inversion setting.
        smooth_scale, inpaint_zeros, prune_zeros
            Passed to ``prepare_heightmap`` when ``prepared=False``.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator with scale metadata inferred from ``result`` and a stored
            height map ready for repeated distance calls.
        """
        if pixel_size is None:
            pixel_size = result.pixel_size
        if voxel_depth is None:
            voxel_depth = getattr(result, "voxel_depth", 1.0)
        if units is None:
            units = getattr(result, "units", "a.u.")
        if heightmap is None:
            prepared_heightmap = getattr(result, "prepared_heightmap", None)
            if prepared_heightmap is None:
                raise ValueError("heightmap is required when result has no prepared_heightmap")
            heightmap = prepared_heightmap
            prepared = True
        if invert_z is None:
            invert_z = False

        calc = cls.from_heightmap(
            heightmap, # type: ignore[arg-type]
            pixel_size=pixel_size, # type: ignore[arg-type]
            voxel_depth=1.0 if prepared else voxel_depth,  # type: ignore[arg-type]
            prepared=prepared,
            units=units,  # type: ignore[arg-type]
            smooth_scale=smooth_scale,
            invert_z=invert_z,
            inpaint_zeros=inpaint_zeros,
            prune_zeros=prune_zeros,
        )
        calc.result = result
        return calc

    @classmethod
    def from_cell_boundaries(
        cls,
        result: DeprojResult,
        *,
        shape: tuple[int, int] | None = None,
        method: str = "linear",
        extrapolation: str = "nearest",
        fill_value: float | None = None,
        units: str | None = None,
    ) -> "SurfaceDistanceCalculator":
        """
        Build a calculator from deprojected cell-boundary xyz points.

        This constructor interpolates the physical z-values stored in
        ``cell.boundary`` arrays back onto an image-like raster. It is useful
        when distance calculations or path visualizations should follow the
        fitted DeProj boundary surface rather than the original height map.

        Parameters
        ----------
        result : DeprojResult
            DeProjPy result-like object with ``epicells`` and ``pixel_size``.
            Each cell boundary is expected to have physical ``(x, y, z)``
            columns. ``pixel_size`` converts boundary ``x/y`` coordinates back
            to pixel coordinates before interpolation.
        shape : tuple[int, int], optional
            Output raster shape in array ``(rows, columns)`` order. If omitted,
            ``result.prepared_heightmap.shape`` is preferred. If that is not
            available, a minimal shape is inferred from boundary coordinates.
        method : {"linear", "clough", "clough_tocher", "nearest"}, optional
            Scattered interpolation method used inside the convex hull of
            boundary points.
        extrapolation : {"linear", "nearest", "constant", "none"}, optional
            How to fill grid points outside the interpolator support. ``"linear"``
            fits a least-squares plane to boundary points, ``"nearest"`` uses
            nearest-neighbor values, ``"constant"`` uses ``fill_value``, and
            ``"none"`` leaves missing values as ``NaN``.
        fill_value : float, optional
            Fill value used when ``extrapolation="constant"``.
        units : str, optional
            Override ``result.units`` for display. Calculations are unaffected.

        Returns
        -------
        SurfaceDistanceCalculator
            Calculator with ``pixel_size`` inferred from ``result``,
            ``voxel_depth=1.0``, and a prepared raster surface in physical
            z units.
        """
        pixel_size = _validate_positive(result.pixel_size, "result.pixel_size")
        if units is None:
            units = getattr(result, "units", "a.u.")
        surface = heightmap_from_cell_boundaries(
            result,
            shape=shape,
            method=method,
            extrapolation=extrapolation,
            fill_value=fill_value,
        )
        return cls(
            heightmap=surface,
            pixel_size=pixel_size,
            voxel_depth=1.0,
            units=units,  # type: ignore[arg-type]
            prepared=True,
            result=result,
        )

    def sample_height(
        self,
        xy: np.ndarray,
        *,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        """
        Sample this calculator's height map at ``(x, y)`` coordinates.

        Parameters
        ----------
        xy : np.ndarray
            Coordinates with shape ``(n, 2)`` in ``(x, y)`` order.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.

        Returns
        -------
        np.ndarray
            Sampled z values in physical units.
        """
        points = _as_pixel_xy(xy, pixel_size=self.pixel_size, input_units=input_units)
        height = sample_height_at_xy(self.heightmap, points, order=1, mode="nearest")
        return height * self.voxel_depth

    def straight_distance(
        self,
        p0_xy,
        p1_xy,
        *,
        input_units: Literal["pixel", "physical"] = "pixel",
        **kwargs,
    ) -> float:
        """
        Return straight-line surface distance between two ``(x, y)`` points.

        Parameters
        ----------
        p0_xy, p1_xy
            Endpoints in ``(x, y)`` order.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.

        Returns
        -------
        float
            Straight-line surface distance in physical units.
        """
        return surface_straight_distance(
            self.heightmap,
            p0_xy,
            p1_xy,
            pixel_size=self.pixel_size,
            voxel_depth=self.voxel_depth,
            input_units=input_units,
            **kwargs,
        )

    def _resolve_points(
        self,
        points_xy: np.ndarray | None,
        result=None,
        *,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        if points_xy is not None:
            points = _as_pixel_xy(points_xy, pixel_size=self.pixel_size, input_units=input_units)
            return _validate_xy_array(points, name="points_xy")
        result = result if result is not None else self.result
        if result is None:
            raise ValueError("points_xy is required unless a result is available")
        return cell_centers_xy_pixels(result)

    def straight_pairwise_distances(
        self,
        points_xy: np.ndarray | None = None,
        *,
        result=None,
        n_samples: int | None = None,
        samples_per_pixel: float = 2.0,
        min_samples: int = 16,
        max_samples: int = 2048,
        block_size: int | None = None,
        show_progress: bool | None = None,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        """
        Return an ``N × N`` matrix of straight-line surface distances.

        The implementation computes only the upper triangle and mirrors it to
        the lower triangle. If ``points_xy`` is omitted, points are pulled from
        the stored result and converted to pixel coordinates automatically.

        Parameters
        ----------
        points_xy : np.ndarray, optional
            Coordinates with shape ``(n, 2)`` in ``(x, y)`` order.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.
        """
        del block_size, show_progress
        points = self._resolve_points(points_xy, result=result, input_units=input_units)
        if len(points) > 2000:
            warnings.warn(
                "straight_pairwise_distances computes an N x N matrix and may "
                "take substantial time and memory for N > 2000",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_samples is not None:
            min_samples = max_samples = int(n_samples)
        return _pairwise_surface_straight_distances_numba(
            self.heightmap,
            points.astype(np.float64),
            float(self.pixel_size),
            float(self.voxel_depth),
            float(samples_per_pixel),
            int(min_samples),
            int(max_samples),
        )

    def straight_distances_for_pairs(
        self,
        points_xy: np.ndarray,
        pairs: np.ndarray,
        *,
        n_samples: int | None = None,
        samples_per_pixel: float = 2.0,
        min_samples: int = 16,
        max_samples: int = 2048,
        input_units: Literal["pixel", "physical"] = "pixel",
    ) -> np.ndarray:
        """
        Compute straight-line surface distances for selected point pairs.

        Parameters
        ----------
        points_xy : np.ndarray
            Coordinates with shape ``(n, 2)`` in ``(x, y)`` order.
        pairs : np.ndarray
            Integer array with shape ``(n_pairs, 2)`` indexing rows of
            ``points_xy``.
        input_units : {"pixel", "physical"}, optional
            Coordinate system of input xy points. Use "pixel" when points are in image
            pixel coordinates, with x as column and y as row. Use "physical" when xy
            points are in the same physical units as result boundaries and centers.
            Distances are always returned in physical units.
        """
        points = _validate_xy_array(
            _as_pixel_xy(points_xy, pixel_size=self.pixel_size, input_units=input_units),
            name="points_xy",
        )
        pair_array = np.asarray(pairs, dtype=np.int64)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        if pair_array.size and (pair_array.min() < 0 or pair_array.max() >= len(points)):
            raise IndexError("pairs contain point indices outside points_xy")
        if n_samples is not None:
            min_samples = max_samples = int(n_samples)
        return _selected_surface_straight_distances_numba(
            self.heightmap,
            points.astype(np.float64),
            pair_array,
            float(self.pixel_size),
            float(self.voxel_depth),
            float(samples_per_pixel),
            int(min_samples),
            int(max_samples),
        )
