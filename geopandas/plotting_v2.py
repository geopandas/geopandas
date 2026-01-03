import warnings

import numpy as np
import pandas as pd

import shapely

import geopandas

from ._compat import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.legend import Legend
    from matplotlib.legend_handler import HandlerPolyCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    class GeoPandasPolyCollection(PatchCollection):
        """Subclass to assign handler without overriding one for PatchCollection."""

    # PatchCollection is not supported by Legend but we can use PolyCollection handler
    # instead in our specific case. Define a subclass and assign a handler.
    Legend.update_default_handler_map(
        {GeoPandasPolyCollection: HandlerPolyCollection()}
    )


def _sanitize_geoms(
    geoms: geopandas.GeoSeries,
) -> tuple[geopandas.GeoSeries, np.ndarray]:
    """Return sanitized geometry with the indices of original geometry.

    1. Normalize all geometry to ensure holes are correctly plotted.
    2. Explode GeometryCollections to individual components. This generates an
       index where values are repeated for all components in the same
       GeometryCollection.
    3. Filter out missing and empty geometry. The resulting index does not contain
       their IDs.

    Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out. The index then maintains 1:1 matching of geometry to value.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    # TODO(shapely) look into simplifying this with
    # shapely.get_parts(geoms, return_index=True) from shapely 2.0
    geoms = geoms.normalize()
    components, component_index = [], []

    if (
        not geoms.geom_type.str.startswith("Geom").any()
        and not geoms.is_empty.any()
        and not geoms.isna().any()
    ):
        return geoms, np.arange(len(geoms))

    for ix, geom in enumerate(geoms):
        if geom is not None and geom.geom_type.startswith("Geom") and not geom.is_empty:
            for poly in geom.geoms:
                components.append(poly)
                component_index.append(ix)
        elif geom is None or geom.is_empty:
            continue
        else:
            components.append(geom)
            component_index.append(ix)

    return geopandas.GeoSeries(components, crs=geoms.crs), np.array(component_index)


def _expand_kwargs(kwargs: dict, multiindex: np.ndarray) -> None:
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    from collections.abc import Iterable

    from matplotlib.colors import is_color_like

    scalar_kwargs = ["marker", "path_effects"]
    for att, value in kwargs.items():
        if "color" in att:  # color(s), edgecolor(s), facecolor(s)
            if is_color_like(value):
                continue
        elif "linestyle" in att:  # linestyle(s)
            # A single linestyle can be 2-tuple of a number and an iterable.
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], Iterable)
            ):
                continue
        elif att in scalar_kwargs:
            # For these attributes, only a single value is allowed, so never expand.
            continue

        if pd.api.types.is_list_like(value):
            kwargs[att] = np.take(value, multiindex, axis=0)


def _PolygonPatch(polygon: shapely.Geometry, **kwargs) -> "PathPatch":
    """Construct a matplotlib patch from a (Multi)Polygon geometry.

    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    To ensure proper rendering on the matplotlib side, winding order of individual
    rings needs to be normalized as the order is what matplotlib uses to determine
    if a Path represents a patch or a hole.

    Example (using Shapely Point and a matplotlib axes)::

        b = shapely.geometry.Point(0, 0).buffer(1.0)
        patch = _PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
        ax.add_patch(patch)

    GeoPandas originally relied on the descartes package by Sean Gillies
    (BSD license, https://pypi.org/project/descartes) for PolygonPatch, but
    this dependency was removed in favor of the below matplotlib code.
    """
    if polygon.geom_type == "Polygon":
        path = Path.make_compound_path(
            Path(np.asarray(polygon.exterior.coords)[:, :2], closed=True),
            *[
                Path(np.asarray(ring.coords)[:, :2], closed=True)
                for ring in polygon.interiors
            ],
        )
    else:
        paths = []
        for part in polygon.geoms:
            # exteriors
            paths.append(Path(np.asarray(part.exterior.coords)[:, :2], closed=True))
            # interiors
            for ring in part.interiors:
                paths.append(Path(np.asarray(ring.coords)[:, :2], closed=True))
        path = Path.make_compound_path(*paths)

    return PathPatch(path, **kwargs)


def _plot_polygon_collection(
    ax,
    geoms,
    values=None,
    cmap=None,
    vmin=None,
    vmax=None,
    autolim=True,
    **kwargs,
):
    """Plot a collection of Polygon and MultiPolygon geometries to `ax`.

    Note that all style keywords, like ``color`` that can be set as an array in
    matplotlib shall be passed directly via kwargs.

    No need to explode geometries to single-parts as _PolygonPatch supports
    MultiPolygons,

    Parameters
    ----------
    ax : _type_
        _description_
    geoms : _type_
        _description_
    values : _type_, optional
        _description_, by default None
    cmap : _type_, optional
        _description_, by default None
    vmin : _type_, optional
        _description_, by default None
    vmax : _type_, optional
        _description_, by default None
    autolim : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    # GeoPandasPolyCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    collection = GeoPandasPolyCollection(
        [_PolygonPatch(poly) for poly in geoms], **kwargs
    )

    if values is not None:
        collection.set_array(np.asarray(values))
        if cmap:
            collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection


def _plot_linestring_collection(
    ax,
    geoms,
    values=None,
    color=None,
    cmap=None,
    vmin=None,
    vmax=None,
    autolim=True,
    **kwargs,
):
    """Plot a collection of LineString and MultiLineString geometries to `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` LineStrings and/or MultiLineStrings (can be
            mixed)
    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
    color : single color or sequence of `N` colors
        Cannot be used together with `values`.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    geoms, multiindex = shapely.get_parts(geoms, return_index=True)
    if values is not None:
        values = np.take(values, multiindex, axis=0)

    # LineCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    # Add to kwargs for easier checking below.
    if color is not None:
        kwargs["color"] = color

    _expand_kwargs(kwargs, multiindex)

    segments = [np.array(linestring.coords)[:, :2] for linestring in geoms]
    collection = LineCollection(segments, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        if cmap:
            collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection


def _plot_point_collection(
    ax,
    geoms,
    values=None,
    color=None,
    cmap=None,
    vmin=None,
    vmax=None,
    marker="o",
    markersize=None,
    **kwargs,
):
    """Plot a collection of Point and MultiPoint geometries to `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : sequence of `N` Points or MultiPoints

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    import shapely

    geoms, multiindex = shapely.get_parts(geoms, return_index=True)

    xy = shapely.get_coordinates(geoms)

    # Add to kwargs for easier checking below.
    if values is not None:
        kwargs["c"] = values
    if markersize is not None:
        kwargs["s"] = markersize
    if color is not None:
        kwargs["color"] = color
    if marker is not None:
        kwargs["marker"] = marker

    _expand_kwargs(kwargs, multiindex)

    # norm cannot be passed alongside vmin and vmax
    if "norm" not in kwargs:
        collection = ax.scatter(
            xy[:, 0], xy[:, 1], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
        )
    else:
        collection = ax.scatter(xy[:, 0], xy[:, 1], cmap=cmap, **kwargs)

    return collection


def plot_series(
    s,
    cmap=None,
    color=None,
    ax=None,
    figsize=None,
    aspect="auto",
    autolim=True,
    **style_kwds,
):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString, Point and MultiPoint
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str, np.array, pd.Series, List (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(s_y * pi/180) with s_y the y coordinate of the middle of
        the GeoSeries (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if s.empty:
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "empty. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
        )
        return ax

    if s.is_empty.all():
        warnings.warn(
            "The GeoSeries you are attempting to plot is "
            "composed of empty geometries. Nothing has been displayed.",
            UserWarning,
            stacklevel=3,
        )
        return ax

    # set correct aspect to preserve proportions in geographic CRS
    if aspect == "auto":
        if s.crs and s.crs.is_geographic:
            bounds = s.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            ax.set_aspect("equal")
    elif aspect is not None:
        ax.set_aspect(aspect)

    # decompose GeometryCollections
    geoms, multiindex = _sanitize_geoms(s)

    values = None
    color_given = False

    # if cmap is specified, create range of colors based on cmap
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, "N"):
            # repeat the cmap rather than expanding it for ListedColormap and likes
            values = values % cmap.N
        style_kwds["vmin"] = values.min()
        style_kwds["vmax"] = values.max()

        # ensure proper mapping of values to components of GeometryCollections
        values = np.take(values, multiindex, axis=0)

    # if color is specified as a list-like, ensure it is properly mapped to components
    elif color is not None:
        color_given = pd.api.types.is_list_like(color) and len(color) == len(s)
        # have colors been given for all geometries?
        if pd.api.types.is_list_like(color) and len(color) == len(s):
            # ensure indexes are consistent
            if isinstance(color, pd.Series):
                color = color.reindex(s.index)
            color = np.take(color, multiindex, axis=0)

    # subdivide by geometry type - each has its own collection
    geom_types = geoms.geom_type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = geoms[poly_idx]
    if not polys.empty:
        # color overrides both face and edgecolor. As we want people to be
        # able to use edgecolor as well, pass color to facecolor
        facecolor = style_kwds.pop("facecolor", None)

        if color is not None:
            facecolor = color[poly_idx] if color_given else color

        values_ = values[poly_idx] if values is not None else None

        _plot_polygon_collection(
            ax,
            polys,
            values_,
            facecolor=facecolor,
            cmap=cmap,
            autolim=autolim,
            **style_kwds,
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = geoms[line_idx]
    if not lines.empty:
        values_ = values[line_idx] if values is not None else None

        color_ = color[line_idx] if color_given else color  # ty:ignore[not-subscriptable]

        _plot_linestring_collection(
            ax, lines, values_, color=color_, cmap=cmap, autolim=autolim, **style_kwds
        )

    # plot all Points in the same collection
    points = geoms[point_idx]
    if not points.empty:
        values_ = values[point_idx] if values is not None else None

        color_ = color[point_idx] if color_given else color  # ty:ignore[not-subscriptable]

        _plot_point_collection(
            ax, points, values_, color=color_, cmap=cmap, **style_kwds
        )

    ax.figure.canvas.draw_idle()

    return ax
