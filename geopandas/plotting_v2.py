import numpy as np
import pandas as pd

import shapely

from ._compat import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    from matplotlib.collections import PatchCollection

    class GeoPandasPolyCollection(PatchCollection):
        """Subclass to assign handler without overriding one for PatchCollection."""

    from matplotlib.legend import Legend
    from matplotlib.legend_handler import HandlerPolyCollection

    # PatchCollection is not supported by Legend but we can use PolyCollection handler
    # instead in our specific case. Define a subclass and assign a handler.
    Legend.update_default_handler_map(
        {GeoPandasPolyCollection: HandlerPolyCollection()}
    )


def _expand_kwargs(kwargs, multiindex):
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


def _PolygonPatch(polygon, **kwargs):
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
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

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
    from matplotlib.collections import LineCollection

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
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=autolim)
    ax.autoscale_view()
    return collection
