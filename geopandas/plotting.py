from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable, Sequence
from itertools import compress
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.plotting import PlotAccessor

import shapely

import geopandas

from ._decorator import doc

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import (
        LineCollection,
        PatchCollection,
        PathCollection,
    )
    from matplotlib.colors import Colormap
    from matplotlib.markers import MarkerStyle
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path


def _set_aspect(
    aspect: float | Literal["auto", "equal", None],
    s: geopandas.GeoSeries | geopandas.GeoDataFrame,
    ax: Axes,
) -> None:
    """Set the aspect ratio of the axis.

    - If `aspect` is "auto" and the CRS is geographic, the aspect ratio is adjusted to
      account for latitude distortion, using a formula ported from the R package 'sp'.
    - If the CRS is not geographic, the aspect is set to "equal".
    - If `aspect` is not None or "auto", its value is passed directly to
      `ax.set_aspect`.

    Parameters
    ----------
    aspect : str or float or None
        The aspect ratio to set. If "auto", the aspect is determined based on the
        geometry's CRS. If a float or other valid matplotlib aspect value, it is passed
        directly to `ax.set_aspect`.
    s : GeoSeries or GeoDataFrame
        The spatial data whose CRS and bounds are used to determine the aspect ratio if
        `aspect` is "auto".
    ax : matplotlib.axes.Axes
        The axes on which to set the aspect ratio.
    """
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


def _sanitize_geoms(
    geoms: geopandas.GeoSeries,
) -> tuple[geopandas.GeoSeries, np.ndarray]:
    """Return sanitized geometry with the indices of original geometry.

    1. Normalize all geometry to ensure holes are correctly plotted.
    2. Explode GeometryCollections to individual components. This generates an
       index where values are repeated for all components in the same
       collection.
    3. Filter out missing and empty geometry. The resulting index does not contain
       their IDs.

    Series like geoms and index, except that any GeometryCollections
    are split into their components and indices are repeated for all component
    in the same collection. At the same time, empty or missing geometries are
    filtered out. The index then maintains 1:1 matching of geometry to value.

    Returns
    -------
    components : list of geometry

    component_index : index array
        indices are repeated for all components in the same collection
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


def _subset_kwds(kwds: dict, index: np.ndarray) -> dict:
    """Subsets list-like keyword arguments based on a given index array.

    Parameters
    ----------
    kwds : dict
        Dictionary of keyword arguments to be subsetted.
    index : np.ndarray
        Array of indices used to subset list-like values in `kwds`.

    Returns
    -------
    dict
        Dictionary with values subsetted according to `index` where applicable.
    """
    subset_kwds = {}
    for key, val in kwds.items():
        # fast indexing for arrays
        if isinstance(val, (np.ndarray, pd.Series, pd.Index, pd.DataFrame)) and (
            len(val) == index.shape[0]
        ):
            subset_kwds[key] = val[index]
        # slowed indexing for lists - can contain mix of floats and tuples, generally
        # unsafe coercing to arrays
        elif pd.api.types.is_list_like(val) and (len(val) == index.shape[0]):
            # corner case caused by the linestyle input being a tuple
            if (
                key.startswith("linestyle")
                and isinstance(val[1], tuple)
                and index.shape[0] == 2
            ):
                subset_kwds[key] = val
            else:
                # compress is like numpy indexing for lists
                compressed = list(compress(val, index))
                # if only one remains, extract scalar
                if len(compressed) == 1:
                    subset_kwds[key] = compressed[0]
                else:
                    subset_kwds[key] = compressed
        else:
            # scalar
            subset_kwds[key] = val
    return subset_kwds


def _PolygonPatch(polygon: shapely.Geometry, **kwargs) -> PathPatch:
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
        parts = [polygon]
    else:
        parts = polygon.geoms
    paths = []
    for part in parts:
        # exteriors
        paths.append(Path(np.asarray(part.exterior.coords)[:, :2], closed=True))
        # interiors
        paths.extend(
            Path(np.asarray(ring.coords)[:, :2], closed=True) for ring in part.interiors
        )
    path = Path.make_compound_path(*paths)

    return PathPatch(path, **kwargs)


def _plot_polygon_collection(
    ax: Axes,
    geoms: geopandas.GeoSeries,
    values: np.ndarray | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
    autolim: bool = True,
    **kwargs,
) -> PatchCollection:
    """Plot a collection of Polygon and MultiPolygon geometries to `ax`.

    Note that all style keywords, like ``color`` that can be set as an array in
    matplotlib shall be passed directly via kwargs.

    No need to explode geometries to single-parts as _PolygonPatch supports
    MultiPolygons,

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to add the collection
    geoms : GeoSeries
        GeoSeries of (Multi)Polygons
    values : np.ndarray, optional
        Values will be mapped to colors using vmin/vmax/norm/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs, by default None
    cmap : str or Colormap, optional
        The colormap recognized by matplotlib, by default None
    vmin : float, optional
        Minimum value of cmap, by default None
    vmax : float, optional
        Maximum value of cmap, by default None
    autolim : bool, optional
        Update axes data limits to contain the new geometries, by default True

    Returns
    -------
    _GeoPandasPolyCollection
        matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.legend import Legend
    from matplotlib.legend_handler import HandlerPolyCollection

    class _GeoPandasPolyCollection(PatchCollection):
        """Subclass to assign handler without overriding one for PatchCollection."""

    # PatchCollection is not supported by Legend but we can use PolyCollection handler
    # instead in our specific case. Define a subclass and assign a handler.
    Legend.update_default_handler_map(
        {_GeoPandasPolyCollection: HandlerPolyCollection()}
    )

    # _GeoPandasPolyCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    collection = _GeoPandasPolyCollection(
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
    ax: Axes,
    geoms: geopandas.GeoSeries,
    values: np.ndarray | None = None,
    color: str | Sequence | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
    autolim: bool = True,
    **kwargs,
) -> LineCollection:
    """Plot a collection of LineString and MultiLineString geometries to `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to add the collection
    geoms : GeoSeries
        GeoSeries of (Multi)Polygons
    values : np.ndarray, optional
        Values will be mapped to colors using vmin/vmax/norm/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs, by default None
    color : single color or sequence of `N` colors
        Color definition understood by matplotlib. Cannot be used together with
        `values`.
    cmap : str or Colormap, optional
        The colormap recognized by matplotlib, by default None
    vmin : float, optional
        Minimum value of cmap, by default None
    vmax : float, optional
        Maximum value of cmap, by default None
    autolim : bool, optional
        Update axes data limits to contain the new geometries, by default True

    Returns
    -------
    LineCollection
        matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import LineCollection

    geoms, multiindex = shapely.get_parts(geoms.values, return_index=True)
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
    ax: Axes,
    geoms: geopandas.GeoSeries,
    values: np.ndarray | None = None,
    color: str | Sequence | None = None,
    cmap: str | Colormap | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
    marker: str | MarkerStyle | Path = "o",
    markersize: float | Sequence[float] | None = None,
    **kwargs,
) -> PathCollection:
    """Plot a collection of Point and MultiPoint geometries to `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to add the collection
    geoms : GeoSeries
        GeoSeries of (Multi)Polygons
    values : np.ndarray, optional
        Values will be mapped to colors using vmin/vmax/norm/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs, by default None
    color : single color or sequence of `N` colors
        Color definition understood by matplotlib. Cannot be used together with
        `values`.
    cmap : str or Colormap, optional
        The colormap recognized by matplotlib, by default None
    vmin : float, optional
        Minimum value of cmap, by default None
    vmax : float, optional
        Maximum value of cmap, by default None
    marker : str, MarkerStyle, Path
        Style of the marker to be used.
    markersize : scalar or array-like, optional
        Size of the markers. Note that under the hood ``scatter`` is
        used, so the specified value will be proportional to the
        area of the marker (size in points^2).

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    geoms, multiindex = shapely.get_parts(geoms.values, return_index=True)

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
    s: geopandas.GeoSeries,
    cmap: str | Colormap | None = None,
    color: str | Sequence | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    aspect: float | Literal["auto", "equal", None] = "auto",
    autolim: bool = True,
    **style_kwds,
) -> Axes:
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
    try:
        import matplotlib  # noqa: F401
        from matplotlib.colors import Colormap
    except ImportError as err:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        ) from err

    import matplotlib.pyplot as plt

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
    _set_aspect(aspect, s, ax)

    # decompose GeometryCollections
    geoms, multiindex = _sanitize_geoms(s)

    values = None
    color_given = False

    # if cmap is specified, create range of colors based on cmap
    if cmap is not None:
        values = np.arange(len(s))
        if isinstance(cmap, Colormap) and hasattr(cmap, "N"):
            # repeat for cmap with limited number of colors
            values = values % cmap.N
        style_kwds["vmin"] = values.min()
        style_kwds["vmax"] = values.max()

        # ensure proper mapping of values to components of GeometryCollections
        values = np.take(values, multiindex, axis=0)

    # if color is specified as a list-like, ensure it is properly mapped to components
    elif color is not None:
        color_given = pd.api.types.is_list_like(color) and len(color) == len(s)
        # have colors been given for all geometries?
        if color_given:
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

        poly_kwds = _subset_kwds(style_kwds, poly_idx)

        _plot_polygon_collection(
            ax,
            polys,
            values_,
            facecolor=facecolor,
            cmap=cmap,
            autolim=autolim,
            **poly_kwds,
        )

    # plot all LineStrings and MultiLineString components in same collection
    lines = geoms[line_idx]
    if not lines.empty:
        values_ = values[line_idx] if values is not None else None

        color_ = color[line_idx] if color_given else color

        lines_kwds = _subset_kwds(style_kwds, line_idx)

        _plot_linestring_collection(
            ax, lines, values_, color=color_, cmap=cmap, autolim=autolim, **lines_kwds
        )

    # plot all Points in the same collection
    points = geoms[point_idx]
    if not points.empty:
        values_ = values[point_idx] if values is not None else None

        color_ = color[point_idx] if color_given else color

        points_kwds = _subset_kwds(style_kwds, point_idx)

        _plot_point_collection(
            ax, points, values_, color=color_, cmap=cmap, **points_kwds
        )

    ax.figure.canvas.draw_idle()

    return ax


def plot_dataframe(
    df: geopandas.GeoDataFrame,
    column: str | np.ndarray | pd.Series | pd.Index | None = None,
    cmap: str | Colormap | None = None,
    color: str | Sequence | None = None,
    ax: Axes | None = None,
    cax: Axes | None = None,
    categorical: bool = False,
    legend: bool = False,
    scheme: str | None = None,
    k: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    markersize: str | float | Sequence | None = None,
    figsize: tuple[float, float] | None = None,
    legend_kwds: dict | None = None,
    categories: Sequence | None = None,
    classification_kwds: dict | None = None,
    missing_kwds: dict | None = None,
    aspect: float | Literal["auto", "equal", None] = "auto",
    autolim: bool = True,
    **style_kwds,
) -> Axes:
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    column : str, np.array, pd.Series, pd.Index (default None)
        The name of the dataframe column, np.array, pd.Series, or pd.Index
        to be plotted. If np.array, pd.Series, or pd.Index are used then it
        must have same length as dataframe. Values are used to color the plot.
        Ignored if `color` is also set.
    kind: str
        The kind of plots to produce. The default is to create a map ("geo").
        Other supported kinds of plots from pandas:

        - 'line' : line plot
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : BoxPlot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot
        - 'hexbin' : hexbin plot.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str, np.array, pd.Series (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    cax : matplotlib.pyplot Artist (default None)
        axes on which to draw the legend in case of color map.
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns, this
        will be set to True.
    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    k : int (default 5)
        Number of classes (ignored if scheme is None)
    vmin : None or float (default None)
        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.
    markersize : str or float or sequence (default None)
        Only applies to point geometries within a frame.
        If a str, will use the values in the column of the frame specified
        by markersize to set the size of markers. Otherwise can be a value
        to apply to all points, or a sequence of the same length as the
        number of points.
    figsize : tuple of integers (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        axes is given explicitly, figsize is ignored.
    legend_kwds : dict (default None)
        Keyword arguments to pass to :func:`matplotlib.pyplot.legend` (e.g. ``labels``,
        or ``frameon``) or :func:`matplotlib.pyplot.colorbar (e.g. ``orientation``).
        Additional accepted keywords when `scheme` is specified:

        fmt : string
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``.
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
        colorbar : boolean (default False)
            An option to control whether the legend should be treated as categorical
            or as a colorbar. When set to True, ``fmt`` and ``interval`` are ignored.

    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    missing_kwds : dict (default None)
        Keyword arguments specifying color options (as style_kwds)
        to be passed on to geometries with missing values in addition to
        or overwriting other style kwds. If None, geometries with missing
        values are not plotted.
    aspect : 'auto', 'equal', None or float (default 'auto')
        Set aspect of axis. If 'auto', the default aspect for map plots is 'equal'; if
        however data are not projected (coordinates are long/lat), the aspect is by
        default set to 1/cos(df_y * pi/180) with df_y the y coordinate of the middle of
        the GeoDataFrame (the mean of the y range of bounding box) so that a long/lat
        square appears square in the middle of the plot. This implies an
        Equirectangular projection. If None, the aspect of `ax` won't be changed. It can
        also be set manually (float) as the ratio of y-unit to x-unit.
    autolim : bool (default True)
        Update axes data limits to contain the new geometries.
    **style_kwds : dict
        Style options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    Examples
    --------
    >>> import geodatasets
    >>> df = geopandas.read_file(geodatasets.get_path("nybb"))
    >>> df.head()  # doctest: +SKIP
       BoroCode  ...                                           geometry
    0         5  ...  MULTIPOLYGON (((970217.022 145643.332, 970227....
    1         4  ...  MULTIPOLYGON (((1029606.077 156073.814, 102957...
    2         3  ...  MULTIPOLYGON (((1021176.479 151374.797, 102100...
    3         1  ...  MULTIPOLYGON (((981219.056 188655.316, 980940....
    4         2  ...  MULTIPOLYGON (((1012821.806 229228.265, 101278...

    >>> df.plot("BoroName", cmap="Set1")  # doctest: +SKIP

    See the User Guide page :doc:`../../user_guide/mapping` for details.

    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, collections, colormaps, colors
    except ImportError:
        raise ImportError(
            "The matplotlib package is required for plotting in geopandas. "
            "You can install it using 'conda install -c conda-forge matplotlib' or "
            "'pip install matplotlib'."
        )

    if column is not None and color is not None:
        warnings.warn(
            "Only specify one of 'column' or 'color'. Using 'color'.",
            UserWarning,
            stacklevel=3,
        )
        column = None

    # Process polymorphic markersize
    if isinstance(markersize, str):
        markersize = df[markersize].values

    # if column is not set, we're showing just geometries -> plot_series
    if column is None:
        return plot_series(
            df.geometry,
            cmap=cmap,
            color=color,
            ax=ax,
            figsize=figsize,
            markersize=markersize,
            aspect=aspect,
            autolim=autolim,
            **style_kwds,
        )

    if ax is None:
        if cax is not None:
            raise ValueError("'ax' can not be None if 'cax' is not.")
        _fig, ax = plt.subplots(figsize=figsize)

    # set correct aspect to preserve proportions in geographic CRS
    _set_aspect(aspect, df, ax)

    # Process polymorphic column argument (column name or array-like)
    if isinstance(column, np.ndarray | pd.Series | pd.Index):
        if column.shape[0] != df.shape[0]:
            raise ValueError(
                "The dataframe and given column have different number of rows."
            )
        elif isinstance(column, pd.Index):
            values = column.values
        else:
            values = column

            # Make sure index of a Series matches index of df
            if isinstance(values, pd.Series):
                values = values.reindex(df.index)
    else:
        values = df[column]

    # Infer categorical variable
    if isinstance(values.dtype, CategoricalDtype):
        if categories is not None:
            raise ValueError(
                "Cannot specify 'categories' when column has categorical dtype"
            )
        categorical = True
    elif (
        pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
        or categories
    ):
        categorical = True

    if legend_kwds is None:
        legend_kwds = {}
    else:
        # if legend_kwds set, copy so we don't update it in place. GH1555
        legend_kwds = legend_kwds.copy()

    nan_idx = np.asarray(pd.isna(values), dtype="bool")

    if scheme:
        try:
            import mapclassify
        except ImportError:
            raise ImportError(
                "The 'mapclassify' package is required to use the 'scheme' keyword."
            )

        if classification_kwds is None:
            classification_kwds = {}
        if "k" not in classification_kwds:
            classification_kwds["k"] = k

        mask = ~nan_idx
        if vmin is not None:
            mask = mask & (values >= vmin)
        if vmax is not None:
            mask = mask & (values <= vmax)

        binning = mapclassify.classify(values[mask], scheme, **classification_kwds)

        # if legend should not be a colorbar we need to treat this as
        # a categorical plot
        if not legend_kwds.pop("colorbar", False):
            # use bin labels generated by mapclassify unless user passes their own
            if "labels" not in legend_kwds:
                classes = binning.get_legend_classes(
                    fmt=legend_kwds.pop("fmt", "{:.2f}")
                )

                if not legend_kwds.pop("interval", False):
                    classes = [c[1:-1] for c in classes]

                legend_kwds["labels"] = classes

            codes = binning.find_bin(values[~nan_idx])
            values = pd.Categorical(
                [np.nan] * len(values), categories=binning.bins, ordered=True
            )
            values[~nan_idx] = pd.Categorical.from_codes(
                codes,
                categories=binning.bins,
                ordered=True,
            )
            categorical = True

    # Plot categorical values via groupby - each category is a group plotted using
    # plot_series
    if categorical:
        if categories is not None:
            values = _check_invalid_categories(categories, values)
        grouped = df.groupby(values, observed=False)
        ngroups = grouped.ngroups

        if cmap is None:
            if scheme:
                cmap = colormaps["viridis"]
            else:
                cmap = colormaps["tab20"] if ngroups > 10 else colormaps["tab10"]
        elif isinstance(cmap, str):
            cmap = colormaps[cmap]

        def _color(i, name, ngroups, cmap):
            """Pull the color from the cmap for group."""
            if isinstance(cmap, colors.Colormap):
                if cmap.N < 32:
                    # For categorical cmaps, iterate over colours for the optimal
                    # contrast. Categorical cmaps have generally lower number of colors.
                    # There's no way of pulling the info on the cmap type directly from
                    # matplotlib.
                    return cmap(i)
                else:
                    # For continuous cmaps, stretch alongside whole range
                    return cmap(i / (ngroups - 1))
            elif isinstance(cmap, dict):
                return cmap[name]
            else:
                raise ValueError(
                    "`cmap` type is not supported. Provide a string mappable "
                    "to matplotlib colormap, `matplotlib.colors.Colormap` or a "
                    "dictionary mapping values to colors."
                )

        # add to style_kwds so it can be mapped to groups
        if markersize is not None:
            style_kwds["markersize"] = markersize

        # get majority geom type to know how to indicate empty value in the legend
        majority_geom_type = df.geom_type.mode().iloc[0]

        # process custom labels if they are provided
        if "labels" in legend_kwds:
            if len(legend_kwds["labels"]) != ngroups:
                raise ValueError(
                    "Number of labels must match number of categories, "
                    f"received {len(legend_kwds['labels'])} labels "
                    f"for {ngroups} categories."
                )
            custom_labels = legend_kwds.pop("labels", None)
        else:
            custom_labels = None

        # looping over groups and adding them to the Axes one by one, each with its
        # own collection and label
        for i, (name, group) in enumerate(grouped["geometry"]):
            # this ensures that any style kwd can be mapped to a value and that
            # list-like kwds are properly split to groups
            group_style_kwds = {}
            for key, val in style_kwds.items():
                if isinstance(val, dict):
                    group_style_kwds[key] = val.get(name)
                elif pd.api.types.is_list_like(val) and len(val) == len(df):
                    group_style_kwds[key] = np.take(val, grouped.indices[name])
                else:
                    group_style_kwds[key] = val

            # extract potential custom label
            label = custom_labels[i] if custom_labels else name

            # categoricals with more categories than observed values might be empty
            # plot nothing to get an item for legend. Determine how to plot nothing
            # based on a majority geom type to get matching handle in the legend
            if group.empty:
                if majority_geom_type.endswith("Polygon"):
                    ax.add_collection(
                        collections.PolyCollection(
                            [],
                            color=_color(i, name, ngroups, cmap),
                            **group_style_kwds,
                            label=label,
                        )
                    )
                elif majority_geom_type.endswith("Point"):
                    ax.scatter(
                        [],
                        [],
                        color=_color(i, name, ngroups, cmap),
                        **group_style_kwds,
                        label=label,
                    )
                else:
                    ax.plot(
                        [],
                        [],
                        color=_color(i, name, ngroups, cmap),
                        **group_style_kwds,
                        label=label,
                    )
            else:
                plot_series(
                    group.geometry,
                    label=label,
                    color=_color(i, name, ngroups, cmap),
                    ax=ax,
                    aspect=None,
                    **group_style_kwds,
                )

        missing_geoms = df.geometry[nan_idx]
        missing_data = not missing_geoms.empty
    else:
        values_min = values[~nan_idx].min()
        values_max = values[~nan_idx].max()
        mn = values_min if vmin is None else vmin
        mx = values_max if vmax is None else vmax

        # classification scheme sets boundary norm for segmented colorbar
        if scheme:
            if "norm" in style_kwds:
                raise ValueError("Cannot set `norm` and `scheme` at the same time.")

            if vmin is not None:
                lowest = vmin
            elif getattr(binning, "lowest", None) is not None:
                lowest = binning.lowest
            elif values_min > binning.bins[0]:
                # we don't know the real lowest value for this scheme
                # e.g. incorrect user_defined scheme without lowest
                # the first bin is zero length to preserve colour mapping
                lowest = binning.bins[0]
            else:
                lowest = values_min
            style_kwds["norm"] = colors.BoundaryNorm(
                boundaries=[lowest] + list(binning.bins),
                ncolors=256,
            )

            # default to proportional spacing of the colorbar when using a scheme
            if "spacing" not in legend_kwds:
                legend_kwds["spacing"] = "proportional"

        # decompose GeometryCollections
        expl_series, multiindex = _sanitize_geoms(df.geometry)
        values = np.take(values, multiindex, axis=0)
        nan_idx = np.take(nan_idx, multiindex, axis=0)
        _expand_kwargs(style_kwds, multiindex)

        geom_types = expl_series.geom_type
        poly_idx = np.asarray(
            (geom_types == "Polygon") | (geom_types == "MultiPolygon")
        )
        line_idx = np.asarray(
            (geom_types == "LineString")
            | (geom_types == "MultiLineString")
            | (geom_types == "LinearRing")
        )
        point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

        # plot all Polygons and all MultiPolygon components in the same collection
        polys_notna = poly_idx & np.invert(nan_idx)
        polys = expl_series[polys_notna]
        if not polys.empty:
            subset = values[polys_notna]
            poly_kwds = _subset_kwds(style_kwds, polys_notna)

            _plot_polygon_collection(
                ax,
                polys,
                subset,
                vmin=mn,
                vmax=mx,
                cmap=cmap,
                autolim=autolim,
                **poly_kwds,
            )

        # plot all LineStrings and MultiLineString components in same collection
        lines_notna = line_idx & np.invert(nan_idx)
        lines = expl_series[lines_notna]
        if not lines.empty:
            subset = values[lines_notna]
            lines_kwds = _subset_kwds(style_kwds, lines_notna)

            _plot_linestring_collection(
                ax,
                lines,
                subset,
                vmin=mn,
                vmax=mx,
                cmap=cmap,
                autolim=autolim,
                **lines_kwds,
            )

        # plot all Points in the same collection
        points_notna = point_idx & np.invert(nan_idx)
        points = expl_series[points_notna]
        if not points.empty:
            subset = values[point_idx & np.invert(nan_idx)]
            points_kwds = _subset_kwds(style_kwds, points_notna)

            _plot_point_collection(
                ax,
                points,
                subset,
                vmin=mn,
                vmax=mx,
                markersize=markersize,
                cmap=cmap,
                **points_kwds,
            )

        if legend:
            # check if the colorbar needs to show value truncation
            if "extend" not in legend_kwds:
                if (mn > values_min) & (mx < values_max):
                    legend_kwds["extend"] = "both"
                elif mn > values_min:
                    legend_kwds["extend"] = "min"
                elif mx < values_max:
                    legend_kwds["extend"] = "max"

            # shrink the colorbar based on the new apect ratio - that way we ensure
            # that it is never much larger than the axis without complicated hacks
            bbox = ax.get_position()
            bbox_orig = ax.get_position(original=True)
            if "shrink" not in legend_kwds:
                if (
                    legend_kwds.get("location", "right")
                    in [
                        "top",
                        "bottom",
                    ]
                    or legend_kwds.get("orientation", "vertical") == "horizontal"
                ):
                    ratio = bbox.width / bbox_orig.width
                else:
                    ratio = bbox.height / bbox_orig.height
                legend_kwds["shrink"] = ratio
                legend_kwds["aspect"] = ratio * 20

            mappable = cm.ScalarMappable(
                norm=style_kwds.get("norm", colors.Normalize(vmin=mn, vmax=mx)),
                cmap=cmap,
            )
            ax.figure.colorbar(
                mappable,
                ax=ax,
                cax=cax,
                **legend_kwds,
            )

        missing_geoms = expl_series[nan_idx]
        missing_data = not missing_geoms.empty

    if missing_kwds is not None and missing_data:
        merged_kwds = style_kwds.copy()
        merged_kwds.update(missing_kwds)

        # ensure we take proper subset of list-like inputs related to missing
        # and clear all the dicts mapping to categories - user shall specify
        # style of missing in missing_kwds
        for key, val in merged_kwds.items():
            if isinstance(val, dict):
                merged_kwds[key] = None
            elif pd.api.types.is_list_like(val) and len(val) == len(df):
                merged_kwds[key] = val[nan_idx]

        plot_series(
            missing_geoms,
            ax=ax,
            aspect=None,
            label=merged_kwds.pop("label", "NaN"),
            **merged_kwds,
        )

    if categorical and legend:
        ax.legend(**legend_kwds)

        # if there is already a colorbar but we want a legend for missing data,
        # user can simply call `ax.legend()` with any custom keywords.

    ax.figure.canvas.draw_idle()
    return ax


def _check_invalid_categories(categories: Collection[Any], values) -> pd.Categorical:
    """
    Pandas 4 compat https://github.com/pandas-dev/pandas/pull/62142
    Could potentially be replaced with a try/except on the above once the warning
    becomes an exception. This logic is derived from
    pandas/core/arrays/categorical.py::_get_codes_for_values.
    """
    dtype = CategoricalDtype._from_values_or_dtype(values, categories)
    categories = dtype.categories
    codes = categories.get_indexer_for(values)
    wrong = (codes == -1) & ~pd.isna(values)
    if wrong.any():
        missing = list(np.unique(values[wrong]))
    else:
        missing = []
        codes_downcast = pd.core.dtypes.cast.coerce_indexer_dtype(codes, categories)
        cat = pd.Categorical.from_codes(codes_downcast, categories)

    if missing:
        raise ValueError(
            "Column contains values not listed in categories. "
            f"Missing categories: {missing}."
        )
    return cat


@doc(plot_dataframe)
class GeoplotAccessor(PlotAccessor):
    _pandas_kinds = PlotAccessor._all_kinds

    def __call__(self, *args, **kwargs):
        data = self._parent.copy()
        kind = kwargs.pop("kind", "geo")
        if kind == "geo":
            return plot_dataframe(data, *args, **kwargs)
        if kind in self._pandas_kinds:
            # Access pandas plots
            return PlotAccessor(data)(kind=kind, **kwargs)
        else:
            # raise error
            raise ValueError(f"{kind} is not a valid plot kind")

    def geo(self, *args, **kwargs):
        return self(kind="geo", *args, **kwargs)  # noqa: B026
