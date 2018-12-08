from __future__ import print_function
from distutils.version import LooseVersion
import warnings

import numpy as np
import pandas as pd

def _flatten_multi_geoms(geoms, colors=None):
    """
    Returns Series like geoms and colors, except that any Multi geometries
    are split into their components and colors are repeated for all component
    in the same Multi geometry.  Maintains 1:1 matching of geometry to color.
    Passing `color` is optional, and when no `color` is passed a list of None
    values is returned as `component_colors`.

    "Colors" are treated opaquely and so can actually contain any values.

    Returns
    -------

    components : list of geometry

    component_colors : list of whatever type `colors` contains
    """
    if colors is None:
        colors = [None] * len(geoms)

    components, component_colors = [], []

    if not geoms.geom_type.str.startswith('Multi').any():
        return geoms, colors

    # precondition, so zip can't short-circuit
    assert len(geoms) == len(colors)
    for geom, color in zip(geoms, colors):
        if geom.type.startswith('Multi'):
            for poly in geom:
                components.append(poly)
                # repeat same color for all components
                component_colors.append(color)
        else:
            components.append(geom)
            component_colors.append(color)

    return components, component_colors


def plot_polygon_collection(ax, geoms, values=None, color=None,
                            cmap=None, vmin=None, vmax=None, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise follows `color` / `facecolor` kwargs.

    edgecolor : single color or sequence of `N` colors
        Color for the edge of the polygons

    facecolor : single color or sequence of `N` colors
        Color to fill the polygons. Cannot be used together with `values`.

    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`

    **kwargs
        Additional keyword arguments passed to the collection

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted
    """

    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError("The descartes package is required"
                          " for plotting polygons in geopandas.")
    from matplotlib.collections import PatchCollection

    geoms, values = _flatten_multi_geoms(geoms, values)
    if None in values:
        values = None

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']

    # color=None overwrites specified facecolor/edgecolor with default color
    if color is not None:
        kwargs['color'] = color

    collection = PatchCollection([PolygonPatch(poly) for poly in geoms],
                                 **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_linestring_collection(ax, geoms, values=None, color=None,
                               cmap=None, vmin=None, vmax=None, **kwargs):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

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

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted

    """
    from matplotlib.collections import LineCollection

    geoms, values = _flatten_multi_geoms(geoms, values)
    if None in values:
        values = None

    # LineCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']

    # color=None gives black instead of default color cycle
    if color is not None:
        kwargs['color'] = color

    segments = [np.array(linestring)[:, :2] for linestring in geoms]
    collection = LineCollection(segments, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_point_collection(ax, geoms, values=None, color=None,
                          cmap=None, vmin=None, vmax=None,
                          marker='o', markersize=None, **kwargs):
    """
    Plots a collection of Point and MultiPoint geometries to `ax`

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
    if values is not None and color is not None:
        raise ValueError("Can only specify one of 'values' and 'color' kwargs")

    geoms, values = _flatten_multi_geoms(geoms, values)
    if None in values:
        values = None
    x = [p.x for p in geoms]
    y = [p.y for p in geoms]

    # matplotlib 1.4 does not support c=None, and < 2.0 does not support s=None
    if values is not None:
        kwargs['c'] = values
    if markersize is not None:
        kwargs['s'] = markersize

    collection = ax.scatter(x, y, color=color, vmin=vmin, vmax=vmax, cmap=cmap,
                            marker=marker, **kwargs)
    return collection


def plot_series(s, cmap=None, color=None, ax=None, figsize=None, **style_kwds):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:

            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2

    color : str (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance
    """
    if 'colormap' in style_kwds:
        warnings.warn("'colormap' is deprecated, please use 'cmap' instead "
                      "(for consistency with matplotlib)", FutureWarning)
        cmap = style_kwds.pop('colormap')
    if 'axes' in style_kwds:
        warnings.warn("'axes' is deprecated, please use 'ax' instead "
                      "(for consistency with pandas)", FutureWarning)
        ax = style_kwds.pop('axes')

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    if s.empty:
        warnings.warn("The GeoSeries you are attempting to plot is "
                      "empty. Nothing has been displayed.", UserWarning)
        return ax

    # if cmap is specified, create range of colors based on cmap
    values = None
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, 'N'):
            values = values % cmap.N
        style_kwds['vmin'] = style_kwds.get('vmin', values.min())
        style_kwds['vmax'] = style_kwds.get('vmax', values.max())

    geom_types = s.geometry.type
    poly_idx = np.asarray((geom_types == 'Polygon')
                          | (geom_types == 'MultiPolygon'))
    line_idx = np.asarray((geom_types == 'LineString')
                          | (geom_types == 'MultiLineString'))
    point_idx = np.asarray((geom_types == 'Point')
                          | (geom_types == 'MultiPoint'))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = s.geometry[poly_idx]

    if not polys.empty:
        # color overrides both face and edgecolor. As we want people to be
        # able to use edgecolor as well, pass color to facecolor
        facecolor = style_kwds.pop('facecolor', None)
        if color is not None:
            facecolor = color
        values_ = values[poly_idx] if cmap else None
        plot_polygon_collection(ax, polys, values_, facecolor=facecolor,
                                cmap=cmap, **style_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    lines = s.geometry[line_idx]
    if not lines.empty:
        values_ = values[line_idx] if cmap else None
        plot_linestring_collection(ax, lines, values_, color=color, cmap=cmap,
                                   **style_kwds)

    # plot all Points in the same collection
    points = s.geometry[point_idx]
    if not points.empty:
        values_ = values[point_idx] if cmap else None
        plot_point_collection(ax, points, values_, color=color, cmap=cmap,
                              **style_kwds)

    plt.draw()
    return ax


def plot_dataframe(df, column=None, cmap=None, color=None, ax=None,
                   categorical=False, legend=False, scheme=None, k=5,
                   vmin=None, vmax=None, markersize=None, figsize=None,
                   legend_kwds=None, **style_kwds):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.

    Parameters
    ----------
    df : GeoDataFrame
        The GeoDataFrame to be plotted.  Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as
        dataframe. Values are used to color the plot. Ignored if `color` is
        also set.
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    color : str (default None)
        If specified, all objects will be colored uniformly.
    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns, this
        will be set to True.
    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.Map_Classifier object will be used
        under the hood. Supported schemes: 'Quantiles',
        'Equal_Interval', 'Fisher_Jenks', 'Fisher_Jenks_Sampled'
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
        Keyword arguments to pass to ax.legend()

    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``edgecolor``, ``facecolor``, ``linewidth``, ``markersize``,
        ``alpha``.

    Returns
    -------
    ax : matplotlib axes instance

    """
    if 'colormap' in style_kwds:
        warnings.warn("'colormap' is deprecated, please use 'cmap' instead "
                      "(for consistency with matplotlib)", FutureWarning)
        cmap = style_kwds.pop('colormap')
    if 'axes' in style_kwds:
        warnings.warn("'axes' is deprecated, please use 'ax' instead "
                      "(for consistency with pandas)", FutureWarning)
        ax = style_kwds.pop('axes')
    if column is not None and color is not None:
        warnings.warn("Only specify one of 'column' or 'color'. Using "
                      "'color'.", UserWarning)
        column = None

    import matplotlib
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    if df.empty:
        warnings.warn("The GeoDataFrame you are attempting to plot is "
                      "empty. Nothing has been displayed.", UserWarning)
        return ax

    if isinstance(markersize, str):
        markersize = df[markersize].values

    if column is None:
        return plot_series(df.geometry, cmap=cmap, color=color, ax=ax,
                           figsize=figsize, markersize=markersize,
                           **style_kwds)

    # To accept pd.Series and np.arrays as column
    if isinstance(column, (np.ndarray, pd.Series)):
        if column.shape[0] != df.shape[0]:
            raise ValueError("The dataframe and given column have different "
                             "number of rows.")
        else:
            values = np.asarray(column)
    else:
        values = np.asarray(df[column])

    if values.dtype is np.dtype('O'):
        categorical = True

    # Define `values` as a Series
    if categorical:
        if cmap is None:
            if LooseVersion(matplotlib.__version__) >= '2.0.1':
                cmap = 'tab10'
            elif LooseVersion(matplotlib.__version__) >= '2.0.0':
                # Erroneous name.
                cmap = 'Vega10'
            else:
                cmap = 'Set1'
        categories = list(set(values))
        categories.sort()
        valuemap = dict((k, v) for (v, k) in enumerate(categories))
        values = np.array([valuemap[k] for k in values])

    if scheme is not None:
        binning = _mapclassify_choro(values, scheme, k=k)
        # set categorical to True for creating the legend
        categorical = True
        binedges = [values.min()] + binning.bins.tolist()
        categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i+1])
                      for i in range(len(binedges)-1)]
        values = np.array(binning.yb)

    mn = values.min() if vmin is None else vmin
    mx = values.max() if vmax is None else vmax

    geom_types = df.geometry.type
    poly_idx = np.asarray((geom_types == 'Polygon')
                          | (geom_types == 'MultiPolygon'))
    line_idx = np.asarray((geom_types == 'LineString')
                          | (geom_types == 'MultiLineString'))
    point_idx = np.asarray((geom_types == 'Point')
                          | (geom_types == 'MultiPoint'))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = df.geometry[poly_idx]
    if not polys.empty:
        plot_polygon_collection(ax, polys, values[poly_idx],
                                vmin=mn, vmax=mx, cmap=cmap, **style_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    lines = df.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, values[line_idx],
                                   vmin=mn, vmax=mx, cmap=cmap, **style_kwds)

    # plot all Points in the same collection
    points = df.geometry[point_idx]
    if not points.empty:
        if isinstance(markersize, np.ndarray):
            markersize = markersize[point_idx]
        plot_point_collection(ax, points, values[point_idx], vmin=mn, vmax=mx,
                              markersize=markersize, cmap=cmap,
                              **style_kwds)

    if legend and not color:
        from matplotlib.lines import Line2D
        from matplotlib.colors import Normalize
        from matplotlib import cm

        norm = Normalize(vmin=mn, vmax=mx)
        n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        if categorical:
            patches = []
            for value, cat in enumerate(categories):
                patches.append(
                    Line2D([0], [0], linestyle="none", marker="o",
                           alpha=style_kwds.get('alpha', 1), markersize=10,
                           markerfacecolor=n_cmap.to_rgba(value),
                           markeredgewidth=0))
            if legend_kwds is None:
                legend_kwds = {}
            legend_kwds.setdefault('numpoints', 1)
            legend_kwds.setdefault('loc', 'best')
            ax.legend(patches, categories, **legend_kwds)
        else:
            n_cmap.set_array([])
            ax.get_figure().colorbar(n_cmap, ax=ax)

    plt.draw()
    return ax


def _mapclassify_choro(values, scheme, k=5):
    """
    Wrapper for choropleth schemes from mapclassify for use with plot_dataframe

    Parameters
    ----------
    values
        Series to be plotted
    scheme : str
        One of mapclassify classification schemes
        Options are 'Quantiles', 'Equal_Interval', 'Fisher_Jenks',
        'Fisher_Jenks_Sampled'
    k : int
        number of classes (2 <= k <=9)

    Returns
    -------
    binning
        Binning objects that holds the Series with values replaced with
        class identifier and the bins.

    """
    try:
        from mapclassify import (
            Quantiles, Equal_Interval, Fisher_Jenks,
            Fisher_Jenks_Sampled)
    except ImportError:
        try:
            from mapclassify.api import (
                Quantiles, Equal_Interval, Fisher_Jenks,
                Fisher_Jenks_Sampled)
        except ImportError:
            raise ImportError(
                "The 'mapclassify' package is required to use the 'scheme' "
                "keyword")

    schemes = {}
    schemes['equal_interval'] = Equal_Interval
    schemes['quantiles'] = Quantiles
    schemes['fisher_jenks'] = Fisher_Jenks
    schemes['fisher_jenks_sampled'] = Fisher_Jenks_Sampled

    scheme = scheme.lower()
    if scheme not in schemes:
        raise ValueError("Invalid scheme. Scheme must be in the"
                         " set: %r" % schemes.keys())
    binning = schemes[scheme](values, k)
    return binning
