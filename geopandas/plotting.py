from __future__ import print_function

import warnings

import numpy as np
from six import next


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


def plot_polygon_collection(ax, geoms, values=None, linewidth=1.0,
                            edgecolor='black', alpha=0.5,
                            vmin=None, vmax=None, cmap=None, **kwargs):
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
    from descartes.patch import PolygonPatch
    from matplotlib.collections import PatchCollection

    geoms, values = _flatten_multi_geoms(geoms, values)
    if None in values:
        values = None

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']

    collection = PatchCollection([PolygonPatch(poly) for poly in geoms],
                                 linewidth=linewidth, edgecolor=edgecolor,
                                 alpha=alpha, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_linestring_collection(ax, geoms, values=None, color=None,
                               vmin=None, vmax=None, cmap=None,
                               linewidth=1.0, **kwargs):
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
    collection = LineCollection(segments, linewidth=linewidth, **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_point_collection(ax, geoms, values=None, color=None,
                          vmin=None, vmax=None, cmap=None,
                          marker='o', markersize=2, **kwargs):
    """
    Plots a collection of Point geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : sequence of `N` Points

    values : a sequence of `N` values, optional
        Values mapped to colors using vmin, vmax, and cmap.
        Cannot be specified together with `color`.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    if values is not None and color is not None:
        raise ValueError("Can only specify one of 'values' and 'color' kwargs")

    x = geoms.x.values
    y = geoms.y.values

    collection = ax.scatter(x, y, s=markersize, c=values, color=color,
                            vmin=vmin, vmax=vmax, cmap=cmap,
                            marker=marker, **kwargs)
    return collection


def _gencolor(N, colormap='Set1'):
    """
    Color generator intended to work with one of the ColorBrewer
    qualitative color scales.

    Suggested values of colormap are the following:

        Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

    (although any matplotlib colormap will work).
    """
    from matplotlib import cm
    # don't use more than 9 discrete colors
    n_colors = min(N, 9)
    cmap = cm.get_cmap(colormap, n_colors)
    colors = cmap(range(n_colors))
    for i in range(N):
        yield colors[i % n_colors]


def plot_series(s, cmap='Set1', color=None, ax=None, linewidth=1.0,
                figsize=None, **color_kwds):
    """
    Plot a GeoSeries.

    Generate a plot of a GeoSeries geometry with matplotlib.

    Parameters
    ----------

    s : Series
        The GeoSeries to be plotted.  Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.

    cmap : str (default 'Set1')
        The name of a colormap recognized by matplotlib.  Any
        colormap will work, but categorical colormaps are
        generally recommended.  Examples of useful discrete
        colormaps include:

            Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

    color : str (default None)
        If specified, all objects will be colored uniformly.

    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot

    linewidth : float (default 1.0)
        Line width for geometries.

    figsize : pair of floats (default None)
        Size of the resulting matplotlib.figure.Figure. If the argument
        ax is given explicitly, figsize is ignored.

    **color_kwds : dict
        Color options to be passed on to the actual plot function

    Returns
    -------

    matplotlib axes instance
    """
    if 'colormap' in color_kwds:
        warnings.warn("'colormap' is deprecated, please use 'cmap' instead "
                      "(for consistency with matplotlib)", FutureWarning)
        cmap = color_kwds.pop('colormap')
    if 'axes' in color_kwds:
        warnings.warn("'axes' is deprecated, please use 'ax' instead "
                      "(for consistency with pandas)", FutureWarning)
        ax = color_kwds.pop('axes')

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

    # if no color specified, create range of colors based on cmap
    num_geoms = len(s.index)
    col_seq = False
    if color is None:
        color_generator = _gencolor(len(s), colormap=cmap)
        color = np.array([next(color_generator) for _ in range(num_geoms)])
        col_seq = True

    geom_types = s.geometry.type
    poly_idx = np.asarray((geom_types == 'Polygon')
                          | (geom_types == 'MultiPolygon'))
    line_idx = np.asarray((geom_types == 'LineString')
                          | (geom_types == 'MultiLineString'))
    point_idx = np.asarray(geom_types == 'Point')

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = s.geometry[poly_idx]

    if not polys.empty:
        # color overrides both face and edgecolor. As we want people to be
        # able to use edgecolor as well, pass color to facecolor
        facecolor = color_kwds.pop('facecolor', None)
        if col_seq:
            if not facecolor:
                facecolor = color[poly_idx] if col_seq else color
        else:
            facecolor = color
        plot_polygon_collection(ax, polys, facecolor=facecolor,
                                linewidth=linewidth, **color_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    lines = s.geometry[line_idx]
    if not lines.empty:
        color_ = color[line_idx] if col_seq else color
        plot_linestring_collection(ax, lines, color=color_,
                                   linewidth=linewidth, **color_kwds)

    # plot all Points in the same collection
    points = s.geometry[point_idx]
    if not points.empty:
        color_ = color[point_idx] if col_seq else color
        plot_point_collection(ax, points, color=color_, **color_kwds)

    plt.draw()
    return ax


def plot_dataframe(df, column=None, cmap=None, color=None, linewidth=1.0,
                   categorical=False, legend=False, ax=None,
                   scheme=None, k=5, vmin=None, vmax=None, figsize=None,
                   **color_kwds):
    """
    Plot a GeoDataFrame.

    Generate a plot of a GeoDataFrame with matplotlib.  If a
    column is specified, the plot coloring will be based on values
    in that column.  Otherwise, a categorical plot of the
    geometries in the `geometry` column will be generated.

    Parameters
    ----------

    df : GeoDataFrame
        The GeoDataFrame to be plotted.  Currently Polygon,
        MultiPolygon, LineString, MultiLineString and Point
        geometries can be plotted.

    column : str (default None)
        The name of the column to be plotted. Ignored if `color` is also set.

    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted.  For non-numerical columns (or if
        column=None), this will be set to True.

    cmap : str (default 'Set1')
        The name of a colormap recognized by matplotlib.

    color : str (default None)
        If specified, all objects will be colored uniformly.

    linewidth : float (default 1.0)
        Line width for geometries.

    legend : bool (default False)
        Plot a legend. Ignored if no `column` is given, or if `color` is given.

    ax : matplotlib.pyplot.Artist (default None)
        axes on which to draw the plot

    scheme : pysal.esda.mapclassify.Map_Classifier
        Choropleth classification schemes (requires PySAL)

    k   : int (default 5)
        Number of classes (ignored if scheme is None)

    vmin : None or float (default None)

        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.

    vmax : None or float (default None)

        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.

    figsize
        Size of the resulting matplotlib.figure.Figure. If the argument
        axes is given explicitly, figsize is ignored.

    **color_kwds : dict
        Color options to be passed on to the actual plot function

    Returns
    -------

    matplotlib axes instance
    """
    if 'colormap' in color_kwds:
        warnings.warn("'colormap' is deprecated, please use 'cmap' instead "
                      "(for consistency with matplotlib)", FutureWarning)
        cmap = color_kwds.pop('colormap')
    if 'axes' in color_kwds:
        warnings.warn("'axes' is deprecated, please use 'ax' instead "
                      "(for consistency with pandas)", FutureWarning)
        ax = color_kwds.pop('axes')
    if column and color:
        warnings.warn("Only specify one of 'column' or 'color'. Using "
                      "'color'.", UserWarning)
        column = None

    import matplotlib.pyplot as plt

    if column is None:
        return plot_series(df.geometry, cmap=cmap, color=color,
                           ax=ax, linewidth=linewidth, figsize=figsize,
                           **color_kwds)

    if df[column].dtype is np.dtype('O'):
        categorical = True

    # Define `values` as a Series
    if categorical:
        if cmap is None:
            cmap = 'Set1'
        categories = list(set(df[column].values))
        categories.sort()
        valuemap = dict([(k, v) for (v, k) in enumerate(categories)])
        values = np.array([valuemap[k] for k in df[column]])
    else:
        values = df[column]
    if scheme is not None:
        binning = __pysal_choro(values, scheme, k=k)
        values = np.array(binning.yb)
        # set categorical to True for creating the legend
        categorical = True
        binedges = [binning.yb.min()] + binning.bins.tolist()
        categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i+1])
                      for i in range(len(binedges)-1)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

    mn = values.min() if vmin is None else vmin
    mx = values.max() if vmax is None else vmax

    geom_types = df.geometry.type
    poly_idx = np.asarray((geom_types == 'Polygon')
                          | (geom_types == 'MultiPolygon'))
    line_idx = np.asarray((geom_types == 'LineString')
                          | (geom_types == 'MultiLineString'))
    point_idx = np.asarray(geom_types == 'Point')

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = df.geometry[poly_idx]
    if not polys.empty:
        plot_polygon_collection(ax, polys, values[poly_idx],
                                vmin=mn, vmax=mx, cmap=cmap,
                                linewidth=linewidth, **color_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    lines = df.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, values[line_idx],
                                   vmin=mn, vmax=mx, cmap=cmap,
                                   linewidth=linewidth, **color_kwds)

    # plot all Points in the same collection
    points = df.geometry[point_idx]
    if not points.empty:
        plot_point_collection(ax, points, values[point_idx],
                              vmin=mn, vmax=mx, cmap=cmap, **color_kwds)

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
                           alpha=color_kwds.get('alpha', 0.5), markersize=10,
                           markerfacecolor=n_cmap.to_rgba(value)))
            ax.legend(patches, categories, numpoints=1, loc='best')
        else:
            n_cmap.set_array([])
            ax.get_figure().colorbar(n_cmap)

    plt.draw()
    return ax


def __pysal_choro(values, scheme, k=5):
    """
    Wrapper for choropleth schemes from PySAL for use with plot_dataframe

    Parameters
    ----------

    values
        Series to be plotted

    scheme
        pysal.esda.mapclassify classificatin scheme
        ['Equal_interval'|'Quantiles'|'Fisher_Jenks']

    k
        number of classes (2 <= k <=9)

    Returns
    -------

    binning
        Binning objects that holds the Series with values replaced with
        class identifier and the bins.

    """
    try:
        from pysal.esda.mapclassify import (
            Quantiles, Equal_Interval, Fisher_Jenks)
        schemes = {}
        schemes['equal_interval'] = Equal_Interval
        schemes['quantiles'] = Quantiles
        schemes['fisher_jenks'] = Fisher_Jenks
        scheme = scheme.lower()
        if scheme not in schemes:
            raise ValueError("Invalid scheme. Scheme must be in the"
                             " set: %r" % schemes.keys())
        binning = schemes[scheme](values, k)
        return binning
    except ImportError:
        raise ImportError("PySAL is required to use the 'scheme' keyword")
