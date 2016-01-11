from __future__ import print_function

import warnings

import numpy as np
import pandas as pd
from six import next
from six.moves import xrange

def _flatten_multi_geoms(geoms, colors):
    """
    Returns Series like geoms and colors, except that any Multi geometries
    are split into their components and colors are repeated for all component
    in the same Multi geometry.  Maintains 1:1 matching of geometry to color.
    """
    components, component_colors = [], []
    assert len(geoms) == len(colors) # precondition, so zip can't short-circuit
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


def plot_polygon_collection(ax, geoms, facecolors, edgecolor='black',
                            alpha=0.5, linewidth=1.0, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------
    geoms : a sequence of shapely Polygons and/or MultiPolygons (can be mixed)
    facecolors : a sequence of RGBA tuples.
        It should have 1:1 correspondence with the geometries (not their components).

    Returns
    -------
    patches : matplotlib.collections.Collection
    """

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    components, component_colors = _flatten_multi_geoms(geoms, facecolors)

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']
    patches = [Polygon(poly.exterior) for poly in components]
    patches = PatchCollection(patches, facecolors=component_colors,
                              linewidth=linewidth, edgecolor=edgecolor,
                              alpha=alpha, **kwargs)
    # TODO: draw polygon interior(s)

    ax.add_collection(patches, autolim=True)
    ax.autoscale_view()
    return patches


def plot_linestring_collection(ax, geoms, colors, linewidth=1.0, **kwargs):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

    Parameters
    ----------
    geoms : a sequence of shapely LineString and/or MultiLineString (can be mixed)
    colors : a sequence of RGBA tuples.
        It should have 1:1 correspondence with the geometries (not their components).

    Returns
    -------
    collection : matplotlib.collections.Collection
    """

    from matplotlib.collections import LineCollection

    components, component_colors = _flatten_multi_geoms(geoms, colors)

    # LineCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']
    segments = [np.array(linestring)[:, :2] for linestring in components]
    collection = LineCollection(segments, color=component_colors,
                                linewidth=linewidth, **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_point_collection(ax, geoms, colors, marker='o', markersize=2, **kwargs):
    """
    Plots a collection of Point geometries to `ax`

    Parameters
    ----------
    geoms : a sequence of Points
    colors : a single color string or sequence of RGBA tuples.

    Returns
    -------
    collection : matplotlib.collections.Collection
    """
    x = [p.x for p in geoms]
    y = [p.y for p in geoms]
    collection = ax.scatter(x, y, c=colors, marker=marker, s=markersize, **kwargs)
    return collection


def gencolor(N, colormap='Set1'):
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
    for i in xrange(N):
        yield colors[i % n_colors]


def plot_series(s, cmap='Set1', color=None, ax=None, linewidth=1.0,
                figsize=None, **color_kwds):
    """ Plot a GeoSeries

        Generate a plot of a GeoSeries geometry with matplotlib.

        Parameters
        ----------

        Series
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
    if 'facecolor' in color_kwds:
        warnings.warn("'facecolor' is deprecated, please use 'color' instead "
                      "(for consistency across geometry types)", FutureWarning)
        color = color_kwds.pop('facecolor') if not color else color

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

    num_geoms = len(s.index)
    if color:
        colors = pd.Series([color] * num_geoms)
    else:
        color_generator = gencolor(len(s), colormap=cmap)
        colors = pd.Series([next(color_generator) for _ in xrange(num_geoms)])

    poly_idx = (s.geometry.type == 'Polygon') | (s.geometry.type == 'MultiPolygon')
    polys = s.geometry[poly_idx]
    if not polys.empty:
        plot_polygon_collection(ax, polys, colors[poly_idx], linewidth=linewidth, **color_kwds)

    line_idx = (s.geometry.type == 'LineString') | (s.geometry.type == 'MultiLineString')
    lines = s.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, colors[line_idx], linewidth=linewidth, **color_kwds)

    point_idx = (s.geometry.type == 'Point')
    points = s.geometry[point_idx]
    if not points.empty:
        plot_point_collection(ax, points, colors[point_idx], **color_kwds)

    plt.draw()
    return ax


def plot_dataframe(s, column=None, cmap=None, color=None, linewidth=1.0,
                   categorical=False, legend=False, ax=None,
                   scheme=None, k=5, vmin=None, vmax=None, figsize=None,
                   **color_kwds):
    """ Plot a GeoDataFrame

        Generate a plot of a GeoDataFrame with matplotlib.  If a
        column is specified, the plot coloring will be based on values
        in that column.  Otherwise, a categorical plot of the
        geometries in the `geometry` column will be generated.

        Parameters
        ----------

        GeoDataFrame
            The GeoDataFrame to be plotted.  Currently Polygon,
            MultiPolygon, LineString, MultiLineString and Point
            geometries can be plotted.

        column : str (default None)
            The name of the column to be plotted.

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
    if 'facecolor' in color_kwds:
        warnings.warn("'facecolor' is deprecated, please use 'color' instead "
                      "(for consistency across geometry types)", FutureWarning)
        color = color_kwds.pop('facecolor') if not color else color

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if column is None:
        return plot_series(s.geometry, cmap=cmap, color=color,
                           ax=ax, linewidth=linewidth, figsize=figsize,
                           **color_kwds)

    if s[column].dtype is np.dtype('O'):
        categorical = True
    if categorical:
        if cmap is None:
            cmap = 'Set1'
        categories = list(set(s[column].values))
        categories.sort()
        valuemap = dict([(k, v) for (v, k) in enumerate(categories)])
        values = [valuemap[k] for k in s[column]]
    else:
        values = s[column]
    if scheme is not None:
        binning = __pysal_choro(values, scheme, k=k)
        values = binning.yb
        # set categorical to True for creating the legend
        categorical = True
        binedges = [binning.yb.min()] + binning.bins.tolist()
        categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i+1])
                      for i in range(len(binedges)-1)]
    cmap = norm_cmap(values, cmap, Normalize, cm, vmin=vmin, vmax=vmax)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

    num_geoms = len(s.index)
    if color:
        colors = pd.Series([color] * num_geoms)
    else:
        colors = pd.Series([cmap.to_rgba(v) for v in values])

    # plot all Polygons and all components of MultiPolygon in the same collection
    poly_idx = (s.geometry.type == 'Polygon') | (s.geometry.type == 'MultiPolygon')
    polys = s.geometry[poly_idx]
    if not polys.empty:
        plot_polygon_collection(ax, polys, colors[poly_idx], linewidth=linewidth, **color_kwds)

    line_idx = (s.geometry.type == 'LineString') | (s.geometry.type == 'MultiLineString')
    lines = s.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, colors[line_idx], linewidth=linewidth, **color_kwds)

    point_idx = (s.geometry.type == 'Point')
    points = s.geometry[point_idx]
    if not points.empty:
        plot_point_collection(ax, points, colors[point_idx], **color_kwds)

    if legend and not color:
        if categorical:
            patches = []
            for value, cat in enumerate(categories):
                patches.append(Line2D([0], [0], linestyle="none",
                                      marker="o", alpha=color_kwds.get('alpha', 0.5),
                                      markersize=10, markerfacecolor=cmap.to_rgba(value)))
            ax.legend(patches, categories, numpoints=1, loc='best')
        else:
            ax.get_figure().colorbar(cmap)

    plt.draw()
    return ax


def __pysal_choro(values, scheme, k=5):
    """ Wrapper for choropleth schemes from PySAL for use with plot_dataframe

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
        from pysal.esda.mapclassify import Quantiles, Equal_Interval, Fisher_Jenks
        schemes = {}
        schemes['equal_interval'] = Equal_Interval
        schemes['quantiles'] = Quantiles
        schemes['fisher_jenks'] = Fisher_Jenks
        s0 = scheme
        scheme = scheme.lower()
        if scheme not in schemes:
            scheme = 'quantiles'
            warnings.warn('Unrecognized scheme "{0}". Using "Quantiles" '
                          'instead'.format(s0), UserWarning, stacklevel=3)
        if k < 2 or k > 9:
            warnings.warn('Invalid k: {0} (2 <= k <= 9), setting k=5 '
                          '(default)'.format(k), UserWarning, stacklevel=3)
            k = 5
        binning = schemes[scheme](values, k)
        return binning
    except ImportError:
        raise ImportError("PySAL is required to use the 'scheme' keyword")


def norm_cmap(values, cmap, normalize, cm, vmin=None, vmax=None):

    """ Normalize and set colormap

        Parameters
        ----------

        values
            Series or array to be normalized

        cmap
            matplotlib Colormap

        normalize
            matplotlib.colors.Normalize

        cm
            matplotlib.cm

        vmin
            Minimum value of colormap. If None, uses min(values).

        vmax
            Maximum value of colormap. If None, uses max(values).

        Returns
        -------
        n_cmap
            mapping of normalized values to colormap (cmap)

    """

    mn = min(values) if vmin is None else vmin
    mx = max(values) if vmax is None else vmax
    norm = normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    n_cmap.set_array([])
    return n_cmap
