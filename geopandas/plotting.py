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

    "Colors" are treated opaquely and so can actually contain any values.

    Returns
    -------

    components : list of geometry

    component_colors : list of whatever type `colors` contains
    """
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


def plot_polygon_collection(ax, geoms, colors_or_values, plot_values,
                            vmin=None, vmax=None, cmap=None,
                            edgecolor='black', alpha=0.5, linewidth=1.0, **kwargs):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)

    colors_or_values : a sequence of `N` values or RGBA tuples
        It should have 1:1 correspondence with the geometries (not their components).

    plot_values : bool
        If True, `colors_or_values` is interpreted as a list of values, and will
        be mapped to colors using vmin/vmax/cmap (which become required).
        Otherwise `colors_or_values` is interpreted as a list of colors.

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted
    """

    from descartes.patch import PolygonPatch
    from matplotlib.collections import PatchCollection

    components, component_colors_or_values = _flatten_multi_geoms(
        geoms, colors_or_values)

    # PatchCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']
    collection = PatchCollection([PolygonPatch(poly) for poly in components],
                                 linewidth=linewidth, edgecolor=edgecolor,
                                 alpha=alpha, **kwargs)

    if plot_values:
        collection.set_array(np.array(component_colors_or_values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)
    else:
        # set_color magically sets the correct combination of facecolor and
        # edgecolor, based on collection type.
        collection.set_color(component_colors_or_values)

        # If the user set facecolor and/or edgecolor explicitly, the previous
        # call to set_color might have overridden it (remember, the 'color' may
        # have come from plot_series, not from the user). The user should be
        # able to override matplotlib's default behavior, by setting them again
        # after set_color.
        if 'facecolor' in kwargs:
            collection.set_facecolor(kwargs['facecolor'])
        if edgecolor:
            collection.set_edgecolor(edgecolor)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_linestring_collection(ax, geoms, colors_or_values, plot_values,
                               vmin=None, vmax=None, cmap=None,
                               linewidth=1.0, **kwargs):
    """
    Plots a collection of LineString and MultiLineString geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : a sequence of `N` LineStrings and/or MultiLineStrings (can be mixed)

    colors_or_values : a sequence of `N` values or RGBA tuples
        It should have 1:1 correspondence with the geometries (not their components).

    plot_values : bool
        If True, `colors_or_values` is interpreted as a list of values, and will
        be mapped to colors using vmin/vmax/cmap (which become required).
        Otherwise `colors_or_values` is interpreted as a list of colors.

    Returns
    -------

    collection : matplotlib.collections.Collection that was plotted
    """

    from matplotlib.collections import LineCollection

    components, component_colors_or_values = _flatten_multi_geoms(
        geoms, colors_or_values)

    # LineCollection does not accept some kwargs.
    if 'markersize' in kwargs:
        del kwargs['markersize']
    segments = [np.array(linestring)[:, :2] for linestring in components]
    collection = LineCollection(segments,
                                linewidth=linewidth, **kwargs)

    if plot_values:
        collection.set_array(np.array(component_colors_or_values))
        collection.set_cmap(cmap)
        collection.set_clim(vmin, vmax)
    else:
        # set_color magically sets the correct combination of facecolor and
        # edgecolor, based on collection type.
        collection.set_color(component_colors_or_values)

        # If the user set facecolor and/or edgecolor explicitly, the previous
        # call to set_color might have overridden it (remember, the 'color' may
        # have come from plot_series, not from the user). The user should be
        # able to override matplotlib's default behavior, by setting them again
        # after set_color.
        if 'facecolor' in kwargs:
            collection.set_facecolor(kwargs['facecolor'])

        if 'edgecolor' in kwargs:
            collection.set_edgecolor(kwargs['edgecolor'])

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def plot_point_collection(ax, geoms, colors_or_values,
                          vmin=None, vmax=None, cmap=None,
                          marker='o', markersize=2, **kwargs):
    """
    Plots a collection of Point geometries to `ax`

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        where shapes will be plotted

    geoms : sequence of `N` Points

    colors_or_values : sequence of color or sequence of numbers
        can be a sequence of color specifications of length `N` or a sequence
        of `N` numbers to be mapped to colors using vmin, vmax, and cmap.

    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    x = [p.x for p in geoms]
    y = [p.y for p in geoms]

    # matplotlib ax.scatter requires RGBA color specifications to be a single 2D
    # array, NOT merely a list of 1D arrays. This reshapes that if necessary,
    # having no effect on 1D arrays of values.
    colors_or_values = np.array([element
                                 for _, element in enumerate(colors_or_values)])
    collection = ax.scatter(x, y, c=colors_or_values,
                            vmin=vmin, vmax=vmax, cmap=cmap,
                            marker=marker, s=markersize, **kwargs)
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

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

    num_geoms = len(s.index)
    if color:
        colors = np.array([color] * num_geoms)
    else:
        color_generator = gencolor(len(s), colormap=cmap)
        colors = np.array([next(color_generator) for _ in xrange(num_geoms)])

    # plot all Polygons and all MultiPolygon components in the same collection
    poly_idx = np.array(
        (s.geometry.type == 'Polygon') | (s.geometry.type == 'MultiPolygon'))
    polys = s.geometry[poly_idx]
    if not polys.empty:
        # Legacy behavior applies alpha to fill but not to edges. This requires
        # plotting them separately (at big performance expense).
        if linewidth > 0 and color_kwds.get('alpha', 0.5) < 1.0:
            # Plot the fill with default or user-specified alpha, but do not
            # draw outlines.
            plot_polygon_collection(ax, polys, colors[poly_idx], False,
                                    linewidth=0, **color_kwds)
            # Draw the edges, fully opaque, but no facecolor.
            edges_kwds = color_kwds.copy()
            edges_kwds['alpha'] = 1
            edges_kwds['facecolor'] = 'none'
            plot_polygon_collection(ax, polys, colors[poly_idx], False,
                                    linewidth=linewidth, **edges_kwds)
        else:
            # Optimization: if no alpha on fill, or if no edges, we can plot
            # everything in one go.
            plot_polygon_collection(ax, polys, colors[poly_idx], False,
                                    linewidth=linewidth, **color_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    line_idx = np.array(
        (s.geometry.type == 'LineString') |
        (s.geometry.type == 'MultiLineString'))
    lines = s.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, colors[line_idx], False,
                                   linewidth=linewidth, **color_kwds)

    point_idx = np.array(s.geometry.type == 'Point')
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
        warnings.warn("Only specify one of 'column' or 'color'. Using 'color'.",
                      SyntaxWarning)
        column = None

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

    # Define `values` as a Series
    if categorical:
        if cmap is None:
            cmap = 'Set1'
        categories = list(set(s[column].values))
        categories.sort()
        valuemap = dict([(k, v) for (v, k) in enumerate(categories)])
        values = np.array([valuemap[k] for k in s[column]])
    else:
        values = s[column]
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

    # plot all Polygons and all MultiPolygon components in the same collection
    poly_idx = np.array(
        (s.geometry.type == 'Polygon') | (s.geometry.type == 'MultiPolygon'))
    polys = s.geometry[poly_idx]
    if not polys.empty:
        # Legacy behavior applies alpha to fill but not to edges. This requires
        # plotting them separately (at big performance expense).
        if linewidth > 0 and color_kwds.get('alpha', 0.5) < 1.0:
            # Plot the fill with default or user-specified alpha, but do not
            # draw outlines.
            plot_polygon_collection(ax, polys, values[poly_idx], True,
                                    vmin=mn, vmax=mx, cmap=cmap,
                                    linewidth=0, **color_kwds)
            # Draw the edges, fully opaque, but no facecolor.
            edges_kwds = color_kwds.copy()
            edges_kwds['alpha'] = 1
            edges_kwds['facecolor'] = 'none'
            # Setting plot_values=False would cause the array values' colors to
            # override edgecolor. By setting color instead, matplotlib will
            # respect edgecolor if set.
            plot_polygon_collection(ax, polys, ['black'] * len(polys), False,
                                    linewidth=linewidth, **edges_kwds)
        else:
            # Optimization: if no alpha on fill, or if no edges, we can plot
            # everything in one go.
            plot_polygon_collection(ax, polys, values[poly_idx], True,
                                    vmin=mn, vmax=mx, cmap=cmap,
                                    linewidth=linewidth, **color_kwds)

    # plot all LineStrings and MultiLineString components in same collection
    line_idx = np.array(
        (s.geometry.type == 'LineString') |
        (s.geometry.type == 'MultiLineString'))
    lines = s.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, values[line_idx], True,
                                   vmin=mn, vmax=mx, cmap=cmap,
                                   linewidth=linewidth, **color_kwds)

    point_idx = np.array(s.geometry.type == 'Point')
    points = s.geometry[point_idx]
    if not points.empty:
        plot_point_collection(ax, points, values[point_idx],
                              vmin=mn, vmax=mx, cmap=cmap, **color_kwds)

    if legend and not color:
        norm = Normalize(vmin=mn, vmax=mx)
        n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        if categorical:
            patches = []
            for value, cat in enumerate(categories):
                patches.append(Line2D([0], [0], linestyle="none",
                                      marker="o", alpha=color_kwds.get('alpha', 0.5),
                                      markersize=10, markerfacecolor=n_cmap.to_rgba(value)))
            ax.legend(patches, categories, numpoints=1, loc='best')
        else:
            n_cmap.set_array([])
            ax.get_figure().colorbar(n_cmap)

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
        scheme = scheme.lower()
        if scheme not in schemes:
            raise ValueError("Invalid scheme. Scheme must be in the set: %r" % schemes.keys())
        binning = schemes[scheme](values, k)
        return binning
    except ImportError:
        raise ImportError("PySAL is required to use the 'scheme' keyword")
