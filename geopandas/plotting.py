from __future__ import print_function

import numpy as np
from six import next
from six.moves import xrange
from shapely.geometry import Polygon


def plot_polygon(ax, poly, facecolor='red', edgecolor='black', alpha=0.5, linewidth=1.0):
    """ Plot a single Polygon geometry """
    from descartes.patch import PolygonPatch
    a = np.asarray(poly.exterior)
    if poly.has_z:
        poly = Polygon(zip(*poly.exterior.xy))

    # without Descartes, we could make a Patch of exterior
    ax.add_patch(PolygonPatch(poly, facecolor=facecolor, linewidth=0, alpha=alpha))  # linewidth=0 because boundaries are drawn separately
    ax.plot(a[:, 0], a[:, 1], color=edgecolor, linewidth=linewidth)
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth)


def plot_multipolygon(ax, geom, facecolor='red', edgecolor='black', alpha=0.5, linewidth=1.0):
    """ Can safely call with either Polygon or Multipolygon geometry
    """
    if geom.type == 'Polygon':
        plot_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            plot_polygon(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)


def plot_linestring(ax, geom, color='black', linewidth=1.0):
    """ Plot a single LineString geometry """
    a = np.array(geom)
    ax.plot(a[:, 0], a[:, 1], color=color, linewidth=linewidth)


def plot_multilinestring(ax, geom, color='red', linewidth=1.0):
    """ Can safely call with either LineString or MultiLineString geometry
    """
    if geom.type == 'LineString':
        plot_linestring(ax, geom, color=color, linewidth=linewidth)
    elif geom.type == 'MultiLineString':
        for line in geom.geoms:
            plot_linestring(ax, line, color=color, linewidth=linewidth)


def plot_point(ax, pt, color='b', marker='o', markersize=2):
    """ Plot a single Point geometry """
    ax.scatter(pt.x, pt.y,
            color=color,
            marker=marker,
            s=markersize,
            edgecolor='')


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


def plot_series(s, colormap='Set1', axes=None, linewidth=1.0, figsize=None, **color_kwds):
    """ Plot a GeoSeries

        Generate a plot of a GeoSeries geometry with matplotlib.

        Parameters
        ----------

        Series
            The GeoSeries to be plotted.  Currently Polygon,
            MultiPolygon, LineString, MultiLineString and Point
            geometries can be plotted.

        colormap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.  Any
            colormap will work, but categorical colormaps are
            generally recommended.  Examples of useful discrete
            colormaps include:

                Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

        axes : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        linewidth : float (default 1.0)
            Line width for geometries.

        figsize : pair of floats (default None)
            Size of the resulting matplotlib.figure.Figure. If the argument
            axes is given explicitly, figsize is ignored.

        **color_kwds : dict
            Color options to be passed on to plot_polygon

        Returns
        -------

        matplotlib axes instance
    """
    import matplotlib.pyplot as plt
    if axes is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
    else:
        ax = axes
    color = gencolor(len(s), colormap=colormap)
    for geom in s:
        if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
            plot_multipolygon(ax, geom, facecolor=next(color), linewidth=linewidth, **color_kwds)
        elif geom.type == 'LineString' or geom.type == 'MultiLineString':
            plot_multilinestring(ax, geom, color=next(color), linewidth=linewidth)
        elif geom.type == 'Point':
            plot_point(ax, geom)
    plt.draw()
    return ax


def plot_dataframe(s, column=None, colormap=None, linewidth=1.0,
                   categorical=False, legend=False, axes=None,
                   scheme=None, k=5, vmin=None, vmax=None, figsize=None,
                   markersize=None,
                   **color_kwds
                   ):
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
            If False, colormap will reflect numerical values of the
            column being plotted.  For non-numerical columns (or if
            column=None), this will be set to True.

        colormap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.

        linewidth : float (default 1.0)
            Line width for geometries.

        legend : bool (default False)
            Plot a legend (Experimental; currently for categorical
            plots only)

        axes : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        scheme : pysal.esda.mapclassify.Map_Classifier
            Choropleth classification schemes

        vmin : float
            Minimum value for color map.

        vmax : float
            Maximum value for color map.

        k   : int (default 5)
            Number of classes (ignored if scheme is None)

        vmin : None or float (default None)

            Minimum value of colormap. If None, the minimum data value
            in the column to be plotted is used.

        vmax : None or float (default None)

            Maximum value of colormap. If None, the maximum data value
            in the column to be plotted is used.

        figsize
            Size of the resulting matplotlib.figure.Figure. If the argument
            axes is given explicitly, figsize is ignored.

        **color_kwds : dict
            Color options to be passed on to plot_polygon

        Returns
        -------

        matplotlib axes instance
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if column is None:
        return plot_series(s.geometry, colormap=colormap, axes=axes, linewidth=linewidth, figsize=figsize, **color_kwds)
    else:
        if s[column].dtype is np.dtype('O'):
            categorical = True
        if categorical:
            if colormap is None:
                colormap = 'Set1'
            categories = list(set(s[column].values))
            categories.sort()
            valuemap = dict([(k, v) for (v, k) in enumerate(categories)])
            values = [valuemap[k] for k in s[column]]
        else:
            values = s[column]
        if scheme is not None:
            values = __pysal_choro(values, scheme, k=k)
        cmap = norm_cmap(values, colormap, Normalize, cm, vmin=vmin, vmax=vmax)
        if axes is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect('equal')
        else:
            ax = axes
        for geom, value in zip(s.geometry, values):
            if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                plot_multipolygon(ax, geom, facecolor=cmap.to_rgba(value), linewidth=linewidth, **color_kwds)
            elif geom.type == 'LineString' or geom.type == 'MultiLineString':
                plot_multilinestring(ax, geom, color=cmap.to_rgba(value), linewidth=linewidth)
            # TODO: color point geometries
            elif geom.type == 'Point':
                plot_point(ax,
                           geom,
                           color=cmap.to_rgba(value),
                           markersize=markersize)
        if legend:
            if categorical:
                patches = []
                for value, cat in enumerate(categories):
                    patches.append(Line2D([0], [0], linestyle="none",
                                          marker="o", alpha=color_kwds.get('alpha', 0.5),
                                          markersize=10, markerfacecolor=cmap.to_rgba(value)))
                ax.legend(patches, categories, numpoints=1, loc='best')
            else:
                # TODO: show a colorbar
                raise NotImplementedError
    plt.draw()
    return ax


def __pysal_choro(values, scheme, k=5):
    """ Wrapper for choropleth schemes from PySAL for use with plot_dataframe

        Parameters
        ----------

        values
            Series to be plotted

        scheme
            pysal.esda.mapclassify classificatin scheme ['Equal_interval'|'Quantiles'|'Fisher_Jenks']

        k
            number of classes (2 <= k <=9)

        Returns
        -------

        values
            Series with values replaced with class identifier if PySAL is available, otherwise the original values are used
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
            print('Unrecognized scheme: ', s0)
            print('Using Quantiles instead')
        if k < 2 or k > 9:
            print('Invalid k: ', k)
            print('2<=k<=9, setting k=5 (default)')
            k = 5
        binning = schemes[scheme](values, k)
        values = binning.yb
    except ImportError:
        print('PySAL not installed, setting map to default')

    return values


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

    mn = vmin or min(values)
    mx = vmax or max(values)
    norm = normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap
