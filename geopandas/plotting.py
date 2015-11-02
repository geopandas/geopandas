from __future__ import print_function

import warnings

import numpy as np

from matplotlib.colorbar import make_axes
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


def plot_point(ax, pt, marker='o', markersize=2, color="black"):
    """ Plot a single Point geometry """
    ax.plot(pt.x, pt.y, marker=marker, markersize=markersize, linewidth=0,
            color=color)


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


def plot_series(s, cmap='Set1', ax=None, linewidth=1.0, figsize=None, **color_kwds):
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

        ax : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        linewidth : float (default 1.0)
            Line width for geometries.

        figsize : pair of floats (default None)
            Size of the resulting matplotlib.figure.Figure. If the argument
            ax is given explicitly, figsize is ignored.

        **color_kwds : dict
            Color options to be passed on to plot_polygon

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
    color = gencolor(len(s), colormap=cmap)
    for geom in s:
        if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
            plot_multipolygon(ax, geom, facecolor=next(color), linewidth=linewidth, **color_kwds)
        elif geom.type == 'LineString' or geom.type == 'MultiLineString':
            plot_multilinestring(ax, geom, color=next(color), linewidth=linewidth)
        elif geom.type == 'Point':
            plot_point(ax, geom, color=next(color))
    plt.draw()
    return ax


def plot_dataframe(s, column=None, cmap=None, linewidth=1.0,
                   categorical=False, legend=False, ax=None,
                   scheme=None, k=5, vmin=None, vmax=None, figsize=None,
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
            If False, cmap will reflect numerical values of the
            column being plotted.  For non-numerical columns (or if
            column=None), this will be set to True.

        cmap : str (default 'Set1')
            The name of a colormap recognized by matplotlib.

        linewidth : float (default 1.0)
            Line width for geometries.

        legend : bool (default False)
            Plot a legend or colorbar (Experimental)

        ax : matplotlib.pyplot.Artist (default None)
            axes on which to draw the plot

        scheme : pysal.esda.mapclassify.Map_Classifier
            Choropleth classification schemes (requires PySAL)

        vmin : float
            Minimum value for color map.

        vmax : float
            Maximum value for color map.

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
            Color options to be passed on to plot_polygon

        Returns
        -------

        ax : matplotlib axes instance
            Axes on which the dataframe was plotted
        cbar : matplotlib.colorbar.Colorbar (optional)
            Colorbar of the noncategorical plot. Returned only if `categorical`
            is False and `legend` is True.
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
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if column is None:
        return plot_series(s.geometry, cmap=cmap, ax=ax, linewidth=linewidth, figsize=figsize, **color_kwds)
    else:
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
        for geom, value in zip(s.geometry, values):
            if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                plot_multipolygon(ax, geom, facecolor=cmap.to_rgba(value), linewidth=linewidth, **color_kwds)
            elif geom.type == 'LineString' or geom.type == 'MultiLineString':
                plot_multilinestring(ax, geom, color=cmap.to_rgba(value), linewidth=linewidth)
            elif geom.type == 'Point':
                plot_point(ax, geom, color=cmap.to_rgba(value))
        cbar = None
        if legend:
            if categorical:
                patches = []
                for value, cat in enumerate(categories):
                    patches.append(Line2D([0], [0], linestyle="none",
                                          marker="o", alpha=color_kwds.get('alpha', 0.5),
                                          markersize=10, markerfacecolor=cmap.to_rgba(value)))
                ax.legend(patches, categories, numpoints=1, loc='best')
            else:
                cax = make_axes(ax)[0]
                cbar = ax.get_figure().colorbar(cmap, cax=cax)
    plt.draw()
    if cbar:
        return ax, cbar
    else:
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

    mn = vmin or min(values)
    mx = vmax or max(values)
    norm = normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    n_cmap.set_array(values)
    return n_cmap
