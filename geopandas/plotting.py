import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from descartes.patch import PolygonPatch

def plot_polygon(ax, poly, facecolor='red', edgecolor='black', alpha=0.5):
    a = np.asarray(poly.exterior)
    # without Descartes, we could make a Patch of exterior
    ax.add_patch(PolygonPatch(poly, facecolor=facecolor, alpha=alpha))
    ax.plot(a[:, 0], a[:, 1], color=edgecolor)
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor)


def plot_multipolygon(ax, geom, facecolor='red'):
    """ Can safely call with either Polygon or Multipolygon geometry
    """
    if geom.type == 'Polygon':
        plot_polygon(ax, geom, facecolor)
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            plot_polygon(ax, poly, facecolor=facecolor)


def plot_linestring(ax, geom, color='black', linewidth=1):
    a = np.array(geom)
    plt.plot(a[:,0], a[:,1], color=color, linewidth=linewidth)


def plot_point(ex, pt, marker='o', markersize=2):
    """ Plot a single Point geometry
    """
    plt.plot(pt.x, pt.y, marker=marker, markersize=markersize, linewidth=0)


def gencolor(N, colormap='Set1'):
    """
    Color generator intended to work with one of the ColorBrewer
    qualitative color scales.

    Suggested values of colormap are the following:

        Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3

    (although any matplotlib colormap will work).
    """
    # don't use more than 9 discrete colors
    n_colors = min(N, 9)
    cmap = cm.get_cmap(colormap, n_colors)
    colors = cmap(range(n_colors))
    for i in xrange(N):
        yield colors[i % n_colors]

def plot_series(s, colormap='Set1'):
    fig = plt.gcf()
    fig.add_subplot(111, aspect='equal')
    ax = plt.gca()
    color = gencolor(len(s), colormap=colormap)
    for geom in s:
        if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
            plot_multipolygon(ax, geom, facecolor=color.next())
        elif geom.type == 'LineString':
            plot_linestring(ax, geom)
        elif geom.type == 'Point':
            plot_point(ax, geom)
    return ax


def plot_dataframe(s, column=None, colormap='Set1'):
    if column is None:
        return s['geometry'].plot()
    else:
        raise NotImplementedError
