import numpy as np

def plot_polygon(ax, poly, facecolor='red', edgecolor='black', alpha=0.5):
    from descartes.patch import PolygonPatch
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
    ax.plot(a[:,0], a[:,1], color=color, linewidth=linewidth)


def plot_point(ax, pt, marker='o', markersize=2):
    """ Plot a single Point geometry
    """
    ax.plot(pt.x, pt.y, marker=marker, markersize=markersize, linewidth=0)


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

def plot_series(s, colormap='Set1', axes=None):
    import matplotlib.pyplot as plt
    if axes == None:
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        ax = plt.gca()
    else:
        ax = plt.gcf()
    color = gencolor(len(s), colormap=colormap)
    for geom in s:
        if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
            plot_multipolygon(ax, geom, facecolor=color.next())
        elif geom.type == 'LineString':
            plot_linestring(ax, geom)
        elif geom.type == 'Point':
            plot_point(ax, geom)
    return ax


def plot_dataframe(s, column=None, colormap=None, alpha=0.5,
                   categorical=False, legend=False, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib import cm
    if column is None:
        return s['geometry'].plot()
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
        mn, mx = min(values), max(values)
        norm = Normalize(vmin=mn, vmax=mx)
        cmap = cm.ScalarMappable(norm=norm, cmap=colormap)
        if axes == None:
            fig = plt.figure()
            fig.add_subplot(111, aspect='equal')
            ax = plt.gca()
        else:
            ax = axes
        for geom, value in zip(s['geometry'], values):
            if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                plot_multipolygon(ax, geom, facecolor=cmap.to_rgba(value, alpha=0.5))
            # TODO: color non-polygon geometries
            elif geom.type == 'LineString':
                plot_linestring(ax, geom)
            elif geom.type == 'Point':
                plot_point(ax, geom)
        if legend:
            if categorical:
                patches = []
                for value, cat in enumerate(categories):
                    patches.append(Line2D([0], [0], linestyle="none",
                                          marker="o", alpha=alpha,
                                          markersize=10, markerfacecolor=cmap.to_rgba(value)))
                ax.legend(patches, categories, numpoints=1, loc='best')
            else:
                # TODO: show a colorbar
                raise NotImplementedError
