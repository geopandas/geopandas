import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import cm

from shapely.geometry import shape, Polygon, Point
import fiona
from descartes.patch import PolygonPatch


def _plot_polygon(ax, poly, facecolor='red', edgecolor='black', alpha=0.5):
    a = np.asarray(poly.exterior)
    # without Descartes, we could make a Patch of exterior
    ax.add_patch(PolygonPatch(poly, facecolor=facecolor, alpha=alpha))
    ax.plot(a[:, 0], a[:, 1], color=edgecolor)
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor)


def _plot_multipolygon(ax, geom, facecolor='red'):
    """ Can safely call with either Polygon or Multipolygon geometry
    """
    if geom.type == 'Polygon':
        _plot_polygon(ax, geom, facecolor)
    elif geom.type == 'MultiPolygon':
        for poly in geom.geoms:
            _plot_polygon(ax, poly, facecolor)


def _gencolor(N, colormap='Accent'):
    """
    Color generator
    """
    cmap = cm.get_cmap(colormap, N)
    colors = cmap(range(N))
    for color in colors:
        yield color

class GeoSeries(Series):
    """
    A Series object designed to store shapely geometry objects.
    """

    def __new__(cls, *args, **kwargs):
        # http://stackoverflow.com/a/11982602/1220158
        arr = Series.__new__(cls, *args, **kwargs)
        return arr.view(GeoSeries)

    @classmethod
    def from_file(cls, filename):
        """
        Alternate constructor to create a GeoSeries from a file
        """
        geoms = []
        with fiona.open(filename) as f:
            for rec in f:
                geoms.append(shape(rec['geometry']))
        return GeoSeries(geoms)

    @property
    def area(self):
        """
        Return the area of each member of the GeoSeries
        """
        return Series([geom.area for geom in self], index=self.index)

    @property
    def boundary(self):
        return GeoSeries([geom.boundary for geom in self], index=self.index)

    @property
    def bounds(self):
        """
        Return a DataFrame of minx, miny, maxx, maxy values of geometry objects
        """
        bounds = np.array([geom.bounds for geom in self])
        return DataFrame(bounds,
                         columns=['minx', 'miny', 'maxx', 'maxy'],
                         index=self.index)

    @property
    def geom_type(self):
        return Series([geom.geom_type for geom in self], index=self.index)

    def contains(self, other):
        """
        Return a Series of boolean values.
        Operates on either a GeoSeries or a Shapely geometry
        """
        if isinstance(other, GeoSeries):
            # TODO: align series
            return Series([s[0].contains(s[1]) for s in zip(self, other)],
                          index=self.index)
        else:
            return Series([s.contains(other) for s in self],
                          index=self.index)

    def plot(self, *args, **kwargs):
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        ax = plt.gca()
        color = _gencolor(len(self))
        for geom in self:
            if geom.type == 'Polygon' or geom.type == 'MultiPolygon':
                _plot_multipolygon(ax, geom, facecolor=color.next(),
                                   *args, **kwargs)
        plt.show()

if __name__ == '__main__':
    p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    g = GeoSeries([p1, p2, p3])
    print g.area
    g.plot()
