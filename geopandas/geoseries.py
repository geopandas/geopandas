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


def _plot_point(ex, geom):
    """ TODO
    """
    pass


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
    def centroid(self):
        """
        Return the centroid of each geometry in the GeoSeries
        """
        return GeoSeries([geom.centroid for geom in self], index=self.index)

    @property
    def convex_hull(self):
        return GeoSeries([geom.convex_hull for geom in self], index=self.index)

    @property
    def geom_type(self):
        return Series([geom.geom_type for geom in self], index=self.index)

    @property
    def type(self):
        return self.geom_type

    @property
    def length(self):
        return Series([geom.length for geom in self], index=self.index)

    @property
    def is_valid(self):
        return Series([geom.is_valid for geom in self], index=self.index)

    @property
    def is_empty(self):
        return Series([geom.is_empty for geom in self], index=self.index)

    @property
    def is_ring(self):
        return Series([geom.exterior.is_ring for geom in self], index=self.index)

    @property
    def is_simple(self):
        return Series([geom.is_simple for geom in self], index=self.index)

    def simplify(self, *args, **kwargs):
        return Series([geom.simplify(*args, **kwargs) for geom in self],
                      index=self.index)

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

    # TODO: refactor to eliminate replications
    def difference(self, other):
        """
        Return a GeoSeries of differences
        Operates on either a GeoSeries or a Shapely geometry
        """
        if isinstance(other, GeoSeries):
            # TODO: align series
            return GeoSeries([s[0].difference(s[1]) for s in zip(self, other)],
                          index=self.index)
        else:
            return GeoSeries([s.difference(other) for s in self],
                          index=self.index)

    def union(self, other):
        """
        Return a GeoSeries of differences
        Operates on either a GeoSeries or a Shapely geometry
        """
        if isinstance(other, GeoSeries):
            # TODO: align series
            return GeoSeries([s[0].union(s[1]) for s in zip(self, other)],
                          index=self.index)
        else:
            return GeoSeries([s.union(other) for s in self],
                          index=self.index)

    def buffer(self, distance, resolution=16):
        return GeoSeries([geom.buffer(distance, resolution) for geom in self],
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
            elif geom.type == 'Point':
                _plot_point(ax, geom)
        return ax

if __name__ == '__main__':
    """ Generate simple examples
    """
    dpi = 300
    p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    g = GeoSeries([p1, p2, p3])
    ax = g.plot()
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([-0.5, 1.5])
    plt.savefig('test.png', dpi=dpi, bbox_inches='tight')
    g.buffer(0.5).plot()
    plt.savefig('test_buffer.png', dpi=dpi, bbox_inches='tight')
    plt.show()
