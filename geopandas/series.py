import numpy as np
from pandas import Series, DataFrame


class GeoSeries(Series):
    """
    A Series object designed to store shapely geometry objects.
    """

    def __new__(cls, *args, **kwargs):
        # http://stackoverflow.com/a/11982602/1220158
        arr = Series.__new__(cls, *args, **kwargs)
        return arr.view(GeoSeries)

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
