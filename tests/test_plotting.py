from contextlib import contextmanager
import os
import tempfile
import unittest

from matplotlib.pyplot import Artist, savefig
from shapely.geometry import Polygon, LineString, Point

from geopandas import GeoSeries


@contextmanager
def get_tempfile():
    f, path = tempfile.mkstemp()
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except:
            pass

class TestSeriesPlot(unittest.TestCase):

    def setUp(self):
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(1, 0), (2, 1), (2, 1)])
        self.polys = GeoSeries([self.t1, self.t2])

    def test_poly_plot(self):
        """ Test plotting a simple series of polygons """
        ax = self.polys.plot()
        self.assertIsInstance(ax, Artist)
        with get_tempfile() as file:
            savefig(file)

if __name__ == '__main__':
    unittest.main()
