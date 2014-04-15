from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

from matplotlib.pyplot import Artist, savefig, clf
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from shapely.geometry import Polygon, LineString, Point

from geopandas import GeoSeries

# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')


class PlotTests(unittest.TestCase):
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        return

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        return

    def _compare_images(self, ax, filename, tol=8):
        """ Helper method to do the comparisons """
        assert isinstance(ax, Artist)
        if GENERATE_BASELINE:
            savefig(os.path.join(BASELINE_DIR, filename))
        savefig(os.path.join(self.tempdir, filename))
        err = compare_images(os.path.join(BASELINE_DIR, filename),
                             os.path.join(self.tempdir, filename),
                             tol, in_decorator=True)
        if err:
            raise ImageComparisonFailure('images not close: %(actual)s '
                                         'vs. %(expected)s '
                                         '(RMS %(rms).3f)' % err)

    def test_poly_plot(self):
        """ Test plotting a simple series of polygons """
        clf()
        filename = 'poly_plot.png'
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        polys = GeoSeries([t1, t2])
        ax = polys.plot()
        self._compare_images(ax=ax, filename=filename)

    def test_point_plot(self):
        """ Test plotting a simple series of points """
        clf()
        filename = 'points_plot.png'
        N = 10
        points = GeoSeries(Point(i, i) for i in xrange(N))
        ax = points.plot()
        self._compare_images(ax=ax, filename=filename)

    def test_line_plot(self):
        """ Test plotting a simple series of lines """
        clf()
        filename = 'lines_plot.png'
        N = 10
        lines = GeoSeries([LineString([(0, i), (9, i)]) for i in xrange(N)])
        ax = lines.plot()
        self._compare_images(ax=ax, filename=filename)

if __name__ == '__main__':
    unittest.main()
