from __future__ import absolute_import

import numpy as np
import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.pyplot import Artist, savefig, clf, cm
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from numpy import cos, sin, pi
from shapely.geometry import Polygon, LineString, Point
from six.moves import xrange

from geopandas import GeoSeries, GeoDataFrame


# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')

TRAVIS = bool(os.environ.get('TRAVIS', False))


class PlotTests(unittest.TestCase):
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        return

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        return

    def _compare_images(self, ax, filename, tol=10):
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

    @unittest.skipIf(TRAVIS, 'Skip on Travis (fails even though it passes locally)')
    def test_plot_GeoDataFrame_with_kwargs(self):
        """
        Test plotting a simple GeoDataFrame consisting of a series of polygons
        with increasing values using various extra kwargs.
        """
        clf()
        filename = 'poly_plot_with_kwargs.png'
        ts = np.linspace(0, 2*pi, 10, endpoint=False)

        # Build GeoDataFrame from a series of triangles wrapping around in a ring
        # and a second column containing a list of increasing values.
        r1 = 1.0  # radius of inner ring boundary
        r2 = 1.5  # radius of outer ring boundary

        def make_triangle(t0, t1):
            return Polygon([(r1*cos(t0), r1*sin(t0)),
                            (r2*cos(t0), r2*sin(t0)),
                            (r1*cos(t1), r1*sin(t1))])

        polys = GeoSeries([make_triangle(t0, t1) for t0, t1 in zip(ts, ts[1:])])
        values = np.arange(len(polys))
        df = GeoDataFrame({'geometry': polys, 'values': values})

        # Plot the GeoDataFrame using various keyword arguments to see if they are honoured
        ax = df.plot(column='values', colormap=cm.RdBu, vmin=+2, vmax=None, figsize=(8, 4))
        self._compare_images(ax=ax, filename=filename)

if __name__ == '__main__':
    unittest.main()
