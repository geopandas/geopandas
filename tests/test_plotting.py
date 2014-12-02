from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.pyplot import Artist, savefig, clf
from matplotlib.colorbar import Colorbar
from matplotlib.backends import backend_agg
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from shapely.geometry import Polygon, LineString, Point
from six.moves import xrange

from geopandas import GeoSeries, read_file
from .util import download_nybb


# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')

class PlotTests(unittest.TestCase):
    
    def setUp(self):
        # hardcode settings for comparison tests
        # settings adapted from ggplot test suite
        matplotlib.rcdefaults() # Start with all defaults
        matplotlib.rcParams['text.hinting'] = True
        matplotlib.rcParams['text.antialiased'] = True
        matplotlib.rcParams['font.sans-serif'] = 'Bitstream Vera Sans'
        backend_agg.RendererAgg._fontd.clear()

        nybb_filename = download_nybb()

        self.df = read_file('/nybb_14a_av/nybb.shp', 
                            vfs='zip://' + nybb_filename)
        self.df['values'] = [0.1, 0.2, 0.1, 0.3, 0.4]
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

    def test_dataframe_plot(self):
        """ Test plotting of a dataframe """
        clf()
        filename = 'df_plot.png'
        ax = self.df.plot()
        self._compare_images(ax=ax, filename=filename)

    def test_dataframe_categorical_plot(self):
        """ Test plotting of a categorical GeoDataFrame with legend """
        clf()
        filename = 'df_cat_leg_plot.png'
        ax = self.df.plot(column='values', categorical=True, legend=True)
        self._compare_images(ax=ax, filename=filename)

    def test_dataframe_noncategorical_plot(self):
        """ Test plotting of a noncategorical GeoDataFrame"""
        clf()
        filename = 'df_noncat_plot.png'
        ax = self.df.plot(column='values', categorical=False)
        self._compare_images(ax=ax, filename=filename)

    def test_dataframe_noncategorical_leg_plot(self):
        """ Test plotting of a noncategorical GeoDataFrame"""
        clf()
        filename = 'df_noncat_leg_plot.png'
        ax, cbar = self.df.plot(column='values', categorical=False, legend=True)
        self._compare_images(ax=ax, filename=filename)
        self.assertTrue(isinstance(cbar, Colorbar))

if __name__ == '__main__':
    unittest.main()
