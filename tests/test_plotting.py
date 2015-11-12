from __future__ import absolute_import, division

import numpy as np
import os
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.pyplot import Artist, savefig, close, cm, get_cmap
from matplotlib.colorbar import Colorbar
from matplotlib.backends import backend_agg
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from numpy import cos, sin, pi
from shapely.geometry import Polygon, LineString, Point
from six.moves import xrange
from .util import download_nybb, unittest

from geopandas import GeoSeries, GeoDataFrame, read_file


# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')

TRAVIS = bool(os.environ.get('TRAVIS', False))


class TestImageComparisons(unittest.TestCase):

    def setUp(self):
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
        close('all')
        filename = 'poly_plot.png'
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        polys = GeoSeries([t1, t2])
        ax = polys.plot()
        self._compare_images(ax=ax, filename=filename)

    def test_point_plot(self):
        """ Test plotting a simple series of points """
        close('all')
        filename = 'points_plot.png'
        N = 10
        points = GeoSeries(Point(i, i) for i in xrange(N))
        ax = points.plot()
        self._compare_images(ax=ax, filename=filename)

    def test_line_plot(self):
        """ Test plotting a simple series of lines """
        close('all')
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
        close('all')
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
        ax = df.plot(column='values', cmap=cm.RdBu, vmin=+2, vmax=None, figsize=(8, 4))
        self._compare_images(ax=ax, filename=filename)



class TestPointPlotting(unittest.TestCase):

    def setUp(self):

        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})

    def test_default_colors(self):

        ## without specifying values -> max 9 different colors

        # GeoSeries
        ax = self.points.plot()
        cmap = get_cmap('Set1', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(ax.get_lines(), expected_colors)

        # GeoDataFrame -> uses 'jet' instead of 'Set1'
        ax = self.df.plot()
        cmap = get_cmap('jet', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(ax.get_lines(), expected_colors)

        ## with specifying values

        ax = self.df.plot(column='values')
        cmap = get_cmap('jet')
        expected_colors = cmap(np.arange(self.N)/(self.N-1))

        _check_colors(ax.get_lines(), expected_colors)

    def test_colormap(self):

        ## without specifying values -> max 9 different colors

        # GeoSeries
        ax = self.points.plot(cmap='RdYlGn')
        cmap = get_cmap('RdYlGn', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(ax.get_lines(), expected_colors)

        # GeoDataFrame -> same as GeoSeries in this case
        ax = self.df.plot(cmap='RdYlGn')
        _check_colors(ax.get_lines(), expected_colors)

        ## with specifying values

        ax = self.df.plot(column='values', cmap='RdYlGn')
        cmap = get_cmap('RdYlGn')
        expected_colors = cmap(np.arange(self.N)/(self.N-1))
        _check_colors(ax.get_lines(), expected_colors)

    def test_single_color(self):

        ax = self.points.plot(color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)

        ax = self.df.plot(color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)

        ax = self.df.plot(column='values', color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)


class TestLineStringPlotting(unittest.TestCase):

    def setUp(self):

        self.N = 10
        values = np.arange(self.N)
        self.lines = GeoSeries([LineString([(0, i), (9, i)]) for i in xrange(self.N)])
        self.df = GeoDataFrame({'geometry': self.lines, 'values': values})

    def test_single_color(self):

        ax = self.lines.plot(color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)

        ax = self.df.plot(color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)

        ax = self.df.plot(column='values', color='green')
        _check_colors(ax.get_lines(), ['green']*self.N)


class TestPolygonPlotting(unittest.TestCase):

    def setUp(self):

        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        self.polys = GeoSeries([t1, t2])
        self.df = GeoDataFrame({'geometry': self.polys, 'values': [0, 1]})
        return

    def tearDown(self):

        close('all')
        return

    def test_single_color(self):

        ax = self.polys.plot(color='green')
        _check_colors(ax.patches, ['green']*2, alpha=0.5)

        ax = self.df.plot(color='green')
        _check_colors(ax.patches, ['green']*2, alpha=0.5)

        ax = self.df.plot(column='values', color='green')
        _check_colors(ax.patches, ['green']*2, alpha=0.5)

    def test_categorical_colors(self):

        ax = self.df.plot(column='values', categorical=True, legend=True)
        cmap = get_cmap('Set1', 2)
        expected_colors = cmap(self.df['values'])
        _check_colors(ax.patches, expected_colors, alpha=0.5)

    def test_noncategorical_colors(self):

        # With custom colormap
        ax = self.df.plot(column='values', cmap='copper')
        cmap = get_cmap('copper', 2)
        expected_colors = cmap(self.df['values'])
        _check_colors(ax.patches, expected_colors, alpha=0.5)

        # With legend
        ax, cbar = self.df.plot(column='values', legend=True)
        cmap = get_cmap('jet', 2)
        expected_colors = cmap(self.df['values'])
        # polygons match the colormap (plus some alpha)
        _check_colors(ax.patches, expected_colors, alpha=0.5)
        # polygons match the colorbar itself
        cbar.cmap._lut[:,-1] = cbar.alpha # move alpha from the cbar to the cmap for this test
        _check_colors(ax.patches, map(cbar.cmap, self.df['values']))

    def test_vmin_vmax(self):

        # when vmin == vmax, all polygons should be the same color
        ax = self.df.plot(column='values', categorical=True, vmin=0, vmax=0)
        cmap = get_cmap('Set1', 2)
        self.assertEqual(ax.patches[0].get_facecolor(), ax.patches[1].get_facecolor())

        # vmin is the max value, so all polygons get plotted the same color.
        val = 1.0
        ax, cbar = self.df.plot(column='values', vmin=val, vmax=2.0, legend=True)
        cmap = get_cmap('jet', 2)
        expected_colors = [cmap(0)] * 2 # the bottom of the colormap
        _check_colors(ax.patches, expected_colors, alpha=0.5)
        # polygons match the colorbar itself
        cbar.cmap._lut[:,-1] = cbar.alpha # move alpha from the cbar to the cmap for this test
        _check_colors(ax.patches, [cbar.to_rgba(val)] * 2)

        # vmin==vmax is the only value, so all polygons get plotted the same color.
        val = 1.0
        ax, cbar = self.df.plot(column='values', vmin=val, vmax=val, legend=True)
        # polygons match the colorbar itself
        # FIXME: This is a bug in geopandas.
        #_check_colors(ax.patches, [cbar.to_rgba(val)] * 2)


class TestPySALPlotting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import pysal as ps
        except ImportError:
            raise unittest.SkipTest("PySAL is not installed")

        pth = ps.examples.get_path("columbus.shp")
        cls.tracts = read_file(pth)

    def test_legend(self):
        ax = self.tracts.plot(column='CRIME', scheme='QUANTILES', k=3,
                         cmap='OrRd', legend=True)

        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [u'0.00 - 26.07', u'26.07 - 41.97', u'41.97 - 68.89']
        self.assertEqual(labels, expected)


def _check_colors(collection, expected_colors, alpha=None):

    from matplotlib.lines import Line2D
    import matplotlib.colors as colors
    conv = colors.colorConverter

    for patch, color in zip(collection, expected_colors):
        if isinstance(patch, Line2D):
            # points/lines
            result = patch.get_color()
        else:
            # polygons
            result = patch.get_facecolor()
        assert conv.to_rgba(result) == conv.to_rgba(color, alpha=alpha)


if __name__ == '__main__':
    unittest.main()
