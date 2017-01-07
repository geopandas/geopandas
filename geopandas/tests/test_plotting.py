from __future__ import absolute_import, division

import itertools
import numpy as np
import warnings

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.pyplot import get_cmap
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon, LineString, Point
from six.moves import xrange
from .util import unittest

from geopandas import GeoSeries, GeoDataFrame, read_file


class TestPointPlotting(unittest.TestCase):

    def setUp(self):

        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N))
        values = np.arange(self.N)
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})

    def test_figsize(self):

        ax = self.points.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))

        ax = self.df.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))

    def test_default_colors(self):

        ## without specifying values -> max 9 different colors

        # GeoSeries
        ax = self.points.plot()
        cmap = get_cmap('Set1', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(self.N, ax.collections[0], expected_colors)

        # GeoDataFrame -> uses 'jet' instead of 'Set1'
        ax = self.df.plot()
        cmap = get_cmap('jet', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(self.N, ax.collections[0], expected_colors)

        ## with specifying values -> different colors for all 10 values
        ax = self.df.plot(column='values')
        cmap = get_cmap('jet')
        expected_colors = cmap(np.arange(self.N)/(self.N-1))
        _check_colors(self.N, ax.collections[0], expected_colors)

    def test_colormap(self):

        ## without specifying values -> max 9 different colors

        # GeoSeries
        ax = self.points.plot(cmap='RdYlGn')
        cmap = get_cmap('RdYlGn', 9)
        expected_colors = cmap(list(range(9))*2)
        _check_colors(self.N, ax.collections[0], expected_colors)

        # GeoDataFrame -> same as GeoSeries in this case
        ax = self.df.plot(cmap='RdYlGn')
        _check_colors(self.N, ax.collections[0], expected_colors)

        ## with specifying values -> different colors for all 10 values
        ax = self.df.plot(column='values', cmap='RdYlGn')
        cmap = get_cmap('RdYlGn')
        expected_colors = cmap(np.arange(self.N)/(self.N-1))
        _check_colors(self.N, ax.collections[0], expected_colors)

    def test_single_color(self):

        ax = self.points.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

        ax = self.df.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'column'
            ax = self.df.plot(column='values', color='green')
            _check_colors(self.N, ax.collections[0], ['green']*self.N)

    def test_style_kwargs(self):

        # markersize
        ax = self.points.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]

        ax = self.df.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]

        ax = self.df.plot(column='values', markersize=10)
        assert ax.collections[0].get_sizes() == [10]

    def test_legend(self):
        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # legend ignored if color is given.
            ax = self.df.plot(column='values', color='green', legend=True)
            assert len(ax.get_figure().axes) == 1  # no separate legend axis

        # legend ignored if no column is given.
        ax = self.df.plot(legend=True)
        assert len(ax.get_figure().axes) == 1  # no separate legend axis

        # Continuous legend
        ## the colorbar matches the Point colors
        ax = self.df.plot(column='values', cmap='RdYlGn', legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = ax.get_figure().axes[1].collections[0].get_facecolors()
        ### first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        ### last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])

        # Categorical legend
        ## the colorbar matches the Point colors
        ax = self.df.plot(column='values', categorical=True, legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = ax.get_legend().axes.collections[0].get_facecolors()
        ### first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        ### last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])


class TestPointZPlotting(unittest.TestCase):

    def setUp(self):
        self.N = 10
        self.points = GeoSeries(Point(i, i, i) for i in range(self.N))
        values = np.arange(self.N)
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})


    def test_plot(self):
        # basic test that points with z coords don't break plotting

        ax = self.df.plot()


class TestLineStringPlotting(unittest.TestCase):

    def setUp(self):

        self.N = 10
        values = np.arange(self.N)
        self.lines = GeoSeries([LineString([(0, i), (4, i+0.5), (9, i)])
                                for i in xrange(self.N)],
                               index=list('ABCDEFGHIJ'))
        self.df = GeoDataFrame({'geometry': self.lines, 'values': values})

    def test_single_color(self):

        ax = self.lines.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

        ax = self.df.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'column'
            ax = self.df.plot(column='values', color='green')
            _check_colors(self.N, ax.collections[0], ['green']*self.N)

    def test_style_kwargs(self):

        def linestyle_tuple_to_string(tup):
            """ Converts a linestyle of the form `(offset, onoffseq)`, as
                documented in `Collections.set_linestyle`, to a string
                representation, namely one of:
                    { 'dashed',  'dotted', 'dashdot', 'solid' }.
            """
            from matplotlib.backend_bases import GraphicsContextBase
            reverse_idx = dict((v, k)
                               for k, v in GraphicsContextBase.dashd.items())
            return reverse_idx[tup]

        # linestyle
        ax = self.lines.plot(linestyle='dashed')
        ls = [linestyle_tuple_to_string(l)
              for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']

        ax = self.df.plot(linestyle='dashed')
        ls = [linestyle_tuple_to_string(l)
              for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']

        ax = self.df.plot(column='values', linestyle='dashed')
        ls = [linestyle_tuple_to_string(l)
              for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']


class TestPolygonPlotting(unittest.TestCase):

    def setUp(self):

        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        self.polys = GeoSeries([t1, t2], index=list('AB'))
        self.df = GeoDataFrame({'geometry': self.polys, 'values': [0, 1]})

        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame({'geometry': [multipoly1, multipoly2],
                                 'values': [0, 1]})
        return

    def test_single_color(self):

        ax = self.polys.plot(color='green')
        _check_colors(2, ax.collections[0], ['green']*2, alpha=0.5)

        ax = self.df.plot(color='green')
        _check_colors(2, ax.collections[0], ['green']*2, alpha=0.5)

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'values'
            ax = self.df.plot(column='values', color='green')
            _check_colors(2, ax.collections[0], ['green']*2, alpha=0.5)

    def test_optimization(self):

        # when linewidth=0 or no alpha, we don't have to plot polys twice

        ax = self.polys.plot(linewidth=0)
        assert len(ax.collections) == 1  # only plotted once

        ax = self.polys.plot(alpha=1)
        assert len(ax.collections) == 1  # only plotted once

        ax = self.df.plot(linewidth=0)
        assert len(ax.collections) == 1  # only plotted once

        ax = self.df.plot(alpha=1)
        assert len(ax.collections) == 1  # only plotted once

    def test_vmin_vmax(self):

        # when vmin == vmax, all polygons should be the same color

        # non-categorical
        ax = self.df.plot(column='values', categorical=False, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])

        # categorical
        ax = self.df.plot(column='values', categorical=True, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])

    def test_style_kwargs(self):

        # facecolor overrides default cmap when color is not set
        ax = self.polys.plot(facecolor='k')
        _check_colors(2, ax.collections[0], ['k']*2, alpha=0.5)

        # facecolor overrides more general-purpose color when both are set
        ax = self.polys.plot(color='red', facecolor='k')
        _check_colors(2, ax.collections[0], ['k']*2, alpha=0.5)

        # edgecolor
        ax = self.polys.plot(edgecolor='red')
        np.testing.assert_array_equal([(1, 0, 0, 0.5)],
                                      ax.collections[0].get_edgecolors())

        ax = self.df.plot('values', edgecolor='red')
        np.testing.assert_array_equal([(1, 0, 0, 0.5)],
                                      ax.collections[0].get_edgecolors())

    def test_multipolygons(self):

        # MultiPolygons
        ax = self.df2.plot()
        assert len(ax.collections[0].get_paths()) == 4
        cmap = get_cmap('jet', 2)
        ## colors are repeated for all components within a MultiPolygon
        expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
        _check_colors(4, ax.collections[0], expected_colors, alpha=0.5)

        ax = self.df2.plot('values')
        ## specifying values -> same as without values in this case.
        _check_colors(4, ax.collections[0], expected_colors, alpha=0.5)


class TestPolygonZPlotting(unittest.TestCase):

    def setUp(self):

        t1 = Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 1)])
        t2 = Polygon([(1, 0, 0), (2, 0, 0), (2, 1, 1)])
        self.polys = GeoSeries([t1, t2], index=list('AB'))
        self.df = GeoDataFrame({'geometry': self.polys, 'values': [0, 1]})

        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame({'geometry': [multipoly1, multipoly2],
                                 'values': [0, 1]})


    def test_plot(self):
        # basic test that points with z coords don't break plotting

        ax = self.df.plot()


class TestNonuniformGeometryPlotting(unittest.TestCase):

    def setUp(self):

        poly = Polygon([(1, 0), (2, 0), (2, 1)])
        line = LineString([(0.5, 0.5), (1, 1), (1, 0.5), (1.5, 1)])
        point = Point(0.75, 0.25)
        self.series = GeoSeries([poly, line, point])
        self.df = GeoDataFrame({'geometry': self.series, 'values': [1, 2, 3]})
        return

    def test_colormap(self):

        ax = self.series.plot(cmap='RdYlGn')
        cmap = get_cmap('RdYlGn', 3)
        # polygon gets extra alpha. See #266
        _check_colors(1, ax.collections[0], [cmap(0)], alpha=0.5)
        # N.B. ax.collections[1] contains the edges of the polygon
        _check_colors(1, ax.collections[2], [cmap(1)], alpha=1)   # line
        _check_colors(1, ax.collections[3], [cmap(2)], alpha=1)   # point

    def test_style_kwargs(self):

        # markersize -> only the Point gets it
        ax = self.series.plot(markersize=10)
        assert ax.collections[3].get_sizes() == [10]

        ax = self.df.plot(markersize=10)
        assert ax.collections[3].get_sizes() == [10]


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

    def test_invalid_scheme(self):
        with self.assertRaises(ValueError):
            scheme = 'invalid_scheme_*#&)(*#'
            ax = self.tracts.plot(column='CRIME', scheme=scheme, k=3,
                                  cmap='OrRd', legend=True)


def _check_colors(N, collection, expected_colors, alpha=None):
    """ Asserts that the members of `collection` match the `expected_colors` (in order)

    Parameters
    ----------
    N : int
        The number of geometries believed to be in collection.
        matplotlib.collection is implemented such that the number of geoms in
        `collection` doesn't have to match the number of colors assignments in
        the collection: the colors will cycle to meet the needs of the geoms.
        `N` helps us resolve this.
    collection : matplotlib.collections.Collection
        The colors of this collection's patches are read from `collection.get_facecolors()`
    expected_colors : sequence of RGBA tuples
    alpha : float (optional)
        If set, this alpha transparency will be applied to the `expected_colors`.
        (Any transparency on the `collecton` is assumed to be set in its own
        facecolor RGBA tuples.)
    """
    import matplotlib.colors as colors
    conv = colors.colorConverter

    # Convert 2D numpy array to a list of RGBA tuples.
    actual_colors = list(collection.get_facecolors())
    actual_colors = map(tuple, actual_colors)
    all_actual_colors = list(itertools.islice(
        itertools.cycle(actual_colors), N))

    for actual, expected in zip(all_actual_colors, expected_colors):
        assert actual == conv.to_rgba(expected, alpha=alpha), \
            '{} != {}'.format(actual, conv.to_rgba(expected, alpha=alpha))


if __name__ == '__main__':
    unittest.main()
