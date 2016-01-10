from __future__ import absolute_import, division

import itertools
import numpy as np
import os
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg', warn=False)
from matplotlib.pyplot import Artist, savefig, clf, cm, get_cmap
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from numpy import cos, sin, pi
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
        # legend ignored if color is given.
        ax = self.df.plot(column='values', color='green', legend=True)
        assert len(ax.get_figure().axes) == 1 # only the plot, no axis w/ legend

        # legend ignored if no column is given.
        ax = self.df.plot(legend=True)
        assert len(ax.get_figure().axes) == 1 # only the plot, no axis w/ legend

        # Continuous legend
        ## the colorbar matches the Point colors
        ax = self.df.plot(column='values', cmap='RdYlGn', legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = ax.get_figure().axes[1].collections[0].get_facecolors()
        ### first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        ### last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])


class TestLineStringPlotting(unittest.TestCase):

    def setUp(self):

        self.N = 10
        values = np.arange(self.N)
        self.lines = GeoSeries([LineString([(0, i), (9, i)]) for i in xrange(self.N)])
        self.df = GeoDataFrame({'geometry': self.lines, 'values': values})

    def test_single_color(self):

        ax = self.lines.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

        ax = self.df.plot(color='green')
        _check_colors(self.N, ax.collections[0], ['green']*self.N)

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
            reverse_idx = dict((v, k) for k, v in GraphicsContextBase.dashd.items())
            return reverse_idx[tup]

        # linestyle
        ax = self.lines.plot(linestyle='dashed')
        ls = [linestyle_tuple_to_string(l) for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']

        ax = self.df.plot(linestyle='dashed')
        ls = [linestyle_tuple_to_string(l) for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']

        ax = self.df.plot(column='values', linestyle='dashed')
        ls = [linestyle_tuple_to_string(l) for l in ax.collections[0].get_linestyles()]
        assert ls == ['dashed']


class TestPolygonPlotting(unittest.TestCase):

    def setUp(self):

        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        self.polys = GeoSeries([t1, t2])
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

        ax = self.df.plot(column='values', color='green')
        _check_colors(2, ax.collections[0], ['green']*2, alpha=0.5)

    def test_vmin_vmax(self):

        # when vmin == vmax, all polygons should be the same color
        ax = self.df.plot(column='values', categorical=True, vmin=0, vmax=0)
        cmap = get_cmap('Set1', 2)
        _check_colors(2, ax.collections[0], cmap([0, 0]), alpha=0.5)

    def test_style_kwargs(self):

        ax = self.polys.plot(facecolor='k')
        _check_colors(2, ax.collections[0], ['k']*2, alpha=0.5)

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


def _check_colors(N, collection, expected_colors, alpha=None):
    """ Asserts that the members of `collection` match the `expected_colors` (in order)

	Parameters
	----------
    N : the number of geometries believed to be in collection.
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
    from matplotlib.lines import Line2D
    import matplotlib.colors as colors
    conv = colors.colorConverter

    # Convert 2D numpy array to a list of RGBA tuples.
    actual_colors = list(collection.get_facecolors())
    actual_colors = map(tuple, actual_colors)
    all_actual_colors = list(itertools.islice(itertools.cycle(actual_colors), N))

    for actual, expected in zip(all_actual_colors, expected_colors):
        assert actual == conv.to_rgba(expected, alpha=alpha), \
            '{} != {}'.format(actual, conv.to_rgba(expected, alpha=alpha))


if __name__ == '__main__':
    unittest.main()
