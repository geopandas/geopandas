import itertools
import warnings

import numpy as np
import pandas as pd

from shapely.affinity import rotate
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.plotting import GeoplotAccessor

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # skipif and importorskip do not work for decorators
    from matplotlib.testing.decorators import check_figures_equal, image_comparison

    MPL_DECORATORS = True
except ImportError:
    MPL_DECORATORS = False


@pytest.fixture(autouse=True)
def close_figures(request):
    yield
    plt.close("all")


try:
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()
    MPL_DFT_COLOR = cycle["color"][0]
except KeyError:
    MPL_DFT_COLOR = matplotlib.rcParams["axes.color_cycle"][0]

plt.rcParams.update({"figure.max_open_warning": 0})


class TestPointPlotting:
    def setup_method(self):
        self.N = 10
        self.points = GeoSeries(Point(i, i) for i in range(self.N))

        values = np.arange(self.N)

        self.df = GeoDataFrame({"geometry": self.points, "values": values})
        self.df["exp"] = (values * 10) ** 3

        multipoint1 = MultiPoint(self.points)
        multipoint2 = rotate(multipoint1, 90)
        self.df2 = GeoDataFrame(
            {"geometry": [multipoint1, multipoint2], "values": [0, 1]}
        )

    def test_figsize(self):
        ax = self.points.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))

        ax = self.df.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))

    def test_default_colors(self):
        # # without specifying values -> uniform color

        # GeoSeries
        ax = self.points.plot()
        _check_colors(
            self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N
        )

        # GeoDataFrame
        ax = self.df.plot()
        _check_colors(
            self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N
        )

        # # with specifying values -> different colors for all 10 values
        ax = self.df.plot(column="values")
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, ax.collections[0].get_facecolors(), expected_colors)

    def test_series_color_no_index(self):
        # Color order with ordered index
        colors_ord = pd.Series(["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"])

        # Plot using Series as color
        ax1 = self.df.plot(colors_ord)

        # Correct answer: Add as column to df and plot
        self.df["colors_ord"] = colors_ord
        ax2 = self.df.plot("colors_ord")

        # Confirm out-of-order index re-sorted
        point_colors1 = ax1.collections[0].get_facecolors()
        point_colors2 = ax2.collections[0].get_facecolors()
        np.testing.assert_array_equal(point_colors1[1], point_colors2[1])

    def test_series_color_index(self):
        # Color order with out-of-order index
        colors_ord = pd.Series(
            ["a", "a", "a", "a", "b", "b", "b", "c", "c", "c"],
            index=[0, 3, 6, 9, 1, 4, 7, 2, 5, 8],
        )

        # Plot using Series as color
        ax1 = self.df.plot(colors_ord)

        # Correct answer: Add as column to df and plot
        self.df["colors_ord"] = colors_ord
        ax2 = self.df.plot("colors_ord")

        # Confirm out-of-order index re-sorted
        point_colors1 = ax1.collections[0].get_facecolors()
        point_colors2 = ax2.collections[0].get_facecolors()
        np.testing.assert_array_equal(point_colors1[1], point_colors2[1])

    def test_colormap(self):
        # without specifying values but cmap specified -> no uniform color
        # but different colors for all points

        # GeoSeries
        ax = self.points.plot(cmap="RdYlGn")
        cmap = plt.get_cmap("RdYlGn")
        exp_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)

        ax = self.df.plot(cmap="RdYlGn")
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)

        # # with specifying values -> different colors for all 10 values
        ax = self.df.plot(column="values", cmap="RdYlGn")
        cmap = plt.get_cmap("RdYlGn")
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)

        # when using a cmap with specified lut -> limited number of different
        # colors
        ax = self.points.plot(cmap=plt.get_cmap("Set1", lut=5))
        cmap = plt.get_cmap("Set1", lut=5)
        exp_colors = cmap(list(range(5)) * 2)
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)

    def test_single_color(self):
        ax = self.points.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_facecolors(), ["green"] * self.N)

        ax = self.df.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_facecolors(), ["green"] * self.N)

        # check rgba tuple GH1178
        ax = self.df.plot(color=(0.5, 0.5, 0.5))
        _check_colors(
            self.N, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * self.N
        )
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(
            self.N, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * self.N
        )
        with pytest.raises((ValueError, TypeError)):
            self.df.plot(color="not color")

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'column'
            ax = self.df.plot(column="values", color="green")
            _check_colors(
                self.N, ax.collections[0].get_facecolors(), ["green"] * self.N
            )

    def test_markersize(self):
        ax = self.points.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]

        ax = self.df.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]

        ax = self.df.plot(column="values", markersize=10)
        assert ax.collections[0].get_sizes() == [10]

        ax = self.df.plot(markersize="values")
        assert (ax.collections[0].get_sizes() == self.df["values"]).all()

        ax = self.df.plot(column="values", markersize="values")
        assert (ax.collections[0].get_sizes() == self.df["values"]).all()

    def test_markerstyle(self):
        ax = self.df2.plot(marker="+")
        expected = _style_to_vertices("+")
        np.testing.assert_array_equal(
            expected, ax.collections[0].get_paths()[0].vertices
        )

    def test_style_kwargs(self):
        ax = self.points.plot(edgecolors="k")
        assert (ax.collections[0].get_edgecolor() == [0, 0, 0, 1]).all()

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=np.linspace(0, 0.0, 1.0, self.N))
        except TypeError:
            # no list allowed for alpha up to matplotlib 3.3
            pass
        else:
            np.testing.assert_array_equal(
                np.linspace(0, 0.0, 1.0, self.N), ax.collections[0].get_alpha()
            )

    def test_legend(self):
        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # legend ignored if color is given.
            ax = self.df.plot(column="values", color="green", legend=True)
            assert len(ax.get_figure().axes) == 1  # no separate legend axis

        # legend ignored if no column is given.
        ax = self.df.plot(legend=True)
        assert len(ax.get_figure().axes) == 1  # no separate legend axis

        # # Continuous legend
        # the colorbar matches the Point colors
        ax = self.df.plot(column="values", cmap="RdYlGn", legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = _get_colorbar_ax(ax.get_figure()).collections[-1].get_facecolors()
        # first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        # last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])

        # # Categorical legend
        # the colorbar matches the Point colors
        ax = self.df.plot(column="values", categorical=True, legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = ax.get_legend().axes.collections[-1].get_facecolors()
        # first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        # last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])

        # # Normalized legend
        # the colorbar matches the Point colors
        norm = matplotlib.colors.LogNorm(
            vmin=self.df[1:].exp.min(), vmax=self.df[1:].exp.max()
        )
        ax = self.df[1:].plot(column="exp", cmap="RdYlGn", legend=True, norm=norm)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = _get_colorbar_ax(ax.get_figure()).collections[-1].get_facecolors()
        # first point == bottom of colorbar
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        # last point == top of colorbar
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])
        # colorbar generated proper long transition
        assert cbar_colors.shape == (256, 4)

    def test_subplots_norm(self):
        # colors of subplots are the same as for plot (norm is applied)
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        ax = self.df.plot(column="values", cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_facecolors()
        exp_colors = cmap(np.arange(10) / (20))
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column="values", ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])

    def test_empty_plot(self):
        s = GeoSeries([Polygon()])
        with pytest.warns(UserWarning):
            ax = s.plot()
        assert len(ax.collections) == 0
        s = GeoSeries([])
        with pytest.warns(UserWarning):
            ax = s.plot()
        assert len(ax.collections) == 0
        df = GeoDataFrame([], columns=["geometry"])
        with pytest.warns(UserWarning):
            ax = df.plot()
        assert len(ax.collections) == 0

    def test_empty_geometry(self):
        s = GeoSeries([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon()])
        ax = s.plot()
        assert len(ax.collections) == 1

        # more complex case with GEOMETRYCOLLECTION EMPTY, POINT EMPTY and NONE
        poly = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1), (-1, -1)])
        point = Point(0, 1)
        point_ = Point(10, 10)
        empty_point = Point()

        gdf = GeoDataFrame(geometry=[point, empty_point, point_])
        gdf["geometry"] = gdf.intersection(poly)
        with warnings.catch_warnings():
            # loc to add row calls concat internally, warning for pandas >=2.1
            warnings.filterwarnings(
                "ignore",
                "The behavior of DataFrame concatenation with empty",
                FutureWarning,
            )
            gdf.loc[3] = [None]
        ax = gdf.plot()
        assert len(ax.collections) == 1

    @pytest.mark.parametrize(
        "geoms",
        [
            [
                box(0, 0, 1, 1),
                box(7, 7, 8, 8),
            ],
            [
                LineString([(1, 1), (1, 2)]),
                LineString([(7, 1), (7, 2)]),
            ],
            [
                Point(1, 1),
                Point(7, 7),
            ],
        ],
    )
    def test_empty_geometry_colors(self, geoms):
        s = GeoSeries(
            geoms,
            index=["r", "b"],
        )
        s2 = s.intersection(box(5, 0, 10, 10))
        ax = s2.plot(color=["red", "blue"])
        blue = np.array([0.0, 0.0, 1.0, 1.0])
        if s.geom_type["r"] == "LineString":
            np.testing.assert_array_equal(ax.get_children()[0].get_edgecolor()[0], blue)
        else:
            np.testing.assert_array_equal(ax.get_children()[0].get_facecolor()[0], blue)

    def test_multipoints(self):
        # MultiPoints
        ax = self.df2.plot()
        _check_colors(4, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 4)

        ax = self.df2.plot(column="values")
        cmap = plt.get_cmap(lut=2)
        expected_colors = [cmap(0)] * self.N + [cmap(1)] * self.N
        _check_colors(20, ax.collections[0].get_facecolors(), expected_colors)

        ax = self.df2.plot(color=["r", "b"])
        # colors are repeated for all components within a MultiPolygon
        _check_colors(20, ax.collections[0].get_facecolors(), ["r"] * 10 + ["b"] * 10)

    def test_multipoints_alpha(self):
        ax = self.df2.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df2.plot(alpha=[0.7, 0.2])
        except TypeError:
            # no list allowed for alpha up to matplotlib 3.3
            pass
        else:
            np.testing.assert_array_equal(
                [0.7] * 10 + [0.2] * 10, ax.collections[0].get_alpha()
            )

    def test_categories(self):
        self.df["cats_object"] = ["cat1", "cat2"] * 5
        self.df["nums"] = [1, 2] * 5
        self.df["singlecat_object"] = ["cat2"] * 10
        self.df["cats"] = pd.Categorical(["cat1", "cat2"] * 5)
        self.df["singlecat"] = pd.Categorical(
            ["cat2"] * 10, categories=["cat1", "cat2"]
        )
        self.df["cats_ordered"] = pd.Categorical(
            ["cat2", "cat1"] * 5, categories=["cat2", "cat1"]
        )
        self.df["bool"] = [False, True] * 5
        self.df["bool_extension"] = pd.array([False, True] * 5)
        self.df["cats_string"] = pd.array(["cat1", "cat2"] * 5, dtype="string")

        ax1 = self.df.plot("cats_object", legend=True)
        ax2 = self.df.plot("cats", legend=True)
        ax3 = self.df.plot("singlecat_object", categories=["cat1", "cat2"], legend=True)
        ax4 = self.df.plot("singlecat", legend=True)
        ax5 = self.df.plot("cats_ordered", legend=True)
        ax6 = self.df.plot("nums", categories=[1, 2], legend=True)
        ax7 = self.df.plot("bool", legend=True)
        ax8 = self.df.plot("bool_extension", legend=True)
        ax9 = self.df.plot("cats_string", legend=True)

        point_colors1 = ax1.collections[0].get_facecolors()
        for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            point_colors2 = ax.collections[0].get_facecolors()
            np.testing.assert_array_equal(point_colors1[1], point_colors2[1])

        legend1 = [x.get_markerfacecolor() for x in ax1.get_legend().get_lines()]
        for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            legend2 = [x.get_markerfacecolor() for x in ax.get_legend().get_lines()]
            np.testing.assert_array_equal(legend1, legend2)

        with pytest.raises(TypeError):
            self.df.plot(column="cats_object", categories="non_list")

        with pytest.raises(
            ValueError, match="Column contains values not listed in categories."
        ):
            self.df.plot(column="cats_object", categories=["cat1"])

        with pytest.raises(
            ValueError, match="Cannot specify 'categories' when column has"
        ):
            self.df.plot(column="cats", categories=["cat1"])

    def test_missing(self):
        self.df.loc[0, "values"] = np.nan
        ax = self.df.plot("values")
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N - 1) / (self.N - 2))
        _check_colors(self.N - 1, ax.collections[0].get_facecolors(), expected_colors)

        ax = self.df.plot("values", missing_kwds={"color": "r"})
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N - 1) / (self.N - 2))
        _check_colors(1, ax.collections[1].get_facecolors(), ["r"])
        _check_colors(self.N - 1, ax.collections[0].get_facecolors(), expected_colors)

        ax = self.df.plot(
            "values", missing_kwds={"color": "r"}, categorical=True, legend=True
        )
        _check_colors(1, ax.collections[1].get_facecolors(), ["r"])
        point_colors = ax.collections[0].get_facecolors()
        nan_color = ax.collections[1].get_facecolors()
        leg_colors = ax.get_legend().axes.collections[0].get_facecolors()
        leg_colors1 = ax.get_legend().axes.collections[1].get_facecolors()
        np.testing.assert_array_equal(point_colors[0], leg_colors[0])
        np.testing.assert_array_equal(nan_color[0], leg_colors1[0])

    def test_no_missing_and_missing_kwds(self):
        # GH2210
        df = self.df.copy()
        df["category"] = df["values"].astype("str")
        df.plot("category", missing_kwds={"facecolor": "none"}, legend=True)

    def test_missing_aspect(self):
        self.df.loc[0, "values"] = np.nan
        ax = self.df.plot(
            "values",
            missing_kwds={"color": "r"},
            categorical=True,
            legend=True,
            aspect=2,
        )
        assert ax.get_aspect() == 2


class TestPointZPlotting:
    def setup_method(self):
        self.N = 10
        self.points = GeoSeries(Point(i, i, i) for i in range(self.N))
        values = np.arange(self.N)
        self.df = GeoDataFrame({"geometry": self.points, "values": values})

    def test_plot(self):
        # basic test that points with z coords don't break plotting
        self.df.plot()


class TestLineStringPlotting:
    def setup_method(self):
        self.N = 10
        values = np.arange(self.N)
        self.lines = GeoSeries(
            [LineString([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)],
            index=list("ABCDEFGHIJ"),
        )
        self.df = GeoDataFrame({"geometry": self.lines, "values": values})

        multiline1 = MultiLineString(self.lines.loc["A":"B"].values)
        multiline2 = MultiLineString(self.lines.loc["C":"D"].values)
        self.df2 = GeoDataFrame(
            {"geometry": [multiline1, multiline2], "values": [0, 1]}
        )

        self.linearrings = GeoSeries(
            [LinearRing([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)],
            index=list("ABCDEFGHIJ"),
        )
        self.df3 = GeoDataFrame({"geometry": self.linearrings, "values": values})

    def test_autolim_false(self):
        """Test linestring plot preserving axes limits."""
        ax = self.lines[: self.N // 2].plot()
        ylim = ax.get_ylim()
        self.lines.plot(ax=ax, autolim=False)
        assert ax.get_ylim() == ylim
        ax = self.df[: self.N // 2].plot()
        ylim = ax.get_ylim()
        self.df.plot(ax=ax, autolim=False)
        assert ax.get_ylim() == ylim

    def test_autolim_true(self):
        """Test linestring plot autoscaling axes limits."""
        ax = self.lines[: self.N // 2].plot()
        ylim = ax.get_ylim()
        self.lines.plot(ax=ax, autolim=True)
        assert ax.get_ylim() != ylim
        ax = self.df[: self.N // 2].plot()
        ylim = ax.get_ylim()
        self.df.plot(ax=ax, autolim=True)
        assert ax.get_ylim() != ylim

    def test_single_color(self):
        ax = self.lines.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_colors(), ["green"] * self.N)

        ax = self.df.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_colors(), ["green"] * self.N)

        ax = self.linearrings.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_colors(), ["green"] * self.N)

        ax = self.df3.plot(color="green")
        _check_colors(self.N, ax.collections[0].get_colors(), ["green"] * self.N)

        # check rgba tuple GH1178
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(
            self.N, ax.collections[0].get_colors(), [(0.5, 0.5, 0.5, 0.5)] * self.N
        )
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(
            self.N, ax.collections[0].get_colors(), [(0.5, 0.5, 0.5, 0.5)] * self.N
        )
        with pytest.raises((TypeError, ValueError)):
            self.df.plot(color="not color")

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'column'
            ax = self.df.plot(column="values", color="green")
            _check_colors(self.N, ax.collections[0].get_colors(), ["green"] * self.N)

    def test_style_kwargs_linestyle(self):
        # single
        for ax in [
            self.lines.plot(linestyle=":", linewidth=1),
            self.df.plot(linestyle=":", linewidth=1),
            self.df.plot(column="values", linestyle=":", linewidth=1),
        ]:
            assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()

        # tuple
        ax = self.lines.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()

        # multiple
        ls = [("dashed", "dotted", "dashdot", "solid")[k % 4] for k in range(self.N)]
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls]
        for ax in [
            self.lines.plot(linestyle=ls, linewidth=1),
            self.lines.plot(linestyles=ls, linewidth=1),
            self.df.plot(linestyle=ls, linewidth=1),
            self.df.plot(column="values", linestyle=ls, linewidth=1),
        ]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        # single
        for ax in [
            self.lines.plot(linewidth=2),
            self.df.plot(linewidth=2),
            self.df.plot(column="values", linewidth=2),
        ]:
            np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())

        # multiple
        lw = [(0, 1, 2, 5.5, 10)[k % 5] for k in range(self.N)]
        for ax in [
            self.lines.plot(linewidth=lw),
            self.lines.plot(linewidths=lw),
            self.df.plot(linewidth=lw),
            self.df.plot(column="values", linewidth=lw),
        ]:
            np.testing.assert_array_equal(lw, ax.collections[0].get_linewidths())

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=np.linspace(0, 0.0, 1.0, self.N))
        except TypeError:
            # no list allowed for alpha up to matplotlib 3.3
            pass
        else:
            np.testing.assert_array_equal(
                np.linspace(0, 0.0, 1.0, self.N), ax.collections[0].get_alpha()
            )

    def test_style_kwargs_path_effects(self):
        from matplotlib.patheffects import withStroke

        effects = [withStroke(linewidth=8, foreground="b")]
        ax = self.df.plot(color="orange", path_effects=effects)
        assert ax.collections[0].get_path_effects()[0].__dict__["_gc"] == {
            "linewidth": 8,
            "foreground": "b",
        }

    def test_subplots_norm(self):
        # colors of subplots are the same as for plot (norm is applied)
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        ax = self.df.plot(column="values", cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_edgecolors()
        exp_colors = cmap(np.arange(10) / (20))
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column="values", ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_edgecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])

    def test_multilinestrings(self):
        # MultiLineStrings
        ax = self.df2.plot()
        assert len(ax.collections[0].get_paths()) == 4
        _check_colors(4, ax.collections[0].get_edgecolors(), [MPL_DFT_COLOR] * 4)

        ax = self.df2.plot("values")
        cmap = plt.get_cmap(lut=2)
        # colors are repeated for all components within a MultiLineString
        expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
        _check_colors(4, ax.collections[0].get_edgecolors(), expected_colors)

        ax = self.df2.plot(color=["r", "b"])
        # colors are repeated for all components within a MultiLineString
        _check_colors(4, ax.collections[0].get_edgecolors(), ["r", "r", "b", "b"])


class TestPolygonPlotting:
    def setup_method(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        self.polys = GeoSeries([t1, t2], index=list("AB"))
        self.df = GeoDataFrame({"geometry": self.polys, "values": [0, 1]})

        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame(
            {"geometry": [multipoly1, multipoly2], "values": [0, 1]}
        )

        t3 = Polygon([(2, 0), (3, 0), (3, 1)])
        df_nan = GeoDataFrame({"geometry": t3, "values": [np.nan]})
        self.df3 = pd.concat([self.df, df_nan])

    def test_autolim_false(self):
        """Test polygon plot preserving axes limits."""
        ax = self.polys[:1].plot()
        xlim = ax.get_xlim()
        self.polys.plot(ax=ax, autolim=False)
        assert ax.get_xlim() == xlim
        ax = self.df[:1].plot()
        xlim = ax.get_xlim()
        self.df.plot(ax=ax, autolim=False)
        assert ax.get_xlim() == xlim

    def test_autolim_true(self):
        """Test polygon plot autoscaling axes limits."""
        ax = self.polys[:1].plot()
        xlim = ax.get_xlim()
        self.polys.plot(ax=ax, autolim=True)
        assert ax.get_xlim() != xlim
        ax = self.df[:1].plot()
        xlim = ax.get_xlim()
        self.df.plot(ax=ax, autolim=True)
        assert ax.get_xlim() != xlim

    def test_single_color(self):
        ax = self.polys.plot(color="green")
        _check_colors(2, ax.collections[0].get_facecolors(), ["green"] * 2)
        # color only sets facecolor
        assert len(ax.collections[0].get_edgecolors()) == 0

        ax = self.df.plot(color="green")
        _check_colors(2, ax.collections[0].get_facecolors(), ["green"] * 2)
        assert len(ax.collections[0].get_edgecolors()) == 0

        # check rgba tuple GH1178
        ax = self.df.plot(color=(0.5, 0.5, 0.5))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
        with pytest.raises((TypeError, ValueError)):
            self.df.plot(color="not color")

        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # 'color' overrides 'values'
            ax = self.df.plot(column="values", color="green")
            _check_colors(2, ax.collections[0].get_facecolors(), ["green"] * 2)

    def test_vmin_vmax(self):
        # when vmin == vmax, all polygons should be the same color

        # non-categorical
        ax = self.df.plot(column="values", categorical=False, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])

        # categorical
        ax = self.df.plot(column="values", categorical=True, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])

        # vmin vmax set correctly for array with NaN (GitHub issue 877)
        ax = self.df3.plot(column="values")
        actual_colors = ax.collections[0].get_facecolors()
        assert np.any(np.not_equal(actual_colors[0], actual_colors[1]))

    def test_style_kwargs_color(self):
        # facecolor overrides default cmap when color is not set
        ax = self.polys.plot(facecolor="k")
        _check_colors(2, ax.collections[0].get_facecolors(), ["k"] * 2)

        # facecolor overrides more general-purpose color when both are set
        ax = self.polys.plot(color="red", facecolor="k")
        # TODO with new implementation, color overrides facecolor
        # _check_colors(2, ax.collections[0], ['k']*2, alpha=0.5)

        # edgecolor
        ax = self.polys.plot(edgecolor="red")
        np.testing.assert_array_equal(
            [(1, 0, 0, 1)], ax.collections[0].get_edgecolors()
        )

        ax = self.df.plot("values", edgecolor="red")
        np.testing.assert_array_equal(
            [(1, 0, 0, 1)], ax.collections[0].get_edgecolors()
        )

        # alpha sets both edge and face
        ax = self.polys.plot(facecolor="g", edgecolor="r", alpha=0.4)
        _check_colors(2, ax.collections[0].get_facecolors(), ["g"] * 2, alpha=0.4)
        _check_colors(2, ax.collections[0].get_edgecolors(), ["r"] * 2, alpha=0.4)

        # check rgba tuple GH1178 for face and edge
        ax = self.df.plot(facecolor=(0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
        _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6)] * 2)

        ax = self.df.plot(
            facecolor=(0.5, 0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6, 0.5)
        )
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
        _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6, 0.5)] * 2)

    def test_style_kwargs_linestyle(self):
        #   single
        ax = self.df.plot(linestyle=":", linewidth=1)
        assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()

        # tuple
        ax = self.df.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()

        #   multiple
        ls = ["dashed", "dotted"]
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls]
        for ax in [
            self.df.plot(linestyle=ls, linewidth=1),
            self.df.plot(linestyles=ls, linewidth=1),
        ]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        #   single
        ax = self.df.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())
        #   multiple
        for ax in [self.df.plot(linewidth=[2, 4]), self.df.plot(linewidths=[2, 4])]:
            np.testing.assert_array_equal([2, 4], ax.collections[0].get_linewidths())

        # alpha
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=[0.7, 0.2])
        except TypeError:
            # no list allowed for alpha up to matplotlib 3.3
            pass
        else:
            np.testing.assert_array_equal([0.7, 0.2], ax.collections[0].get_alpha())

    def test_legend_kwargs(self):
        categories = list(self.df["values"].unique())
        prefix = "LABEL_FOR_"
        ax = self.df.plot(
            column="values",
            categorical=True,
            categories=categories,
            legend=True,
            legend_kwds={
                "labels": [prefix + str(c) for c in categories],
                "frameon": False,
            },
        )
        assert len(categories) == len(ax.get_legend().get_texts())
        assert ax.get_legend().get_texts()[0].get_text().startswith(prefix)
        assert ax.get_legend().get_frame_on() is False

    def test_colorbar_kwargs(self):
        # Test if kwargs are passed to colorbar

        label_txt = "colorbar test"

        ax = self.df.plot(
            column="values",
            categorical=False,
            legend=True,
            legend_kwds={"label": label_txt},
        )
        cax = _get_colorbar_ax(ax.get_figure())
        assert cax.get_ylabel() == label_txt

        ax = self.df.plot(
            column="values",
            categorical=False,
            legend=True,
            legend_kwds={"label": label_txt, "orientation": "horizontal"},
        )

        cax = _get_colorbar_ax(ax.get_figure())
        assert cax.get_xlabel() == label_txt

    if MPL_DECORATORS:
        """Test that geometries are properly normalized so holes appear."""

        @image_comparison(
            ["polygon_with_holes"],
            extensions=["png", "pdf"],
            remove_text=True,
            savefig_kwarg={"dpi": 300, "bbox_inches": "tight"},
        )
        def test_plot_polygon_with_holes(self):
            geoms = [
                Polygon(
                    [(0, 0), (0, 5), (5, 5), (5, 0)],
                    [
                        [(1, 1), (1, 2), (2, 2), (2, 1)],
                        [(3, 2), (3, 3), (4, 3), (4, 2)],
                    ],
                )
            ]

            _df = GeoDataFrame(geometry=geoms)
            _df.plot()
    else:

        def test_plot_polygon_with_holes(self):
            geoms = [
                Polygon(
                    [(0, 0), (0, 5), (5, 5), (5, 0)],
                    [
                        [(1, 1), (1, 2), (2, 2), (2, 1)],
                        [(3, 2), (3, 3), (4, 3), (4, 2)],
                    ],
                )
            ]

            _df = GeoDataFrame(geometry=geoms)
            ax = _df.plot()
            plotted_vertices = ax.collections[0].get_paths()[0].vertices
            expected_vertices = _df.normalize().get_coordinates().to_numpy()
            np.testing.assert_array_equal(plotted_vertices, expected_vertices)

    def test_fmt_ignore(self):
        # test if fmt is removed if scheme is not passed (it would raise Error)
        # GH #1253

        self.df.plot(
            column="values",
            categorical=True,
            legend=True,
            legend_kwds={"fmt": "{:.0f}"},
        )

        self.df.plot(column="values", legend=True, legend_kwds={"fmt": "{:.0f}"})

    def test_multipolygons_color(self):
        # MultiPolygons
        ax = self.df2.plot()
        assert len(ax.collections[0].get_paths()) == 4
        _check_colors(4, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 4)

        ax = self.df2.plot("values")
        cmap = plt.get_cmap(lut=2)
        # colors are repeated for all components within a MultiPolygon
        expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
        _check_colors(4, ax.collections[0].get_facecolors(), expected_colors)

        ax = self.df2.plot(color=["r", "b"])
        # colors are repeated for all components within a MultiPolygon
        _check_colors(4, ax.collections[0].get_facecolors(), ["r", "r", "b", "b"])

    def test_multipolygons_linestyle(self):
        # single
        ax = self.df2.plot(linestyle=":", linewidth=1)
        assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()

        # tuple
        ax = self.df2.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()

        # multiple
        ls = ["dashed", "dotted"]
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls for i in range(2)]
        for ax in [
            self.df2.plot(linestyle=ls, linewidth=1),
            self.df2.plot(linestyles=ls, linewidth=1),
        ]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_multipolygons_linewidth(self):
        # single
        ax = self.df2.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())

        # multiple
        for ax in [self.df2.plot(linewidth=[2, 4]), self.df2.plot(linewidths=[2, 4])]:
            np.testing.assert_array_equal(
                [2, 2, 4, 4], ax.collections[0].get_linewidths()
            )

    def test_multipolygons_alpha(self):
        ax = self.df2.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df2.plot(alpha=[0.7, 0.2])
        except TypeError:
            # no list allowed for alpha up to matplotlib 3.3
            pass
        else:
            np.testing.assert_array_equal(
                [0.7, 0.7, 0.2, 0.2], ax.collections[0].get_alpha()
            )

    if MPL_DECORATORS:
        """Test multipolygons properly orient so holes will appear.
        Test partly derived from shapely PR #1933."""

        @image_comparison(
            ["multipolygon_with_holes"],
            extensions=["png", "pdf"],
            remove_text=True,
            savefig_kwarg={"dpi": 300, "bbox_inches": "tight"},
        )
        def test_multipolygons_with_interior(self):
            poly1 = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
            poly2 = box(3, 3, 6, 6).difference(box(4, 4, 5, 5))
            multipoly = MultiPolygon([poly1, poly2])
            _df = GeoDataFrame(geometry=[multipoly])
            _df.plot()
    else:

        def test_multipolygons_with_interior(self):
            poly1 = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
            poly2 = box(3, 3, 6, 6).difference(box(4, 4, 5, 5))
            multipoly = MultiPolygon([poly1, poly2])
            _df = GeoDataFrame(geometry=[multipoly])
            ax = _df.plot()
            plotted_vertices = np.append(
                ax.collections[0].get_paths()[0].vertices,
                ax.collections[0].get_paths()[1].vertices,
                axis=0,
            )
            expected_vertices = _df.normalize().get_coordinates().to_numpy()
            np.testing.assert_array_equal(plotted_vertices, expected_vertices)

    def test_subplots_norm(self):
        # colors of subplots are the same as for plot (norm is applied)
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
        ax = self.df.plot(column="values", cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_facecolors()
        exp_colors = cmap(np.arange(2) / (10))
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column="values", ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])


class TestPolygonZPlotting:
    def setup_method(self):
        t1 = Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 1)])
        t2 = Polygon([(1, 0, 0), (2, 0, 0), (2, 1, 1)])
        self.polys = GeoSeries([t1, t2], index=list("AB"))
        self.df = GeoDataFrame({"geometry": self.polys, "values": [0, 1]})

        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame(
            {"geometry": [multipoly1, multipoly2], "values": [0, 1]}
        )

    def test_plot(self):
        # basic test that points with z coords don't break plotting
        self.df.plot()


class TestColorParamArray:
    def setup_method(self):
        geom = []
        color = []
        for a, b in [(0, 2), (4, 6)]:
            b = box(a, a, b, b)
            geom += [b, b.buffer(0.8).exterior, b.centroid]
            color += ["red", "green", "blue"]

        self.gdf = GeoDataFrame({"geometry": geom, "color_rgba": color})
        self.mgdf = self.gdf.dissolve(self.gdf.geom_type)

    def test_color_single(self):
        ax = self.gdf.plot(color=self.gdf["color_rgba"])

        _check_colors(
            4,
            np.concatenate([c.get_edgecolor() for c in ax.collections]),
            ["green"] * 2 + ["blue"] * 2,
        )
        _check_colors(
            4,
            np.concatenate([c.get_facecolor() for c in ax.collections]),
            ["red"] * 2 + ["blue"] * 2,
        )

    def test_color_multi(self):
        ax = self.mgdf.plot(color=self.mgdf["color_rgba"])

        _check_colors(
            4,
            np.concatenate([c.get_edgecolor() for c in ax.collections]),
            ["green"] * 2 + ["blue"] * 2,
        )
        _check_colors(
            4,
            np.concatenate([c.get_facecolor() for c in ax.collections]),
            ["red"] * 2 + ["blue"] * 2,
        )


class TestGeometryCollectionPlotting:
    def setup_method(self):
        coll1 = GeometryCollection(
            [
                Polygon([(1, 0), (2, 0), (2, 1)]),
                MultiLineString([((0.5, 0.5), (1, 1)), ((1, 0.5), (1.5, 1))]),
            ]
        )
        coll2 = GeometryCollection(
            [Point(0.75, 0.25), Polygon([(2, 2), (3, 2), (2, 3)])]
        )

        self.series = GeoSeries([coll1, coll2])
        self.df = GeoDataFrame({"geometry": self.series, "values": [1, 2]})

    def test_colors(self):
        # default uniform color
        ax = self.series.plot()
        _check_colors(
            2, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 2
        )  # poly
        _check_colors(
            2, ax.collections[1].get_edgecolors(), [MPL_DFT_COLOR] * 2
        )  # line
        _check_colors(1, ax.collections[2].get_facecolors(), [MPL_DFT_COLOR])  # point

    def test_values(self):
        ax = self.df.plot("values")
        cmap = plt.get_cmap()
        exp_colors = cmap([0.0, 1.0])
        _check_colors(2, ax.collections[0].get_facecolors(), exp_colors)  # poly
        _check_colors(
            2, ax.collections[1].get_edgecolors(), [exp_colors[0]] * 2
        )  # line
        _check_colors(1, ax.collections[2].get_facecolors(), [exp_colors[1]])  # point


class TestNonuniformGeometryPlotting:
    def setup_method(self):
        pytest.importorskip("matplotlib")

        poly = Polygon([(1, 0), (2, 0), (2, 1)])
        line = LineString([(0.5, 0.5), (1, 1), (1, 0.5), (1.5, 1)])
        point = Point(0.75, 0.25)
        self.series = GeoSeries([poly, line, point])
        self.df = GeoDataFrame({"geometry": self.series, "values": [1, 2, 3]})

    def test_colors(self):
        # default uniform color
        ax = self.series.plot()
        _check_colors(1, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR])
        _check_colors(1, ax.collections[1].get_edgecolors(), [MPL_DFT_COLOR])
        _check_colors(1, ax.collections[2].get_facecolors(), [MPL_DFT_COLOR])

        # colormap: different colors
        ax = self.series.plot(cmap="RdYlGn")
        cmap = plt.get_cmap("RdYlGn")
        exp_colors = cmap(np.arange(3) / (3 - 1))
        _check_colors(1, ax.collections[0].get_facecolors(), [exp_colors[0]])
        _check_colors(1, ax.collections[1].get_edgecolors(), [exp_colors[1]])
        _check_colors(1, ax.collections[2].get_facecolors(), [exp_colors[2]])

    def test_style_kwargs(self):
        ax = self.series.plot(markersize=10)
        assert ax.collections[2].get_sizes() == [10]
        ax = self.df.plot(markersize=10)
        assert ax.collections[2].get_sizes() == [10]

    def test_style_kwargs_linestyle(self):
        # single
        for ax in [
            self.series.plot(linestyle=":", linewidth=1),
            self.df.plot(linestyle=":", linewidth=1),
        ]:
            assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()

        # tuple
        ax = self.series.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()

    @pytest.mark.skip(
        reason="array-like style_kwds not supported for mixed geometry types (#1379)"
    )
    def test_style_kwargs_linestyle_listlike(self):
        # multiple
        ls = ["solid", "dotted", "dashdot"]
        exp_ls = [_style_to_linestring_onoffseq(style, 1) for style in ls]
        for ax in [
            self.series.plot(linestyle=ls, linewidth=1),
            self.series.plot(linestyles=ls, linewidth=1),
            self.df.plot(linestyles=ls, linewidth=1),
        ]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        # single
        ax = self.df.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())

    @pytest.mark.skip(
        reason="array-like style_kwds not supported for mixed geometry types (#1379)"
    )
    def test_style_kwargs_linewidth_listlike(self):
        # multiple
        for ax in [
            self.series.plot(linewidths=[2, 4, 5.5]),
            self.series.plot(linewidths=[2, 4, 5.5]),
            self.df.plot(linewidths=[2, 4, 5.5]),
        ]:
            np.testing.assert_array_equal(
                [2, 4, 5.5], ax.collections[0].get_linewidths()
            )

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        # TODO splitting array-like arguments for the different plot types
        # is not yet supported - https://github.com/geopandas/geopandas/issues/1379
        # try:
        #     ax = self.df.plot(alpha=[0.7, 0.2, 0.9])
        # except TypeError:
        #     # no list allowed for alpha up to matplotlib 3.3
        #     pass
        # else:
        #     np.testing.assert_array_equal(
        #         [0.7, 0.2, 0.9], ax.collections[0].get_alpha()
        #     )


@pytest.fixture(scope="class")
def _setup_class_geographic_aspect(naturalearth_lowres, request):
    """Attach naturalearth_lowres class attribute for unittest style setup_method"""
    df = read_file(naturalearth_lowres)
    request.cls.north = df.loc[df.continent == "North America"]
    request.cls.north_proj = request.cls.north.to_crs("ESRI:102008")
    bounds = request.cls.north.total_bounds
    y_coord = np.mean([bounds[1], bounds[3]])
    request.cls.exp = 1 / np.cos(y_coord * np.pi / 180)


@pytest.mark.usefixtures("_setup_class_geographic_aspect")
@pytest.mark.skipif(not compat.HAS_PYPROJ, reason="pyproj not available")
class TestGeographicAspect:
    def test_auto(self):
        ax = self.north.geometry.plot()
        assert ax.get_aspect() == self.exp
        ax2 = self.north_proj.geometry.plot()
        assert ax2.get_aspect() in ["equal", 1.0]
        ax = self.north.plot()
        assert ax.get_aspect() == self.exp
        ax2 = self.north_proj.plot()
        assert ax2.get_aspect() in ["equal", 1.0]
        ax3 = self.north.plot("pop_est")
        assert ax3.get_aspect() == self.exp
        ax4 = self.north_proj.plot("pop_est")
        assert ax4.get_aspect() in ["equal", 1.0]

    def test_manual(self):
        ax = self.north.geometry.plot(aspect="equal")
        assert ax.get_aspect() in ["equal", 1.0]
        self.north.geometry.plot(ax=ax, aspect=None)
        assert ax.get_aspect() in ["equal", 1.0]
        ax2 = self.north.geometry.plot(aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.geometry.plot(ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.geometry.plot(aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.geometry.plot(ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5
        ax = self.north.plot(aspect="equal")
        assert ax.get_aspect() in ["equal", 1.0]
        self.north.plot(ax=ax, aspect=None)
        assert ax.get_aspect() in ["equal", 1.0]
        ax2 = self.north.plot(aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.plot(ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.plot(aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.plot(ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5
        ax = self.north.plot("pop_est", aspect="equal")
        assert ax.get_aspect() in ["equal", 1.0]
        self.north.plot("pop_est", ax=ax, aspect=None)
        assert ax.get_aspect() in ["equal", 1.0]
        ax2 = self.north.plot("pop_est", aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.plot("pop_est", ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.plot("pop_est", aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.plot("pop_est", ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5


@pytest.mark.filterwarnings(
    "ignore:Numba not installed. Using slow pure python version.:UserWarning"
)
class TestMapclassifyPlotting:
    @classmethod
    def setup_class(cls):
        try:
            import mapclassify
        except ImportError:
            pytest.importorskip("mapclassify")
        cls.mc = mapclassify
        cls.classifiers = list(mapclassify.classifiers.CLASSIFIERS)
        cls.classifiers.remove("UserDefined")

    @pytest.fixture
    def df(self, naturalearth_lowres):
        # version of naturalearth_lowres for mapclassify plotting tests
        df = read_file(naturalearth_lowres)
        df["NEGATIVES"] = np.linspace(-10, 10, len(df.index))
        df["low_vals"] = np.linspace(0, 0.3, df.shape[0])
        df["mid_vals"] = np.linspace(0.3, 0.7, df.shape[0])
        df["high_vals"] = np.linspace(0.7, 1.0, df.shape[0])
        df.loc[df.index[:20:2], "high_vals"] = np.nan
        return df

    @pytest.fixture
    def nybb(self, nybb_filename):
        # version of nybb for mapclassify plotting tests
        df = read_file(nybb_filename)
        df["vals"] = [0.001, 0.002, 0.003, 0.004, 0.005]
        return df

    def test_legend(self, df):
        with warnings.catch_warnings(record=True) as _:  # don't print warning
            # warning coming from scipy.stats
            ax = df.plot(
                column="pop_est", scheme="QUANTILES", k=3, cmap="OrRd", legend=True
            )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [
            s.split("|")[0][1:-2]
            for s in str(self.mc.Quantiles(df["pop_est"], k=3)).split("\n")[4:]
        ]
        assert labels == expected

    def test_bin_labels(self, df):
        ax = df.plot(
            column="pop_est",
            scheme="QUANTILES",
            k=3,
            cmap="OrRd",
            legend=True,
            legend_kwds={"labels": ["foo", "bar", "baz"]},
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ["foo", "bar", "baz"]
        assert labels == expected

    def test_invalid_labels_length(self, df):
        with pytest.raises(ValueError):
            df.plot(
                column="pop_est",
                scheme="QUANTILES",
                k=3,
                cmap="OrRd",
                legend=True,
                legend_kwds={"labels": ["foo", "bar"]},
            )

    def test_negative_legend(self, df):
        ax = df.plot(
            column="NEGATIVES", scheme="FISHER_JENKS", k=3, cmap="OrRd", legend=True
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ["-10.00,  -3.41", " -3.41,   3.30", "  3.30,  10.00"]
        assert labels == expected

    def test_fmt(self, df):
        ax = df.plot(
            column="NEGATIVES",
            scheme="FISHER_JENKS",
            k=3,
            cmap="OrRd",
            legend=True,
            legend_kwds={"fmt": "{:.0f}"},
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ["-10,  -3", " -3,   3", "  3,  10"]
        assert labels == expected

    def test_interval(self, df):
        ax = df.plot(
            column="NEGATIVES",
            scheme="FISHER_JENKS",
            k=3,
            cmap="OrRd",
            legend=True,
            legend_kwds={"interval": True},
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ["[-10.00,  -3.41]", "( -3.41,   3.30]", "(  3.30,  10.00]"]
        assert labels == expected

    @pytest.mark.parametrize("scheme", ["FISHER_JENKS", "FISHERJENKS"])
    def test_scheme_name_compat(self, scheme, df):
        ax = df.plot(column="NEGATIVES", scheme=scheme, k=3, legend=True)
        assert len(ax.get_legend().get_texts()) == 3

    def test_schemes(self, df):
        # test if all available classifiers pass
        for scheme in self.classifiers:
            df.plot(column="pop_est", scheme=scheme, legend=True)

    def test_classification_kwds(self, df):
        ax = df.plot(
            column="pop_est",
            scheme="percentiles",
            k=3,
            classification_kwds={"pct": [50, 100]},
            cmap="OrRd",
            legend=True,
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [
            s.split("|")[0][1:-2]
            for s in str(self.mc.Percentiles(df["pop_est"], pct=[50, 100])).split("\n")[
                4:
            ]
        ]

        assert labels == expected

    def test_invalid_scheme(self, df):
        with pytest.raises(ValueError):
            scheme = "invalid_scheme_*#&)(*#"
            df.plot(column="gdp_md_est", scheme=scheme, k=3, cmap="OrRd", legend=True)

    def test_cax_legend_passing(self, df):
        """Pass a 'cax' argument to 'df.plot(.)', that is valid only if 'ax' is
        passed as well (if not, a new figure is created ad hoc, and 'cax' is
        ignored)
        """
        ax = plt.axes()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        with pytest.raises(ValueError):
            ax = df.plot(column="pop_est", cmap="OrRd", legend=True, cax=cax)

    def test_cax_legend_height(self, df):
        """Pass a cax argument to 'df.plot(.)', the legend location must be
        aligned with those of main plot
        """
        # base case
        with warnings.catch_warnings(record=True) as _:  # don't print warning
            ax = df.plot(column="pop_est", cmap="OrRd", legend=True)
        plot_height = _get_ax(ax.get_figure(), "").get_position().height
        legend_height = _get_ax(ax.get_figure(), "<colorbar>").get_position().height
        assert abs(plot_height - legend_height) >= 1e-6
        # fix heights with cax argument
        fig, ax2 = plt.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1, label="fixed_colorbar")
        with warnings.catch_warnings(record=True) as _:
            ax2 = df.plot(column="pop_est", cmap="OrRd", legend=True, cax=cax, ax=ax2)
        plot_height = _get_ax(fig, "").get_position().height
        legend_height = _get_ax(fig, "fixed_colorbar").get_position().height
        assert abs(plot_height - legend_height) < 1e-6

    def test_empty_bins(self, df):
        bins = np.arange(1, 11) / 10
        ax = df.plot(
            "low_vals",
            scheme="UserDefined",
            classification_kwds={"bins": bins},
            legend=True,
        )
        expected = np.array(
            [
                [0.281412, 0.155834, 0.469201, 1.0],
                [0.267004, 0.004874, 0.329415, 1.0],
                [0.244972, 0.287675, 0.53726, 1.0],
            ]
        )
        assert all(
            (z == expected).all(axis=1).any()
            for z in ax.collections[0].get_facecolors()
        )
        labels = [
            "0.00, 0.10",
            "0.10, 0.20",
            "0.20, 0.30",
            "0.30, 0.40",
            "0.40, 0.50",
            "0.50, 0.60",
            "0.60, 0.70",
            "0.70, 0.80",
            "0.80, 0.90",
            "0.90, 1.00",
        ]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == legend

        legend_colors_exp = [
            (0.267004, 0.004874, 0.329415, 1.0),
            (0.281412, 0.155834, 0.469201, 1.0),
            (0.244972, 0.287675, 0.53726, 1.0),
            (0.190631, 0.407061, 0.556089, 1.0),
            (0.147607, 0.511733, 0.557049, 1.0),
            (0.119699, 0.61849, 0.536347, 1.0),
            (0.20803, 0.718701, 0.472873, 1.0),
            (0.430983, 0.808473, 0.346476, 1.0),
            (0.709898, 0.868751, 0.169257, 1.0),
            (0.993248, 0.906157, 0.143936, 1.0),
        ]

        assert [
            line.get_markerfacecolor() for line in ax.get_legend().get_lines()
        ] == legend_colors_exp

        ax2 = df.plot(
            "mid_vals",
            scheme="UserDefined",
            classification_kwds={"bins": bins},
            legend=True,
        )
        expected = np.array(
            [
                [0.244972, 0.287675, 0.53726, 1.0],
                [0.190631, 0.407061, 0.556089, 1.0],
                [0.147607, 0.511733, 0.557049, 1.0],
                [0.119699, 0.61849, 0.536347, 1.0],
                [0.20803, 0.718701, 0.472873, 1.0],
            ]
        )
        assert all(
            (z == expected).all(axis=1).any()
            for z in ax2.collections[0].get_facecolors()
        )

        labels = [
            "-inf, 0.10",
            "0.10, 0.20",
            "0.20, 0.30",
            "0.30, 0.40",
            "0.40, 0.50",
            "0.50, 0.60",
            "0.60, 0.70",
            "0.70, 0.80",
            "0.80, 0.90",
            "0.90, 1.00",
        ]
        legend = [t.get_text() for t in ax2.get_legend().get_texts()]
        assert labels == legend
        assert [
            line.get_markerfacecolor() for line in ax2.get_legend().get_lines()
        ] == legend_colors_exp

        ax3 = df.plot(
            "high_vals",
            scheme="UserDefined",
            classification_kwds={"bins": bins},
            legend=True,
        )
        expected = np.array(
            [
                [0.709898, 0.868751, 0.169257, 1.0],
                [0.993248, 0.906157, 0.143936, 1.0],
                [0.430983, 0.808473, 0.346476, 1.0],
            ]
        )
        assert all(
            (z == expected).all(axis=1).any()
            for z in ax3.collections[0].get_facecolors()
        )

        legend = [t.get_text() for t in ax3.get_legend().get_texts()]
        assert labels == legend

        assert [
            line.get_markerfacecolor() for line in ax3.get_legend().get_lines()
        ] == legend_colors_exp

    def test_equally_formatted_bins(self, nybb):
        ax = nybb.plot(
            "vals",
            scheme="quantiles",
            legend=True,
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [
            "0.00, 0.00",
            "0.00, 0.00",
            "0.00, 0.00",
            "0.00, 0.00",
            "0.00, 0.01",
        ]
        assert labels == expected

        ax2 = nybb.plot(
            "vals", scheme="quantiles", legend=True, legend_kwds={"fmt": "{:.3f}"}
        )
        labels = [t.get_text() for t in ax2.get_legend().get_texts()]
        expected = [
            "0.001, 0.002",
            "0.002, 0.003",
            "0.003, 0.003",
            "0.003, 0.004",
            "0.004, 0.005",
        ]
        assert labels == expected


class TestPlotCollections:
    def setup_method(self):
        self.N = 3
        self.values = np.arange(self.N)
        self.points = GeoSeries(Point(i, i) for i in range(self.N))
        self.lines = GeoSeries(
            [LineString([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)]
        )
        self.polygons = GeoSeries(
            [Polygon([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)]
        )

    def test_points(self):
        from matplotlib.collections import PathCollection

        from geopandas.plotting import _plot_point_collection

        fig, ax = plt.subplots()
        coll = _plot_point_collection(ax, self.points)
        assert isinstance(coll, PathCollection)
        ax.cla()

        # default: single default matplotlib color
        coll = _plot_point_collection(ax, self.points)
        _check_colors(self.N, coll.get_facecolors(), [MPL_DFT_COLOR] * self.N)
        # edgecolor depends on matplotlib version
        # _check_colors(self.N, coll.get_edgecolors(), [MPL_DFT_COLOR]*self.N)
        ax.cla()

        # specify single other color
        coll = _plot_point_collection(ax, self.points, color="g")
        _check_colors(self.N, coll.get_facecolors(), ["g"] * self.N)
        _check_colors(self.N, coll.get_edgecolors(), ["g"] * self.N)
        ax.cla()

        # specify edgecolor/facecolor
        coll = _plot_point_collection(ax, self.points, facecolor="g", edgecolor="r")
        _check_colors(self.N, coll.get_facecolors(), ["g"] * self.N)
        _check_colors(self.N, coll.get_edgecolors(), ["r"] * self.N)
        ax.cla()

        # list of colors
        coll = _plot_point_collection(ax, self.points, color=["r", "g", "b"])
        _check_colors(self.N, coll.get_facecolors(), ["r", "g", "b"])
        _check_colors(self.N, coll.get_edgecolors(), ["r", "g", "b"])
        ax.cla()

        coll = _plot_point_collection(
            ax,
            self.points,
            color=[(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        _check_colors(
            self.N,
            coll.get_facecolors(),
            [(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        _check_colors(
            self.N,
            coll.get_edgecolors(),
            [(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        ax.cla()

        # not a color
        with pytest.raises((TypeError, ValueError)):
            _plot_point_collection(ax, self.points, color="not color")

    def test_points_values(self):
        from geopandas.plotting import _plot_point_collection

        # default colormap
        fig, ax = plt.subplots()
        coll = _plot_point_collection(ax, self.points, self.values)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_facecolors(), expected_colors)
        # edgecolor depends on matplotlib version
        # _check_colors(self.N, coll.get_edgecolors(), expected_colors)

    def test_linestrings(self):
        from matplotlib.collections import LineCollection

        from geopandas.plotting import _plot_linestring_collection

        fig, ax = plt.subplots()
        coll = _plot_linestring_collection(ax, self.lines)
        assert isinstance(coll, LineCollection)
        ax.cla()

        # default: single default matplotlib color
        coll = _plot_linestring_collection(ax, self.lines)
        _check_colors(self.N, coll.get_color(), [MPL_DFT_COLOR] * self.N)
        ax.cla()

        # specify single other color
        coll = _plot_linestring_collection(ax, self.lines, color="g")
        _check_colors(self.N, coll.get_colors(), ["g"] * self.N)
        ax.cla()

        # specify edgecolor / facecolor
        coll = _plot_linestring_collection(ax, self.lines, facecolor="g", edgecolor="r")
        _check_colors(self.N, coll.get_facecolors(), ["g"] * self.N)
        _check_colors(self.N, coll.get_edgecolors(), ["r"] * self.N)
        ax.cla()

        # list of colors
        coll = _plot_linestring_collection(ax, self.lines, color=["r", "g", "b"])
        _check_colors(self.N, coll.get_colors(), ["r", "g", "b"])
        ax.cla()

        coll = _plot_linestring_collection(
            ax,
            self.lines,
            color=[(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        _check_colors(
            self.N,
            coll.get_colors(),
            [(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        ax.cla()

        # pass through of kwargs
        coll = _plot_linestring_collection(ax, self.lines, linestyle="--", linewidth=1)
        exp_ls = _style_to_linestring_onoffseq("dashed", 1)
        res_ls = coll.get_linestyle()[0]
        assert res_ls[0] == exp_ls[0]
        assert res_ls[1] == exp_ls[1]
        ax.cla()

        # not a color
        with pytest.raises((TypeError, ValueError)):
            _plot_linestring_collection(ax, self.lines, color="not color")

    def test_linestrings_values(self):
        from geopandas.plotting import _plot_linestring_collection

        fig, ax = plt.subplots()

        # default colormap
        coll = _plot_linestring_collection(ax, self.lines, self.values)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_color(), expected_colors)
        ax.cla()

        # specify colormap
        coll = _plot_linestring_collection(ax, self.lines, self.values, cmap="RdBu")
        fig.canvas.draw_idle()
        cmap = plt.get_cmap("RdBu")
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_color(), expected_colors)
        ax.cla()

        # specify vmin/vmax
        coll = _plot_linestring_collection(ax, self.lines, self.values, vmin=3, vmax=5)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        expected_colors = [cmap(0)]
        _check_colors(self.N, coll.get_color(), expected_colors * 3)
        ax.cla()

    def test_polygons(self):
        from matplotlib.collections import PatchCollection

        from geopandas.plotting import _plot_polygon_collection

        fig, ax = plt.subplots()
        coll = _plot_polygon_collection(ax, self.polygons)
        assert isinstance(coll, PatchCollection)
        ax.cla()

        # default: single default matplotlib color
        coll = _plot_polygon_collection(ax, self.polygons)
        _check_colors(self.N, coll.get_facecolor(), [MPL_DFT_COLOR] * self.N)
        assert len(coll.get_edgecolor()) == 0
        ax.cla()

        # default: color sets both facecolor and edgecolor
        coll = _plot_polygon_collection(ax, self.polygons, color="g")
        _check_colors(self.N, coll.get_facecolor(), ["g"] * self.N)
        _check_colors(self.N, coll.get_edgecolor(), ["g"] * self.N)
        ax.cla()

        # default: color can be passed as a list
        coll = _plot_polygon_collection(ax, self.polygons, color=["g", "b", "r"])
        _check_colors(self.N, coll.get_facecolor(), ["g", "b", "r"])
        _check_colors(self.N, coll.get_edgecolor(), ["g", "b", "r"])
        ax.cla()

        coll = _plot_polygon_collection(
            ax,
            self.polygons,
            color=[(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        _check_colors(
            self.N,
            coll.get_facecolor(),
            [(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        _check_colors(
            self.N,
            coll.get_edgecolor(),
            [(0.5, 0.5, 0.5, 0.5), (0.1, 0.2, 0.3, 0.5), (0.4, 0.5, 0.6, 0.5)],
        )
        ax.cla()

        # only setting facecolor keeps default for edgecolor
        coll = _plot_polygon_collection(ax, self.polygons, facecolor="g")
        _check_colors(self.N, coll.get_facecolor(), ["g"] * self.N)
        assert len(coll.get_edgecolor()) == 0
        ax.cla()

        # custom facecolor and edgecolor
        coll = _plot_polygon_collection(ax, self.polygons, facecolor="g", edgecolor="r")
        _check_colors(self.N, coll.get_facecolor(), ["g"] * self.N)
        _check_colors(self.N, coll.get_edgecolor(), ["r"] * self.N)
        ax.cla()

        # not a color
        with pytest.raises((TypeError, ValueError)):
            _plot_polygon_collection(ax, self.polygons, color="not color")

    def test_polygons_values(self):
        from geopandas.plotting import _plot_polygon_collection

        fig, ax = plt.subplots()

        # default colormap, edge is still black by default
        coll = _plot_polygon_collection(ax, self.polygons, self.values)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        exp_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_facecolor(), exp_colors)
        # edgecolor depends on matplotlib version
        # _check_colors(self.N, coll.get_edgecolor(), ['k'] * self.N)
        ax.cla()

        # specify colormap
        coll = _plot_polygon_collection(ax, self.polygons, self.values, cmap="RdBu")
        fig.canvas.draw_idle()
        cmap = plt.get_cmap("RdBu")
        exp_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_facecolor(), exp_colors)
        ax.cla()

        # specify vmin/vmax
        coll = _plot_polygon_collection(ax, self.polygons, self.values, vmin=3, vmax=5)
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        exp_colors = [cmap(0)]
        _check_colors(self.N, coll.get_facecolor(), exp_colors * 3)
        ax.cla()

        # override edgecolor
        coll = _plot_polygon_collection(ax, self.polygons, self.values, edgecolor="g")
        fig.canvas.draw_idle()
        cmap = plt.get_cmap()
        exp_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, coll.get_facecolor(), exp_colors)
        _check_colors(self.N, coll.get_edgecolor(), ["g"] * self.N)
        ax.cla()


class TestGeoplotAccessor:
    def setup_method(self):
        geometries = [Polygon([(0, 0), (1, 0), (1, 1)]), Point(1, 3)]
        x = [1, 2]
        y = [10, 20]
        self.gdf = GeoDataFrame(
            {"geometry": geometries, "x": x, "y": y}, crs="EPSG:4326"
        )
        self.df = pd.DataFrame({"x": x, "y": y})

    def compare_figures(self, kind, fig_test, fig_ref, kwargs):
        """Compare Figures."""
        ax_pandas_1 = fig_test.subplots()
        self.df.plot(kind=kind, ax=ax_pandas_1, **kwargs)
        ax_geopandas_1 = fig_ref.subplots()
        self.gdf.plot(kind=kind, ax=ax_geopandas_1, **kwargs)

        ax_pandas_2 = fig_test.subplots()
        getattr(self.df.plot, kind)(ax=ax_pandas_2, **kwargs)
        ax_geopandas_2 = fig_ref.subplots()
        getattr(self.gdf.plot, kind)(ax=ax_geopandas_2, **kwargs)

    _pandas_kinds = GeoplotAccessor._pandas_kinds

    if MPL_DECORATORS:

        @pytest.mark.parametrize("kind", _pandas_kinds)
        @check_figures_equal(extensions=["png", "pdf"])
        def test_pandas_kind(self, kind, fig_test, fig_ref):
            """Test Pandas kind."""
            import importlib

            _scipy_dependent_kinds = ["kde", "density"]  # Needs scipy
            _y_kinds = ["pie"]  # Needs y
            _xy_kinds = ["scatter", "hexbin"]  # Needs x & y
            kwargs = {}
            if kind in _scipy_dependent_kinds:
                if not importlib.util.find_spec("scipy"):
                    with pytest.raises(
                        ModuleNotFoundError, match="No module named 'scipy'"
                    ):
                        self.gdf.plot(kind=kind)
                    return
            elif kind in _y_kinds:
                kwargs = {"y": "y"}
            elif kind in _xy_kinds:
                kwargs = {"x": "x", "y": "y"}
                if kind == "hexbin":  # increase gridsize to reduce duration
                    kwargs["gridsize"] = 10

            self.compare_figures(kind, fig_test, fig_ref, kwargs)
            plt.close("all")

        @check_figures_equal(extensions=["png", "pdf"])
        def test_geo_kind(self, fig_test, fig_ref):
            """Test Geo kind."""
            ax1 = fig_test.subplots()
            self.gdf.plot(ax=ax1)
            ax2 = fig_ref.subplots()
            getattr(self.gdf.plot, "geo")(ax=ax2)
            plt.close("all")

    def test_invalid_kind(self):
        """Test invalid kinds."""
        with pytest.raises(ValueError, match="error is not a valid plot kind"):
            self.gdf.plot(kind="error")
        with pytest.raises(
            AttributeError,
            match="'GeoplotAccessor' object has no attribute 'error'",
        ):
            self.gdf.plot.error()


def test_column_values():
    """
    Check that the dataframe plot method returns same values with an
    input string (column in df), pd.Series, or np.array
    """
    # Build test data
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(1, 0), (2, 0), (2, 1)])
    polys = GeoSeries([t1, t2], index=list("AB"))
    df = GeoDataFrame({"geometry": polys, "values": [0, 1]})
    numeric_index_polys = GeoSeries([t1, t2], index=[0, 1])
    numeric_index_df = GeoDataFrame({"geometry": numeric_index_polys, "values": [0, 1]})

    # Test with continuous values
    ax = df.plot(column="values")
    colors = ax.collections[0].get_facecolors()
    ax = df.plot(column=df["values"])
    colors_series = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(colors, colors_series)
    ax = df.plot(column=df["values"].values)
    colors_array = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(colors, colors_array)

    # Test with categorical values
    ax = df.plot(column="values", categorical=True)
    colors = ax.collections[0].get_facecolors()
    ax = df.plot(column=df["values"], categorical=True)
    colors_series = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(colors, colors_series)
    ax = df.plot(column=df["values"].values, categorical=True)
    colors_array = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(colors, colors_array)

    # Test with pd.Index
    ax = numeric_index_df.plot(column=numeric_index_df.index, categorical=True)
    colors_array = ax.collections[0].get_facecolors()
    np.testing.assert_array_equal(colors, colors_array)

    # Check raised error: is df rows number equal to column length?
    with pytest.raises(ValueError, match="different number of rows"):
        ax = df.plot(column=np.array([1, 2, 3]))


def test_polygon_patch():
    # test adapted from descartes by Sean Gillies
    # (BSD license, https://pypi.org/project/descartes).
    from matplotlib.patches import PathPatch

    from geopandas.plotting import _PolygonPatch

    polygon = (
        Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    )

    patch = _PolygonPatch(polygon)
    assert isinstance(patch, PathPatch)
    path = patch.get_path()
    if compat.GEOS_GE_390:
        assert len(path.vertices) == len(path.codes) == 195
    else:
        assert len(path.vertices) == len(path.codes) == 198


def _check_colors(N, actual_colors, expected_colors, alpha=None):
    """
    Asserts that the members of `collection` match the `expected_colors`
    (in order)

    Parameters
    ----------
    N : int
        The number of geometries believed to be in collection.
        matplotlib.collection is implemented such that the number of geoms in
        `collection` doesn't have to match the number of colors assignments in
        the collection: the colors will cycle to meet the needs of the geoms.
        `N` helps us resolve this.
    collection : matplotlib.collections.Collection
        The colors of this collection's patches are read from
        `collection.get_facecolors()`
    expected_colors : sequence of RGBA tuples
    alpha : float (optional)
        If set, this alpha transparency will be applied to the
        `expected_colors`. (Any transparency on the `collection` is assumed
        to be set in its own facecolor RGBA tuples.)
    """
    from matplotlib import colors

    conv = colors.colorConverter

    # Convert 2D numpy array to a list of RGBA tuples.
    actual_colors = map(tuple, actual_colors)
    all_actual_colors = list(itertools.islice(itertools.cycle(actual_colors), N))

    assert len(all_actual_colors) == len(expected_colors), (
        "Different lengths of actual and expected colors!"
    )

    for actual, expected in zip(all_actual_colors, expected_colors):
        assert actual == conv.to_rgba(expected, alpha=alpha), (
            f"{actual} != {conv.to_rgba(expected, alpha=alpha)}"
        )


def _style_to_linestring_onoffseq(linestyle, linewidth):
    """Converts a linestyle string representation, namely one of:
        ['dashed',  'dotted', 'dashdot', 'solid'],
    documented in `Collections.set_linestyle`,
    to the form `onoffseq`.
    """
    offset, dashes = matplotlib.lines._get_dash_pattern(linestyle)
    return matplotlib.lines._scale_dashes(offset, dashes, linewidth)


def _style_to_vertices(markerstyle):
    """Converts a markerstyle string to a path."""
    # TODO: Vertices values are twice the actual path; unclear, why.
    path = matplotlib.markers.MarkerStyle(markerstyle).get_path()
    return path.vertices / 2


def _get_ax(fig, label):
    """
    Helper function to not rely on the order of `fig.axes`.
    Previously, we did `fig.axes[1]`, but in matplotlib 3.4 the order switched
    and the colorbar ax was first and subplot ax second.
    """
    for ax in fig.axes:
        if ax.get_label() == label:
            return ax
    raise ValueError(f"no ax found with label {label}")


def _get_colorbar_ax(fig):
    return _get_ax(fig, "<colorbar>")
