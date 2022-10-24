import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import warnings

folium = pytest.importorskip("folium")
branca = pytest.importorskip("branca")
matplotlib = pytest.importorskip("matplotlib")
mapclassify = pytest.importorskip("mapclassify")

import matplotlib.cm as cm  # noqa
import matplotlib.colors as colors  # noqa
from branca.colormap import StepColormap  # noqa
import branca as bc  # noqa

BRANCA_05 = Version(branca.__version__) > Version("0.4.2")


class TestExplore:
    def setup_method(self):
        self.nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
        self.world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        self.cities = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
        self.world["range"] = range(len(self.world))
        self.missing = self.world.copy()
        np.random.seed(42)
        self.missing.loc[np.random.choice(self.missing.index, 40), "continent"] = np.nan
        self.missing.loc[np.random.choice(self.missing.index, 40), "pop_est"] = np.nan

    def _fetch_map_string(self, m):
        out = m._parent.render()
        out_str = "".join(out.split())
        return out_str

    def test_simple_pass(self):
        """Make sure default pass"""
        self.nybb.explore()
        self.world.explore()
        self.cities.explore()
        self.world.geometry.explore()

    def test_dependencies(self):
        from unittest import mock
        import sys

        with mock.patch.dict(sys.modules):
            sys.modules["folium"] = None
            with pytest.raises(
                ImportError,
                match=r"^The 'folium', 'matplotlib' and 'mapclassify' packages.*",
            ):
                self.nybb.explore()

        with mock.patch.dict(sys.modules):
            sys.modules["xyzservices"] = None
            self.nybb.explore()

    def test_choropleth_pass(self):
        """Make sure default choropleth pass"""
        self.world.explore(column="pop_est")

    def test_map_settings_default(self):
        """Check default map settings"""
        m = self.world.explore()
        assert m.location == [
            pytest.approx(-3.1774349999999956, rel=1e-6),
            pytest.approx(2.842170943040401e-14, rel=1e-6),
        ]
        assert m.options["zoom"] == 10
        assert m.options["zoomControl"] is True
        assert m.position == "relative"
        assert m.height == (100.0, "%")
        assert m.width == (100.0, "%")
        assert m.left == (0, "%")
        assert m.top == (0, "%")
        assert m.global_switches.no_touch is False
        assert m.global_switches.disable_3d is False
        assert "openstreetmap" in m.to_dict()["children"].keys()

    def test_map_settings_custom(self):
        """Check custom map settings"""
        m = self.nybb.explore(
            zoom_control=False,
            width=200,
            height=200,
        )
        assert m.location == [
            pytest.approx(40.70582377450201, rel=1e-6),
            pytest.approx(-73.9778006856748, rel=1e-6),
        ]
        assert m.options["zoom"] == 10
        assert m.options["zoomControl"] is False
        assert m.height == (200.0, "px")
        assert m.width == (200.0, "px")

        # custom XYZ tiles
        m = self.nybb.explore(
            zoom_control=False,
            width=200,
            height=200,
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google",
        )

        out_str = self._fetch_map_string(m)
        s = '"https://mt1.google.com/vt/lyrs=m\\u0026x={x}\\u0026y={y}\\u0026z={z}"'
        assert s in out_str
        assert '"attribution":"Google"' in out_str

        m = self.nybb.explore(location=(40, 5))
        assert m.location == [40, 5]
        assert m.options["zoom"] == 10

        m = self.nybb.explore(zoom_start=8)
        assert m.location == [
            pytest.approx(40.70582377450201, rel=1e-6),
            pytest.approx(-73.9778006856748, rel=1e-6),
        ]
        assert m.options["zoom"] == 8

        m = self.nybb.explore(location=(40, 5), zoom_start=8)
        assert m.location == [40, 5]
        assert m.options["zoom"] == 8

    def test_simple_color(self):
        """Check color settings"""
        # single named color
        m = self.nybb.explore(color="red")
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"red"' in out_str

        # list of colors
        colors = ["#333333", "#367324", "#95824f", "#fcaa00", "#ffcc33"]
        m2 = self.nybb.explore(color=colors)
        out_str = self._fetch_map_string(m2)
        for c in colors:
            assert f'"fillColor":"{c}"' in out_str

        # column of colors
        df = self.nybb.copy()
        df["colors"] = colors
        m3 = df.explore(color="colors")
        out_str = self._fetch_map_string(m3)
        for c in colors:
            assert f'"fillColor":"{c}"' in out_str

        # line GeoSeries
        m4 = self.nybb.boundary.explore(color="red")
        out_str = self._fetch_map_string(m4)
        assert '"fillColor":"red"' in out_str

    def test_choropleth_linear(self):
        """Check choropleth colors"""
        # default cmap
        m = self.nybb.explore(column="Shape_Leng")
        out_str = self._fetch_map_string(m)
        assert 'color":"#440154"' in out_str
        assert 'color":"#fde725"' in out_str
        assert 'color":"#50c46a"' in out_str
        assert 'color":"#481467"' in out_str
        assert 'color":"#3d4e8a"' in out_str

        # named cmap
        m = self.nybb.explore(column="Shape_Leng", cmap="PuRd")
        out_str = self._fetch_map_string(m)
        assert 'color":"#f7f4f9"' in out_str
        assert 'color":"#67001f"' in out_str
        assert 'color":"#d31760"' in out_str
        assert 'color":"#f0ecf5"' in out_str
        assert 'color":"#d6bedc"' in out_str

    def test_choropleth_mapclassify(self):
        """Mapclassify bins"""
        # quantiles
        m = self.nybb.explore(column="Shape_Leng", scheme="quantiles")
        out_str = self._fetch_map_string(m)
        assert 'color":"#21918c"' in out_str
        assert 'color":"#3b528b"' in out_str
        assert 'color":"#5ec962"' in out_str
        assert 'color":"#fde725"' in out_str
        assert 'color":"#440154"' in out_str

        # headtail
        m = self.world.explore(column="pop_est", scheme="headtailbreaks")
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"#3b528b"' in out_str
        assert '"fillColor":"#21918c"' in out_str
        assert '"fillColor":"#5ec962"' in out_str
        assert '"fillColor":"#fde725"' in out_str
        assert '"fillColor":"#440154"' in out_str
        # custom k
        m = self.world.explore(column="pop_est", scheme="naturalbreaks", k=3)
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"#21918c"' in out_str
        assert '"fillColor":"#fde725"' in out_str
        assert '"fillColor":"#440154"' in out_str

    def test_categorical(self):
        """Categorical maps"""
        # auto detection
        m = self.world.explore(column="continent")
        out_str = self._fetch_map_string(m)
        assert 'color":"#9467bd","continent":"Europe"' in out_str
        assert 'color":"#c49c94","continent":"NorthAmerica"' in out_str
        assert 'color":"#1f77b4","continent":"Africa"' in out_str
        assert 'color":"#98df8a","continent":"Asia"' in out_str
        assert 'color":"#ff7f0e","continent":"Antarctica"' in out_str
        assert 'color":"#9edae5","continent":"SouthAmerica"' in out_str
        assert 'color":"#7f7f7f","continent":"Oceania"' in out_str
        assert 'color":"#dbdb8d","continent":"Sevenseas(openocean)"' in out_str

        # forced categorical
        m = self.nybb.explore(column="BoroCode", categorical=True)
        out_str = self._fetch_map_string(m)
        assert 'color":"#9edae5"' in out_str
        assert 'color":"#c7c7c7"' in out_str
        assert 'color":"#8c564b"' in out_str
        assert 'color":"#1f77b4"' in out_str
        assert 'color":"#98df8a"' in out_str

        # pandas.Categorical
        df = self.world.copy()
        df["categorical"] = pd.Categorical(df["name"])
        m = df.explore(column="categorical")
        out_str = self._fetch_map_string(m)
        for c in np.apply_along_axis(colors.to_hex, 1, cm.tab20(range(20))):
            assert f'"fillColor":"{c}"' in out_str

        # custom cmap
        m = self.nybb.explore(column="BoroName", cmap="Set1")
        out_str = self._fetch_map_string(m)
        assert 'color":"#999999"' in out_str
        assert 'color":"#a65628"' in out_str
        assert 'color":"#4daf4a"' in out_str
        assert 'color":"#e41a1c"' in out_str
        assert 'color":"#ff7f00"' in out_str

        # custom list of colors
        cmap = ["#333432", "#3b6e8c", "#bc5b4f", "#8fa37e", "#efc758"]
        m = self.nybb.explore(column="BoroName", cmap=cmap)
        out_str = self._fetch_map_string(m)
        for c in cmap:
            assert f'"fillColor":"{c}"' in out_str

        # shorter list (to make it repeat)
        cmap = ["#333432", "#3b6e8c"]
        m = self.nybb.explore(column="BoroName", cmap=cmap)
        out_str = self._fetch_map_string(m)
        for c in cmap:
            assert f'"fillColor":"{c}"' in out_str

        with pytest.raises(ValueError, match="`cmap` is not known matplotlib colormap"):
            self.nybb.explore(column="BoroName", cmap="nonsense")

    def test_categories(self):
        m = self.nybb[["BoroName", "geometry"]].explore(
            column="BoroName",
            categories=["Brooklyn", "Staten Island", "Queens", "Bronx", "Manhattan"],
        )
        out_str = self._fetch_map_string(m)
        assert '"Bronx","__folium_color":"#c7c7c7"' in out_str
        assert '"Manhattan","__folium_color":"#9edae5"' in out_str
        assert '"Brooklyn","__folium_color":"#1f77b4"' in out_str
        assert '"StatenIsland","__folium_color":"#98df8a"' in out_str
        assert '"Queens","__folium_color":"#8c564b"' in out_str

        df = self.nybb.copy()
        df["categorical"] = pd.Categorical(df["BoroName"])
        with pytest.raises(ValueError, match="Cannot specify 'categories'"):
            df.explore("categorical", categories=["Brooklyn", "Staten Island"])

    def test_bool(self):
        df = self.nybb.copy()
        df["bool"] = [True, False, True, False, True]
        df["bool_extension"] = pd.array([True, False, True, False, True])
        m1 = df.explore("bool")
        m2 = df.explore("bool_extension")

        out1_str = self._fetch_map_string(m1)
        assert '"__folium_color":"#9edae5","bool":true' in out1_str
        assert '"__folium_color":"#1f77b4","bool":false' in out1_str

        out2_str = self._fetch_map_string(m2)
        assert '"__folium_color":"#9edae5","bool":true' in out2_str
        assert '"__folium_color":"#1f77b4","bool":false' in out2_str

    def test_string(self):
        df = self.nybb.copy()
        df["string"] = pd.array([1, 2, 3, 4, 5], dtype="string")
        m = df.explore("string")
        out_str = self._fetch_map_string(m)
        assert '"__folium_color":"#9edae5","string":"5"' in out_str

    def test_column_values(self):
        """
        Check that the dataframe plot method returns same values with an
        input string (column in df), pd.Series, or np.array
        """
        column_array = np.array(self.world["pop_est"])
        m1 = self.world.explore(column="pop_est")  # column name
        m2 = self.world.explore(column=column_array)  # np.array
        m3 = self.world.explore(column=self.world["pop_est"])  # pd.Series
        assert m1.location == m2.location == m3.location

        m1_fields = self.world.explore(column=column_array, tooltip=True, popup=True)
        out1_fields_str = self._fetch_map_string(m1_fields)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out1_fields_str
        )
        assert (
            'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out1_fields_str
        )

        m2_fields = self.world.explore(
            column=self.world["pop_est"], tooltip=True, popup=True
        )
        out2_fields_str = self._fetch_map_string(m2_fields)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out2_fields_str
        )
        assert (
            'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out2_fields_str
        )

        # GeoDataframe and the given list have different number of rows
        with pytest.raises(ValueError, match="different number of rows"):
            self.world.explore(column=np.array([1, 2, 3]))

    def test_no_crs(self):
        """Naive geometry get no tiles"""
        df = self.world.copy()
        df.crs = None
        m = df.explore()
        assert "openstreetmap" not in m.to_dict()["children"].keys()

    def test_style_kwds(self):
        """Style keywords"""
        m = self.world.explore(
            style_kwds=dict(fillOpacity=0.1, weight=0.5, fillColor="orange")
        )
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"orange","fillOpacity":0.1,"weight":0.5' in out_str
        m = self.world.explore(column="pop_est", style_kwds=dict(color="black"))
        assert '"color":"black"' in self._fetch_map_string(m)

        # custom style_function - geopandas/issues/2350
        m = self.world.explore(
            style_kwds={
                "style_function": lambda x: {
                    "fillColor": "red"
                    if x["properties"]["gdp_md_est"] < 10**6
                    else "green",
                    "color": "black"
                    if x["properties"]["gdp_md_est"] < 10**6
                    else "white",
                }
            }
        )
        # two lines with formatting instructions from style_function.
        # make sure each passes test
        assert all(
            [
                ('"fillColor":"green"' in t and '"color":"white"' in t)
                or ('"fillColor":"red"' in t and '"color":"black"' in t)
                for t in [
                    "".join(line.split())
                    for line in m._parent.render().split("\n")
                    if "return" in line and "color" in line
                ]
            ]
        )

        # style function has to be callable
        with pytest.raises(ValueError, match="'style_function' has to be a callable"):
            self.world.explore(style_kwds={"style_function": "not callable"})

    def test_tooltip(self):
        """Test tooltip"""
        # default with no tooltip or popup
        m = self.world.explore()
        assert "GeoJsonTooltip" in str(m.to_dict())
        assert "GeoJsonPopup" not in str(m.to_dict())

        # True
        m = self.world.explore(tooltip=True, popup=True)
        assert "GeoJsonTooltip" in str(m.to_dict())
        assert "GeoJsonPopup" in str(m.to_dict())
        out_str = self._fetch_map_string(m)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )
        assert (
            'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )

        # True choropleth
        m = self.world.explore(column="pop_est", tooltip=True, popup=True)
        assert "GeoJsonTooltip" in str(m.to_dict())
        assert "GeoJsonPopup" in str(m.to_dict())
        out_str = self._fetch_map_string(m)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )
        assert (
            'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )

        # single column
        m = self.world.explore(tooltip="pop_est", popup="iso_a3")
        out_str = self._fetch_map_string(m)
        assert 'fields=["pop_est"]' in out_str
        assert 'aliases=["pop_est"]' in out_str
        assert 'fields=["iso_a3"]' in out_str
        assert 'aliases=["iso_a3"]' in out_str

        # list
        m = self.world.explore(
            tooltip=["pop_est", "continent"], popup=["iso_a3", "gdp_md_est"]
        )
        out_str = self._fetch_map_string(m)
        assert 'fields=["pop_est","continent"]' in out_str
        assert 'aliases=["pop_est","continent"]' in out_str
        assert 'fields=["iso_a3","gdp_md_est"' in out_str
        assert 'aliases=["iso_a3","gdp_md_est"]' in out_str

        # number
        m = self.world.explore(tooltip=2, popup=2)
        out_str = self._fetch_map_string(m)
        assert 'fields=["pop_est","continent"]' in out_str
        assert 'aliases=["pop_est","continent"]' in out_str

        # keywords tooltip
        m = self.world.explore(
            tooltip=True,
            popup=False,
            tooltip_kwds=dict(aliases=[0, 1, 2, 3, 4, 5], sticky=False),
        )
        out_str = self._fetch_map_string(m)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )
        assert "aliases=[0,1,2,3,4,5]" in out_str
        assert '"sticky":false' in out_str

        # keywords popup
        m = self.world.explore(
            tooltip=False,
            popup=True,
            popup_kwds=dict(aliases=[0, 1, 2, 3, 4, 5]),
        )
        out_str = self._fetch_map_string(m)
        assert (
            'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
            in out_str
        )
        assert "aliases=[0,1,2,3,4,5]" in out_str
        assert "<th>${aliases[i]" in out_str

        # no labels
        m = self.world.explore(
            tooltip=True,
            popup=True,
            tooltip_kwds=dict(labels=False),
            popup_kwds=dict(labels=False),
        )
        out_str = self._fetch_map_string(m)
        assert "<th>${aliases[i]" not in out_str

        # named index
        gdf = self.nybb.set_index("BoroName")
        m = gdf.explore()
        out_str = self._fetch_map_string(m)
        assert "BoroName" in out_str

    def test_default_markers(self):
        # check overridden default for points
        m = self.cities.explore()
        strings = ['"radius":2', '"fill":true', "CircleMarker(latlng,opts)"]
        out_str = self._fetch_map_string(m)
        for s in strings:
            assert s in out_str

        m = self.cities.explore(marker_kwds=dict(radius=5, fill=False))
        strings = ['"radius":5', '"fill":false', "CircleMarker(latlng,opts)"]
        out_str = self._fetch_map_string(m)
        for s in strings:
            assert s in out_str

    def test_custom_markers(self):
        # Markers
        m = self.cities.explore(
            marker_type="marker",
            marker_kwds={"icon": folium.Icon(icon="star")},
        )
        assert ""","icon":"star",""" in self._fetch_map_string(m)

        # Circle Markers
        m = self.cities.explore(marker_type="circle", marker_kwds={"fill_color": "red"})
        assert ""","fillColor":"red",""" in self._fetch_map_string(m)

        # Folium Markers
        m = self.cities.explore(
            marker_type=folium.Circle(
                radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1
            ),
        )
        assert ""","color":"black",""" in self._fetch_map_string(m)

        # Circle
        m = self.cities.explore(marker_type="circle_marker", marker_kwds={"radius": 10})
        assert ""","radius":10,""" in self._fetch_map_string(m)

        # Unsupported Markers
        with pytest.raises(
            ValueError,
            match="Only 'marker', 'circle', and 'circle_marker' are supported",
        ):
            self.cities.explore(marker_type="dummy")

    def test_vmin_vmax(self):
        df = self.world.copy()
        df["range"] = range(len(df))
        m = df.explore("range", vmin=-100, vmax=1000)
        out_str = self._fetch_map_string(m)
        assert 'case"176":return{"color":"#3b528b","fillColor":"#3b528b"' in out_str
        assert 'case"119":return{"color":"#414287","fillColor":"#414287"' in out_str
        assert 'case"3":return{"color":"#482173","fillColor":"#482173"' in out_str

        # test 0
        df2 = self.nybb.copy()
        df2["values"] = df2["BoroCode"] * 10.0
        m = df2[df2["values"] >= 30].explore("values", vmin=0)
        out_str = self._fetch_map_string(m)
        assert 'case"1":return{"color":"#7ad151","fillColor":"#7ad151"' in out_str
        assert 'case"2":return{"color":"#22a884","fillColor":"#22a884"' in out_str

        df2["values_negative"] = df2["BoroCode"] * -10.0
        m = df2[df2["values_negative"] <= 30].explore("values_negative", vmax=0)
        out_str = self._fetch_map_string(m)
        assert 'case"1":return{"color":"#414487","fillColor":"#414487"' in out_str
        assert 'case"2":return{"color":"#2a788e","fillColor":"#2a788e"' in out_str

    def test_missing_vals(self):
        m = self.missing.explore("continent")
        assert '"fillColor":null' in self._fetch_map_string(m)

        m = self.missing.explore("pop_est")
        assert '"fillColor":null' in self._fetch_map_string(m)

        m = self.missing.explore("pop_est", missing_kwds=dict(color="red"))
        assert '"fillColor":"red"' in self._fetch_map_string(m)

        m = self.missing.explore("continent", missing_kwds=dict(color="red"))
        assert '"fillColor":"red"' in self._fetch_map_string(m)

        m = self.missing.explore(
            self.missing["continent"].str[0], cmap=["red", "green"]
        )
        assert '"fillColor":null' in self._fetch_map_string(m)

        m = self.missing.explore(
            "pop_est", cmap=lambda x: "red" if x < 10**7 else "green", legend=False
        )
        assert '"fillColor":null' in self._fetch_map_string(m)

    def test_categorical_legend(self):
        m = self.world.explore("continent", legend=True)
        out_str = self._fetch_map_string(m)
        assert "#1f77b4'></span>Africa" in out_str
        assert "#ff7f0e'></span>Antarctica" in out_str
        assert "#98df8a'></span>Asia" in out_str
        assert "#9467bd'></span>Europe" in out_str
        assert "#c49c94'></span>NorthAmerica" in out_str
        assert "#7f7f7f'></span>Oceania" in out_str
        assert "#dbdb8d'></span>Sevenseas(openocean)" in out_str
        assert "#9edae5'></span>SouthAmerica" in out_str

        m = self.missing.explore(
            "continent", legend=True, missing_kwds={"color": "red"}
        )
        out_str = self._fetch_map_string(m)
        assert "red'></span>NaN" in out_str

    def test_colorbar(self):
        m = self.world.explore("range", legend=True)
        out_str = self._fetch_map_string(m)
        assert "attr(\"id\",'legend')" in out_str
        assert "text('range')" in out_str

        m = self.world.explore(
            "range", legend=True, legend_kwds=dict(caption="my_caption")
        )
        out_str = self._fetch_map_string(m)
        assert "attr(\"id\",'legend')" in out_str
        assert "text('my_caption')" in out_str

        m = self.missing.explore("pop_est", legend=True, missing_kwds=dict(color="red"))
        out_str = self._fetch_map_string(m)
        assert "red'></span>NaN" in out_str

        # do not scale legend
        m = self.world.explore(
            "pop_est",
            legend=True,
            legend_kwds=dict(scale=False),
            scheme="Headtailbreaks",
        )
        out_str = self._fetch_map_string(m)
        assert out_str.count("#440154ff") == 100
        assert out_str.count("#3b528bff") == 100
        assert out_str.count("#21918cff") == 100
        assert out_str.count("#5ec962ff") == 100
        assert out_str.count("#fde725ff") == 100

        # scale legend accordingly
        m = self.world.explore(
            "pop_est",
            legend=True,
            scheme="Headtailbreaks",
        )
        out_str = self._fetch_map_string(m)
        assert out_str.count("#440154ff") == 16
        assert out_str.count("#3b528bff") == 50
        assert out_str.count("#21918cff") == 138
        assert out_str.count("#5ec962ff") == 290
        assert out_str.count("#fde725ff") == 6

        # discrete cmap
        m = self.world.explore("pop_est", legend=True, cmap="Pastel2")
        out_str = self._fetch_map_string(m)

        assert out_str.count("b3e2cdff") == 63
        assert out_str.count("fdcdacff") == 62
        assert out_str.count("cbd5e8ff") == 63
        assert out_str.count("f4cae4ff") == 62
        assert out_str.count("e6f5c9ff") == 62
        assert out_str.count("fff2aeff") == 63
        assert out_str.count("f1e2ccff") == 62
        assert out_str.count("ccccccff") == 63

    @pytest.mark.skipif(not BRANCA_05, reason="requires branca >= 0.5.0")
    def test_colorbar_max_labels(self):
        import re

        # linear
        m = self.world.explore("pop_est", legend_kwds=dict(max_labels=3))
        out_str = self._fetch_map_string(m)
        tick_str = re.search(r"tickValues\(\[[\',\,\.,0-9]*\]\)", out_str).group(0)
        assert (
            tick_str.replace(",''", "")
            == "tickValues([140.0,471386328.07843137,942772516.1568627])"
        )

        # scheme
        m = self.world.explore(
            "pop_est", scheme="headtailbreaks", legend_kwds=dict(max_labels=3)
        )
        out_str = self._fetch_map_string(m)
        assert "tickValues([140.0,'',184117213.1818182,'',1382066377.0,''])" in out_str

        # short cmap
        m = self.world.explore("pop_est", legend_kwds=dict(max_labels=3), cmap="tab10")
        out_str = self._fetch_map_string(m)

        tick_str = re.search(r"tickValues\(\[[\',\,\.,0-9]*\]\)", out_str).group(0)
        assert (
            tick_str
            == "tickValues([140.0,'','','',559086084.0,'','','',1118172028.0,'','',''])"
        )

    def test_xyzservices_providers(self):
        xyzservices = pytest.importorskip("xyzservices")

        m = self.nybb.explore(tiles=xyzservices.providers.CartoDB.PositronNoLabels)
        out_str = self._fetch_map_string(m)

        assert (
            '"https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"'
            in out_str
        )
        assert (
            'attribution":"\\u0026copy;\\u003cahref=\\"https://www.openstreetmap.org'
            in out_str
        )
        assert '"maxNativeZoom":20,"maxZoom":20,"minZoom":0' in out_str

    def test_xyzservices_query_name(self):
        pytest.importorskip("xyzservices")

        m = self.nybb.explore(tiles="CartoDB Positron No Labels")
        out_str = self._fetch_map_string(m)

        assert (
            '"https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"'
            in out_str
        )
        assert (
            'attribution":"\\u0026copy;\\u003cahref=\\"https://www.openstreetmap.org'
            in out_str
        )
        assert '"maxNativeZoom":20,"maxZoom":20,"minZoom":0' in out_str

    def test_linearrings(self):
        rings = self.nybb.explode(index_parts=True).exterior
        m = rings.explore()
        out_str = self._fetch_map_string(m)

        assert out_str.count("LineString") == len(rings)

    def test_mapclassify_categorical_legend(self):
        m = self.missing.explore(
            column="pop_est",
            legend=True,
            scheme="naturalbreaks",
            missing_kwds=dict(color="red", label="missing"),
            legend_kwds=dict(colorbar=False, interval=True),
        )
        out_str = self._fetch_map_string(m)

        strings = [
            "[140.00,21803000.00]",
            "(21803000.00,66834405.00]",
            "(66834405.00,163046161.00]",
            "(163046161.00,328239523.00]",
            "(328239523.00,1397715000.00]",
            "missing",
        ]
        for s in strings:
            assert s in out_str

        # interval=False
        m = self.missing.explore(
            column="pop_est",
            legend=True,
            scheme="naturalbreaks",
            missing_kwds=dict(color="red", label="missing"),
            legend_kwds=dict(colorbar=False, interval=False),
        )
        out_str = self._fetch_map_string(m)

        strings = [
            ">140.00,21803000.00",
            ">21803000.00,66834405.00",
            ">66834405.00,163046161.00",
            ">163046161.00,328239523.00",
            ">328239523.00,1397715000.00",
            "missing",
        ]
        for s in strings:
            assert s in out_str

        # custom labels
        m = self.world.explore(
            column="pop_est",
            legend=True,
            scheme="naturalbreaks",
            k=5,
            legend_kwds=dict(colorbar=False, labels=["s", "m", "l", "xl", "xxl"]),
        )
        out_str = self._fetch_map_string(m)

        strings = [">s<", ">m<", ">l<", ">xl<", ">xxl<"]
        for s in strings:
            assert s in out_str

        # fmt
        m = self.missing.explore(
            column="pop_est",
            legend=True,
            scheme="naturalbreaks",
            missing_kwds=dict(color="red", label="missing"),
            legend_kwds=dict(colorbar=False, fmt="{:.0f}"),
        )
        out_str = self._fetch_map_string(m)

        strings = [
            ">140,21803000",
            ">21803000,66834405",
            ">66834405,163046161",
            ">163046161,328239523",
            ">328239523,1397715000",
            "missing",
        ]
        for s in strings:
            assert s in out_str

    def test_given_m(self):
        "Check that geometry is mapped onto a given folium.Map"
        m = folium.Map()
        self.nybb.explore(m=m, tooltip=False, highlight=False)

        out_str = self._fetch_map_string(m)

        assert out_str.count("BoroCode") == 5
        # should not change map settings
        assert m.options["zoom"] == 1

    def test_highlight(self):
        m = self.nybb.explore(highlight=True)
        out_str = self._fetch_map_string(m)

        assert '"fillOpacity":0.75' in out_str

        m = self.nybb.explore(
            highlight=True, highlight_kwds=dict(fillOpacity=1, color="red")
        )
        out_str = self._fetch_map_string(m)

        assert '{"color":"red","fillOpacity":1}' in out_str

    def test_custom_colormaps(self):

        step = StepColormap(["green", "yellow", "red"], vmin=0, vmax=100000000)

        m = self.world.explore("pop_est", cmap=step, tooltip=["name"], legend=True)

        strings = [
            'fillColor":"#008000ff"',  # Green
            '"fillColor":"#ffff00ff"',  # Yellow
            '"fillColor":"#ff0000ff"',  # Red
        ]

        out_str = self._fetch_map_string(m)
        for s in strings:
            assert s in out_str

        assert out_str.count("008000ff") == 304
        assert out_str.count("ffff00ff") == 188
        assert out_str.count("ff0000ff") == 191

        # Using custom function colormap
        def my_color_function(field):
            """Maps low values to green and high values to red."""
            if field > 100000000:
                return "#ff0000"
            else:
                return "#008000"

        m = self.world.explore("pop_est", cmap=my_color_function, legend=False)

        strings = [
            '"color":"#ff0000","fillColor":"#ff0000"',
            '"color":"#008000","fillColor":"#008000"',
        ]

        for s in strings:
            assert s in self._fetch_map_string(m)

        # matplotlib.Colormap
        cmap = colors.ListedColormap(["red", "green", "blue", "white", "black"])

        m = self.nybb.explore("BoroName", cmap=cmap)
        strings = [
            '"fillColor":"#ff0000"',  # Red
            '"fillColor":"#008000"',  # Green
            '"fillColor":"#0000ff"',  # Blue
            '"fillColor":"#ffffff"',  # White
            '"fillColor":"#000000"',  # Black
        ]

        out_str = self._fetch_map_string(m)
        for s in strings:
            assert s in out_str

    def test_multiple_geoseries(self):
        """
        Additional GeoSeries need to be removed as they cannot be converted to GeoJSON
        """
        gdf = self.nybb
        gdf["boundary"] = gdf.boundary
        gdf["centroid"] = gdf.centroid

        gdf.explore()

    def test_map_kwds(self):
        def check():
            out_str = self._fetch_map_string(m)
            assert "zoomControl:false" in out_str
            assert "dragging:false" in out_str
            assert "scrollWheelZoom:false" in out_str

        # check that folium and leaflet Map() parameters can be passed
        m = self.world.explore(
            zoom_control=False, map_kwds=dict(dragging=False, scrollWheelZoom=False)
        )
        check()
        with pytest.raises(
            ValueError, match="'zoom_control' cannot be specified in 'map_kwds'"
        ):
            self.world.explore(
                map_kwds=dict(dragging=False, scrollWheelZoom=False, zoom_control=False)
            )

    def test_robust(self):
        # this is really only for codecov
        cm = gpd.explore._binning_cmap(
            "Purples", 5, Version(matplotlib.__version__) >= Version("3.6.1")
        )
        assert [colors.to_hex(c) for c in cm(range(cm.N))] == [
            "#fcfbfd",
            "#dadaeb",
            "#9e9ac8",
            "#6a51a3",
            "#3f007d",
        ]
        with pytest.raises(ValueError, match=r".*in this call context$"):
            gpd.explore._binning_cmap(["red", "green"], 5)

        # main test, just use 25 geometries to speed up the test
        robust = RobustHelper(self.world.copy().head(25))
        # 100 still gets 100% codecov
        df = robust.generate(max_tests=100, exclude=[])
        df, maps = robust.run(df)
        robust.expected_result(df)
        # make sure PEP8 formating hasn't broken util function
        robust.html(df[df["__error"].isna()].head(2), maps)


class RobustHelper:
    """Helper for testing permuations of parameters to `explore()`

    Can be used in a JupyterLab notebook::
        import geopandas as gpd
        from geopandas.tests.test_explore import RobustHelper
        from IPython.display import display, HTML

        gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        robust = RobustHelper(gdf)
        df = robust.generate(max_tests=200, exclude=[])
        df, maps = robust.run(df)
        print({c: (~df[c].isna()).sum() for c in
        ["__column_type", "__error", "__warning"]})
        display(HTML(robust.html(df, maps)))
    """

    def __init__(self, gdf, col="pop_est", alpha_col="iso_a3"):
        self.gdf = gdf.copy()
        self.col = "__col"
        self.catcol = "__cat"
        self.alphacol = "__alpha"

        # create columns that drive testing
        self.gdf[self.col] = np.log1p(gdf[col])
        self.gdf[self.catcol] = pd.cut(self.gdf[self.col], bins=5, labels=range(5))
        self.gdf[self.alphacol] = self.gdf[alpha_col].str[1:2]

        # quite a few bugs with nan handling....
        self.gdf.loc[
            self.gdf.sample(int(len(self.gdf) * 0.2), random_state=42).index,
            [self.col, self.catcol, self.alphacol],
        ] = np.nan

        self.plot_opts = dict(
            height=200,
            width=250,
        )

    def generate(
        self,
        max_tests=100,
        exclude=[],
    ):
        import itertools

        gdf = self.gdf

        # colormap list like building blocks
        cl = ["red", "blue", "pink", "yellow", "white"]
        clrgb = [colors.to_rgb(c) for c in cl]
        cs = gdf[self.catcol].replace({i: c for i, c in enumerate(cl)})
        csrgb = gdf[self.catcol].astype(float).map({i: c for i, c in enumerate(clrgb)})
        clb = cl.copy()
        clb[1] = "bluee"
        csb = gdf[self.catcol].replace({i: c for i, c in enumerate(clb)})

        def create_type(t, v):
            if hasattr(v, "tolist"):
                v = v.tolist()
            if t.__name__ == "DataFrame":
                return pd.DataFrame({"c": v})
            elif t.__name__ == "array":
                return np.array(v, dtype=object)
            return t(v)

        cm_c = ["#e1f2f2", "#A8DCDC", "#115F5F"]
        cm_l = [
            None,
            "Purples",
            "bad",
            colors.LinearSegmentedColormap.from_list("custom", cm_c),
            lambda x: "red" if x < 16 else "green",
            bc.colormap.LinearColormap(
                cm_c,
                vmin=5,
                vmax=20,
            ),
            bc.colormap.StepColormap(cl, vmin=13, vmax=20),
        ]

        # generate invalid length array for column arg
        def bad(c):
            return gdf[c].values[1:]

        test_cases = {
            "batch1": {
                "filter": lambda d: d.index == d.index,
                "columns": {
                    "column": [
                        {
                            "__column_type": c + f.__name__,
                            "column": f(c),
                            "__expect_error": f is bad,
                        }
                        for c, f in itertools.product(
                            [self.col, self.catcol, self.alphacol], [str, gdf.get, bad]
                        )
                    ],
                    "cmap": [
                        {
                            "__cmap_type": cm.__class__.__name__
                            if not isinstance(cm, str)
                            else cm,
                            "cmap": cm,
                            "__expect_error": cm == "bad",
                        }
                        for cm in cm_l
                    ]
                    + [
                        {
                            "__cmap_type": (
                                f"{'short' if len(v)<len(self.gdf) else 'full'} "
                                f"{t.__name__} {cat}"
                            ).strip(),
                            "cmap": create_type(t, v),
                            "__expect_error": cat == "bad",
                        }
                        for t in [list, tuple, pd.Series, pd.DataFrame, np.array]
                        for v, cat in zip(
                            [cl, cs, clrgb, csrgb, clb, csb],
                            np.repeat(["", "rgb", "bad"], 2),
                        )
                    ],
                    "color": [None, "red", "bad"],
                    "legend": [True, False, None],
                    "categorical": [True, False, None],
                },
            },
            "batch2": {
                # only used when categorical is False, this also
                # means column is not categorical
                "filter": lambda d: ~d["categorical"].fillna(True)
                & d["__column_type"].str.startswith("__col"),
                "columns": {
                    "scheme": list(mapclassify.classifiers.CLASSIFIERS)[11:12]
                    + [None, "bad"],
                    "k": [5, "bad"],
                    "vmin": [None, 3, "bad"],
                    "vmax": [None, 4, "bad"],
                },
            },
        }

        def expect_cols(df):
            return df.loc[:, [c for c in df.columns if c.startswith("__expect_error")]]

        df = pd.DataFrame(columns=["xref", "__error", "__warning"])
        test_n = 0
        for batch in test_cases.keys():
            for c, v in {
                c: v
                for c, v in test_cases[batch]["columns"].items()
                if c not in exclude
            }.items():
                if isinstance(v, dict) or isinstance(v[0], dict):
                    df_ = pd.DataFrame(v).assign(xref=1)
                else:
                    df_ = pd.DataFrame({c: v}).assign(
                        **{"xref": 1, f"__expect_error_{c}": lambda d: d[c].eq("bad")}
                    )
                # cartesian product of existings tests with new tests. Only if there is
                # not already a expected fail to reduce perm explosion
                mask = expect_cols(df).any(axis=1)
                mask = mask | ~test_cases[batch]["filter"](df)
                df = pd.concat(
                    [
                        df.loc[
                            ~mask,
                        ].merge(df_, on="xref", how="right"),
                        df.loc[
                            mask,
                        ],
                    ]
                ).reset_index(drop=True)
                # concat is bashing dtypes
                for col in expect_cols(df).columns:
                    df[col] = df[col].convert_dtypes()
                test_n += 1

        df = df.drop(columns=["xref"])
        if "k" in df.columns:
            df["k"] = df["k"].fillna(5)
        # nan and None do not work in sample way for explore() args
        # make nan None
        df.loc[:, [c for c in df.columns if not c.startswith("__")]] = df.loc[
            :, [c for c in df.columns if not c.startswith("__")]
        ].replace([np.nan], [None])

        if "cmap" in df.columns:
            df["__cmap_type_agg"] = np.select(
                [
                    df["cmap"].apply(
                        lambda c: pd.api.types.is_list_like(c) and len(c) == len(gdf)
                    ),
                    df["cmap"].apply(pd.api.types.is_list_like),
                ],
                ["full list_like", "short list_like"],
                df["__cmap_type"].fillna("-"),
            )

        # summarise columns to one
        df["__expect_error"] = df.loc[
            :, [c for c in df.columns if c.startswith("__expect_error")]
        ].any(axis=1)
        df = df.drop(columns=[c for c in df.columns if c.startswith("__expect_error_")])

        if max_tests is not None:
            # smart downsample, get a group number which
            # can be used to define random state of each group
            df["__group"] = df.groupby(
                ["__cmap_type_agg", "__expect_error"], group_keys=True
            ).ngroup()
            n = int(max_tests / df["__group"].max())
            df = (
                df.groupby("__group", group_keys=True)
                .apply(
                    lambda d: d
                    if len(d) <= n
                    else d.sample(n=n, random_state=30 + d.name)
                )
                .reset_index(drop=True)
            )

        return df.reset_index(drop=True)

    def run(self, df):
        gdf = self.gdf

        maps = {}

        for i, args in df.iterrows():
            all_opts = {k: v for k, v in args.items() if not k.startswith("__")}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    # matplotlib 3.7 and numpy 1.24 generate a warning that is not in
                    # control of geopandas.  Filter it so number of warnings is
                    # consistent
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                        module="matplotlib",
                        message="NumPy will stop allowing conversion of out-of-bound",
                    )

                    try:
                        m = gdf.explore(**{**all_opts, **self.plot_opts})
                    except (
                        UserWarning,
                        RuntimeWarning,
                        PendingDeprecationWarning,
                        DeprecationWarning,
                    ) as e:
                        df.loc[i, "__warning"] = str(e)
                        df.loc[i, "__warning_class"] = e.__class__.__name__
                        warnings.simplefilter("ignore")
                        m = gdf.explore(**{**all_opts, **self.plot_opts})
                    maps[i] = {"m": m, "opts": all_opts}

            except (ValueError, TypeError, IndexError, AttributeError, KeyError) as e:
                maps[i] = {"opts": all_opts}
                df.loc[i, "__error"] = str(e)
                df.loc[i, "__error_class"] = e.__class__.__name__

        return df, maps

    def expected_result(self, df):
        # total number of attempts, errors and warnings as expected
        assert pd.Series(
            {
                c: (~df[c].isna()).sum()
                for c in ["__column_type", "__error", "__warning"]
            }
        ).equals(pd.Series({"__column_type": 88, "__error": 13, "__warning": 37}))

        expected = {
            "__error": {
                "Invalid RGBA argument: 'bluee'": 1,
                "Invalid scheme. Scheme must be": 2,
                "The GeoDataFrame and given col": 3,
                "`cmap` is invalid. For categor": 2,
                "`cmap` is not known matplotlib": 5,
            },
            "__warning": {
                "`cmap` as callable or list is ": 2,
                "`cmap` invalid for legend": 2,
                "`k` is invalid. Defaulted": 3,
                "`vmax` invalid. Defaulted": 20,
                "`vmin` invalid. Defaulted": 10,
            },
        }

        for col, result in expected.items():
            assert df.groupby(df[col].str[0:30]).size().equals(pd.Series(result))

    def html(self, df=None, maps=None):
        import json

        plot_opts = self.plot_opts
        fmt = (
            '<iframe srcdoc="{}" style="width: {}px; height: {}px; '
            "display:inline-block; width: 24%; margin: 0 auto; border:"
            ' 2px solid black"></iframe>'
        )

        rawhtml = ""
        df["__bc"] = np.where(df["__warning"].isna(), "white", "pink")
        df["__warning"] = df["__warning"].fillna("")
        for i, d in (
            df.sort_values(
                ["__cmap_type_agg", "__cmap_type"]
                + [c for c in ["categorical", "scheme", "legend"] if c in df.columns]
            )
            .loc[lambda d: d["__error"].isna()]
            .groupby("__cmap_type_agg")
        ):
            rawhtml += f"<h2>cmap {i}</h2>"
            for i2, d2 in d.groupby(df["__column_type"].str[:-3]):
                rawhtml += f"<h3>column: {i2}</h3>"
                for i, r in d2.iterrows():
                    m = maps[i]["m"]
                    param_str = json.dumps(
                        {
                            k: v
                            for k, v in r.to_dict().items()
                            if not k[0:2] == "__"
                            and not pd.api.types.is_list_like(v)
                            and not callable(v)
                        }
                    )
                    m.get_root().html.add_child(
                        folium.Element(
                            (
                                f'<h5 style="background-color: {r["__bc"]}">'
                                f'<b>{i}</b> {param_str} {r["__warning"]}</h5>'
                            )
                        )
                    )

                    rawhtml += fmt.format(
                        m.get_root().render().replace('"', "&quot;"),
                        plot_opts["height"],
                        plot_opts["width"],
                    )
        return rawhtml
