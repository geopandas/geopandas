import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version

folium = pytest.importorskip("folium")
branca = pytest.importorskip("branca")
matplotlib = pytest.importorskip("matplotlib")
mapclassify = pytest.importorskip("mapclassify")
geodatasets = pytest.importorskip("geodatasets")

from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap

BRANCA_05 = Version(branca.__version__) > Version("0.4.2")
FOLIUM_G_014 = Version(folium.__version__) > Version("0.14.0")


class TestExplore:
    def setup_method(self):
        self.nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
        self.world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        self.cities = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
        self.chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
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

        # UserDefined overriding default k
        m = self.chicago.explore(
            column="POP2010",
            scheme="UserDefined",
            classification_kwds={"bins": [25000, 50000, 75000, 100000]},
        )
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"#fde725"' in out_str
        assert '"fillColor":"#35b779"' in out_str
        assert '"fillColor":"#31688e"' in out_str
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

        with pytest.raises(ValueError, match="'cmap' is invalid."):
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
            style_kwds={"fillOpacity": 0.1, "weight": 0.5, "fillColor": "orange"}
        )
        out_str = self._fetch_map_string(m)
        assert '"fillColor":"orange","fillOpacity":0.1,"weight":0.5' in out_str
        m = self.world.explore(column="pop_est", style_kwds={"color": "black"})
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
            ('"fillColor":"green"' in t and '"color":"white"' in t)
            or ('"fillColor":"red"' in t and '"color":"black"' in t)
            for t in [
                "".join(line.split())
                for line in m._parent.render().split("\n")
                if "return" in line and "color" in line
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
            tooltip_kwds={"aliases": [0, 1, 2, 3, 4, 5], "sticky": False},
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
            popup_kwds={"aliases": [0, 1, 2, 3, 4, 5]},
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
            tooltip_kwds={"labels": False},
            popup_kwds={"labels": False},
        )
        out_str = self._fetch_map_string(m)
        assert "<th>${aliases[i]" not in out_str

        # named index
        gdf = self.nybb.set_index("BoroName")
        m = gdf.explore()
        out_str = self._fetch_map_string(m)
        assert "BoroName" in out_str

    def test_escape_special_characters(self):
        # check if special characters are escaped
        gdf = self.world.copy()
        gdf["name"] = """{{{what a mess}}} they are so different."""
        m = gdf.explore()
        out_str = self._fetch_map_string(m)
        assert """{{{""" in out_str
        assert """}}}""" in out_str

    def test_default_markers(self):
        # check overridden default for points
        m = self.cities.explore()
        strings = ['"radius":2', '"fill":true', "CircleMarker(latlng,opts)"]
        out_str = self._fetch_map_string(m)
        for s in strings:
            assert s in out_str

        m = self.cities.explore(marker_kwds={"radius": 5, "fill": False})
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
        if FOLIUM_G_014:
            assert 'case"0":return{"color":"#fde725","fillColor":"#fde725"' in out_str
            assert 'case"1":return{"color":"#7ad151","fillColor":"#7ad151"' in out_str
            assert 'default:return{"color":"#22a884","fillColor":"#22a884"' in out_str
        else:
            assert 'case"1":return{"color":"#7ad151","fillColor":"#7ad151"' in out_str
            assert 'case"2":return{"color":"#22a884","fillColor":"#22a884"' in out_str
            assert 'default:return{"color":"#fde725","fillColor":"#fde725"' in out_str

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

        m = self.missing.explore("pop_est", missing_kwds={"color": "red"})
        assert '"fillColor":"red"' in self._fetch_map_string(m)

        m = self.missing.explore("continent", missing_kwds={"color": "red"})
        assert '"fillColor":"red"' in self._fetch_map_string(m)

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
        def quoted_in(find, s):
            return find in s or find.replace("'", '"') in s

        m = self.world.explore("range", legend=True)
        out_str = self._fetch_map_string(m)
        assert "attr(\"id\",'legend')" in out_str
        assert quoted_in("text('range')", out_str)

        m = self.world.explore(
            "range", legend=True, legend_kwds={"caption": "my_caption"}
        )
        out_str = self._fetch_map_string(m)
        assert "attr(\"id\",'legend')" in out_str
        assert quoted_in("text('my_caption')", out_str)

        m = self.missing.explore("pop_est", legend=True, missing_kwds={"color": "red"})
        out_str = self._fetch_map_string(m)
        assert "red'></span>NaN" in out_str

        # do not scale legend
        m = self.world.explore(
            "pop_est",
            legend=True,
            legend_kwds={"scale": False},
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
        m = self.world.explore("pop_est", legend_kwds={"max_labels": 3})
        out_str = self._fetch_map_string(m)
        tick_str = re.search(r"tickValues\(\[[\',\,\.,0-9]*\]\)", out_str).group(0)
        assert (
            tick_str.replace(",''", "")
            == "tickValues([140.0,471386328.07843137,942772516.1568627])"
        )

        # scheme
        m = self.world.explore(
            "pop_est", scheme="headtailbreaks", legend_kwds={"max_labels": 3}
        )
        out_str = self._fetch_map_string(m)
        assert "tickValues([140.0,'',184117213.1818182,'',1382066377.0,''])" in out_str

        # short cmap
        m = self.world.explore("pop_est", legend_kwds={"max_labels": 3}, cmap="tab10")
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

    def test_xyzservices_providers_min_zoom_override(self):
        xyzservices = pytest.importorskip("xyzservices")

        m = self.nybb.explore(
            tiles=xyzservices.providers.CartoDB.PositronNoLabels, min_zoom=3
        )
        out_str = self._fetch_map_string(m)

        assert '"maxNativeZoom":20,"maxZoom":20,"minZoom":3' in out_str

    def test_xyzservices_providers_max_zoom_override(self):
        xyzservices = pytest.importorskip("xyzservices")

        m = self.nybb.explore(
            tiles=xyzservices.providers.CartoDB.PositronNoLabels, max_zoom=12
        )
        out_str = self._fetch_map_string(m)

        assert '"maxNativeZoom":12,"maxZoom":12,"minZoom":0' in out_str

    def test_xyzservices_providers_both_zooms_override(self):
        xyzservices = pytest.importorskip("xyzservices")

        m = self.nybb.explore(
            tiles=xyzservices.providers.CartoDB.PositronNoLabels,
            min_zoom=3,
            max_zoom=12,
        )
        out_str = self._fetch_map_string(m)

        assert '"maxNativeZoom":12,"maxZoom":12,"minZoom":3' in out_str

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
            missing_kwds={"color": "red", "label": "missing"},
            legend_kwds={"colorbar": False, "interval": True},
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
            missing_kwds={"color": "red", "label": "missing"},
            legend_kwds={"colorbar": False, "interval": False},
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
            legend_kwds={"colorbar": False, "labels": ["s", "m", "l", "xl", "xxl"]},
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
            missing_kwds={"color": "red", "label": "missing"},
            legend_kwds={"colorbar": False, "fmt": "{:.0f}"},
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
            highlight=True, highlight_kwds={"fillOpacity": 1, "color": "red"}
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
            zoom_control=False, map_kwds={"dragging": False, "scrollWheelZoom": False}
        )
        check()
        with pytest.raises(
            ValueError, match="'zoom_control' cannot be specified in 'map_kwds'"
        ):
            self.world.explore(
                map_kwds={
                    "dragging": False,
                    "scrollWheelZoom": False,
                    "zoom_control": False,
                }
            )

    def test_none_geometry(self):
        # None and empty geoms are dropped prior plotting
        df = self.nybb.copy()
        df.loc[0, df.geometry.name] = None
        m = df.explore()
        self._fetch_map_string(m)

    def test_empty_geometry(self):
        # None and empty geoms are dropped prior plotting
        df = self.nybb.copy()
        df.loc[0, df.geometry.name] = shapely.Point()
        m = df.explore()
        self._fetch_map_string(m)
