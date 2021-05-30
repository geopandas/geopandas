import folium
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors
from branca.colormap import StepColormap
import contextily
import numpy as np
import pandas as pd
import pytest

from geopandas.explore import _explore

nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
cities = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world["range"] = range(len(world))
missing = world.copy()
np.random.seed(42)
missing.loc[np.random.choice(missing.index, 40), "continent"] = np.nan
missing.loc[np.random.choice(missing.index, 40), "pop_est"] = np.nan


def _fetch_map_string(m):
    out = m._parent.render()
    out_str = "".join(out.split())
    return out_str


def test_simple_pass():
    """Make sure default pass"""
    _explore(nybb)
    _explore(world)
    _explore(cities)
    _explore(world.geometry)


def test_choropleth_pass():
    """Make sure default choropleth pass"""
    _explore(world, column="pop_est")


def test_map_settings_default():
    """Check default map settings"""
    m = _explore(world)
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


def test_map_settings_custom():
    """Check custom map settins"""
    m = _explore(
        nybb, zoom_control=False, width=200, height=200, tiles="CartoDB positron"
    )
    assert m.location == [
        pytest.approx(40.70582377450201, rel=1e-6),
        pytest.approx(-73.9778006856748, rel=1e-6),
    ]
    assert m.options["zoom"] == 10
    assert m.options["zoomControl"] is False
    assert m.height == (200.0, "px")
    assert m.width == (200.0, "px")
    assert "cartodbpositron" in m.to_dict()["children"].keys()

    # custom XYZ tiles
    m = _explore(
        nybb,
        zoom_control=False,
        width=200,
        height=200,
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
    )

    out_str = _fetch_map_string(m)
    assert (
        'Layer("https://mt1.google.com/vt/lyrs=m\\u0026x={x}\\u0026y={y}\\u0026z={z}"'
        in out_str
    )
    assert '"attribution":"Google"' in out_str

    m = _explore(nybb, location=(40, 5))
    assert m.location == [40, 5]
    assert m.options["zoom"] == 10

    m = _explore(nybb, zoom_start=8)
    assert m.location == [
        pytest.approx(40.70582377450201, rel=1e-6),
        pytest.approx(-73.9778006856748, rel=1e-6),
    ]
    assert m.options["zoom"] == 8

    m = _explore(nybb, location=(40, 5), zoom_start=8)
    assert m.location == [40, 5]
    assert m.options["zoom"] == 8


def test_simple_color():
    """Check color settings"""
    # single named color
    m = _explore(nybb, color="red")
    out_str = _fetch_map_string(m)
    assert '"fillColor":"red"' in out_str

    # list of colors
    colors = ["#333333", "#367324", "#95824f", "#fcaa00", "#ffcc33"]
    m2 = _explore(nybb, color=colors)
    out_str = _fetch_map_string(m2)
    for c in colors:
        assert f'"fillColor":"{c}"' in out_str

    # column of colors
    df = nybb.copy()
    df["colors"] = colors
    m3 = _explore(df, color="colors")
    out_str = _fetch_map_string(m3)
    for c in colors:
        assert f'"fillColor":"{c}"' in out_str

    # line GeoSeries
    m4 = _explore(nybb.boundary, color="red")
    out_str = _fetch_map_string(m4)
    assert '"fillColor":"red"' in out_str


def test_choropleth_linear():
    """Check choropleth colors"""
    # default cmap
    m = _explore(nybb, column="Shape_Leng")
    out_str = _fetch_map_string(m)
    assert 'color":"#440154"' in out_str
    assert 'color":"#fde725"' in out_str
    assert 'color":"#50c46a"' in out_str
    assert 'color":"#481467"' in out_str
    assert 'color":"#3d4e8a"' in out_str

    # named cmap
    m = _explore(nybb, column="Shape_Leng", cmap="PuRd")
    out_str = _fetch_map_string(m)
    assert 'color":"#f7f4f9"' in out_str
    assert 'color":"#67001f"' in out_str
    assert 'color":"#d31760"' in out_str
    assert 'color":"#f0ecf5"' in out_str
    assert 'color":"#d6bedc"' in out_str


def test_choropleth_mapclassify():
    """Mapclassify bins"""
    # quantiles
    m = _explore(nybb, column="Shape_Leng", scheme="quantiles")
    out_str = _fetch_map_string(m)
    assert 'color":"#21918c"' in out_str
    assert 'color":"#3b528b"' in out_str
    assert 'color":"#5ec962"' in out_str
    assert 'color":"#fde725"' in out_str
    assert 'color":"#440154"' in out_str

    # headtail
    m = _explore(world, column="pop_est", scheme="headtailbreaks")
    out_str = _fetch_map_string(m)
    assert '"fillColor":"#3b528b"' in out_str
    assert '"fillColor":"#21918c"' in out_str
    assert '"fillColor":"#5ec962"' in out_str
    assert '"fillColor":"#fde725"' in out_str
    assert '"fillColor":"#440154"' in out_str
    # custom k
    m = _explore(world, column="pop_est", scheme="naturalbreaks", k=3)
    out_str = _fetch_map_string(m)
    assert '"fillColor":"#21918c"' in out_str
    assert '"fillColor":"#fde725"' in out_str
    assert '"fillColor":"#440154"' in out_str


def test_categorical():
    """Categorical maps"""
    # auto detection
    m = _explore(world, column="continent")
    out_str = _fetch_map_string(m)
    assert 'color":"#9467bd","continent":"Europe"' in out_str
    assert 'color":"#c49c94","continent":"NorthAmerica"' in out_str
    assert 'color":"#1f77b4","continent":"Africa"' in out_str
    assert 'color":"#98df8a","continent":"Asia"' in out_str
    assert 'color":"#ff7f0e","continent":"Antarctica"' in out_str
    assert 'color":"#9edae5","continent":"SouthAmerica"' in out_str
    assert 'color":"#7f7f7f","continent":"Oceania"' in out_str
    assert 'color":"#dbdb8d","continent":"Sevenseas(openocean)"' in out_str

    # forced categorical
    m = _explore(nybb, column="BoroCode", categorical=True)
    out_str = _fetch_map_string(m)
    assert 'color":"#9edae5"' in out_str
    assert 'color":"#c7c7c7"' in out_str
    assert 'color":"#8c564b"' in out_str
    assert 'color":"#1f77b4"' in out_str
    assert 'color":"#98df8a"' in out_str

    # pandas.Categorical
    df = world.copy()
    df["categorical"] = pd.Categorical(df["name"])
    m = _explore(df, column="categorical")
    out_str = _fetch_map_string(m)
    for c in np.apply_along_axis(colors.to_hex, 1, cm.tab20(range(20))):
        assert f'"fillColor":"{c}"' in out_str

    # custom cmap
    m = _explore(nybb, column="BoroName", cmap="Set1")
    out_str = _fetch_map_string(m)
    assert 'color":"#999999"' in out_str
    assert 'color":"#a65628"' in out_str
    assert 'color":"#4daf4a"' in out_str
    assert 'color":"#e41a1c"' in out_str
    assert 'color":"#ff7f00"' in out_str

    # custom list of colors
    cmap = ["#333432", "#3b6e8c", "#bc5b4f", "#8fa37e", "#efc758"]
    m = _explore(nybb, column="BoroName", cmap=cmap)
    out_str = _fetch_map_string(m)
    for c in cmap:
        assert f'"fillColor":"{c}"' in out_str

    # shorter list (to make it repeat)
    cmap = ["#333432", "#3b6e8c"]
    m = _explore(nybb, column="BoroName", cmap=cmap)
    out_str = _fetch_map_string(m)
    for c in cmap:
        assert f'"fillColor":"{c}"' in out_str

    with pytest.raises(ValueError, match="'cmap' is invalid."):
        _explore(nybb, column="BoroName", cmap="nonsense")


def test_categories():
    m = _explore(
        nybb[["BoroName", "geometry"]],
        column="BoroName",
        categories=["Brooklyn", "Staten Island", "Queens", "Bronx", "Manhattan"],
    )
    out_str = _fetch_map_string(m)
    assert '"Bronx","__folium_color":"#c7c7c7"' in out_str
    assert '"Manhattan","__folium_color":"#9edae5"' in out_str
    assert '"Brooklyn","__folium_color":"#1f77b4"' in out_str
    assert '"StatenIsland","__folium_color":"#98df8a"' in out_str
    assert '"Queens","__folium_color":"#8c564b"' in out_str

    df = nybb.copy()
    df["categorical"] = pd.Categorical(df["BoroName"])
    with pytest.raises(ValueError, match="Cannot specify 'categories'"):
        _explore(df, "categorical", categories=["Brooklyn", "Staten Island"])


def test_column_values():
    """
    Check that the dataframe plot method returns same values with an
    input string (column in df), pd.Series, or np.array
    """
    column_array = np.array(world["pop_est"])
    m1 = _explore(world, column="pop_est")  # column name
    m2 = _explore(world, column=column_array)  # np.array
    m3 = _explore(world, column=world["pop_est"])  # pd.Series
    assert m1.location == m2.location == m3.location

    m1_fields = _explore(world, column=column_array, tooltip=True, popup=True)
    out1_fields_str = _fetch_map_string(m1_fields)
    assert (
        'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
        in out1_fields_str
    )
    assert (
        'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
        in out1_fields_str
    )

    m2_fields = _explore(world, column=world["pop_est"], tooltip=True, popup=True)
    out2_fields_str = _fetch_map_string(m2_fields)
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
        _explore(world, column=np.array([1, 2, 3]))


def test_no_crs():
    """Naive geometry get no tiles"""
    df = world.copy()
    df.crs = None
    m = _explore(df)
    assert "openstreetmap" not in m.to_dict()["children"].keys()


def test_style_kwds():
    """Style keywords"""
    m = _explore(
        world, style_kwds=dict(fillOpacity=0.1, weight=0.5, fillColor="orange")
    )
    out_str = _fetch_map_string(m)
    assert '"fillColor":"orange","fillOpacity":0.1,"weight":0.5' in out_str
    m = _explore(world, column="pop_est", style_kwds=dict(color="black"))
    assert '"color":"black"' in _fetch_map_string(m)


def test_tooltip():
    """Test tooltip"""
    # default with no tooltip or popup
    m = _explore(world)
    assert "GeoJsonTooltip" in str(m.to_dict())
    assert "GeoJsonPopup" not in str(m.to_dict())

    # True
    m = _explore(world, tooltip=True, popup=True)
    assert "GeoJsonTooltip" in str(m.to_dict())
    assert "GeoJsonPopup" in str(m.to_dict())
    out_str = _fetch_map_string(m)
    assert (
        'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    )
    assert (
        'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
        in out_str
    )

    # True choropleth
    m = _explore(world, column="pop_est", tooltip=True, popup=True)
    assert "GeoJsonTooltip" in str(m.to_dict())
    assert "GeoJsonPopup" in str(m.to_dict())
    out_str = _fetch_map_string(m)
    assert (
        'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    )
    assert (
        'aliases=["pop_est","continent","name","iso_a3","gdp_md_est","range"]'
        in out_str
    )

    # single column
    m = _explore(world, tooltip="pop_est", popup="iso_a3")
    out_str = _fetch_map_string(m)
    assert 'fields=["pop_est"]' in out_str
    assert 'aliases=["pop_est"]' in out_str
    assert 'fields=["iso_a3"]' in out_str
    assert 'aliases=["iso_a3"]' in out_str

    # list
    m = _explore(
        world, tooltip=["pop_est", "continent"], popup=["iso_a3", "gdp_md_est"]
    )
    out_str = _fetch_map_string(m)
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str
    assert 'fields=["iso_a3","gdp_md_est"' in out_str
    assert 'aliases=["iso_a3","gdp_md_est"]' in out_str

    # number
    m = _explore(world, tooltip=2, popup=2)
    out_str = _fetch_map_string(m)
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str

    # keywords tooltip
    m = _explore(
        world,
        tooltip=True,
        popup=False,
        tooltip_kwds=dict(aliases=[0, 1, 2, 3, 4, 5], sticky=False),
    )
    out_str = _fetch_map_string(m)
    assert (
        'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    )
    assert "aliases=[0,1,2,3,4,5]" in out_str
    assert '"sticky":false' in out_str

    # keywords popup
    m = _explore(
        world,
        tooltip=False,
        popup=True,
        popup_kwds=dict(aliases=[0, 1, 2, 3, 4, 5]),
    )
    out_str = _fetch_map_string(m)
    assert (
        'fields=["pop_est","continent","name","iso_a3","gdp_md_est","range"]' in out_str
    )
    assert "aliases=[0,1,2,3,4,5]" in out_str
    assert "<th>${aliases[i]" in out_str

    # no labels
    m = _explore(
        world,
        tooltip=True,
        popup=True,
        tooltip_kwds=dict(labels=False),
        popup_kwds=dict(labels=False),
    )
    out_str = _fetch_map_string(m)
    assert "<th>${aliases[i]" not in out_str


def test_custom_markers():
    # Markers
    m = _explore(
        cities,
        marker_type="marker",
        marker_kwds={"icon": folium.Icon(icon="star")},
    )
    assert ""","icon":"star",""" in _fetch_map_string(m)

    # Circle Markers
    m = _explore(cities, marker_type="circle", marker_kwds={"fill_color": "red"})
    assert ""","fillColor":"red",""" in _fetch_map_string(m)

    # Folium Markers
    m = _explore(
        cities,
        marker_type=folium.Circle(
            radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1
        ),
    )
    assert ""","color":"black",""" in _fetch_map_string(m)

    # Circle
    m = _explore(cities, marker_type="circle_marker", marker_kwds={"radius": 10})
    assert ""","radius":10,""" in _fetch_map_string(m)

    # Unsupported Markers
    with pytest.raises(
        ValueError, match="Only 'marker', 'circle', and 'circle_marker' are supported"
    ):
        _explore(cities, marker_type="dummy")


def test_vmin_vmax():
    df = world.copy()
    df["range"] = range(len(df))
    m = _explore(df, "range", vmin=-100, vmax=1000)
    out_str = _fetch_map_string(m)
    assert 'case"176":return{"color":"#3b528b","fillColor":"#3b528b"' in out_str
    assert 'case"119":return{"color":"#414287","fillColor":"#414287"' in out_str
    assert 'case"3":return{"color":"#482173","fillColor":"#482173"' in out_str

    with pytest.warns(UserWarning, match="vmin' cannot be higher than minimum value"):
        m = _explore(df, "range", vmin=100000)

    with pytest.warns(UserWarning, match="'vmax' cannot be lower than maximum value"):
        m = _explore(df, "range", vmax=10)


def test_missing_vals():
    m = _explore(missing, "continent")
    assert '"fillColor":null' in _fetch_map_string(m)

    m = _explore(missing, "pop_est")
    assert '"fillColor":null' in _fetch_map_string(m)

    m = _explore(missing, "pop_est", missing_kwds=dict(color="red"))
    assert '"fillColor":"red"' in _fetch_map_string(m)

    m = _explore(missing, "continent", missing_kwds=dict(color="red"))
    assert '"fillColor":"red"' in _fetch_map_string(m)


def test_categorical_legend():
    m = _explore(world, "continent", legend=True)
    out_str = _fetch_map_string(m)
    assert "#1f77b4'></span>Africa" in out_str
    assert "#ff7f0e'></span>Antarctica" in out_str
    assert "#98df8a'></span>Asia" in out_str
    assert "#9467bd'></span>Europe" in out_str
    assert "#c49c94'></span>NorthAmerica" in out_str
    assert "#7f7f7f'></span>Oceania" in out_str
    assert "#dbdb8d'></span>Sevenseas(openocean)" in out_str
    assert "#9edae5'></span>SouthAmerica" in out_str

    m = _explore(missing, "continent", legend=True, missing_kwds={"color": "red"})
    out_str = _fetch_map_string(m)
    assert "red'></span>NaN" in out_str


def test_colorbar():
    m = _explore(world, "range", legend=True)
    out_str = _fetch_map_string(m)
    assert "attr(\"id\",'legend')" in out_str
    assert "text('range')" in out_str

    m = _explore(world, "range", legend=True, legend_kwds=dict(caption="my_caption"))
    out_str = _fetch_map_string(m)
    assert "attr(\"id\",'legend')" in out_str
    assert "text('my_caption')" in out_str

    m = _explore(missing, "pop_est", legend=True, missing_kwds=dict(color="red"))
    out_str = _fetch_map_string(m)
    assert "red'></span>NaN" in out_str

    # do not scale legend
    m = _explore(
        world,
        "pop_est",
        legend=True,
        legend_kwds=dict(scale=False),
        scheme="Headtailbreaks",
    )
    out_str = _fetch_map_string(m)
    assert out_str.count("#440154ff") == 100
    assert out_str.count("#3b528bff") == 100
    assert out_str.count("#21918cff") == 100
    assert out_str.count("#5ec962ff") == 100
    assert out_str.count("#fde725ff") == 100

    # scale legend accorrdingly
    m = _explore(
        world,
        "pop_est",
        legend=True,
        scheme="Headtailbreaks",
    )
    out_str = _fetch_map_string(m)
    assert out_str.count("#440154ff") == 16
    assert out_str.count("#3b528bff") == 51
    assert out_str.count("#21918cff") == 133
    assert out_str.count("#5ec962ff") == 282
    assert out_str.count("#fde725ff") == 18

    # discrete cmap
    m = _explore(world, "pop_est", legend=True, cmap="Pastel2")
    out_str = _fetch_map_string(m)

    assert out_str.count("b3e2cdff") == 63
    assert out_str.count("fdcdacff") == 62
    assert out_str.count("cbd5e8ff") == 63
    assert out_str.count("f4cae4ff") == 62
    assert out_str.count("e6f5c9ff") == 62
    assert out_str.count("fff2aeff") == 63
    assert out_str.count("f1e2ccff") == 62
    assert out_str.count("ccccccff") == 63


def test_providers():
    m = _explore(nybb, tiles=contextily.providers.CartoDB.PositronNoLabels)
    out_str = _fetch_map_string(m)

    assert (
        '"https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"' in out_str
    )
    assert '"attribution":"(C)OpenStreetMapcontributors(C)CARTO"' in out_str
    assert '"maxNativeZoom":19,"maxZoom":19,"minZoom":0' in out_str


def test_linearrings():
    rings = nybb.explode().exterior
    m = _explore(rings)
    out_str = _fetch_map_string(m)

    assert out_str.count("LineString") == len(rings)


def test_mapclassify_categorical_legend():
    m = _explore(
        missing,
        column="pop_est",
        legend=True,
        scheme="naturalbreaks",
        missing_kwds=dict(color="red", label="Missing"),
        legend_kwds=dict(colorbar=False, interval=True),
    )
    out_str = _fetch_map_string(m)

    strings = [
        "[140.00,33986655.00]",
        "(33986655.00,105350020.00]",
        "(105350020.00,207353391.00]",
        "(207353391.00,326625791.00]",
        "(326625791.00,1379302771.00]",
        "Missing",
    ]
    for s in strings:
        assert s in out_str

    # interval=False
    m = _explore(
        missing,
        column="pop_est",
        legend=True,
        scheme="naturalbreaks",
        missing_kwds=dict(color="red", label="Missing"),
        legend_kwds=dict(colorbar=False, interval=False),
    )
    out_str = _fetch_map_string(m)

    strings = [
        ">140.00,33986655.00",
        ">33986655.00,105350020.00",
        ">105350020.00,207353391.00",
        ">207353391.00,326625791.00",
        ">326625791.00,1379302771.00",
        "Missing",
    ]
    for s in strings:
        assert s in out_str

    # custom labels
    m = _explore(
        world,
        column="pop_est",
        legend=True,
        scheme="naturalbreaks",
        k=5,
        legend_kwds=dict(colorbar=False, labels=["s", "m", "l", "xl", "xxl"]),
    )
    out_str = _fetch_map_string(m)

    strings = [">s<", ">m<", ">l<", ">xl<", ">xxl<"]
    for s in strings:
        assert s in out_str

    # fmt
    m = _explore(
        missing,
        column="pop_est",
        legend=True,
        scheme="naturalbreaks",
        missing_kwds=dict(color="red", label="Missing"),
        legend_kwds=dict(colorbar=False, fmt="{:.0f}"),
    )
    out_str = _fetch_map_string(m)

    strings = [
        ">140,33986655",
        ">33986655,105350020",
        ">105350020,207353391",
        ">207353391,326625791",
        ">326625791,1379302771",
        "Missing",
    ]
    for s in strings:
        assert s in out_str


def test_given_m():
    "Check that geometry is mapped onto a given folium.Map"
    m = folium.Map()
    _explore(nybb, m=m, tooltip=False, highlight=False)

    out_str = _fetch_map_string(m)

    assert out_str.count("BoroCode") == 5
    # should not change map settings
    assert m.options["zoom"] == 1


def test_highlight():
    m = _explore(nybb, highlight=True)
    out_str = _fetch_map_string(m)

    assert '"fillOpacity":0.75' in out_str

    m = _explore(nybb, highlight=True, highlight_kwds=dict(fillOpacity=1, color="red"))
    out_str = _fetch_map_string(m)

    assert '{"color":"red","fillOpacity":1}' in out_str


def test_custom_colormaps():

    step = StepColormap(["green", "yellow", "red"], vmin=0, vmax=100000000)

    m = _explore(world, "pop_est", cmap=step, tooltip=["name"], legend=True)

    strings = [
        'fillColor":"#008000ff"',  # Green
        '"fillColor":"#ffff00ff"',  # Yellow
        '"fillColor":"#ff0000ff"',  # Red
    ]

    out_str = _fetch_map_string(m)
    for s in strings:
        assert s in out_str

    assert out_str.count("008000ff") == 306
    assert out_str.count("ffff00ff") == 187
    assert out_str.count("ff0000ff") == 190

    # Using custom function colormap
    def my_color_function(field):
        """Maps low values to green and high values to red."""
        if field > 100000000:
            return "#ff0000"
        else:
            return "#008000"

    m = _explore(world, "pop_est", cmap=my_color_function, legend=False)

    strings = [
        '"color":"#ff0000","fillColor":"#ff0000"',
        '"color":"#008000","fillColor":"#008000"',
    ]

    for s in strings:
        assert s in _fetch_map_string(m)

    # matplotlib.Colormap
    cmap = colors.ListedColormap(["red", "green", "blue", "white", "black"])

    m = _explore(nybb, "BoroName", cmap=cmap)
    strings = [
        '"fillColor":"#ff0000"',  # Red
        '"fillColor":"#008000"',  # Green
        '"fillColor":"#0000ff"',  # Blue
        '"fillColor":"#ffffff"',  # White
        '"fillColor":"#000000"',  # Black
    ]

    out_str = _fetch_map_string(m)
    for s in strings:
        assert s in out_str
