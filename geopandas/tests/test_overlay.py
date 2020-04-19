import os

import pandas as pd

from shapely.geometry import Point, Polygon, LineString, GeometryCollection
from fiona.errors import DriverError

import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest

DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "overlay")


@pytest.fixture
def dfs(request):
    s1 = GeoSeries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ]
    )
    s2 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": s1})
    df2 = GeoDataFrame({"col2": [1, 2], "geometry": s2})
    return df1, df2


@pytest.fixture(params=["default-index", "int-index", "string-index"])
def dfs_index(request, dfs):
    df1, df2 = dfs
    if request.param == "int-index":
        df1.index = [1, 2]
        df2.index = [0, 2]
    if request.param == "string-index":
        df1.index = ["row1", "row2"]
    return df1, df2


@pytest.fixture(
    params=["union", "intersection", "difference", "symmetric_difference", "identity"]
)
def how(request):
    return request.param


@pytest.fixture(params=[True, False])
def keep_geom_type(request):
    return request.param


def test_overlay(dfs_index, how):
    """
    Basic overlay test with small dummy example dataframes (from docs).
    Results obtained using QGIS 2.16 (Vector -> Geoprocessing Tools ->
    Intersection / Union / ...), saved to GeoJSON
    """
    df1, df2 = dfs_index
    result = overlay(df1, df2, how=how)

    # construction of result

    def _read(name):
        expected = read_file(
            os.path.join(DATA, "polys", "df1_df2-{0}.geojson".format(name))
        )
        expected.crs = None
        return expected

    if how == "identity":
        expected_intersection = _read("intersection")
        expected_difference = _read("difference")
        expected = pd.concat(
            [expected_intersection, expected_difference], ignore_index=True, sort=False
        )
        expected["col1"] = expected["col1"].astype(float)
    else:
        expected = _read(how)

    # TODO needed adaptations to result
    if how == "union":
        result = result.sort_values(["col1", "col2"]).reset_index(drop=True)
    elif how == "difference":
        result = result.reset_index(drop=True)

    assert_geodataframe_equal(result, expected, check_column_type=False)

    # for difference also reversed
    if how == "difference":
        result = overlay(df2, df1, how=how)
        result = result.reset_index(drop=True)
        expected = _read("difference-inverse")
        assert_geodataframe_equal(result, expected, check_column_type=False)


@pytest.mark.filterwarnings("ignore:GeoSeries crs mismatch:UserWarning")
def test_overlay_nybb(how):
    polydf = read_file(geopandas.datasets.get_path("nybb"))

    # construct circles dataframe
    N = 10
    b = [int(x) for x in polydf.total_bounds]
    polydf2 = GeoDataFrame(
        [
            {"geometry": Point(x, y).buffer(10000), "value1": x + y, "value2": x - y}
            for x, y in zip(
                range(b[0], b[2], int((b[2] - b[0]) / N)),
                range(b[1], b[3], int((b[3] - b[1]) / N)),
            )
        ],
        crs=polydf.crs,
    )

    result = overlay(polydf, polydf2, how=how)

    cols = ["BoroCode", "BoroName", "Shape_Leng", "Shape_Area", "value1", "value2"]
    if how == "difference":
        cols = cols[:-2]

    # expected result

    if how == "identity":
        # read union one, further down below we take the appropriate subset
        expected = read_file(os.path.join(DATA, "nybb_qgis", "qgis-union.shp"))
    else:
        expected = read_file(
            os.path.join(DATA, "nybb_qgis", "qgis-{0}.shp".format(how))
        )

    # The result of QGIS for 'union' contains incorrect geometries:
    # 24 is a full original circle overlapping with unioned geometries, and
    # 27 is a completely duplicated row)
    if how == "union":
        expected = expected.drop([24, 27])
        expected.reset_index(inplace=True, drop=True)
    # Eliminate observations without geometries (issue from QGIS)
    expected = expected[expected.is_valid]
    expected.reset_index(inplace=True, drop=True)

    if how == "identity":
        expected = expected[expected.BoroCode.notnull()].copy()

    # Order GeoDataFrames
    expected = expected.sort_values(cols).reset_index(drop=True)

    # TODO needed adaptations to result
    result = result.sort_values(cols).reset_index(drop=True)

    if how in ("union", "identity"):
        # concat < 0.23 sorts, so changes the order of the columns
        # but at least we ensure 'geometry' is the last column
        assert result.columns[-1] == "geometry"
        assert len(result.columns) == len(expected.columns)
        result = result.reindex(columns=expected.columns)

    assert_geodataframe_equal(
        result, expected, check_crs=False, check_column_type=False
    )


def test_overlay_overlap(how):
    """
    Overlay test with overlapping geometries in both dataframes.
    Test files are created with::

        import geopandas
        from geopandas import GeoSeries, GeoDataFrame
        from shapely.geometry import Point, Polygon, LineString

        s1 = GeoSeries([Point(0, 0), Point(1.5, 0)]).buffer(1, resolution=2)
        s2 = GeoSeries([Point(1, 1), Point(2, 2)]).buffer(1, resolution=2)

        df1 = GeoDataFrame({'geometry': s1, 'col1':[1,2]})
        df2 = GeoDataFrame({'geometry': s2, 'col2':[1, 2]})

        ax = df1.plot(alpha=0.5)
        df2.plot(alpha=0.5, ax=ax, color='C1')

        df1.to_file('geopandas/geopandas/tests/data/df1_overlap.geojson',
                    driver='GeoJSON')
        df2.to_file('geopandas/geopandas/tests/data/df2_overlap.geojson',
                    driver='GeoJSON')

    and then overlay results are obtained from using  QGIS 2.16
    (Vector -> Geoprocessing Tools -> Intersection / Union / ...),
    saved to GeoJSON.
    """
    df1 = read_file(os.path.join(DATA, "overlap", "df1_overlap.geojson"))
    df2 = read_file(os.path.join(DATA, "overlap", "df2_overlap.geojson"))

    result = overlay(df1, df2, how=how)

    if how == "identity":
        raise pytest.skip()

    expected = read_file(
        os.path.join(DATA, "overlap", "df1_df2_overlap-{0}.geojson".format(how))
    )

    if how == "union":
        # the QGIS result has the last row duplicated, so removing this
        expected = expected.iloc[:-1]

    # TODO needed adaptations to result
    result = result.reset_index(drop=True)
    if how == "union":
        result = result.sort_values(["col1", "col2"]).reset_index(drop=True)

    assert_geodataframe_equal(
        result, expected, check_column_type=False, check_less_precise=True
    )


@pytest.mark.parametrize("other_geometry", [False, True])
def test_geometry_not_named_geometry(dfs, how, other_geometry):
    # Issue #306
    # Add points and flip names
    df1, df2 = dfs
    df3 = df1.copy()
    df3 = df3.rename(columns={"geometry": "polygons"})
    df3 = df3.set_geometry("polygons")
    if other_geometry:
        df3["geometry"] = df1.centroid.geometry
    assert df3.geometry.name == "polygons"

    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df3, df2, how=how)

    assert df3.geometry.name == "polygons"

    if how == "difference":
        # in case of 'difference', column names of left frame are preserved
        assert res2.geometry.name == "polygons"
        if other_geometry:
            assert "geometry" in res2.columns
            assert_geoseries_equal(
                res2["geometry"], df3["geometry"], check_series_type=False
            )
            res2 = res2.drop(["geometry"], axis=1)
        res2 = res2.rename(columns={"polygons": "geometry"})
        res2 = res2.set_geometry("geometry")

    # TODO if existing column is overwritten -> geometry not last column
    if other_geometry and how == "intersection":
        res2 = res2.reindex(columns=res1.columns)
    assert_geodataframe_equal(res1, res2)

    df4 = df2.copy()
    df4 = df4.rename(columns={"geometry": "geom"})
    df4 = df4.set_geometry("geom")
    if other_geometry:
        df4["geometry"] = df2.centroid.geometry
    assert df4.geometry.name == "geom"

    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df1, df4, how=how)
    assert_geodataframe_equal(res1, res2)


def test_bad_how(dfs):
    df1, df2 = dfs
    with pytest.raises(ValueError):
        overlay(df1, df2, how="spandex")


def test_duplicate_column_name(dfs):
    df1, df2 = dfs
    df2r = df2.rename(columns={"col2": "col1"})
    res = overlay(df1, df2r, how="union")
    assert ("col1_1" in res.columns) and ("col1_2" in res.columns)


def test_geoseries_warning(dfs):
    df1, df2 = dfs
    # Issue #305
    with pytest.raises(NotImplementedError):
        overlay(df1, df2.geometry, how="union")


def test_preserve_crs(dfs, how):
    df1, df2 = dfs
    result = overlay(df1, df2, how=how)
    assert result.crs is None
    crs = "epsg:4326"
    df1.crs = crs
    df2.crs = crs
    result = overlay(df1, df2, how=how)
    assert result.crs == crs


def test_empty_intersection(dfs):
    df1, df2 = dfs
    polys3 = GeoSeries(
        [
            Polygon([(-1, -1), (-3, -1), (-3, -3), (-1, -3)]),
            Polygon([(-3, -3), (-5, -3), (-5, -5), (-3, -5)]),
        ]
    )
    df3 = GeoDataFrame({"geometry": polys3, "col3": [1, 2]})
    expected = GeoDataFrame([], columns=["col1", "col3", "geometry"])
    result = overlay(df1, df3)
    assert_geodataframe_equal(result, expected, check_like=True)


def test_correct_index(dfs):
    # GH883 - case where the index was not properly reset
    df1, df2 = dfs
    polys3 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df3 = GeoDataFrame({"geometry": polys3, "col3": [1, 2, 3]})
    i1 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)])
    i2 = Polygon([(3, 3), (3, 5), (5, 5), (5, 3), (3, 3)])
    expected = GeoDataFrame(
        [[1, 1, i1], [3, 2, i2]], columns=["col3", "col2", "geometry"]
    )
    result = overlay(df3, df2)
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    "geom_types", ["polys", "poly_line", "poly_point", "line_poly", "point_poly"]
)
def test_overlay_strict(how, keep_geom_type, geom_types):
    """
    Test of mixed geometry types on input and output. Expected results initially
    generated using following snippet.

        polys1 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df1 = gpd.GeoDataFrame({'col1': [1, 2], 'geometry': polys1})

        polys2 = gpd.GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
        df2 = gpd.GeoDataFrame({'geometry': polys2, 'col2': [1, 2, 3]})

        lines1 = gpd.GeoSeries([LineString([(2, 0), (2, 4), (6, 4)]),
                                LineString([(0, 3), (6, 3)])])
        df3 = gpd.GeoDataFrame({'col3': [1, 2], 'geometry': lines1})
        points1 = gpd.GeoSeries([Point((2, 2)),
                                 Point((3, 3))])
        df4 = gpd.GeoDataFrame({'col4': [1, 2], 'geometry': points1})

        params=["union", "intersection", "difference", "symmetric_difference",
                "identity"]
        stricts = [True, False]

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df2, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('polys_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')

        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df3, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_line_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
        for p in params:
            for s in stricts:
                exp = gpd.overlay(df1, df4, how=p, keep_geom_type=s)
                if not exp.empty:
                    exp.to_file('poly_point_{p}_{s}.geojson'.format(p=p, s=s),
                                driver='GeoJSON')
    """
    polys1 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": polys1})

    polys2 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df2 = GeoDataFrame({"geometry": polys2, "col2": [1, 2, 3]})
    lines1 = GeoSeries(
        [LineString([(2, 0), (2, 4), (6, 4)]), LineString([(0, 3), (6, 3)])]
    )
    df3 = GeoDataFrame({"col3": [1, 2], "geometry": lines1})
    points1 = GeoSeries([Point((2, 2)), Point((3, 3))])
    df4 = GeoDataFrame({"col4": [1, 2], "geometry": points1})

    if geom_types == "polys":
        result = overlay(df1, df2, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == "poly_line":
        result = overlay(df1, df3, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == "poly_point":
        result = overlay(df1, df4, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == "line_poly":
        result = overlay(df3, df1, how=how, keep_geom_type=keep_geom_type)
    elif geom_types == "point_poly":
        result = overlay(df4, df1, how=how, keep_geom_type=keep_geom_type)

    try:
        expected = read_file(
            os.path.join(
                DATA,
                "strict",
                "{t}_{h}_{s}.geojson".format(t=geom_types, h=how, s=keep_geom_type),
            )
        )
        assert_geodataframe_equal(
            result,
            expected,
            check_column_type=False,
            check_less_precise=True,
            check_crs=False,
            check_dtype=False,
        )

    except DriverError:  # fiona >= 1.8
        assert result.empty

    except OSError:  # fiona < 1.8
        assert result.empty


def test_mixed_geom_error():
    polys1 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": polys1})
    mixed = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            LineString([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    dfmixed = GeoDataFrame({"col1": [1, 2], "geometry": mixed})
    with pytest.raises(NotImplementedError):
        overlay(df1, dfmixed, keep_geom_type=True)


def test_keep_geom_type_error():
    gcol = GeoSeries(
        GeometryCollection(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                LineString([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        )
    )
    dfcol = GeoDataFrame({"col1": [2], "geometry": gcol})
    polys1 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df1 = GeoDataFrame({"col1": [1, 2], "geometry": polys1})
    with pytest.raises(TypeError):
        overlay(dfcol, df1, keep_geom_type=True)
