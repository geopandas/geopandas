import os

import numpy as np
import pandas as pd

from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box

import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import HAS_PYPROJ, PANDAS_GE_30

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

try:
    from fiona.errors import DriverError
except ImportError:

    class DriverError(Exception):
        pass


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
        expected = read_file(os.path.join(DATA, "polys", f"df1_df2-{name}.geojson"))
        expected.geometry.array.crs = None
        for col in expected.columns[expected.dtypes == "int32"]:
            expected[col] = expected[col].astype("int64")
        return expected

    if how == "identity":
        expected_intersection = _read("intersection")
        expected_difference = _read("difference")
        expected = pd.concat(
            [expected_intersection, expected_difference], ignore_index=True, sort=False
        )
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


@pytest.mark.parametrize("keep_geom_type", [True, False])
def test_overlay_invalid_input(keep_geom_type):
    invalid_polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
    square = Polygon(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)))
    df1 = geopandas.GeoDataFrame({"A": [1, 2]}, geometry=[invalid_polygon, square])
    df2 = geopandas.GeoDataFrame({"B": [5, 6]}, geometry=[square, square])

    df1_df2 = df1.overlay(df2, keep_geom_type=keep_geom_type)
    # As overlay only supports inputs with uniform geometry types, `make_valid` of
    # `invalid_polygon` should always result in POLYGON EMPTY, so the number of results
    # will should always be 2.
    assert len(df1_df2) == 2
    if keep_geom_type:
        assert df1_df2.geom_type.isin(["Polygon"]).all()


@pytest.mark.filterwarnings("ignore:GeoSeries crs mismatch:UserWarning")
def test_overlay_nybb(how, nybb_filename):
    polydf = read_file(nybb_filename)

    # The circles have been constructed and saved at the time the expected
    # results were created (exact output of buffer algorithm can slightly
    # change over time -> use saved ones)
    # # construct circles dataframe
    # N = 10
    # b = [int(x) for x in polydf.total_bounds]
    # polydf2 = GeoDataFrame(
    #     [
    #         {"geometry": Point(x, y).buffer(10000), "value1": x + y, "value2": x - y}
    #         for x, y in zip(
    #             range(b[0], b[2], int((b[2] - b[0]) / N)),
    #             range(b[1], b[3], int((b[3] - b[1]) / N)),
    #         )
    #     ],
    #     crs=polydf.crs,
    # )
    polydf2 = read_file(os.path.join(DATA, "nybb_qgis", "polydf2.shp"))

    result = overlay(polydf, polydf2, how=how)

    cols = ["BoroCode", "BoroName", "Shape_Leng", "Shape_Area", "value1", "value2"]
    if how == "difference":
        cols = cols[:-2]

    # expected result

    if how == "identity":
        # read union one, further down below we take the appropriate subset
        expected = read_file(os.path.join(DATA, "nybb_qgis", "qgis-union.shp"))
    else:
        expected = read_file(os.path.join(DATA, "nybb_qgis", f"qgis-{how}.shp"))

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
        boro_code_dtype = result["BoroCode"].dtype
        if boro_code_dtype in ("int32", "int64"):
            # Depending on the pandas version, the dtype might be int32 or int64
            expected["BoroCode"] = expected["BoroCode"].astype(boro_code_dtype)

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

    # the ordering of the spatial index results causes slight deviations
    # in the resultant geometries for multipolygons
    # for more details on the discussion, see:
    # https://github.com/geopandas/geopandas/pull/1338
    # https://github.com/geopandas/geopandas/issues/1337

    # Temporary workaround below:

    # simplify multipolygon geometry comparison
    # since the order of the constituent polygons depends on
    # the ordering of spatial indexing results, we cannot
    # compare symmetric_difference results directly when the
    # resultant geometry is a multipolygon

    # first, check that all bounds and areas are approx equal
    # this is a very rough check for multipolygon equality
    kwargs = {}
    pd.testing.assert_series_equal(
        result.geometry.area, expected.geometry.area, **kwargs
    )
    pd.testing.assert_frame_equal(
        result.geometry.bounds, expected.geometry.bounds, **kwargs
    )

    # There are two cases where the multipolygon have a different number
    # of sub-geometries -> not solved by normalize (and thus drop for now)
    if how == "symmetric_difference":
        expected.loc[9, "geometry"] = None
        result.loc[9, "geometry"] = None

    if how == "union":
        expected.loc[24, "geometry"] = None
        result.loc[24, "geometry"] = None

    # missing values get read as None in read_file for a string column, but
    # are introduced as NaN by overlay
    expected["BoroName"] = expected["BoroName"].fillna(np.nan)

    assert_geodataframe_equal(
        result,
        expected,
        normalize=True,
        check_crs=False,
        check_column_type=False,
        check_less_precise=True,
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
        os.path.join(DATA, "overlap", f"df1_df2_overlap-{how}.geojson")
    )

    if how == "union":
        # the QGIS result has the last row duplicated, so removing this
        expected = expected.iloc[:-1]

    # TODO needed adaptations to result
    result = result.reset_index(drop=True)
    if how == "union":
        result = result.sort_values(["col1", "col2"]).reset_index(drop=True)

    assert_geodataframe_equal(
        result,
        expected,
        normalize=True,
        check_column_type=False,
        check_less_precise=True,
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


def test_duplicate_column_name(dfs, how):
    if how == "difference":
        pytest.skip("Difference uses columns from one df only.")
    df1, df2 = dfs
    df2r = df2.rename(columns={"col2": "col1"})
    res = overlay(df1, df2r, how=how)
    assert ("col1_1" in res.columns) and ("col1_2" in res.columns)


def test_geoseries_warning(dfs):
    df1, df2 = dfs
    # Issue #305
    with pytest.raises(NotImplementedError):
        overlay(df1, df2.geometry, how="union")


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
def test_preserve_crs(dfs, how):
    df1, df2 = dfs
    result = overlay(df1, df2, how=how)
    assert result.crs is None
    crs = "epsg:4326"
    df1.crs = crs
    df2.crs = crs
    result = overlay(df1, df2, how=how)
    assert result.crs == crs


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
def test_crs_mismatch(dfs, how):
    df1, df2 = dfs
    df1.crs = 4326
    df2.crs = 3857
    with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
        overlay(df1, df2, how=how)


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
    assert_geodataframe_equal(result, expected, check_dtype=False)


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
    result = overlay(df3, df2, keep_geom_type=True)
    assert_geodataframe_equal(result, expected)


def test_warn_on_keep_geom_type(dfs):
    df1, df2 = dfs
    polys3 = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df3 = GeoDataFrame({"geometry": polys3})

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
        overlay(df2, df3, keep_geom_type=None)


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
                f"{geom_types}_{how}_{keep_geom_type}.geojson",
            )
        )

        # the order depends on the spatial index used
        # so we sort the resultant dataframes to get a consistent order
        # independently of the spatial index implementation
        assert all(expected.columns == result.columns), "Column name mismatch"
        cols = list(set(result.columns) - {"geometry"})
        expected = expected.sort_values(cols, axis=0).reset_index(drop=True)
        result = result.sort_values(cols, axis=0).reset_index(drop=True)

        # some columns are all-NaN in the result, but get read as object dtype
        # column of None values in read_file
        for col in ["col1", "col3", "col4"]:
            if col in expected.columns and expected[col].isna().all():
                expected[col] = expected[col].astype("float64")

        assert_geodataframe_equal(
            result,
            expected,
            normalize=True,
            check_column_type=False,
            check_less_precise=True,
            check_crs=False,
            check_dtype=False,
        )

    except DriverError:  # fiona >= 1.8
        assert result.empty

    except OSError:  # fiona < 1.8
        assert result.empty

    except RuntimeError:  # pyogrio.DataSourceError
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


def test_keep_geom_type_geometry_collection():
    # GH 1581

    df1 = read_file(os.path.join(DATA, "geom_type", "df1.geojson"))
    df2 = read_file(os.path.join(DATA, "geom_type", "df2.geojson"))

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
        intersection = overlay(df1, df2, keep_geom_type=None)
    assert len(intersection) == 1
    assert (intersection.geom_type == "Polygon").all()

    intersection = overlay(df1, df2, keep_geom_type=True)
    assert len(intersection) == 1
    assert (intersection.geom_type == "Polygon").all()

    intersection = overlay(df1, df2, keep_geom_type=False)
    assert len(intersection) == 1
    assert (intersection.geom_type == "GeometryCollection").all()


def test_keep_geom_type_geometry_collection2():
    polys1 = [
        box(0, 0, 1, 1),
        box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
    ]

    polys2 = [
        box(0, 0, 1, 1),
        box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
    ]
    df1 = GeoDataFrame({"left": [0, 1], "geometry": polys1})
    df2 = GeoDataFrame({"right": [0, 1], "geometry": polys2})

    result1 = overlay(df1, df2, keep_geom_type=True)
    expected1 = GeoDataFrame(
        {
            "left": [0, 1],
            "right": [0, 1],
            "geometry": [box(0, 0, 1, 1), box(4, 3, 5, 4)],
        }
    )
    assert_geodataframe_equal(result1, expected1)

    result1 = overlay(df1, df2, keep_geom_type=False)
    expected1 = GeoDataFrame(
        {
            "left": [0, 1, 1],
            "right": [0, 0, 1],
            "geometry": [
                box(0, 0, 1, 1),
                Point(1, 1),
                GeometryCollection([box(4, 3, 5, 4), LineString([(3, 1), (3, 2)])]),
            ],
        }
    )
    assert_geodataframe_equal(result1, expected1)


def test_keep_geom_type_geomcoll_different_types():
    polys1 = [box(0, 1, 1, 3), box(10, 10, 12, 12)]
    polys2 = [
        Polygon([(1, 0), (3, 0), (3, 3), (1, 3), (1, 2), (2, 2), (2, 1), (1, 1)]),
        box(11, 11, 13, 13),
    ]
    df1 = GeoDataFrame({"left": [0, 1], "geometry": polys1})
    df2 = GeoDataFrame({"right": [0, 1], "geometry": polys2})
    result1 = overlay(df1, df2, keep_geom_type=True)
    expected1 = GeoDataFrame(
        {
            "left": [1],
            "right": [1],
            "geometry": [box(11, 11, 12, 12)],
        }
    )
    assert_geodataframe_equal(result1, expected1)

    result2 = overlay(df1, df2, keep_geom_type=False)
    expected2 = GeoDataFrame(
        {
            "left": [0, 1],
            "right": [0, 1],
            "geometry": [
                GeometryCollection([LineString([(1, 2), (1, 3)]), Point(1, 1)]),
                box(11, 11, 12, 12),
            ],
        }
    )
    assert_geodataframe_equal(result2, expected2)


def test_keep_geom_type_geometry_collection_difference():
    # GH 2163

    polys1 = [
        box(0, 0, 1, 1),
        box(1, 1, 2, 2),
    ]

    # the tiny sliver in the second geometry may be converted to a
    # linestring during the overlay process due to floating point errors
    # on some platforms
    polys2 = [
        box(0, 0, 1, 1),
        box(1, 1, 2, 3).union(box(2, 2, 3, 2.00000000000000001)),
    ]
    df1 = GeoDataFrame({"left": [0, 1], "geometry": polys1})
    df2 = GeoDataFrame({"right": [0, 1], "geometry": polys2})

    result1 = overlay(df2, df1, keep_geom_type=True, how="difference")
    expected1 = GeoDataFrame(
        {
            "right": [1],
            "geometry": [box(1, 2, 2, 3)],
        },
    )

    assert_geodataframe_equal(result1, expected1)


@pytest.mark.parametrize("should_make_valid", [True, False])
def test_overlap_make_valid(should_make_valid):
    bowtie = Polygon([(1, 1), (9, 9), (9, 1), (1, 9), (1, 1)])
    assert not bowtie.is_valid
    fixed_bowtie = make_valid(bowtie)
    assert fixed_bowtie.is_valid

    df1 = GeoDataFrame({"col1": ["region"], "geometry": GeoSeries([box(0, 0, 10, 10)])})
    df_bowtie = GeoDataFrame(
        {"col1": ["invalid", "valid"], "geometry": GeoSeries([bowtie, fixed_bowtie])}
    )

    if should_make_valid:
        df_overlay_bowtie = overlay(df1, df_bowtie, make_valid=should_make_valid)
        assert df_overlay_bowtie.at[0, "geometry"].equals(fixed_bowtie)
        assert df_overlay_bowtie.at[1, "geometry"].equals(fixed_bowtie)
    else:
        with pytest.raises(ValueError, match="1 invalid input geometries"):
            overlay(df1, df_bowtie, make_valid=should_make_valid)


def test_empty_overlay_return_non_duplicated_columns(nybb_filename):
    nybb = geopandas.read_file(nybb_filename)
    nybb2 = nybb.copy()
    nybb2.geometry = nybb2.translate(20000000)

    result = geopandas.overlay(nybb, nybb2)

    expected = GeoDataFrame(
        columns=[
            "BoroCode_1",
            "BoroName_1",
            "Shape_Leng_1",
            "Shape_Area_1",
            "BoroCode_2",
            "BoroName_2",
            "Shape_Leng_2",
            "Shape_Area_2",
            "geometry",
        ],
        crs=nybb.crs,
    )
    assert_geodataframe_equal(result, expected, check_dtype=False)


def test_non_overlapping(how):
    p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p2 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    df1 = GeoDataFrame({"id": [1], "col1": [1], "geometry": [p1]})
    df2 = GeoDataFrame({"id": [2], "col2": [2], "geometry": [p2]})
    result = overlay(df1, df2, how=how)

    if how == "intersection":
        expected = GeoDataFrame(
            {
                "id_1": np.array([], dtype="int64"),
                "col1": np.array([], dtype="int64"),
                "id_2": np.array([], dtype="int64"),
                "col2": np.array([], dtype="int64"),
                "geometry": [],
            }
        )
    elif how == "union":
        expected = GeoDataFrame(
            {
                "id_1": [1, np.nan],
                "col1": [1, np.nan],
                "id_2": [np.nan, 2],
                "col2": [np.nan, 2],
                "geometry": [p1, p2],
            }
        )
    elif how == "identity":
        expected = GeoDataFrame(
            {
                "id_1": [1],
                "col1": [1],
                "id_2": [np.nan],
                "col2": [np.nan],
                "geometry": [p1],
            }
        )
    elif how == "symmetric_difference":
        expected = GeoDataFrame(
            {
                "id_1": [1, np.nan],
                "col1": [1, np.nan],
                "id_2": [np.nan, 2],
                "col2": [np.nan, 2],
                "geometry": [p1, p2],
            }
        )
    elif how == "difference":
        expected = GeoDataFrame(
            {
                "id": [1],
                "col1": [1],
                "geometry": [p1],
            }
        )

    assert_geodataframe_equal(result, expected)


def test_no_intersection():
    # overlapping bounds but non-overlapping geometries
    gs = GeoSeries([Point(x, x).buffer(0.1) for x in range(3)])
    gdf1 = GeoDataFrame({"foo": ["a", "b", "c"]}, geometry=gs)
    gdf2 = GeoDataFrame({"bar": ["1", "3", "5"]}, geometry=gs.translate(1))

    expected = GeoDataFrame(columns=["foo", "bar", "geometry"])
    if PANDAS_GE_30 and pd.options.future.infer_string:
        expected = expected.astype({"foo": "str", "bar": "str"})
    result = overlay(gdf1, gdf2, how="intersection")
    assert_geodataframe_equal(result, expected, check_index_type=False)


def test_zero_len():
    # https://github.com/geopandas/geopandas/issues/3422
    gdf1 = GeoDataFrame(
        {
            "geometry": [
                Polygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]])
            ]
        },
        crs=4326,
    )
    # overlay with empty geodataframe shouldn't throw
    gdf2 = GeoDataFrame({"geometry": []}, crs=4326)
    res = gdf1.overlay(gdf2, how="union")
    assert_geodataframe_equal(res, gdf1)

    gdf2 = GeoDataFrame(geometry=[], crs=4326)
    res = gdf1.overlay(gdf2, how="union")
    assert_geodataframe_equal(res, gdf1)


class TestOverlayWikiExample:
    def setup_method(self):
        self.layer_a = GeoDataFrame(data={"id": ["a"]}, geometry=[box(0, 2, 6, 6)])

        self.layer_b = GeoDataFrame(data={"id": ["b"]}, geometry=[box(4, 0, 10, 4)])

        self.intersection = GeoDataFrame(
            data={"id_1": ["a"], "id_2": ["b"]}, geometry=[box(4, 2, 6, 4)]
        )

        self.union = GeoDataFrame(
            data={"id_1": ["a", "a", np.nan], "id_2": ["b", np.nan, "b"]},
            geometry=[
                box(4, 2, 6, 4),
                Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)]),
                Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)]),
            ],
        )

        self.a_difference_b = GeoDataFrame(
            data={"id": ["a"]},
            geometry=[
                Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)])
            ],
        )

        self.b_difference_a = GeoDataFrame(
            data={"id": ["b"]},
            geometry=[
                Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)])
            ],
        )

        self.symmetric_difference = GeoDataFrame(
            data={"id_1": ["a", np.nan], "id_2": [np.nan, "b"]},
            geometry=[
                Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)]),
                Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)]),
            ],
        )

        self.a_identity_b = GeoDataFrame(
            data={"id_1": ["a", "a"], "id_2": ["b", np.nan]},
            geometry=[
                box(4, 2, 6, 4),
                Polygon([(4, 2), (0, 2), (0, 6), (6, 6), (6, 4), (4, 4), (4, 2)]),
            ],
        )

        self.b_identity_a = GeoDataFrame(
            data={"id_1": ["b", "b"], "id_2": ["a", np.nan]},
            geometry=[
                box(4, 2, 6, 4),
                Polygon([(10, 0), (4, 0), (4, 2), (6, 2), (6, 4), (10, 4), (10, 0)]),
            ],
        )

    def test_intersection(self):
        df_result = overlay(self.layer_a, self.layer_b, how="intersection")
        assert_geodataframe_equal(df_result, self.intersection)

    def test_union(self):
        df_result = overlay(self.layer_a, self.layer_b, how="union")
        assert_geodataframe_equal(df_result, self.union)

    def test_a_difference_b(self):
        df_result = overlay(self.layer_a, self.layer_b, how="difference")
        assert_geodataframe_equal(df_result, self.a_difference_b)

    def test_b_difference_a(self):
        df_result = overlay(self.layer_b, self.layer_a, how="difference")
        assert_geodataframe_equal(df_result, self.b_difference_a)

    def test_symmetric_difference(self):
        df_result = overlay(self.layer_a, self.layer_b, how="symmetric_difference")
        assert_geodataframe_equal(df_result, self.symmetric_difference)

    def test_a_identity_b(self):
        df_result = overlay(self.layer_a, self.layer_b, how="identity")
        assert_geodataframe_equal(df_result, self.a_identity_b)

    def test_b_identity_a(self):
        df_result = overlay(self.layer_b, self.layer_a, how="identity")
        assert_geodataframe_equal(df_result, self.b_identity_a)
