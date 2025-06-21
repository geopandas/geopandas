import json
import os
import shutil
import tempfile
from enum import Enum

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, box

import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal


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


@pytest.fixture(
    params=["union", "intersection", "difference", "symmetric_difference", "identity"]
)
def how(request):
    return request.param


@pytest.mark.usefixtures("_setup_class_nybb_filename")
class TestDataFrame:
    def setup_method(self):
        N = 10
        # self.nybb_filename attached via _setup_class_nybb_filename
        self.df = read_file(self.nybb_filename)
        # TODO re-write instance variables to be fixtures
        self.tempdir = tempfile.mkdtemp()
        self.crs = "epsg:4326"
        self.df2 = GeoDataFrame(
            [
                {"geometry": Point(x, y), "value1": x + y, "value2": x * y}
                for x, y in zip(range(N), range(N))
            ],
            crs=self.crs,
        )
        self.df3 = read_file(
            os.path.join(PACKAGE_DIR, "geopandas", "tests", "data", "null_geom.geojson")
        )

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        assert type(self.df2) is GeoDataFrame
        if compat.HAS_PYPROJ:
            assert self.df2.crs == self.crs

    @pytest.mark.skipif(not compat.HAS_PYPROJ, reason="Requires pyproj")
    def test_different_geo_colname(self):
        from pyproj.exceptions import CRSError

        data = {
            "A": range(5),
            "B": range(-5, 0),
            "location": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        df = GeoDataFrame(data, crs=self.crs, geometry="location")
        locs = GeoSeries(data["location"], crs=self.crs)
        assert_geoseries_equal(df.geometry, locs)
        assert "geometry" not in df
        assert df.geometry.name == "location"
        # internal implementation detail
        assert df._geometry_column_name == "location"

        geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
        with pytest.raises(CRSError):
            df.set_geometry(geom2, crs="dummy_crs")

    @pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
    def test_geo_getitem(self):
        data = {
            "A": range(5),
            "B": range(-5, 0),
            "location": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        df = GeoDataFrame(data, crs=self.crs, geometry="location")
        assert isinstance(df.geometry, GeoSeries)
        df["geometry"] = df["A"]
        assert isinstance(df.geometry, GeoSeries)
        assert df.geometry[0] == data["location"][0]
        # good if this changed in the future
        assert not isinstance(df["geometry"], GeoSeries)
        assert isinstance(df["location"], GeoSeries)

        df["buff"] = df.buffer(1)
        assert isinstance(df["buff"], GeoSeries)

        df["array"] = from_shapely([Point(x, y) for x, y in zip(range(5), range(5))])
        assert isinstance(df["array"], GeoSeries)

        data["geometry"] = [Point(x + 1, y - 1) for x, y in zip(range(5), range(5))]
        df = GeoDataFrame(data, crs=self.crs)
        assert isinstance(df.geometry, GeoSeries)
        assert isinstance(df["geometry"], GeoSeries)
        # good if this changed in the future
        assert not isinstance(df["location"], GeoSeries)

    def test_getitem_no_geometry(self):
        res = self.df2[["value1", "value2"]]
        assert isinstance(res, pd.DataFrame)
        assert not isinstance(res, GeoDataFrame)

        # with different name
        df = self.df2.copy()
        df = df.rename(columns={"geometry": "geom"}).set_geometry("geom")
        assert isinstance(df, GeoDataFrame)
        res = df[["value1", "value2"]]
        assert isinstance(res, pd.DataFrame)
        assert not isinstance(res, GeoDataFrame)

        df["geometry"] = np.arange(len(df))
        res = df[["value1", "value2", "geometry"]]
        assert isinstance(res, pd.DataFrame)
        assert not isinstance(res, GeoDataFrame)

    def test_geo_setitem(self):
        data = {
            "A": range(5),
            "B": np.arange(5.0),
            "geometry": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        df = GeoDataFrame(data)
        s = GeoSeries([Point(x, y + 1) for x, y in zip(range(5), range(5))])

        # setting geometry column
        for vals in [s, s.values]:
            df["geometry"] = vals
            assert_geoseries_equal(df["geometry"], s)
            assert_geoseries_equal(df.geometry, s)

        # non-aligned values
        s2 = GeoSeries([Point(x, y + 1) for x, y in zip(range(6), range(6))])
        df["geometry"] = s2
        assert_geoseries_equal(df["geometry"], s)
        assert_geoseries_equal(df.geometry, s)

        # setting other column with geometry values -> preserve geometry type
        for vals in [s, s.values]:
            df["other_geom"] = vals
            assert isinstance(df["other_geom"].values, GeometryArray)

        # overwriting existing non-geometry column -> preserve geometry type
        data = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": range(5),
            "geometry": [Point(x, y) for x, y in zip(range(5), range(5))],
        }
        df = GeoDataFrame(data)
        for vals in [s, s.values]:
            df["other_geom"] = vals
            assert isinstance(df["other_geom"].values, GeometryArray)

    def test_geometry_property(self):
        assert_geoseries_equal(
            self.df.geometry,
            self.df["geometry"],
            check_dtype=True,
            check_index_type=True,
        )

        df = self.df.copy()
        new_geom = [
            Point(x, y) for x, y in zip(range(len(self.df)), range(len(self.df)))
        ]
        df.geometry = new_geom

        new_geom = GeoSeries(new_geom, index=df.index, crs=df.crs)
        assert_geoseries_equal(df.geometry, new_geom)
        assert_geoseries_equal(df["geometry"], new_geom)

        if compat.HAS_PYPROJ:
            # new crs
            gs = new_geom.to_crs(crs="epsg:3857")
            df.geometry = gs
            assert df.crs == "epsg:3857"

    def test_geometry_property_errors(self):
        with pytest.raises(AttributeError):
            df = self.df.copy()
            del df["geometry"]
            df.geometry

        # list-like error
        with pytest.raises(ValueError):
            df = self.df2.copy()
            df.geometry = "value1"

        # list-like error
        with pytest.raises(ValueError):
            df = self.df.copy()
            df.geometry = "apple"

        # non-geometry error
        with pytest.raises(TypeError):
            df = self.df.copy()
            df.geometry = list(range(df.shape[0]))

        with pytest.raises(KeyError):
            df = self.df.copy()
            del df["geometry"]
            df["geometry"]

        # ndim error
        with pytest.raises(ValueError):
            df = self.df.copy()
            df.geometry = df

    def test_rename_geometry(self):
        assert self.df.geometry.name == "geometry"
        df2 = self.df.rename_geometry("new_name")
        assert df2.geometry.name == "new_name"
        df2 = self.df.rename_geometry("new_name", inplace=True)
        assert df2 is None
        assert self.df.geometry.name == "new_name"

        # existing column error
        msg = "Column named Shape_Area already exists"
        with pytest.raises(ValueError, match=msg):
            df2 = self.df.rename_geometry("Shape_Area")
        with pytest.raises(ValueError, match=msg):
            self.df.rename_geometry("Shape_Area", inplace=True)

    def test_set_geometry(self):
        geom = GeoSeries([Point(x, y) for x, y in zip(range(5), range(5))])
        original_geom = self.df.geometry

        df2 = self.df.set_geometry(geom)
        assert self.df is not df2
        assert_geoseries_equal(df2.geometry, geom, check_crs=False)
        assert_geoseries_equal(self.df.geometry, original_geom)
        assert_geoseries_equal(self.df["geometry"], self.df.geometry)
        # unknown column
        with pytest.raises(ValueError):
            self.df.set_geometry("nonexistent-column")

        # ndim error
        with pytest.raises(ValueError):
            self.df.set_geometry(self.df)

    @pytest.mark.skipif(not compat.HAS_PYPROJ, reason="Requires pyproj")
    def test_set_geometry_crs(self):
        geom = GeoSeries([Point(x, y) for x, y in zip(range(5), range(5))])

        # new crs - setting should default to GeoSeries' crs
        gs = GeoSeries(geom, crs="epsg:3857")
        new_df = self.df.set_geometry(gs)
        assert new_df.crs == "epsg:3857"

        # explicit crs overrides self and dataframe
        new_df = self.df.set_geometry(gs, crs="epsg:26909")
        assert new_df.crs == "epsg:26909"
        assert new_df.geometry.crs == "epsg:26909"

        # Series should use dataframe's
        new_df = self.df.set_geometry(geom.values)
        assert new_df.crs == self.df.crs
        assert new_df.geometry.crs == self.df.crs

    def test_set_geometry_col(self):
        g = self.df.geometry
        g_simplified = g.simplify(100)
        self.df["simplified_geometry"] = g_simplified
        df2 = self.df.set_geometry("simplified_geometry")

        # Drop is false by default
        assert "simplified_geometry" in df2
        assert_geoseries_equal(df2.geometry, g_simplified)

        # If True, drops column and renames to geometry
        with pytest.warns(FutureWarning):
            df3 = self.df.set_geometry("simplified_geometry", drop=True)
        assert "simplified_geometry" not in df3
        assert_geoseries_equal(df3.geometry, g_simplified)

    def test_set_geometry_inplace(self):
        geom = [Point(x, y) for x, y in zip(range(5), range(5))]
        ret = self.df.set_geometry(geom, inplace=True)
        assert ret is None
        geom = GeoSeries(geom, index=self.df.index, crs=self.df.crs)
        assert_geoseries_equal(self.df.geometry, geom)

    def test_set_geometry_series(self):
        # Test when setting geometry with a Series that
        # alignment will occur
        #
        # Reverse the index order
        # Set the Series to be Point(i,i) where i is the index
        self.df.index = range(len(self.df) - 1, -1, -1)

        d = {}
        for i in range(len(self.df)):
            d[i] = Point(i, i)
        g = GeoSeries(d)
        # At this point, the DataFrame index is [4,3,2,1,0] and the
        # GeoSeries index is [0,1,2,3,4]. Make sure set_geometry aligns
        # them to match indexes
        df = self.df.set_geometry(g)

        for i, r in df.iterrows():
            assert i == r["geometry"].x
            assert i == r["geometry"].y

    def test_set_geometry_empty(self):
        df = pd.DataFrame(columns=["a", "geometry"], index=pd.DatetimeIndex([]))
        result = df.set_geometry("geometry")
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_set_geometry_np_int(self):
        self.df.loc[:, 0] = self.df.geometry
        df = self.df.set_geometry(np.int64(0))
        assert df.geometry.name == 0

    def test_get_geometry_invalid(self):
        df = GeoDataFrame()
        # no column "geometry" ever added
        df["geom"] = self.df.geometry
        msg_geo_col_none = "active geometry column to use has not been set. "

        with pytest.raises(AttributeError, match=msg_geo_col_none):
            df.geometry
        # "geometry" originally present but dropped (but still a gdf)
        col_subset_drop_geometry = ["BoroCode", "BoroName", "geom2"]
        df2 = self.df.copy().assign(geom2=self.df.geometry)[col_subset_drop_geometry]
        with pytest.raises(AttributeError, match="is not present."):
            df2.geometry

        msg_other_geo_cols_present = "There are columns with geometry data type"
        msg_no_other_geo_cols = "There are no existing columns with geometry data type"
        with pytest.raises(AttributeError, match=msg_other_geo_cols_present):
            df2.geometry

        with pytest.raises(AttributeError, match=msg_no_other_geo_cols):
            GeoDataFrame().geometry

    def test_get_geometry_geometry_inactive(self):
        # https://github.com/geopandas/geopandas/issues/2574
        df = self.df.assign(geom2=self.df.geometry).set_geometry("geom2")
        df = df.loc[:, ["BoroName", "geometry"]]
        assert df._geometry_column_name == "geom2"
        msg_geo_col_missing = "is not present. "
        # Check that df.geometry raises if active geometry column is missing,
        # it should not fall back to column named "geometry"
        with pytest.raises(AttributeError, match=msg_geo_col_missing):
            df.geometry

    @pytest.mark.skipif(not compat.HAS_PYPROJ, reason="Requires pyproj")
    def test_override_existing_crs_warning(self):
        with pytest.warns(
            DeprecationWarning,
            match="Overriding the CRS of a GeoSeries that already has CRS",
        ):
            self.df.geometry.crs = "epsg:2100"

        with pytest.warns(
            DeprecationWarning,
            match="Overriding the CRS of a GeoDataFrame that already has CRS",
        ):
            self.df.crs = "epsg:4326"

    def test_active_geometry_name(self):
        # default single active called "geometry"
        assert self.df.active_geometry_name == "geometry"

        # one GeoSeries, not active
        no_active = GeoDataFrame({"foo": self.df.BoroName, "bar": self.df.geometry})
        assert no_active.active_geometry_name is None
        assert no_active.set_geometry("bar").active_geometry_name == "bar"

        # multiple, none active
        multiple = GeoDataFrame({"foo": self.df.geometry, "bar": self.df.geometry})
        assert multiple.active_geometry_name is None
        assert multiple.set_geometry("foo").active_geometry_name == "foo"
        assert multiple.set_geometry("bar").active_geometry_name == "bar"

    def test_align(self):
        df = self.df2

        res1, res2 = df.align(df)
        assert_geodataframe_equal(res1, df)
        assert_geodataframe_equal(res2, df)

        res1, res2 = df.align(df.copy())
        assert_geodataframe_equal(res1, df)
        assert_geodataframe_equal(res2, df)

        if compat.HAS_PYPROJ:
            # assert crs is / is not preserved on mixed dataframes
            df_nocrs = df.copy().set_crs(None, allow_override=True)
            res1, res2 = df.align(df_nocrs)
            assert_geodataframe_equal(res1, df)
            assert res1.crs is not None
            assert_geodataframe_equal(res2, df_nocrs)
            assert res2.crs is None

        # mixed GeoDataFrame / DataFrame
        df_nogeom = pd.DataFrame(df.drop("geometry", axis=1))
        res1, res2 = df.align(df_nogeom, axis=0)
        assert_geodataframe_equal(res1, df)
        assert type(res2) is pd.DataFrame
        assert_frame_equal(res2, df_nogeom)

        # same as above but now with actual alignment
        df1 = df.iloc[1:].copy()
        df2 = df.iloc[:-1].copy()

        exp1 = df.copy()
        exp1.iloc[0] = np.nan
        exp2 = df.copy()
        exp2.iloc[-1] = np.nan
        res1, res2 = df1.align(df2)
        assert_geodataframe_equal(res1, exp1)
        assert_geodataframe_equal(res2, exp2)

        if compat.HAS_PYPROJ:
            df2_nocrs = df2.copy().set_crs(None, allow_override=True)
            exp2_nocrs = exp2.copy().set_crs(None, allow_override=True)
            res1, res2 = df1.align(df2_nocrs)
            assert_geodataframe_equal(res1, exp1)
            assert res1.crs is not None
            assert_geodataframe_equal(res2, exp2_nocrs)
            assert res2.crs is None

        df2_nogeom = pd.DataFrame(df2.drop("geometry", axis=1))
        exp2_nogeom = pd.DataFrame(exp2.drop("geometry", axis=1))
        res1, res2 = df1.align(df2_nogeom, axis=0)
        assert_geodataframe_equal(res1, exp1)
        assert type(res2) is pd.DataFrame
        assert_frame_equal(res2, exp2_nogeom)

    @pytest.mark.skipif(not compat.HAS_PYPROJ, reason="Requires pyproj")
    def test_to_json(self):
        text = self.df.to_json(to_wgs84=True)
        data = json.loads(text)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 5
        assert "id" in data["features"][0].keys()

        # check it converts to WGS84
        coord = data["features"][0]["geometry"]["coordinates"][0][0][0]
        np.testing.assert_allclose(coord, [-74.0505080640324, 40.5664220341941])

    def test_to_json_wgs84_false(self):
        text = self.df.to_json()
        data = json.loads(text)
        # check it doesn't convert to WGS84
        coord = data["features"][0]["geometry"]["coordinates"][0][0][0]
        assert coord == [970217.0223999023, 145643.33221435547]

    def test_to_json_no_crs(self):
        self.df.geometry.array.crs = None
        with pytest.raises(ValueError, match="CRS is not set"):
            self.df.to_json(to_wgs84=True)

    @pytest.mark.filterwarnings(
        "ignore:Geometry column does not contain geometry:UserWarning"
    )
    def test_to_json_geom_col(self):
        df = self.df.copy()
        df["geom"] = df["geometry"]
        df["geometry"] = np.arange(len(df))
        df.set_geometry("geom", inplace=True)

        text = df.to_json()
        data = json.loads(text)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 5

    def test_to_json_only_geom_column(self):
        text = self.df[["geometry"]].to_json()
        data = json.loads(text)
        assert len(data["features"]) == 5
        assert "id" in data["features"][0].keys()

    def test_to_json_na(self):
        # Set a value as nan and make sure it's written
        self.df.loc[self.df["BoroName"] == "Queens", "Shape_Area"] = np.nan

        text = self.df.to_json()
        data = json.loads(text)
        assert len(data["features"]) == 5
        for f in data["features"]:
            props = f["properties"]
            assert len(props) == 4
            if props["BoroName"] == "Queens":
                assert props["Shape_Area"] is None

    def test_to_json_bad_na(self):
        # Check that a bad na argument raises error
        with pytest.raises(ValueError):
            self.df.to_json(na="garbage")

    def test_to_json_dropna(self):
        self.df.loc[self.df["BoroName"] == "Queens", "Shape_Area"] = np.nan
        self.df.loc[self.df["BoroName"] == "Bronx", "Shape_Leng"] = np.nan

        text = self.df.to_json(na="drop")
        data = json.loads(text)
        assert len(data["features"]) == 5
        for f in data["features"]:
            props = f["properties"]
            if props["BoroName"] == "Queens":
                assert len(props) == 3
                assert "Shape_Area" not in props
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                assert "Shape_Leng" in props
            elif props["BoroName"] == "Bronx":
                assert len(props) == 3
                assert "Shape_Leng" not in props
                assert "Shape_Area" in props
            else:
                assert len(props) == 4

    def test_to_json_keepna(self):
        self.df.loc[self.df["BoroName"] == "Queens", "Shape_Area"] = np.nan
        self.df.loc[self.df["BoroName"] == "Bronx", "Shape_Leng"] = np.nan

        text = self.df.to_json(na="keep")
        data = json.loads(text)
        assert len(data["features"]) == 5
        for f in data["features"]:
            props = f["properties"]
            assert len(props) == 4
            if props["BoroName"] == "Queens":
                assert np.isnan(props["Shape_Area"])
                # Just make sure setting it to nan in a different row
                # doesn't affect this one
                assert "Shape_Leng" in props
            elif props["BoroName"] == "Bronx":
                assert np.isnan(props["Shape_Leng"])
                assert "Shape_Area" in props

    def test_to_json_drop_id(self):
        text = self.df.to_json(drop_id=True)
        data = json.loads(text)
        assert len(data["features"]) == 5
        for f in data["features"]:
            assert "id" not in f.keys()

    def test_to_json_drop_id_only_geom_column(self):
        text = self.df[["geometry"]].to_json(drop_id=True)
        data = json.loads(text)
        assert len(data["features"]) == 5
        for f in data["features"]:
            assert "id" not in f.keys()

    def test_to_json_with_duplicate_columns(self):
        df = GeoDataFrame(
            data=[[1, 2, 3]], columns=["a", "b", "a"], geometry=[Point(1, 1)]
        )
        with pytest.raises(
            ValueError, match="GeoDataFrame cannot contain duplicated column names."
        ):
            df.to_json()

    def test_copy(self):
        df2 = self.df.copy()
        assert type(df2) is GeoDataFrame
        assert self.df.crs == df2.crs

    def test_empty_copy(self):
        # https://github.com/geopandas/geopandas/issues/2765
        df = GeoDataFrame()
        df2 = df.copy()
        assert type(df2) is GeoDataFrame
        df3 = df.copy(deep=True)
        assert type(df3) is GeoDataFrame

    def test_no_geom_copy(self):
        df = GeoDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
        assert type(df) is GeoDataFrame
        assert type(df.copy()) is GeoDataFrame

    def test_empty(self):
        df = GeoDataFrame({"geometry": []})
        assert df.geometry.dtype == "geometry"
        df = GeoDataFrame({"a": []}, geometry="a")
        assert df.geometry.dtype == "geometry"
        df = GeoDataFrame(geometry=[])
        assert df.geometry.dtype == "geometry"

    def test_bool_index(self):
        # Find boros with 'B' in their name
        df = self.df[self.df["BoroName"].str.contains("B")]
        assert len(df) == 2
        boros = df["BoroName"].values
        assert "Brooklyn" in boros
        assert "Bronx" in boros
        assert type(df) is GeoDataFrame

    def test_coord_slice_points(self):
        assert self.df2.cx[-2:-1, -2:-1].empty
        assert_frame_equal(self.df2, self.df2.cx[:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, :])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[:, 5:])
        assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, 5:])

    def test_from_dict(self):
        data = {"A": [1], "geometry": [Point(0.0, 0.0)]}
        df = GeoDataFrame.from_dict(data, crs=3857)
        if compat.HAS_PYPROJ:
            assert df.crs == "epsg:3857"
        else:
            assert df.crs is None
        assert df._geometry_column_name == "geometry"

        data = {"B": [1], "location": [Point(0.0, 0.0)]}
        df = GeoDataFrame.from_dict(data, geometry="location")
        assert df._geometry_column_name == "location"

    def test_from_features(self, nybb_filename):
        fiona = pytest.importorskip("fiona")
        with fiona.open(nybb_filename) as f:
            features = list(f)
            crs = f.crs_wkt

        df = GeoDataFrame.from_features(features, crs=crs)
        validate_boro_df(df, case_sensitive=True)
        if compat.HAS_PYPROJ:
            assert df.crs == crs
        else:
            assert df.crs is None

    def test_from_features_unaligned_properties(self):
        p1 = Point(1, 1)
        f1 = {
            "type": "Feature",
            "properties": {"a": 0},
            "geometry": p1.__geo_interface__,
        }

        p2 = Point(2, 2)
        f2 = {
            "type": "Feature",
            "properties": {"b": 1},
            "geometry": p2.__geo_interface__,
        }

        p3 = Point(3, 3)
        f3 = {
            "type": "Feature",
            "properties": None,
            "geometry": p3.__geo_interface__,
        }

        df = GeoDataFrame.from_features([f1, f2, f3])

        result = df[["a", "b"]]
        expected = pd.DataFrame.from_dict(
            [{"a": 0, "b": np.nan}, {"a": np.nan, "b": 1}, {"a": np.nan, "b": np.nan}]
        )
        assert_frame_equal(expected, result)

    def test_from_features_empty_properties(self):
        geojson_properties_object = """{
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": {},
              "geometry": {
                "type": "Polygon",
                "coordinates": [
                  [
                    [
                      11.3456529378891,
                      46.49461446367692
                    ],
                    [
                      11.345674395561216,
                      46.494097442978195
                    ],
                    [
                      11.346918940544128,
                      46.49385370294394
                    ],
                    [
                      11.347616314888,
                      46.4938352377453
                    ],
                    [
                      11.347514390945435,
                      46.49466985846028
                    ],
                    [
                      11.3456529378891,
                      46.49461446367692
                    ]
                  ]
                ]
              }
            }
          ]
        }"""

        geojson_properties_null = """{
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": null,
              "geometry": {
                "type": "Polygon",
                "coordinates": [
                  [
                    [
                      11.3456529378891,
                      46.49461446367692
                    ],
                    [
                      11.345674395561216,
                      46.494097442978195
                    ],
                    [
                      11.346918940544128,
                      46.49385370294394
                    ],
                    [
                      11.347616314888,
                      46.4938352377453
                    ],
                    [
                      11.347514390945435,
                      46.49466985846028
                    ],
                    [
                      11.3456529378891,
                      46.49461446367692
                    ]
                  ]
                ]
              }
            }
          ]
        }"""

        # geoJSON with empty properties
        gjson_po = json.loads(geojson_properties_object)
        gdf1 = GeoDataFrame.from_features(gjson_po)

        # geoJSON with null properties
        gjson_null = json.loads(geojson_properties_null)
        gdf2 = GeoDataFrame.from_features(gjson_null)

        assert_frame_equal(gdf1, gdf2)

    def test_from_features_geom_interface_feature(self):
        class Placemark:
            def __init__(self, geom, val):
                self.__geo_interface__ = {
                    "type": "Feature",
                    "properties": {"a": val},
                    "geometry": geom.__geo_interface__,
                }

        p1 = Point(1, 1)
        f1 = Placemark(p1, 0)
        p2 = Point(3, 3)
        f2 = Placemark(p2, 0)
        df = GeoDataFrame.from_features([f1, f2])
        assert sorted(df.columns) == ["a", "geometry"]
        assert df.geometry.tolist() == [p1, p2]

    def test_from_feature_collection(self):
        data = {
            "name": ["a", "b", "c"],
            "lat": [45, 46, 47.5],
            "lon": [-120, -121.2, -122.9],
        }

        df = pd.DataFrame(data)
        geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
        gdf = GeoDataFrame(df, geometry=geometry)
        # from_features returns sorted columns
        expected = gdf[["geometry", "name", "lat", "lon"]]

        # test FeatureCollection
        res = GeoDataFrame.from_features(gdf.__geo_interface__)
        assert_frame_equal(res, expected)

        # test list of Features
        res = GeoDataFrame.from_features(gdf.__geo_interface__["features"])
        assert_frame_equal(res, expected)

        # test __geo_interface__ attribute (a GeoDataFrame has one)
        res = GeoDataFrame.from_features(gdf)
        assert_frame_equal(res, expected)

    def test_dataframe_to_geodataframe(self):
        df = pd.DataFrame(
            {"A": range(len(self.df)), "location": np.array(self.df.geometry)},
            index=self.df.index,
        )
        gf = df.set_geometry("location", crs=self.df.crs)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(gf, GeoDataFrame)
        assert_geoseries_equal(gf.geometry, self.df.geometry)
        assert gf.geometry.name == "location"
        assert "geometry" not in gf

        with pytest.warns(FutureWarning):
            gf2 = df.set_geometry("location", crs=self.df.crs, drop=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(gf2, GeoDataFrame)
        assert gf2.geometry.name == "geometry"
        assert "geometry" in gf2
        assert "location" not in gf2
        assert "location" in df

        # should be a copy
        df.loc[0, "A"] = 100
        assert gf.loc[0, "A"] == 0
        assert gf2.loc[0, "A"] == 0

        with pytest.raises(ValueError):
            df.set_geometry("location", inplace=True)

    def test_dataframe_not_manipulated(self):
        df = pd.DataFrame(
            {
                "A": range(len(self.df)),
                "latitude": self.df.geometry.centroid.y,
                "longitude": self.df.geometry.centroid.x,
            },
            index=self.df.index,
        )
        df_copy = df.copy()
        gf = GeoDataFrame(
            df,
            geometry=points_from_xy(df["longitude"], df["latitude"]),
            crs=self.df.crs,
        )
        assert type(df) is pd.DataFrame
        assert "geometry" not in df
        assert_frame_equal(df, df_copy)
        assert isinstance(gf, GeoDataFrame)
        assert hasattr(gf, "geometry")

        # ensure mutating columns in gf doesn't update df
        gf.loc[0, "A"] = 7
        assert_frame_equal(df, df_copy)
        gf["A"] = 3
        assert_frame_equal(df, df_copy)

    def test_geodataframe_geointerface(self):
        assert self.df.__geo_interface__["type"] == "FeatureCollection"
        assert len(self.df.__geo_interface__["features"]) == self.df.shape[0]

    def test_geodataframe_iterfeatures(self):
        df = self.df.iloc[:1].copy()
        df.loc[0, "BoroName"] = np.nan
        # when containing missing values
        # null: output the missing entries as JSON null
        result = next(iter(df.iterfeatures(na="null")))["properties"]
        assert result["BoroName"] is None
        # drop: remove the property from the feature.
        result = next(iter(df.iterfeatures(na="drop")))["properties"]
        assert "BoroName" not in result.keys()
        # keep: output the missing entries as NaN
        result = next(iter(df.iterfeatures(na="keep")))["properties"]
        assert np.isnan(result["BoroName"])

        # test for checking that the (non-null) features are python scalars and
        # not numpy scalars
        assert type(df.loc[0, "Shape_Leng"]) is np.float64
        # null
        result = next(iter(df.iterfeatures(na="null")))
        assert isinstance(result["properties"]["Shape_Leng"], float)
        # drop
        result = next(iter(df.iterfeatures(na="drop")))
        assert isinstance(result["properties"]["Shape_Leng"], float)
        # keep
        result = next(iter(df.iterfeatures(na="keep")))
        assert isinstance(result["properties"]["Shape_Leng"], float)

        # when only having numerical columns
        df_only_numerical_cols = df[["Shape_Leng", "Shape_Area", "geometry"]]
        assert type(df_only_numerical_cols.loc[0, "Shape_Leng"]) is np.float64
        # null
        result = next(iter(df_only_numerical_cols.iterfeatures(na="null")))
        assert isinstance(result["properties"]["Shape_Leng"], float)
        # drop
        result = next(iter(df_only_numerical_cols.iterfeatures(na="drop")))
        assert isinstance(result["properties"]["Shape_Leng"], float)
        # keep
        result = next(iter(df_only_numerical_cols.iterfeatures(na="keep")))
        assert isinstance(result["properties"]["Shape_Leng"], float)

        with pytest.raises(
            ValueError, match="GeoDataFrame cannot contain duplicated column names."
        ):
            df_with_duplicate_columns = df[
                ["Shape_Leng", "Shape_Leng", "Shape_Area", "geometry"]
            ]
            list(df_with_duplicate_columns.iterfeatures())

        # geometry not set
        df = GeoDataFrame({"values": [0, 1], "geom": [Point(0, 1), Point(1, 0)]})
        with pytest.raises(AttributeError):
            list(df.iterfeatures())

    def test_geodataframe_iterfeatures_non_scalars(self):
        # When some features in geodataframe are non-scalar values
        df = GeoDataFrame(
            {"geometry": [Point(1, 2)], "non-scalar": [[1, 2]], "test_col": None}
        )
        # null
        expected = {"non-scalar": [1, 2], "test_col": None}
        result = next(iter(df.iterfeatures(na="null"))).get("properties")
        assert expected == result
        # drop
        expected = {"non-scalar": [1, 2]}
        result = next(iter(df.iterfeatures(na="drop"))).get("properties")
        assert expected == result
        # keep
        expected = {"non-scalar": [1, 2], "test_col": None}
        result = next(iter(df.iterfeatures(na="keep"))).get("properties")
        assert expected == result

    def test_geodataframe_geojson_no_bbox(self):
        geo = self.df.to_geo_dict(na="null", show_bbox=False)
        assert "bbox" not in geo.keys()
        for feature in geo["features"]:
            assert "bbox" not in feature.keys()

    def test_geodataframe_geojson_bbox(self):
        geo = self.df.to_geo_dict(na="null", show_bbox=True)
        assert "bbox" in geo.keys()
        assert len(geo["bbox"]) == 4
        assert isinstance(geo["bbox"], tuple)
        for bound in geo["bbox"]:
            assert not isinstance(bound, np.float64)
        for feature in geo["features"]:
            assert "bbox" in feature.keys()

    def test_pickle(self):
        import pickle

        df2 = pickle.loads(pickle.dumps(self.df))
        assert_geodataframe_equal(self.df, df2)

    def test_pickle_method(self):
        filename = os.path.join(self.tempdir, "df.pkl")
        self.df.to_pickle(filename)
        unpickled = pd.read_pickle(filename)
        assert_frame_equal(self.df, unpickled)
        assert self.df.crs == unpickled.crs

    def test_estimate_utm_crs(self):
        pyproj = pytest.importorskip("pyproj")

        assert self.df.estimate_utm_crs() == pyproj.CRS("EPSG:32618")
        assert self.df.estimate_utm_crs("NAD83") == pyproj.CRS("EPSG:26918")

    def test_to_wkb(self):
        wkbs0 = [
            (  # POINT (0 0)
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            ),
            (  # POINT (1 1)
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
            ),
        ]
        wkbs1 = [
            (  # POINT (2 2)
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00@\x00\x00\x00\x00\x00\x00\x00@"
            ),
            (  # POINT (3 3)
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x08@\x00\x00\x00\x00\x00\x00\x08@"
            ),
        ]
        gs0 = GeoSeries.from_wkb(wkbs0)
        gs1 = GeoSeries.from_wkb(wkbs1)
        gdf = GeoDataFrame({"geom_col0": gs0, "geom_col1": gs1})

        expected_df = pd.DataFrame({"geom_col0": wkbs0, "geom_col1": wkbs1})
        assert_frame_equal(expected_df, gdf.to_wkb())

    def test_to_wkt(self):
        wkts0 = ["POINT (0 0)", "POINT (1 1)"]
        wkts1 = ["POINT (2 2)", "POINT (3 3)"]
        gs0 = GeoSeries.from_wkt(wkts0)
        gs1 = GeoSeries.from_wkt(wkts1)
        gdf = GeoDataFrame({"gs0": gs0, "gs1": gs1})

        expected_df = pd.DataFrame({"gs0": wkts0, "gs1": wkts1})
        assert_frame_equal(expected_df, gdf.to_wkt())

    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    @pytest.mark.parametrize("predicate", ["intersects", "within", "contains"])
    def test_sjoin(self, how, predicate, naturalearth_cities, naturalearth_lowres):
        """
        Basic test for availability of the GeoDataFrame method. Other
        sjoin tests are located in /tools/tests/test_sjoin.py
        """
        left = read_file(naturalearth_cities)
        right = read_file(naturalearth_lowres)

        expected = geopandas.sjoin(left, right, how=how, predicate=predicate)
        result = left.sjoin(right, how=how, predicate=predicate)
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    @pytest.mark.parametrize("distance", [0, 3])
    @pytest.mark.skipif(
        not compat.GEOS_GE_310,
        reason="`dwithin` requires GEOS 3.10",
    )
    def test_sjoin_dwithin(self, how, distance):
        """
        Basic test for predicate='dwithin' availability of the GeoDataFrame method.
        Other sjoin tests are located in /tools/tests/test_sjoin.py
        """
        left = GeoDataFrame(geometry=points_from_xy([0, 1, 2], [0, 1, 1]))
        right = GeoDataFrame(geometry=[box(0, 0, 1, 1)])

        expected = geopandas.sjoin(
            left, right, how=how, predicate="dwithin", distance=distance
        )
        result = left.sjoin(right, how=how, predicate="dwithin", distance=distance)
        assert_geodataframe_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "inner", "right"])
    @pytest.mark.parametrize("max_distance", [None, 1])
    @pytest.mark.parametrize("distance_col", [None, "distance"])
    @pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS:UserWarning")
    def test_sjoin_nearest(
        self, how, max_distance, distance_col, naturalearth_cities, naturalearth_lowres
    ):
        """
        Basic test for availability of the GeoDataFrame method. Other
        sjoin tests are located in /tools/tests/test_sjoin.py
        """
        left = read_file(naturalearth_cities)
        right = read_file(naturalearth_lowres)

        expected = geopandas.sjoin_nearest(
            left, right, how=how, max_distance=max_distance, distance_col=distance_col
        )
        result = left.sjoin_nearest(
            right, how=how, max_distance=max_distance, distance_col=distance_col
        )
        assert_geodataframe_equal(result, expected)

    def test_clip(self, naturalearth_cities, naturalearth_lowres):
        """
        Basic test for availability of the GeoDataFrame method. Other
        clip tests are located in /tools/tests/test_clip.py
        """
        left = read_file(naturalearth_cities)
        world = read_file(naturalearth_lowres)
        south_america = world[world["continent"] == "South America"]

        expected = geopandas.clip(left, south_america)
        result = left.clip(south_america)
        assert_geodataframe_equal(result, expected)

    def test_clip_sorting(self, naturalearth_cities, naturalearth_lowres):
        """
        Test sorting of geodataframe when clipping.
        """
        cities = read_file(naturalearth_cities)
        world = read_file(naturalearth_lowres)
        south_america = world[world["continent"] == "South America"]

        unsorted_clipped_cities = geopandas.clip(cities, south_america, sort=False)
        sorted_clipped_cities = geopandas.clip(cities, south_america, sort=True)

        expected_sorted_index = pd.Index(
            [55, 59, 62, 88, 101, 114, 122, 169, 181, 189, 210, 230, 236, 238, 239]
        )

        assert not (
            sorted(unsorted_clipped_cities.index) == unsorted_clipped_cities.index
        ).all()
        assert (
            sorted(sorted_clipped_cities.index) == sorted_clipped_cities.index
        ).all()
        assert_index_equal(expected_sorted_index, sorted_clipped_cities.index)

    def test_overlay(self, dfs, how):
        """
        Basic test for availability of the GeoDataFrame method. Other
        overlay tests are located in tests/test_overlay.py
        """
        df1, df2 = dfs

        expected = geopandas.overlay(df1, df2, how=how)
        result = df1.overlay(df2, how=how)
        assert_geodataframe_equal(result, expected)


def check_geodataframe(df, geometry_column="geometry"):
    assert isinstance(df, GeoDataFrame)
    assert isinstance(df.geometry, GeoSeries)
    assert isinstance(df[geometry_column], GeoSeries)
    assert df._geometry_column_name == geometry_column
    assert df.geometry.name == geometry_column
    assert isinstance(df.geometry.values, GeometryArray)
    assert isinstance(df.geometry.dtype, GeometryDtype)


class TestConstructor:
    def test_dict(self):
        data = {
            "A": range(3),
            "B": np.arange(3.0),
            "geometry": [Point(x, x) for x in range(3)],
        }
        df = GeoDataFrame(data)
        check_geodataframe(df)

        # with specifying other kwargs
        df = GeoDataFrame(data, index=list("abc"))
        check_geodataframe(df)
        assert_index_equal(df.index, pd.Index(list("abc")))

        df = GeoDataFrame(data, columns=["B", "A", "geometry"])
        check_geodataframe(df)
        assert_index_equal(df.columns, pd.Index(["B", "A", "geometry"]))

        df = GeoDataFrame(data, columns=["A", "geometry"])
        check_geodataframe(df)
        assert_index_equal(df.columns, pd.Index(["A", "geometry"]))
        assert_series_equal(df["A"], pd.Series(range(3), name="A"))

    def test_dict_of_series(self):
        data = {
            "A": pd.Series(range(3)),
            "B": pd.Series(np.arange(3.0)),
            "geometry": GeoSeries([Point(x, x) for x in range(3)]),
        }

        df = GeoDataFrame(data)
        check_geodataframe(df)

        df = GeoDataFrame(data, index=pd.Index([1, 2]))
        check_geodataframe(df)
        assert_index_equal(df.index, pd.Index([1, 2]))
        assert df["A"].tolist() == [1, 2]

        # one non-series -> length is not correct
        data = {
            "A": pd.Series(range(3)),
            "B": np.arange(3.0),
            "geometry": GeoSeries([Point(x, x) for x in range(3)]),
        }
        with pytest.raises(ValueError):
            GeoDataFrame(data, index=[1, 2])

    def test_dict_specified_geometry(self):
        data = {
            "A": range(3),
            "B": np.arange(3.0),
            "other_geom": [Point(x, x) for x in range(3)],
        }

        df = GeoDataFrame(data, geometry="other_geom")
        check_geodataframe(df, "other_geom")

        with pytest.raises(ValueError):
            df = GeoDataFrame(data, geometry="geometry")

        # when no geometry specified -> works but raises error once
        # trying to access geometry
        df = GeoDataFrame(data)

        with pytest.raises(AttributeError):
            _ = df.geometry

        df = df.set_geometry("other_geom")
        check_geodataframe(df, "other_geom")

        # combined with custom args
        df = GeoDataFrame(data, geometry="other_geom", columns=["B", "other_geom"])
        check_geodataframe(df, "other_geom")
        assert_index_equal(df.columns, pd.Index(["B", "other_geom"]))
        assert_series_equal(df["B"], pd.Series(np.arange(3.0), name="B"))

        df = GeoDataFrame(data, geometry="other_geom", columns=["other_geom", "A"])
        check_geodataframe(df, "other_geom")
        assert_index_equal(df.columns, pd.Index(["other_geom", "A"]))
        assert_series_equal(df["A"], pd.Series(range(3), name="A"))

    def test_array(self):
        data = {
            "A": range(3),
            "B": np.arange(3.0),
            "geometry": [Point(x, x) for x in range(3)],
        }
        a = np.array([data["A"], data["B"], data["geometry"]], dtype=object).T

        df = GeoDataFrame(a, columns=["A", "B", "geometry"])
        check_geodataframe(df)

        df = GeoDataFrame(a, columns=["A", "B", "other_geom"], geometry="other_geom")
        check_geodataframe(df, "other_geom")

    def test_from_frame(self):
        data = {
            "A": range(3),
            "B": np.arange(3.0),
            "geometry": [Point(x, x) for x in range(3)],
        }
        gpdf = GeoDataFrame(data)
        pddf = pd.DataFrame(data)
        check_geodataframe(gpdf)
        assert type(pddf) is pd.DataFrame

        for df in [gpdf, pddf]:
            res = GeoDataFrame(df)
            check_geodataframe(res)

            res = GeoDataFrame(df, index=pd.Index([0, 2]))
            check_geodataframe(res)
            assert_index_equal(res.index, pd.Index([0, 2]))
            assert res["A"].tolist() == [0, 2]

            res = GeoDataFrame(df, columns=["geometry", "B"])
            check_geodataframe(res)
            assert_index_equal(res.columns, pd.Index(["geometry", "B"]))

            with pytest.raises(ValueError):
                GeoDataFrame(df, geometry="other_geom")

    def test_from_frame_specified_geometry(self):
        data = {
            "A": range(3),
            "B": np.arange(3.0),
            "other_geom": [Point(x, x) for x in range(3)],
        }

        gpdf = GeoDataFrame(data, geometry="other_geom")
        check_geodataframe(gpdf, "other_geom")
        pddf = pd.DataFrame(data)

        for df in [gpdf, pddf]:
            res = GeoDataFrame(df, geometry="other_geom")
            check_geodataframe(res, "other_geom")

        # gdf from gdf should preserve active geometry column name
        df = GeoDataFrame(gpdf)
        check_geodataframe(df, "other_geom")

    def test_only_geometry(self):
        exp = GeoDataFrame(
            {"geometry": [Point(x, x) for x in range(3)], "other": range(3)}
        )[["geometry"]]

        df = GeoDataFrame(geometry=[Point(x, x) for x in range(3)])
        check_geodataframe(df)
        assert_geodataframe_equal(df, exp)

        df = GeoDataFrame({"geometry": [Point(x, x) for x in range(3)]})
        check_geodataframe(df)
        assert_geodataframe_equal(df, exp)

        df = GeoDataFrame(
            {"other_geom": [Point(x, x) for x in range(3)]}, geometry="other_geom"
        )
        check_geodataframe(df, "other_geom")
        exp = exp.rename(columns={"geometry": "other_geom"}).set_geometry("other_geom")
        assert_geodataframe_equal(df, exp)

    def test_no_geometries(self):
        # keeps GeoDataFrame class (no DataFrame)
        data = {"A": range(3), "B": np.arange(3.0)}
        df = GeoDataFrame(data)
        assert type(df) is GeoDataFrame

        gdf = GeoDataFrame({"x": [1]})
        assert list(gdf.x) == [1]

    def test_empty(self):
        df = GeoDataFrame()
        assert type(df) is GeoDataFrame

        df = GeoDataFrame({"A": [], "B": []}, geometry=[])
        assert type(df) is GeoDataFrame

    def test_column_ordering(self):
        geoms = [Point(1, 1), Point(2, 2), Point(3, 3)]
        gs = GeoSeries(geoms)
        gdf = GeoDataFrame(
            {"a": [1, 2, 3], "geometry": gs},
            columns=["geometry", "a"],
            geometry="geometry",
        )
        check_geodataframe(gdf)
        assert list(gdf.columns) == ["geometry", "a"]

        # with non-default index
        gdf = GeoDataFrame(
            {"a": [1, 2, 3], "geometry": gs},
            columns=["geometry", "a"],
            index=pd.Index([0, 0, 1]),
            geometry="geometry",
        )
        check_geodataframe(gdf)
        assert list(gdf.columns) == ["geometry", "a"]

    def test_do_not_preserve_series_name_in_constructor(self):
        # GH3337
        # GeoDataFrame(... geometry=...) should always create geom col "geometry"
        geoms = [Point(1, 1), Point(2, 2), Point(3, 3)]
        gs = GeoSeries(geoms)
        gdf = GeoDataFrame({"a": [1, 2, 3]}, geometry=gs)
        check_geodataframe(gdf, geometry_column="geometry")
        # still get "geometry", even with custom geoseries name
        gs = GeoSeries(geoms, name="my_geom")
        gdf = GeoDataFrame({"a": [1, 2, 3]}, geometry=gs)
        check_geodataframe(gdf, geometry_column="geometry")

    def test_overwrite_geometry(self):
        # GH602
        data = pd.DataFrame({"geometry": [1, 2, 3], "col1": [4, 5, 6]})
        geoms = pd.Series([Point(i, i) for i in range(3)])
        # passed geometry kwarg should overwrite geometry column in data
        res = GeoDataFrame(data, geometry=geoms)
        assert_geoseries_equal(res.geometry, GeoSeries(geoms))

    def test_repeat_geo_col(self):
        df = pd.DataFrame(
            [
                {"geometry": Point(x, y), "geom": Point(x, y)}
                for x, y in zip(range(3), range(3))
            ],
        )
        # explicitly prevent construction of gdf with repeat geometry column names
        # two columns called "geometry", geom col inferred
        df2 = df.rename(columns={"geom": "geometry"})
        with pytest.raises(ValueError):
            GeoDataFrame(df2)
        # ensure case is caught when custom geom column name is used
        # two columns called "geom", geom col explicit
        df3 = df.rename(columns={"geometry": "geom"})
        with pytest.raises(ValueError):
            GeoDataFrame(df3, geometry="geom")

    @pytest.mark.parametrize("dtype", ["geometry", "object"])
    def test_multiindex_with_geometry_label(self, dtype):
        # DataFrame with MultiIndex where "geometry" label corresponds to
        # multiple columns
        df = pd.DataFrame([[Point(0, 0), Point(1, 1)], [Point(2, 2), Point(3, 3)]])
        df = df.astype(dtype)
        df.columns = pd.MultiIndex.from_product([["geometry"], [0, 1]])
        # don't error in constructor
        gdf = GeoDataFrame(df)
        with pytest.raises(AttributeError, match=".*geometry .* has not been set.*"):
            gdf.geometry
        res_gdf = gdf.set_geometry(("geometry", 0))
        assert res_gdf.shape == gdf.shape
        assert isinstance(res_gdf.geometry, GeoSeries)

    def test_default_geo_colname_none(self):
        match = "You are adding a column named 'geometry' to a GeoDataFrame"
        gdf = GeoDataFrame({"a": [1, 2]})

        gdf2 = gdf.copy()
        geo_col = GeoSeries.from_xy([1, 3], [3, 3])
        with pytest.warns(FutureWarning, match=match):
            gdf2["geometry"] = geo_col
        assert gdf2._geometry_column_name == "geometry"
        gdf4 = gdf.copy()
        with pytest.warns(FutureWarning, match=match):
            gdf4.geometry = geo_col
        assert gdf4._geometry_column_name == "geometry"

        # geo col name should only change if we add geometry
        gdf5 = gdf.copy()
        with pytest.warns(
            UserWarning, match="Geometry column does not contain geometry"
        ):
            gdf5["geometry"] = "foo"
        assert gdf5._geometry_column_name is None
        with pytest.warns(FutureWarning, match=match):
            gdf3 = gdf.copy().assign(geometry=geo_col)
        assert gdf3._geometry_column_name == "geometry"

        # Check that adding a GeoSeries to a column called "geometry" to a
        # gdf without an active geometry column some time after the init does not
        # warn / set the active geometry column
        gdf6 = gdf.copy()
        gdf6["geom2"] = geo_col
        gdf6["geom3"] = geo_col
        gdf6 = gdf6.set_geometry("geom2")
        subset = gdf6[["a", "geom3"]]  # this has a missing active geometry col
        assert subset._geometry_column_name == "geom2"
        subset["geometry"] = geo_col
        # adding column called geometry shouldn't auto-set
        assert subset._geometry_column_name == "geom2"

    def test_multiindex_geometry_colname_2_level(self):
        # GH1763 https://github.com/geopandas/geopandas/issues/1763
        crs = "EPSG:4326"
        df = pd.DataFrame(
            [[1, 0], [0, 1]], columns=[["location", "location"], ["x", "y"]]
        )
        x_col = df["location", "x"]
        y_col = df["location", "y"]

        gdf = GeoDataFrame(df, crs=crs, geometry=points_from_xy(x_col, y_col))
        if compat.HAS_PYPROJ:
            assert gdf.crs == crs
            assert gdf.geometry.crs == crs
        assert gdf.geometry.dtype == "geometry"
        assert gdf._geometry_column_name == "geometry"
        assert gdf.geometry.name == "geometry"

    def test_multiindex_geometry_colname_3_level(self):
        # GH1763 https://github.com/geopandas/geopandas/issues/1763
        # Note 3-level case uses different code paths in pandas, it is not redundant
        crs = "EPSG:4326"
        df = pd.DataFrame(
            [[1, 0], [0, 1]],
            columns=[
                ["foo", "foo"],
                ["location", "location"],
                ["x", "y"],
            ],
        )

        x_col = df["foo", "location", "x"]
        y_col = df["foo", "location", "y"]

        gdf = GeoDataFrame(df, crs=crs, geometry=points_from_xy(x_col, y_col))
        if compat.HAS_PYPROJ:
            assert gdf.crs == crs
            assert gdf.geometry.crs == crs
        assert gdf.geometry.dtype == "geometry"
        assert gdf._geometry_column_name == "geometry"
        assert gdf.geometry.name == "geometry"

    def test_multiindex_geometry_colname_3_level_new_col(self):
        crs = "EPSG:4326"
        df = pd.DataFrame(
            [[1, 0], [0, 1]],
            columns=[
                ["foo", "foo"],
                ["location", "location"],
                ["x", "y"],
            ],
        )

        x_col = df["foo", "location", "x"]
        y_col = df["foo", "location", "y"]
        df["geometry"] = GeoSeries.from_xy(x_col, y_col)
        df2 = df.copy()
        gdf = df.set_geometry("geometry", crs=crs)
        if compat.HAS_PYPROJ:
            assert gdf.crs == crs
        assert gdf._geometry_column_name == "geometry"
        assert gdf.geometry.name == "geometry"
        # test again setting with tuple col name
        gdf = df2.set_geometry(("geometry", "", ""), crs=crs)
        if compat.HAS_PYPROJ:
            assert gdf.crs == crs
        assert gdf._geometry_column_name == ("geometry", "", "")
        assert gdf.geometry.name == ("geometry", "", "")

    def test_assign_cols_using_index(self, nybb_filename):
        df = read_file(nybb_filename)
        other_df = pd.DataFrame({"foo": range(5), "bar": range(5)})
        expected = pd.concat([df, other_df], axis=1)
        df[other_df.columns] = other_df
        assert_geodataframe_equal(df, expected)

    def test_geometry_colname_enum(self):
        # ensure that other classes to geometry arg in GeoDataFrame
        # with `name` attribute are not assumed to be (Geo)Series
        class Fruit(Enum):
            apple = 1
            pear = 2

        df = pd.DataFrame(
            {Fruit.apple: [1, 2], Fruit.pear: GeoSeries.from_xy([1, 2], [3, 4])}
        )
        res = GeoDataFrame(df, geometry=Fruit.pear)
        assert res.active_geometry_name == Fruit.pear

    def test_geometry_nan_scalar(self):
        gdf = GeoDataFrame(
            data=[[np.nan, np.nan]],
            columns=["geometry", "something"],
            crs="EPSG:4326",
        )
        assert gdf.shape == (1, 2)
        assert gdf.active_geometry_name == "geometry"
        assert gdf.geometry[0] is None
        if compat.HAS_PYPROJ:
            assert gdf.crs == "EPSG:4326"

    def test_geometry_nan_array(self):
        gdf = GeoDataFrame(
            {
                "geometry": [np.nan, None, pd.NA],
                "something": [np.nan, np.nan, np.nan],
            },
            crs="EPSG:4326",
        )
        assert gdf.shape == (3, 2)
        assert gdf.active_geometry_name == "geometry"
        assert gdf.geometry.isna().all()
        if compat.HAS_PYPROJ:
            assert gdf.crs == "EPSG:4326"


@pytest.mark.skipif(not compat.HAS_PYPROJ, reason="pyproj not available")
def test_geodataframe_crs():
    gdf = GeoDataFrame(columns=["geometry"])
    gdf.crs = "IGNF:ETRS89UTM28"
    assert gdf.crs.to_authority() == ("IGNF", "ETRS89UTM28")


def test_geodataframe_nocrs_json():
    # no CRS, no crs field
    gdf = GeoDataFrame(columns=["geometry"])
    gdf_geojson = json.loads(gdf.to_json())
    assert "crs" not in gdf_geojson

    # WGS84, no crs field (default as per spec)
    gdf.crs = 4326
    gdf_geojson = json.loads(gdf.to_json())
    assert "crs" not in gdf_geojson


@pytest.mark.skipif(not compat.HAS_PYPROJ, reason="pyproj not available")
def test_geodataframe_crs_json():
    gdf = GeoDataFrame(columns=["geometry"])
    gdf.crs = 25833
    gdf_geojson = json.loads(gdf.to_json())
    assert "crs" in gdf_geojson
    assert gdf_geojson["crs"] == {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::25833"},
    }
    gdf_geointerface = gdf.__geo_interface__
    assert "crs" not in gdf_geointerface


@pytest.mark.skipif(not compat.HAS_PYPROJ, reason="pyproj not available")
@pytest.mark.parametrize(
    "crs",
    ["+proj=cea +lon_0=0 +lat_ts=45 +x_0=0 +y_0=0 +ellps=WGS84 +units=m", "IGNF:WGS84"],
)
def test_geodataframe_crs_nonrepresentable_json(crs):
    gdf = GeoDataFrame(
        [Point(1000, 1000)],
        columns=["geometry"],
        crs=crs,
    )
    with pytest.warns(
        UserWarning, match="GeoDataFrame's CRS is not representable in URN OGC"
    ):
        gdf_geojson = json.loads(gdf.to_json())
    assert "crs" not in gdf_geojson


def test_geodataframe_crs_colname():
    # https://github.com/geopandas/geopandas/issues/2942
    gdf = GeoDataFrame({"crs": [1], "geometry": [Point(1, 1)]})
    assert gdf.crs is None
    assert gdf["crs"].iloc[0] == 1
    assert getattr(gdf, "crs") is None

    # https://github.com/geopandas/geopandas/issues/3501
    gdf = GeoDataFrame({"crs": [1]}, geometry=[Point(1, 1)])
    assert gdf.crs is None
    assert gdf["crs"].iloc[0] == 1
    assert getattr(gdf, "crs") is None

    # test multiindex handling
    df = pd.DataFrame([[1, 0], [0, 1]], columns=[["crs", "crs"], ["x", "y"]])
    x_col = df["crs", "x"]
    y_col = df["crs", "y"]

    gdf = GeoDataFrame(df, geometry=points_from_xy(x_col, y_col))
    assert gdf.crs is None
    assert gdf["crs"].iloc[0].to_list() == [1, 0]
    assert getattr(gdf, "crs") is None


@pytest.mark.parametrize("geo_col_name", ["geometry", "polygons"])
def test_set_geometry_supply_colname(dfs, geo_col_name):
    df, _ = dfs
    if geo_col_name != "geometry":
        df = df.rename_geometry(geo_col_name)
    df["centroid"] = df.geometry.centroid
    res = df.set_geometry("centroid")
    assert res.active_geometry_name == "centroid"
    assert geo_col_name in res.columns

    # Test that drop=False explicitly warns
    deprecated = "The `drop` keyword argument is deprecated"
    with pytest.warns(FutureWarning, match=deprecated):
        res2 = df.set_geometry("centroid", drop=False)
    assert_geodataframe_equal(res, res2)

    with pytest.warns(FutureWarning, match=deprecated):
        res3 = df.set_geometry("centroid", drop=True)
    # drop=True should preserve previous geometry col name (keep old behaviour)
    assert res3.active_geometry_name == geo_col_name
    assert "centroid" not in res3.columns

    # Test that alternative suggested without using drop=True is equivalent
    assert_geodataframe_equal(
        res3,
        df.set_geometry("centroid")
        .drop(columns=geo_col_name)
        .rename_geometry(geo_col_name),
    )


@pytest.mark.parametrize("geo_col_name", ["geometry", "polygons"])
def test_set_geometry_supply_arraylike(dfs, geo_col_name):
    df, _ = dfs
    if geo_col_name != "geometry":
        df = df.rename_geometry(geo_col_name)
    centroids = df.geometry.centroid
    res = df.set_geometry(centroids)
    assert res.active_geometry_name == geo_col_name
    # drop should do nothing if the column already exists
    match_str = (
        "The `drop` keyword argument is deprecated and has no effect when "
        "`col` is an array-like value"
    )
    with pytest.warns(
        FutureWarning,
        match=match_str,
    ):
        res2 = df.set_geometry(centroids, drop=True)
    assert res2.active_geometry_name == geo_col_name

    centroids = centroids.rename("centroids")
    res3 = df.set_geometry(centroids)
    # Should preserve the geoseries name
    # (and old geometry column should be kept)
    assert res3.active_geometry_name == "centroids"
    assert geo_col_name in res3.columns

    # Drop should not remove previous active geometry colname for arraylike inputs
    with pytest.warns(
        FutureWarning,
        match=match_str,
    ):
        res4 = df.set_geometry(centroids, drop=True)
    assert res4.active_geometry_name == "centroids"
    assert geo_col_name in res4.columns


@pytest.mark.filterwarnings("error::FutureWarning")
def test_reduce_geometry_array():
    """
    Check for a FutureWarning.

    `geopandas.array.GeometryArray._reduce` issues a FutureWarning if
    the parameter `keepdims` is not set.
    `GeometryArray` inherits from `pandas.api.extensions.ExtensionArray`
    and its `_reduce` is overridden in `GeometryArray`.
    This warning is issued with pandas 2.2.2 (tested).
    """
    GeoDataFrame({"geometry": []}).all()


class GDFChild(GeoDataFrame):
    def custom_method(self):
        return "this is a custom output"


def test_inheritance(dfs):
    df, _ = dfs
    df.loc[:, "col2"] = ["a"] * len(df)

    dfc = GDFChild(df)

    dfc2 = dfc.rename_geometry("geometry2")

    children = [
        dfc,
        dfc.iloc[[0]],
        dfc.loc[dfc.col1 == 1],
        dfc.dissolve(),
        dfc[["col2", "geometry"]],
        dfc.copy(),
        dfc2,
        dfc2.iloc[[0]],
        dfc2.loc[dfc.col1 == 1],
        dfc2.dissolve(),
        dfc2[["col2", "geometry2"]],
        dfc2.copy(),
    ]

    for v in children:
        assert isinstance(v, GDFChild)
        assert v.custom_method() == "this is a custom output"

    df2 = dfc2.drop(columns=["geometry2"])

    assert not isinstance(df2, GDFChild)
