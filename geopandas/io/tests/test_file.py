from collections import OrderedDict
import datetime
import io
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd

import fiona
from shapely.geometry import Point, Polygon, box

import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas.io.file import fiona_env

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df

import pytest


_CRS = "epsg:4326"


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path("nybb")
    df = read_file(nybb_path)
    return df


@pytest.fixture
def df_null():
    return read_file(
        os.path.join(PACKAGE_DIR, "geopandas", "tests", "data", "null_geom.geojson")
    )


@pytest.fixture
def file_path():
    return os.path.join(PACKAGE_DIR, "geopandas", "tests", "data", "null_geom.geojson")


@pytest.fixture
def df_points():
    N = 10
    crs = _CRS
    df = GeoDataFrame(
        [
            {"geometry": Point(x, y), "value1": x + y, "value2": x * y}
            for x, y in zip(range(N), range(N))
        ],
        crs=crs,
    )
    return df


# -----------------------------------------------------------------------------
# to_file tests
# -----------------------------------------------------------------------------

driver_ext_pairs = [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson"), ("GPKG", "gpkg")]


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_to_file(tmpdir, df_nybb, df_null, driver, ext):
    """ Test to_file and from_file """
    tempfilename = os.path.join(str(tmpdir), "boros." + ext)
    df_nybb.to_file(tempfilename, driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert "geometry" in df
    assert len(df) == 5
    assert np.alltrue(df["BoroName"].values == df_nybb["BoroName"])

    # Write layer with null geometry out to file
    tempfilename = os.path.join(str(tmpdir), "null_geom." + ext)
    df_null.to_file(tempfilename, driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert "geometry" in df
    assert len(df) == 2
    assert np.alltrue(df["Name"].values == df_null["Name"])


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_to_file_pathlib(tmpdir, df_nybb, df_null, driver, ext):
    """ Test to_file and from_file """
    temppath = pathlib.Path(os.path.join(str(tmpdir), "boros." + ext))
    df_nybb.to_file(temppath, driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(temppath)
    assert "geometry" in df
    assert len(df) == 5
    assert np.alltrue(df["BoroName"].values == df_nybb["BoroName"])


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_to_file_bool(tmpdir, driver, ext):
    """Test error raise when writing with a boolean column (GH #437)."""
    tempfilename = os.path.join(str(tmpdir), "temp.{0}".format(ext))
    df = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": [True, False, True],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        }
    )

    df.to_file(tempfilename, driver=driver)
    result = read_file(tempfilename)
    if driver == "GeoJSON":
        # geojson by default assumes epsg:4326
        result.crs = None
    if driver == "ESRI Shapefile":
        # Shapefile does not support boolean, so is read back as int
        df["b"] = df["b"].astype("int64")
    assert_geodataframe_equal(result, df)


def test_to_file_datetime(tmpdir):
    """Test writing a data file with the datetime column type"""
    tempfilename = os.path.join(str(tmpdir), "test_datetime.gpkg")
    point = Point(0, 0)
    now = datetime.datetime.now()
    df = GeoDataFrame({"a": [1, 2], "b": [now, now]}, geometry=[point, point], crs={})
    df.to_file(tempfilename, driver="GPKG")
    df_read = read_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_to_file_with_point_z(tmpdir, ext, driver):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(str(tmpdir), "test_3Dpoint." + ext)
    point3d = Point(0, 0, 500)
    point2d = Point(1, 1)
    df = GeoDataFrame({"a": [1, 2]}, geometry=[point3d, point2d], crs=_CRS)
    df.to_file(tempfilename, driver=driver)
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_to_file_with_poly_z(tmpdir, ext, driver):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(str(tmpdir), "test_3Dpoly." + ext)
    poly3d = Polygon([[0, 0, 5], [0, 1, 5], [1, 1, 5], [1, 0, 5]])
    poly2d = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    df = GeoDataFrame({"a": [1, 2]}, geometry=[poly3d, poly2d], crs=_CRS)
    df.to_file(tempfilename, driver=driver)
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


def test_to_file_types(tmpdir, df_points):
    """ Test various integer type columns (GH#93) """
    tempfilename = os.path.join(str(tmpdir), "int.shp")
    int_types = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.intp,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]
    geometry = df_points.geometry
    data = dict(
        (str(i), np.arange(len(geometry), dtype=dtype))
        for i, dtype in enumerate(int_types)
    )
    df = GeoDataFrame(data, geometry=geometry)
    df.to_file(tempfilename)


def test_to_file_int64(tmpdir, df_points):
    tempfilename = os.path.join(str(tmpdir), "int64.shp")
    geometry = df_points.geometry
    df = GeoDataFrame(geometry=geometry)
    df["data"] = pd.array([1, np.nan] * 5, dtype=pd.Int64Dtype())
    df.to_file(tempfilename)
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geodataframe_equal(df_read, df, check_dtype=False, check_like=True)


def test_to_file_empty(tmpdir):
    input_empty_df = GeoDataFrame()
    tempfilename = os.path.join(str(tmpdir), "test.shp")
    with pytest.raises(ValueError, match="Cannot write empty DataFrame to file."):
        input_empty_df.to_file(tempfilename)


def test_to_file_privacy(tmpdir, df_nybb):
    tempfilename = os.path.join(str(tmpdir), "test.shp")
    with pytest.warns(DeprecationWarning):
        geopandas.io.file.to_file(df_nybb, tempfilename)


def test_to_file_schema(tmpdir, df_nybb):
    """
    Ensure that the file is written according to the schema
    if it is specified

    """
    tempfilename = os.path.join(str(tmpdir), "test.shp")
    properties = OrderedDict(
        [
            ("Shape_Leng", "float:19.11"),
            ("BoroName", "str:40"),
            ("BoroCode", "int:10"),
            ("Shape_Area", "float:19.11"),
        ]
    )
    schema = {"geometry": "Polygon", "properties": properties}

    # Take the first 2 features to speed things up a bit
    df_nybb.iloc[:2].to_file(tempfilename, schema=schema)

    with fiona.open(tempfilename) as f:
        result_schema = f.schema

    assert result_schema == schema


def test_to_file_column_len(tmpdir, df_points):
    """
    Ensure that a warning about truncation is given when a geodataframe with
    column names longer than 10 characters is saved to shapefile
    """
    tempfilename = os.path.join(str(tmpdir), "test.shp")

    df = df_points.iloc[:1].copy()
    df["0123456789A"] = ["the column name is 11 characters"]

    with pytest.warns(
        UserWarning, match="Column names longer than 10 characters will be truncated"
    ):
        df.to_file(tempfilename, driver="ESRI Shapefile")


@pytest.mark.parametrize("driver,ext", driver_ext_pairs)
def test_append_file(tmpdir, df_nybb, df_null, driver, ext):
    """ Test to_file with append mode and from_file """
    from fiona import supported_drivers

    if "a" not in supported_drivers[driver]:
        return None

    tempfilename = os.path.join(str(tmpdir), "boros." + ext)
    df_nybb.to_file(tempfilename, driver=driver)
    df_nybb.to_file(tempfilename, mode="a", driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert "geometry" in df
    assert len(df) == (5 * 2)
    expected = pd.concat([df_nybb] * 2, ignore_index=True)
    assert_geodataframe_equal(df, expected, check_less_precise=True)

    # Write layer with null geometry out to file
    tempfilename = os.path.join(str(tmpdir), "null_geom." + ext)
    df_null.to_file(tempfilename, driver=driver)
    df_null.to_file(tempfilename, mode="a", driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert "geometry" in df
    assert len(df) == (2 * 2)
    expected = pd.concat([df_null] * 2, ignore_index=True)
    assert_geodataframe_equal(df, expected, check_less_precise=True)


# -----------------------------------------------------------------------------
# read_file tests
# -----------------------------------------------------------------------------


with fiona.open(geopandas.datasets.get_path("nybb")) as f:
    CRS = f.crs["init"] if "init" in f.crs else f.crs_wkt
    NYBB_COLUMNS = list(f.meta["schema"]["properties"].keys())


def test_read_file(df_nybb):
    df = df_nybb.rename(columns=lambda x: x.lower())
    validate_boro_df(df)
    assert df.crs == CRS
    # get lower case columns, and exclude geometry column from comparison
    lower_columns = [c.lower() for c in NYBB_COLUMNS]
    assert (df.columns[:-1] == lower_columns).all()


@pytest.mark.web
def test_read_file_remote_geojson_url():
    url = (
        "https://raw.githubusercontent.com/geopandas/geopandas/"
        "master/geopandas/tests/data/null_geom.geojson"
    )
    gdf = read_file(url)
    assert isinstance(gdf, geopandas.GeoDataFrame)


@pytest.mark.web
def test_read_file_remote_zipfile_url():
    url = (
        "https://raw.githubusercontent.com/geopandas/geopandas/"
        "master/geopandas/datasets/nybb_16a.zip"
    )
    gdf = read_file(url)
    assert isinstance(gdf, geopandas.GeoDataFrame)


def test_read_file_textio(file_path):
    file_text_stream = open(file_path)
    file_stringio = io.StringIO(open(file_path).read())
    gdf_text_stream = read_file(file_text_stream)
    gdf_stringio = read_file(file_stringio)
    assert isinstance(gdf_text_stream, geopandas.GeoDataFrame)
    assert isinstance(gdf_stringio, geopandas.GeoDataFrame)


def test_read_file_bytesio(file_path):
    file_binary_stream = open(file_path, "rb")
    file_bytesio = io.BytesIO(open(file_path, "rb").read())
    gdf_binary_stream = read_file(file_binary_stream)
    gdf_bytesio = read_file(file_bytesio)
    assert isinstance(gdf_binary_stream, geopandas.GeoDataFrame)
    assert isinstance(gdf_bytesio, geopandas.GeoDataFrame)


def test_read_file_raw_stream(file_path):
    file_raw_stream = open(file_path, "rb", buffering=0)
    gdf_raw_stream = read_file(file_raw_stream)
    assert isinstance(gdf_raw_stream, geopandas.GeoDataFrame)


def test_read_file_pathlib(file_path):
    path_object = pathlib.Path(file_path)
    gdf_path_object = read_file(path_object)
    assert isinstance(gdf_path_object, geopandas.GeoDataFrame)


def test_read_file_tempfile():
    temp = tempfile.TemporaryFile()
    temp.write(
        b"""
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [0, 0]
      },
      "properties": {
        "name": "Null Island"
      }
    }
    """
    )
    temp.seek(0)
    gdf_tempfile = geopandas.read_file(temp)
    assert isinstance(gdf_tempfile, geopandas.GeoDataFrame)
    temp.close()


def test_read_binary_file_fsspec():
    fsspec = pytest.importorskip("fsspec")
    # Remove the zip scheme so fsspec doesn't open as a zipped file,
    # instead we want to read as bytes and let fiona decode it.
    path = geopandas.datasets.get_path("nybb")[6:]
    with fsspec.open(path, "rb") as f:
        gdf = read_file(f)
        assert isinstance(gdf, geopandas.GeoDataFrame)


def test_read_text_file_fsspec(file_path):
    fsspec = pytest.importorskip("fsspec")
    with fsspec.open(file_path, "r") as f:
        gdf = read_file(f)
        assert isinstance(gdf, geopandas.GeoDataFrame)


def test_infer_zipped_file():
    # Remove the zip scheme so that the test for a zipped file can
    # check it and add it back.
    path = geopandas.datasets.get_path("nybb")[6:]
    gdf = read_file(path)
    assert isinstance(gdf, geopandas.GeoDataFrame)

    # Check that it can sucessfully add a zip scheme to a path that already has a scheme
    gdf = read_file("file+file://" + path)
    assert isinstance(gdf, geopandas.GeoDataFrame)

    # Check that it can add a zip scheme for a path that includes a subpath
    # within the archive.
    gdf = read_file(path + "!nybb.shp")
    assert isinstance(gdf, geopandas.GeoDataFrame)


def test_allow_legacy_gdal_path():
    # Construct a GDAL-style zip path.
    path = "/vsizip/" + geopandas.datasets.get_path("nybb")[6:]
    gdf = read_file(path)
    assert isinstance(gdf, geopandas.GeoDataFrame)


def test_read_file_filtered(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    bbox = (
        1031051.7879884212,
        224272.49231459625,
        1047224.3104931959,
        244317.30894023244,
    )
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered__rows(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    filtered_df = read_file(nybb_filename, rows=1)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (1, 5)


def test_read_file_filtered__rows_bbox(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    bbox = (
        1031051.7879884212,
        224272.49231459625,
        1047224.3104931959,
        244317.30894023244,
    )
    filtered_df = read_file(nybb_filename, bbox=bbox, rows=slice(-1, None))
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (1, 5)


def test_read_file_filtered__rows_bbox__polygon(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    bbox = box(
        1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244
    )
    filtered_df = read_file(nybb_filename, bbox=bbox, rows=slice(-1, None))
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (1, 5)


def test_read_file_filtered_rows_invalid():
    with pytest.raises(TypeError):
        read_file(geopandas.datasets.get_path("nybb"), rows="not_a_slice")


def test_read_file__ignore_geometry():
    pdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"), ignore_geometry=True
    )
    assert "geometry" not in pdf.columns
    assert isinstance(pdf, pd.DataFrame) and not isinstance(pdf, geopandas.GeoDataFrame)


def test_read_file__ignore_all_fields():
    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres"),
        ignore_fields=["pop_est", "continent", "name", "iso_a3", "gdp_md_est"],
    )
    assert gdf.columns.tolist() == ["geometry"]


def test_read_file_filtered_with_gdf_boundary(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    bbox = geopandas.GeoDataFrame(
        geometry=[
            box(
                1031051.7879884212,
                224272.49231459625,
                1047224.3104931959,
                244317.30894023244,
            )
        ],
        crs=CRS,
    )
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered_with_gdf_boundary__mask(df_nybb):
    gdf_mask = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    gdf = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_cities"),
        mask=gdf_mask[gdf_mask.continent == "Africa"],
    )
    filtered_df_shape = gdf.shape
    assert filtered_df_shape == (50, 2)


def test_read_file_filtered_with_gdf_boundary__mask__polygon(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    mask = box(
        1031051.7879884212, 224272.49231459625, 1047224.3104931959, 244317.30894023244
    )
    filtered_df = read_file(nybb_filename, mask=mask)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered_with_gdf_boundary_mismatched_crs(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    bbox = geopandas.GeoDataFrame(
        geometry=[
            box(
                1031051.7879884212,
                224272.49231459625,
                1047224.3104931959,
                244317.30894023244,
            )
        ],
        crs=CRS,
    )
    bbox.to_crs(epsg=4326, inplace=True)
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered_with_gdf_boundary_mismatched_crs__mask(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path("nybb")
    mask = geopandas.GeoDataFrame(
        geometry=[
            box(
                1031051.7879884212,
                224272.49231459625,
                1047224.3104931959,
                244317.30894023244,
            )
        ],
        crs=CRS,
    )
    mask.to_crs(epsg=4326, inplace=True)
    filtered_df = read_file(nybb_filename, mask=mask.geometry)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_empty_shapefile(tmpdir):

    # create empty shapefile
    meta = {
        "crs": {},
        "crs_wkt": "",
        "driver": "ESRI Shapefile",
        "schema": {
            "geometry": "Point",
            "properties": OrderedDict([("A", "int:9"), ("Z", "float:24.15")]),
        },
    }

    fname = str(tmpdir.join("test_empty.shp"))

    with fiona_env():
        with fiona.open(fname, "w", **meta) as _:  # noqa
            pass

    empty = read_file(fname)
    assert isinstance(empty, geopandas.GeoDataFrame)
    assert all(empty.columns == ["A", "Z", "geometry"])


def test_read_file_privacy(tmpdir, df_nybb):
    with pytest.warns(DeprecationWarning):
        geopandas.io.file.read_file(geopandas.datasets.get_path("nybb"))


class FileNumber(object):
    def __init__(self, tmpdir, base, ext):
        self.tmpdir = str(tmpdir)
        self.base = base
        self.ext = ext
        self.fileno = 0

    def __repr__(self):
        filename = "{0}{1:02d}.{2}".format(self.base, self.fileno, self.ext)
        return os.path.join(self.tmpdir, filename)

    def __next__(self):
        self.fileno += 1
        return repr(self)


@pytest.mark.parametrize(
    "driver,ext", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson")]
)
def test_write_index_to_file(tmpdir, df_points, driver, ext):
    fngen = FileNumber(tmpdir, "check", ext)

    def do_checks(df, index_is_used):
        # check combinations of index=None|True|False on GeoDataFrame/GeoSeries
        other_cols = list(df.columns)
        other_cols.remove("geometry")

        if driver == "ESRI Shapefile":
            # ESRI Shapefile will add FID if no other columns exist
            driver_col = ["FID"]
        else:
            driver_col = []

        if index_is_used:
            index_cols = list(df.index.names)
        else:
            index_cols = [None] * len(df.index.names)

        # replicate pandas' default index names for regular and MultiIndex
        if index_cols == [None]:
            index_cols = ["index"]
        elif len(index_cols) > 1 and not all(index_cols):
            for level, index_col in enumerate(index_cols):
                if index_col is None:
                    index_cols[level] = "level_" + str(level)

        # check GeoDataFrame with default index=None to autodetect
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=None)
        df_check = read_file(tempfilename)
        if len(other_cols) == 0:
            expected_cols = driver_col[:]
        else:
            expected_cols = []
        if index_is_used:
            expected_cols += index_cols
        expected_cols += other_cols + ["geometry"]
        assert list(df_check.columns) == expected_cols

        # similar check on GeoSeries with index=None
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=None)
        df_check = read_file(tempfilename)
        if index_is_used:
            expected_cols = index_cols + ["geometry"]
        else:
            expected_cols = driver_col + ["geometry"]
        assert list(df_check.columns) == expected_cols

        # check GeoDataFrame with index=True
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=True)
        df_check = read_file(tempfilename)
        assert list(df_check.columns) == index_cols + other_cols + ["geometry"]

        # similar check on GeoSeries with index=True
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=True)
        df_check = read_file(tempfilename)
        assert list(df_check.columns) == index_cols + ["geometry"]

        # check GeoDataFrame with index=False
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=False)
        df_check = read_file(tempfilename)
        if len(other_cols) == 0:
            expected_cols = driver_col + ["geometry"]
        else:
            expected_cols = other_cols + ["geometry"]
        assert list(df_check.columns) == expected_cols

        # similar check on GeoSeries with index=False
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=False)
        df_check = read_file(tempfilename)
        assert list(df_check.columns) == driver_col + ["geometry"]

        return

    #
    # Checks where index is not used/saved
    #

    # index is a default RangeIndex
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    do_checks(df, index_is_used=False)

    # index is a RangeIndex, starting from 1
    df.index += 1
    do_checks(df, index_is_used=False)

    # index is a Int64Index regular sequence from 1
    df_p.index = list(range(1, len(df) + 1))
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    do_checks(df, index_is_used=False)

    # index was a default RangeIndex, but delete one row to make an Int64Index
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry).drop(5, axis=0)
    do_checks(df, index_is_used=False)

    # no other columns (except geometry)
    df = GeoDataFrame(geometry=df_p.geometry)
    do_checks(df, index_is_used=False)

    #
    # Checks where index is used/saved
    #

    # named index
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    df.index.name = "foo_index"
    do_checks(df, index_is_used=True)

    # named index, same as pandas' default name after .reset_index(drop=False)
    df.index.name = "index"
    do_checks(df, index_is_used=True)

    # named MultiIndex
    df_p = df_points.copy()
    df_p["value3"] = df_p["value2"] - df_p["value1"]
    df_p.set_index(["value1", "value2"], inplace=True)
    df = GeoDataFrame(df_p, geometry=df_p.geometry)
    do_checks(df, index_is_used=True)

    # partially unnamed MultiIndex
    df.index.names = ["first", None]
    do_checks(df, index_is_used=True)

    # unnamed MultiIndex
    df.index.names = [None, None]
    do_checks(df, index_is_used=True)

    # unnamed Float64Index
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    df.index = df_p.index.astype(float) / 10
    do_checks(df, index_is_used=True)

    # named Float64Index
    df.index.name = "centile"
    do_checks(df, index_is_used=True)

    # index as string
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    df.index = pd.TimedeltaIndex(range(len(df)), "days")
    # TODO: TimedeltaIndex is an invalid field type
    df.index = df.index.astype(str)
    do_checks(df, index_is_used=True)

    # unnamed DatetimeIndex
    df_p = df_points.copy()
    df = GeoDataFrame(df_p["value1"], geometry=df_p.geometry)
    df.index = pd.TimedeltaIndex(range(len(df)), "days") + pd.DatetimeIndex(
        ["1999-12-27"] * len(df)
    )
    if driver == "ESRI Shapefile":
        # Shapefile driver does not support datetime fields
        df.index = df.index.astype(str)
    do_checks(df, index_is_used=True)

    # named DatetimeIndex
    df.index.name = "datetime"
    do_checks(df, index_is_used=True)
