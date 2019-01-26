from __future__ import absolute_import

import os
from collections import OrderedDict
from distutils.version import LooseVersion

import fiona
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas.io.file import fiona_env
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path('nybb')
    df = read_file(nybb_path)
    return df


@pytest.fixture
def df_null():
    return read_file(
            os.path.join(PACKAGE_DIR, 'examples', 'null_geom.geojson'))


@pytest.fixture
def df_points():
    N = 10
    crs = {'init': 'epsg:4326'}
    df = GeoDataFrame([
        {'geometry': Point(x, y), 'value1': x + y, 'value2': x * y}
        for x, y in zip(range(N), range(N))], crs=crs)
    return df


# -----------------------------------------------------------------------------
# to_file tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("driver,ext", [
    ('ESRI Shapefile', 'shp'),
    ('GeoJSON', 'geojson')
])
def test_to_file(tmpdir, df_nybb, df_null, driver, ext):
    """ Test to_file and from_file """
    tempfilename = os.path.join(str(tmpdir), 'boros.' + ext)
    df_nybb.to_file(tempfilename, driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert 'geometry' in df
    assert len(df) == 5
    assert np.alltrue(df['BoroName'].values == df_nybb['BoroName'])

    # Write layer with null geometry out to file
    tempfilename = os.path.join(str(tmpdir), 'null_geom.' + ext)
    df_null.to_file(tempfilename, driver=driver)
    # Read layer back in
    df = GeoDataFrame.from_file(tempfilename)
    assert 'geometry' in df
    assert len(df) == 2
    assert np.alltrue(df['Name'].values == df_null['Name'])


@pytest.mark.parametrize("driver,ext", [
    ('ESRI Shapefile', 'shp'),
    ('GeoJSON', 'geojson')
])
def test_to_file_bool(tmpdir, driver, ext):
    """Test error raise when writing with a boolean column (GH #437)."""
    tempfilename = os.path.join(str(tmpdir), 'temp.{0}'.format(ext))
    df = GeoDataFrame({
        'a': [1, 2, 3], 'b': [True, False, True],
        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]})

    if LooseVersion(fiona.__version__) < LooseVersion('1.8'):
        with pytest.raises(ValueError):
            df.to_file(tempfilename, driver=driver)
    else:
        df.to_file(tempfilename, driver=driver)
        result = read_file(tempfilename)
        if driver == 'GeoJSON':
            # geojson by default assumes epsg:4326
            result.crs = None
        if driver == 'ESRI Shapefile':
            # Shapefile does not support boolean, so is read back as int
            df['b'] = df['b'].astype('int64')
        # PY2: column names 'mixed' instead of 'unicode'
        assert_geodataframe_equal(result, df, check_column_type=False)


def test_to_file_with_point_z(tmpdir):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(str(tmpdir), 'test_3Dpoint.shp')
    point3d = Point(0, 0, 500)
    point2d = Point(1, 1)
    df = GeoDataFrame({'a': [1, 2]}, geometry=[point3d, point2d], crs={})
    df.to_file(tempfilename)
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


def test_to_file_geojson_with_point_z(tmpdir):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(str(tmpdir), 'test_3Dpoint.geojson')
    point3d = Point(0, 0, 500)
    point2d = Point(1, 1)
    df = GeoDataFrame({'a': [1, 2]}, geometry=[point3d, point2d],
                      crs={'init': 'epsg:4326'})
    df.to_file(tempfilename, driver='GeoJSON')
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


def test_to_file_with_poly_z(tmpdir):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(str(tmpdir), 'test_3Dpoly.shp')
    poly3d = Polygon([[0, 0, 5], [0, 1, 5], [1, 1, 5], [1, 0, 5]])
    poly2d = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    df = GeoDataFrame({'a': [1, 2]}, geometry=[poly3d, poly2d], crs={})
    df.to_file(tempfilename)
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


def test_to_file_geojson_with_poly_z(tmpdir):
    """Test that 3D geometries are retained in writes (GH #612)."""

    tempfilename = os.path.join(tmpdir, 'test_3Dpoly.geojson')
    poly3d = Polygon([[0, 0, 5], [0, 1, 5], [1, 1, 5], [1, 0, 5]])
    poly2d = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    df = GeoDataFrame({'a': [1, 2]}, geometry=[poly3d, poly2d],
                      crs={'init': 'epsg:4326'})
    df.to_file(tempfilename, driver='GeoJSON')
    df_read = GeoDataFrame.from_file(tempfilename)
    assert_geoseries_equal(df.geometry, df_read.geometry)


def test_to_file_types(tmpdir, df_points):
    """ Test various integer type columns (GH#93) """
    tempfilename = os.path.join(str(tmpdir), 'int.shp')
    int_types = [np.int, np.int8, np.int16, np.int32, np.int64, np.intp,
                 np.uint8, np.uint16, np.uint32, np.uint64, np.long]
    geometry = df_points.geometry
    data = dict((str(i), np.arange(len(geometry), dtype=dtype))
                for i, dtype in enumerate(int_types))
    df = GeoDataFrame(data, geometry=geometry)
    df.to_file(tempfilename)


def test_to_file_empty(tmpdir):
    input_empty_df = GeoDataFrame()
    tempfilename = os.path.join(str(tmpdir), 'test.shp')
    with pytest.raises(
            ValueError, match="Cannot write empty DataFrame to file."):
        input_empty_df.to_file(tempfilename)


def test_to_file_schema(tmpdir, df_nybb):
    """
    Ensure that the file is written according to the schema
    if it is specified

    """
    tempfilename = os.path.join(str(tmpdir), 'test.shp')
    properties = OrderedDict([
        ('Shape_Leng', 'float:19.11'),
        ('BoroName', 'str:40'),
        ('BoroCode', 'int:10'),
        ('Shape_Area', 'float:19.11'),
    ])
    schema = {'geometry': 'Polygon', 'properties': properties}

    # Take the first 2 features to speed things up a bit
    df_nybb.iloc[:2].to_file(tempfilename, schema=schema)

    with fiona.open(tempfilename) as f:
        result_schema = f.schema

    assert result_schema == schema


# -----------------------------------------------------------------------------
# read_file tests
# -----------------------------------------------------------------------------


with fiona.open(geopandas.datasets.get_path('nybb')) as f:
    CRS = f.crs
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
    url = ("https://raw.githubusercontent.com/geopandas/geopandas/"
           "master/examples/null_geom.geojson")
    gdf = read_file(url)
    assert isinstance(gdf, geopandas.GeoDataFrame)


def test_read_file_filtered(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path('nybb')
    bbox = (1031051.7879884212, 224272.49231459625, 1047224.3104931959,
            244317.30894023244)
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered_with_gdf_boundary(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path('nybb')
    bbox = geopandas.GeoDataFrame(
        geometry=[box(1031051.7879884212, 224272.49231459625,
                      1047224.3104931959, 244317.30894023244)],
        crs=CRS)
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_filtered_with_gdf_boundary_mismatched_crs(df_nybb):
    full_df_shape = df_nybb.shape
    nybb_filename = geopandas.datasets.get_path('nybb')
    bbox = geopandas.GeoDataFrame(
        geometry=[box(1031051.7879884212, 224272.49231459625,
                      1047224.3104931959, 244317.30894023244)],
        crs=CRS)
    bbox.to_crs(epsg=4326, inplace=True)
    filtered_df = read_file(nybb_filename, bbox=bbox)
    filtered_df_shape = filtered_df.shape
    assert full_df_shape != filtered_df_shape
    assert filtered_df_shape == (2, 5)


def test_read_file_empty_shapefile(tmpdir):

    # create empty shapefile
    meta = {'crs': {},
            'crs_wkt': '',
            'driver': 'ESRI Shapefile',
            'schema':
                {'geometry': 'Point',
                    'properties': OrderedDict([('A', 'int:9'),
                                               ('Z', 'float:24.15')])}}

    fname = str(tmpdir.join("test_empty.shp"))

    with fiona_env():
        with fiona.open(fname, 'w', **meta) as _:  # noqa
            pass

    empty = read_file(fname)
    assert isinstance(empty, geopandas.GeoDataFrame)
    assert all(empty.columns == ['A', 'Z', 'geometry'])
