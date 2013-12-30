import os.path
import urllib2

from geopandas import GeoDataFrame, GeoSeries

# Compatibility layer for Python 2.6: try loading unittest2
import sys
if sys.version_info[:2] == (2, 6):
    try:
        import unittest2 as unittest
    except ImportError:
        import unittest

else:
    import unittest

try:
    import psycopg2
    from psycopg2 import OperationalError
except ImportError:
    class OperationalError(Exception):
        pass


def download_nybb():
    """ Returns the path to the NYC boroughs file. Downloads if necessary. """
    # Data from http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip
    # saved as geopandas/examples/nybb_13a.zip.
    filename = os.path.join('examples', 'nybb_13a.zip')
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            response = urllib2.urlopen('http://www.nyc.gov/html/dcp/download/bytes/nybb_13a.zip')
            f.write(response.read())
    return filename


def validate_boro_df(test, df):
    """ Tests a GeoDataFrame that has been read in from the nybb dataset."""
    test.assertTrue(isinstance(df, GeoDataFrame))
    # Make sure all the columns are there and the geometries
    # were properly loaded as MultiPolygons
    test.assertEqual(len(df), 5)
    columns = ('borocode', 'boroname', 'shape_leng', 'shape_area')
    for col in columns:
        test.assertTrue(col in df.columns, 'Column {0} missing'.format(col))
    test.assertTrue(all(df.geometry.type == 'MultiPolygon'))


def connect(dbname):
    try:
        con = psycopg2.connect(dbname=dbname)
    except (NameError, OperationalError):
        return None

    return con


def create_db(df):
    """
    Create a nybb table in the test_geopandas PostGIS database.
    Returns a boolean indicating whether the database table was successfully
    created

    """
    # Try to create the database, skip the db tests if something goes
    # wrong
    # If you'd like these tests to run, create a database called
    # 'test_geopandas' and enable postgis in it:
    # > createdb test_geopandas
    # > psql -c "CREATE EXTENSION postgis" -d test_geopandas
    con = connect('test_geopandas')
    if con is None:
        return False

    try:
        cursor = con.cursor()
        cursor.execute("DROP TABLE IF EXISTS nybb;")

        sql = """CREATE TABLE nybb (
            geom        geometry,
            borocode    integer,
            boroname    varchar(40),
            shape_leng  float,
            shape_area  float
        );"""
        cursor.execute(sql)

        for i, row in df.iterrows():
            sql = """INSERT INTO nybb VALUES (
                ST_GeometryFromText(%s), %s, %s, %s, %s
            );"""
            cursor.execute(sql, (row['geometry'].wkt,
                                 row['BoroCode'],
                                 row['BoroName'],
                                 row['Shape_Leng'],
                                 row['Shape_Area']))
    finally:
        cursor.close()
        con.commit()
        con.close()

    return True


def assert_seq_equal(left, right):
    """Poor man's version of assert_almost_equal which isn't working with Shapely
    objects right now"""
    assert len(left) == len(right), "Mismatched lengths: %d != %d" % (len(left), len(right))

    for elem_left, elem_right in zip(left, right):
        assert elem_left == elem_right, "%r != %r" % (left, right)


def geom_equals(this, that):
    """Test for geometric equality. Empty geometries are considered equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 attribute)
    """

    return (this.geom_equals(that) | (this.is_empty & that.is_empty)).all()


def geom_almost_equals(this, that):
    """Test for 'almost' geometric equality. Empty geometries considered equal.

    Parameters
    ----------
    this, that : arrays of Geo objects (or anything that has an `is_empty`
                 property)
    """

    return (this.geom_almost_equals(that) |
            (this.is_empty & that.is_empty)).all()

# TODO: Remove me when standardizing on pandas 0.13, which already includes
#       this test util.
def assert_isinstance(obj, klass_or_tuple):
    assert isinstance(obj, klass_or_tuple), "type: %r != %r" % (
                                           type(obj).__name__,
                                           getattr(klass_or_tuple, '__name__',
                                                   klass_or_tuple))

def assert_geoseries_equal(left, right, check_dtype=False,
                           check_index_type=False,
                           check_series_type=True,
                           check_less_precise=False,
                           check_geom_type=False,
                           check_crs=True):
    """Test util for checking that two GeoSeries are equal.

    Parameters
    ----------
    left, right : two GeoSeries
    check_dtype : bool, default False
        if True, check geo dtype [only included so it's a drop-in replacement
        for assert_series_equal]
    check_index_type : bool, default False
        check that index types are equal
    check_series_type : bool, default True
        check that both are same type (*and* are GeoSeries). If False,
        will attempt to convert both into GeoSeries.
    check_less_precise : bool, default False
        if True, use geom_almost_equals. if False, use geom_equals.
    check_geom_type : bool, default False
        if True, check that all the geom types are equal.
    check_crs: bool, default True
        if check_series_type is True, then also check that the
        crs matches
    """
    assert len(left) == len(right), "%d != %d" % (len(left), len(right))

    if check_index_type:
        assert_isinstance(left.index, type(right.index))

    if check_dtype:
        assert left.dtype == right.dtype, "dtype: %s != %s" % (left.dtype,
                                                               right.dtype)

    if check_series_type:
        assert isinstance(left, GeoSeries)
        assert_isinstance(left, type(right))

        if check_crs:
            assert(left.crs == right.crs)
    else:
        if not isinstance(left, GeoSeries):
            left = GeoSeries(left)
        if not isinstance(right, GeoSeries):
            right = GeoSeries(right, index=left.index)

    assert left.index.equals(right.index), "index: %s != %s" % (left.index,
                                                                right.index)

    if check_geom_type:
        assert (left.type == right.type).all(), "type: %s != %s" % (left.type,
                                                                    right.type)

    if check_less_precise:
        assert geom_almost_equals(left, right)
    else:
        assert geom_equals(left, right)
