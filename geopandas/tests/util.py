import os.path
import sys
import sqlite3

from geopandas import GeoDataFrame
from geopandas.testing import (
    geom_equals, geom_almost_equals, assert_geoseries_equal)  # flake8: noqa
from pandas import Series

HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(HERE))


try:
    import psycopg2
    from psycopg2 import OperationalError
except ImportError:
    class OperationalError(Exception):
        pass

# mock not used here, but the import from here is used in other modules
try:
    import unittest.mock as mock
except ImportError:
    import mock


def validate_boro_df(df, case_sensitive=False):
    """Tests a GeoDataFrame that has been read in from the nybb dataset."""
    assert isinstance(df, GeoDataFrame)
    # Make sure all the columns are there and the geometries
    # were properly loaded as MultiPolygons
    assert len(df) == 5
    columns = ('BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area')
    if case_sensitive:
        for col in columns:
            assert col in df.columns
    else:
        for col in columns:
            assert col.lower() in (dfcol.lower() for dfcol in df.columns)
    assert Series(df.geometry.type).dropna().eq('MultiPolygon').all()


def connect(dbname, user=None, password=None, host=None, port=None):
    """
    Initiaties a connection to a postGIS database that must already exist.
    See create_postgis for more information.
    """

    user = user or os.environ.get("PGUSER")
    password = password or os.environ.get("PGPASSWORD")
    host = host or os.environ.get("PGHOST")
    port = port or os.environ.get("PGPORT")
    try:
        con = psycopg2.connect(dbname=dbname, user=user, password=password,
                               host=host, port=port)
    except (NameError, OperationalError):
        return None

    return con


def get_srid(df):
    """Return srid from `df.crs`."""
    crs = df.crs
    return (int(crs['init'][5:]) if 'init' in crs
                                 and crs['init'].startswith('epsg:')
            else 0)


def connect_spatialite():
    """
    Return a memory-based SQLite3 connection with SpatiaLite enabled & initialized.

    `The sqlite3 module must be built with loadable extension support <https://docs.python.org/3/library/sqlite3.html#f1>`_ and `SpatiaLite <https://www.gaia-gis.it/fossil/libspatialite/index>`_ must be available on the system as a SQLite module.
    Packages available on Anaconda meet requirements.

    Exceptions
    ----------
    ``AttributeError`` on missing support for loadable SQLite extensions
    ``sqlite3.OperationalError`` on missing SpatiaLite
    """
    try:
        with sqlite3.connect(':memory:') as con:
            con.enable_load_extension(True)
            con.load_extension('mod_spatialite')
            con.execute('SELECT InitSpatialMetaData(TRUE)')
    except Exception:
        con.close()
        raise
    return con

def create_spatialite(con, df):
    """
    Return a SpatiaLite connection containing the nybb table.

    Parameters
    ----------
    `con`: ``sqlite3.Connection``
    `df`: ``GeoDataFrame``
    """

    with con:
        geom_col = df.geometry.name
        srid = get_srid(df)
        con.execute('CREATE TABLE IF NOT EXISTS nybb '
                    '( ogc_fid INTEGER PRIMARY KEY'
                    ', borocode INTEGER'
                    ', boroname TEXT'
                    ', shape_leng REAL'
                    ', shape_area REAL'
                    ')')
        con.execute('SELECT AddGeometryColumn(?, ?, ?, ?)',
                    ('nybb', geom_col, srid, df.geom_type.dropna().iat[0].upper()))
        con.execute('SELECT CreateSpatialIndex(?, ?)', ('nybb', geom_col))
        sql_row = "INSERT INTO nybb VALUES(?, ?, ?, ?, ?, GeomFromText(?, ?))"
        con.executemany(sql_row,
                        ((None,
                          row.BoroCode,
                          row.BoroName,
                          row.Shape_Leng,
                          row.Shape_Area,
                          row.geometry.wkt if row.geometry
                          else None,
                          srid
                         ) for row in df.itertuples(index=False)))
    return con


def create_postgis(df, srid=None, geom_col="geom"):
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

    if srid is not None:
        geom_schema = "geometry(MULTIPOLYGON, {})".format(srid)
        geom_insert = ("ST_SetSRID(ST_GeometryFromText(%s), {})".format(srid))
    else:
        geom_schema = "geometry"
        geom_insert = "ST_GeometryFromText(%s)"
    try:
        cursor = con.cursor()
        cursor.execute("DROP TABLE IF EXISTS nybb;")

        sql = """CREATE TABLE nybb (
            {geom_col}   {geom_schema},
            borocode     integer,
            boroname     varchar(40),
            shape_leng   float,
            shape_area   float
            );""".format(geom_col=geom_col, geom_schema=geom_schema)
        cursor.execute(sql)

        for i, row in df.iterrows():
            sql = """INSERT INTO nybb VALUES ({}, %s, %s, %s, %s
            );""".format(geom_insert)
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
