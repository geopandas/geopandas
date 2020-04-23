"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""
import os

import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis

from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path("nybb")
    df = read_file(nybb_path)
    return df


@pytest.fixture()
def connection_postgis():
    """
    Initiaties a connection to a postGIS database that must already exist.
    See create_postgis for more information.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2 import OperationalError

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    try:
        con = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
    except OperationalError:
        pytest.skip("Cannot connect with postgresql database")

    yield con
    con.close()


@pytest.fixture()
def connection_spatialite():
    """
    Return a memory-based SQLite3 connection with SpatiaLite enabled & initialized.

    `The sqlite3 module must be built with loadable extension support
    <https://docs.python.org/3/library/sqlite3.html#f1>`_ and
    `SpatiaLite <https://www.gaia-gis.it/fossil/libspatialite/index>`_
    must be available on the system as a SQLite module.
    Packages available on Anaconda meet requirements.

    Exceptions
    ----------
    ``AttributeError`` on missing support for loadable SQLite extensions
    ``sqlite3.OperationalError`` on missing SpatiaLite
    """
    sqlite3 = pytest.importorskip("sqlite3")
    try:
        with sqlite3.connect(":memory:") as con:
            con.enable_load_extension(True)
            con.load_extension("mod_spatialite")
            con.execute("SELECT InitSpatialMetaData(TRUE)")
    except Exception:
        con.close()
        pytest.skip("Cannot setup spatialite database")

    yield con
    con.close()


class TestIO:
    def test_read_postgis_default(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con)

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None

    def test_read_postgis_custom_geom_col(self, connection_postgis, df_nybb):
        con = connection_postgis
        geom_col = "the_geom"
        create_postgis(con, df_nybb, geom_col=geom_col)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con, geom_col=geom_col)

        validate_boro_df(df)

    def test_read_postgis_select_geom_as(self, connection_postgis, df_nybb):
        """Tests that a SELECT {geom} AS {some_other_geom} works."""
        con = connection_postgis
        orig_geom = "geom"
        out_geom = "the_geom"
        create_postgis(con, df_nybb, geom_col=orig_geom)

        sql = """SELECT borocode, boroname, shape_leng, shape_area,
                    {} as {} FROM nybb;""".format(
            orig_geom, out_geom
        )
        df = read_postgis(sql, con, geom_col=out_geom)

        validate_boro_df(df)

    def test_read_postgis_get_srid(self, connection_postgis, df_nybb):
        """Tests that an SRID can be read from a geodatabase (GH #451)."""
        con = connection_postgis
        crs = "epsg:4269"
        df_reproj = df_nybb.to_crs(crs)
        create_postgis(con, df_reproj, srid=4269)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con)

        validate_boro_df(df)
        assert df.crs == crs

    def test_read_postgis_override_srid(self, connection_postgis, df_nybb):
        """Tests that a user specified CRS overrides the geodatabase SRID."""
        con = connection_postgis
        orig_crs = df_nybb.crs
        create_postgis(con, df_nybb, srid=4269)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con, crs=orig_crs)

        validate_boro_df(df)
        assert df.crs == orig_crs

    def test_from_postgis_default(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = GeoDataFrame.from_postgis(sql, con)

        validate_boro_df(df, case_sensitive=False)

    def test_from_postgis_custom_geom_col(self, connection_postgis, df_nybb):
        con = connection_postgis
        geom_col = "the_geom"
        create_postgis(con, df_nybb, geom_col=geom_col)

        sql = "SELECT * FROM nybb;"
        df = GeoDataFrame.from_postgis(sql, con, geom_col=geom_col)

        validate_boro_df(df, case_sensitive=False)

    def test_read_postgis_null_geom(self, connection_spatialite, df_nybb):
        """Tests that geometry with NULL is accepted."""
        con = connection_spatialite
        geom_col = df_nybb.geometry.name
        df_nybb.geometry.iat[0] = None
        create_spatialite(con, df_nybb)
        sql = (
            "SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, "
            'AsEWKB("{0}") AS "{0}" FROM nybb'.format(geom_col)
        )
        df = read_postgis(sql, con, geom_col=geom_col)
        validate_boro_df(df)

    def test_read_postgis_binary(self, connection_spatialite, df_nybb):
        """Tests that geometry read as binary is accepted."""
        con = connection_spatialite
        geom_col = df_nybb.geometry.name
        create_spatialite(con, df_nybb)
        sql = (
            "SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, "
            'ST_AsBinary("{0}") AS "{0}" FROM nybb'.format(geom_col)
        )
        df = read_postgis(sql, con, geom_col=geom_col)
        validate_boro_df(df)

    def test_read_postgis_chunksize(self, connection_postgis, df_nybb):
        """Test chunksize argument"""
        chunksize = 10
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = pd.concat(read_postgis(sql, con, chunksize=chunksize))

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None
