"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""
import os

import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis

from geopandas.io.sql import write_postgis
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


def connect_engine(dbname, user=None, password=None, host=None, port=None):
    """
    Initiaties a connection (engine) to a postGIS database that must already exist.
    See create_postgis for more information.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.engine.url import URL

    user = user or os.environ.get("PGUSER")
    password = password or os.environ.get("PGPASSWORD")
    host = host or os.environ.get("PGHOST")
    port = port or os.environ.get("PGPORT")
    try:
        con = sqlalchemy.create_engine(
            URL(
                drivername="postgresql+psycopg2",
                username=user,
                database=dbname,
                password=password,
                host=host,
                port=port,
            )
        )
        con.begin()
    except Exception:
        return None

    return con


def drop_table_if_exists(engine, table):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.declarative import declarative_base

    if engine.has_table(table):
        base = declarative_base()
        metadata = sqlalchemy.MetaData(engine)
        metadata.reflect()
        table = metadata.tables.get(table)
        if table is not None:
            base.metadata.drop_all(engine, [table], checkfirst=True)


@pytest.fixture
def df_mixed_single_and_multi():
    from shapely.geometry import Point, LineString, MultiLineString

    df = geopandas.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
                Point(0, 1),
            ]
        },
        crs="epsg:4326",
    )
    return df


@pytest.fixture
def df_geom_collection():
    from shapely.geometry import Point, LineString, Polygon, GeometryCollection

    df = geopandas.GeoDataFrame(
        {
            "geometry": [
                GeometryCollection(
                    [
                        Polygon([(0, 0), (1, 1), (0, 1)]),
                        LineString([(0, 0), (1, 1)]),
                        Point(0, 0),
                    ]
                )
            ]
        },
        crs="epsg:4326",
    )
    return df


@pytest.fixture
def df_linear_ring():
    from shapely.geometry import LinearRing

    df = geopandas.GeoDataFrame(
        {"geometry": [LinearRing(((0, 0), (0, 1), (1, 1), (1, 0)))]}, crs="epsg:4326"
    )
    return df


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

    def test_write_postgis_default(self, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"

        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        try:
            # Write to db
            write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
            # Validate
            sql = "SELECT * FROM {table};".format(table=table)
            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)
        finally:
            engine.dispose()

    def test_write_postgis_fail_when_table_exists(self, df_nybb):
        """
        Tests that uploading the same table raises error when: if_replace='fail'.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"

        try:
            write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        except ValueError as e:
            if "already exists" in str(e):
                pass
            else:
                raise e
        finally:
            engine.dispose()

    def test_write_postgis_replace_when_table_exists(self, df_nybb):
        """
        Tests that replacing a table is possible when: if_replace='replace'.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"

        try:
            write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
            # Validate
            sql = "SELECT * FROM {table};".format(table=table)
            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)
        except ValueError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_append_when_table_exists(self, df_nybb):
        """
        Tests that appending to existing table produces correct results when:
        if_replace='append'.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"
        try:
            orig_rows, orig_cols = df_nybb.shape
            write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
            write_postgis(df_nybb, con=engine, name=table, if_exists="append")
            # Validate
            sql = "SELECT * FROM {table};".format(table=table)
            df = read_postgis(sql, engine, geom_col="geometry")
            new_rows, new_cols = df.shape
            # There should be twice as many rows in the new table
            assert new_rows == orig_rows * 2, (
                "There should be {target} rows,",
                "found: {current}".format(target=orig_rows * 2, current=new_rows),
            )
            # Number of columns should stay the same
            assert new_cols == orig_cols, (
                "There should be {target} columns,",
                "found: {current}".format(target=orig_cols, current=new_cols),
            )

        except AssertionError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_without_crs(self, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS without CRS information.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"

        try:
            # Write to db
            df_nybb = df_nybb
            df_nybb.crs = None
            write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
            # Validate that srid is -1
            target_srid = engine.execute(
                "SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(
                    schema="public", table=table, geom_col="geometry"
                )
            ).fetchone()[0]
            assert target_srid == 0, "SRID should be 0, found %s" % target_srid
        finally:
            engine.dispose()

    def test_write_postgis_geometry_collection(self, df_geom_collection):
        """
        Tests that writing a mix of different geometry types is possible.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "geomtype_tests"
        try:
            write_postgis(
                df_geom_collection, con=engine, name=table, if_exists="replace"
            )

            # Validate geometry type
            sql = "SELECT DISTINCT(GeometryType(geometry)) FROM {table};".format(
                table=table
            )
            geom_type = engine.execute(sql).fetchone()[0]
            sql = "SELECT * FROM {table};".format(table=table)
            df = read_postgis(sql, engine, geom_col="geometry")

            assert geom_type.upper() == "GEOMETRYCOLLECTION"
            assert df.geom_type.unique()[0] == "GeometryCollection"

        except AssertionError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_mixed_geometry_types(self, df_mixed_single_and_multi):
        """
        Tests that writing a mix of single and MultiGeometries is possible.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "geomtype_tests"
        try:
            write_postgis(
                df_mixed_single_and_multi, con=engine, name=table, if_exists="replace"
            )

            # Validate geometry type
            sql = "SELECT DISTINCT(GeometryType(geometry)) FROM {table};".format(
                table=table
            )
            res = engine.execute(sql).fetchall()
            geom_type_0 = res[0][0]
            geom_type_1 = res[1][0]
            geom_type_2 = res[2][0]
            assert geom_type_0.upper() == "MULTILINESTRING", (
                "Geometry type should be 'MULTILINESTRING',",
                "found: {gt}".format(gt=geom_type_0),
            )
            assert geom_type_1.upper() == "POINT", (
                "Geometry type should be 'POINT',",
                "found: {gt}".format(gt=geom_type_1),
            )
            assert geom_type_2.upper() == "LINESTRING", (
                "Geometry type should be 'LINESTRING',",
                "found: {gt}".format(gt=geom_type_2),
            )

        except AssertionError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_linear_ring(self, df_linear_ring):
        """
        Tests that writing a LinearRing.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "geomtype_tests"
        try:
            write_postgis(df_linear_ring, con=engine, name=table, if_exists="replace")

            # Validate geometry type
            sql = "SELECT DISTINCT(GeometryType(geometry)) FROM {table};".format(
                table=table
            )
            geom_type = engine.execute(sql).fetchone()[0]

            assert geom_type.upper() == "LINESTRING"

        except AssertionError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_in_chunks(self, df_mixed_single_and_multi):
        """
        Tests that writing a LinearRing.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "geomtype_tests"
        try:
            write_postgis(
                df_mixed_single_and_multi,
                con=engine,
                name=table,
                if_exists="replace",
                chunksize=1,
            )
            # Validate row count
            sql = "SELECT COUNT(geometry) FROM {table};".format(table=table)
            row_cnt = engine.execute(sql).fetchone()[0]

            # Validate geometry type
            sql = "SELECT DISTINCT(GeometryType(geometry)) FROM {table};".format(
                table=table
            )
            assert row_cnt == 3

            res = engine.execute(sql).fetchall()
            geom_type_0 = res[0][0]
            geom_type_1 = res[1][0]
            geom_type_2 = res[2][0]
            assert geom_type_0.upper() == "MULTILINESTRING", (
                "Geometry type should be 'MULTILINESTRING',",
                "found: {gt}".format(gt=geom_type_0),
            )
            assert geom_type_1.upper() == "POINT", (
                "Geometry type should be 'POINT',",
                "found: {gt}".format(gt=geom_type_1),
            )
            assert geom_type_2.upper() == "LINESTRING", (
                "Geometry type should be 'LINESTRING',",
                "found: {gt}".format(gt=geom_type_2),
            )

        except AssertionError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_to_different_schema(self, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"
        schema_to_use = "test"
        sql = "CREATE SCHEMA IF NOT EXISTS {schema};".format(schema=schema_to_use)
        engine.execute(sql)

        try:
            write_postgis(
                df_nybb,
                con=engine,
                name=table,
                if_exists="replace",
                schema=schema_to_use,
            )
            # Validate
            sql = "SELECT * FROM {schema}.{table};".format(
                schema=schema_to_use, table=table
            )

            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)
        except ValueError as e:
            raise e
        finally:
            engine.dispose()

    def test_write_postgis_to_different_schema_when_table_exists(self, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        engine = connect_engine("test_geopandas")
        if engine is None:
            raise pytest.skip()

        table = "nybb"
        schema_to_use = "test"
        sql = "CREATE SCHEMA IF NOT EXISTS {schema};".format(schema=schema_to_use)
        engine.execute(sql)

        try:

            write_postgis(
                df_nybb, con=engine, name=table, if_exists="fail", schema=schema_to_use
            )
            # Validate
            sql = "SELECT * FROM {schema}.{table};".format(
                schema=schema_to_use, table=table
            )

            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)
        # Should raise a ValueError when table exists
        except ValueError:
            pass

        # Try with replace flag on
        try:
            write_postgis(
                df_nybb,
                con=engine,
                name=table,
                if_exists="replace",
                schema=schema_to_use,
            )
            # Validate
            sql = "SELECT * FROM {schema}.{table};".format(
                schema=schema_to_use, table=table
            )

            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)

        except ValueError as e:
            raise e
        finally:
            engine.dispose()
