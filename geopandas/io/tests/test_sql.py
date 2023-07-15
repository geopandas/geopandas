"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""
import os
import warnings

import pandas as pd

import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis

import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest

try:
    from sqlalchemy import text
except ImportError:
    # Avoid local imports for text in all sqlalchemy tests
    # all tests using text use engine_postgis, which ensures sqlalchemy is available
    text = str


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path("nybb")
    df = read_file(nybb_path)
    return df


@pytest.fixture()
def connection_postgis():
    """
    Initiates a connection to a postGIS database that must already exist.
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="pandas only supports SQLAlchemy connectable.*"
        )
        yield con
    con.close()


@pytest.fixture()
def engine_postgis():
    """
    Initiates a connection engine to a postGIS database that must already exist.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.engine.url import URL

    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    dbname = "test_geopandas"

    try:
        con = sqlalchemy.create_engine(
            URL.create(
                drivername="postgresql+psycopg2",
                username=user,
                database=dbname,
                password=password,
                host=host,
                port=port,
            )
        )
        con.connect()
    except Exception:
        pytest.skip("Cannot connect with postgresql database")

    yield con
    con.dispose()


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


def drop_table_if_exists(conn_or_engine, table):
    sqlalchemy = pytest.importorskip("sqlalchemy")

    if sqlalchemy.inspect(conn_or_engine).has_table(table):
        metadata = sqlalchemy.MetaData()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Did not recognize type 'geometry' of column.*"
            )
            metadata.reflect(conn_or_engine)
        table = metadata.tables.get(table)
        if table is not None:
            table.drop(conn_or_engine, checkfirst=True)


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


@pytest.fixture
def df_3D_geoms():
    from shapely.geometry import Point, LineString, Polygon

    df = geopandas.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0, 0), (1, 1, 1)]),
                Polygon([(0, 0, 0), (1, 1, 1), (0, 1, 1)]),
                Point(0, 1, 2),
            ]
        },
        crs="epsg:4326",
    )
    return df


class TestIO:
    def test_get_conn(self, engine_postgis):
        Connection = pytest.importorskip("sqlalchemy.engine.base").Connection

        engine = engine_postgis
        with get_conn(engine) as output:
            assert isinstance(output, Connection)
        with engine.connect() as conn:
            with get_conn(conn) as output:
                assert isinstance(output, Connection)
        with pytest.raises(ValueError):
            with get_conn(object()):
                pass

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
        chunksize = 2
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = pd.concat(read_postgis(sql, con, chunksize=chunksize))

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None

    def test_read_postgis_privacy(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        with pytest.warns(FutureWarning):
            geopandas.io.sql.read_postgis(sql, con)

    def test_write_postgis_default(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        engine = engine_postgis
        table = "nybb"

        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        # Write to db
        write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        # Validate
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_write_postgis_uppercase_tablename(self, engine_postgis, df_nybb):
        """Tests writing GeoDataFrame to PostGIS with uppercase tablename."""
        engine = engine_postgis
        table = "aTestTable"

        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        # Write to db
        write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        # Validate
        sql = text('SELECT * FROM "{table}";'.format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_write_postgis_sqlalchemy_connection(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        with engine_postgis.begin() as con:
            table = "nybb_con"

            # If table exists, delete it before trying to write with defaults
            drop_table_if_exists(con, table)

            # Write to db
            write_postgis(df_nybb, con=con, name=table, if_exists="fail")
            # Validate
            sql = text("SELECT * FROM {table};".format(table=table))
            df = read_postgis(sql, con, geom_col="geometry")
            validate_boro_df(df)

    def test_write_postgis_fail_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that uploading the same table raises error when: if_replace='fail'.
        """
        engine = engine_postgis

        table = "nybb"

        # Ensure table exists
        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")

        try:
            write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        except ValueError as e:
            if "already exists" in str(e):
                pass
            else:
                raise e

    def test_write_postgis_replace_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that replacing a table is possible when: if_replace='replace'.
        """
        engine = engine_postgis

        table = "nybb"

        # Ensure table exists
        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
        # Overwrite
        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
        # Validate
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_write_postgis_append_when_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that appending to existing table produces correct results when:
        if_replace='append'.
        """
        engine = engine_postgis

        table = "nybb"

        orig_rows, orig_cols = df_nybb.shape
        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
        write_postgis(df_nybb, con=engine, name=table, if_exists="append")
        # Validate
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        new_rows, new_cols = df.shape

        # There should be twice as many rows in the new table
        assert new_rows == orig_rows * 2, (
            "There should be {target} rows,"
            "found: {current}".format(target=orig_rows * 2, current=new_rows),
        )
        # Number of columns should stay the same
        assert new_cols == orig_cols, (
            "There should be {target} columns,"
            "found: {current}".format(target=orig_cols, current=new_cols),
        )

    def test_write_postgis_without_crs(self, engine_postgis, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS without CRS information.
        """
        engine = engine_postgis

        table = "nybb"

        # Write to db
        df_nybb = df_nybb
        df_nybb.crs = None
        with pytest.warns(UserWarning, match="Could not parse CRS from the GeoDataF"):
            write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
        # Validate that srid is -1
        sql = text(
            "SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(
                schema="public", table=table, geom_col="geometry"
            )
        )
        with engine.connect() as conn:
            target_srid = conn.execute(sql).fetchone()[0]
        assert target_srid == 0, "SRID should be 0, found %s" % target_srid

    def test_write_postgis_with_esri_authority(self, engine_postgis, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS with ESRI Authority
        CRS information (GH #2414).
        """
        engine = engine_postgis

        table = "nybb"

        # Write to db
        df_nybb_esri = df_nybb.to_crs("ESRI:102003")
        write_postgis(df_nybb_esri, con=engine, name=table, if_exists="replace")
        # Validate that srid is 102003
        sql = text(
            "SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(
                schema="public", table=table, geom_col="geometry"
            )
        )
        with engine.connect() as conn:
            target_srid = conn.execute(sql).fetchone()[0]
        assert target_srid == 102003, "SRID should be 102003, found %s" % target_srid

    def test_write_postgis_geometry_collection(
        self, engine_postgis, df_geom_collection
    ):
        """
        Tests that writing a mix of different geometry types is possible.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(df_geom_collection, con=engine, name=table, if_exists="replace")

        # Validate geometry type
        sql = text(
            "SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;".format(
                table=table
            )
        )
        with engine.connect() as conn:
            geom_type = conn.execute(sql).fetchone()[0]
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")

        assert geom_type.upper() == "GEOMETRYCOLLECTION"
        assert df.geom_type.unique()[0] == "GeometryCollection"

    def test_write_postgis_mixed_geometry_types(
        self, engine_postgis, df_mixed_single_and_multi
    ):
        """
        Tests that writing a mix of single and MultiGeometries is possible.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(
            df_mixed_single_and_multi, con=engine, name=table, if_exists="replace"
        )

        # Validate geometry type
        sql = text(
            "SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;".format(
                table=table
            )
        )
        with engine.connect() as conn:
            res = conn.execute(sql).fetchall()
        assert res[0][0].upper() == "LINESTRING"
        assert res[1][0].upper() == "MULTILINESTRING"
        assert res[2][0].upper() == "POINT"

    def test_write_postgis_linear_ring(self, engine_postgis, df_linear_ring):
        """
        Tests that writing a LinearRing.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(df_linear_ring, con=engine, name=table, if_exists="replace")

        # Validate geometry type
        sql = text(
            "SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;".format(
                table=table
            )
        )
        with engine.connect() as conn:
            geom_type = conn.execute(sql).fetchone()[0]

        assert geom_type.upper() == "LINESTRING"

    def test_write_postgis_in_chunks(self, engine_postgis, df_mixed_single_and_multi):
        """
        Tests writing a LinearRing works.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(
            df_mixed_single_and_multi,
            con=engine,
            name=table,
            if_exists="replace",
            chunksize=1,
        )
        # Validate row count
        sql = text("SELECT COUNT(geometry) FROM {table};".format(table=table))
        with engine.connect() as conn:
            row_cnt = conn.execute(sql).fetchone()[0]
        assert row_cnt == 3

        # Validate geometry type
        sql = text(
            "SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;".format(
                table=table
            )
        )
        with engine.connect() as conn:
            res = conn.execute(sql).fetchall()
        assert res[0][0].upper() == "LINESTRING"
        assert res[1][0].upper() == "MULTILINESTRING"
        assert res[2][0].upper() == "POINT"

    def test_write_postgis_to_different_schema(self, engine_postgis, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        engine = engine_postgis

        table = "nybb"
        schema_to_use = "test"
        sql = text("CREATE SCHEMA IF NOT EXISTS {schema};".format(schema=schema_to_use))
        with engine.begin() as conn:
            conn.execute(sql)

        write_postgis(
            df_nybb, con=engine, name=table, if_exists="replace", schema=schema_to_use
        )
        # Validate
        sql = text(
            "SELECT * FROM {schema}.{table};".format(schema=schema_to_use, table=table)
        )

        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_write_postgis_to_different_schema_when_table_exists(
        self, engine_postgis, df_nybb
    ):
        """
        Tests writing data to alternative schema.
        """
        engine = engine_postgis

        table = "nybb"
        schema_to_use = "test"
        sql = text("CREATE SCHEMA IF NOT EXISTS {schema};".format(schema=schema_to_use))
        with engine.begin() as conn:
            conn.execute(sql)

        try:
            write_postgis(
                df_nybb, con=engine, name=table, if_exists="fail", schema=schema_to_use
            )
            # Validate
            sql = text(
                "SELECT * FROM {schema}.{table};".format(
                    schema=schema_to_use, table=table
                )
            )

            df = read_postgis(sql, engine, geom_col="geometry")
            validate_boro_df(df)

        # Should raise a ValueError when table exists
        except ValueError:
            pass

        # Try with replace flag on
        write_postgis(
            df_nybb, con=engine, name=table, if_exists="replace", schema=schema_to_use
        )
        # Validate
        sql = text(
            "SELECT * FROM {schema}.{table};".format(schema=schema_to_use, table=table)
        )

        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_write_postgis_3D_geometries(self, engine_postgis, df_3D_geoms):
        """
        Tests writing a geometries with 3 dimensions works.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(df_3D_geoms, con=engine, name=table, if_exists="replace")

        # Check that all geometries have 3 dimensions
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        assert list(df.geometry.has_z) == [True, True, True]

    def test_row_order(self, engine_postgis, df_nybb):
        """
        Tests that the row order in db table follows the order of the original frame.
        """
        engine = engine_postgis

        table = "row_order_test"
        correct_order = df_nybb["BoroCode"].tolist()

        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")

        # Check that the row order matches
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        assert df["BoroCode"].tolist() == correct_order

    def test_append_before_table_exists(self, engine_postgis, df_nybb):
        """
        Tests that insert works with if_exists='append' when table does not exist yet.
        """
        engine = engine_postgis

        table = "nybb"
        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        write_postgis(df_nybb, con=engine, name=table, if_exists="append")

        # Check that the row order matches
        sql = text("SELECT * FROM {table};".format(table=table))
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    def test_append_with_different_crs(self, engine_postgis, df_nybb):
        """
        Tests that the warning is raised if table CRS differs from frame.
        """
        engine = engine_postgis

        table = "nybb"
        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")

        # Reproject
        df_nybb2 = df_nybb.to_crs(epsg=4326)

        # Should raise error when appending
        with pytest.raises(ValueError, match="CRS of the target table"):
            write_postgis(df_nybb2, con=engine, name=table, if_exists="append")

    @pytest.mark.xfail(
        compat.PANDAS_GE_20 and not compat.PANDAS_GE_21,
        reason="Duplicate columns are dropped in read_sql with pandas 2.0.x",
    )
    def test_duplicate_geometry_column_fails(self, engine_postgis):
        """
        Tests that a ValueError is raised if an SQL query returns two geometry columns.
        """
        engine = engine_postgis

        sql = "select ST_MakePoint(0, 0) as geom, ST_MakePoint(0, 0) as geom;"

        with pytest.raises(ValueError):
            read_postgis(sql, engine, geom_col="geom")
