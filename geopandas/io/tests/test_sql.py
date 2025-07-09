"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""

import os
import warnings
from importlib.util import find_spec

import pandas as pd

import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_postgis
from geopandas._compat import HAS_PYPROJ
from geopandas.io.sql import _get_conn as get_conn
from geopandas.io.sql import _write_postgis as write_postgis

import pytest
from geopandas.tests.util import (
    create_postgis,
    create_spatialite,
    mock,
    validate_boro_df,
)

try:
    from sqlalchemy import text
except ImportError:
    # Avoid local imports for text in all sqlalchemy tests
    # all tests using text use engine_postgis, which ensures sqlalchemy is available
    text = str


@pytest.fixture
def df_nybb(nybb_filename):
    df = read_file(nybb_filename)
    return df


def check_available_postgis_drivers() -> list[str]:
    """Work out which of psycopg2 and psycopg are available.
    This prevents tests running if the relevant package isn't installed
    (rather than being skipped, as skips are treated as failures during postgis CI)
    """
    drivers = []
    if find_spec("psycopg"):
        drivers.append("psycopg")
    if find_spec("psycopg2"):
        drivers.append("psycopg2")
    return drivers


POSTGIS_DRIVERS = check_available_postgis_drivers()


def prepare_database_credentials() -> dict:
    """Gather postgres connection credentials from environment variables."""
    return {
        "dbname": "test_geopandas",
        "user": os.environ.get("PGUSER"),
        "password": os.environ.get("PGPASSWORD"),
        "host": os.environ.get("PGHOST"),
        "port": os.environ.get("PGPORT"),
    }


@pytest.fixture()
def connection_postgis(request):
    """Create a postgres connection using either psycopg2 or psycopg.

    Use this as an indirect fixture, where the request parameter is POSTGIS_DRIVERS."""
    psycopg = pytest.importorskip(request.param)

    try:
        con = psycopg.connect(**prepare_database_credentials())
    except psycopg.OperationalError:
        pytest.skip("Cannot connect with postgresql database")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="pandas only supports SQLAlchemy connectable.*"
        )
        yield con
    con.close()


@pytest.fixture()
def engine_postgis(request):
    """
    Initiate a sqlalchemy connection engine using either psycopg2 or psycopg.

    Use this as an indirect fixture, where the request parameter is POSTGIS_DRIVERS.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.engine.url import URL

    credentials = prepare_database_credentials()
    try:
        con = sqlalchemy.create_engine(
            URL.create(
                drivername=f"postgresql+{request.param}",
                username=credentials["user"],
                database=credentials["dbname"],
                password=credentials["password"],
                host=credentials["host"],
                port=credentials["port"],
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
    from shapely.geometry import LineString, MultiLineString, Point

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
    from shapely.geometry import GeometryCollection, LineString, Point, Polygon

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
    from shapely.geometry import LineString, Point, Polygon

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
    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_default(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con)

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_custom_geom_col(self, connection_postgis, df_nybb):
        con = connection_postgis
        geom_col = "the_geom"
        create_postgis(con, df_nybb, geom_col=geom_col)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con, geom_col=geom_col)

        validate_boro_df(df)

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_select_geom_as(self, connection_postgis, df_nybb):
        """Tests that a SELECT {geom} AS {some_other_geom} works."""
        con = connection_postgis
        orig_geom = "geom"
        out_geom = "the_geom"
        create_postgis(con, df_nybb, geom_col=orig_geom)

        sql = f"""SELECT borocode, boroname, shape_leng, shape_area,
                    {orig_geom} as {out_geom} FROM nybb;"""
        df = read_postgis(sql, con, geom_col=out_geom)

        validate_boro_df(df)

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
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

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_postgis_override_srid(self, connection_postgis, df_nybb):
        """Tests that a user specified CRS overrides the geodatabase SRID."""
        con = connection_postgis
        orig_crs = df_nybb.crs
        create_postgis(con, df_nybb, srid=4269)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con, crs=orig_crs)

        validate_boro_df(df)
        assert df.crs == orig_crs

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_from_postgis_default(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = GeoDataFrame.from_postgis(sql, con)

        validate_boro_df(df, case_sensitive=False)

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
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
            f'AsEWKB("{geom_col}") AS "{geom_col}" FROM nybb'
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
            f'ST_AsBinary("{geom_col}") AS "{geom_col}" FROM nybb'
        )
        df = read_postgis(sql, con, geom_col=geom_col)
        validate_boro_df(df)

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
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

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_default(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        engine = engine_postgis
        table = "nybb"

        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        # Write to db
        write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        # Validate
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_uppercase_tablename(self, engine_postgis, df_nybb):
        """Tests writing GeoDataFrame to PostGIS with uppercase tablename."""
        engine = engine_postgis
        table = "aTestTable"

        # If table exists, delete it before trying to write with defaults
        drop_table_if_exists(engine, table)

        # Write to db
        write_postgis(df_nybb, con=engine, name=table, if_exists="fail")
        # Validate
        sql = text(f'SELECT * FROM "{table}";')
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_sqlalchemy_connection(self, engine_postgis, df_nybb):
        """Tests that GeoDataFrame can be written to PostGIS with defaults."""
        with engine_postgis.begin() as con:
            table = "nybb_con"

            # If table exists, delete it before trying to write with defaults
            drop_table_if_exists(con, table)

            # Write to db
            write_postgis(df_nybb, con=con, name=table, if_exists="fail")
            # Validate
            sql = text(f"SELECT * FROM {table};")
            df = read_postgis(sql, con, geom_col="geometry")
            validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        new_rows, new_cols = df.shape

        # There should be twice as many rows in the new table
        assert new_rows == orig_rows * 2, (
            f"There should be {orig_rows * 2} rows,found: {new_rows}",
        )
        # Number of columns should stay the same
        assert new_cols == orig_cols, (
            f"There should be {orig_cols} columns,found: {new_cols}",
        )

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_without_crs(self, engine_postgis, df_nybb):
        """
        Tests that GeoDataFrame can be written to PostGIS without CRS information.
        """
        engine = engine_postgis

        table = "nybb"

        # Write to db
        df_nybb.geometry.array.crs = None
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
        assert target_srid == 0, f"SRID should be 0, found {target_srid}"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        assert target_srid == 102003, f"SRID should be 102003, found {target_srid}"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;")
        with engine.connect() as conn:
            geom_type = conn.execute(sql).fetchone()[0]
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")

        assert geom_type.upper() == "GEOMETRYCOLLECTION"
        assert df.geom_type.unique()[0] == "GeometryCollection"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;")
        with engine.connect() as conn:
            res = conn.execute(sql).fetchall()
        assert res[0][0].upper() == "LINESTRING"
        assert res[1][0].upper() == "MULTILINESTRING"
        assert res[2][0].upper() == "POINT"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_linear_ring(self, engine_postgis, df_linear_ring):
        """
        Tests that writing a LinearRing.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(df_linear_ring, con=engine, name=table, if_exists="replace")

        # Validate geometry type
        sql = text(f"SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;")
        with engine.connect() as conn:
            geom_type = conn.execute(sql).fetchone()[0]

        assert geom_type.upper() == "LINESTRING"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT COUNT(geometry) FROM {table};")
        with engine.connect() as conn:
            row_cnt = conn.execute(sql).fetchone()[0]
        assert row_cnt == 3

        # Validate geometry type
        sql = text(f"SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;")
        with engine.connect() as conn:
            res = conn.execute(sql).fetchall()
        assert res[0][0].upper() == "LINESTRING"
        assert res[1][0].upper() == "MULTILINESTRING"
        assert res[2][0].upper() == "POINT"

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_to_different_schema(self, engine_postgis, df_nybb):
        """
        Tests writing data to alternative schema.
        """
        engine = engine_postgis

        table = "nybb"
        schema_to_use = "test"
        sql = text(f"CREATE SCHEMA IF NOT EXISTS {schema_to_use};")
        with engine.begin() as conn:
            conn.execute(sql)

        write_postgis(
            df_nybb, con=engine, name=table, if_exists="replace", schema=schema_to_use
        )
        # Validate
        sql = text(f"SELECT * FROM {schema_to_use}.{table};")

        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_to_different_schema_when_table_exists(
        self, engine_postgis, df_nybb
    ):
        """
        Tests writing data to alternative schema.
        """
        engine = engine_postgis

        table = "nybb"
        schema_to_use = "test"
        sql = text(f"CREATE SCHEMA IF NOT EXISTS {schema_to_use};")
        with engine.begin() as conn:
            conn.execute(sql)

        try:
            write_postgis(
                df_nybb, con=engine, name=table, if_exists="fail", schema=schema_to_use
            )
            # Validate
            sql = text(f"SELECT * FROM {schema_to_use}.{table};")

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
        sql = text(f"SELECT * FROM {schema_to_use}.{table};")

        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_write_postgis_3D_geometries(self, engine_postgis, df_3D_geoms):
        """
        Tests writing a geometries with 3 dimensions works.
        """
        engine = engine_postgis

        table = "geomtype_tests"

        write_postgis(df_3D_geoms, con=engine, name=table, if_exists="replace")

        # Check that all geometries have 3 dimensions
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        assert list(df.geometry.has_z) == [True, True, True]

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_row_order(self, engine_postgis, df_nybb):
        """
        Tests that the row order in db table follows the order of the original frame.
        """
        engine = engine_postgis

        table = "row_order_test"
        correct_order = df_nybb["BoroCode"].tolist()

        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")

        # Check that the row order matches
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        assert df["BoroCode"].tolist() == correct_order

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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
        sql = text(f"SELECT * FROM {table};")
        df = read_postgis(sql, engine, geom_col="geometry")
        validate_boro_df(df)

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
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

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_append_without_crs(self, engine_postgis, df_nybb):
        # This test was included in #3328 when the default value for no
        # CRS was changed from an SRID of -1 to 0. This resolves issues
        # of appending dataframes to postgis that have no CRS as postgis
        # no CRS value is 0.
        engine = engine_postgis
        df_nybb = df_nybb.set_crs(None, allow_override=True)
        table = "nybb"

        write_postgis(df_nybb, con=engine, name=table, if_exists="replace")
        # append another dataframe with no crs

        df_nybb2 = df_nybb
        write_postgis(df_nybb2, con=engine, name=table, if_exists="append")

    @pytest.mark.parametrize("engine_postgis", POSTGIS_DRIVERS, indirect=True)
    @pytest.mark.xfail(
        not compat.PANDAS_GE_202,
        reason="Duplicate columns are dropped in read_sql with pandas 2.0.0 and 2.0.1",
    )
    def test_duplicate_geometry_column_fails(self, engine_postgis):
        """
        Tests that a ValueError is raised if an SQL query returns two geometry columns.
        """
        engine = engine_postgis

        sql = "select ST_MakePoint(0, 0) as geom, ST_MakePoint(0, 0) as geom;"

        with pytest.raises(ValueError):
            read_postgis(sql, engine, geom_col="geom")

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_non_epsg_crs(self, connection_postgis, df_nybb):
        con = connection_postgis
        df_nybb = df_nybb.to_crs(crs="esri:54052")
        create_postgis(con, df_nybb, srid=54052)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con)
        validate_boro_df(df)
        assert df.crs == "ESRI:54052"

    @pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
    @mock.patch("shapely.get_srid")
    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_srid_not_in_table(self, mock_get_srid, connection_postgis, df_nybb):
        # mock a non-existent srid for edge case if shapely has an srid
        # not present in postgis table.
        pyproj = pytest.importorskip("pyproj")

        mock_get_srid.return_value = 99999

        con = connection_postgis
        df_nybb = df_nybb.to_crs(crs="epsg:4326")
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        with pytest.raises(pyproj.exceptions.CRSError, match="crs not found"):
            with pytest.warns(UserWarning, match="Could not find srid 99999"):
                read_postgis(sql, con)

    @mock.patch("geopandas.io.sql._get_spatial_ref_sys_df")
    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_no_spatial_ref_sys_table_in_postgis(
        self, mock_get_spatial_ref_sys_df, connection_postgis, df_nybb
    ):
        # mock for a non-existent spatial_ref_sys database

        mock_get_spatial_ref_sys_df.side_effect = pd.errors.DatabaseError

        con = connection_postgis
        df_nybb = df_nybb.to_crs(crs="epsg:4326")
        create_postgis(con, df_nybb, srid=4326)

        sql = "SELECT * FROM nybb;"
        with pytest.warns(
            UserWarning, match="Could not find the spatial reference system table"
        ):
            df = read_postgis(sql, con)

        assert df.crs == "EPSG:4326"

    @pytest.mark.parametrize("connection_postgis", POSTGIS_DRIVERS, indirect=True)
    def test_read_non_epsg_crs_chunksize(self, connection_postgis, df_nybb):
        """Test chunksize argument with non epsg crs"""
        chunksize = 2
        con = connection_postgis
        df_nybb = df_nybb.to_crs(crs="esri:54052")

        create_postgis(con, df_nybb, srid=54052)

        sql = "SELECT * FROM nybb;"
        df = pd.concat(read_postgis(sql, con, chunksize=chunksize))

        validate_boro_df(df)
        assert df.crs == "ESRI:54052"
