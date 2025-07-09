import os.path

from pandas import Series

from geopandas import GeoDataFrame

from geopandas.testing import (  # noqa: F401
    assert_geoseries_equal,
    geom_almost_equals,
    geom_equals,
)

HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(HERE))

_TEST_DATA_DIR = os.path.join(PACKAGE_DIR, "geopandas", "tests", "data")
_NYBB = "zip://" + os.path.join(_TEST_DATA_DIR, "nybb_16a.zip")
_NATURALEARTH_CITIES = os.path.join(
    _TEST_DATA_DIR, "naturalearth_cities", "naturalearth_cities.shp"
)
_NATURALEARTH_LOWRES = os.path.join(
    _TEST_DATA_DIR, "naturalearth_lowres", "naturalearth_lowres.shp"
)


# mock not used here, but the import from here is used in other modules
try:
    from unittest import mock
except ImportError:
    from unittest import mock  # noqa: F401


def validate_boro_df(df, case_sensitive=False):
    """Tests a GeoDataFrame that has been read in from the nybb dataset."""
    assert isinstance(df, GeoDataFrame)
    # Make sure all the columns are there and the geometries
    # were properly loaded as MultiPolygons
    assert len(df) == 5
    columns = ("BoroCode", "BoroName", "Shape_Leng", "Shape_Area")
    if case_sensitive:
        for col in columns:
            assert col in df.columns
    else:
        for col in columns:
            assert col.lower() in (dfcol.lower() for dfcol in df.columns)
    assert Series(df.geometry.geom_type).dropna().eq("MultiPolygon").all()


def get_srid(df):
    """Return srid from `df.crs`."""
    if df.crs is not None:
        return df.crs.to_epsg() or 0
    return 0


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
        con.execute(
            "CREATE TABLE IF NOT EXISTS nybb "
            "( ogc_fid INTEGER PRIMARY KEY"
            ", borocode INTEGER"
            ", boroname TEXT"
            ", shape_leng REAL"
            ", shape_area REAL"
            ")"
        )
        con.execute(
            "SELECT AddGeometryColumn(?, ?, ?, ?)",
            ("nybb", geom_col, srid, df.geom_type.dropna().iat[0].upper()),
        )
        con.execute("SELECT CreateSpatialIndex(?, ?)", ("nybb", geom_col))
        sql_row = "INSERT INTO nybb VALUES(?, ?, ?, ?, ?, GeomFromText(?, ?))"
        con.executemany(
            sql_row,
            (
                (
                    None,
                    row.BoroCode,
                    row.BoroName,
                    row.Shape_Leng,
                    row.Shape_Area,
                    row.geometry.wkt if row.geometry else None,
                    srid,
                )
                for row in df.itertuples(index=False)
            ),
        )


def create_postgis(con, df, srid=None, geom_col="geom"):
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
    if srid is not None:
        geom_schema = f"geometry(MULTIPOLYGON, {srid})"
        geom_insert = f"ST_SetSRID(ST_GeometryFromText(%s), {srid})"
    else:
        geom_schema = "geometry"
        geom_insert = "ST_GeometryFromText(%s)"
    try:
        cursor = con.cursor()
        cursor.execute("DROP TABLE IF EXISTS nybb;")

        sql = f"""CREATE TABLE nybb (
            {geom_col}   {geom_schema},
            borocode     integer,
            boroname     varchar(40),
            shape_leng   float,
            shape_area   float
            );"""
        cursor.execute(sql)

        for i, row in df.iterrows():
            sql = f"""INSERT INTO nybb VALUES ({geom_insert}, %s, %s, %s, %s
            );"""
            cursor.execute(
                sql,
                (
                    row["geometry"].wkt,
                    row["BoroCode"],
                    row["BoroName"],
                    row["Shape_Leng"],
                    row["Shape_Area"],
                ),
            )
    finally:
        cursor.close()
        con.commit()
