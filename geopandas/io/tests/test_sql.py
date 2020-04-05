"""
Tests here include reading/writing to different types of spatial databases.
The spatial database tests may not work without additional system
configuration. postGIS tests require a test database to have been setup;
see geopandas.tests.util for more information.
"""

import geopandas
from geopandas import read_file, read_postgis

from geopandas.tests.util import (
    connect,
    connect_spatialite,
    create_postgis,
    create_spatialite,
    validate_boro_df,
)
import pytest


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path("nybb")
    df = read_file(nybb_path)
    return df


class TestIO:
    def test_read_postgis_default(self, df_nybb):
        con = connect("test_geopandas")
        if con is None or not create_postgis(df_nybb):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None

    def test_read_postgis_custom_geom_col(self, df_nybb):
        con = connect("test_geopandas")
        geom_col = "the_geom"
        if con is None or not create_postgis(df_nybb, geom_col=geom_col):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con, geom_col=geom_col)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_select_geom_as(self, df_nybb):
        """Tests that a SELECT {geom} AS {some_other_geom} works."""
        con = connect("test_geopandas")
        orig_geom = "geom"
        out_geom = "the_geom"
        if con is None or not create_postgis(df_nybb, geom_col=orig_geom):
            raise pytest.skip()

        try:
            sql = """SELECT borocode, boroname, shape_leng, shape_area,
                     {} as {} FROM nybb;""".format(
                orig_geom, out_geom
            )
            df = read_postgis(sql, con, geom_col=out_geom)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_get_srid(self, df_nybb):
        """Tests that an SRID can be read from a geodatabase (GH #451)."""
        crs = "epsg:4269"
        df_reproj = df_nybb.to_crs(crs)
        created = create_postgis(df_reproj, srid=4269)
        con = connect("test_geopandas")
        if con is None or not created:
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df)
        assert df.crs == crs

    def test_read_postgis_override_srid(self, df_nybb):
        """Tests that a user specified CRS overrides the geodatabase SRID."""
        orig_crs = df_nybb.crs
        created = create_postgis(df_nybb, srid=4269)
        con = connect("test_geopandas")
        if con is None or not created:
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con, crs=orig_crs)
        finally:
            con.close()

        validate_boro_df(df)
        assert df.crs == orig_crs

    def test_read_postgis_null_geom(self, df_nybb):
        """Tests that geometry with NULL is accepted."""
        try:
            con = connect_spatialite()
        except Exception:
            raise pytest.skip()
        else:
            geom_col = df_nybb.geometry.name
            df_nybb.geometry.iat[0] = None
            create_spatialite(con, df_nybb)
            sql = (
                "SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, "
                'AsEWKB("{0}") AS "{0}" FROM nybb'.format(geom_col)
            )
            df = read_postgis(sql, con, geom_col=geom_col)
            validate_boro_df(df)
        finally:
            if "con" in locals():
                con.close()

    def test_read_postgis_binary(self, df_nybb):
        """Tests that geometry read as binary is accepted."""
        try:
            con = connect_spatialite()
        except Exception:
            raise pytest.skip()
        else:
            geom_col = df_nybb.geometry.name
            create_spatialite(con, df_nybb)
            sql = (
                "SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, "
                'ST_AsBinary("{0}") AS "{0}" FROM nybb'.format(geom_col)
            )
            df = read_postgis(sql, con, geom_col=geom_col)
            validate_boro_df(df)
        finally:
            if "con" in locals():
                con.close()
