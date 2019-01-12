from __future__ import absolute_import

from collections import OrderedDict

import sys
import fiona
import pytest
from shapely.geometry import box

import geopandas
from geopandas import read_postgis, read_file
from geopandas.io.file import fiona_env
from geopandas.tests.util import connect, connect_spatialite, create_spatialite, create_postgis, validate_boro_df

@pytest.fixture
def nybb_df():
    nybb_path = geopandas.datasets.get_path('nybb')
    df = read_file(nybb_path)

    return df


class TestIO:
    def setup_method(self):
        nybb_zip_path = geopandas.datasets.get_path('nybb')
        self.df = read_file(nybb_zip_path)
        with fiona.open(nybb_zip_path) as f:
            self.crs = f.crs
            self.columns = list(f.meta["schema"]["properties"].keys())

    def test_read_postgis_default(self):
        con = connect('test_geopandas')
        if con is None or not create_postgis(self.df):
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

    def test_read_postgis_custom_geom_col(self):
        con = connect('test_geopandas')
        geom_col = "the_geom"
        if con is None or not create_postgis(self.df, geom_col=geom_col):
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con, geom_col=geom_col)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_select_geom_as(self):
        """Tests that a SELECT {geom} AS {some_other_geom} works."""
        con = connect('test_geopandas')
        orig_geom = "geom"
        out_geom = "the_geom"
        if con is None or not create_postgis(self.df, geom_col=orig_geom):
            raise pytest.skip()

        try:
            sql = """SELECT borocode, boroname, shape_leng, shape_area,
                     {} as {} FROM nybb;""".format(orig_geom, out_geom)
            df = read_postgis(sql, con, geom_col=out_geom)
        finally:
            con.close()

        validate_boro_df(df)

    def test_read_postgis_get_srid(self):
        """Tests that an SRID can be read from a geodatabase (GH #451)."""
        crs = {"init": "epsg:4269"}
        df_reproj = self.df.to_crs(crs)
        created = create_postgis(df_reproj, srid=4269)
        con = connect('test_geopandas')
        if con is None or not created:
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con)
        finally:
            con.close()

        validate_boro_df(df)
        assert(df.crs == crs)

    def test_read_postgis_override_srid(self):
        """Tests that a user specified CRS overrides the geodatabase SRID."""
        orig_crs = self.df.crs
        created = create_postgis(self.df, srid=4269)
        con = connect('test_geopandas')
        if con is None or not created:
            raise pytest.skip()

        try:
            sql = "SELECT * FROM nybb;"
            df = read_postgis(sql, con, crs=orig_crs)
        finally:
            con.close()

        validate_boro_df(df)
        assert(df.crs == orig_crs)

    def test_read_postgis_null_geom(self):
        """Tests that geometry with NULL is accepted."""
        try:
            con = connect_spatialite()
        except Exception:
            raise pytest.skip()
        else:
            geom_col = self.df.geometry.name
            self.df.geometry.iat[0] = None
            create_spatialite(con, self.df)
            sql = 'SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, AsEWKB("{0}") AS "{0}" FROM nybb'.format(geom_col)
            df = read_postgis(sql, con, geom_col=geom_col)
            validate_boro_df(df)
        finally:
            if 'con' in locals():
                con.close()

    def test_read_postgis_binary(self):
        """Tests that geometry read as binary is accepted."""
        try:
            con = connect_spatialite()
        except Exception:
            raise pytest.skip()
        else:
            geom_col = self.df.geometry.name
            create_spatialite(con, self.df)
            sql = 'SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, ST_AsBinary("{0}") AS "{0}" FROM nybb'.format(geom_col)
            df = read_postgis(sql, con, geom_col=geom_col)
            validate_boro_df(df)
        finally:
            if 'con' in locals():
                con.close()

    def test_read_file(self):
        df = self.df.rename(columns=lambda x: x.lower())
        validate_boro_df(df)
        assert df.crs == self.crs
        # get lower case columns, and exclude geometry column from comparison
        lower_columns = [c.lower() for c in self.columns]
        assert (df.columns[:-1] == lower_columns).all()

    @pytest.mark.web
    def test_remote_geojson_url(self):
        url = ("https://raw.githubusercontent.com/geopandas/geopandas/"
               "master/examples/null_geom.geojson")
        gdf = read_file(url)
        assert isinstance(gdf, geopandas.GeoDataFrame)

    def test_filtered_read_file(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = (1031051.7879884212, 224272.49231459625, 1047224.3104931959,
                244317.30894023244)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)

    def test_filtered_read_file_with_gdf_boundary(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = geopandas.GeoDataFrame(
            geometry=[box(1031051.7879884212, 224272.49231459625,
                          1047224.3104931959, 244317.30894023244)],
            crs=self.crs)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)

    def test_filtered_read_file_with_gdf_boundary_mismatched_crs(self):
        full_df_shape = self.df.shape
        nybb_filename = geopandas.datasets.get_path('nybb')
        bbox = geopandas.GeoDataFrame(
            geometry=[box(1031051.7879884212, 224272.49231459625,
                          1047224.3104931959, 244317.30894023244)],
            crs=self.crs)
        bbox.to_crs(epsg=4326, inplace=True)
        filtered_df = read_file(nybb_filename, bbox=bbox)
        filtered_df_shape = filtered_df.shape
        assert full_df_shape != filtered_df_shape
        assert filtered_df_shape == (2, 5)

    def test_empty_shapefile(self, tmpdir):

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
            with fiona.open(fname, 'w', **meta) as _:
                pass

        empty = read_file(fname)
        assert isinstance(empty, geopandas.GeoDataFrame)
        assert all(empty.columns == ['A', 'Z', 'geometry'])



    def test_to_postgis(self):

        con = connect_sqlalchemy()

        try:
            db_name = "nybb_write"
            self.df.to_postgis(db_name, con, if_exists="replace")
            back = read_postgis("SELECT * FROM {};".format(db_name), con,
                                geom_col="geometry")
        finally:
            con.close()
        validate_boro_df(back)

    def test_to_postgis_set_srid(self):

        con = connect_sqlalchemy()

        crs = {"init": "epsg:4326"}
        df_reproj = self.df.to_crs(crs)

        try:
            db_name = "nybb4326_write"
            df_reproj.to_postgis(db_name, con, if_exists="replace")
            back = read_postgis("SELECT * FROM {};".format(db_name), con,
                                geom_col="geometry")
        finally:
            con.close()

        validate_boro_df(back)
        assert back.crs == crs

    def test_to_postgis_no_srid(self):

        con = connect_sqlalchemy()

        df_noproj = self.df.copy()
        df_noproj.crs = None

        try:
            db_name = "nybb_nocrs_write"
            df_noproj.to_postgis(db_name, con, if_exists="replace")
            back = read_postgis("SELECT * FROM {};".format(db_name), con,
                                geom_col="geometry")
        finally:
            con.close()
        validate_boro_df(back)
        assert back.crs is None
