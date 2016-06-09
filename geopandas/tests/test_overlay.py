from __future__ import absolute_import

import tempfile
import shutil

import numpy as np

from pandas.util.testing import assert_frame_equal

from shapely.geometry import Point, Polygon

from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.tests.util import unittest, download_nybb, assert_geoseries_equal
from geopandas import overlay


class TestOverlayNYBB(unittest.TestCase):

    def setup_method(self):
        N = 10

        nybb_filename = geopandas.datasets.get_path('nybb')

        self.polydf = read_file(nybb_filename)
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame(
            [{'geometry': Point(x, y).buffer(10000), 'value1': x + y,
              'value2': x - y}
             for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                             range(b[1], b[3], int((b[3]-b[1])/N)))],
            crs=self.crs)
        self.pointdf = GeoDataFrame(
            [{'geometry': Point(x, y), 'value1': x + y, 'value2': x - y}
             for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                             range(b[1], b[3], int((b[3]-b[1])/N)))],
            crs=self.crs)

        # TODO this appears to be necessary;
        # why is the sindex not generated automatically?
        self.polydf2._generate_sindex()

        self.union_shape = (180, 7)

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        assert type(df) is GeoDataFrame
        assert df.shape == self.union_shape
        assert 'value1' in df.columns and 'Shape_Area' in df.columns

    def test_union_no_index(self):
        # explicitly ignore indices
        dfB = overlay(self.polydf, self.polydf2, how="union", use_sindex=False)
        assert dfB.shape == self.union_shape

        # remove indices from df
        self.polydf._sindex = None
        self.polydf2._sindex = None
        dfC = overlay(self.polydf, self.polydf2, how="union")
        assert dfC.shape == self.union_shape

    def test_union_non_numeric_index(self):
        import string
        letters = list(string.ascii_letters)

        polydf_alpha = self.polydf.copy()
        polydf2_alpha = self.polydf2.copy()
        polydf_alpha.index = letters[:len(polydf_alpha)]
        polydf2_alpha.index = letters[:len(polydf2_alpha)]
        df = overlay(polydf_alpha, polydf2_alpha, how="union")
        assert type(df) is GeoDataFrame
        assert df.shape == self.union_shape
        assert 'value1' in df.columns and 'Shape_Area' in df.columns

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        assert df['BoroName'][0] is not None
        assert df.shape == (68, 7)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        assert df.shape == (154, 7)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        assert df.shape == (122, 7)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        assert df.shape == (86, 7)

    def test_bad_how(self):
        with pytest.raises(ValueError):
            overlay(self.polydf, self.polydf, how="spandex")

    def test_nonpoly(self):
        with pytest.raises(TypeError):
            overlay(self.pointdf, self.polydf, how="union")

    def test_duplicate_column_name(self):
        polydf2r = self.polydf2.rename(columns={'value2': 'Shape_Area'})
        df = overlay(self.polydf, polydf2r, how="union")
        assert 'Shape_Area_2' in df.columns and 'Shape_Area' in df.columns

    def test_geometry_not_named_geometry(self):
        # Issue #306
        # Add points and flip names
        polydf3 = self.polydf.copy()
        polydf3 = polydf3.rename(columns={'geometry': 'polygons'})
        polydf3 = polydf3.set_geometry('polygons')
        polydf3['geometry'] = self.pointdf.geometry.loc[0:4]
        assert polydf3.geometry.name == 'polygons'

        df = overlay(polydf3, self.polydf2, how="union")
        assert type(df) is GeoDataFrame

        df2 = overlay(self.polydf, self.polydf2, how="union")
        assert df.geom_almost_equals(df2).all()

    def test_geoseries_warning(self):
        # Issue #305
        with pytest.raises(NotImplementedError):
            overlay(self.polydf, self.polydf2.geometry, how="union")


class TestOverlay(unittest.TestCase):

    use_sindex = True

    def setUp(self):

        s1 = GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
                        Polygon([(2,2), (4,2), (4,4), (2,4)])])
        s2 = GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
                        Polygon([(3,3), (5,3), (5,5), (3,5)])])

        self.df1 = GeoDataFrame({'geometry': s1, 'col1':[1,2]})
        self.df2 = GeoDataFrame({'geometry': s2, 'col2':[1,2]})

        self.result  = GeoDataFrame(
            {'col1': [1, 1, np.nan, np.nan, 2, 2, 2, np.nan, 2],
             'col2': [np.nan, 1, 1, 1, np.nan, 1, np.nan, 2, 2],
             'geometry': [Polygon([(2, 1), (2, 0), (0, 0), (0, 2), (1, 2), (1, 1), (2, 1)]),
                          Polygon([(2, 1), (1, 1), (1, 2), (2, 2), (2, 1)]),
                          Polygon([(2, 1), (2, 2), (3, 2), (3, 1), (2, 1)]),
                          Polygon([(2, 2), (1, 2), (1, 3), (2, 3), (2, 2)]),
                          Polygon([(3, 2), (3, 3), (4, 3), (4, 2), (3, 2)]),
                          Polygon([(3, 3), (3, 2), (2, 2), (2, 3), (3, 3)]),
                          Polygon([(3, 3), (2, 3), (2, 4), (3, 4), (3, 3)]),
                          Polygon([(4, 3), (4, 4), (3, 4), (3, 5), (5, 5), (5, 3), (4, 3)]),
                          Polygon([(3, 4), (4, 4), (4, 3), (3, 3), (3, 4)])]
            })

    def test_union(self):
        res = overlay(self.df1, self.df2, how='union',
                      use_sindex=self.use_sindex)
        exp = self.result
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_intersection(self):
        res = overlay(self.df1, self.df2, how='intersection',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1', 'col2'], how='any')
        exp = exp.reset_index(drop=True)
        exp[['col1', 'col2']] = exp[['col1', 'col2']].astype('int64')
        print(exp)
        print(res)
        print(self.result.geometry[7])
        print(self.result.geometry[7].representative_point())
        print(list(self.df1.sindex.intersection(self.result.geometry[7].bounds)))
        cent = self.result.geometry[7].representative_point()
        print(cent.intersects(self.df1.geometry[1]))
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_symdiff(self):
        res = overlay(self.df1, self.df2, how='symmetric_difference',
                      use_sindex=self.use_sindex)
        exp = self.result[self.result[['col1', 'col2']].isnull().sum(1) == 1]
        exp = exp.reset_index(drop=True)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_difference(self):
        res = overlay(self.df1, self.df2, how='difference',
                      use_sindex=self.use_sindex)
        exp = self.result.loc[[0, 4, 6]]
        exp = exp.reset_index(drop=True)
        exp['col1'] = exp['col1'].astype('int64')
        exp['col2'] = np.array([None, None, None], dtype='O')
        print(exp)
        print(res)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_identity(self):
        res = overlay(self.df1, self.df2, how='identity',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1'])
        exp = exp.reset_index(drop=True)
        exp['col1'] = exp['col1'].astype('int64')
        print(exp)
        print(res)
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)

    def test_nondefault_index(self):
        df1 = self.df1.copy()
        df1.index = ['row1', 'row2']
        res = overlay(df1, self.df2, how='intersection',
                      use_sindex=self.use_sindex)
        exp = self.result.dropna(subset=['col1', 'col2'], how='any')
        exp = exp.reset_index(drop=True)
        exp[['col1', 'col2']] = exp[['col1', 'col2']].astype('int64')
        assert_frame_equal(res, exp)
        assert_geoseries_equal(res.geometry, exp.geometry)


class TestOverlayNoSIndex(TestOverlay):

    use_sindex = False
