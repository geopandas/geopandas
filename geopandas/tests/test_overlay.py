from __future__ import absolute_import

import os

import pandas as pd
from shapely.geometry import Point, Polygon

import geopandas
from geopandas import GeoDataFrame, GeoSeries, read_file, overlay
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas import datasets

import pytest


DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


# Load qgis overlays
qgispath = datasets._module_path+'/qgis_overlay/'
union_qgis = read_file(qgispath+'union_qgis.shp')
diff_qgis = read_file(qgispath+'diff_qgis.shp')
symdiff_qgis = read_file(qgispath+'symdiff_qgis.shp')
intersect_qgis = read_file(qgispath+'intersect_qgis.shp')
ident_qgis = union_qgis[union_qgis.BoroCode.isnull()==False].copy()
df1 = read_file(qgispath+'polydf.shp')
df2 = read_file(qgispath+'polydf2.shp')
# Eliminate observations without geometries (issue from QGIS)
union_qgis = union_qgis[union_qgis.is_valid]
union_qgis.reset_index(inplace=True, drop=True)
diff_qgis = diff_qgis[diff_qgis.is_valid]
diff_qgis.reset_index(inplace=True, drop=True)
symdiff_qgis = symdiff_qgis[symdiff_qgis.is_valid]
symdiff_qgis.reset_index(inplace=True, drop=True)
intersect_qgis = intersect_qgis[intersect_qgis.is_valid]
intersect_qgis.reset_index(inplace=True, drop=True)
ident_qgis = ident_qgis[ident_qgis.is_valid]
ident_qgis.reset_index(inplace=True, drop=True)
# Order GeoDataFrames
cols = ['BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area', 'value1', 'value2']
union_qgis.sort_values(cols, inplace=True)
union_qgis.reset_index(inplace=True, drop=True)
symdiff_qgis.sort_values(cols, inplace=True)
symdiff_qgis.reset_index(inplace=True, drop=True)
intersect_qgis.sort_values(cols, inplace=True)
intersect_qgis.reset_index(inplace=True, drop=True)
ident_qgis.sort_values(cols, inplace=True)
ident_qgis.reset_index(inplace=True, drop=True)
diff_qgis.sort_values(cols[:-2], inplace=True)
diff_qgis.reset_index(inplace=True, drop=True)


class TestDataFrame():
    def setUp(self):
        # Create original data again
        N = 10
        nybb_filename = geopandas.datasets.get_path('nybb')
        self.polydf = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame([
            {'geometry' : Point(x, y).buffer(10000), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)

class TestOverlayNYBB:

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
        #self.polydf2._generate_sindex()

        self.union_shape = union_qgis.shape

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, union_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/union_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/union_qgis.boundary.length).mean()==1)
        assert type(df) is GeoDataFrame
        assert df.shape == self.union_shape
        assert 'value1' in df.columns and 'Shape_Area' in df.columns

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, intersect_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/intersect_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/intersect_qgis.boundary.length).mean()==1)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, ident_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/ident_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/ident_qgis.boundary.length).mean()==1)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, symdiff_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/symdiff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/symdiff_qgis.boundary.length).mean()==1)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        df.sort_values(cols[:-2], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, diff_qgis.shape)
        self.assertTrue('value1' not in df.columns and 'Shape_Area' in df.columns)
        self.assertTrue((df.area/diff_qgis.area).mean()==1)
        self.assertTrue((df.boundary.length/diff_qgis.boundary.length).mean()==1)

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


@pytest.fixture
def dfs(request):
    s1 = GeoSeries([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])
    s2 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df1 = GeoDataFrame({'geometry': s1, 'col1': [1, 2]})
    df2 = GeoDataFrame({'geometry': s2, 'col2': [1, 2]})
    return df1, df2


@pytest.fixture(params=['default-index', 'int-index', 'string-index'])
def dfs_index(request, dfs):
    df1, df2 = dfs
    if request.param == 'int-index':
        df1.index = [1, 2]
        df2.index = [0, 2]
    if request.param == 'string-index':
        df1.index = ['row1', 'row2']
    return df1, df2


@pytest.fixture(params=['union', 'intersection', 'difference',
                        'symmetric_difference', 'identity'])
def how(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_sindex(request):
    return request.param


def test_overlay(dfs_index, how, use_sindex):
    """
    Basic overlay test with small dummy example dataframes (from docs).
    Results obtained using QGIS 2.16 (Vector -> Geoprocessing Tools ->
    Intersection / Union / ...), saved to GeoJSON
    """
    df1, df2 = dfs_index
    result = overlay(df1, df2, how=how, use_sindex=use_sindex)

    # construction of result

    def _read(name):
        expected = read_file(
            os.path.join(DATA, 'polys', 'df1_df2-{0}.geojson'.format(name)))
        expected.crs = None
        return expected

    if how == 'identity':
        expected_intersection = _read('intersection')
        expected_difference = _read('difference')
        expected = pd.concat([
            expected_intersection,
            expected_difference
        ], ignore_index=True)
    else:
        expected = _read(how)

    # TODO needed adaptations to result
    if how == 'union':
        result = result.drop(['idx1', 'idx2'], axis=1).sort_values(['col1', 'col2']).reset_index(drop=True)
    elif how in ('intersection', 'identity'):
        result = result.drop(['idx1', 'idx2'], axis=1)
    elif how == 'difference':
        result = result.reset_index(drop=True)

    assert_geodataframe_equal(result, expected)

    # for difference also reversed
    if how == 'difference':
        result = overlay(df2, df1, how=how, use_sindex=use_sindex)
        result = result.reset_index(drop=True)
        expected = _read('difference-inverse')
        assert_geodataframe_equal(result, expected)


def test_overlay_overlap(how):
    """
    Overlay test with overlapping geometries in both dataframes.
    Test files are created with::

        import geopandas
        from geopandas import GeoSeries, GeoDataFrame
        from shapely.geometry import Point, Polygon, LineString

        s1 = GeoSeries([Point(0, 0), Point(1.5, 0)]).buffer(1, resolution=2)
        s2 = GeoSeries([Point(1, 1), Point(2, 2)]).buffer(1, resolution=2)

        df1 = GeoDataFrame({'geometry': s1, 'col1':[1,2]})
        df2 = GeoDataFrame({'geometry': s2, 'col2':[1, 2]})

        ax = df1.plot(alpha=0.5)
        df2.plot(alpha=0.5, ax=ax, color='C1')

        df1.to_file('geopandas/geopandas/tests/data/df1_overlap.geojson',
                    driver='GeoJSON')
        df2.to_file('geopandas/geopandas/tests/data/df2_overlap.geojson',
                    driver='GeoJSON')

    and then overlay results are obtained from using  QGIS 2.16
    (Vector -> Geoprocessing Tools -> Intersection / Union / ...),
    saved to GeoJSON.
    """
    df1 = read_file(os.path.join(DATA, 'df1_overlap.geojson'))
    df2 = read_file(os.path.join(DATA, 'df2_overlap.geojson'))

    result = overlay(df1, df2, how=how)

    if how == 'identity':
        raise pytest.skip()

    expected = read_file(os.path.join(DATA, 'df1_df2_overlap-{0}.geojson'.format(how)))


    # TODO needed adaptations to result
    result = result.drop(['idx1', 'idx2'], axis=1, errors='ignore').reset_index(drop=True)
    if how == 'union':
        result = result.sort_values(['col1', 'col2']).reset_index(drop=True)

    if how == 'union':
        # the QGIS result has the last row duplicated, so removing this
        expected = expected.iloc[:-1]

    assert_geodataframe_equal(result, expected,
                              check_less_precise=True)


@pytest.mark.parametrize('other_geometry', [False, True])
def test_geometry_not_named_geometry(dfs, how, other_geometry):
    # Issue #306
    # Add points and flip names
    df1, df2 = dfs
    df3 = df1.copy()
    df3 = df3.rename(columns={'geometry': 'polygons'})
    df3 = df3.set_geometry('polygons')
    if other_geometry:
        df3['geometry'] = df1.centroid.geometry
    assert df3.geometry.name == 'polygons'

    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df3, df2, how=how)

    assert df3.geometry.name == 'polygons'

    if how == 'difference':
        # in case of 'difference', column names of left frame are preserved
        assert res2.geometry.name == 'polygons'
        if other_geometry:
            assert 'geometry' in res2.columns
            assert_geoseries_equal(res2['geometry'], df3['geometry'],
                                   check_series_type=False)
            res2 = res2.drop(['geometry'], axis=1)
        res2 = res2.rename(columns={'polygons':'geometry'})
        res2 = res2.set_geometry('geometry')

    # TODO if existing column is overwritten -> geometry not last column
    if other_geometry and how == 'intersection':
        res2 = res2.reindex(columns=res1.columns)
    assert_geodataframe_equal(res1, res2)

    df4 = df2.copy()
    df4 = df4.rename(columns={'geometry': 'geom'})
    df4 = df4.set_geometry('geom')
    if other_geometry:
        df4['geometry'] = df2.centroid.geometry
    assert df4.geometry.name == 'geom'

    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df1, df4, how=how)
    assert_geodataframe_equal(res1, res2)


def test_bad_how(dfs):
    df1, df2 = dfs
    with pytest.raises(ValueError):
        overlay(df1, df2, how="spandex")


def test_raise_nonpoly(dfs):
    polydf, _ = dfs
    pointdf = polydf.copy()
    pointdf['geometry'] = pointdf.geometry.centroid

    with pytest.raises(TypeError):
        overlay(pointdf, polydf, how="union")


def test_duplicate_column_name(dfs):
    df1, df2 = dfs
    df2r = df2.rename(columns={'col2': 'col1'})
    res = overlay(df1, df2r, how="union")
    assert ('col1_1' in res.columns) and ('col1_2' in res.columns)


def test_geoseries_warning(dfs):
    df1, df2 = dfs
    # Issue #305
    with pytest.raises(NotImplementedError):
        overlay(df1, df2.geometry, how="union")
