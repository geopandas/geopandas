from __future__ import absolute_import

import pandas as pd
from shapely.geometry import Point, Polygon

import geopandas
from geopandas import GeoDataFrame, GeoSeries, read_file, overlay
from geopandas.testing import assert_geodataframe_equal

import pytest


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


@pytest.fixture(params=[False, True], ids=['default-index', 'string-index'])
def dfs(request):
    s1 = GeoSeries([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])
    s2 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df1 = GeoDataFrame({'geometry': s1, 'col1': [1, 2]})
    df2 = GeoDataFrame({'geometry': s2, 'col2': [1, 2]})
    if request.param:
        df1.index = ['row1', 'row2']
    return df1, df2


@pytest.fixture(params=['union', 'intersection', 'difference',
                        'symmetric_difference', 'identity'])
def how(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_sindex(request):
    return request.param


@pytest.fixture
def expected_features():
    expected = {}
    expected['union'] = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": { "col1": 1.0, "col2": 1.0 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 1.0, 2.0 ], [ 2.0, 2.0 ], [ 2.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 2.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 1.0, "col2": None },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 2.0 ], [ 1.0, 2.0 ], [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 0.0 ], [ 0.0, 0.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 2.0, "col2": 1.0 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 2.0, 2.0 ], [ 2.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 2.0 ], [ 2.0, 2.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 2.0, "col2": 2.0 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 3.0, 4.0 ], [ 4.0, 4.0 ], [ 4.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 4.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 2.0, "col2": None },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 2.0, 3.0 ], [ 2.0, 4.0 ], [ 3.0, 4.0 ], [ 3.0, 3.0 ], [ 2.0, 3.0 ] ] ], [ [ [ 4.0, 3.0 ], [ 4.0, 2.0 ], [ 3.0, 2.0 ], [ 3.0, 3.0 ], [ 4.0, 3.0 ] ] ] ] } },
            {"type": "Feature", "properties": { "col1": None, "col2": 1.0 },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 1.0, 2.0 ], [ 1.0, 3.0 ], [ 2.0, 3.0 ], [ 2.0, 2.0 ], [ 1.0, 2.0 ] ] ], [ [ [ 3.0, 2.0 ], [ 3.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 2.0 ], [ 3.0, 2.0 ] ] ] ] } },
            {"type": "Feature", "properties": { "col1": None, "col2": 2.0 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 3.0, 4.0 ], [ 3.0, 5.0 ], [ 5.0, 5.0 ], [ 5.0, 3.0 ], [ 4.0, 3.0 ], [ 4.0, 4.0 ], [ 3.0, 4.0 ] ] ] } }
            ]
        }

    expected['intersection'] = {
        "type": "FeatureCollection",
        "features": [
            { "type": "Feature", "properties": { "col1": 1, "col2": 1 },
            "geometry": { "type": "Polygon", "coordinates": [ [ [ 1.0, 2.0 ], [ 2.0, 2.0 ], [ 2.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 2.0 ] ] ] } },
            { "type": "Feature", "properties": { "col1": 2, "col2": 1 },
            "geometry": { "type": "Polygon", "coordinates": [ [ [ 2.0, 2.0 ], [ 2.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 2.0 ], [ 2.0, 2.0 ] ] ] } },
            { "type": "Feature", "properties": { "col1": 2, "col2": 2 },
            "geometry": { "type": "Polygon", "coordinates": [ [ [ 3.0, 4.0 ], [ 4.0, 4.0 ], [ 4.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 4.0 ] ] ] } }
            ]
        }

    expected['symmetric_difference'] = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": { "col1": 1.0, "col2": None },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 2.0 ], [ 1.0, 2.0 ], [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 0.0 ], [ 0.0, 0.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 2.0, "col2": None },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 2.0, 3.0 ], [ 2.0, 4.0 ], [ 3.0, 4.0 ], [ 3.0, 3.0 ], [ 2.0, 3.0 ] ] ], [ [ [ 4.0, 3.0 ], [ 4.0, 2.0 ], [ 3.0, 2.0 ], [ 3.0, 3.0 ], [ 4.0, 3.0 ] ] ] ] } },
            {"type": "Feature", "properties": { "col1": None, "col2": 1.0 },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 1.0, 2.0 ], [ 1.0, 3.0 ], [ 2.0, 3.0 ], [ 2.0, 2.0 ], [ 1.0, 2.0 ] ] ], [ [ [ 3.0, 2.0 ], [ 3.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 2.0 ], [ 3.0, 2.0 ] ] ] ] } },
            {"type": "Feature", "properties": { "col1": None, "col2": 2.0 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 3.0, 4.0 ], [ 3.0, 5.0 ], [ 5.0, 5.0 ], [ 5.0, 3.0 ], [ 4.0, 3.0 ], [ 4.0, 4.0 ], [ 3.0, 4.0 ] ] ] } }
            ]
        }

    expected['difference'] = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": { "col1": 1 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 2.0 ], [ 1.0, 2.0 ], [ 1.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 0.0 ], [ 0.0, 0.0 ] ] ] } },
            {"type": "Feature", "properties": { "col1": 2 },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 2.0, 3.0 ], [ 2.0, 4.0 ], [ 3.0, 4.0 ], [ 3.0, 3.0 ], [ 2.0, 3.0 ] ] ], [ [ [ 4.0, 3.0 ], [ 4.0, 2.0 ], [ 3.0, 2.0 ], [ 3.0, 3.0 ], [ 4.0, 3.0 ] ] ] ] } }
            ]
        }

    expected['difference_inverse'] = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": { "col2": 1 },
                "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 1.0, 2.0 ], [ 1.0, 3.0 ], [ 2.0, 3.0 ], [ 2.0, 2.0 ], [ 1.0, 2.0 ] ] ], [ [ [ 3.0, 2.0 ], [ 3.0, 1.0 ], [ 2.0, 1.0 ], [ 2.0, 2.0 ], [ 3.0, 2.0 ] ] ] ] } },
            {"type": "Feature", "properties": { "col2": 2 },
                "geometry": { "type": "Polygon", "coordinates": [ [ [ 3.0, 4.0 ], [ 3.0, 5.0 ], [ 5.0, 5.0 ], [ 5.0, 3.0 ], [ 4.0, 3.0 ], [ 4.0, 4.0 ], [ 3.0, 4.0 ] ] ] } }
            ]
        }

    return expected


@pytest.mark.skip(reason="overlay not correctly implemented")
def test_overlay(dfs, how, use_sindex, expected_features):
    """
    Basic overlay test with small dummy example dataframes (from docs).
    Results obtained using QGIS 2.16 (Vector -> Geoprocessing Tools ->
    Intersection / Union / ...), saved to GeoJSON and pasted here
    """
    df1, df2 = dfs
    result = overlay(df1, df2, how=how, use_sindex=use_sindex)

    # construction of result
    if how == 'identity':
        expected = pd.concat([
            GeoDataFrame.from_features(expected_features['intersection']),
            GeoDataFrame.from_features(expected_features['difference'])
        ], ignore_index=True)
    else:
        expected = GeoDataFrame.from_features(expected_features[how])

    # TODO needed adaptations to result
    # if how == 'union':
    #     result = result.drop(['idx1', 'idx2'], axis=1).sort_values(['col1', 'col2']).reset_index(drop=True)
    # elif how in ('intersection', 'identity'):
    #     result = result.drop(['idx1', 'idx2'], axis=1)

    assert_geodataframe_equal(result, expected)

    # for difference also reversed
    if how == 'difference':
        result = overlay(df2, df1, how=how, use_sindex=use_sindex)
        expected = GeoDataFrame.from_features(
            expected_features['difference_inverse'])
        assert_geodataframe_equal(result, expected)
