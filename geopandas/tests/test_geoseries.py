from __future__ import absolute_import

import os
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from shapely.geometry import (Polygon, Point, LineString,
                              MultiPoint, MultiLineString, MultiPolygon)
from shapely.geometry.base import BaseGeometry

from geopandas import GeoSeries

import pytest
from geopandas.tests.util import geom_equals
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal


class TestSeries:

    def setup_method(self):
        self.tempdir = tempfile.mkdtemp()
        self.t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        self.t2 = Polygon([(0, 0), (1, 1), (0, 1)])
        self.sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g1 = GeoSeries([self.t1, self.sq])
        self.g2 = GeoSeries([self.sq, self.t1])
        self.g3 = GeoSeries([self.t1, self.t2])
        self.g3.crs = {'init': 'epsg:4326', 'no_defs': True}
        self.g4 = GeoSeries([self.t2, self.t1])
        self.na = GeoSeries([self.t1, self.t2, Polygon()])
        self.na_none = GeoSeries([self.t1, self.t2, None])
        self.a1 = self.g1.copy()
        self.a1.index = ['A', 'B']
        self.a2 = self.g2.copy()
        self.a2.index = ['B', 'C']
        self.esb = Point(-73.9847, 40.7484)
        self.sol = Point(-74.0446, 40.6893)
        self.landmarks = GeoSeries([self.esb, self.sol],
                                   crs={'init': 'epsg:4326', 'no_defs': True})
        self.l1 = LineString([(0, 0), (0, 1), (1, 1)])
        self.l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.g5 = GeoSeries([self.l1, self.l2])

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_single_geom_constructor(self):
        p = Point(1, 2)
        line = LineString([(2, 3), (4, 5), (5, 6)])
        poly = Polygon([(0, 0), (1, 0), (1, 1)],
                       [[(.1, .1), (.9, .1), (.9, .9)]])
        mp = MultiPoint([(1, 2), (3, 4), (5, 6)])
        mline = MultiLineString([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10)]])

        poly2 = Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1)],
                        [[(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)]])
        mpoly = MultiPolygon([poly, poly2])

        geoms = [p, line, poly, mp, mline, mpoly]
        index = ['a', 'b', 'c', 'd']

        for g in geoms:
            gs = GeoSeries(g)
            assert len(gs) == 1
            assert gs.iloc[0] is g

            gs = GeoSeries(g, index=index)
            assert len(gs) == len(index)
            for x in gs:
                assert x is g

    def test_copy(self):
        gc = self.g3.copy()
        assert type(gc) is GeoSeries
        assert self.g3.name == gc.name
        assert self.g3.crs == gc.crs

    def test_in(self):
        assert self.t1 in self.g1
        assert self.sq in self.g1
        assert self.t1 in self.a1
        assert self.t2 in self.g3
        assert self.sq not in self.g3
        assert 5 not in self.g3

    def test_geom_equals(self):
        assert np.all(self.g1.geom_equals(self.g1))
        assert_array_equal(self.g1.geom_equals(self.sq), [False, True])

    def test_geom_equals_align(self):
        a = self.a1.geom_equals(self.a2)
        exp = pd.Series([False, True, False], index=['A', 'B', 'C'])
        assert_series_equal(a, exp)

    def test_align(self):
        a1, a2 = self.a1.align(self.a2)
        assert a2['A'].is_empty
        assert a1['B'].equals(a2['B'])
        assert a1['C'].is_empty

    def test_geom_almost_equals(self):
        # TODO: test decimal parameter
        assert np.all(self.g1.geom_almost_equals(self.g1))
        assert_array_equal(self.g1.geom_almost_equals(self.sq), [False, True])

    def test_geom_equals_exact(self):
        # TODO: test tolerance parameter
        assert np.all(self.g1.geom_equals_exact(self.g1, 0.001))
        assert_array_equal(self.g1.geom_equals_exact(self.sq, 0.001),
                           [False, True])

    def test_equal_comp_op(self):
        s = GeoSeries([Point(x, x) for x in range(3)])
        res = s == Point(1, 1)
        exp = pd.Series([False, True, False])
        assert_series_equal(res, exp)

    def test_to_file(self):
        """ Test to_file and from_file """
        tempfilename = os.path.join(self.tempdir, 'test.shp')
        self.g3.to_file(tempfilename)
        # Read layer back in?
        s = GeoSeries.from_file(tempfilename)
        assert all(self.g3.geom_equals(s))
        # TODO: compare crs

    def test_to_json(self):
        """
        Test whether GeoSeries.to_json works and returns an actual json file.
        """
        json_str = self.g3.to_json()
        json_dict = json.loads(json_str)
        # TODO : verify the output is a valid GeoJSON.

    def test_representative_point(self):
        assert np.all(self.g1.contains(self.g1.representative_point()))
        assert np.all(self.g2.contains(self.g2.representative_point()))
        assert np.all(self.g3.contains(self.g3.representative_point()))
        assert np.all(self.g4.contains(self.g4.representative_point()))

    def test_transform(self):
        utm18n = self.landmarks.to_crs(epsg=26918)
        lonlat = utm18n.to_crs(epsg=4326)
        assert np.all(self.landmarks.geom_almost_equals(lonlat))
        with pytest.raises(ValueError):
            self.g1.to_crs(epsg=4326)
        with pytest.raises(TypeError):
            self.landmarks.to_crs(crs=None, epsg=None)

    def test_fillna(self):
        # default is to fill with empty geometry
        na = self.na_none.fillna()
        assert isinstance(na[2], BaseGeometry)
        assert na[2].is_empty
        assert geom_equals(self.na_none[:2], na[:2])
        # XXX: method works inconsistently for different pandas versions
        # self.na_none.fillna(method='backfill')

    def test_coord_slice(self):
        """ Test CoordinateSlicer """
        # need some better test cases
        assert geom_equals(self.g3, self.g3.cx[:, :])
        assert geom_equals(self.g3[[True, False]], self.g3.cx[0.9:, :0.1])
        assert geom_equals(self.g3[[False, True]], self.g3.cx[0:0.1, 0.9:1.0])

    def test_coord_slice_with_zero(self):
        # Test that CoordinateSlice correctly handles zero slice (#GH477).

        gs = GeoSeries([Point(x, x) for x in range(-3, 4)])
        assert geom_equals(gs.cx[:0, :0], gs.loc[:3])
        assert geom_equals(gs.cx[:, :0], gs.loc[:3])
        assert geom_equals(gs.cx[:0, :], gs.loc[:3])
        assert geom_equals(gs.cx[0:, 0:], gs.loc[3:])
        assert geom_equals(gs.cx[0:, :], gs.loc[3:])
        assert geom_equals(gs.cx[:, 0:], gs.loc[3:])

    def test_geoseries_geointerface(self):
        assert self.g1.__geo_interface__['type'] == 'FeatureCollection'
        assert len(self.g1.__geo_interface__['features']) == self.g1.shape[0]

    def test_proj4strings(self):
        # As string
        reprojected = self.g3.to_crs('+proj=utm +zone=30N')
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # As dict
        reprojected = self.g3.to_crs({'proj': 'utm', 'zone': '30N'})
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # Set to equivalent string, convert, compare to original
        copy = self.g3.copy()
        copy.crs = '+init=epsg:4326'
        reprojected = copy.to_crs({'proj': 'utm', 'zone': '30N'})
        reprojected_back = reprojected.to_crs(epsg=4326)
        assert np.all(self.g3.geom_almost_equals(reprojected_back))

        # Conversions by different format
        reprojected_string = self.g3.to_crs('+proj=utm +zone=30N')
        reprojected_dict = self.g3.to_crs({'proj': 'utm', 'zone': '30N'})
        assert np.all(reprojected_string.geom_almost_equals(reprojected_dict))
