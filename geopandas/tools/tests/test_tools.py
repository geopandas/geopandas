from distutils.version import LooseVersion

from shapely.geometry import LineString, MultiPoint, Point
import pyproj
from pyproj import CRS

from geopandas import GeoSeries
from geopandas.tools import collect
from geopandas.tools.crs import epsg_from_crs, explicit_crs_from_epsg

import pytest


# pyproj 2.3.1 fixed a segfault for the case working in an environment with
# 'init' dicts (https://github.com/pyproj4/pyproj/issues/415)
PYPROJ_LT_231 = LooseVersion(pyproj.__version__) < LooseVersion("2.3.1")


class TestTools:
    def setup_method(self):
        self.p1 = Point(0, 0)
        self.p2 = Point(1, 1)
        self.p3 = Point(2, 2)
        self.mpc = MultiPoint([self.p1, self.p2, self.p3])

        self.mp1 = MultiPoint([self.p1, self.p2])
        self.line1 = LineString([(3, 3), (4, 4)])

    def test_collect_single(self):
        result = collect(self.p1)
        assert self.p1.equals(result)

    def test_collect_single_force_multi(self):
        result = collect(self.p1, multi=True)
        expected = MultiPoint([self.p1])
        assert expected.equals(result)

    def test_collect_multi(self):
        result = collect(self.mp1)
        assert self.mp1.equals(result)

    def test_collect_multi_force_multi(self):
        result = collect(self.mp1)
        assert self.mp1.equals(result)

    def test_collect_list(self):
        result = collect([self.p1, self.p2, self.p3])
        assert self.mpc.equals(result)

    def test_collect_GeoSeries(self):
        s = GeoSeries([self.p1, self.p2, self.p3])
        result = collect(s)
        assert self.mpc.equals(result)

    def test_collect_mixed_types(self):
        with pytest.raises(ValueError):
            collect([self.p1, self.line1])

    def test_collect_mixed_multi(self):
        with pytest.raises(ValueError):
            collect([self.mpc, self.mp1])

    @pytest.mark.skipif(PYPROJ_LT_231, reason="segfault")
    def test_epsg_from_crs(self):
        with pytest.warns(FutureWarning):
            assert epsg_from_crs({"init": "epsg:4326"}) == 4326
            assert epsg_from_crs({"init": "EPSG:4326"}) == 4326
            assert epsg_from_crs("+init=epsg:4326") == 4326

    @pytest.mark.skipif(PYPROJ_LT_231, reason="segfault")
    def test_explicit_crs_from_epsg(self):
        with pytest.warns(FutureWarning):
            assert explicit_crs_from_epsg(epsg=4326) == CRS.from_epsg(4326)
            assert explicit_crs_from_epsg(epsg="4326") == CRS.from_epsg(4326)
            assert explicit_crs_from_epsg(crs={"init": "epsg:4326"}) == CRS.from_dict(
                {"init": "epsg:4326"}
            )
            assert explicit_crs_from_epsg(crs="+init=epsg:4326") == CRS.from_proj4(
                "+init=epsg:4326"
            )

    @pytest.mark.filterwarnings("ignore:explicit_crs_from_epsg:FutureWarning")
    def test_explicit_crs_from_epsg__missing_input(self):
        with pytest.raises(ValueError):
            explicit_crs_from_epsg()
