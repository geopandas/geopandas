from shapely.geometry import Polygon
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal


class TestUpdateInPlace:
    def setup_method(self):
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
        # descending order of "values"
        self.gdf = GeoDataFrame(
            {"geometry": GeoSeries([poly2, poly1], index=[1, 0]), "values": [2, 1]}
        )
        self.sorted_gdf = GeoDataFrame(
            {"geometry": GeoSeries([poly1, poly2]), "values": [1, 2]}
        )

    def test_order_changing(self):
        gdf = self.gdf.copy()
        old_sindex = gdf.sindex
        # sorting in place
        gdf.sort_values("values", inplace=True)
        # spatial index should be invalidated
        assert not gdf._sindex_generated
        new_sindex = gdf.sindex
        # and should be different
        assert new_sindex != old_sindex

        # sorting should still have happened though
        assert_geodataframe_equal(gdf, self.sorted_gdf)

    def test_order_unchanging(self):
        gdf = self.gdf.copy()
        old_sindex = gdf.sindex
        gdf.rename(columns={"values": "new_values"}, inplace=True)
        assert gdf._sindex_generated
        new_sindex = gdf.sindex
        assert old_sindex == new_sindex
