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

    def test_update_in_place(self):
        old_sindex = self.gdf.sindex
        # sorting in place
        self.gdf.sort_values("values", inplace=True)
        # spatial index should be invalidated
        assert not self.gdf._sindex_generated
        new_sindex = self.gdf.sindex
        # and should be different
        assert new_sindex != old_sindex

        # sorting should still have happened though
        assert_geodataframe_equal(self.gdf, self.sorted_gdf)
