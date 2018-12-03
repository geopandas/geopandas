from collections import OrderedDict
from unittest import TestCase

from hamcrest import has_entries
from hamcrest.core import assert_that
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, LineString, MultiLineString

from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema


class TestInferSchema(TestCase):
    city_hall_boundaries = Polygon((
        (-73.5541107525234, 45.5091983609661),
        (-73.5546126200639, 45.5086813829106),
        (-73.5540185061397, 45.5084409343852),
        (-73.5539986525799, 45.5084323044531),
        (-73.5535801792994, 45.5089539203786),
        (-73.5541107525234, 45.5091983609661)
    ))
    vauquelin_place = Polygon((
        (-73.5542465586147, 45.5081555487952),
        (-73.5540185061397, 45.5084409343852),
        (-73.5546126200639, 45.5086813829106),
        (-73.5548825850032, 45.5084033554357),
        (-73.5542465586147, 45.5081555487952)
    ))

    city_hall_walls = [
        LineString((
            (-73.5541107525234, 45.5091983609661),
            (-73.5546126200639, 45.5086813829106),
            (-73.5540185061397, 45.5084409343852)
        )),
        LineString((
            (-73.5539986525799, 45.5084323044531),
            (-73.5535801792994, 45.5089539203786),
            (-73.5541107525234, 45.5091983609661)
        ))
    ]

    city_hall_entrance = Point(-73.553785, 45.508722)
    city_hall_balcony = Point(-73.554138, 45.509080)
    city_hall_council_chamber = Point(-73.554246, 45.508931)

    def test_infer_schema_when_dataframe_has_only_points(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[self.city_hall_entrance, self.city_hall_balcony]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'Point', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_points_and_multipoints(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPoint([self.city_hall_entrance, self.city_hall_balcony]), self.city_hall_balcony]
        )

        # FIXME : should probably be {'geometry': ['MultiPoint', 'Point'], 'properties': OrderedDict()}
        assert_that(infer_schema(df), has_entries({'geometry': 'Point', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_multipoints(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPoint([self.city_hall_entrance, self.city_hall_balcony, self.city_hall_council_chamber])]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'MultiPoint', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_linestrings(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=self.city_hall_walls
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'LineString', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_linestrings_and_multilinestrings(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiLineString(self.city_hall_walls), self.city_hall_walls[0]]
        )

        # FIXME : should probably be {'geometry': ['MultiLineString', 'LineString'], 'properties': OrderedDict()}
        assert_that(infer_schema(df), has_entries({'geometry': 'LineString', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_multilinestrings(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiLineString(self.city_hall_walls)]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'MultiLineString', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_polygons(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[self.city_hall_boundaries, self.vauquelin_place]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'Polygon', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_polygons_and_multipolygons(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPolygon((self.city_hall_boundaries, self.vauquelin_place)), self.city_hall_boundaries]
        )

        # FIXME : should probably be {'geometry': ['MultiPolygon', 'Polygon'], 'properties': OrderedDict()}
        assert_that(infer_schema(df), has_entries({'geometry': 'Polygon', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_multipolygons(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPolygon((self.city_hall_boundaries, self.vauquelin_place))]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'MultiPolygon', 'properties': OrderedDict()}))
