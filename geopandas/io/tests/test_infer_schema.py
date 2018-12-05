from collections import OrderedDict
from unittest import TestCase

from hamcrest import has_entries, contains_inanyorder
from hamcrest.core import assert_that
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, LineString, MultiLineString

from geopandas import GeoDataFrame
from geopandas.io.file import infer_schema


class TestInferSchema(TestCase):

    # Credit: Polygons below come from Montreal city Open Data portal
    # http://donnees.ville.montreal.qc.ca/dataset/unites-evaluation-fonciere
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

    point_3D = Point(-73.553785, 45.508722, 300)
    linestring_3D = LineString((
        (-73.5541107525234, 45.5091983609661, 300),
        (-73.5546126200639, 45.5086813829106, 300),
        (-73.5540185061397, 45.5084409343852, 300)
    ))
    polygon_3D = Polygon((
        (-73.5541107525234, 45.5091983609661, 300),
        (-73.5535801792994, 45.5089539203786, 300),
        (-73.5541107525234, 45.5091983609661, 300)
    ))

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

        assert_that(infer_schema(df), has_entries({'geometry': contains_inanyorder('MultiPoint', 'Point'), 'properties': OrderedDict()}))

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

        assert_that(infer_schema(df), has_entries({'geometry': contains_inanyorder('MultiLineString', 'LineString'), 'properties': OrderedDict()}))

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

        assert_that(infer_schema(df), has_entries({'geometry': contains_inanyorder('MultiPolygon', 'Polygon'), 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_only_multipolygons(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPolygon((self.city_hall_boundaries, self.vauquelin_place))]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'MultiPolygon', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_multiple_shape_types(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPolygon((self.city_hall_boundaries, self.vauquelin_place)),
                      self.city_hall_boundaries,
                      MultiLineString(self.city_hall_walls),
                      self.city_hall_walls[0],
                      MultiPoint([self.city_hall_entrance, self.city_hall_balcony]),
                      self.city_hall_balcony]
        )

        assert_that(infer_schema(df), has_entries({'geometry': contains_inanyorder('MultiPolygon', 'Polygon', 'MultiLineString', 'LineString', 'MultiPoint', 'Point'),
                                                   'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_a_3D_shape_type(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[MultiPolygon((self.city_hall_boundaries, self.vauquelin_place)),
                      self.city_hall_boundaries,
                      MultiLineString(self.city_hall_walls),
                      self.city_hall_walls[0],
                      MultiPoint([self.city_hall_entrance, self.city_hall_balcony]),
                      self.city_hall_balcony,
                      self.point_3D]
        )

        assert_that(infer_schema(df), has_entries({'geometry': contains_inanyorder('3D MultiPolygon', '3D Polygon', '3D MultiLineString', '3D LineString', '3D MultiPoint', '3D Point'), 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_a_3D_Point(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[self.city_hall_balcony, self.point_3D]
        )

        assert_that(infer_schema(df), has_entries({'geometry': '3D Point', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_a_3D_linestring(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[self.city_hall_walls[0], self.linestring_3D]
        )

        assert_that(infer_schema(df), has_entries({'geometry': '3D LineString', 'properties': OrderedDict()}))

    def test_infer_schema_when_dataframe_has_a_3D_Polygon(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[self.city_hall_boundaries, self.polygon_3D]
        )

        assert_that(infer_schema(df), has_entries({'geometry': '3D Polygon', 'properties': OrderedDict()}))

    def test_infer_schema_when_one_geometry_is_null(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[None, self.city_hall_entrance]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'Unknown', 'properties': OrderedDict()}))

    def test_infer_schema_when_geometries_are_null_and_3D_point(self):
        df = GeoDataFrame(
            {},
            crs={'init': 'epsg:4326', 'no_defs': True},
            geometry=[None, self.point_3D]
        )

        assert_that(infer_schema(df), has_entries({'geometry': 'Unknown', 'properties': OrderedDict()}))
