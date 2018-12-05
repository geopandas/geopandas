import os
import shutil
import tempfile

import pytest
from pandas.util.testing import assert_frame_equal
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, LineString, MultiLineString

import geopandas
from geopandas import GeoDataFrame

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


@pytest.fixture(params=[
    # Points
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=[city_hall_entrance, city_hall_balcony]
    ),
    # Points and MultiPoints
    # FIXME : Fails with GeometryTypeValidationError
    # GeoDataFrame(
    #     {},
    #     crs={'init': 'epsg:4326', 'no_defs': True},
    #     geometry=[MultiPoint([city_hall_entrance, city_hall_balcony]), city_hall_balcony]
    # ),
    # MultiPoints
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=[MultiPoint([city_hall_balcony, city_hall_council_chamber]), MultiPoint([city_hall_entrance, city_hall_balcony, city_hall_council_chamber])]
    ),
    # LineStrings
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=city_hall_walls
    ),
    # LineStrings and MultiLineStrings
    # FIXME : Fails with GeometryTypeValidationError
    # GeoDataFrame(
    #     {},
    #     crs={'init': 'epsg:4326', 'no_defs': True},
    #     geometry=[MultiLineString(city_hall_walls), city_hall_walls[0]]
    # ),
    # MultiLineStrings
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=[MultiLineString(city_hall_walls), MultiLineString(city_hall_walls[1:])]
    ),
    # Polygons
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=[city_hall_boundaries, vauquelin_place]
    ),
    # MultiPolygon and Polygon
    # FIXME : Fails with GeometryTypeValidationError
    # GeoDataFrame(
    #     {},
    #     crs={'init': 'epsg:4326', 'no_defs': True},
    #     geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place)), city_hall_boundaries]
    # ),
    # MultiPolygon
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326', 'no_defs': True},
        geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place))]
    ),
    # all shape types
    # FIXME : Fails with GeometryTypeValidationError
    # GeoDataFrame(
    #     {},
    #     crs={'init': 'epsg:4326', 'no_defs': True},
    #     geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place)),
    #               city_hall_entrance,
    #               MultiLineString(city_hall_walls),
    #               city_hall_walls[0],
    #               MultiPoint([city_hall_entrance, city_hall_balcony]),
    #               city_hall_balcony]
    # )
])
def geodataframe(request):
    return request.param


@pytest.fixture(params=['GeoJSON'])
def ogr_driver(request):
    return request.param


class TestGeoDataFrameToFile():

    def setup_method(self):
        self.output_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.output_dir, "output_file")

    def teardown_method(self):
        shutil.rmtree(self.output_dir)

    def test_geodataframe_to_file(self, geodataframe, ogr_driver):
        geodataframe.to_file(self.output_file, driver=ogr_driver)

        reloaded = geopandas.read_file(self.output_file)

        assert_frame_equal(geodataframe, reloaded)
