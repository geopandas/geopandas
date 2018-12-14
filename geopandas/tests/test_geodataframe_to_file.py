import os
import shutil
import tempfile

import pytest
from shapely.geometry import Point, Polygon, MultiPolygon, MultiPoint, \
    LineString, MultiLineString

import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

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


@pytest.fixture(params=[
    # Points
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[city_hall_entrance, city_hall_balcony]
    ),
    # Points and MultiPoints
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiPoint([city_hall_entrance, city_hall_balcony]),
            city_hall_balcony
        ]
    ),
    # MultiPoints
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiPoint([
                city_hall_balcony,
                city_hall_council_chamber]),
            MultiPoint([
                city_hall_entrance,
                city_hall_balcony,
                city_hall_council_chamber]
            )]
    ),
    # LineStrings
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=city_hall_walls
    ),
    # LineStrings and MultiLineStrings
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[MultiLineString(city_hall_walls), city_hall_walls[0]]
    ),
    # MultiLineStrings
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiLineString(city_hall_walls),
            MultiLineString(city_hall_walls[1:])
        ]
    ),
    # Polygons
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[city_hall_boundaries, vauquelin_place]
    ),
    # MultiPolygon and Polygon
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiPolygon((city_hall_boundaries, vauquelin_place)),
            city_hall_boundaries
        ]
    ),
    # MultiPolygon
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place))]
    ),
    # all shape types
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiPolygon((city_hall_boundaries, vauquelin_place)),
            city_hall_entrance,
            MultiLineString(city_hall_walls),
            city_hall_walls[0],
            MultiPoint([city_hall_entrance, city_hall_balcony]),
            city_hall_balcony
        ]
    ),
    # all 2D shape types and 3D Point
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[
            MultiPolygon((city_hall_boundaries, vauquelin_place)),
            city_hall_entrance,
            MultiLineString(city_hall_walls),
            city_hall_walls[0],
            MultiPoint([city_hall_entrance, city_hall_balcony]),
            city_hall_balcony,
            point_3D
        ]
    ),
    # Null geometry and Point
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[None, city_hall_entrance]
    ),
    # Null geometry and 3D Point
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[None, point_3D]
    ),
    # Null geometries only
    GeoDataFrame(
        {},
        crs={'init': 'epsg:4326'},
        geometry=[None, None]
    )

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

        assert_geodataframe_equal(geodataframe, reloaded)
