from enum import Enum
import os
import sys

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import geopandas
from geopandas import GeoDataFrame
from geopandas.io.file import _FIONA18

from geopandas.testing import assert_geodataframe_equal
import pytest

# Credit: Polygons below come from Montreal city Open Data portal
# http://donnees.ville.montreal.qc.ca/dataset/unites-evaluation-fonciere
city_hall_boundaries = Polygon(
    (
        (-73.5541107525234, 45.5091983609661),
        (-73.5546126200639, 45.5086813829106),
        (-73.5540185061397, 45.5084409343852),
        (-73.5539986525799, 45.5084323044531),
        (-73.5535801792994, 45.5089539203786),
        (-73.5541107525234, 45.5091983609661),
    )
)
vauquelin_place = Polygon(
    (
        (-73.5542465586147, 45.5081555487952),
        (-73.5540185061397, 45.5084409343852),
        (-73.5546126200639, 45.5086813829106),
        (-73.5548825850032, 45.5084033554357),
        (-73.5542465586147, 45.5081555487952),
    )
)

city_hall_walls = [
    LineString(
        (
            (-73.5541107525234, 45.5091983609661),
            (-73.5546126200639, 45.5086813829106),
            (-73.5540185061397, 45.5084409343852),
        )
    ),
    LineString(
        (
            (-73.5539986525799, 45.5084323044531),
            (-73.5535801792994, 45.5089539203786),
            (-73.5541107525234, 45.5091983609661),
        )
    ),
]

city_hall_entrance = Point(-73.553785, 45.508722)
city_hall_balcony = Point(-73.554138, 45.509080)
city_hall_council_chamber = Point(-73.554246, 45.508931)

point_3D = Point(-73.553785, 45.508722, 300)


# *****************************************
# TEST TOOLING


class _Fiona(Enum):
    below_1_8 = "fiona_below_1_8"
    above_1_8 = "fiona_above_1_8"


class _ExpectedError:
    def __init__(self, error_type, error_message_match):
        self.type = error_type
        self.match = error_message_match


class _ExpectedErrorBuilder:
    def __init__(self, composite_key):
        self.composite_key = composite_key

    def to_raise(self, error_type, error_match):
        _expected_exceptions[self.composite_key] = _ExpectedError(
            error_type, error_match
        )


def _expect_writing(gdf, ogr_driver, fiona_version):
    return _ExpectedErrorBuilder(_composite_key(gdf, ogr_driver, fiona_version))


def _composite_key(gdf, ogr_driver, fiona_version):
    return frozenset([id(gdf), ogr_driver, fiona_version.value])


def _expected_error_on(gdf, ogr_driver, is_fiona_above_1_8):
    if is_fiona_above_1_8:
        composite_key = _composite_key(gdf, ogr_driver, _Fiona.above_1_8)
    else:
        composite_key = _composite_key(gdf, ogr_driver, _Fiona.below_1_8)
    return _expected_exceptions.get(composite_key, None)


# *****************************************
# TEST CASES
_geodataframes_to_write = []
_expected_exceptions = {}
_CRS = "epsg:4326"

# ------------------
# gdf with Points
gdf = GeoDataFrame(
    {"a": [1, 2]}, crs=_CRS, geometry=[city_hall_entrance, city_hall_balcony]
)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with MultiPoints
gdf = GeoDataFrame(
    {"a": [1, 2]},
    crs=_CRS,
    geometry=[
        MultiPoint([city_hall_balcony, city_hall_council_chamber]),
        MultiPoint([city_hall_entrance, city_hall_balcony, city_hall_council_chamber]),
    ],
)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with Points and MultiPoints
gdf = GeoDataFrame(
    {"a": [1, 2]},
    crs=_CRS,
    geometry=[MultiPoint([city_hall_entrance, city_hall_balcony]), city_hall_balcony],
)
_geodataframes_to_write.append(gdf)
# 'ESRI Shapefile' driver supports writing LineString/MultiLinestring and
# Polygon/MultiPolygon but does not mention Point/MultiPoint
# see https://www.gdal.org/drv_shapefile.html
for driver in ("ESRI Shapefile", "GPKG"):
    _expect_writing(gdf, driver, _Fiona.below_1_8).to_raise(
        ValueError,
        "Record's geometry type does not match collection schema's geometry "
        "type: 'MultiPoint' != 'Point'",
    )
_expect_writing(gdf, "ESRI Shapefile", _Fiona.above_1_8).to_raise(
    RuntimeError, "Failed to write record"
)

# ------------------
# gdf with LineStrings
gdf = GeoDataFrame({"a": [1, 2]}, crs=_CRS, geometry=city_hall_walls)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with MultiLineStrings
gdf = GeoDataFrame(
    {"a": [1, 2]},
    crs=_CRS,
    geometry=[MultiLineString(city_hall_walls), MultiLineString(city_hall_walls)],
)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with LineStrings and MultiLineStrings
gdf = GeoDataFrame(
    {"a": [1, 2]},
    crs=_CRS,
    geometry=[MultiLineString(city_hall_walls), city_hall_walls[0]],
)
_geodataframes_to_write.append(gdf)
_expect_writing(gdf, "GPKG", _Fiona.below_1_8).to_raise(
    ValueError,
    "Record's geometry type does not match collection schema's geometry "
    "type: 'MultiLineString' != 'LineString'",
)

# ------------------
# gdf with Polygons
gdf = GeoDataFrame(
    {"a": [1, 2]}, crs=_CRS, geometry=[city_hall_boundaries, vauquelin_place]
)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with MultiPolygon
gdf = GeoDataFrame(
    {"a": [1]},
    crs=_CRS,
    geometry=[MultiPolygon((city_hall_boundaries, vauquelin_place))],
)
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with Polygon and MultiPolygon
gdf = GeoDataFrame(
    {"a": [1, 2]},
    crs=_CRS,
    geometry=[
        MultiPolygon((city_hall_boundaries, vauquelin_place)),
        city_hall_boundaries,
    ],
)
_geodataframes_to_write.append(gdf)
_expect_writing(gdf, "GPKG", _Fiona.below_1_8).to_raise(
    ValueError,
    "Record's geometry type does not match collection schema's geometry "
    "type: 'MultiPolygon' != 'Polygon'",
)

# ------------------
# gdf with null geometry and Point
gdf = GeoDataFrame({"a": [1, 2]}, crs=_CRS, geometry=[None, city_hall_entrance])
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with null geometry and 3D Point
gdf = GeoDataFrame({"a": [1, 2]}, crs=_CRS, geometry=[None, point_3D])
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with null geometries only
gdf = GeoDataFrame({"a": [1, 2]}, crs=_CRS, geometry=[None, None])
_geodataframes_to_write.append(gdf)

# ------------------
# gdf with all shape types mixed together
gdf = GeoDataFrame(
    {"a": [1, 2, 3, 4, 5, 6]},
    crs=_CRS,
    geometry=[
        MultiPolygon((city_hall_boundaries, vauquelin_place)),
        city_hall_entrance,
        MultiLineString(city_hall_walls),
        city_hall_walls[0],
        MultiPoint([city_hall_entrance, city_hall_balcony]),
        city_hall_balcony,
    ],
)
_geodataframes_to_write.append(gdf)
# Not supported by 'ESRI Shapefile' driver
for driver in ("ESRI Shapefile", "GPKG"):
    _expect_writing(gdf, driver, _Fiona.below_1_8).to_raise(
        AttributeError, "'list' object has no attribute 'lstrip'"
    )
_expect_writing(gdf, "ESRI Shapefile", _Fiona.above_1_8).to_raise(
    RuntimeError, "Failed to write record"
)

# ------------------
# gdf with all 2D shape types and 3D Point mixed together
gdf = GeoDataFrame(
    {"a": [1, 2, 3, 4, 5, 6, 7]},
    crs=_CRS,
    geometry=[
        MultiPolygon((city_hall_boundaries, vauquelin_place)),
        city_hall_entrance,
        MultiLineString(city_hall_walls),
        city_hall_walls[0],
        MultiPoint([city_hall_entrance, city_hall_balcony]),
        city_hall_balcony,
        point_3D,
    ],
)
_geodataframes_to_write.append(gdf)
# Not supported by 'ESRI Shapefile' driver
for driver in ("ESRI Shapefile", "GPKG"):
    _expect_writing(gdf, driver, _Fiona.below_1_8).to_raise(
        AttributeError, "'list' object has no attribute 'lstrip'"
    )
_expect_writing(gdf, "ESRI Shapefile", _Fiona.above_1_8).to_raise(
    RuntimeError, "Failed to write record"
)


@pytest.fixture(params=_geodataframes_to_write)
def geodataframe(request):
    return request.param


@pytest.fixture(params=["GeoJSON", "ESRI Shapefile", "GPKG"])
def ogr_driver(request):
    return request.param


def test_to_file_roundtrip(tmpdir, geodataframe, ogr_driver):
    output_file = os.path.join(str(tmpdir), "output_file")

    expected_error = _expected_error_on(geodataframe, ogr_driver, _FIONA18)
    if expected_error:
        with pytest.raises(expected_error.type, match=expected_error.match):
            geodataframe.to_file(output_file, driver=ogr_driver)
    else:
        geodataframe.to_file(output_file, driver=ogr_driver)

        reloaded = geopandas.read_file(output_file)

        check_column_type = "equiv"
        if sys.version_info[0] < 3:
            # do not check column types in python 2 (mixed string/unicode)
            check_column_type = False

        assert_geodataframe_equal(
            geodataframe, reloaded, check_column_type=check_column_type
        )
