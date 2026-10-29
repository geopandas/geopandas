import numpy as np

import shapely
from shapely import LineString, MultiPoint, Point


def geodetic_distance(geom1, geom2, crs):
    geod = crs.get_geod()

    match geom1, geom2:
        case Point(), Point():
            return distance_point_point(geom1, geom2, geod)
        case (
            (MultiPoint(), MultiPoint())
            | (MultiPoint(), Point())
            | (Point(), MultiPoint())
        ):
            return distance_multipoint_multipoint(geom1, geom2, geod)
        case Point(), LineString():
            return distance_point_line(geom1, geom2, geod)
        case LineString(), Point():
            return distance_point_line(geom2, geom1, geod)
        case _:
            raise NotImplementedError(
                "Geodetic distance not implemented for these geometry types"
            )


def distance_point_point(point1, point2, geod):
    # TODO handle empty (return value?)
    _, _, dist = geod.inv(point1.x, point1.y, point2.x, point2.y)
    return dist


def distance_multipoint_multipoint(multipoint1, multipoint2, geod):
    coords1 = shapely.get_coordinates(multipoint1)
    lon1 = coords1[:, 0]
    lat1 = coords1[:, 1]
    N1 = len(lon1)

    coords2 = shapely.get_coordinates(multipoint2)
    lon2 = coords2[:, 0]
    lat2 = coords2[:, 1]
    N2 = len(lon2)

    lon1 = np.repeat(lon1, N2)
    lat1 = np.repeat(lat1, N2)
    lon2 = np.tile(lon2, N1)
    lat2 = np.tile(lat2, N1)

    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist.min()


def distance_point_line(point, line, geod):
    # compute the distance to all line segments at once and take the minimum
    coords = shapely.get_coordinates(line)
    lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
    lon2, lat2 = coords[1:, 0], coords[1:, 1]
    dist = distance_to_segment(point, lon1, lat1, lon2, lat2, geod)
    return dist.min(initial=np.inf)


# def haversine_distance(lon1, lat1, lon2, lat2):
#     # Haversine formula
#     dLat = _degrees_to_radians(lat2 - lat1)
#     dLon = _degrees_to_radians(lon2 - lon1)
#     lat1 = _degrees_to_radians(lat1)
#     lat2 = _degrees_to_radians(lat2)

#     a = np.sin(dLat / 2) ** 2 + np.sin(dLon / 2) ** 2 * np.cos(lat1) * np.cos(lat2)
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     R = 6371000  # Earth radius in meters
#     dist = R * c
#     return dist


def distance_to_segment(point, lon1, lat1, lon2, lat2, geod):
    # spherical distance!
    # Haversine formula
    nearest_lon, nearest_lat = nearest_point_on_segment(point, lon1, lat1, lon2, lat2)
    # return haversine_distance(point.x, point.y, nearest_lon, nearest_lat)
    _, _, dist = geod.inv(point.x, point.y, nearest_lon, nearest_lat)
    return dist


def _degrees_to_radians(degrees):
    # % 360 degrees in case someone passes value > 360
    normalised_degrees = degrees % 360
    return (normalised_degrees * np.pi) / 180


def _lon_lat_to_vector(lon, lat):
    # returns unit vectors with shape (..., 3)
    lon_rad = _degrees_to_radians(lon)
    lat_rad = _degrees_to_radians(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


def _vector_to_lon_lat(vector):
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    lat_rad = np.arcsin(np.clip(z, -1, 1))
    lon_rad = np.arctan2(y, x)
    lat = (lat_rad * 180) / np.pi
    lon = (lon_rad * 180) / np.pi
    return lon, lat


# tolerance on the norm of cross products of unit vectors (sine of the angle
# between them), ~6 micrometer on the Earth's surface
_EPS = 1e-12


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def nearest_point_on_segment(point, lon1, lat1, lon2, lat2):
    # vectorized over the segments: lon1, lat1, lon2, lat2 can be arrays
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(lon1, lat1, lon2, lat2)

    v = _lon_lat_to_vector(point.x, point.y)
    v1 = _lon_lat_to_vector(lon1, lat1)
    v2 = _lon_lat_to_vector(lon2, lat2)

    segment_axis = np.cross(v1, v2)
    segment_axis_len = np.linalg.norm(segment_axis, axis=-1)
    target_axis = np.cross(segment_axis, v)

    # zero-length segment (or antipodal end points, for which the great
    # circle is undefined): the segment is reduced to its start point
    degenerate = segment_axis_len < _EPS
    # point is at a pole of the segment's great circle: every point of the
    # segment is at the same distance, so take its start point
    at_pole = np.linalg.norm(target_axis, axis=-1) < _EPS * segment_axis_len

    # the computations below give nan for the degenerate cases (division by
    # zero), but those are overwritten with the start point at the end
    with np.errstate(invalid="ignore", divide="ignore"):
        intersection_axis = np.cross(segment_axis, target_axis)
        intersection_axis /= np.linalg.norm(intersection_axis, axis=-1)[..., None]
        # choose the intersection on the same side of the globe as the point
        intersection_axis *= np.where(_dot(v, intersection_axis) > 0, 1, -1)[..., None]

        segment_axis_norm = segment_axis / segment_axis_len[..., None]
        cmp1 = _dot(np.cross(v1, intersection_axis), segment_axis_norm)
        cmp2 = _dot(np.cross(intersection_axis, v2), segment_axis_norm)

    on_segment = (cmp1 >= 0) & (cmp2 >= 0)
    nearest_lon, nearest_lat = _vector_to_lon_lat(intersection_axis)

    closest_to_start = _dot(v1, v) > _dot(v2, v)
    endpoint_lon = np.where(closest_to_start, lon1, lon2)
    endpoint_lat = np.where(closest_to_start, lat1, lat2)

    nearest_lon = np.where(on_segment, nearest_lon, endpoint_lon)
    nearest_lat = np.where(on_segment, nearest_lat, endpoint_lat)

    use_start = degenerate | at_pole
    nearest_lon = np.where(use_start, lon1, nearest_lon)
    nearest_lat = np.where(use_start, lat1, nearest_lat)
    return nearest_lon, nearest_lat
