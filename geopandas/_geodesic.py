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
    point_lon = np.broadcast_to(point.x, lon1.shape)
    point_lat = np.broadcast_to(point.y, lon1.shape)
    # TODO this also works with passing scalar point lon/lat -> is that faster
    # if we don't need the broadcast for the distance calculation?
    nearest_lon, nearest_lat = nearest_point_on_segment(
        point_lon, point_lat, lon1, lat1, lon2, lat2
    )
    # TODO check how much speed up it would give by using haversine distance
    # to determine the closest point, and only use geodetic distance for that point
    # return haversine_distance(point.x, point.y, nearest_lon, nearest_lat)
    _, _, dist = geod.inv(point_lon, point_lat, nearest_lon, nearest_lat)
    return dist


def _lonlat_to_vector(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude and latitude values to 3D unit vectors.

    Gets 1D lon/lat arrays with length N, and returns unit vectors with shape (N, 3).

    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    https://en.wikipedia.org/wiki/N-vector#Converting_latitude/longitude_to_n-vector
    """
    # (lonlat in degrees) * np.pi / 180
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


def _vector_to_lonlat(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D unit vectors to longitude and latitude values.

    Input vector does not need to be normalized, since the arctan2 conversion
    is scale-invariant.

    See _lonlat_to_vector docstring.
    """
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    lon_rad = np.arctan2(y, x)
    # np.hypot(x, y) == np.sqrt(x**2 + y**2)
    lat_rad = np.arctan2(z, np.hypot(x, y))
    # (lonlat in radians) * 180 / np.pi
    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)
    return lon, lat


def _dot(a, b):
    # row-wise dot product
    # slightly faster version of np.sum(a * b, axis=-1)
    return np.einsum("...i,...i -> ...", a, b)


def nearest_point_on_segment(
    point_lon: np.ndarray,
    point_lat: np.ndarray,
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherical approximation to get the nearest point on a segment.

    The segment is the shortest great-circle path defined by p1 and p2, and
    we look for the nearest point on this segment to target point p.

    Vectorized over the points and segments. This assumes that all the input arrays
    are already broadcasted correctly and have the same shape (all 1d arrays).

    Based on:
    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    https://github.com/Turfjs/turf/blob/0c20329dfc2b63555ddd5d5388c4858669df6037/packages/turf-nearest-point-on-line/index.ts#L233-L320
    """
    v = _lonlat_to_vector(point_lon, point_lat)
    v1 = _lonlat_to_vector(lon1, lat1)
    v2 = _lonlat_to_vector(lon2, lat2)

    # The normal of the great circle through p1 and p2 (containing the
    # line segment), i.e. the vector perpendicular to the plane formed by the
    # great circle
    c1 = np.cross(v1, v2)

    # The normal of the great circle through the target point p and the
    # segment's normal c1
    c2 = np.cross(v, c1)

    # The intersection of both great circles is the projection of p onto the
    # great circle of the p1 - p2 segment.
    # This point is determined by the cross product of their normals (perpendicular
    # to both normals, and thus lying on both great circles, thus the intersection),
    # and the order ensures we pick the intersection on p's side (cross(c2, c1)
    # would give the antipode)
    n = np.cross(c1, c2)
    projected_lon, projected_lat = _vector_to_lonlat(n)

    # The intersection point lives on the great circle defined by p1 and p2
    # (and gives the nearest point to p on that great circle), but does not
    # necessarily live on the segment (p1, p2) itself. Thus, we need to
    # check if n is between p1 and p2.

    # along-track direction at p1 (towards p2) and at p2 (towards p1)
    t1 = np.cross(c1, v1)
    t2 = np.cross(v2, c1)
    # a positive dot product indicates that p is on the same side as p2 of the
    # perpendicular plane through p1 (or as p1 of plane through p2) -> if both
    # are positive, the target point p is between p1 and p2
    on_segment = (_dot(v, t1) >= 0) & (_dot(v, t2) >= 0)

    # Some degenerate cases where the on-segment closest point gives the wrong
    # result:
    # - zero-length segments: v1 == v2, c1 becomes the zero vector. Later on,
    #   the code picks the start point for this case
    # - the point p is exactly at a pole of the segment's great circle: any
    #   point on the segment would be equally valid. n becomes the zero vector,
    #   so this also falls back to always the start point of the segment.
    # - antipodal end points (for which the great circle is undefined): v1 and
    #   v2 are opposite, c1 also becomes the zero vector
    # - TODO intersection point at the poles (where the cross product might be zero)

    # n is the zero vector if the segment is degenerate (p1 == p2) or the point
    # sits exactly at a pole of the segment's great circle: fall back to endpoints

    on_segment &= np.any(n != 0, axis=-1)

    # If the nearest point is not on the segment, we want the closest endpoint
    # p1 or p2 instead -> use squared chord length (straight line through the
    # sphere, monotonic with actual distance but cheaper)
    d1, d2 = v - v1, v - v2
    closest_to_start = _dot(d1, d1) <= _dot(d2, d2)

    # Get the actual nearest point coordinates by choosing the intersection
    # point or start point or end point accordingly (in a vectorized way)
    endpoint_lon = np.where(closest_to_start, lon1, lon2)
    endpoint_lat = np.where(closest_to_start, lat1, lat2)

    nearest_lon = np.where(on_segment, projected_lon, endpoint_lon)
    nearest_lat = np.where(on_segment, projected_lat, endpoint_lat)

    # mask invalid input data in result
    # TODO verify if this is needed (this might be caught higher up in the
    # distance calculation)
    invalid = ~np.isfinite(point_lon + point_lat + lon1 + lat1 + lon2 + lat2)
    nearest_lon = np.where(invalid, np.nan, nearest_lon)
    nearest_lat = np.where(invalid, np.nan, nearest_lat)

    return nearest_lon, nearest_lat
