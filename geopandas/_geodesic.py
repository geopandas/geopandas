import numpy as np

import shapely
from shapely import LineString, MultiPoint, Point


def geodetic_distance(geom1, geom2, crs):
    geod = crs.get_geod()

    match geom1, geom2:
        case Point(), Point():
            return distance_point_point(geom1, geom2, geod)
        case MultiPoint(), MultiPoint() | MultiPoint(), Point() | Point(), MultiPoint():
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
    # iterate over line segments and find the minimum distance
    coords = shapely.get_coordinates(line)
    min_dist = float("inf")
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dist = distance_to_segment(point, lon1, lat1, lon2, lat2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def haversine_distance(lon1, lat1, lon2, lat2):
    # Haversine formula
    dLat = _degrees_to_radians(lat2 - lat1)
    dLon = _degrees_to_radians(lon2 - lon1)
    lat1 = _degrees_to_radians(lat1)
    lat2 = _degrees_to_radians(lat2)

    a = np.sin(dLat / 2) ** 2 + np.sin(dLon / 2) ** 2 * np.cos(lat1) * np.cos(lat2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371000  # Earth radius in meters
    dist = R * c
    return dist


def distance_to_segment(point, lon1, lat1, lon2, lat2):
    # spherical distance!
    # Haversine formula
    nearest_lon, nearest_lat = nearest_point_on_segment(point, lon1, lat1, lon2, lat2)
    return haversine_distance(point.x, point.y, nearest_lon, nearest_lat)


def _degrees_to_radians(degrees):
    # % 360 degrees in case someone passes value > 360
    normalised_degrees = degrees % 360
    return (normalised_degrees * np.pi) / 180


def _lon_lat_to_vector(lon, lat):
    lon_rad = _degrees_to_radians(lon)
    lat_rad = _degrees_to_radians(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    # TODO generalize for multiple points at once
    return np.array([x, y, z])


def _vector_to_lon_lat(vector):
    x, y, z = vector
    lat_rad = np.arcsin(z)
    lon_rad = np.arctan2(y, x)
    lat = (lat_rad * 180) / np.pi
    lon = (lon_rad * 180) / np.pi
    return lon, lat


def nearest_point_on_segment(point, lon1, lat1, lon2, lat2):

    v = _lon_lat_to_vector(point.x, point.y)
    v1 = _lon_lat_to_vector(lon1, lat1)
    v2 = _lon_lat_to_vector(lon2, lat2)

    segment_axis = np.cross(v1, v2)
    target_axis = np.cross(segment_axis, v)
    intersection_axis1 = np.cross(segment_axis, target_axis)
    intersection_axis1 /= np.linalg.norm(intersection_axis1)
    intersection_axis2 = -intersection_axis1

    if np.dot(v, intersection_axis1) > np.dot(v, intersection_axis2):
        intersection_axis = intersection_axis1
    else:
        intersection_axis = intersection_axis2

    segment_axis_norm = segment_axis / np.linalg.norm(segment_axis)
    cmp1 = np.dot(np.cross(v1, intersection_axis), segment_axis_norm)
    cmp2 = np.dot(np.cross(intersection_axis, v2), segment_axis_norm)
    if cmp1 >= 0 and cmp2 >= 0:
        return _vector_to_lon_lat(intersection_axis)
    elif np.dot(v1, v) > np.dot(v2, v):
        return lon1, lat1
    else:
        return lon2, lat2
