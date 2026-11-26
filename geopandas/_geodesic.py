import numpy as np

import shapely
from shapely import LineString, MultiLineString, MultiPoint, Point


def geodetic_distance_vectorized(geoms1, geoms2, crs):

    geom_types1 = set(np.unique(shapely.get_type_id(geoms1)))
    geom_types2 = set(np.unique(shapely.get_type_id(geoms2)))

    if geom_types1 == {1, 5} and geom_types2 == {0}:
        return distance_point_line_vectorized(geoms2, geoms1, crs.get_geod())
    elif geom_types1 == {0} and geom_types2 == {1, 5}:
        return distance_point_line_vectorized(geoms1, geoms2, crs.get_geod())

    return np.array([geodetic_distance(g1, g2, crs) for g1, g2 in zip(geoms1, geoms2)])


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
        case (Point(), LineString()) | (Point(), MultiLineString()):
            return distance_point_line(geom1, geom2, geod)
        case (LineString(), Point()) | (MultiLineString(), Point()):
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
    if isinstance(line, LineString):
        coords = shapely.get_coordinates(line)
        lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
        lon2, lat2 = coords[1:, 0], coords[1:, 1]
    else:
        # MultiLineString
        parts = shapely.get_parts(line)
        coords, idx = shapely.get_coordinates(parts, return_index=True)

        # False where one part ends and the next begins
        same_part = idx[:-1] == idx[1:]
        lon1, lat1 = coords[:-1, 0][same_part], coords[:-1, 1][same_part]
        lon2, lat2 = coords[1:, 0][same_part], coords[1:, 1][same_part]
    dist = distance_to_segments(point, lon1, lat1, lon2, lat2, geod)
    return dist


def distance_point_line_vectorized(points, lines, geod):
    # compute the distance to all line segments at once and take the minimum
    # TODO test with z/m/zm coords
    _, coords, offsets = shapely.to_ragged_array(
        lines, include_z=False, include_m=False
    )
    idx_lines, idx_geoms = offsets
    n_vertices = len(coords)

    # For taking the segment start/ends coords, we optimize this by using all
    # start/end pairs (including pairs that are not actually a segment but cross
    # lines). That means we calculate the distance metric for those invalid pairs
    # unnecessarily (and those distances will be set to inf later), but it is
    # generally more efficient than copying the coordinates for each segment.
    # (the lines vector can then just be sliced (for the chunking below), instead
    #  of having to 'take' the coords (i.e. fancy indexing) for only the valid
    #  segment pairs)
    n_pairs = n_vertices - 1
    is_segment = np.ones(n_pairs, dtype=bool)
    is_segment[idx_lines[1:-1] - 1] = False

    idx_pairs = idx_lines[idx_geoms]
    idx_pairs[-1] = n_pairs
    n_pairs_per_geom = np.diff(idx_pairs)

    if isinstance(points, Point):
        points_idx = np.zeros(n_pairs, dtype=int)
        points_vector = _lonlat_to_vector([points.x], [points.y])
        # TODO make this more efficient
        point_lon = np.broadcast_to(points.x, lines.shape)
        point_lat = np.broadcast_to(points.y, lines.shape)
    else:
        # indices into `points` to match each segment pair for vectorized computation
        points_idx = np.repeat(np.arange(len(points)), n_pairs_per_geom)

        point_lon = shapely.get_x(points)
        point_lat = shapely.get_y(points)
        points_vector = _lonlat_to_vector(point_lon, point_lat)
        # TODO empty lines would give negative repeats -> errors

    lon, lat = coords[:, 0], coords[:, 1]
    lines_vector = _lonlat_to_vector(lon, lat)

    # Calculate distance metric for each point/segment pair
    # (chunked to improve performance and reduce memory usage)
    chunksize = 16_000
    distance_metric = np.empty(n_pairs, dtype=np.float64)
    for i in range(0, n_pairs, chunksize):
        j = min(i + chunksize, n_pairs)

        # Now actually select the point and segment coords for this chunk
        pidx = points_idx[i:j]
        segment_starts = slice(i, j)
        segment_ends = slice(i + 1, j + 1)
        distance_metric[i:j] = _nearest_point_on_segment_metric(
            points_vector[pidx],
            lines_vector[segment_starts],
            lines_vector[segment_ends],
        )

    # Mark distance metric for non-segment pairs as inf so they are ignored for
    # the minimum distance step
    distance_metric[~is_segment] = np.inf

    # Find segment with minimum point/segment distance per geometry
    # (mimicking an "argmin reduceat" operation)
    mins = np.minimum.reduceat(distance_metric, idx_pairs[:-1])
    pos = np.arange(0, n_pairs)
    cand = np.where(distance_metric == np.repeat(mins, n_pairs_per_geom), pos, n_pairs)
    idx_min = np.minimum.reduceat(cand, idx_pairs[:-1])

    # Geometries where the minimum distance could not be found, e.g. because the
    # min distance was NaN and thus `distance_metric == min` was all False for that
    # set of segments, and `cand` gets set to `n_pairs` (an out-of-bounds index)
    # -> set them to a valid index now to avoid an error (actual distance gets
    # marked as invalid at the end)
    invalid = idx_min == n_pairs
    any_invalid = np.any(invalid)
    if any_invalid:
        idx_min[invalid] = 0

    # Given the selected segments with the nearest point, actually calculate
    # the nearest points for each input point / line geometry
    nearest_lon, nearest_lat = nearest_point_on_segment(
        point_lon,
        point_lat,
        lon[idx_min],
        lat[idx_min],
        lon[idx_min + 1],
        lat[idx_min + 1],
    )

    # Compute the actual geodetic distance between the original point and
    # the (spherical) nearest point
    _, _, dist = geod.inv(point_lon, point_lat, nearest_lon, nearest_lat)
    if any_invalid:
        dist[invalid] = np.nan
    return dist


def distance_to_segments(
    point: Point,
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    geod,
) -> float:
    point_lon = np.broadcast_to(point.x, lon1.shape)
    point_lat = np.broadcast_to(point.y, lon1.shape)
    # TODO this also works with passing scalar point lon/lat -> is that faster
    # if we don't need the broadcast for the distance calculation?

    point_vector = _lonlat_to_vector(point_lon, point_lat)
    segment_starts_vector = _lonlat_to_vector(lon1, lat1)
    segment_ends_vector = _lonlat_to_vector(lon2, lat2)

    # use haversine distance (approximated by squared chord distance) to more
    # cheaply find the closest segment
    metric = _nearest_point_on_segment_metric(
        point_vector, segment_starts_vector, segment_ends_vector
    )
    idx = np.argmin(metric)
    nearest_lon, nearest_lat = nearest_point_on_segment(
        point_lon, point_lat, lon1[[idx]], lat1[[idx]], lon2[[idx]], lat2[[idx]]
    )

    # then only compute the geodetic distance for the closest points
    _, _, dist = geod.inv(point_lon, point_lat, nearest_lon, nearest_lat)
    return dist[0]


class Vector3D:
    """
    Small helper class to more easily work with 3D vectors represented
    as 3 separate numpy arrays instead of a single (n, 3) ndarray.

    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        # allow easy unpacking like a tuple (x, y, z = vector)
        return iter((self.x, self.y, self.z))

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def is_nonzero(self) -> np.ndarray:
        return (self.x != 0) | (self.y != 0) | (self.z != 0)

    def __getitem__(self, key) -> np.ndarray:
        assert isinstance(key, (slice, np.ndarray))
        return Vector3D(self.x[key], self.y[key], self.z[key])

    def __setitem__(self, key, value: "Vector3D") -> None:
        assert isinstance(key, (slice, np.ndarray))
        assert isinstance(value, Vector3D)
        self.x[key] = value.x
        self.y[key] = value.y
        self.z[key] = value.z

    def create_empty(size: int) -> "Vector3D":
        return Vector3D(
            np.empty(size, dtype=np.float64),
            np.empty(size, dtype=np.float64),
            np.empty(size, dtype=np.float64),
        )

    def normalize(self) -> "Vector3D":
        norm = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector3D(self.x / norm, self.y / norm, self.z / norm)


def _lonlat_to_vector(lon: np.ndarray, lat: np.ndarray) -> Vector3D:
    """
    Convert longitude and latitude values to 3D unit vectors.

    Gets 1D lon/lat arrays with length N, and returns unit vectors as separate
    x, y, z arrays.

    https://www.movable-type.co.uk/scripts/latlong-vectors.html
    https://en.wikipedia.org/wiki/N-vector#Converting_latitude/longitude_to_n-vector
    """
    # (lonlat in degrees) * np.pi / 180
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    with np.errstate(invalid="ignore"):
        cos_lat = np.cos(lat_rad)
        x = cos_lat * np.cos(lon_rad)
        y = cos_lat * np.sin(lon_rad)
        z = np.sin(lat_rad)
    return Vector3D(x, y, z)


def _vector_to_lonlat(vector: Vector3D) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D unit vectors to longitude and latitude values.

    Input vector does not need to be normalized, since the arctan2 conversion
    is scale-invariant.

    See _lonlat_to_vector docstring.
    """
    x, y, z = vector
    lon_rad = np.arctan2(y, x)
    # np.hypot(x, y) == np.sqrt(x**2 + y**2)
    lat_rad = np.arctan2(z, np.hypot(x, y))
    # (lonlat in radians) * 180 / np.pi
    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)
    return lon, lat


def _cross(a: Vector3D, b: Vector3D) -> Vector3D:
    # optimized version of numpy's cross product, avoiding the generic setup (shape
    # and axis checking, broadcasting) and astype of the inputs (causing a copy)

    # create local aliases for readability
    a0, a1, a2 = a
    b0, b1, b2 = b

    # cp0 = a1 * b2 - a2 * b1
    # cp1 = a2 * b0 - a0 * b2
    # cp2 = a0 * b1 - a1 * b0
    cp0 = np.multiply(a1, b2)
    tmp = a2 * b1
    cp0 -= tmp
    cp1 = np.multiply(a2, b0)
    np.multiply(a0, b2, out=tmp)
    cp1 -= tmp
    cp2 = np.multiply(a0, b1)
    np.multiply(a1, b0, out=tmp)
    cp2 -= tmp

    return Vector3D(cp0, cp1, cp2)


def _dot(a: Vector3D, b: Vector3D) -> np.ndarray:
    # row-wise dot product
    # custom version for Vector3D with separate xyz arrays (with a 2D array,
    # this would be np.sum(a * b, axis=-1), or np.einsum("...i,...i -> ...", a, b)
    a0, a1, a2 = a
    b0, b1, b2 = b

    # a0 * b0 + a1 * b1 + a2 * b2
    out = np.multiply(a0, b0)
    tmp = np.multiply(a1, b1)
    out += tmp
    np.multiply(a2, b2, out=tmp)
    out += tmp

    return out


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
    c1 = _cross(v1, v2)

    # The normal of the great circle through the target point p and the
    # segment's normal c1
    c2 = _cross(v, c1)

    # The intersection of both great circles is the projection of p onto the
    # great circle of the p1 - p2 segment.
    # This point is determined by the cross product of their normals (perpendicular
    # to both normals, and thus lying on both great circles, thus the intersection),
    # and the order ensures we pick the intersection on p's side (cross(c2, c1)
    # would give the antipode)
    n = _cross(c1, c2)

    # The intersection point lives on the great circle defined by p1 and p2
    # (and gives the nearest point to p on that great circle), but does not
    # necessarily live on the segment (p1, p2) itself. Thus, we need to
    # check if n is between p1 and p2.

    # along-track direction at p1 (towards p2) and at p2 (towards p1)
    t1 = _cross(c1, v1)
    t2 = _cross(v2, c1)
    # a positive dot product indicates that p is on the same side as p2 of the
    # perpendicular plane through p1 (or as p1 of plane through p2) -> if both
    # are positive, the target point p is between p1 and p2
    on_segment = (_dot(n, t1) >= 0) & (_dot(n, t2) >= 0)

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
    on_segment &= n.is_nonzero()

    # If the nearest point is not on the segment, we want the closest endpoint
    # p1 or p2 instead -> use squared chord length (straight line through the
    # sphere, monotonic with actual distance but cheaper)
    d1, d2 = v - v1, v - v2
    dist1, dist2 = _dot(d1, d1), _dot(d2, d2)
    closest_to_start = dist1 <= dist2

    # If the point is exactly one of the start/end points of the segment, prefer
    # that end point instead of the projected-on-segment point (using this projected
    # point means a lonlat -> vector -> lonlat roundtrip, and because of floating
    # point rounding errors, this roundtripped lonlat is not exactly equal to the
    # original, and this can then give a tiny non-zero distance instead of
    # exactly zero)
    # (in such a case, v - v1 or v - v2 would be exactly zero -> dist1 or dist2
    #  would be 0 -> check for strictly positive distance)
    on_segment &= np.minimum(dist1, dist2) > 0

    # Get the actual nearest point coordinates by choosing the intersection
    # point or start point or end point accordingly (in a vectorized way)
    nearest_lon = np.where(closest_to_start, lon1, lon2)
    nearest_lat = np.where(closest_to_start, lat1, lat2)

    projected_lon, projected_lat = _vector_to_lonlat(n[on_segment])
    nearest_lon[on_segment] = projected_lon
    nearest_lat[on_segment] = projected_lat

    # mask invalid input data in result
    # TODO verify if this is needed (this might be caught higher up in the
    # distance calculation)
    invalid = ~np.isfinite(point_lon + point_lat + lon1 + lat1 + lon2 + lat2)
    nearest_lon = np.where(invalid, np.nan, nearest_lon)
    nearest_lat = np.where(invalid, np.nan, nearest_lat)

    return nearest_lon, nearest_lat


def _nearest_point_on_segment_metric(
    points: Vector3D, segment_starts: Vector3D, segment_ends: Vector3D
) -> np.ndarray:
    """
    Subset of nearest_point_on_segment that only computes the metric to determine
    which segment has the closest point, without computing the actual nearest point.

    Also assumes it already gets vectors in.

    See the more elaborately commented code in the function above for details.
    Here only commented about differences with the version above.
    """
    v = points
    v1 = segment_starts
    v2 = segment_ends

    c1 = _cross(v1, v2)
    c2 = _cross(v, c1)
    n = _cross(c1, c2)

    t1 = _cross(c1, v1)
    t2 = _cross(v2, c1)

    on_segment = (_dot(n, t1) >= 0) & (_dot(n, t2) >= 0)
    on_segment &= n.is_nonzero()

    d1, d2 = v - v1, v - v2
    dist1, dist2 = _dot(d1, d1), _dot(d2, d2)
    # NaN in input v/v1/v2 always propagate to the metric
    metric = np.minimum(dist1, dist2)

    projected = n[on_segment].normalize()
    d = v[on_segment] - projected
    metric[on_segment] = _dot(d, d)

    return metric
