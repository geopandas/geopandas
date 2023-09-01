import numpy as np


def _hilbert_distance(geoms, total_bounds=None, level=16):
    """
    Calculate the distance along a Hilbert curve.

    The distances are calculated for the midpoints of the geometries in the
    GeoDataFrame.

    Parameters
    ----------
    geoms : GeometryArray
    total_bounds : 4-element array
        Total bounds of geometries - array
    level : int (1 - 16), default 16
        Determines the precision of the curve (points on the curve will
        have coordinates in the range [0, 2^level - 1]).

    Returns
    -------
    np.ndarray
        Array containing distances along the Hilbert curve

    """
    if geoms.is_empty.any() | geoms.isna().any():
        raise ValueError(
            "Hilbert distance cannot be computed on a GeoSeries with empty or "
            "missing geometries.",
        )
    # Calculate bounds as numpy array
    bounds = geoms.bounds

    # Calculate discrete coords based on total bounds and bounds
    x, y = _continuous_to_discrete_coords(bounds, level, total_bounds)
    # Compute distance along hilbert curve
    distances = _encode(level, x, y)

    return distances


def _continuous_to_discrete_coords(bounds, level, total_bounds):
    """
    Calculates mid points & ranges of geoms and returns
    as discrete coords

    Parameters
    ----------

    bounds : Bounds of each geometry - array

    p : The number of iterations used in constructing the Hilbert curve

    total_bounds : Total bounds of geometries - array

    Returns
    -------
    Discrete two-dimensional numpy array
    Two-dimensional array Array of hilbert distances for each geom

    """
    # Hilbert Side length
    side_length = (2**level) - 1

    # Calculate mid points for x and y bound coords - returns array
    x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
    y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0

    # Calculate x and y range of total bound coords - returns array
    if total_bounds is None:
        total_bounds = (
            np.nanmin(x_mids),
            np.nanmin(y_mids),
            np.nanmax(x_mids),
            np.nanmax(y_mids),
        )

    xmin, ymin, xmax, ymax = total_bounds

    # Transform continuous value to discrete integer for each dimension
    x_int = _continuous_to_discrete(x_mids, (xmin, xmax), side_length)
    y_int = _continuous_to_discrete(y_mids, (ymin, ymax), side_length)

    return x_int, y_int


def _continuous_to_discrete(vals, val_range, n):
    """
    Convert a continuous one-dimensional array to discrete integer values
    based their ranges

    Parameters
    ----------
    vals : Array of continuous values

    val_range : Tuple containing range of continuous values

    n : Number of discrete values

    Returns
    -------
    One-dimensional array of discrete ints

    """
    width = val_range[1] - val_range[0]
    if width == 0:
        return np.zeros_like(vals, dtype=np.uint32)
    res = (vals - val_range[0]) * (n / width)

    np.clip(res, 0, n, out=res)
    return res.astype(np.uint32)


# Fast Hilbert curve algorithm by http://threadlocalmutex.com/
# From C++ https://github.com/rawrunprotected/hilbert_curves
# (public domain)


MAX_LEVEL = 16


def _interleave(x):
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


def _encode(level, x, y):
    x = np.asarray(x, dtype="uint32")
    y = np.asarray(y, dtype="uint32")

    if level > MAX_LEVEL:
        raise ValueError("Level out of range")

    x = x << (16 - level)
    y = y << (16 - level)

    # Initial prefix scan round, prime with x and y
    a = x ^ y
    b = 0xFFFF ^ a
    c = 0xFFFF ^ (x | y)
    d = x & (y ^ 0xFFFF)

    A = a | (b >> 1)
    B = (a >> 1) ^ a
    C = ((c >> 1) ^ (b & (d >> 1))) ^ c
    D = ((a & (c >> 1)) ^ (d >> 1)) ^ d

    a = A.copy()
    b = B.copy()
    c = C.copy()
    d = D.copy()

    A = (a & (a >> 2)) ^ (b & (b >> 2))
    B = (a & (b >> 2)) ^ (b & ((a ^ b) >> 2))
    C ^= (a & (c >> 2)) ^ (b & (d >> 2))
    D ^= (b & (c >> 2)) ^ ((a ^ b) & (d >> 2))

    a = A.copy()
    b = B.copy()
    c = C.copy()
    d = D.copy()

    A = (a & (a >> 4)) ^ (b & (b >> 4))
    B = (a & (b >> 4)) ^ (b & ((a ^ b) >> 4))
    C ^= (a & (c >> 4)) ^ (b & (d >> 4))
    D ^= (b & (c >> 4)) ^ ((a ^ b) & (d >> 4))

    # Final round and projection
    a = A.copy()
    b = B.copy()
    c = C.copy()
    d = D.copy()

    C ^= (a & (c >> 8)) ^ (b & (d >> 8))
    D ^= (b & (c >> 8)) ^ ((a ^ b) & (d >> 8))

    # Undo transformation prefix scan
    a = C ^ (C >> 1)
    b = D ^ (D >> 1)

    # Recover index bits
    i0 = x ^ y
    i1 = b | (0xFFFF ^ (i0 | a))

    return ((_interleave(i1) << 1) | _interleave(i0)) >> (32 - 2 * level)
