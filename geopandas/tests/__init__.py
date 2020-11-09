from geopandas import sindex


def has_sindex_backend():
    """Dynamically checks for ability to generate spatial index."""
    try:
        sindex._get_sindex_class()
        return True
    except ImportError:
        return False


# TODO: get rid of this once we have a permanent sindex backend in Shapely 2.0
sindex.has_sindex_backend = has_sindex_backend
