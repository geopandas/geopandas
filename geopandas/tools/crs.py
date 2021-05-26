import warnings

from pyproj import CRS


def explicit_crs_from_epsg(crs=None, epsg=None):
    """
    Gets full/explicit CRS from EPSG code provided.

    Parameters
    ----------
    crs : dict or string, default None
        An existing crs dict or Proj string with the 'init' key specifying an EPSG code
    epsg : string or int, default None
       The EPSG code to lookup
    """
    warnings.warn(
        "explicit_crs_from_epsg is deprecated. "
        "You can set the epsg on the GeoDataFrame (gdf) using gdf.crs=epsg",
        FutureWarning,
        stacklevel=2,
    )
    if crs is not None:
        return CRS.from_user_input(crs)
    elif epsg is not None:
        return CRS.from_epsg(epsg)
    raise ValueError("Must pass either crs or epsg.")


def epsg_from_crs(crs):
    """
    Returns an epsg code from a crs dict or Proj string.

    Parameters
    ----------
    crs : dict or string, default None
        A crs dict or Proj string

    """
    warnings.warn(
        "epsg_from_crs is deprecated. "
        "You can get the epsg code from GeoDataFrame (gdf) "
        "using gdf.crs.to_epsg()",
        FutureWarning,
        stacklevel=2,
    )
    crs = CRS.from_user_input(crs)
    if "init=epsg" in crs.to_string().lower():
        epsg_code = crs.to_epsg(0)
    else:
        epsg_code = crs.to_epsg()
    return epsg_code


def get_epsg_file_contents():
    warnings.warn("get_epsg_file_contents is deprecated.", FutureWarning, stacklevel=2)
    return ""
