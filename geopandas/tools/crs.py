import os
import re
import warnings
from distutils.version import LooseVersion

import pyproj
import fiona.crs


_PYPROJ22 = LooseVersion(pyproj.__version__) >= LooseVersion('2.2.0')

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
    if _PYPROJ22:
        warnings.warn(
            "explicit_crs_from_epsg is deprecated. "
            "You can set the epsg on the GeoDataFrame (gdf) using gdf.crs=epsg",
            DeprecationWarning,
        )
    if epsg is None and crs is not None:
        epsg = epsg_from_crs(crs)
    if epsg is None:
        raise ValueError('No epsg code provided or epsg code could not be identified from the provided crs.')

    _crs = re.search(r'\n<{}>\s*(.+?)\s*<>'.format(epsg), get_epsg_file_contents())
    if _crs is None:
        raise ValueError('EPSG code "{}" not found.'.format(epsg))
    _crs = fiona.crs.from_string(_crs.group(1))
    # preserve the epsg code for future reference
    _crs['init'] = 'epsg:{}'.format(epsg)
    return _crs


def epsg_from_crs(crs):
    """
    Returns an epsg code from a crs dict or Proj string.

    Parameters
    ----------
    crs : dict or string, default None
        A crs dict or Proj string

    """
    if _PYPROJ22:
        warnings.warn(
            "epsg_from_crs is deprecated. "
            "You can get the epsg code from GeoDataFrame (gdf) "
            "using gdf.crs.to_epsg()",
            DeprecationWarning,
        )
    if crs is None:
        raise ValueError('No crs provided.')
    if isinstance(crs, str):
        crs = fiona.crs.from_string(crs)
    if not crs:
        raise ValueError('Empty or invalid crs provided')
    if 'init' in crs and crs['init'].lower().startswith('epsg:'):
        return int(crs['init'].split(':')[1])


def get_epsg_file_contents():
    with open(os.path.join(pyproj.pyproj_datadir, 'epsg')) as f:
        return f.read()
