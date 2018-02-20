import pyproj
import re
import os
import fiona.crs


def explicit_crs_from_epsg(crs=None, epsg=None):
    """
    Gets full/explicit CRS from EPSG code provided.

    Parameters
    ----------
     crs : dict or string, default None
         An existing crs dict or string with the 'init' key specifying an EPSG code
     epsg : string or int, default None
        The EPSG code to lookup
    """

    if epsg is None:
        if crs is not None:
            if isinstance(crs, str):
                crs = fiona.crs.from_string(crs)
            if 'init' in crs and crs['init'].lower().startswith('epsg:'):
                epsg = crs['init'].split(':')[1]

    if epsg is None:
        raise ValueError('No epsg code provided.')

    with open(os.path.join(pyproj.pyproj_datadir, 'epsg')) as f:
        data = f.read()

    _crs = re.search('\n<{}>\s*(.+?)\s*<>'.format(epsg), data)
    if _crs is None:
        raise ValueError('EPSG code "{}" not found.'.format(epsg))
    _crs = fiona.crs.from_string(_crs.group(1))
    # preserve the epsg code for future reference
    _crs['init'] = 'epsg:{}'.format(epsg)
    return _crs
