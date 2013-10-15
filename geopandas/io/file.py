from collections import defaultdict

from shapely.geometry import shape
import fiona

from geopandas import GeoSeries, GeoDataFrame

def read_file(filename, **kwargs):
    """
    Returns a GeoDataFrame from a file.

    *filename* is either the absolute or relative path to the file to be
    opened and *kwargs* are keyword args to be passed to the method when
    opening the file.

    Note: This method does not attempt to align rows.
    Properties that are not present in all features of the source
    file will not be properly aligned.  This should be fixed.
    """
    geoms = []
    columns = defaultdict(lambda: [])
    bbox = kwargs.pop('bbox', None)
    with fiona.open(filename, **kwargs) as f:
        crs = f.crs
        if bbox != None:
            assert len(bbox)==4
            f_filt = f.filter(bbox=bbox)
        else:
            f_filt = f
        for rec in f_filt:
            geoms.append(shape(rec['geometry']))
            for key, value in rec['properties'].iteritems():
                columns[key].append(value)
    geom = GeoSeries(geoms)
    df = GeoDataFrame(columns)
    df['geometry'] = geom
    df.crs = crs
    return df
