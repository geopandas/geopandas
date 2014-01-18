from geopandas import GeoDataFrame

def read_file(filename, **kwargs):
    """
    Returns a GeoDataFrame from a file.

    *filename* is either the absolute or relative path to the file to be
    opened and *kwargs* are keyword args to be passed to the method when
    opening the file.
    """
    import fiona
    bbox = kwargs.pop('bbox', None)
    with fiona.open(filename, **kwargs) as f:
        crs = f.crs
        if bbox != None:
            assert len(bbox)==4
            f_filt = f.filter(bbox=bbox)
        else:
            f_filt = f
        gdf = GeoDataFrame.from_features(f)

    return gdf
