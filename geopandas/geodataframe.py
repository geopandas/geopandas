from collections import defaultdict, OrderedDict
import json
import os

import fiona
import numpy as np
from pandas import DataFrame
from shapely.geometry import mapping, shape

from geopandas import GeoSeries
from geopandas.plotting import plot_dataframe


class GeoDataFrame(DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    named 'geometry' which is a GeoSeries.
    """

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop('crs', None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)
        self.crs = crs

    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Alternate constructor to create a GeoDataFrame from a file.

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

    def to_json(self, **kwargs):
        """Returns a GeoJSON representation of the GeoDataFrame.
        
        The *kwargs* are passed to json.dumps().
        """
        def feature(i, row):
            return {
                'id': str(i),
                'type': 'Feature',
                'properties': {
                    k: v for k, v in row.iteritems() if k != 'geometry'},
                'geometry': mapping(row['geometry']) }

        return json.dumps(
            {'type': 'FeatureCollection',
             'features': [feature(i, row) for i, row in self.iterrows()]},
            **kwargs )
            
    def to_file(self, filename, driver="ESRI Shapefile", **kwargs):
        """
        Write this GeoDataFrame to an OGR data source
        
        A dictionary of supported OGR providers is available via:
        >>> import fiona
        >>> fiona.supported_drivers

        Parameters
        ----------
        filename : string 
            File path or file handle to write to.
        driver : string, default 'ESRI Shapefile'
            The OGR format driver used to write the vector file.

        The *kwargs* are passed to fiona.open and can be used to write 
        to multi-layer data, store data within archives (zip files), etc.
        """
        def convert_type(in_type):
            if in_type == object:
                return 'str'
            return type(np.asscalar(np.zeros(1, in_type))).__name__
            
        def feature(i, row):
            return {
                'id': str(i),
                'type': 'Feature',
                'properties': {
                    k: v for k, v in row.iteritems() if k != 'geometry'},
                'geometry': mapping(row['geometry']) }
        
        properties = OrderedDict([(col, convert_type(_type)) for col, _type 
            in zip(self.columns, self.dtypes) if col!='geometry'])
        # Need to check geom_types before we write to file... 
        # Some (most?) providers expect a single geometry type: 
        # Point, LineString, or Polygon
        geom_types = self['geometry'].geom_type.unique()
        from os.path import commonprefix # To find longest common prefix
        geom_type = commonprefix([g[::-1] for g in geom_types])[::-1]  # Reverse
        if geom_type == '': # No common suffix = mixed geometry types
            raise ValueError("Geometry column cannot contains mutiple "
                             "geometry types when writing to file.")
        schema = {'geometry': geom_type, 'properties': properties}
        filename = os.path.abspath(os.path.expanduser(filename))
        with fiona.open(filename, 'w', driver=driver, crs=self.crs, 
                        schema=schema, **kwargs) as c:
            for i, row in self.iterrows():
                c.write(feature(i, row))

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system

        This method will transform all points in all objects.  It has
        no notion or projecting entire geometries.  All segments
        joining points are assumed to be lines in the current
        projection, not geodesics.  Objects crossing the dateline (or
        other projection boundary) will have undesirable behavior.
        """
        if inplace:
            df = self
        else:
            df = self.copy()
            df.crs = self.crs
        geom = df.geometry.to_crs(crs=crs, epsg=epsg)
        df.geometry = geom
        if not inplace:
            return df

    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with a 'geometry' column, return a
        GeoDataFrame.
        """
        result = super(GeoDataFrame, self).__getitem__(key)
        if isinstance(key, basestring) and key == 'geometry':
            result.__class__ = GeoSeries
            result.crs = self.crs
        elif isinstance(result, DataFrame) and 'geometry' in result:
            result.__class__ = GeoDataFrame
            result.crs = self.crs
        return result

    def plot(self, *args, **kwargs):
        return plot_dataframe(self, *args, **kwargs)
