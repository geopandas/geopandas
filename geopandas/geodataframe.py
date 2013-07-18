from collections import defaultdict
import json

import fiona
from pandas import DataFrame
from shapely.geometry import mapping, shape

from geopandas import GeoSeries
from plotting import plot_dataframe


class GeoDataFrame(DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    named 'geometry' which is a GeoSeries.
    """

    def __init__(self, *args, **kwargs):
        crs = kwargs.pop('crs', None)
        super(GeoDataFrame, self).__init__(*args, **kwargs)
        self.crs = None

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
        The geometry column is not stored as a GeoSeries, so need to make sure
        that it is returned as one
        """
        col = super(GeoDataFrame, self).__getitem__(key)
        if key == 'geometry':
            col.__class__ = GeoSeries
            col.crs = self.crs
        return col

    def plot(self, *args, **kwargs):
        return plot_dataframe(self, *args, **kwargs)
