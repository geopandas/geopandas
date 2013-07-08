from collections import defaultdict
import json

import fiona
from pandas import DataFrame
from shapely.geometry import mapping, shape

from geopandas import GeoSeries


class GeoDataFrame(DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    named 'geometry' which is a GeoSeries.
    """

    def __init__(self, *args, **kwargs):
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
        
        with fiona.open(filename, **kwargs) as f:
            crs = f.crs
            for rec in f:
                geoms.append(shape(rec['geometry']))
                for key, value in rec['properties'].iteritems():
                    columns[key].append(value)
        geom = GeoSeries(geoms)
        df = GeoDataFrame(columns)
        df['geometry'] = geom
        df.crs = crs
        return df

    def to_json(self, **kwargs):
        """Returns a GeoJSON formatted representation of the GeoDataFrame.
        
        *kwargs* are passed to json.dumps().
        """
        def feature(i, row):
            return {
                'id': str(i),
                'type': 'Feature',
                'properties': {k: v for k, v in row.iteritems() if k != 'geometry'},
                'geometry': mapping(row['geometry']) }
        return json.dumps(
            {'type': 'FeatureCollection',
             'features': [feature(i, row) for i, row in self.iterrows()]},
            **kwargs )

    def __getitem__(self, key):
        """
        The geometry column is not stored as a GeoSeries, so need to convert it back
        """
        col = super(GeoDataFrame, self).__getitem__(key)
        if key == 'geometry':
            g = GeoSeries(col)
            # TODO: set crs in GeoSeries constructor rather than here
            g.crs = self.crs
            return g
        else:
            return col

    def plot(self, *args, **kwargs):
        # TODO: pass in argument to color geometries
        return self['geometry'].plot(*args, **kwargs)
