from collections import defaultdict
from pandas import DataFrame
from shapely.geometry import shape
import fiona
from geopandas import GeoSeries


class GeoDataFrame(DataFrame):
    """
    A GeoDataFrame object is a pandas.DataFrame that has a column
    named 'geometry' which is a GeoSeries.
    """

    @classmethod
    def from_file(cls, filename):
        """
        Alternate constructor to create a GeoDataFrame from a file

        Note: This method does not attempt to align rows.
        Properties that are not present in all features of the source
        file will not be properly aligned.  This should be fixed.
        """
        geoms = []
        columns = defaultdict(lambda: [])
        with fiona.open(filename) as f:
            for rec in f:
                geoms.append(shape(rec['geometry']))
                for key, value in rec['properties'].iteritems():
                    columns[key].append(value)
        geom = GeoSeries(geoms)
        df = GeoDataFrame(columns)
        df['geometry'] = geom
        return df

    def __getitem__(self, key):
        """
        The geometry column is not stored as a GeoSeries, so need to convert it back
        """
        col = super(GeoDataFrame, self).__getitem__(key)
        if key == 'geometry':
            return GeoSeries(col)
        else:
            return col

    def plot(self, colormap='Accent'):
        # TODO: pass in argument to color geometries
        return self['geometry'].plot(colormap)
