from geopandas import GeoDataFrame, GeoSeries, read_file, datasets, overlay
from shapely.geometry import Polygon


class Countries:

    param_names = ['op']
    params = [('intersection', 'union', 'identity', 'symmetric_difference',
               'difference')]

    def setup(self, *args):
        world = read_file(datasets.get_path('naturalearth_lowres'))
        capitals = read_file(datasets.get_path('naturalearth_cities'))
        countries = world[['geometry', 'name']]
        countries = countries.to_crs('+init=epsg:3395')[
            countries.name != "Antarctica"]
        capitals = capitals.to_crs('+init=epsg:3395')
        capitals['geometry'] = capitals.buffer(500000)

        self.countries = countries
        self.capitals = capitals

    def time_overlay(self, op):
        overlay(self.countries, self.capitals, how=op)


class Small:

    param_names = ['op']
    params = [('intersection', 'union', 'identity', 'symmetric_difference',
               'difference')]

    def setup(self, *args):
        polys1 = GeoSeries([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])])
        polys2 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])

        df1 = GeoDataFrame({'geometry': polys1, 'df1': [1, 2]})
        df2 = GeoDataFrame({'geometry': polys2, 'df2': [1, 2]})

        self.df1, self.df2 = df1, df2

    def time_overlay(self, op):
        overlay(self.df1, self.df2, how=op)
