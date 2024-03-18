import geopandas


class Bench:
    def setup(self, *args):
        df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        self.df = df

    def time_constructor_from_manager(self):
        geopandas.GeoDataFrame(self.df._data)

    def time_constructor_from_geodataframe(self):
        geopandas.GeoDataFrame(self.df)

    def time_copy(self):
        self.df.copy()
