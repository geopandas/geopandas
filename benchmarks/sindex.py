from geopandas import read_file, datasets
from geopandas.sindex import VALID_QUERY_PREDICATES


class Bench:
    def setup(self, *args):
        world = read_file(datasets.get_path("naturalearth_lowres"))
        capitals = read_file(datasets.get_path("naturalearth_cities"))
        countries = world.to_crs("epsg:3395")[["geometry"]]
        capitals = capitals.to_crs("epsg:3395")[["geometry"]]

        # save dfs
        self.df, self.df2, self.data = (
            capitals,
            countries,
            countries.geometry.values.data,
        )
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = [c.bounds for c in capitals.geometry] + [
            c.bounds for c in countries.geometry
        ]

    def time_sindex_index_creation(self, *args):
        """Time creation of spatial index.

        Note: pygeos will only create the index once; this benchmark
        is not intended to be used to compare rtree and pygeos.
        """
        self.df._invalidate_sindex()
        self.df._generate_sindex()

    def time_sindex_intersects(self, *args):
        for bounds in self.bounds:
            self.df.sindex.intersection(bounds)

    def time_sindex_intersects_objects(self, *args):
        for bounds in self.bounds:
            self.df.sindex.intersection(bounds, objects=True)


class BenchQuery:

    param_names = ["predicate"]
    params = [*VALID_QUERY_PREDICATES]

    def setup(self, *args):
        world = read_file(datasets.get_path("naturalearth_lowres"))
        capitals = read_file(datasets.get_path("naturalearth_cities"))
        countries = world.to_crs("epsg:3395")
        countries = countries[["geometry"]]
        capitals = capitals.to_crs("epsg:3395")
        capitals = capitals[["geometry"]]

        # save dfs
        self.df, self.df2, self.data = (
            countries,
            capitals,
            capitals.geometry.values.data,
        )
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = [c.bounds for c in capitals.geometry] + [
            c.bounds for c in countries.geometry
        ]

    def time_query_bulk_data(self, predicate):
        self.df.sindex.query_bulk(self.data, predicate=predicate)

    def time_query_bulk(self, predicate):
        self.df.sindex.query_bulk(self.df2.geometry, predicate=predicate)

    def time_query(self, predicate):
        for geo in self.data:
            self.df.sindex.query(geo, predicate=predicate)
