import numpy as np

from geopandas import read_file, datasets
from geopandas.sindex import VALID_QUERY_PREDICATES


def generate_test_df():
    world = read_file(datasets.get_path("naturalearth_lowres"))
    capitals = read_file(datasets.get_path("naturalearth_cities"))
    countries = world.to_crs("epsg:3395")[["geometry"]]
    capitals = capitals.to_crs("epsg:3395")[["geometry"]]
    mixed = capitals.append(countries)  # get a mix of geometries
    points = capitals
    polygons = countries
    # filter out invalid geometries
    data = {
        "mixed": mixed[mixed.is_valid],
        "points": points[points.is_valid],
        "polygons": polygons[polygons.is_valid],
    }
    return data


class Bench:

    param_names = ["input_geom_type", "tree_geom_type"]
    params = [
        ["mixed", "points", "polygons"],
        ["mixed", "points", "polygons"],
    ]

    def setup(self, *args):
        self.data = generate_test_df()
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = {
            data_type: [g.bounds for g in self.data[data_type].geometry]
            for data_type in self.data.keys()
        }
        # ensure index is pre-generated
        for data_type in self.data.keys():
            self.data[data_type].sindex.query(
                self.data[data_type].geometry.values.data[0]
            )
        np.random.seed(0)  # set numpy random seed for reproducible results

    def time_intersects(self, tree_geom_type, input_geom_type):
        for bounds in self.bounds[input_geom_type]:
            self.data[tree_geom_type].sindex.intersection(bounds)


class BenchIndexCreation:

    param_names = ["tree_geom_type"]
    params = [["mixed", "points", "polygons"]]

    def setup(self, *args):
        self.data = generate_test_df()
        # ensure index is pre-generated
        for data_type in self.data.keys():
            self.data[data_type].sindex.query(
                self.data[data_type].geometry.values.data[0]
            )
        np.random.seed(0)  # set numpy random seed for reproducible results

    def time_index_creation(self, tree_geom_type):
        """Time creation of spatial index.
        """
        self.data[tree_geom_type].geometry.values._sindex = None
        self.data[tree_geom_type].sindex


class BenchQuery:

    param_names = ["predicate", "input_geom_type", "tree_geom_type"]
    params = [
        [*VALID_QUERY_PREDICATES],
        ["mixed", "points", "polygons"],
        ["mixed", "points", "polygons"],
    ]

    def setup(self, *args):
        self.data = generate_test_df()
        # ensure index is pre-generated
        for data_type in self.data.keys():
            self.data[data_type].sindex.query(
                self.data[data_type].geometry.values.data[0]
            )
        np.random.seed(0)  # set numpy random seed for reproducible results

    def time_query_bulk(self, predicate, input_geom_type, tree_geom_type):
        self.data[tree_geom_type].sindex.query_bulk(
            self.data[input_geom_type].geometry.values.data, predicate=predicate
        )

    def time_query(self, predicate, input_geom_type, tree_geom_type):
        self.data[tree_geom_type].sindex.query(
            np.random.choice(self.data[input_geom_type].geometry.values.data),
            predicate=predicate,
        )
