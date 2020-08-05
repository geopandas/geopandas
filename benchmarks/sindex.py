import numpy as np
import random

from geopandas import read_file, datasets
from geopandas.sindex import VALID_QUERY_PREDICATES


# set random seeds for deterministic results
np.random.seed(0)
random.seed(0)

predicates = sorted(VALID_QUERY_PREDICATES, key=lambda x: (x is None, x))
geom_types = ("mixed", "points", "polygons")


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
    # ensure index is pre-generated
    for data_type in data.keys():
        data[data_type].sindex.query(data[data_type].geometry.values.data[0])
    return data


class BenchIntersection:

    param_names = ["input_geom_type", "tree_geom_type"]
    params = [
        geom_types,
        geom_types,
    ]

    def setup(self, *args):
        self.data = generate_test_df()
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = {
            data_type: [g.bounds for g in self.data[data_type].geometry]
            for data_type in self.data.keys()
        }

    def time_intersects(self, input_geom_type, tree_geom_type):
        tree = self.data[tree_geom_type].sindex
        for bounds in self.bounds[input_geom_type]:
            tree.intersection(bounds)


class BenchIndexCreation:

    param_names = ["tree_geom_type"]
    params = [
        geom_types,
    ]

    def setup(self, *args):
        self.data = generate_test_df()

    def time_index_creation(self, tree_geom_type):
        """Time creation of spatial index.

        Note: requires running a single query to ensure that
        lazy-building indexes are actually built.
        """
        self.data[tree_geom_type]._sindex_generated = None
        self.data[tree_geom_type].geometry.values._sindex = None
        tree = self.data[tree_geom_type].sindex
        # also do a single query to ensure the index is actually
        # generated and used
        tree.query(
            self.data[tree_geom_type].geometry.values.data[0]
        )


class BenchQuery:

    param_names = ["predicate", "input_geom_type", "tree_geom_type"]
    params = [
        predicates,
        geom_types,
        geom_types,
    ]

    def setup(self, *args):
        self.data = generate_test_df()

    def time_query_bulk(self, predicate, input_geom_type, tree_geom_type):
        self.data[tree_geom_type].sindex.query_bulk(
            self.data[input_geom_type].geometry.values.data,
            predicate=predicate,
        )

    def time_query(self, predicate, input_geom_type, tree_geom_type):
        tree = self.data[tree_geom_type].sindex
        for geom in self.data[input_geom_type].geometry.values.data:
            tree.query(
                geom,
                predicate=predicate
            )
