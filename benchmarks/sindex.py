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

    param_names = ["tree_geom_type"]
    params = [["mixed", "points", "polygons"]]

    def setup(self, *args):
        self.data = generate_test_df()
        # cache bounds so that bound creation is not counted in benchmarks
        self.bounds = [g.bounds for g in self.data["mixed"].geometry]

    def time_index_creation(self, tree_geom_type):
        """Time creation of spatial index.

        Note: pygeos will only create the index once; this benchmark
        is not intended to be used to compare rtree and pygeos.
        """
        self.data[tree_geom_type]._invalidate_sindex()
        self.data[tree_geom_type]._generate_sindex()

    def time_intersects(self, tree_geom_type):
        for bounds in self.bounds:
            self.data[tree_geom_type].sindex.intersection(bounds)

    def time_intersects_objects(self, tree_geom_type):
        for bounds in self.bounds:
            self.data[tree_geom_type].sindex.intersection(bounds, objects=True)


class BenchQuery:

    param_names = ["predicate", "input_geom_type", "tree_geom_type"]
    params = [
        [*VALID_QUERY_PREDICATES],
        ["mixed", "points", "polygons"],
        ["mixed", "points", "polygons"],
    ]

    def setup(self, *args):
        self.data = generate_test_df()

    def time_query_bulk(self, predicate, input_geom_type, tree_geom_type):
        self.data[tree_geom_type].sindex.query_bulk(
            self.data[input_geom_type].geometry, predicate=predicate
        )

    def time_query(self, predicate, input_geom_type, tree_geom_type):
        for geo in self.data[input_geom_type].geometry.sample(10, random_state=0):
            self.data[tree_geom_type].sindex.query(geo, predicate=predicate)
