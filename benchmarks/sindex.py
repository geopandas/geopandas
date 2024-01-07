from shapely.geometry import Point
from geopandas.tests.util import _NATURALEARTH_LOWRES
from geopandas.tests.util import _NATURALEARTH_CITIES
from geopandas import read_file, GeoSeries


# Derive list of valid query predicates based on underlying index backend;
# we have to create a non-empty instance of the index to get these
index = GeoSeries([Point(0, 0)]).sindex
predicates = sorted(p for p in index.valid_query_predicates if p is not None)

geom_types = ("mixed", "points", "polygons")


def generate_test_df():
    world = read_file(_NATURALEARTH_LOWRES)
    capitals = read_file(_NATURALEARTH_CITIES)
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
        data[data_type].sindex.query(data[data_type].geometry.values[0])
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
        # Note: the GeoDataFram._sindex_generated attribute will
        # be removed by GH#1444 but is kept here (in the benchmarks
        # so that we can compare pre GH#1444 to post GH#1444 if needed
        self.data[tree_geom_type]._sindex_generated = None
        self.data[tree_geom_type].geometry.values._sindex = None
        tree = self.data[tree_geom_type].sindex
        # also do a single query to ensure the index is actually
        # generated and used
        tree.query(self.data[tree_geom_type].geometry.values[0])


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
        self.data[tree_geom_type].sindex.query(
            self.data[input_geom_type].geometry.values,
            predicate=predicate,
        )

    def time_query(self, predicate, input_geom_type, tree_geom_type):
        tree = self.data[tree_geom_type].sindex
        for geom in self.data[input_geom_type].geometry.values:
            tree.query(geom, predicate=predicate)
