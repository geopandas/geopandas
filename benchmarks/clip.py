from geopandas import read_file, datasets, clip
from shapely.geometry import box


class Bench:
    def setup(self, *args):
        world = read_file(datasets.get_path("naturalearth_lowres"))
        capitals = read_file(datasets.get_path("naturalearth_cities"))
        self.bounds = [box(*geom.bounds) for geom in world.geometry]
        self.points = capitals

    def time_clip(self):
        for bound in self.bounds:
            clip(self.points, bound)
