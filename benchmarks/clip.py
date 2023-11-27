from geopandas import read_file, clip
from shapely.geometry import box
from geopandas.tests.util import _NATURALEARTH_LOWRES
from geopandas.tests.util import _NATURALEARTH_CITIES


class Bench:
    def setup(self, *args):
        world = read_file(_NATURALEARTH_LOWRES)
        capitals = read_file(_NATURALEARTH_CITIES)
        self.bounds = [box(*geom.bounds) for geom in world.geometry]
        self.points = capitals

    def time_clip(self):
        for bound in self.bounds:
            clip(self.points, bound)
