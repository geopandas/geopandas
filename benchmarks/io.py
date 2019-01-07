import os
import shutil
import tempfile

from geopandas import GeoDataFrame, GeoSeries
import numpy as np
from shapely.geometry import Point


class Bench:

    # extensions for different file types to test
    params = [".shp", ".json", ".gpkg"]
    param_names = ["ext"]

    def setup(self, ext):

        self.driver_dict = {".shp": "ESRI Shapefile",
                            ".json": "GeoJSON",
                            ".gpkg": "GPKG"}
        driver = self.driver_dict[ext]

        num_points = 20000
        xs = np.random.rand(num_points)
        ys = np.random.rand(num_points)

        self.points = GeoSeries([Point(x, y) for (x, y) in zip(xs, ys)])
        self.df = GeoDataFrame({"geometry": self.points, "x": xs, "y": ys,
                                "s": np.zeros(num_points, dtype="object")})

        self.tmpdir = tempfile.mkdtemp()
        self.series_filename = os.path.join(self.tmpdir, "series" + ext)
        self.frame_filename = os.path.join(self.tmpdir, "frame" + ext)
        self.points.to_file(self.series_filename, driver=driver)
        self.df.to_file(self.frame_filename, driver=driver)

    def teardown(self, ext):
        shutil.rmtree(self.tmpdir)

    def time_write_frame(self, ext):
        driver = self.driver_dict[ext]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "frame" + ext)
            self.df.to_file(out_filename, driver=driver)

    def time_write_series(self, ext):
        driver = self.driver_dict[ext]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "series" + ext)
            self.points.to_file(out_filename, driver=driver)

    def time_read_frame(self, ext):
        frame = GeoDataFrame.from_file(self.frame_filename)

    def time_read_series(self, ext):
        points = GeoSeries.from_file(self.series_filename)

    def time_read_series_from_frame(self, ext):
        points = GeoSeries.from_file(self.frame_filename)
