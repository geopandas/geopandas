import os
import shutil
import tempfile

from geopandas import GeoDataFrame, GeoSeries
import numpy as np
from shapely.geometry import Point


class Bench:

    def setup(self):

        num_points = 50000
        xs = np.random.rand(num_points)
        ys = np.random.rand(num_points)

        self.points = GeoSeries([Point(x, y) for (x, y) in zip(xs, ys)])
        self.df = GeoDataFrame({"geometry": self.points, "x": xs, "y": ys,
                                "s": np.zeros(num_points, dtype="object")})

        self.tmpdir = tempfile.mkdtemp()
        self.series_filename = os.path.join(self.tmpdir, "series.shp")
        self.frame_filename = os.path.join(self.tmpdir, "frame.shp")
        self.points.to_file(self.series_filename)
        self.df.to_file(self.frame_filename)

    def teardown(self):
        shutil.rmtree(self.tmpdir)

    def time_write_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "frame.shp")
            self.df.to_file(out_filename)

    def time_write_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "series.shp")
            self.points.to_file(out_filename)

    def time_read_frame(self):
        frame = GeoDataFrame.from_file(self.frame_filename)

    def time_read_series(self):
        points = GeoSeries.from_file(self.series_filename)

    def time_read_series_from_frame(self):
        points = GeoSeries.from_file(self.frame_filename)
