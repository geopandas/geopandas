import os
import shutil
import tempfile
import warnings

from geopandas import GeoDataFrame, GeoSeries, read_parquet
import numpy as np
from shapely.geometry import Point


# TEMP: hide warning from to_parquet
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")


class Bench:
    def setup(self):
        num_points = 20000
        xs = np.random.rand(num_points)
        ys = np.random.rand(num_points)

        self.points = GeoSeries([Point(x, y) for (x, y) in zip(xs, ys)])
        self.df = GeoDataFrame(
            {
                "geometry": self.points,
                "x": xs,
                "y": ys,
                "s": np.zeros(num_points, dtype="object"),
            }
        )

        self.tmpdir = tempfile.mkdtemp()
        self.filename = os.path.join(self.tmpdir, "frame.pq")

        self.df.to_parquet(self.filename)

    def teardown(self):
        shutil.rmtree(self.tmpdir)

    def time_read_parquet(self):
        read_parquet(self.filename)

    def time_write_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.df.to_parquet(os.path.join(tmpdir, "frame.pq"))
