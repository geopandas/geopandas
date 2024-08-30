import os
import shutil
import tempfile
import warnings

import numpy as np

from shapely.geometry import Point

from geopandas import GeoDataFrame, GeoSeries, read_file, read_parquet, read_feather


# TEMP: hide warning from to_parquet
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")


format_dict = {
    "ESRI Shapefile": (
        ".shp",
        lambda gdf, filename: gdf.to_file(filename, driver="ESRI Shapefile"),
        lambda filename: read_file(filename, driver="ESRI Shapefile"),
    ),
    "GeoJSON": (
        ".json",
        lambda gdf, filename: gdf.to_file(filename, driver="GeoJSON"),
        lambda filename: read_file(filename, driver="GeoJSON"),
    ),
    "GPKG": (
        ".gpkg",
        lambda gdf, filename: gdf.to_file(filename, driver="GeoJSON"),
        lambda filename: read_file(filename, driver="GeoJSON"),
    ),
    "Parquet": (
        ".parquet",
        lambda gdf, filename: gdf.to_parquet(filename),
        lambda filename: read_parquet(filename),
    ),
    "Feather": (
        ".feather",
        lambda gdf, filename: gdf.to_feather(filename),
        lambda filename: read_feather(filename),
    ),
}


class Bench:
    params = ["ESRI Shapefile", "GeoJSON", "GPKG", "Parquet", "Feather"]
    param_names = ["file_format"]

    def setup(self, file_format):
        self.ext, self.writer, self.reader = format_dict[file_format]

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
        self.filename = os.path.join(self.tmpdir, "frame" + self.ext)
        self.writer(self.df, self.filename)

    def teardown(self, file_format):
        shutil.rmtree(self.tmpdir)


class BenchFrame(Bench):
    params = ["ESRI Shapefile", "GeoJSON", "GPKG", "Parquet", "Feather"]
    param_names = ["file_format"]

    def time_write(self, file_format):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "frame" + self.ext)
            self.writer(self.df, out_filename)

    def time_read(self, file_format):
        self.reader(self.filename)


class BenchSeries(Bench):
    params = ["ESRI Shapefile", "GeoJSON", "GPKG"]
    param_names = ["file_format"]

    def setup(self, file_format):
        super().setup(file_format)
        self.filename_series = os.path.join(self.tmpdir, "series" + self.ext)
        self.writer(self.points, self.filename_series)

    def time_write_series(self, file_format):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_filename = os.path.join(tmpdir, "series" + self.ext)
            self.writer(self.points, out_filename)

    def time_read_series(self, file_format):
        GeoSeries.from_file(self.filename_series)

    def time_read_series_from_frame(self, file_format):
        GeoSeries.from_file(self.filename)
