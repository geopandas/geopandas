import concurrent.futures
import dataclasses
import os
import time

import pandas as pd
import shapely
import shapely.geometry

import geopandas as gpd

from .generators import RandomPolyGenerator


@dataclasses.dataclass(frozen=True)
class TestCase:
    generator: RandomPolyGenerator
    seed: int
    gdf1: gpd.GeoDataFrame = dataclasses.field(init=False)
    gdf2: gpd.GeoDataFrame = dataclasses.field(init=False)
    poly1: shapely.geometry.MultiPolygon = dataclasses.field(init=False)
    poly2: shapely.geometry.MultiPolygon = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "gdf1", self.generator(self.seed))
        object.__setattr__(self, "gdf2", self.generator(self.seed + 1_000_000_000))
        object.__setattr__(self, "poly1", self.gdf1.unary_union)
        object.__setattr__(self, "poly2", self.gdf2.unary_union)
        if self.generator.precision is not None:
            assert (
                shapely.get_precision(self.gdf1.geometry) == self.generator.precision
            ).all()
            assert (
                shapely.get_precision(self.gdf2.geometry) == self.generator.precision
            ).all()
            assert shapely.get_precision(self.poly1) == self.generator.precision
            assert shapely.get_precision(self.poly2) == self.generator.precision
        _ = (
            self.gdf1.sindex,
            self.gdf2.sindex,
        )  # pre-calculate to avoid impacting timings

    def gdf1_is_valid(self) -> bool:
        return self.gdf1.is_valid.all()

    def gdf2_is_valid(self) -> bool:
        return self.gdf2.is_valid.all()

    def _return_gdf(self, gdf):
        if self.generator.precision is not None:
            assert (
                shapely.get_precision(gdf.geometry) == self.generator.precision
            ).all()
        return gdf

    def gdf1_gdf1_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf1, how="intersection", keep_geom_type=True)
        )

    def gdf1_gdf2_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf2, how="intersection", keep_geom_type=True)
        )

    def gdf2_gdf2_overlay_intersection(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf2, self.gdf2, how="intersection", keep_geom_type=True)
        )

    def gdf1_gdf1_overlay_union(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf1, how="union", keep_geom_type=True)
        )

    def gdf1_gdf2_overlay_union(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf2, how="union", keep_geom_type=True)
        )

    def gdf2_gdf2_overlay_union(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf2, self.gdf2, how="union", keep_geom_type=True)
        )

    def gdf1_gdf1_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf1, how="difference", keep_geom_type=True)
        )

    def gdf1_gdf2_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf1, self.gdf2, how="difference", keep_geom_type=True)
        )

    def gdf2_gdf2_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf2, self.gdf2, how="difference", keep_geom_type=True)
        )

    def gdf2_gdf1_overlay_difference(self) -> gpd.GeoDataFrame:
        return self._return_gdf(
            gpd.overlay(self.gdf2, self.gdf1, how="difference", keep_geom_type=True)
        )

    def poly1_is_valid(self) -> bool:
        return self.poly1.is_valid

    def poly2_is_valid(self) -> bool:
        return self.poly2.is_valid

    def _return_poly(self, poly):
        if self.generator.precision is not None:
            assert shapely.get_precision(poly) == self.generator.precision
        return poly

    def poly1_poly1_overlay_intersection(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.intersection(self.poly1))

    def poly1_poly2_overlay_intersection(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.intersection(self.poly2))

    def poly2_poly2_overlay_intersection(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly2.intersection(self.poly2))

    def poly1_poly1_overlay_union(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.union(self.poly1))

    def poly1_poly2_overlay_union(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.union(self.poly2))

    def poly2_poly2_overlay_union(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly2.union(self.poly2))

    def poly1_poly1_overlay_difference(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.difference(self.poly1))

    def poly1_poly2_overlay_difference(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly1.difference(self.poly2))

    def poly2_poly2_overlay_difference(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly2.difference(self.poly2))

    def poly2_poly1_overlay_difference(self) -> shapely.geometry.MultiPolygon:
        return self._return_poly(self.poly2.difference(self.poly1))

    def time_all(self) -> pd.DataFrame:
        functions = [
            f for f in dir(self) if f.startswith(("gdf1_", "gdf2_", "poly1_", "poly2_"))
        ]
        results = []
        for func in functions:
            t0 = time.monotonic()
            try:
                result = getattr(self, func)()
                if (
                    isinstance(result, shapely.geometry.base.BaseGeometry)
                    and not result.is_valid
                ):
                    err = "Invalid geometry"
                elif isinstance(result, gpd.GeoDataFrame) and not result.is_valid.all():
                    err = "Invalid geometries"
                else:
                    err = None
            except Exception as e:
                err = str(e)
            finally:
                elapsed = time.monotonic() - t0
            results.append(
                {
                    "shapely_version": shapely.__version__,
                    "generator": self.generator.name(),
                    "seed": self.seed,
                    "function": func,
                    "error": err,
                    "time_ms": 1000 * elapsed,
                }
            )
        return pd.DataFrame(results)


@dataclasses.dataclass(frozen=True)
class ExecutionAndTimingTest:
    generator: RandomPolyGenerator

    def __call__(self, seed: int) -> pd.DataFrame:
        return TestCase(self.generator, seed).time_all()


def run_execution_and_timing_test(
    generator: RandomPolyGenerator,
    n_procs: int = os.cpu_count(),
    n_tests: int = 10 * os.cpu_count(),
) -> pd.DataFrame:
    test = ExecutionAndTimingTest(generator)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(n_procs, n_tests)
    ) as pool:
        return pd.concat(pool.map(test, range(n_tests)), ignore_index=True)
