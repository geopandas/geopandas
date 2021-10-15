import abc
import concurrent.futures
import dataclasses
import os
import time
import uuid
import warnings
from pprint import pprint
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.geometry
import shapely.ops
from shapely.validation import explain_validity

GenericPoly = Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]
warnings.filterwarnings("ignore", "`keep_geom_type=True` in overlay", UserWarning)


def strings_to_uuid_v5(*args: str) -> str:
    """Creates a reproducible hash of the argument strings into a v5 uuid string"""
    if not args:
        raise ValueError("You must pass at least one string")
    _NAMESPACE_UUID = uuid.UUID("29384bda-193e-4c22-9f4d-85d91bcef08c")
    return str(uuid.uuid5(_NAMESPACE_UUID, "_".join(args)))


@dataclasses.dataclass(frozen=True)
class RandomPolyGenerator(abc.ABC):
    top_left_tile: Tuple[int, int] = (18807, 23776)  # chosen only so that it can be loaded into QGIS easily (@z=16)
    px_per_tile: int = 1024

    def __call__(self, seed: int) -> gpd.GeoDataFrame:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""
        np.random.seed(seed)
        polys = self._fill_unit_square()
        polys = self._scale(polys)
        polys = self._translate(polys)
        polys = self._clip_by_confidence(polys)
        return gpd.GeoDataFrame(polys, columns=["uid", "confidence", "geometry"], geometry="geometry")

    @abc.abstractmethod
    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """Use this method to fill the unit square with polygons and their associated 'confidence'"""

    def _scale(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[float, GenericPoly]]:
        return [
            (confidence, shapely.affinity.scale(geom, xfact=self.px_per_tile, yfact=self.px_per_tile))
            for confidence, geom in polygons
        ]

    def _translate(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[float, GenericPoly]]:
        return [
            (confidence, shapely.affinity.translate(geom, xoff=self.top_left_tile[0], yoff=self.top_left_tile[1]))
            for confidence, geom in polygons
        ]

    def _clip_by_confidence(self, polygons: List[Tuple[float, GenericPoly]]) -> List[Tuple[str, float, GenericPoly]]:
        """Clips the polygons based on confidence ordering, also creates 'row' UIDs for each polygon"""
        polygons.sort(reverse=True)
        new_polygons = []
        region = None
        for i, (confidence, poly) in enumerate(polygons):
            uid = strings_to_uuid_v5(str(i))
            if i == 0:
                newpoly = poly
                region = poly  # highest confidence, no need to do anything
            else:
                newpoly = poly.difference(region)  # clip to region
                region = region.union(poly)  # update region
            if not newpoly.is_valid:
                raise RuntimeError(
                    f"Invalid geometry after clip: {explain_validity(poly)=}, {explain_validity(newpoly)=}"
                )
            new_polygons.append((uid, confidence, newpoly))
        return new_polygons


@dataclasses.dataclass(frozen=True)
class RandomTargetsGenerator(RandomPolyGenerator):
    min_arc_length: float = 0.01
    radius_step: float = 0.01

    def _fill_unit_square(self) -> List[Tuple[float, GenericPoly]]:
        """See RandomExample._fill_unit_square"""
        polygons = []
        radius = 0.5 - self.radius_step
        while radius > 4 * self.radius_step:
            polygons.append(self._make_ring(radius))
            radius -= 2 * self.radius_step
        return polygons

    def _make_ring(self, radius: float) -> Tuple[float, GenericPoly]:
        outer = self._make_circle(radius)
        for _ in range(10):
            inner_radius = radius - self.radius_step
            inner = self._make_circle(inner_radius)
            poly = outer.difference(inner)
            if poly.is_valid:
                return np.random.uniform(0.5, 1), poly
        raise RuntimeError(f"Couldn't make ring for {radius=}")

    def _make_circle(self, radius: float) -> GenericPoly:
        delta_theta = self.min_arc_length / radius
        n_steps = int(np.round(2 * np.pi / delta_theta))
        theta = np.linspace(0, 2 * np.pi, num=n_steps, endpoint=False)
        radii = radius + np.random.uniform(-0.6 * self.radius_step, 0.6 * self.radius_step, size=n_steps)
        xy = radii * np.vstack((np.cos(theta), np.sin(theta)))
        # shift into the unit square
        xy[0, :] += 0.5
        xy[1, :] += 0.5
        return shapely.geometry.Polygon(xy.T)


@dataclasses.dataclass(frozen=True)
class ExecutionTest:
    data_generator: RandomPolyGenerator

    def __call__(self, seed: int, stop_at_breakpoint: bool = False) -> Dict:
        gdf1 = self.data_generator(seed)
        gdf2 = self.data_generator(seed + 1_000_000_000)
        _ = gdf1.sindex, gdf2.sindex  # pre-calculate to avoid impacting timings
        funcs = {
            "gdf1_gdf1_overlay_intersection": lambda: gpd.overlay(gdf1, gdf1, how="intersection"),
            "gdf1_gdf2_overlay_intersection": lambda: gpd.overlay(gdf1, gdf2, how="intersection"),
            "gdf2_gdf2_overlay_intersection": lambda: gpd.overlay(gdf2, gdf2, how="intersection"),
            "gdf1_gdf1_overlay_union": lambda: gpd.overlay(gdf1, gdf1, how="union"),
            "gdf1_gdf2_overlay_union": lambda: gpd.overlay(gdf1, gdf2, how="union"),
            "gdf2_gdf2_overlay_union": lambda: gpd.overlay(gdf2, gdf2, how="union"),
            "gdf1_gdf1_overlay_difference": lambda: gpd.overlay(gdf1, gdf1, how="difference"),
            "gdf1_gdf2_overlay_difference": lambda: gpd.overlay(gdf1, gdf2, how="difference"),
            "gdf2_gdf2_overlay_difference": lambda: gpd.overlay(gdf2, gdf2, how="difference"),
        }
        result = {"data_generator": str(self.data_generator), "seed": seed}
        for name, func in funcs.items():
            errcol, timecol = f"err_{name}", f"time_{name}"
            t0 = time.monotonic()
            try:
                gdf = func()
                result[errcol] = None if gdf.is_valid.all() else "Invalid geometries"
            except Exception as e:
                result[errcol] = str(e)
            finally:
                result[timecol] = time.monotonic() - t0
        if stop_at_breakpoint:
            breakpoint()
        return result


def main():
    test = ExecutionTest(RandomTargetsGenerator())

    n_workers = os.cpu_count()
    n_tests = 10 * n_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        cf_result = list(pool.map(test, range(n_tests)))
    result = pd.DataFrame(cf_result)

    time_cols = [c for c in result.columns if c.startswith("time_")]
    print("*" * 80)
    print(result[time_cols].describe().T)

    err_cols = [c for c in result.columns if c.startswith("err_")]
    print("*" * 80)
    pprint({c: result[c].dropna().unique() for c in err_cols})
    breakpoint()

    # To dig into the details of one of the failing cases, just use the seed value
    # test(0, stop_at_breakpoint=True)


if __name__ == "__main__":
    main()
