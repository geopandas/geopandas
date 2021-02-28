import warnings
from functools import reduce

import numpy as np
import pandas as pd

from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn


def _ensure_geometry_column(df):
    """
    Helper function to ensure the geometry column is called 'geometry'.
    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == "geometry":
        if "geometry" in df.columns:
            df.drop("geometry", axis=1, inplace=True)
        df.rename(
            columns={df._geometry_column_name: "geometry"}, copy=False, inplace=True
        )
        df.set_geometry("geometry", inplace=True)


def _overlay_intersection(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """
    # Spatial Index to create intersections
    idx1, idx2 = df2.sindex.query_bulk(df1.geometry, predicate="intersects", sort=True)
    # Create pairs of geometries in both dataframes to be intersected
    if idx1.size > 0 and idx2.size > 0:
        left = df1.geometry.take(idx1)
        left.reset_index(drop=True, inplace=True)
        right = df2.geometry.take(idx2)
        right.reset_index(drop=True, inplace=True)
        intersections = left.intersection(right)
        poly_ix = intersections.type.isin(["Polygon", "MultiPolygon"])
        intersections.loc[poly_ix] = intersections[poly_ix].buffer(0)

        # only keep actual intersecting geometries
        pairs_intersect = pd.DataFrame({"__idx1": idx1, "__idx2": idx2})
        geom_intersect = intersections

        # merge data for intersecting geometries
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = pairs_intersect.merge(
            df1.drop(df1._geometry_column_name, axis=1),
            left_on="__idx1",
            right_index=True,
        )
        dfinter = dfinter.merge(
            df2.drop(df2._geometry_column_name, axis=1),
            left_on="__idx2",
            right_index=True,
            suffixes=("_1", "_2"),
        )

        return GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs)
    else:
        return GeoDataFrame(
            [],
            columns=list(set(df1.columns).union(df2.columns)) + ["__idx1", "__idx2"],
            crs=df1.crs,
        )


def _overlay_difference(df1, df2):
    """
    Overlay Difference operation used in overlay function
    """
    # spatial index query to find intersections
    idx1, idx2 = df2.sindex.query_bulk(df1.geometry, predicate="intersects", sort=True)
    idx1_unique, idx1_unique_indices = np.unique(idx1, return_index=True)
    idx2_split = np.split(idx2, idx1_unique_indices[1:])
    sidx = [
        idx2_split.pop(0) if idx in idx1_unique else []
        for idx in range(df1.geometry.size)
    ]
    # Create differences
    new_g = []
    for geom, neighbours in zip(df1.geometry, sidx):
        new = reduce(
            lambda x, y: x.difference(y), [geom] + list(df2.geometry.iloc[neighbours])
        )
        new_g.append(new)
    differences = GeoSeries(new_g, index=df1.index, crs=df1.crs)
    poly_ix = differences.type.isin(["Polygon", "MultiPolygon"])
    differences.loc[poly_ix] = differences[poly_ix].buffer(0)
    geom_diff = differences[~differences.is_empty].copy()
    dfdiff = df1[~differences.is_empty].copy()
    dfdiff[dfdiff._geometry_column_name] = geom_diff
    return dfdiff


def _overlay_symmetric_diff(df1, df2):
    """
    Overlay Symmetric Difference operation used in overlay function
    """
    dfdiff1 = _overlay_difference(df1, df2)
    dfdiff2 = _overlay_difference(df2, df1)
    dfdiff1["__idx1"] = range(len(dfdiff1))
    dfdiff2["__idx2"] = range(len(dfdiff2))
    dfdiff1["__idx2"] = np.nan
    dfdiff2["__idx1"] = np.nan
    # ensure geometry name (otherwise merge goes wrong)
    _ensure_geometry_column(dfdiff1)
    _ensure_geometry_column(dfdiff2)
    # combine both 'difference' dataframes
    dfsym = dfdiff1.merge(
        dfdiff2, on=["__idx1", "__idx2"], how="outer", suffixes=("_1", "_2")
    )
    geometry = dfsym.geometry_1.copy()
    geometry.name = "geometry"
    # https://github.com/pandas-dev/pandas/issues/26468 use loc for now
    geometry.loc[dfsym.geometry_1.isnull()] = dfsym.loc[
        dfsym.geometry_1.isnull(), "geometry_2"
    ]
    dfsym.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = GeoDataFrame(dfsym, geometry=geometry, crs=df1.crs)
    return dfsym


def _overlay_union(df1, df2):
    """
    Overlay Union operation used in overlay function
    """
    dfinter = _overlay_intersection(df1, df2)
    dfsym = _overlay_symmetric_diff(df1, df2)
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    # keep geometry column last
    columns = list(dfunion.columns)
    columns.remove("geometry")
    columns = columns + ["geometry"]
    return dfunion.reindex(columns=columns)


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """Perform spatial overlay between two GeoDataFrames.

    Currently only supports data GeoDataFrames with uniform geometry types,
    i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
    combination of (Multi)LineString and LinearRing shapes.
    Implements several methods that are all effectively subsets of the union.

    See the User Guide page :doc:`../../user_guide/set_operations` for details.

    Parameters
    ----------
    df1 : GeoDataFrame
    df2 : GeoDataFrame
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None,
        which will set keep_geom_type to True but warn upon dropping
        geometries.
    make_valid : bool, default True
        If True, any invalid input geometries are corrected with a call to `buffer(0)`,
        if False, a `ValueError` is raised if any input geometries are invalid.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
    ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
    >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
    ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
    >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
    >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

    >>> geopandas.overlay(df1, df2, how='union')
       df1_data  df2_data                                           geometry
    0       1.0       1.0  POLYGON ((1.00000 2.00000, 2.00000 2.00000, 2....
    1       2.0       1.0  POLYGON ((3.00000 2.00000, 2.00000 2.00000, 2....
    2       2.0       2.0  POLYGON ((3.00000 4.00000, 4.00000 4.00000, 4....
    3       1.0       NaN  POLYGON ((2.00000 1.00000, 2.00000 0.00000, 0....
    4       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...
    5       NaN       1.0  MULTIPOLYGON (((2.00000 2.00000, 3.00000 2.000...
    6       NaN       2.0  POLYGON ((3.00000 4.00000, 3.00000 5.00000, 5....

    >>> geopandas.overlay(df1, df2, how='intersection')
       df1_data  df2_data                                           geometry
    0         1         1  POLYGON ((1.00000 2.00000, 2.00000 2.00000, 2....
    1         2         1  POLYGON ((3.00000 2.00000, 2.00000 2.00000, 2....
    2         2         2  POLYGON ((3.00000 4.00000, 4.00000 4.00000, 4....

    >>> geopandas.overlay(df1, df2, how='symmetric_difference')
       df1_data  df2_data                                           geometry
    0       1.0       NaN  POLYGON ((2.00000 1.00000, 2.00000 0.00000, 0....
    1       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...
    2       NaN       1.0  MULTIPOLYGON (((2.00000 2.00000, 3.00000 2.000...
    3       NaN       2.0  POLYGON ((3.00000 4.00000, 3.00000 5.00000, 5....

    >>> geopandas.overlay(df1, df2, how='difference')
                                                geometry  df1_data
    0  POLYGON ((2.00000 1.00000, 2.00000 0.00000, 0....         1
    1  MULTIPOLYGON (((2.00000 3.00000, 2.00000 4.000...         2

    >>> geopandas.overlay(df1, df2, how='identity')
       df1_data  df2_data                                           geometry
    0       1.0       1.0  POLYGON ((1.00000 2.00000, 2.00000 2.00000, 2....
    1       2.0       1.0  POLYGON ((3.00000 2.00000, 2.00000 2.00000, 2....
    2       2.0       2.0  POLYGON ((3.00000 4.00000, 4.00000 4.00000, 4....
    3       1.0       NaN  POLYGON ((2.00000 1.00000, 2.00000 0.00000, 0....
    4       2.0       NaN  MULTIPOLYGON (((3.00000 3.00000, 4.00000 3.000...

    See also
    --------
    sjoin : spatial join

    Notes
    ------
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(
            "`how` was '{0}' but is expected to be in {1}".format(how, allowed_hows)
        )

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for " "GeoDataFrames"
        )

    if not _check_crs(df1, df2):
        _crs_mismatch_warn(df1, df2, stacklevel=3)

    if keep_geom_type is None:
        keep_geom_type = True
        keep_geom_type_warning = True
    else:
        keep_geom_type_warning = False

    polys = ["Polygon", "MultiPolygon"]
    lines = ["LineString", "MultiLineString", "LinearRing"]
    points = ["Point", "MultiPoint"]
    for i, df in enumerate([df1, df2]):
        poly_check = df.geom_type.isin(polys).any()
        lines_check = df.geom_type.isin(lines).any()
        points_check = df.geom_type.isin(points).any()
        if sum([poly_check, lines_check, points_check]) > 1:
            raise NotImplementedError(
                "df{} contains mixed geometry types.".format(i + 1)
            )

    # Computations
    def _make_valid(df):
        df = df.copy()
        if df.geom_type.isin(polys).all():
            mask = ~df.geometry.is_valid
            col = df._geometry_column_name
            if make_valid:
                df.loc[mask, col] = df.loc[mask, col].buffer(0)
            elif mask.any():
                raise ValueError(
                    "You have passed make_valid=False along with "
                    f"{mask.sum()} invalid input geometries. "
                    "Use make_valid=True or make sure that all geometries "
                    "are valid before using overlay."
                )
        return df

    df1 = _make_valid(df1)
    df2 = _make_valid(df2)

    with warnings.catch_warnings():  # CRS checked above, supress array-level warning
        warnings.filterwarnings("ignore", message="CRS mismatch between the CRS")
        if how == "difference":
            return _overlay_difference(df1, df2)
        elif how == "intersection":
            result = _overlay_intersection(df1, df2)
        elif how == "symmetric_difference":
            result = _overlay_symmetric_diff(df1, df2)
        elif how == "union":
            result = _overlay_union(df1, df2)
        elif how == "identity":
            dfunion = _overlay_union(df1, df2)
            result = dfunion[dfunion["__idx1"].notnull()].copy()

    if keep_geom_type:
        key_order = result.keys()
        exploded = result.reset_index(drop=True).explode()
        exploded = exploded.reset_index(level=0)

        orig_num_geoms = result.shape[0]
        geom_type = df1.geom_type.iloc[0]
        if geom_type in polys:
            exploded = exploded.loc[exploded.geom_type.isin(polys)]
        elif geom_type in lines:
            exploded = exploded.loc[exploded.geom_type.isin(lines)]
        elif geom_type in points:
            exploded = exploded.loc[exploded.geom_type.isin(points)]
        else:
            raise TypeError("`keep_geom_type` does not support {}.".format(geom_type))

        # level_0 created with above reset_index operation
        # and represents the original geometry collections
        result = exploded.dissolve(by="level_0")[key_order]

        if (result.shape[0] != orig_num_geoms) and keep_geom_type_warning:
            num_dropped = orig_num_geoms - result.shape[0]
            warnings.warn(
                "`keep_geom_type=True` in overlay resulted in {} dropped "
                "geometries of different geometry types than df1 has. "
                "Set `keep_geom_type=False` to retain all "
                "geometries".format(num_dropped),
                UserWarning,
                stacklevel=2,
            )

    result.reset_index(drop=True, inplace=True)
    result.drop(["__idx1", "__idx2"], axis=1, inplace=True)
    return result
