# Changelog

## Version 1.1.3 (Feburary XX, 2026)

Bug fixes:
- Improved compatibility with pandas 3.0 Copy-on-Write feature, making use of deferred copies where possible (#3298).
- Fix `GeoSeries.sample_points` not accepting list-like `size` when generating points using
  `pointpaterns` (#3710).
- Fix `from_wkt/wkb` to correctly handle missing values with pandas 3 (where the new `str` dtype is used) (#3714).
- Fix `to_postgis` to correctly handle missing values with pandas 3 (where the new `str` dtype is used) (#3722).

## Version 1.1.2 (December 22, 2025)

Bug fixes:

- Fix an issue that caused an error in `GeoDataFrame.from_features` when there is no `properties` field (#3599).
- Fix `read_file` and `to_file` errors (#3682)
- Fix `read_parquet` with `to_pandas_kwargs` for complex (list/struct) arrow types (#3640)
- `value_counts` on GeoSeries now preserves CRS in index (#3669)
- Fix f-string placeholders appearing in error messages when `pyogrio` cannot be imported (#3682).
- Fix `read_parquet` with `to_pandas_kwargs` for complex (list/struct) arrow types (#3640).
- `.to_json` now provides a clearer error message when called on a GeoDataFrame without an active geometry
  column (#3648).
- Calling `del gdf["geometry"]` now will downcast to a `pd.DataFrame` if there are no geometry columns left
  in the dataframe (#3648).
- Fix SQL injection in `to_postgis` via geometry column name (#3681).

## Version 1.1.1 (June 27, 2025)

Bug fixes:

- Fix regression in the GeoDataFrame constructor when np.nan is given as an only geometry (#3591).
- Fix regression in `overlay` with `how="identity"` when input dataframes have column
  names that are equal (#3596).

## Version 1.1.0 (June 1, 2025)

Notes on dependencies:

- GeoPandas 1.1 now requires Python 3.10 or greater and pandas 2.0, numpy 1.24, pyproj 3.5,
  are now the minimum required version for these dependencies.
  Furthermore, the minimum tested version for optional dependencies has been updated to
  fiona 1.8.21, scipy 1.9, matplotlib 3.7, mapclassify 2.5, folium 0.12 and
  SQLAlchemy 2.0. Older versions of these libraries may continue to work, but are no longer
  considered supported (#3371).

New features and improvements:

- Added options to return the result of `SpatialIndex.query` in a form of a dense or a
  sparse boolean array. This adds optional dependency on `scipy` for the sparse output.
  Note that this also changes the previously undocumented behaviour of the `output_format`
  keyword (#1674).
- Add ``grid_size`` parameter to ``union_all`` and ``dissolve`` (#3445).
- `GeoDataFrame.plot` now supports `pd.Index` as an input for the `column` keyword (#3463).
- Added `disjoint_subset` union algorithm for `union_all` and `dissolve` (#3534).
- Added `constrained_delaunay_triangles` method to GeoSeries/GeoDataFrame (#3552).
- Added `to_pandas_kwargs` argument to `from_arrow`, `read_parquet` and `read_feather`
  to allow better control of conversion of non-geometric Arrow data to DataFrames (#3466).
- Added `is_valid_coverage` and `invalid_coverage_edges` to GeoSeries/GeoDataFrame to
  allow validation of polygonal  coverage (#3545).
- Added `maximum_inscribed_circle` method from shapely to GeoSeries/GeoDataFrame (#3544).
- Added `minimum_clearance_line` method from shapely to GeoSeries/GeoDataFrame (#3543).
- Added `orient_polygons` method from shapely to GeoSeries/GeoDataFrame (#3559).
- Added ``method`` and ``keep_collapsed`` argument to ``make_valid`` (#3548).
- Added `simplify_coverage` method for topological simplification of polygonal coverages
  to GeoSeries/GeoDataFrame (#3541).
- Added initial support of M coordinates (`m` and `has_m` properties, `include_m` in `get_coordinates`) (#3561).
- Added `geom_equals_identical` method exposing `equals_identical` from shapely to GeoSeries/GeoDataFrame (#3560).
- GeoPandas now attempts to use a range request when reading from an URL even if the header
  does not directly indicate its support (#3572).
- Added `geopandas.accessors` module. Import this module to register a
  `pandas.Series.geo` accessor, which exposes GeoSeries methods via pandas's
  extension mechanism (#3272).
- Improve performance of `overlay` with `how=identity` (#3504).
- A warning message is raised in `read_file` when a GeoDataFrame or GeoSeries mask
  and/or the source dataset is missing a defined CRS (#3464).
- GeoDataFrame no longer hard-codes the class internally, allowing easier subclassing (#3505).

Bug fixes:

- Fix an issue that showed numpy dtypes in bbox in `to_geo_dict` and `__geo_interface__`. (#3436).
- Fix an issue in `sample_points` that could occasionally result in non-uniform distribution (#3470).
- Fix unspecified layer warning being emitted while reading multilayer datasets, even
  when layer is specified when using the mask or bbox keywords (#3378).
- Properly support named aggregations over a geometry column in `GroupBy.agg` (#3368).
- Support GeoDataFrame constructor receiving arguments to `geometry` which are not
  (Geo)Series, but instead should be interpreted as column names, like Enums (#3384).
- Fix regression where constructing a GeoSeries from a pd.Series with GeometryDtype values
  failed when `crs` was provided (#3383).
- Fix regression where `overlay` with `keep_geom_type` returns wrong results if the
  input contains invalid geometries (#3395).
- Fix the dtype of the GeometryArray backing data being incorrect for zero length
  GeoDataFrames causing errors in `overlay` (#3424).
- Fix regression where constructing a GeoSeries from a pd.Series with GeometryDtype values
  failed when `crs` was provided (#3383).
- Fix plotting of polygons with holes by normalizing the coordinate order prior to plotting (#3483).
- Fix an issue in plotting when polygon patches were not closed (#3576).
- Fix ambiguous error when GeoDataFrame is initialised with a column called "crs" (#3502).
- Avoid change of the plot aspect when plotting missing values (#3438).

Deprecations and compatibility notes:

- The `GeoSeries.select` method wrapping the pandas `Series.select` method has been removed.
  The upstream method no longer exists in all supported version of pandas (#3394).
- The deprecated `geom_almost_equals` method has been removed. Use
  `geom_equals_exact` instead (#3522).

## Version 1.0.1 (July 2, 2024)

Bug fixes:

- Support a named datetime or object dtype index in `explore()` (#3360, #3364).
- Fix a regression preventing a Series as an argument for geometric methods (#3363).

## Version 1.0.0 (June 24, 2024)

Notes on dependencies:

- GeoPandas 1.0 drops support for shapely<2 and PyGEOS. The only geometry engine that is
  currently supported is shapely >= 2. As a consequence, spatial indexing based on the
  rtree package has also been removed (#3035).
- The I/O engine now defaults to Pyogrio which is now installed with GeoPandas instead
  of Fiona (#3223).

New methods:

- Added `count_geometries` method from shapely to GeoSeries/GeoDataFrame (#3154).
- Added `count_interior_rings` method from shapely to GeoSeries/GeoDataFrame (#3154).
- Added `relate_pattern` method from shapely to GeoSeries/GeoDataFrame (#3211).
- Added `intersection_all` method from shapely to GeoSeries/GeoDataFrame (#3228).
- Added `line_merge` method from shapely to GeoSeries/GeoDataFrame (#3214).
- Added `set_precision` and `get_precision` methods from shapely to GeoSeries/GeoDataFrame (#3175).
- Added `count_coordinates` method from shapely to GeoSeries/GeoDataFrame (#3026).
- Added `minimum_clearance` method from shapely to GeoSeries/GeoDataFrame (#2989).
- Added `shared_paths` method from shapely to GeoSeries/GeoDataFrame (#3215).
- Added `is_ccw` method from shapely to GeoSeries/GeoDataFrame (#3027).
- Added `is_closed` attribute from shapely to GeoSeries/GeoDataFrame (#3092).
- Added `force_2d` and `force_3d` methods from shapely to GeoSeries/GeoDataFrame (#3090).
- Added `voronoi_polygons` method from shapely to GeoSeries/GeoDataFrame (#3177).
- Added `contains_properly` method from shapely to GeoSeries/GeoDataFrame (#3105).
- Added `build_area` method exposing `build_area` shapely to GeoSeries/GeoDataFrame (#3202).
- Added `snap` method from shapely to GeoSeries/GeoDataFrame (#3086).
- Added `transform` method from shapely to GeoSeries/GeoDataFrame (#3075).
- Added `get_geometry` method from shapely to GeoSeries/GeoDataFrame (#3287).
- Added `dwithin` method to check for a "distance within" predicate on
  GeoSeries/GeoDataFrame (#3153).
- Added `to_geo_dict` method to generate GeoJSON-like dictionary from a GeoDataFrame (#3132).
- Added `polygonize` method exposing both `polygonize` and `polygonize_full` from
  shapely to GeoSeries/GeoDataFrame (#2963).
- Added `is_valid_reason` method from shapely to GeoSeries/GeoDataFrame (#3176).
- Added `to_arrow` method and `from_arrow` class method to
  GeoSeries/GeoDataFrame to export and import to/from Arrow data with GeoArrow
  extension types (#3219, #3301).

New features and improvements:

- Added ``predicate="dwithin"`` option and ``distance`` argument to the ``sindex.query()`` method
 and ``sjoin`` (#2882).
- GeoSeries and GeoDataFrame `__repr__` now trims trailing zeros for a more readable
  output (#3087).
- Add `on_invalid` parameter to `from_wkt` and `from_wkb` (#3110).
- `make_valid` option in `overlay` now uses the `make_valid` method instead of
  `buffer(0)` (#3113).
- Passing `"geometry"` as `dtype` to `pd.read_csv` will now return a GeoSeries for
  the specified columns (#3101).
- Added support to ``read_file`` for the ``mask`` keyword for the pyogrio engine (#3062).
- Added support to ``read_file`` for the ``columns`` keyword for the fiona engine (#3133).
- Added support to ``to_parquet`` and ``read_parquet`` for writing and reading files
  using the GeoArrow-based native geometry encoding of GeoParquet 1.1 (#3253, #3275).
- Add `sort` keyword to `clip` method for GeoSeries and GeoDataFrame to allow optional
  preservation of the original order of observations (#3233).
- Added `show_bbox`, `drop_id` and `to_wgs84` arguments to allow further customization of
  `GeoSeries.to_json` (#3226).
- `explore` now supports `GeoDataFrame`s with additional columns containing datetimes, uuids and
  other non JSON serializable objects (#3261).
- The `GeoSeries.fillna` method now supports the `limit` keyword (#3290).
- Added ``on_attribute`` option argument to the ``sjoin()``
  method, allowing to restrict joins to the observations with
  matching attributes (#3231).
- Added support for `bbox` covering encoding in geoparquet. Can filter reading of parquet
files based on a bounding box, and write out a bounding box column to parquet files (#3282).
- `align` keyword in binary methods now defaults to `None`, treated as True. Explicit True
  will silence the warning about mismatched indices (#3212).
- `GeoSeries.set_crs` can now be used to remove CRS information by passing
  `crs=None, allow_override=True` (#3316).
- Added ``autolim`` keyword argument to ``GeoSeries.plot()`` and ``GeoDataFrame.plot()`` (#2817).
- Added `metadata` parameter to `GeoDataFrame.to_file` (#2850).
- Updated documentation to clarify that passing a named (Geo)Series as the `geometry`
  argument to the GeoDataFrame constructor will not use the name but will always
  produce a GeoDataFrame with an active geometry column named "geometry" (#3337).
- `read_postgis` will query the spatial_ref_sys table to determine the CRS authority
  instead of its current behaviour of assuming EPSG. In the event the spiatal_ref_sys
  table is not present, or the SRID is not present, `read_postgis` will fallback
  on assuming EPSG CRS authority (#3329).
- Added ``GeoDataFrame.active_geometry_name`` property returning the active geometry column's name or None if no active geometry column is set (#2943).

Backwards incompatible API changes:

- The `sjoin` method will now preserve the name of the index of the right
  GeoDataFrame, if it has one, instead of always using `"index_right"` as the
  name for the resulting column in the return value (#846, #2144).
- GeoPandas now raises a ValueError when an unaligned Series is passed as a method
  argument to avoid confusion of whether the automatic alignment happens or not (#3271).
- The deprecated default value of GeoDataFrame/ GeoSeries `explode(.., index_parts=True)` is now
  set to false for consistency with pandas (#3174).
- The behaviour of `set_geometry` has been changed when passed a (Geo)Series `ser` with a name.
  The new active geometry column name in this case will be `ser.name`, if not None, rather than
  the previous active geometry column name. This means that if the new and old names are
  different, then both columns will be preserved in the GeoDataFrame. To replicate the previous
  behaviour, you can instead call `gdf.set_geometry(ser.rename(gdf.active_geometry_name))` (#3237).
  Note that this behaviour change does not affect the `GeoDataFrame` constructor, passing a named
  GeoSeries `ser` to `GeoDataFrame(df, geometry=ser)` will always produce a GeoDataFrame with a
  geometry column named "geometry" to preserve backwards compatibility. If you would like to
  instead propagate the name of `ser` when constructing a GeoDataFrame, you can instead call
  `df.set_geometry(ser)` or `GeoDataFrame(df, geometry=ser).rename_geometry(ser.name)` (#3337).
- `delaunay_triangles` now considers all geometries together when creating the Delaunay triangulation
  instead of performing the operation element-wise. If you want to generate Delaunay
  triangles for each geometry separately, use ``shapely.delaunay_triangles`` instead. (#3273)
- Reading a data source that does not have a geometry field using ``read_file``
  now returns a Pandas DataFrame instead of a GeoDataFrame with an empty
  ``geometry`` column.

Enforced deprecations:

- The deprecation of `geopandas.datasets` has been enforced and the module has been
  removed. New sample datasets are now available in the
  [geodatasets](https://geodatasets.readthedocs.io/en/latest/) package (#3084).
- Many longstanding deprecated functions, methods and properties have been removed (#3174), (#3190)
  - Removed deprecated functions
    `geopandas.io.read_file`, `geopandas.io.to_file` and `geopandas.io.sql.read_postgis`.
    `geopandas.read_file`, `geopandas.read_postgis` and the GeoDataFrame/GeoSeries `to_file(..)`
    method should be used instead.
  - Removed deprecated `GeometryArray.data` property, `np.asarray(..)` or the `to_numpy()`
    method should be used instead.
  - Removed deprecated `sindex.query_bulk` method, using `sindex.query` instead.
  - Removed deprecated `sjoin` parameter `op`, `predicate` should be supplied instead.
  - Removed deprecated GeoSeries/ GeoDataFrame methods `__xor__`, `__or__`, `__and__` and
    `__sub__`. Instead use methods `symmetric_difference`, `union`, `intersection` and
    `difference` respectively.
  - Removed deprecated plotting functions `plot_polygon_collection`,
    `plot_linestring_collection` and `plot_point_collection`, use the GeoSeries/GeoDataFrame `.plot`
    method directly instead.
  - Removed deprecated GeoSeries/GeoDataFrame `.plot` parameters `axes` and `colormap`, instead use
    `ax` and `cmap` respectively.
  - Removed compatibility for specifying the `version` keyword in `to_parquet` and `to_feather`.
    This keyword will now be passed through to pyarrow and use `schema_version` to specify the GeoParquet specification version (#3334).

New deprecations:

- `unary_union` attribute is now deprecated and replaced by the `union_all()` method (#3007) allowing
  opting for a faster union algorithm for coverages (#3151).
- The ``include_fields`` and ``ignore_fields`` keywords in ``read_file()`` are deprecated
  for the default pyogrio engine. Currently those are translated to the ``columns`` keyword
  for backwards compatibility, but you should directly use the ``columns`` keyword instead
  to select which columns to read (#3133).
- The `drop` keyword in `set_geometry` has been deprecated, and in future the `drop=True`
  behaviour will be removed (#3237). To prepare for this change, you should remove any explicit
  `drop=False` calls in your code (the default behaviour already is the same as `drop=False`).
  To replicate the previous `drop=True` behaviour you should replace
  `gdf.set_geometry(new_geo_col, drop=True)` with

  ```python
  geo_col_name = gdf.active_geometry_name
  gdf.set_geometry(new_geo_col).drop(columns=geo_col_name).rename_geometry(geo_col_name)
  ```
- The `geopandas.use_pygeos` option has been deprecated and will be removed in GeoPandas
  1.1 (#3283)
- Manual overriding of an existing CRS of a GeoSeries or GeoDataFrame by setting the `crs` property has been deprecated
  and will be disabled in future. Use the `set_crs()` method instead (#3085).

Bug fixes:

- Fix `GeoDataFrame.merge()` incorrectly returning a `DataFrame` instead of a
  `GeoDataFrame` when the `suffixes` argument is applied to the active
  geometry column (#2933).
- Fix bug in `GeoDataFrame` constructor where if `geometry` is given a named
  `GeoSeries` the name was not used as the active geometry column name (#3237).
- Fix bug in `GeoSeries` constructor when passing a Series and specifying a `crs` to not change the original input data (#2492).
- Fix regression preventing reading from file paths containing hashes in `read_file`
  with the fiona engine (#3280). An analogous fix for pyogrio is included in
  pyogrio 0.8.1.
- Fix `to_parquet` to write correct metadata in case of 3D geometries (#2824).
- Fixes for compatibility with psycopg (#3167).
- Fix to allow appending dataframes with no CRS to PostGIS tables with no CRS (#3328)
- Fix plotting of all-empty GeoSeries using `explore` (#3316).

## Version 0.14.4 (April 26, 2024)

- Several fixes for compatibility with the upcoming pandas 3.0, numpy 2.0 and
  fiona 1.10 releases.

## Version 0.14.3 (Jan 31, 2024)

- Several fixes for compatibility with the latest pandas 2.2 release.
- Fix bug in `pandas.concat` CRS consistency checking where CRS differing by WKT
  whitespace only were treated as incompatible (#3023).

## Version 0.14.2 (Jan 4, 2024)

- Fix regression in `overlay` where using `buffer(0)` instead of `make_valid` internally
  produced invalid results (#3074).
- Fix `explore()` method when the active geometry contains missing and empty geometries (#3094).

## Version 0.14.1 (Nov 11, 2023)

- The Parquet and Feather IO functions now support the latest 1.0.0 version
  of the GeoParquet specification (geoparquet.org) (#2663).
- Fix `read_parquet` and `read_feather` for [CVE-2023-47248](https://www.cve.org/CVERecord?id=CVE-2023-47248>) (#3070).

## Version 0.14 (Sep 15, 2023)

GeoPandas will use Shapely 2.0 by default instead of PyGEOS when both Shapely >= 2.0 and
PyGEOS are installed.  PyGEOS will continue to be used by default when PyGEOS is
installed alongside Shapely < 2.0.  Support for PyGEOS and Shapely < 2.0 will be removed
in GeoPandas 1.0. (#2999)

API changes:

- ``seed`` keyword in ``sample_points`` is deprecated. Use ``rng`` instead. (#2913).

New methods:

- Added ``concave_hull`` method from shapely to GeoSeries/GeoDataFrame (#2903).
- Added ``delaunay_triangles`` method from shapely to GeoSeries/GeoDataFrame (#2907).
- Added ``extract_unique_points`` method from shapely to GeoSeries/GeoDataFrame (#2915).
- Added ``frechet_distance()`` method from shapely to GeoSeries/GeoDataFrame (#2929).
- Added ``hausdorff_distance`` method from shapely to GeoSeries/GeoDataFrame (#2909).
- Added ``minimum_rotated_rectangle`` method from shapely to GeoSeries/GeoDataFrame (#2541).
- Added ``offset_curve`` method from shapely to GeoSeries/GeoDataFrame (#2902).
- Added ``remove_repeated_points`` method from shapely to GeoSeries/GeoDataFrame (#2940).
- Added ``reverse`` method from shapely to GeoSeries/GeoDataFrame (#2988).
- Added ``segmentize`` method from shapely to GeoSeries/GeoDataFrame (#2910).
- Added ``shortest_line`` method from shapely to GeoSeries/GeoDataFrame (#2960).

New features and improvements:

- Added ``exclusive`` parameter to ``sjoin_nearest`` method for Shapely >= 2.0 (#2877)
- The ``to_file()`` method will now automatically detect the FlatGeoBuf driver
  for files with the `.fgb` extension (#2958)

Bug fixes:

- Fix ambiguous error when GeoDataFrame is initialized with a column called ``"crs"`` (#2944)
- Fix a color assignment in ``explore`` when using ``UserDefined`` bins (#2923)
- Fix bug in `apply` with `axis=1` where the given user defined function returns nested
  data in the geometry column (#2959)
- Properly infer schema for ``np.int32`` and ``pd.Int32Dtype`` columns (#2950)
- ``assert_geodataframe_equal`` now handles GeoDataFrames with no active geometry (#2498)

Notes on (optional) dependencies:

- GeoPandas 0.14 drops support for Python 3.8 and pandas 1.3 and below (the minimum
  supported pandas version is now 1.4). Further, the minimum required versions for the
  listed dependencies have now changed to shapely 1.8.0, fiona 1.8.21, pyproj 3.3.0 and
  matplotlib 3.5.0 (#3001)

Deprecations and compatibility notes:

- `geom_almost_equals()` methods have been deprecated and
   `geom_equals_exact()` should be used instead (#2604).

## Version 0.13.2 (Jun 6, 2023)

Bug fix:

- Fix a regression in reading from local file URIs (``file://..``) using
  ``geopandas.read_file`` (#2948).

## Version 0.13.1 (Jun 5, 2023)

Bug fix:

- Fix a regression in reading from URLs using ``geopandas.read_file`` (#2908). This
  restores the behaviour to download all data up-front before passing it to the
  underlying engine (fiona or pyogrio), except if the server supports partial requests
  (to support reading a subset of a large file).

## Version 0.13 (May 6, 2023)

New methods:

- Added ``sample_points`` method to sample random points from Polygon or LineString
  geometries (#2860).
- New ``hilbert_distance()`` method that calculates the distance along a Hilbert curve
  for each geometry in a GeoSeries/GeoDataFrame (#2297).
- Support for sorting geometries (for example, using ``sort_values()``) based on
  the distance along the Hilbert curve (#2070).
- Added ``get_coordinates()`` method from shapely to GeoSeries/GeoDataFrame (#2624).
- Added ``minimum_bounding_circle()`` method from shapely to GeoSeries/GeoDataFrame (#2621).
- Added `minimum_bounding_radius()` as GeoSeries method (#2827).

Other new features and improvements:

- The Parquet and Feather IO functions now support the latest 1.0.0-beta.1 version
  of the GeoParquet specification (<geoparquet.org>) (#2663).
- Added support to fill missing values in `GeoSeries.fillna` via another `GeoSeries` (#2535).
- Support specifying ``min_zoom`` and ``max_zoom`` inside the ``map_kwds`` argument for ``.explore()`` (#2599).
- Added support for append (``mode="a"`` or ``append=True``) in ``to_file()``
  using ``engine="pyogrio"`` (#2788).
- Added a ``to_wgs84`` keyword to ``to_json`` allowing automatic re-projecting to follow
  the 2016 GeoJSON specification (#416).
- ``to_json`` output now includes a ``"crs"`` field if the CRS is not the default WGS84 (#1774).
- Improve error messages when accessing the `geometry` attribute of GeoDataFrame without an active geometry column
  related to the default name `"geometry"` being provided in the constructor (#2577)

Deprecations and compatibility notes:

- Added warning that ``unary_union`` will return ``'GEOMETRYCOLLECTION EMPTY'`` instead
  of None for all-None GeoSeries. (#2618)
- The ``query_bulk()`` method of the spatial index `.sindex` property is deprecated
  in favor of ``query()`` (#2823).

Bug fixes:

- Ensure that GeoDataFrame created from DataFrame is a copy, not a view (#2667)
- Fix mismatch between geometries and colors in ``plot()`` if an empty or missing
  geometry is present (#2224)
- Escape special characters to avoid TemplateSyntaxError in ``explore()`` (#2657)
- Fix `to_parquet`/`to_feather` to not write an invalid bbox (with NaNs) in the
  metadata in case of an empty GeoDataFrame (#2653)
- Fix `to_parquet`/`to_feather` to use correct WKB flavor for 3D geometries (#2654)
- Fix `read_file` to avoid reading all file bytes prior to calling Fiona or
  Pyogrio if provided a URL as input (#2796)
- Fix `copy()` downcasting GeoDataFrames without an active geometry column to a
  DataFrame (#2775)
- Fix geometry column name propagation when GeoDataFrame columns are a multiindex (#2088)
- Fix `iterfeatures()` method of GeoDataFrame to correctly handle non-scalar values
  when `na='drop'` is specified (#2811)
- Fix issue with passing custom legend labels to `plot` (#2886)

Notes on (optional) dependencies:

- GeoPandas 0.13 drops support pandas 1.0.5 (the minimum supported
  pandas version is now 1.1). Further, the minimum required versions for the listed
  dependencies have now changed to shapely 1.7.1, fiona 1.8.19, pyproj 3.0.1 and
  matplotlib 3.3.4 (#2655)

## Version 0.12.2 (December 10, 2022)

Bug fixes:

- Correctly handle geometries with Z dimension in ``to_crs()`` when using PyGEOS or
  Shapely >= 2.0 (previously the z coordinates were lost) (#1345).
- Assign Crimea to Ukraine in the ``naturalearth_lowres`` built-in dataset (#2670)

## Version 0.12.1 (October 29, 2022)

Small bug-fix release removing the shapely<2 pin in the installation requirements.

## Version 0.12 (October 24, 2022)

The highlight of this release is the support for Shapely 2.0. This makes it possible to
test Shapely 2.0 (currently 2.0b1) alongside GeoPandas.

Note that if you also have PyGEOS installed, you need to set an environment variable
(`USE_PYGEOS=0`) before importing geopandas to actually test Shapely 2.0 features instead of PyGEOS. See
<https://geopandas.org/en/latest/getting_started/install.html#using-the-optional-pygeos-dependency>
for more details.

New features and improvements:

- Added ``normalize()`` method from shapely to GeoSeries/GeoDataFrame (#2537).
- Added ``make_valid()`` method from shapely to GeoSeries/GeoDataFrame (#2539).
- Added ``where`` filter to ``read_file`` (#2552).
- Updated the distributed natural earth datasets (*naturalearth_lowres* and
  *naturalearth_cities*) to version 5.1 (#2555).

Deprecations and compatibility notes:

- Accessing the `crs` of a `GeoDataFrame` without active geometry column was deprecated
  and this now raises an AttributeError (#2578).
- Resolved colormap-related warning in ``.explore()`` for recent Matplotlib versions
  (#2596).

Bug fixes:

- Fix cryptic error message in ``geopandas.clip()`` when clipping with an empty geometry (#2589).
- Accessing `gdf.geometry` where the active geometry column is missing, and a column
  named `"geometry"` is present will now raise an `AttributeError`, rather than
  returning `gdf["geometry"]` (#2575).
- Combining GeoSeries/GeoDataFrames with ``pandas.concat`` will no longer silently
  override CRS information if not all inputs have the same CRS (#2056).

## Version 0.11.1 (July 24, 2022)

Small bug-fix release:

- Fix regression (RecursionError) in reshape methods such as ``unstack()``
  and ``pivot()`` involving MultiIndex, or GeoDataFrame construction with
  MultiIndex (#2486).
- Fix regression in ``GeoDataFrame.explode()`` with non-default
  geometry column name.
- Fix regression in ``apply()`` causing row-wise all nan float columns to be
  casted to GeometryDtype (#2482).
- Fix a crash in datetime column reading where the file contains mixed timezone
  offsets (#2479). These will be read as UTC localized values.
- Fix a crash in datetime column reading where the file contains datetimes
  outside the range supported by [ns] precision (#2505).
- Fix regression in passing the Parquet or Feather format ``version`` in
  ``to_parquet`` and ``to_feather``. As a result, the ``version`` parameter
  for the ``to_parquet`` and ``to_feather`` methods has been replaced with
  ``schema_version``. ``version`` will be passed directly to underlying
  feather or parquet writer. ``version`` will only be used to set
  ``schema_version`` if ``version`` is one of 0.1.0 or 0.4.0 (#2496).

Version 0.11 (June 20, 2022)
----------------------------

Highlights of this release:

- The ``geopandas.read_file()`` and `GeoDataFrame.to_file()` methods to read
  and write GIS file formats can now optionally use the
  [pyogrio](https://github.com/geopandas/pyogrio/) package under the hood
  through the ``engine="pyogrio"`` keyword. The pyogrio package implements
  vectorized IO for GDAL/OGR vector data sources, and is faster compared to
  the ``fiona``-based engine (#2225).
- GeoParquet support updated to implement
  [v0.4.0](https://github.com/opengeospatial/geoparquet/releases/tag/v0.4.0) of the
  OpenGeospatial/GeoParquet specification (#2441). Backwards compatibility with v0.1.0 of
  the metadata spec (implemented in the previous releases of GeoPandas) is guaranteed,
  and reading and writing Parquet and Feather files will no longer produce a ``UserWarning``
  (#2327).

New features and improvements:

- Improved handling of GeoDataFrame when the active geometry column is
  lost from the GeoDataFrame. Previously, square bracket indexing ``gdf[[...]]`` returned
  a GeoDataFrame when the active geometry column was retained and a DataFrame was
  returned otherwise. Other pandas indexing methods (``loc``, ``iloc``, etc) did not follow
  the same rules. The new behaviour for all indexing/reshaping operations is now as
  follows (#2329, #2060):
  - If operations produce a ``DataFrame`` containing the active geometry column, a
    GeoDataFrame is returned
  - If operations produce a ``DataFrame`` containing ``GeometryDtype`` columns, but not the
    active geometry column, a ``GeoDataFrame`` is returned, where the active geometry
    column is set to ``None`` (set the new geometry column with ``set_geometry()``)
  - If operations produce a ``DataFrame`` containing no ``GeometryDtype`` columns, a
    ``DataFrame`` is returned (this can be upcast again by calling ``set_geometry()`` or the
    ``GeoDataFrame`` constructor)
  - If operations produce a ``Series`` of ``GeometryDtype``, a ``GeoSeries`` is returned,
    otherwise ``Series`` is returned.
  - Error messages for having an invalid geometry column
    have been improved, indicating the name of the last valid active geometry column set
    and whether other geometry columns can be promoted to the active geometry column
    (#2329).

- Datetime fields are now read and written correctly for GIS formats which support them
  (e.g. GPKG, GeoJSON) with fiona 1.8.14 or higher. Previously, datetimes were read as
  strings (#2202).
- ``folium.Map`` keyword arguments can now be specified as the ``map_kwds`` argument to
  ``GeoDataFrame.explore()`` method (#2315).
- Add a new parameter ``style_function`` to ``GeoDataFrame.explore()`` to enable plot styling
  based on GeoJSON properties (#2377).
- It is now possible to write an empty ``GeoDataFrame`` to a file for supported formats
  (#2240). Attempting to do so will now emit a ``UserWarning`` instead of a ``ValueError``.
- Fast rectangle clipping has been exposed as ``GeoSeries/GeoDataFrame.clip_by_rect()``
  (#1928).
- The ``mask`` parameter of ``GeoSeries/GeoDataFrame.clip()`` now accepts a rectangular mask
  as a list-like to perform fast rectangle clipping using the new
  ``GeoSeries/GeoDataFrame.clip_by_rect()`` (#2414).
- Bundled demo dataset ``naturalearth_lowres`` has been updated to version 5.0.1 of the
  source, with field ``ISO_A3`` manually corrected for some cases (#2418).

Deprecations and compatibility notes:

- The active development branch of geopandas on GitHub has been renamed from master to
  main (#2277).
- Deprecated methods ``GeometryArray.equals_exact()`` and ``GeometryArray.almost_equals()``
  have been removed. They should
  be replaced with ``GeometryArray.geom_equals_exact()`` and
  ``GeometryArray.geom_almost_equals()`` respectively (#2267).
- Deprecated CRS functions ``explicit_crs_from_epsg()``, ``epsg_from_crs()`` and
  ``get_epsg_file_contents()`` were removed (#2340).
- Warning about the behaviour change to ``GeoSeries.isna()`` with empty
  geometries present has been removed (#2349).
- Specifying a CRS in the ``GeoDataFrame/GeoSeries`` constructor which contradicted the
  underlying ``GeometryArray`` now raises a ``ValueError`` (#2100).
- Specifying a CRS in the ``GeoDataFrame`` constructor when no geometry column is provided
  and calling ``GeoDataFrame. set_crs`` on a ``GeoDataFrame`` without an active geometry
  column now raise a ``ValueError`` (#2100)
- Passing non-geometry data to the``GeoSeries`` constructor is now fully deprecated and
  will raise a ``TypeError`` (#2314). Previously, a ``pandas.Series`` was returned for
  non-geometry data.
- Deprecated ``GeoSeries/GeoDataFrame`` set operations ``__xor__()``,
  ``__or__()``, ``__and__()`` and ``__sub__()``, ``geopandas.io.file.read_file``/``to_file`` and
  ``geopandas.io.sql.read_postgis`` now emit ``FutureWarning`` instead of
  ``DeprecationWarning`` and will be completely removed in a future release.
- Accessing the ``crs`` of a ``GeoDataFrame`` without active geometry column is deprecated and will be removed in GeoPandas 0.12 (#2373).

Bug fixes:

- ``GeoSeries.to_frame`` now creates a ``GeoDataFrame`` with the geometry column name set
  correctly (#2296)
- Fix pickle files created with pygeos installed can not being readable when pygeos is
  not installed (#2237).
- Fixed ``UnboundLocalError`` in ``GeoDataFrame.plot()`` using ``legend=True`` and
  ``missing_kwds`` (#2281).
- Fix ``explode()`` incorrectly relating index to columns, including where the input index
  is not unique (#2292)
- Fix ``GeoSeries.[xyz]`` raising an ``IndexError`` when the underlying GeoSeries contains
  empty points (#2335). Rows corresponding to empty points now contain ``np.nan``.
- Fix ``GeoDataFrame.iloc`` raising a ``TypeError`` when indexing a ``GeoDataFrame`` with only
  a single column of ``GeometryDtype`` (#1970).
- Fix ``GeoDataFrame.iterfeatures()`` not returning features with the same field order as
  ``GeoDataFrame.columns`` (#2396).
- Fix ``GeoDataFrame.from_features()`` to support reading GeoJSON with null properties
  (#2243).
- Fix ``GeoDataFrame.to_parquet()`` not intercepting ``engine`` keyword argument, breaking
  consistency with pandas (#2227)
- Fix ``GeoDataFrame.explore()`` producing an error when ``column`` is of boolean dtype
  (#2403).
- Fix an issue where ``GeoDataFrame.to_postgis()`` output the wrong SRID for ESRI
  authority CRS (#2414).
- Fix ``GeoDataFrame.from_dict/from_features`` classmethods using ``GeoDataFrame`` rather
  than ``cls`` as the constructor.
- Fix ``GeoDataFrame.plot()`` producing incorrect colors with mixed geometry types when
  ``colors`` keyword is provided. (#2420)

Notes on (optional) dependencies:

- GeoPandas 0.11 drops support for Python 3.7 and pandas 0.25 (the minimum supported
  pandas version is now 1.0.5). Further, the minimum required versions for the listed
  dependencies have now changed to shapely 1.7, fiona 1.8.13.post1, pyproj 2.6.1.post1,
  matplotlib 3.2, mapclassify 2.4.0 (#2358, #2391)

Version 0.10.2 (October 16, 2021)
---------------------------------

Small bug-fix release:

- Fix regression in ``overlay()`` in case no geometries are intersecting (but
  have overlapping total bounds) (#2172).
- Fix regression in ``overlay()`` with ``keep_geom_type=True`` in case the
  overlay of two geometries in a GeometryCollection with other geometry types
  (#2177).
- Fix ``overlay()`` to honor the ``keep_geom_type`` keyword for the
  ``op="difference"`` case (#2164).
- Fix regression in ``plot()`` with a mapclassify ``scheme`` in case the
  formatted legend labels have duplicates (#2166).
- Fix a bug in the ``explore()`` method ignoring the ``vmin`` and ``vmax`` keywords
  in case they are set to 0 (#2175).
- Fix ``unary_union`` to correctly handle a GeoSeries with missing values (#2181).
- Avoid internal deprecation warning in ``clip()`` (#2179).

Version 0.10.1 (October 8, 2021)
--------------------------------

Small bug-fix release:

- Fix regression in ``overlay()`` with non-overlapping geometries and a
  non-default ``how`` (i.e. not "intersection") (#2157).

Version 0.10.0 (October 3, 2021)
--------------------------------

Highlights of this release:

- A new ``sjoin_nearest()`` method to join based on proximity, with the
  ability to set a maximum search radius (#1865). In addition, the ``sindex``
  attribute gained a new method for a "nearest" spatial index query (#1865,
  #2053).
- A new ``explore()`` method on GeoDataFrame and GeoSeries with native support
  for interactive visualization based on folium / leaflet.js (#1953)
- The ``geopandas.sjoin()``/``overlay()``/``clip()`` functions are now also
  available as methods on the GeoDataFrame (#2141, #1984, #2150).

New features and improvements:

- Add support for pandas' ``value_counts()`` method for geometry dtype (#2047).
- The ``explode()`` method has a new ``ignore_index`` keyword (consistent with
  pandas' explode method) to reset the index in the result, and a new
  ``index_parts`` keywords to control whether a cumulative count indexing the
  parts of the exploded multi-geometries should be added (#1871).
- ``points_from_xy()`` is now available as a GeoSeries method ``from_xy`` (#1936).
- The ``to_file()`` method will now attempt to detect the driver (if not
  specified) based on the extension of the provided filename, instead of
  defaulting to ESRI Shapefile (#1609).
- Support for the ``storage_options`` keyword in ``read_parquet()`` for
  specifying filesystem-specific options (e.g. for S3) based on fsspec (#2107).
- The read/write functions now support ``~`` (user home directory) expansion (#1876).
- Support the ``convert_dtypes()`` method from pandas to preserve the
  GeoDataFrame class (#2115).
- Support WKB values in the hex format in ``GeoSeries.from_wkb()`` (#2106).
- Update the ``estimate_utm_crs()`` method to handle crossing the antimeridian
  with pyproj 3.1+ (#2049).
- Improved heuristic to decide how many decimals to show in the repr based on
  whether the CRS is projected or geographic (#1895).
- Switched the default for ``geocode()`` from GeoCode.Farm to the Photon
  geocoding API (<https://photon.komoot.io>) (#2007).

Deprecations and compatibility notes:

- The ``op=`` keyword of ``sjoin()`` to indicate which spatial predicate to use
  for joining is being deprecated and renamed in favor of a new ``predicate=``
  keyword (#1626).
- The ``cascaded_union`` attribute is deprecated, use ``unary_union`` instead (#2074).
- Constructing a GeoDataFrame with a duplicated "geometry" column is now
  disallowed. This can also raise an error in the ``pd.concat(.., axis=1)``
  function if this results in duplicated active geometry columns (#2046).
- The ``explode()`` method currently returns a GeoSeries/GeoDataFrame with a
  MultiIndex, with an additional level with indices of the parts of the
  exploded multi-geometries. For consistency with pandas, this will change in
  the future and the new ``index_parts`` keyword is added to control this.

Bug fixes:

- Fix in the ``clip()`` function to correctly clip MultiPoints instead of
  leaving them intact when partly outside of the clip bounds (#2148).
- Fix ``GeoSeries.isna()`` to correctly return a boolean Series in case of an
  empty GeoSeries (#2073).
- Fix the GeoDataFrame constructor to preserve the geometry name when the
  argument is already a GeoDataFrame object (i.e. ``GeoDataFrame(gdf)``) (#2138).
- Fix loss of the values' CRS when setting those values as a column
  (``GeoDataFrame.__setitem__``) (#1963)
- Fix in ``GeoDataFrame.apply()`` to preserve the active geometry column name
  (#1955).
- Fix in ``sjoin()`` to not ignore the suffixes in case of a right-join
  (``how="right``) (#2065).
- Fix ``GeoDataFrame.explode()`` with a MultiIndex (#1945).
- Fix the handling of missing values in ``to/from_wkb`` and ``to_from_wkt`` (#1891).
- Fix ``to_file()`` and ``to_json()`` when DataFrame has duplicate columns to
  raise an error (#1900).
- Fix bug in the colors shown with user-defined classification scheme (#2019).
- Fix handling of the ``path_effects`` keyword in ``plot()`` (#2127).
- Fix ``GeoDataFrame.explode()`` to preserve ``attrs`` (#1935)

Notes on (optional) dependencies:

- GeoPandas 0.10.0 dropped support for Python 3.6 and pandas 0.24. Further,
  the minimum required versions are numpy 1.18, shapely 1.6, fiona 1.8,
  matplotlib 3.1 and pyproj 2.2.
- Plotting with a classification schema now requires mapclassify version >=
  2.4 (#1737).
- Compatibility fixes for the latest numpy in combination with Shapely 1.7 (#2072)
- Compatibility fixes for the upcoming Shapely 1.8 (#2087).
- Compatibility fixes for the latest PyGEOS (#1872, #2014) and matplotlib
  (colorbar issue, #2066).

Version 0.9.0 (February 28, 2021)
---------------------------------

Many documentation improvements and a restyled and restructured website with
a new logo (#1564, #1579, #1617, #1668, #1731, #1750, #1757, #1759).

New features and improvements:

- The ``geopandas.read_file`` function now accepts more general
  file-like objects (e.g. ``fsspec`` open file objects). It will now also
  automatically recognize zipped files (#1535).
- The ``GeoDataFrame.plot()`` method now provides access to the pandas plotting
  functionality for the non-geometry columns, either using the ``kind`` keyword
  or the accessor method (e.g. ``gdf.plot(kind="bar")`` or ``gdf.plot.bar()``)
  (#1465).
- New ``from_wkt()``, ``from_wkb()``, ``to_wkt()``, ``to_wkb()`` methods for
  GeoSeries to construct a GeoSeries from geometries in WKT or WKB
  representation, or to convert a GeoSeries to a pandas Seriew with WKT or WKB
  values (#1710).
- New ``GeoSeries.z`` attribute to access the z-coordinates of Point geometries
  (similar to the existing ``.x`` and ``.y`` attributes) (#1773).
- The ``to_crs()`` method now handles missing values (#1618).
- Support for pandas' new ``.attrs`` functionality (#1658).
- The ``dissolve()`` method now allows dissolving by no column (``by=None``) to
  create a union of all geometries (single-row GeoDataFrame) (#1568).
- New ``estimate_utm_crs()`` method on GeoSeries/GeoDataFrame to determine the
  UTM CRS based on the bounds (#1646).
- ``GeoDataFrame.from_dict()`` now accepts ``geometry`` and ``crs`` keywords
  (#1619).
- ``GeoDataFrame.to_postgis()`` and ``geopandas.read_postgis()`` now supports
  both sqlalchemy engine and connection objects (#1638).
- The ``GeoDataFrame.explode()`` method now allows exploding based on a
  non-geometry column, using the pandas implementation (#1720).
- Performance improvement in ``GeoDataFrame/GeoSeries.explode()`` when using
  the PyGEOS backend (#1693).
- The binary operation and predicate methods (eg ``intersection()``,
  ``intersects()``) have a new ``align`` keyword which allows optionally not
  aligning on the index before performing the operation with ``align=False``
  (#1668).
- The ``GeoDataFrame.dissolve()`` method now supports all relevant keywords of
  ``groupby()``, i.e. the ``level``, ``sort``, ``observed`` and ``dropna`` keywords
  (#1845).
- The ``geopandas.overlay()`` function now accepts ``make_valid=False`` to skip
  the step to ensure the input geometries are valid using ``buffer(0)`` (#1802).
- The ``GeoDataFrame.to_json()`` method gained a ``drop_id`` keyword to
  optionally not write the GeoDataFrame's index as the "id" field in the
  resulting JSON (#1637).
- A new ``aspect`` keyword in the plotting methods to optionally allow retaining
  the original aspect (#1512)
- A new ``interval`` keyword in the ``legend_kwds`` group of the ``plot()`` method
  to control the appearance of the legend labels when using a classification
  scheme (#1605).
- The spatial index of a GeoSeries (accessed with the ``sindex`` attribute) is
  now stored on the underlying array. This ensures that the spatial index is
  preserved in more operations where possible, and that multiple geometry
  columns of a GeoDataFrame can each have a spatial index (#1444).
- Addition of a ``has_sindex`` attribute on the GeoSeries/GeoDataFrame to check
  if a spatial index has already been initialized (#1627).
- The ``geopandas.testing.assert_geoseries_equal()`` and ``assert_geodataframe_equal()``
  testing utilities now have a ``normalize`` keyword (False by default) to
  normalize geometries before comparing for equality (#1826). Those functions
  now also give a more informative error message when failing (#1808).

Deprecations and compatibility notes:

- The ``is_ring`` attribute currently returns True for Polygons. In the future,
  this will be False (#1631). In addition, start to check it for LineStrings
  and LinearRings (instead of always returning False).
- The deprecated ``objects`` keyword in the ``intersection()`` method of the
  ``GeoDataFrame/GeoSeries.sindex`` spatial index object has been removed
  (#1444).

Bug fixes:

- Fix regression in the ``plot()`` method raising an error with empty
  geometries (#1702, #1828).
- Fix ``geopandas.overlay()`` to preserve geometries of the correct type which
  are nested within a GeometryCollection as a result of the overlay
  operation (#1582). In addition, a warning will now be raised if geometries
  of different type are dropped from the result (#1554).
- Fix the repr of an empty GeoSeries to not show spurious warnings (#1673).
- Fix the ``.crs`` for empty GeoDataFrames (#1560).
- Fix ``geopandas.clip`` to preserve the correct geometry column name (#1566).
- Fix bug in ``plot()`` method when using ``legend_kwds`` with multiple subplots
  (#1583)
- Fix spurious warning with ``missing_kwds`` keyword of the ``plot()`` method
  when there are no areas with missing data (#1600).
- Fix the ``plot()`` method to correctly align values passed to the ``column``
  keyword as a pandas Series (#1670).
- Fix bug in plotting MultiPoints when passing values to determine the color
  (#1694)
- The ``rename_geometry()`` method now raises a more informative error message
  when a duplicate column name is used (#1602).
- Fix ``explode()`` method to preserve the CRS (#1655)
- Fix the ``GeoSeries.apply()`` method to again accept the ``convert_dtype``
  keyword to be consistent with pandas (#1636).
- Fix ``GeoDataFrame.apply()`` to preserve the CRS when possible (#1848).
- Fix bug in containment test as ``geom in geoseries`` (#1753).
- The ``shift()`` method of a GeoSeries/GeoDataFrame now preserves the CRS
  (#1744).
- The PostGIS IO functionality now quotes table names to ensure it works with
  case-sensitive names (#1825).
- Fix the ``GeoSeries`` constructor without passing data but only an index (#1798).

Notes on (optional) dependencies:

- GeoPandas 0.9.0 dropped support for Python 3.5. Further, the minimum
  required versions are pandas 0.24, numpy 1.15 and shapely 1.6 and fiona 1.8.
- The ``descartes`` package is no longer required for plotting polygons. This
  functionality is now included by default in GeoPandas itself, when
  matplotlib is available (#1677).
- Fiona is now only imported when used in ``read_file``/``to_file``. This means
  you can now force geopandas to install without fiona installed (although it
  is still a default requirement) (#1775).
- Compatibility with the upcoming Shapely 1.8 (#1659, #1662, #1819).

Version 0.8.2 (January 25, 2021)
--------------------------------

Small bug-fix release for compatibility with PyGEOS 0.9.

Version 0.8.1 (July 15, 2020)
-----------------------------

Small bug-fix release:

- Fix a regression in the ``plot()`` method when visualizing with a
  JenksCaspallSampled or FisherJenksSampled scheme (#1486).
- Fix spurious warning in ``GeoDataFrame.to_postgis`` (#1497).
- Fix the un-pickling with ``pd.read_pickle`` of files written with older
  GeoPandas versions (#1511).

Version 0.8.0 (June 24, 2020)
-----------------------------

**Experimental**: optional use of PyGEOS to speed up spatial operations (#1155).
PyGEOS is a faster alternative for Shapely (being contributed back to a future
version of Shapely), and is used in element-wise spatial operations and for
spatial index in e.g. ``sjoin`` (#1343, #1401, #1421, #1427, #1428). See the
[installation docs](https://geopandas.readthedocs.io/en/latest/install.html#using-the-optional-pygeos-dependency)
for more info and how to enable it.

New features and improvements:

- IO enhancements:

  - New ``GeoDataFrame.to_postgis()`` method to write to PostGIS database (#1248).
  - New Apache Parquet and Feather file format support (#1180, #1435)
  - Allow appending to files with ``GeoDataFrame.to_file`` (#1229).
  - Add support for the ``ignore_geometry`` keyword in ``read_file`` to only read
    the attribute data. If set to True, a pandas DataFrame without geometry is
    returned (#1383).
  - ``geopandas.read_file`` now supports reading from file-like objects (#1329).
  - ``GeoDataFrame.to_file`` now supports specifying the CRS to write to the file
    (#802). By default it still uses the CRS of the GeoDataFrame.
  - New ``chunksize`` keyword in ``geopandas.read_postgis`` to read a query in
    chunks (#1123).

- Improvements related to geometry columns and CRS:

  - Any column of the GeoDataFrame that has a "geometry" dtype is now returned
    as a GeoSeries. This means that when having multiple geometry columns, not
    only the "active" geometry column is returned as a GeoSeries, but also
    accessing another geometry column (``gdf["other_geom_column"]``) gives a
    GeoSeries (#1336).
  - Multiple geometry columns in a GeoDataFrame can now each have a different
    CRS. The global ``gdf.crs`` attribute continues to returns the CRS of the
    "active" geometry column. The CRS of other geometry columns can be accessed
    from the column itself (eg ``gdf["other_geom_column"].crs``) (#1339).
  - New ``set_crs()`` method on GeoDataFrame/GeoSeries to set the CRS of naive
    geometries (#747).

- Improvements related to plotting:

  - The y-axis is now scaled depending on the center of the plot when using a
    geographic CRS, instead of using an equal aspect ratio (#1290).
  - When passing a column of categorical dtype to the ``column=`` keyword of the
    GeoDataFrame ``plot()``, we now honor all categories and its order (#1483).
    In addition, a new ``categories`` keyword allows to specify all categories
    and their order otherwise (#1173).
  - For choropleths using a classification scheme (using ``scheme=``), the
    ``legend_kwds`` accept two new keywords to control the formatting of the
    legend: ``fmt`` with a format string for the bin edges (#1253), and ``labels``
    to pass fully custom class labels (#1302).

- New ``covers()`` and ``covered_by()`` methods on GeoSeries/GeoDataFrame for the
  equivalent spatial predicates (#1460, #1462).
- GeoPandas now warns when using distance-based methods with data in a
  geographic projection (#1378).

Deprecations:

- When constructing a GeoSeries or GeoDataFrame from data that already has a
  CRS, a deprecation warning is raised when both CRS don't match, and in the
  future an error will be raised in such a case. You can use the new ``set_crs``
  method to override an existing CRS. See
  [the docs](https://geopandas.readthedocs.io/en/latest/projections.html#projection-for-multiple-geometry-columns).
- The helper functions in the ``geopandas.plotting`` module are deprecated for
  public usage (#656).
- The ``geopandas.io`` functions are deprecated, use the top-level ``read_file`` and
  ``to_file`` instead (#1407).
- The set operators (``&``, ``|``, ``^``, ``-``) are deprecated, use the
  ``intersection()``, ``union()``, ``symmetric_difference()``, ``difference()`` methods
  instead (#1255).
- The ``sindex`` for empty dataframe will in the future return an empty spatial
  index instead of ``None`` (#1438).
- The ``objects`` keyword in the ``intersection`` method of the spatial index
  returned by the ``sindex`` attribute is deprecated and will be removed in the
  future (#1440).

Bug fixes:

- Fix the ``total_bounds()`` method to ignore missing and empty geometries (#1312).
- Fix ``geopandas.clip`` when masking with non-overlapping area resulting in an
  empty GeoDataFrame (#1309, #1365).
- Fix error in ``geopandas.sjoin`` when joining on an empty geometry column (#1318).
- CRS related fixes: ``pandas.concat`` preserves CRS when concatenating GeoSeries
  objects (#1340), preserve the CRS in ``geopandas.clip`` (#1362) and in
  ``GeoDataFrame.astype`` (#1366).
- Fix bug in ``GeoDataFrame.explode()`` when 'level_1' is one of the column names
  (#1445).
- Better error message when rtree is not installed (#1425).
- Fix bug in ``GeoSeries.equals()`` (#1451).
- Fix plotting of multi-part geometries with additional style keywords (#1385).

And we now have a [Code of Conduct](https://github.com/geopandas/geopandas/blob/main/CODE_OF_CONDUCT.md)!

GeoPandas 0.8.0 is the last release to support Python 3.5. The next release
will require Python 3.6, pandas 0.24, numpy 1.15 and shapely 1.6 or higher.

Version 0.7.0 (February 16, 2020)
---------------------------------

Support for Python 2.7 has been dropped. GeoPandas now works with Python >= 3.5.

The important API change of this release is that GeoPandas now requires
PROJ > 6 and pyproj > 2.2, and that the ``.crs`` attribute of a GeoSeries and
GeoDataFrame no longer stores the CRS information as a proj4 string or dict,
but as a ``pyproj.CRS`` object (#1101).

This gives a better user interface and integrates improvements from pyproj and
PROJ 6, but might also require some changes in your code. Check the
[migration guide](https://geopandas.readthedocs.io/en/latest/projections.html#upgrading-to-geopandas-0-7-with-pyproj-2-2-and-proj-6)
in the documentation.

Other API changes;

- The ``GeoDataFrame.to_file`` method will now also write the GeoDataFrame index
  to the file, if the index is named and/or non-integer. You can use the
  ``index=True/False`` keyword to overwrite this default inference (#1059).

New features and improvements:

- A new ``geopandas.clip`` function to clip a GeoDataFrame to the spatial extent
  of another shape (#1128).
- The ``geopandas.overlay`` function now works for all geometry types, including
  points and linestrings in addition to polygons (#1110).
- The ``plot()`` method gained support for missing values (in the column that
  determines the colors). By default it doesn't plot the corresponding
  geometries, but using the new ``missing_kwds`` argument you can specify how to
  style those geometries (#1156).
- The ``plot()`` method now also supports plotting GeometryCollection and
  LinearRing objects (#1225).
- Added support for filtering with a geometry or reading a subset of the rows in
  ``geopandas.read_file`` (#1160).
- Added support for the new nullable integer data type of pandas in
  ``GeoDataFrame.to_file`` (#1220).

Bug fixes:

- ``GeoSeries.reset_index()`` now correctly results in a GeoDataFrame instead of DataFrame (#1252).
- Fixed the ``geopandas.sjoin`` function to handle MultiIndex correctly (#1159).
- Fixed the ``geopandas.sjoin`` function to preserve the index name of the left GeoDataFrame (#1150).

Version 0.6.3 (February 6, 2020)
---------------------------------

Small bug-fix release:

- Compatibility with Shapely 1.7 and pandas 1.0 (#1244).
- Fix ``GeoDataFrame.fillna`` to accept non-geometry values again when there are
  no missing values in the geometry column. This should make it easier to fill
  the numerical columns of the GeoDataFrame (#1279).

Version 0.6.2 (November 18, 2019)
---------------------------------

Small bug-fix release fixing a few regressions:

- Fix a regression in passing an array of RRB(A) tuples to the ``.plot()``
  method (#1178, #1211).
- Fix the ``bounds`` and ``total_bounds`` attributes for empty GeoSeries, which
  also fixes the repr of an empty or all-NA GeoSeries (#1184, #1195).
- Fix filtering of a GeoDataFrame to preserve the index type when ending up
  with an empty result (#1190).

Version 0.6.1 (October 12, 2019)
--------------------------------

Small bug-fix release fixing a few regressions:

- Fix ``astype`` when converting to string with Multi geometries (#1145) or when converting a dataframe without geometries (#1144).
- Fix ``GeoSeries.fillna`` to accept ``np.nan`` again (#1149).

Version 0.6.0 (September 27, 2019)
----------------------------------

Important note! This will be the last release to support Python 2.7 (#1031)

API changes:

- A refactor of the internals based on the pandas ExtensionArray interface (#1000). The main user visible changes are:

  - The ``.dtype`` of a GeoSeries is now a ``'geometry'`` dtype (and no longer a numpy ``object`` dtype).
  - The ``.values`` of a GeoSeries now returns a custom ``GeometryArray``, and no longer a numpy array. To get back a numpy array of Shapely scalars, you can convert explicitly using ``np.asarray(..)``.

- The ``GeoSeries`` constructor now raises a warning when passed non-geometry data. Currently the constructor falls back to return a pandas ``Series``, but in the future this will raise an error (#1085).
- The missing value handling has been changed to now separate the concepts of missing geometries and empty geometries (#601, 1062). In practice this means that (see [the docs](https://geopandas.readthedocs.io/en/v0.6.0/missing_empty.html) for more details):

  - ``GeoSeries.isna`` now considers only missing values, and if you want to check for empty geometries, you can use ``GeoSeries.is_empty`` (``GeoDataFrame.isna`` already only looked at missing values).
  - ``GeoSeries.dropna`` now actually drops missing values (before it didn't drop either missing or empty geometries)
  - ``GeoSeries.fillna`` only fills missing values (behaviour unchanged).
  - ``GeoSeries.align`` uses missing values instead of empty geometries by default to fill non-matching index entries.

New features and improvements:

- Addition of a ``GeoSeries.affine_transform`` method, equivalent of Shapely's function (#1008).
- Addition of a ``GeoDataFrame.rename_geometry`` method to easily rename the active geometry column (#1053).
- Addition of ``geopandas.show_versions()`` function, which can be used to give an overview of the installed libraries in bug reports (#899).
- The ``legend_kwds`` keyword of the ``plot()`` method can now also be used to specify keywords for the color bar (#1102).
- Performance improvement in the ``sjoin()`` operation by re-using existing spatial index of the input dataframes, if available (#789).
- Updated documentation to work with latest version of geoplot and contextily (#1044, #1088).
- A new ``geopandas.options`` configuration, with currently a single option to control the display precision of the coordinates (``options.display_precision``). The default is now to show less coordinates (3 for projected and 5 for geographic coordinates), but the default can be overridden with the option.

Bug fixes:

- Also try to use ``pysal`` instead of ``mapclassify`` if available (#1082).
- The ``GeoDataFrame.astype()`` method now correctly returns a ``GeoDataFrame`` if the geometry column is preserved (#1009).
- The ``to_crs`` method now uses ``always_xy=True`` to ensure correct lon/lat order handling for pyproj>=2.2.0 (#1122).
- Fixed passing list-like colors in the ``plot()`` method in case of "multi" geometries (#1119).
- Fixed the coloring of shapes and colorbar when passing a custom ``norm`` in the ``plot()`` method (#1091, #1089).
- Fixed ``GeoDataFrame.to_file`` to preserve VFS file paths (e.g. when a "s3://" path is specified) (#1124).
- Fixed failing case in ``geopandas.sjoin`` with empty geometries (#1138).

In addition, the minimum required versions of some dependencies have been increased: GeoPandas now requires pandas >=0.23.4 and matplotlib >=2.0.1 (#1002).

Version 0.5.1 (July 11, 2019)
-----------------------------

- Compatibility with latest mapclassify version 2.1.0 (#1025).

Version 0.5.0 (April 25, 2019)
------------------------------

Improvements:

- Significant performance improvement (around 10x) for ``GeoDataFrame.iterfeatures``,
  which also improves ``GeoDataFrame.to_file`` (#864).
- File IO enhancements based on Fiona 1.8:

  - Support for writing bool dtype (#855) and datetime dtype, if the file format supports it (#728).
  - Support for writing dataframes with multiple geometry types, if the file format allows it (e.g. GeoJSON for all types, or ESRI Shapefile for Polygon+MultiPolygon) (#827, #867, #870).

- Compatibility with pyproj >= 2 (#962).
- A new ``geopandas.points_from_xy()`` helper function to convert x and y coordinates to Point objects (#896).
- The ``buffer`` and ``interpolate`` methods now accept an array-like to specify a variable distance for each geometry (#781).
- Addition of a ``relate`` method, corresponding to the shapely method that returns the DE-9IM matrix (#853).
- Plotting improvements:

  - Performance improvement in plotting by only flattening the geometries if there are actually 'Multi' geometries (#785).
  - Choropleths: access to all ``mapclassify`` classification schemes and addition of the ``classification_kwds`` keyword in the ``plot`` method to specify options for the scheme (#876).
  - Ability to specify a matplotlib axes object on which to plot the color bar with the ``cax`` keyword, in order to have more control over the color bar placement (#894).

- Changed the default provider in ``geopandas.tools.geocode`` from Google (now requires an API key) to Geocode.Farm (#907, #975).

Bug fixes:

- Remove the edge in the legend marker (#807).
- Fix the ``align`` method to preserve the CRS (#829).
- Fix ``geopandas.testing.assert_geodataframe_equal`` to correctly compare left and right dataframes (#810).
- Fix in choropleth mapping when the values contain missing values (#877).
- Better error message in ``sjoin`` if the input is not a GeoDataFrame (#842).
- Fix in ``read_postgis`` to handle nullable (missing) geometries (#856).
- Correctly passing through the ``parse_dates`` keyword in ``read_postgis`` to the underlying pandas method (#860).
- Fixed the shape of Antarctica in the included demo dataset 'naturalearth_lowres'
  (by updating to the latest version) (#804).

Version 0.4.1 (March 5, 2019)
-----------------------------

Small bug-fix release for compatibility with the latest Fiona and PySAL
releases:

- Compatibility with Fiona 1.8: fix deprecation warning (#854).
- Compatibility with PySAL 2.0: switched to ``mapclassify`` instead of ``PySAL`` as
  dependency for choropleth mapping with the ``scheme`` keyword (#872).
- Fix for new ``overlay`` implementation in case the intersection is empty (#800).

Version 0.4.0 (July 15, 2018)
-----------------------------

Improvements:

- Improved ``overlay`` function (better performance, several incorrect behaviours fixed) (#429)
- Pass keywords to control legend behavior (``legend_kwds``) to ``plot`` (#434)
- Add basic support for reading remote datasets in ``read_file`` (#531)
- Pass kwargs for ``buffer`` operation on GeoSeries (#535)
- Expose all geopy services as options in geocoding (#550)
- Faster write speeds to GeoPackage (#605)
- Permit ``read_file`` filtering with a bounding box from a GeoDataFrame (#613)
- Set CRS on GeoDataFrame returned by ``read_postgis`` (#627)
- Permit setting markersize for Point GeoSeries plots with column values (#633)
- Started an example gallery (#463, #690, #717)
- Support for plotting MultiPoints (#683)
- Testing functionality (e.g. ``assert_geodataframe_equal``) is now publicly exposed (#707)
- Add ``explode`` method to GeoDataFrame (similar to the GeoSeries method) (#671)
- Set equal aspect on active axis on multi-axis figures (#718)
- Pass array of values to column argument in ``plot`` (#770)

Bug fixes:

- Ensure that colorbars are plotted on the correct axis (#523)
- Handle plotting empty GeoDataFrame (#571)
- Save z-dimension when writing files (#652)
- Handle reading empty shapefiles (#653)
- Correct dtype for empty result of spatial operations (#685)
- Fix empty ``sjoin`` handling for pandas>=0.23 (#762)

Version 0.3.0 (August 29, 2017)
-------------------------------

Improvements:

- Improve plotting performance using ``matplotlib.collections`` (#267)
- Improve default plotting appearance. The defaults now follow the new matplotlib defaults (#318, #502, #510)
- Provide access to x/y coordinates as attributes for Point GeoSeries (#383)
- Make the NYBB dataset available through ``geopandas.datasets`` (#384)
- Enable ``sjoin`` on non-integer-index GeoDataFrames (#422)
- Add ``cx`` indexer to GeoDataFrame (#482)
- ``GeoDataFrame.from_features`` now also accepts a Feature Collection (#225, #507)
- Use index label instead of integer id in output of ``iterfeatures`` and
  ``to_json`` (#421)
- Return empty data frame rather than raising an error when performing a spatial join with non overlapping geodataframes (#335)

Bug fixes:

- Compatibility with shapely 1.6.0 (#512)
- Fix ``fiona.filter`` results when bbox is not None (#372)
- Fix ``dissolve`` to retain CRS (#389)
- Fix ``cx`` behavior when using index of 0 (#478)
- Fix display of lower bin in legend label of choropleth plots using a PySAL scheme (#450)

Version 0.2.0
-------------

Improvements:

- Complete overhaul of the documentation
- Addition of ``overlay`` to perform spatial overlays with polygons (#142)
- Addition of ``sjoin`` to perform spatial joins (#115, #145, #188)
- Addition of ``__geo_interface__`` that returns a python data structure
  to represent the ``GeoSeries`` as a GeoJSON-like ``FeatureCollection`` (#116)
  and ``iterfeatures`` method (#178)
- Addition of the ``explode`` (#146) and ``dissolve`` (#310, #311) methods.
- Addition of the ``sindex`` attribute, a Spatial Index using the optional
  dependency ``rtree`` (``libspatialindex``) that can be used to speed up
  certain operations such as overlays (#140, #141).
- Addition of the ``GeoSeries.cx`` coordinate indexer to slice a GeoSeries based
  on a bounding box of the coordinates (#55).
- Improvements to plotting: ability to specify edge colors (#173), support for
  the ``vmin``, ``vmax``, ``figsize``, ``linewidth`` keywords (#207), legends
  for chloropleth plots (#210), color points by specifying a colormap (#186) or
  a single color (#238).
- Larger flexibility of ``to_crs``, accepting both dicts and proj strings (#289)
- Addition of embedded example data, accessible through
  ``geopandas.datasets.get_path``.

API changes:

- In the ``plot`` method, the ``axes`` keyword is renamed to ``ax`` for
  consistency with pandas, and the ``colormap`` keyword is renamed to ``cmap``
  for consistency with matplotlib (#208, #228, #240).

Bug fixes:

- Properly handle rows with missing geometries (#139, #193).
- Fix ``GeoSeries.to_json`` (#263).
- Correctly serialize metadata when pickling (#199, #206).
- Fix ``merge`` and ``concat`` to return correct GeoDataFrame (#247, #320, #322).
