Changelog
=========

Version 0.9.0 (February 28, 2021)
---------------------------------

Many documentation improvements and a restyled and restructured website with
a new logo (#1564, #1579, #1617, #1668, #1731, #1750, #1757, #1759).

New features and improvements:

- The `geopandas.read_file` function now accepts more general
  file-like objects (e.g. `fsspec` open file objects). It will now also
  automatically recognize zipped files (#1535).
- The `GeoDataFrame.plot()` method now provides access to the pandas plotting
  functionality for the non-geometry columns, either using the `kind` keyword
  or the accessor method (e.g. `gdf.plot(kind="bar")` or `gdf.plot.bar()`)
  (#1465).
- New `from_wkt()`, `from_wkb()`, `to_wkt()`, `to_wkb()` methods for
  GeoSeries to construct a GeoSeries from geometries in WKT or WKB
  representation, or to convert a GeoSeries to a pandas Seriew with WKT or WKB
  values (#1710).
- New `GeoSeries.z` attribute to access the z-coordinates of Point geometries
  (similar to the existing `.x` and `.y` attributes) (#1773).
- The `to_crs()` method now handles missing values (#1618).
- Support for pandas' new `.attrs` functionality (#1658).
- The `dissolve()` method now allows dissolving by no column (`by=None`) to
  create a union of all geometries (single-row GeoDataFrame) (#1568).
- New `estimate_utm_crs()` method on GeoSeries/GeoDataFrame to determine the
  UTM CRS based on the bounds (#1646).
- `GeoDataFrame.from_dict()` now accepts `geometry` and `crs` keywords
  (#1619).
- `GeoDataFrame.to_postgis()` and `geopandas.read_postgis()` now supports
  both sqlalchemy engine and connection objects (#1638).
- The `GeoDataFrame.explode()` method now allows exploding based on a
  non-geometry column, using the pandas implementation (#1720).
- Performance improvement in `GeoDataFrame/GeoSeries.explode()` when using
  the PyGEOS backend (#1693).
- The binary operation and predicate methods (eg `intersection()`,
  `intersects()`) have a new `align` keyword which allows optionally not
  aligning on the index before performing the operation with `align=False`
  (#1668).
- The `GeoDataFrame.dissolve()` method now supports all relevant keywords of
  `groupby()`, i.e. the `level`, `sort`, `observed` and `dropna` keywords
  (#1845).
- The `geopandas.overlay()` function now accepts `make_valid=False` to skip
  the step to ensure the input geometries are valid using `buffer(0)` (#1802).
- The `GeoDataFrame.to_json()` method gained a `drop_id` keyword to
  optionally not write the GeoDataFrame's index as the "id" field in the
  resulting JSON (#1637).
- A new `aspect` keyword in the plotting methods to optionally allow retaining
  the original aspect (#1512)
- A new `interval` keyword in the `legend_kwds` group of the `plot()` method
  to control the appearance of the legend labels when using a classification
  scheme (#1605).
- The spatial index of a GeoSeries (accessed with the `sindex` attribute) is
  now stored on the underlying array. This ensures that the spatial index is
  preserved in more operations where possible, and that multiple geometry
  columns of a GeoDataFrame can each have a spatial index (#1444).
- Addition of a `has_sindex` attribute on the GeoSeries/GeoDataFrame to check
  if a spatial index has already been initialized (#1627).
- The `geopandas.testing.assert_geoseries_equal()` and `assert_geodataframe_equal()`
  testing utilities now have a `normalize` keyword (False by default) to
  normalize geometries before comparing for equality (#1826). Those functions
  now also give a more informative error message when failing (#1808).

Deprecations and compatibility notes:

- The `is_ring` attribute currently returns True for Polygons. In the future,
  this will be False (#1631). In addition, start to check it for LineStrings
  and LinearRings (instead of always returning False).
- The deprecated `objects` keyword in the `intersection()` method of the
  `GeoDataFrame/GeoSeries.sindex` spatial index object has been removed
  (#1444).

Bug fixes:

- Fix regression in the `plot()` method raising an error with empty
  geometries (#1702, #1828).
- Fix `geopandas.overlay()` to preserve geometries of the correct type which
  are nested within a GeometryCollection as a result of the overlay
  operation (#1582). In addition, a warning will now be raised if geometries
  of different type are dropped from the result (#1554).
- Fix the repr of an empty GeoSeries to not show spurious warnings (#1673).
- Fix the `.crs` for empty GeoDataFrames (#1560).
- Fix `geopandas.clip` to preserve the correct geometry column name (#1566). 
- Fix bug in `plot()` method when using `legend_kwds` with multiple subplots
  (#1583)
- Fix spurious warning with `missing_kwds` keyword of the `plot()` method
  when there are no areas with missing data (#1600).
- Fix the `plot()` method to correctly align values passed to the `column`
  keyword as a pandas Series (#1670).
- Fix bug in plotting MultiPoints when passing values to determine the color
  (#1694)
- The `rename_geometry()` method now raises a more informative error message
  when a duplicate column name is used (#1602).
- Fix `explode()` method to preserve the CRS (#1655)
- Fix the `GeoSeries.apply()` method to again accept the `convert_dtype`
  keyword to be consistent with pandas (#1636).
- Fix `GeoDataFrame.apply()` to preserve the CRS when possible (#1848).
- Fix bug in containment test as `geom in geoseries` (#1753).
- The `shift()` method of a GeoSeries/GeoDataFrame now preserves the CRS
  (#1744).
- The PostGIS IO functionality now quotes table names to ensure it works with
  case-sensitive names (#1825).
- Fix the `GeoSeries` constructor without passing data but only an index (#1798).

Notes on (optional) dependencies:

- GeoPandas 0.9.0 dropped support for Python 3.5. Further, the minimum
  required versions are pandas 0.24, numpy 1.15 and shapely 1.6 and fiona 1.8.
- The `descartes` package is no longer required for plotting polygons. This
  functionality is now included by default in GeoPandas itself, when
  matplotlib is available (#1677).
- Fiona is now only imported when used in `read_file`/`to_file`. This means
  you can now force geopandas to install without fiona installed (although it
  is still a default requirement) (#1775).
- Compatibility with the upcoming Shapely 1.8 (#1659, #1662, #1819).


Version 0.8.2 (January 25, 2021)
--------------------------------

Small bug-fix release for compatibility with PyGEOS 0.9.


Version 0.8.1 (July 15, 2020)
-----------------------------

Small bug-fix release:

- Fix a regression in the `plot()` method when visualizing with a
  JenksCaspallSampled or FisherJenksSampled scheme (#1486).
- Fix spurious warning in `GeoDataFrame.to_postgis` (#1497).
- Fix the un-pickling with `pd.read_pickle` of files written with older
  GeoPandas versions (#1511).


Version 0.8.0 (June 24, 2020)
-----------------------------

**Experimental**: optional use of PyGEOS to speed up spatial operations (#1155).
PyGEOS is a faster alternative for Shapely (being contributed back to a future
version of Shapely), and is used in element-wise spatial operations and for
spatial index in e.g. `sjoin` (#1343, #1401, #1421, #1427, #1428). See the
[installation docs](https://geopandas.readthedocs.io/en/latest/install.html#using-the-optional-pygeos-dependency)
for more info and how to enable it.

New features and improvements:

- IO enhancements:
  - New `GeoDataFrame.to_postgis()` method to write to PostGIS database (#1248).
  - New Apache Parquet and Feather file format support (#1180, #1435)
  - Allow appending to files with `GeoDataFrame.to_file` (#1229).
  - Add support for the `ignore_geometry` keyword in `read_file` to only read
    the attribute data. If set to True, a pandas DataFrame without geometry is
    returned (#1383).
  - `geopandas.read_file` now supports reading from file-like objects (#1329).
  - `GeoDataFrame.to_file` now supports specifying the CRS to write to the file
  (#802). By default it still uses the CRS of the GeoDataFrame.
  - New `chunksize` keyword in `geopandas.read_postgis` to read a query in
    chunks (#1123).
- Improvements related to geometry columns and CRS:
  - Any column of the GeoDataFrame that has a "geometry" dtype is now returned
    as a GeoSeries. This means that when having multiple geometry columns, not
    only the "active" geometry column is returned as a GeoSeries, but also
    accessing another geometry column (`gdf["other_geom_column"]`) gives a
    GeoSeries (#1336).
  - Multiple geometry columns in a GeoDataFrame can now each have a different
    CRS. The global `gdf.crs` attribute continues to returns the CRS of the
    "active" geometry column. The CRS of other geometry columns can be accessed
    from the column itself (eg `gdf["other_geom_column"].crs`) (#1339).
  - New `set_crs()` method on GeoDataFrame/GeoSeries to set the CRS of naive
    geometries (#747).
- Improvements related to plotting:
  - The y-axis is now scaled depending on the center of the plot when using a
    geographic CRS, instead of using an equal aspect ratio (#1290).
  - When passing a column of categorical dtype to the `column=` keyword of the
    GeoDataFrame `plot()`, we now honor all categories and its order (#1483).
    In addition, a new `categories` keyword allows to specify all categories
    and their order otherwise (#1173).
  - For choropleths using a classification scheme (using `scheme=`), the
    `legend_kwds` accept two new keywords to control the formatting of the
    legend: `fmt` with a format string for the bin edges (#1253), and `labels`
    to pass fully custom class labels (#1302).
- New `covers()` and `covered_by()` methods on GeoSeries/GeoDataframe for the
  equivalent spatial predicates (#1460, #1462).
- GeoPandas now warns when using distance-based methods with data in a
  geographic projection (#1378).

Deprecations:

- When constructing a GeoSeries or GeoDataFrame from data that already has a
  CRS, a deprecation warning is raised when both CRS don't match, and in the
  future an error will be raised in such a case. You can use the new `set_crs`
  method to override an existing CRS. See
  [the docs](https://geopandas.readthedocs.io/en/latest/projections.html#projection-for-multiple-geometry-columns).  
- The helper functions in the `geopandas.plotting` module are deprecated for
  public usage (#656).
- The `geopandas.io` functions are deprecated, use the top-level `read_file` and
  `to_file` instead (#1407).
- The set operators (`&`, `|`, `^`, `-`) are deprecated, use the
  `intersection()`, `union()`, `symmetric_difference()`, `difference()` methods
  instead (#1255).
- The `sindex` for empty dataframe will in the future return an empty spatial
  index instead of `None` (#1438).
- The `objects` keyword in the `intersection` method of the spatial index
  returned by the `sindex` attribute is deprecated and will be removed in the
  future (#1440).

Bug fixes:

- Fix the `total_bounds()` method to ignore missing and empty geometries (#1312).
- Fix `geopandas.clip` when masking with non-overlapping area resulting in an
  empty GeoDataFrame (#1309, #1365).
- Fix error in `geopandas.sjoin` when joining on an empty geometry column (#1318).
- CRS related fixes: `pandas.concat` preserves CRS when concatenating GeoSeries
  objects (#1340), preserve the CRS in `geopandas.clip` (#1362) and in
  `GeoDataFrame.astype` (#1366).
- Fix bug in `GeoDataFrame.explode()` when 'level_1' is one of the column names
  (#1445).
- Better error message when rtree is not installed (#1425).
- Fix bug in `GeoSeries.equals()` (#1451).
- Fix plotting of multi-part geometries with additional style keywords (#1385).

And we now have a [Code of Conduct](https://github.com/geopandas/geopandas/blob/master/CODE_OF_CONDUCT.md)!

GeoPandas 0.8.0 is the last release to support Python 3.5. The next release
will require Python 3.6, pandas 0.24, numpy 1.15 and shapely 1.6 or higher.


Version 0.7.0 (February 16, 2020)
---------------------------------

Support for Python 2.7 has been dropped. GeoPandas now works with Python >= 3.5.

The important API change of this release is that GeoPandas now requires
PROJ > 6 and pyproj > 2.2, and that the `.crs` attribute of a GeoSeries and
GeoDataFrame no longer stores the CRS information as a proj4 string or dict,
but as a ``pyproj.CRS`` object (#1101).

This gives a better user interface and integrates improvements from pyproj and
PROJ 6, but might also require some changes in your code. Check the
[migration guide](https://geopandas.readthedocs.io/en/latest/projections.html#upgrading-to-geopandas-0-7-with-pyproj-2-2-and-proj-6)
in the documentation.

Other API changes;

- The `GeoDataFrame.to_file` method will now also write the GeoDataFrame index
  to the file, if the index is named and/or non-integer. You can use the
  `index=True/False` keyword to overwrite this default inference (#1059).

New features and improvements:

- A new `geopandas.clip` function to clip a GeoDataFrame to the spatial extent
  of another shape (#1128).
- The `geopandas.overlay` function now works for all geometry types, including
  points and linestrings in addition to polygons (#1110).
- The `plot()` method gained support for missing values (in the column that
  determines the colors). By default it doesn't plot the corresponding
  geometries, but using the new `missing_kwds` argument you can specify how to
  style those geometries (#1156).
- The `plot()` method now also supports plotting GeometryCollection and
  LinearRing objects (#1225).
- Added support for filtering with a geometry or reading a subset of the rows in
  `geopandas.read_file` (#1160).
- Added support for the new nullable integer data type of pandas in
  `GeoDataFrame.to_file` (#1220).

Bug fixes:

- `GeoSeries.reset_index()` now correctly results in a GeoDataFrame instead of DataFrame (#1252).
- Fixed the `geopandas.sjoin` function to handle MultiIndex correctly (#1159).
- Fixed the `geopandas.sjoin` function to preserve the index name of the left GeoDataFrame (#1150).


Version 0.6.3 (February 6, 2020)
---------------------------------

Small bug-fix release:

- Compatibility with Shapely 1.7 and pandas 1.0 (#1244).
- Fix `GeoDataFrame.fillna` to accept non-geometry values again when there are
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

- Fix `astype` when converting to string with Multi geometries (#1145) or when converting a dataframe without geometries (#1144).
- Fix `GeoSeries.fillna` to accept `np.nan` again (#1149).


Version 0.6.0 (September 27, 2019)
----------------------------------

Important note! This will be the last release to support Python 2.7 (#1031)

API changes:

- A refactor of the internals based on the pandas ExtensionArray interface (#1000). The main user visible changes are:
  - The `.dtype` of a GeoSeries is now a `'geometry'` dtype (and no longer a numpy `object` dtype).
  - The `.values` of a GeoSeries now returns a custom `GeometryArray`, and no longer a numpy array. To get back a numpy array of Shapely scalars, you can convert explicitly using `np.asarray(..)`.
- The `GeoSeries` constructor now raises a warning when passed non-geometry data. Currently the constructor falls back to return a pandas `Series`, but in the future this will raise an error (#1085).
- The missing value handling has been changed to now separate the concepts of missing geometries and empty geometries (#601, 1062). In practice this means that (see [the docs](https://geopandas.readthedocs.io/en/v0.6.0/missing_empty.html) for more details):
  - `GeoSeries.isna` now considers only missing values, and if you want to check for empty geometries, you can use `GeoSeries.is_empty` (`GeoDataFrame.isna` already only looked at missing values).
  - `GeoSeries.dropna` now actually drops missing values (before it didn't drop either missing or empty geometries)
  - `GeoSeries.fillna` only fills missing values (behaviour unchanged).
  - `GeoSeries.align` uses missing values instead of empty geometries by default to fill non-matching index entries.

New features and improvements:

- Addition of a `GeoSeries.affine_transform` method, equivalent of Shapely's function (#1008).
- Addition of a `GeoDataFrame.rename_geometry` method to easily rename the active geometry column (#1053).
- Addition of `geopandas.show_versions()` function, which can be used to give an overview of the installed libraries in bug reports (#899).
- The `legend_kwds` keyword of the `plot()` method can now also be used to specify keywords for the color bar (#1102).
- Performance improvement in the `sjoin()` operation by re-using existing spatial index of the input dataframes, if available (#789).
- Updated documentation to work with latest version of geoplot and contextily (#1044, #1088).
- A new ``geopandas.options`` configuration, with currently a single option to control the display precision of the coordinates (``options.display_precision``). The default is now to show less coordinates (3 for projected and 5 for geographic coordinates), but the default can be overridden with the option.

Bug fixes:

- Also try to use `pysal` instead of `mapclassify` if available (#1082).
- The `GeoDataFrame.astype()` method now correctly returns a `GeoDataFrame` if the geometry column is preserved (#1009).
- The `to_crs` method now uses `always_xy=True` to ensure correct lon/lat order handling for pyproj>=2.2.0 (#1122).
- Fixed passing list-like colors in the `plot()` method in case of "multi" geometries (#1119).
- Fixed the coloring of shapes and colorbar when passing a custom `norm` in the `plot()` method (#1091, #1089).
- Fixed `GeoDataFrame.to_file` to preserve VFS file paths (e.g. when a "s3://" path is specified) (#1124).
- Fixed failing case in ``geopandas.sjoin`` with empty geometries (#1138).


In addition, the minimum required versions of some dependencies have been increased: GeoPandas now requirs pandas >=0.23.4 and matplotlib >=2.0.1 (#1002).


Version 0.5.1 (July 11, 2019)
-----------------------------

- Compatibility with latest mapclassify version 2.1.0 (#1025).

Version 0.5.0 (April 25, 2019)
------------------------------

Improvements:

* Significant performance improvement (around 10x) for `GeoDataFrame.iterfeatures`,
  which also improves `GeoDataFrame.to_file` (#864).
* File IO enhancements based on Fiona 1.8:
    * Support for writing bool dtype (#855) and datetime dtype, if the file format supports it (#728).
    * Support for writing dataframes with multiple geometry types, if the file format allows it (e.g. GeoJSON for all types, or ESRI Shapefile for Polygon+MultiPolygon) (#827, #867, #870).
* Compatibility with pyproj >= 2 (#962).
* A new `geopandas.points_from_xy()` helper function to convert x and y coordinates to Point objects (#896).
* The `buffer` and `interpolate` methods now accept an array-like to specify a variable distance for each geometry (#781). 
* Addition of a `relate` method, corresponding to the shapely method that returns the DE-9IM matrix (#853).
* Plotting improvements:
    * Performance improvement in plotting by only flattening the geometries if there are actually 'Multi' geometries (#785).
    * Choropleths: access to all `mapclassify` classification schemes and addition of the `classification_kwds` keyword in the `plot` method to specify options for the scheme (#876).
    * Ability to specify a matplotlib axes object on which to plot the color bar with the `cax` keyword, in order to have more control over the color bar placement (#894).
* Changed the default provider in ``geopandas.tools.geocode`` from Google (now requires an API key) to Geocode.Farm (#907, #975).

Bug fixes:

- Remove the edge in the legend marker (#807).
- Fix the `align` method to preserve the CRS (#829).
- Fix `geopandas.testing.assert_geodataframe_equal` to correctly compare left and right dataframes (#810).
- Fix in choropleth mapping when the values contain missing values (#877).
- Better error message in `sjoin` if the input is not a GeoDataFrame (#842).
- Fix in `read_postgis` to handle nullable (missing) geometries (#856).
- Correctly passing through the `parse_dates` keyword in `read_postgis` to the underlying pandas method (#860).
- Fixed the shape of Antarctica in the included demo dataset 'naturalearth_lowres'
  (by updating to the latest version) (#804).


Version 0.4.1 (March 5, 2019)
-----------------------------

Small bug-fix release for compatibility with the latest Fiona and PySAL
releases:

* Compatibility with Fiona 1.8: fix deprecation warning (#854).
* Compatibility with PySAL 2.0: switched to `mapclassify` instead of `PySAL` as
  dependency for choropleth mapping with the `scheme` keyword (#872).
* Fix for new `overlay` implementation in case the intersection is empty (#800).


Version 0.4.0 (July 15, 2018)
-----------------------------

Improvements:

* Improved `overlay` function (better performance, several incorrect behaviours fixed) (#429)
* Pass keywords to control legend behavior (`legend_kwds`) to `plot` (#434)
* Add basic support for reading remote datasets in `read_file` (#531)
* Pass kwargs for `buffer` operation on GeoSeries (#535)
* Expose all geopy services as options in geocoding (#550)
* Faster write speeds to GeoPackage (#605)
* Permit `read_file` filtering with a bounding box from a GeoDataFrame (#613)
* Set CRS on GeoDataFrame returned by `read_postgis` (#627)
* Permit setting markersize for Point GeoSeries plots with column values (#633)
* Started an example gallery (#463, #690, #717)
* Support for plotting MultiPoints (#683)
* Testing functionalty (e.g. `assert_geodataframe_equal`) is now publicly exposed (#707)
* Add `explode` method to GeoDataFrame (similar to the GeoSeries method) (#671)
* Set equal aspect on active axis on multi-axis figures (#718)
* Pass array of values to column argument in `plot` (#770)

Bug fixes:

* Ensure that colorbars are plotted on the correct axis (#523)
* Handle plotting empty GeoDataFrame (#571)
* Save z-dimension when writing files (#652)
* Handle reading empty shapefiles (#653)
* Correct dtype for empty result of spatial operations (#685)
* Fix empty `sjoin` handling for pandas>=0.23 (#762)


Version 0.3.0 (August 29, 2017)
-------------------------------

Improvements:

* Improve plotting performance using ``matplotlib.collections`` (#267)
* Improve default plotting appearance. The defaults now follow the new matplotlib defaults (#318, #502, #510)
* Provide access to x/y coordinates as attributes for Point GeoSeries (#383)
* Make the NYBB dataset available through ``geopandas.datasets`` (#384)
* Enable ``sjoin`` on non-integer-index GeoDataFrames (#422)
* Add ``cx`` indexer to GeoDataFrame (#482)
* ``GeoDataFrame.from_features`` now also accepts a Feature Collection (#225, #507)
* Use index label instead of integer id in output of ``iterfeatures`` and
  ``to_json`` (#421)
* Return empty data frame rather than raising an error when performing a spatial join with non overlapping geodataframes (#335)

Bug fixes:

* Compatibility with shapely 1.6.0 (#512)
* Fix ``fiona.filter`` results when bbox is not None (#372)
* Fix ``dissolve`` to retain CRS (#389)
* Fix ``cx`` behavior when using index of 0 (#478)
* Fix display of lower bin in legend label of choropleth plots using a PySAL scheme (#450)


Version 0.2.0
-------------

Improvements:

* Complete overhaul of the documentation
* Addition of ``overlay`` to perform spatial overlays with polygons (#142)
* Addition of ``sjoin`` to perform spatial joins (#115, #145, #188)
* Addition of ``__geo_interface__`` that returns a python data structure
  to represent the ``GeoSeries`` as a GeoJSON-like ``FeatureCollection`` (#116)
  and ``iterfeatures`` method (#178)
* Addition of the ``explode`` (#146) and ``dissolve`` (#310, #311) methods.
* Addition of the ``sindex`` attribute, a Spatial Index using the optional
  dependency ``rtree`` (``libspatialindex``) that can be used to speed up
  certain operations such as overlays (#140, #141).
* Addition of the ``GeoSeries.cx`` coordinate indexer to slice a GeoSeries based
  on a bounding box of the coordinates (#55).
* Improvements to plotting: ability to specify edge colors (#173), support for
  the ``vmin``, ``vmax``, ``figsize``, ``linewidth`` keywords (#207), legends
  for chloropleth plots (#210), color points by specifying a colormap (#186) or
  a single color (#238).
* Larger flexibility of ``to_crs``, accepting both dicts and proj strings (#289)
* Addition of embedded example data, accessible through
  ``geopandas.datasets.get_path``.

API changes:

* In the ``plot`` method, the ``axes`` keyword is renamed to ``ax`` for
  consistency with pandas, and the ``colormap`` keyword is renamed to ``cmap``
  for consistency with matplotlib (#208, #228, #240).

Bug fixes:

* Properly handle rows with missing geometries (#139, #193).
* Fix ``GeoSeries.to_json`` (#263).
* Correctly serialize metadata when pickling (#199, #206).
* Fix ``merge`` and ``concat`` to return correct GeoDataFrame (#247, #320, #322).
