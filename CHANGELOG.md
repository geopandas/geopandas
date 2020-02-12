Changes
=======

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
