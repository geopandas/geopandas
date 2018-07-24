Changes
=======

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

Bug fixes :

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
