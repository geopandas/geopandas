Changes
=======

Next
----

Improvements:

* Pass keywords to control legend behavior (`legend_kwds`) to `plot` (#434)
* Add basic support for reading remote datasets in `read_file` (#531)
* Pass kwargs for `buffer` operation on GeoSeries (#535)
* Expose all geopy services as options in geocoding (#550)

Bug fixes :

* Ensure that colorbars are plotted on the correct axis (#523)


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
