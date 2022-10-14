# Roadmap

This page provides an overview of the strategic goals for development of GeoPandas. Some
of the tasks may happen sooner given the appropriate funding, other later with no
specified date, and some may not happen at all if the implementation proves to be
against the will of the community or face technical issues preventing their inclusion in
the code base.

The current roadmap is divided into two milestones. The first milestone aims at a
release of the first major version of GeoPandas, while the second milestone is a
longer-term vision covering enhancements that should happen in subsequent releases.

## Roadmap for GeoPandas 1.0

### Fully vectorized geometry engine

GeoPandas uses `shapely` as its geometry engine, based on scalar geometries, requiring a
loop-based implementation of most GeoPandas methods. That comes at a significant
performance cost, which is being resolved in shapely 2.0, a new major release resulting
from a complete rewrite of the internals using the vectorized implementation prototyped
in the `PyGEOS` project. At this moment, GeoPandas supports `shapely<2.0`,
`shapely>=2.0`, and `PyGEOS` as possible geometry engines, which causes friction in the
development process and uneven performance on the user side based on what geometry
engine the user happens to be using.

GeoPandas 1.0 will require `shapely>=2.0` and deprecate both older shapely and `PyGEOS`
engines. This change should simplify the code base allowing more manageable maintenance
and a lower barrier to entry for new contributors.

### Feature parity with shapely

Even though GeoPandas uses shapely as the geometry engine, not all its functions are
exposed at a GeoPandas level. This has resulted in a less convenient API and a need to
switch between `GeoSeries` objects and lists or arrays of geometries, potentially
risking the data loss or corruption as the CRS is not included in such operations. In
the first phase, all element-wise operations (e.g. `segmentize`, or
`minimum_bounding_circle`) should be exposed as `GeoSeries` methods. The feature parity
should be reached in the second phase, covering all relevant functions.

### Clarity of the API

The first version of the GeoPandas API is nearly ten years old. The PyData ecosystem has
significantly changed in the meantime, and some of the early decisions may no longer be
future-proof. Ahead of GeoPandas 1.0, the API will be revised to ensure that all the
necessary deprecations occur before the major release to provide the stability of the
API for the coming years.

### Pruned dependencies

GeoPandas offers functionality for every step of a typical geospatial workflow, from
reading of the GIS file formats to geometry operations and handling of Coordinate
Reference Systems (CRS) and transformation of geometries between them. However, GIS I/O
depends on a relatively heavy C++ library `GDAL` and CRS management on another C++
library `PROJ`, even though not every application based on GeoPandas is necessarily
geospatial. GeoPandas 1.0 should eliminate the hard dependency on both `GDAL` and `PROJ`
and offer the basic capability of a GeoDataFrame with a minimal set of dependencies
limited to `pandas` and `shapely`.

## Beyond GeoPandas 1.0

Additional work is planned for a longer time frame, stretching beyond GeoPandas 1.0
without a specific target release.

### S2 geometry engine

The geometry engine used in GeoPandas is `shapely`, which serves as a Python API for
`GEOS`. It means that all geometry operations in GeoPandas are planar, using (possibly)
projected coordinate reference systems. Some applications focusing on the global context
may find planar operations limiting as they come with troubles around anti-meridian and
poles. One solution is an implementation of a spherical geometry engine, namely `S2`,
that should eliminate these limitations and offer an alternative to `GEOS`.

The GeoPandas community is currently working together with the R-spatial community that
has already exposed `S2` in an R counterpart of GeoPandas `sf` on Python bindings for
`S2`, that should be used as a secondary geometry engine in GeoPandas.

### Lighter-weight geospatial I/O

In order to support lighter-weight installations of GeoPandas that do not depend on
heavier and difficult to install libraries such as GDAL, additional I/O libraries should
be developed and integrated into GeoPandas as optional dependencies.  These should be
simpler to install and not require binary dependencies, which would lower the barrier to
entry for GeoPandas users that need basic I/O support for a limited number of GIS
formats such as ESRI Shapefiles or GeoPackages.

### Prepared geometries

GeoPandas is using spatial indexing for the operations that may benefit from it. Further
performance gains can be achieved using prepared geometries. Preparation creates a
spatial index of individual line segments of geometries, greatly enhancing the speed of
spatial predicates like `intersects` or `contains`. Given that the preparation has
become less computationally expensive in `shapely` 2.0, GeoPandas should expose the
preparation to the user but, more importantly, use smart automatic geometry preparation
under the hood.

### Static plotting improvements

GeoPandas currently covers a broad range of geospatial tasks, from data exploration to
advanced analysis. However, one moment may tempt the user to use different software -
plotting. GeoPandas can create static maps based on ``matplotlib``, but they are a bit
basic at the moment. It isn't straightforward to generate a complex map in a
production-quality which can go straight to an academic journal or an infographic. We
want to change this and remove barriers which we currently have and make it simple to
create beautiful maps.
