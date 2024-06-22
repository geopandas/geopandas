# Roadmap

This page provides an overview of the strategic goals for development of GeoPandas. Some
of the tasks may happen sooner given the appropriate funding, other later with no
specified date, and some may not happen at all if the implementation proves to be
against the will of the community or face technical issues preventing their inclusion in
the code base.

The current roadmap reflects longer-term vision covering enhancements that should happen
in upcoming releases.

## S2 geometry engine

The geometry engine used in GeoPandas is `shapely`, which serves as a Python API for
`GEOS`. It means that all geometry operations in GeoPandas are planar, using (possibly)
projected coordinate reference systems. Some applications focusing on the global context
may find planar operations limiting as they come with troubles around anti-meridian and
poles. One solution is an implementation of a spherical geometry engine, namely `S2`,
that should eliminate these limitations and offer an alternative to `GEOS`.

The GeoPandas community is currently working together with the R-spatial community that
has already exposed `S2` in an R counterpart of GeoPandas `sf` on Python bindings for
`S2`, that should be used as a secondary geometry engine in GeoPandas.

## Prepared geometries

GeoPandas is using spatial indexing for the operations that may benefit from it. Further
performance gains can be achieved using prepared geometries. Preparation creates a
spatial index of individual line segments of geometries, greatly enhancing the speed of
spatial predicates like `intersects` or `contains`. Given that the preparation has
become less computationally expensive in `shapely` 2.0, GeoPandas should expose the
preparation to the user but, more importantly, use smart automatic geometry preparation
under the hood.

## Static plotting improvements

GeoPandas currently covers a broad range of geospatial tasks, from data exploration to
advanced analysis. However, one moment may tempt the user to use different software -
plotting. GeoPandas can create static maps based on ``matplotlib``, but they are a bit
basic at the moment. It isn't straightforward to generate a complex map in a
production-quality which can go straight to an academic journal or an infographic. We
want to change this and remove barriers which we currently have and make it simple to
create beautiful maps.
