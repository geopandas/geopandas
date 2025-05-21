=========
GeoSeries
=========
.. currentmodule:: geopandas

Constructor
-----------
.. autosummary::
   :toctree: api/

   GeoSeries

General methods and attributes
------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.area
   GeoSeries.boundary
   GeoSeries.bounds
   GeoSeries.total_bounds
   GeoSeries.length
   GeoSeries.geom_type
   GeoSeries.offset_curve
   GeoSeries.distance
   GeoSeries.hausdorff_distance
   GeoSeries.frechet_distance
   GeoSeries.representative_point
   GeoSeries.exterior
   GeoSeries.interiors
   GeoSeries.minimum_bounding_radius
   GeoSeries.minimum_clearance
   GeoSeries.x
   GeoSeries.y
   GeoSeries.z
   GeoSeries.m
   GeoSeries.get_coordinates
   GeoSeries.count_coordinates
   GeoSeries.count_geometries
   GeoSeries.count_interior_rings
   GeoSeries.set_precision
   GeoSeries.get_precision
   GeoSeries.get_geometry

Unary predicates
----------------

.. autosummary::
   :toctree: api/

   GeoSeries.is_closed
   GeoSeries.is_empty
   GeoSeries.is_ring
   GeoSeries.is_simple
   GeoSeries.is_valid
   GeoSeries.is_valid_reason
   GeoSeries.is_valid_coverage
   GeoSeries.invalid_coverage_edges
   GeoSeries.has_m
   GeoSeries.has_z
   GeoSeries.is_ccw


Binary predicates
-----------------

.. autosummary::
   :toctree: api/

   GeoSeries.contains
   GeoSeries.contains_properly
   GeoSeries.crosses
   GeoSeries.disjoint
   GeoSeries.dwithin
   GeoSeries.geom_equals
   GeoSeries.geom_equals_exact
   GeoSeries.geom_equals_identical
   GeoSeries.intersects
   GeoSeries.overlaps
   GeoSeries.touches
   GeoSeries.within
   GeoSeries.covers
   GeoSeries.covered_by
   GeoSeries.relate
   GeoSeries.relate_pattern


Set-theoretic methods
---------------------

.. autosummary::
   :toctree: api/

   GeoSeries.clip_by_rect
   GeoSeries.difference
   GeoSeries.intersection
   GeoSeries.symmetric_difference
   GeoSeries.union

Constructive methods and attributes
-----------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.boundary
   GeoSeries.buffer
   GeoSeries.centroid
   GeoSeries.concave_hull
   GeoSeries.convex_hull
   GeoSeries.envelope
   GeoSeries.extract_unique_points
   GeoSeries.force_2d
   GeoSeries.force_3d
   GeoSeries.make_valid
   GeoSeries.minimum_bounding_circle
   GeoSeries.maximum_inscribed_circle
   GeoSeries.minimum_clearance
   GeoSeries.minimum_clearance_line
   GeoSeries.minimum_rotated_rectangle
   GeoSeries.normalize
   GeoSeries.orient_polygons
   GeoSeries.remove_repeated_points
   GeoSeries.reverse
   GeoSeries.sample_points
   GeoSeries.segmentize
   GeoSeries.shortest_line
   GeoSeries.simplify
   GeoSeries.simplify_coverage
   GeoSeries.snap
   GeoSeries.transform

Affine transformations
----------------------

.. autosummary::
   :toctree: api/

   GeoSeries.affine_transform
   GeoSeries.rotate
   GeoSeries.scale
   GeoSeries.skew
   GeoSeries.translate

Linestring operations
---------------------

.. autosummary::
   :toctree: api/

   GeoSeries.interpolate
   GeoSeries.line_merge
   GeoSeries.project
   GeoSeries.shared_paths

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.build_area
   GeoSeries.constrained_delaunay_triangles
   GeoSeries.delaunay_triangles
   GeoSeries.explode
   GeoSeries.intersection_all
   GeoSeries.polygonize
   GeoSeries.union_all
   GeoSeries.voronoi_polygons


Serialization / IO / conversion
-------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.from_arrow
   GeoSeries.from_file
   GeoSeries.from_wkb
   GeoSeries.from_wkt
   GeoSeries.from_xy
   GeoSeries.to_arrow
   GeoSeries.to_file
   GeoSeries.to_json
   GeoSeries.to_wkb
   GeoSeries.to_wkt

Projection handling
-------------------

.. autosummary::
   :toctree: api/

   GeoSeries.crs
   GeoSeries.set_crs
   GeoSeries.to_crs
   GeoSeries.estimate_utm_crs

Missing values
--------------

.. autosummary::
   :toctree: api/

   GeoSeries.fillna
   GeoSeries.isna
   GeoSeries.notna

Overlay operations
------------------

.. autosummary::
   :toctree: api/

   GeoSeries.clip

Plotting
--------

.. autosummary::
   :toctree: api/

   GeoSeries.plot
   GeoSeries.explore


Spatial index
-------------

.. autosummary::
   :toctree: api/

   GeoSeries.sindex
   GeoSeries.has_sindex

Indexing
--------

.. autosummary::
   :toctree: api/

   GeoSeries.cx

Interface
---------

.. autosummary::
   :toctree: api/

   GeoSeries.__geo_interface__


Methods of pandas ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result when appropriate. The methods
``isna()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.
