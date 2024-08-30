.. _how_to:

How to...
=========

Drop duplicate geometry in all situations
-----------------------------------------

Using the standard Pandas :meth:`~pandas.DataFrame.drop_duplicates` function on a geometry column can lead to some duplicate
geometries not being dropped, in certain circumstances. When used on a geometry columnm, the Pandas function compares the
WKB of each geometry object. This is sensitive to the orders of various components of the geometry - for example, a line
with co-ordinates in the order left-to-right should be equal to a line with the same co-ordinates in the order right-to-left,
but the WKB representations will be different. The same applies for the order of rings of polygons and parts in multipart
geometries.

To deal with this problem, use the :meth:`~geopandas.GeoSeries.normalize` method first to order the co-ordinates in a canonincal form,
and then use the standard :meth:`~pandas.DataFrame.drop_duplicates` method::

    gdf["geometry"] = gdf.normalize()
    gdf.drop_duplicates()

The effect of the :meth:`~geopandas.GeoSeries.normalize` method can be seen in the following example::

    >>> geopandas.GeoSeries([
    ...     shapely.LineString([(0, 0), (1, 0), (2, 0)]),
    ...     shapely.LineString([(2, 0), (1, 0), (0, 0)]),
    ... ]).normalize().to_wkt()
    0    LINESTRING (0 0, 1 0, 2 0)
    1    LINESTRING (0 0, 1 0, 2 0)
    dtype: object
