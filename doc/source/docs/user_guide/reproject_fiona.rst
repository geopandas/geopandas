Re-projecting with Fiona
============================

The simplest method of re-projecting is :meth:`GeoDataFrame.to_crs`.
It uses ``pyproj`` as the engine and transforms the points within the geometries.

This example demonstrates how to use ``Fiona`` as the engine to re-project your data.
Fiona is powered by GDAL and with algorithms that consider the geometry instead of
just the points the geometry contains. This is particularly useful for antimeridian cutting.
However, this also means the transformation is not as fast.

.. code-block:: python

    from functools import partial

    import fiona
    import geopandas
    from fiona.transform import transform_geom
    from packaging import version
    from pyproj import CRS
    from pyproj.enums import WktVersion
    from shapely.geometry import mapping, shape


    # set up Fiona transformer
    def crs_to_fiona(proj_crs):
        proj_crs = CRS.from_user_input(proj_crs)
        if version.parse(fiona.__gdal_version__) < version.parse("3.0.0"):
            fio_crs = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ can use WKT2
            fio_crs = proj_crs.to_wkt()
        return fio_crs

    def base_transformer(geom, src_crs, dst_crs):
        return shape(
            transform_geom(
                src_crs=crs_to_fiona(src_crs),
                dst_crs=crs_to_fiona(dst_crs),
                geom=mapping(geom),
                antimeridian_cutting=True,
            )
        )

    # load example data
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    destination_crs = "EPSG:3395"
    forward_transformer = partial(base_transformer, src_crs=world.crs, dst_crs=destination_crs)

    # Reproject to Mercator (after dropping Antartica)
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]
    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION="YES"):
        mercator_world = world.set_geometry(world.geometry.apply(forward_transformer), crs=destination_crs)