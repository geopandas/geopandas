.. currentmodule:: geopandas

.. ipython:: python
   :suppress:

   import geopandas as gpd


Managing Projections
=========================================



Coordinate Reference Systems
-----------------------------

CRS are important because the geometric shapes in a GeoSeries or GeoDataFrame object are simply a collection of coordinates in an arbitrary space. A CRS tells Python how those coordinates related to places on the Earth.

CRS are referred to using codes called `proj4 strings <https://en.wikipedia.org/wiki/PROJ.4>`_. You can find the codes for most commonly used projections from `www.spatialreference.org <http://spatialreference.org/>`_ or `remotesensing.org <http://www.remotesensing.org/geotiff/proj_list/>`_.

The same CRS can often be referred to in many ways. For example, one of the most commonly used CRS is the WGS84 latitude-longitude projection. One `proj4` representation of this projection is: ``"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"``. But common projections can also be referred to by `EPSG` codes, so this same projection can also called using the `proj4` string ``"+init=epsg:4326"``.

*geopandas* can accept lots of representations of CRS, including the `proj4` string itself (``"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"``) or parameters broken out in a dictionary: ``{'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}``). In addition, some functions will take `EPSG` codes directly.

For reference, a few very common projections and their proj4 strings:

* WGS84 Latitude/Longitude: ``"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"`` or ``"+init=epsg:4326"``
* UTM Zones (North): ``"+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"``
* UTM Zones (South): ``"+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south"``

Setting a Projection
----------------------

There are two relevant operations for projections: setting a projection and re-projecting.

Setting a projection may be necessary when for some reason *geopandas* has coordinate data (x-y values), but no information about how those coordinates refer to locations in the real world. Setting a projection is how one tells *geopandas* how to interpret coordinates. If no CRS is set, *geopandas* geometry operations will still work, but coordinate transformations will not be possible and exported files may not be interpreted correctly by other software.

Be aware that **most of the time** you don't have to set a projection. Data loaded from a reputable source (using the ``from_file()`` command) *should* always include projection information. You can see an objects current CRS through the ``crs`` attribute: ``my_geoseries.crs``.

From time to time, however, you may get data that does not include a projection. In this situation, you have to set the CRS so *geopandas* knows how to interpret the coordinates.

For example, if you convert a spreadsheet of latitudes and longitudes into a GeoSeries by hand, you would set the projection by assigning the WGS84 latitude-longitude CRS to the ``crs`` attribute:

.. sourcecode:: python

   my_geoseries.crs = {'init' :'epsg:4326'}


Re-Projecting
----------------

Re-projecting is the process of changing the representation of locations from one coordinate system to another. All projections of locations on the Earth into a two-dimensional plane `are distortions <https://en.wikipedia.org/wiki/Map_projection#Which_projection_is_best.3F>`_, the projection that is best for your application may be different from the projection associated with the data you import. In these cases, data can be re-projected using the ``to_crs`` command:

.. ipython:: python

    # load example data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Check original projection
    # (it's Platte Carre! x-y are long and lat)
    world.crs

    # Visualize
    @savefig world_starting.png width=3in
    world.plot();

    # Reproject to Mercator (after dropping Antartica)
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]
    world = world.to_crs({'init': 'epsg:3395'}) # world.to_crs(epsg=3395) would also work
    @savefig world_reproj.png width=3in
    world.plot();
