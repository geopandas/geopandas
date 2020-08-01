Getting Started
---------------

Installation
============

GeoPandas is written in pure Python, but has several dependecies written in C 
(`GEOS`_, `GDAL`_, `PROJ`_).  Those base C libraries can sometimes be a challenge to 
install. Therefore, we advise you to closely follow the recommendations below to avoid 
installation problems.

Easy way
^^^^^^^^

The best way to install GeoPandas is using ``conda`` and ``conda-forge`` channel::

  conda install -c conda-forge geopandas

Detailed instructions
^^^^^^^^^^^^^^^^^^^^^

Do you prefer ``pip install`` or installation from source? Or specific version? See 
:doc:`detailed instructions <getting_started/install>`.

What now?
=========

If you don't have GeoPandas yet, check :doc:`Installation <getting_started/install>`. 
If you have never used GeoPandas and want to get familiar with it and its core 
functionality quickly, see :doc:`Getting Started Tutorial <getting_started/introduction>`. 
Detailed illustration how to work with different parts of GeoPandas, how to make maps,
manage projections, spatially merge data or geocode are part of our 
:doc:`User Guide <documentation/user_guide>`. And if you are interested in the complete 
documentation of all classes, functions, method and attributes GeoPandas offers, 
:doc:`API Reference <documentation/reference>` is here for you.

.. raw:: html

  <a href="#" class="tutorial">Installation</a> <a href="#" class="tutorial">Tutorial</a> <br />
  <a href="#" class="tutorial">User Guide</a> <a href="#" class="tutorial">API Reference</a>


Get in touch
============

Haven't found what you were looking for?

- Ask usage questions ("How do I?") on `StackOverflow`_ or `GIS StackExchange`_.
- Report bugs, suggest features or view the source code `on GitHub`_.
- For a quick question about a bug report or feature request, or Pull Request,
  head over to the `gitter channel`_.
- For less well defined questions or ideas, or to announce other projects of
  interest to GeoPandas users, ... use the `mailing list`_.

.. _StackOverflow: https://stackoverflow.com/questions/tagged/geopandas
.. _GIS StackExchange: https://gis.stackexchange.com/questions/tagged/geopandas
.. _on GitHub: https://github.com/geopandas/geopandas
.. _gitter channel: https://gitter.im/geopandas/geopandas
.. _mailing list: https://groups.google.com/forum/#!forum/geopandas


.. toctree::
  :maxdepth: 2
  :caption: Getting Started
  :hidden:

  Installation <getting_started/install>
  Introduction to GeoPandas <getting_started/introduction>
  Examples Gallery <gallery/index>


.. _GDAL: https://www.gdal.org/

.. _GEOS: https://geos.osgeo.org

.. _PROJ: https://proj.org/
