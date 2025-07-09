:html_theme.sidebar_secondary.remove:

GeoPandas |version|
===================

GeoPandas is an open source project to make working with geospatial
data in python easier.  GeoPandas extends the datatypes used by
`pandas`_ to allow spatial operations on geometric types.  Geometric
operations are performed by `shapely`_.  Geopandas further depends on
`pyogrio`_ for file access and `matplotlib`_ for plotting.

.. _pandas: http://pandas.pydata.org
.. _shapely: https://shapely.readthedocs.io
.. _pyogrio: https://pyogrio.readthedocs.io
.. _matplotlib: http://matplotlib.org

Description
-----------

The goal of GeoPandas is to make working with geospatial data in
python easier.  It combines the capabilities of pandas and shapely,
providing geospatial operations in pandas and a high-level interface
to multiple geometries to shapely.  GeoPandas enables you to easily do
operations in python that would otherwise require a spatial database
such as PostGIS.

.. toctree::
   :hidden:

   Home <self>
   About <about>
   Getting started <getting_started>
   Documentation <docs>
   Community <community>

.. container:: button

    :doc:`Getting started <getting_started>` :doc:`Documentation <docs>`
    :doc:`About GeoPandas <about>` :doc:`Community <community>`

Useful links
------------

`Binary Installers (PyPI) <https://pypi.org/project/geopandas/>`_ | `Source Repository (GitHub) <https://github.com/geopandas/geopandas>`_ | `Issues & Ideas <https://github.com/geopandas/geopandas/issues>`_ | `Q&A Support <https://stackoverflow.com/questions/tagged/geopandas>`_

|pypi| |Actions Status| |Coverage Status| |Join the chat at https://gitter.im/geopandas/geopandas| |Binder| |DOI| |Powered by NumFOCUS|

.. |pypi| image:: https://img.shields.io/pypi/v/geopandas.svg
   :target: https://pypi.python.org/pypi/geopandas/
.. |Actions Status| image:: https://github.com/geopandas/geopandas/workflows/Tests/badge.svg
   :target: https://github.com/geopandas/geopandas/actions?query=workflow%3ATests
.. |Coverage Status| image:: https://codecov.io/gh/geopandas/geopandas/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/geopandas/geopandas
.. |Join the chat at https://gitter.im/geopandas/geopandas| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/geopandas/geopandas?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/geopandas/geopandas/main
.. |DOI| image:: https://zenodo.org/badge/11002815.svg
   :target: https://zenodo.org/badge/latestdoi/11002815
.. |Powered by NumFOCUS| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: https://numfocus.org

Supported by
------------

The GeoPandas project uses an
`open governance model <https://github.com/geopandas/governance/blob/main/Governance.md>`__ and is fiscally
sponsored by `NumFOCUS <https://numfocus.org/>`__. Consider making a
`tax-deductible donation <https://numfocus.org/donate-for-geopandas>`__
to help the project pay for developer time, professional services,
travel, workshops, and a variety of other needs.

.. image:: _static/SponsoredProject.svg
    :alt: numfocus
    :width: 400
    :target: https://numfocus.org/project/geopandas
