Contributing to GeoPandas
=========================

(Contribution guidelines largely copied from `pandas <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_)

Overview
--------

Contributions to GeoPandas are very welcome.  They are likely to
be accepted more quickly if they follow these guidelines.

At this stage of GeoPandas development, the priorities are to define a
simple, usable, and stable API and to have clean, maintainable,
readable code.  Performance matters, but not at the expense of those
goals.

In general, GeoPandas follows the conventions of the pandas project
where applicable.

In particular, when submitting a pull request:

- All existing tests should pass.  Please make sure that the test
  suite passes, both locally and on
  `Travis CI <https://travis-ci.org/geopandas/geopandas>`_.  Status on
  Travis will be visible on a pull request.  If you want to enable
  Travis CI on your own fork, please read the pandas guidelines link
  above or the
  `getting started docs <http://about.travis-ci.org/docs/user/getting-started/>`_.

- New functionality should include tests.  Please write reasonable
  tests for your code and make sure that they pass on your pull request.

- Classes, methods, functions, etc. should have docstrings.  The first
  line of a docstring should be a standalone summary.  Parameters and
  return values should be ducumented explicitly.

- GeoPandas supports python 2 (2.6+) and python 3 (3.2+) with a single
  code base.  Use modern python idioms when possible that are
  compatibile with both major versions, and use the
  `six <https://pythonhosted.org/six>`_ library where helpful to smooth
  over the differences.  Use ``from __future__ import`` statements where
  appropriate.  Test code locally in both python 2 and python 3 when
  possible (all supported versions will be automatically tested on
  Travis CI).

- Follow PEP 8 when possible.

- Imports should be grouped with standard library imports first,
  3rd-party libraries next, and geopandas imports third.  Within each
  grouping, imports should be alphabetized.  Always use absolute
  imports when possible, and explicit relative imports for local
  imports when necessary in tests.


Seven Steps for Contributing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are seven basic steps to contributing to *geopandas*:

1) Fork the *geopandas* git repository
2) Create a development environment
3) Install *geopandas* dependencies
4) Make a ``development`` build of *geopandas*
5) Make changes to code and add tests
6) Update the documentation
7) Submit a Pull Request

Each of these 7 steps is detailed below.


1) Forking the *geopandas* repository using Git
------------------------------------------------

To the new user, working with Git is one of the more daunting aspects of contributing to *geopandas**.
It can very quickly become overwhelming, but sticking to the guidelines below will help keep the process
straightforward and mostly trouble free.  As always, if you are having difficulties please
feel free to ask for help.

The code is hosted on `GitHub <https://github.com/geopandas/geopandas>`_. To
contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* Software Carpentry's `Git Tutorial <http://swcarpentry.github.io/git-novice/>`_
* `Atlassian <https://www.atlassian.com/git/tutorials/what-is-version-control>`_
* the `GitHub help pages <http://help.github.com/>`_.
* Matthew Brett's `Pydagogue <http://matthew-brett.github.com/pydagogue/>`_.

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

.. _contributing.forking:

Forking
~~~~~~~~

You will need your own fork to work on the code. Go to the `geopandas project
page <https://github.com/geopandas/geopandas>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone git@github.com:your-user-name/geopandas.git geopandas-yourname
    cd geopandas-yourname
    git remote add upstream git://github.com/geopandas/geopandas.git

This creates the directory `geopandas-yourname` and connects your repository to
the upstream (main project) *geopandas* repository.

The testing suite will run automatically on Travis-CI once your pull request is
submitted.  However, if you wish to run the test suite on a branch prior to
submitting the pull request, then Travis-CI needs to be hooked up to your
GitHub repository.  Instructions for doing so are `here
<http://about.travis-ci.org/docs/user/getting-started/>`__.

Creating a branch
~~~~~~~~~~~~~~~~~~

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *geopandas*. You can have many shiny-new-features
and switch in between them using the git checkout command.

To update this branch, you need to retrieve the changes from the master branch::

    git fetch upstream
    git rebase upstream/master

This will replay your commits on top of the latest geopandas git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``stash`` them prior
to updating.  This will effectively store your changes and they can be reapplied
after updating.

.. _contributing.dev_env:

2) Creating a development environment
---------------------------------------
A development environment is a virtual space where you can keep an independent installation of *geopandas*.
This makes it easy to keep both a stable version of python in one place you use for work, and a development
version (which you may break while playing with code) in another.

An easy way to create a *geopandas* development environment is as follows:

- Install either `Anaconda <http://docs.continuum.io/anaconda/>`_ or
  `miniconda <http://conda.pydata.org/miniconda.html>`_
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *geopandas** source directory

Tell conda to create a new environment, named ``geopandas_dev``, or any other name you would like
for this environment, by running::

      conda create -n geopandas_dev

For a python 3 environment::

      conda create -n geopandas_dev python=3.4

This will create the new environment, and not touch any of your existing environments,
nor any existing python installation.

To work in this environment, Windows users should ``activate`` it as follows::

      activate geopandas_dev

Mac OSX and Linux users should use::

      source activate geopandas_dev

You will then see a confirmation message to indicate you are in the new development environment.

To view your environments::

      conda info -e

To return to you home root environment::

      deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.

At this point you can easily do a *development* install, as detailed in the next sections.

3) Installing Dependencies
--------------------------

To run *geopandas* in an development environment, you must first install
*geopandas*'s dependencies. We suggest doing so using the following commands
(executed after your development environment has been activated)::

    conda install -c conda-forge fiona shapely pyproj rtree
    conda install pandas


This should install all necessary dependencies.

4) Making a development build
-----------------------------

Once dependencies are in place, make an in-place build by navigating to the git
clone of the *geopandas* repository and running::

    python setup.py develop


5) Making changes and writing tests
-------------------------------------

*geopandas* is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests.  Often the test can be
taken from the original GitHub issue.  However, it is always worth considering additional
use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to *geopandas*.  Therefore,
it is worth getting in the habit of writing tests ahead of time so this is never an issue.

*geopandas* uses the `pytest testing system
<http://doc.pytest.org/en/latest/>`_ and the convenient
extensions in `numpy.testing
<http://docs.scipy.org/doc/numpy/reference/routines.testing.html>`_.

Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` directory. This folder contains many
current examples of tests, and we suggest looking to these for inspiration.

The ``.util`` module has some special ``assert`` functions that
make it easier to make statements about whether GeoSeries or GeoDataFrame
objects are equivalent. The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result, using eg the function ``assert_geoseries_equal``.

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

The tests can then be run directly inside your Git clone (without having to
install *geopandas*) by typing::

    pytest

6) Updating the Documentation
-----------------------------

*geopandas* documentation resides in the `doc` folder. Changes to the docs are
make by modifying the appropriate file in the `source` folder within `doc`.
*geopandas* docs us reStructuredText syntax, `which is explained here <http://www.sphinx-doc.org/en/stable/rest.html#rst-primer>`_
and the docstrings follow the `Numpy Docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Once you have made your changes, you can build the docs by navigating to the `doc` folder and typing::

    make html

The resulting html pages will be located in `doc/build/html`.


7) Submitting a Pull Request
------------------------------

Once you've made changes and pushed them to your forked repository, you then
submit a pull request to have them integrated into the *geopandas* code base.

You can find a pull request (or PR) tutorial in the `GitHub's Help Docs <https://help.github.com/articles/using-pull-requests/>`_.
