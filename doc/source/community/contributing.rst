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
  `GitHub Actions <https://github.com/geopandas/geopandas/actions>`_.  Status on
  GHA will be visible on a pull request. GHA are automatically enabled
  on your own fork as well. To trigger a check, make a PR to your own fork.

- New functionality should include tests.  Please write reasonable
  tests for your code and make sure that they pass on your pull request.

- Classes, methods, functions, etc. should have docstrings.  The first
  line of a docstring should be a standalone summary.  Parameters and
  return values should be documented explicitly.

- Follow PEP 8 when possible. We use `ruff <https://docs.astral.sh/ruff/>`_
  to ensure a consistent code format throughout the project. For more details
  see :ref:`below <contributing_style>`.

- Imports should be grouped with standard library imports first,
  3rd-party libraries next, and GeoPandas imports third.  Within each
  grouping, imports should be alphabetized.  Always use absolute
  imports when possible, and explicit relative imports for local
  imports when necessary in tests.

- GeoPandas supports Python 3.9+ only. The last version of GeoPandas
  supporting Python 2 is 0.6.

- Unless your PR implements minor changes or internal work only, make sure
  it contains a note describing the changes in the `CHANGELOG.md` file.


Seven Steps for Contributing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are seven basic steps to contributing to GeoPandas:

1) Fork the GeoPandas git repository
2) Create a development environment
3) Install GeoPandas dependencies
4) Make a ``development`` build of GeoPandas
5) Make changes to code and add tests
6) Update the documentation
7) Submit a Pull Request

Each of these 7 steps is detailed below.


1) Forking the GeoPandas repository using Git
------------------------------------------------

To the new user, working with Git is one of the more daunting aspects of contributing to GeoPandas.
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
* Matthew Brett's `Pydagogue <https://matthew-brett.github.io/pydagogue/>`_.

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions <http://help.github.com/set-up-git-redirect>`__ for installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

.. _contributing.forking:

Forking
~~~~~~~

You will need your own fork to work on the code. Go to the `GeoPandas project
page <https://github.com/geopandas/geopandas>`_ and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git clone git@github.com:your-user-name/geopandas.git geopandas-yourname
    cd geopandas-yourname
    git remote add upstream git://github.com/geopandas/geopandas.git

This creates the directory ``geopandas-yourname`` and connects your repository to
the upstream (main project) GeoPandas repository.

The testing suite will run automatically on GitHub Actions once your pull request is
submitted. The test suite will also automatically run on your branch so you can
check it prior to submitting the pull request.

Creating a branch
~~~~~~~~~~~~~~~~~~

You want your main branch to reflect only production-ready code, so create a
feature branch for making your changes. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to::

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to GeoPandas. You can have many shiny-new-features
and switch in between them using the git checkout command.

To update this branch, you need to retrieve the changes from the main branch::

    git fetch upstream
    git rebase upstream/main

This will replay your commits on top of the latest GeoPandas git main.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``stash`` them prior
to updating.  This will effectively store your changes and they can be reapplied
after updating.

.. _contributing.dev_env:

2) Creating a development environment
---------------------------------------
A development environment is a virtual space where you can keep an independent installation of GeoPandas.
This makes it easy to keep both a stable version of python in one place you use for work, and a development
version (which you may break while playing with code) in another.

An easy way to create a GeoPandas development environment is as follows:

- Install either `Anaconda <http://docs.continuum.io/anaconda/>`_ or
  `miniconda <http://conda.pydata.org/miniconda.html>`_
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the ``geopandas`` source directory

Using the provided environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoPandas provides an environment which includes the required dependencies for development.
The environment file is located in the top level of the repo and is named ``environment-dev.yml``.
You can create this environment by navigating to the the ``geopandas`` source directory
and running::

      conda env create -f environment-dev.yml

This will create a new conda environment named ``geopandas_dev``.

Creating the environment manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, it is possible to create a development environment manually.  To do this,
tell conda to create a new environment named ``geopandas_dev``, or any other name you would like
for this environment, by running::

      conda create -n geopandas_dev python

This will create the new environment, and not touch any of your existing environments,
nor any existing python installation.

Working with the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work in this environment, you need to ``activate`` it. The instructions below
should work for both Windows, Mac and Linux::

      conda activate geopandas_dev

Once your environment is activated, you will see a confirmation message to
indicate you are in the new development environment.

To view your environments::

      conda info -e

To return to you home root environment::

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`__.

At this point you can easily do a *development* install, as detailed in the next sections.


3) Installing Dependencies
--------------------------

To run GeoPandas in an development environment, you must first install the dependencies of
GeoPandas. If you used the provided environment in section 2, skip this
step and continue to section 4. If you created the environment manually, we suggest installing
dependencies using the following commands (executed after your development environment has been activated)::

    conda install -c conda-forge pandas pyogrio shapely pyproj pytest

This should install all necessary dependencies.


4) Making a development build
-----------------------------

Once dependencies are in place, make an in-place build by navigating to the git
clone of the GeoPandas repository and running::

    python -m pip install -e .


5) Making changes and writing tests
-------------------------------------

GeoPandas is serious about testing and strongly encourages contributors to embrace
`test-driven development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_.
This development process "relies on the repetition of a very short development cycle:
first the developer writes an (initially failing) automated test case that defines a desired
improvement or new function, then produces the minimum amount of code to pass that test."
So, before actually writing any code, you should write your tests.  Often the test can be
taken from the original GitHub issue.  However, it is always worth considering additional
use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to GeoPandas.  Therefore,
it is worth getting in the habit of writing tests ahead of time so this is never an issue.

GeoPandas uses the `pytest testing system
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
install GeoPandas) by typing::

    pytest

6) Updating the Documentation
-----------------------------

GeoPandas documentation resides in the ``doc`` folder. Changes to the docs are made by
modifying the appropriate file in the ``source`` folder within ``doc``. GeoPandas docs use
mixture of reStructuredText syntax for ``rst`` files, `which is explained here
<http://www.sphinx-doc.org/en/stable/rest.html#rst-primer>`_ and MyST syntax for ``md``
files `explained here <https://myst-parser.readthedocs.io/en/latest/index.html>`_.
The docstrings follow the `Numpy Docstring standard
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_. Some pages
and examples are Jupyter notebooks converted to docs using `nbsphinx
<https://nbsphinx.readthedocs.io/>`_. Jupyter notebooks should be stored without the output.

We highly encourage you to follow the `Google developer documentation style guide
<https://developers.google.com/style/highlights>`_ when updating or creating new documentation.

Once you have made your changes, you may try if they render correctly by
building the docs using sphinx. To do so, you can navigate to the `doc` folder::

    cd doc

and type::

    make html

The resulting html pages will be located in ``doc/build/html``.

In case of any errors, you can try to use ``make html`` within a new environment based on
environment.yml specification in the ``doc`` folder. You may need to register Jupyter kernel as
``geopandas_docs``. Using conda::

    cd doc
    conda env create -f environment.yml
    conda activate geopandas_docs
    python -m ipykernel install --user --name geopandas_docs
    make html

For minor updates, you can skip the ``make html`` part as reStructuredText and MyST
syntax are usually quite straightforward.


7) Submitting a Pull Request
------------------------------

Once you've made changes and pushed them to your forked repository, you then
submit a pull request to have them integrated into the GeoPandas code base.

You can find a pull request (or PR) tutorial in the `GitHub's Help Docs <https://help.github.com/articles/using-pull-requests/>`_.

.. _contributing_style:

Style Guide & Linting
---------------------

GeoPandas follows the `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_
standard and uses `ruff <https://docs.astral.sh/ruff/>`_ to ensure a consistent
code format throughout the project.

Continuous Integration (GitHub Actions) will run those tools and
report any stylistic errors in your code. Therefore, it is helpful before
submitting code to run the check yourself::

   ruff format geopandas
   ruff check geopandas
   git diff upstream/main -u -- "*.py" | ruff .

to auto-format your code. Additionally, many editors have plugins that will
apply ``ruff`` as you edit files.

Optionally (but recommended), you can setup `pre-commit hooks <https://pre-commit.com/>`_
to automatically run ``ruff`` when you make a git commit. If you did not
use the provided development environment in ``environment-dev.yml``, you must
first install ``pre-commit``::

   $ python -m pip install pre-commit

From the root of the geopandas repository, you should then install the
``pre-commit`` included in GeoPandas::

   $ pre-commit install

Then ``ruff`` will be run automatically each time you commit changes. You can
skip these checks with ``git commit --no-verify``.

Commit message conventions
--------------------------

Commit your changes to your local repository with an explanatory message. GeoPandas
uses the pandas convention for commit message prefixes and layout. Here are
some common prefixes along with general guidelines for when to use them:

* ENH: Enhancement, new functionality
* BUG: Bug fix
* DOC: Additions/updates to documentation
* TST: Additions/updates to tests
* BLD: Updates to the build process/scripts
* PERF: Performance improvement
* TYP: Type annotations
* CLN: Code cleanup

The following defines how a commit message should be structured. Please refer to the
relevant GitHub issues in your commit message using GH1234 or #1234. Either style
is fine, but the former is generally preferred:

* a subject line with `< 80` chars.
* One blank line.
* Optionally, a commit message body.

Now you can commit your changes in your local repository::

    git commit -m
