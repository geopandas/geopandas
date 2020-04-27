Guidelines
==========

Contributions to GeoPandas are very welcome. They are likely to
be accepted more quickly if they follow these guidelines.

At this stage of GeoPandas development, the priorities are to define a
simple, usable, and stable API and to have clean, maintainable,
readable code. Performance matters, but not at the expense of those
goals.

In general, GeoPandas follows the conventions of the pandas project
where applicable. Please read the [pandas contributing
guidelines](http://pandas.pydata.org/pandas-docs/stable/contributing.html).


In particular, when submitting a pull request:

- Install the requirements for the development environment (one can do this
  with either conda, and the environment.yml file, or pip, and the
  requirements-dev.txt file, and can use the pandas contributing guidelines
  as a guide). 
- All existing tests should pass. Please make sure that the test
  suite passes, both locally and on
  [Travis CI](https://travis-ci.org/geopandas/geopandas).  Status on
  Travis will be visible on a pull request. If you want to enable
  Travis CI on your own fork, please read the pandas guidelines link
  above or the
  [getting started docs](http://about.travis-ci.org/docs/user/getting-started/).

- New functionality should include tests. Please write reasonable
  tests for your code and make sure that they pass on your pull request.

- Classes, methods, functions, etc. should have docstrings. The first
  line of a docstring should be a standalone summary. Parameters and
  return values should be documented explicitly.

Improving the documentation and testing for code already in GeoPandas
is a great way to get started if you'd like to make a contribution.

Style
-----

- GeoPandas supports Python 3.5+ only. The last version of GeoPandas
  supporting Python 2 is 0.6.

- GeoPandas follows [the PEP 8
  standard](http://www.python.org/dev/peps/pep-0008/) and uses
  [Black](https://black.readthedocs.io/en/stable/) and
  [Flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent
  code format throughout the project.

- Imports should be grouped with standard library imports first,
  third-party libraries next, and GeoPandas imports third. Within each
  grouping, imports should be alphabetized. Always use absolute
  imports when possible, and explicit relative imports for local
  imports when necessary in tests.

- You can set up [pre-commit hooks](https://pre-commit.com/) to
  automatically run `black` and `flake8` when you make a git
  commit. This can be done by installing `pre-commit`:

    $ python -m pip install pre-commit

  From the root of the geopandas repository, you should then install
  `pre-commit`:

    $ pre-commit install

  Then `black` and `flake8` will be run automatically each time you
  commit changes. You can skip these checks with `git commit
  --no-verify`.
