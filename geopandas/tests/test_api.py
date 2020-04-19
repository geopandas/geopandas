import subprocess
import sys

from geopandas._compat import PANDAS_GE_10


def test_no_additional_imports():
    # test that 'import geopandas' does not import any of the optional or
    # development dependencies
    blacklist = {
        "pytest",
        "py",
        "ipython",
        # 'matplotlib',  # matplotlib gets imported by pandas, see below
        "descartes",
        "mapclassify",
        # 'rtree',  # rtree actually gets imported if installed
        "sqlalchemy",
        "psycopg2",
        "geopy",
    }
    if PANDAS_GE_10:
        # pandas > 0.25 stopped importing matplotlib by default
        blacklist.add("matplotlib")

    code = """
import sys
import geopandas
blacklist = {0!r}

mods = blacklist & set(m.split('.')[0] for m in sys.modules)
if mods:
    sys.stderr.write('err: geopandas should not import: {{}}'.format(', '.join(mods)))
    sys.exit(len(mods))
""".format(
        blacklist
    )
    call = [sys.executable, "-c", code]
    returncode = subprocess.run(call).returncode
    assert returncode == 0
