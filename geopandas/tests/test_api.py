import subprocess
import sys


def test_no_additional_imports():
    # test that 'import geopandas' does not import any of the optional or
    # development dependencies
    blacklist = {
        "pytest",
        "py",
        "ipython",
        # fiona actually gets imported if installed (but error suppressed until used)
        # "fiona",
        # "matplotlib",  # matplotlib gets imported by pandas, see below
        "mapclassify",
        "sqlalchemy",
        "psycopg",
        "psycopg2",
        "geopy",
        "geoalchemy2",
        "matplotlib",
    }

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
    returncode = subprocess.run(call, check=False).returncode
    assert returncode == 0
