import subprocess
import sys


def test_no_additional_imports():
    # test that 'import geopandas' does not import any of the optional or
    # development dependencies
    code = """
import sys
import geopandas
blacklist = {'pytest', 'py', 'ipython',
             'matplotlib' 'descartes','mapclassify',
             # 'rtree',  # rtree actually gets imported if installed
             'sqlalchemy', 'psycopg2', 'geopy'}
mods = blacklist & set(m.split('.')[0] for m in sys.modules)
if mods:
    sys.stderr.write('err: geopandas should not import: {}'.format(', '.join(mods)))
    sys.exit(len(mods))
"""
    call = [sys.executable, "-c", code]
    returncode = subprocess.run(call).returncode
    assert returncode == 0
