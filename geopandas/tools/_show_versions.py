import platform
import sys
import importlib


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace('\n', ' ')

    blob = [
        ("python", python),
        ('executable', sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "geopandas",
        "pandas",
        "fiona",
        "osgeo.gdal",
        "numpy",
        "shapely",
        "rtree",
        "pyproj",
        "matplotlib",
        "mapclassify",
        "pysal",
        "geopy",
        "psycopg2",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """
    Print system information and installed module versions.

    Example
    -------
    > python -c "import geopandas; geopandas.show_versions()"
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    maxlen = max(len(x) for x in deps_info)
    tpl = "{{k:<{maxlen}}}: {{stat}}".format(maxlen=maxlen)
    print("\nSYSTEM INFO")
    print("-----------")
    for k, stat in sys_info.items():
        print(tpl.format(k=k, stat=stat))
    print("\nPYTHON DEPENDENCIES")
    print("-------------------")
    for k, stat in deps_info.items():
        print(tpl.format(k=k, stat=stat))
