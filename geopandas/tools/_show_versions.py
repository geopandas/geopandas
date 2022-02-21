import importlib
import platform
import sys


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_C_info():
    """Information on system PROJ, GDAL, GEOS
    Returns
    -------
    c_info: dict
        system PROJ information
    """
    try:
        import pyproj

        proj_version = pyproj.proj_version_str
    except Exception:
        proj_version = None
    try:
        import pyproj

        proj_dir = pyproj.datadir.get_data_dir()
    except Exception:
        proj_dir = None

    try:
        import shapely._buildcfg

        geos_version = "{}.{}.{}".format(*shapely._buildcfg.geos_version)
        geos_dir = shapely._buildcfg.geos_library_path
    except Exception:
        geos_version = None
        geos_dir = None

    try:
        import fiona

        gdal_version = fiona.env.get_gdal_release_name()
    except Exception:
        gdal_version = None
    try:
        import fiona

        gdal_dir = fiona.env.GDALDataFinder().search()
    except Exception:
        gdal_dir = None

    blob = [
        ("GEOS", geos_version),
        ("GEOS lib", geos_dir),
        ("GDAL", gdal_version),
        ("GDAL data dir", gdal_dir),
        ("PROJ", proj_version),
        ("PROJ data dir", proj_dir),
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
        "numpy",
        "shapely",
        "rtree",
        "pyproj",
        "matplotlib",
        "mapclassify",
        "geopy",
        "psycopg2",
        "geoalchemy2",
        "pyarrow",
        "pygeos",
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
        except Exception:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """
    Print system information and installed module versions.

    Examples
    --------

    ::

        $ python -c "import geopandas; geopandas.show_versions()"
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()
    proj_info = _get_C_info()

    maxlen = max(len(x) for x in deps_info)
    tpl = "{{k:<{maxlen}}}: {{stat}}".format(maxlen=maxlen)
    print("\nSYSTEM INFO")
    print("-----------")
    for k, stat in sys_info.items():
        print(tpl.format(k=k, stat=stat))
    print("\nGEOS, GDAL, PROJ INFO")
    print("---------------------")
    for k, stat in proj_info.items():
        print(tpl.format(k=k, stat=stat))
    print("\nPYTHON DEPENDENCIES")
    print("-------------------")
    for k, stat in deps_info.items():
        print(tpl.format(k=k, stat=stat))
