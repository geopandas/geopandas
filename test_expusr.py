import pathlib

from operator import countOf
from geopandas.io import file
from geopandas.io.file import coverage_branches_expusr


# --------
# Func: _expand_user(path)
# Path: geopandas/io/file.py
# --------

def test_expand_user_str(): 
    res = file._expand_user("any/path")
    assert(isinstance(res, str))

def test_expand_user_path(): 
    res = file._expand_user(pathlib.Path("any/path"))
    assert(isinstance(res, pathlib.Path))


# --- OWN COVERAGE TOOL ---

def print_coverage(flags: dict):
    for branch, hit in flags.items():
        print(f"{branch} was {'hit' if hit else 'not hit'}")
    print(f"Coverage: {float(countOf(flags.values(), True) / len(flags)) * 100}%")


# Main

test_expand_user_str()
test_expand_user_path()

print_coverage(coverage_branches_expusr)

