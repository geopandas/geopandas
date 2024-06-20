import pytest

from operator import countOf
from unittest.mock import patch, DEFAULT
from geopandas.io import file
from geopandas.io.file import coverage_branches_cheng

# --------
# Func: _check_engine(engine, func)
# Path: geopandas/io/file.py
# --------

engine_res_pairs_mock_pyogrio = [
    (None, "fiona"),
    ("fiona", "fiona"),
    ("pyogrio", "pyogrio"),
    ("some_text", "some_text"),
]

engine_res_pairs = [
    (None, "pyogrio"),
    ("fiona", "fiona"),
    ("pyogrio", "pyogrio"),
    ("some_text", "some_text"),
]

@pytest.mark.parametrize("engine, excpt_res", engine_res_pairs_mock_pyogrio)
def test_bcov_check_engine_mock_pyogrio(engine, excpt_res):
    with patch.multiple("geopandas.io.file", _import_pyogrio = DEFAULT, _check_pyogrio = DEFAULT ):
        res = file._check_engine(engine, "'read_file' function")
        
        assert(excpt_res == res)

@pytest.mark.parametrize("engine, excpt_res", engine_res_pairs)
def test_bcov_check_engine(engine, excpt_res):
    res = file._check_engine(engine, "'read_file' function")

    assert(excpt_res == res)

# --- OWN COVERAGE TOOL ---

def print_coverage(flags: dict):
    for branch, hit in flags.items():
        print(f"{branch} was {'hit' if hit else 'not hit'}")
    print(f"Coverage: {float(countOf(flags.values(), True) / len(flags)) * 100}%")


# Main

for args in engine_res_pairs_mock_pyogrio:
    test_bcov_check_engine_mock_pyogrio(args[0], args[1])

for args in engine_res_pairs:
    test_bcov_check_engine(args[0], args[1])

print_coverage(coverage_branches_cheng)
