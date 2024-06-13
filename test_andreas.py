import pytest
from unittest.mock import patch, DEFAULT
from geopandas.io import file

# --------
# Unit test
#
# Func: def _check_engine(engine, func)
# Path: geopandas/io/file.py
# --------

cov_br_list = [None] * 10

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

# --- PYTEST ---

@pytest.mark.parametrize("engine, excpt_res", engine_res_pairs_mock_pyogrio)
def test_bcov_check_engine_mock_pyogrio(engine, excpt_res):
    with patch.multiple("geopandas.io.file", _import_pyogrio = DEFAULT, _check_pyogrio = DEFAULT ):
        res = file._check_engine(engine, "'read_file' function", cov_br_list)
        
        assert(excpt_res == res)

@pytest.mark.parametrize("engine, excpt_res", engine_res_pairs)
def test_bcov_check_engine(engine, excpt_res):
    res = file._check_engine(engine, "'read_file' function", cov_br_list)

    assert(excpt_res == res)


# --- OWN COVERAGE TOOL ---

for args in engine_res_pairs_mock_pyogrio:
    test_bcov_check_engine_mock_pyogrio(args[0], args[1])

for args in engine_res_pairs:
    test_bcov_check_engine(args[0], args[1]) 

print("Branch coverage: " + str(int(cov_br_list.count(1) / len(cov_br_list) * 100)) + "%")