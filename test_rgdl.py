from operator import countOf
from geopandas import _version
from geopandas._version import coverage_branches_rgdl


# --------
# Func: render_git_describe_long(pieces: Dict[str, Any])
# Path: geopandas/_version.py
# --------

def test_git_describe_long_ct(): 
    test_params = {
        "closest-tag": "",
        "short": "",
        "dirty": True
    }
    res = _version.render_git_describe_long(test_params)
    print("long_res: " + res)

def test_git_describe_long_short():
    test_params = {
        "closest-tag": False,
        "short": "10",
        "dirty": False
    }
    res = _version.render_git_describe_long(test_params)
    print("short_res: " + res)


# --- OWN COVERAGE TOOL ---

def print_coverage(flags: dict):
    for branch, hit in flags.items():
        print(f"{branch} was {'hit' if hit else 'not hit'}")
    print(f"Coverage: {float(countOf(flags.values(), True) / len(flags)) * 100}%")


# Main

test_git_describe_long_ct()
test_git_describe_long_short()

print_coverage(coverage_branches_rgdl)

