import os
import folium
from shapely import Point
import geopandas as gp

#os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.sys.path.append('../explore.py')
os.sys.path.append('../_version.py')
from geopandas.explore import coverage_tool_tip, _tooltip_popup
from geopandas._version import coverage_render_pep440_pre, coverage_render_pep440_old ,render_pep440_pre, render_pep440_old

def print_percentage(flags):
   percentage = sum(flags.values()) / len(flags) 
   print(f"Coverage: {percentage*100}%\n")

def print_coverage(coverage_dict):
    for branch, hit in coverage_dict.items():
        print(f"{branch} was {'hit' if hit else 'not hit'}")

def test_render_pep440_pre():
#    testDict = {
#        "closest-tag": None,
#        "distance": 5
#    }
#    assert render_pep440_pre(testDict) == "0.post0.dev5"

#    testDict = {
#        "closest-tag": "1.0.0",
#        "distance": 0
#    }
#    assert render_pep440_pre(testDict) == "1.0.0"

#    testDict = {
#         "closest-tag": "1.0.0",
#        "distance": 3
#    }
#    assert render_pep440_pre(testDict) == "1.0.0.post0.dev3"

#    testDict = {
#        "closest-tag": "1.0.0.post2",
#        "distance": 4
#    }
#    assert render_pep440_pre(testDict) == "1.0.0.post3.dev4"

    print()
    print("render_pep440_pre coverage:")
    print_coverage(coverage_render_pep440_pre)
    print_percentage(coverage_render_pep440_pre)

def test_render_pep440_old():
#    testDict = {
#        "closest-tag": "v1.2.3",
#        "distance": 0,
#        "dirty": False
#    }
#    assert render_pep440_old(testDict) == "v1.2.3"

#    testDict = {
#        "closest-tag": "v1.2.3",
#        "distance": 5,
#        "dirty": False
#    }
#    assert render_pep440_old(testDict) == "v1.2.3.post5"

#    testDict = {
#        "closest-tag": "v1.2.3",
#        "distance": 5,
#        "dirty": True
#    }
#    assert render_pep440_old(testDict) == "v1.2.3.post5.dev0"


#    testDict = {
#        "closest-tag": None,
#        "distance": 7,
#        "dirty": False
#    }
#    assert render_pep440_old(testDict) == "0.post7"

#    testDict = {
#        "closest-tag": None,
#        "distance": 7,
#        "dirty": True
#    }
#    assert render_pep440_old(testDict) == "0.post7.dev0"

    print()
    print("render_pep440_old coverage:")
    print_coverage(coverage_render_pep440_old)
    print_percentage(coverage_render_pep440_old)

