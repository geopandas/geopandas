import os
import folium
from shapely import Point
import geopandas as gp

#os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.sys.path.append('../explore.py')
os.sys.path.append('../_version.py')
from explore import coverage_tool_tip, _tooltip_popup
from _version import coverage_render_pep440_pre, render_pep440_pre

def print_percentage(flags):
   percentage = sum(flags.values()) / len(flags) 
   print(f"Coverage: {percentage*100}%\n")

def print_coverage(coverage_dict):
    for branch, hit in coverage_dict.items():
        print(f"{branch} was {'hit' if hit else 'not hit'}")

def test_tooltip_popup():
    data = {
        'name': ['A', 'B'],
        'value': [1, 2],
        'geometry': [Point(1, 2), Point(3, 4)],
        '__folium_color':['green','yellow']
    }
    df = gp.GeoDataFrame(data, crs=4326)

    assert _tooltip_popup('tooltip', False, df) is None
    assert _tooltip_popup('tooltip', None, df) is None
    assert _tooltip_popup('tooltip', 0, df) is None
    
    res = _tooltip_popup('tooltip', 2, df)
    assert isinstance(res, folium.GeoJsonTooltip)

    res = _tooltip_popup('popup', True, df)
    assert isinstance(res, folium.GeoJsonPopup)

    res = _tooltip_popup("popup", "name", df)
    assert isinstance(res, folium.GeoJsonPopup)

    print()
    print("tooltip_popup coverage:")
    print_coverage(coverage_tool_tip)
    print_percentage(coverage_tool_tip)

def test_render_pep440_pre():
    testDict = {
        "closest-tag": None,
        "distance": 5
    }
    assert render_pep440_pre(testDict) == "0.post0.dev5"

    testDict = {
        "closest-tag": "1.0.0",
        "distance": 0
    }
    assert render_pep440_pre(testDict) == "1.0.0"

    testDict = {
        "closest-tag": "1.0.0",
        "distance": 3
    }
    assert render_pep440_pre(testDict) == "1.0.0.post0.dev3"

    testDict = {
        "closest-tag": "1.0.0.post2",
        "distance": 4
    }
    assert render_pep440_pre(testDict) == "1.0.0.post3.dev4"

    print()
    print("render_pep440_pre coverage:")
    print_coverage(coverage_render_pep440_pre)
    print_percentage(coverage_render_pep440_pre)