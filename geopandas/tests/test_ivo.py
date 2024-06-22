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

