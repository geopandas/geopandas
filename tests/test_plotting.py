import os
import unittest

from matplotlib.pyplot import Artist, savefig
from matplotlib.testing.decorators import image_comparison
from shapely.geometry import Polygon, LineString, Point

from geopandas import GeoSeries

# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')

def save_baseline_image(filename):
    """ save a baseline image """
    savefig(os.path.join(BASELINE_DIR, filename))

@image_comparison(baseline_images=['poly_plot'], extensions=['png'])
def test_poly_plot():
    """ Test plotting a simple series of polygons """
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(1, 0), (2, 1), (2, 1)])
    polys = GeoSeries([t1, t2])
    ax = polys.plot()
    assert isinstance(ax, Artist)
    if GENERATE_BASELINE:
        save_baseline_image('poly_plot.png')

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'])
