import os
import tempfile
import unittest

from matplotlib.pyplot import Artist, savefig
from matplotlib.testing.noseclasses import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from shapely.geometry import Polygon, LineString, Point

from geopandas import GeoSeries

# If set to True, generate images rather than perform tests (all tests will pass!)
GENERATE_BASELINE = False

TEMPDIR = tempfile.gettempdir()
BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline_images', 'test_plotting')


class PlotTests(unittest.TestCase):
    
    def test_poly_plot(self, tol=8):
        """ Test plotting a simple series of polygons """
        filename = 'poly_plot.png'
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        polys = GeoSeries([t1, t2])
        ax = polys.plot()
        assert isinstance(ax, Artist)
        if GENERATE_BASELINE:
            savefig(os.path.join(BASELINE_DIR, filename))
        savefig(os.path.join(TEMPDIR, filename))
        err = compare_images(os.path.join(BASELINE_DIR, filename),
                             os.path.join(TEMPDIR, filename),
                             tol, in_decorator=True)
        try:
            if err:
                raise ImageComparisonFailure('images not close: %(actual)s '
                                             'vs. %(expected)s '
                                             '(RMS %(rms).3f)' % err)
        finally:
            os.remove(os.path.join(TEMPDIR, filename))

if __name__ == '__main__':
    unittest.main()
