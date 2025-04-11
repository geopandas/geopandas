import geopandas as gpd, numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
import matplotlib.pyplot as plt

# %%

geom = Polygon(
    [(0, 0), (0, 5), (5, 5), (5, 0)],
    [
        [(1, 1), (1, 2), (2, 2), (2, 1)],
        [(3, 2), (3, 3), (4, 3), (4, 2)],
    ],
)

df = gpd.GeoDataFrame(geometry=[geom])
ax = df.plot(edgecolor='black');
# plt.show();
plotted_vertices = ax.collections[0].get_paths()[0].vertices
expected_vertices = df.normalize().get_coordinates().to_numpy()

np.testing.assert_array_equal(plotted_vertices, expected_vertices)

# %%

poly1 = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
poly2 = box(3, 3, 6, 6).difference(box(4, 4, 5, 5))
multipoly = MultiPolygon([poly1, poly2])
_df = gpd.GeoDataFrame(geometry=[multipoly])
ax = _df.plot()
# plt.show();

# TODO: there's more than one patch in collections. Find out how to append together.
plotted_vertices = np.append(
    ax.collections[0].get_paths()[0].vertices,
    ax.collections[0].get_paths()[1].vertices,
    axis=0
)
expected_vertices = _df.normalize().get_coordinates().to_numpy()

print(plotted_vertices)
print()
print(expected_vertices)
print()

np.testing.assert_array_equal(plotted_vertices, expected_vertices)