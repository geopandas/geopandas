"""
Illustrating make_grid options.
"""
import geopandas
import geodatasets

import matplotlib.pyplot as plt


world = geopandas.read_file(
    geodatasets.get_path('naturalearth land'))
madagascar = world.cx[45:50, -25:-15]

sq_grid = geopandas.make_grid(madagascar, cell_size=1)
sq_grid_centers = geopandas.make_grid(madagascar, cell_size=1, what="centers")
sq_grid_corners = geopandas.make_grid(madagascar, cell_size=1, what="corners")

hex_grid = geopandas.make_grid(madagascar, cell_size=1, cell_type="hexagon")


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
# ax0.grid(zorder=-10)
# ax0.set_axisbelow(True)
# ax1.grid(zorder=-10)
# ax1.set_axisbelow(True)

madagascar.plot(ax=ax0, alpha=0.5)
sq_grid.plot(ax=ax0, facecolor="none", alpha=0.5)
sq_grid_centers.plot(ax=ax0, color="C1", marker="o", label='what="centers"')
sq_grid_corners.plot(ax=ax0, color="C2", marker="x", label='what="corners"')
ax0.legend(loc="upper left")

madagascar.plot(ax=ax1, alpha=0.5)
hex_grid.plot(ax=ax1, facecolor="none", edgecolor="C2")

ax0.set_title('cell_type="square"')
ax1.set_title('cell_type="hexagon"')

fig.tight_layout()
