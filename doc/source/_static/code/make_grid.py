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
sq_grid2 = geopandas.make_grid(madagascar, cell_size=1, offset=(30, -30))


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
ax0.grid(zorder=-10)
ax0.set_axisbelow(True)
ax1.grid(zorder=-10)
ax1.set_axisbelow(True)

madagascar.plot(ax=ax0, alpha=0.5)
sq_grid.plot(ax=ax0, facecolor="none", edgecolor="C2")

madagascar.plot(ax=ax1, alpha=0.5)
sq_grid2.plot(ax=ax1, facecolor="none", edgecolor="C2")

ax0.set_title("Default offset")
ax1.set_title("Custom offset")

fig.tight_layout()
