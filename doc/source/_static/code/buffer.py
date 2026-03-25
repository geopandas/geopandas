"""
Create an illustrative figure for different kwargs
in buffer method.
"""


import geopandas
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon

s = geopandas.GeoSeries(
    [
        Point(0, 0),
        LineString([(1, -1), (1, 0), (2, 0), (2, 1)]),
        Polygon([(3, -1), (3, 0), (4, 0), (4, 1), (5, 1), (5, -1)])
    ]
)

fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
for ax in axs.flatten():
    s.plot(ax=ax)
    ax.set(xticks=[], yticks=[])

s.buffer(0.2).plot(ax=axs[0, 0], alpha=0.6)
axs[0, 0].set_title("s.buffer(0.2)")

s.buffer(0.2, quad_segs=2).plot(ax=axs[0, 1], alpha=0.6)
axs[0, 1].set_title("s.buffer(0.2, quad_segs=2)")

s.buffer(-0.2).plot(ax=axs[0, 2], color="white", alpha=0.4)
axs[0, 2].set_title("s.buffer(-0.2)")

s.buffer(0.2, cap_style="square").plot(ax=axs[1, 0], alpha=0.6)
axs[1, 0].set_title('s.buffer(0.2, cap_style="square")')

s.buffer(0.2, cap_style="flat").plot(ax=axs[1, 1], alpha=0.6)
axs[1, 1].set_title('s.buffer(0.2, cap_style="flat")')

s.buffer(-0.2, join_style="mitre").plot(ax=axs[1, 2], color="white", alpha=0.4)
axs[1, 2].set_title('s.buffer(-0.2, join_style="mitre")')

s.buffer(0.2, join_style="mitre").plot(ax=axs[2, 0], alpha=0.6)
axs[2, 0].set_title('s.buffer(0.2, join_style="mitre")')

s.buffer(0.2, join_style="bevel").plot(ax=axs[2, 1], alpha=0.6)
axs[2, 1].set_title('s.buffer(0.2, join_style="bevel")')

s.buffer(-0.2, join_style="bevel").plot(ax=axs[2, 2], color="white", alpha=0.4)
axs[2, 2].set_title('s.buffer(-0.2, join_style="bevel")')

fig.tight_layout()
