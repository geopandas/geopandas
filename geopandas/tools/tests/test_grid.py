import geopandas
import pytest
from shapely import geometry
from scipy.spatial import distance
import numpy

from geopandas.tools.grids import make_grid
from geopandas import _compat as compat

nybb = geopandas.read_file(geopandas.datasets.get_path("nybb")).to_crs(epsg=4326)

box = geometry.box(-1, -1, 1, 1)

staten = nybb.iloc[0].geometry

rect = geometry.box(0, 0, 0.5, 1)


def find_spacing(input, return_distances=False):
    x, y = input.x, input.y
    first = numpy.array([[x[0], y[0]]])
    rest = numpy.column_stack((x[1:], y[1:]))
    distances = distance.cdist(first, rest)
    if return_distances:
        return distances.min(), distances
    else:
        return distances.min()


@pytest.mark.parametrize(
    ["geom", "size", "spacing", "tile"],
    [
        (geom, size, spacing, tile)
        for geom in (None, box, rect, nybb, staten)
        for size in ((5, 8), 10, None)
        for spacing in (None, 0.05)
        for tile in ("hex", "square")
        if (not ((size is not None) & (spacing is not None)))
    ],
)
@pytest.mark.skipif(
    not (compat.HAS_PYGEOS | compat.HAS_RTREE),
    reason="Sampling requires a spatial index",
)
def test_grid(
    geom,
    size,
    spacing,
    tile,
):
    clipped = make_grid(
        geom=geom,
        size=size,
        spacing=spacing,
        tile=tile,
        as_polygons=False,
        clip=True,
    )

    implied_spacing = find_spacing(clipped)

    assert isinstance(clipped, geopandas.GeoSeries)
    if size is not None:
        n_obs = clipped.shape[0]
        if tile == "square":
            implied_size = size**2 if isinstance(size, int) else (size[0] * size[1])
            assert n_obs <= implied_size, (
                f"The clipped grid is not smaller than "
                f"the implied size: {n_obs} !< {implied_size}"
            )

    if spacing is not None:
        numpy.testing.assert_allclose(
            spacing,
            implied_spacing,
            err_msg=f"Spacing was provided to make_grid()"
            f"but output spacing differs from input spacing:"
            f"\noutput:{implied_spacing}\tinput:{spacing}",
        )

    not_clipped = make_grid(
        geom=geom,
        size=size,
        spacing=spacing,
        tile=tile,
        as_polygons=False,
        clip=False,
    )

    assert len(clipped) <= len(
        not_clipped
    ), "Clipped output is larger than the unclipped output."
    implied_spacing_unclipped = find_spacing(not_clipped)
    numpy.testing.assert_allclose(
        implied_spacing_unclipped,
        implied_spacing,
        err_msg=f"Spacing differs between clipped and unclipped data:\n"
        f"clipped: {implied_spacing}, unclipped:{implied_spacing_unclipped}",
    )

    if spacing is not None:
        dmat = distance.squareform(
            distance.pdist(numpy.column_stack((not_clipped.x, not_clipped.y)))
        )
        min_dists = numpy.ones_like(dmat) * implied_spacing_unclipped
        n_at_minimum = numpy.isclose(min_dists, dmat).sum(axis=1)
        if tile == "hex":
            assert n_at_minimum.max() == 6, (
                f"hexagonal gridpoints "
                f"should have 6 neighbors, maximum was {n_at_minimum.max()}"
            )
            assert n_at_minimum.min() == 3, (
                f"hexagonal gridpoints "
                f"should have 3 neighbors, minimum was {n_at_minimum.min()}"
            )
        else:
            assert n_at_minimum.max() == 4, (
                f"square gridpoints "
                f"should have 4 neighbors, maximum was {n_at_minimum.max()}"
            )
            assert n_at_minimum.min() == 2, (
                f"square gridpoints "
                f"should have 2 neighbors, minimum was {n_at_minimum.min()}"
            )
