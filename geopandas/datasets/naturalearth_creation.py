"""
Script that generates the included dataset 'naturalearth_lowres.shp'
and 'naturalearth_cities.shp'.

Raw data: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
Current version used: see code
"""  # noqa (E501 link is longer than max line length)

import geopandas as gpd
import requests
from pathlib import Path
from zipfile import ZipFile
import tempfile
from shapely.geometry import box

version = "latest"
urlbase = "https://www.naturalearthdata.com/"
urlbase += "http//www.naturalearthdata.com/download/110m/cultural/"


def countries_override(world_raw):
    # not ideal - fix some country codes
    mask = world_raw["ISO_A3"].eq("-99") & world_raw["TYPE"].isin(
        ["Sovereign country", "Country"]
    )
    world_raw.loc[mask, "ISO_A3"] = world_raw.loc[mask, "ADM0_A3"]
    # backwards compatibility
    return world_raw.rename(columns={"GDP_MD": "GDP_MD_EST"})


# any change between versions?
def df_same(new, old, dataset, log):
    assert (new.columns == old.columns).all(), "columns should be the same"
    if new.shape != old.shape:
        dfc = old.merge(new, on="name", how="outer", suffixes=("_old", "_new")).loc[
            lambda d: d.isna().any(axis=1)
        ]
        log.append(f"### {dataset} row count changed ###\n{dfc.to_markdown()}")
        return False
    dfc = new.compare(old)
    if len(dfc) > 0:
        log.append(f"### {dataset} data changed ###\n{dfc.to_markdown()}")
    return len(dfc) == 0


config = [
    {
        "file": "ne_110m_populated_places.zip",
        "cols": ["NAME", "geometry"],
        "current": gpd.datasets.get_path("naturalearth_cities"),
    },
    {
        "file": "ne_110m_admin_0_countries.zip",
        "cols": ["POP_EST", "CONTINENT", "NAME", "ISO_A3", "GDP_MD_EST", "geometry"],
        "override": countries_override,
        "current": gpd.datasets.get_path("naturalearth_lowres"),
    },
]

downloads = {}
log = []
for dl in config:
    with tempfile.TemporaryDirectory() as tmpdirname:
        url = urlbase + dl["file"]
        r = requests.get(
            url,
            stream=True,
            headers={"User-Agent": "XY"},
            params=None if version == "latest" else {"version": version},
        )
        assert (
            r.status_code == 200
        ), f"version: {version} does not exist. status: {r.status_code}"

        f = Path(tmpdirname).joinpath(dl["file"])
        with open(f, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
        # extract the natural earth version
        z = ZipFile(f)
        version_f = [i for i in z.infolist() if "VERSION" in i.filename]
        assert len(version_f) == 1, "failed to find VERSION file"
        with open(z.extract(version_f[0], Path(tmpdirname).joinpath("v.txt"))) as f_:
            dl_version = f_.read().strip()

        # extract geodataframe from zip
        gdf = gpd.read_file(f)
        # maintain structure that geopandas distributes
        if "override" in dl.keys():
            gdf = dl["override"](gdf)
        gdf = gdf.loc[:, dl["cols"]]
        gdf = gdf.rename(columns={c: c.lower() for c in gdf.columns})

        # override Crimea #2382
        if dl["file"] == "ne_110m_admin_0_countries.zip":
            crimean_bbox = box(32.274, 44.139, 36.65, 46.704)
            crimea_only = (
                gdf.loc[gdf.name == "Russia", "geometry"]
                .iloc[0]
                .intersection(crimean_bbox)
            )
            complete_ukraine = (
                gdf.loc[gdf.name == "Ukraine", "geometry"].iloc[0].union(crimea_only)
            )
            correct_russia = (
                gdf.loc[gdf.name == "Russia", "geometry"]
                .iloc[0]
                .difference(crimean_bbox)
            )
            r_ix = gdf.loc[gdf.name == "Russia"].index[0]
            gdf.at[r_ix, "geometry"] = correct_russia

            u_ix = gdf.loc[gdf.name == "Ukraine"].index[0]
            gdf.at[u_ix, "geometry"] = complete_ukraine

        # get changes between current version and new version
        if not df_same(gdf, gpd.read_file(dl["current"]), dl["file"], log):
            downloads[dl["file"]] = gdf


# create change log that can be pasted into PR
with open(f"CHANGE_{dl_version}.md", "w") as f:
    f.write("\n\n".join(log))

# save downloaded geodataframe to appropriate place
for k, gdf_ in downloads.items():
    f = [Path(c["current"]) for c in config if c["file"] == k][0]
    gdf_.to_file(driver="ESRI Shapefile", filename=Path(f.parent.name).joinpath(f.name))
