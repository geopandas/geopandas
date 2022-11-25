import json
from pathlib import Path

locks = list(Path.home().joinpath("geopandas/ci/lock").glob("*.lock"))
envs = list(Path.home().joinpath("geopandas/ci/envs").glob("*.yaml"))

d = [
    {"file": str(Path(*fn.parts[-3:])), "name": fn.stem}
    for fn in locks + [e for e in envs if e.stem not in [l.stem for l in locks]]
    if fn.suffix == ".lock"
]
d = d[0:1]
print(f"::set-output name=env::{json.dumps(d)}")