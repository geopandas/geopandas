import json
from pathlib import Path

locks = list(Path.home().joinpath("geopandas/ci/lock").glob("*.lock"))
envs = list(Path.home().joinpath("geopandas/ci/envs").glob("*.yaml"))

d = [
    {"env": str(Path(*fn.parts[-3:])), "env_name": fn.stem}
    for fn in locks + [e for e in envs if e.stem not in [l.stem for l in locks]]
    # if fn.suffix == ".yaml"
]
# d = d[0:1]
print(f"::set-output name=env::{json.dumps(d)}")