import json, os, random
from pathlib import Path

# is runtime env nektos/act?
ACT = "ACT" in os.environ and os.environ["ACT"] == "true"
ADD_INCLUDE = json.loads(os.environ["ADD_INCLUDE"]) if "ADD_INCLUDE" in os.environ else True
ENV_YAML = os.environ["ENV_YAML"] if "ENV_YAML" in os.environ else "*.yaml"

# special case development envs
dev = ["310-dev"]

# environment that are tested on macos and windows as well
all_os = ["38-latest-conda-forge", "39-latest-conda-forge"]

# need the name for conda activation and filename for building
# conda env
env = [
    {"file": str(Path(*fn.parts[-3:])), "name": fn.stem}
    for fn in Path.cwd().joinpath("ci/envs").glob(ENV_YAML)
    if fn.stem not in dev
]

# add macos and windows environments to the matrix
inc = [
    {"os": runner, "env": e, "postgis": False, "dev": False}
    for e in env
    if e["name"] in all_os
    for runner in ["windows-latest", "macos-latest"]
]

# add the dev environments
inc += [
    {"os": "ubuntu-latest", "env": e, "postgis": False, "dev": True}
    for e in [{"name": n, "file": f"ci/envs/{n}.yaml"} for n in dev]
]

matrix = {
    "env": env,
    "os": ["ubuntu-latest"],
    "postgis": [False],
    "dev": [False],
    # "include": inc,
}
if ADD_INCLUDE:
    matrix["include"] = inc

if ACT:
    opts = {
        "miniforge-version": "",
        "miniforge-variant": "",
        "run-post": "false",
    }
else:
    opts = {
        "miniforge-version": "latest",
        "miniforge-variant": "Mambaforge",
        "run-post": "true",
    }

print(f"opts={json.dumps(opts)}")
print(f"matrix={json.dumps(matrix)}")

