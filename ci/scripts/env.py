from pathlib import Path
import json

f = list(Path.home().joinpath("geopandas/ci/lock").glob("*.lock"))
# d = {"env":[str(Path(*fn.parts[-3:])) for fn in f],
# "env_name":[fn.stem for fn in f]}

d = [{"env":str(Path(*fn.parts[-3:])), "env_name":fn.name} for fn in f]

print(f"::set-output name=env::{json.dumps(d)}")