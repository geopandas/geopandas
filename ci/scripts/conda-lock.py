from pathlib import Path
import subprocess

def cmd(cmd="ls", assertfail=True):
    up = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    o = []
    for iob in up.stdout:
        o.append(iob.rstrip().decode())

    exitcode = up.wait()

    if assertfail:
        assert exitcode == 0, f"[{exitcode}] {cmd} {str}"

    return exitcode,  o

d = Path.cwd().joinpath("../geopandas")
print(d, d.exists())
ld = d.joinpath("ci/lock")
ld.mkdir(exist_ok=True)

for f in d.joinpath("ci/envs").glob("*.yaml"):
    lf = ld.joinpath(f"{f.stem}.lock")
    acmd = f"conda-lock --kind explicit -p linux-64 --mamba -f {f} --filename-template {lf}"
    if not lf.exists():
        print(f"\033[1m{acmd}\033[0m")
        exitcode, o = cmd(acmd, assertfail=False)
        if exitcode != 0:
            for n in range(1,len(o)):
                if o[n].startswith("Traceback"):
                    break
            print("\n".join(o[0:n]))
        
